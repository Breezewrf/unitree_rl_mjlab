"""Export a tracking policy with its reference motion embedded in ONNX.

The exported model follows the BeyondMimic deployment contract:

  inputs:  obs, time_step
  outputs: actions, joint_pos, joint_vel, body_pos_w, body_quat_w,
           body_lin_vel_w, body_ang_vel_w

Both an RSL-RL checkpoint (``.pt``) and a policy ONNX file are accepted. CSV
motions are converted through ``scripts/csv_to_npz.py`` before export.
"""

from __future__ import annotations

import copy
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np
import onnx
import tyro
from onnx import TensorProto, helper, numpy_helper

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import get_base_metadata, list_to_csv_str
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommand, MotionCommandCfg

import src.tasks  # noqa: F401  # Populate the task registry.

_MOTION_OUTPUT_NAMES = (
  "joint_pos",
  "joint_vel",
  "body_pos_w",
  "body_quat_w",
  "body_lin_vel_w",
  "body_ang_vel_w",
)


@dataclass
class ExportConfig:
  checkpoint_file: str
  """Input RSL-RL checkpoint (.pt) or policy model (.onnx)."""
  motion_file: str
  """Reference motion in tracking NPZ or retargeted CSV format."""
  output_file: str = ""
  """Destination ONNX. Empty writes <checkpoint_stem>_motion.onnx."""
  task: str = "Agibot-X2-Tracking"
  """Tracking task whose observation layout and robot metadata match the policy."""
  device: str = "cpu"
  """Device used to restore PT checkpoints and convert CSV motion."""
  csv_robot: Literal["g1", "g1_23dof", "x2"] | None = None
  """CSV joint layout. Empty infers it from the task name."""
  input_fps: float = 30.0
  """Input CSV frame rate."""
  output_fps: float = 50.0
  """CSV interpolation and exported motion frame rate."""


def _default_output_path(checkpoint_path: Path) -> Path:
  return checkpoint_path.with_name(f"{checkpoint_path.stem}_motion.onnx")


def _infer_csv_robot(task: str) -> Literal["g1", "g1_23dof", "x2"]:
  normalized = task.lower()
  if "x2" in normalized:
    return "x2"
  if "23dof" in normalized:
    return "g1_23dof"
  if "g1" in normalized:
    return "g1"
  raise ValueError(
    f"Cannot infer the CSV robot layout from task {task!r}; set --csv-robot."
  )


def _convert_csv_motion(
  csv_path: Path,
  output_path: Path,
  *,
  task: str,
  csv_robot: Literal["g1", "g1_23dof", "x2"] | None,
  input_fps: float,
  output_fps: float,
  device: str,
) -> None:
  converter = Path(__file__).with_name("csv_to_npz.py")
  robot = csv_robot or _infer_csv_robot(task)
  command = [
    sys.executable,
    str(converter),
    "--robot",
    robot,
    "--input-file",
    str(csv_path),
    "--output-name",
    str(output_path.resolve()),
    "--input-fps",
    str(input_fps),
    "--output-fps",
    str(output_fps),
    "--device",
    device,
  ]
  print(f"[INFO] Converting CSV motion with layout {robot!r}...", flush=True)
  subprocess.run(command, check=True, env=os.environ.copy())


def _prepare_motion_path(cfg: ExportConfig, temp_dir: Path) -> Path:
  motion_path = Path(cfg.motion_file).expanduser().resolve()
  if not motion_path.is_file():
    raise FileNotFoundError(f"Motion file not found: {motion_path}")
  if motion_path.suffix.lower() == ".npz":
    return motion_path
  if motion_path.suffix.lower() == ".csv":
    converted_path = temp_dir / f"{motion_path.stem}.npz"
    _convert_csv_motion(
      motion_path,
      converted_path,
      task=cfg.task,
      csv_robot=cfg.csv_robot,
      input_fps=cfg.input_fps,
      output_fps=cfg.output_fps,
      device=cfg.device,
    )
    return converted_path
  raise ValueError(
    f"Unsupported motion format: {motion_path.suffix}; use .npz or .csv"
  )


def _find_policy_io(model: onnx.ModelProto) -> tuple[str, str]:
  input_names = [item.name for item in model.graph.input]
  output_names = [item.name for item in model.graph.output]
  policy_inputs = [name for name in input_names if name != "time_step"]
  if "obs" in policy_inputs:
    obs_name = "obs"
  elif len(policy_inputs) == 1:
    obs_name = policy_inputs[0]
  else:
    raise ValueError(f"Cannot determine policy observation input from {input_names}")
  actions_name = "actions" if "actions" in output_names else output_names[0]
  return obs_name, actions_name


def _rename_graph_value(graph: onnx.GraphProto, old: str, new: str) -> None:
  if old == new:
    return
  for collection in (graph.input, graph.output, graph.value_info, graph.initializer):
    for item in collection:
      if item.name == old:
        item.name = new
  for node in graph.node:
    for index, name in enumerate(node.input):
      if name == old:
        node.input[index] = new
    for index, name in enumerate(node.output):
      if name == old:
        node.output[index] = new


def _default_opset(model: onnx.ModelProto) -> int:
  for item in model.opset_import:
    if item.domain in ("", "ai.onnx"):
      return item.version
  raise ValueError("The ONNX model has no default-domain opset import")


def _static_feature_dim(value: onnx.ValueInfoProto) -> int | None:
  dims = value.type.tensor_type.shape.dim
  if len(dims) != 2:
    raise ValueError(
      f"ONNX value {value.name!r} must have rank 2, got rank {len(dims)}"
    )
  feature_dim = dims[-1]
  return feature_dim.dim_value if feature_dim.HasField("dim_value") else None


def _validate_policy_dimensions(
  model: onnx.ModelProto,
  obs_name: str,
  actions_name: str,
  *,
  expected_obs_dim: int,
  expected_action_dim: int,
) -> None:
  obs_info = next(item for item in model.graph.input if item.name == obs_name)
  actions_info = next(item for item in model.graph.output if item.name == actions_name)
  obs_dim = _static_feature_dim(obs_info)
  action_dim = _static_feature_dim(actions_info)
  if obs_dim is not None and obs_dim != expected_obs_dim:
    raise ValueError(
      f"Policy observation width {obs_dim} does not match the task width "
      f"{expected_obs_dim}"
    )
  if action_dim is not None and action_dim != expected_action_dim:
    raise ValueError(
      f"Policy action width {action_dim} does not match the task width "
      f"{expected_action_dim}"
    )


def _bundle_policy_onnx(
  policy_path: Path,
  output_path: Path,
  motion_arrays: dict[str, np.ndarray],
  *,
  expected_obs_dim: int,
  expected_action_dim: int,
) -> None:
  source_model = onnx.load(str(policy_path))
  onnx.checker.check_model(source_model)
  obs_name, actions_name = _find_policy_io(source_model)
  _validate_policy_dimensions(
    source_model,
    obs_name,
    actions_name,
    expected_obs_dim=expected_obs_dim,
    expected_action_dim=expected_action_dim,
  )

  # Extracting obs -> actions also removes any reference motion already bundled.
  model = onnx.utils.Extractor(source_model).extract_model(
    [obs_name], [actions_name]
  )
  graph = model.graph
  _rename_graph_value(graph, obs_name, "obs")

  if actions_name != "actions":
    original_output = copy.deepcopy(graph.output[0])
    del graph.output[:]
    graph.node.append(
      helper.make_node(
        "Identity",
        inputs=[actions_name],
        outputs=["actions"],
        name="bundle_motion/rename_actions",
      )
    )
    original_output.name = "actions"
    graph.output.append(original_output)

  frame_counts = {array.shape[0] for array in motion_arrays.values()}
  if len(frame_counts) != 1:
    raise ValueError(f"Motion arrays have inconsistent frame counts: {frame_counts}")
  num_frames = frame_counts.pop()
  if num_frames < 1:
    raise ValueError("Reference motion is empty")

  opset = _default_opset(model)
  graph.input.append(
    helper.make_tensor_value_info("time_step", TensorProto.FLOAT, [1, 1])
  )
  if opset >= 13:
    graph.initializer.append(
      numpy_helper.from_array(
        np.asarray([1], dtype=np.int64), "bundle_motion/squeeze_axes"
      )
    )
  if opset >= 12:
    graph.initializer.append(
      numpy_helper.from_array(
        np.asarray(num_frames - 1, dtype=np.int64),
        "bundle_motion/max_time_step",
      )
    )
  graph.node.append(
    helper.make_node(
      "Cast",
      inputs=["time_step"],
      outputs=["bundle_motion/time_step_int64"],
      to=TensorProto.INT64,
      name="bundle_motion/cast_time_step",
    )
  )
  if opset >= 13:
    graph.node.append(
      helper.make_node(
        "Squeeze",
        inputs=["bundle_motion/time_step_int64", "bundle_motion/squeeze_axes"],
        outputs=["bundle_motion/time_step_squeezed"],
        name="bundle_motion/squeeze_time_step",
      )
    )
  else:
    graph.node.append(
      helper.make_node(
        "Squeeze",
        inputs=["bundle_motion/time_step_int64"],
        outputs=["bundle_motion/time_step_squeezed"],
        axes=[1],
        name="bundle_motion/squeeze_time_step",
      )
    )
  time_step_index = "bundle_motion/time_step_clamped"
  if opset >= 12:
    graph.node.append(
      helper.make_node(
        "Clip",
        inputs=[
          "bundle_motion/time_step_squeezed",
          "",
          "bundle_motion/max_time_step",
        ],
        outputs=[time_step_index],
        name="bundle_motion/clamp_time_step",
      )
    )
  else:
    graph.node.append(
      helper.make_node(
        "Cast",
        inputs=["bundle_motion/time_step_squeezed"],
        outputs=["bundle_motion/time_step_float"],
        to=TensorProto.FLOAT,
        name="bundle_motion/cast_time_step_float",
      )
    )
    if opset >= 11:
      graph.initializer.append(
        numpy_helper.from_array(
          np.asarray(num_frames - 1, dtype=np.float32),
          "bundle_motion/max_time_step_float",
        )
      )
      clip_inputs = [
        "bundle_motion/time_step_float",
        "",
        "bundle_motion/max_time_step_float",
      ]
      clip_attributes = {}
    else:
      clip_inputs = ["bundle_motion/time_step_float"]
      clip_attributes = {"max": float(num_frames - 1)}
    graph.node.append(
      helper.make_node(
        "Clip",
        inputs=clip_inputs,
        outputs=["bundle_motion/time_step_clamped_float"],
        name="bundle_motion/clamp_time_step",
        **clip_attributes,
      )
    )
    graph.node.append(
      helper.make_node(
        "Cast",
        inputs=["bundle_motion/time_step_clamped_float"],
        outputs=[time_step_index],
        to=TensorProto.INT64,
        name="bundle_motion/cast_time_step_index",
      )
    )

  for output_name in _MOTION_OUTPUT_NAMES:
    array = np.ascontiguousarray(motion_arrays[output_name], dtype=np.float32)
    initializer_name = f"bundle_motion/reference_{output_name}"
    graph.initializer.append(numpy_helper.from_array(array, initializer_name))
    graph.node.append(
      helper.make_node(
        "Gather",
        inputs=[initializer_name, time_step_index],
        outputs=[output_name],
        axis=0,
        name=f"bundle_motion/gather_{output_name}",
      )
    )
    graph.output.append(
      helper.make_tensor_value_info(
        output_name, TensorProto.FLOAT, [1, *array.shape[1:]]
      )
    )

  model.producer_name = "unitree_rl_mjlab"
  onnx.checker.check_model(model)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  onnx.save(model, str(output_path))


def _replace_metadata(
  onnx_path: Path, metadata: dict[str, list | str | float]
) -> None:
  model = onnx.load(str(onnx_path))
  merged = {item.key: item.value for item in model.metadata_props}
  for key, value in metadata.items():
    merged[key] = list_to_csv_str(value) if isinstance(value, list) else str(value)
  del model.metadata_props[:]
  for key, value in merged.items():
    entry = model.metadata_props.add()
    entry.key = key
    entry.value = value
  onnx.checker.check_model(model)
  onnx.save(model, str(onnx_path))


def _validate_npz_layout(motion_path: Path) -> float:
  with np.load(motion_path) as data:
    missing = (set(_MOTION_OUTPUT_NAMES) | {"fps"}).difference(data.files)
    if missing:
      raise ValueError(f"Motion NPZ is missing fields: {sorted(missing)}")
    fps = np.asarray(data["fps"]).reshape(-1)
    if fps.size != 1 or not np.isfinite(fps[0]) or fps[0] <= 0:
      raise ValueError(f"Motion NPZ has invalid fps: {data['fps']}")
    frame_counts = {data[name].shape[0] for name in _MOTION_OUTPUT_NAMES}
    if len(frame_counts) != 1:
      raise ValueError(f"Motion NPZ frame counts differ: {frame_counts}")
    for name in _MOTION_OUTPUT_NAMES:
      if not np.isfinite(data[name]).all():
        raise ValueError(f"Motion field {name!r} contains NaN or Inf")
    return float(fps[0])


def _export(cfg: ExportConfig, motion_path: Path, output_path: Path) -> None:
  motion_fps = _validate_npz_layout(motion_path)

  env_cfg = load_env_cfg(cfg.task)
  if "motion" not in env_cfg.commands or not isinstance(
    env_cfg.commands["motion"], MotionCommandCfg
  ):
    raise ValueError(f"Task {cfg.task!r} is not a tracking task")
  env_cfg.scene.num_envs = 1
  env_cfg.commands["motion"].motion_file = str(motion_path)
  agent_cfg = load_rl_cfg(cfg.task)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device)
  try:
    motion_term = cast(MotionCommand, env.command_manager.get_term("motion"))
    motion_arrays = {
      name: getattr(motion_term.motion, name).detach().cpu().numpy()
      for name in _MOTION_OUTPUT_NAMES
    }
    robot = env.scene["robot"]
    if motion_arrays["joint_pos"].shape[1] != len(robot.joint_names):
      raise ValueError(
        "Motion joint width does not match the task robot: "
        f"{motion_arrays['joint_pos'].shape[1]} != {len(robot.joint_names)}"
      )
    actor_obs_shape = env.observation_manager.group_obs_dim["actor"]
    if not isinstance(actor_obs_shape, tuple) or len(actor_obs_shape) != 1:
      raise ValueError(f"Expected a flat actor observation, got {actor_obs_shape}")
    expected_obs_dim = actor_obs_shape[0]
    expected_action_dim = env.action_manager.total_action_dim

    checkpoint_path = Path(cfg.checkpoint_file).expanduser().resolve()
    if checkpoint_path.suffix.lower() == ".onnx":
      _bundle_policy_onnx(
        checkpoint_path,
        output_path,
        motion_arrays,
        expected_obs_dim=expected_obs_dim,
        expected_action_dim=expected_action_dim,
      )
    elif checkpoint_path.suffix.lower() == ".pt":
      wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
      runner_cls = load_runner_cls(cfg.task) or MjlabOnPolicyRunner
      runner = runner_cls(wrapped_env, asdict(agent_cfg), device=cfg.device)
      runner.load(
        str(checkpoint_path),
        load_cfg={"actor": True},
        strict=True,
        map_location=cfg.device,
      )
      export_fn = getattr(runner, "export_motion_policy_to_onnx", None)
      if export_fn is None:
        raise TypeError(
          f"Runner {type(runner).__name__} cannot export a bundled motion policy"
        )
      output_path.parent.mkdir(parents=True, exist_ok=True)
      export_fn(str(output_path.parent), output_path.name)
    else:
      raise ValueError(
        f"Unsupported checkpoint format: {checkpoint_path.suffix}; use .pt or .onnx"
      )

    metadata = get_base_metadata(env, str(checkpoint_path))
    metadata.update(
      {
        "anchor_body_name": motion_term.cfg.anchor_body_name,
        "body_names": list(motion_term.cfg.body_names),
        "motion_num_frames": str(motion_arrays["joint_pos"].shape[0]),
        "motion_fps": str(motion_fps),
        "task": cfg.task,
      }
    )
    _replace_metadata(output_path, metadata)
  finally:
    env.close()


def main(cfg: ExportConfig) -> None:
  checkpoint_path = Path(cfg.checkpoint_file).expanduser().resolve()
  if not checkpoint_path.is_file():
    raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
  output_path = (
    Path(cfg.output_file).expanduser().resolve()
    if cfg.output_file
    else _default_output_path(checkpoint_path)
  )
  if output_path.suffix.lower() != ".onnx":
    output_path = output_path.with_suffix(".onnx")

  with tempfile.TemporaryDirectory(prefix="tracking_onnx_export_") as temp_dir:
    motion_path = _prepare_motion_path(cfg, Path(temp_dir))
    _export(cfg, motion_path, output_path)

  model = onnx.load(str(output_path), load_external_data=False)
  onnx.checker.check_model(model)
  print(f"[INFO] Exported bundled tracking policy: {output_path}")
  print(f"[INFO] Inputs: {[item.name for item in model.graph.input]}")
  print(f"[INFO] Outputs: {[item.name for item in model.graph.output]}")


if __name__ == "__main__":
  main(tyro.cli(ExportConfig))
