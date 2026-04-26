from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .velocity_command import UniformVelocityCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_SCENE_CFG = SceneEntityCfg("robot")


class VelocityStage(TypedDict):
  step: int
  lin_vel_x: tuple[float, float] | None
  lin_vel_y: tuple[float, float] | None
  ang_vel_z: tuple[float, float] | None


class RewardWeightStage(TypedDict):
  step: int
  weight: float


class RecoveryStage(TypedDict):
  step: int
  recovery_prob: float


class Stage2Transition(TypedDict):
  min_steps: int
  stand_success_threshold: float


class StandSuccessCfg(TypedDict):
  max_tilt_deg: float
  min_base_height: float
  ema_alpha: float


def terrain_levels_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]

  terrain = env.scene.terrain
  assert terrain is not None
  terrain_generator = terrain.cfg.terrain_generator
  assert terrain_generator is not None

  command = env.command_manager.get_command(command_name)
  assert command is not None

  # Compute the distance the robot walked.
  distance = torch.norm(
    asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1
  )

  # Robots that walked far enough progress to harder terrains.
  move_up = distance > terrain_generator.size[0] / 2

  # Robots that walked less than half of their required distance go to simpler
  # terrains.
  move_down = (
    distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
  )
  move_down *= ~move_up

  # Update terrain levels.
  terrain.update_env_origins(env_ids, move_up, move_down)

  return torch.mean(terrain.terrain_levels.float())


def commands_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  velocity_stages: list[VelocityStage],
) -> dict[str, torch.Tensor]:
  del env_ids  # Unused.
  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None
  cfg = cast(UniformVelocityCommandCfg, command_term.cfg)
  for stage in velocity_stages:
    if env.common_step_counter >= stage["step"]:
      if "lin_vel_x" in stage and stage["lin_vel_x"] is not None:
        cfg.ranges.lin_vel_x = stage["lin_vel_x"]
      if "lin_vel_y" in stage and stage["lin_vel_y"] is not None:
        cfg.ranges.lin_vel_y = stage["lin_vel_y"]
      if "ang_vel_z" in stage and stage["ang_vel_z"] is not None:
        cfg.ranges.ang_vel_z = stage["ang_vel_z"]
  return {
    # "lin_vel_x_min": torch.tensor(cfg.ranges.lin_vel_x[0]),
    # "lin_vel_x_max": torch.tensor(cfg.ranges.lin_vel_x[1]),
    # "lin_vel_y_min": torch.tensor(cfg.ranges.lin_vel_y[0]),
    # "lin_vel_y_max": torch.tensor(cfg.ranges.lin_vel_y[1]),
    # "ang_vel_z_min": torch.tensor(cfg.ranges.ang_vel_z[0]),
    # "ang_vel_z_max": torch.tensor(cfg.ranges.ang_vel_z[1]),
  }


def reward_weight(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  reward_name: str,
  weight_stages: list[RewardWeightStage],
) -> torch.Tensor:
  """Update a reward term's weight based on training step stages."""
  del env_ids  # Unused.
  reward_term_cfg = env.reward_manager.get_term_cfg(reward_name)
  for stage in weight_stages:
    if env.common_step_counter >= stage["step"]:
      reward_term_cfg.weight = stage["weight"]
  return torch.tensor([reward_term_cfg.weight])


def _apply_velocity_stage(
  cfg: UniformVelocityCommandCfg,
  velocity_stages: list[VelocityStage],
  step: int,
) -> None:
  for stage in velocity_stages:
    if step >= stage["step"]:
      if "lin_vel_x" in stage and stage["lin_vel_x"] is not None:
        cfg.ranges.lin_vel_x = stage["lin_vel_x"]
      if "lin_vel_y" in stage and stage["lin_vel_y"] is not None:
        cfg.ranges.lin_vel_y = stage["lin_vel_y"]
      if "ang_vel_z" in stage and stage["ang_vel_z"] is not None:
        cfg.ranges.ang_vel_z = stage["ang_vel_z"]


def _apply_recovery_stage(
  cfg: UniformVelocityCommandCfg,
  recovery_stages: list[RecoveryStage],
  step: int,
) -> float:
  recovery_prob = 0.0
  for stage in recovery_stages:
    if step >= stage["step"]:
      recovery_prob = stage["recovery_prob"]
  recovery_prob = max(0.0, min(1.0, recovery_prob))
  cfg.standing_task_weight = (1.0 - recovery_prob, recovery_prob)
  return recovery_prob


def _apply_reward_stages(
  env: ManagerBasedRlEnv,
  reward_weight_stages: dict[str, list[RewardWeightStage]],
  step: int,
) -> dict[str, float]:
  active_weights: dict[str, float] = {}
  for reward_name, stages in reward_weight_stages.items():
    reward_term_cfg = env.reward_manager.get_term_cfg(reward_name)
    for stage in stages:
      if step >= stage["step"]:
        reward_term_cfg.weight = stage["weight"]
    active_weights[reward_name] = reward_term_cfg.weight
  return active_weights


def standup_then_track_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  stand_success: StandSuccessCfg,
  stage2_transition: Stage2Transition,
  stage1_velocity_stages: list[VelocityStage],
  stage2_velocity_stages: list[VelocityStage],
  stage1_recovery_stages: list[RecoveryStage],
  stage2_recovery_stages: list[RecoveryStage],
  stage1_reward_weight_stages: dict[str, list[RewardWeightStage]],
  stage2_reward_weight_stages: dict[str, list[RewardWeightStage]],
  asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
) -> dict[str, torch.Tensor]:
  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None
  command_cfg = cast(UniformVelocityCommandCfg, command_term.cfg)

  state = getattr(env, "_standup_curriculum_state", None)
  if state is None:
    state = {
      "stage": 1,
      "stage2_start_step": -1,
      "stand_success_ema": 0.0,
    }
    setattr(env, "_standup_curriculum_state", state)

  asset: Entity = env.scene[asset_cfg.name]
  reset_ids = env_ids
  if isinstance(env_ids, slice):
    reset_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)

  max_tilt_rad = torch.deg2rad(
    torch.tensor(stand_success["max_tilt_deg"], device=env.device)
  )
  tilt = torch.acos(-asset.data.projected_gravity_b[reset_ids, 2]).abs()
  base_height = asset.data.root_link_pos_w[reset_ids, 2]
  is_upright = tilt < max_tilt_rad
  is_tall_enough = base_height > stand_success["min_base_height"]
  stand_success_rate = torch.mean((is_upright & is_tall_enough).float()).item()

  alpha = stand_success["ema_alpha"]
  prev_ema = float(state["stand_success_ema"])
  stand_success_ema = (1.0 - alpha) * prev_ema + alpha * stand_success_rate
  state["stand_success_ema"] = stand_success_ema

  if state["stage"] == 1:
    has_min_steps = env.common_step_counter >= stage2_transition["min_steps"]
    ready_to_switch = (
      has_min_steps
      and stand_success_ema >= stage2_transition["stand_success_threshold"]
    )
    if ready_to_switch:
      state["stage"] = 2
      state["stage2_start_step"] = env.common_step_counter

  current_stage = int(state["stage"])
  if current_stage == 1:
    local_step = env.common_step_counter
    _apply_velocity_stage(command_cfg, stage1_velocity_stages, local_step)
    recovery_prob = _apply_recovery_stage(
      command_cfg, stage1_recovery_stages, local_step
    )
    active_weights = _apply_reward_stages(
      env, stage1_reward_weight_stages, local_step
    )
  else:
    stage2_start_step = int(state["stage2_start_step"])
    local_step = max(0, env.common_step_counter - stage2_start_step)
    _apply_velocity_stage(command_cfg, stage2_velocity_stages, local_step)
    recovery_prob = _apply_recovery_stage(
      command_cfg, stage2_recovery_stages, local_step
    )
    active_weights = _apply_reward_stages(
      env, stage2_reward_weight_stages, local_step
    )

  state["active_weights"] = active_weights

  return {
    "stage": torch.tensor(float(state["stage"]), device=env.device),
    "stand_success": torch.tensor(stand_success_rate, device=env.device),
    "stand_success_ema": torch.tensor(stand_success_ema, device=env.device),
    "recovery_prob": torch.tensor(recovery_prob, device=env.device),
    "stage_step": torch.tensor(float(local_step), device=env.device),
  }
