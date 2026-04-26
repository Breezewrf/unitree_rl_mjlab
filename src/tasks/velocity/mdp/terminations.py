from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def illegal_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    return (force_mag > force_threshold).any(dim=-1).any(dim=-1)  # [B]
  assert data.found is not None
  return torch.any(data.found, dim=-1)


from mjlab.managers.scene_entity_config import SceneEntityCfg
import math
if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

def _apply_termination_tolerance(
  env: ManagerBasedRlEnv,
  violation: torch.Tensor,
  tolerance_time_s: float,
  term_key: str,
) -> torch.Tensor:
  """Apply optional grace time before returning a terminal condition."""
  if tolerance_time_s <= 0.0:
    return violation

  tolerance_steps = max(1, math.ceil(tolerance_time_s / env.step_dt))
  counters = getattr(env, "_termination_tolerance_counters", None)
  if counters is None:
    counters = {}
    setattr(env, "_termination_tolerance_counters", counters)

  if term_key not in counters:
    counters[term_key] = torch.zeros(
      env.num_envs,
      device=env.device,
      dtype=torch.int32,
    )
  violation_counts = counters[term_key]

  violation_counts[env.episode_length_buf == 0] = 0
  violation_counts[violation] += 1
  violation_counts[~violation] = 0
  return violation_counts >= tolerance_steps

def bad_orientation_tolerance(
  env: ManagerBasedRlEnv,
  limit_angle: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  tolerance_time_s: float = 0.0,
  tolerance_key: str | None = None,
  command_name: str = "twist",
) -> torch.Tensor:
  """Terminate when orientation exceeds the limit angle.

  Tolerance is applied only for standing-task environments
  (``command_term.is_standing_task_env == True``). Non-standing environments use
  immediate termination on violation.
  """
  asset: Entity = env.scene[asset_cfg.name]
  projected_gravity = asset.data.projected_gravity_b
  violation = torch.acos(-projected_gravity[:, 2]).abs() > limit_angle

  if tolerance_time_s <= 0.0:
    return violation

  command_term = env.command_manager.get_term(command_name)
  standing_mask = None
  if command_term is not None:
    standing_mask = getattr(command_term, "is_standing_task_env", None)

  if standing_mask is None:
    return violation

  standing_mask = standing_mask.to(device=env.device, dtype=torch.bool)
  if standing_mask.shape != violation.shape:
    return violation

  standing_violation = violation & standing_mask
  term_key = (
    tolerance_key
    or f"bad_orientation:{asset_cfg.name}:{limit_angle:.6f}"
  )
  standing_terminated = _apply_termination_tolerance(
    env,
    standing_violation,
    tolerance_time_s=tolerance_time_s,
    term_key=f"{term_key}:standing",
  )
  return torch.where(standing_mask, standing_terminated, violation)