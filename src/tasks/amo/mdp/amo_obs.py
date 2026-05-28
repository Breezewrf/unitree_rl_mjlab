from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def _quat_to_euler_xyz(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """Convert quaternion (qw, qx, qy, qz) to Euler angles (roll, pitch, yaw).

    Uses the XYZ intrinsic convention (roll about X, pitch about Y, yaw about Z).
    Returns tensor of shape (..., 3).
    """
    qw, qx, qy, qz = quat_wxyz.unbind(-1)

    # Roll (X-axis rotation).
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # Pitch (Y-axis rotation).
    sinp = 2.0 * (qw * qy - qz * qx)
    sinp = torch.clamp(sinp, -1.0, 1.0)
    pitch = torch.asin(sinp)

    # Yaw (Z-axis rotation).
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack([roll, pitch, yaw], dim=-1)


def base_orientation(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return base orientation as Euler angles (roll, pitch, yaw)."""
    asset = env.scene[asset_cfg.name]
    quat_wxyz = asset.data.root_link_quat_w  # (N, 4) in (qw, qx, qy, qz) order.
    return _quat_to_euler_xyz(quat_wxyz)


def amo_ref_lower(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return AMO MLP lower-body reference joint angles.

    During a normal step the reward manager runs first and populates the cache
    (via ``_get_amo_ref_lower``); this function reads from it.  A fallback
    path handles the case where the function is called before the first step
    (e.g. during ``_prepare_terms()``).

    Requires ``env._amo_module`` to be set by a startup event.
    """
    # Fast path: cache was populated earlier this step by reward computation.
    cached_step = getattr(env, "_amo_ref_lower_step", -1)
    if cached_step == env.common_step_counter and env._amo_ref_lower_cache is not None:
        return env._amo_ref_lower_cache

    amo_module = getattr(env, "_amo_module", None)
    if amo_module is None:
        # During _prepare_terms() the startup event hasn't run yet.
        asset = env.scene[asset_cfg.name]
        return torch.zeros(asset.data.joint_pos.shape[0], 12, device=env.device)

    # Fallback: compute on device (first step, or called outside normal step loop).
    asset = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos
    upper_indices = env._amo_upper_indices
    q_upper = joint_pos[:, upper_indices]
    amo_cmd = env.command_manager.get_command("amo")
    rpy_cmd = amo_cmd[:, 3:6]
    h_cmd = amo_cmd[:, 6:7]
    x = torch.cat([q_upper, rpy_cmd, h_cmd], dim=1)
    result = amo_module.batch_predict(x)

    env._amo_ref_lower_cache = result
    env._amo_ref_lower_step = env.common_step_counter
    return result


def amo_phase(
    env: ManagerBasedRlEnv,
    period: float,
    command_name: str,
) -> torch.Tensor:
    """Gait phase signal that zeros out for standing commands.

    Unlike the base ``phase`` function, this only checks the velocity portion
    of the AMO command (first 3 elements) to determine standing status,
    ignoring rpy and height commands.
    """
    global_phase = (env.episode_length_buf * env.step_dt) % period / period
    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    cmd = env.command_manager.get_command(command_name)
    stand_mask = torch.linalg.norm(cmd[:, :3], dim=1) < 0.1
    phase = torch.where(stand_mask.unsqueeze(1), torch.zeros_like(phase), phase)
    return phase


def whole_body_actions(
    env: ManagerBasedRlEnv,
    n_upper_joints: int = 17,
) -> torch.Tensor:
    """Return whole-body last actions with upper-body padding.

    The policy only controls lower-body joints (12 DOF).  This function
    pads the upper-body portion with zeros so that the observation has the
    full 29-DOF action vector: [a_lower (12), a_upper_zero (17)].
    """
    lower_actions = env.action_manager.action  # (N, 12)
    upper_zeros = torch.zeros(
        lower_actions.shape[0], n_upper_joints, device=lower_actions.device
    )
    return torch.cat([lower_actions, upper_zeros], dim=-1)
