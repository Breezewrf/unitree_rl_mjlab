from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import numpy as np

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

    Reads the current upper-body joint positions, the commanded rpy + height
    from the AMO command term, runs MLP inference on CPU, and returns the
    predicted lower-body joint angles (12 DOF) in robot joint order.

    Requires ``env._amo_module`` to be set by a startup event.
    """
    amo_module = getattr(env, "_amo_module", None)
    if amo_module is None:
        # During _prepare_terms() the startup event hasn't run yet.
        # Return a zero tensor with the expected shape (N, 12).
        asset = env.scene[asset_cfg.name]
        return torch.zeros(asset.data.joint_pos.shape[0], 12, device=env.device)

    asset = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos  # (N, n_joints) absolute positions.

    # Map robot joint positions to checkpoint's upper_names ordering.
    upper_indices = env._amo_upper_indices  # indices into robot joint_pos
    q_upper = joint_pos[:, upper_indices].cpu().numpy()  # (N, n_upper)

    # Get rpy + height from the amo command term.
    amo_cmd = env.command_manager.get_command("amo")
    rpy_cmd = amo_cmd[:, 3:6].cpu().numpy()  # (N, 3)
    h_cmd = amo_cmd[:, 6:7].cpu().numpy()  # (N, 1)

    # Batched MLP inference on CPU.
    n_envs = joint_pos.shape[0]
    x = np.concatenate(
        [q_upper, rpy_cmd, h_cmd], axis=1
    ).astype(np.float32)  # (N, n_upper+4)
    x = (x - amo_module.in_mean) / amo_module.in_std
    x_t = torch.from_numpy(x).to(amo_module.device)
    with torch.no_grad():
        y_t = amo_module.model(x_t)
    y = y_t.cpu().numpy() * amo_module.out_std + amo_module.out_mean  # (N, n_lower)

    # Reorder from checkpoint lower_names to robot joint order.
    lower_indices = env._amo_lower_indices
    result = torch.from_numpy(y).float().to(env.device)
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
