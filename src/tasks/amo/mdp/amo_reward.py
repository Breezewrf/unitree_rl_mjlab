from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

from src.tasks.amo.mdp.amo_obs import _quat_to_euler_xyz

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def track_rpy(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Gaussian reward for tracking commanded roll, pitch, yaw.

    The commanded rpy is at indices 3:6 of the AMO command vector.
    """
    cmd = env.command_manager.get_command(command_name)
    rpy_cmd = cmd[:, 3:6]  # (N, 3)

    asset = env.scene[asset_cfg.name]
    quat_wxyz = asset.data.root_link_quat_w
    rpy_actual = _quat_to_euler_xyz(quat_wxyz)  # (N, 3)

    rpy_error = torch.sum((rpy_cmd - rpy_actual) ** 2, dim=-1)
    return torch.exp(-rpy_error / (2.0 * std ** 2))


def track_height(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Gaussian reward for tracking commanded base height.

    The commanded height is at index 6 of the AMO command vector.
    """
    cmd = env.command_manager.get_command(command_name)
    h_cmd = cmd[:, 6]  # (N,)

    asset = env.scene[asset_cfg.name]
    h_actual = asset.data.root_link_pos_w[:, 2]  # (N,)

    h_error = (h_cmd - h_actual) ** 2
    return torch.exp(-h_error / (2.0 * std ** 2))
