from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mjlab.managers.command_manager import CommandTermCfg

from src.tasks.amo.mdp.velocity_command import (
    UniformVelocityCommand,
    UniformVelocityCommandCfg,
)

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


class AmoCommand(UniformVelocityCommand):
    """Velocity + rpy + height command for AMO task.

    Extends ``UniformVelocityCommand`` with torso orientation (roll, pitch, yaw)
    and height targets.  The ``command`` property returns a 7-D vector:
    ``[vx, vy, vyaw, roll, pitch, yaw, height]``.
    """

    cfg: AmoCommandCfg

    def __init__(self, cfg: AmoCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.rpy_command = torch.zeros(self.num_envs, 3, device=self.device)
        self.height_command = torch.zeros(self.num_envs, 1, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return torch.cat(
            [self.vel_command_b, self.rpy_command, self.height_command], dim=-1
        )

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        r = torch.empty(len(env_ids), device=self.device)
        self.rpy_command[env_ids, 0] = r.uniform_(*self.cfg.ranges.roll)
        self.rpy_command[env_ids, 1] = r.uniform_(*self.cfg.ranges.pitch)
        self.rpy_command[env_ids, 2] = r.uniform_(*self.cfg.ranges.yaw)
        self.height_command[env_ids, 0] = r.uniform_(*self.cfg.ranges.height)

    def _update_command(self) -> None:
        super()._update_command()
        # Zero rpy/height for standing envs.
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.rpy_command[standing_env_ids, :] = 0.0
        self.height_command[standing_env_ids, :] = 0.0


@dataclass(kw_only=True)
class AmoCommandCfg(UniformVelocityCommandCfg):
    """Configuration for AMO velocity + rpy + height command."""

    @dataclass
    class Ranges(UniformVelocityCommandCfg.Ranges):
        roll: tuple[float, float] = (-0.0, 0.0)
        pitch: tuple[float, float] = (-0.0, 0.0)
        yaw: tuple[float, float] = (-0.0, 0.0)
        height: tuple[float, float] = (0.78, 0.78)

    ranges: Ranges = field(default_factory=Ranges)

    def build(self, env: ManagerBasedRlEnv) -> AmoCommand:
        return AmoCommand(self, env)
