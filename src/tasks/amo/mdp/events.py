from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse
from .velocity_command import UniformVelocityCommand
import math
if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def apply_standing_upright_force(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    command_name: str,
    force_magnitude: float,
    asset_cfg: SceneEntityCfg,
    trunk_max_tilt_deg: float = 25.0,
    feet_body_names: tuple[str, ...] = (),
) -> None:
    """Apply upward external force to standing-task envs whose trunk is near-vertical.
 
    Mirrors HoST ``host_slope.py`` (L105-133) with two enhancements:
 
    1. **Per-env force** — if the curriculum has populated ``_force_buf`` in the
       event params, each env applies its own individually decayed force rather
       than a single shared scalar.  This directly mirrors HoST's
       ``self.force`` tensor (shape ``(num_envs, 1)``).
 
    2. **Trunk-tilt gate** — force is withheld when the torso tilts beyond
       ``trunk_max_tilt_deg`` from vertical, equivalent to HoST's
       ``projected_gravity_z < -0.8`` gate (L113).
 
    Also updates ``env._standing_old_headheight`` every step so the curriculum
    function always has fresh feet-relative head heights at reset time (mirrors
    HoST ``_reward_head_height`` L1292-1293).
 
    Args:
        env: The RL environment instance.
        env_ids: Unused – step-mode events run on all envs.
        command_name: Name of the command term (UniformVelocityCommand).
        force_magnitude: Fallback scalar force (N) used when ``_force_buf`` has
            not yet been written by the curriculum (first few steps of training).
        asset_cfg: Scene entity config; ``body_ids[0]`` is the trunk link used
            both for the tilt check and as the head-height proxy.
        trunk_max_tilt_deg: Max trunk tilt from vertical (degrees) before force
            is withheld.  Mirrors HoST's gravity-z < -0.8 threshold (~37°).
        feet_body_names: Names of the foot links used to compute feet mean height
            for the ``old_headheight`` tracker.  If empty, raw world-z of the
            trunk is stored instead (less accurate but safe as a fallback).
    """
    del env_ids  # Step-mode: runs on all envs every step.
 
    command_term = env.command_manager.get_term(command_name)
    assert command_term is not None
    command_term = cast("UniformVelocityCommand", command_term)
 
    asset = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids
    trunk_body_id = body_ids[0] if isinstance(body_ids, (list, tuple)) else body_ids
 
    num_bodies = (
        len(body_ids) if isinstance(body_ids, list) else asset.num_bodies
    )
 
    forces  = torch.zeros((env.num_envs, num_bodies, 3), device=env.device)
    torques = torch.zeros_like(forces)
 
    # ------------------------------------------------------------------
    # Step 1 – update old_headheight buffer (HoST L1292-1293).
    # Feet-relative trunk height is written every step so the curriculum
    # reads the episode's final height at reset time.
    # ------------------------------------------------------------------
    if not hasattr(env, "_standing_old_headheight"):
        env._standing_old_headheight = torch.zeros(
            (env.num_envs, 1), dtype=torch.float32, device=env.device
        )
 
    trunk_z = asset.data.body_link_pos_w[:, trunk_body_id, 2]  # (num_envs,)
 
    if feet_body_names:
        feet_ids = [
            asset.body_names.index(n) for n in feet_body_names
            if n in asset.body_names
        ]
        if feet_ids:
            feet_z      = asset.data.body_link_pos_w[:, feet_ids, 2]  # (num_envs, n)
            feet_mean_z = feet_z.mean(dim=-1)                          # (num_envs,)
        else:
            feet_mean_z = torch.zeros_like(trunk_z)
    else:
        feet_mean_z = torch.zeros_like(trunk_z)
 
    env._standing_old_headheight[:, 0] = trunk_z - feet_mean_z        # in-place
 
    # ------------------------------------------------------------------
    # Step 2 – determine per-env force magnitudes.
    # The curriculum writes env._standing_force_buf (shape (num_envs, 1))
    # mirroring HoST's self.force.  Fall back to the scalar param on the
    # first few steps before the curriculum has run for the first time.
    # NOTE: we intentionally do NOT read from event_term_cfg.params here —
    # the event manager splats params as **kwargs, so any key stored there
    # would be passed as an unexpected argument and raise a TypeError.
    # ------------------------------------------------------------------
    _force_buf: torch.Tensor | None = getattr(env, "_standing_force_buf", None)
 
    if _force_buf is not None:
        # shape (num_envs, 1) → squeeze to (num_envs,)
        per_env_force = _force_buf.squeeze(1)                          # (num_envs,)
    else:
        per_env_force = torch.full(
            (env.num_envs,), fill_value=force_magnitude,
            dtype=torch.float32, device=env.device,
        )
 
    # ------------------------------------------------------------------
    # Step 3 – build apply mask: standing AND trunk near-vertical.
    # Mirrors HoST L113: force_tensor *= (projected_gravity_z < -0.8)
    # ------------------------------------------------------------------
    is_standing = command_term.is_standing_task_env                    # (num_envs,) bool
 
    # Trunk-tilt check via projected gravity in body frame.
    body_quat_w = asset.data.body_link_quat_w[:, trunk_body_id, :]    # (num_envs, 4)
    gravity_w   = asset.data.gravity_vec_w.expand(env.num_envs, 3)    # (num_envs, 3)
    projected   = quat_apply_inverse(body_quat_w, gravity_w)          # (num_envs, 3)
 
    proj_norm   = torch.clamp(torch.norm(projected, dim=1), min=1e-6)
    xy_norm     = torch.norm(projected[:, :2], dim=1)
    xy_ratio    = xy_norm / proj_norm
 
    sin_max_tilt = math.sin(math.radians(trunk_max_tilt_deg))
    near_vertical = xy_ratio <= sin_max_tilt                           # (num_envs,) bool
 
    apply_mask    = is_standing & near_vertical                        # (num_envs,) bool
    apply_env_ids = apply_mask.nonzero(as_tuple=False).squeeze(-1)
 
    # ------------------------------------------------------------------
    # Step 4 – write per-env forces (HoST L107-108).
    # Only envs passing both gates receive a non-zero upward force; all
    # others get zeros written explicitly to clear any stale wrenches.
    # ------------------------------------------------------------------
    if len(apply_env_ids) > 0:
        forces[apply_env_ids, :, 2] = per_env_force[apply_env_ids].unsqueeze(-1)
 
    asset.write_external_wrench_to_sim(
        forces, torques, env_ids=None, body_ids=asset_cfg.body_ids
    )

def hold_upper_body_default(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
) -> None:
    """Set upper-body joint position targets to default pose (PD hold).

    Run as a step-mode event so the actuators' stiffness/damping hold the
    upper body steady at the default configuration every step.
    """
    del env_ids  # Step-mode: runs on all envs every step.

    asset = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    default_pos = asset.data.default_joint_pos[:, joint_ids]
    encoder_bias = asset.data.encoder_bias[:, joint_ids]
    target = default_pos - encoder_bias
    asset.set_joint_position_target(target, joint_ids=joint_ids)