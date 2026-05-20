"""Symmetry augmentation helpers for AMO tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from tensordict import TensorDict


def _is_lateral_or_yaw_joint(joint_name: str) -> bool:
  return ("_roll_" in joint_name) or ("_yaw_" in joint_name)


@dataclass
class _TermSlice:
  start: int
  end: int


class _SymmetryAugmentor:
  """Builds and applies left-right mirroring for actor/critic observations/actions."""

  def __init__(self, env: Any):
    self._env = env
    self._device = torch.device(env.device)
    self._joint_index_map, self._joint_sign_mask = self._build_joint_mirror()
    self._slices = self._build_term_slices()

  def _build_joint_mirror(self) -> tuple[torch.Tensor, torch.Tensor]:
    joint_action = self._env.unwrapped.action_manager.get_term("joint_pos")
    joint_names = list(joint_action.target_names)

    name_to_idx = {name: idx for idx, name in enumerate(joint_names)}
    mapped_indices: list[int] = []
    signs: list[float] = []

    for name in joint_names:
      if name.startswith("left_"):
        mirror_name = "right_" + name[len("left_") :]
      elif name.startswith("right_"):
        mirror_name = "left_" + name[len("right_") :]
      else:
        mirror_name = name

      mapped_indices.append(name_to_idx.get(mirror_name, name_to_idx[name]))
      signs.append(-1.0 if _is_lateral_or_yaw_joint(name) else 1.0)

    return (
      torch.tensor(mapped_indices, device=self._device, dtype=torch.long),
      torch.tensor(signs, device=self._device, dtype=torch.float32),
    )

  def _build_term_slices(self) -> dict[str, dict[str, _TermSlice]]:
    manager = self._env.unwrapped.observation_manager
    slices: dict[str, dict[str, _TermSlice]] = {}

    for group_name, term_names in manager.active_terms.items():
      term_dims = manager.group_obs_term_dim[group_name]
      group_slices: dict[str, _TermSlice] = {}
      offset = 0
      for term_name, term_dim in zip(term_names, term_dims, strict=False):
        flat_dim = int(torch.tensor(term_dim).prod().item())
        group_slices[term_name] = _TermSlice(offset, offset + flat_dim)
        offset += flat_dim
      slices[group_name] = group_slices

    return slices

  def _mirror_joint_like(self, tensor: torch.Tensor) -> torch.Tensor:
    return tensor[..., self._joint_index_map] * self._joint_sign_mask

  def _mirror_command(self, tensor: torch.Tensor) -> torch.Tensor:
    mirrored = tensor.clone()
    # 3D velocity: [vx, vy, vyaw] -> negate vy (1) and vyaw (2).
    if mirrored.shape[-1] >= 2:
      mirrored[..., 1] = -mirrored[..., 1]
    if mirrored.shape[-1] >= 3:
      mirrored[..., 2] = -mirrored[..., 2]
    # 7D AMO: [vx, vy, vyaw, roll, pitch, yaw, height]
    # -> negate roll (3) and yaw (5), keep pitch (4) and height (6).
    if mirrored.shape[-1] >= 4:
      mirrored[..., 3] = -mirrored[..., 3]
    if mirrored.shape[-1] >= 6:
      mirrored[..., 5] = -mirrored[..., 5]
    return mirrored

  def _mirror_phase(self, tensor: torch.Tensor) -> torch.Tensor:
    mirrored = tensor.clone()
    if mirrored.shape[-1] >= 1:
      mirrored[..., 0] = -mirrored[..., 0]
    if mirrored.shape[-1] >= 2:
      mirrored[..., 1] = -mirrored[..., 1]
    return mirrored

  def _mirror_term(self, term_name: str, values: torch.Tensor) -> torch.Tensor:
    mirrored = values.clone()

    if term_name == "base_lin_vel":
      if mirrored.shape[-1] >= 2:
        mirrored[..., 1] = -mirrored[..., 1]
      return mirrored

    if term_name == "base_ang_vel":
      if mirrored.shape[-1] >= 1:
        mirrored[..., 0] = -mirrored[..., 0]
      if mirrored.shape[-1] >= 3:
        mirrored[..., 2] = -mirrored[..., 2]
      return mirrored

    if term_name == "projected_gravity":
      if mirrored.shape[-1] >= 2:
        mirrored[..., 1] = -mirrored[..., 1]
      return mirrored

    if term_name == "command":
      return self._mirror_command(mirrored)

    if term_name == "phase":
      return self._mirror_phase(mirrored)

    if term_name in ("joint_pos", "joint_vel", "actions"):
      if mirrored.shape[-1] == self._joint_index_map.shape[0]:
        return self._mirror_joint_like(mirrored)
      return mirrored

    if term_name == "base_orientation":
      # (roll, pitch, yaw): negate roll and yaw, keep pitch.
      if mirrored.shape[-1] >= 1:
        mirrored[..., 0] = -mirrored[..., 0]
      if mirrored.shape[-1] >= 3:
        mirrored[..., 2] = -mirrored[..., 2]
      return mirrored

    if term_name == "amo_ref_lower":
      if mirrored.shape[-1] == self._joint_index_map.shape[0]:
        return self._mirror_joint_like(mirrored)
      return mirrored

    if term_name == "whole_body_actions":
      # Mirror lower-body part (first n_joints dims), leave upper-body zeros.
      n_joints = self._joint_index_map.shape[0]
      if mirrored.shape[-1] >= n_joints:
        mirrored[..., :n_joints] = self._mirror_joint_like(mirrored[..., :n_joints])
      return mirrored

    if term_name in ("foot_height", "foot_air_time", "foot_contact") and mirrored.shape[-1] >= 2:
      mirrored[..., 0], mirrored[..., 1] = values[..., 1], values[..., 0]
      return mirrored

    return mirrored

  def mirror_observations(self, obs: TensorDict) -> TensorDict:
    mirrored_obs = obs.clone()

    for group_name, group_terms in self._slices.items():
      if group_name not in mirrored_obs.keys():
        continue
      group_tensor = mirrored_obs[group_name]
      if group_tensor.ndim != 2:
        continue

      updated = group_tensor.clone()
      for term_name, term_slice in group_terms.items():
        chunk = group_tensor[:, term_slice.start : term_slice.end]
        updated[:, term_slice.start : term_slice.end] = self._mirror_term(term_name, chunk)

      mirrored_obs[group_name] = updated

    return mirrored_obs

  def augment(
    self,
    obs: TensorDict | None,
    actions: torch.Tensor | None,
  ) -> tuple[TensorDict | None, torch.Tensor | None]:
    if obs is None and actions is None:
      raise ValueError("At least one of obs/actions must be provided.")

    out_obs: TensorDict | None = None
    if obs is not None:
      mirrored_obs = self.mirror_observations(obs)
      out_obs = torch.cat((obs, mirrored_obs), dim=0)

    out_actions: torch.Tensor | None = None
    if actions is not None:
      mirrored_actions = self._mirror_joint_like(actions)
      out_actions = torch.cat((actions, mirrored_actions), dim=0)

    return out_obs, out_actions


_AUGMENTORS: dict[int, _SymmetryAugmentor] = {}


def g1_symmetry_augmentation(
  obs: TensorDict | None,
  actions: torch.Tensor | None,
  env: Any,
) -> tuple[TensorDict | None, torch.Tensor | None]:
  """Symmetry augmentation function used by RSL-RL PPO.

  This function mirrors left-right terms in observations and actions and
  concatenates mirrored samples to the original batch.
  """
  key = id(env)
  augmentor = _AUGMENTORS.get(key)
  if augmentor is None:
    augmentor = _SymmetryAugmentor(env)
    _AUGMENTORS[key] = augmentor
  return augmentor.augment(obs=obs, actions=actions)
