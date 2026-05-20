#!/usr/bin/env python
"""Interactive MLP-driven G1 visualization in MuJoCo.

Load a trained MLP checkpoint and allow real-time adjustment of torso rpy/height.
The MLP predicts lower-body joint angles from (q_upper, rpy, h).

Controls:
  W/S        pitch up/down (±0.02 rad)
  A/D        roll  left/right  (±0.02 rad)
  Q/E        yaw   left/right  (±0.02 rad)
  Up/Down    height up/down    (±0.01 m)
  R          reset rpy=0, h=0.75
  ←/→        cycle through AMASS motion frames (q_upper)
  0          use default standing pose for upper body
  Space      pause/resume AMASS auto-play
  Esc        quit

Usage:
  cd /home/breeze/workspace/unitree_rl_mjlab
  source .venv/bin/activate
  cd scripts/amo_dataset
  python interactive_mlp.py --checkpoint output/mlp_model.pt
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from pathlib import Path

import numpy as np
import mujoco

# ── Project imports ────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

try:
    import torch
    import torch.nn as nn
except ImportError:
    raise ImportError("PyTorch required. Install: pip install torch")

# ── Paths ──────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
_SCENE_XML = str(
    _PROJECT_ROOT / "src" / "assets" / "robots" / "unitree_g1"
    / "xmls" / "scene_g1.xml"
)
_LAFAN1_DIR = str(
    _SCRIPT_DIR / "assets" / "g1" / "motions" / "LAFAN1_Retargeting_Dataset"
)

# Joint names in MuJoCo order (same as play_motion.py)
_JOINT_NAMES_29DOF = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]


def _quat_from_euler(rpy: np.ndarray) -> np.ndarray:
    """euler XYZ (rad) → quaternion [qx, qy, qz, qw] (Pinocchio order)."""
    cr, sr = np.cos(rpy * 0.5), np.sin(rpy * 0.5)
    qw = cr[0] * cr[1] * cr[2] + sr[0] * sr[1] * sr[2]
    qx = sr[0] * cr[1] * cr[2] - cr[0] * sr[1] * sr[2]
    qy = cr[0] * sr[1] * cr[2] + sr[0] * cr[1] * sr[2]
    qz = cr[0] * cr[1] * sr[2] - sr[0] * sr[1] * cr[2]
    return np.array([qx, qy, qz, qw])


class MotionMLP(nn.Module):
    """MLP: same architecture as train_mlp.py."""

    def __init__(self, in_dim: int, out_dim: int,
                 hidden_dims: list[int] | None = None,
                 dropout: float = 0.1):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 512, 256]
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_checkpoint(path: str, device: torch.device):
    """Load MLP checkpoint and return (model, ckpt dict)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    hidden = ckpt["args"].get("hidden", [256, 512, 256])
    drop = ckpt["args"].get("dropout", 0.1)
    in_dim = len(ckpt["upper_names"]) + 4
    out_dim = len(ckpt["lower_names"])

    model = MotionMLP(in_dim, out_dim, hidden, drop).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"Loaded checkpoint: {path}")
    print(f"  epoch={ckpt['epoch']}  val_loss={ckpt.get('val_loss', '?')}")
    print(f"  in_dim={in_dim}  out_dim={out_dim}  hidden={hidden}")
    return model, ckpt


class AmoModule:
    """Lightweight MLP wrapper for training-time inference.

    Loads a trained checkpoint and provides batched prediction of lower-body
    joint angles from (q_upper, rpy, h).  Designed to be stored on the env
    object (``env._amo_module``) and called from observation terms.
    """

    def __init__(self, checkpoint_path: str, device: torch.device | str = "cpu"):
        self.device = torch.device(device)
        self.model, ckpt = load_checkpoint(checkpoint_path, self.device)

        self.upper_names: list[str] = list(ckpt["upper_names"])
        self.lower_names: list[str] = list(ckpt["lower_names"])

        # Normalization stats as numpy (converted from tensor if needed).
        self.in_mean = self._to_numpy(ckpt["in_mean"]).squeeze()
        self.in_std = self._to_numpy(ckpt["in_std"]).squeeze()
        self.out_mean = self._to_numpy(ckpt["out_mean"]).squeeze()
        self.out_std = self._to_numpy(ckpt["out_std"]).squeeze()

    @staticmethod
    def _to_numpy(x):
        if torch.is_tensor(x):
            return x.cpu().numpy()
        return np.asarray(x)

    def predict(
        self,
        q_upper: np.ndarray,  # (n_upper,)
        rpy: np.ndarray,      # (3,)
        h: float,
    ) -> np.ndarray:
        """Run MLP inference and return q_lower (n_lower,)."""
        x = np.concatenate([q_upper, rpy, [h]]).astype(np.float32)
        x = (x - self.in_mean) / self.in_std
        x_t = torch.from_numpy(x).unsqueeze(0).to(self.device)
        with torch.no_grad():
            y_t = self.model(x_t)
        y = y_t.cpu().numpy().squeeze()
        y = y * self.out_std + self.out_mean
        return y.astype(np.float64)


def load_amass_frames(
    upper_names: list[str],
    num_files: int = 5,
) -> dict[str, np.ndarray]:
    """Load a few AMASS motion files, extract upper-body frames.

    AMASS CSV layout: 36 columns = 7 base + 29 joints (in _JOINT_NAMES_29DOF order).
    Upper joints are a subset of the 29; we map by name.
    """
    csv_files = sorted(glob.glob(os.path.join(_LAFAN1_DIR, "*.csv")))[:num_files]
    n_upper = len(upper_names)
    frames: dict[str, np.ndarray] = {}

    # Build mapping: index in 29-dof array → index in upper_names
    name_to_29idx = {name: i for i, name in enumerate(_JOINT_NAMES_29DOF)}
    upper_29_indices = np.array(
        [name_to_29idx[n] for n in upper_names], dtype=np.int32,
    )

    for fp in csv_files:
        name = os.path.basename(fp).replace(".csv", "")
        data = np.loadtxt(fp, delimiter=",", dtype=np.float64)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        # Columns 7:36 = 29 joints
        q_29dof = data[:, 7:36]
        q_upper_seq = q_29dof[:, upper_29_indices]
        frames[name] = q_upper_seq

    print(f"Loaded {sum(v.shape[0] for v in frames.values())} frames "
          f"from {len(frames)} motion files")
    return frames


def predict_lower(
    model: MotionMLP,
    ckpt: dict,
    q_upper: np.ndarray,   # (n_upper,)
    rpy: np.ndarray,        # (3,)
    h: float,
    device: torch.device,
) -> np.ndarray:
    """MLP inference: (q_upper, rpy, h) → q_lower."""
    x = np.concatenate([q_upper, rpy, [h]]).astype(np.float32)

    # Normalize/denormalize using checkpoint statistics. The checkpoint may
    # store these as PyTorch tensors or numpy arrays; convert tensors to
    # numpy to avoid dtype/type mismatches with `x` (a numpy array).
    in_mean = ckpt["in_mean"].squeeze()
    in_std = ckpt["in_std"].squeeze()
    out_mean = ckpt["out_mean"].squeeze()
    out_std = ckpt["out_std"].squeeze()

    if torch.is_tensor(in_mean):
        in_mean = in_mean.cpu().numpy()
    if torch.is_tensor(in_std):
        in_std = in_std.cpu().numpy()
    if torch.is_tensor(out_mean):
        out_mean = out_mean.cpu().numpy()
    if torch.is_tensor(out_std):
        out_std = out_std.cpu().numpy()

    x = (x - in_mean) / in_std
    x_t = torch.from_numpy(x).unsqueeze(0).to(device)
    with torch.no_grad():
        y_t = model(x_t)
    y = y_t.cpu().numpy().squeeze()
    y = y * out_std + out_mean
    return y.astype(np.float64)


def build_qpos(
    model: mujoco.MjModel,
    rpy: np.ndarray,
    h: float,
    q_upper: np.ndarray,
    q_lower: np.ndarray,
    upper_names: list[str],
    lower_names: list[str],
) -> np.ndarray:
    """Build MuJoCo qpos from rpy/h + upper/lower joint angles."""
    qpos = np.zeros(model.nq)
    # Base: [x, y, z, qw, qx, qy, qz]
    quat = _quat_from_euler(rpy)
    qpos[0:3] = [0.0, 0.0, h]
    qpos[3] = quat[3]   # Pinocchio qw → MuJoCo qw
    qpos[4] = quat[0]   # Pinocchio qx → MuJoCo qx
    qpos[5] = quat[1]   # Pinocchio qy → MuJoCo qy
    qpos[6] = quat[2]   # Pinocchio qz → MuJoCo qz

    # Map joint names → values
    jv = {}
    for i, name in enumerate(upper_names):
        jv[name] = q_upper[i]
    for i, name in enumerate(lower_names):
        jv[name] = q_lower[i]

    for j_idx, j_name in enumerate(_JOINT_NAMES_29DOF):
        if j_name in jv:
            qpos[7 + j_idx] = jv[j_name]

    return qpos

