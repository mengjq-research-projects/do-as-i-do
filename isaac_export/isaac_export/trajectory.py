"""Build and save the explicit, simulator-independent replay trajectory."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .coordinate import euler_xyz_to_quaternion, normalize_quaternions
from .source import SourceTrajectory


@dataclass(frozen=True)
class StandardTrajectory:
    timestamps: np.ndarray
    hand_root_position: np.ndarray
    hand_root_quaternion: np.ndarray
    hand_joint_positions: np.ndarray
    object_position: np.ndarray
    object_quaternion: np.ndarray
    valid_mask: np.ndarray
    source_frame_indices: np.ndarray

    @property
    def frame_count(self) -> int:
        return int(self.timestamps.shape[0])


def build_standard_trajectory(
    source: SourceTrajectory, skip_warmup: bool
) -> StandardTrajectory:
    """Convert Retargeting qpos into the Level A canonical array protocol."""
    if not np.isfinite(source.qpos).all():
        raise ValueError("Source qpos contains NaN or Inf values.")

    qpos = source.qpos
    root_position = qpos[:, 0:3]
    root_quaternion = euler_xyz_to_quaternion(qpos[:, 3:6])
    finger_indices = [spec.qpos_start for spec in source.protocol.finger_joints]
    finger_positions = qpos[:, finger_indices]
    object_start = source.protocol.object_joint.qpos_start
    object_position = qpos[:, object_start : object_start + 3]
    object_quaternion = normalize_quaternions(
        qpos[:, object_start + 3 : object_start + 7]
    )

    source_frame_indices = np.arange(qpos.shape[0], dtype=np.int64)
    valid_mask = source_frame_indices >= source.warmup_steps
    start = source.warmup_steps if skip_warmup else 0
    if start >= qpos.shape[0]:
        raise ValueError(
            f"warmup_steps={source.warmup_steps} removes all {qpos.shape[0]} frames."
        )
    frame_count = qpos.shape[0] - start
    timestamps = np.arange(frame_count, dtype=np.float64) * source.sim_dt

    return StandardTrajectory(
        timestamps=timestamps,
        hand_root_position=np.ascontiguousarray(root_position[start:]),
        hand_root_quaternion=np.ascontiguousarray(root_quaternion[start:]),
        hand_joint_positions=np.ascontiguousarray(finger_positions[start:]),
        object_position=np.ascontiguousarray(object_position[start:]),
        object_quaternion=np.ascontiguousarray(object_quaternion[start:]),
        valid_mask=np.ascontiguousarray(valid_mask[start:]),
        source_frame_indices=np.ascontiguousarray(source_frame_indices[start:]),
    )


def save_standard_trajectory(trajectory: StandardTrajectory, path: Path) -> None:
    """Write the canonical trajectory using stable, documented array names."""
    np.savez_compressed(
        path,
        timestamps=trajectory.timestamps,
        hand_root_position=trajectory.hand_root_position,
        hand_root_quaternion=trajectory.hand_root_quaternion,
        hand_joint_positions=trajectory.hand_joint_positions,
        object_position=trajectory.object_position,
        object_quaternion=trajectory.object_quaternion,
        valid_mask=trajectory.valid_mask,
        source_frame_indices=trajectory.source_frame_indices,
    )