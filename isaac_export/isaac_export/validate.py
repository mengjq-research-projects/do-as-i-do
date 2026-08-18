"""Automatic checks for the canonical Level A trajectory package."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

from .source import SourceTrajectory
from .trajectory import StandardTrajectory


def _max_step_norm(values: np.ndarray) -> float:
    if values.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(values, axis=0), axis=-1).max())


def validate_trajectory(
    trajectory: StandardTrajectory,
    source: SourceTrajectory,
    position_jump_warning_m: float,
    joint_jump_warning_rad: float,
) -> dict:
    """Return a JSON-ready quality report; errors fail export, warnings do not."""
    errors: list[str] = []
    warnings: list[str] = []
    arrays = {
        "timestamps": trajectory.timestamps,
        "hand_root_position": trajectory.hand_root_position,
        "hand_root_quaternion": trajectory.hand_root_quaternion,
        "hand_joint_positions": trajectory.hand_joint_positions,
        "object_position": trajectory.object_position,
        "object_quaternion": trajectory.object_quaternion,
    }
    for name, values in arrays.items():
        if not np.isfinite(values).all():
            errors.append(f"{name} contains NaN or Inf values")

    expected_frames = trajectory.frame_count
    for name, values in arrays.items():
        if values.shape[0] != expected_frames:
            errors.append(
                f"{name} has {values.shape[0]} frames, expected {expected_frames}"
            )
    if trajectory.hand_joint_positions.shape[1:] != (
        len(source.protocol.finger_joints),
    ):
        errors.append(
            "hand_joint_positions width does not match the scene finger joint count"
        )
    if expected_frames < 1:
        errors.append("trajectory is empty")
    elif expected_frames > 1:
        timestamp_steps = np.diff(trajectory.timestamps)
        if np.any(timestamp_steps <= 0.0):
            errors.append("timestamps are not strictly increasing")
        elif not np.allclose(timestamp_steps, source.sim_dt, rtol=0.0, atol=1e-9):
            errors.append("timestamp spacing does not match sim_dt")

    quaternion_norm_errors = {}
    for name, values in (
        ("hand_root_quaternion", trajectory.hand_root_quaternion),
        ("object_quaternion", trajectory.object_quaternion),
    ):
        error = float(np.max(np.abs(np.linalg.norm(values, axis=1) - 1.0)))
        quaternion_norm_errors[name] = error
        if error > 1e-6:
            errors.append(f"{name} is not normalized (max norm error {error:.3g})")

    max_jumps = {
        "hand_root_position_m": _max_step_norm(trajectory.hand_root_position),
        "hand_joint_position_rad": _max_step_norm(
            trajectory.hand_joint_positions
        ),
        "object_position_m": _max_step_norm(trajectory.object_position),
    }
    for name in ("hand_root_position_m", "object_position_m"):
        if max_jumps[name] > position_jump_warning_m:
            warnings.append(
                f"{name} frame jump {max_jumps[name]:.6g} exceeds "
                f"{position_jump_warning_m:.6g}"
            )
    if max_jumps["hand_joint_position_rad"] > joint_jump_warning_rad:
        warnings.append(
            "hand_joint_position_rad frame jump "
            f"{max_jumps['hand_joint_position_rad']:.6g} exceeds "
            f"{joint_jump_warning_rad:.6g}"
        )

    joint_limit_violations = []
    for column, spec in enumerate(source.protocol.finger_joints):
        if spec.lower is None or spec.upper is None:
            continue
        values = trajectory.hand_joint_positions[:, column]
        below = float(max(spec.lower - float(values.min()), 0.0))
        above = float(max(float(values.max()) - spec.upper, 0.0))
        if max(below, above) > 1e-5:
            joint_limit_violations.append(
                {
                    "joint": spec.name,
                    "max_violation_rad": max(below, above),
                }
            )
    if joint_limit_violations:
        warnings.append(
            f"{len(joint_limit_violations)} finger joints exceed MuJoCo limits"
        )

    chunk_contiguous = None
    if source.chunk_end_steps and source.chunk_size is not None:
        expected_end_steps = np.arange(
            source.chunk_size,
            source.chunk_size * (len(source.chunk_end_steps) + 1),
            source.chunk_size,
        )
        chunk_contiguous = bool(
            np.array_equal(np.asarray(source.chunk_end_steps), expected_end_steps)
        )
        if not chunk_contiguous:
            warnings.append("source sim_step chunk endpoints are not contiguous")

    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "warnings": warnings,
        "checks": {
            "frame_count": expected_frames,
            "valid_frame_count": int(trajectory.valid_mask.sum()),
            "all_finite": not any("NaN or Inf" in error for error in errors),
            "quaternion_max_norm_error": quaternion_norm_errors,
            "max_frame_jumps": max_jumps,
            "joint_limit_violations": joint_limit_violations,
            "source_chunks_contiguous": chunk_contiguous,
            "hand_position_bounds_m": {
                "min": trajectory.hand_root_position.min(axis=0).tolist(),
                "max": trajectory.hand_root_position.max(axis=0).tolist(),
            },
            "object_position_bounds_m": {
                "min": trajectory.object_position.min(axis=0).tolist(),
                "max": trajectory.object_position.max(axis=0).tolist(),
            },
        },
    }