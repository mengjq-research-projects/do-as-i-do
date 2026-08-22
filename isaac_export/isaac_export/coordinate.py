"""Coordinate conversions shared by trajectory export and Isaac replay."""

from __future__ import annotations

import numpy as np


def multiply_quaternions(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return the Hamilton product of ``wxyz`` quaternions."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    lw, lx, ly, lz = np.moveaxis(left, -1, 0)
    rw, rx, ry, rz = np.moveaxis(right, -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def normalize_quaternions(quaternions: np.ndarray) -> np.ndarray:
    """Normalize ``wxyz`` quaternions and keep adjacent signs continuous."""
    quaternions = np.asarray(quaternions, dtype=np.float64)
    if quaternions.shape[-1] != 4:
        raise ValueError(
            f"Expected quaternions with a final dimension of 4, got {quaternions.shape}."
        )

    norms = np.linalg.norm(quaternions, axis=-1, keepdims=True)
    if np.any(norms <= np.finfo(np.float64).eps):
        raise ValueError("Cannot normalize a zero-length quaternion.")
    normalized = quaternions / norms

    if normalized.ndim == 1:
        return -normalized if normalized[0] < 0.0 else normalized
    if normalized.ndim != 2:
        raise ValueError(
            "Quaternion continuity currently expects one quaternion or a (T, 4) array."
        )

    normalized = normalized.copy()
    if normalized[0, 0] < 0.0:
        normalized[0] *= -1.0
    for frame in range(1, normalized.shape[0]):
        if np.dot(normalized[frame - 1], normalized[frame]) < 0.0:
            normalized[frame] *= -1.0
    return normalized


def euler_xyz_to_quaternion(euler_xyz: np.ndarray) -> np.ndarray:
    """Convert the nested Retargeting ``Rx * Ry * Rz`` chain to ``wxyz``."""
    euler_xyz = np.asarray(euler_xyz, dtype=np.float64)
    if euler_xyz.ndim != 2 or euler_xyz.shape[1] != 3:
        raise ValueError(f"Expected Euler angles shaped (T, 3), got {euler_xyz.shape}.")

    half = euler_xyz * 0.5
    cosines = np.cos(half)
    sines = np.sin(half)
    zeros = np.zeros(euler_xyz.shape[0], dtype=np.float64)

    quat_x = np.stack((cosines[:, 0], sines[:, 0], zeros, zeros), axis=-1)
    quat_y = np.stack((cosines[:, 1], zeros, sines[:, 1], zeros), axis=-1)
    quat_z = np.stack((cosines[:, 2], zeros, zeros, sines[:, 2]), axis=-1)
    return normalize_quaternions(
        multiply_quaternions(quat_x, multiply_quaternions(quat_y, quat_z))
    )


def quaternion_to_euler_xyz(quaternions: np.ndarray) -> np.ndarray:
    """Convert ``wxyz`` quaternions to the nested ``Rx * Ry * Rz`` chain.

    The returned middle angle is in ``[-pi/2, pi/2]``.  This is the inverse
    convention required by the six scalar wrist joints in the Sharpa MJCF.
    """
    quaternions = normalize_quaternions(quaternions)
    if quaternions.ndim == 1:
        quaternions = quaternions[None, :]

    w, x, y, z = np.moveaxis(quaternions, -1, 0)
    r00 = 1.0 - 2.0 * (y * y + z * z)
    r01 = 2.0 * (x * y - w * z)
    r02 = 2.0 * (x * z + w * y)
    r12 = 2.0 * (y * z - w * x)
    r22 = 1.0 - 2.0 * (x * x + y * y)

    angle_y = np.arcsin(np.clip(r02, -1.0, 1.0))
    angle_x = np.arctan2(-r12, r22)
    angle_z = np.arctan2(-r01, r00)
    return np.stack((angle_x, angle_y, angle_z), axis=-1)
