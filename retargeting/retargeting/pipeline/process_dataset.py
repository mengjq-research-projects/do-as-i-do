"""Process reconstruction-pipeline output into the retargeting pipeline format.

Load raw hand + object pose -> gravity-align to Z-up -> velocity-capped spike
cleaning -> write qpos trajectory. Input layout (relative to raw_dir) is written
by reconstruction/run_pipeline.sh.
"""

import glob
import json
import os
import shutil

import loguru
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from retargeting.capture_metadata import normalize_capture_metadata
from retargeting.utils.in_hand import (
    compute_in_hand_mask,
    find_freeze_indices,
    in_hand_at_endpoint,
)
from retargeting.utils.io import get_mesh_dir, get_processed_data_dir

# OpenPose 21-joint indices per finger (thumb, index, middle, ring, pinky).
# Thumb has no anatomical PIP/DIP — its CMC/MCP/IP/tip are mapped onto the same
# slots: PIP-analog = MCP (idx 2), DIP-analog = IP (idx 3).
FINGERTIP_JOINT_IDX = [4, 8, 12, 16, 20]
PIP_JOINT_IDX = [2, 6, 10, 14, 18]
DIP_JOINT_IDX = [3, 7, 11, 15, 19]

# ---------------------------------------------------------------------------
# Velocity-capped spike detection + interpolation.
#
# Fixed per-signal thresholds (derived from a dataset-wide noise analysis).
# A frame-to-frame edge is flagged
# when its velocity exceeds `min(median + K_MAD * MAD, V_CAP)`.  The edge mask
# is converted to a frame mask by OR-ing across both endpoints; short gaps are
# merged; masked regions longer than MAX_BURST are left alone (presumed real
# motion); remaining regions are interpolated (linear / SLERP).  No low-pass
# smoothing — retargeting absorbs residual jitter.
# ---------------------------------------------------------------------------
CLEAN_CONFIG = {
    # name   k_mad, v_cap (m/fr or rad/fr), window (for local median+MAD)
    "pos": dict(k_mad=8.0, v_cap=0.20, window=31),
    "rot": dict(k_mad=8.0, v_cap=0.40, window=31),
}
# Post-processing of the shared mask (after OR across all signals).
SHARED_GAP_MERGE = 1
SHARED_MAX_BURST = 10


def _finite_frame_mask(*arrays: np.ndarray) -> np.ndarray:
    """Return frames whose values are finite in every supplied array.

    HaWoR can emit NaN joints, vertices, translations, or shape coefficients
    for frames where no hand was reconstructed while still setting its
    ``*_valid`` flag to True.  Retargeting must treat numerical validity as an
    additional hard requirement before spike detection and interpolation.
    """
    if not arrays:
        raise ValueError("_finite_frame_mask requires at least one array")
    n_frames = np.asarray(arrays[0]).shape[0]
    finite = np.ones(n_frames, dtype=bool)
    for array in arrays:
        array = np.asarray(array)
        if array.shape[0] != n_frames:
            raise ValueError(
                f"Per-frame array length mismatch: expected {n_frames}, "
                f"got {array.shape[0]}"
            )
        finite &= np.isfinite(array.reshape(n_frames, -1)).all(axis=1)
    return finite


def _require_finite(name: str, array: np.ndarray) -> None:
    """Fail early with a useful message if interpolation left invalid data."""
    if np.isfinite(array).all():
        return
    bad_frames = np.flatnonzero(
        ~np.isfinite(array.reshape(array.shape[0], -1)).all(axis=1)
    )
    raise ValueError(
        f"{name} remains non-finite after cleaning at frames "
        f"{bad_frames[:20].tolist()}"
    )


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return [(start, end_exclusive), ...] for True runs in a 1-D bool mask."""
    n = len(mask)
    out = []
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def _post_process_mask(bad: np.ndarray, gap_merge: int, max_burst: int) -> np.ndarray:
    """Merge short good-runs sandwiched between bad-runs, then unmask
    bad-runs longer than ``max_burst``.  Clip boundaries act as implicit bad
    neighbors so a short good prefix/suffix adjacent to a real bad run also
    merges — otherwise the interpolation ramp leaves the wrong-valued boundary
    frames untouched and re-introduces the jump we wanted to smooth over.
    """
    out = bad.copy()
    n = len(out)
    if not out.any():
        return out
    i = 0
    while i < n:
        if not out[i]:
            j = i
            while j < n and not out[j]:
                j += 1
            if (j - i) <= gap_merge:
                out[i:j] = True
            i = j
        else:
            i += 1
    for s, e in _runs(out):
        if e - s > max_burst:
            out[s:e] = False
    return out


def _interp_positions(x: np.ndarray, bad: np.ndarray) -> np.ndarray:
    """Linear interp of ``x[bad]``; boundary bad frames clamp to nearest good."""
    if not bad.any():
        return x
    out = x.astype(np.float64, copy=True)
    good_idx = np.where(~bad)[0]
    if len(good_idx) == 0:
        return out
    bad_idx = np.where(bad)[0]
    for d in range(out.shape[-1]):
        out[bad_idx, d] = np.interp(bad_idx, good_idx, out[good_idx, d])
    return out


def _interp_rotations(
    rot: np.ndarray, bad: np.ndarray, is_quat_wxyz: bool
) -> np.ndarray:
    """SLERP masked frames; ``rot`` is (N, 3) rotvec or (N, 4) wxyz quat."""
    if not bad.any():
        return rot
    out = rot.astype(np.float64, copy=True)
    good_idx = np.where(~bad)[0]
    if len(good_idx) == 0:
        return out
    if len(good_idx) == 1:
        # A single good frame gives Slerp nothing to interpolate (it needs
        # >= 2 key rotations) — hold that orientation constant instead.
        out[bad] = rot[good_idx[0]]
        return out

    if is_quat_wxyz:
        R_good = Rotation.from_quat(out[good_idx][:, [1, 2, 3, 0]])
    else:
        R_good = Rotation.from_rotvec(out[good_idx])
    slerp = Slerp(good_idx, R_good)

    bad_idx = np.where(bad)[0]
    in_range = (bad_idx >= good_idx[0]) & (bad_idx <= good_idx[-1])
    if in_range.any():
        R_interp = slerp(bad_idx[in_range])
        if is_quat_wxyz:
            qxyzw = R_interp.as_quat()
            out[bad_idx[in_range]] = qxyzw[:, [3, 0, 1, 2]]
        else:
            out[bad_idx[in_range]] = R_interp.as_rotvec()
    for i in bad_idx[~in_range]:
        nearest = good_idx[0] if i < good_idx[0] else good_idx[-1]
        out[i] = rot[nearest]
    return out


def _extrapolate_terminal_hand_translation(
    joints: np.ndarray,
    vertices: np.ndarray,
    valid_mask: np.ndarray,
    velocity_window: int = 12,
    max_speed_per_frame: float = 0.06,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rigidly continue a hand's final observed retreat through a lost suffix.

    Monocular hand trackers commonly lose the hand as it exits the image. A
    boundary-clamped interpolation then leaves a robot hand frozen beside the
    placed object. This explicit task prior estimates robust wrist velocity
    from the final valid observations and translates the last reconstructed
    hand rigidly through the invalid suffix. It is intentionally opt-in: loss
    of tracking does not always imply retreat.

    Returns corrected joints, vertices, and the per-frame translation offsets.
    """
    joints = np.asarray(joints, dtype=np.float64)
    vertices = np.asarray(vertices, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    if joints.ndim != 3 or joints.shape[-1] != 3:
        raise ValueError("joints must have shape (N, J, 3)")
    if vertices.ndim != 3 or vertices.shape[0] != joints.shape[0] or vertices.shape[-1] != 3:
        raise ValueError("vertices must have shape (N, V, 3)")
    if valid.shape != (joints.shape[0],):
        raise ValueError("valid_mask must have shape (N,)")

    offsets = np.zeros((joints.shape[0], 3), dtype=np.float64)
    valid_indices = np.flatnonzero(valid)
    if len(valid_indices) < 3:
        return joints.copy(), vertices.copy(), offsets
    last = int(valid_indices[-1])
    if last >= joints.shape[0] - 1:
        return joints.copy(), vertices.copy(), offsets

    recent = valid_indices[-max(3, int(velocity_window)) :]
    # A tracker often repeats its last estimate for several still-marked-valid
    # frames before it finally reports loss.  A median of adjacent differences
    # then collapses to zero even though the hand was clearly retreating just
    # before the plateau.  Use a component-wise Theil-Sen slope over every
    # pair in the recent window instead; it is robust to both a short frozen
    # tail and isolated monocular-position spikes.
    pairwise_slopes = []
    for i in range(len(recent) - 1):
        for j in range(i + 1, len(recent)):
            dt = float(recent[j] - recent[i])
            if dt > 0.0:
                pairwise_slopes.append(
                    (joints[recent[j], 0, :] - joints[recent[i], 0, :]) / dt
                )
    if not pairwise_slopes:
        return joints.copy(), vertices.copy(), offsets
    velocity = np.median(np.asarray(pairwise_slopes), axis=0)
    speed = float(np.linalg.norm(velocity))
    if not np.isfinite(speed) or speed < 1e-6:
        return joints.copy(), vertices.copy(), offsets
    if speed > max_speed_per_frame:
        velocity *= max_speed_per_frame / speed

    suffix_steps = np.arange(1, joints.shape[0] - last, dtype=np.float64)
    offsets[last + 1 :] = suffix_steps[:, None] * velocity[None]
    out_joints = joints.copy()
    out_vertices = vertices.copy()
    out_joints[last + 1 :] = joints[last] + offsets[last + 1 :, None, :]
    out_vertices[last + 1 :] = vertices[last] + offsets[last + 1 :, None, :]
    return out_joints, out_vertices, offsets


def _stabilize_hand_grasp_anchor(
    joints: np.ndarray,
    vertices: np.ndarray,
    obj_trans: np.ndarray,
    obj_quat_wxyz: np.ndarray,
    anchor_local: tuple[float, float, float] | np.ndarray,
    aperture_thresh: float = 0.06,
    acquisition_dist_thresh: float = 0.04,
    fade_frames: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int] | None]:
    """Keep the thumb-index pinch point on a semantic object-local anchor.

    Hand and object monocular reconstructions can be individually correct in
    image space yet drift by several centimetres relative to one another in
    3-D.  For a known handle grasp, this opt-in prior identifies the closed
    thumb/index interval near ``anchor_local`` and rigidly translates the hand
    so the pinch midpoint stays on that anchor.  Finger articulation and the
    full object trajectory are unchanged.

    A fixed boundary correction is faded immediately before/after the grasp
    interval to avoid introducing jumps while still allowing the hand to open
    and retreat naturally after release.
    """
    joints = np.asarray(joints, dtype=np.float64)
    vertices = np.asarray(vertices, dtype=np.float64)
    obj_trans = np.asarray(obj_trans, dtype=np.float64)
    obj_quat_wxyz = np.asarray(obj_quat_wxyz, dtype=np.float64)
    anchor = np.asarray(anchor_local, dtype=np.float64)
    if joints.ndim != 3 or joints.shape[-1] != 3:
        raise ValueError("joints must have shape (N, J, 3)")
    if vertices.ndim != 3 or vertices.shape[0] != joints.shape[0] or vertices.shape[-1] != 3:
        raise ValueError("vertices must have shape (N, V, 3)")
    if obj_trans.shape != (joints.shape[0], 3):
        raise ValueError("obj_trans must have shape (N, 3)")
    if obj_quat_wxyz.shape != (joints.shape[0], 4):
        raise ValueError("obj_quat_wxyz must have shape (N, 4)")
    if anchor.shape != (3,) or not np.isfinite(anchor).all():
        raise ValueError("anchor_local must contain three finite values")

    rotations = Rotation.from_quat(obj_quat_wxyz[:, [1, 2, 3, 0]])
    thumb = joints[:, FINGERTIP_JOINT_IDX[0], :]
    index = joints[:, FINGERTIP_JOINT_IDX[1], :]
    pinch_world = 0.5 * (thumb + index)
    pinch_local = rotations.inv().apply(pinch_world - obj_trans)
    aperture = np.linalg.norm(thumb - index, axis=1)
    anchor_dist = np.linalg.norm(pinch_local - anchor[None, :], axis=1)
    acquired = (aperture < float(aperture_thresh)) & (
        anchor_dist < float(acquisition_dist_thresh)
    )
    acquired_idx = np.flatnonzero(acquired)
    offsets = np.zeros((joints.shape[0], 3), dtype=np.float64)
    if acquired_idx.size == 0:
        return joints.copy(), vertices.copy(), offsets, None

    start = int(acquired_idx[0])
    end = int(acquired_idx[-1])
    correction_local = anchor[None, :] - pinch_local
    correction_world = rotations.apply(correction_local)
    offsets[start : end + 1] = correction_world[start : end + 1]

    fade = max(0, int(fade_frames))
    for step in range(1, fade + 1):
        weight = 1.0 - step / float(fade + 1)
        before = start - step
        after = end + step
        if before >= 0:
            offsets[before] = weight * correction_world[start]
        if after < len(offsets):
            offsets[after] = weight * correction_world[end]

    out_joints = joints + offsets[:, None, :]
    out_vertices = vertices + offsets[:, None, :]
    return out_joints, out_vertices, offsets, (start, end)


def _axis_from_label(label: str) -> np.ndarray:
    """Convert a signed mesh-axis label (for example ``+y``) to XYZ."""
    if label not in {"+x", "-x", "+y", "-y", "+z", "-z"}:
        raise ValueError(
            "object_up_axis must be one of +x, -x, +y, -y, +z, -z; "
            f"got {label!r}"
        )
    axis = np.zeros(3, dtype=np.float64)
    axis[{"x": 0, "y": 1, "z": 2}[label[1]]] = 1.0 if label[0] == "+" else -1.0
    return axis


def _resolve_object_up(
    local_up_axis: str,
    local_up_vector: tuple[float, float, float] | np.ndarray | None = None,
) -> np.ndarray:
    """Resolve and normalize an exact or signed-axis semantic up vector."""
    if local_up_vector is None:
        return _axis_from_label(local_up_axis)
    vector = np.asarray(local_up_vector, dtype=np.float64)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError("object_up_vector must contain three finite values")
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        raise ValueError("object_up_vector must be non-zero")
    return vector / norm


def _project_object_quaternions_upright(
    quat_wxyz: np.ndarray,
    local_up_axis: str = "+z",
    local_up_vector: tuple[float, float, float] | np.ndarray | None = None,
) -> np.ndarray:
    """Remove object roll/pitch while preserving its horizontal heading.

    This is an explicit task prior for objects such as a mug in a pick/place
    demonstration where a known mesh-local semantic axis remains parallel to
    gravity. It must not be enabled for pouring or other tasks with intended
    object tilt. ``local_up_axis`` identifies the direction pointing out of an
    upright object's top/opening; reconstructed meshes do not necessarily use
    local +Z for that direction.

    Heading is measured using a stable local direction perpendicular to the
    semantic up axis. Returned quaternions use MuJoCo's wxyz convention and
    are sign-aligned over time for deterministic interpolation downstream.
    """
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64)
    if quat_wxyz.ndim != 2 or quat_wxyz.shape[1] != 4:
        raise ValueError(
            "quat_wxyz must have shape (N, 4); "
            f"got {quat_wxyz.shape}"
        )
    if not np.isfinite(quat_wxyz).all():
        raise ValueError("quat_wxyz contains non-finite values")

    axis = _resolve_object_up(local_up_axis, local_up_vector)
    canonical_upright, _ = Rotation.align_vectors(
        np.asarray([[0.0, 0.0, 1.0]]), axis[None]
    )
    # Pick the local direction which canonical_upright maps onto world +X.
    # It is therefore perpendicular to the requested semantic up axis and
    # remains a valid yaw reference for every supported signed axis.
    local_heading = canonical_upright.inv().apply([1.0, 0.0, 0.0])
    rotations = Rotation.from_quat(quat_wxyz[:, [1, 2, 3, 0]])
    heading_world = rotations.apply(
        np.broadcast_to(local_heading, (len(rotations), 3))
    )
    heading_norm = np.linalg.norm(heading_world[:, :2], axis=1)
    if np.any(heading_norm < 1e-8):
        raise ValueError("Object rotation has no stable horizontal heading")
    yaw = np.arctan2(heading_world[:, 1], heading_world[:, 0])
    upright_xyzw = (
        Rotation.from_euler("z", yaw[:, None]) * canonical_upright
    ).as_quat()
    upright_wxyz = upright_xyzw[:, [3, 0, 1, 2]]
    for i in range(1, len(upright_wxyz)):
        if np.dot(upright_wxyz[i], upright_wxyz[i - 1]) < 0.0:
            upright_wxyz[i] *= -1.0
    return upright_wxyz


def _rotate_points_per_frame(
    points: np.ndarray,
    origins: np.ndarray,
    rotations: Rotation,
) -> np.ndarray:
    """Apply one world rotation per frame around matching frame origins."""
    points = np.asarray(points, dtype=np.float64)
    origins = np.asarray(origins, dtype=np.float64)
    if points.ndim < 2 or points.shape[0] != origins.shape[0]:
        raise ValueError("points and origins must have matching frame counts")
    if points.shape[-1] != 3 or origins.shape != (points.shape[0], 3):
        raise ValueError("points must end in XYZ and origins must have shape (N, 3)")

    frame_count = points.shape[0]
    centered = points.reshape(frame_count, -1, 3) - origins[:, None, :]
    rotated = np.einsum("nij,nkj->nki", rotations.as_matrix(), centered)
    return (rotated + origins[:, None, :]).reshape(points.shape)


def _velocity_position(x: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.diff(x, axis=0), axis=-1)


def _velocity_rotvec(rv: np.ndarray) -> np.ndarray:
    """Angular magnitude between consecutive rotvec frames."""
    R_rel = Rotation.from_rotvec(rv[1:]) * Rotation.from_rotvec(rv[:-1]).inv()
    return R_rel.magnitude()


def _velocity_quat_wxyz(q: np.ndarray) -> np.ndarray:
    """Angular magnitude between consecutive wxyz quaternions (sign-aligned)."""
    q2 = q.copy()
    for i in range(1, len(q2)):
        if np.dot(q2[i], q2[i - 1]) < 0:
            q2[i] = -q2[i]
    dots = np.clip(np.sum(q2[:-1] * q2[1:], axis=-1), -1.0, 1.0)
    return 2.0 * np.arccos(np.abs(dots))


def _edge_to_frame_mask(edge_bad: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros(n, dtype=bool)
    out[:-1] |= edge_bad
    out[1:] |= edge_bad
    return out


def _rolling_median_mad(v: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-edge median and MAD over a centered window, clipped at the ends."""
    n = len(v)
    half = window // 2
    med = np.empty(n)
    mad = np.empty(n)
    for i in range(n):
        w = v[max(0, i - half): min(n, i + half + 1)]
        m = np.median(w)
        med[i] = m
        mad[i] = np.median(np.abs(w - m))
    return med, mad


def _detect_raw_mask(
    signal: np.ndarray,
    valid: np.ndarray | None,
    cfg: dict,
    kind: str,              # "pos" | "rotvec" | "quat_wxyz"
) -> np.ndarray:
    """Iterative per-signal spike detection; returns velocity-spike frames only.

    Gap-merge / max_burst are applied later on the OR-combined shared mask.
    `valid=False` frames seed the mask (so their garbage doesn't pollute the
    velocity/MAD passes) but are excluded from the return value: the caller
    re-applies validity afterward, so max_burst never un-flags a long invalid run.
    """
    n = len(signal)
    if n < 3:
        return np.zeros(n, dtype=bool)
    seed = np.zeros(n, dtype=bool) if valid is None else ~valid.astype(bool)
    bad = seed.copy()

    vel_fn = {"pos": _velocity_position,
              "rotvec": _velocity_rotvec,
              "quat_wxyz": _velocity_quat_wxyz}[kind]
    is_rot = kind != "pos"

    for _ in range(2):
        if bad.any():
            work = (_interp_rotations(signal, bad, kind == "quat_wxyz")
                    if is_rot else _interp_positions(signal, bad))
        else:
            work = signal
        v = vel_fn(work)
        med, mad = _rolling_median_mad(v, cfg["window"])
        thresh = np.minimum(
            med + cfg["k_mad"] * np.maximum(mad, 1e-10), cfg["v_cap"]
        )
        vel_bad = _edge_to_frame_mask(v > thresh, n)
        new_bad = bad | vel_bad
        if np.array_equal(new_bad, bad):
            break
        bad = new_bad
    return bad & ~seed


def _log_mask(mask: np.ndarray, name: str) -> None:
    n_masked = int(mask.sum())
    if n_masked == 0:
        loguru.logger.debug(f"clean[{name}]: no frames flagged")
        return
    runs = _runs(mask)
    run_lens = [e - s for s, e in runs]
    loguru.logger.info(
        f"clean[{name}]: {n_masked} frames in {len(runs)} runs "
        f"(max {max(run_lens)}, lens {run_lens})"
    )


def compute_wrist_rotation(joints: np.ndarray, is_right: bool = True) -> Rotation:
    """Compute wrist rotation(s) from hand joint positions.

    Accepts (21, 3) or (..., 21, 3); returns a matching-shape Rotation.

    Coordinate frame:
    - z: middle MCP (joint 9) -> wrist (joint 0)
    - y_aux: index MCP (joint 5) -> ring MCP (joint 13) [right hand]
             ring MCP (joint 13) -> index MCP (joint 5) [left hand]
    - x = cross(y_aux, z), y = cross(z, x)
    """
    def _norm(v: np.ndarray) -> np.ndarray:
        return v / np.linalg.norm(v, axis=-1, keepdims=True)

    z = _norm(joints[..., 9, :] - joints[..., 0, :])
    y_src = (
        joints[..., 5, :] - joints[..., 13, :]
        if is_right
        else joints[..., 13, :] - joints[..., 5, :]
    )
    y_aux = _norm(y_src)
    x = _norm(np.cross(y_aux, z))
    y = _norm(np.cross(z, x))
    return Rotation.from_matrix(np.stack([x, y, z], axis=-1))


def _identity_qpos(shape: tuple[int, ...]) -> np.ndarray:
    q = np.zeros(shape)
    q[..., 3] = 1.0
    return q


def _wrist_frame_offset(
    joints: np.ndarray, rotvec: np.ndarray, is_right: bool
) -> Rotation:
    """Fixed rotation mapping MANO canonical wrist frame to the MCP-based
    geometric frame used historically by compute_wrist_rotation.

    MCP joints are fixed in the MANO wrist frame (they are roots of finger
    chains, unaffected by hand_pose), so R_offset = R_mano^-1 @ R_geom is
    frame-invariant by construction. We average across frames via quaternion
    mean to denoise joint-position noise.
    """
    R_geom = compute_wrist_rotation(joints, is_right)
    R_mano = Rotation.from_rotvec(rotvec)
    q = (R_mano.inv() * R_geom).as_quat()  # (N, 4) xyzw
    q = q * np.sign(q @ q[0])[:, None]  # align signs before averaging
    _, eigvecs = np.linalg.eigh(q.T @ q)
    return Rotation.from_quat(eigvecs[:, -1])


def _build_side_qpos(
    joints: np.ndarray,
    fingertips: np.ndarray,
    rotvec: np.ndarray,
    world_offset: np.ndarray,
    is_right: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised (qpos_wrist, qpos_finger) for one hand side.

    Wrist orientation = MANO global_orient * fitted frame-offset, placing it
    in the same frame as compute_wrist_rotation (MCP-based).
    """
    N = joints.shape[0]
    wrist_pos = joints[:, 0, :] - world_offset
    R_off = _wrist_frame_offset(joints, rotvec, is_right)
    q_xyzw = (Rotation.from_rotvec(rotvec) * R_off).as_quat()
    qpos_wrist = np.concatenate([wrist_pos, q_xyzw[:, [3, 0, 1, 2]]], axis=-1)

    ft_pos = fingertips - world_offset
    ft_quat = np.broadcast_to([1.0, 0.0, 0.0, 0.0], (N, 5, 4))
    qpos_finger = np.concatenate([ft_pos, ft_quat], axis=-1)
    return qpos_wrist, qpos_finger


def _load_obj_verts(path: str) -> np.ndarray:
    verts = []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                verts.append([float(x) for x in parts[1:4]])
    return np.array(verts) if verts else np.zeros((0, 3))


def _scale_and_save_obj(src_path: str, scale: float, dst_path: str) -> None:
    out_lines = []
    with open(src_path) as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                xyz = [float(x) * scale for x in parts[1:4]]
                extra = parts[4:]  # vertex colors if present
                new_line = "v " + " ".join(f"{v:.8f}" for v in xyz)
                if extra:
                    new_line += " " + " ".join(extra)
                out_lines.append(new_line + "\n")
            else:
                out_lines.append(line)
    with open(dst_path, "w") as f:
        f.writelines(out_lines)


def _copy_obj_texture(src_obj: str, dst_dir: str) -> str | None:
    """Copy the OBJ's ``map_Kd`` texture image (if any) into dst_dir.

    MuJoCo ignores OBJ ``mtllib``, so this image is never loaded by the
    sim/optimizer/visualizer; only the figure renderer uses it, attaching it via
    MJCF on a throwaway spec copy at snapshot time (see ``viser_viewer``). Saved as
    ``visual_texture.<ext>`` next to ``visual.obj``. Returns the basename or None.
    """
    src_dir = os.path.dirname(src_obj)
    mtl = os.path.join(src_dir, "material.mtl")
    if not os.path.exists(mtl):
        return None
    tex_name = None
    with open(mtl) as f:
        for line in f:
            if line.lower().startswith("map_kd"):
                tex_name = line.split(maxsplit=1)[1].strip()
                break
    if not tex_name or not os.path.exists(os.path.join(src_dir, tex_name)):
        return None
    dst = os.path.join(dst_dir, "visual_texture" + os.path.splitext(tex_name)[1])
    shutil.copyfile(os.path.join(src_dir, tex_name), dst)
    return os.path.basename(dst)


def main(
    raw_dir: str = "../reconstruction/whisking",
    output_root_dir: str = "outputs",
    task: str = "whisking",
    data_id: int = 0,
    embodiment_type: str = "auto",
    dataset_name: str = "do_as_i_do",
    force: bool = False,
    start_frame: int = 0,
    object_upright: bool = False,
    object_up_axis: str = "+z",
    object_up_vector: tuple[float, float, float] | None = None,
    terminal_hand_retreat: bool = False,
    hand_grasp_anchor_vector: tuple[float, float, float] | None = None,
) -> str:
    output_root_dir = os.path.abspath(output_root_dir)
    raw_dir = os.path.abspath(raw_dir)

    if start_frame < 0:
        raise ValueError(f"start_frame must be >= 0; got {start_frame}")

    if embodiment_type == "auto":
        # anchor_hand from the raw config.json; bimanual if absent.
        config_path = os.path.join(raw_dir, "config.json")
        if os.path.exists(config_path):
            embodiment_type = json.load(open(config_path)).get("anchor_hand", "bimanual")
        else:
            embodiment_type = "bimanual"

    out_path = os.path.join(
        get_processed_data_dir(output_root_dir, dataset_name, "mano", embodiment_type, task, data_id),
        "trajectory_keypoints.npz",
    )
    if not force and os.path.exists(out_path):
        loguru.logger.info(f"Skipping process_dataset.py (output exists: {out_path})")
        return task

    process_right = embodiment_type in ["right", "bimanual"]
    process_left = embodiment_type in ["left", "bimanual"]

    # ------------------------------------------------------------------
    # 1. Load hand data
    # ------------------------------------------------------------------
    # HaWoR writes hand meshes under a subdirectory named after the video
    # (`{raw_dir}/{task}/all_hand_meshes.npz`). Fall back to a glob over
    # immediate subdirs in case the inner directory name differs.
    npz_path = f"{raw_dir}/{task}/all_hand_meshes.npz"
    if not os.path.exists(npz_path):
        candidates = sorted(glob.glob(f"{raw_dir}/*/all_hand_meshes.npz"))
        if not candidates:
            raise FileNotFoundError(f"Hand meshes NPZ not found: {npz_path}")
        if len(candidates) > 1:
            loguru.logger.warning(
                f"Multiple all_hand_meshes.npz under {raw_dir}; using {candidates[0]}"
            )
        npz_path = candidates[0]
        loguru.logger.info(f"Hand meshes NPZ resolved via glob: {npz_path}")
    meshes = np.load(npz_path)

    if process_right:
        right_joints = meshes["right_joints"].copy()      # (N, 21, 3) in camera space
        right_valid_mask = np.asarray(meshes["right_valid"], dtype=bool).copy()
        right_rot = meshes["right_rot"].copy()             # (N, 3) MANO global_orient
        right_vertices = meshes["right_vertices"].astype(np.float64)   # (N, 778, 3)
        right_faces = meshes["right_faces"].astype(np.int32)           # (1552, 3)
        # 45-D MANO finger axis-angle (joints 1..15 in wrist-local frame).
        # Invariant under world-frame gravity alignment.
        right_hand_pose = meshes["right_hand_pose"].astype(np.float64) # (N, 45)
        right_betas = meshes["right_betas"].astype(np.float64)         # (N, 10)
        N = right_joints.shape[0]
        right_finite_mask = _finite_frame_mask(
            right_joints, right_vertices, right_rot, right_hand_pose, right_betas
        )
        mislabeled = right_valid_mask & ~right_finite_mask
        if mislabeled.any():
            loguru.logger.warning(
                f"HaWoR right_valid marked {int(mislabeled.sum())} non-finite "
                "frames as valid; forcing interpolation."
            )
        right_valid_mask &= right_finite_mask

    if process_left:
        left_joints = meshes["left_joints"].copy()        # (N, 21, 3) in camera space
        left_valid_mask = np.asarray(meshes["left_valid"], dtype=bool).copy()
        left_rot = meshes["left_rot"].copy()               # (N, 3) MANO global_orient
        left_vertices = meshes["left_vertices"].astype(np.float64)     # (N, 778, 3)
        left_faces = meshes["left_faces"].astype(np.int32)             # (1552, 3)
        left_hand_pose = meshes["left_hand_pose"].astype(np.float64)   # (N, 45)
        left_betas = meshes["left_betas"].astype(np.float64)           # (N, 10)
        N = left_joints.shape[0]
        left_finite_mask = _finite_frame_mask(
            left_joints, left_vertices, left_rot, left_hand_pose, left_betas
        )
        mislabeled = left_valid_mask & ~left_finite_mask
        if mislabeled.any():
            loguru.logger.warning(
                f"HaWoR left_valid marked {int(mislabeled.sum())} non-finite "
                "frames as valid; forcing interpolation."
            )
        left_valid_mask &= left_finite_mask

    loguru.logger.info(f"Loaded hand data: {N} frames from {npz_path}")

    # ------------------------------------------------------------------
    # 1b. Sanity guard: a hand needs >= 2 valid frames to interpolate through.
    # ------------------------------------------------------------------
    if process_right:
        n_valid = int(np.asarray(right_valid_mask, dtype=bool).sum())
        if n_valid < 2:
            raise ValueError(
                f"Right hand unusable: only {n_valid}/{N} valid frames — "
                f"too few to interpolate."
            )
    if process_left:
        n_valid = int(np.asarray(left_valid_mask, dtype=bool).sum())
        if n_valid < 2:
            raise ValueError(
                f"Left hand unusable: only {n_valid}/{N} valid frames — "
                f"too few to interpolate."
            )

    # ------------------------------------------------------------------
    # 2. Load object trajectory from layout json
    # ------------------------------------------------------------------
    # The reconstruction pipeline writes the object name to config.json and the
    # tracked object poses to obj_tracking_out/{object}/.
    config_path = f"{raw_dir}/config.json"
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"config.json not found at {config_path} — raw_dir must point at a "
            "reconstruction pipeline output directory (the video's directory)."
        )
    cfg = json.load(open(config_path))
    capture_metadata = normalize_capture_metadata(cfg)
    object_name = cfg["object_names"][0]
    gpp_dir = f"{raw_dir}/obj_tracking_out/{object_name}"
    if not os.path.isdir(gpp_dir):
        raise FileNotFoundError(
            f"Cannot find object pose data at {gpp_dir} — expected the "
            "reconstruction pipeline's obj_tracking_out/{object}/ output."
        )
    loguru.logger.info(f"Object name: {object_name}, tracking dir: {gpp_dir}")
    layout_path = (
        f"{gpp_dir}/combined_visualization/layout_camera_frame_optimized.json"
    )
    layout = json.load(open(layout_path))

    # Build per-frame object arrays + obj_valid mask.  `layout["objects"]` lists
    # only the frames with an object pose; frames absent from it (or with
    # negative depth) keep an identity placeholder for the spike cleaner below
    # to interpolate through.
    obj_trans_cam = np.zeros((N, 3))
    obj_quat_cam = np.zeros((N, 4))
    obj_quat_cam[:, 0] = 1.0
    obj_valid = np.zeros(N, dtype=bool)
    for obj in layout["objects"]:
        fi = obj.get("frame_idx", obj.get("frame_index"))
        if not (0 <= fi < N):
            continue
        ls = obj["local_to_scene"]
        t = np.array(ls["translation_camera_frame"])
        q = np.array(ls["quat_wxyz_camera_frame"])
        if t[2] > 0:
            obj_trans_cam[fi] = t
            obj_quat_cam[fi] = q
            obj_valid[fi] = True
        else:
            loguru.logger.warning(f"Frame {fi}: object Z={t[2]:.4f} (negative depth) — marking invalid")

    # ------------------------------------------------------------------
    # 2b. Drop the first `start_frame` reference frames.
    # ------------------------------------------------------------------
    # All per-frame arrays are sliced in lockstep, so frame 0 below becomes the
    # new start: centering, the floor lift, the in-hand freeze window, and
    # speed resampling all then operate over the trimmed trajectory.
    if start_frame > 0:
        if start_frame >= N - 1:
            raise ValueError(
                f"start_frame={start_frame} leaves < 2 of {N} frames"
            )
        obj_trans_cam = obj_trans_cam[start_frame:]
        obj_quat_cam = obj_quat_cam[start_frame:]
        obj_valid = obj_valid[start_frame:]
        if process_right:
            right_joints = right_joints[start_frame:]
            right_valid_mask = right_valid_mask[start_frame:]
            right_rot = right_rot[start_frame:]
            right_vertices = right_vertices[start_frame:]
            right_hand_pose = right_hand_pose[start_frame:]
            right_betas = right_betas[start_frame:]
        if process_left:
            left_joints = left_joints[start_frame:]
            left_valid_mask = left_valid_mask[start_frame:]
            left_rot = left_rot[start_frame:]
            left_vertices = left_vertices[start_frame:]
            left_hand_pose = left_hand_pose[start_frame:]
            left_betas = left_betas[start_frame:]
        N -= start_frame
        loguru.logger.info(f"start_frame={start_frame}: trimmed to {N} frames")

    # Object frames needing force-interpolation (negative-depth / missing).
    # Kept separate so the shared-mask max_burst step can't later treat a long
    # run of them as real motion (see step 4 below).
    obj_invalid = ~obj_valid
    n_obj_valid = int(obj_valid.sum())
    if n_obj_valid < 2:
        raise ValueError(
            f"Object trajectory unusable: only {n_obj_valid}/{N} frames have "
            f"an object pose — too few to interpolate."
        )

    # ------------------------------------------------------------------
    # 3. Gravity alignment — rotate all camera-frame data so that Z points up
    # ------------------------------------------------------------------
    gravity_cache = f"{raw_dir}/gravity.json"
    if not os.path.exists(gravity_cache):
        raise FileNotFoundError(
            f"gravity.json not found at {gravity_cache} — it is written by the "
            "reconstruction pipeline's gravity-estimation step "
            "(reconstruction/scripts/predict_video_gravity.py). Re-run the "
            "reconstruction pipeline, or run that script on "
            f"{raw_dir}/all_frames manually."
        )

    gravity_data = json.load(open(gravity_cache))
    gravity_up = np.array(gravity_data["vec3d"])  # world-up direction in camera frame
    R_align, _ = Rotation.align_vectors([[0, 0, 1]], [gravity_up])
    loguru.logger.info(
        f"Gravity: up={gravity_up.round(3).tolist()}  "
        f"roll={gravity_data['roll_deg']:.2f}°  pitch={gravity_data['pitch_deg']:.2f}°  "
        f"→ R_align magnitude={np.degrees(R_align.magnitude()):.1f}°"
    )

    # Apply R_align to all camera-frame arrays (vectorised).
    obj_trans_cam = R_align.apply(obj_trans_cam)                              # (N, 3)
    obj_quat_xyzw = obj_quat_cam[:, [1, 2, 3, 0]]
    obj_quat_xyzw = (R_align * Rotation.from_quat(obj_quat_xyzw)).as_quat()
    obj_quat_cam = obj_quat_xyzw[:, [3, 0, 1, 2]]                            # back to wxyz

    if process_right:
        right_joints = R_align.apply(right_joints.reshape(-1, 3)).reshape(N, 21, 3)
        right_rot = (R_align * Rotation.from_rotvec(right_rot)).as_rotvec()
        right_vertices = R_align.apply(right_vertices.reshape(-1, 3)).reshape(
            N, right_vertices.shape[1], 3)

    if process_left:
        left_joints = R_align.apply(left_joints.reshape(-1, 3)).reshape(N, 21, 3)
        left_rot = (R_align * Rotation.from_rotvec(left_rot)).as_rotvec()
        left_vertices = R_align.apply(left_vertices.reshape(-1, 3)).reshape(
            N, left_vertices.shape[1], 3)

    # ------------------------------------------------------------------
    # 4. Velocity-capped spike cleaning (shared hand/object mask).
    # ------------------------------------------------------------------
    # Per-signal detection, then OR the masks so a spike in any signal flags
    # the same frame for all of them.  Hand and object stay coherent: when
    # either tracker glitches, both get interpolated through.  Thresholds come
    # from a dataset-wide noise analysis.
    per_signal_masks = []
    per_signal_masks.append((
        "obj_pos",
        _detect_raw_mask(obj_trans_cam, obj_valid, CLEAN_CONFIG["pos"], "pos"),
    ))
    per_signal_masks.append((
        "obj_rot",
        _detect_raw_mask(obj_quat_cam, obj_valid, CLEAN_CONFIG["rot"], "quat_wxyz"),
    ))
    if process_right:
        per_signal_masks.append((
            "right_wrist_pos",
            _detect_raw_mask(right_joints[:, 0, :], right_valid_mask,
                             CLEAN_CONFIG["pos"], "pos"),
        ))
        per_signal_masks.append((
            "right_wrist_rot",
            _detect_raw_mask(right_rot, right_valid_mask,
                             CLEAN_CONFIG["rot"], "rotvec"),
        ))
    if process_left:
        per_signal_masks.append((
            "left_wrist_pos",
            _detect_raw_mask(left_joints[:, 0, :], left_valid_mask,
                             CLEAN_CONFIG["pos"], "pos"),
        ))
        per_signal_masks.append((
            "left_wrist_rot",
            _detect_raw_mask(left_rot, left_valid_mask,
                             CLEAN_CONFIG["rot"], "rotvec"),
        ))
    for name, m in per_signal_masks:
        _log_mask(m, name)

    # max_burst's "long run = real motion" heuristic is only valid for
    # velocity spikes, so post-process the spike masks alone...
    shared_mask = _post_process_mask(
        np.logical_or.reduce([m for _, m in per_signal_masks]),
        gap_merge=SHARED_GAP_MERGE, max_burst=SHARED_MAX_BURST,
    )
    # ...then OR in every tracker-flagged-invalid frame: low-confidence hand
    # tracking plus missing / negative-depth object poses. These must always be
    # interpolated; keeping them out of the post-process step above stops
    # max_burst from un-flagging a long invalid run.
    invalid_mask = obj_invalid.copy()
    if process_right:
        invalid_mask = invalid_mask | ~np.asarray(right_valid_mask, dtype=bool)
    if process_left:
        invalid_mask = invalid_mask | ~np.asarray(left_valid_mask, dtype=bool)
    shared_mask = shared_mask | invalid_mask
    _log_mask(shared_mask, "shared")

    obj_trans_cam = _interp_positions(obj_trans_cam, shared_mask)
    obj_quat_cam = _interp_rotations(obj_quat_cam, shared_mask, is_quat_wxyz=True)
    if process_right:
        for j in range(right_joints.shape[1]):
            right_joints[:, j, :] = _interp_positions(right_joints[:, j, :], shared_mask)
        for v in range(right_vertices.shape[1]):
            right_vertices[:, v, :] = _interp_positions(right_vertices[:, v, :], shared_mask)
        right_rot = _interp_rotations(right_rot, shared_mask, is_quat_wxyz=False)
        right_hand_pose = right_hand_pose.reshape(N, 15, 3)
        for j in range(15):
            right_hand_pose[:, j, :] = _interp_rotations(
                right_hand_pose[:, j, :], shared_mask, is_quat_wxyz=False
            )
        right_hand_pose = right_hand_pose.reshape(N, 45)
        right_betas = _interp_positions(right_betas, shared_mask)
        right_fingertips = right_joints[:, FINGERTIP_JOINT_IDX, :]
    if process_left:
        for j in range(left_joints.shape[1]):
            left_joints[:, j, :] = _interp_positions(left_joints[:, j, :], shared_mask)
        for v in range(left_vertices.shape[1]):
            left_vertices[:, v, :] = _interp_positions(left_vertices[:, v, :], shared_mask)
        left_rot = _interp_rotations(left_rot, shared_mask, is_quat_wxyz=False)
        left_hand_pose = left_hand_pose.reshape(N, 15, 3)
        for j in range(15):
            left_hand_pose[:, j, :] = _interp_rotations(
                left_hand_pose[:, j, :], shared_mask, is_quat_wxyz=False
            )
        left_hand_pose = left_hand_pose.reshape(N, 45)
        left_betas = _interp_positions(left_betas, shared_mask)
        left_fingertips = left_joints[:, FINGERTIP_JOINT_IDX, :]

    if terminal_hand_retreat:
        for side in ("right", "left"):
            if side == "right" and process_right:
                right_joints, right_vertices, retreat_offsets = (
                    _extrapolate_terminal_hand_translation(
                        right_joints, right_vertices, right_valid_mask
                    )
                )
                right_fingertips = right_joints[:, FINGERTIP_JOINT_IDX, :]
            elif side == "left" and process_left:
                left_joints, left_vertices, retreat_offsets = (
                    _extrapolate_terminal_hand_translation(
                        left_joints, left_vertices, left_valid_mask
                    )
                )
                left_fingertips = left_joints[:, FINGERTIP_JOINT_IDX, :]
            else:
                continue
            moved = np.linalg.norm(retreat_offsets, axis=1) > 0.0
            if moved.any():
                loguru.logger.info(
                    f"Applied terminal {side}-hand retreat prior over "
                    f"{int(moved.sum())} tracker-lost frames; final rigid "
                    f"translation={retreat_offsets[-1].round(4).tolist()}"
                )

    if object_upright:
        object_rotations_before = Rotation.from_quat(
            obj_quat_cam[:, [1, 2, 3, 0]]
        )
        semantic_up = _resolve_object_up(object_up_axis, object_up_vector)
        up_before = object_rotations_before.apply(
            np.broadcast_to(semantic_up, (N, 3))
        )
        tilt_before_deg = np.degrees(
            np.arccos(
                np.clip(up_before[:, 2], -1.0, 1.0)
            )
        )
        obj_quat_cam = _project_object_quaternions_upright(
            obj_quat_cam, object_up_axis, object_up_vector
        )
        object_rotations_after = Rotation.from_quat(
            obj_quat_cam[:, [1, 2, 3, 0]]
        )
        correction = object_rotations_after * object_rotations_before.inv()
        if process_right:
            right_joints = _rotate_points_per_frame(
                right_joints, obj_trans_cam, correction
            )
            right_vertices = _rotate_points_per_frame(
                right_vertices, obj_trans_cam, correction
            )
            right_rot = (
                correction * Rotation.from_rotvec(right_rot)
            ).as_rotvec()
            right_fingertips = right_joints[:, FINGERTIP_JOINT_IDX, :]
        if process_left:
            left_joints = _rotate_points_per_frame(
                left_joints, obj_trans_cam, correction
            )
            left_vertices = _rotate_points_per_frame(
                left_vertices, obj_trans_cam, correction
            )
            left_rot = (
                correction * Rotation.from_rotvec(left_rot)
            ).as_rotvec()
            left_fingertips = left_joints[:, FINGERTIP_JOINT_IDX, :]
        loguru.logger.info(
            "Applied object-upright prior (mesh-local semantic up "
            f"{semantic_up.round(5).tolist()} "
            "→ world +Z); "
            f"removed tilt median={np.median(tilt_before_deg):.2f}°, "
            f"max={np.max(tilt_before_deg):.2f}° while preserving yaw and "
            "hand-object relative geometry"
        )

    if hand_grasp_anchor_vector is not None:
        anchor = np.asarray(hand_grasp_anchor_vector, dtype=np.float64)
        for side in ("right", "left"):
            if side == "right" and process_right:
                right_joints, right_vertices, anchor_offsets, anchor_span = (
                    _stabilize_hand_grasp_anchor(
                        right_joints,
                        right_vertices,
                        obj_trans_cam,
                        obj_quat_cam,
                        anchor,
                    )
                )
                right_fingertips = right_joints[:, FINGERTIP_JOINT_IDX, :]
            elif side == "left" and process_left:
                left_joints, left_vertices, anchor_offsets, anchor_span = (
                    _stabilize_hand_grasp_anchor(
                        left_joints,
                        left_vertices,
                        obj_trans_cam,
                        obj_quat_cam,
                        anchor,
                    )
                )
                left_fingertips = left_joints[:, FINGERTIP_JOINT_IDX, :]
            else:
                continue
            if anchor_span is None:
                loguru.logger.warning(
                    f"{side}: hand grasp anchor {anchor.round(5).tolist()} "
                    "was never acquired; reference unchanged"
                )
            else:
                moved = np.linalg.norm(anchor_offsets, axis=1)
                loguru.logger.info(
                    f"Applied {side}-hand grasp anchor "
                    f"{anchor.round(5).tolist()} over frames "
                    f"{anchor_span[0]}..{anchor_span[1]}; "
                    f"max rigid correction={moved.max():.4f} m"
                )

    # Downstream wrist-frame construction calls SciPy Rotation/SVD and cannot
    # produce a meaningful result from NaN inputs.  Keep the failure here
    # explicit in case a future producer introduces a new invalid field that
    # is not covered by the effective validity mask above.
    if process_right:
        _require_finite("right_joints", right_joints)
        _require_finite("right_vertices", right_vertices)
        _require_finite("right_rot", right_rot)
        _require_finite("right_hand_pose", right_hand_pose)
        _require_finite("right_betas", right_betas)
    if process_left:
        _require_finite("left_joints", left_joints)
        _require_finite("left_vertices", left_vertices)
        _require_finite("left_rot", left_rot)
        _require_finite("left_hand_pose", left_hand_pose)
        _require_finite("left_betas", left_betas)

    # ------------------------------------------------------------------
    # 5. Resolve object mesh and compute the world-frame shift
    # ------------------------------------------------------------------
    # The shift combines (a) centering the object's frame-0 xy at the origin
    # and (b) lifting so the lowest point of any geometry (object mesh + hand
    # vertices) over the *entire* trajectory sits at z=0. The floor in
    # the simulator is a plane at z=0, so any reference target dipping below
    # that becomes physically unreachable — checking only frame 0 (as the
    # original code did) lets later frames sneak under the floor.
    #
    # Gravity-aligned Z-up axes are preserved — NOT rotated into the object's
    # local frame — because MuJoCo's floor sits at Z=0 with gravity along -Z,
    # and rotating into the object frame would map horizontal object axes to Z.
    # Canonical (reference-frame) object mesh, emitted by the reconstruction
    # pipeline at the configured init_frame (so the frame index varies).
    mesh_candidates = sorted(glob.glob(
        f"{raw_dir}/video_segmentation/masks/frame_*_masks/{object_name}/{object_name}.obj"
    ))
    mesh_src = next((m for m in mesh_candidates if os.path.exists(m)), None)
    if mesh_src is None:
        raise FileNotFoundError(f"Cannot find object mesh: tried {mesh_candidates}")
    mesh_scale = layout["translation_scale_optimization"]["mesh_scale"]
    loguru.logger.info(f"Object mesh scale: {mesh_scale}")

    centering_offset = obj_trans_cam[0]
    verts = _load_obj_verts(mesh_src) * mesh_scale

    # Object world-frame z over all frames: for each frame i, z-component of
    # (R_i @ v) + t_i is (R_i[2, :] @ v) + t_i[2]. Vectorized over (N, V).
    obj_min_z = np.inf
    if len(verts) > 0:
        R_all = Rotation.from_quat(obj_quat_cam[:, [1, 2, 3, 0]]).as_matrix()
        z_axes = R_all[:, 2, :]  # (N, 3) — third row of each rotation
        obj_z_world = z_axes @ verts.T + obj_trans_cam[:, 2:3]  # (N, V)
        obj_min_z = float(obj_z_world.min())

    hand_min_z = np.inf
    if process_right and right_vertices.size > 0:
        hand_min_z = min(hand_min_z, float(right_vertices[..., 2].min()))
    if process_left and left_vertices.size > 0:
        hand_min_z = min(hand_min_z, float(left_vertices[..., 2].min()))

    traj_min_z = min(obj_min_z, hand_min_z)
    if not np.isfinite(traj_min_z):
        traj_min_z = float(centering_offset[2])

    world_offset = np.array(
        [float(centering_offset[0]), float(centering_offset[1]), traj_min_z]
    )
    # The reconstruction coordinates are camera-relative. After gravity
    # alignment the original camera origin is still zero, then world_offset is
    # subtracted from every trajectory point. Persist its resulting world pose
    # so ego presentation does not have to infer it from the hand trajectory.
    camera_origin_world = -world_offset
    camera_forward_world = R_align.apply(np.array([0.0, 0.0, 1.0]))
    loguru.logger.info(
        f"world_offset={world_offset.round(4)} "
        f"(centering_xy={centering_offset[:2].round(4)}, "
        f"traj_min_z={traj_min_z:.4f}, "
        f"obj_min_z={obj_min_z:.4f}, hand_min_z={hand_min_z:.4f})"
    )

    # ------------------------------------------------------------------
    # 6. Build trajectory in gravity-aligned Z-up frame, shifted by world_offset
    # ------------------------------------------------------------------
    qpos_obj = np.concatenate([obj_trans_cam - world_offset, obj_quat_cam], axis=-1)

    if process_right:
        qpos_wrist_right, qpos_finger_right = _build_side_qpos(
            right_joints, right_fingertips, right_rot, world_offset, is_right=True
        )
        qpos_pip_right = right_joints[:, PIP_JOINT_IDX, :] - world_offset
        qpos_dip_right = right_joints[:, DIP_JOINT_IDX, :] - world_offset
    else:
        qpos_wrist_right = _identity_qpos((N, 7))
        qpos_finger_right = _identity_qpos((N, 5, 7))
        qpos_pip_right = np.zeros((N, 5, 3))
        qpos_dip_right = np.zeros((N, 5, 3))

    if process_left:
        qpos_wrist_left, qpos_finger_left = _build_side_qpos(
            left_joints, left_fingertips, left_rot, world_offset, is_right=False
        )
        qpos_pip_left = left_joints[:, PIP_JOINT_IDX, :] - world_offset
        qpos_dip_left = left_joints[:, DIP_JOINT_IDX, :] - world_offset
    else:
        qpos_wrist_left = _identity_qpos((N, 7))
        qpos_finger_left = _identity_qpos((N, 5, 7))
        qpos_pip_left = np.zeros((N, 5, 3))
        qpos_dip_left = np.zeros((N, 5, 3))

    qpos_obj_right = qpos_obj.copy() if process_right else _identity_qpos((N, 7))
    qpos_obj_left = qpos_obj.copy() if process_left else _identity_qpos((N, 7))

    # Hand meshes in the same world frame as qpos (shift vertices by world_offset).
    # Unprocessed sides save empty arrays.
    if process_right:
        mano_verts_right = (right_vertices - world_offset).astype(np.float32)
        mano_faces_right = right_faces
    else:
        mano_verts_right = np.zeros((0, 0, 3), dtype=np.float32)
        mano_faces_right = np.zeros((0, 3), dtype=np.int32)
    if process_left:
        mano_verts_left = (left_vertices - world_offset).astype(np.float32)
        mano_faces_left = left_faces
    else:
        mano_verts_left = np.zeros((0, 0, 3), dtype=np.float32)
        mano_faces_left = np.zeros((0, 3), dtype=np.int32)

    # 45-D MANO finger axis-angle (joints 1..15, wrist-local frame).
    # Invariant under gravity rotation. Kept for reference/diagnostics.
    mano_finger_pose_right = (
        right_hand_pose.astype(np.float32)
        if process_right else np.zeros((0, 45), dtype=np.float32)
    )
    mano_finger_pose_left = (
        left_hand_pose.astype(np.float32)
        if process_left else np.zeros((0, 45), dtype=np.float32)
    )
    # Post-gravity-align, post-spike-clean global_orient (MANO body rotation
    # in data world frame). Kept for reference/diagnostics.
    mano_global_orient_right = (
        right_rot.astype(np.float32)
        if process_right else np.zeros((0, 3), dtype=np.float32)
    )
    mano_global_orient_left = (
        left_rot.astype(np.float32)
        if process_left else np.zeros((0, 3), dtype=np.float32)
    )
    # Per-frame MANO shape blendshape coefficients (raw from upstream predictor;
    # not modified by gravity alignment or world_offset).
    mano_betas_right = (
        right_betas.astype(np.float32)
        if process_right else np.zeros((0, 10), dtype=np.float32)
    )
    mano_betas_left = (
        left_betas.astype(np.float32)
        if process_left else np.zeros((0, 10), dtype=np.float32)
    )

    # ------------------------------------------------------------------
    # 6b. Save scaled object mesh
    # ------------------------------------------------------------------
    mesh_dir = get_mesh_dir(
        output_root_dir=output_root_dir,
        dataset_name=dataset_name,
        object_name=task,
    )
    os.makedirs(mesh_dir, exist_ok=True)
    mesh_dst = f"{mesh_dir}/visual.obj"
    _scale_and_save_obj(mesh_src, mesh_scale, mesh_dst)
    loguru.logger.info(f"Saved scaled mesh → {mesh_dst}")
    tex_name = _copy_obj_texture(mesh_src, mesh_dir)
    if tex_name:
        loguru.logger.info(f"Copied object texture → {mesh_dir}/{tex_name}")

    # ------------------------------------------------------------------
    # 6.5. Freeze object reference outside the in-hand window
    # ------------------------------------------------------------------
    # For each side+endpoint where the in-hand endpoint check fails (i.e.
    # the post-IK pedestal step will add a stabilizing pedestal there),
    # find the earliest/latest frame that passes the per-frame in-hand
    # mask and clamp the surrounding frames to that pose. Holds the
    # object static during the approach (and release) so the pedestal
    # placed under it (post-IK, in ``retargeting/pipeline/resolve_pedestal.py``) matches a
    # stable reference position.
    #
    # Uses the raw visual mesh (``verts``, mesh-local) here rather than
    # the convex-decomp output, which doesn't exist yet at this stage.
    # The in-hand mask is robust to mesh source at the 10 cm threshold.
    for side, hand_verts_side, qpos_obj_side in (
        ("right", mano_verts_right, qpos_obj_right),
        ("left",  mano_verts_left,  qpos_obj_left),
    ):
        if hand_verts_side.shape[0] == 0 or len(verts) == 0:
            continue
        needs = {}
        for ep_name, frame_idx in (("start", 0), ("end", -1)):
            in_hand, *_ = in_hand_at_endpoint(
                hand_verts_world=hand_verts_side,
                qpos_obj=qpos_obj_side,
                obj_verts=verts,
                frame=frame_idx,
            )
            needs[ep_name] = not in_hand
        if not (needs["start"] or needs["end"]):
            continue
        mask = compute_in_hand_mask(hand_verts_side, qpos_obj_side, verts)
        i_start, j_end = find_freeze_indices(mask)
        if i_start is None:
            loguru.logger.warning(
                f"{side}: no frame passes the in-hand check; "
                f"skipping reference freeze"
            )
            continue
        if needs["start"] and i_start > 0:
            qpos_obj_side[:i_start] = qpos_obj_side[i_start]
            loguru.logger.info(
                f"{side}: froze qpos_obj[0:{i_start}] = qpos_obj[{i_start}]"
            )
        if needs["end"] and j_end < qpos_obj_side.shape[0] - 1:
            qpos_obj_side[j_end + 1 :] = qpos_obj_side[j_end]
            loguru.logger.info(
                f"{side}: froze qpos_obj[{j_end + 1}:] = qpos_obj[{j_end}]"
            )

    # ------------------------------------------------------------------
    # 7. Save trajectory_keypoints.npz
    # ------------------------------------------------------------------
    out_data_dir = get_processed_data_dir(
        output_root_dir=output_root_dir,
        dataset_name=dataset_name,
        robot_type="mano",
        embodiment_type=embodiment_type,
        task=task,
        data_id=data_id,
    )
    os.makedirs(out_data_dir, exist_ok=True)

    np.savez(
        f"{out_data_dir}/trajectory_keypoints.npz",
        qpos_wrist_right=qpos_wrist_right,
        qpos_finger_right=qpos_finger_right,
        qpos_pip_right=qpos_pip_right,
        qpos_dip_right=qpos_dip_right,
        qpos_obj_right=qpos_obj_right,
        qpos_wrist_left=qpos_wrist_left,
        qpos_finger_left=qpos_finger_left,
        qpos_pip_left=qpos_pip_left,
        qpos_dip_left=qpos_dip_left,
        qpos_obj_left=qpos_obj_left,
        contact_right=np.zeros((N, 10)),
        contact_pos_right=np.zeros((10, 3)),
        contact_left=np.zeros((N, 10)),
        contact_pos_left=np.zeros((10, 3)),
        centering_offset=centering_offset,
        world_offset=world_offset,
        camera_origin_world=camera_origin_world,
        camera_forward_world=camera_forward_world,
        mano_verts_right=mano_verts_right,
        mano_faces_right=mano_faces_right,
        mano_verts_left=mano_verts_left,
        mano_faces_left=mano_faces_left,
        mano_finger_pose_right=mano_finger_pose_right,
        mano_finger_pose_left=mano_finger_pose_left,
        mano_global_orient_right=mano_global_orient_right,
        mano_global_orient_left=mano_global_orient_left,
        mano_betas_right=mano_betas_right,
        mano_betas_left=mano_betas_left,
        capture_viewpoint=np.asarray(capture_metadata["viewpoint"]),
        capture_camera_motion=np.asarray(capture_metadata["camera_motion"]),
        object_upright=np.asarray(object_upright),
        object_up_axis=np.asarray(object_up_axis),
        object_up_vector=np.asarray(
            semantic_up if object_upright else _resolve_object_up(object_up_axis)
        ),
        terminal_hand_retreat=np.asarray(terminal_hand_retreat),
        hand_grasp_anchor_vector=np.asarray(
            hand_grasp_anchor_vector
            if hand_grasp_anchor_vector is not None
            else np.empty((0,), dtype=np.float64)
        ),
    )
    loguru.logger.info(f"Saved trajectory_keypoints.npz → {out_data_dir}")

    # task_info.json lives one level above the data_id directory
    task_dir = os.path.dirname(out_data_dir)
    mesh_dir_relative = os.path.relpath(mesh_dir, output_root_dir)
    task_info = {
        "task": task,
        "dataset_name": dataset_name,
        "robot_type": "mano",
        "embodiment_type": embodiment_type,
        "data_id": data_id,
        "capture": capture_metadata,
        "object_upright": object_upright,
        "object_up_axis": object_up_axis,
        "object_up_vector": (
            semantic_up.tolist()
            if object_upright
            else _resolve_object_up(object_up_axis).tolist()
        ),
        "terminal_hand_retreat": terminal_hand_retreat,
        "hand_grasp_anchor_vector": (
            list(hand_grasp_anchor_vector)
            if hand_grasp_anchor_vector is not None
            else None
        ),
        # decompose_mesh.py prepends output_root_dir, so store a relative path.
        # For bimanual with a single shared object, only right_object_mesh_dir
        # is set (left_object_mesh_dir=None signals a shared object).
        "right_object_mesh_dir": mesh_dir_relative if process_right else None,
        "left_object_mesh_dir": mesh_dir_relative if (process_left and not process_right) else None,
    }
    task_info_path = f"{task_dir}/task_info.json"
    with open(task_info_path, "w") as f:
        json.dump(task_info, f, indent=2)
    loguru.logger.info(f"Saved task_info.json → {task_info_path}")

    return task
