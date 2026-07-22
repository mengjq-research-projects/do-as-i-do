"""Smooth per-frame object trajectory from an optimized layout JSON.

Filters out OOD frames (jump-and-jump-back outliers) via interpolation,
then applies Savitzky-Golay filtering to the translation and rotation
(quaternion, wxyz) fields. Only those two fields are modified; every other
field in the layout is passed through unchanged.

By default this operates on the camera-frame pose produced by
`convert_layout_to_camera_frame.py` + `optimize_translation_scale.py`
(`translation_camera_frame` / `quat_wxyz_camera_frame`), which is what the
downstream visualizer/retargeter consumes. Use --trans-key / --quat-key to
smooth the pose-frame fields (`translation` / `new_quat`) instead.
"""

import argparse
import json
import copy
import os
import numpy as np
from pathlib import Path
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation, Slerp

# --- Defaults ---
WINDOW_SIZE = 7  # must be odd
POLY_ORDER = 3
# OOD detection: frame is outlier if both its incoming and outgoing velocity
# exceed this many times the median velocity
OOD_MULTIPLIER = 3.0
# Field keys (camera-frame pose is the pipeline's final consumed pose)
DEFAULT_TRANS_KEY = "translation_camera_frame"
DEFAULT_QUAT_KEY = "quat_wxyz_camera_frame"


def detect_ood_frames(values: np.ndarray, threshold_mult: float) -> list[int]:
    """Detect frames that jump out and back (OOD spikes).

    A frame i is OOD if the change from i-1->i AND i->i+1 are both
    above threshold_mult * median change.
    """
    deltas = np.linalg.norm(np.diff(values, axis=0), axis=1)
    median_delta = np.median(deltas)
    threshold = max(threshold_mult * median_delta, 1e-6)
    ood = []
    for i in range(len(deltas) - 1):
        if deltas[i] > threshold and deltas[i + 1] > threshold:
            ood.append(i + 1)  # the frame that jumped
    return ood


def interpolate_ood_translation(trans: np.ndarray, ood_frames: list[int]) -> np.ndarray:
    """Replace OOD frames with linearly interpolated values."""
    result = trans.copy()
    for i in ood_frames:
        # Find nearest non-OOD neighbors
        lo = i - 1
        while lo in ood_frames and lo > 0:
            lo -= 1
        hi = i + 1
        while hi in ood_frames and hi < len(trans) - 1:
            hi += 1
        if lo == hi:
            continue
        t = (i - lo) / (hi - lo)
        result[i] = (1 - t) * result[lo] + t * result[hi]
    return result


def interpolate_ood_rotation(quats_xyzw: np.ndarray, ood_frames: list[int]) -> np.ndarray:
    """Replace OOD frames with SLERP-interpolated rotations."""
    result = quats_xyzw.copy()
    ood_set = set(ood_frames)
    for i in ood_frames:
        lo = i - 1
        while lo in ood_set and lo > 0:
            lo -= 1
        hi = i + 1
        while hi in ood_set and hi < len(quats_xyzw) - 1:
            hi += 1
        if lo == hi:
            continue
        rots = Rotation.from_quat([result[lo], result[hi]])
        slerp = Slerp([0, 1], rots)
        t = (i - lo) / (hi - lo)
        result[i] = slerp([t]).as_quat()[0]
    return result


def smooth_trajectory(input_path: Path, output_path: Path, trans_key: str, quat_key: str):
    with open(input_path) as f:
        data = json.load(f)

    def get_frame_idx(o):
        return o.get("frame_index", o.get("frame_idx"))

    # Separate dynamic (has a frame index) and static objects
    dynamic = [o for o in data["objects"] if get_frame_idx(o) is not None]
    static = [o for o in data["objects"] if get_frame_idx(o) is None]

    # Sort by frame index
    dynamic.sort(key=get_frame_idx)
    n = len(dynamic)
    print(f"Dynamic frames: {n}, Static objects: {len(static)}")
    print(f"Smoothing fields: translation='{trans_key}', quaternion='{quat_key}'")

    # Extract arrays
    trans = np.array([o["local_to_scene"][trans_key] for o in dynamic])       # (N, 3)
    quats_wxyz = np.array([o["local_to_scene"][quat_key] for o in dynamic])   # (N, 4) wxyz

    # Convert quat wxyz -> scipy xyzw
    quats_xyzw = quats_wxyz[:, [1, 2, 3, 0]]

    # Ensure quaternion continuity
    for i in range(1, n):
        if np.dot(quats_xyzw[i], quats_xyzw[i - 1]) < 0:
            quats_xyzw[i] = -quats_xyzw[i]

    # --- Detect OOD frames ---
    rotvec = Rotation.from_quat(quats_xyzw).as_rotvec()
    ood_trans = detect_ood_frames(trans, OOD_MULTIPLIER)
    ood_rot = detect_ood_frames(rotvec, OOD_MULTIPLIER)
    ood_all = sorted(set(ood_trans + ood_rot))
    print(f"OOD translation frames: {ood_trans}")
    print(f"OOD rotation frames: {ood_rot}")
    print(f"Combined OOD frames: {ood_all}")

    # --- Replace OOD frames with interpolated values ---
    trans_clean = interpolate_ood_translation(trans, ood_all)
    quats_clean = interpolate_ood_rotation(quats_xyzw, ood_all)

    # --- Smooth translation ---
    trans_smooth = np.zeros_like(trans_clean)
    for i in range(3):
        trans_smooth[:, i] = savgol_filter(trans_clean[:, i], WINDOW_SIZE, POLY_ORDER)

    # --- Smooth quaternion directly in R^4 and renormalize ---
    # Works because sign-continuity is enforced above; no angle-wrap issues.
    quats_smooth = np.zeros_like(quats_clean)
    for i in range(4):
        quats_smooth[:, i] = savgol_filter(quats_clean[:, i], WINDOW_SIZE, POLY_ORDER)
    quats_smooth /= np.linalg.norm(quats_smooth, axis=1, keepdims=True)

    quats_xyzw_smooth = quats_smooth
    quats_wxyz_smooth = quats_xyzw_smooth[:, [3, 0, 1, 2]]

    # --- Build output (only modify the translation and quaternion fields) ---
    output = copy.deepcopy(data)
    smoothed_dynamic = copy.deepcopy(dynamic)
    for i, obj in enumerate(smoothed_dynamic):
        pose = obj["local_to_scene"]
        pose[trans_key] = trans_smooth[i].tolist()
        pose[quat_key] = quats_wxyz_smooth[i].tolist()

    output["objects"] = smoothed_dynamic + copy.deepcopy(static)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nWrote smoothed trajectory to {output_path}")

    # --- Summary statistics ---
    trans_diff = np.abs(trans_smooth - trans)
    print(f"\nTranslation smoothing (mean abs change per axis):")
    print(f"  x: {trans_diff[:, 0].mean():.6f}  y: {trans_diff[:, 1].mean():.6f}  z: {trans_diff[:, 2].mean():.6f}")
    print(f"  max change: {trans_diff.max():.6f}")

    angle_diff = [
        (Rotation.from_quat(quats_xyzw[i]).inv() * Rotation.from_quat(quats_xyzw_smooth[i])).magnitude()
        for i in range(n)
    ]
    angle_diff = np.array(angle_diff)
    print(f"\nQuaternion smoothing (angle change in radians):")
    print(f"  mean: {angle_diff.mean():.6f}  max: {angle_diff.max():.6f}")

    print(f"\nSample translations (frame: original -> smoothed):")
    for idx in [0, n // 4, n // 2, 3 * n // 4, n - 1]:
        print(f"  frame {get_frame_idx(dynamic[idx])}: "
              f"{trans[idx].tolist()} -> {trans_smooth[idx].tolist()}")


def parse_args():
    p = argparse.ArgumentParser(description="Smooth per-frame object trajectory")
    p.add_argument("--input", type=str, required=True, help="Input layout JSON path")
    p.add_argument("--output", type=str, default=None,
                   help="Output smoothed layout JSON path (default: <input>_smooth.json)")
    p.add_argument("--trans-key", type=str, default=DEFAULT_TRANS_KEY,
                   help="local_to_scene translation field to smooth "
                        f"(default: {DEFAULT_TRANS_KEY}; use 'translation' for pose frame)")
    p.add_argument("--quat-key", type=str, default=DEFAULT_QUAT_KEY,
                   help="local_to_scene quaternion (wxyz) field to smooth "
                        f"(default: {DEFAULT_QUAT_KEY}; use 'new_quat' for pose frame)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output = args.output
    if output is None:
        stem, ext = os.path.splitext(args.input)
        output = f"{stem}_smooth{ext}"
    smooth_trajectory(args.input, output, args.trans_key, args.quat_key)
