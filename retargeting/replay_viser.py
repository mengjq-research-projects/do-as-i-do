#!/usr/bin/env python3
"""replay_viser.py — replay a retargeted trajectory in a viser viewer.

Loads a generated MuJoCo scene (`scene.xml`) and an optimized trajectory
(`trajectory_mjwp.npz`) from a retargeting run directory and plays it back in the
browser — reusing the pipeline's own `retargeting.utils.viser_viewer` geometry /
posing code. This is a *view-only* replay of an existing result: it does NOT
re-run the optimization (unlike `launch.py`, whose viewer is tied to Stage 5).

Alongside the retargeted trajectory it overlays up to two reference layers,
when their files exist in/for the run directory:

1. **MANO reference** (orange) — the reconstruction-pipeline input: deforming
   MANO hand mesh + tracked object pose, from the Stage-1 output
   ``outputs/mano/<embodiment>/<task>/<id>/trajectory_keypoints.npz`` (already
   gravity-aligned and world-offset into the scene frame by process_dataset).
2. **IK reference** (transparent blue) — the kinematic solution
   ``trajectory_kinematic.npz``, drawn with viser_viewer's built-in ref-ghost
   scene copy.

Each layer has a GUI checkbox; Frame slider / Play / FPS drive all layers on a
shared timeline (the mjwp sim_dt timeline; references advance every
``ref_dt/sim_dt`` sim steps, holding frame 0 during warmup). Missing reference
files simply disable their layer — the shipped whisking demo (retargeted
trajectory only) plays back as before.

Run from the `retargeting/` directory in the `retargeting` conda env:

    conda activate retargeting

    # defaults to the whisking demo output on port 8081
    python replay_viser.py

    # or point at any run dir / explicit files / different port
    python replay_viser.py --run-dir outputs/sharpa/right/whisking/0 --port 8081
    python replay_viser.py --scene path/to/scene.xml --traj path/to/trajectory_mjwp.npz
    python replay_viser.py --no-skip-warmup        # also show the 600 warmup frames

Then open http://localhost:<port> and use the Frame slider / Play button.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import mujoco
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from retargeting.utils import viser_viewer

MANO_COLOR = (255, 150, 40)  # orange: raw reconstruction reference
MANO_OPACITY = 0.55


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--run-dir",
        default="outputs/sharpa/right/whisking/0",
        help="Run dir containing scene.xml + trajectory_mjwp.npz + config.yaml.",
    )
    p.add_argument("--scene", default=None, help="Override path to scene.xml.")
    p.add_argument("--traj", default=None, help="Override path to trajectory .npz.")
    p.add_argument("--port", type=int, default=8081, help="Viser server port.")
    p.add_argument("--fps", type=float, default=60.0, help="Initial playback FPS.")
    p.add_argument(
        "--viewpoint",
        choices=("auto", "ego", "exo"),
        default=None,
        help=(
            "Input observation viewpoint. By default read from "
            "capture_metadata.json beside the trajectory."
        ),
    )
    p.add_argument(
        "--camera-motion",
        choices=("auto", "moving", "static"),
        default=None,
        help=(
            "Input camera motion metadata. This does not synthesize missing "
            "per-frame camera extrinsics."
        ),
    )
    p.add_argument(
        "--camera-mode",
        choices=("auto", "ego", "scene", "top-down"),
        default="auto",
        help=(
            "Viewer camera preset. auto selects ego for ego input and the "
            "ordinary scene camera otherwise; top-down is a canonical world "
            "-Z view independent of the source camera."
        ),
    )
    p.add_argument(
        "--ego-fov",
        type=float,
        default=60.0,
        help="Vertical field of view in degrees for the ego camera preset.",
    )
    p.add_argument(
        "--ego-distance",
        type=float,
        default=0.0,
        help=(
            "Distance in metres from the ego camera to the trajectory centre. "
            "Values <= 0 select a distance automatically."
        ),
    )
    p.add_argument(
        "--display-object-upright",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Project the displayed object's semantic up axis onto world +Z "
            "and apply the same rigid correction to the hand root, preserving "
            "hand-object relative geometry. Disabled by default; production "
            "upright correction belongs upstream."
        ),
    )
    p.add_argument(
        "--display-object-up-axis",
        choices=("+x", "-x", "+y", "-y", "+z", "-z"),
        default="+z",
        help=(
            "Mesh-local direction that points out of the object's top/opening. "
            "For this reconstructed mug use +z. Only affects upright display."
        ),
    )
    p.add_argument(
        "--display-object-up-vector",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help=(
            "Exact mesh-local semantic up vector. Overrides "
            "--display-object-up-axis; useful when reconstruction leaves the "
            "mesh geometry tilted relative to its coordinate axes."
        ),
    )
    p.add_argument(
        "--display-hand-level",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Diagnostic-only rotation of the displayed hand/reference scene "
            "toward the world XY plane. This changes its pose independently "
            "of the object and must not be used to judge contact. Disabled by "
            "default."
        ),
    )
    p.add_argument(
        "--skip-warmup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop the leading warmup frames (default: skip).",
    )
    return p.parse_args()


def _load_capture_metadata(run_dir: Path, keypoints_path: Path | None) -> dict[str, str]:
    """Load capture metadata with compatibility for older processed runs."""
    metadata = {"viewpoint": "auto", "camera_motion": "auto"}
    sidecar = run_dir / "capture_metadata.json"
    if sidecar.is_file():
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
            for key in metadata:
                value = payload.get(key)
                if isinstance(value, str):
                    metadata[key] = value
        except (OSError, json.JSONDecodeError):
            pass

    if keypoints_path is not None and keypoints_path.is_file():
        try:
            with np.load(str(keypoints_path), allow_pickle=False) as data:
                for key, npz_key in (
                    ("viewpoint", "capture_viewpoint"),
                    ("camera_motion", "capture_camera_motion"),
                ):
                    if metadata[key] == "auto" and npz_key in data.files:
                        metadata[key] = str(np.asarray(data[npz_key]).item())
        except (OSError, ValueError):
            pass
    if metadata["viewpoint"] not in {"auto", "ego", "exo"}:
        metadata["viewpoint"] = "auto"
    if metadata["camera_motion"] not in {"auto", "moving", "static"}:
        metadata["camera_motion"] = "auto"
    return metadata


def _ego_camera_preset(
    qpos: np.ndarray,
    start: int,
    keypoints_path: Path | None,
    requested_distance: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a stable, downward-looking ego presentation camera.

    Reconstruction has no per-frame camera extrinsics.  The keypoint package
    does retain the gravity-aligned optical axis, however, so preserve its
    horizontal heading while placing a stabilized camera above the combined
    hand/object trajectory.  ``camera_origin_world`` is intentionally not used:
    its height is tied to the arbitrary floor lift and can therefore land below
    the Retargeting scene.
    """
    roots = np.asarray(qpos[start:, :3], dtype=np.float64)
    roots = roots[np.all(np.isfinite(roots), axis=1)]
    focus_groups = [roots] if len(roots) else []
    forward: np.ndarray | None = None

    if keypoints_path is not None and keypoints_path.is_file():
        try:
            with np.load(str(keypoints_path), allow_pickle=False) as data:
                if "camera_forward_world" in data.files:
                    candidate = np.asarray(
                        data["camera_forward_world"], dtype=np.float64
                    )
                    if candidate.shape == (3,) and np.all(np.isfinite(candidate)):
                        forward = candidate
                for key in (
                    "qpos_wrist_right",
                    "qpos_wrist_left",
                    "qpos_obj_right",
                    "qpos_obj_left",
                ):
                    if key not in data.files:
                        continue
                    xyz = np.asarray(data[key], dtype=np.float64)
                    if xyz.ndim != 2 or xyz.shape[1] < 3:
                        continue
                    xyz = xyz[:, :3]
                    xyz = xyz[np.all(np.isfinite(xyz), axis=1)]
                    if len(xyz) and np.any(np.linalg.norm(xyz, axis=1) > 1e-6):
                        focus_groups.append(xyz)
        except (OSError, ValueError):
            pass

    if focus_groups:
        focus = np.concatenate(focus_groups, axis=0)
        lower = np.percentile(focus, 5.0, axis=0)
        upper = np.percentile(focus, 95.0, axis=0)
        target = (lower + upper) * 0.5
        span = float(np.linalg.norm(upper - lower))
    else:
        target = np.zeros(3, dtype=np.float64)
        span = 0.45

    if forward is None or not np.all(np.isfinite(forward)):
        forward = np.array([0.0, 1.0, -0.65], dtype=np.float64)
    else:
        forward = forward.copy()
        # GeoCalib's gravity convention can leave the saved optical axis with
        # the correct table heading but an upward Z sign. Ego manipulation
        # footage must look down towards the work surface.
        forward[2] = -abs(float(forward[2]))
        if abs(float(forward[2])) < 0.25:
            forward[2] = -0.45
    forward_norm = float(np.linalg.norm(forward))
    if not np.isfinite(forward_norm) or forward_norm < 1e-8:
        forward = np.array([0.0, 1.0, -0.65], dtype=np.float64)
        forward_norm = float(np.linalg.norm(forward))
    forward /= forward_norm

    if np.isfinite(requested_distance) and requested_distance > 0.0:
        distance = float(requested_distance)
    else:
        distance = float(np.clip(1.6 * max(span, 0.40), 0.65, 1.25))
    origin = target - forward * distance

    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(forward, world_up)
    right_norm = float(np.linalg.norm(right))
    if right_norm < 1e-8:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        right /= right_norm
    camera_up = np.cross(right, forward)
    camera_up /= np.linalg.norm(camera_up)
    return origin, target, camera_up


def _top_down_camera_preset(
    qpos: np.ndarray,
    start: int,
    keypoints_path: Path | None,
    *,
    vertical_fov_deg: float,
    aspect_ratio: float = 16.0 / 9.0,
    margin: float = 1.20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Frame the hand/object path with a canonical world -Z camera.

    The orientation is deterministic, while the look-at point and height are
    fitted to robust trajectory bounds. This deliberately does not pretend to
    recover unknown source-camera intrinsics or extrinsics.
    """
    focus_groups: list[np.ndarray] = []
    roots = np.asarray(qpos[start:, :3], dtype=np.float64)
    roots = roots[np.all(np.isfinite(roots), axis=1)]
    if len(roots):
        focus_groups.append(roots)

    if keypoints_path is not None and keypoints_path.is_file():
        try:
            with np.load(str(keypoints_path), allow_pickle=False) as data:
                for key in (
                    "qpos_wrist_right",
                    "qpos_wrist_left",
                    "qpos_obj_right",
                    "qpos_obj_left",
                ):
                    if key not in data.files:
                        continue
                    xyz = np.asarray(data[key], dtype=np.float64)
                    if xyz.ndim != 2 or xyz.shape[1] < 3:
                        continue
                    xyz = xyz[:, :3]
                    xyz = xyz[np.all(np.isfinite(xyz), axis=1)]
                    if len(xyz) and np.any(np.linalg.norm(xyz, axis=1) > 1e-6):
                        focus_groups.append(xyz)
        except (OSError, ValueError):
            pass

    if focus_groups:
        focus = np.concatenate(focus_groups, axis=0)
        lower = np.percentile(focus, 5.0, axis=0)
        upper = np.percentile(focus, 95.0, axis=0)
    else:
        lower = np.array([-0.25, -0.25, 0.0], dtype=np.float64)
        upper = np.array([0.25, 0.25, 0.2], dtype=np.float64)

    target = (lower + upper) * 0.5
    half_x = max(0.5 * float(upper[0] - lower[0]), 0.20)
    half_y = max(0.5 * float(upper[1] - lower[1]), 0.20)
    fov = float(np.clip(vertical_fov_deg, 10.0, 120.0))
    fit_half_height = max(half_y, half_x / max(aspect_ratio, 1e-6)) * margin
    distance = fit_half_height / np.tan(0.5 * np.deg2rad(fov))
    distance = max(float(distance), 0.65)
    origin = target + np.array([0.0, 0.0, distance], dtype=np.float64)
    camera_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    return origin, target, camera_up


def _load_run_config(config_yaml: Path) -> dict:
    """Best-effort read of the run's saved config.yaml (warmup_steps, sim_dt, ref_dt)."""
    if not config_yaml.exists():
        return {}
    try:
        from omegaconf import OmegaConf

        return dict(OmegaConf.load(str(config_yaml)))
    except Exception:
        return {}


def load_qpos(traj_path: Path, model_nq: int) -> np.ndarray:
    """Load + flatten the optimized qpos to (T, nq), ordered by sim step.

    `trajectory_mjwp.npz` stores qpos chunked as (n_chunks, steps_per_chunk, nq);
    chunks are sequential executed segments, so we order by `sim_step` and flatten.
    """
    d = np.load(str(traj_path), allow_pickle=True)
    if "qpos" not in d.files:
        raise SystemExit(f"'qpos' not in {traj_path} (keys: {list(d.files)})")
    qpos = np.asarray(d["qpos"])
    if qpos.ndim == 3:
        if "sim_step" in d.files and len(d["sim_step"]) == qpos.shape[0]:
            qpos = qpos[np.argsort(np.asarray(d["sim_step"]).ravel())]
        qpos = qpos.reshape(-1, qpos.shape[-1])
    if qpos.shape[-1] != model_nq:
        raise SystemExit(
            f"qpos width {qpos.shape[-1]} != model.nq {model_nq} — "
            "scene.xml and trajectory .npz are from different runs?"
        )
    return np.ascontiguousarray(qpos, dtype=np.float64)


def _project_object_qpos_upright(
    qpos: np.ndarray,
    object_qpos_addresses: list[int],
    local_up_axis: np.ndarray | None = None,
) -> np.ndarray:
    """Make a mesh-local semantic up axis vertical while preserving heading."""
    out = np.asarray(qpos, dtype=np.float64).copy()
    axis = np.asarray(
        [0.0, 0.0, 1.0] if local_up_axis is None else local_up_axis,
        dtype=np.float64,
    )
    if axis.shape != (3,) or not np.isfinite(axis).all():
        raise ValueError("local_up_axis must be a finite XYZ vector")
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-8:
        raise ValueError("local_up_axis must be non-zero")
    axis /= axis_norm
    canonical_upright, _ = Rotation.align_vectors(
        np.asarray([[0.0, 0.0, 1.0]]), axis[None]
    )
    local_heading = canonical_upright.inv().apply([1.0, 0.0, 0.0])
    for qadr in object_qpos_addresses:
        quat_wxyz = out[:, qadr + 3 : qadr + 7]
        rotations = Rotation.from_quat(quat_wxyz[:, [1, 2, 3, 0]])
        heading_world = rotations.apply(
            np.broadcast_to(local_heading, (len(rotations), 3))
        )
        yaw = np.arctan2(heading_world[:, 1], heading_world[:, 0])
        upright_rotation = (
            Rotation.from_euler("z", yaw[:, None]) * canonical_upright
        )
        upright_xyzw = upright_rotation.as_quat()
        upright_wxyz = upright_xyzw[:, [3, 0, 1, 2]]
        for i in range(1, len(upright_wxyz)):
            if np.dot(upright_wxyz[i], upright_wxyz[i - 1]) < 0.0:
                upright_wxyz[i] *= -1.0
        out[:, qadr + 3 : qadr + 7] = upright_wxyz
    return out


def _project_scene_qpos_upright(
    qpos: np.ndarray,
    object_qpos_addresses: list[int],
    hand_root_addresses: list[tuple[str, int, int]],
    local_up_axis: np.ndarray | None = None,
) -> np.ndarray:
    """Upright an object and rigidly carry robot hand roots with it."""
    if not object_qpos_addresses:
        return np.asarray(qpos, dtype=np.float64).copy()
    source = np.asarray(qpos, dtype=np.float64)
    out = _project_object_qpos_upright(
        source, object_qpos_addresses, local_up_axis
    )
    object_qadr = object_qpos_addresses[0]
    origins = source[:, object_qadr : object_qadr + 3]
    before_wxyz = source[:, object_qadr + 3 : object_qadr + 7]
    after_wxyz = out[:, object_qadr + 3 : object_qadr + 7]
    before = Rotation.from_quat(before_wxyz[:, [1, 2, 3, 0]])
    after = Rotation.from_quat(after_wxyz[:, [1, 2, 3, 0]])
    correction = after * before.inv()

    for _side, pos_qadr, rot_qadr in hand_root_addresses:
        relative_position = source[:, pos_qadr : pos_qadr + 3] - origins
        out[:, pos_qadr : pos_qadr + 3] = origins + correction.apply(
            relative_position
        )
        root_rotation = Rotation.from_euler(
            "XYZ", source[:, rot_qadr : rot_qadr + 3]
        )
        angles = (correction * root_rotation).as_euler("XYZ")
        out[:, rot_qadr : rot_qadr + 3] = np.unwrap(angles, axis=0)
    return out


def _axis_from_label(label: str) -> np.ndarray:
    sign = 1.0 if label[0] == "+" else -1.0
    axis = np.zeros(3, dtype=np.float64)
    axis[{"x": 0, "y": 1, "z": 2}[label[1]]] = sign
    return axis


def _rotations_to_horizontal(vectors: np.ndarray) -> tuple[Rotation, np.ndarray]:
    """Return shortest rotations that place direction vectors in world XY."""
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError(f"vectors must have shape (N, 3); got {vectors.shape}")
    lengths = np.linalg.norm(vectors, axis=1)
    if np.any(lengths < 1e-8) or not np.isfinite(vectors).all():
        raise ValueError("vectors must be finite and non-zero")
    directions = vectors / lengths[:, None]
    horizontal = directions.copy()
    horizontal[:, 2] = 0.0
    horizontal_lengths = np.linalg.norm(horizontal, axis=1)
    if np.any(horizontal_lengths < 1e-8):
        raise ValueError("cannot preserve heading for a vertical direction")
    horizontal /= horizontal_lengths[:, None]

    axes = np.cross(directions, horizontal)
    axis_lengths = np.linalg.norm(axes, axis=1)
    dots = np.sum(directions * horizontal, axis=1)
    angles = np.arccos(np.clip(dots, -1.0, 1.0))
    rotvecs = np.zeros_like(vectors)
    moving = axis_lengths > 1e-10
    rotvecs[moving] = (
        axes[moving] / axis_lengths[moving, None] * angles[moving, None]
    )
    elevation_deg = np.degrees(np.arcsin(np.clip(np.abs(directions[:, 2]), 0.0, 1.0)))
    return Rotation.from_rotvec(rotvecs), elevation_deg


def _hand_root_qpos_addresses(
    model: mujoco.MjModel,
) -> list[tuple[str, int, int]]:
    """Return (side, translation qadr, intrinsic-XYZ rotation qadr)."""
    roots: list[tuple[str, int, int]] = []
    for side in ("right", "left"):
        names = [
            f"{side}_pos_x",
            f"{side}_pos_y",
            f"{side}_pos_z",
            f"{side}_rot_x",
            f"{side}_rot_y",
            f"{side}_rot_z",
        ]
        ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in names
        ]
        if any(joint_id < 0 for joint_id in ids):
            continue
        addresses = [int(model.jnt_qposadr[joint_id]) for joint_id in ids]
        if addresses == list(range(addresses[0], addresses[0] + 6)):
            roots.append((side, addresses[0], addresses[3]))
    return roots


def _level_robot_fingers(
    qpos: np.ndarray,
    model: mujoco.MjModel,
    hand_root_addresses: list[tuple[str, int, int]],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Level each robot hand's wrist-to-middle-tip direction per frame."""
    out = np.asarray(qpos, dtype=np.float64).copy()
    elevations: dict[str, np.ndarray] = {}
    for side, _pos_qadr, rot_qadr in hand_root_addresses:
        base_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_hand_C_MC"
        )
        tip_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_middle_tip"
        )
        if base_id < 0 or tip_id < 0:
            continue
        data = mujoco.MjData(model)
        directions = np.empty((len(out), 3), dtype=np.float64)
        for frame, pose in enumerate(out):
            data.qpos[:] = pose
            mujoco.mj_kinematics(model, data)
            directions[frame] = data.site_xpos[tip_id] - data.xpos[base_id]
        correction, elevation_deg = _rotations_to_horizontal(directions)
        root_rotation = Rotation.from_euler("XYZ", out[:, rot_qadr : rot_qadr + 3])
        angles = (correction * root_rotation).as_euler("XYZ")
        out[:, rot_qadr : rot_qadr + 3] = np.unwrap(angles, axis=0)
        elevations[side] = elevation_deg
    return out, elevations


def _object_freejoint_qpos_addresses(model: mujoco.MjModel) -> list[int]:
    """Find free-joint qpos addresses for tracked object bodies."""
    addresses: list[int] = []
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            continue
        body_id = int(model.jnt_bodyid[joint_id])
        body_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, body_id
        )
        if body_name and body_name.endswith("_object"):
            addresses.append(int(model.jnt_qposadr[joint_id]))
    return addresses


def _keypoints_request_upright_display(keypoints_path: Path | None) -> bool:
    if keypoints_path is None or not keypoints_path.is_file():
        return False
    try:
        with np.load(str(keypoints_path), allow_pickle=False) as data:
            if "object_upright" not in data.files:
                return False
            return bool(np.asarray(data["object_upright"]).item())
    except (OSError, ValueError):
        return False


class ManoOverlay:
    """Orange MANO hand mesh(es) + tracked-object mesh from trajectory_keypoints.npz.

    Hand vertices deform per frame, so the hand mesh is re-uploaded on frame
    change (778 verts — cheap). The object mesh is rigid: uploaded once, posed
    via handle.position/wxyz.
    """

    def __init__(
        self,
        keypoints_path: Path,
        outputs_root: Path,
        task: str,
        level_fingers: bool = False,
        object_upright: bool = False,
        object_up_axis: np.ndarray | None = None,
    ):
        self.hands: dict[str, tuple[np.ndarray, np.ndarray]] = {}  # side -> (verts, faces)
        self.obj_qpos: np.ndarray | None = None
        self.obj_mesh: trimesh.Trimesh | None = None
        self.n_frames = 0
        self.visible = True
        self._hand_handles: dict[str, object] = {}
        self._hand_frame: dict[str, int] = {}
        self._obj_handle = None

        if not keypoints_path.exists():
            print(f"MANO reference not found ({keypoints_path}) — layer disabled.")
            return

        d = np.load(str(keypoints_path))
        for side in ("right", "left"):
            v, f = f"mano_verts_{side}", f"mano_faces_{side}"
            if v in d.files and d[v].size > 0:
                self.hands[side] = (
                    d[v].astype(np.float32),
                    d[f].astype(np.uint32),
                )
                self.n_frames = max(self.n_frames, d[v].shape[0])

        # Tracked object pose: sides carry identical copies; prefer a side we render.
        for side in (*self.hands.keys(), "right", "left"):
            key = f"qpos_obj_{side}"
            if key in d.files and d[key].size > 0:
                self.obj_qpos = d[key].astype(np.float64)
                self.n_frames = max(self.n_frames, self.obj_qpos.shape[0])
                break

        if level_fingers:
            for side, (vertices, faces) in tuple(self.hands.items()):
                wrist_key = f"qpos_wrist_{side}"
                fingertip_key = f"qpos_finger_{side}"
                if wrist_key not in d.files or fingertip_key not in d.files:
                    continue
                wrists = np.asarray(d[wrist_key][:, :3], dtype=np.float64)
                middle_tips = np.asarray(
                    d[fingertip_key][:, 2, :3], dtype=np.float64
                )
                correction, _ = _rotations_to_horizontal(middle_tips - wrists)
                origins = wrists[: vertices.shape[0], None, :]
                relative = vertices - origins
                leveled = origins + np.einsum(
                    "tij,tvj->tvi",
                    correction.as_matrix()[: vertices.shape[0]],
                    relative,
                )
                self.hands[side] = (leveled.astype(np.float32), faces)
        if object_upright and self.obj_qpos is not None:
            source_obj_qpos = self.obj_qpos.copy()
            self.obj_qpos = _project_object_qpos_upright(
                source_obj_qpos, [0], object_up_axis
            )
            before = Rotation.from_quat(source_obj_qpos[:, [4, 5, 6, 3]])
            after = Rotation.from_quat(self.obj_qpos[:, [4, 5, 6, 3]])
            correction = after * before.inv()
            origins = source_obj_qpos[:, None, :3]
            for side, (vertices, faces) in tuple(self.hands.items()):
                n = min(vertices.shape[0], source_obj_qpos.shape[0])
                corrected = vertices.copy()
                corrected[:n] = origins[:n] + np.einsum(
                    "tij,tvj->tvi",
                    correction.as_matrix()[:n],
                    vertices[:n] - origins[:n],
                )
                self.hands[side] = (corrected.astype(np.float32), faces)

        mesh_path = outputs_root / "assets" / "objects" / task / "visual.obj"
        if self.obj_qpos is not None and mesh_path.exists():
            try:
                m = trimesh.load_mesh(str(mesh_path))
                if not isinstance(m, trimesh.Trimesh):
                    m = m.dump(concatenate=True)
                self.obj_mesh = m
            except Exception as e:
                print(f"Could not load object mesh {mesh_path}: {e}")

    @property
    def available(self) -> bool:
        return bool(self.hands) or self.obj_qpos is not None

    def build(self, server) -> None:
        self._server = server
        if self.obj_mesh is not None:
            self._obj_handle = server.scene.add_mesh_simple(
                "/mano_ref/object",
                vertices=np.asarray(self.obj_mesh.vertices, dtype=np.float32),
                faces=np.asarray(self.obj_mesh.faces, dtype=np.uint32),
                color=MANO_COLOR,
                opacity=MANO_OPACITY,
            )

    def show_frame(self, frame_idx: int) -> None:
        if not self.available or not self.visible:
            return
        fi = max(0, min(self.n_frames - 1, int(frame_idx)))
        for side, (verts, faces) in self.hands.items():
            f = min(fi, verts.shape[0] - 1)
            if self._hand_frame.get(side) == f:
                continue
            # Deforming mesh: re-add at the same path replaces the old handle.
            self._hand_handles[side] = self._server.scene.add_mesh_simple(
                f"/mano_ref/hand_{side}",
                vertices=verts[f],
                faces=faces,
                color=MANO_COLOR,
                opacity=MANO_OPACITY,
            )
            self._hand_frame[side] = f
        if self._obj_handle is not None and self.obj_qpos is not None:
            q = self.obj_qpos[min(fi, self.obj_qpos.shape[0] - 1)]
            self._obj_handle.position = tuple(q[:3])
            self._obj_handle.wxyz = tuple(q[3:7])

    def set_visible(self, value: bool, frame_idx: int) -> None:
        self.visible = value
        for h in self._hand_handles.values():
            h.visible = value
        if self._obj_handle is not None:
            self._obj_handle.visible = value
        if value:
            # Hand meshes may be stale (frames skipped while hidden).
            self._hand_frame.clear()
            self.show_frame(frame_idx)


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)  # flush progress even when piped to a file
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    scene_path = Path(args.scene).resolve() if args.scene else run_dir / "scene.xml"
    traj_path = Path(args.traj).resolve() if args.traj else run_dir / "trajectory_mjwp.npz"
    kin_path = run_dir / "trajectory_kinematic.npz"
    for pth in (scene_path, traj_path):
        if not pth.exists():
            raise SystemExit(f"Not found: {pth}")

    # outputs/<robot>/<embodiment>/<task>/<id> -> outputs root + task + embodiment.
    # Run dirs that don't follow this layout just get the MANO layer disabled.
    keypoints_path = None
    outputs_root = None
    task = ""
    try:
        data_id, task, embodiment = (
            run_dir.name, run_dir.parent.name, run_dir.parent.parent.name
        )
        outputs_root = run_dir.parents[3]
        keypoints_path = (
            outputs_root / "mano" / embodiment / task / data_id / "trajectory_keypoints.npz"
        )
    except IndexError:
        pass

    cfg = _load_run_config(run_dir / "config.yaml")
    warmup_steps = int(cfg.get("warmup_steps", 0) or 0)
    sim_dt = float(cfg.get("sim_dt", 0.005))
    ref_dt = float(cfg.get("ref_dt", 0.0333))
    ref_steps = max(1, int(round(ref_dt / sim_dt)))  # sim steps per reference frame

    # Build the MuJoCo model + data (spec is needed by build_and_log_scene_from_spec).
    spec = mujoco.MjSpec.from_file(str(scene_path))
    model = spec.compile()
    data = mujoco.MjData(model)

    qpos = load_qpos(traj_path, model.nq)
    display_object_upright = bool(args.display_object_upright)
    object_qpos_addresses = _object_freejoint_qpos_addresses(model)
    object_up_axis = (
        np.asarray(args.display_object_up_vector, dtype=np.float64)
        if args.display_object_up_vector is not None
        else _axis_from_label(args.display_object_up_axis)
    )
    hand_root_addresses = _hand_root_qpos_addresses(model)
    display_hand_level = bool(args.display_hand_level)
    if display_hand_level:
        print(
            "WARNING: --display-hand-level independently rotates the hand and "
            "alters hand-object relative geometry; do not use it to judge contact."
        )
    upright_qpos = None
    hand_level_elevation: dict[str, np.ndarray] = {}
    if display_object_upright and object_qpos_addresses:
        upright_qpos = _project_scene_qpos_upright(
            qpos,
            object_qpos_addresses,
            hand_root_addresses,
            object_up_axis,
        )
    if display_hand_level:
        upright_qpos, hand_level_elevation = _level_robot_fingers(
            upright_qpos if upright_qpos is not None else qpos,
            model,
            hand_root_addresses,
        )
    upright_display = {"on": upright_qpos is not None}

    # IK reference (blue ghost) — optional.
    kin_qpos = None
    data_ref = None
    if kin_path.exists():
        kd = np.load(str(kin_path))
        kin_qpos = np.asarray(kd["qpos"], dtype=np.float64)
        if kin_qpos.ndim == 3:
            kin_qpos = kin_qpos.reshape(-1, kin_qpos.shape[-1])
        if kin_qpos.shape[-1] != model.nq:
            print(
                f"Kinematic qpos width {kin_qpos.shape[-1]} != model.nq {model.nq}"
                " — IK reference disabled."
            )
            kin_qpos = None
        else:
            data_ref = mujoco.MjData(model)
            if display_object_upright and object_qpos_addresses:
                kin_qpos = _project_scene_qpos_upright(
                    kin_qpos,
                    object_qpos_addresses,
                    hand_root_addresses,
                    object_up_axis,
                )
            if display_hand_level:
                kin_qpos, _ = _level_robot_fingers(
                    kin_qpos, model, hand_root_addresses
                )

    start = warmup_steps if (args.skip_warmup and warmup_steps > 0) else 0
    start = min(start, max(0, len(qpos) - 1))
    n_frames = len(qpos) - start
    if n_frames <= 0:
        raise SystemExit("No frames to play.")

    capture = _load_capture_metadata(run_dir, keypoints_path)
    viewpoint = args.viewpoint or capture["viewpoint"]
    camera_motion = args.camera_motion or capture["camera_motion"]
    camera_mode = args.camera_mode
    if camera_mode == "auto":
        camera_mode = "ego" if viewpoint == "ego" else "scene"

    def ref_frame_for(display_idx: int) -> int:
        """Map a display frame to the kinematic/MANO reference frame index."""
        sim_i = display_idx + start
        return max(0, sim_i - warmup_steps) // ref_steps

    # Spin up viser and upload the scene geometry (ref ghost only if IK data exists).
    viser_viewer.init_viser(app_name="retargeting-replay", port=args.port)
    server = viser_viewer._get_server()
    body_ids = viser_viewer.build_and_log_scene_from_spec(
        spec, model, xml_path=scene_path,
        build_ref=kin_qpos is not None, build_gui=False,
    )
    # Convex decomposition geoms are useful for physics, but they seal hollow
    # objects such as mugs and obscure the opening in a presentation replay.
    for handle in viser_viewer._STATE.collision_geom_handles:
        handle.visible = False
    for handle, _body_id in viser_viewer._STATE.ref_body_handles:
        handle.visible = False
    for handle in viser_viewer._STATE.ref_geom_handles:
        handle.visible = False

    camera_preset = None
    if camera_mode in {"ego", "top-down"}:
        if camera_mode == "ego":
            camera_position, camera_target, camera_up = _ego_camera_preset(
                qpos,
                start,
                keypoints_path,
                requested_distance=args.ego_distance,
            )
        else:
            camera_position, camera_target, camera_up = _top_down_camera_preset(
                upright_qpos if upright_qpos is not None else qpos,
                start,
                keypoints_path,
                vertical_fov_deg=args.ego_fov,
            )
        camera_preset = (camera_position, camera_target, camera_up)

        def apply_camera_preset(client) -> None:
            client.camera.position = camera_position
            client.camera.look_at = camera_target
            client.camera.up_direction = camera_up
            client.camera.fov = np.deg2rad(args.ego_fov)

        @server.on_client_connect
        def _set_camera_preset(client) -> None:
            apply_camera_preset(client)

        for client in server.get_clients().values():
            apply_camera_preset(client)

    mano = (
        ManoOverlay(
            keypoints_path,
            outputs_root,
            task,
            level_fingers=bool(display_hand_level),
            object_upright=bool(display_object_upright),
            object_up_axis=object_up_axis,
        )
        if keypoints_path is not None
        else None
    )
    if mano is not None and mano.available:
        mano.build(server)
        mano.set_visible(False, 0)
    else:
        mano = None

    def show_frame(frame_idx: int) -> None:
        fi = max(0, min(n_frames - 1, int(frame_idx)))
        shown_qpos = (
            upright_qpos if upright_display["on"] and upright_qpos is not None
            else qpos
        )
        data.qpos[:] = shown_qpos[fi + start]
        mujoco.mj_kinematics(model, data)  # populate body xpos/xquat for log_frame
        rf = ref_frame_for(fi)
        if kin_qpos is not None:
            data_ref.qpos[:] = kin_qpos[min(rf, len(kin_qpos) - 1)]
            mujoco.mj_kinematics(model, data_ref)
        viser_viewer.log_frame(
            data, sim_time=0.0, viewer_body_entity_and_ids=body_ids,
            data_ref=data_ref, record=False,
        )
        if mano is not None:
            mano.show_frame(rf)

    # --- GUI: layer toggles + Frame slider + Play/Pause + FPS ---
    with server.gui.add_folder("Layers"):
        if mano is not None:
            cb_mano = server.gui.add_checkbox(
                "MANO reference (orange)", initial_value=False
            )
        if kin_qpos is not None:
            cb_ik = server.gui.add_checkbox(
                "IK reference (blue)", initial_value=False
            )
        cb_robot = server.gui.add_checkbox(
            "Retargeted Sharpa hand + object", initial_value=True
        )
        if upright_qpos is not None:
            alignment_label = (
                "Hand/table + object alignment (display only)"
                if display_hand_level
                else "Object/hand rigid alignment (display only)"
            )
            cb_upright = server.gui.add_checkbox(
                alignment_label, initial_value=True
            )

    frame_slider = server.gui.add_slider(
        "Frame", min=0, max=n_frames - 1, step=1, initial_value=0
    )
    play_button = server.gui.add_button("Play")
    fps_slider = server.gui.add_slider(
        "FPS", min=1, max=120, step=1, initial_value=int(args.fps)
    )

    if mano is not None:

        @cb_mano.on_update
        def _(_) -> None:
            mano.set_visible(cb_mano.value, ref_frame_for(int(frame_slider.value)))

    if kin_qpos is not None:

        @cb_ik.on_update
        def _(_) -> None:
            # Ghost body frames parent their geoms, so toggling frames suffices.
            for h, _bid in viser_viewer._STATE.ref_body_handles:
                h.visible = cb_ik.value
            for gh in viser_viewer._STATE.ref_geom_handles:
                gh.visible = cb_ik.value

    @cb_robot.on_update
    def _(_) -> None:
        # Skip worldbody (bid 0) so the floor stays visible.
        for h, bid in body_ids:
            if bid != 0:
                h.visible = cb_robot.value

    if upright_qpos is not None:

        @cb_upright.on_update
        def _(_) -> None:
            upright_display["on"] = cb_upright.value
            show_frame(int(frame_slider.value))

    playing = {"on": False}
    suppress_cb = {"on": False}  # don't double-render during programmatic advance

    def set_button_label(text: str) -> None:
        for attr in ("name", "label"):  # viser version compatibility
            try:
                setattr(play_button, attr, text)
                return
            except Exception:
                pass

    @frame_slider.on_update
    def _(_) -> None:
        if not suppress_cb["on"]:
            show_frame(frame_slider.value)

    @play_button.on_click
    def _(_) -> None:
        playing["on"] = not playing["on"]
        set_button_label("Pause" if playing["on"] else "Play")

    def playback_loop() -> None:
        while True:
            if playing["on"]:
                nxt = int(frame_slider.value) + 1
                if nxt >= n_frames:
                    nxt = 0
                suppress_cb["on"] = True
                frame_slider.value = nxt
                suppress_cb["on"] = False
                show_frame(nxt)
                time.sleep(1.0 / max(1.0, float(fps_slider.value)))
            else:
                time.sleep(0.05)

    threading.Thread(target=playback_loop, daemon=True).start()

    show_frame(0)
    print(f"Scene:       {scene_path}")
    print(f"Retargeted:  {traj_path}")
    print(f"IK ref:      {kin_path if kin_qpos is not None else 'not found'}")
    print(f"MANO ref:    {keypoints_path if mano is not None else 'not found'}")
    print(
        "Presentation: hand-only "
        f"(viewpoint={viewpoint}, camera_motion={camera_motion}, camera={camera_mode})"
    )
    if camera_preset is not None:
        camera_position, camera_target, camera_up = camera_preset
        print(
            f"{camera_mode} camera:  position="
            f"{np.round(camera_position, 4).tolist()}, "
            f"look_at={np.round(camera_target, 4).tolist()}, "
            f"up={np.round(camera_up, 4).tolist()}"
        )
    if camera_motion == "moving":
        print(
            "NOTE: this replay uses one stabilized ego camera. Per-frame camera "
            "extrinsics are not produced by the current reconstruction pipeline."
        )
    if upright_qpos is not None:
        alignment_notes = []
        if display_object_upright:
            up_description = (
                np.round(object_up_axis, 5).tolist()
                if args.display_object_up_vector is not None
                else args.display_object_up_axis
            )
            alignment_notes.append(
                f"object semantic up {up_description} and the hand root are "
                "rigidly aligned with world +Z"
            )
        if display_hand_level:
            initial_elevation = max(
                (float(values[start]) for values in hand_level_elevation.values()),
                default=0.0,
            )
            alignment_notes.append(
                f"robot finger elevation {initial_elevation:.1f}° is independently "
                "projected onto the table"
            )
        print(
            "NOTE: display alignment is enabled: "
            + "; ".join(alignment_notes)
            + ". The saved MJWP trajectory is unchanged."
        )
    print(
        f"Playing {n_frames} frames"
        + (f" (skipped {start} warmup frames; --no-skip-warmup to include)" if start else "")
    )
    print(f"Viewer running at http://localhost:{args.port}  (Ctrl+C to exit)")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nShutting down.")


if __name__ == "__main__":
    main()
