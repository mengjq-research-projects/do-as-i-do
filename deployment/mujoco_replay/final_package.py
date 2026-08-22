"""Pure data helpers for the final UR3e + Sharpa + object replay package.

This module deliberately has no MuJoCo/viser dependency so package conversion
and validation can run in a lightweight Python environment.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

TRAJECTORY_FILENAME = "trajectory_dual_ur3e.npz"
MANIFEST_FILENAME = "deployment_manifest.json"
SCENE_FILENAME = "scene_dual_ur3e_object.xml"
OBJECT_QPOS_SLICE = slice(28, 35)


def load_source_qpos(path: Path) -> np.ndarray:
    """Load and chronologically flatten a Retargeting qpos trajectory."""
    with np.load(path, allow_pickle=False) as data:
        if "qpos" not in data.files:
            raise ValueError(f"{path} does not contain a qpos array.")
        raw = np.asarray(data["qpos"], dtype=np.float64)
        if raw.ndim == 2:
            qpos = raw
        elif raw.ndim == 3:
            order = np.arange(raw.shape[0])
            if "sim_step" in data.files:
                sim_step = np.asarray(data["sim_step"]).reshape(-1)
                if sim_step.shape != (raw.shape[0],):
                    raise ValueError(
                        f"sim_step shape {sim_step.shape} does not match "
                        f"{raw.shape[0]} trajectory chunks."
                    )
                order = np.argsort(sim_step, kind="stable")
            qpos = raw[order].reshape(-1, raw.shape[-1])
        else:
            raise ValueError(f"qpos must be 2D or 3D, got shape {raw.shape}.")
    if qpos.shape[1] < OBJECT_QPOS_SLICE.stop:
        raise ValueError(
            f"qpos width {qpos.shape[1]} does not contain the expected "
            "22-finger + 7-DoF object protocol."
        )
    if not np.isfinite(qpos).all():
        raise ValueError("Source qpos contains NaN or Inf values.")
    return np.ascontiguousarray(qpos)


def _normalize_quaternions(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError("Object trajectory contains a zero quaternion.")
    return values / norms


def _quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Vectorized Hamilton product for wxyz quaternions."""
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


def workspace_quaternion(
    yaw_deg: float, pitch_deg: float = 0.0, roll_deg: float = 0.0
) -> np.ndarray:
    """Return RotZ(yaw) * RotY(pitch) * RotX(roll), in wxyz order."""

    def axis_quaternion(axis: int, degrees: float) -> np.ndarray:
        half = np.deg2rad(degrees) / 2.0
        value = np.zeros(4, dtype=np.float64)
        value[0] = np.cos(half)
        value[axis + 1] = np.sin(half)
        return value

    qz = axis_quaternion(2, yaw_deg)
    qy = axis_quaternion(1, pitch_deg)
    qx = axis_quaternion(0, roll_deg)
    return _quat_multiply(qz, _quat_multiply(qy, qx))


def transform_object_trajectory(
    qpos: np.ndarray,
    *,
    wrist_anchor: np.ndarray,
    workspace_xyz: np.ndarray,
    yaw_deg: float,
    pitch_deg: float = 0.0,
    roll_deg: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Move the source object with the exact same workspace rigid transform.

    ``wrist_anchor`` is the source wrist position at the chosen start frame.
    Subtracting that anchor preserves every hand/object relative translation.
    """
    qpos = np.asarray(qpos, dtype=np.float64)
    object_qpos = qpos[:, OBJECT_QPOS_SLICE]
    rotation = workspace_quaternion(yaw_deg, pitch_deg, roll_deg)
    rotation_batch = np.broadcast_to(rotation, (qpos.shape[0], 4))
    relative_position = object_qpos[:, :3] - np.asarray(wrist_anchor)
    pure_vectors = np.concatenate(
        (np.zeros((qpos.shape[0], 1)), relative_position), axis=1
    )
    rotated = _quat_multiply(
        _quat_multiply(rotation_batch, pure_vectors),
        np.broadcast_to(
            np.array([rotation[0], -rotation[1], -rotation[2], -rotation[3]]),
            rotation_batch.shape,
        ),
    )[:, 1:]
    position = rotated + np.asarray(workspace_xyz, dtype=np.float64)
    quaternion = _normalize_quaternions(
        _quat_multiply(rotation_batch, _normalize_quaternions(object_qpos[:, 3:7]))
    )
    return np.ascontiguousarray(position), np.ascontiguousarray(quaternion)


def discover_object_mesh(trajectory_path: Path, side: str) -> Path | None:
    """Resolve the visual mesh recorded by the source Retargeting scene."""
    scene_path = trajectory_path.resolve().parent / "scene.xml"
    if scene_path.is_file():
        root = ElementTree.parse(scene_path).getroot()
        body = root.find(f".//body[@name='{side}_object']")
        mesh_name = None
        if body is not None:
            geoms = list(body.findall(".//geom"))
            for geom in geoms:
                candidate = geom.get("mesh")
                if candidate and (
                    "visual" in candidate or geom.get("group") in {"0", "1"}
                ):
                    mesh_name = candidate
                    break
            if mesh_name is None:
                mesh_name = next(
                    (geom.get("mesh") for geom in geoms if geom.get("mesh")), None
                )
        if mesh_name:
            mesh = root.find(f"./asset/mesh[@name='{mesh_name}']")
            if mesh is not None and mesh.get("file"):
                compiler = root.find("./compiler")
                mesh_dir = Path(
                    compiler.get("meshdir", ".") if compiler is not None else "."
                )
                if not mesh_dir.is_absolute():
                    mesh_dir = scene_path.parent / mesh_dir
                candidate = (mesh_dir / mesh.get("file")).resolve()
                if candidate.is_file():
                    return candidate
    return None


def discover_object_texture(mesh_path: Path | None) -> Path | None:
    if mesh_path is None:
        return None
    for name in (
        "visual_texture.png",
        "material_0.png",
        "texture.png",
        "visual_texture.jpg",
        "material_0.jpg",
    ):
        candidate = mesh_path.parent / name
        if candidate.is_file():
            return candidate.resolve()
    return None


def save_final_package(
    *,
    output_path: Path,
    scene_xml: str,
    arm_qpos: np.ndarray,
    finger_qpos: np.ndarray,
    object_position: np.ndarray,
    object_quaternion: np.ndarray,
    arm_joint_names: list[str],
    finger_joint_names: list[str],
    arm_joint_limits: np.ndarray,
    finger_joint_limits: np.ndarray,
    dt: float,
    start_frame: int,
    workspace_xyz: np.ndarray,
    workspace_yaw_deg: float,
    workspace_pitch_deg: float,
    workspace_roll_deg: float,
    ik_seed_qpos: np.ndarray,
    side: str,
    source_traj: Path,
    object_mesh: Path | None,
    object_texture: Path | None,
    capture_metadata: dict[str, str] | None = None,
) -> tuple[Path, Path, Path]:
    """Write compatible NPZ plus simulator-independent metadata and MJCF."""
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = int(arm_qpos.shape[0])
    for name, value, expected in (
        ("finger_qpos", finger_qpos, (frame_count, len(finger_joint_names))),
        ("object_position", object_position, (frame_count, 3)),
        ("object_quaternion", object_quaternion, (frame_count, 4)),
    ):
        if np.asarray(value).shape != expected:
            raise ValueError(f"{name} has shape {np.asarray(value).shape}; expected {expected}.")
    timestamps = np.arange(frame_count, dtype=np.float64) * float(dt)
    source_frame_indices = np.arange(frame_count, dtype=np.int64)
    valid_mask = source_frame_indices >= int(start_frame)
    capture_metadata = capture_metadata or {
        "viewpoint": "auto",
        "camera_motion": "auto",
    }
    np.savez_compressed(
        output_path,
        arm_qpos=np.asarray(arm_qpos, dtype=np.float64),
        finger_qpos=np.asarray(finger_qpos, dtype=np.float64),
        object_position=np.asarray(object_position, dtype=np.float64),
        object_quaternion=_normalize_quaternions(object_quaternion),
        timestamps=timestamps,
        valid_mask=valid_mask,
        source_frame_indices=source_frame_indices,
        arm_joint_names=np.asarray(arm_joint_names),
        finger_joint_names=np.asarray(finger_joint_names),
        dt=np.float64(dt),
        start_frame=np.int64(start_frame),
        workspace_xyz=np.asarray(workspace_xyz, dtype=np.float64),
        workspace_yaw_deg=np.float64(workspace_yaw_deg),
        workspace_pitch_deg=np.float64(workspace_pitch_deg),
        workspace_roll_deg=np.float64(workspace_roll_deg),
        ik_seed_qpos=np.asarray(ik_seed_qpos, dtype=np.float64),
        side=str(side),
        source_traj=str(source_traj.resolve()),
        capture_viewpoint=str(capture_metadata.get("viewpoint", "auto")),
        capture_camera_motion=str(
            capture_metadata.get("camera_motion", "auto")
        ),
    )

    scene_path = output_path.parent / SCENE_FILENAME
    scene_path.write_text(scene_xml, encoding="utf-8")
    manifest_path = output_path.parent / MANIFEST_FILENAME

    def relative_or_absolute(path: Path | None) -> str | None:
        if path is None:
            return None
        try:
            return str(path.resolve().relative_to(output_path.parent))
        except ValueError:
            return str(path.resolve())

    def joint_specs(names: list[str], limits: np.ndarray) -> list[dict]:
        return [
            {
                "name": name,
                "lower_rad": float(bound[0]),
                "upper_rad": float(bound[1]),
            }
            for name, bound in zip(names, np.asarray(limits), strict=True)
        ]

    manifest = {
        "schema_version": "deployment.robot_scene.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "robot": {
            "type": "dual_ur3e_sharpa",
            "embodiment": side,
            "arm_joints": joint_specs(arm_joint_names, arm_joint_limits),
            "finger_joints": joint_specs(
                finger_joint_names, finger_joint_limits
            ),
        },
        "object": {
            "body_name": f"{side}_object",
            "free_joint": f"{side}_object_joint",
            "source_mesh": relative_or_absolute(object_mesh),
            "source_texture": relative_or_absolute(object_texture),
            "playback_mode": "kinematic_reference",
        },
        "scene": {"mjcf": scene_path.name},
        "source": {
            "trajectory": str(source_traj.resolve()),
            "capture": {
                "viewpoint": str(capture_metadata.get("viewpoint", "auto")),
                "camera_motion": str(
                    capture_metadata.get("camera_motion", "auto")
                ),
            },
        },
        "timing": {
            "sim_dt_seconds": float(dt),
            "frame_count": frame_count,
            "start_frame": int(start_frame),
        },
        "coordinate_system": {
            "world": "MuJoCo/Isaac right-handed Z-up",
            "length_unit": "meter",
            "angle_unit": "radian",
            "quaternion_order": "wxyz",
            "workspace_rotation": "RotZ(yaw) * RotY(pitch) * RotX(roll)",
        },
        "trajectory": {"file": output_path.name},
        "trajectory_arrays": {
            "arm_qpos": [frame_count, len(arm_joint_names)],
            "finger_qpos": [frame_count, len(finger_joint_names)],
            "object_position": [frame_count, 3],
            "object_quaternion": [frame_count, 4],
            "timestamps": [frame_count],
            "valid_mask": [frame_count],
            "source_frame_indices": [frame_count],
        },
        "isaac_sim": {
            "version": "unverified",
            "asset_conversion_status": "not_run",
            "state_readback_validation_status": "not_run",
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    return output_path, manifest_path, scene_path


def validate_final_package(directory: Path) -> dict:
    """Validate a final package without importing MuJoCo or Isaac Sim."""
    directory = directory.resolve()
    manifest_path = directory / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing {manifest_path}.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    trajectory_path = directory / manifest["trajectory"]["file"]
    frame_count = int(manifest["timing"]["frame_count"])
    arm_count = len(manifest["robot"]["arm_joints"])
    finger_count = len(manifest["robot"]["finger_joints"])
    expected = {
        "arm_qpos": (frame_count, arm_count),
        "finger_qpos": (frame_count, finger_count),
        "object_position": (frame_count, 3),
        "object_quaternion": (frame_count, 4),
        "timestamps": (frame_count,),
        "valid_mask": (frame_count,),
        "source_frame_indices": (frame_count,),
    }
    with np.load(trajectory_path, allow_pickle=False) as data:
        for name, shape in expected.items():
            if name not in data.files or data[name].shape != shape:
                actual = None if name not in data.files else data[name].shape
                raise ValueError(f"{name} has shape {actual}; expected {shape}.")
            if data[name].dtype.kind in "fc" and not np.isfinite(data[name]).all():
                raise ValueError(f"{name} contains NaN or Inf values.")
        norms = np.linalg.norm(data["object_quaternion"], axis=1)
        if not np.allclose(norms, 1.0, atol=1e-6):
            raise ValueError("object_quaternion is not normalized.")
        start_frame = int(manifest["timing"]["start_frame"])
        valid_mask = np.asarray(data["valid_mask"], dtype=bool)
        if valid_mask[:start_frame].any() or not valid_mask[start_frame:].all():
            raise ValueError("valid_mask does not match start_frame.")
    scene_path = directory / manifest["scene"]["mjcf"]
    if not scene_path.is_file():
        raise FileNotFoundError(f"Missing {scene_path}.")
    return {
        "status": "pass",
        "frame_count": frame_count,
        "arm_joint_count": arm_count,
        "finger_joint_count": finger_count,
        "object_body": manifest["object"]["body_name"],
        "trajectory": str(trajectory_path),
        "scene": str(scene_path),
    }
