"""Read and make explicit the Retargeting run data protocol."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import yaml


@dataclass(frozen=True)
class JointSpec:
    name: str
    qpos_start: int
    qpos_width: int
    lower: float | None
    upper: float | None


@dataclass(frozen=True)
class SceneProtocol:
    root_joints: tuple[JointSpec, ...]
    finger_joints: tuple[JointSpec, ...]
    object_joint: JointSpec
    object_mesh_path: Path | None
    qpos_width: int


@dataclass(frozen=True)
class SourceTrajectory:
    run_dir: Path
    trajectory_path: Path
    scene_path: Path
    object_mesh_path: Path | None
    qpos: np.ndarray
    source_time: np.ndarray | None
    trajectory_kind: str
    sim_dt: float
    ref_dt: float | None
    warmup_steps: int
    task: str
    robot_type: str
    embodiment_type: str
    protocol: SceneProtocol
    source_keys: tuple[str, ...]
    chunk_end_steps: tuple[int, ...]
    chunk_size: int | None


def _load_yaml_mapping(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required run config not found: {path}")
    with path.open(encoding="utf-8") as stream:
        values = yaml.safe_load(stream) or {}
    if not isinstance(values, dict):
        raise ValueError(f"Expected a YAML mapping in {path}.")
    return values


def _joint_width(element: ElementTree.Element) -> int:
    if element.tag == "freejoint" or element.get("type") == "free":
        return 7
    if element.get("type") == "ball":
        return 4
    return 1


def _joint_range(element: ElementTree.Element) -> tuple[float | None, float | None]:
    value = element.get("range")
    if value is None:
        return None, None
    limits = value.split()
    if len(limits) != 2:
        raise ValueError(
            f"Joint {element.get('name')!r} has an invalid range: {value!r}."
        )
    return float(limits[0]), float(limits[1])


def _resolve_object_mesh(
    root: ElementTree.Element, scene_path: Path, side: str
) -> Path | None:
    object_body = root.find(f".//body[@name='{side}_object']")
    if object_body is None:
        return None

    mesh_name = None
    for geom in object_body.findall(".//geom"):
        candidate = geom.get("mesh")
        if candidate and (geom.get("group") == "1" or candidate.endswith("visual")):
            mesh_name = candidate
            break
    if mesh_name is None:
        for geom in object_body.findall(".//geom"):
            if geom.get("mesh"):
                mesh_name = geom.get("mesh")
                break
    if mesh_name is None:
        return None

    mesh_element = root.find(f"./asset/mesh[@name='{mesh_name}']")
    if mesh_element is None or mesh_element.get("file") is None:
        return None
    compiler = root.find("./compiler")
    mesh_dir_value = compiler.get("meshdir", ".") if compiler is not None else "."
    mesh_dir = Path(mesh_dir_value)
    if not mesh_dir.is_absolute():
        mesh_dir = scene_path.parent / mesh_dir
    return (mesh_dir / mesh_element.get("file")).resolve()


def parse_scene_protocol(
    scene_path: Path, side: str, expected_finger_joints: int
) -> SceneProtocol:
    """Derive qpos addresses and names from the run's actual MuJoCo scene."""
    if not scene_path.is_file():
        raise FileNotFoundError(f"Required MuJoCo scene not found: {scene_path}")
    root = ElementTree.parse(scene_path).getroot()

    specs: list[JointSpec] = []
    qpos_start = 0
    for element in root.iter():
        if element.tag not in {"joint", "freejoint"}:
            continue
        name = element.get("name")
        if not name:
            continue
        width = _joint_width(element)
        lower, upper = _joint_range(element)
        specs.append(JointSpec(name, qpos_start, width, lower, upper))
        qpos_start += width

    root_names = tuple(
        f"{side}_{suffix}"
        for suffix in ("pos_x", "pos_y", "pos_z", "rot_x", "rot_y", "rot_z")
    )
    by_name = {spec.name: spec for spec in specs}
    missing_root = [name for name in root_names if name not in by_name]
    if missing_root:
        raise ValueError(f"Scene is missing expected root joints: {missing_root}")
    root_joints = tuple(by_name[name] for name in root_names)
    if tuple(spec.qpos_start for spec in root_joints) != tuple(range(6)):
        raise ValueError("The six hand root joints are not the first six qpos values.")

    object_candidates = [
        spec
        for spec in specs
        if spec.name == f"{side}_object_joint" or spec.qpos_width == 7
    ]
    if len(object_candidates) != 1:
        names = [spec.name for spec in object_candidates]
        raise ValueError(f"Expected exactly one object free joint, found {names}.")
    object_joint = object_candidates[0]
    if object_joint.qpos_width != 7:
        raise ValueError(f"Object joint {object_joint.name} is not a free joint.")

    root_name_set = set(root_names)
    finger_joints = tuple(
        spec
        for spec in specs
        if spec.name not in root_name_set and spec.name != object_joint.name
    )
    if len(finger_joints) != expected_finger_joints:
        raise ValueError(
            f"Expected {expected_finger_joints} finger joints, "
            f"found {len(finger_joints)}: {[spec.name for spec in finger_joints]}"
        )
    if any(spec.qpos_width != 1 for spec in finger_joints):
        raise ValueError("All Sharpa finger joints must have one qpos value.")
    expected_finger_addresses = tuple(range(6, 6 + expected_finger_joints))
    if tuple(spec.qpos_start for spec in finger_joints) != expected_finger_addresses:
        raise ValueError("Finger joints are not contiguous after the hand root joints.")
    if object_joint.qpos_start != 6 + expected_finger_joints:
        raise ValueError("The object free joint is not directly after the finger joints.")

    return SceneProtocol(
        root_joints=root_joints,
        finger_joints=finger_joints,
        object_joint=object_joint,
        object_mesh_path=_resolve_object_mesh(root, scene_path, side),
        qpos_width=qpos_start,
    )


def _chunk_order(data: np.lib.npyio.NpzFile, chunk_count: int) -> np.ndarray:
    if "sim_step" not in data.files:
        return np.arange(chunk_count)
    sim_steps = np.asarray(data["sim_step"]).reshape(-1)
    if sim_steps.shape[0] != chunk_count:
        raise ValueError(
            f"sim_step has {sim_steps.shape[0]} entries for {chunk_count} chunks."
        )
    if len(np.unique(sim_steps)) != len(sim_steps):
        raise ValueError("sim_step contains duplicate chunk endpoints.")
    return np.argsort(sim_steps, kind="stable")


def _flatten_frames(array: np.ndarray, order: np.ndarray) -> np.ndarray:
    if array.ndim == 3:
        return array[order].reshape(-1, array.shape[-1])
    if array.ndim == 2 and order.shape[0] > 1:
        return array[order].reshape(-1)
    return array


def load_source_trajectory(
    run_dir: Path, trajectory_filename: str, expected_finger_joints: int
) -> SourceTrajectory:
    """Load a right-hand Sharpa run and flatten sequential optimization chunks."""
    run_dir = run_dir.resolve()
    scene_path = run_dir / "scene.xml"
    trajectory_path = run_dir / trajectory_filename
    run_config = _load_yaml_mapping(run_dir / "config.yaml")
    if not trajectory_path.is_file():
        raise FileNotFoundError(f"Required trajectory not found: {trajectory_path}")

    robot_type = str(run_config.get("robot_type", ""))
    embodiment_type = str(run_config.get("embodiment_type", ""))
    protocol = parse_scene_protocol(
        scene_path, embodiment_type, expected_finger_joints
    )

    with np.load(trajectory_path, allow_pickle=False) as data:
        if "qpos" not in data.files:
            raise ValueError(
                f"Trajectory has no qpos array. Available keys: {list(data.files)}"
            )
        raw_qpos = np.asarray(data["qpos"])
        if raw_qpos.ndim not in {2, 3}:
            raise ValueError(f"qpos must be 2D or chunked 3D, got {raw_qpos.shape}.")
        chunk_count = raw_qpos.shape[0] if raw_qpos.ndim == 3 else 1
        order = _chunk_order(data, chunk_count)
        qpos = _flatten_frames(raw_qpos, order)
        source_time = None
        if "time" in data.files:
            source_time = _flatten_frames(np.asarray(data["time"]), order)
        if "frequency" in data.files:
            frequency = float(np.asarray(data["frequency"]))
            if not np.isfinite(frequency) or frequency <= 0.0:
                raise ValueError(f"Trajectory frequency must be positive, got {frequency}.")
            trajectory_kind = "kinematic"
            sim_dt = 1.0 / frequency
            warmup_steps = 0
        else:
            trajectory_kind = "physics_optimized"
            sim_dt = float(run_config["sim_dt"])
            warmup_steps = int(run_config.get("warmup_steps", 0))
        source_keys = tuple(data.files)
        if "sim_step" in data.files:
            sim_steps = np.asarray(data["sim_step"]).reshape(-1)[order]
            chunk_end_steps = tuple(int(value) for value in sim_steps)
        else:
            chunk_end_steps = ()

    if qpos.ndim != 2 or qpos.shape[1] != protocol.qpos_width:
        raise ValueError(
            f"Flattened qpos shape {qpos.shape} does not match scene width "
            f"{protocol.qpos_width}."
        )
    if source_time is not None and source_time.shape != (qpos.shape[0],):
        raise ValueError(
            f"Flattened time shape {source_time.shape} does not match "
            f"{qpos.shape[0]} qpos frames."
        )

    return SourceTrajectory(
        run_dir=run_dir,
        trajectory_path=trajectory_path,
        scene_path=scene_path,
        object_mesh_path=protocol.object_mesh_path,
        qpos=np.ascontiguousarray(qpos, dtype=np.float64),
        source_time=(
            np.ascontiguousarray(source_time, dtype=np.float64)
            if source_time is not None
            else None
        ),
        trajectory_kind=trajectory_kind,
        sim_dt=sim_dt,
        ref_dt=(
            float(run_config["ref_dt"]) if run_config.get("ref_dt") is not None else None
        ),
        warmup_steps=warmup_steps,
        task=str(run_config.get("task", run_dir.parent.name)),
        robot_type=robot_type,
        embodiment_type=embodiment_type,
        protocol=protocol,
        source_keys=source_keys,
        chunk_end_steps=chunk_end_steps,
        chunk_size=raw_qpos.shape[1] if raw_qpos.ndim == 3 else None,
    )