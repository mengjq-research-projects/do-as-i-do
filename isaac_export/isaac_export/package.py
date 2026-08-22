"""Load and validate an Isaac-independent Level A replay package."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .coordinate import quaternion_to_euler_xyz
from .trajectory import StandardTrajectory


@dataclass(frozen=True)
class RobotSceneTrajectory:
    timestamps: np.ndarray
    arm_joint_positions: np.ndarray
    hand_joint_positions: np.ndarray
    object_position: np.ndarray
    object_quaternion: np.ndarray
    valid_mask: np.ndarray
    source_frame_indices: np.ndarray

    @property
    def frame_count(self) -> int:
        return int(self.timestamps.shape[0])


@dataclass(frozen=True)
class ReplayPackage:
    directory: Path
    manifest_path: Path
    manifest: dict
    trajectory: StandardTrajectory | RobotSceneTrajectory
    root_joint_names: tuple[str, ...]
    arm_joint_names: tuple[str, ...]
    finger_joint_names: tuple[str, ...]
    clipped_finger_positions: np.ndarray
    clip_count: int
    max_clip_rad: float

    @property
    def frame_count(self) -> int:
        return self.trajectory.frame_count

    @property
    def dt(self) -> float:
        return float(self.manifest["timing"]["sim_dt_seconds"])

    def hand_joint_state(self, frame: int) -> np.ndarray:
        """Return all driven robot joints in scene-name order."""
        if isinstance(self.trajectory, RobotSceneTrajectory):
            return np.concatenate(
                (
                    self.trajectory.arm_joint_positions[frame],
                    self.clipped_finger_positions[frame],
                )
            )
        root_euler = quaternion_to_euler_xyz(
            self.trajectory.hand_root_quaternion[frame]
        )[0]
        return np.concatenate(
            (
                self.trajectory.hand_root_position[frame],
                root_euler,
                self.clipped_finger_positions[frame],
            )
        )

    @property
    def driven_joint_names(self) -> tuple[str, ...]:
        return (
            self.arm_joint_names + self.finger_joint_names
            if self.arm_joint_names
            else self.root_joint_names + self.finger_joint_names
        )

    @property
    def is_robot_scene(self) -> bool:
        return bool(self.arm_joint_names)


def _array(data: np.lib.npyio.NpzFile, name: str, shape: tuple) -> np.ndarray:
    if name not in data:
        raise ValueError(f"trajectory.npz is missing {name!r}.")
    value = np.asarray(data[name])
    if len(value.shape) != len(shape) or any(
        expected is not None and actual != expected
        for actual, expected in zip(value.shape, shape, strict=True)
    ):
        raise ValueError(f"{name} has shape {value.shape}; expected {shape}.")
    if value.dtype.kind in "fc" and not np.isfinite(value).all():
        raise ValueError(f"{name} contains NaN or Inf values.")
    return value


def _load_level_a_package(directory: Path, manifest_path: Path) -> ReplayPackage:
    directory = directory.resolve()
    trajectory_path = directory / "trajectory.npz"
    if not manifest_path.is_file() or not trajectory_path.is_file():
        raise FileNotFoundError(
            f"Expected manifest.json and trajectory.npz in {directory}."
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frame_count = int(manifest["timing"]["frame_count"])
    finger_specs = manifest["robot"]["finger_joints"]
    finger_count = len(finger_specs)
    with np.load(trajectory_path, allow_pickle=False) as data:
        trajectory = StandardTrajectory(
            timestamps=_array(data, "timestamps", (frame_count,)),
            hand_root_position=_array(
                data, "hand_root_position", (frame_count, 3)
            ),
            hand_root_quaternion=_array(
                data, "hand_root_quaternion", (frame_count, 4)
            ),
            hand_joint_positions=_array(
                data, "hand_joint_positions", (frame_count, finger_count)
            ),
            object_position=_array(data, "object_position", (frame_count, 3)),
            object_quaternion=_array(
                data, "object_quaternion", (frame_count, 4)
            ),
            valid_mask=_array(data, "valid_mask", (frame_count,)),
            source_frame_indices=_array(
                data, "source_frame_indices", (frame_count,)
            ),
        )

    lower = np.asarray([joint["lower_rad"] for joint in finger_specs])
    upper = np.asarray([joint["upper_rad"] for joint in finger_specs])
    clipped = np.clip(trajectory.hand_joint_positions, lower, upper)
    difference = np.abs(clipped - trajectory.hand_joint_positions)
    return ReplayPackage(
        directory=directory,
        manifest_path=manifest_path,
        manifest=manifest,
        trajectory=trajectory,
        root_joint_names=tuple(manifest["robot"]["root_source_joints"]),
        arm_joint_names=(),
        finger_joint_names=tuple(joint["name"] for joint in finger_specs),
        clipped_finger_positions=clipped,
        clip_count=int(np.count_nonzero(difference)),
        max_clip_rad=float(difference.max(initial=0.0)),
    )


def _load_robot_scene_package(directory: Path, manifest_path: Path) -> ReplayPackage:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "deployment.robot_scene.v1":
        raise ValueError(
            f"Unsupported Deployment package schema: {manifest.get('schema_version')!r}."
        )
    trajectory_path = directory / manifest["trajectory"]["file"]
    frame_count = int(manifest["timing"]["frame_count"])
    arm_specs = manifest["robot"]["arm_joints"]
    finger_specs = manifest["robot"]["finger_joints"]
    with np.load(trajectory_path, allow_pickle=False) as data:
        trajectory = RobotSceneTrajectory(
            timestamps=_array(data, "timestamps", (frame_count,)),
            arm_joint_positions=_array(
                data, "arm_qpos", (frame_count, len(arm_specs))
            ),
            hand_joint_positions=_array(
                data, "finger_qpos", (frame_count, len(finger_specs))
            ),
            object_position=_array(data, "object_position", (frame_count, 3)),
            object_quaternion=_array(
                data, "object_quaternion", (frame_count, 4)
            ),
            valid_mask=_array(data, "valid_mask", (frame_count,)),
            source_frame_indices=_array(
                data, "source_frame_indices", (frame_count,)
            ),
        )
    lower = np.asarray([joint["lower_rad"] for joint in finger_specs])
    upper = np.asarray([joint["upper_rad"] for joint in finger_specs])
    clipped = np.clip(trajectory.hand_joint_positions, lower, upper)
    difference = np.abs(clipped - trajectory.hand_joint_positions)
    return ReplayPackage(
        directory=directory,
        manifest_path=manifest_path,
        manifest=manifest,
        trajectory=trajectory,
        root_joint_names=(),
        arm_joint_names=tuple(joint["name"] for joint in arm_specs),
        finger_joint_names=tuple(joint["name"] for joint in finger_specs),
        clipped_finger_positions=clipped,
        clip_count=int(np.count_nonzero(difference)),
        max_clip_rad=float(difference.max(initial=0.0)),
    )


def load_replay_package(directory: Path) -> ReplayPackage:
    """Load either the original Level A package or the final robot package."""
    directory = directory.resolve()
    deployment_manifest = directory / "deployment_manifest.json"
    level_a_manifest = directory / "manifest.json"
    if deployment_manifest.is_file():
        return _load_robot_scene_package(directory, deployment_manifest)
    if level_a_manifest.is_file():
        return _load_level_a_package(directory, level_a_manifest)
    raise FileNotFoundError(
        f"Expected deployment_manifest.json or manifest.json in {directory}."
    )
