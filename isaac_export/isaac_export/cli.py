"""Command-line entry point for Level A export."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from .config import load_export_config
from .environment import inspect_environment
from .source import SourceTrajectory, load_source_trajectory
from .trajectory import build_standard_trajectory, save_standard_trajectory
from .validate import validate_trajectory

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PACKAGE_ROOT / "config" / "default.yaml"
DEFAULT_RUN_DIR = (
    PACKAGE_ROOT.parent
    / "retargeting"
    / "outputs"
    / "sharpa"
    / "right"
    / "whisking"
    / "0"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a Retargeting run into the canonical Level A trajectory "
            "and manifest used by Isaac Sim replay."
        )
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--trajectory", help="Override the trajectory filename.")
    parser.add_argument(
        "--include-warmup",
        action="store_true",
        help="Keep warmup frames and mark them invalid instead of dropping them.",
    )
    parser.add_argument(
        "--isaac-version",
        default="unverified",
        help="Isaac Sim version to record in manifest.json.",
    )
    return parser.parse_args()


def _relative_or_absolute(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path.resolve())


def _joint_manifest(source: SourceTrajectory) -> list[dict]:
    return [
        {
            "name": spec.name,
            "source_qpos_index": spec.qpos_start,
            "lower_rad": spec.lower,
            "upper_rad": spec.upper,
        }
        for spec in source.protocol.finger_joints
    ]


def _build_manifest(
    source: SourceTrajectory,
    output_dir: Path,
    frame_count: int,
    skip_warmup: bool,
    isaac_version: str,
) -> dict:
    duration = max(frame_count - 1, 0) * source.sim_dt
    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "task": source.task,
        "robot": {
            "type": source.robot_type,
            "embodiment": source.embodiment_type,
            "finger_joint_count": len(source.protocol.finger_joints),
            "finger_joints": _joint_manifest(source),
            "root_source_joints": [
                spec.name for spec in source.protocol.root_joints
            ],
        },
        "object": {
            "source_mesh": (
                _relative_or_absolute(source.object_mesh_path, output_dir)
                if source.object_mesh_path is not None
                else None
            ),
            "source_free_joint": source.protocol.object_joint.name,
            "mesh_scale": 1.0,
            "mesh_to_body_transform": "identity",
        },
        "source": {
            "run_directory": _relative_or_absolute(source.run_dir, output_dir),
            "scene": _relative_or_absolute(source.scene_path, output_dir),
            "trajectory": _relative_or_absolute(
                source.trajectory_path, output_dir
            ),
            "trajectory_kind": source.trajectory_kind,
            "source_keys": list(source.source_keys),
            "source_frame_count": int(source.qpos.shape[0]),
            "chunk_end_steps": list(source.chunk_end_steps),
        },
        "coordinate_system": {
            "source_world": "Retargeting right-handed Z-up",
            "target_world": "Isaac right-handed Z-up",
            "length_unit": "meter",
            "angle_unit": "radian",
            "quaternion_order": "wxyz",
            "hand_root_euler_source_order": "Rx * Ry * Rz",
            "hand_root_transform": "identity",
            "object_transform": "identity",
        },
        "timing": {
            "sim_dt_seconds": source.sim_dt,
            "reference_dt_seconds": source.ref_dt,
            "frame_count": frame_count,
            "duration_seconds": duration,
            "timestamps_start_at_zero": True,
            "warmup_source_steps": source.warmup_steps,
            "warmup_skipped": skip_warmup,
            "warmup_exported_steps": 0 if skip_warmup else source.warmup_steps,
        },
        "trajectory_arrays": {
            "timestamps": [frame_count],
            "hand_root_position": [frame_count, 3],
            "hand_root_quaternion": [frame_count, 4],
            "hand_joint_positions": [
                frame_count,
                len(source.protocol.finger_joints),
            ],
            "object_position": [frame_count, 3],
            "object_quaternion": [frame_count, 4],
            "valid_mask": [frame_count],
            "source_frame_indices": [frame_count],
        },
        "isaac_sim": {
            "version": isaac_version,
            "asset_conversion_status": "not_run",
            "state_readback_validation_status": "not_run",
        },
    }


def main() -> None:
    args = _parse_args()
    config = load_export_config(args.config)
    trajectory_filename = args.trajectory or config.trajectory_filename
    source = load_source_trajectory(
        args.run_dir, trajectory_filename, config.expected_finger_joints
    )
    if source.robot_type != config.expected_robot:
        raise SystemExit(
            f"Expected robot {config.expected_robot!r}, got {source.robot_type!r}."
        )
    if source.embodiment_type != config.expected_embodiment:
        raise SystemExit(
            "Level A currently supports only "
            f"{config.expected_embodiment!r}, got {source.embodiment_type!r}."
        )

    skip_warmup = config.skip_warmup and not args.include_warmup
    trajectory = build_standard_trajectory(source, skip_warmup=skip_warmup)
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else PACKAGE_ROOT / "outputs" / source.task
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    trajectory_path = output_dir / "trajectory.npz"
    manifest_path = output_dir / "manifest.json"
    report_path = output_dir / "quality_report.json"
    environment_path = output_dir / "environment_report.json"
    save_standard_trajectory(trajectory, trajectory_path)

    manifest = _build_manifest(
        source,
        output_dir,
        trajectory.frame_count,
        skip_warmup,
        args.isaac_version,
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    report = validate_trajectory(
        trajectory,
        source,
        config.position_jump_warning_m,
        config.joint_jump_warning_rad,
    )
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    environment = inspect_environment(args.isaac_version)
    environment_path.write_text(
        json.dumps(environment, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    print(f"Exported {trajectory.frame_count} frames to {output_dir}")
    print(f"Quality status: {report['status']} ({len(report['warnings'])} warnings)")
    print(f"Isaac environment: {environment['status']}")
    if report["status"] != "pass":
        raise SystemExit("Export failed quality checks; see quality_report.json.")


if __name__ == "__main__":
    main()