#!/usr/bin/env python3
"""Replay a Level A trajectory in Isaac Sim."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
import time
from pathlib import Path

from isaac_export.package import load_replay_package


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--scene", type=Path, help="Defaults to PACKAGE/scene.usd")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--record-dir",
        type=Path,
        help="Write PNG frames and replay.mp4 (requires ffmpeg).",
    )
    parser.add_argument("--record-fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    return parser.parse_args()


def load_scene_mapping(package_dir: Path) -> dict:
    path = package_dir / "isaac_scene_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run isaac_export/build_usd.py first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def encode_video(record_dir: Path, fps: float) -> None:
    output = record_dir / "replay.mp4"
    command = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(record_dir / "frame_%06d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output),
    ]
    subprocess.run(command, check=True)
    print(f"Saved replay video: {output}")


def main() -> None:
    args = parse_args()
    if args.stride < 1:
        raise SystemExit("--stride must be at least 1.")
    package = load_replay_package(args.package_dir)
    mapping = load_scene_mapping(package.directory)
    scene_path = (args.scene or package.directory / "scene.usd").resolve()
    if not scene_path.is_file():
        raise SystemExit(f"Scene does not exist: {scene_path}")

    try:
        from isaacsim import SimulationApp
    except ModuleNotFoundError as error:
        raise SystemExit(
            "Isaac Sim is not installed in this Python environment. Run with "
            "the dedicated Isaac Sim Python environment."
        ) from error

    simulation_app = SimulationApp(
        {
            "headless": args.headless,
            "width": args.width,
            "height": args.height,
            # Isaac Sim 6 fast shutdown can abort with a busy TaskGroup and
            # terminate before the replay report below is written.
            "fast_shutdown": False,
        }
    )
    interrupted = False
    recorded = 0
    readback_samples = []
    try:
        import numpy as np
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
        from isaacsim.core.utils.stage import open_stage

        open_stage(str(scene_path))
        world = World(
            physics_dt=package.dt,
            rendering_dt=package.dt,
            stage_units_in_meters=1.0,
        )
        hand = world.scene.add(
            SingleArticulation(
                prim_path=mapping.get(
                    "robot_articulation_prim", mapping["hand_articulation_prim"]
                ),
                name="replay_robot",
            )
        )
        obj = world.scene.add(
            SingleRigidPrim(
                prim_path=mapping["object_prim"],
                name="replay_object",
                reset_xform_properties=False,
            )
        )

        camera = None
        if args.record_dir is not None:
            from isaacsim.sensors.camera import Camera
            from pxr import UsdGeom
            import omni.usd

            camera_paths = [
                str(prim.GetPath())
                for prim in omni.usd.get_context().get_stage().Traverse()
                if prim.IsA(UsdGeom.Camera)
            ]
            if not camera_paths:
                raise RuntimeError("Imported scene has no camera for recording.")
            args.record_dir.mkdir(parents=True, exist_ok=True)
            camera = Camera(
                prim_path=camera_paths[0], resolution=(args.width, args.height)
            )

        world.reset()
        if camera is not None:
            camera.initialize()
        joint_names = package.driven_joint_names
        available_dof_names = list(hand.dof_names or [])
        missing = [name for name in joint_names if name not in available_dof_names]
        root_world_x_name = (
            package.root_joint_names[0] if package.root_joint_names else None
        )
        allowed_missing = [root_world_x_name] if root_world_x_name else []
        if missing not in ([], allowed_missing):
            raise RuntimeError(
                "Isaac articulation is missing joints: "
                + ", ".join(missing)
                + "; available DOFs: "
                + ", ".join(available_dof_names)
            )
        root_x_is_world_transform = bool(root_world_x_name) and missing == [
            root_world_x_name
        ]
        runtime_joint_names = (
            joint_names[1:] if root_x_is_world_transform else joint_names
        )
        joint_indices = np.asarray(
            [hand.get_dof_index(name) for name in runtime_joint_names],
            dtype=np.int32,
        )
        if np.any(joint_indices < 0):
            missing = [
                name for name, index in zip(joint_names, joint_indices) if index < 0
            ]
            raise RuntimeError("Isaac articulation is missing joints: " + ", ".join(missing))

        frame_indices = [
            frame
            for frame in range(0, package.frame_count, args.stride)
            if bool(package.trajectory.valid_mask[frame])
        ]
        if args.max_frames is not None:
            frame_indices = frame_indices[: args.max_frames]
        if not frame_indices:
            raise ValueError("No replay frames selected; check --max-frames and --stride.")
        capture_period = 1.0 / args.record_fps
        next_capture_time = 0.0

        print(
            f"Replaying {len(frame_indices)} frames; dt={package.dt:.6f}s; "
            f"joint clips={package.clip_count}, max={package.max_clip_rad:.6g} rad"
        )
        validation_frames = {
            frame_indices[0],
            frame_indices[len(frame_indices) // 2],
            frame_indices[-1],
        }
        while simulation_app.is_running():
            for frame in frame_indices:
                started = time.monotonic()
                expected_hand_state = package.hand_joint_state(frame)
                # Isaac Sim 6 promotes the first world prismatic joint to the
                # articulation root transform instead of exposing it as a DOF.
                if root_x_is_world_transform:
                    root_position, root_orientation = hand.get_world_pose()
                    root_position[0] = expected_hand_state[0]
                    hand.set_world_pose(
                        position=root_position, orientation=root_orientation
                    )
                    runtime_hand_state = expected_hand_state[1:]
                else:
                    runtime_hand_state = expected_hand_state
                hand.set_joint_positions(
                    runtime_hand_state, joint_indices=joint_indices
                )
                obj.set_world_pose(
                    position=package.trajectory.object_position[frame],
                    orientation=package.trajectory.object_quaternion[frame],
                )
                # This is a Level A kinematic replay. Rendering without a
                # physics step prevents gravity/contact from modifying the
                # explicitly authored reference pose.
                world.render()

                if frame in validation_frames:
                    actual_joints = hand.get_joint_positions(
                        joint_indices=joint_indices
                    )
                    if root_x_is_world_transform:
                        actual_root_position, _ = hand.get_world_pose()
                        actual_hand_state = np.concatenate(
                            ([actual_root_position[0]], actual_joints)
                        )
                    else:
                        actual_hand_state = actual_joints
                    actual_position, actual_quaternion = obj.get_world_pose()
                    expected_quaternion = package.trajectory.object_quaternion[frame]
                    quaternion_error = min(
                        np.linalg.norm(actual_quaternion - expected_quaternion),
                        np.linalg.norm(actual_quaternion + expected_quaternion),
                    )
                    readback_samples.append(
                        {
                            "frame": frame,
                            "hand_joint_max_abs_error": float(
                                np.max(
                                    np.abs(
                                        actual_hand_state - expected_hand_state
                                    )
                                )
                            ),
                            "object_position_error_m": float(
                                np.linalg.norm(
                                    actual_position
                                    - package.trajectory.object_position[frame]
                                )
                            ),
                            "object_quaternion_l2_error": float(quaternion_error),
                        }
                    )

                timestamp = float(package.trajectory.timestamps[frame])
                if camera is not None and timestamp + package.dt / 2 >= next_capture_time:
                    from PIL import Image

                    rgba = camera.get_rgba()
                    if rgba is not None and rgba.size:
                        image = np.asarray(rgba)
                        if image.dtype != np.uint8:
                            image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
                        Image.fromarray(image).save(
                            args.record_dir / f"frame_{recorded:06d}.png"
                        )
                        recorded += 1
                        next_capture_time += capture_period
                if args.realtime:
                    time.sleep(max(package.dt * args.stride - (time.monotonic() - started), 0.0))
            if not args.loop:
                break
            next_capture_time = 0.0
    except KeyboardInterrupt:
        interrupted = True
        print("Replay interrupted by user.")
    finally:
        simulation_app.close()

    if args.record_dir is not None and recorded and not interrupted:
        encode_video(args.record_dir.resolve(), args.record_fps)
    if not interrupted:
        max_joint_error = max(
            sample["hand_joint_max_abs_error"] for sample in readback_samples
        )
        max_position_error = max(
            sample["object_position_error_m"] for sample in readback_samples
        )
        max_quaternion_error = max(
            sample["object_quaternion_l2_error"] for sample in readback_samples
        )
        passed = (
            max_joint_error <= 1e-5
            and max_position_error <= 1e-5
            and max_quaternion_error <= 1e-5
        )
        report = {
            "status": "pass" if passed else "fail",
            "isaac_sim_version": importlib.metadata.version("isaacsim"),
            "frames_replayed": len(frame_indices),
            "joint_clip_count": package.clip_count,
            "max_joint_clip_rad": package.max_clip_rad,
            "readback_samples": readback_samples,
        }
        report_path = package.directory / "isaac_replay_report.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        package.manifest["isaac_sim"].update(
            {
                "version": report["isaac_sim_version"],
                "state_readback_validation_status": report["status"],
            }
        )
        package.manifest_path.write_text(
            json.dumps(package.manifest, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Saved Isaac readback report: {report_path} ({report['status']})")
        print("Replay finished.")


if __name__ == "__main__":
    main()
