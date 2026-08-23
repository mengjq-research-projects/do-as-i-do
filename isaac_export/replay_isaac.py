#!/usr/bin/env python3
"""Replay a Level A trajectory in Isaac Sim."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from isaac_export.package import load_replay_package
from isaac_export.runtime import isaac_sim_version


# UR3e's conventional folded home pose.  An all-zero UR arm extends through
# the tabletop in this dual-arm layout, so the side without a trajectory must
# be held here instead.  The same joint-space pose works for either side
# because their mirrored placement is encoded in the base transforms.
IDLE_ARM_STANDBY_QPOS = (
    -1.5707963267948966,
    -1.5707963267948966,
    1.5707963267948966,
    -1.5707963267948966,
    -1.5707963267948966,
    0.0,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--scene", type=Path, help="Defaults to PACKAGE/scene.usd")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--realtime", action="store_true")
    playback_mode = parser.add_mutually_exclusive_group()
    playback_mode.add_argument(
        "--loop",
        action="store_true",
        help="Loop until the window closes or Ctrl+C (the GUI default).",
    )
    playback_mode.add_argument(
        "--once",
        action="store_true",
        help="Play one pass and exit, including in GUI mode.",
    )
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


def prepare_record_directory(record_dir: Path) -> None:
    """Remove only recorder-owned artifacts so repeated runs cannot mix frames."""
    record_dir.mkdir(parents=True, exist_ok=True)
    for frame_path in record_dir.glob("frame_*.png"):
        frame_path.unlink()
    (record_dir / "replay.mp4").unlink(missing_ok=True)


def compute_replay_camera_pose(package) -> tuple:
    """Frame the valid hand/object workspace with a stable three-quarter view."""
    import numpy as np

    valid_mask = np.asarray(package.trajectory.valid_mask, dtype=bool)
    object_workspace = package.trajectory.object_position[valid_mask]
    # Level A packages contain a free hand root trajectory.  The robot-scene
    # package instead contains six UR3e joint positions, so there is no
    # hand_root_position field; use the tool workspace and a wider framing
    # distance that also includes the fixed robot base.
    hand_workspace = getattr(package.trajectory, "hand_root_position", None)
    if hand_workspace is not None:
        workspace = np.concatenate(
            (hand_workspace[valid_mask], object_workspace), axis=0
        )
    else:
        workspace = object_workspace
    center = (workspace.min(axis=0) + workspace.max(axis=0)) / 2.0
    radius = float(np.max(np.linalg.norm(workspace - center, axis=1)))
    distance = max(0.9 if hand_workspace is None else 0.45, radius * 3.2)
    eye = center + np.asarray([distance, -distance, distance * 0.7])
    return eye, center


def should_loop_replay(args: argparse.Namespace) -> bool:
    """Keep GUI viewing sessions alive; batch and recording runs stay finite."""
    return bool(
        args.loop
        or (
            not args.headless
            and args.record_dir is None
            and not args.once
        )
    )


def robot_sides(package) -> tuple[str, str] | tuple[None, None]:
    """Return (driven, idle) sides for a dual-robot deployment package."""
    if not package.is_robot_scene:
        return None, None
    driven = package.manifest["robot"]["embodiment"]
    if driven not in {"left", "right"}:
        raise ValueError(f"Unsupported robot embodiment: {driven!r}")
    return driven, "left" if driven == "right" else "right"


def idle_standby_state(dof_names, reset_state, side: str, np):
    """Replace only the idle UR3e arm DOFs with its folded standby pose."""
    arm_names = (
        f"{side}_shoulder_pan_joint",
        f"{side}_shoulder_lift_joint",
        f"{side}_elbow_joint",
        f"{side}_wrist_1_joint",
        f"{side}_wrist_2_joint",
        f"{side}_wrist_3_joint",
    )
    indices_by_name = {name: index for index, name in enumerate(dof_names)}
    missing = [name for name in arm_names if name not in indices_by_name]
    if missing:
        raise RuntimeError(
            "Idle articulation is missing UR3e joints: " + ", ".join(missing)
        )
    standby = np.asarray(reset_state, dtype=np.float32).copy()
    standby[
        np.asarray([indices_by_name[name] for name in arm_names], dtype=np.int32)
    ] = np.asarray(IDLE_ARM_STANDBY_QPOS, dtype=np.float32)
    return standby


def set_kinematic_articulation_state(articulation, positions, joint_indices, np):
    """Teleport an articulation and clear dynamics that could move it again."""
    articulation.set_joint_positions(positions, joint_indices=joint_indices)
    articulation.set_joint_velocities(
        np.zeros_like(positions), joint_indices=joint_indices
    )
    articulation.set_linear_velocity(np.zeros(3, dtype=np.float32))
    articulation.set_angular_velocity(np.zeros(3, dtype=np.float32))


def save_readback_report(package, frames_replayed: int, readback_samples: list) -> dict:
    """Persist validation before Isaac shutdown can terminate the interpreter."""
    max_joint_error = max(
        sample["hand_joint_max_abs_error"] for sample in readback_samples
    )
    max_position_error = max(
        sample["object_position_error_m"] for sample in readback_samples
    )
    max_quaternion_error = max(
        sample["object_quaternion_l2_error"] for sample in readback_samples
    )
    max_idle_joint_error = max(
        (sample.get("idle_joint_max_abs_error", 0.0) for sample in readback_samples),
        default=0.0,
    )
    max_object_visual_error = max(
        (
            sample.get("object_visual_position_error_m", 0.0)
            for sample in readback_samples
        ),
        default=0.0,
    )
    passed = (
        max_joint_error <= 1e-5
        and max_idle_joint_error <= 1e-5
        and max_position_error <= 1e-5
        and max_quaternion_error <= 1e-5
        and max_object_visual_error <= 1e-5
    )
    report = {
        "status": "pass" if passed else "fail",
        "isaac_sim_version": isaac_sim_version(),
        "frames_replayed": frames_replayed,
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
    print(f"Saved Isaac readback report: {report_path} ({report['status']})", flush=True)
    return report


def main() -> None:
    args = parse_args()
    if args.stride < 1:
        raise SystemExit("--stride must be at least 1.")
    package = load_replay_package(args.package_dir)
    mapping = load_scene_mapping(package.directory)
    scene_path = (args.scene or package.directory / "scene.usd").resolve()
    if not scene_path.is_file():
        raise SystemExit(f"Scene does not exist: {scene_path}")
    if args.loop and args.record_dir is not None:
        raise SystemExit("--loop and --record-dir cannot be used together.")
    loop_replay = should_loop_replay(args)

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
            # Reports and videos are persisted before close(), so fast shutdown
            # cannot discard them and avoids Isaac 5's long teardown path.
            "fast_shutdown": True,
        }
    )
    interrupted = False
    recorded = 0
    readback_samples = []
    try:
        import numpy as np
        from isaacsim.core.api import World
        from isaacsim.core.prims import (
            SingleArticulation,
            SingleRigidPrim,
            SingleXFormPrim,
        )
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
        driven_side, idle_side = robot_sides(package)
        idle_robot = None
        if idle_side is not None:
            articulation_paths = mapping.get("articulation_prims_by_side", {})
            idle_prim_path = articulation_paths.get(
                idle_side, f"/World/{idle_side}_base/{idle_side}_base"
            )
            idle_robot = world.scene.add(
                SingleArticulation(
                    prim_path=idle_prim_path,
                    name="replay_idle_robot",
                )
            )
        obj = world.scene.add(
            SingleRigidPrim(
                prim_path=mapping["object_prim"],
                name="replay_object",
                reset_xform_properties=False,
            )
        )
        object_visual = None
        object_visual_path = mapping.get("task_object_appearance", {}).get(
            "xform_prim"
        )
        if object_visual_path:
            object_visual = world.scene.add(
                SingleXFormPrim(
                    prim_path=object_visual_path,
                    name="replay_object_visual",
                    reset_xform_properties=True,
                )
            )

        camera = None
        camera_eye, camera_target = compute_replay_camera_pose(package)
        if args.record_dir is not None:
            from isaacsim.sensors.camera import Camera

            prepare_record_directory(args.record_dir)
            camera_path = "/OmniverseKit_Persp"
            camera = Camera(
                prim_path=camera_path,
                resolution=(args.width, args.height),
            )
        world.reset()
        idle_joint_indices = None
        idle_home_state = None
        if idle_robot is not None:
            idle_home_state = idle_standby_state(
                list(idle_robot.dof_names or []),
                idle_robot.get_joint_positions(),
                idle_side,
                np,
            ).copy()
            idle_joint_indices = np.arange(
                len(idle_home_state), dtype=np.int32
            )
            set_kinematic_articulation_state(
                idle_robot, idle_home_state, idle_joint_indices, np
            )
            print(
                f"Driving {driven_side} robot; holding {idle_side} robot at "
                f"its folded {len(idle_home_state)}-DOF standby state.",
                flush=True,
            )
        if camera is not None:
            from isaacsim.core.utils.viewports import set_camera_view

            camera.initialize()
            set_camera_view(
                camera_eye,
                camera_target,
                camera_prim_path=camera_path,
            )
            simulation_app.update()
        elif not args.headless:
            from isaacsim.core.utils.viewports import set_camera_view

            set_camera_view(camera_eye, camera_target)
            simulation_app.update()
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
            f"joint clips={package.clip_count}, max={package.max_clip_rad:.6g} rad",
            flush=True,
        )
        if loop_replay:
            print(
                "Interactive replay will loop continuously; close the window "
                "or press Ctrl+C to stop.",
                flush=True,
            )
        validation_frames = {
            frame_indices[0],
            frame_indices[len(frame_indices) // 2],
            frame_indices[-1],
        }
        report_saved = False
        while simulation_app.is_running():
            completed_pass = True
            for frame in frame_indices:
                if not simulation_app.is_running():
                    completed_pass = False
                    break
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
                set_kinematic_articulation_state(
                    hand, runtime_hand_state, joint_indices, np
                )
                if idle_robot is not None:
                    set_kinematic_articulation_state(
                        idle_robot, idle_home_state, idle_joint_indices, np
                    )
                obj.set_world_pose(
                    position=package.trajectory.object_position[frame],
                    orientation=package.trajectory.object_quaternion[frame],
                )
                if object_visual is not None:
                    object_visual.set_world_pose(
                        position=package.trajectory.object_position[frame],
                        orientation=package.trajectory.object_quaternion[frame],
                    )
                # Propagate the teleported articulation state through PhysX so
                # link transforms and attached visual meshes reach the renderer.
                # A render-only update reads the authored DOFs back correctly
                # but leaves link geometry at its previous/default pose. Reapply
                # the reference after the synchronization step so gravity and
                # joint dynamics cannot perturb this kinematic replay.
                world.step(render=False)
                if root_x_is_world_transform:
                    root_position, root_orientation = hand.get_world_pose()
                    root_position[0] = expected_hand_state[0]
                    hand.set_world_pose(
                        position=root_position, orientation=root_orientation
                    )
                set_kinematic_articulation_state(
                    hand, runtime_hand_state, joint_indices, np
                )
                if idle_robot is not None:
                    set_kinematic_articulation_state(
                        idle_robot, idle_home_state, idle_joint_indices, np
                    )
                obj.set_world_pose(
                    position=package.trajectory.object_position[frame],
                    orientation=package.trajectory.object_quaternion[frame],
                )
                if object_visual is not None:
                    object_visual.set_world_pose(
                        position=package.trajectory.object_position[frame],
                        orientation=package.trajectory.object_quaternion[frame],
                    )
                world.render()

                if not report_saved and frame in validation_frames:
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
                    sample = {
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
                    if idle_robot is not None:
                        actual_idle_state = idle_robot.get_joint_positions(
                            joint_indices=idle_joint_indices
                        )
                        sample["idle_side"] = idle_side
                        sample["idle_joint_max_abs_error"] = float(
                            np.max(np.abs(actual_idle_state - idle_home_state))
                        )
                    if object_visual is not None:
                        visual_position, _ = object_visual.get_world_pose()
                        sample["object_visual_position_error_m"] = float(
                            np.linalg.norm(visual_position - actual_position)
                        )
                    readback_samples.append(sample)

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
            if completed_pass and not report_saved:
                save_readback_report(
                    package,
                    frames_replayed=len(frame_indices),
                    readback_samples=readback_samples,
                )
                report_saved = True
                if loop_replay:
                    print("Validated first pass; continuing loop.", flush=True)
            if not loop_replay or not simulation_app.is_running():
                break
            next_capture_time = 0.0
        if report_saved and not loop_replay:
            print("Replay finished.", flush=True)
    except KeyboardInterrupt:
        interrupted = True
        print("Replay stopped by user.", flush=True)
    finally:
        try:
            if args.record_dir is not None and recorded and not interrupted:
                encode_video(args.record_dir.resolve(), args.record_fps)
        finally:
            simulation_app.close()


if __name__ == "__main__":
    main()
