#!/usr/bin/env python3
"""Capture deterministic, GUI-free frames from a Viser retargeting replay.

This utility renders an already-optimized ``trajectory_mjwp.npz``. It never
runs IK or physics optimization. A browser client is still required because
Viser renders with WebGL; ``ClientHandle.get_render()`` returns only the canvas,
without the GUI.

Example, from ``retargeting/``::

    ./run_pipeline.sh capture-replay \
      --run-dir outputs/sharpa/right/mug_pickplace_10s/5 \
      --camera-mode top-down \
      --source-frame-count 150 \
      --source-fps 15 \
      --anchors 0:0,20:100,30:200,40:300,50:392,60:471,100:706,110:785,120:850,130:942,140:1020,149:1099 \
      --output-dir /tmp/mug_retarget_frames \
      --port 8084

The anchor coordinates are ``source_frame:post_warmup_replay_frame`` pairs.
They define a monotonic piecewise-linear time warp.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import imageio.v3 as iio
import mujoco
import numpy as np

from replay_viser import (
    _load_run_config,
    _top_down_camera_preset,
    load_qpos,
)
from retargeting.utils import viser_viewer


def _parse_anchors(text: str) -> np.ndarray:
    pairs: list[tuple[int, int]] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            source_text, replay_text = item.split(":", maxsplit=1)
            pairs.append((int(source_text), int(replay_text)))
        except (TypeError, ValueError) as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid anchor {item!r}; expected SOURCE_FRAME:REPLAY_FRAME."
            ) from exc
    if len(pairs) < 2:
        raise argparse.ArgumentTypeError("At least two time-warp anchors are required.")
    anchors = np.asarray(pairs, dtype=np.int64)
    if np.any(np.diff(anchors[:, 0]) <= 0):
        raise argparse.ArgumentTypeError("Source-frame anchors must be strictly increasing.")
    if np.any(np.diff(anchors[:, 1]) < 0):
        raise argparse.ArgumentTypeError("Replay-frame anchors must be monotonic.")
    return anchors


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Completed retargeting run.")
    parser.add_argument("--scene", default=None, help="Optional scene.xml override.")
    parser.add_argument("--traj", default=None, help="Optional trajectory NPZ override.")
    parser.add_argument("--output-dir", required=True, help="Frame output directory.")
    parser.add_argument("--port", type=int, default=8084, help="Temporary Viser port.")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fov", type=float, default=50.0, help="Vertical FOV in degrees.")
    parser.add_argument(
        "--camera-mode",
        choices=("top-down",),
        default="top-down",
        help="Capture currently supports the validated canonical top-down view.",
    )
    parser.add_argument("--source-frame-count", type=int, required=True)
    parser.add_argument("--source-fps", type=float, required=True)
    parser.add_argument(
        "--anchors",
        type=_parse_anchors,
        required=True,
        help="Comma-separated SOURCE_FRAME:REPLAY_FRAME pairs.",
    )
    parser.add_argument(
        "--format",
        choices=("png", "jpeg"),
        default="png",
        dest="image_format",
    )
    parser.add_argument(
        "--client-timeout",
        type=float,
        default=90.0,
        help="Seconds to wait for a browser WebGL client.",
    )
    parser.add_argument(
        "--settle-time",
        type=float,
        default=0.05,
        help="Seconds to let WebGL draw after each scene update.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing frames already present in output-dir.",
    )
    return parser.parse_args()


def _keypoints_path_for_run(run_dir: Path) -> Path | None:
    try:
        data_id = run_dir.name
        task = run_dir.parent.name
        embodiment = run_dir.parent.parent.name
        outputs_root = run_dir.parents[3]
    except IndexError:
        return None
    candidate = (
        outputs_root
        / "mano"
        / embodiment
        / task
        / data_id
        / "trajectory_keypoints.npz"
    )
    return candidate if candidate.is_file() else None


def _capture_indices(
    source_frame_count: int,
    anchors: np.ndarray,
    replay_frame_count: int,
) -> np.ndarray:
    if source_frame_count <= 0:
        raise ValueError("--source-frame-count must be positive.")
    if anchors[0, 0] != 0 or anchors[-1, 0] != source_frame_count - 1:
        raise ValueError(
            "Time-warp anchors must include source frames 0 and "
            f"{source_frame_count - 1}."
        )
    if anchors[0, 1] < 0 or anchors[-1, 1] >= replay_frame_count:
        raise ValueError(
            f"Replay anchors must lie in [0, {replay_frame_count - 1}]."
        )
    source_frames = np.arange(source_frame_count, dtype=np.float64)
    mapped = np.rint(
        np.interp(source_frames, anchors[:, 0], anchors[:, 1])
    ).astype(np.int64)
    if np.any(np.diff(mapped) < 0):
        raise ValueError("Generated replay-frame mapping is not monotonic.")
    return mapped


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    args = _parse_args()

    run_dir = Path(args.run_dir).resolve()
    scene_path = Path(args.scene).resolve() if args.scene else run_dir / "scene.xml"
    traj_path = (
        Path(args.traj).resolve()
        if args.traj
        else run_dir / "trajectory_mjwp.npz"
    )
    output_dir = Path(args.output_dir).resolve()
    for path in (scene_path, traj_path):
        if not path.is_file():
            raise SystemExit(f"Not found: {path}")

    extension = ".png" if args.image_format == "png" else ".jpg"
    if output_dir.exists() and not args.overwrite:
        existing = next(output_dir.glob(f"frame_*{extension}"), None)
        if existing is not None:
            raise SystemExit(
                f"Capture frames already exist in {output_dir}; use --overwrite."
            )
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = _load_run_config(run_dir / "config.yaml")
    warmup_steps = int(cfg.get("warmup_steps", 0) or 0)
    spec = mujoco.MjSpec.from_file(str(scene_path))
    model = spec.compile()
    data = mujoco.MjData(model)
    qpos = load_qpos(traj_path, model.nq)
    start = min(warmup_steps, max(0, len(qpos) - 1))
    replay_frame_count = len(qpos) - start
    indices = _capture_indices(
        args.source_frame_count,
        args.anchors,
        replay_frame_count,
    )

    keypoints_path = _keypoints_path_for_run(run_dir)
    camera_position, camera_target, camera_up = _top_down_camera_preset(
        qpos,
        start,
        keypoints_path,
        vertical_fov_deg=args.fov,
        aspect_ratio=float(args.width) / float(args.height),
    )

    viser_viewer.init_viser(app_name="retargeting-capture", port=args.port)
    server = viser_viewer._get_server()
    body_ids = viser_viewer.build_and_log_scene_from_spec(
        spec,
        model,
        xml_path=scene_path,
        build_ref=False,
        build_gui=False,
    )
    # Match replay_viser.py: collision decomposition seals hollow mugs.
    for handle in viser_viewer._STATE.collision_geom_handles:
        handle.visible = False

    def apply_camera(client) -> None:
        client.camera.position = camera_position
        client.camera.look_at = camera_target
        client.camera.up_direction = camera_up
        client.camera.fov = np.deg2rad(args.fov)

    @server.on_client_connect
    def _set_camera(client) -> None:
        apply_camera(client)

    def show_frame(display_index: int) -> None:
        data.qpos[:] = qpos[start + int(display_index)]
        mujoco.mj_kinematics(model, data)
        viser_viewer.log_frame(
            data,
            sim_time=0.0,
            viewer_body_entity_and_ids=body_ids,
            data_ref=None,
            record=False,
        )

    show_frame(int(indices[0]))
    print(f"Capture server: http://localhost:{args.port}")
    print(f"Scene:          {scene_path}")
    print(f"Trajectory:     {traj_path}")
    print(f"Warmup skipped: {start} frames")
    print(f"Replay frames:  {replay_frame_count}")
    print(f"Output frames:  {len(indices)} at {args.source_fps:g} FPS")
    print(
        "Top-down camera: position="
        f"{np.round(camera_position, 4).tolist()}, "
        f"look_at={np.round(camera_target, 4).tolist()}, "
        f"up={np.round(camera_up, 4).tolist()}"
    )
    print("Waiting for a browser WebGL client...")

    deadline = time.monotonic() + args.client_timeout
    client = None
    while time.monotonic() < deadline:
        clients = server.get_clients()
        if clients:
            client = clients[sorted(clients)[0]]
            break
        time.sleep(0.1)
    if client is None:
        raise SystemExit(
            f"No browser connected within {args.client_timeout:g} seconds."
        )

    apply_camera(client)
    client.flush()
    time.sleep(0.25)
    # A freshly-connected browser can answer its first get_render request before
    # the camera messages have affected the WebGL canvas. Prime the round trip
    # twice and discard those images so output frame zero is deterministic.
    for _ in range(2):
        apply_camera(client)
        show_frame(int(indices[0]))
        client.flush()
        client.get_render(
            height=args.height,
            width=args.width,
            transport_format="jpeg",
        )
        time.sleep(0.05)
    for output_index, replay_index in enumerate(indices):
        show_frame(int(replay_index))
        client.flush()
        # Scene messages and get_render requests share a websocket, but Three.js
        # applies a scene mutation on its next animation frame. Without this
        # short wait, fast captures can repeat the preceding trajectory frame.
        time.sleep(max(0.0, args.settle_time))
        image = client.get_render(
            height=args.height,
            width=args.width,
            transport_format=args.image_format,
        )
        if image.ndim == 3 and image.shape[-1] == 4:
            image = image[..., :3]
        output_path = output_dir / f"frame_{output_index:06d}{extension}"
        iio.imwrite(output_path, image)
        if output_index == 0 or (output_index + 1) % 10 == 0:
            print(
                f"Captured {output_index + 1}/{len(indices)} "
                f"(replay frame {int(replay_index)})"
            )

    manifest = {
        "schema_version": "retargeting-capture.v1",
        "run_dir": str(run_dir),
        "scene": str(scene_path),
        "trajectory": str(traj_path),
        "warmup_steps": start,
        "replay_frame_count": replay_frame_count,
        "source_frame_count": args.source_frame_count,
        "source_fps": args.source_fps,
        "width": args.width,
        "height": args.height,
        "camera_mode": args.camera_mode,
        "camera_position": camera_position.tolist(),
        "camera_look_at": camera_target.tolist(),
        "camera_up": camera_up.tolist(),
        "vertical_fov_degrees": args.fov,
        "anchors": args.anchors.tolist(),
        "replay_indices": indices.tolist(),
        "image_format": args.image_format,
        "settle_time_seconds": args.settle_time,
    }
    (output_dir / "capture_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"CAPTURE_COMPLETE {output_dir}")


if __name__ == "__main__":
    main()
