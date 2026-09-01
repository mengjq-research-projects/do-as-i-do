#!/usr/bin/env python3
"""Render and compose an event-aligned original/retargeting comparison video.

The source MP4 is the master clock. For each source frame, a monotonic
piecewise-linear mapping selects one post-warmup state from an existing MJWP
trajectory. Rendering is offline through MuJoCo; IK and physics optimization
are never re-run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import cv2
import mujoco
import numpy as np

from replay_viser import _load_run_config, _top_down_camera_preset, load_qpos


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
    anchors = np.asarray(pairs, dtype=np.int64)
    if anchors.shape[0] < 2:
        raise argparse.ArgumentTypeError("At least two anchors are required.")
    if np.any(np.diff(anchors[:, 0]) <= 0):
        raise argparse.ArgumentTypeError("Source anchors must strictly increase.")
    if np.any(np.diff(anchors[:, 1]) < 0):
        raise argparse.ArgumentTypeError("Replay anchors must be monotonic.")
    return anchors


def _parse_affine(text: str) -> np.ndarray:
    try:
        values = [float(value.strip()) for value in text.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Affine transform must contain six comma-separated numbers."
        ) from exc
    if len(values) != 6:
        raise argparse.ArgumentTypeError(
            "Affine transform must be m00,m01,m02,m10,m11,m12."
        )
    return np.asarray(values, dtype=np.float32).reshape(2, 3)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-video", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--anchors", type=_parse_anchors, required=True)
    parser.add_argument("--panel-width", type=int, default=1280)
    parser.add_argument("--panel-height", type=int, default=720)
    parser.add_argument("--fov", type=float, default=50.0)
    parser.add_argument(
        "--visual-style",
        choices=("viser", "mujoco"),
        default="viser",
        help=(
            "Use the light hand-body/green-fingertip presentation colors from "
            "the Viser replay, or retain MuJoCo's darker default lighting."
        ),
    )
    parser.add_argument(
        "--background-image",
        help=(
            "Optional clean-plate image for the retargeting panel. When set, "
            "only the rendered hand and object are composited over this image."
        ),
    )
    parser.add_argument(
        "--foreground-affine",
        type=_parse_affine,
        default=np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float32),
        help="2D render-to-background affine: m00,m01,m02,m10,m11,m12.",
    )
    parser.add_argument(
        "--shadow-opacity",
        type=float,
        default=0.12,
        help="Soft presentation shadow opacity when a background is used.",
    )
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", default="medium")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _draw_label(image: np.ndarray, text: str, *, align_right: bool = False) -> None:
    height, width = image.shape[:2]
    del height
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (width, 52), (0, 0, 0), thickness=-1)
    cv2.addWeighted(overlay, 0.55, image, 0.45, 0.0, dst=image)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.78
    thickness = 2
    (text_width, _), _ = cv2.getTextSize(text, font, scale, thickness)
    x = width - text_width - 18 if align_right else 18
    cv2.putText(
        image,
        text,
        (x, 35),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def _apply_viser_visual_style(model: mujoco.MjModel) -> None:
    """Approximate the presentation material used by the Viser replay.

    Viser displays the hand's visual meshes with a bright, mostly unlit
    material.  MuJoCo's default headlight makes the same meshes appear dark
    blue-grey from this top-down view.  Keep the green elastomer fingertips,
    whiten the remaining group-1 hand visuals, and use balanced ambient light
    so mesh shape remains visible instead of clipping to flat white.
    """
    for geom_id in range(model.ngeom):
        if int(model.geom_group[geom_id]) != 1:
            continue
        geom_name = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        )
        if "elastomer" not in geom_name.lower():
            model.geom_rgba[geom_id, :3] = 1.0

    model.vis.headlight.ambient[:] = 0.55
    model.vis.headlight.diffuse[:] = 0.45
    model.vis.headlight.specular[:] = 0.10


def _geom_name(model: mujoco.MjModel, geom_id: int) -> str:
    return mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        geom_id,
    ) or ""


def _prepare_background_compositing(
    model: mujoco.MjModel,
) -> np.ndarray:
    """Hide display-only supports and return hand/object visual geom IDs."""
    foreground_ids: list[int] = []
    for geom_id in range(model.ngeom):
        name = _geom_name(model, geom_id).lower()
        if "pedestal" in name or "support" in name:
            model.geom_group[geom_id] = 5
            continue
        if int(model.geom_group[geom_id]) == 1 or name.endswith("_object_visual"):
            foreground_ids.append(geom_id)
    if not foreground_ids:
        raise SystemExit("No hand/object visual geoms found for compositing.")
    return np.asarray(foreground_ids, dtype=np.int32)


def _composite_over_background(
    render_rgb: np.ndarray,
    segmentation: np.ndarray,
    foreground_geom_ids: np.ndarray,
    background_rgb: np.ndarray,
    affine: np.ndarray,
    shadow_opacity: float,
) -> np.ndarray:
    height, width = render_rgb.shape[:2]
    is_geom = segmentation[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
    foreground = is_geom & np.isin(
        segmentation[..., 0],
        foreground_geom_ids,
    )
    alpha = foreground.astype(np.float32)
    premultiplied = render_rgb.astype(np.float32) * alpha[..., None]
    warped_alpha = cv2.warpAffine(
        alpha,
        affine,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    warped_premultiplied = cv2.warpAffine(
        premultiplied,
        affine,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    warped_alpha = np.clip(warped_alpha, 0.0, 1.0)

    background = background_rgb.astype(np.float32)
    if shadow_opacity > 0.0:
        shadow_transform = np.asarray(
            [[1, 0, 7], [0, 1, 10]],
            dtype=np.float32,
        )
        shadow = cv2.warpAffine(
            warped_alpha,
            shadow_transform,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        shadow = cv2.GaussianBlur(shadow, (0, 0), sigmaX=7.0, sigmaY=7.0)
        shadow *= 1.0 - warped_alpha
        background *= (
            1.0 - float(shadow_opacity) * np.clip(shadow, 0.0, 1.0)
        )[..., None]

    composited = (
        warped_premultiplied
        + background * (1.0 - warped_alpha[..., None])
    )
    return np.clip(composited, 0, 255).astype(np.uint8)


def main() -> None:
    args = _parse_args()
    source_path = Path(args.source_video).resolve()
    run_dir = Path(args.run_dir).resolve()
    output_path = Path(args.output).resolve()
    background_path = (
        Path(args.background_image).resolve() if args.background_image else None
    )
    scene_path = run_dir / "scene.xml"
    traj_path = run_dir / "trajectory_mjwp.npz"
    required_paths = [source_path, scene_path, traj_path]
    if background_path is not None:
        required_paths.append(background_path)
    for path in required_paths:
        if not path.is_file():
            raise SystemExit(f"Not found: {path}")
    if args.panel_width <= 0 or args.panel_height <= 0:
        raise SystemExit("Panel dimensions must be positive.")

    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise SystemExit(f"Could not open source video: {source_path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    source_frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    source_width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    source_height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if source_fps <= 0.0 or source_frame_count <= 0:
        raise SystemExit("Source video has invalid FPS or frame count.")

    cfg = _load_run_config(run_dir / "config.yaml")
    warmup_steps = int(cfg.get("warmup_steps", 0) or 0)
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    if args.visual_style == "viser":
        _apply_viser_visual_style(model)
    foreground_geom_ids = None
    background_rgb = None
    if background_path is not None:
        if not 0.0 <= args.shadow_opacity <= 1.0:
            raise SystemExit("--shadow-opacity must lie in [0, 1].")
        foreground_geom_ids = _prepare_background_compositing(model)
        background_bgr = cv2.imread(str(background_path), cv2.IMREAD_COLOR)
        if background_bgr is None:
            raise SystemExit(f"Could not read background image: {background_path}")
        background_bgr = cv2.resize(
            background_bgr,
            (args.panel_width, args.panel_height),
            interpolation=cv2.INTER_AREA,
        )
        background_rgb = cv2.cvtColor(background_bgr, cv2.COLOR_BGR2RGB)
    data = mujoco.MjData(model)
    qpos = load_qpos(traj_path, model.nq)
    start = min(warmup_steps, max(0, len(qpos) - 1))
    replay_qpos = qpos[start:]

    anchors = args.anchors
    if anchors[0, 0] != 0 or anchors[-1, 0] != source_frame_count - 1:
        raise SystemExit(
            f"Anchors must include source frames 0 and {source_frame_count - 1}."
        )
    if anchors[0, 1] < 0 or anchors[-1, 1] >= len(replay_qpos):
        raise SystemExit(
            f"Replay anchors must lie in [0, {len(replay_qpos) - 1}]."
        )
    replay_indices = np.rint(
        np.interp(
            np.arange(source_frame_count, dtype=np.float64),
            anchors[:, 0],
            anchors[:, 1],
        )
    ).astype(np.int64)

    keypoints_path = _keypoints_path_for_run(run_dir)
    camera_position, camera_target, camera_up = _top_down_camera_preset(
        qpos,
        start,
        keypoints_path,
        vertical_fov_deg=args.fov,
        aspect_ratio=float(args.panel_width) / float(args.panel_height),
    )
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = camera_target
    camera.distance = float(np.linalg.norm(camera_position - camera_target))
    # Canonical world -Z view; azimuth=90 keeps world +Y at the top of frame.
    camera.azimuth = 90.0
    camera.elevation = -89.9

    model.vis.global_.offwidth = max(
        int(model.vis.global_.offwidth), args.panel_width
    )
    model.vis.global_.offheight = max(
        int(model.vis.global_.offheight), args.panel_height
    )
    renderer = mujoco.Renderer(
        model,
        height=args.panel_height,
        width=args.panel_width,
    )
    scene_options = mujoco.MjvOption()
    mujoco.mjv_defaultOption(scene_options)
    # Match replay_viser.py: presentation replay hides group-3 collision
    # decomposition meshes (and all higher diagnostic-only groups).
    scene_options.geomgroup[3:] = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_width = args.panel_width * 2
    final_height = args.panel_height
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{final_width}x{final_height}",
        "-r",
        f"{source_fps:.12g}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        args.preset,
        "-crf",
        str(args.crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    encoder = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    encoded_frames = 0
    try:
        for source_index, replay_index in enumerate(replay_indices):
            ok, source_bgr = capture.read()
            if not ok:
                raise RuntimeError(
                    f"Source decode stopped at frame {source_index}/{source_frame_count}."
                )
            source_rgb = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2RGB)
            source_rgb = cv2.resize(
                source_rgb,
                (args.panel_width, args.panel_height),
                interpolation=cv2.INTER_AREA,
            )

            data.qpos[:] = replay_qpos[int(replay_index)]
            mujoco.mj_forward(model, data)
            renderer.update_scene(
                data,
                camera=camera,
                scene_option=scene_options,
            )
            retarget_rgb = renderer.render().copy()
            if background_rgb is not None and foreground_geom_ids is not None:
                renderer.enable_segmentation_rendering()
                segmentation = renderer.render().copy()
                renderer.disable_segmentation_rendering()
                retarget_rgb = _composite_over_background(
                    retarget_rgb,
                    segmentation,
                    foreground_geom_ids,
                    background_rgb,
                    args.foreground_affine,
                    args.shadow_opacity,
                )

            _draw_label(source_rgb, "Original ego video")
            _draw_label(
                retarget_rgb,
                "Retargeting data-id 5 (event-aligned)",
                align_right=True,
            )
            composed = np.concatenate((source_rgb, retarget_rgb), axis=1)
            composed[:, args.panel_width - 2 : args.panel_width + 2] = 255
            if encoder.stdin is None:
                raise RuntimeError("FFmpeg stdin is unavailable.")
            encoder.stdin.write(np.ascontiguousarray(composed).tobytes())
            encoded_frames += 1
            if encoded_frames == 1 or encoded_frames % 15 == 0:
                print(
                    f"Rendered {encoded_frames}/{source_frame_count} frames "
                    f"(source={source_index}, replay={int(replay_index)})",
                    flush=True,
                )
    finally:
        capture.release()
        renderer.close()
        if encoder.stdin is not None:
            encoder.stdin.close()

    return_code = encoder.wait()
    if return_code != 0:
        raise SystemExit(f"FFmpeg failed with exit code {return_code}.")
    if encoded_frames != source_frame_count:
        raise SystemExit(
            f"Encoded {encoded_frames} frames; expected {source_frame_count}."
        )

    manifest = {
        "schema_version": "aligned-comparison.v2",
        "source_video": str(source_path),
        "source_video_sha256": _sha256(source_path),
        "source_width": source_width,
        "source_height": source_height,
        "source_frame_count": source_frame_count,
        "source_fps": source_fps,
        "duration_seconds": source_frame_count / source_fps,
        "retargeting_run": str(run_dir),
        "retargeting_trajectory": str(traj_path),
        "warmup_steps_skipped": start,
        "post_warmup_replay_frames": len(replay_qpos),
        "anchors": anchors.tolist(),
        "replay_indices": replay_indices.tolist(),
        "camera_mode": "top-down",
        "camera_position": camera_position.tolist(),
        "camera_look_at": camera_target.tolist(),
        "camera_up": camera_up.tolist(),
        "vertical_fov_degrees": args.fov,
        "visual_style": args.visual_style,
        "hand_visual_color": (
            "white body with original green elastomer fingertips"
            if args.visual_style == "viser"
            else "scene.xml material under MuJoCo default lighting"
        ),
        "background_image": (
            str(background_path) if background_path is not None else None
        ),
        "background_image_sha256": (
            _sha256(background_path) if background_path is not None else None
        ),
        "foreground_affine": args.foreground_affine.tolist(),
        "shadow_opacity": args.shadow_opacity if background_path else None,
        "render_backend": "MuJoCo via remote software OpenGL/Xvfb",
        "output_video": str(output_path),
        "output_video_sha256": _sha256(output_path),
        "output_width": final_width,
        "output_height": final_height,
        "output_frame_count": encoded_frames,
        "output_fps": source_fps,
    }
    manifest_path = output_path.with_suffix(".json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"COMPARISON_COMPLETE {output_path}", flush=True)
    print(f"MANIFEST {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
