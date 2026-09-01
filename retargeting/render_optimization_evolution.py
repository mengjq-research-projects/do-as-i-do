#!/usr/bin/env python3
"""Render a synchronized 2x2 optimization-evolution research demo.

The source ego video is the master clock.  Three saved Retargeting stages are
shown beside it, so this script never runs IK, MJWP, MPC, or reinforcement
learning.  It only renders existing qpos trajectories and composes a video.

For a moving ego source, table-plane homographies are estimated directly from
the source frames.  The same per-frame transform is applied to the clean table
plate and every simulated stage, preserving a fair shared camera motion.  A
larger clean mosaic is reconstructed from the registered source frames so the
camera never exposes reflected or otherwise duplicated border pixels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import warnings
from dataclasses import dataclass
from pathlib import Path

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from render_aligned_comparison import (
    _apply_viser_visual_style,
    _composite_over_background,
    _geom_name,
    _keypoints_path_for_run,
    _parse_affine,
    _parse_anchors,
    _prepare_background_compositing,
)
from replay_viser import _load_run_config, _top_down_camera_preset, load_qpos


DEFAULT_ANCHORS = _parse_anchors(
    "0:0,24:160,30:200,40:280,45:320,50:350,55:392,60:450,"
    "95:640,108:693,114:726,121:819,130:942,144:1060,149:1099"
)
DEFAULT_AFFINE = np.asarray(
    [[0.8432, -0.1315, 227.6], [0.1315, 0.8432, 237.25]],
    dtype=np.float32,
)
MOSAIC_ALPHA_EPS = 1e-4


@dataclass(frozen=True)
class StageSpec:
    label: str
    run_dir: Path
    trajectory_name: str
    timing: str


def _parse_stage(text: str) -> StageSpec:
    """Parse LABEL|RUN_DIR|TRAJECTORY|TIMING."""
    fields = [field.strip() for field in text.split("|")]
    if len(fields) != 4:
        raise argparse.ArgumentTypeError(
            "Stage must be LABEL|RUN_DIR|TRAJECTORY|TIMING."
        )
    label, run_dir, trajectory_name, timing = fields
    if not label or not run_dir or not trajectory_name:
        raise argparse.ArgumentTypeError("Stage fields must not be empty.")
    if timing not in {"physics", "kinematic"}:
        raise argparse.ArgumentTypeError(
            "Stage timing must be 'physics' or 'kinematic'."
        )
    return StageSpec(label, Path(run_dir), trajectory_name, timing)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-video", required=True)
    parser.add_argument("--background-image", required=True)
    parser.add_argument(
        "--mask-root",
        required=True,
        help=(
            "Directory containing frame_NNNNNN_masks/mug.png and optional "
            "right_hand_0.png masks from Reconstruction."
        ),
    )
    parser.add_argument(
        "--stage",
        action="append",
        type=_parse_stage,
        required=True,
        help=(
            "Exactly three LABEL|RUN_DIR|TRAJECTORY|TIMING specs. TIMING is "
            "physics (skip warmup and use anchors) or kinematic (linear map)."
        ),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--anchors", type=_parse_anchors, default=DEFAULT_ANCHORS)
    parser.add_argument("--panel-width", type=int, default=960)
    parser.add_argument("--panel-height", type=int, default=540)
    parser.add_argument("--fov", type=float, default=50.0)
    parser.add_argument(
        "--final-stage-yaw-degrees",
        type=float,
        default=0.0,
        help=(
            "Presentation-only world-Z yaw applied rigidly to the final "
            "stage's object and hand around the object centre. This preserves "
            "their relative contact geometry while matching source-view "
            "handle and approach directions."
        ),
    )
    parser.add_argument(
        "--first-stage-yaw-start-degrees",
        type=float,
        default=0.0,
        help="Presentation yaw for the first simulated stage at source frame 0.",
    )
    parser.add_argument(
        "--first-stage-yaw-end-degrees",
        type=float,
        default=None,
        help=(
            "Presentation yaw for the first simulated stage at the final "
            "source frame; defaults to --first-stage-yaw-start-degrees."
        ),
    )
    parser.add_argument(
        "--final-stage-yaw-start-degrees",
        type=float,
        default=None,
        help=(
            "Presentation yaw for the final simulated stage at source frame "
            "0; defaults to --final-stage-yaw-degrees."
        ),
    )
    parser.add_argument(
        "--final-stage-yaw-end-degrees",
        type=float,
        default=None,
        help=(
            "Presentation yaw for the final simulated stage at the final "
            "source frame; defaults to its start yaw."
        ),
    )
    parser.add_argument(
        "--narrative",
        choices=("measured-optimization", "target-rl"),
        default="measured-optimization",
        help=(
            "measured-optimization describes the saved trajectories literally; "
            "target-rl presents them as visual proxies for the intended future "
            "RL learning progression."
        ),
    )
    parser.add_argument("--reference-frame", type=int, default=75)
    parser.add_argument(
        "--foreground-affine",
        type=_parse_affine,
        default=DEFAULT_AFFINE,
        help=(
            "Render-to-clean-plate affine calibrated at --affine-reference-* "
            "resolution. It is used as a fallback when auto-alignment is off."
        ),
    )
    parser.add_argument("--affine-reference-width", type=int, default=1280)
    parser.add_argument("--affine-reference-height", type=int, default=720)
    parser.add_argument(
        "--auto-align-object",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Refine the global foreground similarity transform from source mug "
            "masks and the final stage's rendered object trajectory."
        ),
    )
    parser.add_argument(
        "--dynamic-camera",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply source-derived table homographies to all simulated panels.",
    )
    parser.add_argument(
        "--homography-smooth-radius",
        type=int,
        default=2,
        help="Temporal radius for light homography smoothing; 0 keeps raw motion.",
    )
    parser.add_argument(
        "--mosaic-padding",
        type=int,
        default=8,
        help="Unobserved guard pixels around the expanded clean mosaic.",
    )
    parser.add_argument(
        "--mosaic-mask-dilation",
        type=int,
        default=18,
        help="Panel-resolution dilation around source hand and mug masks.",
    )
    parser.add_argument(
        "--mosaic-feather",
        type=int,
        default=24,
        help="Feather width when inserting the supplied reference clean plate.",
    )
    parser.add_argument(
        "--mosaic-cleanplate-mode",
        choices=("temporal", "feathered-input"),
        default="temporal",
        help=(
            "temporal trusts the registered multi-frame clean mosaic; "
            "feathered-input inserts --background-image into the reference "
            "view for compatibility with an externally verified clean plate."
        ),
    )
    parser.add_argument(
        "--mosaic-tile-height",
        type=int,
        default=32,
        help="Row tile height used by the robust temporal mosaic reducer.",
    )
    parser.add_argument("--shadow-opacity", type=float, default=0.12)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", default="medium")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_source_frames(
    path: Path,
    width: int,
    height: int,
) -> tuple[list[np.ndarray], float, tuple[int, int]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise SystemExit(f"Could not open source video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    source_size = (
        int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))),
        int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))),
    )
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(
            cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        )
    capture.release()
    if fps <= 0.0 or not frames:
        raise SystemExit("Source video has invalid FPS or no decodable frames.")
    return frames, fps, source_size


def _mask_path(mask_root: Path, frame_index: int, name: str) -> Path:
    return mask_root / f"frame_{frame_index:06d}_masks" / name


def _read_mask(
    mask_root: Path,
    frame_index: int,
    name: str,
    size: tuple[int, int],
) -> np.ndarray | None:
    path = _mask_path(mask_root, frame_index, name)
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    return cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST) > 0


def _feature_mask(
    mask_root: Path,
    frame_index: int,
    size: tuple[int, int],
) -> np.ndarray:
    dynamic = np.zeros((size[1], size[0]), dtype=np.uint8)
    for name in ("mug.png", "right_hand_0.png"):
        mask = _read_mask(mask_root, frame_index, name, size)
        if mask is not None:
            dynamic[mask] = 255
    radius = max(5, int(round(size[0] / 96)))
    kernel_size = radius * 2 + 1
    dynamic = cv2.dilate(
        dynamic,
        np.ones((kernel_size, kernel_size), dtype=np.uint8),
    )
    return 255 - dynamic


def _smooth_homographies(
    homographies: np.ndarray,
    reference_frame: int,
    radius: int,
) -> np.ndarray:
    if radius <= 0:
        return homographies
    radius = int(radius)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    sigma = max(radius / 1.5, 0.8)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(
        homographies,
        ((radius, radius), (0, 0), (0, 0)),
        mode="edge",
    )
    smoothed = np.empty_like(homographies)
    for index in range(len(homographies)):
        smoothed[index] = np.tensordot(
            kernel,
            padded[index : index + len(kernel)],
            axes=(0, 0),
        )
        smoothed[index] /= smoothed[index, 2, 2]
    for index in range(len(smoothed)):
        determinant = float(np.linalg.det(smoothed[index]))
        condition = float(np.linalg.cond(smoothed[index]))
        if (
            not np.isfinite(smoothed[index]).all()
            or not np.isfinite(determinant)
            or abs(determinant) < 1e-8
            or not np.isfinite(condition)
            or condition > 1e8
        ):
            smoothed[index] = homographies[index]
    reference = smoothed[reference_frame]
    if (
        not np.isfinite(reference).all()
        or abs(float(np.linalg.det(reference))) < 1e-8
    ):
        raise SystemExit("Reference-frame homography became singular.")
    reference_inverse = np.linalg.inv(reference)
    smoothed = smoothed @ reference_inverse
    smoothed /= smoothed[:, 2:3, 2:3]
    return smoothed


def _estimate_table_homographies(
    frames: list[np.ndarray],
    mask_root: Path,
    reference_frame: int,
    smooth_radius: int,
) -> tuple[np.ndarray, dict[str, object]]:
    if not 0 <= reference_frame < len(frames):
        raise SystemExit(
            f"Reference frame {reference_frame} outside [0, {len(frames) - 1}]."
        )
    height, width = frames[0].shape[:2]
    size = (width, height)
    cv2.setRNGSeed(0)
    orb = cv2.ORB_create(
        nfeatures=7000,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=19,
        fastThreshold=7,
    )
    reference_gray = cv2.cvtColor(frames[reference_frame], cv2.COLOR_BGR2GRAY)
    reference_keypoints, reference_descriptors = orb.detectAndCompute(
        reference_gray,
        _feature_mask(mask_root, reference_frame, size),
    )
    if reference_descriptors is None or len(reference_keypoints) < 100:
        raise SystemExit("Too few static reference features for camera recovery.")

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    estimates: list[np.ndarray | None] = []
    match_counts: list[int] = []
    inlier_counts: list[int] = []
    rmses: list[float] = []
    for frame_index, frame in enumerate(frames):
        if frame_index == reference_frame:
            estimates.append(np.eye(3, dtype=np.float64))
            match_counts.append(len(reference_keypoints))
            inlier_counts.append(len(reference_keypoints))
            rmses.append(0.0)
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = orb.detectAndCompute(
            gray,
            _feature_mask(mask_root, frame_index, size),
        )
        if descriptors is None or len(keypoints) < 40:
            estimates.append(None)
            match_counts.append(0)
            inlier_counts.append(0)
            rmses.append(float("nan"))
            continue
        pairs = matcher.knnMatch(reference_descriptors, descriptors, k=2)
        matches = []
        for pair in pairs:
            if len(pair) != 2:
                continue
            first, second = pair
            if first.distance < 0.78 * second.distance:
                matches.append(first)
        match_counts.append(len(matches))
        if len(matches) < 40:
            estimates.append(None)
            inlier_counts.append(0)
            rmses.append(float("nan"))
            continue
        source_points = np.float32(
            [reference_keypoints[match.queryIdx].pt for match in matches]
        )
        target_points = np.float32(
            [keypoints[match.trainIdx].pt for match in matches]
        )
        homography, inliers = cv2.findHomography(
            source_points,
            target_points,
            cv2.RANSAC,
            2.5,
            maxIters=5000,
            confidence=0.999,
        )
        if homography is None or inliers is None or int(inliers.sum()) < 30:
            estimates.append(None)
            inlier_counts.append(0)
            rmses.append(float("nan"))
            continue
        homography = np.asarray(homography, dtype=np.float64)
        homography /= homography[2, 2]
        keep = inliers.ravel().astype(bool)
        projected = cv2.perspectiveTransform(
            source_points[keep, None, :], homography
        )[:, 0, :]
        error = np.linalg.norm(projected - target_points[keep], axis=1)
        estimates.append(homography)
        inlier_counts.append(int(keep.sum()))
        rmses.append(float(np.sqrt(np.mean(error**2))))

    valid_indices = [index for index, value in enumerate(estimates) if value is not None]
    if len(valid_indices) < max(2, int(0.9 * len(frames))):
        raise SystemExit(
            f"Camera recovery succeeded on only {len(valid_indices)}/{len(frames)} frames."
        )
    parameters = np.full((len(frames), 8), np.nan, dtype=np.float64)
    for index in valid_indices:
        homography = estimates[index]
        assert homography is not None
        parameters[index] = homography.reshape(-1)[:8]
    frame_axis = np.arange(len(frames), dtype=np.float64)
    valid_axis = np.asarray(valid_indices, dtype=np.float64)
    for column in range(parameters.shape[1]):
        parameters[:, column] = np.interp(
            frame_axis,
            valid_axis,
            parameters[valid_indices, column],
        )
    homographies = np.concatenate(
        (
            parameters,
            np.ones((len(frames), 1), dtype=np.float64),
        ),
        axis=1,
    ).reshape(-1, 3, 3)
    homographies = _smooth_homographies(
        homographies,
        reference_frame,
        smooth_radius,
    )
    for frame_index, homography in enumerate(homographies):
        determinant = float(np.linalg.det(homography))
        condition = float(np.linalg.cond(homography))
        if (
            not np.isfinite(homography).all()
            or not np.isfinite(determinant)
            or abs(determinant) < 1e-8
            or not np.isfinite(condition)
            or condition > 1e8
        ):
            raise SystemExit(
                f"Invalid table homography at source frame {frame_index}."
            )
    finite_rmses = np.asarray([value for value in rmses if np.isfinite(value)])
    metrics: dict[str, object] = {
        "reference_frame": reference_frame,
        "valid_frames": len(valid_indices),
        "interpolated_frames": len(frames) - len(valid_indices),
        "reference_keypoints": len(reference_keypoints),
        "matches_min": int(min(match_counts)),
        "matches_median": float(np.median(match_counts)),
        "inliers_min": int(min(inlier_counts)),
        "inliers_median": float(np.median(inlier_counts)),
        "reprojection_rmse_median_px": float(np.median(finite_rmses)),
        "reprojection_rmse_p95_px": float(np.percentile(finite_rmses, 95)),
        "smooth_radius": int(smooth_radius),
    }
    return homographies, metrics


def _scale_affine(
    affine: np.ndarray,
    reference_size: tuple[int, int],
    output_size: tuple[int, int],
) -> np.ndarray:
    reference_width, reference_height = reference_size
    output_width, output_height = output_size
    source_scale = np.diag(
        [output_width / reference_width, output_height / reference_height, 1.0]
    )
    affine_3x3 = np.vstack((affine, [0.0, 0.0, 1.0]))
    scaled = source_scale @ affine_3x3 @ np.linalg.inv(source_scale)
    return scaled[:2].astype(np.float32)


def _centroid(mask: np.ndarray) -> np.ndarray | None:
    moments = cv2.moments(mask.astype(np.uint8), binaryImage=True)
    if moments["m00"] <= 0.0:
        return None
    return np.asarray(
        [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]],
        dtype=np.float32,
    )


def _transform_point(point: np.ndarray, homography: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(
        point.reshape(1, 1, 2).astype(np.float32),
        homography.astype(np.float64),
    )[0, 0]


def _source_dynamic_mask(
    source_bgr: np.ndarray,
    mask_root: Path,
    frame_index: int,
    size: tuple[int, int],
    dilation: int,
) -> np.ndarray:
    """Return a conservative source mask for the mug, hand, and forearm."""
    width, height = size
    dynamic = np.zeros((height, width), dtype=np.uint8)
    hand = np.zeros((height, width), dtype=np.uint8)
    for name in ("mug.png", "right_hand_0.png"):
        mask = _read_mask(mask_root, frame_index, name, size)
        if mask is not None:
            dynamic[mask] = 255
            if name == "right_hand_0.png":
                hand[mask] = 255

    # Reconstruction's hand mask often ends at the wrist.  Recover the
    # connected skin-coloured forearm without treating the whole grey tabletop
    # as dynamic: retain only components anchored to the hand mask or entering
    # from the lower image boundary, with a conservative component-size cap.
    ycrcb = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2HSV)
    cr = ycrcb[..., 1]
    cb = ycrcb[..., 2]
    hue = hsv[..., 0]
    saturation = hsv[..., 1]
    value = hsv[..., 2]
    skin = (
        (cr >= 135)
        & (cr <= 185)
        & (cb >= 75)
        & (cb <= 145)
        & (saturation >= 22)
        & (value >= 45)
        & ((hue <= 28) | (hue >= 170))
    ).astype(np.uint8)
    skin = cv2.morphologyEx(
        skin,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    )
    hand_anchor = cv2.dilate(
        hand,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)),
    ) > 0
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        skin,
        connectivity=8,
    )
    forearm = np.zeros((height, width), dtype=bool)
    maximum_component_area = int(round(width * height * 0.18))
    for component in range(1, component_count):
        x, y, component_width, component_height, area = stats[component]
        component_mask = labels == component
        anchored_to_hand = bool(np.any(component_mask & hand_anchor))
        enters_from_bottom = (
            y + component_height >= height - 2
            and component_width <= int(round(width * 0.45))
        )
        if (
            80 <= int(area) <= maximum_component_area
            and (anchored_to_hand or enters_from_bottom)
        ):
            forearm |= component_mask
    dynamic[forearm] = 255
    if dilation > 0:
        kernel_size = int(dilation) * 2 + 1
        dynamic = cv2.dilate(
            dynamic,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (kernel_size, kernel_size),
            ),
        )
    return dynamic > 0


def _normalized_warp_rgb(
    image_rgb: np.ndarray,
    alpha: np.ndarray,
    transform: np.ndarray,
    size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Warp premultiplied RGB and normalize away constant-border darkening."""
    alpha_float = alpha.astype(np.float32)
    premultiplied = image_rgb.astype(np.float32) * alpha_float[..., None]
    warped_premultiplied = cv2.warpPerspective(
        premultiplied,
        transform,
        size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    warped_alpha = cv2.warpPerspective(
        alpha_float,
        transform,
        size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    normalized = np.zeros_like(warped_premultiplied)
    np.divide(
        warped_premultiplied,
        np.maximum(warped_alpha[..., None], MOSAIC_ALPHA_EPS),
        out=normalized,
        where=warped_alpha[..., None] > MOSAIC_ALPHA_EPS,
    )
    return normalized, warped_alpha


def _expanded_mosaic_bounds(
    homographies: np.ndarray,
    size: tuple[int, int],
    padding: int,
) -> tuple[np.ndarray, tuple[int, int], list[int]]:
    """Return reference-to-mosaic transform, canvas size, and reference bbox."""
    width, height = size
    corners = np.asarray(
        [[0.0, 0.0], [width - 1.0, 0.0], [width - 1.0, height - 1.0], [0.0, height - 1.0]],
        dtype=np.float32,
    )
    reference_corners = []
    for homography in homographies:
        reference_corners.append(
            cv2.perspectiveTransform(
                corners[None, ...],
                np.linalg.inv(homography),
            )[0]
        )
    stacked = np.concatenate(reference_corners, axis=0)
    lower = np.floor(stacked.min(axis=0)).astype(np.int64) - int(padding)
    upper = np.ceil(stacked.max(axis=0)).astype(np.int64) + int(padding)
    mosaic_width = int(upper[0] - lower[0] + 1)
    mosaic_height = int(upper[1] - lower[1] + 1)
    if mosaic_width < width or mosaic_height < height:
        raise SystemExit("Expanded mosaic bounds are unexpectedly smaller than one frame.")
    reference_to_mosaic = np.asarray(
        [[1.0, 0.0, -float(lower[0])], [0.0, 1.0, -float(lower[1])], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return (
        reference_to_mosaic,
        (mosaic_width, mosaic_height),
        [int(lower[0]), int(lower[1]), int(upper[0]), int(upper[1])],
    )


def _build_expanded_background_mosaic(
    source_frames: list[np.ndarray],
    homographies: np.ndarray,
    mask_root: Path,
    reference_background_rgb: np.ndarray,
    output_path: Path,
    padding: int,
    mask_dilation: int,
    feather: int,
    tile_height: int,
    cleanplate_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Build a clean reference-coordinate mosaic from real source observations.

    Dynamic hand/mug masks are excluded before registration.  A temporal median
    is used in well-observed areas, while low-coverage boundary pixels use the
    valid observation farthest from dynamic content with sharpness as a small
    tie-break.  The supplied high-quality clean plate is feathered into the
    reference-frame area.  No reflected or synthesized pixels are introduced,
    and all per-frame view footprints must be covered.  The default temporal
    mode deliberately does not reinsert an external clean plate, because an
    imperfect plate can contain fixed hand/forearm ghosts.
    """
    if padding < 0 or mask_dilation < 0 or feather < 0 or tile_height <= 0:
        raise SystemExit("Mosaic padding/dilation/feather must be non-negative and tile height positive.")
    height, width = source_frames[0].shape[:2]
    size = (width, height)
    reference_to_mosaic, mosaic_size, bounds = _expanded_mosaic_bounds(
        homographies,
        size,
        padding,
    )
    mosaic_width, mosaic_height = mosaic_size
    frame_count = len(source_frames)
    estimated_stack_bytes = frame_count * mosaic_width * mosaic_height * 4
    if estimated_stack_bytes > 8 * 1024**3:
        raise SystemExit(
            "Expanded mosaic would require more than 8 GiB for registered "
            f"samples ({mosaic_width}x{mosaic_height}x{frame_count}); check "
            "the recovered homographies."
        )
    print(
        f"Building real-source background mosaic {mosaic_width}x{mosaic_height} "
        f"from {frame_count} frames",
        flush=True,
    )

    registered = np.zeros(
        (frame_count, mosaic_height, mosaic_width, 3),
        dtype=np.uint8,
    )
    registered_valid = np.zeros(
        (frame_count, mosaic_height, mosaic_width),
        dtype=bool,
    )
    required = np.zeros((mosaic_height, mosaic_width), dtype=np.uint8)
    best_score = np.full((mosaic_height, mosaic_width), -np.inf, dtype=np.float32)
    best_rgb = np.zeros((mosaic_height, mosaic_width, 3), dtype=np.uint8)
    source_footprint = np.full((height, width), 255, dtype=np.uint8)

    for frame_index, (source_bgr, homography) in enumerate(
        zip(source_frames, homographies, strict=True)
    ):
        source_rgb = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2RGB)
        dynamic = _source_dynamic_mask(
            source_bgr,
            mask_root,
            frame_index,
            size,
            mask_dilation,
        )
        static_valid = (~dynamic).astype(np.uint8) * 255
        source_alpha = static_valid.astype(np.float32) / 255.0
        source_to_mosaic = reference_to_mosaic @ np.linalg.inv(homography)
        warped_rgb_float, warped_alpha = _normalized_warp_rgb(
            source_rgb,
            source_alpha,
            source_to_mosaic,
            mosaic_size,
        )
        warped_rgb = np.clip(np.rint(warped_rgb_float), 0.0, 255.0).astype(np.uint8)
        warped_valid = warped_alpha > MOSAIC_ALPHA_EPS
        warped_required = cv2.warpPerspective(
            source_footprint,
            source_to_mosaic,
            mosaic_size,
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        required = cv2.max(required, warped_required)
        registered[frame_index] = warped_rgb
        registered_valid[frame_index] = warped_valid

        gray = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.GaussianBlur(
            np.abs(cv2.Laplacian(gray, cv2.CV_32F)),
            (0, 0),
            1.0,
        )
        distance_from_dynamic = cv2.distanceTransform(
            static_valid,
            cv2.DIST_L2,
            3,
        )
        candidate_score = np.minimum(distance_from_dynamic, 128.0) + 0.02 * sharpness
        warped_score_premultiplied = cv2.warpPerspective(
            candidate_score * source_alpha,
            source_to_mosaic,
            mosaic_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
        warped_score = np.zeros_like(warped_score_premultiplied)
        np.divide(
            warped_score_premultiplied,
            np.maximum(warped_alpha, MOSAIC_ALPHA_EPS),
            out=warped_score,
            where=warped_alpha > MOSAIC_ALPHA_EPS,
        )
        take = warped_valid & (warped_score > best_score)
        best_score[take] = warped_score[take]
        best_rgb[take] = warped_rgb[take]
        if frame_index == 0 or (frame_index + 1) % 30 == 0:
            print(
                f"Registered mosaic source {frame_index + 1}/{frame_count}",
                flush=True,
            )

    observation_count = registered_valid.sum(axis=0, dtype=np.uint16)
    required_bool = required > 0
    holes = required_bool & (observation_count == 0)
    hole_count = int(holes.sum())
    if hole_count:
        raise SystemExit(
            f"Expanded mosaic has {hole_count} required pixels without a real static observation."
        )

    mosaic_rgb = np.zeros((mosaic_height, mosaic_width, 3), dtype=np.uint8)
    for y0 in range(0, mosaic_height, tile_height):
        y1 = min(y0 + tile_height, mosaic_height)
        pixels = registered[:, y0:y1].astype(np.float32)
        valid = registered_valid[:, y0:y1, :, None]
        pixels[~np.broadcast_to(valid, pixels.shape)] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            median = np.nanmedian(pixels, axis=0)
        median = np.nan_to_num(median, nan=0.0)
        mosaic_rgb[y0:y1] = np.clip(np.rint(median), 0.0, 255.0).astype(np.uint8)
        print(
            f"Reduced mosaic rows {y1}/{mosaic_height}",
            flush=True,
        )
    low_coverage = required_bool & (observation_count <= 5)
    mosaic_rgb[low_coverage] = best_rgb[low_coverage]

    origin_x = int(round(reference_to_mosaic[0, 2]))
    origin_y = int(round(reference_to_mosaic[1, 2]))
    reference_roi = mosaic_rgb[origin_y : origin_y + height, origin_x : origin_x + width]
    if reference_roi.shape != reference_background_rgb.shape:
        raise SystemExit("Reference clean plate does not fit inside the expanded mosaic.")
    if cleanplate_mode == "feathered-input":
        if feather == 0:
            alpha = np.ones((height, width), dtype=np.float32)
        else:
            x_distance = np.minimum(np.arange(width) + 1, np.arange(width, 0, -1))
            y_distance = np.minimum(np.arange(height) + 1, np.arange(height, 0, -1))
            distance = np.minimum(y_distance[:, None], x_distance[None, :])
            alpha = np.clip(
                distance.astype(np.float32) / float(feather),
                0.0,
                1.0,
            )
            alpha = cv2.GaussianBlur(
                alpha,
                (0, 0),
                max(feather / 6.0, 0.5),
            )
            alpha = np.clip(alpha, 0.0, 1.0)
        blended_reference = (
            reference_roi.astype(np.float32) * (1.0 - alpha[..., None])
            + reference_background_rgb.astype(np.float32) * alpha[..., None]
        )
        reference_roi[:] = np.clip(
            np.rint(blended_reference),
            0.0,
            255.0,
        ).astype(np.uint8)
    elif cleanplate_mode != "temporal":
        raise SystemExit(f"Unsupported mosaic clean-plate mode: {cleanplate_mode}")
    reference_background_rgb = reference_roi.copy()

    mosaic_path = output_path.with_name(f"{output_path.stem}_background_mosaic.png")
    coverage_path = output_path.with_name(f"{output_path.stem}_mosaic_coverage.png")
    alpha_path = output_path.with_name(f"{output_path.stem}_mosaic_alpha.png")
    if not cv2.imwrite(
        str(mosaic_path),
        cv2.cvtColor(mosaic_rgb, cv2.COLOR_RGB2BGR),
    ):
        raise SystemExit(f"Could not write mosaic image: {mosaic_path}")
    coverage_scale = np.log1p(observation_count.astype(np.float32))
    coverage_scale /= max(float(coverage_scale.max()), 1.0)
    coverage_image = cv2.applyColorMap(
        np.clip(np.rint(coverage_scale * 255.0), 0.0, 255.0).astype(np.uint8),
        cv2.COLORMAP_VIRIDIS,
    )
    coverage_image[~required_bool] = 0
    if not cv2.imwrite(str(coverage_path), coverage_image):
        raise SystemExit(f"Could not write mosaic coverage image: {coverage_path}")
    if not cv2.imwrite(str(alpha_path), required):
        raise SystemExit(f"Could not write mosaic alpha image: {alpha_path}")

    mosaic_valid = observation_count > 0
    uncovered_fractions = []
    mosaic_to_reference = np.linalg.inv(reference_to_mosaic)
    for homography in homographies:
        mosaic_to_frame = homography @ mosaic_to_reference
        coverage = cv2.warpPerspective(
            mosaic_valid.astype(np.float32),
            mosaic_to_frame,
            size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
        uncovered_fractions.append(
            float(np.mean(coverage <= MOSAIC_ALPHA_EPS))
        )
    maximum_uncovered = float(max(uncovered_fractions))
    if maximum_uncovered > 1e-4:
        raise SystemExit(
            "Expanded mosaic does not cover every dynamic frame: "
            f"max uncovered fraction={maximum_uncovered:.6f}."
        )
    counts = observation_count[required_bool]
    metrics: dict[str, object] = {
        "method": (
            "registered real-source observations with hand/mug/forearm "
            "exclusion; robust median plus farthest-from-dynamic, sharpness-"
            "tiebroken low-coverage sample"
        ),
        "border_policy": "BORDER_CONSTANT only; no reflection, replication, or inpainting",
        "mosaic_path": str(mosaic_path),
        "mosaic_sha256": _sha256(mosaic_path),
        "coverage_path": str(coverage_path),
        "coverage_sha256": _sha256(coverage_path),
        "alpha_path": str(alpha_path),
        "alpha_sha256": _sha256(alpha_path),
        "size": [mosaic_width, mosaic_height],
        "reference_bounds_xyxy": bounds,
        "reference_origin_xy": [origin_x, origin_y],
        "mask_dilation_pixels": int(mask_dilation),
        "reference_cleanplate_mode": cleanplate_mode,
        "reference_cleanplate_feather_pixels": int(feather),
        "required_pixels": int(required_bool.sum()),
        "unobserved_required_pixels": hole_count,
        "observation_count_min": int(counts.min()),
        "observation_count_p01": float(np.percentile(counts, 1)),
        "observation_count_p05": float(np.percentile(counts, 5)),
        "observation_count_median": float(np.median(counts)),
        "frame_uncovered_fraction_max": maximum_uncovered,
        "frame_uncovered_fraction_median": float(np.median(uncovered_fractions)),
    }
    print(
        "Expanded mosaic coverage verified: "
        f"{hole_count} required holes, max frame uncovered={maximum_uncovered:.6f}",
        flush=True,
    )
    return (
        mosaic_rgb,
        reference_to_mosaic,
        reference_background_rgb,
        mosaic_valid,
        metrics,
    )


def _prepare_dynamic_camera_fill(
    mosaic_rgb: np.ndarray,
    reference_to_mosaic: np.ndarray,
    mosaic_valid: np.ndarray,
    homography: np.ndarray,
    size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample the real-source mosaic without reflecting any border pixels."""
    width, height = size
    mosaic_to_frame = homography @ np.linalg.inv(reference_to_mosaic)
    dynamic_background, mosaic_coverage = _normalized_warp_rgb(
        mosaic_rgb,
        mosaic_valid.astype(np.float32),
        mosaic_to_frame,
        (width, height),
    )
    reference_coverage = cv2.warpPerspective(
        np.ones((height, width), dtype=np.float32),
        homography,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    return (
        dynamic_background,
        np.clip(reference_coverage, 0.0, 1.0),
        mosaic_coverage > MOSAIC_ALPHA_EPS,
    )


def _warp_composite_without_foreground_reflection(
    reference_composite: np.ndarray,
    homography: np.ndarray,
    dynamic_background: np.ndarray,
    coverage: np.ndarray,
) -> np.ndarray:
    """Apply camera motion while clipping foreground at the frame boundary."""
    height, width = reference_composite.shape[:2]
    warped_composite = cv2.warpPerspective(
        reference_composite.astype(np.float32),
        homography,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    # With a zero border, warped_composite is already premultiplied by the
    # interpolation coverage at the image edge. Add only the missing clean
    # background fraction; multiplying warped_composite by coverage again
    # would create a dark halo.
    filled = warped_composite + dynamic_background * (1.0 - coverage[..., None])
    return np.clip(filled, 0.0, 255.0).astype(np.uint8)


def _rotate_scene_about_object_yaw(
    qpos: np.ndarray,
    model: mujoco.MjModel,
    yaw_degrees: float | np.ndarray,
) -> np.ndarray:
    """Rigidly yaw the tracked object and free-root hand for presentation.

    ``yaw_degrees`` may be one shared angle or one angle per trajectory row.
    In both cases the hand and object receive the same rigid correction, so
    their object-relative position and orientation remain invariant.
    """
    source = np.asarray(qpos, dtype=np.float64)
    out = source.copy()
    yaw = np.asarray(yaw_degrees, dtype=np.float64)
    if yaw.ndim == 0:
        if not np.isfinite(yaw):
            raise ValueError("yaw_degrees must be finite")
        if abs(float(yaw)) < 1e-12:
            return out
        correction = Rotation.from_euler("z", float(yaw), degrees=True)
    else:
        if yaw.shape != (len(source),) or not np.isfinite(yaw).all():
            raise ValueError(
                "per-frame yaw_degrees must be finite with shape "
                f"({len(source)},); got {yaw.shape}"
            )
        if np.all(np.abs(yaw) < 1e-12):
            return out
        correction = Rotation.from_euler("z", yaw[:, None], degrees=True)

    if not len(source):
        return out

    object_qadr = None
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            continue
        body_id = int(model.jnt_bodyid[joint_id])
        body_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, body_id
        )
        if body_name and body_name.endswith("_object"):
            object_qadr = int(model.jnt_qposadr[joint_id])
            break
    if object_qadr is None:
        raise ValueError("No tracked-object free joint found for yaw correction.")

    origins = source[:, object_qadr : object_qadr + 3]
    object_wxyz = source[:, object_qadr + 3 : object_qadr + 7]
    object_rotation = Rotation.from_quat(object_wxyz[:, [1, 2, 3, 0]])
    corrected_object_xyzw = (correction * object_rotation).as_quat()
    out[:, object_qadr + 3 : object_qadr + 7] = corrected_object_xyzw[
        :, [3, 0, 1, 2]
    ]

    for side in ("right", "left"):
        names = [
            f"{side}_pos_x",
            f"{side}_pos_y",
            f"{side}_pos_z",
            f"{side}_rot_x",
            f"{side}_rot_y",
            f"{side}_rot_z",
        ]
        joint_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in names
        ]
        if any(joint_id < 0 for joint_id in joint_ids):
            continue
        addresses = [int(model.jnt_qposadr[joint_id]) for joint_id in joint_ids]
        if addresses != list(range(addresses[0], addresses[0] + 6)):
            raise ValueError(f"Non-contiguous {side} hand root qpos addresses.")
        pos_qadr, rot_qadr = addresses[0], addresses[3]
        relative_position = source[:, pos_qadr : pos_qadr + 3] - origins
        out[:, pos_qadr : pos_qadr + 3] = origins + correction.apply(
            relative_position
        )
        hand_rotation = Rotation.from_euler(
            "XYZ", source[:, rot_qadr : rot_qadr + 3]
        )
        corrected_angles = (correction * hand_rotation).as_euler("XYZ")
        out[:, rot_qadr : rot_qadr + 3] = np.unwrap(corrected_angles, axis=0)
    return np.ascontiguousarray(out)


class StageRenderer:
    def __init__(
        self,
        spec: StageSpec,
        frame_count: int,
        anchors: np.ndarray,
        width: int,
        height: int,
        presentation_yaw_degrees: float = 0.0,
        presentation_yaw_end_degrees: float | None = None,
    ) -> None:
        self.spec = spec
        self.run_dir = spec.run_dir.resolve()
        self.scene_path = self.run_dir / "scene.xml"
        self.trajectory_path = self.run_dir / spec.trajectory_name
        for path in (self.scene_path, self.trajectory_path):
            if not path.is_file():
                raise SystemExit(f"Not found: {path}")
        self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        _apply_viser_visual_style(self.model)
        self.foreground_geom_ids = _prepare_background_compositing(self.model)
        self.object_geom_ids = np.asarray(
            [
                geom_id
                for geom_id in range(self.model.ngeom)
                if _geom_name(self.model, geom_id).lower().endswith("_object_visual")
            ],
            dtype=np.int32,
        )
        if not len(self.object_geom_ids):
            raise SystemExit(f"No object visual geom found in {self.scene_path}")
        self.data = mujoco.MjData(self.model)
        self.presentation_yaw_start_degrees = float(presentation_yaw_degrees)
        self.presentation_yaw_end_degrees = float(
            presentation_yaw_degrees
            if presentation_yaw_end_degrees is None
            else presentation_yaw_end_degrees
        )
        self.presentation_yaw_degrees = (
            self.presentation_yaw_start_degrees
            if self.presentation_yaw_start_degrees
            == self.presentation_yaw_end_degrees
            else None
        )
        self.qpos = load_qpos(self.trajectory_path, self.model.nq)
        self.start = 0
        if spec.timing == "physics":
            cfg = _load_run_config(self.run_dir / "config.yaml")
            self.start = min(
                int(cfg.get("warmup_steps", 0) or 0),
                max(0, len(self.qpos) - 1),
            )
            replay_qpos = self.qpos[self.start :]
            if anchors[0, 0] != 0 or anchors[-1, 0] != frame_count - 1:
                raise SystemExit(
                    "Physics anchors must include the first and last source frames."
                )
            if anchors[0, 1] < 0 or anchors[-1, 1] >= len(replay_qpos):
                raise SystemExit(
                    f"Physics anchors outside {spec.label!r} replay range "
                    f"[0, {len(replay_qpos) - 1}]."
                )
            relative = np.rint(
                np.interp(
                    np.arange(frame_count, dtype=np.float64),
                    anchors[:, 0],
                    anchors[:, 1],
                )
            ).astype(np.int64)
            self.frame_indices = relative + self.start
        else:
            self.frame_indices = np.rint(
                np.linspace(0, len(self.qpos) - 1, frame_count)
            ).astype(np.int64)

        source_yaw = np.linspace(
            self.presentation_yaw_start_degrees,
            self.presentation_yaw_end_degrees,
            frame_count,
            dtype=np.float64,
        )
        unique_indices, first_occurrences = np.unique(
            self.frame_indices, return_index=True
        )
        trajectory_yaw = np.interp(
            np.arange(len(self.qpos), dtype=np.float64),
            unique_indices,
            source_yaw[first_occurrences],
            left=source_yaw[0],
            right=source_yaw[-1],
        )
        self.qpos = _rotate_scene_about_object_yaw(
            self.qpos,
            self.model,
            trajectory_yaw,
        )

        self.model.vis.global_.offwidth = max(
            int(self.model.vis.global_.offwidth), width
        )
        self.model.vis.global_.offheight = max(
            int(self.model.vis.global_.offheight), height
        )
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.scene_options = mujoco.MjvOption()
        mujoco.mjv_defaultOption(self.scene_options)
        self.scene_options.geomgroup[3:] = 0

    def render(
        self,
        source_frame: int,
        camera: mujoco.MjvCamera,
    ) -> tuple[np.ndarray, np.ndarray]:
        qpos_index = int(self.frame_indices[source_frame])
        self.data.qpos[:] = self.qpos[qpos_index]
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(
            self.data,
            camera=camera,
            scene_option=self.scene_options,
        )
        rgb = self.renderer.render().copy()
        self.renderer.enable_segmentation_rendering()
        segmentation = self.renderer.render().copy()
        self.renderer.disable_segmentation_rendering()
        return rgb, segmentation

    def object_mask(self, segmentation: np.ndarray) -> np.ndarray:
        return (
            (segmentation[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM))
            & np.isin(segmentation[..., 0], self.object_geom_ids)
        )

    def close(self) -> None:
        self.renderer.close()


def _fit_foreground_affine(
    final_stage: StageRenderer,
    camera: mujoco.MjvCamera,
    homographies: np.ndarray,
    mask_root: Path,
    anchors: np.ndarray,
    size: tuple[int, int],
) -> tuple[np.ndarray, dict[str, object]]:
    rendered_points: list[np.ndarray] = []
    source_reference_points: list[np.ndarray] = []
    used_frames: list[int] = []
    frame_count = len(final_stage.frame_indices)
    candidate_frames = sorted(
        {
            0,
            frame_count - 1,
            *[int(value) for value in anchors[:, 0]],
            *range(15, frame_count, 15),
        }
    )
    for frame_index in candidate_frames:
        source_mask = _read_mask(mask_root, frame_index, "mug.png", size)
        if source_mask is None:
            continue
        source_center = _centroid(source_mask)
        if source_center is None:
            continue
        _, segmentation = final_stage.render(frame_index, camera)
        rendered_center = _centroid(final_stage.object_mask(segmentation))
        if rendered_center is None:
            continue
        source_in_reference = _transform_point(
            source_center,
            np.linalg.inv(homographies[frame_index]),
        )
        rendered_points.append(rendered_center)
        source_reference_points.append(source_in_reference)
        used_frames.append(frame_index)
    if len(rendered_points) < 6:
        raise SystemExit(
            f"Only {len(rendered_points)} object correspondences for auto-alignment."
        )
    source_array = np.asarray(rendered_points, dtype=np.float32)
    target_array = np.asarray(source_reference_points, dtype=np.float32)
    affine, inliers = cv2.estimateAffinePartial2D(
        source_array,
        target_array,
        method=cv2.RANSAC,
        ransacReprojThreshold=18.0,
        maxIters=5000,
        confidence=0.999,
        refineIters=50,
    )
    if affine is None or inliers is None or int(inliers.sum()) < 5:
        raise SystemExit("Could not fit a stable foreground similarity transform.")
    affine = np.asarray(affine, dtype=np.float32)
    scale = float(np.hypot(affine[0, 0], affine[1, 0]))
    rotation_degrees = float(np.degrees(np.arctan2(affine[1, 0], affine[0, 0])))
    if not 0.5 <= scale <= 1.4:
        raise SystemExit(f"Auto-alignment produced unreasonable scale {scale:.3f}.")
    homogeneous_points = np.concatenate(
        (source_array, np.ones((len(source_array), 1), dtype=np.float32)),
        axis=1,
    )
    predicted = homogeneous_points @ affine.T
    errors = np.linalg.norm(predicted - target_array, axis=1)
    keep = inliers.ravel().astype(bool)
    metrics: dict[str, object] = {
        "used_frames": used_frames,
        "correspondences": len(used_frames),
        "inliers": int(keep.sum()),
        "scale": scale,
        "rotation_degrees_image_coordinates": rotation_degrees,
        "translation": affine[:, 2].tolist(),
        "inlier_error_median_px": float(np.median(errors[keep])),
        "inlier_error_p95_px": float(np.percentile(errors[keep], 95)),
    }
    return affine, metrics


def _draw_panel_header(
    image: np.ndarray,
    text: str,
    color: tuple[int, int, int],
) -> None:
    height, width = image.shape[:2]
    del height
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (width, 48), (18, 22, 28), thickness=-1)
    cv2.addWeighted(overlay, 0.78, image, 0.22, 0.0, dst=image)
    cv2.rectangle(image, (0, 0), (7, 48), color, thickness=-1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.68
    thickness = 2
    (text_width, _), _ = cv2.getTextSize(text, font, scale, thickness)
    if text_width > width - 44:
        scale *= (width - 44) / text_width
    cv2.putText(
        image,
        text,
        (22, 32),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def _draw_badge(
    image: np.ndarray,
    text: str,
    color: tuple[int, int, int],
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.58
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, scale, thickness
    )
    height, width = image.shape[:2]
    x0 = 14
    y1 = height - 14
    y0 = y1 - text_height - baseline - 16
    x1 = min(width - 14, x0 + text_width + 24)
    overlay = image.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (15, 18, 22), thickness=-1)
    cv2.addWeighted(overlay, 0.78, image, 0.22, 0.0, dst=image)
    cv2.circle(image, (x0 + 11, (y0 + y1) // 2), 4, color, thickness=-1)
    cv2.putText(
        image,
        text,
        (x0 + 22, y1 - baseline - 7),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def _phase(frame_index: int) -> str:
    if frame_index < 45:
        return "Approach"
    if frame_index <= 114:
        return "Grasp and transport"
    if frame_index <= 124:
        return "Placement"
    return "Release and retreat"


def _stage_badge(
    stage_index: int,
    frame_index: int,
    narrative: str,
) -> tuple[str, tuple[int, int, int]]:
    if narrative == "target-rl":
        if stage_index == 0:
            if frame_index < 45:
                return "Before RL: untrained motion", (235, 105, 80)
            if frame_index <= 114:
                return "Before RL: affordance missed", (235, 105, 80)
            return "Before RL: no physical policy", (235, 105, 80)
        if stage_index == 1:
            if frame_index < 45:
                return "Learning: exploration", (242, 178, 70)
            if frame_index <= 114:
                return "Learning: body / rim contact", (242, 178, 70)
            return "Learning: unstable release", (242, 178, 70)
        if frame_index < 45:
            return "Converged policy: approach", (85, 210, 115)
        if frame_index <= 114:
            return "Converged policy: handle grasp", (85, 210, 115)
        return "Converged policy: stable place", (85, 210, 115)
    if frame_index < 45:
        return _phase(frame_index), (225, 190, 60)
    if stage_index == 0:
        if frame_index <= 114:
            return "No handle-affordance alignment", (235, 105, 80)
        return "Reference-only release", (235, 105, 80)
    if stage_index == 1:
        if frame_index <= 114:
            return "Body / rim contact", (235, 105, 80)
        return "Residual object contact", (235, 105, 80)
    if frame_index <= 114:
        return "Handle-aligned contact", (85, 210, 115)
    return "Clean release", (85, 210, 115)


def _make_camera(
    final_stage: StageRenderer,
    width: int,
    height: int,
    fov: float,
) -> tuple[mujoco.MjvCamera, np.ndarray, np.ndarray, np.ndarray]:
    keypoints_path = _keypoints_path_for_run(final_stage.run_dir)
    position, target, up = _top_down_camera_preset(
        final_stage.qpos,
        final_stage.start,
        keypoints_path,
        vertical_fov_deg=fov,
        aspect_ratio=float(width) / float(height),
    )
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = target
    camera.distance = float(np.linalg.norm(position - target))
    camera.azimuth = 90.0
    camera.elevation = -89.9
    return camera, position, target, up


def main() -> None:
    args = _parse_args()
    if len(args.stage) != 3:
        raise SystemExit(f"Exactly three --stage values are required; got {len(args.stage)}.")
    stage_timings = [spec.timing for spec in args.stage]
    if stage_timings != ["kinematic", "physics", "physics"]:
        raise SystemExit(
            "This 2x2 narrative requires stage timings: kinematic, physics, physics."
        )
    if args.panel_width <= 0 or args.panel_height <= 0:
        raise SystemExit("Panel dimensions must be positive.")
    if not 0.0 <= args.shadow_opacity <= 1.0:
        raise SystemExit("--shadow-opacity must lie in [0, 1].")
    first_yaw_start = args.first_stage_yaw_start_degrees
    first_yaw_end = (
        first_yaw_start
        if args.first_stage_yaw_end_degrees is None
        else args.first_stage_yaw_end_degrees
    )
    final_yaw_start = (
        args.final_stage_yaw_degrees
        if args.final_stage_yaw_start_degrees is None
        else args.final_stage_yaw_start_degrees
    )
    final_yaw_end = (
        final_yaw_start
        if args.final_stage_yaw_end_degrees is None
        else args.final_stage_yaw_end_degrees
    )
    yaw_values = (
        first_yaw_start,
        first_yaw_end,
        final_yaw_start,
        final_yaw_end,
    )
    if not np.isfinite(yaw_values).all():
        raise SystemExit("Presentation yaw values must be finite.")
    if args.mosaic_padding < 0:
        raise SystemExit("--mosaic-padding must be non-negative.")
    if args.mosaic_mask_dilation < 0:
        raise SystemExit("--mosaic-mask-dilation must be non-negative.")
    if args.mosaic_feather < 0:
        raise SystemExit("--mosaic-feather must be non-negative.")
    if args.mosaic_tile_height <= 0:
        raise SystemExit("--mosaic-tile-height must be positive.")
    source_path = Path(args.source_video).resolve()
    background_path = Path(args.background_image).resolve()
    mask_root = Path(args.mask_root).resolve()
    output_path = Path(args.output).resolve()
    for path in (source_path, background_path, mask_root):
        if not path.exists():
            raise SystemExit(f"Not found: {path}")
    if not mask_root.is_dir():
        raise SystemExit(f"Mask root is not a directory: {mask_root}")
    if args.auto_align_object and not args.dynamic_camera:
        raise SystemExit(
            "--auto-align-object requires --dynamic-camera so source mask "
            "centres can be returned to the clean-plate reference frame."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    source_frames, source_fps, source_size = _load_source_frames(
        source_path,
        args.panel_width,
        args.panel_height,
    )
    frame_count = len(source_frames)
    mug_mask_count = sum(
        _mask_path(mask_root, frame_index, "mug.png").is_file()
        for frame_index in range(frame_count)
    )
    if mug_mask_count < max(6, int(np.ceil(0.9 * frame_count))):
        raise SystemExit(
            f"Found only {mug_mask_count}/{frame_count} source mug masks."
        )
    if args.dynamic_camera:
        homographies, homography_metrics = _estimate_table_homographies(
            source_frames,
            mask_root,
            args.reference_frame,
            args.homography_smooth_radius,
        )
    else:
        homographies = np.repeat(
            np.eye(3, dtype=np.float64)[None, ...], frame_count, axis=0
        )
        homography_metrics = {
            "reference_frame": args.reference_frame,
            "valid_frames": frame_count,
            "interpolated_frames": 0,
            "dynamic_camera": False,
        }

    stages: list[StageRenderer] = []
    try:
        for stage_index, spec in enumerate(args.stage):
            if stage_index == 0:
                yaw_start, yaw_end = first_yaw_start, first_yaw_end
            elif stage_index == len(args.stage) - 1:
                yaw_start, yaw_end = final_yaw_start, final_yaw_end
            else:
                yaw_start = yaw_end = 0.0
            stages.append(
                StageRenderer(
                    spec,
                    frame_count,
                    args.anchors,
                    args.panel_width,
                    args.panel_height,
                    presentation_yaw_degrees=yaw_start,
                    presentation_yaw_end_degrees=yaw_end,
                )
            )
        camera, camera_position, camera_target, camera_up = _make_camera(
            stages[-1],
            args.panel_width,
            args.panel_height,
            args.fov,
        )
        affine = _scale_affine(
            args.foreground_affine,
            (args.affine_reference_width, args.affine_reference_height),
            (args.panel_width, args.panel_height),
        )
        affine_metrics: dict[str, object] = {"mode": "configured fallback"}
        if args.auto_align_object:
            affine, fitted_metrics = _fit_foreground_affine(
                stages[-1],
                camera,
                homographies,
                mask_root,
                args.anchors,
                (args.panel_width, args.panel_height),
            )
            affine_metrics = {
                "mode": "source mug masks vs final render",
                **fitted_metrics,
            }

        background_bgr = cv2.imread(str(background_path), cv2.IMREAD_COLOR)
        if background_bgr is None:
            raise SystemExit(f"Could not read background image: {background_path}")
        background_rgb = cv2.cvtColor(
            cv2.resize(
                background_bgr,
                (args.panel_width, args.panel_height),
                interpolation=cv2.INTER_AREA,
            ),
            cv2.COLOR_BGR2RGB,
        )
        if args.dynamic_camera:
            (
                mosaic_rgb,
                reference_to_mosaic,
                background_rgb,
                mosaic_valid,
                mosaic_metrics,
            ) = _build_expanded_background_mosaic(
                source_frames,
                homographies,
                mask_root,
                background_rgb,
                output_path,
                args.mosaic_padding,
                args.mosaic_mask_dilation,
                args.mosaic_feather,
                args.mosaic_tile_height,
                args.mosaic_cleanplate_mode,
            )
        else:
            mosaic_rgb = background_rgb
            reference_to_mosaic = np.eye(3, dtype=np.float64)
            mosaic_valid = np.ones(background_rgb.shape[:2], dtype=bool)
            mosaic_metrics = {
                "method": "disabled with static camera",
                "border_policy": "not applicable",
                "size": [args.panel_width, args.panel_height],
            }
    except BaseException:
        for stage in stages:
            stage.close()
        raise

    output_width = args.panel_width * 2
    output_height = args.panel_height * 2
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
        f"{output_width}x{output_height}",
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
    try:
        encoder = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
    except BaseException:
        for stage in stages:
            stage.close()
        raise
    stage_colors = ((242, 178, 70), (235, 105, 80), (85, 210, 115))
    encoded_frames = 0
    same_frame_source_guard_pixels = 0
    same_frame_source_guard_pixels_max = 0
    render_error: BaseException | None = None
    encoder_close_error: BrokenPipeError | None = None
    try:
        for frame_index, source_bgr in enumerate(source_frames):
            source_rgb = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2RGB)
            dynamic_background = None
            dynamic_coverage = None
            dynamic_mosaic_coverage = None
            if args.dynamic_camera:
                (
                    dynamic_background,
                    dynamic_coverage,
                    dynamic_mosaic_coverage,
                ) = _prepare_dynamic_camera_fill(
                    mosaic_rgb,
                    reference_to_mosaic,
                    mosaic_valid,
                    homographies[frame_index],
                    (args.panel_width, args.panel_height),
                )
                missing_mask = ~dynamic_mosaic_coverage
                missing = int(missing_mask.sum())
                maximum_guard_pixels = max(
                    8,
                    int(np.ceil(args.panel_width * args.panel_height * 1e-5)),
                )
                if missing > maximum_guard_pixels:
                    raise RuntimeError(
                        f"Mosaic coverage regression at frame {frame_index}: "
                        f"{missing} output pixels are undefined."
                    )
                if missing:
                    # A perspective-warp raster can place a corner pixel just
                    # outside the discrete mosaic alpha even though the same
                    # source frame directly observes it.  Use that exact real
                    # pixel as a tiny guard instead of reflecting a neighbour.
                    dynamic_background[missing_mask] = source_rgb[missing_mask]
                    same_frame_source_guard_pixels += missing
                    same_frame_source_guard_pixels_max = max(
                        same_frame_source_guard_pixels_max,
                        missing,
                    )
            source_header = (
                "Human Ego Demonstration (Task Goal)"
                if args.narrative == "target-rl"
                else "Human ego demonstration"
            )
            _draw_panel_header(source_rgb, source_header, (90, 175, 245))
            _draw_badge(source_rgb, _phase(frame_index), (90, 175, 245))
            panels = [source_rgb]
            for stage_index, stage in enumerate(stages):
                render_rgb, segmentation = stage.render(frame_index, camera)
                panel = _composite_over_background(
                    render_rgb,
                    segmentation,
                    stage.foreground_geom_ids,
                    background_rgb,
                    affine,
                    args.shadow_opacity,
                )
                if args.dynamic_camera:
                    assert dynamic_background is not None
                    assert dynamic_coverage is not None
                    panel = _warp_composite_without_foreground_reflection(
                        panel,
                        homographies[frame_index],
                        dynamic_background,
                        dynamic_coverage,
                    )
                color = stage_colors[stage_index]
                _draw_panel_header(panel, stage.spec.label, color)
                badge, badge_color = _stage_badge(
                    stage_index,
                    frame_index,
                    args.narrative,
                )
                _draw_badge(panel, badge, badge_color)
                panels.append(panel)
            top = np.concatenate((panels[0], panels[1]), axis=1)
            bottom = np.concatenate((panels[2], panels[3]), axis=1)
            composed = np.concatenate((top, bottom), axis=0)
            composed[
                args.panel_height - 2 : args.panel_height + 2,
                :,
            ] = 255
            composed[
                :,
                args.panel_width - 2 : args.panel_width + 2,
            ] = 255
            if encoder.stdin is None:
                raise RuntimeError("FFmpeg stdin is unavailable.")
            encoder.stdin.write(np.ascontiguousarray(composed).tobytes())
            encoded_frames += 1
            if encoded_frames == 1 or encoded_frames % 15 == 0:
                print(f"Rendered {encoded_frames}/{frame_count} frames", flush=True)
    except BaseException as exc:
        render_error = exc
        raise
    finally:
        for stage in stages:
            stage.close()
        if encoder.stdin is not None:
            try:
                encoder.stdin.close()
            except BrokenPipeError as exc:
                encoder_close_error = exc
        if render_error is not None and encoder.poll() is None:
            encoder.terminate()
            encoder.wait()
    return_code = encoder.wait()
    if encoder_close_error is not None:
        raise SystemExit("FFmpeg closed its input pipe before encoding completed.")
    if return_code != 0:
        raise SystemExit(f"FFmpeg failed with exit code {return_code}.")
    if encoded_frames != frame_count:
        raise SystemExit(f"Encoded {encoded_frames}; expected {frame_count} frames.")

    stage_manifest = []
    for stage in stages:
        stage_manifest.append(
            {
                "label": stage.spec.label,
                "run_dir": str(stage.run_dir),
                "trajectory": str(stage.trajectory_path),
                "trajectory_sha256": _sha256(stage.trajectory_path),
                "timing": stage.spec.timing,
                "presentation_yaw_degrees": stage.presentation_yaw_degrees,
                "presentation_yaw_start_degrees": (
                    stage.presentation_yaw_start_degrees
                ),
                "presentation_yaw_end_degrees": (
                    stage.presentation_yaw_end_degrees
                ),
                "presentation_yaw_interpolation": "linear in source frame index",
                "warmup_steps_skipped": stage.start,
                "trajectory_indices": stage.frame_indices.tolist(),
            }
        )
    mosaic_metrics["same_frame_source_guard_pixels_total"] = int(
        same_frame_source_guard_pixels
    )
    mosaic_metrics["same_frame_source_guard_pixels_max_per_frame"] = int(
        same_frame_source_guard_pixels_max
    )
    manifest = {
        "schema_version": (
            "target-rl-concept-demo.v1"
            if args.narrative == "target-rl"
            else "optimization-evolution-demo.v1"
        ),
        "narrative": args.narrative,
        "terminology": (
            "Target/concept storyboard for a planned RL system. The present "
            "panels use saved kinematic and MJWP/MPC retargeting trajectories "
            "as visual proxies; they are not actual RL training checkpoints or "
            "evidence of a trained policy."
            if args.narrative == "target-rl"
            else (
                "Saved kinematic and MJWP/MPC physics-optimization stages; this "
                "is not reinforcement-learning training."
            )
        ),
        "source_video": str(source_path),
        "source_video_sha256": _sha256(source_path),
        "source_size": list(source_size),
        "source_frame_count": frame_count,
        "source_fps": source_fps,
        "duration_seconds": frame_count / source_fps,
        "background_image": str(background_path),
        "background_image_sha256": _sha256(background_path),
        "mask_root": str(mask_root),
        "stages": stage_manifest,
        "anchors": args.anchors.tolist(),
        "camera": {
            "base_mode": "canonical top-down",
            "base_position": camera_position.tolist(),
            "base_target": camera_target.tolist(),
            "base_up": camera_up.tolist(),
            "vertical_fov_degrees": args.fov,
            "dynamic_table_homography": bool(args.dynamic_camera),
            "dynamic_border_policy": (
                "expanded real-source clean mosaic with constant-border "
                "sampling; no reflected, replicated, or inpainted pixels"
            ),
            "expanded_background_mosaic": mosaic_metrics,
            "homography_metrics": homography_metrics,
            "homographies_reference_to_frame": homographies.tolist(),
        },
        "foreground_alignment": {
            **affine_metrics,
            "affine": affine.tolist(),
        },
        "shadow_opacity": args.shadow_opacity,
        "output_video": str(output_path),
        "output_video_sha256": _sha256(output_path),
        "output_size": [output_width, output_height],
        "output_frame_count": encoded_frames,
        "output_fps": source_fps,
        "render_backend": "MuJoCo offline rendering; no optimization rerun",
    }
    manifest_path = output_path.with_suffix(".json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"EVOLUTION_DEMO_COMPLETE {output_path}", flush=True)
    print(f"MANIFEST {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
