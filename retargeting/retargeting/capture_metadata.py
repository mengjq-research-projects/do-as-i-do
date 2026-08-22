"""Capture metadata shared by Reconstruction and Retargeting.

Viewpoint and camera motion describe the observation source. They are kept
separate from robot workspace placement: an ego video is not itself an IK
branch or a fixed robot coordinate.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


VALID_VIEWPOINTS = {"auto", "ego", "exo"}
VALID_CAMERA_MOTIONS = {"auto", "moving", "static"}


def normalize_capture_metadata(config: dict[str, Any]) -> dict[str, str]:
    """Return validated metadata while accepting pre-metadata configs."""
    capture = config.get("capture", {})
    if not isinstance(capture, dict):
        capture = {}
    viewpoint = str(capture.get("viewpoint", config.get("viewpoint", "auto")))
    camera_motion = str(
        capture.get("camera_motion", config.get("camera_motion", "auto"))
    )
    if viewpoint not in VALID_VIEWPOINTS:
        raise ValueError(
            f"capture.viewpoint must be one of {sorted(VALID_VIEWPOINTS)}; "
            f"got {viewpoint!r}"
        )
    if camera_motion not in VALID_CAMERA_MOTIONS:
        raise ValueError(
            "capture.camera_motion must be one of "
            f"{sorted(VALID_CAMERA_MOTIONS)}; got {camera_motion!r}"
        )
    return {"viewpoint": viewpoint, "camera_motion": camera_motion}


def load_capture_metadata(raw_dir: str | Path) -> dict[str, str]:
    """Load capture metadata from a Reconstruction output directory."""
    config_path = Path(raw_dir) / "config.json"
    with config_path.open(encoding="utf-8") as file:
        config = json.load(file)
    return normalize_capture_metadata(config)


def write_capture_metadata(raw_dir: str | Path, output_dir: str | Path) -> Path:
    """Write the deployment-readable sidecar beside Retargeting outputs."""
    capture = load_capture_metadata(raw_dir)
    output_path = Path(output_dir) / "capture_metadata.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "capture.v1",
        **capture,
        "source_config": str((Path(raw_dir) / "config.json").resolve()),
    }
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_path
