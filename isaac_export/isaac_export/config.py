"""Configuration for the standalone Isaac export stage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ExportConfig:
    trajectory_filename: str = "trajectory_mjwp.npz"
    skip_warmup: bool = True
    expected_robot: str = "sharpa"
    expected_embodiment: str = "right"
    expected_finger_joints: int = 22
    position_jump_warning_m: float = 0.25
    joint_jump_warning_rad: float = 1.0


def load_export_config(path: Path) -> ExportConfig:
    """Load the small, Isaac-independent export configuration."""
    if not path.is_file():
        raise FileNotFoundError(f"Export config not found: {path}")
    with path.open(encoding="utf-8") as stream:
        values = yaml.safe_load(stream) or {}
    if not isinstance(values, dict):
        raise ValueError(f"Export config must contain a mapping: {path}")
    return ExportConfig(**values)