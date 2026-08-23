"""Helpers for wheel-based and standalone Isaac Sim runtimes."""

from __future__ import annotations

import importlib.metadata
import os
from pathlib import Path


def isaac_sim_version() -> str:
    """Return the Isaac Sim version for wheels or extracted standalone installs."""
    try:
        return importlib.metadata.version("isaacsim")
    except importlib.metadata.PackageNotFoundError:
        pass

    isaac_path = os.environ.get("ISAAC_PATH")
    if isaac_path:
        version_path = Path(isaac_path) / "VERSION"
        if version_path.is_file():
            version = version_path.read_text(encoding="utf-8").strip()
            if version:
                return version

    return "unknown"
