"""Detect the Isaac Sim capabilities available to the export process."""

from __future__ import annotations

import importlib.util
import platform
import shutil
import sys
from datetime import UTC, datetime


def _module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def inspect_environment(requested_isaac_version: str) -> dict:
    """Return a JSON-ready report without importing or starting Isaac Sim."""
    modules = {
        module_name: _module_available(module_name)
        for module_name in ("isaacsim", "omni.usd", "pxr")
    }
    executables = {
        name: shutil.which(name) for name in ("isaac-sim", "isaaclab")
    }
    ready = modules["isaacsim"] and modules["omni.usd"] and modules["pxr"]
    missing = [name for name, available in modules.items() if not available]
    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "status": "ready" if ready else "unavailable",
        "requested_isaac_sim_version": requested_isaac_version,
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
        },
        "platform": platform.platform(),
        "executables": executables,
        "modules": modules,
        "missing_requirements": missing,
        "capabilities": {
            "standard_trajectory_export": True,
            "usd_asset_conversion": ready,
            "isaac_kinematic_replay": ready,
            "headless_rendering": ready,
        },
    }