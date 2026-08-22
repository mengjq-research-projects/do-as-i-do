#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "replay_dispatch.py"
SPEC = importlib.util.spec_from_file_location("replay_dispatch", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
replay_dispatch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay_dispatch)


class ResolveRenderModeTest(unittest.TestCase):
    def test_auto_ego_uses_hand_only(self) -> None:
        self.assertEqual(
            replay_dispatch.resolve_render_mode("auto", "ego"), "hand-only"
        )

    def test_auto_exo_uses_full_arm(self) -> None:
        self.assertEqual(
            replay_dispatch.resolve_render_mode("auto", "exo"), "full-arm"
        )

    def test_legacy_auto_preserves_full_arm(self) -> None:
        self.assertEqual(
            replay_dispatch.resolve_render_mode("auto", "auto"), "full-arm"
        )

    def test_explicit_override_wins(self) -> None:
        self.assertEqual(
            replay_dispatch.resolve_render_mode("full-arm", "ego"), "full-arm"
        )
        self.assertEqual(
            replay_dispatch.resolve_render_mode("hand-only", "exo"), "hand-only"
        )

    def test_capture_viewpoint_reads_valid_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "capture_metadata.json").write_text(
                json.dumps({"viewpoint": "ego"}), encoding="utf-8"
            )
            self.assertEqual(
                replay_dispatch._capture_viewpoint(run_dir / "trajectory_mjwp.npz"),
                "ego",
            )

    def test_capture_viewpoint_rejects_invalid_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "capture_metadata.json").write_text(
                json.dumps({"viewpoint": "inside-out"}), encoding="utf-8"
            )
            self.assertEqual(
                replay_dispatch._capture_viewpoint(run_dir / "trajectory_mjwp.npz"),
                "auto",
            )


if __name__ == "__main__":
    unittest.main()
