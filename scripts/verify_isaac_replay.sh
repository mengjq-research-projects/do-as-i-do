#!/usr/bin/env bash
# Fast, offline source-level verification for the Isaac replay branch.
set -Eeuo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPOSITORY_ROOT"

for required in \
    deployment/mujoco_replay/assets/universal_robots_ur3e/ur3e.xml \
    retargeting/outputs/sharpa/right/whisking/0/trajectory_mjwp.npz \
    retargeting/outputs/assets/objects/whisking/visual.obj \
    retargeting/outputs/assets/objects/whisking/visual_texture.png; do
    if [[ ! -f "$required" ]]; then
        echo "Missing required replay input: $required" >&2
        exit 1
    fi
done

bash -n \
    deployment/run_pipeline.sh \
    isaac_export/run_pipeline.sh \
    isaac_export/run_pipeline_local.sh \
    scripts/package_isaac_replay.sh \
    scripts/verify_isaac_replay.sh

if rg -n '/home/hym|/data/jiaqimeng' \
    deployment/run_pipeline.sh \
    deployment/mujoco_replay/*.py \
    isaac_export/*.py \
    isaac_export/*.sh; then
    echo "Machine-specific absolute path found in replay source." >&2
    exit 1
fi

PYTHON_BIN="${TEST_PYTHON:-python3}"
"$PYTHON_BIN" - <<'PY'
import hashlib
import json
from pathlib import Path

bundle = Path("isaac_export/prebuilt/whisking")
required = (
    "scene.usd",
    "trajectory_dual_ur3e.npz",
    "deployment_manifest.json",
    "isaac_scene_manifest.json",
    "isaac_assets/visual_texture.png",
    "source_assets/visual.obj",
    "source_assets/trajectory_mjwp.npz",
    "PORTABLE_BUNDLE.json",
)
for name in required:
    if not (bundle / name).is_file():
        raise SystemExit(f"Missing portable replay file: {bundle / name}")
for path in bundle.rglob("*"):
    if path.is_file() and any(
        marker in path.read_bytes()
        for marker in (b"/home/", b"/data/", b"/tmp/", b"file://")
    ):
        raise SystemExit(f"Machine-specific path remains in {path}")
manifest = json.loads((bundle / "PORTABLE_BUNDLE.json").read_text())
if manifest["mode"] != "direct_isaac_replay":
    raise SystemExit("Portable bundle mode is invalid")
for name, expected in manifest["files"].items():
    path = bundle / name
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected["sha256"] or path.stat().st_size != expected["bytes"]:
        raise SystemExit(f"Portable bundle checksum failed: {path}")
PY
(
    cd deployment/mujoco_replay
    "$PYTHON_BIN" -m pytest -q tests
)
(
    cd isaac_export
    "$PYTHON_BIN" -m pytest -q tests
)

git diff --check
echo "Isaac replay source verification passed."
