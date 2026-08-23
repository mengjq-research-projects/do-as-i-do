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
    scripts/package_isaac_replay.sh

if rg -n '/home/hym|/data/jiaqimeng' \
    deployment/run_pipeline.sh \
    deployment/mujoco_replay/*.py \
    isaac_export/*.py \
    isaac_export/*.sh; then
    echo "Machine-specific absolute path found in replay source." >&2
    exit 1
fi

PYTHON_BIN="${TEST_PYTHON:-python3}"
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
