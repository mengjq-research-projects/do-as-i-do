#!/usr/bin/env bash
# Generate a UR3e + Sharpa + task-object replay package outside the checkout.
set -Eeuo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${1:-$REPOSITORY_ROOT/retargeting/outputs/sharpa/right/whisking/0}"
if [[ $# -gt 0 ]]; then
    shift
fi

if [[ ! -f "$RUN_DIR/trajectory_mjwp.npz" ]]; then
    echo "Missing Retargeting trajectory: $RUN_DIR/trajectory_mjwp.npz" >&2
    exit 1
fi

DATA_ROOT="${XDG_DATA_HOME:-$HOME/.local/share}"
PACKAGE_DIR="${DO_AS_I_DO_REPLAY_PACKAGE_DIR:-$DATA_ROOT/do-as-i-do/isaac_replay/whisking}"
mkdir -p "$PACKAGE_DIR"

"$REPOSITORY_ROOT/deployment/run_pipeline.sh" mujoco-replay \
    --side right \
    --traj "$RUN_DIR/trajectory_mjwp.npz" \
    --output "$PACKAGE_DIR/trajectory_dual_ur3e.npz" \
    --solve-only \
    --save-on-solve \
    "$@"

"$REPOSITORY_ROOT/deployment/run_pipeline.sh" validate-package "$PACKAGE_DIR"

cat <<EOF
Replay package ready: $PACKAGE_DIR

Build and replay it with:
  ./isaac_export/run_pipeline_local.sh build-usd --package-dir "$PACKAGE_DIR" --headless
  ./isaac_export/run_pipeline_local.sh replay --package-dir "$PACKAGE_DIR" --realtime
EOF
