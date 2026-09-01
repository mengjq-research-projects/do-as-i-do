#!/usr/bin/env bash
# Stable shell entry point for the Retargeting pipeline.
set -Eeuo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="${DO_AS_I_DO_PACKAGE_ROOT:-/data/jiaqimeng/retargeting_dev}"
RELEASE="${DO_AS_I_DO_RELEASE:-current}"
if [[ "$RELEASE" = /* ]]; then
    RELEASE_DIR="$RELEASE"
else
    RELEASE_DIR="$PACKAGE_ROOT/$RELEASE"
fi

PYTHON_CANDIDATES=(
    "${RETARGETING_PYTHON:-}"
    "$RELEASE_DIR/installed-envs/venv/retargeting/bin/python"
    "$HERE/.venv/bin/python"
)

ENTRYPOINT="launch.py"
NEEDS_HEADLESS_GL=0
case "${1:-}" in
    replay)
        ENTRYPOINT="replay_viser.py"
        shift
        ;;
    capture-replay)
        ENTRYPOINT="capture_replay_frames.py"
        shift
        ;;
    render-evolution)
        ENTRYPOINT="render_optimization_evolution.py"
        NEEDS_HEADLESS_GL=1
        shift
        ;;
esac

for candidate in "${PYTHON_CANDIDATES[@]}"; do
    if [[ -n "$candidate" && -x "$candidate" ]]; then
        cd "$HERE"
        if [[ "$NEEDS_HEADLESS_GL" -eq 1 && -z "${DISPLAY:-}" && -z "${MUJOCO_GL:-}" ]]; then
            if ! command -v xvfb-run >/dev/null 2>&1; then
                cat >&2 <<'EOF'
Headless MuJoCo rendering requires xvfb-run when DISPLAY and MUJOCO_GL are unset.
Install Xvfb, or explicitly configure a working MuJoCo GL backend.
EOF
                exit 1
            fi
            exec xvfb-run -a env MUJOCO_GL=glx "$candidate" "$ENTRYPOINT" "$@"
        fi
        exec "$candidate" "$ENTRYPOINT" "$@"
    fi
done

if command -v uv >/dev/null 2>&1; then
    cd "$HERE"
    exec uv run --frozen python "$ENTRYPOINT" "$@"
fi

cat >&2 <<'EOF'
Retargeting Python environment was not found.
Run this from the repository root first:
  ./setup_all.sh --managed-offline
EOF
exit 1
