#!/usr/bin/env bash
# Stable shell entry point for simulation preview and optional robot deployment.
set -Eeuo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="${DO_AS_I_DO_PACKAGE_ROOT:-/data/jiaqimeng/retargeting_dev}"
RELEASE="${DO_AS_I_DO_RELEASE:-current}"
if [[ "$RELEASE" = /* ]]; then
    RELEASE_DIR="$RELEASE"
else
    RELEASE_DIR="${DO_AS_I_DO_RELEASE_DIR:-$PACKAGE_ROOT/$RELEASE}"
fi

case "${1:-}" in
    mujoco-replay)
        ACTION="$1"
        ENTRYPOINT="$HERE/mujoco_replay/replay_retarget.py"
        shift
        ;;
    validate-package)
        ACTION="$1"
        ENTRYPOINT="$HERE/mujoco_replay/validate_package.py"
        shift
        ;;
    robot-replay)
        ACTION="$1"
        ENTRYPOINT="$HERE/robot_replay/run_npz.py"
        shift
        ;;
    home)
        ACTION="$1"
        ENTRYPOINT="$HERE/robot_replay/home.py"
        shift
        ;;
    *)
        cat <<'EOF'
Usage:
  ./deployment/run_pipeline.sh mujoco-replay [REPLAY_ARGS...]
  ./deployment/run_pipeline.sh validate-package PACKAGE_DIR
  ./deployment/run_pipeline.sh robot-replay [ROBOT_ARGS...]
  ./deployment/run_pipeline.sh home [HOME_ARGS...]

The robot-replay and home actions can command real hardware. Validate with
robot-replay --dry-run and keep an emergency stop available.
EOF
        [[ -z "${1:-}" || "${1:-}" == "-h" || "${1:-}" == "--help" ]] && exit 0
        echo "Unknown deployment action: $1" >&2
        exit 2
        ;;
esac

PYTHON_CANDIDATES=(
    "${DEPLOYMENT_PYTHON:-}"
)

# The simulation-only preview uses the same public Python dependencies as
# Retargeting, so the managed offline Retargeting venv is sufficient.  Keep
# the proprietary real-hardware actions on the dedicated Deployment env.
if [[ "$ACTION" == "mujoco-replay" || "$ACTION" == "validate-package" ]]; then
    PYTHON_CANDIDATES+=(
        "${RETARGETING_PYTHON:-}"
        "${RETARGETING_VENV:+${RETARGETING_VENV}/bin/python}"
        "$RELEASE_DIR/installed-envs/venv/retargeting/bin/python"
    )
fi

PYTHON_CANDIDATES+=(
    "/home/jiaqimeng/miniforge3/envs/deployment/bin/python"
    "$HOME/miniforge3/envs/deployment/bin/python"
    "$HOME/miniconda3/envs/deployment/bin/python"
)

if [[ "${CONDA_DEFAULT_ENV:-}" == "deployment" && -n "${CONDA_PREFIX:-}" ]]; then
    PYTHON_CANDIDATES=("$CONDA_PREFIX/bin/python" "${PYTHON_CANDIDATES[@]}")
fi

for candidate in "${PYTHON_CANDIDATES[@]}"; do
    if [[ -n "$candidate" && -x "$candidate" ]]; then
        exec "$candidate" "$ENTRYPOINT" "$@"
    fi
done

if [[ "$ACTION" == "mujoco-replay" || "$ACTION" == "validate-package" ]]; then
    cat >&2 <<'EOF'
Managed Retargeting Python environment was not found.
Restore it from the repository root:
  ./setup_all.sh --managed-offline
EOF
else
    cat >&2 <<'EOF'
Deployment Python environment was not found.
Create it first from the repository root:
  ./setup_all.sh --with-deployment
EOF
fi
exit 1
