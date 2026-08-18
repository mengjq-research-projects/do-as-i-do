#!/usr/bin/env bash
# Stable shell entry point for simulation preview and optional robot deployment.
set -Eeuo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "${1:-}" in
    mujoco-replay)
        ACTION="$1"
        ENTRYPOINT="$HERE/mujoco_replay/replay_retarget.py"
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

cat >&2 <<'EOF'
Deployment Python environment was not found.
Create it first from the repository root:
  ./setup_all.sh --with-deployment
EOF
exit 1
