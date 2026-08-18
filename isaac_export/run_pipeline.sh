#!/usr/bin/env bash
# Stable shell entry point for export, USD generation, and Isaac replay.
set -Eeuo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="${DO_AS_I_DO_PACKAGE_ROOT:-/data/jiaqimeng/retargeting_dev}"
RELEASE="${DO_AS_I_DO_RELEASE:-current}"
if [[ "$RELEASE" = /* ]]; then
    RELEASE_DIR="$RELEASE"
else
    RELEASE_DIR="$PACKAGE_ROOT/$RELEASE"
fi

ACTION="export"
case "${1:-}" in
    export|build-usd|replay)
        ACTION="$1"
        shift
        ;;
    -h|--help)
        cat <<'EOF'
Usage:
  ./isaac_export/run_pipeline.sh export [EXPORT_ARGS...]
  ./isaac_export/run_pipeline.sh build-usd [BUILD_ARGS...]
  ./isaac_export/run_pipeline.sh replay [REPLAY_ARGS...]

With no action, arguments are passed to the standard export command.
EOF
        exit 0
        ;;
esac

PYTHON_CANDIDATES=(
    "${ISAAC_PYTHON:-}"
    "$RELEASE_DIR/installed-envs/venv/isaac/bin/python"
    "$HERE/.venv-isaac/bin/python"
    "$HERE/.venv/bin/python"
)

PYTHON_BIN=""
for candidate in "${PYTHON_CANDIDATES[@]}"; do
    if [[ -n "$candidate" && -x "$candidate" ]]; then
        PYTHON_BIN="$candidate"
        break
    fi
done

if [[ -z "$PYTHON_BIN" && "$ACTION" == "export" ]] && command -v uv >/dev/null 2>&1; then
    exec uv run --project "$HERE" --frozen python "$HERE/export.py" "$@"
fi
if [[ -z "$PYTHON_BIN" ]]; then
    cat >&2 <<'EOF'
Isaac Export Python environment was not found.
Run this from the repository root first:
  ./setup_all.sh --managed-offline
EOF
    exit 1
fi

case "$ACTION" in
    export)
        ENTRYPOINT="$HERE/export.py"
        ;;
    build-usd)
        ENTRYPOINT="$HERE/build_usd.py"
        ;;
    replay)
        ENTRYPOINT="$HERE/replay_isaac.py"
        ;;
esac

if [[ "$ACTION" != "export" ]]; then
    if ! "$PYTHON_BIN" -c \
        'import importlib.metadata as m; m.version("isaacsim")' >/dev/null 2>&1; then
        echo "$ACTION requires the managed/full Isaac Sim environment." >&2
        echo "Run: ./setup_all.sh --managed-offline" >&2
        exit 1
    fi
fi

exec "$PYTHON_BIN" "$ENTRYPOINT" "$@"
