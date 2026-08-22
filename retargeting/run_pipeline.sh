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
if [[ "${1:-}" == "replay" ]]; then
    ENTRYPOINT="replay_viser.py"
    shift
fi

for candidate in "${PYTHON_CANDIDATES[@]}"; do
    if [[ -n "$candidate" && -x "$candidate" ]]; then
        cd "$HERE"
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
