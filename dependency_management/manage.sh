#!/usr/bin/env bash
# Stable shell entry point for versioned dependency/resource management.
set -Eeuo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_CANDIDATES=(
    "${DO_AS_I_DO_MANAGER_PYTHON:-}"
    "/home/jiaqimeng/miniforge3/bin/python"
    "$HOME/miniforge3/bin/python"
    "$HOME/miniconda3/bin/python"
    "$(command -v python3 2>/dev/null || true)"
)

for candidate in "${PYTHON_CANDIDATES[@]}"; do
    if [[ -n "$candidate" && -x "$candidate" ]]; then
        exec "$candidate" "$HERE/manage.py" "$@"
    fi
done

echo "Python 3 was not found; set DO_AS_I_DO_MANAGER_PYTHON explicitly." >&2
exit 1
