#!/usr/bin/env bash
# Local wrapper for Isaac export with a writable default output directory.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -n "${DO_AS_I_DO_ISAAC_EXPORT_OUTPUT_DIR:-}" ]]; then
    export DO_AS_I_DO_ISAAC_EXPORT_OUTPUT_DIR
else
    DEFAULT_OUTPUT_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/do-as-i-do/isaac_export/outputs"
    if mkdir -p "${DEFAULT_OUTPUT_DIR}" 2>/dev/null; then
        export DO_AS_I_DO_ISAAC_EXPORT_OUTPUT_DIR="${DEFAULT_OUTPUT_DIR}"
    else
        FALLBACK_OUTPUT_DIR="/tmp/${USER:-user}/do-as-i-do/isaac_export/outputs"
        mkdir -p "${FALLBACK_OUTPUT_DIR}"
        export DO_AS_I_DO_ISAAC_EXPORT_OUTPUT_DIR="${FALLBACK_OUTPUT_DIR}"
    fi
fi

exec "${HERE}/run_pipeline.sh" "$@"
