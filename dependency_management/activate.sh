#!/usr/bin/env bash
# Source this file to use the versioned runtime without placing environments in Git.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "This script must be sourced: source dependency_management/activate.sh" >&2
    exit 2
fi

_MANAGED_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_MANAGED_REPO_ROOT="$(cd "$_MANAGED_SCRIPT_DIR/.." && pwd)"

export DO_AS_I_DO_PACKAGE_ROOT="${DO_AS_I_DO_PACKAGE_ROOT:-/data/jiaqimeng/retargeting_dev}"
export DO_AS_I_DO_RELEASE="${DO_AS_I_DO_RELEASE:-current}"
if [[ "$DO_AS_I_DO_RELEASE" = /* ]]; then
    export DO_AS_I_DO_RELEASE_DIR="$DO_AS_I_DO_RELEASE"
else
    export DO_AS_I_DO_RELEASE_DIR="$DO_AS_I_DO_PACKAGE_ROOT/$DO_AS_I_DO_RELEASE"
fi

export DO_AS_I_DO_UV="$DO_AS_I_DO_RELEASE_DIR/tools/uv/uv"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$DO_AS_I_DO_PACKAGE_ROOT/cache/uv}"
export RETARGETING_VENV="$DO_AS_I_DO_RELEASE_DIR/installed-envs/venv/retargeting"
export ISAAC_VENV="$DO_AS_I_DO_RELEASE_DIR/installed-envs/venv/isaac"
export RETARGETING_PYTHON="$RETARGETING_VENV/bin/python"
export ISAAC_PYTHON="$ISAAC_VENV/bin/python"
export DO_AS_I_DO_CONDA_BASE="${DO_AS_I_DO_CONDA_BASE:-/home/jiaqimeng/miniforge3}"

# This also exports the managed model paths and the four full Conda prefixes.
# shellcheck source=../reconstruction/config/paths.sh
source "$_MANAGED_REPO_ROOT/reconstruction/config/paths.sh"

_managed_missing=0
for _managed_required in \
    "$DO_AS_I_DO_UV" \
    "$RETARGETING_PYTHON" \
    "$ISAAC_PYTHON"; do
    if [[ ! -x "$_managed_required" ]]; then
        echo "Managed runtime is incomplete: $_managed_required" >&2
        _managed_missing=1
    fi
done

unset _MANAGED_SCRIPT_DIR _MANAGED_REPO_ROOT _managed_required
if [[ "$_managed_missing" -ne 0 ]]; then
    unset _managed_missing
    return 1
fi
unset _managed_missing
