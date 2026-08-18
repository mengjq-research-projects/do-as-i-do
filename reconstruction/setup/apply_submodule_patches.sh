#!/usr/bin/env bash
# Apply the project-specific changes kept outside the upstream submodules.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATCH_ROOT="$ROOT/patches"

apply_patch_once() {
    local module="$1"
    local patch="$2"

    if [[ ! -d "$module/.git" && ! -f "$module/.git" ]]; then
        echo "[patches] Missing initialized submodule: $module" >&2
        exit 1
    fi
    if [[ ! -f "$patch" ]]; then
        echo "[patches] Missing patch file: $patch" >&2
        exit 1
    fi

    if git -C "$module" apply --check "$patch" >/dev/null 2>&1; then
        git -C "$module" apply "$patch"
        echo "[patches] Applied: $(basename "$patch")"
    elif git -C "$module" apply --reverse --check "$patch" >/dev/null 2>&1; then
        echo "[patches] Already applied: $(basename "$patch")"
    else
        echo "[patches] Cannot apply cleanly: $patch" >&2
        echo "[patches] Restore the pinned submodule checkout and rerun setup." >&2
        exit 1
    fi
}

apply_patch_once \
    "$ROOT/modules/Fast-SAM3D" \
    "$PATCH_ROOT/Fast-SAM3D.patch"
apply_patch_once \
    "$ROOT/modules/HaWoR" \
    "$PATCH_ROOT/HaWoR.patch"
apply_patch_once \
    "$ROOT/modules/sam-3d-objects" \
    "$PATCH_ROOT/sam-3d-objects.patch"
apply_patch_once \
    "$ROOT/modules/HaWoR/thirdparty/DROID-SLAM/thirdparty/lietorch" \
    "$PATCH_ROOT/lietorch.patch"

echo "[patches] Project submodule patches: PASS"
