# ───────────────────────── reconstruction pipeline paths ─────────────────────────
# Sourced by run_pipeline.sh; the Python scripts read these via os.environ

# Root of this extracted project 
RECON_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export RECON_ROOT

# ── Modules ──
export SAM3D_DIR="$RECON_ROOT/modules/sam-3d-objects"
export FASTSAM3D_DIR="$RECON_ROOT/modules/Fast-SAM3D"
export HAWOR_DIR="$RECON_ROOT/modules/HaWoR"
export TAPNET_DIR="$RECON_ROOT/modules/tapnet"
export SAM3_PKG_DIR="$RECON_ROOT/modules/sam3"

export SCRIPTS_DIR="$RECON_ROOT/scripts"

# ── Which repo root the MoGe pointmap script imports from ──
export SAM3D_REPO_ROOT="$FASTSAM3D_DIR"         

# ── Versioned external resources ──
# Production machines use the release selected below. A checkout that does not
# have the managed package yet continues to work with the legacy repo-local
# paths, so migration is reversible.
export DO_AS_I_DO_PACKAGE_ROOT="${DO_AS_I_DO_PACKAGE_ROOT:-/data/jiaqimeng/retargeting_dev}"
export DO_AS_I_DO_RELEASE="${DO_AS_I_DO_RELEASE:-current}"
if [[ "$DO_AS_I_DO_RELEASE" = /* ]]; then
    export DO_AS_I_DO_RELEASE_DIR="$DO_AS_I_DO_RELEASE"
else
    export DO_AS_I_DO_RELEASE_DIR="$DO_AS_I_DO_PACKAGE_ROOT/$DO_AS_I_DO_RELEASE"
fi
export DO_AS_I_DO_RUNTIME_DIR="$DO_AS_I_DO_RELEASE_DIR/runtime"

_prefer_managed_dir() {
    if [[ -d "$1" ]]; then
        printf '%s\n' "$1"
    else
        printf '%s\n' "$2"
    fi
}

_prefer_managed_file() {
    if [[ -s "$1" ]]; then
        printf '%s\n' "$1"
    else
        printf '%s\n' "$2"
    fi
}

_REPO_ROOT="$(cd "$RECON_ROOT/.." && pwd)"
_MANAGED_ASSETS="$DO_AS_I_DO_RELEASE_DIR/assets"

export WEIGHTS_DIR="${WEIGHTS_DIR:-$(_prefer_managed_dir \
    "$_MANAGED_ASSETS/weights/reconstruction" "$RECON_ROOT/weights")}"
export HAWOR_WEIGHTS_DIR="${HAWOR_WEIGHTS_DIR:-$(_prefer_managed_dir \
    "$_MANAGED_ASSETS/weights/hawor" "$HAWOR_DIR/weights")}"
export METRIC3D_WEIGHTS_DIR="${METRIC3D_WEIGHTS_DIR:-$(_prefer_managed_dir \
    "$_MANAGED_ASSETS/weights/metric3d" "$HAWOR_DIR/thirdparty/Metric3D/weights")}"
export DEPLOYMENT_ASSETS_DIR="${DEPLOYMENT_ASSETS_DIR:-$(_prefer_managed_dir \
    "$_MANAGED_ASSETS/third-party/deployment-assets" "$_REPO_ROOT/deployment_assets")}"

export TAPNET_CKPT="${TAPNET_CKPT:-$WEIGHTS_DIR/tapnet/bootstapir_checkpoint_v2.pt}"
export SAM3_CHECKPOINT="${SAM3_CHECKPOINT:-$WEIGHTS_DIR/sam3/sam3.pt}"
export MOGE_CHECKPOINT="${MOGE_CHECKPOINT:-$WEIGHTS_DIR/moge-vitl/model.pt}"
export GEOCALIB_CHECKPOINT="${GEOCALIB_CHECKPOINT:-$WEIGHTS_DIR/geocalib/pinhole.tar}"
export DINOV2_REPO="${DINOV2_REPO:-$DEPLOYMENT_ASSETS_DIR/dinov2}"

export SAM3D_CONFIG="${SAM3D_CONFIG:-$(_prefer_managed_file \
    "$DO_AS_I_DO_RUNTIME_DIR/sam3d/checkpoints/hf/pipeline.yaml" \
    "$SAM3D_DIR/checkpoints/hf/pipeline.yaml")}"
export FASTSAM3D_CONFIG="${FASTSAM3D_CONFIG:-$(_prefer_managed_file \
    "$DO_AS_I_DO_RUNTIME_DIR/fast-sam3d/checkpoints/hf/pipeline.yaml" \
    "$FASTSAM3D_DIR/checkpoints/hf/pipeline.yaml")}"

export HAWOR_CKPT="${HAWOR_CKPT:-$HAWOR_WEIGHTS_DIR/hawor/checkpoints/hawor.ckpt}"
export HAWOR_INFILLER_CKPT="${HAWOR_INFILLER_CKPT:-$HAWOR_WEIGHTS_DIR/hawor/checkpoints/infiller.pt}"
export HAWOR_DETECTOR_CKPT="${HAWOR_DETECTOR_CKPT:-$HAWOR_WEIGHTS_DIR/external/detector.pt}"
export HAWOR_DROID_CKPT="${HAWOR_DROID_CKPT:-$HAWOR_WEIGHTS_DIR/external/droid.pth}"
export METRIC3D_CKPT="${METRIC3D_CKPT:-$METRIC3D_WEIGHTS_DIR/metric_depth_vit_large_800k.pth}"

export MANO_RIGHT="${MANO_RIGHT:-$(_prefer_managed_file \
    "$_MANAGED_ASSETS/licensed/mano/MANO_RIGHT.pkl" \
    "$HAWOR_DIR/_DATA/data/mano/MANO_RIGHT.pkl")}"
export MANO_LEFT="${MANO_LEFT:-$(_prefer_managed_file \
    "$_MANAGED_ASSETS/licensed/mano/MANO_LEFT.pkl" \
    "$HAWOR_DIR/_DATA/data_left/mano_left/MANO_LEFT.pkl")}"

if [[ -d "$_MANAGED_ASSETS/weights/reconstruction" ]]; then
    export DO_AS_I_DO_RESOURCE_MODE=managed
else
    export DO_AS_I_DO_RESOURCE_MODE=repo-fallback
fi

# ── Conda env names (4 separate environments) ──
# Defaults match what `setup/01_create_envs.sh` creates. If you are reusing the
# pre-existing local envs, your sam3d env may be named differently
# (sam3 / hawor / tapnet already match) — set ENV_SAM3D accordingly.
_MANAGED_CONDA_DIR="$DO_AS_I_DO_RELEASE_DIR/installed-envs/conda"
_prefer_managed_env() {
    if [[ -x "$1/bin/python" ]]; then
        printf '%s\n' "$1"
    else
        printf '%s\n' "$2"
    fi
}
export ENV_SAM3="${ENV_SAM3:-$(_prefer_managed_env "$_MANAGED_CONDA_DIR/sam3" sam3)}"
export ENV_SAM3D="${ENV_SAM3D:-$(_prefer_managed_env "$_MANAGED_CONDA_DIR/sam3d" sam3d)}"
export ENV_HAWOR="${ENV_HAWOR:-$(_prefer_managed_env "$_MANAGED_CONDA_DIR/hawor" hawor)}"
export ENV_TAPNET="${ENV_TAPNET:-$(_prefer_managed_env "$_MANAGED_CONDA_DIR/tapnet" tapnet)}"

unset _REPO_ROOT _MANAGED_ASSETS _MANAGED_CONDA_DIR
unset -f _prefer_managed_dir _prefer_managed_file _prefer_managed_env

# ── Host / GPU ──
# Respect an explicit per-run GPU selection and otherwise default to GPU 0.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# Optional X display used by the click-based SAM3 segmentation UI (Stage 1).
# Do not invent a display on headless hosts: pass point coordinates to
# run_pipeline.sh or explicitly export DISPLAY/SAM3_DISPLAY after configuring X.
export SAM3_DISPLAY="${SAM3_DISPLAY:-}"
