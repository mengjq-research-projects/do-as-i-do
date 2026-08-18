#!/bin/bash
# [01] Create the 4 conda envs the pipeline switches between. Names come from config/paths.sh.
# Also see env/README.md.
#
# NOTE: these recreate the envs from original source. For an exact reproduction of a working set up, see env/README.md
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/config/paths.sh"
source "$(conda info --base)/etc/profile.d/conda.sh"

NO_PROXY=0
PYPI_MIRROR=""
PYPI_MIRROR_URL=""
PYTORCH_WHEEL_SOURCE_ROOT="https://download.pytorch.org/whl"
PYTORCH_WHEEL_SOURCE_MODE="index"
PROXY_ENV_VARS=(
  http_proxy
  https_proxy
  HTTP_PROXY
  HTTPS_PROXY
  all_proxy
  ALL_PROXY
  ftp_proxy
  FTP_PROXY
)

usage() {
  echo "usage: $0 [--no-proxy] [--pypi-mirror tsinghua] [all|sam3|sam3d|hawor|tapnet]" >&2
}

proxy_configured () {
  local var_name
  for var_name in "${PROXY_ENV_VARS[@]}"; do
    if [[ -n "${!var_name:-}" ]]; then
      return 0
    fi
  done
  return 1
}

print_active_proxy_vars () {
  local var_name
  for var_name in "${PROXY_ENV_VARS[@]}"; do
    if [[ -n "${!var_name:-}" ]]; then
      printf '    - %s=%q\n' "$var_name" "${!var_name}"
    fi
  done
}

clear_proxy_env () {
  local var_name
  for var_name in "${PROXY_ENV_VARS[@]}"; do
    unset "$var_name"
  done
}

clear_inherited_pip_indexes () {
  local var_name
  local cleared=0
  for var_name in PIP_INDEX_URL PIP_EXTRA_INDEX_URL PIP_FIND_LINKS; do
    if [[ -n "${!var_name:-}" ]]; then
      printf '    - ignoring inherited %s=%q\n' "$var_name" "${!var_name}"
      unset "$var_name"
      cleared=1
    fi
  done
  if [[ "$cleared" -eq 1 ]]; then
    echo "=== [01] using only package indexes required by this setup ==="
  fi
}

only="all"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-proxy)
      NO_PROXY=1
      shift
      ;;
    --pypi-mirror)
      if [[ "${2:-}" != "tsinghua" ]]; then
        echo "error: supported PyPI mirror: tsinghua" >&2
        usage
        exit 1
      fi
      PYPI_MIRROR="tsinghua"
      PYPI_MIRROR_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
      PYTORCH_WHEEL_SOURCE_ROOT="https://mirrors.aliyun.com/pytorch-wheels"
      PYTORCH_WHEEL_SOURCE_MODE="find-links"
      shift 2
      ;;
    all|sam3|sam3d|hawor|tapnet)
      only="$1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage
      exit 1
      ;;
  esac
done

if proxy_configured; then
  if [[ "$NO_PROXY" -eq 1 ]]; then
    echo "=== [01] ignoring proxy environment variables for Conda/pip ==="
    print_active_proxy_vars
    clear_proxy_env
  else
    cat <<'EOF'
=== [01] proxy environment variables detected ===
If Conda or pip fails with ConnectTimeoutError or ProxyError, rerun with:
  ./setup/01_create_envs.sh --no-proxy

Also check Conda-specific proxy settings with:
  conda config --show proxy_servers
EOF
    print_active_proxy_vars
  fi
fi

# A shell-wide PIP_EXTRA_INDEX_URL or global pip config can inject unrelated
# indexes (for example, the retired pypi.ngc.nvidia.com) into every install.
# Keep the index selection deterministic; mk_sam3d sets its CUDA index explicitly.
clear_inherited_pip_indexes
export PIP_CONFIG_FILE=/dev/null
if [[ -n "$PYPI_MIRROR_URL" ]]; then
  export PIP_INDEX_URL="$PYPI_MIRROR_URL"
  echo "=== [01] using PyPI mirror: $PIP_INDEX_URL ==="
  echo "=== [01] using PyTorch wheel mirror: $PYTORCH_WHEEL_SOURCE_ROOT ==="
fi
export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-900}"
export PIP_RETRIES="${PIP_RETRIES:-10}"
unset PIP_NO_CACHE_DIR
export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=http.version
export GIT_CONFIG_VALUE_0=HTTP/1.1
export MAX_JOBS="${MAX_JOBS:-4}"

ensure_conda_env () {
  local environment_name="$1"
  local python_version="$2"
  if conda env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -Fxq "$environment_name"; then
    echo "=== [01] reusing $environment_name with Python $python_version ==="
    conda install -y -n "$environment_name" "python=$python_version"
  else
    conda create -y -n "$environment_name" "python=$python_version"
  fi
}

activate_conda_env () {
  local activation_status
  set +e
  conda activate "$1"
  activation_status=$?
  set -e
  if [[ "$activation_status" -ne 0 ]]; then
    echo "error: failed to activate Conda environment '$1'." >&2
    return "$activation_status"
  fi
}

deactivate_conda_env () {
  local deactivation_status
  set +e
  conda deactivate
  deactivation_status=$?
  set -e
  if [[ "$deactivation_status" -ne 0 ]]; then
    echo "error: failed to deactivate the current Conda environment." >&2
    return "$deactivation_status"
  fi
}

pip_install_with_git_retries () {
  local attempt
  local max_attempts=3
  for ((attempt = 1; attempt <= max_attempts; attempt++)); do
    if pip install "$@"; then
      return 0
    fi
    if [[ "$attempt" -lt "$max_attempts" ]]; then
      echo "=== [01] pip install failed; retrying ($((attempt + 1))/$max_attempts) ===" >&2
    fi
  done
  return 1
}

pip_install_pytorch_wheels () {
  local cuda_variant="$1"
  shift
  if [[ "$PYTORCH_WHEEL_SOURCE_MODE" == "find-links" ]]; then
    pip_install_with_git_retries "$@" \
      --find-links "$PYTORCH_WHEEL_SOURCE_ROOT/$cuda_variant"
  else
    pip_install_with_git_retries "$@" \
      --index-url "$PYTORCH_WHEEL_SOURCE_ROOT/$cuda_variant"
  fi
}

install_moge_from_archives () {
  local requirements_without_moge
  requirements_without_moge="$(mktemp)"
  # The upstream requirements file combines inference dependencies with
  # training, quantization, text-model, cloud, documentation, and interactive
  # visualization tooling. The excluded packages are not imported by the
  # run_pipeline.sh reconstruction path. In particular, omitting the CUDA
  # compiler wheel avoids duplicating the Conda CUDA toolkit already installed
  # by environments/default.yml.
  sed -E '/^(MoGe @ git\+https:\/\/github\.com\/microsoft\/MoGe\.git@|astor==|async-timeout==|auto_gptq==|autoflake==|av==|bitsandbytes==|black==|bpy==|colorama==|conda-pack==|crcmod==|cuda-python==|dataclasses==|decord==|deprecation==|einops-exts==|exceptiongroup==|fastavro==|fasteners==|flake8==|Flask==|fqdn==|ftfy==|fvcore==|gdown==|hdfs==|httplib2==|hydra-submitit-launcher==|imath==|isoduration==|jsonlines==|jsonpickle==|jsonpointer==|jupyter==|librosa==|mosaicml-streaming==|nvidia-cuda-nvcc-cu12==|nvidia-pyindex==|objsize==|OpenEXR==|optimum==|optree==|orjson==|panda3d-gltf==|pdoc3==|peft==|pip-system-certs==|point-cloud-utils==|polyscope==|pycocotools==|pydot==|pymongo==|pyrender==|PySocks==|pytest==|python-pycg==|randomname==|roma==|rootutils==|sagemaker==|sentence-transformers==|simplejson==|smplx==|tensorboard==|tomli==|torchaudio==|uri-template==|usort==|wandb==|webcolors==|webdataset==|Werkzeug==)/d' \
    "$SAM3D_DIR/requirements.txt" > "$requirements_without_moge"

  pip_install_with_git_retries -r "$requirements_without_moge"
  rm -f "$requirements_without_moge"

  # Download release snapshots over HTTPS instead of invoking Git clone/fetch.
  # MoGe's own pyproject points utils3d at a Git URL, so install that exact
  # revision first and install MoGe without dependency resolution afterward.
  pip_install_with_git_retries setuptools wheel
  pip_install_with_git_retries \
    "utils3d @ https://codeload.github.com/EasternJournalist/utils3d/tar.gz/3913c65d81e05e47b9f367250cf8c0f7462a0900" \
    --no-build-isolation
  # MoGe only requires setuptools and wheel to build. Reuse the Conda
  # environment instead of downloading an isolated build environment.
  pip_install_with_git_retries \
    "MoGe @ https://codeload.github.com/microsoft/MoGe/tar.gz/a8c37341bc0325ca99b9d57981cc3bb2bd3e255b" \
    --no-deps --no-build-isolation
}

install_gsplat_from_archive () {
  local inference_requirements_without_gsplat
  inference_requirements_without_gsplat="$(mktemp)"
  sed '/^gsplat @ git+https:\/\/github\.com\/nerfstudio-project\/gsplat\.git@/d' \
    "$SAM3D_DIR/requirements.inference.txt" > "$inference_requirements_without_gsplat"

  pip_install_with_git_retries -r "$inference_requirements_without_gsplat"
  rm -f "$inference_requirements_without_gsplat"

  # gsplat is required by Gaussian rendering, but cloning its source over Git
  # is unreliable on this connection. Build the same pinned revision from its
  # HTTPS source archive, without a second isolated Torch build environment.
  pip_install_with_git_retries ninja
  pip_install_with_git_retries \
    "gsplat @ https://codeload.github.com/nerfstudio-project/gsplat/tar.gz/2323de5905d5e90e035f792fe65bad0fedd413e7" \
    --no-build-isolation
}

install_geocalib_from_archive () {
  pip_install_with_git_retries setuptools wheel
  pip_install_with_git_retries \
    "geocalib @ https://codeload.github.com/cvg/GeoCalib/tar.gz/97b8968e7798a66bf04fcf791fb535624241bda7" \
    --no-build-isolation
}

install_hawor_vcs_dependencies () {
  pip_install_with_git_retries iopath ninja setuptools wheel
  pip_install_with_git_retries \
    "pytorch3d @ https://codeload.github.com/facebookresearch/pytorch3d/tar.gz/75ebeeaea0908c5527e7b1e305fbc7681382db47" \
    --no-deps --no-build-isolation
  pip_install_with_git_retries \
    "chumpy @ https://codeload.github.com/mattloper/chumpy/tar.gz/580566eafc9ac68b2614b64d6f7aaa84eebb70da" \
    --no-deps --no-build-isolation
}

configure_hawor_build_toolchain () {
  local compiler_path
  local cuda_header
  local nvidia_package_dir
  local python_site_packages
  local wheel_include_paths=()
  local wheel_library_paths=()
  export CUDA_HOME="$CONDA_PREFIX"
  export FORCE_CUDA=1
  export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"

  for compiler_path in \
    "$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc" \
    "$CONDA_PREFIX/bin/gcc"; do
    if [[ -x "$compiler_path" ]]; then
      export CC="$compiler_path"
      break
    fi
  done
  for compiler_path in \
    "$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++" \
    "$CONDA_PREFIX/bin/g++"; do
    if [[ -x "$compiler_path" ]]; then
      export CXX="$compiler_path"
      break
    fi
  done

  if [[ -z "${CC:-}" || -z "${CXX:-}" || ! -x "$CUDA_HOME/bin/nvcc" ]]; then
    echo "error: HaWoR requires GCC/G++ and NVCC from its Conda environment." >&2
    return 1
  fi

  # PyTorch's cu128 wheels install CUDA libraries and development headers in
  # site-packages/nvidia. Reuse those domestic-mirror downloads rather than
  # downloading a duplicate cuda-libraries-dev stack from NVIDIA Conda.
  python_site_packages="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
  for nvidia_package_dir in "$python_site_packages"/nvidia/*; do
    [[ "$(basename "$nvidia_package_dir")" == "cu13" ]] && continue
    [[ -d "$nvidia_package_dir/include" ]] \
      && wheel_include_paths+=("$nvidia_package_dir/include")
    [[ -d "$nvidia_package_dir/lib" ]] \
      && wheel_library_paths+=("$nvidia_package_dir/lib")
  done
  if [[ "${#wheel_include_paths[@]}" -gt 0 ]]; then
    export CPATH="$(IFS=:; echo "${wheel_include_paths[*]}")${CPATH:+:$CPATH}"
  fi
  if [[ "${#wheel_library_paths[@]}" -gt 0 ]]; then
    export LIBRARY_PATH="$(IFS=:; echo "${wheel_library_paths[*]}")${LIBRARY_PATH:+:$LIBRARY_PATH}"
    export LD_LIBRARY_PATH="$(IFS=:; echo "${wheel_library_paths[*]}")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
  for cuda_header in cublas_v2.h cusparse.h; do
    if ! find "${wheel_include_paths[@]}" \
      "$CUDA_HOME/targets/x86_64-linux/include" "$CUDA_HOME/include" \
      -maxdepth 1 -name "$cuda_header" -print -quit 2>/dev/null | grep -q .; then
      echo "error: missing CUDA development header: $cuda_header" >&2
      return 1
    fi
  done
  echo "=== [01] HaWoR build toolchain: CC=$CC CXX=$CXX NVCC=$CUDA_HOME/bin/nvcc ==="
}

remove_stale_hawor_cuda_packages () {
  local package_name
  local stale_packages=()
  while IFS= read -r package_name; do
    [[ -n "$package_name" ]] && stale_packages+=("$package_name")
  done < <(
    python -m pip list --format=freeze \
      | sed -nE 's/^(nvidia-[a-z0-9-]+-cu13)==.*/\1/p'
  )
  if [[ "${#stale_packages[@]}" -gt 0 ]]; then
    echo "=== [01] removing stale CUDA 13 packages from $ENV_HAWOR ==="
    python -m pip uninstall -y "${stale_packages[@]}"
  fi
}

patch_lietorch_dispatch () {
  local dispatch_header="$HAWOR_DIR/thirdparty/DROID-SLAM/thirdparty/lietorch/lietorch/include/dispatch.h"
  if grep -Fq '::detail::scalar_type(the_type)' "$dispatch_header"; then
    sed -i 's/::detail::scalar_type(the_type)/the_type.scalarType()/' "$dispatch_header"
  fi
}

mk_sam3d () {
  echo "=== [01] sam3d ($ENV_SAM3D) ==="
  # NOTE: default.yml/requirements pin CUDA 12.1 / torch ~2.5; the working env (env/sam3d.yml)
  # runs torch 2.8+cu128. On CUDA 12.8 GPUs prefer:  conda env create -f env/sam3d.yml
  conda env create -n "$ENV_SAM3D" -f "$SAM3D_DIR/environments/default.yml" \
    || conda env update -n "$ENV_SAM3D" -f "$SAM3D_DIR/environments/default.yml"
  activate_conda_env "$ENV_SAM3D"
  conda install -y -c conda-forge ffmpeg   
  (
    # Required by the upstream cu121 pins (for example
    # torchaudio==2.5.1+cu121) and the Kaolin inference dependency.
    if [[ "$PYTORCH_WHEEL_SOURCE_MODE" == "find-links" ]]; then
      unset PIP_EXTRA_INDEX_URL
      export PIP_FIND_LINKS="$PYTORCH_WHEEL_SOURCE_ROOT/cu121 https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"
    else
      export PIP_EXTRA_INDEX_URL="$PYTORCH_WHEEL_SOURCE_ROOT/cu121"
      export PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"
    fi
    install_moge_from_archives
    [ -f "$SAM3D_DIR/requirements.inference.txt" ] && install_gsplat_from_archive
    # Requirements are installed explicitly above. Register the local package
    # without resolving them a second time (which would reclone MoGe).
    pip install hatchling hatch-requirements-txt
    pip install -e "$SAM3D_DIR" --no-deps --no-build-isolation
    pip install viser                       # optional Stage 4 viz (scripts/visualize_3d.py)
    install_geocalib_from_archive            # Stage 2 gravity (scripts/predict_video_gravity.py)
  )
  deactivate_conda_env
}

mk_hawor () {
  local hawor_constraints
  local local_hawor_requirements
  echo "=== [01] hawor ($ENV_HAWOR) — verified CUDA 12.8 / torch 2.9 build ==="
  ensure_conda_env "$ENV_HAWOR" 3.10
  # DROID-SLAM is pinned to Blackwell sm_120 in this repository. Its CUDA
  # extensions therefore require the same CUDA 12.8 + GCC 14 toolchain as the
  # known-working reconstruction/env/hawor.yml, not the upstream cu117 recipe.
  conda install -y -n "$ENV_HAWOR" \
    -c nvidia/label/cuda-12.8.1 -c defaults \
    "cuda-compiler=12.8.1" "gxx_linux-64=14.3.0"
  conda install -y -n "$ENV_HAWOR" -c conda-forge ffmpeg
  activate_conda_env "$ENV_HAWOR"
  remove_stale_hawor_cuda_packages

  pip_install_with_git_retries numpy==1.26.4 pillow requests typing-extensions
  pip_install_pytorch_wheels cu128 \
    torch==2.9.0+cu128 torchvision==0.24.0+cu128
  configure_hawor_build_toolchain

  hawor_constraints="$(mktemp)"
  cat > "$hawor_constraints" <<'EOF'
torch==2.9.0+cu128
torchvision==0.24.0+cu128
numpy==1.26.4
EOF
  # Omit mmcv (unused by the runtime pipeline), VCS sources handled below,
  # and torch-scatter, whose build must see Torch in the active environment.
  local_hawor_requirements="$(mktemp)"
  grep -vE "mmcv==1.3.9|chumpy@|git\+https://github.com/facebookresearch/pytorch3d.git@stable|torch-scatter==2.1.2" \
    "$HAWOR_DIR/requirements.txt" > "$local_hawor_requirements"
  pip_install_with_git_retries -r "$local_hawor_requirements" -c "$hawor_constraints"
  rm -f "$local_hawor_requirements"
  pip_install_with_git_retries setuptools wheel ninja
  pip install torch-scatter==2.1.2 --no-build-isolation -c "$hawor_constraints"
  install_hawor_vcs_dependencies
  pip install "setuptools<81"   # pytorch-lightning 2.2.4 needs pkg_resources (removed in setuptools>=81)
  pip install pytorch-lightning==2.2.4 --no-deps
  pip install lightning-utilities torchmetrics==1.4.0
  rm -f "$hawor_constraints"
  patch_lietorch_dispatch
  ( cd "$HAWOR_DIR/thirdparty/DROID-SLAM" && python setup.py install )
  # torch>=2.6 defaults torch.load(weights_only=True), which rejects HaWoR's checkpoints
  # (they embed an omegaconf DictConfig). These are the official/trusted weights, so restore
  # the pre-2.6 default for this env (no code change needed); applied on every activate.
  conda env config vars set TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 -n "$ENV_HAWOR"
  deactivate_conda_env
}

mk_tapnet () {
  echo "=== [01] tapnet ($ENV_TAPNET) — torch-only BootsTAPIR ==="
  ensure_conda_env "$ENV_TAPNET" 3.10
  activate_conda_env "$ENV_TAPNET"
  conda install -y -c conda-forge ffmpeg   
  pip_install_pytorch_wheels cu128 \
    torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0
  pip install -e "$TAPNET_DIR[torch]"     # IMPORTANT: editable install, NOT sys.path (tapnet/torch shadows torch)
  pip install einops tqdm mediapy
  deactivate_conda_env
}

mk_sam3 () {
  echo "=== [01] sam3 ($ENV_SAM3) — Stage 1 segmentation (cu128) ==="
  ensure_conda_env "$ENV_SAM3" 3.12
  activate_conda_env "$ENV_SAM3"
  conda install -y -c conda-forge ffmpeg   # run_pipeline.sh extracts frames with the sam3 env's ffmpeg
  pip_install_pytorch_wheels cu128 \
    torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
  # The default video pipeline needs OpenCV in its wrapper and psutil in the
  # predictor. Notebook, development, and training extras are not imported.
  pip_install_with_git_retries -e "$SAM3_PKG_DIR"
  pip_install_with_git_retries opencv-python psutil
  deactivate_conda_env
}

case "$only" in
  all)    mk_sam3; mk_sam3d; mk_hawor; mk_tapnet ;;
  sam3)   mk_sam3 ;;
  sam3d)  mk_sam3d ;;
  hawor)  mk_hawor ;;
  tapnet) mk_tapnet ;;
  *) usage; exit 1 ;;
esac

echo "[01] Done."
