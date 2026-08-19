#!/usr/bin/env bash
# Prepare the environments used by Do as I Do. The default setup covers the
# four Reconstruction Conda environments plus the workspace, Retargeting, and
# Isaac Level A uv projects. Model weights and Deployment are opt-in. Isaac Sim
# itself is not installed by this script.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRY_RUN=0
SKIP_SUBMODULES=0
SKIP_RECONSTRUCTION=0
SKIP_RETARGETING=0
SKIP_ISAAC_EXPORT=0
SKIP_WORKSPACE=0
WITH_WEIGHTS=0
WEIGHTS_LOCAL_DIR=""
WITH_DEPLOYMENT=0
NO_PROXY=0
PYPI_MIRROR=""
PYPI_MIRROR_URL=""
MANAGED_OFFLINE=0
MANAGED_PACKAGE_ROOT="${DO_AS_I_DO_PACKAGE_ROOT:-/data/jiaqimeng/retargeting_dev}"
MANAGED_RELEASE="${DO_AS_I_DO_MANAGED_RELEASE:-2026.08-a100}"
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
    cat <<'EOF'
Usage: ./setup_all.sh [options]

Create all default project environments in one pass. Existing Reconstruction
Conda environments are reused, so the command is safe to rerun after a partial
failure.

Options:
  --managed-offline           Restore the validated external runtime package.
                              This mode never contacts package indexes or GitHub.
  --package-root DIR          Managed package root (default:
                              /data/jiaqimeng/retargeting_dev).
  --release ID                Managed release ID (default: 2026.08-a100).
  --no-proxy                 Ignore *_PROXY environment variables for setup commands.
                             Use this when Conda/uv should connect directly.
  --pypi-mirror NAME         Use domestic mirrors for Python package downloads.
                             `tsinghua` uses Tsinghua PyPI and Aliyun PyTorch wheels.
  --with-weights              Download Reconstruction model weights after setup.
                              Requires Hugging Face access/login; MANO remains a
                              manual download because of its license.
  --weights-from-local DIR    Copy Reconstruction weights from a local source tree.
  --with-deployment           Also create the optional deployment Conda environment.
  --skip-submodules           Do not initialize/update git submodules.
  --skip-reconstruction       Do not create sam3/sam3d/hawor/tapnet environments.
  --skip-retargeting          Do not run uv sync in retargeting/.
  --skip-isaac-export         Do not run uv sync in isaac_export/.
  --skip-workspace            Do not create the root development-tool environment.
  --dry-run                   Print the operations without changing the machine.
  -h, --help                  Show this help.

Not installed here:
  - Source mode: install a frozen Isaac Sim version separately for USD/replay.
  - Managed mode: Isaac Sim 6.0.1.0 is restored from its offline wheelhouse.
  - MANO models: download manually from https://mano.is.tue.mpg.de.
  - Sharpa Wave SDK: proprietary; only needed for real-hardware deployment.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --managed-offline)
            MANAGED_OFFLINE=1
            shift
            ;;
        --package-root)
            [[ $# -ge 2 ]] || { echo "error: --package-root needs DIR" >&2; exit 2; }
            MANAGED_PACKAGE_ROOT="$2"
            shift 2
            ;;
        --release)
            [[ $# -ge 2 ]] || { echo "error: --release needs ID" >&2; exit 2; }
            MANAGED_RELEASE="$2"
            shift 2
            ;;
        --with-weights)
            WITH_WEIGHTS=1
            shift
            ;;
        --weights-from-local)
            [[ $# -ge 2 ]] || { echo "error: --weights-from-local needs DIR" >&2; exit 2; }
            WITH_WEIGHTS=1
            WEIGHTS_LOCAL_DIR="$2"
            shift 2
            ;;
        --with-deployment)
            WITH_DEPLOYMENT=1
            shift
            ;;
        --no-proxy)
            NO_PROXY=1
            shift
            ;;
        --pypi-mirror)
            [[ $# -ge 2 ]] || { echo "error: --pypi-mirror needs a mirror name." >&2; exit 2; }
            case "$2" in
                tsinghua)
                    PYPI_MIRROR="tsinghua"
                    PYPI_MIRROR_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
                    ;;
                *)
                    echo "error: unsupported PyPI mirror: $2 (supported: tsinghua)" >&2
                    exit 2
                    ;;
            esac
            shift 2
            ;;
        --skip-submodules)
            SKIP_SUBMODULES=1
            shift
            ;;
        --skip-reconstruction)
            SKIP_RECONSTRUCTION=1
            shift
            ;;
        --skip-retargeting)
            SKIP_RETARGETING=1
            shift
            ;;
        --skip-isaac-export)
            SKIP_ISAAC_EXPORT=1
            shift
            ;;
        --skip-workspace)
            SKIP_WORKSPACE=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "error: unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

log() {
    printf '\n==> %s\n' "$1"
}

proxy_configured() {
    local var_name
    for var_name in "${PROXY_ENV_VARS[@]}"; do
        if [[ -n "${!var_name:-}" ]]; then
            return 0
        fi
    done
    return 1
}

print_active_proxy_vars() {
    local var_name
    for var_name in "${PROXY_ENV_VARS[@]}"; do
        if [[ -n "${!var_name:-}" ]]; then
            printf '    - %s=%q\n' "$var_name" "${!var_name}"
        fi
    done
}

clear_proxy_env() {
    local var_name
    for var_name in "${PROXY_ENV_VARS[@]}"; do
        unset "$var_name"
    done
}

configure_proxy_mode() {
    if ! proxy_configured; then
        return
    fi

    if [[ "$NO_PROXY" -eq 1 ]]; then
        log "Ignoring proxy environment variables for setup commands"
        print_active_proxy_vars
        clear_proxy_env
        return
    fi

    log "Detected proxy environment variables"
    print_active_proxy_vars
    cat <<'EOF'
    ! Networked setup steps will use those proxies.
    ! If Conda or uv fails with ConnectTimeoutError or ProxyError, rerun:
        ./setup_all.sh --no-proxy
    ! Also check Conda-specific proxy settings:
        conda config --show proxy_servers
EOF
}

print_command() {
    printf '    +'
    printf ' %q' "$@"
    printf '\n'
}

run() {
    print_command "$@"
    if [[ "$DRY_RUN" -eq 0 ]]; then
        "$@"
    fi
}

run_in_dir() {
    local directory="$1"
    shift
    printf '    + cd %q &&' "$directory"
    printf ' %q' "$@"
    printf '\n'
    if [[ "$DRY_RUN" -eq 0 ]]; then
        (cd "$directory" && "$@")
    fi
}

run_uv_sync() {
    local directory="$1"
    shift
    if [[ -n "$PYPI_MIRROR_URL" ]]; then
        # Retargeting names its ordinary PyPI index `pypi-public`; keep its
        # explicit NVIDIA index untouched for warp-lang.
        run_in_dir "$directory" env \
            "UV_DEFAULT_INDEX=$PYPI_MIRROR_URL" \
            "UV_INDEX_PYPI_PUBLIC_URL=$PYPI_MIRROR_URL" \
            "UV_HTTP_TIMEOUT=900" \
            "GIT_CONFIG_COUNT=1" \
            "GIT_CONFIG_KEY_0=http.version" \
            "GIT_CONFIG_VALUE_0=HTTP/1.1" \
            "$UV_BIN" sync "$@"
    else
        run_in_dir "$directory" env \
            "UV_HTTP_TIMEOUT=900" \
            "GIT_CONFIG_COUNT=1" \
            "GIT_CONFIG_KEY_0=http.version" \
            "GIT_CONFIG_VALUE_0=HTTP/1.1" \
            "$UV_BIN" sync "$@"
    fi
}

require_command() {
    local command_name="$1"
    local install_hint="$2"
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "error: required command '$command_name' was not found." >&2
        echo "       $install_hint" >&2
        exit 1
    fi
}

select_manager_python() {
    if [[ -n "${DO_AS_I_DO_MANAGER_PYTHON:-}" ]]; then
        MANAGER_PYTHON="$DO_AS_I_DO_MANAGER_PYTHON"
    elif [[ -x /home/jiaqimeng/miniforge3/bin/python ]]; then
        MANAGER_PYTHON=/home/jiaqimeng/miniforge3/bin/python
    elif command -v python3 >/dev/null 2>&1; then
        MANAGER_PYTHON="$(command -v python3)"
    else
        echo "error: Python 3 is required to restore the managed package." >&2
        exit 1
    fi
}

ensure_conda_available() {
    if command -v conda >/dev/null 2>&1; then
        return
    fi
    local candidate
    for candidate in \
        "${DO_AS_I_DO_CONDA_EXE:-}" \
        /home/jiaqimeng/miniforge3/bin/conda \
        "$HOME/miniforge3/bin/conda" \
        "$HOME/miniconda3/bin/conda"; do
        if [[ -n "$candidate" && -x "$candidate" ]]; then
            export PATH="$(dirname "$candidate"):$PATH"
            return
        fi
    done
}

run_managed() {
    run "$MANAGER_PYTHON" "$ROOT/dependency_management/manage.py" "$@" \
        --root "$MANAGED_PACKAGE_ROOT" --release "$MANAGED_RELEASE"
}

setup_managed_offline() {
    local managed_mano_dir="$MANAGED_PACKAGE_ROOT/releases/$MANAGED_RELEASE/assets/licensed/mano"
    local managed_mano_summary
    select_manager_python
    log "Using managed offline release: $MANAGED_PACKAGE_ROOT/releases/$MANAGED_RELEASE"
    echo "    = No GitHub, PyPI, Hugging Face, or NVIDIA requests are permitted."
    if [[ "$WITH_DEPLOYMENT" -eq 1 ]]; then
        echo "error: Deployment is not part of the managed offline release." >&2
        exit 2
    fi
    if [[ -n "$WEIGHTS_LOCAL_DIR" ]]; then
        echo "error: --weights-from-local cannot be combined with --managed-offline." >&2
        exit 2
    fi
    if [[ "$WITH_WEIGHTS" -eq 1 ]]; then
        echo "    = --with-weights is unnecessary; release weights are already managed."
    fi
    if [[ -n "$PYPI_MIRROR" || "$NO_PROXY" -eq 1 ]]; then
        echo "    = Network mirror/proxy options are unnecessary in offline mode."
    fi

    if [[ "$DRY_RUN" -eq 0 ]]; then
        require_command "$MANAGER_PYTHON" "Set DO_AS_I_DO_MANAGER_PYTHON to Python 3."
        require_command tar "Install tar before restoring Conda environment packs."
        require_command zstd "Install zstd before restoring Conda environment packs."
        if [[ "$SKIP_SUBMODULES" -eq 0 ]]; then
            require_command git "Sync the complete Git checkout from the relay first."
        fi
        if [[ "$SKIP_RECONSTRUCTION" -eq 0 ]]; then
            ensure_conda_available
            require_command conda "Install Miniforge/Conda before running Reconstruction."
        fi
    fi

    if [[ "$SKIP_SUBMODULES" -eq 0 ]]; then
        log "Verifying maintained fork pins without network access"
        run_managed verify-submodules
    fi

    if [[ "$SKIP_RECONSTRUCTION" -eq 0 ]]; then
        log "Preparing managed Reconstruction resources"
        run_managed verify-assets
        run_managed prepare-runtime
        run_managed verify-runtime

        log "Restoring managed Reconstruction Conda environments"
        run_managed install-conda
        run_managed verify-installed-conda
    fi

    local venv_selection=()
    if [[ "$SKIP_RETARGETING" -eq 0 ]]; then
        venv_selection+=(--environment retargeting)
    fi
    if [[ "$SKIP_ISAAC_EXPORT" -eq 0 ]]; then
        venv_selection+=(--environment isaac)
    fi
    if [[ "${#venv_selection[@]}" -gt 0 ]]; then
        log "Restoring managed Retargeting and Isaac environments"
        run_managed verify-uv-cache
        run_managed install-venvs "${venv_selection[@]}"
        run_managed verify-installed-venvs "${venv_selection[@]}"
    fi

    log "Checking user-provided licensed MANO assets"
    if [[ -s "$managed_mano_dir/MANO_LEFT.pkl" && \
          -s "$managed_mano_dir/MANO_RIGHT.pkl" ]]; then
        echo "MANO_LEFT.pkl: PASS ($managed_mano_dir/MANO_LEFT.pkl)"
        echo "MANO_RIGHT.pkl: PASS ($managed_mano_dir/MANO_RIGHT.pkl)"
        managed_mano_summary="MANO: available from the user-provided licensed asset directory."
    else
        echo "MANO: not installed (required only for uncached HaWoR reconstruction)."
        echo "Install both licensed files under: $managed_mano_dir"
        managed_mano_summary="MANO: manual licensed asset not installed; cached demos can still run."
    fi

    log "Managed offline setup complete"
    cat <<EOF
Managed release: $MANAGED_PACKAGE_ROOT/releases/$MANAGED_RELEASE

Activate convenient interpreter variables in the current shell:
  source dependency_management/activate.sh

$managed_mano_summary
Existing repo-local environments were not modified or removed.
EOF
}

if [[ "$MANAGED_OFFLINE" -eq 1 ]]; then
    setup_managed_offline
    exit 0
fi

configure_proxy_mode

ensure_uv() {
    if command -v uv >/dev/null 2>&1; then
        UV_BIN="$(command -v uv)"
        return
    fi
    if [[ -x "$HOME/.local/bin/uv" ]]; then
        UV_BIN="$HOME/.local/bin/uv"
        return
    fi

    log "Installing uv in the user account"
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "    + download and run https://astral.sh/uv/install.sh"
        UV_BIN="uv"
        return
    fi

    require_command curl "Install curl, then rerun this script."
    local installer
    installer="$(mktemp)"
    trap 'rm -f "$installer"' RETURN
    curl -LsSf https://astral.sh/uv/install.sh -o "$installer"
    sh "$installer"
    rm -f "$installer"
    trap - RETURN

    if [[ -x "$HOME/.local/bin/uv" ]]; then
        UV_BIN="$HOME/.local/bin/uv"
    elif command -v uv >/dev/null 2>&1; then
        UV_BIN="$(command -v uv)"
    else
        echo "error: uv installation completed but the executable was not found." >&2
        exit 1
    fi
}

conda_env_exists() {
    local environment_name="$1"
    conda env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -Fxq "$environment_name"
}

if [[ "$DRY_RUN" -eq 0 ]]; then
    require_command git "Install git before setting up this repository."
    if [[ "$SKIP_RECONSTRUCTION" -eq 0 || "$WITH_DEPLOYMENT" -eq 1 ]]; then
        require_command conda "Install Miniconda or Miniforge, then rerun this script."
    fi
fi

if [[ "$SKIP_SUBMODULES" -eq 0 ]]; then
    log "Initializing pinned third-party submodules"
    run bash "$ROOT/reconstruction/setup/00_init_submodules.sh"
fi

if [[ "$SKIP_RECONSTRUCTION" -eq 0 ]]; then
    log "Preparing Reconstruction Conda environments"
    # shellcheck source=reconstruction/config/paths.sh
    source "$ROOT/reconstruction/config/paths.sh"
    reconstruction_specs=(
        "sam3:$ENV_SAM3"
        "sam3d:$ENV_SAM3D"
        "hawor:$ENV_HAWOR"
        "tapnet:$ENV_TAPNET"
    )
    for spec in "${reconstruction_specs[@]}"; do
        setup_target="${spec%%:*}"
        environment_name="${spec#*:}"
        if [[ "$DRY_RUN" -eq 0 ]] && conda_env_exists "$environment_name"; then
            echo "    = Conda environment '$environment_name' already exists; reconciling its packages."
        fi
        reconstruction_setup_args=()
        if [[ -n "$PYPI_MIRROR" ]]; then
            reconstruction_setup_args+=(--pypi-mirror "$PYPI_MIRROR")
        fi
        run bash "$ROOT/reconstruction/setup/01_create_envs.sh" \
            "${reconstruction_setup_args[@]}" "$setup_target"
    done
fi

if [[ "$SKIP_WORKSPACE" -eq 0 || "$SKIP_RETARGETING" -eq 0 || "$SKIP_ISAAC_EXPORT" -eq 0 ]]; then
    ensure_uv
    log "Using uv: $UV_BIN"
    run "$UV_BIN" --version
    if [[ -n "$PYPI_MIRROR" ]]; then
        log "Using PyPI mirror for uv: $PYPI_MIRROR_URL"
    fi
fi

if [[ "$SKIP_WORKSPACE" -eq 0 ]]; then
    log "Preparing root development tools"
    run_uv_sync "$ROOT" --group dev
fi

if [[ "$SKIP_RETARGETING" -eq 0 ]]; then
    log "Preparing Retargeting"
    run_uv_sync "$ROOT/retargeting" --locked
fi

if [[ "$SKIP_ISAAC_EXPORT" -eq 0 ]]; then
    log "Preparing Isaac Level A standard exporter"
    run_uv_sync "$ROOT/isaac_export"
fi

if [[ "$WITH_WEIGHTS" -eq 1 ]]; then
    log "Preparing Reconstruction model weights"
    if [[ -n "$WEIGHTS_LOCAL_DIR" ]]; then
        run bash "$ROOT/reconstruction/setup/02_fetch_weights.sh" \
            --from-local "$WEIGHTS_LOCAL_DIR"
    else
        run bash "$ROOT/reconstruction/setup/02_fetch_weights.sh" --download
    fi
fi

if [[ "$WITH_DEPLOYMENT" -eq 1 ]]; then
    log "Preparing optional Deployment environment"
    if [[ "$DRY_RUN" -eq 0 ]] && conda_env_exists deployment; then
        echo "    = Conda environment 'deployment' already exists; reusing it."
    else
        run conda env create -f "$ROOT/deployment/env/deployment.yml"
    fi
fi

log "Environment setup complete"
cat <<'EOF'
Next checks:
  conda info --envs
  uv run --directory retargeting python -c "import retargeting"
  uv run --directory isaac_export python export.py

For Reconstruction model downloads, rerun with --with-weights after completing
Hugging Face authentication. MANO files and Isaac Sim remain manual installs.
EOF
