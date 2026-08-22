#!/usr/bin/env bash
# Download Isaac Sim on this networked host and install it on PhysicsAssets.
set -Eeuo pipefail

LOCAL_PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE="${REMOTE:-PhysicsAssets}"
REMOTE_PROJECT="${REMOTE_PROJECT:-projects/do-as-i-do}"
REMOTE_CONDA_BASE="${REMOTE_CONDA_BASE:-/home/jiaqimeng/miniforge3}"
ISAAC_VERSION="${ISAAC_VERSION:-6.0.1.0}"
WHEELHOUSE="$LOCAL_PROJECT/deployment_assets/isaac_sim_${ISAAC_VERSION}_wheels"
REMOTE_WHEELHOUSE="do-as-i-do-isaac-wheelhouse-${ISAAC_VERSION}"

mkdir -p "$WHEELHOUSE"

# NVIDIA publishes Isaac Sim itself only on its package index. Ordinary Python
# dependencies use Aliyun; all files are downloaded here and sent by rsync so
# the remote GPU host never needs international network access.
uv pip install --python "$LOCAL_PROJECT/isaac_export/.venv/bin/python" pip
PIP_CONFIG_FILE=/dev/null \
    "$LOCAL_PROJECT/isaac_export/.venv/bin/python" -m pip download \
    --dest "$WHEELHOUSE" \
    --index-url https://mirrors.aliyun.com/pypi/simple \
    --extra-index-url https://pypi.nvidia.com \
    "isaacsim[all,extscache]==$ISAAC_VERSION" wheel

uv build --wheel "$LOCAL_PROJECT/isaac_export" \
    --out-dir "$LOCAL_PROJECT/deployment_assets/wheels"

ssh "$REMOTE" "mkdir -p ~/$REMOTE_WHEELHOUSE"
rsync -a --partial --info=progress2 \
    "$WHEELHOUSE/" "$REMOTE:$REMOTE_WHEELHOUSE/"
rsync -a --partial --info=progress2 \
    "$LOCAL_PROJECT/deployment_assets/wheels/do_as_i_do_isaac_export-0.1.0-py3-none-any.whl" \
    "$REMOTE:$REMOTE_WHEELHOUSE/"

ssh "$REMOTE" bash -s -- \
    "$REMOTE_PROJECT" "$REMOTE_CONDA_BASE" "$REMOTE_WHEELHOUSE" \
    "$ISAAC_VERSION" <<'REMOTE_INSTALL'
set -Eeuo pipefail
remote_project="$1"
conda_base="$2"
wheelhouse="$HOME/$3"
isaac_version="$4"
root="$HOME/$remote_project"
python312="$conda_base/envs/sam3/bin/python"
target="$root/isaac_export/.venv-isaac"

rm -rf "$target"
"$python312" -m venv "$target"
PIP_CONFIG_FILE=/dev/null "$target/bin/python" -m pip install \
    --no-index --find-links "$wheelhouse" \
    "isaacsim[all,extscache]==$isaac_version"
PIP_CONFIG_FILE=/dev/null "$target/bin/python" -m pip install \
    --no-index --find-links "$wheelhouse" --no-deps \
    do-as-i-do-isaac-export==0.1.0

# Importing isaacsim is deliberately omitted: first run presents NVIDIA's EULA.
"$target/bin/python" --version
echo "Isaac Sim files installed at: $target"
echo "EULA has not been accepted or bypassed."
REMOTE_INSTALL
