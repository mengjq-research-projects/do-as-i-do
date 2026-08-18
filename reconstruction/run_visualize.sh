#!/usr/bin/env bash
# Shell entry point for the optional Reconstruction 3D viewer.
set -Eeuo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=config/paths.sh
source "$HERE/config/paths.sh"

if [[ -x "$ENV_SAM3D/bin/python" ]]; then
    exec "$ENV_SAM3D/bin/python" "$HERE/scripts/visualize_3d.py" "$@"
fi

if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
elif [[ -f /home/jiaqimeng/miniforge3/etc/profile.d/conda.sh ]]; then
    CONDA_BASE=/home/jiaqimeng/miniforge3
elif [[ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]]; then
    CONDA_BASE="$HOME/miniforge3"
else
    echo "Conda was not found. Run ./setup_all.sh --managed-offline first." >&2
    exit 1
fi

# shellcheck disable=SC1090
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_SAM3D"
exec python "$HERE/scripts/visualize_3d.py" "$@"
