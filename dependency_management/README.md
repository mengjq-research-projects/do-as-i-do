# Dependency release management

The runtime code and large external dependencies are released separately.
Tracked manifests in `releases/` describe the exact environment and asset
baseline; binary artifacts live outside Git under
`/data/jiaqimeng/retargeting_dev`.

The `2026.08-a100` release is the first A100 asset baseline. Runtime code
prefers `/data/jiaqimeng/retargeting_dev/current` and falls back to legacy
repo-local paths when a managed artifact is not present. This keeps the
migration reversible while removing large model files from the code checkout.

For a normal developer checkout on `PhysicsAssets`, restore and verify the
entire managed runtime with one offline command:

```bash
./setup_all.sh --managed-offline
```

This command never initializes/downloads submodules and never contacts a
package index. It already runs runtime preparation and verification internally;
no `manage.sh` command is required afterward. Use `--managed-offline --dry-run`
to inspect every operation first.

## Release maintainer and troubleshooting commands

The commands below are granular building blocks used to publish a release or
diagnose one failed setup phase. They are not additional newcomer setup steps.

Copy the asset baseline without changing runtime paths:

```bash
./dependency_management/manage.sh sync-assets
```

Verify every copied file against the generated SHA256 manifest:

```bash
./dependency_management/manage.sh verify-assets
```

Build the small SAM3D runtime config/link trees after cloning or updating the
code. This does not copy model weights again and does not modify submodules:

```bash
./dependency_management/manage.sh prepare-runtime
./dependency_management/manage.sh verify-runtime
```

Inspect the paths that the reconstruction pipeline will actually use:

```bash
source reconstruction/config/paths.sh
printf 'mode=%s\nrelease=%s\nsam3=%s\nsam3d=%s\nhawor=%s\ndinov2=%s\n' \
  "$DO_AS_I_DO_RESOURCE_MODE" "$DO_AS_I_DO_RELEASE_DIR" \
  "$SAM3_CHECKPOINT" "$SAM3D_CONFIG" "$HAWOR_CKPT" "$DINOV2_REPO"
```

Set `DO_AS_I_DO_PACKAGE_ROOT` or `DO_AS_I_DO_RELEASE` before sourcing
`paths.sh` to select another package root or release. Individual paths such as
`SAM3_CHECKPOINT` and `MANO_RIGHT` can also be overridden explicitly.

The managed release covers model weights, the offline DINO source, the Isaac
wheelhouse, relocatable packs for the four Reconstruction Conda environments,
and reproducible external venvs for Retargeting and Isaac. Nothing under a
repo-local `.venv` is copied into the release.

## Reconstruction Conda environment packs

Release maintainers build the four relocatable Reconstruction environments
from the validated source prefixes. `conda-pack` itself is pinned and retained
in the release tool wheelhouse, so rebuilding does not need network access:

```bash
PIP_CONFIG_FILE=/dev/null /home/jiaqimeng/miniforge3/bin/python -m pip install \
  --no-index \
  --find-links /data/jiaqimeng/retargeting_dev/current/wheelhouse/tools \
  conda-pack==0.9.2

./dependency_management/manage.sh pack-conda
```

New machines do not need `conda-pack`. They verify and install the archives
using the base Python plus system `tar`/`zstd`:

```bash
./dependency_management/manage.sh verify-conda-packs
./dependency_management/manage.sh install-conda
./dependency_management/manage.sh verify-installed-conda
```

Installed prefixes live under `current/installed-envs/conda/`. After they are
verified, `reconstruction/config/paths.sh` automatically supplies their full
prefixes to `conda activate`; otherwise it falls back to the traditional
environment names. Use `--environment sam3d` (repeatable) to operate on only
selected environments. Archive rebuilds require an explicit `--force` and
never overwrite an installed prefix.

| Environment | Python | Pack size | Validated CUDA |
|---|---:|---:|---:|
| `sam3` | 3.12.13 | 3.53 GB | 12.8 |
| `sam3d` | 3.11.0 | 6.52 GB | 12.1 |
| `hawor` | 3.10.20 | 4.97 GB | 12.8 |
| `tapnet` | 3.10.20 | 3.31 GB | 12.8 |

The tracked release manifest pins each archive size and SHA256. Installed
verification checks the Python version, stage-specific imports, CUDA
availability, and GPU compute capability; this baseline passed on capability
8.0 (A100).

## Retargeting and Isaac offline venvs

Retargeting and Isaac virtual environments are rebuilt rather than copied:
venv launchers contain absolute interpreter paths and are not portable. The
managed `sam3` Python is used as their Python 3.12 base, while the resulting
environments live outside Git under `current/installed-envs/venv/`.

A release maintainer refreshes the pinned uv executable and shared package
cache after updating either `uv.lock`. This is the only command in this stage
that reads the builder's user cache:

```bash
./dependency_management/manage.sh sync-uv-cache
```

On a new checkout or machine, environment installation is entirely offline:

```bash
./dependency_management/manage.sh verify-uv-cache
./dependency_management/manage.sh install-venvs
./dependency_management/manage.sh verify-installed-venvs
```

The first venv uses the frozen Retargeting lock, including the cached pinned
`mujoco-warp` Git commit. The second uses the frozen Isaac Export lock and then
installs `isaacsim[all,extscache]==6.0.1.0` from the release's 167-file local
wheelhouse. All install commands set uv offline mode; they cannot fall back to
PyPI, NVIDIA, or GitHub. Development extras such as `ruff` are included.

Use the stage shell entry points; they select the managed interpreters:

```bash
./retargeting/run_pipeline.sh --help
./isaac_export/run_pipeline.sh --help
```

`--environment retargeting` or `--environment isaac` limits installation or
verification to one venv. An existing target without matching release metadata
is never overwritten; publish a new release or inspect/remove that exact target
explicitly before reinstalling.
