# Reproduce the UR3e + Sharpa Isaac replay

This branch contains the MuJoCo scene assembly and Isaac replay fixes for the
`whisking` trajectory. Generated USD, videos, caches, and locally regenerated
meshes are deliberately excluded from the commit.

## 1. Check out and prepare the repository

```bash
git fetch origin
git switch isaac-replay-whisking
./setup_all.sh --managed-offline
```

Isaac Sim is not installed by Git. For an extracted Isaac Sim installation,
set `ISAAC_SIM_ROOT=/path/to/isaacsim`; the local wrapper also checks common
locations below `$HOME` and `/opt/isaacsim`.

## 2. Verify the source checkout

```bash
./scripts/verify_isaac_replay.sh
```

The check confirms that the tracked UR3e, Sharpa, tool, and trajectory inputs
exist, rejects developer-specific absolute paths, and runs the MuJoCo-package
and Isaac-export unit tests without starting Isaac Sim.

## Direct replay from the prebuilt portable package

The branch also contains a prebuilt package that does not require MuJoCo or an
MJCF-to-USD conversion step. With Isaac Sim 5.1 available, run:

```bash
./isaac_export/run_pipeline_local.sh replay \
  --package-dir isaac_export/prebuilt/whisking --realtime
```

The package contains a composed binary USD, trajectory, manifests, tool mesh,
and texture. It intentionally omits the 126 MiB of intermediate converted STL
meshes and is therefore for direct replay, not USD rebuilding.

## 3. Generate the final MuJoCo package locally

```bash
./scripts/package_isaac_replay.sh
```

By default this reads the tracked right-hand `whisking/0` trajectory and writes
generated files outside the Git checkout under:

```text
$XDG_DATA_HOME/do-as-i-do/isaac_replay/whisking
```

If `XDG_DATA_HOME` is unset, `$HOME/.local/share` is used. Override the location
with `DO_AS_I_DO_REPLAY_PACKAGE_DIR`. An alternative Retargeting run directory
may be supplied as the first positional argument; remaining arguments are
passed to `replay_retarget.py` for workspace/IK tuning.

## 4. Build and replay in Isaac Sim

```bash
PACKAGE_DIR="${DO_AS_I_DO_REPLAY_PACKAGE_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/do-as-i-do/isaac_replay/whisking}"

./isaac_export/run_pipeline_local.sh build-usd \
  --package-dir "$PACKAGE_DIR" --headless

./isaac_export/run_pipeline_local.sh replay \
  --package-dir "$PACKAGE_DIR" --realtime
```

For a short noninteractive smoke test:

```bash
./isaac_export/run_pipeline_local.sh replay \
  --package-dir "$PACKAGE_DIR" --headless --max-frames 30
```

Expected behavior: the active UR3e drives the recorded trajectory, its Sharpa
hand remains welded to the wrist flange, the whisk moves with the recorded
object pose, the unused arm stays in a folded pose, and robot/table/tool
collision proxies remain stable after simulation reset.

## What belongs in Git

Commit source code, tests, shell entry points, lockfiles, and this guide. Do not
commit `scene.usd`, rendered frames/videos, `*_tmp/` converted meshes, or local
changes below `retargeting/outputs/`; those are generated or machine-specific
artifacts and make reviews unnecessarily large.
