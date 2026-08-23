# Isaac Level A Export

For the tested UR3e + Sharpa + whisk end-to-end reproduction path, start with
[`REPRODUCE_ISAAC.md`](../REPRODUCE_ISAAC.md).

This module is an independent downstream stage. It does not modify or rerun
Reconstruction or Retargeting.

It accepts two package protocols:

- the original Level A package (`manifest.json` + `trajectory.npz`), containing
  a free Sharpa hand root, fingers, and object;
- the final robot-scene package (`deployment_manifest.json` +
  `trajectory_dual_ur3e.npz`), containing UR3e arm joints, Sharpa fingers, and
  the same transformed object trajectory.

The current implementation completes the Isaac-independent data boundary:

- reads a right-hand Sharpa Retargeting run;
- derives the real qpos and 22-joint order from `scene.xml`;
- sorts and flattens `trajectory_mjwp.npz` chunks by `sim_step`;
- converts the nested wrist `Rx * Ry * Rz` chain to normalized `wxyz` quaternions;
- explicitly handles warmup and timestamps;
- writes `trajectory.npz`, `manifest.json`, `quality_report.json`, and an Isaac
  capability `environment_report.json`.

## Run the bundled whisking export

From the repository root:

```bash
cd "$(git rev-parse --show-toplevel)"
./isaac_export/run_pipeline.sh export
```

The default input and output paths are repository-root-relative. If your shell
is currently in `reconstruction/` or `retargeting/`, return to the repository
root before running this command.

The output is written to `isaac_export/outputs/whisking/`. To export another
Retargeting result without changing code:

```bash
cd "$(git rev-parse --show-toplevel)"
./isaac_export/run_pipeline.sh export \
  --run-dir retargeting/outputs/sharpa/right/<task>/<id> \
  --output-dir isaac_export/outputs/<task>
```

Warmup frames are skipped by default. Use `--include-warmup` to retain them;
they are then marked `false` in `valid_mask`. Use `--trajectory
trajectory_kinematic.npz` for an IK comparison export. Its own `frequency`
field defines timestamps and MJWP warmup settings are not applied.

`environment_report.json` is produced before an Isaac application is started.
With some wheel-based Isaac Sim installations, `omni.usd` becomes importable
only after `SimulationApp` initializes. Consequently, the standard export can
print `Isaac environment: unavailable` even when the report already finds both
`isaacsim` and `pxr`. This does not invalidate `trajectory.npz` or its quality
report. Use the `build-usd` command below as the authoritative end-to-end Isaac
capability check.

## Canonical trajectory protocol

`trajectory.npz` contains:

| Array | Shape | Convention |
| --- | --- | --- |
| `timestamps` | `(T,)` | seconds, starts at zero |
| `hand_root_position` | `(T, 3)` | meters, Retargeting/Isaac Z-up world |
| `hand_root_quaternion` | `(T, 4)` | normalized `wxyz` |
| `hand_joint_positions` | `(T, 22)` | radians, order recorded in manifest |
| `object_position` | `(T, 3)` | meters, Retargeting/Isaac Z-up world |
| `object_quaternion` | `(T, 4)` | normalized `wxyz` |
| `valid_mask` | `(T,)` | excludes retained warmup frames |
| `source_frame_indices` | `(T,)` | frame index in flattened source NPZ |

## Build the Isaac USD scene

Isaac Sim uses a separate Python environment. Prepare it together with all
other project environments by running the single offline setup command from
the repository root:

```bash
./setup_all.sh --managed-offline
```

This command restores and verifies the managed Isaac environment internally;
new developers do not need to run `dependency_management/manage.sh`.

For a local extracted Isaac Sim installation, use the local wrapper. It checks
`$ISAAC_SIM_ROOT`, then common locations including
`$HOME/code/rl_sim/isaacsim`, and invokes that installation's `python.sh`:

```bash
./isaac_export/run_pipeline_local.sh export \
  --output-dir isaac_export/outputs/whisking

./isaac_export/run_pipeline_local.sh build-usd \
  --package-dir isaac_export/outputs/whisking \
  --run-dir retargeting/outputs/sharpa/right/whisking/0 \
  --headless

./isaac_export/run_pipeline_local.sh replay \
  --package-dir isaac_export/outputs/whisking \
  --headless --max-frames 30
```

Set `ISAAC_SIM_ROOT=/path/to/isaacsim` when the installation is elsewhere.
On Isaac Sim 5, `build-usd` creates an `isaac5_compat/scene.xml` plus sanitized
copies of binary STL files whose headers look like ASCII STL. The original
Retargeting assets are never modified.

Release maintainers use `setup_remote_isaac.sh` only when refreshing the pinned
Isaac wheelhouse. NVIDIA publishes the Isaac Sim wheels only on its own package
index, so that refresh happens on the networked relay workstation. Normal
developer setup does not access NVIDIA, PyPI, or GitHub.

The first Isaac Sim launch displays NVIDIA's EULA. Review and accept it in the
interactive prompt yourself; the setup script deliberately does not accept it
on your behalf. Then generate the scene on the remote host:

```bash
cd "$(git rev-parse --show-toplevel)"

./isaac_export/run_pipeline.sh build-usd \
  --package-dir isaac_export/outputs/whisking \
  --run-dir retargeting/outputs/sharpa/right/whisking/0 \
  --headless
```

This imports the complete Retargeting MJCF, keeps collision physics while
hiding collision render proxies and MJCF site markers, adds a light, and
writes:

```text
isaac_export/outputs/whisking/scene.usd
isaac_export/outputs/whisking/isaac_scene_manifest.json
```

## Replay

Open an interactive Isaac window (X11/VNC display required):

```bash
./isaac_export/run_pipeline_local.sh replay \
  --package-dir isaac_export/outputs/whisking \
  --realtime
```

GUI replay loops by default so the window does not disappear after the first
five-second trajectory pass. Close the window or press `Ctrl+C` to stop. Add
`--once` when a single interactive pass is desired.

Or run headless and render an MP4:

```bash
./isaac_export/run_pipeline_local.sh replay \
  --package-dir isaac_export/outputs/whisking \
  --headless \
  --record-dir isaac_export/outputs/whisking/render
```

The replayer drives the six scalar wrist joints, 22 Sharpa finger joints, and
the object world pose by name. Values outside the MJCF finger limits are clipped
before they reach PhysX. Recording fits the standard viewport camera to the valid
hand/object workspace, and repeated recordings replace only recorder-owned
frames in the chosen directory. `Ctrl+C` stops either replay mode safely.

## Full UR3e + Sharpa + object scene

First save the final package from `deployment/mujoco-replay` (or use its
`--save-on-solve` option). Then point the same Isaac commands directly at the
Retargeting run directory:

```bash
RUN_DIR=retargeting/outputs/sharpa/right/whisking/0

./isaac_export/run_pipeline.sh build-usd \
  --package-dir "$RUN_DIR" \
  --headless

./isaac_export/run_pipeline.sh replay \
  --package-dir "$RUN_DIR" \
  --realtime
```

Package detection is automatic. In this mode Isaac imports
`scene_dual_ur3e_object.xml`, drives the active six-axis UR3e articulation plus
22 Sharpa joints, and writes the object world pose each frame. Invalid warmup
frames before `start_frame` are skipped. This remains a deterministic kinematic
reference replay rather than a contact-driven object simulation.
