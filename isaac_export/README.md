# Isaac Level A Export

This module is an independent downstream stage. It does not modify or rerun
Reconstruction or Retargeting.

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

This imports the complete Retargeting MJCF, keeps the Sharpa visual and
collision meshes, adds a ground collider and light, and writes:

```text
isaac_export/outputs/whisking/scene.usd
isaac_export/outputs/whisking/isaac_scene_manifest.json
```

## Replay

Open an interactive Isaac window (X11/VNC display required):

```bash
./isaac_export/run_pipeline.sh replay \
  --package-dir isaac_export/outputs/whisking \
  --realtime
```

Or run headless and render an MP4:

```bash
./isaac_export/run_pipeline.sh replay \
  --package-dir isaac_export/outputs/whisking \
  --headless \
  --record-dir isaac_export/outputs/whisking/render
```

The replayer drives the six scalar wrist joints, 22 Sharpa finger joints, and
the object world pose by name. Values outside the MJCF finger limits are clipped
before they reach PhysX. `Ctrl+C` stops either replay mode safely.
