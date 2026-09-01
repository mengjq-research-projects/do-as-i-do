# Deployment

Present a retargeted demo in simulation, then optionally run it on real hardware.
The simulation entry separates the input viewpoint from the output embodiment:

- `viewpoint=ego|exo|auto` describes the source video.
- `render-mode=hand-only|full-arm|auto` selects what the viewer shows.
- `camera-mode=auto|scene|ego|top-down` selects how that output is presented.
- `auto` maps ego input to a first-person free-root Sharpa hand + object and exo
  input to the complete UR3e + Sharpa + object scene. Legacy/unknown input keeps
  the historical full-arm behavior.

The stages are:

1. **`mujoco_replay/`** — load a `trajectory_mjwp.npz` from Retargeting. In
   hand-only mode it directly presents the free-root Sharpa hand and object. In
   full-arm mode it places + IK-solves the motion onto dual UR3e with
   collision-aware [`mink`](https://github.com/kevinzakka/mink) IK and can save
   a complete arm + hand + object scene package.
2. **`robot_replay/`** — stream `trajectory_dual_ur3e.npz` to a real UR3e arm +
   Sharpa Wave hand at 50 Hz.

## Setup

The simulation-only `mujoco-replay` action reuses the managed Retargeting
environment restored by the standard offline setup:

```bash
./setup_all.sh --managed-offline
```

No separate Deployment environment is required for that action. Real-hardware
`robot-replay` and `home` additionally require the optional Deployment Conda
environment and proprietary Sharpa Wave SDK:

```bash
conda env create -f env/deployment.yml   # see env/README.md
conda activate deployment
```

The Sharpa Wave hand SDK is proprietary and not shipped — drop it into
`robot_replay/Sharpa/`, then copy `robot_replay/config.example.yaml` to
`robot_replay/config.yaml` and fill in your robot IPs / hand serials.

## 1. mujoco_replay (sim → trajectory)

```bash
./deployment/run_pipeline.sh mujoco-replay \
  --render-mode auto --side left --traj /path/to/trajectory_mjwp.npz
```

The automatic mode reads `capture_metadata.json` beside the trajectory. An
explicit CLI value takes priority:

```bash
# First-person dexterous-hand presentation; no UR3e IK is run.
./deployment/run_pipeline.sh mujoco-replay \
  --viewpoint ego --render-mode hand-only \
  --traj /path/to/trajectory_mjwp.npz --port 8081

# Complete robot presentation, including UR3e IK.
./deployment/run_pipeline.sh mujoco-replay \
  --viewpoint exo --render-mode full-arm --side right \
  --traj /path/to/trajectory_mjwp.npz

# Canonical overhead validation, independent of unknown source-camera parameters.
./deployment/run_pipeline.sh mujoco-replay \
  --render-mode full-arm --camera-mode top-down --side right \
  --traj /path/to/trajectory_mjwp.npz
```

Both modes open a viser web GUI at the printed URL. Hand-only mode reuses
Retargeting's `scene.xml` and does not alter or regenerate the trajectory. Its
ego camera uses the persisted reconstruction camera origin for new runs and a
compatible approximation for old runs. A moving-camera input is currently
shown through one stabilized camera because Reconstruction does not yet produce
per-frame camera extrinsics.

`--camera-mode top-down` fixes the optical axis to world `-Z` and keeps world
`+Y` at the top of the image. The viewer derives its center and height from the
full tabletop plus the current wrist/object path, so the preset is repeatable
without hard-coding a task-specific camera pose or requiring source-camera
calibration. Recomputing the workspace also refreshes this framing.

In full-arm mode, tune the workspace placement
(x/y/z + yaw/pitch/roll), the start frame, and collision avoidance; hit
**Recompute IK**, then **Save retarget**. Three files are written next to the
input trajectory:

```text
trajectory_dual_ur3e.npz
deployment_manifest.json
scene_dual_ur3e_object.xml
```

The NPZ retains the original `arm_qpos`, `finger_qpos`, and `dt` fields used by
real-hardware replay, and adds the transformed object pose, timestamps, source
indices, valid-frame mask, selected IK seed, and capture metadata. The object is
kinematically authored and visual only; it does not alter the collision-aware
arm IK solve.

For a new task with no saved package and no explicit `--workspace-*` flag, the
command centers the complete wrist path in front of the selected arm, evaluates
several workspace/IK-seed candidates on sampled frames, and solves the full
trajectory once using the best candidate. The `ego/exo` metadata is preserved
but is not used as a robot-coordinate shortcut. An existing
`trajectory_dual_ur3e.npz` or any explicit workspace flag takes priority. Use
`--no-auto-place` to select the legacy fixed preset.

When tuning the workspace placement, adjust x/y/z and yaw. Only tune pitch and
roll if the gravity alignment (GeoCalib) from reconstruction was significantly
incorrect — otherwise leave them at zero.

### Try it on the whisk demo

Replay the shipped `whisk` retarget (from the [`retargeting/`](../retargeting/README.md)
demo) without re-running anything. The right-arm workspace preset already defaults
to this clip's saved placement, so it comes up positioned in front of the arm:

```bash
# From the repository root — right arm + Sharpa hand, 0.25x playback
./deployment/run_pipeline.sh mujoco-replay \
    --render-mode full-arm \
    --side right \
    --traj retargeting/outputs/sharpa/right/whisking/0/trajectory_mjwp.npz \
    --speed 0.25
```

`--speed 0.25` starts the viser playback at quarter speed (change it live with the
Speed buttons). Open the printed URL, then tweak the workspace / collision controls
and **Save retarget** as usual.

Headless solve (no GUI) to check IK residuals / collision clearance:

```bash
./deployment/run_pipeline.sh mujoco-replay \
  --render-mode full-arm --side left --traj .../trajectory_mjwp.npz --solve-only
```

Add `--save-on-solve` to write the final package after a headless solve. Validate
an existing package without opening MuJoCo or Isaac:

```bash
./deployment/run_pipeline.sh validate-package \
  retargeting/outputs/sharpa/right/whisking/0
```

See the root [`README.md`](../README.md#分阶段入口) for the general staged entry
points, and [`EGO_DEMO_WALKTHROUGH.md`](../EGO_DEMO_WALKTHROUGH.md) for the
fully audited Ego Mug command history.

## 2. robot_replay (trajectory → hardware)

```bash
cp deployment/robot_replay/config.example.yaml \
  deployment/robot_replay/config.yaml     # then edit arm_ip / hand_sn

./deployment/run_pipeline.sh robot-replay --side left trajectory_dual_ur3e.npz
./deployment/run_pipeline.sh robot-replay --side left trajectory_dual_ur3e.npz --both
./deployment/run_pipeline.sh robot-replay --side left trajectory_dual_ur3e.npz --speed 0.5
./deployment/run_pipeline.sh robot-replay --side left trajectory_dual_ur3e.npz --dry-run
```

The script connects, homes, moves to the trajectory start, then waits for Enter
before streaming. Home a robot independently with:

```bash
./deployment/run_pipeline.sh home --side left
./deployment/run_pipeline.sh home --side right --arm-only
```

## Safety

The `robot-replay` and `home` actions command real hardware. Verify the home
pose, keep an e-stop in reach, and start with `--dry-run` then a low `--speed`.
