# Deployment

Run a retargeted demo on the real robot. Two stages:

1. **`mujoco_replay/`** — load a spider `trajectory_mjwp.npz` (from the
   `retargeting/` stage), place + IK-solve it onto the dual-UR3e scene with
   collision-aware [`mink`](https://github.com/kevinzakka/mink) IK, preview/tune
   in a viser GUI, and save `trajectory_dual_ur3e.npz` (arm + finger joint
   trajectories).
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
  --side left --traj /path/to/trajectory_mjwp.npz
```

Opens a viser web GUI at the printed URL. Tune the workspace placement
(x/y/z + yaw/pitch/roll), the start frame, and collision avoidance; hit
**Recompute IK**, then **Save retarget** to write `trajectory_dual_ur3e.npz`
next to the input trajectory.

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
  --side left --traj .../trajectory_mjwp.npz --solve-only
```

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
