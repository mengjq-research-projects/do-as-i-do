#!/bin/bash
# ════════════════════════════ do-as-i-do reconstruction ════════════════════════════
# End-to-end object reconstruction + pose tracking from a hand-object demo video.
# Stages (each runs in its own conda env; see config/paths.sh):
#   0  extract frames (ffmpeg)
#   1  SAM3 video segmentation — objects (click) + anchor hand (text)        [sam3]
#   2  masks -> 3D meshes ; MoGe pointmap (ref frame) ; HaWoR hands ; gravity (GeoCalib)  [sam3d/hawor]
#   2.5 TAPIR velocity tracking                                              [tapnet]
#   3  object tracking using guided pose prediction ; project mesh ; layout -> camera frame        [sam3d]
#   4  optimize translation/scale (+ optional viser viz)                     [sam3d]
#
# Usage:  ./run_pipeline.sh VIDEO_PATH [FRAME_N] [OBJECT] [ANCHOR_HAND] [OBJECT_POINTS] [POINT_LABELS] [VIEWPOINT] [CAMERA_MOTION]
# Example: ./run_pipeline.sh /data/pickplan_pan/pickplan_pan.mp4 28 pan right "620,410" "1" ego moving
set -eo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config/paths.sh"

# ──────────────────────────── Per-run inputs (args) ────────────────────────────
VIDEO_PATH="${1:?usage: run_pipeline.sh VIDEO_PATH [FRAME_N] [OBJECT] [ANCHOR_HAND]}"
# Resolve to an ABSOLUTE path up front: later stages `cd "$SCRIPTS_DIR"` and into the
# module dirs, so a RELATIVE VIDEO_PATH would stop resolving (run_sam3_video.py would
# then load 0 frames and crash with `IndexError: list index out of range`). This also
# makes the derived VIDEO_DIR / FRAME_PATH / MASKS_DIR absolute.
VIDEO_PATH="$(realpath "$VIDEO_PATH")"
n="${2:-28}"
OBJECT_NAMES=("${3:-pan}")
ANCHOR_HAND="${4:-right}"
OBJECT_POINTS="${5:-}"
POINT_LABELS="${6:-1}"
VIEWPOINT="${7:-auto}"
CAMERA_MOTION="${8:-auto}"

case "$VIEWPOINT" in
    auto|ego|exo) ;;
    *)
        echo "ERROR: VIEWPOINT must be one of: auto, ego, exo (got '$VIEWPOINT')." >&2
        exit 2
        ;;
esac
case "$CAMERA_MOTION" in
    auto|moving|static) ;;
    *)
        echo "ERROR: CAMERA_MOTION must be one of: auto, moving, static (got '$CAMERA_MOTION')." >&2
        exit 2
        ;;
esac

# ──────────────────────────── Derived paths ────────────────────────────
VIDEO_DIR="$(dirname "$VIDEO_PATH")"
VIDEO_BASENAME="$(basename "$VIDEO_PATH")"
VIDEO_NAME="${VIDEO_BASENAME%%.*}"
FRAME_PATH="$VIDEO_DIR/$(printf "%04d.png" "$n")"
POINTMAP_PATH="$VIDEO_DIR/$(printf "%04d_pointmap.npy" "$n")"
INTRINSICS_PATH="$VIDEO_DIR/$(printf "%04d_intrinsics.txt" "$n")"
MASKS_DIR="$VIDEO_DIR/video_segmentation/masks/frame_$(printf "%06d" "$n")_masks"
VIDEO_MASKS_DIR="$VIDEO_DIR/video_segmentation/masks"
HAND_MESHES_PATH="$VIDEO_DIR/$VIDEO_NAME/all_hand_meshes.npz"

if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
elif [[ -n "${DO_AS_I_DO_CONDA_BASE:-}" && -f "$DO_AS_I_DO_CONDA_BASE/etc/profile.d/conda.sh" ]]; then
    CONDA_BASE="$DO_AS_I_DO_CONDA_BASE"
elif [[ -f /home/jiaqimeng/miniforge3/etc/profile.d/conda.sh ]]; then
    CONDA_BASE=/home/jiaqimeng/miniforge3
elif [[ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]]; then
    CONDA_BASE="$HOME/miniforge3"
else
    echo "Conda initialization script was not found." >&2
    echo "Set DO_AS_I_DO_CONDA_BASE to the Miniforge/Conda installation." >&2
    exit 1
fi
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Some Conda packages (notably HaWoR's binutils package) run activation hooks
# containing harmless probe commands that return non-zero.  With this script's
# `set -e`, Bash would abort inside the hook before `conda activate` could
# finish, even though activation succeeds when allowed to complete.  Disable
# errexit only around Conda's activation machinery, then restore it and check
# Conda's final status explicitly.
activate_conda_env() {
    local env_name="$1"
    local activate_status

    set +e
    conda activate "$env_name"
    activate_status=$?
    set -e

    if [[ "$activate_status" -ne 0 ]]; then
        echo "ERROR: failed to activate Conda environment: $env_name" >&2
        return "$activate_status"
    fi
}

# Frame extraction (Steps 0 & 1) uses ffmpeg from the sam3 env — activate it FIRST so
# no system/base ffmpeg is required and the pipeline runs directly on a clip.
activate_conda_env "$ENV_SAM3"

# ──────────────── Step 0: Extract all frames ───────────────────────────
echo "=== Extracting all frames ==="
mkdir -p "$VIDEO_DIR/all_frames"
ffmpeg -y -i "$VIDEO_PATH" -fps_mode passthrough -start_number 0 \
    "$VIDEO_DIR/all_frames/%06d.png"

# ──────────────── Step 1: Save config, extract ref frame, run SAM3 ────
echo "=== Saving config and extracting reference frame ==="
OBJ_ARRAY=$(printf ', "%s"' "${OBJECT_NAMES[@]}")
OBJ_ARRAY="[${OBJ_ARRAY:2}]"
cat > "$VIDEO_DIR/config.json" <<EOF
{
    "schema_version": 2,
    "frame_number": $n,
    "object_names": $OBJ_ARRAY,
    "anchor_hand": "$ANCHOR_HAND",
    "capture": {
        "viewpoint": "$VIEWPOINT",
        "camera_motion": "$CAMERA_MOTION"
    }
}
EOF

echo "Capture metadata: viewpoint=$VIEWPOINT, camera_motion=$CAMERA_MOTION"

ffmpeg -y -i "$VIDEO_PATH" -vf "select=eq(n\,${n})" \
    -fps_mode passthrough -frames:v 1 -update 1 "$FRAME_PATH"

# sam3 env already active (from Step 0) — provides both ffmpeg and the SAM3 model
cd "$SCRIPTS_DIR"
mkdir -p "$VIDEO_MASKS_DIR"
TOTAL_FRAME_COUNT="$(find "$VIDEO_DIR/all_frames" -maxdepth 1 -type f -name '*.png' | wc -l)"

echo "=== Running SAM3 video segmentation (objects, click-based) ==="
for OBJ_NAME in "${OBJECT_NAMES[@]}"; do
    OBJ_ID="${OBJ_NAME// /_}"
    OBJECT_MASK_COUNT="$(find "$VIDEO_MASKS_DIR" -type f -name "$OBJ_ID.png" | wc -l)"
    if [[ "$TOTAL_FRAME_COUNT" -gt 0 && "$OBJECT_MASK_COUNT" -eq "$TOTAL_FRAME_COUNT" ]]; then
        echo "Skipping SAM3 for $OBJ_ID: found $OBJECT_MASK_COUNT/$TOTAL_FRAME_COUNT masks."
        continue
    fi
    if [[ -n "$OBJECT_POINTS" ]]; then
        python run_sam3_video.py \
            --video "$VIDEO_PATH" \
            --points "$OBJECT_POINTS" \
            --point_labels "$POINT_LABELS" \
            --obj_id "$OBJ_ID" \
            --frame_idx "$n"
    else
        INTERACTIVE_DISPLAY="${DISPLAY:-${SAM3_DISPLAY:-}}"
        if [[ -z "$INTERACTIVE_DISPLAY" ]] || \
           { command -v xdpyinfo >/dev/null 2>&1 && \
             ! DISPLAY="$INTERACTIVE_DISPLAY" xdpyinfo >/dev/null 2>&1; }; then
            echo "ERROR: SAM3 interactive clicking needs a working X display" \
                 "(received '${INTERACTIVE_DISPLAY:-<unset>}')." >&2
            echo "This host is headless. Pass object point coordinates instead:" >&2
            echo "  ./run_pipeline.sh '$VIDEO_PATH' '$n' '$OBJ_NAME' '$ANCHOR_HAND' 'X,Y' '1'" >&2
            echo "Multiple points use 'x1,y1;x2,y2' with labels such as '1;0'." >&2
            echo "Reference frame for choosing pixels: $FRAME_PATH" >&2
            exit 1
        fi
        DISPLAY="$INTERACTIVE_DISPLAY" python run_sam3_video.py \
            --video "$VIDEO_PATH" \
            --click \
            --obj_id "$OBJ_ID" \
            --frame_idx "$n"
    fi
done

echo "=== Running SAM3 video segmentation (hands, text-based) ==="
HAND_NAME="$ANCHOR_HAND hand"
HAND_ID="${ANCHOR_HAND}_hand_0"
HAND_MASK_COUNT="$(find "$VIDEO_MASKS_DIR" -type f -name "$HAND_ID.png" | wc -l)"
if [[ "$TOTAL_FRAME_COUNT" -gt 0 && "$HAND_MASK_COUNT" -eq "$TOTAL_FRAME_COUNT" ]]; then
    echo "Skipping SAM3 for $HAND_ID: found $HAND_MASK_COUNT/$TOTAL_FRAME_COUNT masks."
else
    python run_sam3_video.py \
        --video "$VIDEO_PATH" \
        --text "$HAND_NAME" \
        --obj_id "$HAND_ID" \
        --frame_idx "$n"
fi

# ──────────────── Step 2: 3D reconstruction, pointmaps, HaWoR ────────
echo "=== Running batch masks to meshes ==="
activate_conda_env "$ENV_SAM3D"

REFERENCE_MESH_IDS=()
for OBJ_NAME in "${OBJECT_NAMES[@]}"; do
    REFERENCE_MESH_IDS+=("${OBJ_NAME// /_}")
done
REFERENCE_MESH_IDS+=("$HAND_ID")

REFERENCE_MESHES_COMPLETE=true
for MESH_ID in "${REFERENCE_MESH_IDS[@]}"; do
    MESH_ROOT="$MASKS_DIR/$MESH_ID"
    if [[ ! -s "$MESH_ROOT/$MESH_ID.obj" || \
          ! -s "$MESH_ROOT/material.mtl" || \
          ! -s "$MESH_ROOT/material_0.png" ]]; then
        REFERENCE_MESHES_COMPLETE=false
        break
    fi
done

if [[ "$REFERENCE_MESHES_COMPLETE" == true ]]; then
    echo "Skipping SAM3D mesh generation: all reference meshes are complete."
else
    cd "$SAM3D_DIR"                              # overlay: generate_mesh_sam3d.py
    python generate_mesh_sam3d.py \
        --config "$SAM3D_CONFIG" \
        --image_path "$FRAME_PATH" \
        --masks_dir "$MASKS_DIR"
fi

cd "$SCRIPTS_DIR"
echo "=== Computing pointmap for reference frame ==="
if python - "$POINTMAP_PATH" "$INTRINSICS_PATH" <<'PY'
import pathlib
import sys

import numpy as np

pointmap_path = pathlib.Path(sys.argv[1])
intrinsics_path = pathlib.Path(sys.argv[2])
try:
    pointmap = np.load(pointmap_path, allow_pickle=False)
    intrinsics = [float(value) for value in intrinsics_path.read_text().split()]
    complete = pointmap.size > 0 and len(intrinsics) == 4
except (OSError, ValueError):
    complete = False
raise SystemExit(0 if complete else 1)
PY
then
    echo "Skipping reference pointmap: existing pointmap and intrinsics are valid."
else
    python get_pointmap_dir.py --image "$FRAME_PATH" --output "$POINTMAP_PATH"
fi

echo "=== Running HaWoR ==="
if python - "$HAND_MESHES_PATH" "$TOTAL_FRAME_COUNT" <<'PY'
import sys
import numpy as np

path, expected_frames = sys.argv[1], int(sys.argv[2])
try:
    data = np.load(path, allow_pickle=False)
    temporal_keys = (
        "left_vertices", "left_joints", "right_vertices", "right_joints",
        "left_trans", "right_trans", "left_valid", "right_valid",
    )
    valid = all(key in data and data[key].shape[0] == expected_frames for key in temporal_keys)
except (OSError, ValueError):
    valid = False
raise SystemExit(0 if valid else 1)
PY
then
    echo "Skipping HaWoR: found complete $TOTAL_FRAME_COUNT-frame result at $HAND_MESHES_PATH"
else
    if [[ ! -s "$MANO_RIGHT" || ! -s "$MANO_LEFT" ]]; then
        echo "ERROR: HaWoR needs licensed MANO files when no complete cached result exists:" >&2
        echo "  $MANO_RIGHT" >&2
        echo "  $MANO_LEFT" >&2
        exit 1
    fi
    activate_conda_env "$ENV_HAWOR"
    cd "$HAWOR_DIR"                              # patched: demo.py
    IMG_FOCAL=$(head -n 1 "$INTRINSICS_PATH")
    if [[ "$CAMERA_MOTION" == "moving" ]]; then
        echo "NOTE: camera_motion=moving is preserved as metadata, but HaWoR currently" \
             "uses its validated static-camera compatibility path. Dynamic-camera" \
             "rotation/gravity compensation remains a separate pipeline stage."
    fi
    python demo.py \
        --video_path "$VIDEO_PATH" \
        --vis_mode cam \
        --img_focal "$IMG_FOCAL" \
        --checkpoint "$HAWOR_CKPT" \
        --infiller_weight "$HAWOR_INFILLER_CKPT" \
        --static_camera
fi

echo "=== Computing pointmaps for all frames ==="
cd "$SCRIPTS_DIR"
activate_conda_env "$ENV_SAM3D"
python get_pointmap_dir.py --image_dir "$VIDEO_DIR/all_frames"

echo "=== Estimating gravity (GeoCalib) ==="
python predict_video_gravity.py "$VIDEO_DIR/all_frames" --output_path "$VIDEO_DIR/gravity.json"

# ──────────────── Step 2.5: TAPIR velocity tracking ───────────────────
activate_conda_env "$ENV_TAPNET"
echo "=== Running TAPIR velocity tracking ==="
cd "$SCRIPTS_DIR"
for OBJ_NAME in "${OBJECT_NAMES[@]}"; do
    OBJECT_ID="${OBJ_NAME// /_}"
    PYTHONPATH="$TAPNET_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    python tapir_velocity_tracking.py \
        --video "$VIDEO_PATH" \
        --mask-dir "$VIDEO_MASKS_DIR" \
        --object "$OBJECT_ID" \
        --checkpoint "$TAPNET_CKPT"
done

# ──────────────── Step 3: Object Tracking using guided pose prediction & projection ─────────
activate_conda_env "$ENV_SAM3D"
echo "=== Running guided pose prediction for object tracking ==="
cd "$FASTSAM3D_DIR"                              # overlay: track_object.py
for OBJ_NAME in "${OBJECT_NAMES[@]}"; do
    OBJECT_ID="${OBJ_NAME// /_}"
    python track_object.py \
        --config "$FASTSAM3D_CONFIG" \
        --vid_dir "$VIDEO_DIR" \
        --masks_root "$VIDEO_MASKS_DIR" \
        --object_name "$OBJECT_ID" \
        --init_frame "$n" \
        --output_dir "$VIDEO_DIR/obj_tracking_out/$OBJECT_ID" \
        --guidance_strength 1 \
        --save_layout \
        --fix_scale_to_init_frame \
        --pose_guidance_strength 0.5 \
        --num_pose_samples 25 \
        --scoring_metric render_iou \
        --pose_selection cluster \
        --cluster_dist_thresh 0.3 \
        --cluster_min_size 3 \
        --cluster_w_rot 1.5 \
        --chain_poses \
        --post_optimize \
        --no-enable_shape_icp \
        --chain_on_diffusion \
        --enable_ss_cache \
        --torch_compile \
        --euler_steps 25 \
        --rotvel_json "$VIDEO_DIR/perframe_tracking_$OBJECT_ID/motion_stats.json"

    cd "$SCRIPTS_DIR"

    echo "=== Projecting mesh for $OBJ_NAME ==="
    python run_project_mesh_combined.py \
        --video "$VIDEO_PATH" \
        --mesh "$MASKS_DIR/$OBJECT_ID/${OBJECT_ID}.obj" \
        --json "$VIDEO_DIR/obj_tracking_out/$OBJECT_ID/combined_visualization/layout.json" \
        --output-base "$VIDEO_DIR/obj_tracking_out/$OBJECT_ID/combined_visualization/projected"

    echo "=== Converting layout to camera frame for $OBJ_NAME ==="
    python convert_layout_to_camera_frame.py \
        --input "$VIDEO_DIR/obj_tracking_out/$OBJECT_ID/combined_visualization/layout.json" \
        --output "$VIDEO_DIR/obj_tracking_out/$OBJECT_ID/combined_visualization/layout_camera_frame.json"

    cd "$FASTSAM3D_DIR"
done

# ──────────────── Step 4: Optimize translation/scale & visualize ──────
cd "$SCRIPTS_DIR"
activate_conda_env "$ENV_SAM3D"

for OBJ_NAME in "${OBJECT_NAMES[@]}"; do
    OBJECT_ID="${OBJ_NAME// /_}"
    LAYOUT_JSON_CF="$VIDEO_DIR/obj_tracking_out/$OBJECT_ID/combined_visualization/layout_camera_frame.json"
    LAYOUT_JSON_OPT="$VIDEO_DIR/obj_tracking_out/$OBJECT_ID/combined_visualization/layout_camera_frame_optimized.json"

    echo "=== Optimizing translation/scale for $OBJ_NAME ==="
    python optimize_translation_scale.py \
        --video-dir "$VIDEO_DIR" \
        --layout-json "$LAYOUT_JSON_CF" \
        --anchor-hand "$ANCHOR_HAND" \
        --ref-frame "$n"

    # Optional trajectory smoothing to minimize depth inconsistencies
    # Writes <..._optimized>_smooth.json; point visualize_3d.py at it below to use.
    # LAYOUT_JSON_SMOOTH="${LAYOUT_JSON_OPT%.json}_smooth.json"
    # python smooth_trajectory.py \
    #     --input "$LAYOUT_JSON_OPT" \
    #     --output "$LAYOUT_JSON_SMOOTH"

    # Optional interactive 3D visualization (needs viser in the sam3d env):
    # MESH_SCALE="$(python3 -c "import json; d=json.load(open('$LAYOUT_JSON_OPT')); print(d['translation_scale_optimization']['mesh_scale'])")"
    # python visualize_3d.py \
    #     --frames-dir "$VIDEO_DIR/all_frames" \
    #     --layout-json "$LAYOUT_JSON_OPT" \
    #     --mesh "$VIDEO_DIR/tracking_output_every_frame/$OBJECT_ID/frame_000000/$OBJECT_ID/${OBJECT_ID}.obj" \
    #     --scale "$MESH_SCALE" \
    #     --translation-scale 1.0 \
    #     --hand-meshes "$HAND_MESHES_PATH" \
    #     --port 8080
done

echo "=== Pipeline complete ==="
