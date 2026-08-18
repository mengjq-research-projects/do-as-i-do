# Do as I Do · Reconstruction

Hand and object **reconstruction + 6-DoF pose tracking** from a single hand-object demo video. Given a
video, a reference frame, an object name, and the anchor hand, the pipeline segments the object
and hand, reconstructs a 3D mesh for the object, estimates per-frame object pose, and reconstructs the hand (HaWoR).

## Layout

```
reconstruction/
├── run_pipeline.sh          # the driver 
├── config/paths.sh          # ← the ONLY file you edit to relocate things
├── scripts/                 # 7 standalone first-party stage scripts
├── modules/             # submodules with our changes
├── weights/                 # model weights 
├── env/                     # conda env docs (see env/README.md)
└── setup/                   # 00 submodules · 01 envs · 02 weights
```

## Requirements
- An NVIDIA GPU with ≥ 32 GB VRAM.
- HuggingFace auth with access to the repos `facebook/sam-3d-objects` and `facebook/sam3`; Plus a MANO download
  (https://mano.is.tue.mpg.de) for HaWoR.
- SAM 3 segmentation is implemented with a click-based GUI needing an X display
  (`config/paths.sh` sets `SAM3_DISPLAY=:1`; on a headless host, use forwarding or try text based prompting).

## Setup (one time)

To prepare the complete repository from its root, use the one-shot setup entry:

```bash
cd do-as-i-do
./setup_all.sh --managed-offline
```

This restores and verifies all four Reconstruction environments, model weights,
offline sources, Retargeting, and Isaac. It also prepares the runtime links, so
no additional setup command is required. MANO remains a separately licensed
manual asset. See [`../ENVIRONMENT.md`](../ENVIRONMENT.md).


## Run

```bash
./run_pipeline.sh VIDEO_PATH [FRAME_N] [OBJECT] [ANCHOR_HAND]
# e.g.
./run_pipeline.sh whisking/whisking.mp4 125 whisk right
```

### Details on Pipeline Stages
| # | stage | script | env |
|---|---|---|---|
| 0 | extract frames | ffmpeg | sam3 |
| 1 | SAM3 segmentation (object click + hand text) | `scripts/run_sam3_video.py` | sam3 |
| 2 | masks → 3D mesh | `modules/sam-3d-objects/generate_mesh_sam3d.py` | sam3d |
| 2 | MoGe pointmaps (reference frame + all frames) | `scripts/get_pointmap_dir.py` | sam3d |
| 2 | hand reconstruction | `modules/HaWoR/demo.py` | hawor |
| 2 | gravity estimation (GeoCalib) | `scripts/predict_video_gravity.py` | sam3d |
| 2.5 | velocity tracking | `scripts/tapir_velocity_tracking.py` | tapnet |
| 3 | guided pose prediction | `modules/Fast-SAM3D/track_object.py` | sam3d |
| 3 | project mesh / to camera frame | `scripts/run_project_mesh_combined.py`, `scripts/convert_layout_to_camera_frame.py` | sam3d |
| 4 | optimize translation/scale | `scripts/optimize_translation_scale.py` | sam3d |
| 4 | (optional) 3D viser viz | `scripts/visualize_3d.py` | sam3d |


### Outputs (under the video's directory)
`video_segmentation/masks/…`, per-object `.obj` meshes, `*_pointmap.npy` / `*_intrinsics.txt`,
HaWoR `all_hand_meshes.npz`, `gravity.json` (camera-frame up direction from GeoCalib),
and `obj_tracking_out/<object>/combined_visualization/`
with `layout.json → layout_camera_frame.json → layout_camera_frame_optimized.json` and projected
frames. This directory is the input to the [`retargeting/`](../retargeting/README.md) pipeline.


### Visualize the processed output yourself

To explore the processed `whisk` demo (`whisking/`) in an interactive 3D (viser) viewer
without re-running the pipeline, first extract the video frames. Use the **same** ffmpeg
invocation as the pipeline's Step 0 so the frame numbering matches the tracking output:

```bash
# from the reconstruction/ directory — same flags as run_pipeline.sh Step 0
VIDEO_DIR=whisking
mkdir -p "$VIDEO_DIR/all_frames"
ffmpeg -i "$VIDEO_DIR/whisking.mp4" -vsync 0 -start_number 0 "$VIDEO_DIR/all_frames/%06d.png"
```

Then launch the viewer:

```bash
VIDEO_DIR=whisking
OBJECT_ID=whisk
LAYOUT_JSON_OPT="$VIDEO_DIR/obj_tracking_out/$OBJECT_ID/combined_visualization/layout_camera_frame_optimized.json"
./run_visualize.sh \
    --frames-dir "$VIDEO_DIR/all_frames" \
    --layout-json "$LAYOUT_JSON_OPT" \
    --mesh "$VIDEO_DIR/video_segmentation/masks/frame_000125_masks/$OBJECT_ID/$OBJECT_ID.obj" \
    --hand-meshes "$VIDEO_DIR/whisking/all_hand_meshes.npz" \
    --scale 0.1808 \
    --translation-scale 1.0 \
    --hands both \
    --port 8080
```


## Credits & licenses
The `modules/` contain external references with our changes to the sources, and each of them retain their upstream `LICENSE`. We gratefully
acknowledge the original authors.

| fork | upstream @ pinned commit | fork commit | license |
|---|---|---|---|
| `malik-group/sam-3d-objects` | facebookresearch/sam-3d-objects @ `81a8237` | `875b010` | SAM License (Meta) |
| `malik-group/Fast-SAM3D`     | wlfeng0509/Fast-SAM3D @ `c0f99e8`           | `823d478` | MIT (+ embedded SAM-3D under SAM License) |
| `malik-group/HaWoR`          | ThunderVVV/HaWoR @ `de90272`                | `2c3fa0c` | CC BY-NC-ND 4.0 |
| `malik-group/tapnet`         | google-deepmind/tapnet @ `96d3f84`          | `f2f8888` | Apache-2.0 |
| `malik-group/sam3`           | facebookresearch/sam3 @ `757bbb0`           | `b8e18f5` | SAM License (Meta) |
