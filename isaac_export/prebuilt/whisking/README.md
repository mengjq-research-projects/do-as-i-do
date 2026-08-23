# Portable whisking replay bundle

This directory is a prebuilt direct-replay package for Isaac Sim 5.1. It does
not contain the 126 MiB of intermediate MJCF-conversion meshes, so use the
repository source pipeline when rebuilding `scene.usd` is required.

From the repository root:

```bash
./isaac_export/run_pipeline_local.sh replay \
  --package-dir isaac_export/prebuilt/whisking --realtime
```

For a finite smoke test, add `--headless --max-frames 30` and remove
`--realtime`.
