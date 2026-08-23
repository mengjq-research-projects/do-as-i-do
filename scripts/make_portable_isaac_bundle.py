#!/usr/bin/env python3
"""Assemble a self-contained, direct-replay Isaac package."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np


MACHINE_PATH_MARKERS = (b"/home/", b"/data/", b"/tmp/", b"file://")


def copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Missing portable-bundle input: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_obj(source: Path, destination: Path) -> None:
    """Copy an OBJ with exactly one terminal newline for clean Git diffs."""
    if not source.is_file():
        raise FileNotFoundError(f"Missing portable-bundle input: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        source.read_text(encoding="utf-8", errors="strict").rstrip() + "\n",
        encoding="utf-8",
    )


def rewrite_trajectory(source: Path, destination: Path) -> None:
    """Copy all trajectory arrays while making provenance package-relative."""
    with np.load(source, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    if "source_traj" in arrays:
        arrays["source_traj"] = np.asarray(
            "source_assets/trajectory_mjwp.npz"
        )
    np.savez_compressed(destination, **arrays)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_portable(paths: list[Path]) -> None:
    for path in paths:
        contents = path.read_bytes()
        markers = [
            marker.decode("ascii")
            for marker in MACHINE_PATH_MARKERS
            if marker in contents
        ]
        if markers:
            raise RuntimeError(
                f"Machine-specific path marker(s) {markers} remain in {path}"
            )


def assemble(source_directory: Path, output_directory: Path) -> None:
    source_directory = source_directory.expanduser().resolve()
    output_directory = output_directory.expanduser().resolve()
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(
            f"Portable-bundle output must be empty: {output_directory}"
        )
    output_directory.mkdir(parents=True, exist_ok=True)

    deployment_path = source_directory / "deployment_manifest.json"
    scene_mapping_path = source_directory / "isaac_scene_manifest.json"
    deployment = json.loads(deployment_path.read_text(encoding="utf-8"))
    scene_mapping = json.loads(scene_mapping_path.read_text(encoding="utf-8"))

    object_mesh = Path(deployment["object"]["source_mesh"]).expanduser().resolve()
    object_texture = Path(
        deployment["object"]["source_texture"]
    ).expanduser().resolve()
    source_trajectory = Path(
        deployment["source"]["trajectory"]
    ).expanduser().resolve()

    copy_file(source_directory / "scene.usd", output_directory / "scene.usd")
    copy_file(
        object_texture,
        output_directory / "isaac_assets" / "visual_texture.png",
    )
    copy_obj(
        object_mesh,
        output_directory / "source_assets" / "visual.obj",
    )
    copy_file(
        source_trajectory,
        output_directory / "source_assets" / "trajectory_mjwp.npz",
    )
    rewrite_trajectory(
        source_directory / deployment["trajectory"]["file"],
        output_directory / "trajectory_dual_ur3e.npz",
    )

    deployment["object"]["source_mesh"] = "source_assets/visual.obj"
    deployment["object"]["source_texture"] = (
        "isaac_assets/visual_texture.png"
    )
    deployment["source"]["trajectory"] = (
        "source_assets/trajectory_mjwp.npz"
    )
    deployment["scene"] = {
        "usd": "scene.usd",
        "mjcf": None,
        "rebuild_assets_included": False,
    }
    deployment["portable_bundle"] = {
        "mode": "direct_isaac_replay",
        "requires_mjcf_rebuild": False,
    }
    portable_deployment = output_directory / "deployment_manifest.json"
    portable_deployment.write_text(
        json.dumps(deployment, indent=2) + "\n", encoding="utf-8"
    )

    scene_mapping["scene_usd"] = "scene.usd"
    appearance = scene_mapping.get("task_object_appearance", {})
    if appearance.get("texture"):
        appearance["texture"] = "isaac_assets/visual_texture.png"
    portable_mapping = output_directory / "isaac_scene_manifest.json"
    portable_mapping.write_text(
        json.dumps(scene_mapping, indent=2) + "\n", encoding="utf-8"
    )

    readme = output_directory / "README.md"
    readme.write_text(
        """# Portable whisking replay bundle

This directory is a prebuilt direct-replay package for Isaac Sim 5.1. It does
not contain the 126 MiB of intermediate MJCF-conversion meshes, so use the
repository source pipeline when rebuilding `scene.usd` is required.

From the repository root:

```bash
./isaac_export/run_pipeline_local.sh replay \\
  --package-dir isaac_export/prebuilt/whisking --realtime
```

For a finite smoke test, add `--headless --max-frames 30` and remove
`--realtime`.
""",
        encoding="utf-8",
    )

    payload_paths = sorted(
        path
        for path in output_directory.rglob("*")
        if path.is_file()
    )
    assert_portable(payload_paths)
    checksums = {
        str(path.relative_to(output_directory)): {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in payload_paths
    }
    bundle_manifest = {
        "schema_version": "do-as-i-do.portable-isaac-replay.v1",
        "isaac_sim_version": deployment["isaac_sim"]["version"],
        "mode": "direct_isaac_replay",
        "files": checksums,
    }
    (output_directory / "PORTABLE_BUNDLE.json").write_text(
        json.dumps(bundle_manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Created portable Isaac bundle with {len(payload_paths)} payload files: "
        f"{output_directory}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-package", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    assemble(arguments.source_package, arguments.output_dir)
