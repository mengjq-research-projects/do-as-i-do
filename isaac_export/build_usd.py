#!/usr/bin/env python3
"""Convert the Retargeting MJCF scene to a replayable Isaac Sim USD stage."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from datetime import UTC, datetime
from pathlib import Path

from isaac_export.package import load_replay_package


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Retargeting run containing scene.xml; overrides the recorded path.",
    )
    parser.add_argument("--output", type=Path, help="Defaults to PACKAGE/scene.usd")
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


def resolve_scene_xml(package, run_dir: Path | None) -> Path:
    candidates = []
    if run_dir is not None:
        candidates.append(run_dir.expanduser().resolve() / "scene.xml")
    recorded = Path(package.manifest["source"]["scene"]).expanduser()
    candidates.append(recorded if recorded.is_absolute() else package.directory / recorded)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not find source scene.xml. Pass --run-dir explicitly. Tried: "
        + ", ".join(str(path) for path in candidates)
    )


def nearest_articulation_root(prim, UsdPhysics):
    while prim and prim.IsValid():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            return str(prim.GetPath())
        prim = prim.GetParent()
    return None


def inspect_stage(stage, package, UsdPhysics) -> dict:
    prims = list(stage.Traverse())
    by_name = {prim.GetName(): prim for prim in prims}
    required_joints = package.root_joint_names + package.finger_joint_names
    missing = [name for name in required_joints if name not in by_name]
    if missing:
        raise RuntimeError(
            "MJCF import did not preserve required joint names: " + ", ".join(missing)
        )

    articulation_roots = {
        nearest_articulation_root(by_name[name], UsdPhysics)
        for name in required_joints
    }
    articulation_roots.discard(None)
    if len(articulation_roots) != 1:
        raise RuntimeError(
            "Expected all Sharpa joints under one articulation root, found "
            f"{sorted(articulation_roots)}."
        )

    # Isaac Sim 6 can emit both the rigid-body Xform and a child geometry prim
    # with the same MJCF name.  Replay must target the rigid body, not its mesh.
    object_candidates = [
        prim
        for prim in prims
        if prim.GetName() == "right_object"
        and prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    if len(object_candidates) != 1:
        raise RuntimeError(
            "Expected one imported prim named 'right_object', found "
            f"{len(object_candidates)}."
        )
    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "scene_usd": str(Path(stage.GetRootLayer().realPath).resolve()),
        "hand_articulation_prim": next(iter(articulation_roots)),
        "object_prim": str(object_candidates[0].GetPath()),
        "root_joint_names": list(package.root_joint_names),
        "finger_joint_names": list(package.finger_joint_names),
        "joint_value_unit": "meter for root translations; radian otherwise",
        "quaternion_order": "wxyz",
    }


def main() -> None:
    args = parse_args()
    package = load_replay_package(args.package_dir)
    scene_xml = resolve_scene_xml(package, args.run_dir)
    output = (args.output or package.directory / "scene.usd").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        from isaacsim import SimulationApp
    except ModuleNotFoundError as error:
        raise SystemExit(
            "Isaac Sim is not installed in this Python environment. Run this script "
            "with the dedicated Isaac Sim Python environment."
        ) from error

    simulation_app = SimulationApp(
        {"headless": args.headless, "fast_shutdown": False}
    )
    try:
        from isaacsim.core.utils.extensions import enable_extension
        from isaacsim.core.utils.stage import open_stage
        from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics

        enable_extension("isaacsim.asset.importer.mjcf")
        from isaacsim.asset.importer.mjcf import MJCFImporter, MJCFImporterConfig

        config = MJCFImporterConfig()
        config.mjcf_path = str(scene_xml)
        # Isaac Sim 6 treats ``usd_path`` as an output directory and returns
        # the actual generated stage path from ``import_mjcf()``.  Keep those
        # importer assets beside the replay package, then export the composed
        # stage to the stable scene.usd path used by the replay tools.
        import_directory = package.directory / "mjcf_import"
        import_directory.mkdir(parents=True, exist_ok=True)
        config.usd_path = str(import_directory)
        config.import_scene = True
        config.merge_mesh = False
        config.collision_from_visuals = False
        config.allow_self_collision = True
        config.fix_base = False
        config.run_asset_transformer = False
        importer = MJCFImporter(config)
        imported_output = Path(importer.import_mjcf()).resolve()
        if not imported_output.is_file():
            raise RuntimeError(
                f"MJCF importer did not create its reported output {imported_output}."
            )

        open_stage(str(imported_output))
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        if not stage.GetPrimAtPath("/World/ReplayGround").IsValid():
            ground = UsdGeom.Cube.Define(stage, "/World/ReplayGround")
            ground.CreateSizeAttr(1.0)
            ground.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.025))
            ground.AddScaleOp().Set(Gf.Vec3f(5.0, 5.0, 0.025))
            UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
        if not stage.GetPrimAtPath("/World/ReplayLight").IsValid():
            light = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/ReplayLight"))
            light.CreateIntensityAttr(1000.0)
        if output.is_dir():
            raise RuntimeError(
                f"Cannot write scene file because a directory exists at {output}. "
                "Move that legacy directory aside and rerun."
            )
        if not stage.GetRootLayer().Export(str(output)):
            raise RuntimeError(f"Failed to export composed Isaac stage to {output}.")

        # Reopen the stable exported path so the recorded root-layer path is
        # scene.usd rather than the importer's intermediate scene.usda.
        open_stage(str(output))
        stage = omni.usd.get_context().get_stage()

        scene_manifest = inspect_stage(stage, package, UsdPhysics)
        manifest_path = package.directory / "isaac_scene_manifest.json"
        manifest_path.write_text(
            json.dumps(scene_manifest, indent=2) + "\n", encoding="utf-8"
        )
        package.manifest["isaac_sim"].update(
            {
                "version": importlib.metadata.version("isaacsim"),
                "asset_conversion_status": "pass",
            }
        )
        (package.directory / "manifest.json").write_text(
            json.dumps(package.manifest, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Created Isaac scene: {output}")
        print(f"Created scene mapping: {manifest_path}")
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
