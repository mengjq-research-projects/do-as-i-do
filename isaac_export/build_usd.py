#!/usr/bin/env python3
"""Convert the Retargeting MJCF scene to a replayable Isaac Sim USD stage."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

from isaac_export.package import load_replay_package
from isaac_export.runtime import isaac_sim_version


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Retargeting run containing scene.xml; overrides the recorded path.",
    )
    parser.add_argument(
        "--scene",
        type=Path,
        help="Explicit MJCF scene to import; useful for an Isaac-safe sanitized XML.",
    )
    parser.add_argument("--output", type=Path, help="Defaults to PACKAGE/scene.usd")
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


def resolve_scene_xml(
    package, run_dir: Path | None, scene_override: Path | None = None
) -> Path:
    candidates = []

    def add_recorded_candidate(candidate: Path) -> None:
        """Prefer the preconverted Isaac-safe scene when it accompanies MJCF."""
        if package.is_robot_scene and not candidate.stem.endswith("_isaac"):
            candidates.append(candidate.with_name(candidate.stem + "_isaac.xml"))
        candidates.append(candidate)

    if scene_override is not None:
        candidates.append(scene_override.expanduser().resolve())
    if run_dir is not None:
        explicit_directory = run_dir.expanduser().resolve()
        add_recorded_candidate(
            explicit_directory
            / (
                package.manifest["scene"]["mjcf"]
                if package.is_robot_scene
                else "scene.xml"
            )
        )
    recorded_value = (
        package.manifest["scene"]["mjcf"]
        if package.is_robot_scene
        else package.manifest["source"]["scene"]
    )
    recorded = Path(recorded_value).expanduser()
    add_recorded_candidate(
        recorded if recorded.is_absolute() else package.directory / recorded
    )
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


def articulation_roots_for_joint(prim, stage, UsdPhysics) -> set[str]:
    """Resolve articulation roots for both hierarchical and flat joint layouts."""
    roots = set()
    hierarchical_root = nearest_articulation_root(prim, UsdPhysics)
    if hierarchical_root is not None:
        roots.add(hierarchical_root)

    # Isaac Sim 5's MJCF importer authors joints below /World/joints, outside
    # the rigid-body hierarchy. Follow physics:body0/body1 to find the root.
    joint = UsdPhysics.Joint(prim)
    for relationship in (joint.GetBody0Rel(), joint.GetBody1Rel()):
        for target in relationship.GetTargets():
            body_root = nearest_articulation_root(
                stage.GetPrimAtPath(target), UsdPhysics
            )
            if body_root is not None:
                roots.add(body_root)
    return roots


def inspect_stage(stage, package, UsdPhysics) -> dict:
    prims = list(stage.Traverse())
    by_name = {prim.GetName(): prim for prim in prims}
    required_joints = package.driven_joint_names
    missing = [name for name in required_joints if name not in by_name]
    if missing:
        raise RuntimeError(
            "MJCF import did not preserve required joint names: " + ", ".join(missing)
        )

    articulation_roots = set().union(
        *(
            articulation_roots_for_joint(by_name[name], stage, UsdPhysics)
            for name in required_joints
        )
    )
    if len(articulation_roots) != 1:
        global_roots = [
            str(prim.GetPath())
            for prim in prims
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
        ]
        joint_paths = {
            name: str(by_name[name].GetPath()) for name in required_joints
        }
        raise RuntimeError(
            "Expected all driven robot joints under one articulation root, found "
            f"{sorted(articulation_roots)}; stage roots={global_roots}; "
            f"driven joint paths={joint_paths}."
        )

    # Isaac Sim 6 can emit both the rigid-body Xform and a child geometry prim
    # with the same MJCF name. Replay must target the rigid body, not its mesh.
    object_name = package.manifest["object"].get("body_name", "right_object")
    object_candidates = [
        prim
        for prim in prims
        if prim.GetName() == object_name
        and prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    if len(object_candidates) != 1:
        raise RuntimeError(
            f"Expected one imported prim named {object_name!r}, found "
            f"{len(object_candidates)}."
        )
    root_layer_path = Path(stage.GetRootLayer().realPath).resolve()
    try:
        scene_usd = str(root_layer_path.relative_to(package.directory))
    except ValueError:
        scene_usd = str(root_layer_path)
    mapping = {
        "schema_version": "1.0",
        "generated_at": datetime.now(UTC).isoformat(),
        # Keep packages movable when scene.usd lives below the package root.
        "scene_usd": scene_usd,
        "robot_articulation_prim": next(iter(articulation_roots)),
        # Backward-compatible alias used by existing Level A packages.
        "hand_articulation_prim": next(iter(articulation_roots)),
        "object_prim": str(object_candidates[0].GetPath()),
        "package_kind": "robot_scene" if package.is_robot_scene else "level_a",
        "arm_joint_names": list(package.arm_joint_names),
        "root_joint_names": list(package.root_joint_names),
        "finger_joint_names": list(package.finger_joint_names),
        "joint_value_unit": "meter for root translations; radian otherwise",
        "quaternion_order": "wxyz",
    }
    if package.is_robot_scene:
        articulation_prims_by_side = {}
        for side in ("left", "right"):
            path = f"/World/{side}_base/{side}_base"
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid() or not prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                raise RuntimeError(
                    f"Dual-arm scene is missing the {side} articulation root: {path}"
                )
            articulation_prims_by_side[side] = path
        mapping["articulation_prims_by_side"] = articulation_prims_by_side
    return mapping


def mjcf_auxiliary_visual_names(scene_xml: Path) -> tuple[set[str], set[str]]:
    """Return importer names for collision geoms and non-rendered MJCF sites."""
    root = ET.parse(scene_xml).getroot()
    collision_names = set()
    for geom in root.findall(".//geom"):
        if geom.get("group") != "3":
            continue
        # Isaac 5 names primitive visuals after the geom, but mesh visuals
        # after the referenced mesh asset (for example right_object_0 becomes
        # right_0). Record both forms so either importer layout is covered.
        collision_names.update(
            value for value in (geom.get("name"), geom.get("mesh")) if value
        )
    site_names = {
        site.get("name")
        for site in root.findall(".//site")
        if site.get("name")
    }
    return collision_names, site_names


def hide_auxiliary_visuals(stage, scene_xml: Path, UsdGeom) -> list[str]:
    """Hide imported collision proxies and site markers without removing physics."""
    collision_names, site_names = mjcf_auxiliary_visual_names(scene_xml)
    hidden_paths = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        name = prim.GetName()
        is_collision_visual = name in collision_names
        # Site prototypes live below /meshes. Restrict by path because site
        # names can equal real body names (notably ``right_object``).
        is_site_visual = path.startswith("/meshes/") and name in site_names
        if not (is_collision_visual or is_site_visual):
            continue
        imageable = UsdGeom.Imageable(prim)
        if imageable:
            imageable.MakeInvisible()
            hidden_paths.append(path)
    return hidden_paths


def repair_default_mesh_materials(stage, UsdShade) -> int:
    """Let visual-link materials override the importer's mesh default.

    Isaac Sim 5 authors ``DefaultMaterial`` directly on nearly every imported
    mesh.  A direct mesh binding has stronger USD precedence than the colour
    binding on the visual-link Xform, making the whole robot scene grey.  The
    MuJoCo material colours are already imported into ``/World/Looks``; clear
    only the unwanted default bindings so the mesh inherits those colours.
    """
    repaired = 0
    for prim in stage.Traverse():
        if prim.GetTypeName() != "Mesh":
            continue
        direct = UsdShade.MaterialBindingAPI(prim).GetDirectBinding()
        if str(direct.GetMaterialPath()) != "/World/Looks/DefaultMaterial":
            continue
        direct.GetBindingRel().ClearTargets(False)
        repaired += 1
    return repaired


def load_obj_face_varying_uvs(mesh_path: Path) -> list[tuple[float, float]]:
    """Read OBJ texture coordinates in face-vertex order for a USD primvar."""
    texture_coordinates = []
    face_texture_indices = []
    for raw_line in mesh_path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = raw_line.split()
        if not fields:
            continue
        if fields[0] == "vt" and len(fields) >= 3:
            texture_coordinates.append((float(fields[1]), float(fields[2])))
        elif fields[0] == "f":
            for vertex in fields[1:]:
                components = vertex.split("/")
                if len(components) < 2 or not components[1]:
                    raise ValueError(
                        f"OBJ face has no texture-coordinate index: {vertex!r}"
                    )
                index = int(components[1])
                # OBJ indices are one-based; negative indices are relative to
                # the texture-coordinate list at the point of use.
                face_texture_indices.append(
                    index - 1 if index > 0 else len(texture_coordinates) + index
                )
    if not texture_coordinates or not face_texture_indices:
        raise ValueError(f"OBJ has no usable UV coordinates: {mesh_path}")
    return [texture_coordinates[index] for index in face_texture_indices]


def resolve_package_asset(package_directory: Path, value: str | None) -> Path:
    """Resolve a manifest asset relative to its replay package."""
    path = Path(value or "").expanduser()
    if not path.is_absolute():
        path = package_directory / path
    return path.resolve()


def repair_task_object_appearance(
    stage, package, UsdGeom, UsdShade, Sdf, Gf
) -> dict[str, object]:
    """Restore the task object's OBJ UVs and texture lost by MJCF import.

    Isaac Sim 5 does not reliably propagate a dynamic rigid-body transform to
    the nested visual instance produced by its in-memory MJCF importer.  Build
    one ordinary, top-level replay visual that the player drives with the same
    pose as the physical object, so RTX and PhysX remain explicitly aligned.
    """
    object_manifest = package.manifest.get("object", {})
    mesh_path = resolve_package_asset(
        package.directory, object_manifest.get("source_mesh")
    )
    texture_path = resolve_package_asset(
        package.directory, object_manifest.get("source_texture")
    )
    if not mesh_path.is_file() or not texture_path.is_file():
        return {"status": "skipped", "reason": "source mesh or texture unavailable"}

    mesh_prim_path = "/meshes/task_object_visual_mesh/task_object_visual_mesh"
    mesh_prim = stage.GetPrimAtPath(mesh_prim_path)
    if not mesh_prim.IsValid():
        raise RuntimeError(f"Imported task object mesh is missing: {mesh_prim_path}")
    mesh = UsdGeom.Mesh(mesh_prim)
    face_vertex_count = sum(mesh.GetFaceVertexCountsAttr().Get() or [])
    face_varying_uvs = load_obj_face_varying_uvs(mesh_path)
    if len(face_varying_uvs) != face_vertex_count:
        raise RuntimeError(
            "Task object UV count does not match imported mesh topology: "
            f"{len(face_varying_uvs)} UVs for {face_vertex_count} face vertices."
        )

    primvar = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
    )
    primvar.Set([Gf.Vec2f(u, v) for u, v in face_varying_uvs])

    # Isaac Sim 5 marks the imported OBJ prototype Mesh invisible even though
    # its referencing visual Xform remains visible.  The recorded object pose
    # is therefore correct but the tool is absent from the rendered hand.
    # Author a stronger opinion on the actual prototype Mesh.
    UsdGeom.Imageable(mesh_prim).CreateVisibilityAttr().Set(
        UsdGeom.Tokens.inherited
    )
    # The Isaac-safe STL is a triangle soup.  Preserve it from renderer-side
    # back-face elimination when individual converted triangle windings do not
    # agree with the source OBJ normals.
    mesh.CreateDoubleSidedAttr(True)

    texture_directory = package.directory / "isaac_assets"
    texture_directory.mkdir(parents=True, exist_ok=True)
    packaged_texture = texture_directory / texture_path.name
    shutil.copy2(texture_path, packaged_texture)

    material = UsdShade.Material.Define(
        stage, "/World/Looks/ReplayTaskObjectMaterial"
    )
    surface = UsdShade.Shader.Define(
        stage, "/World/Looks/ReplayTaskObjectMaterial/PreviewSurface"
    )
    surface.CreateIdAttr("UsdPreviewSurface")
    surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.45)
    surface.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)

    texture = UsdShade.Shader.Define(
        stage, "/World/Looks/ReplayTaskObjectMaterial/DiffuseTexture"
    )
    texture.CreateIdAttr("UsdUVTexture")
    packaged_texture_asset = Path("isaac_assets") / packaged_texture.name
    texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath(packaged_texture_asset.as_posix())
    )
    texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)

    reader = UsdShade.Shader.Define(
        stage, "/World/Looks/ReplayTaskObjectMaterial/PrimvarReader"
    )
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        reader.ConnectableAPI(), "result"
    )
    surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        texture.ConnectableAPI(), "rgb"
    )
    surface.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(
        surface.ConnectableAPI(), "surface"
    )
    UsdShade.MaterialBindingAPI.Apply(mesh_prim).Bind(material)

    object_name = object_manifest.get("body_name", "right_object")
    rigid_body_paths = (
        f"/World/{object_name}/{object_name}",
        f"/World/{object_name}",
    )
    rigid_body_path = next(
        (path for path in rigid_body_paths if stage.GetPrimAtPath(path).IsValid()),
        None,
    )
    if rigid_body_path is None:
        raise RuntimeError(
            f"Could not find imported task object body; tried {rigid_body_paths}."
        )
    replay_visual_root_path = "/World/ReplayTaskObjectVisual"
    UsdGeom.Xform.Define(stage, replay_visual_root_path)
    replay_mesh_path = replay_visual_root_path + "/Geometry"
    replay_mesh = UsdGeom.Mesh.Define(stage, replay_mesh_path)
    replay_mesh.CreatePointsAttr().Set(mesh.GetPointsAttr().Get())
    replay_mesh.CreateFaceVertexCountsAttr().Set(
        mesh.GetFaceVertexCountsAttr().Get()
    )
    replay_mesh.CreateFaceVertexIndicesAttr().Set(
        mesh.GetFaceVertexIndicesAttr().Get()
    )
    extent = mesh.GetExtentAttr().Get()
    if extent is not None:
        replay_mesh.CreateExtentAttr().Set(extent)
    normals = mesh.GetNormalsAttr().Get()
    if normals is not None and len(normals):
        replay_mesh.CreateNormalsAttr().Set(normals)
        replay_mesh.SetNormalsInterpolation(mesh.GetNormalsInterpolation())
    replay_mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    replay_mesh.CreateDoubleSidedAttr().Set(True)
    replay_primvar = UsdGeom.PrimvarsAPI(replay_mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
    )
    replay_primvar.Set([Gf.Vec2f(u, v) for u, v in face_varying_uvs])
    UsdShade.MaterialBindingAPI.Apply(replay_mesh.GetPrim()).Bind(material)

    imported_visual_root_path = rigid_body_path + "/visuals"
    imported_visual_root = stage.GetPrimAtPath(imported_visual_root_path)
    if imported_visual_root.IsValid():
        UsdGeom.Imageable(imported_visual_root).MakeInvisible()
    return {
        "status": "repaired",
        "source_mesh_prim": mesh_prim_path,
        "xform_prim": replay_visual_root_path,
        "mesh_prim": replay_mesh_path,
        "hidden_imported_visual_root": imported_visual_root_path,
        "face_varying_uv_count": len(face_varying_uvs),
        "texture": packaged_texture_asset.as_posix(),
        "visibility": "inherited",
        "double_sided": True,
    }


def add_replay_collision_proxies(stage, UsdGeom, UsdPhysics, Sdf, Gf) -> int:
    """Add stable primitive colliders for the imported tabletop and tool.

    The MJCF importer keeps the table collision references in an instance
    branch and the whisking tool is visual-only in MuJoCo.  Isaac Sim 5 can
    consequently omit those shapes from the live PhysX scene.  These simple
    primitives are authored in the exported root layer: the table proxies are
    static world geometry and the tool proxy is under the referenced collision
    prototype, so it follows the imported rigid body.
    """
    created = 0

    def cube(path, translate, half_extents):
        nonlocal created
        if stage.GetPrimAtPath(path).IsValid():
            return
        prim = UsdGeom.Cube.Define(stage, Sdf.Path(path))
        prim.CreateSizeAttr(2.0)
        prim.AddTranslateOp().Set(Gf.Vec3d(*translate))
        prim.AddScaleOp().Set(Gf.Vec3d(*half_extents))
        UsdPhysics.CollisionAPI.Apply(prim.GetPrim())
        UsdGeom.Imageable(prim).MakeInvisible()
        created += 1

    # The MuJoCo scene uses a 1.1176 m square tabletop, 12.7 mm thick.
    table_root = "/World/ReplayCollisionProxies"
    if not stage.GetPrimAtPath(table_root).IsValid():
        UsdGeom.Xform.Define(stage, Sdf.Path(table_root))
    cube(table_root + "/table_center", (0.0, 0.0, 0.62865), (0.5588, 0.5588, 0.00635))
    cube(table_root + "/table_edge_px", (0.5588, 0.0, 0.62865), (0.0508, 0.5588, 0.00635))
    cube(table_root + "/table_edge_nx", (-0.5588, 0.0, 0.62865), (0.0508, 0.5588, 0.00635))
    cube(table_root + "/table_edge_py", (0.0, 0.5588, 0.62865), (0.5588, 0.0508, 0.00635))
    cube(table_root + "/table_edge_ny", (0.0, -0.5588, 0.62865), (0.5588, 0.0508, 0.00635))

    # The whisking tool has no MuJoCo collision geom.  Keep this proxy in the
    # global prototype referenced by /World/right_object/.../collisions.
    object_root = "/collisions/right_object"
    if not stage.GetPrimAtPath(object_root).IsValid():
        UsdGeom.Xform.Define(stage, Sdf.Path(object_root))
    cube(object_root + "/right_object_collision_proxy", (0.0, 0.0, 0.0), (0.0312, 0.0894, 0.0277))
    return created


def repair_root_fixed_joints(stage, UsdPhysics, Gf) -> int:
    """Repair MJCF world welds so imported robot bases stay mounted.

    Isaac Sim 5 leaves the world side of these fixed joints empty while the
    base rigid body is authored at its world pose.  PhysX interprets that as a
    disjointed joint and snaps the base to the origin on reset.  Put the
    authored base transform on joint body0 and keep body1 at the link origin.
    """
    repaired = 0
    for side in ("left", "right"):
        joint_prim = stage.GetPrimAtPath(f"/World/joints/rootJoint_{side}_base")
        body_prim = stage.GetPrimAtPath(f"/World/{side}_base/{side}_base")
        if not joint_prim.IsValid() or not body_prim.IsValid():
            continue
        position = body_prim.GetAttribute("xformOp:translate").Get()
        orientation = body_prim.GetAttribute("xformOp:orient").Get()
        if position is None or orientation is None:
            continue
        joint = UsdPhysics.FixedJoint(joint_prim)
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*position))
        joint.CreateLocalRot0Attr().Set(
            Gf.Quatf(orientation.GetReal(), *orientation.GetImaginary())
        )
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        repaired += 1
    return repaired


def validate_hand_mount_fixed_joints(stage, UsdPhysics) -> dict[str, dict[str, str]]:
    """Fail export if either Sharpa palm is not welded to its UR3e flange.

    The MJCF importer represents an ``MjSpec.attach`` as a fixed joint.  A USD
    can still look superficially complete when that joint is absent or targets
    the wrong rigid body, so make this assembly invariant part of export.
    """
    mounts = {}
    for side in ("left", "right"):
        joint_path = f"/World/joints/{side}_hand_{side}_hand_C_MC"
        expected_body0 = f"/World/{side}_base/{side}_wrist_3_link"
        expected_body1 = f"/World/{side}_base/{side}_hand_{side}_hand_C_MC"
        joint_prim = stage.GetPrimAtPath(joint_path)
        if not joint_prim.IsValid():
            raise RuntimeError(f"Missing Sharpa hand mount fixed joint: {joint_path}")
        joint = UsdPhysics.FixedJoint(joint_prim)
        body0 = [str(path) for path in joint.GetBody0Rel().GetTargets()]
        body1 = [str(path) for path in joint.GetBody1Rel().GetTargets()]
        if body0 != [expected_body0] or body1 != [expected_body1]:
            raise RuntimeError(
                f"Invalid {side} Sharpa hand mount: body0={body0}, body1={body1}; "
                f"expected {[expected_body0]} -> {[expected_body1]}."
            )
        mounts[side] = {
            "joint_prim": joint_path,
            "ur3e_flange_body": expected_body0,
            "sharpa_palm_body": expected_body1,
        }
    return mounts


def prepare_legacy_mjcf(scene_xml: Path, package_dir: Path) -> Path:
    """Sanitize binary STL headers that Isaac Sim 5 mistakes for ASCII STL."""
    tree = ET.parse(scene_xml)
    root = tree.getroot()
    compiler = root.find("compiler")
    mesh_directory = scene_xml.parent
    if compiler is not None and compiler.get("meshdir"):
        mesh_directory = (scene_xml.parent / compiler.get("meshdir")).resolve()

    compatibility_directory = package_dir / "isaac5_compat"
    sanitized_assets = {}
    mesh_sources = []
    for mesh in root.findall("./asset/mesh"):
        file_value = mesh.get("file")
        if not file_value:
            continue
        source = Path(file_value)
        if not source.is_absolute():
            source = mesh_directory / source
        source = source.resolve()
        mesh_sources.append((mesh, source))
        if source.suffix.lower() != ".stl" or not source.is_file():
            continue
        data = source.read_bytes()
        if len(data) < 84 or data[:5].lower() != b"solid":
            continue
        triangle_count = int.from_bytes(data[80:84], byteorder="little")
        if len(data) != 84 + triangle_count * 50:
            continue

        destination = sanitized_assets.get(source)
        if destination is None:
            compatibility_directory.mkdir(parents=True, exist_ok=True)
            destination = compatibility_directory / (
                f"{len(sanitized_assets):03d}_{source.name}"
            )
            binary_header = b"Binary STL sanitized for Isaac Sim 5"
            destination.write_bytes(binary_header.ljust(80, b" ") + data[80:])
            sanitized_assets[source] = destination

    if not sanitized_assets:
        return scene_xml
    for mesh, source in mesh_sources:
        target = sanitized_assets.get(source, source)
        mesh.set("file", os.path.relpath(target, compatibility_directory))
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    compiler.set("meshdir", ".")
    compatibility_scene = compatibility_directory / "scene.xml"
    tree.write(compatibility_scene, encoding="utf-8", xml_declaration=True)
    print(
        f"Sanitized {len(sanitized_assets)} ambiguous binary STL asset(s): "
        f"{compatibility_scene}",
        flush=True,
    )
    return compatibility_scene


def import_mjcf_stage(scene_xml, package_dir, simulation_app, open_stage):
    """Import MJCF with either the Isaac Sim 6 or Isaac Sim 5 API."""
    import isaacsim.asset.importer.mjcf as mjcf

    if hasattr(mjcf, "MJCFImporter"):
        config = mjcf.MJCFImporterConfig()
        config.mjcf_path = str(scene_xml)
        import_directory = package_dir / "mjcf_import"
        import_directory.mkdir(parents=True, exist_ok=True)
        config.usd_path = str(import_directory)
        config.import_scene = True
        config.merge_mesh = False
        config.collision_from_visuals = False
        config.allow_self_collision = True
        config.fix_base = True
        config.run_asset_transformer = False
        imported_output = Path(mjcf.MJCFImporter(config).import_mjcf()).resolve()
        print(f"MJCF importer output: {imported_output}", flush=True)
        if not imported_output.is_file():
            raise RuntimeError(
                f"MJCF importer did not create its reported output {imported_output}."
            )
        open_stage(str(imported_output))
        import omni.usd

        return omni.usd.get_context().get_stage()

    # Isaac Sim 5.x exposes the importer through Kit commands and authors the
    # result into the current in-memory stage.
    import omni.kit.commands
    import omni.usd

    context = omni.usd.get_context()
    context.new_stage()
    simulation_app.update()
    status, config = omni.kit.commands.execute("MJCFCreateImportConfig")
    if not status:
        raise RuntimeError("Isaac Sim did not create an MJCF import config.")
    # The UR3e bases are mounted to the tabletop in the MuJoCo scene.  A
    # floating Isaac articulation otherwise gets reset as a dynamic body and
    # falls under gravity before replay starts.
    config.set_fix_base(True)
    config.set_import_inertia_tensor(True)
    config.set_merge_fixed_joints(False)
    config.set_convex_decomp(False)
    # Replay drives the recorded joint positions directly; enabling the
    # imported MJCF self-collision set makes PhysX accumulate impulses during
    # teleported frames and can eventually produce NaN link transforms.  The
    # task only needs robot/table and tool/table contacts, not robot self hits.
    config.set_self_collision(False)
    config.set_make_default_prim(True)
    config.set_distance_scale(1.0)
    scene_xml = prepare_legacy_mjcf(scene_xml, package_dir)
    status, prim_path = omni.kit.commands.execute(
        "MJCFCreateAsset",
        mjcf_path=str(scene_xml),
        import_config=config,
        prim_path="/World",
    )
    if not status:
        raise RuntimeError("Isaac Sim 5 MJCFCreateAsset command failed.")
    simulation_app.update()
    print(f"MJCF importer prim: {prim_path}", flush=True)
    return context.get_stage()


def main() -> None:
    args = parse_args()
    package = load_replay_package(args.package_dir)
    scene_xml = resolve_scene_xml(package, args.run_dir, args.scene)
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
        {
            "headless": args.headless,
            # USD and manifests are exported before close(); avoid Isaac 5's
            # lengthy extension-by-extension teardown after those writes.
            "fast_shutdown": True,
        }
    )
    try:
        from isaacsim.core.utils.extensions import enable_extension
        from isaacsim.core.utils.stage import open_stage
        from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics, UsdShade
        import omni.usd

        enable_extension("isaacsim.asset.importer.mjcf")
        print(f"Importing MJCF scene: {scene_xml}", flush=True)
        stage = import_mjcf_stage(
            scene_xml, package.directory, simulation_app, open_stage
        )
        hidden_auxiliary_paths = hide_auxiliary_visuals(stage, scene_xml, UsdGeom)
        print(
            f"Hidden {len(hidden_auxiliary_paths)} collision/site visual prim(s).",
            flush=True,
        )
        repaired_material_count = repair_default_mesh_materials(stage, UsdShade)
        print(
            f"Repaired {repaired_material_count} direct DefaultMaterial mesh binding(s).",
            flush=True,
        )
        task_object_appearance = repair_task_object_appearance(
            stage, package, UsdGeom, UsdShade, Sdf, Gf
        )
        print(
            f"Task object appearance: {task_object_appearance['status']}",
            flush=True,
        )
        collision_proxy_count = add_replay_collision_proxies(
            stage, UsdGeom, UsdPhysics, Sdf, Gf
        )
        print(
            f"Added {collision_proxy_count} stable replay collision proxy/proxies.",
            flush=True,
        )
        repaired_root_joint_count = repair_root_fixed_joints(stage, UsdPhysics, Gf)
        print(
            f"Repaired {repaired_root_joint_count} robot base fixed joint(s).",
            flush=True,
        )
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
        scene_manifest["validated_hand_mounts"] = validate_hand_mount_fixed_joints(
            stage, UsdPhysics
        )
        scene_manifest["hidden_auxiliary_visual_prim_count"] = len(
            hidden_auxiliary_paths
        )
        scene_manifest["replay_collision_proxy_count"] = collision_proxy_count
        scene_manifest["repaired_root_fixed_joint_count"] = repaired_root_joint_count
        scene_manifest["task_object_appearance"] = task_object_appearance
        manifest_path = package.directory / "isaac_scene_manifest.json"
        manifest_path.write_text(
            json.dumps(scene_manifest, indent=2) + "\n", encoding="utf-8"
        )
        package.manifest["isaac_sim"].update(
            {
                "version": isaac_sim_version(),
                "asset_conversion_status": "pass",
            }
        )
        package.manifest_path.write_text(
            json.dumps(package.manifest, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Created Isaac scene: {output}", flush=True)
        print(f"Created scene mapping: {manifest_path}", flush=True)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
