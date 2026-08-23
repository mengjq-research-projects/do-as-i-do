import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

from build_usd import (
    hide_auxiliary_visuals,
    inspect_stage,
    load_obj_face_varying_uvs,
    mjcf_auxiliary_visual_names,
    prepare_legacy_mjcf,
    resolve_package_asset,
    resolve_scene_xml,
    validate_hand_mount_fixed_joints,
)


class FakeRelationship:
    def __init__(self, targets=()):
        self._targets = targets

    def GetTargets(self):
        return self._targets


class FakePrim:
    def __init__(
        self,
        name,
        path,
        parent=None,
        *,
        articulation=False,
        rigid=False,
        body_targets=(),
    ):
        self._name = name
        self._path = path
        self._parent = parent
        self._articulation = articulation
        self._rigid = rigid
        self.body_targets = body_targets

    def GetName(self):
        return self._name

    def GetPath(self):
        return self._path

    def GetParent(self):
        return self._parent

    def IsValid(self):
        return True

    def HasAPI(self, api):
        return (
            api is FakeUsdPhysics.ArticulationRootAPI and self._articulation
        ) or (api is FakeUsdPhysics.RigidBodyAPI and self._rigid)


class FakeUsdPhysics:
    class ArticulationRootAPI:
        pass

    class RigidBodyAPI:
        pass

    class Joint:
        def __init__(self, prim):
            self._prim = prim

        def GetBody0Rel(self):
            return FakeRelationship(self._prim.body_targets[:1])

        def GetBody1Rel(self):
            return FakeRelationship(self._prim.body_targets[1:])

    FixedJoint = Joint


def test_resolve_scene_xml_prefers_isaac_safe_robot_scene(tmp_path):
    package_dir = tmp_path / "package"
    package_dir.mkdir()
    original = package_dir / "scene_dual_ur3e_object.xml"
    isaac_safe = package_dir / "scene_dual_ur3e_object_isaac.xml"
    original.write_text("<mujoco/>", encoding="utf-8")
    isaac_safe.write_text("<mujoco/>", encoding="utf-8")
    package = SimpleNamespace(
        directory=package_dir,
        is_robot_scene=True,
        manifest={"scene": {"mjcf": original.name}},
    )

    assert resolve_scene_xml(package, None) == isaac_safe.resolve()


def test_resolve_scene_xml_honors_explicit_override(tmp_path):
    package_dir = tmp_path / "package"
    package_dir.mkdir()
    override = tmp_path / "explicit.xml"
    override.write_text("<mujoco/>", encoding="utf-8")
    package = SimpleNamespace(
        directory=package_dir,
        is_robot_scene=True,
        manifest={"scene": {"mjcf": "missing.xml"}},
    )

    assert resolve_scene_xml(package, None, override) == override.resolve()


def test_validate_hand_mount_fixed_joints_checks_both_welds():
    joints = {}
    for side in ("left", "right"):
        path = f"/World/joints/{side}_hand_{side}_hand_C_MC"
        joints[path] = FakePrim(
            f"{side}_hand_{side}_hand_C_MC",
            path,
            body_targets=(
                f"/World/{side}_base/{side}_wrist_3_link",
                f"/World/{side}_base/{side}_hand_{side}_hand_C_MC",
            ),
        )
    stage = SimpleNamespace(GetPrimAtPath=lambda path: joints[str(path)])

    mounts = validate_hand_mount_fixed_joints(stage, FakeUsdPhysics)

    assert set(mounts) == {"left", "right"}
    assert mounts["right"]["ur3e_flange_body"].endswith("right_wrist_3_link")


def test_inspect_stage_returns_replay_mapping(tmp_path):
    root = FakePrim("right_base", "/World/right_base", articulation=True)
    joint = FakePrim("right_finger_0", "/World/right_base/right_finger_0", root)
    replay_object = FakePrim(
        "right_object", "/World/right_object", rigid=True
    )
    prims = (root, joint, replay_object)
    stage = SimpleNamespace(
        Traverse=lambda: iter(prims),
        GetPrimAtPath=lambda path: next(p for p in prims if p.GetPath() == path),
        GetRootLayer=lambda: SimpleNamespace(realPath=str(tmp_path / "scene.usd")),
    )
    package = SimpleNamespace(
        directory=tmp_path,
        driven_joint_names=("right_finger_0",),
        manifest={"object": {"body_name": "right_object"}},
        is_robot_scene=False,
        arm_joint_names=(),
        root_joint_names=(),
        finger_joint_names=("right_finger_0",),
    )

    mapping = inspect_stage(stage, package, FakeUsdPhysics)

    assert mapping["scene_usd"] == "scene.usd"
    assert mapping["robot_articulation_prim"] == "/World/right_base"
    assert mapping["object_prim"] == "/World/right_object"


def test_resolve_package_asset_supports_relative_manifest_paths(tmp_path):
    asset = tmp_path / "source_assets" / "tool.obj"

    assert resolve_package_asset(tmp_path, "source_assets/tool.obj") == asset.resolve()


def test_inspect_stage_follows_flat_joint_body_relationships(tmp_path):
    root = FakePrim("right_base", "/World/right_base", articulation=True)
    joints_scope = FakePrim("joints", "/World/joints")
    joint = FakePrim(
        "right_finger_0",
        "/World/joints/right_finger_0",
        joints_scope,
        body_targets=("/World/right_base",),
    )
    replay_object = FakePrim("right_object", "/World/right_object", rigid=True)
    prims = (root, joints_scope, joint, replay_object)
    stage = SimpleNamespace(
        Traverse=lambda: iter(prims),
        GetPrimAtPath=lambda path: next(p for p in prims if p.GetPath() == str(path)),
        GetRootLayer=lambda: SimpleNamespace(realPath=str(tmp_path / "scene.usd")),
    )
    package = SimpleNamespace(
        directory=tmp_path,
        driven_joint_names=("right_finger_0",),
        manifest={"object": {"body_name": "right_object"}},
        is_robot_scene=False,
        arm_joint_names=(),
        root_joint_names=(),
        finger_joint_names=("right_finger_0",),
    )

    mapping = inspect_stage(stage, package, FakeUsdPhysics)

    assert mapping["robot_articulation_prim"] == "/World/right_base"


def test_prepare_legacy_mjcf_sanitizes_ambiguous_binary_stl(tmp_path):
    assets = tmp_path / "assets"
    assets.mkdir()
    mesh = assets / "thumb.stl"
    plain_mesh = assets / "plain.obj"
    plain_mesh.write_text("v 0 0 0\n", encoding="utf-8")
    triangle = bytes(50)
    original = (
        b"solid non-ascii \xff".ljust(80, b" ")
        + (1).to_bytes(4, "little")
        + triangle
    )
    mesh.write_bytes(original)
    scene = tmp_path / "scene.xml"
    scene.write_text(
        '<mujoco><compiler meshdir="assets"/><asset>'
        '<mesh name="thumb" file="thumb.stl"/>'
        '<mesh name="thumb_collision" file="thumb.stl"/>'
        '<mesh name="plain" file="plain.obj"/>'
        "</asset></mujoco>",
        encoding="utf-8",
    )
    package_dir = tmp_path / "package"

    compatible_scene = prepare_legacy_mjcf(scene, package_dir)

    compatible_root = ET.parse(compatible_scene).getroot()
    rewritten = compatible_root.findall("./asset/mesh")
    sanitized = (compatible_scene.parent / rewritten[0].get("file")).read_bytes()
    assert compatible_scene == package_dir / "isaac5_compat" / "scene.xml"
    assert compatible_root.find("compiler").get("meshdir") == "."
    assert rewritten[0].get("file") == rewritten[1].get("file")
    assert (compatible_scene.parent / rewritten[2].get("file")).resolve() == plain_mesh
    assert len(list((package_dir / "isaac5_compat").glob("*.stl"))) == 1
    assert not sanitized.startswith(b"solid")
    assert sanitized[80:] == original[80:]


def test_load_obj_face_varying_uvs_follows_face_indices(tmp_path):
    mesh = tmp_path / "tool.obj"
    mesh.write_text(
        "\n".join(
            (
                "v 0 0 0",
                "v 1 0 0",
                "v 0 1 0",
                "vt 0.1 0.2",
                "vt 0.8 0.2",
                "vt 0.1 0.9",
                "f 1/2/1 2/3/1 3/1/1",
            )
        ),
        encoding="utf-8",
    )

    assert load_obj_face_varying_uvs(mesh) == [
        (0.8, 0.2),
        (0.1, 0.9),
        (0.1, 0.2),
    ]


def test_mjcf_auxiliary_visual_names_includes_geom_and_mesh_names(tmp_path):
    scene = tmp_path / "scene.xml"
    scene.write_text(
        """<mujoco><worldbody><body name="right_object">
        <geom name="right_object_0" mesh="right_0" group="3"/>
        <geom name="right_object_visual" mesh="right_visual" group="1"/>
        <site name="right_object"/><site name="trace_right_object" group="4"/>
        </body></worldbody></mujoco>""",
        encoding="utf-8",
    )

    collision_names, site_names = mjcf_auxiliary_visual_names(scene)

    assert collision_names == {"right_object_0", "right_0"}
    assert site_names == {"right_object", "trace_right_object"}


def test_hide_auxiliary_visuals_preserves_body_with_site_name(tmp_path):
    scene = tmp_path / "scene.xml"
    scene.write_text(
        """<mujoco><worldbody><body name="right_object">
        <geom name="right_object_0" mesh="right_0" group="3"/>
        <geom name="right_object_visual" mesh="right_visual" group="1"/>
        <site name="right_object"/>
        </body></worldbody></mujoco>""",
        encoding="utf-8",
    )
    prims = [
        FakePrim("right_object", "/World/right_object"),
        FakePrim("right_0", "/visuals/right_object/right_0"),
        FakePrim("right_visual", "/visuals/right_object/right_visual"),
        FakePrim("right_object", "/meshes/right_object"),
    ]

    class FakeImageable:
        def __init__(self, prim):
            self.prim = prim

        def __bool__(self):
            return True

        def MakeInvisible(self):
            self.prim.invisible = True

    fake_usd_geom = SimpleNamespace(Imageable=FakeImageable)
    stage = SimpleNamespace(Traverse=lambda: iter(prims))

    hidden = hide_auxiliary_visuals(stage, scene, fake_usd_geom)

    assert hidden == [
        "/visuals/right_object/right_0",
        "/meshes/right_object",
    ]
    assert not hasattr(prims[0], "invisible")
    assert not hasattr(prims[2], "invisible")
