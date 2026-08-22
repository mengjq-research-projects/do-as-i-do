import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from final_package import (
    save_final_package,
    transform_object_trajectory,
    validate_final_package,
)


class FinalPackageTest(unittest.TestCase):
    def test_object_uses_same_workspace_transform_as_wrist(self) -> None:
        qpos = np.zeros((2, 35), dtype=np.float64)
        qpos[:, 31] = 1.0
        qpos[0, 28:31] = [2.0, 1.0, 0.0]
        qpos[1, 28:31] = [3.0, 1.0, 0.0]
        position, quaternion = transform_object_trajectory(
            qpos,
            wrist_anchor=np.array([1.0, 1.0, 0.0]),
            workspace_xyz=np.array([10.0, 20.0, 30.0]),
            yaw_deg=90.0,
        )
        np.testing.assert_allclose(
            position, [[10.0, 21.0, 30.0], [10.0, 22.0, 30.0]], atol=1e-12
        )
        np.testing.assert_allclose(np.linalg.norm(quaternion, axis=1), 1.0)

    def test_saved_package_validates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            mesh = directory / "object.obj"
            mesh.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
            frames = 4
            output, manifest, scene = save_final_package(
                output_path=directory / "trajectory_dual_ur3e.npz",
                scene_xml="<mujoco/>",
                arm_qpos=np.zeros((frames, 6)),
                finger_qpos=np.zeros((frames, 22)),
                object_position=np.zeros((frames, 3)),
                object_quaternion=np.tile([1.0, 0.0, 0.0, 0.0], (frames, 1)),
                arm_joint_names=[f"arm_{index}" for index in range(6)],
                finger_joint_names=[f"finger_{index}" for index in range(22)],
                arm_joint_limits=np.tile([-3.0, 3.0], (6, 1)),
                finger_joint_limits=np.tile([-1.0, 1.0], (22, 1)),
                dt=0.005,
                start_frame=2,
                workspace_xyz=np.zeros(3),
                workspace_yaw_deg=0.0,
                workspace_pitch_deg=0.0,
                workspace_roll_deg=0.0,
                ik_seed_qpos=np.zeros(6),
                side="right",
                source_traj=directory / "source.npz",
                object_mesh=mesh,
                object_texture=None,
                capture_metadata={"viewpoint": "ego", "camera_motion": "moving"},
            )
            self.assertTrue(output.is_file())
            self.assertTrue(manifest.is_file())
            self.assertTrue(scene.is_file())
            report = validate_final_package(directory)
            self.assertEqual(report["status"], "pass")
            with np.load(output, allow_pickle=False) as data:
                np.testing.assert_array_equal(
                    data["valid_mask"], [False, False, True, True]
                )
                self.assertEqual(str(data["capture_viewpoint"]), "ego")


if __name__ == "__main__":
    unittest.main()
