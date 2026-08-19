import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from isaac_export.package import load_replay_package


class ReplayPackageTest(unittest.TestCase):
    def test_loads_and_clips_joint_limits(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            manifest = {
                "timing": {"frame_count": 2, "sim_dt_seconds": 0.01},
                "robot": {
                    "root_source_joints": [f"root_{index}" for index in range(6)],
                    "finger_joints": [
                        {"name": "finger", "lower_rad": -1.0, "upper_rad": 1.0}
                    ],
                },
            }
            (directory / "manifest.json").write_text(json.dumps(manifest))
            np.savez(
                directory / "trajectory.npz",
                timestamps=np.array([0.0, 0.01]),
                hand_root_position=np.zeros((2, 3)),
                hand_root_quaternion=np.tile([1.0, 0.0, 0.0, 0.0], (2, 1)),
                hand_joint_positions=np.array([[0.0], [1.1]]),
                object_position=np.zeros((2, 3)),
                object_quaternion=np.tile([1.0, 0.0, 0.0, 0.0], (2, 1)),
                valid_mask=np.ones(2, dtype=bool),
                source_frame_indices=np.arange(2),
            )

            package = load_replay_package(directory)

            self.assertEqual(package.clip_count, 1)
            self.assertAlmostEqual(package.max_clip_rad, 0.1)
            self.assertEqual(package.hand_joint_state(0).shape, (7,))

    def test_loads_final_robot_scene_package(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            arm_names = [f"arm_{index}" for index in range(6)]
            finger_names = [f"finger_{index}" for index in range(22)]
            manifest = {
                "schema_version": "deployment.robot_scene.v1",
                "timing": {"frame_count": 3, "sim_dt_seconds": 0.005},
                "trajectory": {"file": "trajectory_dual_ur3e.npz"},
                "robot": {
                    "arm_joints": [
                        {"name": name, "lower_rad": -3.0, "upper_rad": 3.0}
                        for name in arm_names
                    ],
                    "finger_joints": [
                        {"name": name, "lower_rad": -1.0, "upper_rad": 1.0}
                        for name in finger_names
                    ],
                },
            }
            (directory / "deployment_manifest.json").write_text(json.dumps(manifest))
            np.savez(
                directory / "trajectory_dual_ur3e.npz",
                timestamps=np.arange(3) * 0.005,
                arm_qpos=np.zeros((3, 6)),
                finger_qpos=np.zeros((3, 22)),
                object_position=np.zeros((3, 3)),
                object_quaternion=np.tile([1.0, 0.0, 0.0, 0.0], (3, 1)),
                valid_mask=np.array([False, True, True]),
                source_frame_indices=np.arange(3),
            )

            package = load_replay_package(directory)

            self.assertTrue(package.is_robot_scene)
            self.assertEqual(package.driven_joint_names, tuple(arm_names + finger_names))
            self.assertEqual(package.hand_joint_state(1).shape, (28,))


if __name__ == "__main__":
    unittest.main()
