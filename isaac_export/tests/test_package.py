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


if __name__ == "__main__":
    unittest.main()
