import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from isaac_export.source import load_source_trajectory
from isaac_export.trajectory import build_standard_trajectory


FINGER_NAMES = [f"right_finger_{index}" for index in range(22)]


def _scene_xml() -> str:
    root_joints = "\n".join(
        f'<joint name="right_{name}" range="-6.28 6.28"/>'
        for name in ("pos_x", "pos_y", "pos_z", "rot_x", "rot_y", "rot_z")
    )
    finger_joints = "\n".join(
        f'<joint name="{name}" range="-1 1"/>' for name in FINGER_NAMES
    )
    return f"""
<mujoco>
  <compiler meshdir="assets"/>
  <asset><mesh name="right_visual" file="object.obj"/></asset>
  <worldbody>
    <body name="right_base">{root_joints}{finger_joints}</body>
    <body name="right_object">
      <joint name="right_object_joint" type="free"/>
      <geom mesh="right_visual" group="1"/>
    </body>
  </worldbody>
</mujoco>
"""


class SourceTest(unittest.TestCase):
    def test_chunks_are_sorted_and_warmup_is_removed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_dir = Path(temporary_directory)
            (run_dir / "scene.xml").write_text(_scene_xml(), encoding="utf-8")
            (run_dir / "config.yaml").write_text(
                yaml.safe_dump(
                    {
                        "robot_type": "sharpa",
                        "embodiment_type": "right",
                        "task": "test",
                        "sim_dt": 0.01,
                        "ref_dt": 0.02,
                        "warmup_steps": 2,
                    }
                ),
                encoding="utf-8",
            )
            first = np.zeros((2, 35), dtype=np.float64)
            second = np.zeros((2, 35), dtype=np.float64)
            first[:, 0] = [0.0, 1.0]
            second[:, 0] = [2.0, 3.0]
            first[:, 31] = 1.0
            second[:, 31] = 1.0
            np.savez(
                run_dir / "trajectory_mjwp.npz",
                qpos=np.stack((second, first)),
                time=np.array([[0.03, 0.04], [0.01, 0.02]]),
                sim_step=np.array([4, 2]),
            )

            source = load_source_trajectory(run_dir, "trajectory_mjwp.npz", 22)
            trajectory = build_standard_trajectory(source, skip_warmup=True)

            np.testing.assert_array_equal(source.qpos[:, 0], [0.0, 1.0, 2.0, 3.0])
            np.testing.assert_array_equal(trajectory.hand_root_position[:, 0], [2.0, 3.0])
            np.testing.assert_allclose(trajectory.timestamps, [0.0, 0.01])
            np.testing.assert_array_equal(trajectory.source_frame_indices, [2, 3])
            self.assertTrue(trajectory.valid_mask.all())
            self.assertEqual(
                [spec.name for spec in source.protocol.finger_joints], FINGER_NAMES
            )

    def test_kinematic_frequency_overrides_mjwp_timing_and_warmup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_dir = Path(temporary_directory)
            (run_dir / "scene.xml").write_text(_scene_xml(), encoding="utf-8")
            (run_dir / "config.yaml").write_text(
                yaml.safe_dump(
                    {
                        "robot_type": "sharpa",
                        "embodiment_type": "right",
                        "task": "test",
                        "sim_dt": 0.005,
                        "ref_dt": 0.0333,
                        "warmup_steps": 600,
                    }
                ),
                encoding="utf-8",
            )
            qpos = np.zeros((3, 35), dtype=np.float64)
            qpos[:, 31] = 1.0
            np.savez(
                run_dir / "trajectory_kinematic.npz",
                qpos=qpos,
                frequency=np.array(50.0),
            )

            source = load_source_trajectory(
                run_dir, "trajectory_kinematic.npz", 22
            )
            trajectory = build_standard_trajectory(source, skip_warmup=True)

            self.assertEqual(source.trajectory_kind, "kinematic")
            self.assertEqual(source.warmup_steps, 0)
            self.assertAlmostEqual(source.sim_dt, 0.02)
            np.testing.assert_allclose(trajectory.timestamps, [0.0, 0.02, 0.04])


if __name__ == "__main__":
    unittest.main()