import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from replay_viser import (
    _project_object_qpos_upright,
    _project_scene_qpos_upright,
    _rotations_to_horizontal,
)


class ReplayUprightDisplayTest(unittest.TestCase):
    def test_projects_only_object_quaternion(self) -> None:
        qpos = np.arange(70, dtype=np.float64).reshape(2, 35)
        rotations = Rotation.from_euler(
            "zyx",
            [[30.0, 25.0, -10.0], [-80.0, -20.0, 15.0]],
            degrees=True,
        )
        quat_xyzw = rotations.as_quat()
        qpos[:, 31:35] = quat_xyzw[:, [3, 0, 1, 2]]

        corrected = _project_object_qpos_upright(qpos, [28])
        matrices = Rotation.from_quat(
            corrected[:, [32, 33, 34, 31]]
        ).as_matrix()

        np.testing.assert_allclose(
            matrices[:, :, 2],
            np.broadcast_to([0.0, 0.0, 1.0], (2, 3)),
            atol=1e-12,
        )
        expected_heading = np.arctan2(
            rotations.as_matrix()[:, 1, 0], rotations.as_matrix()[:, 0, 0]
        )
        actual_heading = np.arctan2(matrices[:, 1, 0], matrices[:, 0, 0])
        np.testing.assert_allclose(actual_heading, expected_heading, atol=1e-12)
        np.testing.assert_array_equal(corrected[:, :31], qpos[:, :31])

    def test_projects_finger_directions_onto_world_xy(self) -> None:
        directions = np.asarray(
            [[1.0, 2.0, 3.0], [-2.0, 1.0, -0.5]], dtype=np.float64
        )

        correction, elevation_deg = _rotations_to_horizontal(directions)
        corrected = correction.apply(directions)

        np.testing.assert_allclose(corrected[:, 2], 0.0, atol=1e-12)
        expected_heading = np.arctan2(directions[:, 1], directions[:, 0])
        actual_heading = np.arctan2(corrected[:, 1], corrected[:, 0])
        np.testing.assert_allclose(actual_heading, expected_heading, atol=1e-12)
        self.assertTrue(np.all(elevation_deg > 0.0))

    def test_projects_semantic_negative_y_axis_upright(self) -> None:
        qpos = np.zeros((2, 35), dtype=np.float64)
        qpos[:, 31] = 1.0

        corrected = _project_object_qpos_upright(
            qpos, [28], np.asarray([0.0, -1.0, 0.0])
        )
        rotation = Rotation.from_quat(corrected[:, [32, 33, 34, 31]])

        np.testing.assert_allclose(
            rotation.apply(np.broadcast_to([0.0, -1.0, 0.0], (2, 3))),
            np.broadcast_to([0.0, 0.0, 1.0], (2, 3)),
            atol=1e-12,
        )

    def test_scene_projection_preserves_hand_object_relative_position(self) -> None:
        qpos = np.zeros((2, 20), dtype=np.float64)
        qpos[:, :3] = [[0.3, -0.2, 0.5], [-0.1, 0.4, 0.8]]
        qpos[:, 3:6] = [[0.2, -0.1, 0.3], [-0.2, 0.4, -0.1]]
        object_qadr = 8
        qpos[:, object_qadr : object_qadr + 3] = [
            [0.1, -0.1, 0.2],
            [-0.2, 0.1, 0.3],
        ]
        object_rotation = Rotation.from_euler(
            "xyz", [[0.4, 0.2, -0.3], [-0.1, 0.5, 0.2]]
        )
        object_xyzw = object_rotation.as_quat()
        qpos[:, object_qadr + 3 : object_qadr + 7] = object_xyzw[:, [3, 0, 1, 2]]

        corrected = _project_scene_qpos_upright(
            qpos,
            [object_qadr],
            [("right", 0, 3)],
            np.asarray([0.0, 1.0, 0.0]),
        )

        corrected_object = Rotation.from_quat(
            corrected[:, object_qadr + np.asarray([4, 5, 6, 3])]
        )
        np.testing.assert_allclose(
            corrected_object.apply([[0.0, 1.0, 0.0]] * 2),
            [[0.0, 0.0, 1.0]] * 2,
            atol=1e-12,
        )
        before_relative = object_rotation.inv().apply(
            qpos[:, :3] - qpos[:, object_qadr : object_qadr + 3]
        )
        after_relative = corrected_object.inv().apply(
            corrected[:, :3]
            - corrected[:, object_qadr : object_qadr + 3]
        )
        np.testing.assert_allclose(after_relative, before_relative, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
