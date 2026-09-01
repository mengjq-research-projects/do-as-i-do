import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from retargeting.pipeline.process_dataset import (
    _extrapolate_terminal_hand_translation,
    _project_object_quaternions_upright,
    _rotate_points_per_frame,
    _stabilize_hand_grasp_anchor,
)
from retargeting.pipeline.resolve_pedestal import _object_bottom_footprint


class ObjectUprightTest(unittest.TestCase):
    def test_terminal_hand_retreat_extrapolates_invalid_suffix_rigidly(self) -> None:
        joints = np.zeros((7, 2, 3), dtype=np.float64)
        vertices = np.zeros((7, 3, 3), dtype=np.float64)
        for frame in range(4):
            offset = np.asarray([0.02 * frame, -0.01 * frame, 0.0])
            joints[frame] = np.asarray([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]) + offset
            vertices[frame] = np.asarray(
                [[0.0, 0.0, 0.0], [0.0, 0.1, 0.0], [0.1, 0.0, 0.0]]
            ) + offset
        # Mimic boundary interpolation before applying the retreat prior.
        joints[4:] = joints[3]
        vertices[4:] = vertices[3]
        valid = np.asarray([True, True, True, True, False, False, False])

        out_joints, out_vertices, offsets = _extrapolate_terminal_hand_translation(
            joints, vertices, valid
        )

        np.testing.assert_allclose(offsets[-1], [0.06, -0.03, 0.0])
        np.testing.assert_allclose(
            out_joints[-1] - out_joints[3],
            np.broadcast_to(offsets[-1], (2, 3)),
        )
        np.testing.assert_allclose(
            out_vertices[-1] - out_vertices[3],
            np.broadcast_to(offsets[-1], (3, 3)),
        )

    def test_terminal_hand_retreat_ignores_short_frozen_valid_tail(self) -> None:
        joints = np.zeros((17, 2, 3), dtype=np.float64)
        vertices = np.zeros((17, 3, 3), dtype=np.float64)
        velocity = np.asarray([0.01, -0.02, 0.005])
        for frame in range(10):
            offset = frame * velocity
            joints[frame] = np.asarray(
                [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]
            ) + offset
            vertices[frame] = np.asarray(
                [[0.0, 0.0, 0.0], [0.0, 0.1, 0.0], [0.1, 0.0, 0.0]]
            ) + offset
        # The tracker repeats its last observed pose but still marks two frames
        # valid before reporting loss.
        joints[10:] = joints[9]
        vertices[10:] = vertices[9]
        valid = np.asarray([True] * 12 + [False] * 5)

        out_joints, out_vertices, offsets = _extrapolate_terminal_hand_translation(
            joints, vertices, valid
        )

        self.assertGreater(np.linalg.norm(offsets[-1]), 0.05)
        np.testing.assert_allclose(
            out_joints[-1] - out_joints[11],
            np.broadcast_to(offsets[-1], (2, 3)),
        )
        np.testing.assert_allclose(
            out_vertices[-1] - out_vertices[11],
            np.broadcast_to(offsets[-1], (3, 3)),
        )

    def test_grasp_anchor_stabilizes_closed_pinch_and_preserves_shape(self) -> None:
        n_frames = 12
        joints = np.zeros((n_frames, 21, 3), dtype=np.float64)
        vertices = np.zeros((n_frames, 4, 3), dtype=np.float64)
        obj_trans = np.zeros((n_frames, 3), dtype=np.float64)
        obj_quat = np.zeros((n_frames, 4), dtype=np.float64)
        obj_quat[:, 0] = 1.0
        anchor = np.asarray([0.0, 0.0, -0.05])
        # Closed thumb/index from frames 3..8. Their midpoint drifts while the
        # internal hand geometry remains rigid.
        for frame in range(n_frames):
            drift = np.asarray([0.003 * frame, -0.001 * frame, 0.0])
            joints[frame] = drift
            vertices[frame] = np.asarray(
                [[0.0, 0.0, 0.0], [0.02, 0.0, 0.0], [0.0, 0.02, 0.0], [0.0, 0.0, 0.02]]
            ) + drift
            if 3 <= frame <= 8:
                joints[frame, 4] = anchor + [-0.01, 0.0, 0.0] + drift
                joints[frame, 8] = anchor + [0.01, 0.0, 0.0] + drift
            else:
                joints[frame, 4] = anchor + [-0.05, 0.0, 0.0] + drift
                joints[frame, 8] = anchor + [0.05, 0.0, 0.0] + drift

        out_joints, out_vertices, offsets, span = _stabilize_hand_grasp_anchor(
            joints,
            vertices,
            obj_trans,
            obj_quat,
            anchor,
            fade_frames=0,
        )

        self.assertEqual(span, (3, 8))
        pinch = 0.5 * (out_joints[3:9, 4] + out_joints[3:9, 8])
        np.testing.assert_allclose(
            pinch, np.broadcast_to(anchor, pinch.shape), atol=1e-12
        )
        np.testing.assert_allclose(
            out_vertices[3:9, 1] - out_vertices[3:9, 0],
            vertices[3:9, 1] - vertices[3:9, 0],
        )
        self.assertGreater(np.linalg.norm(offsets[8]), 0.0)

    def test_removes_tilt_and_preserves_projected_x_heading(self) -> None:
        rotations = Rotation.from_euler(
            "zyx",
            [
                [20.0, 30.0, -15.0],
                [80.0, -25.0, 10.0],
                [-140.0, 12.0, 32.0],
            ],
            degrees=True,
        )
        matrices = rotations.as_matrix()
        quat_xyzw = rotations.as_quat()
        quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]]

        corrected = _project_object_quaternions_upright(quat_wxyz)
        corrected_matrices = Rotation.from_quat(
            corrected[:, [1, 2, 3, 0]]
        ).as_matrix()

        np.testing.assert_allclose(
            corrected_matrices[:, :, 2],
            np.broadcast_to([0.0, 0.0, 1.0], (3, 3)),
            atol=1e-12,
        )
        expected_heading = np.arctan2(matrices[:, 1, 0], matrices[:, 0, 0])
        corrected_heading = np.arctan2(
            corrected_matrices[:, 1, 0], corrected_matrices[:, 0, 0]
        )
        np.testing.assert_allclose(corrected_heading, expected_heading, atol=1e-12)

    def test_sign_aligns_output_quaternions(self) -> None:
        yaw = np.deg2rad([170.0, 179.0, 181.0, 190.0])
        quat_xyzw = Rotation.from_euler("z", yaw[:, None]).as_quat()
        quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]]
        quat_wxyz[2:] *= -1.0

        corrected = _project_object_quaternions_upright(quat_wxyz)

        self.assertTrue(
            np.all(np.sum(corrected[1:] * corrected[:-1], axis=1) > 0.0)
        )

    def test_supports_mesh_local_positive_y_as_semantic_up(self) -> None:
        rotations = Rotation.from_euler(
            "xyz", [[25.0, -10.0, 70.0], [-30.0, 15.0, -40.0]], degrees=True
        )
        quat_xyzw = rotations.as_quat()
        quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]]

        corrected = _project_object_quaternions_upright(
            quat_wxyz, local_up_axis="+y"
        )
        corrected_rotations = Rotation.from_quat(
            corrected[:, [1, 2, 3, 0]]
        )

        np.testing.assert_allclose(
            corrected_rotations.apply([[0.0, 1.0, 0.0]] * 2),
            [[0.0, 0.0, 1.0]] * 2,
            atol=1e-12,
        )

    def test_rejects_unknown_semantic_up_axis(self) -> None:
        with self.assertRaisesRegex(ValueError, "object_up_axis"):
            _project_object_quaternions_upright(
                np.asarray([[1.0, 0.0, 0.0, 0.0]]),
                local_up_axis="up",
            )

    def test_supports_exact_semantic_up_vector(self) -> None:
        semantic_up = np.asarray([0.03, 0.31, -0.95])
        semantic_up /= np.linalg.norm(semantic_up)
        rotation = Rotation.from_euler("xyz", [[0.3, -0.4, 0.2]])
        xyzw = rotation.as_quat()
        wxyz = xyzw[:, [3, 0, 1, 2]]

        corrected = _project_object_quaternions_upright(
            wxyz,
            local_up_vector=semantic_up,
        )
        corrected_rotation = Rotation.from_quat(
            corrected[:, [1, 2, 3, 0]]
        )

        np.testing.assert_allclose(
            corrected_rotation.apply(semantic_up[None]),
            [[0.0, 0.0, 1.0]],
            atol=1e-12,
        )

    def test_rejects_bad_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape"):
            _project_object_quaternions_upright(np.zeros((3, 3)))

    def test_upright_support_is_centered_below_com(self) -> None:
        vertices = np.array(
            [
                [-1.0, -1.0, 0.0],
                [-1.0, 1.0, 0.0],
                [1.0, -1.0, 0.0],
                [1.0, 1.0, 0.0],
            ]
        )
        qpos = np.array([0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0])
        com_local = np.array([0.8, 0.0, 1.0])

        center, radius, bottom_z = _object_bottom_footprint(
            vertices,
            qpos,
            com_local=com_local,
            center_on_com=True,
        )

        np.testing.assert_allclose(center, [0.8, 0.0])
        self.assertAlmostEqual(radius, np.sqrt(1.8**2 + 1.0**2))
        self.assertEqual(bottom_z, 2.0)

    def test_frame_rotation_preserves_object_relative_points(self) -> None:
        object_positions = np.array([[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0]])
        object_before = Rotation.from_euler(
            "xyz", [[20.0, 30.0, 40.0], [-15.0, 25.0, 80.0]], degrees=True
        )
        object_after = Rotation.from_euler(
            "z", np.deg2rad([[45.0], [90.0]])
        )
        correction = object_after * object_before.inv()
        local_points = np.array(
            [
                [[0.1, 0.2, 0.3], [-0.3, 0.1, 0.2]],
                [[0.2, -0.2, 0.4], [0.0, 0.5, -0.1]],
            ]
        )
        world_points = (
            np.einsum("nij,nkj->nki", object_before.as_matrix(), local_points)
            + object_positions[:, None, :]
        )

        corrected_points = _rotate_points_per_frame(
            world_points, object_positions, correction
        )
        recovered_local = np.einsum(
            "nij,nkj->nki",
            object_after.inv().as_matrix(),
            corrected_points - object_positions[:, None, :],
        )

        np.testing.assert_allclose(recovered_local, local_points, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
