import unittest

import numpy as np

from isaac_export.coordinate import (
    euler_xyz_to_quaternion,
    normalize_quaternions,
    quaternion_to_euler_xyz,
)


class CoordinateTest(unittest.TestCase):
    def test_xyz_euler_chain_uses_wxyz(self) -> None:
        angles = np.array(
            [
                [0.0, 0.0, 0.0],
                [np.pi / 2.0, 0.0, 0.0],
                [0.0, np.pi / 2.0, 0.0],
                [0.0, 0.0, np.pi / 2.0],
            ]
        )
        root_two = np.sqrt(2.0) / 2.0
        expected = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [root_two, root_two, 0.0, 0.0],
                [root_two, 0.0, root_two, 0.0],
                [root_two, 0.0, 0.0, root_two],
            ]
        )

        np.testing.assert_allclose(
            euler_xyz_to_quaternion(angles), expected, atol=1e-12
        )

    def test_quaternion_signs_are_continuous(self) -> None:
        quaternions = np.array(
            [
                [-2.0, 0.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ]
        )

        normalized = normalize_quaternions(quaternions)

        np.testing.assert_allclose(normalized, np.ones((3, 1)) * [1.0, 0.0, 0.0, 0.0])

    def test_zero_quaternion_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "zero-length"):
            normalize_quaternions(np.zeros((1, 4)))

    def test_xyz_euler_round_trip(self) -> None:
        angles = np.array(
            [[0.2, -0.3, 0.4], [-1.1, 0.7, 2.2], [2.5, -1.0, -2.1]]
        )
        recovered = quaternion_to_euler_xyz(euler_xyz_to_quaternion(angles))
        np.testing.assert_allclose(recovered, angles, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
