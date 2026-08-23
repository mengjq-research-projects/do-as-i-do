from types import SimpleNamespace

import numpy as np

from replay_isaac import (
    IDLE_ARM_STANDBY_QPOS,
    compute_replay_camera_pose,
    idle_standby_state,
    prepare_record_directory,
    robot_sides,
    should_loop_replay,
)


def test_compute_replay_camera_pose_frames_valid_workspace():
    package = SimpleNamespace(
        trajectory=SimpleNamespace(
            valid_mask=np.asarray([False, True, True]),
            hand_root_position=np.asarray(
                [[100.0, 100.0, 100.0], [-0.2, 0.0, 0.1], [-0.1, 0.2, 0.2]]
            ),
            object_position=np.asarray(
                [[100.0, 100.0, 100.0], [0.0, 0.0, 0.1], [0.1, 0.1, 0.2]]
            ),
        )
    )

    eye, target = compute_replay_camera_pose(package)

    np.testing.assert_allclose(target, [-0.05, 0.1, 0.15])
    assert eye[0] > target[0]
    assert eye[1] < target[1]
    assert eye[2] > target[2]


def test_prepare_record_directory_removes_only_recorder_artifacts(tmp_path):
    record_dir = tmp_path / "render"
    record_dir.mkdir()
    (record_dir / "frame_000000.png").write_bytes(b"old frame")
    (record_dir / "replay.mp4").write_bytes(b"old video")
    keep = record_dir / "notes.txt"
    keep.write_text("keep", encoding="utf-8")

    prepare_record_directory(record_dir)

    assert not (record_dir / "frame_000000.png").exists()
    assert not (record_dir / "replay.mp4").exists()
    assert keep.read_text(encoding="utf-8") == "keep"


def test_gui_replay_loops_by_default():
    args = SimpleNamespace(
        loop=False,
        once=False,
        headless=False,
        record_dir=None,
    )

    assert should_loop_replay(args)


def test_batch_recording_and_once_modes_do_not_loop():
    base = {
        "loop": False,
        "once": False,
        "headless": True,
        "record_dir": None,
    }
    assert not should_loop_replay(SimpleNamespace(**base))
    assert not should_loop_replay(
        SimpleNamespace(**{**base, "headless": False, "record_dir": "render"})
    )
    assert not should_loop_replay(
        SimpleNamespace(**{**base, "headless": False, "once": True})
    )


def test_robot_sides_holds_the_side_without_a_trajectory():
    package = SimpleNamespace(
        is_robot_scene=True,
        manifest={"robot": {"embodiment": "right"}},
    )

    assert robot_sides(package) == ("right", "left")
    assert robot_sides(SimpleNamespace(is_robot_scene=False)) == (None, None)


def test_idle_standby_state_folds_arm_and_preserves_hand_joints():
    arm_names = [
        "left_shoulder_pan_joint",
        "left_shoulder_lift_joint",
        "left_elbow_joint",
        "left_wrist_1_joint",
        "left_wrist_2_joint",
        "left_wrist_3_joint",
    ]
    dof_names = ["left_hand_joint_0", *arm_names, "left_hand_joint_1"]
    reset_state = np.arange(len(dof_names), dtype=np.float32)

    standby = idle_standby_state(dof_names, reset_state, "left", np)

    np.testing.assert_allclose(standby[1:7], IDLE_ARM_STANDBY_QPOS)
    assert standby[0] == reset_state[0]
    assert standby[-1] == reset_state[-1]
    np.testing.assert_array_equal(reset_state, np.arange(8, dtype=np.float32))
