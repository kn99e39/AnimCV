from pose.constraint_target_builder import build_constraint_targets
from pose.root_motion import RootMotionFrame, RootMotionSequence


def test_constraint_targets_emit_pole_vectors_and_disable_unsafe_limb():
    points = {
        "left_hip": (-1, 0, 0), "left_knee": (-1, 1, -1), "left_ankle": (-1, 0, -2),
        "right_hip": (1, 0, 0), "right_knee": (1, 1, -1), "right_ankle": (1, 0, -2),
        "left_shoulder": (-1, 0, 2), "left_elbow": (-1, 1, 1), "left_wrist": (-1, 0, 0),
        "right_shoulder": (1, 0, 2), "right_elbow": (1, 1, 1), "right_wrist": (1, 0, 0),
    }
    root = RootMotionSequence(frames=[RootMotionFrame(0, 0, 0, (0, 1, 0), (1, 0, 0), 1, None, points)])
    uncertainty = {"limb_unsafe_rate": {"left_thigh": 0, "left_calf": 0, "right_thigh": 0, "right_calf": 0,
                                          "left_upper_arm": 0, "left_forearm": .3, "right_upper_arm": 0, "right_forearm": 0},
                   "frames": [{"frame_index": 0, "joints": {name: {"unsafe": False} for name in points}}]}
    result = build_constraint_targets(root, uncertainty)
    assert result["frames"][0]["chains"]["left_leg"]["pole_vector"] is not None
    left_arm = result["frames"][0]["chains"]["left_arm"]
    assert not left_arm["enabled"]
    assert left_arm["disabled_reason"] == "limb_unsafe_rate"
