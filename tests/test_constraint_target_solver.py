from retarget.constraint_target_solver import solve_constraint_target_animation
from rig.bone_mapping import BoneMappingEntry, BoneMappingProfile
from rig.rig_profile import BoneInfo, RigProfile
from pose.root_motion import RootMotionFrame, RootMotionSequence


def test_constraint_target_solver_holds_rejected_chain_and_rotates_valid_chain():
    root = RootMotionSequence(frames=[
        RootMotionFrame(0, 0, 0, (0, 1, 0), (1, 0, 0), 1, None, {"left_shoulder": (0, 0, 0), "left_elbow": (1, 0, 0)}),
        RootMotionFrame(1, .02, 0, (0, 1, 0), (1, 0, 0), 1, None, {"left_shoulder": (0, 0, 0), "left_elbow": (0, 1, 0)}),
    ], source_fps=50)
    targets = {"frames": [{"frame_index": 0, "chains": {"left_arm": {"enabled": True}}}, {"frame_index": 1, "chains": {"left_arm": {"enabled": False}}}]}
    rig = RigProfile("rig", "rig.fbx", {"upperarm_l": BoneInfo("upperarm_l", None)})
    mapping = BoneMappingProfile("rig", [BoneMappingEntry("upperarm_l", "landmark", ["left_shoulder", "left_elbow"], "direction")])
    clip = solve_constraint_target_animation(root, targets, rig, mapping)
    samples = clip.tracks["upperarm_l"].samples
    assert samples[0].rotation == (0.0, 0.0, 0.0, 1.0)
    assert samples[1].confidence == 0.0
    assert samples[1].rotation == samples[0].rotation
