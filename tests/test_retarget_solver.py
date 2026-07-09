import math

import pytest

from motion.motion_graph import MotionFrame, MotionGraph, MotionPoint
from retarget.axis_utils import quaternion_from_axis_angle
from retarget.solver import (
    AnimationClip,
    AnimationTrack,
    BoneTransformSample,
    RetargetSolver,
    load_animation_clip,
    save_animation_clip,
)
from rig.bone_mapping import BoneMappingEntry, BoneMappingProfile
from rig.rig_profile import BoneInfo, RigProfile


def _sample_clip() -> AnimationClip:
    sample = BoneTransformSample(
        frame_index=0,
        bone_name="upper_arm.L",
        location=None,
        rotation=(0.0, 0.0, 0.0, 1.0),
        scale=None,
        confidence=0.9,
    )
    track = AnimationTrack(bone_name="upper_arm.L", samples=[sample])
    return AnimationClip(
        name="Generated_Motion",
        fps=24.0,
        tracks={"upper_arm.L": track},
        frame_start=0,
        frame_end=0,
    )


def test_animation_clip_roundtrip():
    clip = _sample_clip()

    restored = AnimationClip.from_dict(clip.to_dict())

    assert restored == clip


def test_animation_clip_json_roundtrip(tmp_path):
    clip = _sample_clip()
    path = tmp_path / "animation.json"

    save_animation_clip(clip, path)
    restored = load_animation_clip(path)

    assert restored == clip


def _point(name, x, y, frame_index):
    return MotionPoint(
        semantic_name=name,
        frame_index=frame_index,
        position_2d=(x, y),
        position_3d=None,
        confidence=0.9,
        visible=True,
    )


def _swinging_arm_motion_graph() -> MotionGraph:
    # left_shoulder fixed; left_elbow sweeps 0deg -> 90deg over 2 frames.
    positions = [(1.0, 0.0), (0.0, 1.0)]
    frames = []
    for i, (ex, ey) in enumerate(positions):
        frames.append(
            MotionFrame(
                frame_index=i,
                timestamp=i / 24.0,
                points={
                    "left_shoulder": _point("left_shoulder", 0.0, 0.0, i),
                    "left_elbow": _point("left_elbow", ex, ey, i),
                },
            )
        )
    return MotionGraph(frames=frames, tracks={}, fps=24.0)


def _rig_with_bone(name: str) -> RigProfile:
    return RigProfile(
        rig_id="character_01",
        source_path="character.fbx",
        bones={name: BoneInfo(name=name, parent=None)},
    )


def test_solve_moves_mapped_chain_according_to_pose_direction():
    motion_graph = _swinging_arm_motion_graph()
    rig_profile = _rig_with_bone("upper_arm.L")
    mapping = BoneMappingProfile(
        rig_id="character_01",
        entries=[
            BoneMappingEntry(
                target_bone="upper_arm.L",
                source_type="landmark",
                source_names=["left_shoulder", "left_elbow"],
                mapping_mode="direction",
            )
        ],
    )

    clip = RetargetSolver().solve(motion_graph, rig_profile, mapping)

    assert clip.fps == 24.0
    assert clip.frame_start == 0 and clip.frame_end == 1
    track = clip.tracks["upper_arm.L"]
    assert len(track.samples) == 2
    assert track.samples[0].rotation == pytest.approx((0.0, 0.0, 0.0, 1.0))
    assert track.samples[1].rotation == pytest.approx(
        quaternion_from_axis_angle((0.0, 0.0, 1.0), math.pi / 2)
    )
    # The chain actually moved -- not left at rest -- per Milestone 5's
    # acceptance criteria.
    assert track.samples[1].rotation != track.samples[0].rotation


def test_solve_skips_bones_not_present_in_rig():
    motion_graph = _swinging_arm_motion_graph()
    rig_profile = _rig_with_bone("upper_arm.L")  # note: no "forearm.L"
    mapping = BoneMappingProfile(
        rig_id="character_01",
        entries=[
            BoneMappingEntry(
                target_bone="forearm.L",
                source_type="landmark",
                source_names=["left_shoulder", "left_elbow"],
                mapping_mode="direction",
            )
        ],
    )

    clip = RetargetSolver().solve(motion_graph, rig_profile, mapping)

    assert clip.tracks == {}


def test_solve_skips_unsupported_mapping_mode():
    motion_graph = _swinging_arm_motion_graph()
    rig_profile = _rig_with_bone("upper_arm.L")
    mapping = BoneMappingProfile(
        rig_id="character_01",
        entries=[
            BoneMappingEntry(
                target_bone="upper_arm.L",
                source_type="landmark",
                source_names=["left_shoulder"],
                mapping_mode="some_future_mode",
            )
        ],
    )

    clip = RetargetSolver().solve(motion_graph, rig_profile, mapping)

    assert clip.tracks == {}


def test_solve_handles_landmark_anchor_mapping():
    motion_graph = _swinging_arm_motion_graph()
    rig_profile = _rig_with_bone("head")
    mapping = BoneMappingProfile(
        rig_id="character_01",
        entries=[
            BoneMappingEntry(
                target_bone="head",
                source_type="landmark",
                source_names=["left_shoulder"],
                mapping_mode="landmark",
            )
        ],
    )

    clip = RetargetSolver().solve(motion_graph, rig_profile, mapping)

    assert "head" in clip.tracks
    assert clip.tracks["head"].samples[0].location == pytest.approx((0.0, 0.0, 0.0))
