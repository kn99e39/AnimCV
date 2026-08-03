"""Safe 3D direction bake from R3/R5 constraint targets.

This is the bridge to Blender's FK channels, not a claim of rig-space analytic
IK: it holds an entire limb whenever R5 rejects its target and uses R3's
stable bend plane as a prerequisite for the 3D bone direction rotation.
"""

from __future__ import annotations

import math
from typing import Any

from retarget.axis_utils import quaternion_from_axis_angle, quaternion_from_vectors
from retarget.solver import AnimationClip, AnimationTrack, BoneTransformSample
from rig.bone_mapping import BoneMappingProfile
from rig.rig_profile import RigProfile


_IDENTITY = (0.0, 0.0, 0.0, 1.0)
_PAIR_TO_CHAIN = {
    ("left_hip", "left_knee"): "left_leg", ("left_knee", "left_ankle"): "left_leg",
    ("right_hip", "right_knee"): "right_leg", ("right_knee", "right_ankle"): "right_leg",
    ("left_shoulder", "left_elbow"): "left_arm", ("left_elbow", "left_wrist"): "left_arm",
    ("right_shoulder", "right_elbow"): "right_arm", ("right_elbow", "right_wrist"): "right_arm",
}


def solve_constraint_target_animation(
    root_motion, constraint_targets: dict[str, Any], rig: RigProfile, mapping: BoneMappingProfile
) -> AnimationClip:
    frames = {frame["frame_index"]: frame for frame in constraint_targets["frames"]}
    tracks: dict[str, AnimationTrack] = {}
    for entry in mapping.entries:
        if entry.mapping_mode != "direction" or len(entry.source_names) != 2 or entry.target_bone not in rig.bones:
            continue
        pair = tuple(entry.source_names)
        chain = _PAIR_TO_CHAIN.get(pair)
        if chain is None:
            continue
        reference = None
        last_rotation = _IDENTITY
        samples = []
        for root_frame in root_motion.frames:
            target_frame = frames.get(root_frame.frame_index)
            if target_frame is None:
                raise ValueError(f"missing constraint target frame {root_frame.frame_index}")
            chain_target = target_frame["chains"][chain]
            a, b = (root_frame.character_points[name] for name in pair)
            direction = _unit(_subtract(b, a))
            if not chain_target["enabled"] or direction is None:
                samples.append(BoneTransformSample(root_frame.frame_index, entry.target_bone, None, last_rotation, None, 0.0))
                continue
            if reference is None:
                reference = direction
            rotation = quaternion_from_vectors(reference, direction)
            last_rotation = rotation
            samples.append(BoneTransformSample(root_frame.frame_index, entry.target_bone, None, rotation, None, 1.0))
        tracks[entry.target_bone] = AnimationTrack(entry.target_bone, samples)
    if rig.root_bone in rig.bones and root_motion.frames:
        reference_yaw = root_motion.frames[0].root_yaw_radians
        tracks[rig.root_bone] = AnimationTrack(rig.root_bone, [
            BoneTransformSample(frame.frame_index, rig.root_bone, None,
                                quaternion_from_axis_angle((0.0, 0.0, 1.0), frame.root_yaw_radians - reference_yaw),
                                None, frame.confidence)
            for frame in root_motion.frames
        ])
    indices = [frame.frame_index for frame in root_motion.frames]
    return AnimationClip("Constraint_Target_Motion", root_motion.source_fps, tracks, min(indices), max(indices))


def _subtract(a, b): return tuple(x - y for x, y in zip(a, b))
def _unit(vector):
    length = math.sqrt(sum(value * value for value in vector))
    return tuple(value / length for value in vector) if length > 1e-6 else None
