"""Build rig-independent 3D IK targets from prepared pose and uncertainty."""

from __future__ import annotations

import math
from typing import Any

from pose.root_motion import RootMotionSequence


_CHAINS = {
    "left_leg": ("left_hip", "left_knee", "left_ankle", ("left_thigh", "left_calf")),
    "right_leg": ("right_hip", "right_knee", "right_ankle", ("right_thigh", "right_calf")),
    "left_arm": ("left_shoulder", "left_elbow", "left_wrist", ("left_upper_arm", "left_forearm")),
    "right_arm": ("right_shoulder", "right_elbow", "right_wrist", ("right_upper_arm", "right_forearm")),
}


def build_constraint_targets(
    root_motion: RootMotionSequence,
    uncertainty: dict[str, Any],
    max_limb_unsafe_rate: float = 0.20,
) -> dict[str, Any]:
    """Create end-effector/pole targets, refusing unreliable limbs.

    Coordinates are character-root-relative metres.  The resulting pole
    vectors deliberately stay rig-independent; a rig adapter later maps them
    through the selected bone's actual hinge axis/rest basis.
    """
    uncertainty_frames = {frame["frame_index"]: frame["joints"] for frame in uncertainty["frames"]}
    limb_rates = uncertainty["limb_unsafe_rate"]
    records = []
    for frame in root_motion.frames:
        joints = uncertainty_frames.get(frame.frame_index)
        if joints is None:
            raise ValueError(f"missing uncertainty frame {frame.frame_index}")
        chains = {}
        for name, (root, mid, end, limbs) in _CHAINS.items():
            root_point, mid_point, end_point = (frame.character_points[joint] for joint in (root, mid, end))
            upper, lower = _subtract(mid_point, root_point), _subtract(end_point, mid_point)
            pole = _unit(_cross(upper, lower))
            bend = _angle_degrees(upper, lower)
            limb_rate = max(limb_rates[limb] for limb in limbs)
            joint_unsafe = any(joints[joint]["unsafe"] for joint in (root, mid, end))
            enabled = limb_rate <= max_limb_unsafe_rate and not joint_unsafe and pole is not None
            chains[name] = {
                "root": list(root_point), "mid": list(mid_point), "end_effector": list(end_point),
                "pole_vector": list(pole) if pole is not None else None,
                "bend_degrees": bend,
                "limb_unsafe_rate": limb_rate,
                "enabled": enabled,
                "disabled_reason": _disabled_reason(limb_rate, max_limb_unsafe_rate, joint_unsafe, pole),
            }
        records.append({"frame_index": frame.frame_index, "timestamp": frame.timestamp, "chains": chains})
    return {
        "coordinate_frame": "character_root_relative",
        "units": "metres",
        "method": "R3 bend-stabilized end-effector and pole targets gated by R5 quality score",
        "max_limb_unsafe_rate": max_limb_unsafe_rate,
        "frame_count": len(records),
        "chain_enabled_rate": {
            chain: sum(frame["chains"][chain]["enabled"] for frame in records) / len(records)
            for chain in _CHAINS
        },
        "frames": records,
    }


def _disabled_reason(rate, threshold, joint_unsafe, pole):
    if rate > threshold:
        return "limb_unsafe_rate"
    if joint_unsafe:
        return "unsafe_joint_sample"
    if pole is None:
        return "ambiguous_bend_plane"
    return None


def _subtract(a, b): return tuple(x - y for x, y in zip(a, b))
def _cross(a, b): return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])
def _length(a): return math.sqrt(sum(value * value for value in a))
def _unit(a):
    length = _length(a)
    return tuple(value / length for value in a) if length > 1e-6 else None
def _angle_degrees(a, b):
    denominator = _length(a) * _length(b)
    return math.degrees(math.acos(max(-1.0, min(1.0, sum(x * y for x, y in zip(a, b)) / denominator)))) if denominator else 0.0
