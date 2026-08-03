"""Traceable quality/uncertainty sidecar for prepared 3D pose targets.

This is a quality score, not a statistical posterior variance.  Each value is
decomposed so a future constraint retargeter can reject unsafe limb targets
without pretending a monocular lifter supplies calibrated depth variance.
"""

from __future__ import annotations

import math
import statistics
from typing import Any

from pose.pose_lifter import LiftedPoseSequence
from pose.root_motion import RootMotionSequence


_LIMBS = {
    "left_thigh": ("left_hip", "left_knee"), "left_calf": ("left_knee", "left_ankle"),
    "right_thigh": ("right_hip", "right_knee"), "right_calf": ("right_knee", "right_ankle"),
    "left_upper_arm": ("left_shoulder", "left_elbow"), "left_forearm": ("left_elbow", "left_wrist"),
    "right_upper_arm": ("right_shoulder", "right_elbow"), "right_forearm": ("right_elbow", "right_wrist"),
}


def audit_pose_uncertainty(
    raw: LiftedPoseSequence,
    prepared: LiftedPoseSequence,
    root_motion: RootMotionSequence,
    unsafe_threshold: float = 0.55,
    correction_reference_metres: float = 0.10,
    acceleration_reference_metres: float = 0.08,
    min_bend_degrees: float = 12.0,
) -> dict[str, Any]:
    if not (len(raw.frames) == len(prepared.frames) == len(root_motion.frames)):
        raise ValueError("raw pose, prepared pose, and root motion frame counts must match")
    if not 0.0 < unsafe_threshold <= 1.0:
        raise ValueError("unsafe threshold must be in (0, 1]")
    records = []
    component_values: dict[str, list[float]] = {key: [] for key in _component_keys()}
    for index, (raw_frame, prepared_frame, root_frame) in enumerate(zip(raw.frames, prepared.frames, root_motion.frames)):
        if raw_frame.frame_index != prepared_frame.frame_index or prepared_frame.frame_index != root_frame.frame_index:
            raise ValueError("frame indices must match")
        ambiguities = _bend_ambiguities(root_frame, min_bend_degrees)
        yaw_disagreement = min(1.0, root_frame.yaw_agreement_degrees / 90.0)
        joints = {}
        for name, point in prepared_frame.points.items():
            raw_point = raw_frame.points[name]
            components = {
                "source_confidence": 1.0 - point.confidence,
                "invalid_observation": float(not point.observation_valid),
                "temporal_acceleration": _temporal_acceleration(raw, index, name, acceleration_reference_metres),
                "kinematic_correction": min(1.0, _distance(point.position, raw_point.position) / correction_reference_metres),
                "bend_ambiguity": ambiguities.get(name, 0.0),
                "torso_axis_disagreement": yaw_disagreement,
            }
            score = 1.0 if components["invalid_observation"] else min(
                1.0,
                0.25 * components["source_confidence"]
                + 0.20 * components["temporal_acceleration"]
                + 0.25 * components["kinematic_correction"]
                + 0.20 * components["bend_ambiguity"]
                + 0.10 * components["torso_axis_disagreement"],
            )
            joints[name] = {"quality_score": score, "unsafe": score >= unsafe_threshold, "components": components}
            for key, value in components.items():
                component_values[key].append(value)
        records.append({"frame_index": prepared_frame.frame_index, "timestamp": prepared_frame.timestamp, "joints": joints})
    limb_rates = {
        limb: _unsafe_rate(records, joints, unsafe_threshold)
        for limb, joints in _LIMBS.items()
    }
    all_scores = [joint["quality_score"] for frame in records for joint in frame["joints"].values()]
    return {
        "method": "traceable monocular 3D quality score; not posterior variance",
        "frame_count": len(records),
        "point_count": len(all_scores),
        "unsafe_threshold": unsafe_threshold,
        "unsafe_point_rate": sum(score >= unsafe_threshold for score in all_scores) / len(all_scores),
        "median_quality_score": statistics.median(all_scores),
        "p95_quality_score": sorted(all_scores)[round(0.95 * (len(all_scores) - 1))],
        "component_mean": {key: statistics.fmean(values) for key, values in component_values.items()},
        "limb_unsafe_rate": limb_rates,
        "all_points_traceable": all(set(joint["components"]) == set(_component_keys())
                                    for frame in records for joint in frame["joints"].values()),
        "retarget_gate": {"can_reject_limb_when_unsafe_rate_exceeds": 0.20},
        "frames": records,
    }


def _component_keys():
    return ("source_confidence", "invalid_observation", "temporal_acceleration", "kinematic_correction", "bend_ambiguity", "torso_axis_disagreement")


def _temporal_acceleration(sequence, index, name, reference):
    if index == 0 or index == len(sequence.frames) - 1:
        return 0.0
    before = sequence.frames[index - 1].points[name].position
    current = sequence.frames[index].points[name].position
    after = sequence.frames[index + 1].points[name].position
    acceleration = _distance(tuple(a - 2 * b + c for a, b, c in zip(before, current, after)), (0.0, 0.0, 0.0))
    return min(1.0, acceleration / reference)


def _bend_ambiguities(root_frame, minimum):
    result = {}
    for root, mid, end in (("left_hip", "left_knee", "left_ankle"), ("right_hip", "right_knee", "right_ankle"),
                           ("left_shoulder", "left_elbow", "left_wrist"), ("right_shoulder", "right_elbow", "right_wrist")):
        points = root_frame.character_points
        upper = _subtract(points[mid], points[root])
        lower = _subtract(points[end], points[mid])
        angle = math.degrees(math.acos(max(-1.0, min(1.0, _dot(upper, lower) / (_length(upper) * _length(lower))))))
        ambiguity = max(0.0, min(1.0, (minimum - angle) / minimum))
        result[mid] = ambiguity
        result[end] = ambiguity
    return result


def _unsafe_rate(records, joints, threshold):
    samples = [frame["joints"][joint]["quality_score"] for frame in records for joint in joints]
    return sum(score >= threshold for score in samples) / len(samples)


def _subtract(a, b): return tuple(x - y for x, y in zip(a, b))
def _dot(a, b): return sum(x * y for x, y in zip(a, b))
def _length(a): return math.sqrt(_dot(a, a))
def _distance(a, b): return _length(_subtract(a, b))
