"""Numerical acceptance audit for the 3D lifting and root-orientation stages."""

from __future__ import annotations

from dataclasses import dataclass
import math
import statistics
from typing import Any

from pose.pose_lifter import LiftedPoseSequence
from pose.root_motion import RootMotionSequence


_LIMB_SEGMENTS = (
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
)


@dataclass(frozen=True)
class Pose3DQualityConfig:
    max_limb_length_cv: float = 0.10
    max_yaw_step_degrees: float = 15.0
    max_low_yaw_confidence_rate: float = 0.10
    max_yaw_held_rate: float = 0.10
    max_knee_bend_direction_flips: int = 0
    min_bend_degrees: float = 12.0
    min_frames: int = 30


def audit_3d_pose(
    lifted: LiftedPoseSequence,
    root_motion: RootMotionSequence,
    config: Pose3DQualityConfig | None = None,
) -> dict[str, Any]:
    config = config or Pose3DQualityConfig()
    failures: list[str] = []
    if lifted.coordinate_frame != "camera_root_relative" or lifted.units != "metres":
        failures.append("lifted pose does not declare the required camera-root-relative metre frame")
    if root_motion.coordinate_frame != "character_root_relative":
        failures.append("root motion does not declare character-root-relative coordinates")
    if len(lifted.frames) != len(root_motion.frames):
        failures.append("lifted pose and root motion frame counts differ")
    if len(lifted.frames) < config.min_frames:
        failures.append(f"need at least {config.min_frames} frames")

    length_cv = _limb_length_cv(lifted)
    unstable = {name: value for name, value in length_cv.items() if value > config.max_limb_length_cv}
    if unstable:
        failures.append(f"limb length CV exceeds {config.max_limb_length_cv:.0%}: {unstable}")

    max_yaw_step = _max_yaw_step_degrees(root_motion)
    if max_yaw_step > config.max_yaw_step_degrees:
        failures.append(f"root yaw step {max_yaw_step:.2f} exceeds {config.max_yaw_step_degrees:.2f} degrees")
    low_confidence_rate = _low_yaw_confidence_rate(root_motion)
    if low_confidence_rate > config.max_low_yaw_confidence_rate:
        failures.append("too many low-confidence root-yaw frames")
    yaw_held_rate = _yaw_held_rate(root_motion)
    if yaw_held_rate > config.max_yaw_held_rate:
        failures.append("too many held root-yaw frames")

    bend_flips = {
        f"{side}_{joint}": _bend_direction_flips(root_motion, side, root, joint, end, config.min_bend_degrees)
        for side, root, joint, end in (
            ("left", "hip", "knee", "ankle"), ("right", "hip", "knee", "ankle"),
            ("left", "shoulder", "elbow", "wrist"), ("right", "shoulder", "elbow", "wrist"),
        )
    }
    knee_flips = {side: bend_flips[f"{side}_knee"] for side in ("left", "right")}
    if any(value > config.max_knee_bend_direction_flips for value in bend_flips.values()):
        failures.append(f"limb bend direction flips: {bend_flips}")

    observation = _observation_integrity(lifted)
    if observation["silently_accepted_invalid_count"]:
        failures.append("invalid 2D observations were emitted as accepted 3D targets")

    return {
        "passed": not failures,
        "failures": failures,
        "frame_count": len(lifted.frames),
        "coordinate_contract": {
            "lifted_frame": lifted.coordinate_frame,
            "units": lifted.units,
            "root_frame": root_motion.coordinate_frame,
            "global_translation_available": root_motion.translation_observable,
            "observation_confidence_threshold": lifted.observation_confidence_threshold,
            "root_observation_confidence_threshold": root_motion.observation_confidence_threshold,
        },
        "limb_length_cv": length_cv,
        "max_yaw_step_degrees": max_yaw_step,
        "low_yaw_confidence_rate": low_confidence_rate,
        "yaw_held_rate": yaw_held_rate,
        "knee_bend_direction_flips": knee_flips,
        "bend_direction_flips": bend_flips,
        "observation_integrity": observation,
        "thresholds": {
            "max_limb_length_cv": config.max_limb_length_cv,
            "max_yaw_step_degrees": config.max_yaw_step_degrees,
            "max_low_yaw_confidence_rate": config.max_low_yaw_confidence_rate,
            "max_yaw_held_rate": config.max_yaw_held_rate,
            "max_knee_bend_direction_flips": config.max_knee_bend_direction_flips,
            "min_bend_degrees": config.min_bend_degrees,
        },
    }


def _limb_length_cv(lifted: LiftedPoseSequence) -> dict[str, float]:
    results = {}
    for start, end in _LIMB_SEGMENTS:
        lengths = [_distance(frame.points[start].position, frame.points[end].position) for frame in lifted.frames]
        mean = statistics.fmean(lengths)
        results[f"{start}->{end}"] = statistics.pstdev(lengths) / mean if mean else float("inf")
    return results


def _max_yaw_step_degrees(root_motion: RootMotionSequence) -> float:
    yaws = [frame.root_yaw_radians for frame in root_motion.frames]
    return max((abs(b - a) * 180.0 / math.pi for a, b in zip(yaws, yaws[1:])), default=0.0)


def _low_yaw_confidence_rate(root_motion: RootMotionSequence) -> float:
    if not root_motion.frames:
        return 1.0
    return sum(frame.confidence < 0.5 for frame in root_motion.frames) / len(root_motion.frames)


def _yaw_held_rate(root_motion: RootMotionSequence) -> float:
    if not root_motion.frames:
        return 1.0
    return sum(frame.yaw_held for frame in root_motion.frames) / len(root_motion.frames)


def _bend_direction_flips(
    root_motion: RootMotionSequence,
    side: str,
    root: str,
    joint: str,
    end: str,
    min_bend_degrees: float = 12.0,
) -> int:
    signs: list[int] = []
    for frame in root_motion.frames:
        points = frame.character_points
        start, middle, finish = (points[f"{side}_{name}"] for name in (root, joint, end))
        upper = _unit(_subtract(middle, start))
        lower = _unit(_subtract(finish, middle))
        cosine = max(-1.0, min(1.0, _dot(upper, lower)))
        bend = math.degrees(math.acos(cosine))
        # character_points are already root-normalized, so their hinge axis is
        # the canonical character +X axis, not the camera-space lateral vector.
        signed = _dot(_cross(upper, lower), (1.0, 0.0, 0.0))
        if bend >= min_bend_degrees and abs(signed) > 1e-6:
            signs.append(1 if signed > 0 else -1)
    return sum(a != b for a, b in zip(signs, signs[1:]))


def _observation_integrity(lifted: LiftedPoseSequence) -> dict[str, int]:
    points = [point for frame in lifted.frames for point in frame.points.values()]
    invalid = [point for point in points if not point.observation_valid]
    return {
        "point_count": len(points),
        "invalid_observation_count": len(invalid),
        "interpolated_input_count": sum(point.interpolated_input for point in points),
        "silently_accepted_invalid_count": sum(point.confidence >= 0.3 for point in invalid),
    }


def _distance(a, b) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _subtract(a, b):
    return tuple(x - y for x, y in zip(a, b))


def _unit(vector):
    length = math.sqrt(sum(value * value for value in vector))
    if length == 0:
        raise ValueError("coincident joints cannot define a limb direction")
    return tuple(value / length for value in vector)


def _dot(a, b) -> float:
    return sum(x * y for x, y in zip(a, b))


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])
