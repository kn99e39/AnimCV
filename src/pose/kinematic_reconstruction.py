"""Subject-specific fixed-length reconstruction of temporally lifted poses."""

from __future__ import annotations

from dataclasses import replace
import math
import statistics

from pose.pose_lifter import LiftedPoseFrame, LiftedPosePoint, LiftedPoseSequence


_CHAINS = (
    ("left_hip", "left_knee", "left_ankle"),
    ("right_hip", "right_knee", "right_ankle"),
    ("left_shoulder", "left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow", "right_wrist"),
)


def reconstruct_kinematic_pose(
    lifted: LiftedPoseSequence, min_confidence: float = 0.3
) -> LiftedPoseSequence:
    """Make limb lengths constant while retaining each lifted chain direction.

    This is intentionally not an IK or pole-vector solver. It removes the
    scale jitter that makes a lifted skeleton unusable as a rig target; R3
    subsequently resolves the remaining mirror/bend-plane ambiguity.
    """
    lengths = _estimate_lengths(lifted, min_confidence)
    reconstructed: list[LiftedPoseFrame] = []
    previous_directions: dict[tuple[str, str], tuple[float, float, float]] = {}
    for frame in lifted.frames:
        points = dict(frame.points)
        for root, mid, end in _CHAINS:
            root_position = points[root].position
            raw_mid = points[mid].position
            upper_direction = _stable_direction(
                _subtract(raw_mid, root_position), previous_directions.get((root, mid))
            )
            previous_directions[(root, mid)] = upper_direction
            mid_position = _add(root_position, _scale(upper_direction, lengths[(root, mid)]))

            raw_end = points[end].position
            lower_direction = _stable_direction(
                _subtract(raw_end, raw_mid), previous_directions.get((mid, end))
            )
            previous_directions[(mid, end)] = lower_direction
            end_position = _add(mid_position, _scale(lower_direction, lengths[(mid, end)]))
            points[mid] = replace(points[mid], position=mid_position)
            points[end] = replace(points[end], position=end_position)
        reconstructed.append(LiftedPoseFrame(frame.frame_index, frame.timestamp, points))
    return LiftedPoseSequence(
        frames=reconstructed,
        source_fps=lifted.source_fps,
        coordinate_frame=lifted.coordinate_frame,
        units=lifted.units,
        backend=f"{lifted.backend}+fixed_length_kinematic",
    )


def _estimate_lengths(lifted: LiftedPoseSequence, min_confidence: float) -> dict[tuple[str, str], float]:
    lengths = {}
    for root, mid, end in _CHAINS:
        for pair in ((root, mid), (mid, end)):
            values = [
                _length(frame.points[pair[0]].position, frame.points[pair[1]].position)
                for frame in lifted.frames
                if frame.points[pair[0]].confidence >= min_confidence
                and frame.points[pair[1]].confidence >= min_confidence
                and frame.points[pair[0]].observation_valid
                and frame.points[pair[1]].observation_valid
            ]
            if not values:
                raise ValueError(f"no reliable observations for bone length {pair[0]}->{pair[1]}")
            lengths[pair] = statistics.median(values)
    return lengths


def _stable_direction(raw, previous):
    length = _length(raw, (0.0, 0.0, 0.0))
    if length > 1e-6:
        return _scale(raw, 1.0 / length)
    if previous is None:
        raise ValueError("coincident lifted joints without a previous direction")
    return previous


def _subtract(a, b):
    return tuple(x - y for x, y in zip(a, b))


def _add(a, b):
    return tuple(x + y for x, y in zip(a, b))


def _scale(vector, factor):
    return tuple(value * factor for value in vector)


def _length(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
