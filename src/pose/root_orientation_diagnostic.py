"""Root Orientation diagnostic (section 5 of docs/51).

Independent of pose.root_motion's production hold/smoothing policy — this
module does not import it and does not modify it. It only measures what the
*raw* bilateral yaw observation looks like across a sequence, using the
shared canonical geometry owner (common.canonical_pose.YAW_PAIRS), so a later
decision about whether root_motion.py's max_yaw_step_degrees heuristic is
still the right owner for body heading can be made from evidence rather than
assumption.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from common.canonical_pose import YAW_PAIRS
from pose.pose_lifter import LiftedPoseSequence


@dataclass(frozen=True)
class YawObservation:
    frame_index: int
    yaw_radians: float | None  # None == UNKNOWN, no bilateral pair with defined geometry
    per_source_yaw_radians: dict[str, float] = field(default_factory=dict)
    disagreement_degrees: float | None = None
    # True only if every source that contributed to yaw_radians was fully
    # observation_valid. Deliberately NOT used to gate whether yaw is
    # computed (unlike root_motion.py's production weighting) — this
    # diagnostic needs yaw computed on unreliable frames too, so failure
    # rates can be measured conditioned on reliability instead of the
    # unreliable frames simply vanishing into UNKNOWN.
    reliable: bool = False


@dataclass(frozen=True)
class RootOrientationReport:
    sequence_id: str
    frame_count: int
    unknown_count: int
    flip_count: int  # raw |delta| > 90deg between consecutive KNOWN frames
    mean_abs_delta_degrees: float | None
    median_abs_delta_degrees: float | None
    mean_disagreement_degrees: float | None
    unreliable_frame_count: int
    flip_rate_among_unreliable: float | None
    flip_rate_among_reliable: float | None


def observe_yaw(sequence: LiftedPoseSequence) -> list[YawObservation]:
    observations = []
    labels = ("shoulders", "hips")
    for frame in sequence.frames:
        per_source: dict[str, float] = {}
        weighted: list[tuple[float, float, bool]] = []
        for label, (left_name, right_name) in zip(labels, YAW_PAIRS):
            left = frame.points.get(left_name)
            right = frame.points.get(right_name)
            if left is None or right is None:
                continue
            dx = right.position[0] - left.position[0]
            dy = right.position[1] - left.position[1]
            length = math.hypot(dx, dy)
            if length <= 1e-6:
                continue
            angle = math.atan2(dy, dx)
            per_source[label] = angle
            valid = left.observation_valid and right.observation_valid
            confidence = (left.confidence + right.confidence) / 2.0
            weight = length * confidence
            if weight > 1e-6:
                weighted.append((angle, weight, valid))
        if not weighted:
            observations.append(YawObservation(frame.frame_index, None, per_source, None, False))
            continue
        x = sum(math.cos(angle) * weight for angle, weight, _ in weighted)
        y = sum(math.sin(angle) * weight for angle, weight, _ in weighted)
        yaw = math.atan2(y, x) if math.hypot(x, y) > 1e-6 else None
        reliable = all(valid for _, _, valid in weighted)
        disagreement = None
        if len(per_source) >= 2:
            values = list(per_source.values())
            disagreement = max(
                abs(_angle_delta(a, b)) * 180.0 / math.pi for a in values for b in values
            )
        observations.append(YawObservation(frame.frame_index, yaw, per_source, disagreement, reliable))
    return observations


def summarize(sequence_id: str, observations: list[YawObservation]) -> RootOrientationReport:
    known = [o for o in observations if o.yaw_radians is not None]
    unknown_count = len(observations) - len(known)
    deltas = []
    flip_frame_indices = set()
    for previous, current in zip(known, known[1:]):
        delta = abs(_angle_delta(current.yaw_radians, previous.yaw_radians)) * 180.0 / math.pi
        deltas.append(delta)
        if delta > 90.0:
            flip_frame_indices.add(current.frame_index)

    def _flip_rate(subset: list[YawObservation]) -> float | None:
        indices = {o.frame_index for o in subset}
        total = sum(1 for _, current in zip(known, known[1:]) if current.frame_index in indices)
        if total == 0:
            return None
        hits = sum(1 for index in indices if index in flip_frame_indices)
        return hits / total

    disagreements = [o.disagreement_degrees for o in known if o.disagreement_degrees is not None]
    unreliable = [o for o in known if not o.reliable]
    reliable_ok = [o for o in known if o.reliable]
    sorted_deltas = sorted(deltas)
    return RootOrientationReport(
        sequence_id=sequence_id,
        frame_count=len(observations),
        unknown_count=unknown_count,
        flip_count=len(flip_frame_indices),
        mean_abs_delta_degrees=(sum(deltas) / len(deltas)) if deltas else None,
        median_abs_delta_degrees=(sorted_deltas[len(sorted_deltas) // 2] if sorted_deltas else None),
        mean_disagreement_degrees=(sum(disagreements) / len(disagreements)) if disagreements else None,
        unreliable_frame_count=len(unreliable),
        flip_rate_among_unreliable=_flip_rate(unreliable),
        flip_rate_among_reliable=_flip_rate(reliable_ok),
    )


def _angle_delta(a: float, b: float) -> float:
    return (a - b + math.pi) % (2 * math.pi) - math.pi
