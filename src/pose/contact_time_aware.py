"""FramePose contact-like motion on elapsed physical time (separate from docs/51)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pose.contact import ContactState, _finite_diff_positions, _percentile
from pose.pose_lifter import LiftedPoseSequence

RULE_VERSION = "framepose_elapsed_time_contact_v1"


@dataclass(frozen=True)
class TimeAwareThresholds:
    contact_speed_m_s: float
    moving_speed_m_s: float
    height_std_threshold_m: float
    height_radius_seconds: float
    min_run_span_seconds: float

    def __post_init__(self) -> None:
        if not (0 <= self.contact_speed_m_s < self.moving_speed_m_s):
            raise ValueError("invalid speed thresholds")
        if self.height_std_threshold_m < 0 or self.height_radius_seconds <= 0 or self.min_run_span_seconds <= 0:
            raise ValueError("invalid height or run duration")


def elapsed_times(sequence: LiftedPoseSequence) -> list[float]:
    """Use complete timestamps, or frame_index/fps when timestamps are absent."""
    raw = [frame.timestamp for frame in sequence.frames]
    if all(value is not None for value in raw):
        times = [float(value) for value in raw]
    elif all(value is None for value in raw):
        if not sequence.source_fps or sequence.source_fps <= 0:
            raise ValueError("positive source_fps required without timestamps")
        times = [frame.frame_index / sequence.source_fps for frame in sequence.frames]
    else:
        raise ValueError("partially missing timestamps")
    if any(not math.isfinite(value) for value in times) or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("timestamps must be finite and strictly increasing")
    return times


def kinematics(sequence: LiftedPoseSequence, foot: str, height_radius_seconds: float) -> tuple[list[float | None], list[float | None]]:
    times = elapsed_times(sequence)
    positions = _finite_diff_positions(sequence, foot)
    speeds: list[float | None] = []
    heights: list[float | None] = []
    for index, current in enumerate(positions):
        previous = positions[index - 1] if index else None
        following = positions[index + 1] if index + 1 < len(positions) else None
        if current is None:
            speeds.append(None)
        elif previous is not None and following is not None:
            speeds.append(float(np.linalg.norm(following - previous) / (times[index + 1] - times[index - 1])))
        elif previous is not None:
            speeds.append(float(np.linalg.norm(current - previous) / (times[index] - times[index - 1])))
        elif following is not None:
            speeds.append(float(np.linalg.norm(following - current) / (times[index + 1] - times[index])))
        else:
            speeds.append(None)
        window = [p[2] for time, p in zip(times, positions)
                  if p is not None and abs(time - times[index]) <= height_radius_seconds + 1e-9]
        heights.append(float(np.std(window)) if len(window) >= 2 else None)
    return speeds, heights


def fit_time_aware_thresholds(train_sequences: list[LiftedPoseSequence], foot: str,
                              height_radius_seconds: float, min_run_span_seconds: float) -> TimeAwareThresholds:
    speeds, heights = [], []
    for sequence in train_sequences:
        sequence_speeds, sequence_heights = kinematics(sequence, foot, height_radius_seconds)
        speeds.extend(value for value in sequence_speeds if value is not None)
        heights.extend(value for value in sequence_heights if value is not None)
    if len(speeds) < 10 or len(heights) < 10:
        raise ValueError("not enough valid TRAIN observations")
    speeds.sort()
    heights.sort()
    contact = _percentile(speeds, 15.0)
    moving = _percentile(speeds, 60.0)
    if moving <= contact:
        moving = contact + 1e-3
    return TimeAwareThresholds(contact, moving, _percentile(heights, 40.0),
                               height_radius_seconds, min_run_span_seconds)


def classify_time_aware_contact(sequence: LiftedPoseSequence, foot: str,
                                thresholds: TimeAwareThresholds) -> list[ContactState]:
    times = elapsed_times(sequence)
    speeds, heights = kinematics(sequence, foot, thresholds.height_radius_seconds)
    states = []
    for speed, height in zip(speeds, heights):
        if speed is None or height is None:
            states.append(ContactState.UNKNOWN)
        elif speed <= thresholds.contact_speed_m_s and height <= thresholds.height_std_threshold_m:
            states.append(ContactState.CONTACT)
        elif speed >= thresholds.moving_speed_m_s:
            states.append(ContactState.MOVING)
        else:
            states.append(ContactState.UNKNOWN)
    output = list(states)
    index = 0
    while index < len(states):
        end = index + 1
        while end < len(states) and states[end] is states[index]:
            end += 1
        if states[index] in (ContactState.CONTACT, ContactState.MOVING) and (
            end - index < 2 or times[end - 1] - times[index] + 1e-9 < thresholds.min_run_span_seconds
        ):
            output[index:end] = [ContactState.UNKNOWN] * (end - index)
        index = end
    return output
