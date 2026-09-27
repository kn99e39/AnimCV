"""Foot-contact candidate (Layer D territory) — CONTACT / MOVING / UNKNOWN only.

This is a bounded diagnostic candidate, not foot locking and not a claim
about ground-truth contact (see contact_reference.py and docs/51). It scores
one signal that is honestly available from a root-relative Frame Pose
sequence today: how much a foot moves *relative to the pelvis*, plus a
short-window height-consistency check. That is closer to a stance/swing-phase
detector than to true world-space "planted on the ground" state, because
knowing the latter requires knowing whether the body itself is translating —
which is exactly the Root Motion question this batch leaves open (see
motion_ownership.py). It refuses (UNKNOWN) whenever the evidence needed to
decide is itself missing or unreliable, rather than guessing.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import numpy as np

from pose.pose_lifter import LiftedPoseSequence


class ContactState(Enum):
    CONTACT = "contact"
    MOVING = "moving"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ContactThresholds:
    """Cutoffs fit from TRAIN sequences only — never from validation/test."""

    contact_speed_m_s: float
    moving_speed_m_s: float
    height_std_threshold_m: float
    min_reliable_run: int = 2

    def __post_init__(self) -> None:
        if not (0.0 <= self.contact_speed_m_s < self.moving_speed_m_s):
            raise ValueError("thresholds must satisfy 0 <= contact_speed_m_s < moving_speed_m_s")
        if self.height_std_threshold_m < 0.0:
            raise ValueError("height_std_threshold_m must be non-negative")
        if self.min_reliable_run < 1:
            raise ValueError("min_reliable_run must be >= 1")


def fit_contact_thresholds(
    train_sequences: list[LiftedPoseSequence],
    foot: str,
    *,
    contact_percentile: float = 15.0,
    moving_percentile: float = 60.0,
    height_percentile: float = 40.0,
    min_reliable_run: int = 2,
) -> ContactThresholds:
    """Derive speed/height cutoffs from TRAIN ankle kinematics only.

    Calling this with validation or test sequences would tune the detector on
    the split it is later scored against; the caller is responsible for only
    ever passing a train split.
    """
    all_speeds: list[float] = []
    all_height_stds: list[float] = []
    for sequence in train_sequences:
        positions = _finite_diff_positions(sequence, foot)
        speeds = _ankle_speeds(sequence, foot)
        all_speeds.extend(speed for speed in speeds if speed is not None)
        all_height_stds.extend(
            std for std in (_height_std(positions, i) for i in range(len(positions))) if std is not None
        )
    if len(all_speeds) < 10 or len(all_height_stds) < 10:
        raise ValueError("not enough valid train observations to fit contact thresholds")
    all_speeds.sort()
    all_height_stds.sort()
    contact_cut = _percentile(all_speeds, contact_percentile)
    moving_cut = _percentile(all_speeds, moving_percentile)
    if moving_cut <= contact_cut:
        moving_cut = contact_cut + 1e-3
    height_cut = _percentile(all_height_stds, height_percentile)
    return ContactThresholds(
        contact_speed_m_s=contact_cut, moving_speed_m_s=moving_cut,
        height_std_threshold_m=height_cut, min_reliable_run=min_reliable_run,
    )


def classify_foot_contact(
    sequence: LiftedPoseSequence, foot: str, thresholds: ContactThresholds,
) -> list[ContactState]:
    """Per-frame CONTACT/MOVING/UNKNOWN candidate from local ankle kinematics."""
    positions = _finite_diff_positions(sequence, foot)
    speeds = _ankle_speeds(sequence, foot)
    states = []
    for index in range(len(positions)):
        speed = speeds[index]
        height_std = _height_std(positions, index)
        if speed is None or height_std is None:
            states.append(ContactState.UNKNOWN)
        elif speed <= thresholds.contact_speed_m_s and height_std <= thresholds.height_std_threshold_m:
            states.append(ContactState.CONTACT)
        elif speed >= thresholds.moving_speed_m_s:
            states.append(ContactState.MOVING)
        else:
            states.append(ContactState.UNKNOWN)
    return _suppress_short_runs(states, thresholds.min_reliable_run)


def _finite_diff_positions(sequence: LiftedPoseSequence, foot: str) -> list[np.ndarray | None]:
    joint = f"{foot}_ankle"
    positions: list[np.ndarray | None] = []
    for frame in sequence.frames:
        point = frame.points.get(joint)
        if point is not None and point.observation_valid:
            positions.append(np.asarray(point.position, dtype=float))
        else:
            positions.append(None)
    return positions


def _ankle_speeds(sequence: LiftedPoseSequence, foot: str) -> list[float | None]:
    positions = _finite_diff_positions(sequence, foot)
    if not sequence.source_fps:
        return [None] * len(positions)
    dt = 1.0 / sequence.source_fps
    speeds: list[float | None] = [None] * len(positions)
    for index in range(len(positions)):
        current = positions[index]
        if current is None:
            continue
        previous = positions[index - 1] if index - 1 >= 0 else None
        following = positions[index + 1] if index + 1 < len(positions) else None
        if previous is not None and following is not None:
            speeds[index] = float(np.linalg.norm(following - previous) / (2 * dt))
        elif previous is not None:
            speeds[index] = float(np.linalg.norm(current - previous) / dt)
        elif following is not None:
            speeds[index] = float(np.linalg.norm(following - current) / dt)
    return speeds


def _height_std(positions: list[np.ndarray | None], index: int, radius: int = 2) -> float | None:
    window = [
        p[2] for p in positions[max(0, index - radius):index + radius + 1] if p is not None
    ]
    if len(window) < 2:
        return None
    return float(np.std(window))


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        raise ValueError("no values to compute percentile")
    position = (len(sorted_values) - 1) * (percentile / 100.0)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return sorted_values[int(position)]
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (position - lower)


def _suppress_short_runs(states: list[ContactState], min_run: int) -> list[ContactState]:
    """A CONTACT/MOVING run shorter than min_run has no temporal support."""
    output = list(states)
    index = 0
    while index < len(output):
        end = index
        while end < len(output) and output[end] == output[index]:
            end += 1
        if output[index] in (ContactState.CONTACT, ContactState.MOVING) and (end - index) < min_run:
            for position in range(index, end):
                output[position] = ContactState.UNKNOWN
        index = end
    return output
