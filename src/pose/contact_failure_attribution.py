"""Attribute O-vs-H0 contact-candidate disagreements to one of five causes.

Section 7 of the H0 replay batch (docs/52): when the oracle-geometry contact
candidate and the H0-driven contact candidate disagree at the same frame,
this module classifies *why*, using only evidence already computed elsewhere
(H0-vs-oracle joint position error, H0's own observation validity, and each
candidate's agreement with the world-frame KINEMATIC_PROXY reference). It
does not invent a sixth category.

Precedence matters: a frame where even the ORACLE candidate already
disagreed with the reference is pre-existing root-relative/world ambiguity
(docs/51's own finding), not something H0 introduced, so that check runs
first and pre-empts the others.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from common.canonical_pose import JOINT_NAMES
from pose.contact import ContactState
from pose.pose_lifter import LiftedPoseSequence


class FailureCategory(Enum):
    POSE_POSITION_ERROR = "pose_position_error"
    TEMPORAL_JITTER = "temporal_jitter"
    OBSERVATION_INVALIDITY = "observation_invalidity"
    ANKLE_SPECIFIC_FAILURE = "ankle_specific_failure"
    BODY_ROOT_RELATIVE_MOTION_AMBIGUITY = "body_root_relative_motion_ambiguity"


@dataclass(frozen=True)
class FailureAttribution:
    frame_index: int  # position within the matched O/H subset, not a raw dataset frame index
    category: FailureCategory
    ankle_position_error_m: float | None
    mean_other_joint_position_error_m: float | None


def fit_position_error_threshold(train_errors: list[float], *, percentile: float = 90.0) -> float:
    """A TRAIN-only per-joint-per-frame H0-vs-oracle position-error cutoff."""
    values = sorted(error for error in train_errors if error is not None)
    if len(values) < 10:
        raise ValueError("not enough TRAIN errors to fit a position error threshold")
    position = (len(values) - 1) * (percentile / 100.0)
    lower, upper = int(np.floor(position)), int(np.ceil(position))
    if lower == upper:
        return float(values[int(position)])
    return float(values[lower] + (values[upper] - values[lower]) * (position - lower))


def per_frame_joint_errors(
    oracle: LiftedPoseSequence, h0: LiftedPoseSequence,
) -> list[dict[str, float | None]]:
    """H0-vs-oracle per-joint Euclidean error for each aligned frame pair.

    ``oracle`` and ``h0`` must already be the same matched-frame subset, same
    length and order (the replay script aligns them before calling this).
    """
    if len(oracle.frames) != len(h0.frames):
        raise ValueError("oracle and h0 sequences must be the same, matched length")
    errors = []
    for oracle_frame, h0_frame in zip(oracle.frames, h0.frames):
        frame_errors: dict[str, float | None] = {}
        for name in JOINT_NAMES:
            oracle_point = oracle_frame.points.get(name)
            h0_point = h0_frame.points.get(name)
            if oracle_point is None or h0_point is None or not h0_point.observation_valid:
                frame_errors[name] = None
                continue
            frame_errors[name] = float(np.linalg.norm(
                np.asarray(h0_point.position) - np.asarray(oracle_point.position)
            ))
        errors.append(frame_errors)
    return errors


def attribute_disagreements(
    joint_errors: list[dict[str, float | None]],
    foot: str,
    candidate_oracle: list[ContactState],
    candidate_h0: list[ContactState],
    reference: list[ContactState],
    position_error_threshold_m: float,
    *,
    ankle_dominance_ratio: float = 2.0,
) -> list[FailureAttribution]:
    ankle_joint = f"{foot}_ankle"
    lengths = {len(joint_errors), len(candidate_oracle), len(candidate_h0), len(reference)}
    if len(lengths) != 1:
        raise ValueError("all inputs to attribute_disagreements must be the same, matched length")

    attributions: list[FailureAttribution] = []
    for index in range(len(candidate_oracle)):
        if candidate_oracle[index] is candidate_h0[index]:
            continue
        ankle_error = joint_errors[index].get(ankle_joint)
        other_errors = [value for name, value in joint_errors[index].items() if name != ankle_joint and value is not None]
        mean_other_error = float(np.mean(other_errors)) if other_errors else None

        if reference[index] is not ContactState.UNKNOWN and candidate_oracle[index] is not reference[index]:
            # Even oracle geometry disagreed with the world-frame proxy here --
            # a pre-existing ambiguity docs/51 already documented, not something
            # H0 introduced.
            category = FailureCategory.BODY_ROOT_RELATIVE_MOTION_AMBIGUITY
        elif ankle_error is None:
            # per_frame_joint_errors returns None exactly when H0 marked this
            # joint's own observation invalid.
            category = FailureCategory.OBSERVATION_INVALIDITY
        elif ankle_error > position_error_threshold_m:
            if mean_other_error is not None and ankle_error > ankle_dominance_ratio * mean_other_error:
                category = FailureCategory.ANKLE_SPECIFIC_FAILURE
            else:
                category = FailureCategory.POSE_POSITION_ERROR
        else:
            # Position itself is within the TRAIN-typical error band, so the
            # disagreement must come from an unstable frame-to-frame velocity
            # estimate rather than a bad single-frame position.
            category = FailureCategory.TEMPORAL_JITTER

        attributions.append(FailureAttribution(
            frame_index=index, category=category,
            ankle_position_error_m=ankle_error, mean_other_joint_position_error_m=mean_other_error,
        ))
    return attributions


def summarize_attribution(attributions: list[FailureAttribution]) -> dict[str, int]:
    counts = {category.value: 0 for category in FailureCategory}
    for attribution in attributions:
        counts[attribution.category.value] += 1
    counts["total_disagreements"] = len(attributions)
    return counts
