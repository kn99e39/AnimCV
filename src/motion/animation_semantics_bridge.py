"""FramePose H0 -> AnimationSemantics bridge (docs/53).

One deterministic, sequence-local conversion from

    frozen FramePose H0 + FrameBank input observation metadata
    + the historical Root Orientation estimator (docs/52: usable on H0)
    + the docs/51 contact rule at an H0-TRAIN calibration (docs/52 T1)

to motion.animation_semantics.AnimationSemantics. It designs no estimator:
every signal comes from an existing, unmodified function.

Reads only `bank.arrays["input_2d"]`, `bank.arrays["input_valid"]` and the
caller's H0 array (through pose.framepose_bridge) — never `target_3d` or
`target_valid`. Root translation and ground height are never computed.
"""

from __future__ import annotations

import statistics
from typing import Any

import numpy as np

from framepose.contract import COORDINATE_FRAME, FrameBank, JOINT_NAMES
from framepose.nonlinear_reconstruction import in_frame_mask
from motion.animation_semantics import (
    CONTACT_MIN_RELIABLE_RUN, CONTACT_RULE_PERCENTILES, CONTACT_RULE_VERSION, FOOT_SIDES,
    ROOT_ORIENTATION_NOTES, ROOT_ORIENTATION_POLICY, UNKNOWN_ORIENTATION, AnimationSemantics,
    ContactCalibration, FootMotion, LocalArticulation, ObservationReliability, RootOrientation,
    SemanticFrame, SemanticsProvenance,
)
from pose.contact import ContactThresholds, classify_foot_contact, fit_contact_thresholds
from pose.framepose_bridge import (
    H0Identity, build_h0_lifted_sequence, sequence_frame_positions, sequence_ids_in_split,
)
from pose.pose_lifter import LiftedPoseSequence
from pose.root_motion import estimate_root_motion

HISTORICAL_SMOOTHING_WINDOW = 5
HISTORICAL_MAX_YAW_STEP_DEGREES = 20.0


# ---------------------------------------------------------------- calibration --

def calibrate_contact_from_h0_train(bank: FrameBank, h0: np.ndarray, identity: H0Identity) -> ContactCalibration:
    """docs/52 T1, exactly: the unchanged docs/51 fit on every H0 TRAIN sequence.

    This is a model/domain calibration of the H0 output distribution. It never
    sees a validation or test sequence, and is never re-fit per video.
    """
    if "train" not in identity.split_sha256:
        raise ValueError("H0 identity has no train split; contact calibration is TRAIN-only")
    sequence_ids = sequence_ids_in_split(bank, "train")
    sequences = [build_h0_lifted_sequence(bank, h0, sequence_id) for sequence_id in sequence_ids]
    thresholds = {
        side: fit_contact_thresholds(
            sequences, side,
            contact_percentile=CONTACT_RULE_PERCENTILES["contact"],
            moving_percentile=CONTACT_RULE_PERCENTILES["moving"],
            height_percentile=CONTACT_RULE_PERCENTILES["height"],
            min_reliable_run=CONTACT_MIN_RELIABLE_RUN,
        )
        for side in FOOT_SIDES
    }
    return ContactCalibration(
        thresholds={side: _thresholds_dict(value) for side, value in thresholds.items()},
        source={
            "split": "train",
            "frame_pose_output": "frozen_h0",
            "bank_content_digest": identity.bank_content_digest,
            "h0_train_sha256": identity.split_sha256["train"],
            "sequence_count": len(sequences),
            "frame_count": sum(len(sequence.frames) for sequence in sequences),
            "regime": bank.regime(),
            "procedure": "docs52_T1_h0_train_refit",
        },
        sampling=_sampling_domain(sequences),
    )


def _thresholds_dict(value: ContactThresholds) -> dict[str, float | int]:
    return {"contact_speed_m_s": value.contact_speed_m_s, "moving_speed_m_s": value.moving_speed_m_s,
            "height_std_threshold_m": value.height_std_threshold_m, "min_reliable_run": value.min_reliable_run}


def _thresholds(calibration: ContactCalibration, side: str) -> ContactThresholds:
    return ContactThresholds(**calibration.thresholds[side])


def _row_strides(sequence: LiftedPoseSequence) -> list[int]:
    indices = [frame.frame_index for frame in sequence.frames]
    return [b - a for a, b in zip(indices, indices[1:])]


def _sampling_domain(sequences: list[LiftedPoseSequence]) -> dict[str, Any]:
    strides = [stride for sequence in sequences for stride in _row_strides(sequence)]
    fps = sorted({sequence.source_fps for sequence in sequences})
    return {
        "rule_time_basis": "row_neighbours_at_source_fps",
        "source_fps": fps[0] if len(fps) == 1 else fps,
        "median_row_stride_frames": statistics.median(strides) if strides else None,
    }


# --------------------------------------------------------------------- bridge --

def build_animation_semantics(
    bank: FrameBank, h0: np.ndarray, identity: H0Identity, sequence_id: str,
    contact_calibration: ContactCalibration,
) -> AnimationSemantics:
    """Sequence-local semantics for one bank sequence. Deterministic."""
    positions = sequence_frame_positions(bank, sequence_id)
    lifted = build_h0_lifted_sequence(bank, h0, sequence_id)
    joint_valid = [
        tuple(frame.points[name].observation_valid for name in JOINT_NAMES) for frame in lifted.frames
    ]
    in_frame = in_frame_mask(bank.arrays["input_2d"][positions], bank.arrays["input_valid"][positions])
    orientation = _root_orientation(lifted)
    contact = {side: classify_foot_contact(lifted, side, _thresholds(contact_calibration, side))
               for side in FOOT_SIDES}

    frames = tuple(
        SemanticFrame(
            frame_index=frame.frame_index,
            timestamp=frame.timestamp,
            articulation=LocalArticulation(tuple(frame.points[name].position for name in JOINT_NAMES)),
            root_orientation=orientation[row],
            foot_motion=FootMotion(contact["left"][row], contact["right"][row]),
            reliability=ObservationReliability(joint_valid[row], tuple(bool(v) for v in in_frame[row])),
        )
        for row, frame in enumerate(lifted.frames)
    )
    sequence_sampling = _sampling_domain([lifted])
    samples = [bank.samples[position] for position in positions]
    provenance = SemanticsProvenance(
        frame_pose={
            "source": "frozen_h0",
            "bank_content_digest": identity.bank_content_digest,
            "h0_split_sha256": dict(identity.split_sha256),
            "split": _single(sample.split for sample in samples),
            "coordinate_frame": COORDINATE_FRAME,
            "units": lifted.units,
        },
        root_orientation={
            "policy": ROOT_ORIENTATION_POLICY,
            "estimator": "pose.root_motion.estimate_root_motion",
            "smoothing_window": HISTORICAL_SMOOTHING_WINDOW,
            "max_yaw_step_degrees": HISTORICAL_MAX_YAW_STEP_DEGREES,
            "max_yaw_step_status": "historical_default_not_proven_optimal",
            "yaw_reference_frame": "camera",
            "notes": ROOT_ORIENTATION_NOTES,
            "evidence": "docs/52",
        },
        contact={
            "rule_version": CONTACT_RULE_VERSION,
            "calibration": contact_calibration.to_dict(),
            "sequence_sampling": sequence_sampling,
            "sampling_matches_calibration": (
                sequence_sampling["source_fps"] == contact_calibration.sampling["source_fps"]
                and sequence_sampling["median_row_stride_frames"]
                == contact_calibration.sampling["median_row_stride_frames"]
            ),
        },
        reliability={
            "joint_observation_valid": "bank.input_valid AND finite Frame Pose output",
            "joint_in_frame": "framepose.nonlinear_reconstruction.in_frame_mask(input_2d, input_valid)",
            "observation_provenance": _single_provenance(samples),
            "regime": _single(sample.observation.regime for sample in samples),
            "learned_reliability": None,  # docs/50: Qwen VLM reliability rejected, not carried
        },
    )
    return AnimationSemantics(
        sequence_id=sequence_id,
        source_fps=float(lifted.source_fps),
        coordinate_frame=COORDINATE_FRAME,
        joint_names=tuple(JOINT_NAMES),
        frames=frames,
        provenance=provenance,
    )


def _root_orientation(lifted: LiftedPoseSequence) -> list[RootOrientation]:
    """Historical estimator per maximal computable run; UNKNOWN elsewhere.

    estimate_root_motion raises for a whole sequence if any one frame lacks a
    valid bilateral torso pair. Rather than altering it, it is run unchanged on
    each maximal run of frames it can compute; with no gaps that is exactly
    one call on the whole sequence.
    """
    computable = [_yaw_computable(lifted, row) for row in range(len(lifted.frames))]
    output: list[RootOrientation] = [UNKNOWN_ORIENTATION] * len(lifted.frames)
    row = 0
    while row < len(computable):
        if not computable[row]:
            row += 1
            continue
        end = row
        while end < len(computable) and computable[end]:
            end += 1
        run = LiftedPoseSequence(frames=lifted.frames[row:end], source_fps=lifted.source_fps,
                                 coordinate_frame=lifted.coordinate_frame, units=lifted.units,
                                 backend=lifted.backend)
        estimated = estimate_root_motion(run, smoothing_window=HISTORICAL_SMOOTHING_WINDOW,
                                         max_yaw_step_degrees=HISTORICAL_MAX_YAW_STEP_DEGREES)
        for offset, frame in enumerate(estimated.frames):
            output[row + offset] = RootOrientation(
                known=True, yaw_radians=float(frame.root_yaw_radians),
                historical_confidence=float(frame.confidence), yaw_held=bool(frame.yaw_held),
            )
        row = end
    return output


def _yaw_computable(lifted: LiftedPoseSequence, row: int) -> bool:
    single = LiftedPoseSequence(frames=[lifted.frames[row]], source_fps=lifted.source_fps)
    try:
        estimate_root_motion(single, smoothing_window=1, max_yaw_step_degrees=HISTORICAL_MAX_YAW_STEP_DEGREES)
    except ValueError:
        return False
    return True


def _single(values) -> Any:
    distinct = sorted(set(values))
    if len(distinct) != 1:
        raise ValueError(f"expected one value within a sequence, got {distinct}")
    return distinct[0]


def _single_provenance(samples) -> dict[str, Any]:
    payloads = {str(sorted(sample.observation.to_dict().items())): sample.observation.to_dict() for sample in samples}
    if len(payloads) != 1:
        raise ValueError("a sequence mixes observation provenance; refusing to summarize it as one")
    return next(iter(payloads.values()))
