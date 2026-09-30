"""Frozen FramePose H0 to current-policy AnimationSemantics v2."""

from __future__ import annotations

import statistics

import numpy as np

from framepose.contract import COORDINATE_FRAME, FrameBank, JOINT_NAMES
from framepose.nonlinear_reconstruction import in_frame_mask
from motion.animation_semantics import FootMotion, LocalArticulation, ObservationReliability, SemanticsProvenance
from motion.animation_semantics_bridge import _single, _single_provenance
from motion.animation_semantics_v2 import (
    AnimationSemanticsV2, CurrentRootOrientation, ROOT_POLICY_V2, SemanticFrameV2, TimeAwareCalibration,
)
from pose.contact_time_aware import (
    RULE_VERSION, TimeAwareThresholds, classify_time_aware_contact, fit_time_aware_thresholds,
)
from pose.framepose_bridge import H0Identity, build_h0_lifted_sequence, sequence_frame_positions, sequence_ids_in_split
from pose.root_orientation_diagnostic import observe_yaw


def calibrate_time_aware_contact_from_h0_train(
    bank: FrameBank, h0: np.ndarray, identity: H0Identity,
) -> TimeAwareCalibration:
    if "train" not in identity.split_sha256:
        raise ValueError("H0 identity has no TRAIN split")
    sequence_ids = sequence_ids_in_split(bank, "train")
    sequences = [build_h0_lifted_sequence(bank, h0, sid) for sid in sequence_ids]
    fps_values = {sequence.source_fps for sequence in sequences}
    if len(fps_values) != 1 or not fps_values or next(iter(fps_values)) <= 0:
        raise ValueError("TRAIN time derivation requires one positive source fps")
    fps = next(iter(fps_values))
    strides = [b.frame_index - a.frame_index for sequence in sequences
               for a, b in zip(sequence.frames, sequence.frames[1:])]
    if not strides or any(stride <= 0 for stride in strides):
        raise ValueError("TRAIN requires positive row strides")
    median_stride = statistics.median(strides)
    row_seconds = median_stride / fps
    height_radius = 2 * row_seconds  # historical radius of two TRAIN rows
    min_run_span = (2 - 1) * row_seconds  # historical minimum of two TRAIN rows
    thresholds = {side: fit_time_aware_thresholds(sequences, side, height_radius, min_run_span)
                  for side in ("left", "right")}
    return TimeAwareCalibration(
        thresholds={side: vars(value) for side, value in thresholds.items()},
        source={"split": "train", "frame_pose_output": "frozen_h0",
                "bank_content_digest": identity.bank_content_digest,
                "h0_train_sha256": identity.split_sha256["train"],
                "sequence_count": len(sequences),
                "frame_count": sum(len(sequence.frames) for sequence in sequences),
                "regime": bank.regime(), "procedure": "framepose_elapsed_time_h0_train_fit"},
        sampling={"time_basis": "elapsed_seconds", "source_fps": fps,
                  "train_median_row_stride_frames": median_stride,
                  "historical_height_radius_train_rows": 2,
                  "historical_min_run_train_rows": 2,
                  "height_radius_seconds": height_radius,
                  "min_run_span_seconds": min_run_span,
                  "derivation": "radius=2*median_stride/source_fps; minimum run span=(2-1)*median_stride/source_fps"},
    )


def current_root_orientation(lifted) -> list[CurrentRootOrientation]:
    """docs/55 current fused yaw, requiring a valid bilateral pair for production."""
    output = []
    for frame, observation in zip(lifted.frames, observe_yaw(lifted)):
        # A diagnostic can compute a yaw from invalid joints to measure its
        # failures. Production refuses a frame with no valid bilateral pair.
        has_valid_pair = any(
            frame.points.get(left) is not None and frame.points.get(right) is not None
            and frame.points[left].observation_valid and frame.points[right].observation_valid
            for left, right in (("left_shoulder", "right_shoulder"), ("left_hip", "right_hip"))
        )
        yaw = observation.yaw_radians if has_valid_pair else None
        output.append(CurrentRootOrientation(yaw is not None, yaw))
    return output


def build_animation_semantics_v2(
    bank: FrameBank, h0: np.ndarray, identity: H0Identity, sequence_id: str,
    contact_calibration: TimeAwareCalibration,
) -> AnimationSemanticsV2:
    positions = sequence_frame_positions(bank, sequence_id)
    lifted = build_h0_lifted_sequence(bank, h0, sequence_id)
    samples = [bank.samples[position] for position in positions]
    in_frame = in_frame_mask(bank.arrays["input_2d"][positions], bank.arrays["input_valid"][positions])
    orientations = current_root_orientation(lifted)
    contacts = {side: classify_time_aware_contact(lifted, side, TimeAwareThresholds(**contact_calibration.thresholds[side]))
                for side in ("left", "right")}
    frames = tuple(SemanticFrameV2(
        frame_index=frame.frame_index, timestamp=frame.timestamp,
        articulation=LocalArticulation(tuple(frame.points[name].position for name in JOINT_NAMES)),
        root_orientation=orientations[row],
        foot_motion=FootMotion(contacts["left"][row], contacts["right"][row]),
        reliability=ObservationReliability(
            tuple(frame.points[name].observation_valid for name in JOINT_NAMES),
            tuple(bool(value) for value in in_frame[row])),
    ) for row, frame in enumerate(lifted.frames))
    provenance = SemanticsProvenance(
        frame_pose={"source": "frozen_h0", "bank_content_digest": identity.bank_content_digest,
                    "h0_split_sha256": dict(identity.split_sha256),
                    "split": _single(sample.split for sample in samples),
                    "coordinate_frame": COORDINATE_FRAME, "units": lifted.units},
        root_orientation={"policy": ROOT_POLICY_V2,
                          "estimator": "pose.root_orientation_diagnostic.observe_yaw",
                          "yaw_reference_frame": "camera", "evidence": "docs/55 H0-CURRENT"},
        contact={"rule_version": RULE_VERSION, "calibration": contact_calibration.to_dict(),
                 "sequence_time_basis": "timestamps_or_frame_index_over_source_fps"},
        reliability={"joint_observation_valid": "bank.input_valid AND finite Frame Pose output",
                     "joint_in_frame": "framepose.nonlinear_reconstruction.in_frame_mask(input_2d, input_valid)",
                     "observation_provenance": _single_provenance(samples),
                     "regime": _single(sample.observation.regime for sample in samples),
                     "learned_reliability": None},
    )
    return AnimationSemanticsV2(sequence_id, float(lifted.source_fps), COORDINATE_FRAME,
                                tuple(JOINT_NAMES), frames, provenance)
