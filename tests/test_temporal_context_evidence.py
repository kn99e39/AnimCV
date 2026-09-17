from types import SimpleNamespace

import numpy as np
import pytest

from scripts.diagnose_temporal_context_evidence import (
    _linear_residual,
    classify_temporal_rows,
    failure_mask,
    interpolate_recovery,
    local_neighbor_rows,
    neighbor_evidence,
    observation_gap_metrics,
    nearest_temporal_support,
    sequence_rows,
)


def _sample(sequence: str, frame: int, timestamp: float):
    return SimpleNamespace(sequence_id=sequence, frame_index=frame, timestamp=timestamp,
                           fps=30.0, sample_id=f"{sequence}#{frame:06d}")


def test_neighbor_indexing_is_sequence_local_and_timestamp_sorted():
    samples = [_sample("A", 2, 0.2), _sample("B", 0, 0.0), _sample("A", 0, 0.0),
               _sample("B", 1, 0.1), _sample("A", 1, 0.1)]
    positions = np.arange(len(samples))

    grouped = sequence_rows(samples, positions)
    before, after = local_neighbor_rows(grouped, len(samples), max_steps=2)

    assert grouped["A"].tolist() == [2, 4, 0]
    assert before[2] == [] and after[2] == [4, 0]
    assert before[4] == [2] and after[4] == [0]
    assert before[0] == [4, 2] and after[0] == []
    assert before[1] == [] and after[1] == [3]
    assert 1 not in before[2] + after[2]


def test_timestamp_interpolation_uses_actual_spacing_not_frame_count():
    left = np.asarray([0.0, 0.0, 0.0])
    center = np.asarray([2.0, 0.0, 0.0])
    right = np.asarray([4.0, 0.0, 0.0])

    assert _linear_residual(left, center, right, 0.0, 0.5, 1.0) == 0.0
    assert _linear_residual(left, center, right, 0.0, 0.25, 1.0) == pytest.approx(1.0)


def test_missing_neighbor_is_unresolved_and_recovery_only_changes_eligible_joint():
    h0 = np.asarray([[[0.0, 0.0, 0.0], [9.0, 0.0, 0.0]],
                     [[8.0, 0.0, 0.0], [8.0, 0.0, 0.0]],
                     [[4.0, 0.0, 0.0], [7.0, 0.0, 0.0]]])
    in_frame = np.asarray([[True, True], [False, True], [True, False]])
    before = [[], [0], [1, 0]]
    after = [[1, 2], [2], []]
    timestamps = np.asarray([0.0, 0.25, 1.0])

    left, right, support = nearest_temporal_support(in_frame, h0, timestamps, before, after)
    eligible = np.zeros_like(support)
    eligible[1, 0] = support[1, 0]
    recovered = interpolate_recovery(h0, timestamps, left, right, eligible)

    assert support[1, 0]
    assert not support[1, 1]
    assert recovered[1, 0, 0] == pytest.approx(1.0)
    np.testing.assert_array_equal(recovered[1, 1], h0[1, 1])
    np.testing.assert_array_equal(recovered[0], h0[0])


def test_temporal_classes_leave_stable_controls_unlabelled():
    failure = np.asarray([[True, True, True, True, True, False]])
    in_frame = np.asarray([[True, True, False, True, True, True]])
    stable = np.asarray([[False, False, False, True, False, True]])
    eligible = np.asarray([[True, True, False, False, False, True]])
    target_valid = np.ones_like(failure)
    before = np.asarray([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
    after = np.asarray([[0.5, 1.5, np.nan, 1.0, np.nan, 1.0]])

    labels = classify_temporal_rows(failure, in_frame, stable, eligible, target_valid, before, after)

    assert labels[0].tolist() == [
        "T1_TEMPORALLY_RECOVERABLE_OBSERVATION_GAP",
        "T2_TEMPORAL_EVIDENCE_SIMPLE_RECOVERY_INSUFFICIENT",
        "T4_OBSERVATION_FAILURE_WITHOUT_TEMPORAL_SUPPORT",
        "T3_NON_TEMPORAL_FRAME_POSE_FAILURE",
        "T5_MIXED_OR_UNRESOLVED",
        "",
    ]


def test_stable_control_wins_over_a_quantile_tie_and_cannot_be_probed():
    cohorts = {
        "current_frame_observation_loss": np.asarray([[False, False]]),
        "observation_degradation": np.asarray([[True, False]]),
        "h0_jitter_stable_observation": np.asarray([[False, True]]),
        "distal_joint_failure": np.asarray([[False, False]]),
        "implausible_articulation": np.asarray([[False, False]]),
        "stable_control": np.asarray([[True, False]]),
    }

    result = failure_mask(cohorts)

    np.testing.assert_array_equal(result, [[False, True]])


def test_neighbor_evidence_and_gap_accounting_stay_local_to_sequence():
    in_frame = np.asarray([[True], [False], [False], [True], [True]])
    confidence = np.asarray([[0.2], [0.1], [0.1], [0.9], [0.3]])
    h0_error = np.asarray([[0.4], [0.8], [0.7], [0.3], [0.5]])
    before = [[], [0], [1, 0], [2, 1], [3, 2]]
    after = [[1, 2], [2, 3], [3, 4], [4], []]
    evidence = neighbor_evidence(in_frame, confidence, h0_error, before, after)

    assert evidence["observed_before"][1, 0] == 0
    assert evidence["observed_after"][1, 0] == 3
    assert evidence["two_sided_observation"][1, 0]
    assert evidence["higher_confidence_neighbor"][1, 0]
    assert evidence["lower_oracle_error_neighbor"][1, 0]

    grouped = {"A": np.arange(5)}
    target = np.asarray([[[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]], [[2.0, 0.0, 0.0]],
                         [[3.0, 0.0, 0.0]], [[4.0, 0.0, 0.0]]])
    gaps = observation_gap_metrics(in_frame, target, np.ones_like(in_frame), np.arange(5.0), grouped)
    assert gaps["observation_gap_length_frames"][1, 0] == 2
    assert gaps["observation_gap_length_frames"][2, 0] == 2
    assert gaps["oracle_gap_residual"][1, 0] == pytest.approx(0.0)
