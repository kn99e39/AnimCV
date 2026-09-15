import hashlib
import numpy as np
import pytest

from framepose.sign_advisor import prompt_text
from framepose.signs import SIGN_FIELD_NAMES, UNKNOWN
from scripts.evaluate_current_sign_advisor import (
    EXPECTED_MODEL_ID,
    EXPECTED_MODEL_REVISION,
    EXPECTED_PROMPT_SHA256,
    EXPECTED_SEED,
    EXPECTED_WEIGHT_FINGERPRINT,
    EXPECTED_MAX_NEW_TOKENS,
    different_sequence_donors,
    field_metrics,
    paired_bootstrap_accuracy_difference,
    paired_bootstrap_hinge_metrics,
    summarize_predictions,
    validate_frozen_backend,
)
from framepose.sign_advisor import ADVISOR_CROP_RESOLUTION


def test_frozen_backend_provenance_and_prompt_are_pinned():
    snapshot = {
        "resolved_commit": EXPECTED_MODEL_REVISION,
        "weight_fingerprint": EXPECTED_WEIGHT_FINGERPRINT,
        "file_count": 11,
    }
    actual_prompt_sha = hashlib.sha256(prompt_text().encode("utf-8")).hexdigest()

    assert actual_prompt_sha == EXPECTED_PROMPT_SHA256
    validate_frozen_backend(
        EXPECTED_MODEL_ID, EXPECTED_MODEL_REVISION, snapshot, actual_prompt_sha,
        max_new_tokens=EXPECTED_MAX_NEW_TOKENS, seed=EXPECTED_SEED,
        crop_resolution=ADVISOR_CROP_RESOLUTION)
    with pytest.raises(ValueError, match="prompt hash"):
        validate_frozen_backend(
            EXPECTED_MODEL_ID, EXPECTED_MODEL_REVISION, snapshot, "drifted",
            max_new_tokens=EXPECTED_MAX_NEW_TOKENS, seed=EXPECTED_SEED,
            crop_resolution=ADVISOR_CROP_RESOLUTION)


def test_strict_parse_failure_unknown_is_counted_wrong_on_labeled_rows():
    reference = np.asarray([1, -1, 1, UNKNOWN], dtype=np.int8)
    predicted = np.asarray([1, 0, -1, 0], dtype=np.int8)
    parsed_valid = np.asarray([True, False, True, False])

    metrics = field_metrics(predicted, parsed_valid, reference)

    assert metrics["labeled_rows"] == 3
    assert metrics["strict_parse_failures"] == 2
    assert metrics["strict_parse_success_rate_on_oracle_readable_rows"] == 2 / 3
    assert metrics["accuracy_on_labeled_rows_parse_failures_count_wrong"] == 1 / 3
    assert metrics["coverage_on_labeled_rows"] == 2 / 3
    assert metrics["wrong_sign_count"] == 1
    assert metrics["abstention_count_on_labeled_rows"] == 1
    assert metrics["selective_accuracy_conditioned_on_answering_sign"] == 1 / 2
    assert metrics["confusion_matrix_oracle_rows_vs_prediction_columns"]["-1"]["UNKNOWN"] == 1


def test_shuffle_control_always_uses_a_different_sequence():
    sequence_ids = ["seq-a", "seq-a", "seq-b", "seq-c", "seq-b", "seq-c"]

    donors = different_sequence_donors(sequence_ids)

    assert np.array_equal(donors, different_sequence_donors(sequence_ids))
    assert all(sequence_ids[index] != sequence_ids[donor]
               for index, donor in enumerate(donors))


def test_paired_bootstrap_uses_same_rows_for_real_and_shuffled():
    reference = np.asarray([1, 1, -1, -1, 1], dtype=np.int8)
    real = np.asarray([1, 1, -1, -1, 0], dtype=np.int8)
    shuffled = np.asarray([1, -1, -1, -1, 0], dtype=np.int8)

    first = paired_bootstrap_accuracy_difference(real, shuffled, reference, replicates=1000)
    second = paired_bootstrap_accuracy_difference(real, shuffled, reference, replicates=1000)

    assert first == second
    assert first["rows"] == 5
    assert first["accuracy_difference_real_minus_shuffled"] == pytest.approx(0.2)
    assert first["paired_correctness_cells"]["real_only_correct"] == 1
    assert first["paired_correctness_cells"]["shuffled_only_correct"] == 0


def test_hinge_bootstrap_is_paired_stratified_and_reports_requested_intervals():
    reference = np.asarray([1, 1, -1, -1, 1, -1], dtype=np.int8)
    real = np.asarray([1, 0, -1, 1, -1, -1], dtype=np.int8)
    shuffled = np.asarray([1, -1, 0, -1, 1, -1], dtype=np.int8)

    first = paired_bootstrap_hinge_metrics(real, shuffled, reference, replicates=500)
    second = paired_bootstrap_hinge_metrics(real, shuffled, reference, replicates=500)

    assert first == second
    assert first["class_counts"] == {1: 3, -1: 3}
    assert "real_effective_accuracy" in first["percentile_95_ci"]
    assert "real_wrong_sign_rate" in first["percentile_95_ci"]
    assert "real_balanced_accuracy" in first["percentile_95_ci"]
    assert "real_minus_shuffled_effective_accuracy" in first["percentile_95_ci"]


def test_summary_keeps_docs45_populations_separate_and_pools_hinge_rows():
    reference = np.tile(np.asarray([1, 1, -1, -1], dtype=np.int8)[:, None],
                        (1, len(SIGN_FIELD_NAMES)))
    records = []
    for row in range(len(reference)):
        real = reference[row].copy()
        shuffled = reference[row].copy()
        if row == 1:
            real[0] = 0
        if row == 2:
            shuffled[0] *= -1
        records.append({
            "real": {"state": real.tolist(), "valid": True},
            "shuffled": {"state": shuffled.tolist(), "valid": True},
        })
    rows = np.arange(len(reference), dtype=np.int64)
    hinge_fields = [field for field in SIGN_FIELD_NAMES if field.endswith("_forward_bend")]
    c_readable_wrong = {field: rows for field in hinge_fields}
    c_h0_unknown = {field: rows[:2] for field in hinge_fields}

    result = summarize_predictions(records, reference, c_readable_wrong, c_h0_unknown)

    assert result["populations"]["C_READABLE_WRONG"]["hinge_chain_row_count"] == 16
    assert result["populations"]["C_H0_UNKNOWN"]["hinge_chain_row_count"] == 8
    assert "pooled_hinge" in result["populations"]["C_H0_UNKNOWN"]["by_mode"]["real"]
    assert "C_H0_UNKNOWN::left_elbow_forward_bend" in result["paired_real_vs_shuffled"]
    assert "uncertainty" in result["paired_real_vs_shuffled"]["C_H0_UNKNOWN::pooled_hinge"]
