import hashlib
from contextlib import nullcontext
import numpy as np
import pytest

from framepose.sign_advisor import parse_response, prompt_text
from framepose.signs import SIGN_FIELD_NAMES, UNKNOWN
from scripts.evaluate_current_sign_advisor import (
    EXPECTED_MODEL_ID,
    EXPECTED_MODEL_REVISION,
    EXPECTED_PROMPT_SHA256,
    EXPECTED_SEED,
    EXPECTED_WEIGHT_FINGERPRINT,
    EXPECTED_MAX_NEW_TOKENS,
    _fixed_frame_batches,
    different_sequence_donors,
    field_metrics,
    paired_bootstrap_accuracy_difference,
    paired_bootstrap_hinge_metrics,
    summarize_predictions,
    validate_frozen_backend,
)
from framepose.sign_advisor import ADVISOR_CROP_RESOLUTION
from scripts.sign_advisor_batching import (
    OrderedBatchPreparer,
    PreparedPairBatch,
    compare_output_rows,
    crop_sha256,
    generate_prepared_batch,
    interleave_pairs,
    prefetch_ordered_batches,
    split_interleaved,
    validate_resume_identity,
)


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


def test_real_and_shuffled_requests_keep_receiver_pair_order():
    real = ["real-a", "real-b", "real-c"]
    shuffled = ["shuffled-a", "shuffled-b", "shuffled-c"]

    interleaved = interleave_pairs(real, shuffled)
    restored_real, restored_shuffled = split_interleaved(interleaved)

    assert interleaved == ["real-a", "shuffled-a", "real-b", "shuffled-b",
                           "real-c", "shuffled-c"]
    assert restored_real == real
    assert restored_shuffled == shuffled
    assert list(_fixed_frame_batches(range(7), 3)) == [
        [0, 1, 2], [3, 4, 5], [6]]


def test_crop_digest_matches_sequential_c_order_rgb_hash_without_mutation():
    crop = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)[:, ::-1]
    before = crop.copy()

    assert crop_sha256(crop) == hashlib.sha256(crop.tobytes()).hexdigest()
    assert np.array_equal(crop, before)


def test_prefetched_processor_batches_remain_bounded_and_in_receiver_order():
    class FakeProcessor:
        def __call__(self, *, text, images, return_tensors):
            assert return_tensors == "pt"
            return {"text": text, "pixels": [image.getpixel((0, 0)) for image in images]}

    def read_pair(entry):
        real = np.full((448, 448, 3), entry + 1, dtype=np.uint8)
        shuffled = np.full((448, 448, 3), entry + 101, dtype=np.uint8)
        return real, shuffled

    preparer = OrderedBatchPreparer(read_pair, FakeProcessor(), "same frozen prompt")
    try:
        batches = list(prefetch_ordered_batches([[0, 1], [2]], preparer))
    finally:
        preparer.close()

    assert [batch.entries for batch in batches] == [[0, 1], [2]]
    assert [batch.inputs["pixels"] for batch in batches] == [
        [(1, 1, 1), (101, 101, 101), (2, 2, 2), (102, 102, 102)],
        [(3, 3, 3), (103, 103, 103)],
    ]
    assert all(len(batch.crop_hash_pairs) == len(batch.entries) for batch in batches)


def test_batched_output_equivalence_reports_raw_parse_and_state_separately():
    reference = {
        "sample-a": {
            "real": {"raw": '{"sign":"left"}', "valid": True,
                     "reason": None, "state": [1]},
            "shuffled": {"raw": "malformed", "valid": False,
                         "reason": "invalid json", "state": [0]},
        },
    }
    candidate = [{
        "sample_id": "sample-a",
        "real": {"raw": '{ "sign": "left" }', "valid": True,
                 "reason": None, "state": [1]},
        "shuffled": {"raw": "malformed", "valid": False,
                     "reason": "invalid json", "state": [0]},
    }]

    result = compare_output_rows(reference, candidate)

    assert result["rows"] == 1
    assert result["requests"] == 2
    assert result["raw_exact_match_rate"] == 0.5
    assert result["parse_status_exact_match_rate"] == 1.0
    assert result["state_exact_match_rate"] == 1.0
    assert result["evaluation_relevant_exact_match_rate"] == 1.0
    assert result["raw_mismatches"] == [{"sample_id": "sample-a", "mode": "real"}]
    assert result["evaluation_relevant_mismatches"] == []


def test_resume_identity_pins_batch_mode_and_equivalence_evidence():
    identity = {
        "record_type": "run_identity",
        "run_started_utc": "first start",
        "execution_mode": "batched_pair",
        "frame_batch_size": 8,
        "request_batch_size": 16,
        "equivalence_report_sha256": "evidence-sha",
    }

    validate_resume_identity(identity, {**identity, "run_started_utc": "resume"})
    with pytest.raises(ValueError, match="frame_batch_size"):
        validate_resume_identity(identity, {**identity, "frame_batch_size": 4})
    with pytest.raises(ValueError, match="equivalence_report_sha256"):
        validate_resume_identity(identity, {
            **identity, "equivalence_report_sha256": "different-evidence"})


def test_batched_generation_keeps_malformed_responses_as_parser_failures():
    class FakeTensor:
        shape = (2, 3)

        def __getitem__(self, _):
            return self

    class FakeInputs(dict):
        def to(self, _device):
            return self

    class FakeProcessor:
        def batch_decode(self, _tokens, skip_special_tokens):
            assert skip_special_tokens is True
            return ["not json", '{"not_a_sign_field": "left"}']

    class FakeModel:
        device = "cpu"

        def generate(self, **kwargs):
            assert kwargs["do_sample"] is False
            assert kwargs["max_new_tokens"] == EXPECTED_MAX_NEW_TOKENS
            return FakeTensor()

    class FakeCuda:
        @staticmethod
        def is_available():
            return False

    class FakeTorch:
        cuda = FakeCuda()

        @staticmethod
        def inference_mode():
            return nullcontext()

    prepared = PreparedPairBatch(
        entries=[(0, 1, 2)], crop_pairs=[], crop_hash_pairs=[],
        inputs=FakeInputs(input_ids=FakeTensor()), crop_read_seconds=0.0,
        pil_conversion_seconds=0.0, processor_seconds=0.0)

    parsed, raw, _timings = generate_prepared_batch(
        prepared, processor=FakeProcessor(), model=FakeModel(), torch=FakeTorch(),
        parse_response=parse_response,
        fields=SIGN_FIELD_NAMES, max_new_tokens=EXPECTED_MAX_NEW_TOKENS)

    assert raw == ["not json", '{"not_a_sign_field": "left"}']
    assert len(parsed) == 2
    assert all(not response.valid for response in parsed)
    assert all(np.all(response.state == 0) for response in parsed)


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
            "real": {"state": real.tolist(), "valid": row != 3},
            "shuffled": {"state": shuffled.tolist(), "valid": row != 2},
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
    paired = result["paired_real_vs_shuffled"]["all_readable_test::torso_facing"]
    assert paired["real_metrics"]["strict_parse_failures"] == 1
    assert paired["shuffled_metrics"]["strict_parse_failures"] == 1
    pooled = result["paired_real_vs_shuffled"]["all_readable_test::pooled_hinge"]
    assert pooled["real_metrics"]["strict_parse_failures"] == 4
    assert pooled["shuffled_metrics"]["strict_parse_failures"] == 4
