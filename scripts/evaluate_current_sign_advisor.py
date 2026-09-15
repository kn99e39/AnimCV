#!/usr/bin/env python3
"""Full-test, strict-contract audit of the currently configured Sign Advisor.

The frozen combined prompt is evaluated on the same receiver rows with their
real crop and a deterministic different-sequence donor crop. Strict parse
failures remain UNKNOWN sensor outputs. No prompt, crop, decoder, or model
tuning is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common.serialization import write_json
import diagnose_pose_reconciliation_attribution as attribution
from framepose.bank import load_bank
from framepose.features import read_crop
from sign_advisor_batching import (
    OrderedBatchPreparer,
    generate_prepared_batch,
    prefetch_ordered_batches,
    split_interleaved,
    validate_resume_identity,
)
from framepose.sign_advisor import (
    ADVISOR_CROP_RESOLUTION,
    PROMPT_SCHEMA_VERSION,
    parse_response,
    prompt_provenance,
    prompt_text,
    resolve_snapshot,
)
from framepose.signs import (
    SIGN_FIELD_NAMES,
    UNKNOWN,
    oracle_sign_states,
    sign_state,
)
import framepose.features as feature_module
import framepose.sign_advisor as sign_advisor_module


BOOTSTRAP_REPLICATES = 10_000
EXPECTED_BANK_DIGEST = "75519e6394a764e3749ddaa30555b58b73a01db582ccee14f661374b9a0ed536"
EXPECTED_H0_SHA256 = "6df04c2031b2bde6a4f0af37fdae0a7f0228e9fc23401e15e1aa2b3bd35695a5"
EXPECTED_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
EXPECTED_MODEL_REVISION = "895c3a49bc3fa70a340399125c650a463535e71c"
EXPECTED_WEIGHT_FINGERPRINT = "840bc66b30f80632ade2f1fd6f415c34bb2f89ff0460ecee98a572c8d77eabc0"
EXPECTED_PROMPT_SHA256 = "4529750c6835fbf4ae11b7d2874cdbd61d6ed9ec5e4ca8d13062d2783405446f"
EXPECTED_ADVISOR_SOURCE_SHA256 = "16892c8ed7e7de161e4eefb42f739dd6f3913b1c71f94a6f0db93030e098f99f"
EXPECTED_FEATURE_SOURCE_SHA256 = "c0ae1f94e1e2086cd6d5a0296be567c3628072e5e374430643a06b48f90348fa"
EXPECTED_DOCKERFILE_SHA256 = "73f2315c5eeb8ea493b3645e5f3e9aa6286a62f306c6bd73d16fb698317f140a"
EXPECTED_MAX_NEW_TOKENS = 160
EXPECTED_SEED = 1337


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def different_sequence_donors(sequence_ids: list[str]) -> np.ndarray:
    """Assign each row a deterministic crop donor from another sequence."""
    groups: dict[str, list[int]] = {}
    for order, sequence_id in enumerate(sequence_ids):
        groups.setdefault(sequence_id, []).append(order)
    sequences = sorted(groups)
    if len(sequences) < 2:
        raise ValueError("the shuffled-image control requires at least two sequences")
    donors = np.empty(len(sequence_ids), dtype=np.int64)
    for sequence_index, sequence_id in enumerate(sequences):
        donor_sequence = sequences[(sequence_index + 1) % len(sequences)]
        donor_orders = groups[donor_sequence]
        for local_index, receiver_order in enumerate(groups[sequence_id]):
            donors[receiver_order] = donor_orders[local_index % len(donor_orders)]
    if any(sequence_ids[index] == sequence_ids[int(donor)]
           for index, donor in enumerate(donors)):
        raise AssertionError("a shuffled crop donor came from the receiver sequence")
    return donors


def validate_frozen_backend(model_id: str, revision: str,
                            snapshot: dict[str, Any] | None,
                            prompt_sha256: str, *, max_new_tokens: int,
                            seed: int, crop_resolution: int) -> None:
    """Refuse any backend, prompt, crop, or decoding drift from the adopted run."""
    if model_id != EXPECTED_MODEL_ID or revision != EXPECTED_MODEL_REVISION:
        raise ValueError("model identity differs from the currently adopted backend")
    if (snapshot is None or snapshot.get("resolved_commit") != EXPECTED_MODEL_REVISION
            or snapshot.get("weight_fingerprint") != EXPECTED_WEIGHT_FINGERPRINT
            or snapshot.get("file_count") != 11):
        raise ValueError("model snapshot/fingerprint differs from the recovered frozen weights")
    if prompt_sha256 != EXPECTED_PROMPT_SHA256:
        raise ValueError("frozen Sign Advisor prompt hash changed")
    if max_new_tokens != EXPECTED_MAX_NEW_TOKENS or seed != EXPECTED_SEED:
        raise ValueError("generation settings differ from the adopted Sign Advisor configuration")
    if crop_resolution != ADVISOR_CROP_RESOLUTION or crop_resolution != 448:
        raise ValueError("advisor crop resolution differs from the frozen contract")


def _entropy(predicted: np.ndarray) -> float:
    counts = np.asarray([(predicted == value).sum() for value in (-1, 0, 1)], dtype=float)
    total = float(counts.sum())
    if total == 0:
        return 0.0
    probabilities = counts[counts > 0] / total
    return float(-np.sum(probabilities * np.log2(probabilities)))


def field_metrics(predicted: np.ndarray, parsed_valid: np.ndarray,
                  reference: np.ndarray) -> dict[str, Any]:
    """Count invalid/UNKNOWN sensor answers as failure on every labeled row."""
    predicted = np.asarray(predicted, dtype=np.int8)
    parsed_valid = np.asarray(parsed_valid, dtype=bool)
    reference = np.asarray(reference, dtype=np.int8)
    scored = reference != UNKNOWN
    truth, guess = reference[scored], predicted[scored]
    parse_valid_labeled = parsed_valid[scored]
    if len(guess):
        positive, negative = int((truth == 1).sum()), int((truth == -1).sum())
        majority = max(positive, negative) / len(truth)
        recall_positive = float((guess[truth == 1] == 1).mean()) if positive else None
        recall_negative = float((guess[truth == -1] == -1).mean()) if negative else None
        balanced = (float(np.mean([recall_positive, recall_negative]))
                    if recall_positive is not None and recall_negative is not None else None)
        accuracy = float((guess == truth).mean())
        coverage = float((guess != UNKNOWN).mean())
        correct = int((guess == truth).sum())
        wrong_sign = int((guess == -truth).sum())
        abstentions = int((guess == UNKNOWN).sum())
        selective_accuracy = float(correct / (len(guess) - abstentions)) if len(guess) > abstentions else None
        confusion = {
            str(label): {
                prediction_name: int(((truth == label) & (guess == prediction_value)).sum())
                for prediction_name, prediction_value in
                (("+1", 1), ("-1", -1), ("UNKNOWN", UNKNOWN))
            }
            for label in (1, -1)
        }
    else:
        positive = negative = 0
        majority = accuracy = coverage = balanced = None
        recall_positive = recall_negative = None
        correct = wrong_sign = abstentions = 0
        selective_accuracy = None
        confusion = {str(label): {name: 0 for name in ("+1", "-1", "UNKNOWN")}
                     for label in (1, -1)}
    counts = {str(value): int((predicted == value).sum()) for value in (-1, 0, 1)}
    return {
        "rows": int(len(reference)),
        "reference_positive": int((reference == 1).sum()),
        "reference_negative": int((reference == -1).sum()),
        "reference_unknown": int((reference == UNKNOWN).sum()),
        "labeled_rows": int(scored.sum()),
        "parsed_valid_rows": int(parsed_valid.sum()),
        "strict_parse_failures": int((~parsed_valid).sum()),
        "strict_parse_failure_rate_all_rows": float((~parsed_valid).mean()) if len(parsed_valid) else None,
        "strict_parse_success_rate_all_rows": float(parsed_valid.mean()) if len(parsed_valid) else None,
        "strict_parse_failures_on_oracle_readable_rows": int((~parse_valid_labeled).sum()),
        "strict_parse_success_rate_on_oracle_readable_rows": (
            float(parse_valid_labeled.mean()) if len(parse_valid_labeled) else None),
        "predicted_distribution": counts,
        "answer_entropy_bits": _entropy(predicted),
        "coverage_on_labeled_rows": coverage,
        "abstention_count_on_labeled_rows": abstentions,
        "abstention_rate_on_labeled_rows": (
            float(abstentions / len(guess)) if len(guess) else None),
        "correct_count": correct,
        "wrong_sign_count": wrong_sign,
        "wrong_sign_rate_over_oracle_readable_rows": (
            float(wrong_sign / len(guess)) if len(guess) else None),
        "wrong_sign_rate_conditioned_on_covered_rows": (
            float(wrong_sign / (len(guess) - abstentions))
            if len(guess) > abstentions else None),
        "accuracy_on_labeled_rows_parse_failures_count_wrong": accuracy,
        "effective_accuracy": accuracy,
        "selective_accuracy_conditioned_on_answering_sign": selective_accuracy,
        "balanced_accuracy_on_labeled_rows": balanced,
        "confusion_matrix_oracle_rows_vs_prediction_columns": confusion,
        "positive_recall": recall_positive,
        "negative_recall": recall_negative,
        "majority_class_baseline": majority,
    }


def paired_bootstrap_accuracy_difference(real_prediction: np.ndarray,
                                         shuffled_prediction: np.ndarray,
                                         reference: np.ndarray,
                                         *, seed: int = 20260915,
                                         replicates: int = BOOTSTRAP_REPLICATES
                                         ) -> dict[str, Any]:
    """Paired bootstrap CI from the four matched correctness cells."""
    real_prediction = np.asarray(real_prediction, dtype=np.int8)
    shuffled_prediction = np.asarray(shuffled_prediction, dtype=np.int8)
    reference = np.asarray(reference, dtype=np.int8)
    mask = reference != UNKNOWN
    truth = reference[mask]
    real_correct = real_prediction[mask] == truth
    shuffled_correct = shuffled_prediction[mask] == truth
    n = int(len(truth))
    if n == 0:
        return {"rows": 0, "accuracy_difference_real_minus_shuffled": None}
    cells = np.asarray([
        np.sum(real_correct & shuffled_correct),
        np.sum(real_correct & ~shuffled_correct),
        np.sum(~real_correct & shuffled_correct),
        np.sum(~real_correct & ~shuffled_correct),
    ], dtype=np.int64)
    rng = np.random.default_rng(seed)
    draw = rng.multinomial(n, cells / n, size=replicates)
    differences = (draw[:, 1] - draw[:, 2]) / n
    point = float(real_correct.mean() - shuffled_correct.mean())
    return {
        "rows": n,
        "paired_correctness_cells": {
            "both_correct": int(cells[0]),
            "real_only_correct": int(cells[1]),
            "shuffled_only_correct": int(cells[2]),
            "both_incorrect": int(cells[3]),
        },
        "accuracy_difference_real_minus_shuffled": point,
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "percentile_95_ci": [float(np.percentile(differences, 2.5)),
                              float(np.percentile(differences, 97.5))],
    }


def paired_bootstrap_hinge_metrics(real_prediction: np.ndarray,
                                  shuffled_prediction: np.ndarray,
                                  reference: np.ndarray,
                                  *, seed: int = 20260915,
                                  replicates: int = BOOTSTRAP_REPLICATES
                                  ) -> dict[str, Any]:
    """Stratified paired bootstrap for effective, wrong-sign, and balanced scores."""
    real_prediction = np.asarray(real_prediction, dtype=np.int8)
    shuffled_prediction = np.asarray(shuffled_prediction, dtype=np.int8)
    reference = np.asarray(reference, dtype=np.int8)
    mask = reference != UNKNOWN
    truth = reference[mask]
    real = real_prediction[mask]
    shuffled = shuffled_prediction[mask]
    n = len(truth)
    class_counts = {label: int(np.sum(truth == label)) for label in (1, -1)}
    if not n or not all(class_counts.values()):
        return {"rows": n, "class_counts": class_counts,
                "method": "stratified paired oracle-class bootstrap",
                "percentile_95_ci": {}}

    rng = np.random.default_rng(seed)
    category_predictions = (-1, UNKNOWN, 1)
    class_draws: dict[int, np.ndarray] = {}
    observed_cells: dict[str, Any] = {}
    for label in (1, -1):
        class_mask = truth == label
        cells = np.asarray([
            np.sum(class_mask & (real == real_value) & (shuffled == shuffled_value))
            for real_value in category_predictions
            for shuffled_value in category_predictions
        ], dtype=np.int64)
        count = class_counts[label]
        draws = rng.multinomial(count, cells / count, size=replicates)
        class_draws[label] = draws.reshape(replicates, 3, 3)
        observed_cells[str(label)] = {
            f"real_{real_value}|shuffled_{shuffled_value}": int(cells[real_index * 3 + shuffled_index])
            for real_index, real_value in enumerate(category_predictions)
            for shuffled_index, shuffled_value in enumerate(category_predictions)
        }

    boot: dict[str, dict[str, np.ndarray]] = {
        mode: {metric: np.zeros(replicates, dtype=np.float64)
               for metric in ("effective_accuracy", "wrong_sign_rate", "balanced_accuracy")}
        for mode in ("real", "shuffled")
    }
    for mode in ("real", "shuffled"):
        correct_by_class = {}
        wrong_by_class = {}
        for label in (1, -1):
            draws = class_draws[label]
            correct_index = category_predictions.index(label)
            wrong_index = category_predictions.index(-label)
            if mode == "real":
                correct_by_class[label] = draws[:, correct_index, :].sum(axis=1)
                wrong_by_class[label] = draws[:, wrong_index, :].sum(axis=1)
            else:
                correct_by_class[label] = draws[:, :, correct_index].sum(axis=1)
                wrong_by_class[label] = draws[:, :, wrong_index].sum(axis=1)
        total_correct = correct_by_class[1] + correct_by_class[-1]
        total_wrong = wrong_by_class[1] + wrong_by_class[-1]
        boot[mode]["effective_accuracy"] = total_correct / n
        boot[mode]["wrong_sign_rate"] = total_wrong / n
        boot[mode]["balanced_accuracy"] = 0.5 * (
            correct_by_class[1] / class_counts[1]
            + correct_by_class[-1] / class_counts[-1])

    differences = {metric: boot["real"][metric] - boot["shuffled"][metric]
                   for metric in boot["real"]}

    def interval(values: np.ndarray) -> list[float]:
        return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]

    return {
        "rows": n,
        "class_counts": class_counts,
        "method": "stratified paired oracle-class bootstrap; percentile 95% intervals",
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "paired_joint_prediction_cells": observed_cells,
        "percentile_95_ci": {
            f"{mode}_{metric}": interval(boot[mode][metric])
            for mode in ("real", "shuffled") for metric in boot[mode]
        } | {f"real_minus_shuffled_{metric}": interval(values)
             for metric, values in differences.items()},
    }


def _paired_control(real: np.ndarray, shuffled: np.ndarray, reference: np.ndarray,
                    *, real_valid: np.ndarray | None = None,
                    shuffled_valid: np.ndarray | None = None,
                    seed: int, bootstrap: bool) -> dict[str, Any]:
    real = np.asarray(real, dtype=np.int8)
    shuffled = np.asarray(shuffled, dtype=np.int8)
    reference = np.asarray(reference, dtype=np.int8)
    if real_valid is None:
        real_valid = np.ones(len(real), dtype=bool)
    if shuffled_valid is None:
        shuffled_valid = np.ones(len(shuffled), dtype=bool)
    real_valid = np.asarray(real_valid, dtype=bool)
    shuffled_valid = np.asarray(shuffled_valid, dtype=bool)
    if len(real_valid) != len(real) or len(shuffled_valid) != len(shuffled):
        raise ValueError("paired validity arrays must align with paired predictions")
    real_metrics = field_metrics(real, real_valid, reference)
    shuffled_metrics = field_metrics(shuffled, shuffled_valid, reference)
    scored = reference != UNKNOWN
    result = {
        "rows": int(len(reference)),
        "oracle_readable_rows": int(scored.sum()),
        "real_image_accuracy": real_metrics["effective_accuracy"],
        "shuffled_image_accuracy": shuffled_metrics["effective_accuracy"],
        "real_image_balanced_accuracy": real_metrics["balanced_accuracy_on_labeled_rows"],
        "shuffled_image_balanced_accuracy": shuffled_metrics["balanced_accuracy_on_labeled_rows"],
        "output_change_rate_real_vs_shuffled_all_rows": (
            float(np.mean(real != shuffled)) if len(real) else None),
        "coverage_change_real_minus_shuffled": (
            real_metrics["coverage_on_labeled_rows"]
            - shuffled_metrics["coverage_on_labeled_rows"]
            if scored.any() else None),
        "wrong_sign_rate_change_real_minus_shuffled": (
            real_metrics["wrong_sign_rate_over_oracle_readable_rows"]
            - shuffled_metrics["wrong_sign_rate_over_oracle_readable_rows"]
            if scored.any() else None),
        "real_metrics": real_metrics,
        "shuffled_metrics": shuffled_metrics,
    }
    if bootstrap:
        result["uncertainty"] = paired_bootstrap_hinge_metrics(
            real, shuffled, reference, seed=seed)
    else:
        result["accuracy_difference_uncertainty"] = paired_bootstrap_accuracy_difference(
            real, shuffled, reference, seed=seed)
    return result


def summarize_predictions(records: list[dict[str, Any]],
                          reference: np.ndarray,
                          c_readable_wrong: dict[str, np.ndarray],
                          c_h0_unknown: dict[str, np.ndarray]) -> dict[str, Any]:
    result: dict[str, Any] = {"populations": {}, "paired_real_vs_shuffled": {}}
    modes = ("real", "shuffled")
    prediction_matrices = {
        mode: np.asarray([record[mode]["state"] for record in records], dtype=np.int8)
        for mode in modes
    }
    valid_matrices = {
        mode: np.asarray([record[mode]["valid"] for record in records], dtype=bool)
        for mode in modes
    }
    all_rows = np.arange(len(records), dtype=np.int64)
    populations: dict[str, dict[str, np.ndarray]] = {
        "all_readable_test": {field: all_rows for field in SIGN_FIELD_NAMES},
        "C_READABLE_WRONG": c_readable_wrong,
        "C_H0_UNKNOWN": c_h0_unknown,
    }
    hinge_fields = [field for field in SIGN_FIELD_NAMES if field.endswith("_forward_bend")]
    for population_name, rows_by_field in populations.items():
        population: dict[str, Any] = {
            "unique_frame_count": (len(records) if population_name == "all_readable_test" else None),
            "hinge_chain_row_count": int(sum(len(rows_by_field[field]) for field in hinge_fields
                                             if field in rows_by_field)),
            "by_mode": {mode: {} for mode in modes},
        }
        for field, rows in rows_by_field.items():
            column = SIGN_FIELD_NAMES.index(field)
            rows = np.asarray(rows, dtype=np.int64)
            for mode in modes:
                population["by_mode"][mode][field] = field_metrics(
                    prediction_matrices[mode][rows, column],
                    valid_matrices[mode][rows], reference[rows, column])
            is_hinge_cohort = population_name != "all_readable_test"
            key = f"{population_name}::{field}"
            result["paired_real_vs_shuffled"][key] = _paired_control(
                prediction_matrices["real"][rows, column],
                prediction_matrices["shuffled"][rows, column],
                reference[rows, column],
                real_valid=valid_matrices["real"][rows],
                shuffled_valid=valid_matrices["shuffled"][rows],
                seed=20260915 + column
                + (100 if population_name == "C_READABLE_WRONG" else
                   200 if population_name == "C_H0_UNKNOWN" else 0),
                bootstrap=is_hinge_cohort)
        pooled_rows = [np.asarray(rows_by_field[field], dtype=np.int64)
                       for field in hinge_fields if field in rows_by_field]
        if pooled_rows:
            pooled_real, pooled_shuffled, pooled_reference = [], [], []
            pooled_real_valid, pooled_shuffled_valid = [], []
            for field, rows in ((field, np.asarray(rows_by_field[field], dtype=np.int64))
                                for field in hinge_fields if field in rows_by_field):
                column = SIGN_FIELD_NAMES.index(field)
                pooled_real.extend(prediction_matrices["real"][rows, column].tolist())
                pooled_shuffled.extend(prediction_matrices["shuffled"][rows, column].tolist())
                pooled_reference.extend(reference[rows, column].tolist())
                pooled_real_valid.extend(valid_matrices["real"][rows].tolist())
                pooled_shuffled_valid.extend(valid_matrices["shuffled"][rows].tolist())
            pooled_real = np.asarray(pooled_real, dtype=np.int8)
            pooled_shuffled = np.asarray(pooled_shuffled, dtype=np.int8)
            pooled_reference = np.asarray(pooled_reference, dtype=np.int8)
            pooled_real_valid = np.asarray(pooled_real_valid, dtype=bool)
            pooled_shuffled_valid = np.asarray(pooled_shuffled_valid, dtype=bool)
            population["pooled_hinge_rows"] = int(len(pooled_reference))
            for mode, prediction in (("real", pooled_real), ("shuffled", pooled_shuffled)):
                valid = pooled_real_valid if mode == "real" else pooled_shuffled_valid
                population["by_mode"][mode]["pooled_hinge"] = field_metrics(
                    prediction, valid, pooled_reference)
            result["paired_real_vs_shuffled"][f"{population_name}::pooled_hinge"] = _paired_control(
                pooled_real, pooled_shuffled, pooled_reference,
                real_valid=pooled_real_valid, shuffled_valid=pooled_shuffled_valid,
                seed=20260915 + (100 if population_name == "C_READABLE_WRONG" else
                                 200 if population_name == "C_H0_UNKNOWN" else 300),
                bootstrap=True)
        result["populations"][population_name] = population
    return result


def exact_docs45_c_populations(replay_report: dict[str, Any],
                               reference: np.ndarray,
                               h0_signs: np.ndarray
                               ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    expected = {
        "left_elbow_forward_bend": (406, 353),
        "right_elbow_forward_bend": (306, 285),
        "left_knee_forward_bend": (268, 324),
        "right_knee_forward_bend": (410, 314),
    }
    readable_wrong: dict[str, np.ndarray] = {}
    h0_unknown: dict[str, np.ndarray] = {}
    for field, expected_counts in expected.items():
        column = SIGN_FIELD_NAMES.index(field)
        rows = attribution._cohort_rows_from_report(replay_report, "C", field)
        partition = attribution.partition_c_rows(
            rows, reference[rows, column], h0_signs[rows, column])
        actual = (len(partition["readable_wrong"]), len(partition["h0_unknown"]))
        if actual != expected_counts:
            raise ValueError(f"docs/45 C partition changed for {field}: {actual}")
        readable_wrong[field] = partition["readable_wrong"]
        h0_unknown[field] = partition["h0_unknown"]
    if sum(map(len, readable_wrong.values())) != 1390:
        raise AssertionError("docs/45 C_READABLE_WRONG pooled identity changed")
    if sum(map(len, h0_unknown.values())) != 1276:
        raise AssertionError("docs/45 C_H0_UNKNOWN pooled identity changed")
    return readable_wrong, h0_unknown


def _write_jsonl_line(stream, value: dict[str, Any]) -> None:
    stream.write(json.dumps(value, sort_keys=True, separators=(",", ":"),
                            ensure_ascii=False, allow_nan=False) + "\n")
    stream.flush()


def _fixed_frame_batches(entries, batch_size: int):
    """Yield bounded receiver chunks while preserving the frozen test order."""
    chunk = []
    for entry in entries:
        chunk.append(entry)
        if len(chunk) == batch_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _load_equivalence_evidence(path: Path, batch_size: int, *, resume: bool,
                              evaluator_sha256: str, batching_sha256: str
                               ) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise ValueError("batched inference requires a saved throughput/equivalence report")
    digest = _sha256(path)
    evidence = json.loads(path.read_text(encoding="utf-8"))
    if evidence.get("evaluator_script_sha256") != evaluator_sha256:
        raise ValueError("equivalence evidence was produced by a different evaluator revision")
    if evidence.get("batching_helper_sha256") != batching_sha256:
        raise ValueError("equivalence evidence was produced by a different batching helper")
    if evidence.get("selected_frame_batch_size") != batch_size:
        raise ValueError("equivalence evidence was recorded for a different frame batch size")
    equivalence = evidence.get("selected_equivalence", {})
    if equivalence.get("rows", 0) < 64 or equivalence.get("requests", 0) < 128:
        raise ValueError("batched path requires the frozen 64-frame/two-mode equivalence control")
    if not evidence.get("selected_batch_safe") or not evidence.get("selected_material_gain"):
        raise ValueError("selected batch lacks safe-memory and material-throughput evidence")
    if resume and equivalence.get("raw_exact_match_rate") != 1.0:
        raise ValueError("non-byte-identical batched output cannot resume a partial run")
    return evidence, digest


def run(args) -> dict[str, Any]:
    import torch
    import transformers
    from PIL import Image
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    if transformers.__version__ != "4.49.0":
        raise ValueError(f"Sign Advisor requires frozen transformers 4.49.0, got {transformers.__version__}")

    frame_batch_size = getattr(args, "frame_batch_size", 1)
    if frame_batch_size < 1:
        raise ValueError("frame batch size must be positive")
    equivalence_evidence = None
    equivalence_sha = None
    if frame_batch_size > 1:
        evidence_path = getattr(args, "equivalence_report", None)
        if evidence_path is None:
            raise ValueError("batched inference requires --equivalence-report")
        equivalence_evidence, equivalence_sha = _load_equivalence_evidence(
            evidence_path, frame_batch_size, resume=args.resume,
            evaluator_sha256=_sha256(Path(__file__).resolve()),
            batching_sha256=_sha256(Path(__file__).with_name("sign_advisor_batching.py")))
    elif getattr(args, "equivalence_report", None) is not None:
        raise ValueError("--equivalence-report is only valid for a batched execution")

    if args.out.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite existing run: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    roots = dict(value.split("=", 1) for value in args.image_root)
    bank = load_bank(args.bank)
    positions = bank.indices("test")
    if len(positions) != 7076:
        raise ValueError(f"expected the docs/44 test population 7076, got {len(positions)}")
    if bank.content_digest() != EXPECTED_BANK_DIGEST:
        raise ValueError("FrameBank digest differs from the docs/44 recovered artifact")
    if _sha256(args.h0_prediction) != EXPECTED_H0_SHA256:
        raise ValueError("H0 prediction bytes differ from the docs/44 O_BILATERAL artifact")
    replay_sha = _sha256(args.replay_report)
    if replay_sha != attribution.EXPECTED_REPLAY_SHA256:
        raise ValueError(f"docs/44 replay SHA mismatch: {replay_sha}")
    replay_report = json.loads(args.replay_report.read_text(encoding="utf-8"))
    attribution_report = json.loads(args.attribution_report.read_text(encoding="utf-8"))
    attribution_sha = _sha256(args.attribution_report)
    if (attribution_report.get("identity", {}).get("docs44_replay_sha256") != replay_sha
            or attribution_report.get("identity", {}).get("bank_content_digest")
            != EXPECTED_BANK_DIGEST):
        raise ValueError("docs/45 attribution identity differs from the frozen replay or FrameBank")
    reference_all = oracle_sign_states(bank.arrays["target_3d"], bank.arrays["target_valid"])
    reference = reference_all[positions]
    h0 = np.load(args.h0_prediction).astype(np.float64)
    if h0.shape != (len(positions), len(replay_joint_names()), 3):
        raise ValueError(f"H0 prediction shape mismatch: {h0.shape}")
    h0_valid = bank.arrays["target_valid"][positions]
    h0_signs = np.stack([sign_state(pose, valid) for pose, valid in zip(h0, h0_valid)])
    c_readable_wrong, c_h0_unknown = exact_docs45_c_populations(
        replay_report, reference, h0_signs)
    instruction = prompt_text()
    snapshot = resolve_snapshot(args.model, args.hf_cache)
    prompt_sha = hashlib.sha256(instruction.encode("utf-8")).hexdigest()
    validate_frozen_backend(
        args.model, args.revision, snapshot, prompt_sha,
        max_new_tokens=args.max_new_tokens, seed=args.seed,
        crop_resolution=ADVISOR_CROP_RESOLUTION)
    feature_sha = _sha256(Path(feature_module.__file__).resolve())
    advisor_sha = _sha256(Path(sign_advisor_module.__file__).resolve())
    repository_root = Path(sign_advisor_module.__file__).resolve().parents[2]
    dockerfile_sha = _sha256(repository_root / "Dockerfile.signadvisor")
    if feature_sha != EXPECTED_FEATURE_SOURCE_SHA256 or advisor_sha != EXPECTED_ADVISOR_SOURCE_SHA256:
        raise ValueError("crop/Sign Advisor contract source differs from the frozen adopted code")
    if dockerfile_sha != EXPECTED_DOCKERFILE_SHA256:
        raise ValueError("Sign Advisor runtime Dockerfile differs from the frozen environment")
    sequence_ids = [bank.samples[int(position)].sequence_id for position in positions]
    donor_orders = different_sequence_donors(sequence_ids)

    run_manifest_path = args.out / "run_manifest.json"
    jsonl_path = args.out / "full_test_responses.jsonl"
    existing: dict[str, dict[str, Any]] = {}
    if args.resume and jsonl_path.is_file():
        with jsonl_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if row.get("record_type") == "frame":
                    existing[row["sample_id"]] = row

    torch.manual_seed(args.seed)
    processor = AutoProcessor.from_pretrained(args.model, revision=args.revision)
    batch_preparation_processor = (
        AutoProcessor.from_pretrained(args.model, revision=args.revision)
        if frame_batch_size > 1 else processor)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model, revision=args.revision, torch_dtype=torch.float16,
        device_map=args.device)
    model.eval()
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    if parameter_count != 2_208_985_600:
        raise ValueError(f"model parameter count differs from the recovered backend: {parameter_count}")
    generation_config = {
        **model.generation_config.to_dict(),
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "torch_seed": args.seed,
    }
    run_identity = {
        "record_type": "run_identity",
        "schema": "animcv_sign_advisor_full_test_run_v1",
        "evaluation_script_sha256": _sha256(Path(__file__).resolve()),
        "bank_content_digest": bank.content_digest(),
        "test_rows": int(len(positions)),
        "docs44_replay_sha256": replay_sha,
        "docs45_attribution_sha256": attribution_sha,
        "docs45_population_counts": {
            "C_READABLE_WRONG": {field: int(len(rows))
                                  for field, rows in c_readable_wrong.items()},
            "C_H0_UNKNOWN": {field: int(len(rows))
                             for field, rows in c_h0_unknown.items()},
        },
        "model_id": args.model,
        "resolved_model_commit": snapshot["resolved_commit"],
        "snapshot_path": snapshot["snapshot_dir"],
        "weight_fingerprint": snapshot["weight_fingerprint"],
        "weight_file_count": snapshot["file_count"],
        "weight_files": snapshot["files"],
        "processor_class": f"{processor.__class__.__module__}.{processor.__class__.__name__}",
        "batch_preparation_processor_class": (
            f"{batch_preparation_processor.__class__.__module__}."
            f"{batch_preparation_processor.__class__.__name__}"),
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "dtype": "float16",
        "generation": generation_config,
        "prompt_schema": PROMPT_SCHEMA_VERSION,
        "prompt_sha256": prompt_sha,
        "prompt_provenance": prompt_provenance(),
        "crop_resolution": ADVISOR_CROP_RESOLUTION,
        "crop_source": "framepose.features.read_crop / crop_box contract",
        "crop_source_sha256": feature_sha,
        "sign_advisor_source_sha256": advisor_sha,
        "runtime_dockerfile_sha256": dockerfile_sha,
        "execution_mode": "sequential" if frame_batch_size == 1 else "batched_pair",
        "frame_batch_size": frame_batch_size,
        "request_batch_size": 2 * frame_batch_size,
        "prefetch_crop_workers": 0 if frame_batch_size == 1 else 4,
        "prefetch_max_pending_batches": 0 if frame_batch_size == 1 else 1,
        "equivalence_report_sha256": equivalence_sha,
        "shuffle_rule": (
            "receiver sequence maps to next lexicographic test sequence; donor frames cycle in test order"),
        "strict_parser": "parse_response; invalid responses are UNKNOWN sensor failures",
    }
    if args.resume and jsonl_path.is_file() and jsonl_path.stat().st_size:
        with jsonl_path.open("r", encoding="utf-8") as stream:
            first_line = json.loads(next(stream))
        validate_resume_identity(first_line, run_identity)

    def ask(crop: np.ndarray) -> tuple[Any, str]:
        messages = [{"role": "user", "content": [{"type": "image"},
                                                     {"type": "text", "text": instruction}]}]
        rendered = processor.apply_chat_template(messages, tokenize=False,
                                                 add_generation_prompt=True)
        inputs = processor(text=[rendered], images=[Image.fromarray(crop)],
                           return_tensors="pt").to(model.device)
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                       do_sample=False)
        raw = processor.batch_decode(
            generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        return parse_response(raw, fields=tuple(SIGN_FIELD_NAMES)), raw

    elapsed_started = time.perf_counter()
    processed_this_run = 0
    stage_seconds = {
        "crop_read_seconds": 0.0,
        "pil_conversion_seconds": 0.0,
        "processor_seconds": 0.0,
        "host_to_device_seconds": 0.0,
        "generate_seconds": 0.0,
        "decode_parse_seconds": 0.0,
    }

    def emit_record(entry, crop_hash_pair,
                    real_response, real_raw, shuffled_response, shuffled_raw,
                    stream) -> None:
        nonlocal processed_this_run
        order, position, donor_position = entry
        sample = bank.samples[int(position)]
        donor = bank.samples[int(donor_position)]
        record = {
            "record_type": "frame",
            "bank_position": int(position),
            "test_order": int(order),
            "sample_id": sample.sample_id,
            "sequence_id": sample.sequence_id,
            "frame_index": int(sample.frame_index),
            "image_reference": sample.image_reference.to_dict()
            if sample.image_reference else None,
            "image_digest_real_crop_rgb": crop_hash_pair[0],
            "shuffled_donor_sample_id": donor.sample_id,
            "shuffled_donor_sequence_id": donor.sequence_id,
            "shuffled_donor_image_reference": donor.image_reference.to_dict()
            if donor.image_reference else None,
            "image_digest_shuffled_crop_rgb": crop_hash_pair[1],
            "oracle_sign_state": [int(value) for value in reference[order]],
            "h0_sign_state": [int(value) for value in h0_signs[order]],
            "real": {"valid": bool(real_response.valid),
                     "reason": real_response.reason,
                     "state": [int(value) for value in real_response.state],
                     "raw": real_raw},
            "shuffled": {"valid": bool(shuffled_response.valid),
                         "reason": shuffled_response.reason,
                         "state": [int(value) for value in shuffled_response.state],
                         "raw": shuffled_raw},
        }
        _write_jsonl_line(stream, record)
        existing[sample.sample_id] = record
        processed_this_run += 1
        if processed_this_run % 20 == 0:
            elapsed = max(time.perf_counter() - elapsed_started, 1e-9)
            rate = processed_this_run / elapsed
            print(f"completed {order + 1}/{len(positions)} selected rows; "
                  f"{processed_this_run} new rows at {rate:.3f} rows/s", flush=True)

    with jsonl_path.open("a" if args.resume else "w", encoding="utf-8") as stream:
        if not args.resume or not jsonl_path.stat().st_size:
            run_identity["run_started_utc"] = args.started_utc
            _write_jsonl_line(stream, run_identity)
        entries = (
            (order, int(position), int(positions[int(donor_orders[order])]))
            for order, position in enumerate(positions)
            if bank.samples[int(position)].sample_id not in existing
        )
        if frame_batch_size == 1:
            for entry in entries:
                order, position, donor_position = entry
                real_crop = read_crop(bank, position, roots, ADVISOR_CROP_RESOLUTION)
                shuffled_crop = read_crop(bank, donor_position, roots,
                                          ADVISOR_CROP_RESOLUTION)
                real_response, real_raw = ask(real_crop)
                shuffled_response, shuffled_raw = ask(shuffled_crop)
                emit_record(
                    entry,
                    (hashlib.sha256(real_crop.tobytes()).hexdigest(),
                     hashlib.sha256(shuffled_crop.tobytes()).hexdigest()),
                    real_response, real_raw, shuffled_response, shuffled_raw, stream)
        else:
            messages = [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": instruction}]}]
            rendered_prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)

            def read_pair(entry):
                _, position, donor_position = entry
                return (
                    read_crop(bank, position, roots, ADVISOR_CROP_RESOLUTION),
                    read_crop(bank, donor_position, roots, ADVISOR_CROP_RESOLUTION),
                )

            preparer = OrderedBatchPreparer(
                read_pair, batch_preparation_processor, rendered_prompt, crop_workers=4,
                crop_resolution=ADVISOR_CROP_RESOLUTION)
            try:
                for prepared in prefetch_ordered_batches(
                        _fixed_frame_batches(entries, frame_batch_size), preparer):
                    parsed, raw, timings = generate_prepared_batch(
                        prepared, processor=processor, model=model, torch=torch,
                        parse_response=parse_response, fields=SIGN_FIELD_NAMES,
                        max_new_tokens=args.max_new_tokens)
                    real_parsed, shuffled_parsed = split_interleaved(parsed)
                    real_raw, shuffled_raw = split_interleaved(raw)
                    for index, entry in enumerate(prepared.entries):
                        emit_record(
                            entry, prepared.crop_hash_pairs[index],
                            real_parsed[index], real_raw[index],
                            shuffled_parsed[index], shuffled_raw[index], stream)
                    for name, value in timings.items():
                        stage_seconds[name] += value
            finally:
                preparer.close()

    records = [existing[bank.samples[int(position)].sample_id]
               for position in positions
               if bank.samples[int(position)].sample_id in existing]
    complete = len(records) == len(positions)
    elapsed_seconds = time.perf_counter() - elapsed_started
    report = {
        "schema": "animcv_sign_advisor_full_test_report_v1",
        "run_status": "COMPLETE" if complete else "PARTIAL",
        "frame_count": len(records),
        "expected_test_frame_count": int(len(positions)),
        "bank_content_digest": bank.content_digest(),
        "model": {
            "id": args.model,
            "requested_and_resolved_revision": snapshot["resolved_commit"],
            "snapshot_path": snapshot["snapshot_dir"],
            "weight_fingerprint": snapshot["weight_fingerprint"],
            "file_count": snapshot["file_count"],
            "weight_files": snapshot["files"],
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "dtype": "float16",
            "processor_class": run_identity["processor_class"],
            "parameter_count": parameter_count,
            "generation": generation_config,
        },
        "implementation_identity": {
            "evaluation_script_sha256": _sha256(Path(__file__).resolve()),
            "sign_advisor_source_sha256": advisor_sha,
            "crop_source_sha256": feature_sha,
            "runtime_dockerfile_sha256": dockerfile_sha,
        },
        "prompt": {
            "schema": PROMPT_SCHEMA_VERSION,
            "sha256": prompt_sha,
            "provenance": prompt_provenance(),
        },
        "crop": {"resolution": ADVISOR_CROP_RESOLUTION,
                 "contract": "unchanged framepose read_crop / crop_box",
                 "feature_source_sha256": run_identity["crop_source_sha256"]},
        "source_identity": {
            "bank_content_digest": bank.content_digest(),
            "docs44_replay_sha256": replay_sha,
            "docs45_attribution_sha256": attribution_sha,
            "prediction_sha256": EXPECTED_H0_SHA256,
            "evaluation_scope": "all 7,076 test rows; no answer-dependent subsampling",
        },
        "population_identity": run_identity["docs45_population_counts"],
        "execution": {
            "mode": run_identity["execution_mode"],
            "frame_batch_size": frame_batch_size,
            "request_batch_size": 2 * frame_batch_size,
            "prefetch_crop_workers": run_identity["prefetch_crop_workers"],
            "prefetch_max_pending_batches": run_identity["prefetch_max_pending_batches"],
            "equivalence_report_sha256": equivalence_sha,
            "equivalence_control": equivalence_evidence.get("selected_equivalence")
            if equivalence_evidence else None,
        },
        "performance_this_process": {
            "frames_processed": processed_this_run,
            "vlm_requests_processed": 2 * processed_this_run,
            "elapsed_seconds": elapsed_seconds,
            "frames_per_second": processed_this_run / elapsed_seconds
            if elapsed_seconds else None,
            "vlm_requests_per_second": (2 * processed_this_run) / elapsed_seconds
            if elapsed_seconds else None,
            "stage_seconds": stage_seconds,
        },
        "population_selection": "exact docs/44 C rows partitioned by the unchanged docs/45 canonical H0 SignState",
        "control": {
            "real_vs_shuffled": "same receiver rows, same frozen prompt and exact model weights; only donor crop changes",
            "shuffle_rule": run_identity["shuffle_rule"],
            "parse_failures_count_as_sensor_failures": True,
        },
        "metrics": summarize_predictions(
            records,
            np.asarray([record["oracle_sign_state"] for record in records], dtype=np.int8),
            c_readable_wrong, c_h0_unknown),
        "response_jsonl_sha256": _sha256(jsonl_path),
        "elapsed_seconds_this_process": elapsed_seconds,
    }
    write_json(args.out / "full_test_metrics.json", report)
    write_json(run_manifest_path, {
        "schema": "animcv_sign_advisor_full_test_manifest_v1",
        "status": report["run_status"],
        "model": report["model"],
        "prompt": report["prompt"],
        "crop": report["crop"],
        "bank_content_digest": report["bank_content_digest"],
        "docs44_replay_sha256": replay_sha,
        "docs45_attribution_sha256": attribution_sha,
        "response_jsonl_sha256": report["response_jsonl_sha256"],
        "metrics_sha256": _sha256(args.out / "full_test_metrics.json"),
        "execution": report["execution"],
        "records": len(records),
        "expected_records": len(positions),
        "raw_root_data_modified": False,
        "historical_outputs_modified": False,
    })
    return report


def replay_joint_names() -> tuple[str, ...]:
    from framepose.contract import JOINT_NAMES
    return tuple(JOINT_NAMES)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--h0-prediction", required=True, type=Path)
    parser.add_argument("--replay-report", required=True, type=Path)
    parser.add_argument("--attribution-report", required=True, type=Path)
    parser.add_argument("--image-root", action="append", required=True, help="KEY=PATH")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--model", default=EXPECTED_MODEL_ID)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--hf-cache", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=EXPECTED_MAX_NEW_TOKENS)
    parser.add_argument("--seed", type=int, default=EXPECTED_SEED)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--started-utc", default="not-recorded")
    parser.add_argument("--frame-batch-size", type=int, default=1,
                        help="receiver frames per call; each contributes real+shuffled requests")
    parser.add_argument("--equivalence-report", type=Path,
                        help="saved bounded benchmark/equivalence report required for batching")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({"run_status": result["run_status"],
                      "frame_count": result["frame_count"],
                      "response_jsonl_sha256": result["response_jsonl_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
