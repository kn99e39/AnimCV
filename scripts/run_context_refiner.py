#!/usr/bin/env python3
"""Train and evaluate the bounded support-gated context refiner.

The command consumes frozen H0 predictions for each split.  It never changes
the bank, H0 checkpoint, Frame Pose Core, Pose Reconciliation or SignState.
The only learned output is a residual for the target frame.  The gate is fit
from train-only runtime-observable signals and is frozen before validation/test
evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from common.canonical_pose import similarity_align
from common.serialization import write_json
from framepose.bank import load_bank
from framepose.context_refiner import (
    EPSILON,
    GateDecision,
    RefinerConfig,
    WINDOW_OFFSETS,
    apply_gate,
    build_context_features,
    fit_gate_thresholds,
    interpolate_gated_h0,
    load_checkpoint,
    parameter_report,
    predict,
    runtime_signals,
    train_refiner,
)
from framepose.contract import JOINT_COUNT, JOINT_NAMES
from framepose.evaluate import evaluate_predictions

from diagnose_temporal_context_evidence import (
    build_cohorts,
    local_neighbor_rows,
    sequence_rows,
)


SCHEMA = "animcv_context_refiner_experiment_v1"
FIXED_BANK_DIGEST = "75519e6394a764e3749ddaa30555b58b73a01db582ccee14f661374b9a0ed536"
FIXED_H0_TEST_SHA256 = "6df04c2031b2bde6a4f0af37fdae0a7f0228e9fc23401e15e1aa2b3bd35695a5"
OWNER_SEEDS = (
    ("arguing_550_left_elbow", "3dpw:downtown_arguing_00:actor0#000550", "left_elbow"),
    ("sitOnStairs_45_left_knee", "3dpw:downtown_sitOnStairs_00:actor0#000045", "left_knee"),
    ("bar_985_right_elbow", "3dpw:downtown_bar_00:actor0#000985", "right_elbow"),
    ("warmWelcome_330_right_knee", "3dpw:downtown_warmWelcome_00:actor0#000330", "right_knee"),
    ("car_507_right_knee", "3dpw:downtown_car_00:actor1#000507", "right_knee"),
    ("windowShopping_588_right_ankle", "3dpw:downtown_windowShopping_00:actor0#000588", "right_ankle"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_split(path: Path, count: int, label: str) -> np.ndarray:
    values = np.asarray(np.load(path), dtype=np.float32)
    expected = (count, JOINT_COUNT, 3)
    if values.shape != expected:
        raise ValueError(f"{label} H0 prediction has shape {values.shape}, expected {expected}")
    return values


def _assemble_h0(bank, paths: dict[str, Path]) -> tuple[np.ndarray, dict[str, Any]]:
    result = np.full((len(bank), JOINT_COUNT, 3), np.nan, dtype=np.float32)
    identity = {}
    for split, path in paths.items():
        positions = bank.indices(split)
        values = _load_split(path, len(positions), split)
        result[positions] = values
        identity[split] = {"path": str(path), "sha256": _sha256(path), "shape": list(values.shape)}
    if not np.isfinite(result).all():
        raise ValueError("H0 predictions do not cover every bank row with finite values")
    return result, identity


def _stats(values: np.ndarray) -> dict[str, Any] | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return None
    return {"mean": float(finite.mean()), "median": float(np.median(finite)),
            "p95": float(np.quantile(finite, 0.95)), "count": int(len(finite))}


def _pose_metrics(target: np.ndarray, valid: np.ndarray, prediction: np.ndarray,
                  row_joint_mask: np.ndarray) -> dict[str, Any]:
    selected = row_joint_mask & valid & np.isfinite(target).all(axis=-1) & np.isfinite(prediction).all(axis=-1)
    errors = np.linalg.norm(prediction - target, axis=-1) * 1000.0
    frame_errors = errors[selected]
    aligned = []
    for row in range(len(target)):
        indices = np.flatnonzero(selected[row])
        if len(indices) >= 3:
            aligned.extend(np.linalg.norm(
                similarity_align(prediction[row, indices], target[row, indices]) - target[row, indices], axis=1) * 1000.0)
    return {
        "total_joint_rows": int(row_joint_mask.sum()),
        "evaluated_joint_rows": int(selected.sum()),
        "mpjpe_mm": _stats(frame_errors),
        "pa_mpjpe_mm": _stats(np.asarray(aligned)),
        "per_joint_error_mm": {
            name: _stats(errors[:, joint][selected[:, joint]]) for joint, name in enumerate(JOINT_NAMES)
        },
    }


def _delta_stats(delta: np.ndarray) -> dict[str, Any]:
    result = _stats(delta)
    negative = -delta[delta < -EPSILON]
    result = result or {"count": 0}
    result["damage_tail_p95_mm"] = float(np.quantile(negative, 0.95)) if len(negative) else 0.0
    return result


def _gate_accounting(name: str, mask: np.ndarray, candidate: np.ndarray, baseline: np.ndarray,
                     target: np.ndarray, target_valid: np.ndarray, decision: GateDecision,
                     bank, positions: np.ndarray) -> dict[str, Any]:
    finite = np.isfinite(target).all(axis=-1) & np.isfinite(baseline).all(axis=-1) & np.isfinite(candidate).all(axis=-1)
    evaluated = mask & target_valid & finite
    gate_on = mask & decision.gate_on
    gate_off = mask & ~decision.gate_on
    on_evaluated = evaluated & decision.gate_on
    baseline_error = np.linalg.norm(baseline - target, axis=-1) * 1000.0
    candidate_error = np.linalg.norm(candidate - target, axis=-1) * 1000.0
    delta = baseline_error - candidate_error
    on_delta = delta[on_evaluated]
    all_delta = delta[evaluated]
    unchanged_off = np.all(candidate[gate_off] == baseline[gate_off], axis=-1) if gate_off.any() else np.asarray([], dtype=bool)
    by_joint = {}
    for joint, joint_name in enumerate(JOINT_NAMES):
        selected = mask[:, joint]
        selected_eval = evaluated[:, joint]
        selected_on = on_evaluated[:, joint]
        values = delta[:, joint][selected_eval]
        by_joint[joint_name] = {
            "total_rows": int(selected.sum()),
            "gate_on_count": int(gate_on[:, joint].sum()),
            "gate_activation_rate": float(gate_on[:, joint].sum() / selected.sum()) if selected.any() else 0.0,
            "improved_when_on": int((delta[:, joint][selected_on] > EPSILON).sum()),
            "worsened_when_on": int((delta[:, joint][selected_on] < -EPSILON).sum()),
            "delta_mm": _delta_stats(values),
        }
    by_sequence = {}
    for sequence in sorted({bank.samples[int(position)].sequence_id for position in positions}):
        row_mask = np.asarray([bank.samples[int(position)].sequence_id == sequence for position in positions])
        selected = mask & row_mask[:, None]
        selected_eval = evaluated & row_mask[:, None]
        values = delta[selected_eval]
        by_sequence[sequence] = {
            "total_rows": int(selected.sum()),
            "gate_on_count": int((selected & decision.gate_on).sum()),
            "gate_activation_rate": float((selected & decision.gate_on).sum() / selected.sum()) if selected.any() else 0.0,
            "delta_mm": _delta_stats(values),
        }
    refusal = {}
    for value in decision.reason[mask].tolist():
        refusal[str(value)] = refusal.get(str(value), 0) + 1
    return {
        "candidate": name,
        "total_rows": int(mask.sum()),
        "evaluated_rows": int(evaluated.sum()),
        "gate_on_count": int(gate_on.sum()),
        "gate_off_count": int(gate_off.sum()),
        "gate_activation_rate": float(gate_on.sum() / mask.sum()) if mask.any() else 0.0,
        "improved_when_on": int((on_delta > EPSILON).sum()),
        "worsened_when_on": int((on_delta < -EPSILON).sum()),
        "unchanged_when_off": int(unchanged_off.sum()),
        "unchanged_when_off_rate": float(unchanged_off.mean()) if len(unchanged_off) else None,
        "target_error_delta_mm": _delta_stats(all_delta),
        "target_error_delta_when_on_mm": _delta_stats(on_delta),
        "false_intervention_on_stable_control": {
            "gate_on_count": int(gate_on.sum()) if name else 0,
            "changed_count": int((~unchanged_off).sum()) if name else 0,
        },
        "gate_activation_by_joint": by_joint,
        "gate_activation_by_sequence": by_sequence,
        "support_refusal_reasons": refusal,
    }


def _owner_replay(bank, positions, baseline, c1, c2, decision) -> list[dict[str, Any]]:
    row_by_id = {bank.samples[int(position)].sample_id: row for row, position in enumerate(positions)}
    target = bank.arrays["target_3d"][positions]
    valid = bank.arrays["target_valid"][positions]
    output = []
    for label, sample_id, joint_name in OWNER_SEEDS:
        if sample_id not in row_by_id:
            continue
        row = row_by_id[sample_id]
        joint = JOINT_NAMES.index(joint_name)
        h0_error = float(np.linalg.norm(baseline[row, joint] - target[row, joint]) * 1000.0) if valid[row, joint] else None
        output.append({
            "label": label, "sample_id": sample_id, "joint": joint_name,
            "gate_on": bool(decision.gate_on[row, joint]),
            "gate_reason": str(decision.reason[row, joint]),
            "h0_error_mm": h0_error,
            "c1_error_mm": float(np.linalg.norm(c1[row, joint] - target[row, joint]) * 1000.0) if valid[row, joint] else None,
            "c2_error_mm": float(np.linalg.norm(c2[row, joint] - target[row, joint]) * 1000.0) if valid[row, joint] else None,
        })
    return output


def _verdict(slice_reports: dict[str, Any], accounting: dict[str, Any]) -> str:
    """Conservative machine verdict; selected-cohort gain alone can never yield A."""
    all_report = slice_reports["ALL TEST"]["C2"]
    jitter = slice_reports["STABLE-2D / H0-JITTER"]["C2"]
    stable = accounting["STABLE CONTROL"]["C2"]
    high_instability = accounting["HIGH 2D INSTABILITY"]["C2"]
    if all_report["mpjpe_mm"] is None or jitter["mpjpe_mm"] is None:
        return "E — MIXED / INSUFFICIENT EVIDENCE"
    if stable["gate_on_count"] > 0 or high_instability["gate_on_count"] > 0:
        return "C — GATING ABSTRACTION INSUFFICIENT"
    all_delta = all_report["mpjpe_mm"]["mean"] - slice_reports["ALL TEST"]["C0"]["mpjpe_mm"]["mean"]
    jitter_delta = jitter["mpjpe_mm"]["mean"] - slice_reports["STABLE-2D / H0-JITTER"]["C0"]["mpjpe_mm"]["mean"]
    # Metrics are errors, hence both improvements are negative here. A requires
    # whole-test preservation/improvement, intended-regime improvement, and no
    # stable false activation; an isolated jitter win is insufficient.
    if all_delta < 0.0 and jitter_delta < 0.0 and stable["changed_count"] == 0:
        return "A — ADOPT SUPPORT-GATED CONTEXT LAYER"
    if jitter_delta < 0.0:
        return "B — CONTEXT SIGNAL REAL, CURRENT REFINER INSUFFICIENT"
    return "D — TEMPORAL CONTEXT BENEFIT DOES NOT SURVIVE LEARNED CONTROL"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--h0-train", required=True, type=Path)
    parser.add_argument("--h0-validation", required=True, type=Path)
    parser.add_argument("--h0-test", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-mixed-precision", action="store_true")
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"refusing to overwrite context-refiner output: {args.out}")
    bank = load_bank(args.bank)
    bank.assert_split_isolation()
    h0, h0_identity = _assemble_h0(bank, {
        "train": args.h0_train, "validation": args.h0_validation, "test": args.h0_test,
    })
    if bank.content_digest() == FIXED_BANK_DIGEST and h0_identity["test"]["sha256"] != FIXED_H0_TEST_SHA256:
        raise ValueError("the fixed test bank requires the byte-identified frozen H0 test prediction")
    contexts, windows = build_context_features(bank, h0)
    signals = runtime_signals(bank, h0)
    thresholds = fit_gate_thresholds(signals, bank.indices("train"))
    decision = apply_gate(signals, thresholds)
    config = RefinerConfig(name="C2_support_gated_context_refiner", epochs=args.epochs,
                           batch_size=args.batch_size, learning_rate=args.learning_rate,
                           weight_decay=args.weight_decay, seed=args.seed, device=args.device,
                           mixed_precision=not args.no_mixed_precision)
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "gate_thresholds.json", thresholds.to_dict())
    training = train_refiner(bank, h0, thresholds, config, contexts=contexts, gate=decision,
                             checkpoint_path=args.out / "checkpoint.pt")
    write_json(args.out / "training_report.json", training)
    device = args.device if (args.device != "cuda" or _torch_available()) else "cpu"
    model, checkpoint = load_checkpoint(args.out / "checkpoint.pt", device=device)
    predictions = {"C0": h0, "C1": interpolate_gated_h0(h0, signals, decision.gate_on)}
    started = perf_counter()
    predictions["C2"] = predict(model, contexts, h0, decision.gate_on, np.arange(len(bank)), device=device)
    inference_seconds = perf_counter() - started
    np.save(args.out / "prediction_test_C0.npy", predictions["C0"][bank.indices("test")].astype(np.float32))
    np.save(args.out / "prediction_test_C1.npy", predictions["C1"][bank.indices("test")].astype(np.float32))
    np.save(args.out / "prediction_test_C2.npy", predictions["C2"][bank.indices("test")].astype(np.float32))
    for candidate, prediction in predictions.items():
        report = evaluate_predictions(bank, bank.indices("test"), prediction[bank.indices("test")], candidate=candidate)
        write_json(args.out / f"evaluation_test_{candidate}.json", report)

    test_positions = bank.indices("test")
    test_h0 = h0[test_positions]
    test_observation = bank.arrays["input_2d"][test_positions].astype(np.float64)
    test_observation_valid = bank.arrays["input_valid"][test_positions]
    test_target = bank.arrays["target_3d"][test_positions].astype(np.float64)
    test_target_valid = bank.arrays["target_valid"][test_positions]
    grouped = sequence_rows(bank.samples, test_positions)
    before, after = local_neighbor_rows(grouped, len(test_positions))
    timestamps = np.asarray([signals["timestamps"][position] for position in test_positions], dtype=np.float64)
    cohorts, _ = build_cohorts(test_observation, test_observation_valid, test_h0, test_target,
                               test_target_valid, timestamps, before, after)
    slice_masks = {"ALL TEST": test_target_valid.copy(),
                   "STABLE CONTROL": cohorts["stable_control"],
                   "STABLE-2D / H0-JITTER": cohorts["h0_jitter_stable_observation"],
                   "HIGH 2D INSTABILITY": cohorts["observation_degradation"],
                   "CURRENT OBSERVATION LOSS": cohorts["current_frame_observation_loss"],
                   "DISTAL ANKLE": cohorts["distal_joint_failure"],
                   "ARTICULATION-MISMATCH": cohorts["implausible_articulation"]}
    slice_reports: dict[str, Any] = {}
    accounting: dict[str, Any] = {}
    for label, mask in slice_masks.items():
        slice_reports[label] = {}
        accounting[label] = {}
        for candidate, prediction in predictions.items():
            slice_reports[label][candidate] = _pose_metrics(
                test_target, test_target_valid, prediction[test_positions], mask)
            accounting[label][candidate] = _gate_accounting(
                candidate, mask, prediction[test_positions], test_h0, test_target,
                test_target_valid, GateDecision(
                    decision.gate_on[test_positions], decision.reason[test_positions],
                    decision.observation_residual[test_positions], decision.h0_residual[test_positions],
                    decision.left[test_positions], decision.right[test_positions]), bank, test_positions)
    write_json(args.out / "slice_metrics.json", slice_reports)
    write_json(args.out / "gate_accounting.json", accounting)
    write_json(args.out / "owner_seed_replay.json", _owner_replay(
        bank, test_positions, test_h0, predictions["C1"][test_positions], predictions["C2"][test_positions],
        GateDecision(decision.gate_on[test_positions], decision.reason[test_positions],
                     decision.observation_residual[test_positions], decision.h0_residual[test_positions],
                     decision.left[test_positions], decision.right[test_positions])))
    report = {
        "schema": SCHEMA,
        "bank": {"path": str(args.bank), "content_digest": bank.content_digest(),
                 "split_counts": {split: int(len(bank.indices(split))) for split in ("train", "validation", "test")}},
        "frozen_h0": h0_identity,
        "architecture": parameter_report(model),
        "context": {"window_offsets": list(WINDOW_OFFSETS),
                     "context_dimension": int(contexts.shape[1]), "target_frame_only": True,
                     "sequence_window_shape": list(windows.shape)},
        "runtime_gate": {"thresholds": thresholds.to_dict(), "train_only_fit": True,
                         "test_target_error_access": False,
                         "activation_rate_by_split": {
                             split: float(decision.gate_on[bank.indices(split)].mean())
                             for split in ("train", "validation", "test")}},
        "candidates": {candidate: {"whole_test": slice_reports["ALL TEST"][candidate],
                                   "evaluation_path": str(args.out / f"evaluation_test_{candidate}.json")}
                      for candidate in predictions},
        "inference": {"candidate": "C2", "seconds": inference_seconds,
                      "milliseconds_per_frame": inference_seconds / max(len(bank), 1) * 1000.0},
        "slice_metrics_path": str(args.out / "slice_metrics.json"),
        "gate_accounting_path": str(args.out / "gate_accounting.json"),
        "owner_seed_replay_path": str(args.out / "owner_seed_replay.json"),
        "verdict": _verdict(slice_reports, accounting),
        "preserved": ["Frame Pose Core", "H0", "FrameBank", "Pose Reconciliation", "SignState",
                      "MINIMUM_NORM/R_SWIVEL evidence", "canonical coordinates"],
    }
    write_json(args.out / "experiment_report.json", report)
    print(json.dumps({"out": str(args.out), "verdict": report["verdict"],
                      "h0_test_sha256": h0_identity["test"]["sha256"],
                      "c2_test_mpjpe_mm": slice_reports["ALL TEST"]["C2"]["mpjpe_mm"]},
                     indent=2, sort_keys=True))
    return 0


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
