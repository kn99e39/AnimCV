#!/usr/bin/env python3
"""Diagnose why the frozen C2 context refiner did or did not help.

This is an evidence-only script.  It never retrains C2, changes an existing
artifact, or uses target 3D values to build the context or runtime gate.  The
target is read only by the analysis functions after the frozen gate and
checkpoint have been replayed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from common.canonical_pose import BONE_INDICES, HINGE_INDICES, TORSO_INDICES
from common.serialization import write_json
from framepose.bank import load_bank
from framepose.context_refiner import (
    GateDecision,
    GateThresholds,
    apply_gate,
    build_context_features,
    interpolate_gated_h0,
    load_checkpoint,
    predict,
    runtime_signals,
)
from framepose.contract import JOINT_COUNT, JOINT_NAMES
from framepose.evaluate import evaluate_predictions

try:
    from diagnose_temporal_context_evidence import build_cohorts, local_neighbor_rows, sequence_rows
except ModuleNotFoundError:  # importable from repository-root test runners
    from scripts.diagnose_temporal_context_evidence import build_cohorts, local_neighbor_rows, sequence_rows


SCHEMA = "animcv_context_refiner_failure_diagnosis_v1"
SPLITS = ("train", "validation", "test")
EPSILON = 1e-12
FIXED_BANK_DIGEST = "75519e6394a764e3749ddaa30555b58b73a01db582ccee14f661374b9a0ed536"
FIXED_C2_TEST_SHA256 = "1bbbd35fd189585d6ac79b4d94735908cf6d9fc57df3267c8f2f5c5eeaa8fa49"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_h0(path: Path, count: int, label: str) -> np.ndarray:
    values = np.asarray(np.load(path), dtype=np.float32)
    expected = (count, JOINT_COUNT, 3)
    if values.shape != expected:
        raise ValueError(f"{label} H0 shape {values.shape}, expected {expected}")
    return values


def assemble_h0(bank, paths: dict[str, Path]) -> tuple[np.ndarray, dict[str, Any]]:
    result = np.full((len(bank), JOINT_COUNT, 3), np.nan, dtype=np.float32)
    identity: dict[str, Any] = {}
    for split, path in paths.items():
        positions = bank.indices(split)
        values = load_h0(path, len(positions), split)
        result[positions] = values
        identity[split] = {"path": str(path), "sha256": sha256(path), "shape": list(values.shape)}
    if not np.isfinite(result).all():
        raise ValueError("H0 does not cover every bank row with finite values")
    return result, identity


def stats(values: np.ndarray, *, scale: float = 1.0) -> dict[str, Any]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)] * scale
    if not len(finite):
        return {"count": 0}
    return {
        "count": int(len(finite)),
        "mean": float(finite.mean()),
        "median": float(np.median(finite)),
        "p05": float(np.quantile(finite, 0.05)),
        "p50": float(np.quantile(finite, 0.50)),
        "p95": float(np.quantile(finite, 0.95)),
    }


def pose_metrics(target: np.ndarray, valid: np.ndarray, prediction: np.ndarray,
                 mask: np.ndarray) -> dict[str, Any]:
    selected = mask & valid & np.isfinite(target).all(axis=-1) & np.isfinite(prediction).all(axis=-1)
    errors = np.linalg.norm(prediction - target, axis=-1) * 1000.0
    aligned: list[float] = []
    from common.canonical_pose import similarity_align

    for row in range(len(target)):
        joints = np.flatnonzero(selected[row])
        if len(joints) >= 3:
            aligned.extend(np.linalg.norm(
                similarity_align(prediction[row, joints], target[row, joints])
                - target[row, joints], axis=1) * 1000.0)
    return {
        "selected_joint_rows": int(mask.sum()),
        "evaluated_joint_rows": int(selected.sum()),
        "mpjpe_mm": stats(errors[selected]),
        "pa_mpjpe_mm": stats(np.asarray(aligned)),
        "per_joint_error_mm": {
            name: stats(errors[:, joint][selected[:, joint]])
            for joint, name in enumerate(JOINT_NAMES)
        },
    }


def residual_summary(target: np.ndarray, valid: np.ndarray, h0: np.ndarray,
                     candidate: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    """Attribute a candidate residual against the exact same target residual."""
    finite = (mask & valid & np.isfinite(target).all(axis=-1)
              & np.isfinite(h0).all(axis=-1) & np.isfinite(candidate).all(axis=-1))
    target_residual = target - h0
    learned_residual = candidate - h0
    target_norm = np.linalg.norm(target_residual, axis=-1)
    learned_norm = np.linalg.norm(learned_residual, axis=-1)
    denom = target_norm
    valid_metric = finite & (denom > EPSILON) & (learned_norm >= 0.0)
    target_unit = np.divide(target_residual, denom[..., None],
                            out=np.zeros_like(target_residual), where=denom[..., None] > EPSILON)
    signed_projection = np.sum(learned_residual * target_unit, axis=-1)
    orthogonal = learned_residual - signed_projection[..., None] * target_unit
    cosine = np.divide(np.sum(learned_residual * target_residual, axis=-1),
                       learned_norm * denom, out=np.full_like(denom, np.nan),
                       where=(learned_norm > EPSILON) & (denom > EPSILON))
    ratio = np.divide(learned_norm, denom, out=np.full_like(denom, np.nan), where=denom > EPSILON)
    h0_error = np.linalg.norm(h0 - target, axis=-1)
    candidate_error = np.linalg.norm(candidate - target, axis=-1)
    delta = (h0_error - candidate_error) * 1000.0
    result: dict[str, Any] = {
        "joint_rows": int(mask.sum()),
        "evaluated_rows": int(finite.sum()),
        "metric_rows": int(valid_metric.sum()),
        "target_residual_norm_mm": stats(target_norm[finite], scale=1000.0),
        "learned_residual_norm_mm": stats(learned_norm[finite], scale=1000.0),
        "norm_ratio": stats(ratio[valid_metric]),
        "cosine_alignment": stats(cosine[valid_metric]),
        "signed_projection_mm": stats(signed_projection[valid_metric], scale=1000.0),
        "orthogonal_residual_mm": stats(np.linalg.norm(orthogonal, axis=-1)[valid_metric], scale=1000.0),
        "target_error_delta_mm": stats(delta[finite]),
        "improved_count": int((delta[finite] > EPSILON).sum()),
        "worsened_count": int((delta[finite] < -EPSILON).sum()),
        "axis_target_residual_mm": {},
        "axis_learned_residual_mm": {},
    }
    for axis, name in enumerate(("x", "y", "z")):
        result["axis_target_residual_mm"][name] = stats(target_residual[..., axis][finite], scale=1000.0)
        result["axis_learned_residual_mm"][name] = stats(learned_residual[..., axis][finite], scale=1000.0)
    return result


def gate_on_metrics(target: np.ndarray, valid: np.ndarray, h0: np.ndarray,
                    c1: np.ndarray, c2: np.ndarray, gate: np.ndarray) -> dict[str, Any]:
    mask = gate.copy()
    baseline = pose_metrics(target, valid, h0, mask)
    control = pose_metrics(target, valid, c1, mask)
    learned = pose_metrics(target, valid, c2, mask)
    h0_error = np.linalg.norm(h0 - target, axis=-1) * 1000.0
    c1_error = np.linalg.norm(c1 - target, axis=-1) * 1000.0
    c2_error = np.linalg.norm(c2 - target, axis=-1) * 1000.0
    evaluated = mask & valid & np.isfinite(target).all(axis=-1)
    return {
        "gate_on_joint_rows": int(mask.sum()),
        "evaluated_joint_rows": int(evaluated.sum()),
        "C0": baseline,
        "C1": control,
        "C2": learned,
        "C1_delta_vs_C0_mm": stats((h0_error - c1_error)[evaluated]),
        "C2_delta_vs_C0_mm": stats((h0_error - c2_error)[evaluated]),
        "C1_improved_count": int((h0_error[evaluated] - c1_error[evaluated] > EPSILON).sum()),
        "C1_worsened_count": int((h0_error[evaluated] - c1_error[evaluated] < -EPSILON).sum()),
        "C2_improved_count": int((h0_error[evaluated] - c2_error[evaluated] > EPSILON).sum()),
        "C2_worsened_count": int((h0_error[evaluated] - c2_error[evaluated] < -EPSILON).sum()),
    }


def make_gate_decision(decision, positions: np.ndarray) -> GateDecision:
    return GateDecision(decision.gate_on[positions], decision.reason[positions],
                        decision.observation_residual[positions], decision.h0_residual[positions],
                        decision.left[positions], decision.right[positions])


def test_cohort_masks(bank, h0: np.ndarray, signals: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    positions = bank.indices("test")
    test_h0 = h0[positions]
    observation = bank.arrays["input_2d"][positions].astype(np.float64)
    observation_valid = bank.arrays["input_valid"][positions]
    target = bank.arrays["target_3d"][positions].astype(np.float64)
    target_valid = bank.arrays["target_valid"][positions]
    grouped = sequence_rows(bank.samples, positions)
    before, after = local_neighbor_rows(grouped, len(positions))
    timestamps = np.asarray([signals["timestamps"][position] for position in positions], dtype=np.float64)
    cohorts, _ = build_cohorts(observation, observation_valid, test_h0, target,
                               target_valid, timestamps, before, after)
    return {
        "ALL GATE-ON": np.ones_like(target_valid, dtype=bool),
        "STABLE CONTROL": cohorts["stable_control"],
        "STABLE-2D / H0-JITTER": cohorts["h0_jitter_stable_observation"],
        "HIGH 2D INSTABILITY": cohorts["observation_degradation"],
        "OBSERVATION LOSS": cohorts["current_frame_observation_loss"],
        "DISTAL ANKLE": cohorts["distal_joint_failure"],
        "ARTICULATION-MISMATCH": cohorts["implausible_articulation"],
    }


def parity_report(h0: np.ndarray, c1: np.ndarray, c2: np.ndarray,
                  gate: np.ndarray, positions: np.ndarray) -> dict[str, Any]:
    c1_changed = np.any(c1 != h0, axis=-1)
    c2_changed = np.any(c2 != h0, axis=-1)
    c1_applied = c1_changed
    c2_applied = c2_changed
    same_gate = np.array_equal(c1_applied, gate) and np.array_equal(c2_applied, gate)
    return {
        "split": str({"positions": int(len(positions))}),
        "same_gate_applied_by_C1_and_C2": bool(same_gate),
        "C1_applied_count": int(c1_applied.sum()),
        "C2_applied_count": int(c2_applied.sum()),
        "C1_only": int((c1_applied & ~c2_applied).sum()),
        "C2_only": int((c2_applied & ~c1_applied).sum()),
        "both": int((c1_applied & c2_applied).sum()),
        "neither": int((~c1_applied & ~c2_applied).sum()),
        "C1_gate_mismatch": int((c1_applied != gate).sum()),
        "C2_gate_mismatch": int((c2_applied != gate).sum()),
    }


def _term_group(on: np.ndarray) -> np.ndarray:
    return np.where(on, "gate_on_or_coupled", "gate_off_only")


def loss_accounting_split(torch, prediction: np.ndarray, target: np.ndarray,
                          valid: np.ndarray, gate: np.ndarray) -> dict[str, Any]:
    """Decompose the exact C2 loss reduction without changing its objective."""
    pred = torch.as_tensor(prediction, dtype=torch.float32)
    truth = torch.as_tensor(target, dtype=torch.float32)
    valid_t = torch.as_tensor(valid, dtype=torch.bool)
    gate_t = torch.as_tensor(gate, dtype=torch.bool)
    coordinate_element = torch.nn.functional.smooth_l1_loss(pred, truth, reduction="none")
    coordinate_joint = coordinate_element.sum(dim=-1) * valid_t
    coordinate_denominator = float((valid_t.sum() * 3).item())

    def structural(values, indices, mode: str):
        first, second = zip(*indices) if mode != "hinge" else ((), ())
        if mode == "hinge":
            proximal, joint, distal = zip(*indices)
            chain_valid = valid_t[:, proximal] & valid_t[:, joint] & valid_t[:, distal]
            axis = pred[:, distal] - pred[:, proximal]
            target_axis = truth[:, distal] - truth[:, proximal]
            projection = ((pred[:, joint] - pred[:, proximal]) * axis).sum(-1, keepdim=True) / axis.square().sum(-1, keepdim=True).clamp_min(1e-8)
            target_projection = ((truth[:, joint] - truth[:, proximal]) * target_axis).sum(-1, keepdim=True) / target_axis.square().sum(-1, keepdim=True).clamp_min(1e-8)
            predicted = pred[:, joint] - (pred[:, proximal] + projection * axis)
            expected = truth[:, joint] - (truth[:, proximal] + target_projection * target_axis)
            errors = torch.nn.functional.smooth_l1_loss(predicted, expected, reduction="none").mean(dim=-1)
            active = gate_t[:, proximal] | gate_t[:, joint] | gate_t[:, distal]
        else:
            first, second = zip(*indices)
            pair_valid = valid_t[:, first] & valid_t[:, second]
            predicted = pred[:, first] - pred[:, second] if mode == "bone" else pred[:, second] - pred[:, first]
            expected = truth[:, first] - truth[:, second] if mode == "bone" else truth[:, second] - truth[:, first]
            errors = torch.nn.functional.smooth_l1_loss(predicted, expected, reduction="none").mean(dim=-1)
            chain_valid = pair_valid
            active = gate_t[:, first] | gate_t[:, second]
        counts = chain_valid.sum(dim=0)
        active_chains = counts > 0
        chain_denominator = active_chains.sum().clamp_min(1)
        total = (errors * chain_valid).sum(dim=0) / counts.clamp_min(1)
        raw = (total * active_chains).sum() / chain_denominator
        on_num = (errors * chain_valid * active).sum(dim=0) / counts.clamp_min(1)
        off_num = (errors * chain_valid * ~active).sum(dim=0) / counts.clamp_min(1)
        on = (on_num * active_chains).sum() / chain_denominator
        off = (off_num * active_chains).sum() / chain_denominator
        return float(raw.item()), float(on.item()), float(off.item()), int(active_chains.sum().item()), int(chain_valid.sum().item())

    bone_raw, bone_on, bone_off, bone_chains, bone_valid = structural(BONE_INDICES, BONE_INDICES, "bone")
    torso_raw, torso_on, torso_off, torso_chains, torso_valid = structural(TORSO_INDICES, TORSO_INDICES, "torso")
    hinge_raw, hinge_on, hinge_off, hinge_chains, hinge_valid = structural(HINGE_INDICES, HINGE_INDICES, "hinge")
    coordinate_raw = float(coordinate_joint.sum().item() / max(coordinate_denominator, 1.0))
    coordinate_on = float((coordinate_joint * gate_t).sum().item() / max(coordinate_denominator, 1.0))
    coordinate_off = float((coordinate_joint * ~gate_t).sum().item() / max(coordinate_denominator, 1.0))
    terms = {
        "coordinate": (coordinate_raw, coordinate_on, coordinate_off, int(valid_t.sum().item()), int(valid_t.sum().item())),
        "bone": (bone_raw, bone_on, bone_off, bone_chains, bone_valid),
        "torso": (torso_raw, torso_on, torso_off, torso_chains, torso_valid),
        "hinge": (hinge_raw, hinge_on, hinge_off, hinge_chains, hinge_valid),
    }
    weights = {"coordinate": 1.0, "bone": 0.25, "torso": 0.15, "hinge": 0.15}
    output: dict[str, Any] = {"gate_on_joint_rows": int((gate & valid).sum()),
                              "gate_off_joint_rows": int((~gate & valid).sum()), "terms": {}}
    weighted_total = 0.0
    for name, (raw, on, off, chains, valid_count) in terms.items():
        weighted = raw * weights[name]
        weighted_on = on * weights[name]
        weighted_off = off * weights[name]
        weighted_total += weighted
        output["terms"][name] = {
            "weight": weights[name],
            "raw_value": raw,
            "weighted_value": weighted,
            "gate_on_or_coupled_value": on,
            "gate_off_only_value": off,
            "gate_on_or_coupled_fraction": float(on / raw) if raw > 0 else None,
            "gate_off_only_fraction": float(off / raw) if raw > 0 else None,
            "fraction_sum": float((on + off) / raw) if raw > 0 else None,
            "structural_chain_count": chains,
            "valid_pair_or_chain_rows": valid_count,
        }
    output["weighted_total"] = weighted_total
    output["attribution_policy"] = {
        "coordinate": "joint-level gate-on versus gate-off numerator with the exact full valid-coordinate denominator",
        "bone_torso_hinge": "a pair/chain is gate-on-or-coupled when any member joint is on; otherwise gate-off-only; exact full chain reduction retained",
        "coupling_warning": "structural terms are not separable per joint; gate-on-or-coupled is reported separately from gate-off-only",
    }
    return output


def write_residual_rows(path: Path, split_reports: dict[str, Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        fields = ("split", "population", "candidate", "joint_rows", "metric_rows",
                  "target_residual_norm_mean_mm", "learned_residual_norm_mean_mm",
                  "norm_ratio_mean", "cosine_alignment_mean", "signed_projection_mean_mm",
                  "orthogonal_residual_mean_mm", "target_error_delta_mean_mm")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for split, populations in split_reports.items():
            for population, candidates in populations.items():
                for candidate, report in candidates.items():
                    def mean(name: str, scale: float = 1.0):
                        value = report.get(name, {}).get("mean")
                        return None if value is None else value * scale
                    writer.writerow({
                        "split": split, "population": population, "candidate": candidate,
                        "joint_rows": report.get("joint_rows"), "metric_rows": report.get("metric_rows"),
                        "target_residual_norm_mean_mm": mean("target_residual_norm_mm"),
                        "learned_residual_norm_mean_mm": mean("learned_residual_norm_mm"),
                        "norm_ratio_mean": mean("norm_ratio"),
                        "cosine_alignment_mean": mean("cosine_alignment"),
                        "signed_projection_mean_mm": mean("signed_projection_mm", 1000.0),
                        "orthogonal_residual_mean_mm": mean("orthogonal_residual_mm", 1000.0),
                        "target_error_delta_mean_mm": mean("target_error_delta_mm"),
                    })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--h0-train", required=True, type=Path)
    parser.add_argument("--h0-validation", required=True, type=Path)
    parser.add_argument("--h0-test", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--frozen-c2-test", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"refusing to overwrite diagnosis output: {args.out}")

    bank = load_bank(args.bank)
    bank.assert_split_isolation()
    if bank.content_digest() != FIXED_BANK_DIGEST:
        raise ValueError(f"unexpected bank digest {bank.content_digest()}")
    h0, h0_identity = assemble_h0(bank, {"train": args.h0_train, "validation": args.h0_validation, "test": args.h0_test})
    stored_c2 = np.asarray(np.load(args.frozen_c2_test), dtype=np.float32)
    expected_shape = (len(bank.indices("test")), JOINT_COUNT, 3)
    if stored_c2.shape != expected_shape:
        raise ValueError(f"frozen C2 shape {stored_c2.shape}, expected {expected_shape}")
    if sha256(args.frozen_c2_test) != FIXED_C2_TEST_SHA256:
        raise ValueError("frozen C2 test prediction hash does not match the accepted C2 artifact")

    contexts, windows = build_context_features(bank, h0)
    signals = runtime_signals(bank, h0)
    threshold_path = args.checkpoint.parent / "gate_thresholds.json"
    thresholds = GateThresholds.from_dict(json.loads(threshold_path.read_text(encoding="utf-8")))
    decision = apply_gate(signals, thresholds)
    device = args.device
    model, checkpoint = load_checkpoint(args.checkpoint, device=device)
    all_positions = np.arange(len(bank), dtype=np.int64)
    c0 = h0.copy()
    c1 = interpolate_gated_h0(h0, signals, decision.gate_on)
    c2 = predict(model, contexts, h0, decision.gate_on, all_positions, device=device)
    replay_c2 = c2[bank.indices("test")]
    if not np.array_equal(replay_c2, stored_c2):
        raise AssertionError("all-bank frozen C2 replay is not bitwise identical to stored C2 test prediction")

    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "input_identity.json", {
        "schema": "animcv_context_refiner_failure_input_identity_v1",
        "bank": {"path": str(args.bank), "content_digest": bank.content_digest()},
        "h0": h0_identity,
        "checkpoint": {"path": str(args.checkpoint), "sha256": sha256(args.checkpoint),
                        "payload_schema": checkpoint.get("schema"), "candidate": checkpoint.get("candidate")},
        "frozen_c2_test": {"path": str(args.frozen_c2_test), "sha256": sha256(args.frozen_c2_test),
                           "shape": list(stored_c2.shape), "bitwise_replay": True},
        "thresholds": {"path": str(threshold_path), "sha256": sha256(threshold_path)},
    })

    gate_rates = {
        split: {
            "frames": int(len(bank.indices(split))),
            "joint_rows": int(len(bank.indices(split)) * JOINT_COUNT),
            "gate_on_joint_rows": int(decision.gate_on[bank.indices(split)].sum()),
            "gate_on_rate": float(decision.gate_on[bank.indices(split)].mean()),
        }
        for split in SPLITS
    }
    contract = {
        "all_train_rows_enter_training": True,
        "train_frame_count": int(len(bank.indices("train"))),
        "target_valid_mask_used_for_loss": True,
        "residual_output_shape": [JOINT_COUNT, 3],
        "residual_predicted_for_all_joints": True,
        "gate_applied_during_training": True,
        "gate_off_refined_value": "exact H0 in _apply_residual",
        "gate_off_joints_still_contribute_coordinate_loss": True,
        "structural_terms_couple_on_off_joints": True,
        "validation_selection": "whole-frame validation MPJPE over target-valid joints",
        "gate_rates": gate_rates,
        "loss_contract": checkpoint.get("candidate", {}).get("loss_contract", "baseline_geometry_v1"),
        "weights": {"coordinate": 1.0, "bone": 0.25, "torso": 0.15, "hinge": 0.15},
    }
    write_json(args.out / "training_runtime_contract.json", contract)

    positions_by_split = {split: bank.indices(split) for split in SPLITS}
    predictions = {"C0": c0, "C1": c1, "C2": c2}
    whole_split: dict[str, Any] = {}
    gate_on_split: dict[str, Any] = {}
    residual_split: dict[str, Any] = {}
    parity: dict[str, Any] = {}
    residual_rows_source: dict[str, Any] = {}
    for split, positions in positions_by_split.items():
        target = bank.arrays["target_3d"][positions].astype(np.float64)
        valid = bank.arrays["target_valid"][positions]
        finite_target = np.isfinite(target).all(axis=-1)
        valid = valid & finite_target
        split_predictions = {name: value[positions] for name, value in predictions.items()}
        whole_split[split] = {
            name: evaluate_predictions(bank, positions, value, candidate=name)
            for name, value in split_predictions.items()
        }
        split_gate = decision.gate_on[positions]
        gate_on_split[split] = gate_on_metrics(target, valid, split_predictions["C0"],
                                               split_predictions["C1"], split_predictions["C2"], split_gate)
        residual_split[split] = {"ALL GATE-ON": {}}
        for candidate in ("C1", "C2"):
            residual_split[split]["ALL GATE-ON"][candidate] = residual_summary(
                target, valid, split_predictions["C0"], split_predictions[candidate], split_gate)
        parity[split] = parity_report(split_predictions["C0"], split_predictions["C1"],
                                      split_predictions["C2"], split_gate, positions)
        residual_rows_source[split] = {"target": target, "valid": valid,
                                       "h0": split_predictions["C0"],
                                       "c1": split_predictions["C1"], "c2": split_predictions["C2"],
                                       "gate": split_gate}

    test_positions = positions_by_split["test"]
    test_target = residual_rows_source["test"]["target"]
    test_valid = residual_rows_source["test"]["valid"]
    test_populations = test_cohort_masks(bank, h0, signals)
    for population, population_mask in test_populations.items():
        gate_mask = population_mask & decision.gate_on[test_positions]
        residual_split["test"].setdefault(population, {})
        for candidate in ("C1", "C2"):
            residual_split["test"][population][candidate] = residual_summary(
                test_target, test_valid, residual_rows_source["test"]["h0"],
                residual_rows_source["test"][candidate.lower()], gate_mask)
    write_json(args.out / "whole_split_metrics.json", whole_split)
    write_json(args.out / "gate_on_metrics.json", gate_on_split)
    write_json(args.out / "gate_parity.json", parity)
    write_json(args.out / "residual_attribution.json", residual_split)
    write_residual_rows(args.out / "residual_attribution.csv", residual_split)

    import torch
    loss_accounting = {
        split: loss_accounting_split(torch, predictions["C2"][positions],
                                     bank.arrays["target_3d"][positions],
                                     bank.arrays["target_valid"][positions],
                                     decision.gate_on[positions])
        for split, positions in positions_by_split.items()
    }
    write_json(args.out / "loss_accounting.json", loss_accounting)

    summary = {
        "schema": SCHEMA,
        "input_identity_path": str(args.out / "input_identity.json"),
        "training_runtime_contract_path": str(args.out / "training_runtime_contract.json"),
        "gate_rates": gate_rates,
        "C1_C2_gate_identity": all(item["same_gate_applied_by_C1_and_C2"] for item in parity.values()),
        "C2_stored_test_replay_bitwise": True,
        "whole_split_metrics_path": str(args.out / "whole_split_metrics.json"),
        "gate_on_metrics_path": str(args.out / "gate_on_metrics.json"),
        "gate_parity_path": str(args.out / "gate_parity.json"),
        "residual_attribution_path": str(args.out / "residual_attribution.json"),
        "loss_accounting_path": str(args.out / "loss_accounting.json"),
        "no_c3_run": True,
        "no_existing_c2_artifact_modified": True,
    }
    write_json(args.out / "diagnosis_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
