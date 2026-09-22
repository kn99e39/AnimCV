#!/usr/bin/env python3
"""Complete the C3 slice and gate accounting audit without retraining."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from common.serialization import write_json
from framepose.bank import load_bank
from framepose.context_refiner import GateThresholds, apply_gate, runtime_signals
from framepose.contract import JOINT_COUNT

try:
    from diagnose_context_refiner_failure import test_cohort_masks
except ModuleNotFoundError:
    from scripts.diagnose_context_refiner_failure import test_cohort_masks


def stats(values: np.ndarray) -> dict[str, Any]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {"count": 0}
    return {"count": int(len(finite)), "mean": float(finite.mean()),
            "median": float(np.median(finite)),
            "p05": float(np.quantile(finite, 0.05)),
            "p95": float(np.quantile(finite, 0.95))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--h0-train", required=True, type=Path)
    parser.add_argument("--h0-validation", required=True, type=Path)
    parser.add_argument("--h0-test", required=True, type=Path)
    parser.add_argument("--gate-thresholds", required=True, type=Path)
    parser.add_argument("--c3-out", required=True, type=Path)
    args = parser.parse_args()
    bank = load_bank(args.bank)
    positions = bank.indices("test")
    h0 = np.asarray(np.load(args.h0_test), dtype=np.float32)
    if h0.shape != (len(positions), JOINT_COUNT, 3):
        raise ValueError("unexpected H0 test shape")
    # The signal builder needs all split H0 rows.  Reconstruct them from the
    # explicitly supplied frozen arrays, without reading targets.
    split_paths = {
        "train": args.h0_train,
        "validation": args.h0_validation,
        "test": args.h0_test,
    }
    all_h0 = np.full((len(bank), JOINT_COUNT, 3), np.nan, dtype=np.float32)
    for split, path in split_paths.items():
        values = np.asarray(np.load(path), dtype=np.float32)
        all_h0[bank.indices(split)] = values
    signals = runtime_signals(bank, all_h0)
    thresholds = GateThresholds.from_dict(json.loads(
        args.gate_thresholds.read_text(encoding="utf-8")))
    decision = apply_gate(signals, thresholds)
    target = bank.arrays["target_3d"][positions]
    valid = bank.arrays["target_valid"][positions]
    predictions = {
        name: np.asarray(np.load(args.c3_out / f"prediction_test_{name}.npy"), dtype=np.float32)
        for name in ("C0", "C1", "C2", "C3")
    }
    raw = test_cohort_masks(bank, all_h0, signals)
    masks = {"ALL TEST": np.ones_like(valid, dtype=bool),
             "ALL GATE-ON": raw["ALL GATE-ON"] & decision.gate_on[positions]}
    for name, mask in raw.items():
        if name != "ALL GATE-ON":
            masks[name] = mask
    slice_metrics: dict[str, Any] = {}
    gate_accounting: dict[str, Any] = {}
    base_error = np.linalg.norm(predictions["C0"] - target, axis=-1)
    gate = decision.gate_on[positions]
    for label, mask in masks.items():
        selected = mask & valid & np.isfinite(target).all(axis=-1)
        slice_metrics[label] = {}
        gate_accounting[label] = {}
        for candidate, prediction in predictions.items():
            errors = np.linalg.norm(prediction - target, axis=-1)
            values = errors[selected] * 1000.0
            changed = np.any(prediction != predictions["C0"], axis=-1)
            on = selected & gate
            delta = (base_error - errors) * 1000.0
            slice_metrics[label][candidate] = {
                "joint_rows": int(mask.sum()), "evaluated_rows": int(selected.sum()),
                "mpjpe_mm": float(values.mean()) if len(values) else None,
            }
            gate_accounting[label][candidate] = {
                "gate_on_count": int((mask & gate).sum()),
                "gate_off_count": int((mask & ~gate).sum()),
                "changed_gate_off_count": int((changed & mask & ~gate).sum()),
                "gate_on_delta_mm": stats(delta[on]),
                "gate_on_improved_count": int((delta[on] > 0).sum()),
                "gate_on_worsened_count": int((delta[on] < 0).sum()),
            }
    write_json(args.c3_out / "c3_audit_slice_metrics.json", slice_metrics)
    write_json(args.c3_out / "c3_audit_gate_accounting.json", gate_accounting)
    write_json(args.c3_out / "c3_audit_report.json", {
        "schema": "animcv_context_refiner_c3_audit_v1",
        "same_runtime_gate_as_C2": True,
        "slices": list(masks),
        "slice_metrics_path": str(args.c3_out / "c3_audit_slice_metrics.json"),
        "gate_accounting_path": str(args.c3_out / "c3_audit_gate_accounting.json"),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
