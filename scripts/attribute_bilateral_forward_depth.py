#!/usr/bin/env python3
"""A9-vs-A14 forward-depth attribution on the 3DPW official-test hard set.

Diagnostic only: trains nothing, changes no checkpoint or evaluation
semantics. Reuses the production ``_bilateral_forward_depth_diagnostics``
helper (Section 12/13, docs/10 A14) on real 3DPW official-test predictions
from both checkpoints, sliced by the A9 evaluator's own per-frame root-yaw
ranking (top-5%, top-1%, and the remaining non-hard frames) -- the hard-set
definition is fixed from A9 alone and never redefined after observing A14.

Usage:
  python3 scripts/attribute_bilateral_forward_depth.py \
    --a9-checkpoint /output/experiments/ablation_a9_fingerprinted_baseline_10e/reports/direct_mix.pth \
    --a14-checkpoint /output/experiments/ablation_a14_bilateral_forward_depth_10e_v2/reports/direct_mix.pth \
    --holdout /data/3dpw/prepared/holdout.json \
    --out /output/experiments/a14_bilateral_forward_depth_diagnosis/test_attribution.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from training.temporal_lifter import (
    _arrays, _bilateral_forward_depth_diagnostics, _model, _predict_batched, _root_yaw_error_degrees,
    _torch, load_dataset,
)


def _predict(torch, nn, checkpoint_path: Path, dataset: dict[str, Any], device: str):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    inputs, targets, valid, offsets = _arrays(
        dataset, int(checkpoint["window"]),
        coordinate_normalization=checkpoint.get("input_coordinate_normalization", "image_v1"),
    )
    model = _model(nn, int(checkpoint["channels"]), checkpoint.get("architecture", "legacy_tcn_v1")).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    x = torch.as_tensor(inputs, dtype=torch.float32, device=device)
    with torch.no_grad():
        prediction = _predict_batched(
            model, x, torch.as_tensor(offsets, dtype=torch.long, device=device), 1024, device.startswith("cuda"),
        )
    return prediction.cpu().numpy(), targets, valid


def _per_frame_yaw(predictions: np.ndarray, targets: np.ndarray, valid: np.ndarray) -> np.ndarray:
    errors = np.full(len(predictions), np.nan, dtype=np.float64)
    for index, (estimate, reference, frame_valid) in enumerate(zip(predictions, targets, valid)):
        yaw = _root_yaw_error_degrees(estimate, reference, frame_valid)
        if yaw is not None:
            errors[index] = yaw
    return errors


def _subset_diagnostics(torch, predictions: np.ndarray, targets: np.ndarray, valid: np.ndarray, indices: np.ndarray) -> dict[str, float]:
    if len(indices) == 0:
        return {}
    prediction_t = torch.as_tensor(predictions[indices], dtype=torch.float32)
    target_t = torch.as_tensor(targets[indices], dtype=torch.float32)
    valid_t = torch.as_tensor(valid[indices], dtype=torch.bool)
    result = _bilateral_forward_depth_diagnostics(torch, prediction_t, target_t, valid_t)
    result["frame_count"] = int(len(indices))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a9-checkpoint", required=True, type=Path)
    parser.add_argument("--a14-checkpoint", required=True, type=Path)
    parser.add_argument("--holdout", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    torch, nn = _torch()
    dataset = load_dataset(args.holdout)

    a9_prediction, targets, valid = _predict(torch, nn, args.a9_checkpoint, dataset, args.device)
    a14_prediction, targets_check, valid_check = _predict(torch, nn, args.a14_checkpoint, dataset, args.device)
    assert np.array_equal(targets, targets_check) and np.array_equal(valid, valid_check), \
        "A9 and A14 must evaluate on identical targets/validity (same holdout, same _arrays call)"

    # Hard-set definition is fixed from the A9 evaluator alone (Section 13:
    # "Reuse the existing A9 hard-set definition. Do not redefine hard
    # examples after observing the candidate.").
    a9_yaw = _per_frame_yaw(a9_prediction, targets, valid)
    eligible = np.flatnonzero(np.isfinite(a9_yaw))
    order = eligible[np.argsort(-a9_yaw[eligible])]  # descending yaw error
    top5_count = max(1, (len(order) + 19) // 20)
    top1_count = max(1, (len(order) + 99) // 100)
    hard_top5 = order[:top5_count]
    hard_top1 = order[:top1_count]
    non_hard = order[top5_count:]

    subsets = {
        "all_eligible": eligible,
        "a9_hard_top5pct": hard_top5,
        "a9_hard_top1pct": hard_top1,
        "non_hard": non_hard,
    }

    report: dict[str, Any] = {
        "holdout": str(args.holdout),
        "frame_count": len(a9_yaw),
        "eligible_frame_count": int(len(eligible)),
        "a9_hard_top5pct_yaw_cutoff_degrees": float(a9_yaw[hard_top5[-1]]) if len(hard_top5) else None,
        "a9_hard_top1pct_yaw_cutoff_degrees": float(a9_yaw[hard_top1[-1]]) if len(hard_top1) else None,
        "a9": {name: _subset_diagnostics(torch, a9_prediction, targets, valid, indices) for name, indices in subsets.items()},
        "a14": {name: _subset_diagnostics(torch, a14_prediction, targets, valid, indices) for name, indices in subsets.items()},
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"eligible_frame_count": report["eligible_frame_count"], "out": str(args.out)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
