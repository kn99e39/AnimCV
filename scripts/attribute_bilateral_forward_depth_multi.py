#!/usr/bin/env python3
"""Multi-checkpoint forward-depth attribution on the 3DPW official-test hard
set (docs/21) -- generalizes attribute_bilateral_forward_depth.py (which
compared exactly A9 vs A14) to an arbitrary set of labeled checkpoints, so
the clean compiled-A9-control-vs-corrected-candidate comparison can include
the historical eager A9 and historical (denominator-contaminated) A14 as
reference without conflating any of them as "the" baseline.

Diagnostic only: trains nothing, changes no checkpoint or evaluation
semantics. Reuses the production ``_bilateral_forward_depth_diagnostics``
helper. The hard-set definition is fixed from ONE nominated checkpoint's
evaluator ranking (``--hard-set-label``, historical A9 by default) and never
redefined after observing any other checkpoint (docs/21 Section 15).

Usage:
  python3 scripts/attribute_bilateral_forward_depth_multi.py \
    --checkpoint historical_a9=/output/experiments/ablation_a9_fingerprinted_baseline_10e/reports/direct_mix.pth \
    --checkpoint compiled_a9_control=/output/experiments/ablation_a15_compiled_a9_control_10e/reports/direct_mix.pth \
    --checkpoint corrected_candidate=/output/experiments/ablation_a16_bilateral_forward_depth_corrected_10e/reports/direct_mix.pth \
    --checkpoint historical_a14_contaminated=/output/experiments/ablation_a14_bilateral_forward_depth_10e_v2/reports/direct_mix.pth \
    --hard-set-label historical_a9 \
    --holdout /data/3dpw/prepared/holdout.json \
    --out /output/experiments/a21_historical_diagnostic_repair/test_attribution_multi.json
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


def _parse_checkpoint_arg(value: str) -> tuple[str, Path]:
    label, _sep, path = value.partition("=")
    if not _sep:
        raise argparse.ArgumentTypeError(f"expected label=path, got {value!r}")
    return label, Path(path)


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
    parser.add_argument("--checkpoint", action="append", required=True, type=_parse_checkpoint_arg,
                         dest="checkpoints", metavar="LABEL=PATH")
    parser.add_argument("--hard-set-label", required=True,
                         help="which --checkpoint label's evaluator ranking fixes the hard-set definition")
    parser.add_argument("--holdout", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    checkpoints = dict(args.checkpoints)
    if args.hard_set_label not in checkpoints:
        raise ValueError(f"--hard-set-label {args.hard_set_label!r} is not among --checkpoint labels {list(checkpoints)}")

    torch, nn = _torch()
    dataset = load_dataset(args.holdout)

    predictions: dict[str, np.ndarray] = {}
    reference_targets: np.ndarray | None = None
    reference_valid: np.ndarray | None = None
    for label, checkpoint_path in checkpoints.items():
        prediction, targets, valid = _predict(torch, nn, checkpoint_path, dataset, args.device)
        if reference_targets is None:
            reference_targets, reference_valid = targets, valid
        else:
            assert np.array_equal(targets, reference_targets) and np.array_equal(valid, reference_valid), \
                f"{label} must evaluate on identical targets/validity (same holdout, same _arrays call)"
        predictions[label] = prediction

    # Hard-set definition is fixed from ONE nominated checkpoint's evaluator
    # ranking and never redefined after observing any other checkpoint.
    hard_set_yaw = _per_frame_yaw(predictions[args.hard_set_label], reference_targets, reference_valid)
    eligible = np.flatnonzero(np.isfinite(hard_set_yaw))
    order = eligible[np.argsort(-hard_set_yaw[eligible])]
    top5_count = max(1, (len(order) + 19) // 20)
    top1_count = max(1, (len(order) + 99) // 100)
    hard_top5 = order[:top5_count]
    hard_top1 = order[:top1_count]
    non_hard = order[top5_count:]

    subsets = {"all_eligible": eligible, "hard_top5pct": hard_top5, "hard_top1pct": hard_top1, "non_hard": non_hard}

    report: dict[str, Any] = {
        "holdout": str(args.holdout),
        "frame_count": len(hard_set_yaw),
        "eligible_frame_count": int(len(eligible)),
        "hard_set_label": args.hard_set_label,
        "hard_top5pct_yaw_cutoff_degrees": float(hard_set_yaw[hard_top5[-1]]) if len(hard_top5) else None,
        "hard_top1pct_yaw_cutoff_degrees": float(hard_set_yaw[hard_top1[-1]]) if len(hard_top1) else None,
        "checkpoints": {
            label: {name: _subset_diagnostics(torch, predictions[label], reference_targets, reference_valid, indices)
                    for name, indices in subsets.items()}
            for label in checkpoints
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"eligible_frame_count": report["eligible_frame_count"], "out": str(args.out)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
