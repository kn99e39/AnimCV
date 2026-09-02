#!/usr/bin/env python3
"""Frozen-representation linear probe: does A15's temporal latent already
contain recoverable shoulder/hip forward-depth state? (docs/22 Section
12-13, run only because Sections 4-11 left temporal branch-state failure
as a substantial unresolved explanation.)

Diagnostic-only: does NOT retrain the lifter. Loads A15's frozen checkpoint
(compiled A9 control -- the clean baseline) and, without touching its
weights, extracts two center-frame representations from the SAME forward
pass computation the model already does:

  - "full_temporal": the stable channel vector immediately before the
    prediction head (encoded[:, :, center] after all 5 dilated residual
    blocks -- receptive field 127, i.e. the full 81-frame window).
  - "local_control": the same channel width immediately after the stem
    (one Conv1d, kernel 3, padding 1 -- receptive field 3), a comparable-
    capacity but near-center-frame-only control.

A single linear probe (closed-form least squares, no deep MLP, no
backbone gradients) is fit per representation, predicting the continuous
target:

    q_shoulder = (y_right_shoulder - y_left_shoulder) / sqrt(2)
    q_hip      = (y_right_hip - y_left_hip) / sqrt(2)

(the exact corrected-SRD target, not root yaw) from GT. Train/test split
is by sequence (not by frame) to avoid temporal leakage. Reports held-out
R^2 for each representation; the comparison answers whether the existing
temporal representation already contains recoverable orientation-branch
information the current output head fails to exploit (P1: full_temporal
materially beats local_control) or does not (P2: comparable/no better).

Usage:
  python3 scripts/probe_a15_temporal_orientation_state.py \
    --a15-checkpoint /output/experiments/ablation_a15_compiled_a9_control_10e/reports/direct_mix.pth \
    --holdout /data/3dpw/prepared/holdout.json \
    --out /output/experiments/a22_a16_generalization_diagnosis/temporal_probe.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from training.temporal_lifter import FORWARD_DEPTH_AXIS, TORSO_INDICES, _arrays, _frame_metadata, _model, _torch, load_dataset

_NORMALIZATION = 1.0 / np.sqrt(2.0)


def _extract_features(torch, model, x, offsets, batch_size: int):
    """Run the frozen model's own stem/blocks up to (but not through) the
    head, batched, no_grad -- never modifies or retrains the backbone."""
    stem_features, block_features = [], []
    with torch.no_grad():
        for batch_offsets in offsets.split(batch_size):
            windows = x[batch_offsets]
            batch, frames, joints, features = windows.shape
            center = frames // 2
            stem_out = model.stem(windows.reshape(batch, frames, joints * features).transpose(1, 2))
            stem_features.append(stem_out[:, :, center].cpu())
            encoded = stem_out
            for block in model.blocks:
                encoded = block(encoded)
            block_features.append(encoded[:, :, center].cpu())
    return torch.cat(stem_features, dim=0).numpy(), torch.cat(block_features, dim=0).numpy()


def _q_targets(targets: np.ndarray) -> np.ndarray:
    """(N, 2) [q_shoulder, q_hip] from GT target_3d, matching the corrected
    SRD candidate's exact definition."""
    output = []
    for left, right in TORSO_INDICES:  # (shoulder, hip), same pair convention as YAW_INDICES
        output.append((targets[:, right, FORWARD_DEPTH_AXIS] - targets[:, left, FORWARD_DEPTH_AXIS]) * _NORMALIZATION)
    return np.stack(output, axis=1)


def _sequence_split(metadata: list[dict[str, Any]], test_fraction: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    """Split by distinct sequence ('action' label), not by frame, so the
    probe's held-out evaluation never sees frames from a sequence it fit
    on -- avoids trivial temporal leakage within a clip."""
    sequence_ids = [meta.get("action") or "unknown" for meta in metadata]
    distinct = sorted(set(sequence_ids))
    test_count = max(1, int(round(len(distinct) * test_fraction)))
    test_sequences = set(distinct[::max(1, len(distinct) // test_count)][:test_count])
    is_test = np.array([sequence_id in test_sequences for sequence_id in sequence_ids])
    return np.flatnonzero(~is_test), np.flatnonzero(is_test)


def _fit_linear_probe(features_train: np.ndarray, targets_train: np.ndarray):
    """Closed-form least squares with an intercept column -- the 'minimal
    diagnostic linear probe' the instruction calls for, not a trained MLP."""
    design = np.concatenate([features_train, np.ones((len(features_train), 1))], axis=1)
    weights, _residuals, _rank, _singular = np.linalg.lstsq(design, targets_train, rcond=None)
    return weights


def _apply_probe(weights: np.ndarray, features: np.ndarray) -> np.ndarray:
    design = np.concatenate([features, np.ones((len(features), 1))], axis=1)
    return design @ weights


def _r_squared(predicted: np.ndarray, actual: np.ndarray) -> float:
    residual = actual - predicted
    ss_res = float((residual ** 2).sum())
    ss_tot = float(((actual - actual.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a15-checkpoint", required=True, type=Path)
    parser.add_argument("--holdout", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    torch, nn = _torch()
    checkpoint = torch.load(args.a15_checkpoint, map_location=args.device, weights_only=True)
    dataset = load_dataset(args.holdout)
    metadata = _frame_metadata(dataset)
    inputs, targets, valid, offsets = _arrays(
        dataset, int(checkpoint["window"]),
        coordinate_normalization=checkpoint.get("input_coordinate_normalization", "image_v1"),
    )
    model = _model(nn, int(checkpoint["channels"]), checkpoint.get("architecture", "legacy_tcn_v1")).to(args.device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()  # frozen backbone: no gradients, no weight updates, ever

    x = torch.as_tensor(inputs, dtype=torch.float32, device=args.device)
    offset_tensor = torch.as_tensor(offsets, dtype=torch.long, device=args.device)
    stem_features, block_features = _extract_features(torch, model, x, offset_tensor, batch_size=1024)

    left_indices = [pair[0] for pair in TORSO_INDICES]
    right_indices = [pair[1] for pair in TORSO_INDICES]
    pair_valid = valid[:, left_indices] & valid[:, right_indices]  # (N, 2) [shoulder, hip]
    q = _q_targets(targets)
    train_indices, test_indices = _sequence_split(metadata)

    report: dict[str, Any] = {
        "holdout": str(args.holdout), "frame_count": len(targets),
        "train_frame_count": int(len(train_indices)), "test_frame_count": int(len(test_indices)),
        "representations": {},
    }
    for representation_name, features in (("local_control", stem_features), ("full_temporal", block_features)):
        result: dict[str, Any] = {"feature_dim": int(features.shape[1])}
        for pair_index, pair_name in enumerate(("shoulder", "hip")):
            valid_mask = pair_valid[:, pair_index]
            train_mask = valid_mask[train_indices]
            test_mask = valid_mask[test_indices]
            train_rows, test_rows = train_indices[train_mask], test_indices[test_mask]
            if len(train_rows) < 10 or len(test_rows) < 10:
                result[pair_name] = {"error": "insufficient valid frames"}
                continue
            weights = _fit_linear_probe(features[train_rows], q[train_rows, pair_index])
            test_prediction = _apply_probe(weights, features[test_rows])
            train_prediction = _apply_probe(weights, features[train_rows])
            result[pair_name] = {
                "train_r_squared": _r_squared(train_prediction, q[train_rows, pair_index]),
                "test_r_squared": _r_squared(test_prediction, q[test_rows, pair_index]),
                "test_mae": float(np.abs(test_prediction - q[test_rows, pair_index]).mean()),
                "train_frame_count": int(len(train_rows)), "test_frame_count": int(len(test_rows)),
            }
        report["representations"][representation_name] = result

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
