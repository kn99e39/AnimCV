#!/usr/bin/env python3
"""Repair two historical diagnostic-only measurement bugs (docs/21).

Diagnostic-only: no new training, no change to any historical checkpoint,
report, or evaluator/loss semantics. Reuses the exact fixed-batch replay
infrastructure from ``scripts/diagnose_yaw_tail_gradients.py`` (same seed,
augmentation, source-balanced permutation, model states) so results are
directly comparable to the historical A11/A12 diagnosis.

Repair 1 (docs/21 Section 2) -- A11 selector granularity re-diagnosis:
the historical "frame-level evaluator-aligned selector"
(``_yaw_tail_loss_frame_level``) ranked frames by the mean of the raw
training-loss (1-cos) PAIR error, not by the actual evaluator's per-pair
angular-degree error (``_root_yaw_error_degrees``'s own arctan2 +
angle-wrap math). This adds an exact-evaluator-angle frame selector and
compares selected-set overlap and gradient behavior against both the
production pooled selector and the old (1-cos) frame selector.

Repair 2 (docs/21 Section 3) -- A12 yaw-association re-diagnosis: the
historical ``_yaw_association`` used ``(1 - cos(theta)) * 180/pi`` as a
"yaw-degree" surrogate for correlating candidate residuals with orientation
error. That is not a degree quantity (for small theta it scales as
theta^2, not theta) and is not what the production evaluator measures.
This recomputes the same Pearson correlations against the real per-frame
evaluator-degree error.

Usage:
  python3 scripts/repair_historical_selector_diagnostics.py \
    --train-dataset /output/experiments/ablation_a9_fingerprinted_baseline_10e/datasets/direct_mix_train.json \
    --a9-checkpoint /output/experiments/ablation_a9_fingerprinted_baseline_10e/reports/direct_mix.pth \
    --a11-checkpoint /output/experiments/ablation_a11_yaw_tail_10e/reports/direct_mix.pth \
    --a12-checkpoint /output/experiments/ablation_a12_cartesian_torso_tail_10e/reports/direct_mix.pth \
    --out /output/experiments/a21_historical_diagnostic_repair/repair.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnose_yaw_tail_gradients import (  # noqa: E402
    A9_STRUCTURAL_WEIGHTS, SOURCE_NAMES, _first_epoch_batches, _grad_of, _load_model_state, _pearson,
    _torso_attribution, _yaw_axis_error_grid, _yaw_tail_loss_frame_level,
)
from training.temporal_lifter import (  # noqa: E402
    TrainingConfig, YAW_INDICES, _arrays, _supervision_loss, _torch, _yaw_tail_loss, load_dataset,
)


def _exact_evaluator_yaw_degree_grid(torch, prediction, target, valid):
    """(batch, pair) grid of the EXACT production per-pair angular-degree
    error (_root_yaw_error_degrees's own arctan2 + wrapped-difference math),
    vectorized -- not the (1-cos) proxy the training loss and the historical
    diagnostic used."""
    left, right = zip(*YAW_INDICES)
    pair_valid = valid[:, left] & valid[:, right]
    predicted_axis = prediction[:, right, :2] - prediction[:, left, :2]
    target_axis = target[:, right, :2] - target[:, left, :2]
    predicted_length = torch.linalg.vector_norm(predicted_axis, dim=-1)
    target_length = torch.linalg.vector_norm(target_axis, dim=-1)
    stable = pair_valid & (predicted_length > 1e-6) & (target_length > 1e-6)
    predicted_angle = torch.atan2(predicted_axis[..., 1], predicted_axis[..., 0])
    target_angle = torch.atan2(target_axis[..., 1], target_axis[..., 0])
    wrapped = (predicted_angle - target_angle + torch.pi) % (2 * torch.pi) - torch.pi
    degrees = wrapped.abs() * 180.0 / torch.pi
    return degrees, stable


def _exact_evaluator_frame_combined_grid(torch, prediction, target, valid):
    """Per-frame degree error, averaged over available pairs -- exactly how
    the production evaluator's _root_yaw_error_degrees combines shoulder
    and hip before any ranking happens."""
    degrees, stable = _exact_evaluator_yaw_degree_grid(torch, prediction, target, valid)
    count = stable.sum(dim=-1).clamp_min(1)
    combined = (degrees * stable).sum(dim=-1) / count
    frame_stable = stable.any(dim=-1)
    return combined, frame_stable


def _exact_evaluator_frame_selector_loss(torch, prediction, target, valid):
    frame_errors, frame_stable = _exact_evaluator_frame_combined_grid(torch, prediction, target, valid)
    tail_count = ((frame_stable.sum() + 19) // 20).clamp_min(1)
    maximum_tail = max(1, (frame_errors.numel() + 19) // 20)
    selected = torch.topk(frame_errors.masked_fill(~frame_stable, 0.0), maximum_tail).values
    chosen = torch.arange(maximum_tail, device=prediction.device) < tail_count
    return selected.masked_select(chosen).mean()


def _frame_selected_set(torch, frame_errors, frame_stable) -> set[int]:
    tail_count = int(((frame_stable.sum() + 19) // 20).clamp_min(1).item())
    maximum_tail = max(1, (frame_errors.numel() + 19) // 20)
    _values, indices = torch.topk(frame_errors.masked_fill(~frame_stable, 0.0), maximum_tail)
    return set(indices[:tail_count].detach().cpu().numpy().tolist())


def _pooled_selected_frame_set(torch, prediction, target, valid) -> set[int]:
    """The production pooled selector operates on (frame, pair) observations
    directly, not one-per-frame; map its selected pair-observations back to
    the frame indices they touched, for a like-for-like overlap comparison."""
    errors, stable = _yaw_axis_error_grid(torch, prediction, target, valid)
    flattened_errors, flattened_stable = errors.flatten(), stable.flatten()
    tail_count = int(((flattened_stable.sum() + 19) // 20).clamp_min(1).item())
    maximum_tail = max(1, (flattened_errors.numel() + 19) // 20)
    _values, indices = torch.topk(flattened_errors.masked_fill(~flattened_stable, 0.0), maximum_tail)
    selected_indices = indices[:tail_count].detach().cpu().numpy()
    frame_indices = selected_indices // 2
    return set(frame_indices.tolist())


def _selector_repair(torch, model, prediction, target, valid, source_ids) -> dict[str, Any]:
    a9_config = TrainingConfig(window=81, channels=256, epochs=1, batch_size=1, **A9_STRUCTURAL_WEIGHTS)
    mask = valid.unsqueeze(-1).float()
    base_loss = _supervision_loss(torch, prediction, target, mask, a9_config)

    pooled_loss = _yaw_tail_loss(torch, prediction, target, valid)
    old_frame_loss = _yaw_tail_loss_frame_level(torch, prediction, target, valid)
    exact_frame_loss = _exact_evaluator_frame_selector_loss(torch, prediction, target, valid)

    g_base = _grad_of(torch, base_loss, model)

    def stats(loss):
        g = _grad_of(torch, loss, model)
        cosine = torch.nn.functional.cosine_similarity(g_base.unsqueeze(0), g.unsqueeze(0)).item()
        return {
            "raw_loss": float(loss.item()),
            "base_norm": float(g_base.norm().item()),
            "selector_norm": float(g.norm().item()),
            "cosine": float(cosine),
            "selector_to_base_ratio": float(g.norm().item() / g_base.norm().clamp_min(1e-12).item()),
        }

    pooled_frames = _pooled_selected_frame_set(torch, prediction, target, valid)
    exact_degrees, exact_stable = _exact_evaluator_frame_combined_grid(torch, prediction, target, valid)
    old_degrees_proxy, old_stable = _yaw_axis_error_grid(torch, prediction, target, valid)
    old_count = old_stable.sum(dim=-1).clamp_min(1)
    old_frame_combined = (old_degrees_proxy * old_stable).sum(dim=-1) / old_count
    old_frame_stable = old_stable.any(dim=-1)

    old_frames = _frame_selected_set(torch, old_frame_combined, old_frame_stable)
    exact_frames = _frame_selected_set(torch, exact_degrees, exact_stable)

    def jaccard(a: set[int], b: set[int]) -> float:
        union = a | b
        return len(a & b) / len(union) if union else 1.0

    return {
        "pooled_production_selector": stats(pooled_loss),
        "old_frame_1_minus_cos_selector": stats(old_frame_loss),
        "exact_evaluator_angle_frame_selector": stats(exact_frame_loss),
        "selected_set_overlap": {
            "pooled_vs_old_frame_jaccard": jaccard(pooled_frames, old_frames),
            "pooled_vs_exact_frame_jaccard": jaccard(pooled_frames, exact_frames),
            "old_frame_vs_exact_frame_jaccard": jaccard(old_frames, exact_frames),
            "pooled_selected_frame_count": len(pooled_frames),
            "old_frame_selected_count": len(old_frames),
            "exact_frame_selected_count": len(exact_frames),
        },
    }


def _yaw_association_repair(torch, prediction, target, valid, source_ids) -> dict[str, Any]:
    attribution = _torso_attribution(torch, prediction, target, valid, source_ids)
    exact_frame_yaw, exact_frame_stable = _exact_evaluator_frame_combined_grid(torch, prediction, target, valid)

    def frame_average(values, stable):
        # values/stable are (batch, pair) -- shoulder/hip -- for every
        # attribution metric this function is called with.
        common = stable & exact_frame_stable.unsqueeze(-1)
        count = common.sum(dim=-1).clamp_min(1)
        frame_values = (values * common).sum(dim=-1) / count
        frame_valid = common.any(dim=-1)
        return frame_values, frame_valid

    metrics = {
        "a12_cartesian": (attribution["a12_errors"], attribution["geometry"]["pair_valid"]),
        "magnitude": (attribution["magnitude_loss"], attribution["geometry"]["pair_valid"]),
        "direction_scale_restored": (attribution["direction_errors"], attribution["direction_stable"]),
    }
    output: dict[str, Any] = {"frame_count": int(exact_frame_stable.sum().item())}
    for name, (values, stable) in metrics.items():
        frame_values, frame_valid = frame_average(values, stable)
        common_valid = frame_valid & exact_frame_stable
        pearson = _pearson(
            frame_values[common_valid].detach().cpu().numpy(), exact_frame_yaw[common_valid].detach().cpu().numpy(),
        )
        output[name] = {"pearson_r_vs_exact_evaluator_degrees": pearson, "valid_frame_count": int(common_valid.sum().item())}
    return output


def _summarize(records: list[dict[str, Any]], keys: list[str]) -> dict[str, Any]:
    def mean_of(path):
        values = []
        for record in records:
            node = record
            for key in path:
                node = node[key]
            if node is not None and np.isfinite(node):
                values.append(node)
        return float(np.mean(values)) if values else None
    return {key: mean_of(key.split(".")) for key in keys}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-dataset", required=True, type=Path)
    parser.add_argument("--a9-checkpoint", required=True, type=Path)
    parser.add_argument("--a11-checkpoint", required=True, type=Path)
    parser.add_argument("--a12-checkpoint", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    torch, nn = _torch()
    dataset = load_dataset(args.train_dataset)
    inputs, targets, valid, offsets, source_ids, sequence_ranges = _arrays(
        dataset, window=81, include_metadata=True, coordinate_normalization="pelvis_torso_v1",
    )
    device = args.device
    x = torch.as_tensor(inputs, dtype=torch.float32, device=device)
    y = torch.as_tensor(targets, dtype=torch.float32, device=device)
    valid_tensor = torch.as_tensor(valid, dtype=torch.bool, device=device)
    offset_tensor = torch.as_tensor(offsets, dtype=torch.long, device=device)
    source_tensor = torch.as_tensor(source_ids, dtype=torch.long, device=device)

    replay_config = TrainingConfig(
        window=81, channels=256, epochs=1, batch_size=128, seed=args.seed, source_balanced_sampling=True,
        input_jitter_std=0.015, input_dropout_probability=0.05, confidence_jitter_std=0.08,
        input_global_scale_std=0.04, input_translation_std=0.03, input_rotation_degrees=12.0,
        temporal_occlusion_probability=0.10, temporal_occlusion_frames=9,
        input_coordinate_normalization="pelvis_torso_v1", **A9_STRUCTURAL_WEIGHTS,
    )
    epoch_inputs, batches = _first_epoch_batches(
        torch, x, y, valid_tensor, offset_tensor, source_tensor, sequence_ranges, replay_config, args.batch_count,
    )

    model_states = {
        "init": (None, args.seed), "a9_trained": (args.a9_checkpoint, None),
        "a11_trained": (args.a11_checkpoint, None), "a12_trained": (args.a12_checkpoint, None),
    }

    selector_repair_by_state: dict[str, Any] = {}
    yaw_association_by_state: dict[str, Any] = {}
    for state_name, (checkpoint_path, seed) in model_states.items():
        model = _load_model_state(torch, nn, state_name, checkpoint_path, seed or args.seed, device)
        model.train()
        selector_batches, association_batches = [], []
        for batch in batches:
            windows = epoch_inputs[offset_tensor[batch]]
            prediction = model(windows)
            target_batch = y[batch]
            valid_batch = valid_tensor[batch]
            batch_source_ids = source_tensor[batch]
            selector_batches.append(_selector_repair(torch, model, prediction, target_batch, valid_batch, batch_source_ids))
            association_batches.append(_yaw_association_repair(torch, prediction, target_batch, valid_batch, batch_source_ids))
        selector_repair_by_state[state_name] = {
            "batches": selector_batches,
            "summary": {
                "pooled_production_selector": _summarize(selector_batches, [
                    "pooled_production_selector.selector_to_base_ratio", "pooled_production_selector.cosine",
                ]),
                "old_frame_1_minus_cos_selector": _summarize(selector_batches, [
                    "old_frame_1_minus_cos_selector.selector_to_base_ratio", "old_frame_1_minus_cos_selector.cosine",
                ]),
                "exact_evaluator_angle_frame_selector": _summarize(selector_batches, [
                    "exact_evaluator_angle_frame_selector.selector_to_base_ratio",
                    "exact_evaluator_angle_frame_selector.cosine",
                ]),
                "selected_set_overlap": _summarize(selector_batches, [
                    "selected_set_overlap.pooled_vs_old_frame_jaccard", "selected_set_overlap.pooled_vs_exact_frame_jaccard",
                    "selected_set_overlap.old_frame_vs_exact_frame_jaccard",
                ]),
            },
        }
        yaw_association_by_state[state_name] = {
            "batches": association_batches,
            "summary": {
                name: _summarize(association_batches, [f"{name}.pearson_r_vs_exact_evaluator_degrees"])
                for name in ("a12_cartesian", "magnitude", "direction_scale_restored")
            },
        }

    report = {
        "batch_count": len(batches),
        "a11_selector_repair": selector_repair_by_state,
        "a12_yaw_association_repair": yaw_association_by_state,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"batch_count": len(batches), "out": str(args.out)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
