#!/usr/bin/env python3
"""A11 selector-ONLY isolation repair (docs/22 Section 1).

docs/21's "exact_evaluator_angle_frame_selector" both *ranked* and
*differentiated* the real evaluator degree value -- which conflates two
separate questions ("which frames get selected" vs "what penalty scale
those frames are trained against"). This script isolates them with three
paths, replayed on the same fixed batches/model states as every prior A11
diagnosis:

P1 -- production pooled selector (candidates = pair observations,
      ranking = existing production (1-cos), differentiated penalty =
      the same (1-cos)).
P2 -- historical frame-level proxy (candidates = frames, ranking =
      frame-combined (1-cos), differentiated penalty = the same
      frame-combined (1-cos)).
P3 -- EXACT-RANKING CONTROL (candidates = frames, ranking ONLY = exact
      production evaluator root-yaw angular error, computed under
      torch.no_grad() so no gradient ever flows through it; differentiated
      penalty = the SAME frame-combined (1-cos) quantity P2 uses, gathered
      at the P3-selected indices).

P3 isolates selection semantics from penalty representation/scale. It is
NOT the same thing as docs/21's angle-penalty diagnostic (which remains
valid as a separate "what if the penalty itself were exact degrees"
question, kept here as `angle_penalty_diagnostic` and explicitly not used
to support or reject the selector-structure Case).

Diagnostic-only: no new training, no checkpoint/report changes.

Usage:
  python3 scripts/repair_a11_selector_isolation.py \
    --train-dataset /output/experiments/ablation_a9_fingerprinted_baseline_10e/datasets/direct_mix_train.json \
    --a9-checkpoint /output/experiments/ablation_a9_fingerprinted_baseline_10e/reports/direct_mix.pth \
    --a11-checkpoint /output/experiments/ablation_a11_yaw_tail_10e/reports/direct_mix.pth \
    --out /output/experiments/a22_a16_generalization_diagnosis/a11_selector_isolation.json
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
    A9_STRUCTURAL_WEIGHTS, _first_epoch_batches, _grad_of, _load_model_state, _yaw_axis_error_grid,
)
from repair_historical_selector_diagnostics import (  # noqa: E402
    _exact_evaluator_frame_combined_grid, _exact_evaluator_frame_selector_loss,
)
from training.temporal_lifter import TrainingConfig, _arrays, _supervision_loss, _torch, _yaw_tail_loss, load_dataset  # noqa: E402


def _p2_old_frame_combined_grid(torch, prediction, target, valid):
    proxy, stable = _yaw_axis_error_grid(torch, prediction, target, valid)
    count = stable.sum(dim=-1).clamp_min(1)
    combined = (proxy * stable).sum(dim=-1) / count
    frame_stable = stable.any(dim=-1)
    return combined, frame_stable


def _frame_selected_indices(torch, ranking_errors, ranking_stable) -> "torch.Tensor":
    tail_count = int(((ranking_stable.sum() + 19) // 20).clamp_min(1).item())
    maximum_tail = max(1, (ranking_errors.numel() + 19) // 20)
    _values, indices = torch.topk(ranking_errors.masked_fill(~ranking_stable, 0.0), maximum_tail)
    return indices[:tail_count]


def _p3_exact_ranking_fixed_penalty(torch, prediction, target, valid):
    """P3: select frames by the exact evaluator-angle ranking (no gradient
    through that ranking value), then differentiate the SAME frame-combined
    (1-cos) quantity P2 uses, gathered at those selected indices."""
    with torch.no_grad():
        exact_degrees, exact_stable = _exact_evaluator_frame_combined_grid(torch, prediction, target, valid)
    old_frame_combined, old_frame_stable = _p2_old_frame_combined_grid(torch, prediction, target, valid)
    selection_stable = exact_stable & old_frame_stable
    with torch.no_grad():
        selected_indices = _frame_selected_indices(torch, exact_degrees, selection_stable)
    selected_penalty = old_frame_combined[selected_indices]
    return selected_penalty.mean() if selected_penalty.numel() else prediction.new_zeros(()), selected_indices


def _selected_set(indices) -> set[int]:
    return set(indices.detach().cpu().numpy().tolist())


def _p1_selected_frame_indices(torch, prediction, target, valid) -> set[int]:
    """Map P1's pooled (frame, pair) selection back to the frame indices it
    touched, for like-for-like overlap with P2/P3 (each frame-granular)."""
    errors, stable = _yaw_axis_error_grid(torch, prediction, target, valid)
    flattened_errors, flattened_stable = errors.flatten(), stable.flatten()
    tail_count = int(((flattened_stable.sum() + 19) // 20).clamp_min(1).item())
    maximum_tail = max(1, (flattened_errors.numel() + 19) // 20)
    _values, indices = torch.topk(flattened_errors.masked_fill(~flattened_stable, 0.0), maximum_tail)
    selected_indices = indices[:tail_count].detach().cpu().numpy()
    return set((selected_indices // 2).tolist())


def _isolation_repair(torch, model, prediction, target, valid) -> dict[str, Any]:
    a9_config = TrainingConfig(window=81, channels=256, epochs=1, batch_size=1, **A9_STRUCTURAL_WEIGHTS)
    mask = valid.unsqueeze(-1).float()
    base_loss = _supervision_loss(torch, prediction, target, mask, a9_config)
    g_base = _grad_of(torch, base_loss, model)

    p1_loss = _yaw_tail_loss(torch, prediction, target, valid)
    p2_loss = _p2_old_frame_combined_grid(torch, prediction, target, valid)
    p2_frame_combined, p2_frame_stable = p2_loss
    from repair_historical_selector_diagnostics import _yaw_tail_loss_frame_level
    p2_loss_value = _yaw_tail_loss_frame_level(torch, prediction, target, valid)
    p3_loss_value, p3_indices = _p3_exact_ranking_fixed_penalty(torch, prediction, target, valid)

    # docs/21's angle-penalty diagnostic, kept separate and NOT used for the
    # selector-structure Case A/B/C/D conclusion here.
    angle_penalty_loss = _exact_evaluator_frame_selector_loss(torch, prediction, target, valid)

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

    p1_frames = _p1_selected_frame_indices(torch, prediction, target, valid)
    p2_frames = _selected_set(_frame_selected_indices(torch, p2_frame_combined, p2_frame_stable))
    p3_frames = _selected_set(p3_indices)

    def jaccard(a: set[int], b: set[int]) -> float:
        union = a | b
        return len(a & b) / len(union) if union else 1.0

    return {
        "p1_pooled_production_selector": stats(p1_loss),
        "p2_historical_frame_proxy": stats(p2_loss_value),
        "p3_exact_ranking_fixed_penalty": stats(p3_loss_value),
        "angle_penalty_diagnostic_NOT_a_selector_conclusion": stats(angle_penalty_loss),
        "selected_set_overlap": {
            "p1_vs_p2_jaccard": jaccard(p1_frames, p2_frames),
            "p1_vs_p3_jaccard": jaccard(p1_frames, p3_frames),
            "p2_vs_p3_jaccard": jaccard(p2_frames, p3_frames),
            "p1_selected_frame_count": len(p1_frames),
            "p2_selected_frame_count": len(p2_frames),
            "p3_selected_frame_count": len(p3_frames),
        },
    }


def _mean(records: list[dict[str, Any]], path: list[str]):
    values = []
    for record in records:
        node = record
        for key in path:
            node = node[key]
        if node is not None and np.isfinite(node):
            values.append(node)
    return float(np.mean(values)) if values else None


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "p1_selector_to_base_ratio_mean": _mean(records, ["p1_pooled_production_selector", "selector_to_base_ratio"]),
        "p2_selector_to_base_ratio_mean": _mean(records, ["p2_historical_frame_proxy", "selector_to_base_ratio"]),
        "p3_selector_to_base_ratio_mean": _mean(records, ["p3_exact_ranking_fixed_penalty", "selector_to_base_ratio"]),
        "angle_penalty_ratio_mean_NOT_a_selector_conclusion": _mean(
            records, ["angle_penalty_diagnostic_NOT_a_selector_conclusion", "selector_to_base_ratio"],
        ),
        "p1_cosine_mean": _mean(records, ["p1_pooled_production_selector", "cosine"]),
        "p2_cosine_mean": _mean(records, ["p2_historical_frame_proxy", "cosine"]),
        "p3_cosine_mean": _mean(records, ["p3_exact_ranking_fixed_penalty", "cosine"]),
        "p1_vs_p2_jaccard_mean": _mean(records, ["selected_set_overlap", "p1_vs_p2_jaccard"]),
        "p1_vs_p3_jaccard_mean": _mean(records, ["selected_set_overlap", "p1_vs_p3_jaccard"]),
        "p2_vs_p3_jaccard_mean": _mean(records, ["selected_set_overlap", "p2_vs_p3_jaccard"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-dataset", required=True, type=Path)
    parser.add_argument("--a9-checkpoint", required=True, type=Path)
    parser.add_argument("--a11-checkpoint", required=True, type=Path)
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
        "init": (None, args.seed), "a9_trained": (args.a9_checkpoint, None), "a11_trained": (args.a11_checkpoint, None),
    }
    report: dict[str, Any] = {"batch_count": len(batches), "states": {}}
    for state_name, (checkpoint_path, seed) in model_states.items():
        model = _load_model_state(torch, nn, state_name, checkpoint_path, seed or args.seed, device)
        model.train()
        batch_reports = []
        for batch in batches:
            windows = epoch_inputs[offset_tensor[batch]]
            prediction = model(windows)
            batch_reports.append(_isolation_repair(torch, model, prediction, y[batch], valid_tensor[batch]))
        report["states"][state_name] = {"batches": batch_reports, "summary": _summarize(batch_reports)}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"batch_count": len(batches), "out": str(args.out)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
