#!/usr/bin/env python3
"""Actual Delta_G diagnostic for the docs/21 corrected bilateral
forward-depth candidate (distinct from historical A14's own standalone
gradient diagnostic).

Diagnostic-only: no training, no change to any checkpoint. Reuses the exact
fixed-batch replay infrastructure from
``scripts/diagnose_bilateral_forward_depth_gradients.py`` (same seed,
augmentation, source-balanced permutation, model states) so results are
directly comparable to the historical A14 gradient diagnosis.

At each model state (fresh init, A9 checkpoint) and each of the same 10
fixed first-epoch batches, computes:

    G_A9         = gradient of the unmodified historical A9 objective
    G_candidate  = gradient of A9 + corrected relational term
                   (S_coord/D_coord + S_relational/D_coord + structural)
    Delta_G      = G_candidate - G_A9
    G_relational = gradient of S_relational/D_coord alone

and verifies Delta_G == G_relational within numerical tolerance -- i.e. the
corrected candidate's only mathematical difference from A9 is exactly the
new relational term, with no attenuation of the existing coordinate
gradient (the bug found in historical A14).

Usage:
  python3 scripts/diagnose_corrected_srd_gradient_delta.py \
    --train-dataset /output/experiments/ablation_a9_fingerprinted_baseline_10e/datasets/direct_mix_train.json \
    --a9-checkpoint /output/experiments/ablation_a9_fingerprinted_baseline_10e/reports/direct_mix.pth \
    --out /output/experiments/a21_historical_diagnostic_repair/corrected_gradient_delta.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnose_bilateral_forward_depth_gradients import (  # noqa: E402
    A9_STRUCTURAL_WEIGHTS, SOURCE_NAMES, _first_epoch_batches, _flat_grad, _load_model_state,
)
from training.temporal_lifter import (  # noqa: E402
    TrainingConfig, _arrays, _bilateral_forward_depth_residual_sum, _supervision_loss, _torch, load_dataset,
)


def _grad_of(torch, loss, model):
    model.zero_grad(set_to_none=True)
    loss.backward(retain_graph=True)
    grad = _flat_grad(model.parameters())
    model.zero_grad(set_to_none=True)
    return grad


def _relational_term_loss(torch, prediction, target, mask, valid):
    """S_relational / D_coord alone, using the SAME D_coord (mask.sum()) the
    base coordinate term uses -- exactly what the corrected candidate adds
    to A9's numerator."""
    relational_sum, _count = _bilateral_forward_depth_residual_sum(torch, prediction, target, valid)
    return relational_sum / mask.sum().clamp_min(1.0)


def _source_restricted_relational_loss(torch, prediction, target, mask, valid, source_ids, source_id: int):
    from training.temporal_lifter import _bilateral_forward_depth_grid

    q_pred, q_target, pair_valid = _bilateral_forward_depth_grid(torch, prediction, target, valid)
    source_mask = (source_ids == source_id).unsqueeze(-1).expand_as(pair_valid)
    restricted_valid = (pair_valid & source_mask).to(q_pred.dtype)
    residual = torch.nn.functional.smooth_l1_loss(q_pred, q_target, reduction="none") * restricted_valid
    return residual.sum() / mask.sum().clamp_min(1.0)


def _gradient_delta(torch, model, prediction, target, valid, source_ids) -> dict[str, Any]:
    a9_config = TrainingConfig(window=81, channels=256, epochs=1, batch_size=1, **A9_STRUCTURAL_WEIGHTS)
    candidate_config = TrainingConfig(
        window=81, channels=256, epochs=1, batch_size=1,
        bilateral_forward_depth_supervision_corrected=True, **A9_STRUCTURAL_WEIGHTS,
    )
    mask = valid.unsqueeze(-1).float()

    loss_a9 = _supervision_loss(torch, prediction, target, mask, a9_config)
    loss_candidate = _supervision_loss(torch, prediction, target, mask, candidate_config)
    loss_relational = _relational_term_loss(torch, prediction, target, mask, valid)

    g_a9 = _grad_of(torch, loss_a9, model)
    g_candidate = _grad_of(torch, loss_candidate, model)
    g_relational = _grad_of(torch, loss_relational, model)

    delta_g = g_candidate - g_a9
    equivalence_abs_diff = (delta_g - g_relational).norm().item()
    equivalence_rel_diff = equivalence_abs_diff / g_relational.norm().clamp_min(1e-12).item()

    cosine_a9_delta = torch.nn.functional.cosine_similarity(g_a9.unsqueeze(0), delta_g.unsqueeze(0)).item()

    per_source = {}
    for source_id, name in enumerate(SOURCE_NAMES):
        if not (source_ids == source_id).any():
            continue
        source_relational_loss = _source_restricted_relational_loss(torch, prediction, target, mask, valid, source_ids, source_id)
        g_source_relational = _grad_of(torch, source_relational_loss, model)
        per_source[name] = {
            "raw_loss": float(source_relational_loss.item()),
            "gradient_norm": float(g_source_relational.norm().item()),
        }

    return {
        "loss_a9": float(loss_a9.item()),
        "loss_candidate": float(loss_candidate.item()),
        "loss_relational_alone": float(loss_relational.item()),
        "loss_decomposition_abs_diff": abs(float(loss_candidate.item()) - (float(loss_a9.item()) + float(loss_relational.item()))),
        "g_a9_norm": float(g_a9.norm().item()),
        "g_candidate_norm": float(g_candidate.norm().item()),
        "delta_g_norm": float(delta_g.norm().item()),
        "g_relational_norm": float(g_relational.norm().item()),
        "delta_g_vs_g_relational_abs_diff": equivalence_abs_diff,
        "delta_g_vs_g_relational_rel_diff": equivalence_rel_diff,
        "cosine_g_a9_delta_g": cosine_a9_delta,
        "delta_g_to_g_a9_ratio": float(delta_g.norm().item() / g_a9.norm().clamp_min(1e-12).item()),
        "per_source_relational": per_source,
    }


def _mean(values):
    values = [value for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(values)) if values else None


def _summarize(batch_reports):
    return {
        "g_a9_norm_mean": _mean([b["g_a9_norm"] for b in batch_reports]),
        "delta_g_norm_mean": _mean([b["delta_g_norm"] for b in batch_reports]),
        "g_candidate_norm_mean": _mean([b["g_candidate_norm"] for b in batch_reports]),
        "cosine_g_a9_delta_g_mean": _mean([b["cosine_g_a9_delta_g"] for b in batch_reports]),
        "delta_g_to_g_a9_ratio_mean": _mean([b["delta_g_to_g_a9_ratio"] for b in batch_reports]),
        "max_delta_g_vs_g_relational_rel_diff": max((b["delta_g_vs_g_relational_rel_diff"] for b in batch_reports), default=None),
        "max_loss_decomposition_abs_diff": max((b["loss_decomposition_abs_diff"] for b in batch_reports), default=None),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-dataset", required=True, type=Path)
    parser.add_argument("--a9-checkpoint", required=True, type=Path)
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

    model_states = {"init": (None, args.seed), "a9_trained": (args.a9_checkpoint, None)}
    report: dict[str, Any] = {"batch_count": len(batches), "states": {}}
    for state_name, (checkpoint_path, seed) in model_states.items():
        model = _load_model_state(torch, nn, checkpoint_path, seed or args.seed, device)
        model.train()
        batch_reports = []
        for batch in batches:
            windows = epoch_inputs[offset_tensor[batch]]
            prediction = model(windows)
            target_batch = y[batch]
            valid_batch = valid_tensor[batch]
            batch_source_ids = source_tensor[batch]
            batch_reports.append(_gradient_delta(torch, model, prediction, target_batch, valid_batch, batch_source_ids))
        report["states"][state_name] = {"batches": batch_reports, "summary": _summarize(batch_reports)}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"batch_count": len(batches), "out": str(args.out)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
