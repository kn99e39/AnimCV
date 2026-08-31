#!/usr/bin/env python3
"""Fixed-batch gradient diagnosis for the A14 bilateral forward-depth candidate.

Diagnostic instrumentation only -- trains nothing new. Replays the exact
fixed batches A9/A11/A12 all saw in their first epoch (same seed, same
augmentation, same source-balanced permutation as ``train()``) through two
real model states -- fresh init and A9's final checkpoint -- and measures
whether the A14 candidate (docs/10) recreates the A11-style gradient-scale
pathology or behaves like A12's stable Cartesian reconstruction.

Usage:
  python3 scripts/diagnose_bilateral_forward_depth_gradients.py \
    --train-dataset /output/experiments/ablation_a9_fingerprinted_baseline_10e/datasets/direct_mix_train.json \
    --a9-checkpoint /output/experiments/ablation_a9_fingerprinted_baseline_10e/reports/direct_mix.pth \
    --out /output/experiments/a14_bilateral_forward_depth_diagnosis/gradient_diagnosis.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from training.temporal_lifter import (
    TrainingConfig, _arrays, _augment_inputs, _bilateral_forward_depth_diagnostics,
    _bilateral_forward_depth_grid, _bilateral_forward_depth_residual_sum, _model,
    _source_balanced_permutation, _supervision_loss, _torch, load_dataset,
)

# A9's exact structural-loss configuration (bilateral_forward_depth_supervision
# is intentionally NOT included here -- this diagnostic isolates the
# candidate's own gradient against the unmodified A9 base objective).
A9_STRUCTURAL_WEIGHTS = {"bone_loss_weight": 0.25, "torso_loss_weight": 0.15, "hinge_loss_weight": 0.15}
SOURCE_NAMES = ("MPI-INF-3DHP", "3DPW", "AMASS")  # first-seen order in direct_mix (see _arrays/_source_frame_counts)


def _flat_grad(parameters):
    import torch
    pieces = [p.grad.detach().reshape(-1) for p in parameters if p.grad is not None]
    return torch.cat(pieces) if pieces else None


def _grad_of(torch, loss, model):
    model.zero_grad(set_to_none=True)
    loss.backward(retain_graph=True)
    grad = _flat_grad(model.parameters())
    model.zero_grad(set_to_none=True)
    return grad


def _candidate_loss(torch, prediction, target, valid):
    """Raw (coefficient=1.0) A14 candidate: pooled sum / pooled count over
    every valid shoulder+hip pair, exactly what folding into the base
    coordinate mean would add (per-pair, not yet added to the base term)."""
    total, count = _bilateral_forward_depth_residual_sum(torch, prediction, target, valid)
    return total / count.clamp_min(1.0)


def _source_restricted_candidate_loss(torch, prediction, target, valid, source_ids, source_id: int):
    """Isolate one source's contribution to the all-frame candidate mean."""
    q_pred, q_target, pair_valid = _bilateral_forward_depth_grid(torch, prediction, target, valid)
    source_mask = (source_ids == source_id).unsqueeze(-1).expand_as(pair_valid)
    restricted_valid = (pair_valid & source_mask).to(q_pred.dtype)
    residual = torch.nn.functional.smooth_l1_loss(q_pred, q_target, reduction="none") * restricted_valid
    return residual.sum() / restricted_valid.sum().clamp_min(1.0)


def _endpoint_gradient(torch, prediction, loss):
    """Gradient directly on the four shoulder/hip forward-depth endpoints
    (Section 8): confirms the candidate produces the intended anti-symmetric
    correction rather than diffusing into unrelated coordinates."""
    from training.temporal_lifter import FORWARD_DEPTH_AXIS, TORSO_INDICES

    gradient = torch.autograd.grad(loss, prediction, retain_graph=True, allow_unused=True)[0]
    if gradient is None:
        return {"left_shoulder": 0.0, "right_shoulder": 0.0, "left_hip": 0.0, "right_hip": 0.0,
                "non_endpoint_l1": 0.0, "endpoint_l1_share": None}
    left, right = zip(*TORSO_INDICES)
    endpoint_indices = set(left) | set(right)
    total_l1 = float(gradient.abs().sum().item())
    endpoint_l1 = float(gradient[:, list(endpoint_indices), :].abs().sum().item())
    non_forward_depth_l1 = float(
        gradient[:, list(endpoint_indices), :].abs().sum().item()
        - gradient[:, list(endpoint_indices), FORWARD_DEPTH_AXIS].abs().sum().item()
    )
    return {
        "shoulder_forward_depth_grad_l1": float(gradient[:, [left[0], right[0]], FORWARD_DEPTH_AXIS].abs().sum().item()),
        "hip_forward_depth_grad_l1": float(gradient[:, [left[1], right[1]], FORWARD_DEPTH_AXIS].abs().sum().item()),
        "endpoint_non_forward_depth_grad_l1": non_forward_depth_l1,
        "total_grad_l1": total_l1,
        "endpoint_l1_share": endpoint_l1 / total_l1 if total_l1 > 0 else None,
    }


def _gradient_interaction(torch, model, prediction, target, valid, source_ids) -> dict[str, Any]:
    a9_config = TrainingConfig(window=81, channels=256, epochs=1, batch_size=1, **A9_STRUCTURAL_WEIGHTS)
    mask = valid.unsqueeze(-1).float()
    base_loss = _supervision_loss(torch, prediction, target, mask, a9_config)
    candidate_loss = _candidate_loss(torch, prediction, target, valid)

    g_base = _grad_of(torch, base_loss, model)
    g_candidate = _grad_of(torch, candidate_loss, model)

    def stats(g_component, loss=None):
        if g_base is None or g_component is None:
            return {"base_norm": None, "component_norm": None, "combined_norm": None, "cosine": None}
        cosine = torch.nn.functional.cosine_similarity(g_base.unsqueeze(0), g_component.unsqueeze(0)).item()
        result = {
            "base_norm": float(g_base.norm().item()),
            "component_norm": float(g_component.norm().item()),
            "combined_norm": float((g_base + g_component).norm().item()),
            "cosine": float(cosine),
            "component_to_base_ratio": float(g_component.norm().item() / g_base.norm().clamp_min(1e-12).item()),
        }
        if loss is not None:
            result["endpoint_gradient"] = _endpoint_gradient(torch, prediction, loss)
        return result

    per_source_candidate = {}
    for source_id, name in enumerate(SOURCE_NAMES):
        if not (source_ids == source_id).any():
            continue
        composition_share = float((source_ids == source_id).float().mean().item())
        candidate_source_loss = _source_restricted_candidate_loss(torch, prediction, target, valid, source_ids, source_id)
        g_candidate_source = _grad_of(torch, candidate_source_loss, model)
        per_source_candidate[name] = stats(g_candidate_source) | {"batch_composition_share": composition_share,
                                                                    "raw_loss": float(candidate_source_loss.item())}

    return {
        "bilateral_forward_depth_candidate": stats(g_candidate, candidate_loss),
        "per_source_isolated": per_source_candidate,
    }


def _load_model_state(torch, nn, checkpoint_path: Path | None, seed: int, device: str):
    if checkpoint_path is None:
        torch.manual_seed(seed)
        return _model(nn, 256, "dilated_tcn_v1").to(device)
    model = _model(nn, 256, "dilated_tcn_v1").to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    return model


def _first_epoch_batches(torch, x, y, valid_tensor, offsets, source_tensor, sequence_ranges, config, batch_count: int):
    generator = torch.Generator(device=x.device).manual_seed(config.seed)
    epoch_inputs = _augment_inputs(torch, x, config, generator, sequence_ranges)
    indices = torch.arange(len(offsets), device=x.device)
    permutation = _source_balanced_permutation(torch, indices, source_tensor, generator)
    batches = list(permutation.split(config.batch_size))[:batch_count]
    return epoch_inputs, batches


def _mean(values):
    values = [value for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(values)) if values else None


def _percentile(values, percentile):
    values = [value for value in values if value is not None and np.isfinite(value)]
    return float(np.percentile(values, percentile)) if values else None


def _source_wise_diagnostics(torch, prediction, target, valid, source_ids) -> dict[str, Any]:
    """Section 9: per-source valid-pair count, raw residual, and gradient
    norm for the all-frame candidate. No source weighting is derived from
    this -- reporting only."""
    q_pred, q_target, pair_valid = _bilateral_forward_depth_grid(torch, prediction, target, valid)
    output = {}
    for source_id, name in enumerate(SOURCE_NAMES):
        mask = (pair_valid & (source_ids == source_id).unsqueeze(-1).expand_as(pair_valid))
        count = int(mask.sum().item())
        if count == 0:
            output[name] = {"valid_pair_count": 0, "raw_residual_mean": None}
            continue
        residual = torch.nn.functional.smooth_l1_loss(q_pred, q_target, reduction="none")
        output[name] = {
            "valid_pair_count": count,
            "raw_residual_mean": float(residual.masked_select(mask).mean().item()),
        }
    return output


def _summarize_state(batch_reports):
    ratios = [batch["gradients"]["bilateral_forward_depth_candidate"].get("component_to_base_ratio") for batch in batch_reports]
    cosines = [batch["gradients"]["bilateral_forward_depth_candidate"].get("cosine") for batch in batch_reports]
    diagnostics_shoulder_abs = [batch["diagnostics"]["shoulder_forward_depth_abs_residual_m"] for batch in batch_reports]
    diagnostics_hip_abs = [batch["diagnostics"]["hip_forward_depth_abs_residual_m"] for batch in batch_reports]
    diagnostics_shoulder_sign = [batch["diagnostics"]["shoulder_forward_depth_sign_disagreement"] for batch in batch_reports]
    diagnostics_hip_sign = [batch["diagnostics"]["hip_forward_depth_sign_disagreement"] for batch in batch_reports]
    return {
        "batch_count": len(batch_reports),
        "mean_raw_candidate_loss": _mean([batch["candidate_raw_loss"] for batch in batch_reports]),
        "mean_component_to_base_ratio": _mean(ratios),
        "p95_component_to_base_ratio": _percentile(ratios, 95),
        "maximum_component_to_base_ratio": max([r for r in ratios if r is not None], default=None),
        "mean_cosine": _mean(cosines),
        "diagnostics_mean": {
            "shoulder_forward_depth_abs_residual_m": _mean(diagnostics_shoulder_abs),
            "hip_forward_depth_abs_residual_m": _mean(diagnostics_hip_abs),
            "shoulder_forward_depth_sign_disagreement": _mean(diagnostics_shoulder_sign),
            "hip_forward_depth_sign_disagreement": _mean(diagnostics_hip_sign),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-dataset", required=True, type=Path, help="A9-A12's materialized direct_mix_train.json")
    parser.add_argument("--a9-checkpoint", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-count", type=int, default=10, help="how many of epoch 1's real batches to replay")
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

            candidate_loss = _candidate_loss(torch, prediction, target_batch, valid_batch)
            diagnostics = _bilateral_forward_depth_diagnostics(torch, prediction, target_batch, valid_batch)
            gradients = _gradient_interaction(torch, model, prediction, target_batch, valid_batch, batch_source_ids)
            source_wise = _source_wise_diagnostics(torch, prediction, target_batch, valid_batch, batch_source_ids)
            batch_reports.append({
                "candidate_raw_loss": float(candidate_loss.item()),
                "diagnostics": diagnostics,
                "gradients": gradients,
                "source_wise": source_wise,
            })
        report["states"][state_name] = {
            "batches": batch_reports,
            "summary": _summarize_state(batch_reports),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"batch_count": len(batches), "out": str(args.out)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
