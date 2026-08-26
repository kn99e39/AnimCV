#!/usr/bin/env python3
"""Diagnose why A11's isolated yaw_tail_loss destabilized general 3D training.

Diagnostic instrumentation only -- trains nothing new. Replays the exact
fixed batches A9/A11 both saw in their first epoch (same seed, same
augmentation, same source-balanced permutation as ``train()``) through three
real model states -- fresh init, A9's final checkpoint, A11's final
checkpoint -- and decomposes loss magnitude and gradient interaction between
the base A9 objective and the isolated yaw-tail term.

Usage:
  python3 scripts/diagnose_yaw_tail_gradients.py \
    --train-dataset /output/experiments/ablation_a9_fingerprinted_baseline_10e/datasets/direct_mix_train.json \
    --a9-checkpoint /output/experiments/ablation_a9_fingerprinted_baseline_10e/reports/direct_mix.pth \
    --a11-checkpoint /output/experiments/ablation_a11_yaw_tail_10e/reports/direct_mix.pth \
    --out audit/a11_gradient_diagnosis.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from training.temporal_lifter import (
    TrainingConfig, _arrays, _augment_inputs, _cartesian_torso_tail_loss, _hinge_loss, _model,
    _source_balanced_permutation, _supervision_loss, _torch, _torso_vector_error_grid, _vector_loss,
    _yaw_axis_error_grid, _yaw_tail_loss, BONE_INDICES, TORSO_INDICES, load_dataset,
)

# A9's exact structural-loss configuration; only yaw_tail_loss_weight differs
# between A9 (0.0) and A11 (0.05). Reconstructed here rather than imported
# from any single experiment's recorded config, since this script must build
# a config independent of which checkpoint it happens to be probing.
A9_STRUCTURAL_WEIGHTS = {"bone_loss_weight": 0.25, "torso_loss_weight": 0.15, "hinge_loss_weight": 0.15}
SOURCE_NAMES = ("MPI-INF-3DHP", "3DPW", "AMASS")  # first-seen order in direct_mix (see _arrays/_source_frame_counts)


def _frame_combined_error_grid(torch, prediction, target, valid):
    """Frame-combined yaw error: average the available (shoulder, hip) pair
    errors *per frame first* -- matching how the evaluator's own
    ``_root_yaw_error_degrees`` combines pairs before ranking frames, unlike
    the production ``_yaw_tail_loss`` which ranks pooled pair observations
    directly. Returns (per-frame error, per-frame stable-mask).
    """
    errors, stable = _yaw_axis_error_grid(torch, prediction, target, valid)  # (batch, 2)
    count = stable.sum(dim=-1).clamp_min(1)
    combined = (errors * stable).sum(dim=-1) / count
    frame_stable = stable.any(dim=-1)
    return combined, frame_stable


def _yaw_tail_loss_frame_level(torch, prediction, target, valid):
    """Counterfactual, diagnostic-only tail selector: same CVaR-5% mechanism
    and the same underlying (1-cos) error as the production loss, but ranks
    *frames* by their combined error instead of pooling every (frame, pair)
    observation. Never used for training in this batch (Section 6: diagnostic
    only)."""
    frame_errors, frame_stable = _frame_combined_error_grid(torch, prediction, target, valid)
    tail_count = ((frame_stable.sum() + 19) // 20).clamp_min(1)
    maximum_tail = max(1, (frame_errors.numel() + 19) // 20)
    selected = torch.topk(frame_errors.masked_fill(~frame_stable, 0.0), maximum_tail).values
    chosen = torch.arange(maximum_tail, device=prediction.device) < tail_count
    return selected.masked_select(chosen).mean()


def _pooled_selection_detail(torch, prediction, target, valid, source_ids) -> dict[str, Any]:
    """Replicate _yaw_tail_loss's exact selection mechanics but return which
    (frame, pair) entries were chosen, instead of only the loss scalar --
    needed to characterize the real production selector (Section 5) and to
    attribute the real pooled loss to sources (Section 4) without changing
    what gets selected.
    """
    errors, stable = _yaw_axis_error_grid(torch, prediction, target, valid)  # (batch, 2)
    batch_size = errors.shape[0]
    flattened_errors, flattened_stable = errors.flatten(), stable.flatten()
    tail_count = int(((flattened_stable.sum() + 19) // 20).clamp_min(1).item())
    maximum_tail = max(1, (flattened_errors.numel() + 19) // 20)
    values, indices = torch.topk(flattened_errors.masked_fill(~flattened_stable, 0.0), maximum_tail)
    selected_values = values[:tail_count]
    selected_indices = indices[:tail_count].detach().cpu().numpy()
    frame_indices = selected_indices // 2
    pair_indices = selected_indices % 2  # 0=shoulder, 1=hip (YAW_INDICES order)
    selected_sources = source_ids[frame_indices].detach().cpu().numpy() if len(frame_indices) else np.asarray([])

    shoulder_only = int(((pair_indices == 0).sum()))
    hip_only = int(((pair_indices == 1).sum()))
    both_frames = len(set(frame_indices.tolist())) if len(frame_indices) else 0

    source_share: dict[str, float] = {}
    total = float(selected_values.sum().item()) if len(selected_values) else 0.0
    for source_id, name in enumerate(SOURCE_NAMES):
        mask = selected_sources == source_id
        share = float(selected_values[:tail_count][mask].sum().item()) / total if total > 0 and mask.any() else 0.0
        source_share[name] = share

    return {
        "candidate_count": int(flattened_errors.numel()),
        "stable_candidate_count": int(flattened_stable.sum().item()),
        "selected_count": tail_count,
        "selected_frame_count": both_frames,
        "shoulder_only_selections": shoulder_only,
        "hip_only_selections": hip_only,
        "selected_frame_indices": frame_indices.tolist(),
        "loss_share_by_source": source_share,
        "batch_size": batch_size,
    }


def _source_restricted_yaw_tail_loss(torch, prediction, target, valid, source_ids, source_id: int):
    """Isolate the gradient a single source's samples would contribute if
    they were the *only* candidates in the tail pool -- a deliberate
    construction (not the real joint-pool selection) used only to compare
    per-source gradient magnitude/direction (Section 4). Distinct from
    ``_pooled_selection_detail``'s source attribution of the real,
    jointly-selected pool.
    """
    errors, stable = _yaw_axis_error_grid(torch, prediction, target, valid)
    source_mask = (source_ids == source_id).unsqueeze(-1).expand_as(stable)
    restricted_stable = stable & source_mask
    flattened_errors, flattened_stable = errors.flatten(), restricted_stable.flatten()
    tail_count = ((flattened_stable.sum() + 19) // 20).clamp_min(1)
    maximum_tail = max(1, (flattened_errors.numel() + 19) // 20)
    selected = torch.topk(flattened_errors.masked_fill(~flattened_stable, 0.0), maximum_tail).values
    chosen = torch.arange(maximum_tail, device=prediction.device) < tail_count
    return selected.masked_select(chosen).mean()


def _torso_pooled_selection_detail(torch, prediction, target, valid, source_ids) -> dict[str, Any]:
    """Same accounting as ``_pooled_selection_detail``, but over the
    Cartesian torso-vector grid instead of the angular yaw grid -- so the
    Section 2 candidate's real selection/attribution can be compared to the
    angular loss's on identical terms (Section 5-6)."""
    errors, stable = _torso_vector_error_grid(torch, prediction, target, valid)  # (batch, 2)
    batch_size = errors.shape[0]
    flattened_errors, flattened_stable = errors.flatten(), stable.flatten()
    tail_count = int(((flattened_stable.sum() + 19) // 20).clamp_min(1).item())
    maximum_tail = max(1, (flattened_errors.numel() + 19) // 20)
    values, indices = torch.topk(flattened_errors.masked_fill(~flattened_stable, 0.0), maximum_tail)
    selected_values = values[:tail_count]
    selected_indices = indices[:tail_count].detach().cpu().numpy()
    frame_indices = selected_indices // 2
    pair_indices = selected_indices % 2  # 0=shoulder, 1=hip (TORSO_INDICES order)
    selected_sources = source_ids[frame_indices].detach().cpu().numpy() if len(frame_indices) else np.asarray([])

    shoulder_only = int(((pair_indices == 0).sum()))
    hip_only = int(((pair_indices == 1).sum()))
    both_frames = len(set(frame_indices.tolist())) if len(frame_indices) else 0

    source_share: dict[str, float] = {}
    total = float(selected_values.sum().item()) if len(selected_values) else 0.0
    for source_id, name in enumerate(SOURCE_NAMES):
        mask = selected_sources == source_id
        share = float(selected_values[:tail_count][mask].sum().item()) / total if total > 0 and mask.any() else 0.0
        source_share[name] = share

    return {
        "candidate_count": int(flattened_errors.numel()),
        "stable_candidate_count": int(flattened_stable.sum().item()),
        "selected_count": tail_count,
        "selected_frame_count": both_frames,
        "shoulder_only_selections": shoulder_only,
        "hip_only_selections": hip_only,
        "selected_frame_indices": frame_indices.tolist(),
        "loss_share_by_source": source_share,
        "batch_size": batch_size,
    }


def _source_restricted_cartesian_torso_tail_loss(torch, prediction, target, valid, source_ids, source_id: int):
    """Cartesian-candidate counterpart to ``_source_restricted_yaw_tail_loss``
    -- isolates one source's contribution to the tail-selected Cartesian
    torso penalty (Section 6), same deliberate single-source-pool
    construction, not the real joint-pool selection."""
    errors, stable = _torso_vector_error_grid(torch, prediction, target, valid)
    source_mask = (source_ids == source_id).unsqueeze(-1).expand_as(stable)
    restricted_stable = stable & source_mask
    flattened_errors, flattened_stable = errors.flatten(), restricted_stable.flatten()
    tail_count = ((flattened_stable.sum() + 19) // 20).clamp_min(1)
    maximum_tail = max(1, (flattened_errors.numel() + 19) // 20)
    selected = torch.topk(flattened_errors.masked_fill(~flattened_stable, 0.0), maximum_tail).values
    chosen = torch.arange(maximum_tail, device=prediction.device) < tail_count
    return selected.masked_select(chosen).mean()


def _component_losses(torch, prediction, target, valid) -> dict[str, float]:
    """Raw (unweighted, coefficient=1.0) structural-loss magnitudes, reusing
    the production helpers directly so these numbers cannot drift from what
    ``_supervision_loss`` actually computes."""
    mask = valid.unsqueeze(-1).float()
    coordinate = (torch.nn.functional.smooth_l1_loss(prediction, target, reduction="none") * mask).sum() / mask.sum().clamp_min(1.0)
    bone = _vector_loss(torch, prediction, target, valid, BONE_INDICES, lambda first, second: first - second)
    torso = _vector_loss(torch, prediction, target, valid, TORSO_INDICES, lambda first, second: second - first)
    hinge = _hinge_loss(torch, prediction, target, valid)
    yaw_tail = _yaw_tail_loss(torch, prediction, target, valid)
    yaw_tail_frame_level = _yaw_tail_loss_frame_level(torch, prediction, target, valid)
    cartesian_torso_tail = _cartesian_torso_tail_loss(torch, prediction, target, valid)
    return {
        "coordinate": float(coordinate.item()),
        "bone": float(bone.item()),
        "torso": float(torso.item()),
        "hinge": float(hinge.item()),
        "yaw_tail_pooled": float(yaw_tail.item()),
        "yaw_tail_frame_level": float(yaw_tail_frame_level.item()),
        "cartesian_torso_tail": float(cartesian_torso_tail.item()),
    }


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


def _gradient_interaction(torch, model, prediction, target, valid, source_ids) -> dict[str, Any]:
    a9_config = TrainingConfig(window=81, channels=256, epochs=1, batch_size=1, **A9_STRUCTURAL_WEIGHTS)
    mask = valid.unsqueeze(-1).float()
    base_loss = _supervision_loss(torch, prediction, target, mask, a9_config)
    yaw_loss_pooled = _yaw_tail_loss(torch, prediction, target, valid)
    yaw_loss_frame = _yaw_tail_loss_frame_level(torch, prediction, target, valid)
    candidate_loss = _cartesian_torso_tail_loss(torch, prediction, target, valid)

    g_base = _grad_of(torch, base_loss, model)
    g_yaw_pooled = _grad_of(torch, yaw_loss_pooled, model)
    g_yaw_frame = _grad_of(torch, yaw_loss_frame, model)
    g_candidate = _grad_of(torch, candidate_loss, model)

    def stats(g_yaw):
        if g_base is None or g_yaw is None:
            return {"base_norm": None, "yaw_norm": None, "combined_norm": None, "cosine": None}
        cosine = torch.nn.functional.cosine_similarity(g_base.unsqueeze(0), g_yaw.unsqueeze(0)).item()
        return {
            "base_norm": float(g_base.norm().item()),
            "yaw_norm": float(g_yaw.norm().item()),
            "combined_norm": float((g_base + g_yaw).norm().item()),
            "cosine": float(cosine),
        }

    per_source_yaw = {}
    per_source_candidate = {}
    for source_id, name in enumerate(SOURCE_NAMES):
        if not (source_ids == source_id).any():
            continue
        composition_share = float((source_ids == source_id).float().mean().item())
        yaw_source_loss = _source_restricted_yaw_tail_loss(torch, prediction, target, valid, source_ids, source_id)
        g_yaw_source = _grad_of(torch, yaw_source_loss, model)
        # This source's share of *this batch* (expected ~= 1/3 under
        # source-balanced sampling) -- NOT its share of the real pooled tail
        # selection, which is reported separately in loss_share_by_source.
        per_source_yaw[name] = stats(g_yaw_source) | {"batch_composition_share": composition_share}
        candidate_source_loss = _source_restricted_cartesian_torso_tail_loss(
            torch, prediction, target, valid, source_ids, source_id,
        )
        g_candidate_source = _grad_of(torch, candidate_source_loss, model)
        per_source_candidate[name] = stats(g_candidate_source) | {"batch_composition_share": composition_share}

    return {
        "pooled_selector": stats(g_yaw_pooled),
        "frame_level_selector": stats(g_yaw_frame),
        "cartesian_torso_tail_candidate": stats(g_candidate),
        "per_source_isolated": per_source_yaw,
        "per_source_isolated_candidate": per_source_candidate,
    }


def _load_model_state(torch, nn, name: str, checkpoint_path: Path | None, seed: int, device: str):
    model = _model(nn, 256, "dilated_tcn_v1").to(device)
    if checkpoint_path is None:
        torch.manual_seed(seed)
        model = _model(nn, 256, "dilated_tcn_v1").to(device)
    else:
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-dataset", required=True, type=Path, help="A9/A11's materialized direct_mix_train.json")
    parser.add_argument("--a9-checkpoint", required=True, type=Path)
    parser.add_argument("--a11-checkpoint", required=True, type=Path)
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

    model_states = {
        "init": (None, args.seed),
        "a9_trained": (args.a9_checkpoint, None),
        "a11_trained": (args.a11_checkpoint, None),
    }

    report: dict[str, Any] = {"batch_count": len(batches), "states": {}}
    for state_name, (checkpoint_path, seed) in model_states.items():
        model = _load_model_state(torch, nn, state_name, checkpoint_path, seed or args.seed, device)
        model.train()
        batch_reports = []
        for batch in batches:
            windows = epoch_inputs[offset_tensor[batch]]
            prediction = model(windows)
            target_batch = y[batch]
            valid_batch = valid_tensor[batch]
            batch_source_ids = source_tensor[batch]

            components = _component_losses(torch, prediction, target_batch, valid_batch)
            selection = _pooled_selection_detail(torch, prediction, target_batch, valid_batch, batch_source_ids)
            candidate_selection = _torso_pooled_selection_detail(
                torch, prediction, target_batch, valid_batch, batch_source_ids,
            )
            gradients = _gradient_interaction(torch, model, prediction, target_batch, valid_batch, batch_source_ids)
            batch_reports.append({
                "components_raw": components,
                "components_weighted": {
                    "bone": components["bone"] * A9_STRUCTURAL_WEIGHTS["bone_loss_weight"],
                    "torso": components["torso"] * A9_STRUCTURAL_WEIGHTS["torso_loss_weight"],
                    "hinge": components["hinge"] * A9_STRUCTURAL_WEIGHTS["hinge_loss_weight"],
                    "yaw_tail_pooled_at_0.05": components["yaw_tail_pooled"] * 0.05,
                    "cartesian_torso_tail_at_0.05": components["cartesian_torso_tail"] * 0.05,
                },
                "pooled_selection": selection,
                "cartesian_torso_tail_selection": candidate_selection,
                "gradients": gradients,
            })
        report["states"][state_name] = batch_reports

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"batch_count": len(batches), "out": str(args.out)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
