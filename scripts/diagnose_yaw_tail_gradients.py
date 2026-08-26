#!/usr/bin/env python3
"""Diagnose why A11's isolated yaw_tail_loss destabilized general 3D training.

Diagnostic instrumentation only -- trains nothing new. Replays the exact
fixed batches A9/A11/A12 all saw in their first epoch (same seed, same
augmentation, same source-balanced permutation as ``train()``) through four
real model states -- fresh init, A9's final checkpoint, A11's final
checkpoint, and A12's final checkpoint -- and decomposes A12's Cartesian
torso-tail signal into magnitude and direction contributions.

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
    _scale_restored_direction_torso_error_grid, _scale_restored_direction_torso_tail_loss,
    _source_balanced_permutation, _supervision_loss, _torch, _torso_vector_error_grid, _vector_loss,
    _torso_vector_geometry_grid, _yaw_axis_error_grid, _yaw_tail_loss,
    BONE_INDICES, TORSO_INDICES, load_dataset,
)

# A9's exact structural-loss configuration; only yaw_tail_loss_weight differs
# between A9 (0.0) and A11 (0.05). Reconstructed here rather than imported
# from any single experiment's recorded config, since this script must build
# a config independent of which checkpoint it happens to be probing.
A9_STRUCTURAL_WEIGHTS = {"bone_loss_weight": 0.25, "torso_loss_weight": 0.15, "hinge_loss_weight": 0.15}
SOURCE_NAMES = ("MPI-INF-3DHP", "3DPW", "AMASS")  # first-seen order in direct_mix (see _arrays/_source_frame_counts)


def _pooled_selection_mask(torch, errors, stable):
    """Return the exact top-5% pooled-tail mask used by A12-style losses."""
    flattened_errors, flattened_stable = errors.flatten(), stable.flatten()
    tail_count = int(((flattened_stable.sum() + 19) // 20).clamp_min(1).item())
    maximum_tail = max(1, (flattened_errors.numel() + 19) // 20)
    _values, indices = torch.topk(flattened_errors.masked_fill(~flattened_stable, 0.0), maximum_tail)
    selected = torch.zeros_like(flattened_stable)
    selected[indices[:tail_count]] = True
    return selected.reshape_as(stable), tail_count


def _selected_mean(torch, values, selected):
    chosen = values.masked_select(selected)
    return chosen.mean() if chosen.numel() else values.sum() * 0.0


def _source_shares(torch, selected, source_ids, values: dict[str, Any]) -> dict[str, Any]:
    """Attribute selected pair observations by source and by loss mass."""
    source_grid = source_ids.unsqueeze(-1).expand_as(selected)
    total_count = int(selected.sum().item())
    total_by_value = {
        name: float(tensor.masked_select(selected).sum().item())
        for name, tensor in values.items()
    }
    output = {}
    for source_id, source_name in enumerate(SOURCE_NAMES):
        mask = selected & (source_grid == source_id)
        item = {"selected_count": int(mask.sum().item())}
        item["selected_fraction"] = item["selected_count"] / total_count if total_count else 0.0
        for name, tensor in values.items():
            mass = float(tensor.masked_select(mask).sum().item())
            item[f"{name}_loss_mass"] = mass
            item[f"{name}_loss_share"] = mass / total_by_value[name] if total_by_value[name] > 0 else 0.0
        output[source_name] = item
    return output


def _torso_attribution(torch, prediction, target, valid, source_ids):
    """Decompose A12's selected vector residual and the direction counterfactual."""
    geometry = _torso_vector_geometry_grid(torch, prediction, target, valid)
    a12_errors, a12_stable = _torso_vector_error_grid(torch, prediction, target, valid)
    a12_selected, a12_selected_count = _pooled_selection_mask(torch, a12_errors, a12_stable)

    predicted_lengths = geometry["predicted_lengths"]
    target_lengths = geometry["target_lengths"]
    magnitude_residual = predicted_lengths - target_lengths
    magnitude_abs = magnitude_residual.abs()
    magnitude_loss = torch.nn.functional.smooth_l1_loss(
        predicted_lengths, target_lengths, reduction="none",
    )
    chord = geometry["direction_chord"]
    chord_norm = torch.linalg.vector_norm(chord, dim=-1)
    direction_residual = target_lengths.detach().unsqueeze(-1) * chord
    direction_loss = torch.nn.functional.smooth_l1_loss(
        direction_residual, torch.zeros_like(direction_residual), reduction="none",
    ).mean(dim=-1)

    # Exact Euclidean energy identity for attribution:
    # ||v_pred-v_gt||^2 = (||v_pred||-||v_gt||)^2
    #                    + ||v_pred|| ||v_gt|| ||u_pred-u_gt||^2.
    magnitude_energy = magnitude_residual.square()
    direction_energy = predicted_lengths * target_lengths * chord_norm.square()
    total_energy = (geometry["predicted_vectors"] - geometry["target_vectors"]).square().sum(dim=-1)

    direction_errors, direction_stable, direction_residual, direction_geometry = (
        _scale_restored_direction_torso_error_grid(torch, prediction, target, valid)
    )
    direction_selected, direction_selected_count = _pooled_selection_mask(
        torch, direction_errors, direction_stable,
    )

    def selected_summary(selected, stable, errors):
        selected_magnitude_energy = magnitude_energy.masked_select(selected)
        selected_direction_energy = direction_energy.masked_select(selected)
        energy_denominator = selected_magnitude_energy.sum() + selected_direction_energy.sum()
        selected_a12 = errors.masked_select(selected)
        return {
            "candidate_count": int(a12_errors.numel()),
            "stable_candidate_count": int(stable.sum().item()),
            "selected_count": int(selected.sum().item()),
            "selected_fraction_of_stable": float(selected.sum().item() / stable.sum().clamp_min(1).item()),
            "a12_vector_loss": float(selected_a12.mean().item()) if selected_a12.numel() else 0.0,
            "predicted_length_mean": float(geometry["predicted_lengths"].masked_select(selected).mean().item()) if selected.any() else 0.0,
            "target_length_mean": float(geometry["target_lengths"].masked_select(selected).mean().item()) if selected.any() else 0.0,
            "magnitude_residual_abs_mean": float(magnitude_abs.masked_select(selected).mean().item()) if selected.any() else 0.0,
            "magnitude_loss": float(magnitude_loss.masked_select(selected).mean().item()) if selected.any() else 0.0,
            "direction_chord_norm_mean": float(chord_norm.masked_select(selected).mean().item()) if selected.any() else 0.0,
            "direction_loss_scale_restored": float(direction_loss.masked_select(selected).mean().item()) if selected.any() else 0.0,
            "magnitude_energy": float(selected_magnitude_energy.mean().item()) if selected_magnitude_energy.numel() else 0.0,
            "direction_energy": float(selected_direction_energy.mean().item()) if selected_direction_energy.numel() else 0.0,
            "magnitude_energy_fraction": float((selected_magnitude_energy.sum() / energy_denominator.clamp_min(1e-12)).item()),
            "direction_energy_fraction": float((selected_direction_energy.sum() / energy_denominator.clamp_min(1e-12)).item()),
        }

    a12_values = {
        "a12": a12_errors,
        "magnitude": magnitude_loss,
        "direction": direction_loss,
        "magnitude_energy": magnitude_energy,
        "direction_energy": direction_energy,
    }
    direction_values = {
        "direction": direction_errors,
        "magnitude": magnitude_loss,
        "direction_energy": direction_energy,
    }
    return {
        "geometry": geometry,
        "a12_errors": a12_errors,
        "a12_selected": a12_selected,
        "a12_selected_count": a12_selected_count,
        "direction_errors": direction_errors,
        "direction_stable": direction_stable,
        "direction_residual": direction_residual,
        "direction_geometry": direction_geometry,
        "direction_selected": direction_selected,
        "direction_selected_count": direction_selected_count,
        "magnitude_residual": magnitude_residual,
        "magnitude_abs": magnitude_abs,
        "magnitude_loss": magnitude_loss,
        "direction_chord": chord,
        "direction_chord_norm": chord_norm,
        "direction_loss": direction_loss,
        "magnitude_energy": magnitude_energy,
        "direction_energy": direction_energy,
        "total_energy": total_energy,
        "a12_summary": selected_summary(a12_selected, a12_stable, a12_errors),
        "direction_summary": selected_summary(direction_selected, direction_stable, direction_errors),
        "a12_source_shares": _source_shares(torch, a12_selected, source_ids, a12_values),
        "direction_source_shares": _source_shares(torch, direction_selected, source_ids, direction_values),
    }


def _pearson(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 2 or np.std(x[finite]) <= 1e-12 or np.std(y[finite]) <= 1e-12:
        return None
    return float(np.corrcoef(x[finite], y[finite])[0, 1])


def _yaw_association(torch, prediction, target, valid):
    """Compare pair residuals with the evaluator's per-frame root-yaw error."""
    attribution = _torso_attribution(torch, prediction, target, valid, torch.zeros(
        (prediction.shape[0],), dtype=torch.long, device=prediction.device,
    ))
    yaw_errors, yaw_stable = _yaw_axis_error_grid(torch, prediction, target, valid)
    yaw_count = yaw_stable.sum(dim=-1).clamp_min(1)
    frame_yaw = (yaw_errors * yaw_stable).sum(dim=-1) / yaw_count * 180.0 / torch.pi
    frame_valid = yaw_stable.any(dim=-1)

    def frame_average(values, stable):
        common = stable & yaw_stable
        count = common.sum(dim=-1).clamp_min(1)
        return (values * common).sum(dim=-1) / count, common.any(dim=-1) & frame_valid

    metrics = {
        "a12_cartesian": attribution["a12_errors"],
        "magnitude": attribution["magnitude_loss"],
        "direction_scale_restored": attribution["direction_errors"],
        "historical_angular": yaw_errors,
    }
    output = {"frame_count": int(frame_valid.sum().item())}
    for name, values in metrics.items():
        frame_values, stable = frame_average(values, attribution["geometry"]["pair_valid"] if name != "direction_scale_restored" else attribution["direction_stable"])
        output[name] = {"pearson_r": _pearson(frame_values[stable].detach().cpu().numpy(), frame_yaw[stable].detach().cpu().numpy()),
                        "valid_frame_count": int(stable.sum().item())}
    return output


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


def _source_restricted_direction_torso_tail_loss(torch, prediction, target, valid, source_ids, source_id: int):
    """Isolate one source's target-scale-restored direction tail gradient."""
    errors, stable, _residual, _geometry = _scale_restored_direction_torso_error_grid(
        torch, prediction, target, valid,
    )
    source_mask = (source_ids == source_id).unsqueeze(-1).expand_as(stable)
    restricted_stable = stable & source_mask
    flattened_errors, flattened_stable = errors.flatten(), restricted_stable.flatten()
    tail_count = ((flattened_stable.sum() + 19) // 20).clamp_min(1)
    maximum_tail = max(1, (flattened_errors.numel() + 19) // 20)
    selected = torch.topk(flattened_errors.masked_fill(~flattened_stable, 0.0), maximum_tail).values
    chosen = torch.arange(maximum_tail, device=prediction.device) < tail_count
    return selected.masked_select(chosen).mean()


def _component_losses(torch, prediction, target, valid, attribution=None) -> dict[str, float]:
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
    direction_torso_tail = _scale_restored_direction_torso_tail_loss(torch, prediction, target, valid)
    if attribution is None:
        attribution = _torso_attribution(torch, prediction, target, valid, torch.zeros(
            (prediction.shape[0],), dtype=torch.long, device=prediction.device,
        ))
    a12_selected = attribution["a12_selected"]
    return {
        "coordinate": float(coordinate.item()),
        "bone": float(bone.item()),
        "torso": float(torso.item()),
        "hinge": float(hinge.item()),
        "yaw_tail_pooled": float(yaw_tail.item()),
        "yaw_tail_frame_level": float(yaw_tail_frame_level.item()),
        "cartesian_torso_tail": float(cartesian_torso_tail.item()),
        "scale_restored_direction_torso_tail": float(direction_torso_tail.item()),
        "a12_selected_magnitude_loss": float(_selected_mean(torch, attribution["magnitude_loss"], a12_selected).item()),
        "a12_selected_direction_loss": float(_selected_mean(torch, attribution["direction_loss"], a12_selected).item()),
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


def _prediction_gradient_concentration(torch, prediction, loss, source_ids):
    """Report where a torso loss sends gradient in prediction space."""
    gradient = torch.autograd.grad(loss, prediction, retain_graph=True, allow_unused=True)[0]
    if gradient is None:
        return {"total_l1": 0.0, "torso_endpoint_l1_share": None, "by_pair_l1_share": {}, "by_source_l1_share": {}}
    joint_mass = gradient.abs().sum(dim=-1)
    total = float(joint_mass.sum().item())
    first, second = zip(*TORSO_INDICES)
    pair_mass = torch.stack(
        (joint_mass[:, first].sum(dim=-1), joint_mass[:, second].sum(dim=-1)),
        dim=-1,
    )
    torso_mass = float(pair_mass.sum().item())
    source_grid = source_ids.unsqueeze(-1).expand_as(joint_mass)
    by_source = {}
    for source_id, source_name in enumerate(SOURCE_NAMES):
        mass = float(joint_mass.masked_select(source_grid == source_id).sum().item())
        by_source[source_name] = {"l1": mass, "share": mass / total if total > 0 else 0.0}
    return {
        "total_l1": total,
        "torso_endpoint_l1_share": torso_mass / total if total > 0 else None,
        "by_pair_l1_share": {
            name: float(pair_mass[:, index].sum().item()) / total if total > 0 else 0.0
            for index, name in enumerate(("shoulder", "hip"))
        },
        "by_source_l1_share": by_source,
    }


def _gradient_interaction(torch, model, prediction, target, valid, source_ids) -> dict[str, Any]:
    a9_config = TrainingConfig(window=81, channels=256, epochs=1, batch_size=1, **A9_STRUCTURAL_WEIGHTS)
    mask = valid.unsqueeze(-1).float()
    base_loss = _supervision_loss(torch, prediction, target, mask, a9_config)
    yaw_loss_pooled = _yaw_tail_loss(torch, prediction, target, valid)
    yaw_loss_frame = _yaw_tail_loss_frame_level(torch, prediction, target, valid)
    candidate_loss = _cartesian_torso_tail_loss(torch, prediction, target, valid)
    direction_candidate_loss = _scale_restored_direction_torso_tail_loss(torch, prediction, target, valid)
    attribution = _torso_attribution(torch, prediction, target, valid, source_ids)
    magnitude_loss = _selected_mean(torch, attribution["magnitude_loss"], attribution["a12_selected"])
    direction_loss = _selected_mean(torch, attribution["direction_loss"], attribution["a12_selected"])

    g_base = _grad_of(torch, base_loss, model)
    g_yaw_pooled = _grad_of(torch, yaw_loss_pooled, model)
    g_yaw_frame = _grad_of(torch, yaw_loss_frame, model)
    g_candidate = _grad_of(torch, candidate_loss, model)
    g_direction_candidate = _grad_of(torch, direction_candidate_loss, model)
    g_magnitude = _grad_of(torch, magnitude_loss, model)
    g_direction = _grad_of(torch, direction_loss, model)

    def stats(g_yaw, loss=None):
        if g_base is None or g_yaw is None:
            return {"base_norm": None, "yaw_norm": None, "combined_norm": None, "cosine": None}
        cosine = torch.nn.functional.cosine_similarity(g_base.unsqueeze(0), g_yaw.unsqueeze(0)).item()
        result = {
            "base_norm": float(g_base.norm().item()),
            "yaw_norm": float(g_yaw.norm().item()),
            "combined_norm": float((g_base + g_yaw).norm().item()),
            "cosine": float(cosine),
            "component_to_base_ratio": float(g_yaw.norm().item() / g_base.norm().clamp_min(1e-12).item()),
        }
        if loss is not None:
            result["prediction_gradient_concentration"] = _prediction_gradient_concentration(
                torch, prediction, loss, source_ids,
            )
        return result

    per_source_yaw = {}
    per_source_candidate = {}
    per_source_direction = {}
    for source_id, name in enumerate(SOURCE_NAMES):
        if not (source_ids == source_id).any():
            continue
        composition_share = float((source_ids == source_id).float().mean().item())
        yaw_source_loss = _source_restricted_yaw_tail_loss(torch, prediction, target, valid, source_ids, source_id)
        g_yaw_source = _grad_of(torch, yaw_source_loss, model)
        # This source's share of *this batch* (expected ~= 1/3 under
        # source-balanced sampling) -- NOT its share of the real pooled tail
        # selection, which is reported separately in loss_share_by_source.
        per_source_yaw[name] = stats(g_yaw_source, yaw_source_loss) | {"batch_composition_share": composition_share}
        candidate_source_loss = _source_restricted_cartesian_torso_tail_loss(
            torch, prediction, target, valid, source_ids, source_id,
        )
        g_candidate_source = _grad_of(torch, candidate_source_loss, model)
        per_source_candidate[name] = stats(g_candidate_source, candidate_source_loss) | {"batch_composition_share": composition_share}
        direction_source_loss = _source_restricted_direction_torso_tail_loss(
            torch, prediction, target, valid, source_ids, source_id,
        )
        g_direction_source = _grad_of(torch, direction_source_loss, model)
        per_source_direction[name] = stats(g_direction_source, direction_source_loss) | {"batch_composition_share": composition_share}

    return {
        "pooled_selector": stats(g_yaw_pooled, yaw_loss_pooled),
        "frame_level_selector": stats(g_yaw_frame, yaw_loss_frame),
        "cartesian_torso_tail_candidate": stats(g_candidate, candidate_loss),
        "scale_restored_direction_candidate": stats(g_direction_candidate, direction_candidate_loss),
        "a12_magnitude_component": stats(g_magnitude, magnitude_loss),
        "a12_direction_component": stats(g_direction, direction_loss),
        "per_source_isolated": per_source_yaw,
        "per_source_isolated_candidate": per_source_candidate,
        "per_source_isolated_direction": per_source_direction,
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


def _mean(values):
    values = [value for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(values)) if values else None


def _percentile(values, percentile):
    values = [value for value in values if value is not None and np.isfinite(value)]
    return float(np.percentile(values, percentile)) if values else None


def _summarize_state(batch_reports):
    """Make the machine-readable report useful without re-running the replay."""
    component_names = (
        "coordinate", "bone", "torso", "hinge", "yaw_tail_pooled", "cartesian_torso_tail",
        "scale_restored_direction_torso_tail", "a12_selected_magnitude_loss", "a12_selected_direction_loss",
    )
    gradient_names = (
        "pooled_selector", "cartesian_torso_tail_candidate", "scale_restored_direction_candidate",
        "a12_magnitude_component", "a12_direction_component",
    )
    components = {
        name: _mean([batch["components_raw"].get(name) for batch in batch_reports])
        for name in component_names
    }
    gradients = {}
    for name in gradient_names:
        records = [batch["gradients"].get(name, {}) for batch in batch_reports]
        ratios = [record.get("component_to_base_ratio") for record in records]
        gradients[name] = {
            "mean_base_norm": _mean([record.get("base_norm") for record in records]),
            "mean_component_norm": _mean([record.get("yaw_norm") for record in records]),
            "mean_combined_norm": _mean([record.get("combined_norm") for record in records]),
            "mean_cosine": _mean([record.get("cosine") for record in records]),
            "mean_component_to_base_ratio": _mean(ratios),
            "p95_component_to_base_ratio": _percentile(ratios, 95),
            "maximum_component_to_base_ratio": max(ratios) if ratios else None,
        }

    def aggregate_source(kind):
        source_maps = [batch["torso_attribution"][kind]["source_shares"] for batch in batch_reports]
        total_selected = sum(
            record["selected_count"]
            for source_map in source_maps
            for record in source_map.values()
        )
        total_masses = {
            key: sum(
                record[key]
                for source_map in source_maps
                for record in source_map.values()
                if key in record
            )
            for key in source_maps[0][SOURCE_NAMES[0]]
            if key.endswith("_loss_mass")
        }
        output = {}
        for source_name in SOURCE_NAMES:
            records = [item[source_name] for item in source_maps]
            selected_count = sum(record["selected_count"] for record in records)
            output[source_name] = {"selected_count": selected_count}
            for key in records[0]:
                if key == "selected_count":
                    continue
                if key.endswith("_loss_mass"):
                    total_mass = sum(record[key] for record in records)
                    output[source_name][key] = total_mass
                    output[source_name][key.replace("_mass", "_share")] = (
                        total_mass / total_masses[key] if total_masses[key] > 0 else 0.0
                    )
            output[source_name]["selected_fraction"] = selected_count / max(total_selected, 1)
        return output

    def aggregate_association(name):
        records = [batch["yaw_association"].get(name, {}) for batch in batch_reports]
        return {
            "mean_pearson_r": _mean([record.get("pearson_r") for record in records]),
            "minimum_pearson_r": min(
                [record["pearson_r"] for record in records if record.get("pearson_r") is not None],
                default=None,
            ),
            "maximum_pearson_r": max(
                [record["pearson_r"] for record in records if record.get("pearson_r") is not None],
                default=None,
            ),
            "total_valid_frame_count": sum(record.get("valid_frame_count", 0) for record in records),
        }

    def mean_attribution(kind):
        summaries = [batch["torso_attribution"][kind]["summary"] for batch in batch_reports]
        return {key: _mean([summary.get(key) for summary in summaries]) for key in summaries[0]}

    return {
        "batch_count": len(batch_reports),
        "components_raw_mean": components,
        "a12_tail_mean": mean_attribution("a12_tail"),
        "direction_tail_mean": mean_attribution("direction_tail"),
        "source_wise_a12_tail": aggregate_source("a12_tail"),
        "source_wise_direction_tail": aggregate_source("direction_tail"),
        "gradients": gradients,
        "yaw_association": {
            name: aggregate_association(name)
            for name in ("a12_cartesian", "magnitude", "direction_scale_restored", "historical_angular")
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-dataset", required=True, type=Path, help="A9-A12's materialized direct_mix_train.json")
    parser.add_argument("--a9-checkpoint", required=True, type=Path)
    parser.add_argument("--a11-checkpoint", required=True, type=Path)
    parser.add_argument("--a12-checkpoint", required=True, type=Path)
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
        "a12_trained": (args.a12_checkpoint, None),
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
            attribution = _torso_attribution(
                torch, prediction, target_batch, valid_batch, batch_source_ids,
            )
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
                    "a12_selected_magnitude_at_0.05": components["a12_selected_magnitude_loss"] * 0.05,
                    "a12_selected_direction_at_0.05": components["a12_selected_direction_loss"] * 0.05,
                    "scale_restored_direction_torso_tail_at_0.05": components["scale_restored_direction_torso_tail"] * 0.05,
                },
                "pooled_selection": selection,
                "cartesian_torso_tail_selection": candidate_selection,
                "torso_attribution": {
                    "a12_tail": {
                        "summary": attribution["a12_summary"],
                        "source_shares": attribution["a12_source_shares"],
                    },
                    "direction_tail": {
                        "summary": attribution["direction_summary"],
                        "source_shares": attribution["direction_source_shares"],
                    },
                },
                "yaw_association": _yaw_association(
                    torch, prediction, target_batch, valid_batch,
                ),
                "gradients": gradients,
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
