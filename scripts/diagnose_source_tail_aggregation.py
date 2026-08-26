#!/usr/bin/env python3
"""Diagnose source reweighting caused by A12's global Cartesian tail.

This is diagnostic-only instrumentation.  It keeps the A12 Cartesian
torso-vector residual, coefficient, and pooled top-5% selector unchanged,
then compares the existing global aggregation with a generic per-source tail
mean on the same fixed batches.  It also reports source-specific error
distributions and 3DPW split coverage without training a model.

The source-stratified path is intentionally not wired into ``TrainingConfig``
or the production training loop.  It is a counterfactual used only to decide
whether one controlled follow-up experiment is justified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from training.temporal_lifter import (
    BONE_INDICES,
    TORSO_INDICES,
    TrainingConfig,
    _arrays,
    _augment_inputs,
    _cartesian_torso_tail_loss,
    _frame_metadata,
    _hinge_loss,
    _model,
    _predict_batched,
    _root_yaw_error_degrees,
    _source_balanced_permutation,
    _supervision_loss,
    _torch,
    _torso_vector_error_grid,
    _vector_loss,
    load_dataset,
)


A9_STRUCTURAL_WEIGHTS = {
    "bone_loss_weight": 0.25,
    "torso_loss_weight": 0.15,
    "hinge_loss_weight": 0.15,
}
TAIL_WEIGHT = 0.05  # A12's recorded coefficient; not a tuning parameter here.
REPLAY_BATCH_SIZE = 128
SOURCE_BALANCED_REPLAY_SEED = 1337
PAIR_NAMES = ("shoulder", "hip")


def _source_label_order(dataset: dict[str, Any]) -> list[str]:
    """Return generic first-seen source/group labels, matching ``_arrays``."""
    labels: list[str] = []
    default_source = dataset.get("source", {})
    for sequence in dataset.get("sequences", [{"frames": dataset["frames"]}]):
        source = {**default_source, **sequence.get("source", {})}
        label = str(source.get("dataset", "unknown"))
        if label not in labels:
            labels.append(label)
    return labels


def _stats(values: list[float] | np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"count": 0, "mean": None, "median": None, "p90": None, "p95": None, "p99": None}
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "median": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
    }


def _pearson(x: list[float] | np.ndarray, y: list[float] | np.ndarray) -> float | None:
    x_array, y_array = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    finite = np.isfinite(x_array) & np.isfinite(y_array)
    if finite.sum() < 2 or np.std(x_array[finite]) <= 1e-12 or np.std(y_array[finite]) <= 1e-12:
        return None
    return float(np.corrcoef(x_array[finite], y_array[finite])[0, 1])


def _top_tail_mask(torch, errors, stable):
    """Select ceil(valid/20) observations from one candidate pool.

    For the complete grid this has the same selected entries as A12's
    ``_pooled_tail_mean``.  For a per-source pool, the grid is compressed to
    that source first, so the same 5% fraction is applied within the source.
    """
    flat_errors, flat_stable = errors.flatten(), stable.flatten()
    valid_indices = torch.nonzero(flat_stable, as_tuple=False).flatten()
    valid_count = int(valid_indices.numel())
    selected = torch.zeros_like(flat_stable)
    if not valid_count:
        return selected.reshape_as(stable), 0
    tail_count = max(1, (valid_count + 19) // 20)
    values = flat_errors[valid_indices]
    order = torch.topk(values, min(tail_count, values.numel())).indices
    selected[valid_indices[order]] = True
    return selected.reshape_as(stable), tail_count


def _global_tail_mask(torch, errors, stable):
    """Replicate A12's exact pooled selector, including invalid zero fill."""
    flat_errors, flat_stable = errors.flatten(), stable.flatten()
    tail_count = max(1, (int(flat_stable.sum().item()) + 19) // 20)
    maximum_tail = max(1, (flat_errors.numel() + 19) // 20)
    _values, indices = torch.topk(flat_errors.masked_fill(~flat_stable, 0.0), maximum_tail)
    selected = torch.zeros_like(flat_stable)
    selected[indices[:tail_count]] = True
    return selected.reshape_as(stable), tail_count


def _source_tail_accounting(
    torch, errors, stable, source_ids, source_labels: list[str], weight: float = TAIL_WEIGHT,
) -> tuple[dict[str, Any], Any]:
    """Account for the unchanged A12 global tail by generic source ID."""
    selected, tail_count = _global_tail_mask(torch, errors, stable)
    source_grid = source_ids.unsqueeze(-1).expand_as(stable)
    total_raw = float(errors.masked_select(selected).sum().item())
    total_selected = int(selected.sum().item())
    source_report: dict[str, Any] = {}
    for source_id, label in enumerate(source_labels):
        source_mask = source_grid == source_id
        candidate_count = int((stable & source_mask).sum().item())
        selected_count = int((selected & source_mask).sum().item())
        raw_mass = float(errors.masked_select(selected & source_mask).sum().item())
        source_report[label] = {
            "candidate_count_before_selection": candidate_count,
            "selected_tail_count": selected_count,
            "selected_fraction_within_source": selected_count / candidate_count if candidate_count else 0.0,
            "fraction_of_total_selected": selected_count / total_selected if total_selected else 0.0,
            "raw_auxiliary_loss_mass": raw_mass,
            "weighted_auxiliary_loss_mass": raw_mass * weight,
            "raw_auxiliary_loss_share": raw_mass / total_raw if total_raw > 0 else 0.0,
            "weighted_auxiliary_loss_share": raw_mass / total_raw if total_raw > 0 else 0.0,
        }
    return {
        "candidate_count_before_selection": int(stable.sum().item()),
        "selected_tail_count": total_selected,
        "tail_fraction": total_selected / int(stable.sum().item()) if stable.any() else 0.0,
        "raw_auxiliary_loss": total_raw / total_selected if total_selected else 0.0,
        "weighted_auxiliary_loss": (total_raw / total_selected * weight) if total_selected else 0.0,
        "source": source_report,
    }, selected


def _source_stratified_tail(
    torch, errors, stable, source_ids, source_labels: list[str], weight: float = TAIL_WEIGHT,
) -> dict[str, Any]:
    """Diagnostic-only equal mean of each active source's own A12-style tail."""
    source_grid = source_ids.unsqueeze(-1).expand_as(stable)
    selected_union = torch.zeros_like(stable)
    active: list[tuple[int, str, Any, Any, int, int]] = []
    for source_id, label in enumerate(source_labels):
        source_stable = stable & (source_grid == source_id)
        source_errors = errors.masked_select(source_stable)
        candidate_count = int(source_errors.numel())
        if not candidate_count:
            continue
        # Compress to this source so ceil(N_source/20), not ceil(N_total/20),
        # is selected.  No source gets a manual weight.
        local_selected_values, local_tail_count = _top_tail_mask(
            torch, source_errors, torch.ones_like(source_errors, dtype=torch.bool),
        )
        local_selected_values = local_selected_values.reshape_as(source_errors)
        local_loss = source_errors.masked_select(local_selected_values).mean()
        source_positions = torch.nonzero(source_stable, as_tuple=False)
        selected_positions = source_positions[local_selected_values]
        if selected_positions.numel():
            selected_union[selected_positions[:, 0], selected_positions[:, 1]] = True
        active.append((source_id, label, local_loss, local_selected_values, candidate_count, local_tail_count))

    active_count = len(active)
    aggregate_loss = torch.stack([item[2] for item in active]).mean() if active else errors.sum() * 0.0
    source_report: dict[str, Any] = {}
    for _source_id, label, local_loss, _selected_values, candidate_count, local_tail_count in active:
        raw_loss = float(local_loss.item())
        source_report[label] = {
            "candidate_count_before_selection": candidate_count,
            "selected_tail_count": local_tail_count,
            "selected_fraction_within_source": local_tail_count / candidate_count,
            "fraction_of_total_selected": local_tail_count / int(selected_union.sum().item()) if selected_union.any() else 0.0,
            "raw_auxiliary_loss": raw_loss,
            "weighted_auxiliary_loss": raw_loss * weight / active_count,
            "aggregate_loss_share": 1.0 / active_count,
        }
    return {
        "active_source_count": active_count,
        "candidate_count_before_selection": int(stable.sum().item()),
        "selected_tail_count": int(selected_union.sum().item()),
        "raw_auxiliary_loss": float(aggregate_loss.item()),
        "weighted_auxiliary_loss": float((aggregate_loss * weight).item()),
        "source": source_report,
        "aggregate_loss": aggregate_loss,
        "selected_mask": selected_union,
        "local_losses": {label: loss for _id, label, loss, _values, _count, _tail in active},
    }


def _flat_grad(torch, model):
    pieces = [parameter.grad.detach().reshape(-1) for parameter in model.parameters() if parameter.grad is not None]
    return torch.cat(pieces) if pieces else None


def _grad_of(torch, model, loss):
    model.zero_grad(set_to_none=True)
    loss.backward(retain_graph=True)
    gradient = _flat_grad(torch, model)
    model.zero_grad(set_to_none=True)
    return gradient


def _cosine(torch, first, second) -> float | None:
    if first is None or second is None:
        return None
    denominator = first.norm() * second.norm()
    if float(denominator.item()) <= 1e-12:
        return None
    return float((first @ second / denominator).item())


def _gradient_metrics(torch, base_gradient, auxiliary_gradient, base_loss=None, auxiliary_loss=None):
    if base_gradient is None or auxiliary_gradient is None:
        return {"base_norm": None, "auxiliary_norm": None, "combined_norm": None, "cosine_with_base": None}
    return {
        "base_norm": float(base_gradient.norm().item()),
        "auxiliary_norm": float(auxiliary_gradient.norm().item()),
        "combined_norm": float((base_gradient + auxiliary_gradient).norm().item()),
        "cosine_with_base": _cosine(torch, base_gradient, auxiliary_gradient),
    }


def _gradient_comparison(torch, model, prediction, target, valid, source_ids, source_labels):
    mask = valid.unsqueeze(-1).float()
    base_config = TrainingConfig(window=81, channels=256, epochs=1, batch_size=1, **A9_STRUCTURAL_WEIGHTS)
    base_loss = _supervision_loss(torch, prediction, target, mask, base_config)
    errors, stable = _torso_vector_error_grid(torch, prediction, target, valid)
    global_detail, global_selected = _source_tail_accounting(
        torch, errors, stable, source_ids, source_labels,
    )
    stratified_detail = _source_stratified_tail(
        torch, errors, stable, source_ids, source_labels,
    )
    global_loss = _cartesian_torso_tail_loss(torch, prediction, target, valid)
    stratified_loss = stratified_detail["aggregate_loss"]
    base_gradient = _grad_of(torch, model, base_loss)
    global_gradient = _grad_of(torch, model, global_loss)
    stratified_gradient = _grad_of(torch, model, stratified_loss)

    source_grid = source_ids.unsqueeze(-1).expand_as(stable)
    global_source_gradients: dict[str, Any] = {}
    local_source_gradients: dict[str, Any] = {}
    stratified_source_gradients: dict[str, Any] = {}
    global_selected_count = max(int(global_selected.sum().item()), 1)
    active_count = max(int(stratified_detail["active_source_count"]), 1)
    for source_id, label in enumerate(source_labels):
        source_mask = source_grid == source_id
        global_source_loss = (errors * (global_selected & source_mask)).sum() / global_selected_count
        local_loss = stratified_detail["local_losses"].get(label)
        if local_loss is None:
            local_loss = errors.sum() * 0.0
        global_source_gradient = _grad_of(torch, model, global_source_loss)
        local_source_gradient = _grad_of(torch, model, local_loss)
        stratified_source_gradient = _grad_of(torch, model, local_loss / active_count)
        global_source_gradients[label] = global_source_gradient
        local_source_gradients[label] = local_source_gradient
        stratified_source_gradients[label] = stratified_source_gradient

    def source_metrics(gradients):
        result = {}
        for label, gradient in gradients.items():
            result[label] = {
                "gradient_norm": float(gradient.norm().item()) if gradient is not None else None,
                "cosine_with_base": _cosine(torch, base_gradient, gradient),
                "cosine_with_global_auxiliary": _cosine(torch, global_gradient, gradient),
            }
        return result

    pairwise = {}
    for first_index, first_label in enumerate(source_labels):
        for second_label in source_labels[first_index + 1:]:
            pairwise[f"{first_label}__{second_label}"] = _cosine(
                torch, local_source_gradients[first_label], local_source_gradients[second_label],
            )
    return {
        "global_tail": _gradient_metrics(torch, base_gradient, global_gradient),
        "source_stratified": _gradient_metrics(torch, base_gradient, stratified_gradient),
        "global_selected_source_contribution": source_metrics(global_source_gradients),
        "source_local_tail": source_metrics(local_source_gradients),
        "source_stratified_contribution": source_metrics(stratified_source_gradients),
        "source_local_pairwise_cosine": pairwise,
        "global_tail_detail": global_detail,
        "source_stratified_detail": {
            key: value for key, value in stratified_detail.items()
            if key not in ("aggregate_loss", "selected_mask", "local_losses")
        },
    }


def _source_balance(torch, source_ids, source_labels, seed: int, batch_size: int, batch_count: int, generator=None):
    indices = torch.arange(len(source_ids), device=source_ids.device)
    if generator is None:
        generator = torch.Generator(device=source_ids.device).manual_seed(seed)
    permutation = _source_balanced_permutation(torch, indices, source_ids, generator)

    def counts(values):
        return {
            label: int((values == source_id).sum().item())
            for source_id, label in enumerate(source_labels)
        }

    batches = list(permutation.split(batch_size))[:batch_count]
    representative = torch.cat(batches) if batches else permutation[:0]
    return {
        "raw_input_frame_mass": counts(source_ids),
        "sampled_epoch_mass": counts(source_ids[permutation]),
        "representative_batch_mass": counts(source_ids[representative]),
        "epoch_sample_count": int(permutation.numel()),
        "representative_sample_count": int(representative.numel()),
        "batch_size": batch_size,
        "batch_count": len(batches),
        "permutation": permutation,
        "batches": batches,
    }


def _load_model(torch, nn, checkpoint_path: Path | None, seed: int, device: str):
    torch.manual_seed(seed)
    model = _model(nn, 256, "dilated_tcn_v1").to(device)
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["state_dict"])
    model.train()
    return model


def _fixed_batch_state(
    torch, nn, checkpoint_path, seed, device, epoch_inputs, offset_tensor, targets, valid_tensor,
    source_tensor, source_labels, batches,
):
    model = _load_model(torch, nn, checkpoint_path, seed, device)
    reports = []
    for batch in batches:
        prediction = model(epoch_inputs[offset_tensor[batch]])
        target_batch = targets[batch]
        valid_batch = valid_tensor[batch]
        source_batch = source_tensor[batch]
        errors, stable = _torso_vector_error_grid(torch, prediction, target_batch, valid_batch)
        global_detail, _selected = _source_tail_accounting(
            torch, errors, stable, source_batch, source_labels,
        )
        stratified_detail = _source_stratified_tail(
            torch, errors, stable, source_batch, source_labels,
        )
        gradients = _gradient_comparison(
            torch, model, prediction, target_batch, valid_batch, source_batch, source_labels,
        )
        reports.append({
            "batch_source_composition": {
                label: int((source_batch == source_id).sum().item())
                for source_id, label in enumerate(source_labels)
            },
            "global_tail": global_detail,
            "source_stratified": {
                key: value for key, value in stratified_detail.items()
                if key not in ("aggregate_loss", "selected_mask", "local_losses")
            },
            "gradients": gradients,
        })
    return reports


def _numpy_yaw(estimate, reference, frame_valid):
    angles = []
    for left, right in ((11, 14), (4, 1)):
        if not (frame_valid[left] and frame_valid[right]):
            continue
        predicted_axis = estimate[right, :2] - estimate[left, :2]
        target_axis = reference[right, :2] - reference[left, :2]
        predicted_length = np.linalg.norm(predicted_axis)
        target_length = np.linalg.norm(target_axis)
        if min(predicted_length, target_length) <= 1e-6:
            continue
        delta = (np.arctan2(predicted_axis[1], predicted_axis[0]) -
                 np.arctan2(target_axis[1], target_axis[0]) + np.pi) % (2 * np.pi) - np.pi
        angles.append(abs(delta) * 180.0 / np.pi)
    return float(np.mean(angles)) if angles else None


def _numpy_torso_residuals(prediction, targets, valid):
    values = np.full((len(prediction), 2), np.nan, dtype=np.float64)
    for pair_index, (left, right) in enumerate(((11, 14), (1, 4))):
        pair_valid = valid[:, left] & valid[:, right]
        residual = (prediction[:, right] - prediction[:, left]) - (targets[:, right] - targets[:, left])
        absolute = np.abs(residual)
        smooth_l1 = np.where(absolute < 1.0, 0.5 * residual * residual, absolute - 0.5).mean(axis=-1)
        values[pair_valid, pair_index] = smooth_l1[pair_valid]
    return values


def _numpy_torso_orientation(targets, valid):
    orientations = np.full(len(targets), np.nan, dtype=np.float64)
    for index, (frame, frame_valid) in enumerate(zip(targets, valid)):
        axes = []
        for left, right in ((11, 14), (1, 4)):
            if not (frame_valid[left] and frame_valid[right]):
                continue
            axis = frame[right, :2] - frame[left, :2]
            length = np.linalg.norm(axis)
            if length > 1e-6:
                axes.append(axis / length)
        if axes:
            mean_axis = np.mean(axes, axis=0)
            orientations[index] = np.degrees(np.arctan2(mean_axis[1], mean_axis[0]))
    return orientations


def _turn_deltas(orientations, metadata):
    deltas = []
    previous_action = None
    previous_orientation = None
    for orientation, meta in zip(orientations, metadata):
        action = meta.get("action") or "unknown"
        if action != previous_action:
            previous_action, previous_orientation = action, None
        if previous_orientation is not None and np.isfinite(orientation):
            delta = (np.radians(orientation) - previous_orientation + np.pi) % (2 * np.pi) - np.pi
            deltas.append(abs(np.degrees(delta)))
        previous_orientation = np.radians(orientation) if np.isfinite(orientation) else None
    return deltas


def _distribution_report(prediction, targets, valid, inputs, metadata, source_ids, source_labels):
    yaws = [_numpy_yaw(estimate, reference, frame_valid) for estimate, reference, frame_valid in zip(prediction, targets, valid)]
    torso_residuals = _numpy_torso_residuals(prediction, targets, valid)
    orientations = _numpy_torso_orientation(targets, valid)
    source_report: dict[str, Any] = {}
    source_grid = np.repeat(source_ids[:, None], 2, axis=1)
    for source_id, label in enumerate(source_labels):
        frame_mask = source_ids == source_id
        pair_mask = source_grid == source_id
        pair_values = torso_residuals[pair_mask]
        pair_values = pair_values[np.isfinite(pair_values)]
        source_errors = np.asarray([value for value, keep in zip(yaws, frame_mask) if keep and value is not None])
        source_pair_grid = np.where(pair_mask & np.isfinite(torso_residuals), torso_residuals, -np.inf)
        valid_pair_values = source_pair_grid[np.isfinite(source_pair_grid)]
        local_tail_count = max(1, (len(valid_pair_values) + 19) // 20) if len(valid_pair_values) else 0
        tail_values = np.sort(valid_pair_values)[-local_tail_count:] if local_tail_count else np.asarray([])
        source_report[label] = {
            "frame_count": int(frame_mask.sum()),
            "root_yaw_error_degrees": _stats(source_errors),
            "cartesian_torso_residual": _stats(pair_values),
            "cartesian_torso_local_tail_residual": _stats(tail_values),
            "gt_torso_orientation_degrees": _stats(orientations[frame_mask]),
            "input_confidence_mean": _stats(inputs[frame_mask, :, 2].mean(axis=1)),
            "input_confidence_positive_joint_fraction": _stats((inputs[frame_mask, :, 2] > 0).mean(axis=1)),
        }
    return source_report


def _coverage_report(predictions, targets, valid, inputs, metadata):
    orientations = _numpy_torso_orientation(targets, valid)
    actions = [meta.get("action") or "unknown" for meta in metadata]
    views = [meta.get("view") for meta in metadata]
    action_counts: dict[str, int] = {}
    for action in actions:
        action_counts[action] = action_counts.get(action, 0) + 1
    top_actions = sorted(action_counts.items(), key=lambda item: (-item[1], item[0]))[:10]
    result = {
        "frame_count": len(metadata),
        "sequence_or_action_count": len(action_counts),
        "top_sequence_or_action_frame_counts": dict(top_actions),
        "view_labels": sorted({view for view in views if view is not None}),
        "view_metadata_available": any(view is not None for view in views),
        "gt_torso_orientation_degrees": _stats(orientations),
        "gt_torso_turn_delta_degrees": _stats(_turn_deltas(orientations, metadata)),
        "input_confidence_mean": _stats(inputs[:, :, 2].mean(axis=1)),
        "input_confidence_positive_joint_fraction": _stats((inputs[:, :, 2] > 0).mean(axis=1)),
    }
    for name, prediction in predictions.items():
        yaws = [_numpy_yaw(estimate, reference, frame_valid) for estimate, reference, frame_valid in zip(prediction, targets, valid)]
        result[f"{name}_root_yaw_error_degrees"] = _stats([value for value in yaws if value is not None])
    return result


def _predict_dataset(torch, nn, checkpoint_path, dataset, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    inputs, targets, valid, offsets = _arrays(
        dataset, int(checkpoint["window"]), coordinate_normalization=checkpoint.get("input_coordinate_normalization", "image_v1"),
    )
    model = _model(nn, int(checkpoint["channels"]), checkpoint.get("architecture", "legacy_tcn_v1")).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    x = torch.as_tensor(inputs, dtype=torch.float32, device=device)
    with torch.no_grad():
        prediction = _predict_batched(
            model, x, torch.as_tensor(offsets, dtype=torch.long, device=device), 1024, device.startswith("cuda"),
        ).cpu().numpy()
    return prediction, targets, valid, inputs, _frame_metadata(dataset)


def _fingerprint(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest(), "byte_size": path.stat().st_size}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dataset", required=True, type=Path)
    parser.add_argument("--three-dpw-train", required=True, type=Path)
    parser.add_argument("--three-dpw-validation", required=True, type=Path)
    parser.add_argument("--three-dpw-holdout", required=True, type=Path)
    parser.add_argument("--a9-checkpoint", required=True, type=Path)
    parser.add_argument("--a12-checkpoint", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-count", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=REPLAY_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=SOURCE_BALANCED_REPLAY_SEED)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    torch, nn = _torch()
    train_dataset = load_dataset(args.train_dataset)
    source_labels = _source_label_order(train_dataset)
    inputs, targets, valid, offsets, source_ids, sequence_ranges = _arrays(
        train_dataset, window=81, include_metadata=True, coordinate_normalization="pelvis_torso_v1",
    )
    device = args.device
    x = torch.as_tensor(inputs, dtype=torch.float32, device=device)
    y = torch.as_tensor(targets, dtype=torch.float32, device=device)
    valid_tensor = torch.as_tensor(valid, dtype=torch.bool, device=device)
    offset_tensor = torch.as_tensor(offsets, dtype=torch.long, device=device)
    source_tensor = torch.as_tensor(source_ids, dtype=torch.long, device=device)
    replay_config = TrainingConfig(
        window=81, channels=256, epochs=1, batch_size=args.batch_size, seed=args.seed,
        source_balanced_sampling=True, input_jitter_std=0.015,
        input_dropout_probability=0.05, confidence_jitter_std=0.08,
        input_global_scale_std=0.04, input_translation_std=0.03,
        input_rotation_degrees=12.0, temporal_occlusion_probability=0.10,
        temporal_occlusion_frames=9, input_coordinate_normalization="pelvis_torso_v1",
        **A9_STRUCTURAL_WEIGHTS,
    )
    generator = torch.Generator(device=x.device).manual_seed(args.seed)
    epoch_inputs = _augment_inputs(torch, x, replay_config, generator, sequence_ranges)
    # ``train()`` consumes the seeded RNG in this exact order: augmentation
    # first, then source-balanced permutation.  Reusing one generator keeps
    # these fixed batches identical to A9/A12's prior diagnostic replay.
    balance = _source_balance(
        torch, source_tensor, source_labels, args.seed, args.batch_size, args.batch_count,
        generator=generator,
    )
    batches = balance["batches"]

    model_states = {
        "init": None,
        "a9_trained": args.a9_checkpoint,
        "a12_trained": args.a12_checkpoint,
    }
    fixed_batch_report = {}
    for state_name, checkpoint_path in model_states.items():
        fixed_batch_report[state_name] = _fixed_batch_state(
            torch, nn, checkpoint_path, args.seed, device, epoch_inputs, offset_tensor,
            y, valid_tensor, source_tensor, source_labels, batches,
        )

    # The 3DPW train split is already present inside direct_mix_train.  Reuse
    # that prediction slice instead of loading a second copy of the large
    # training JSON; validation/test are loaded one at a time below.
    coverage_datasets = {
        "3dpw_validation": args.three_dpw_validation,
        "3dpw_test": args.three_dpw_holdout,
    }
    direct_predictions: dict[str, Any] = {}
    source_error_distributions: dict[str, Any] = {}
    coverage: dict[str, Any] = {}
    for state_name, checkpoint_path in (("a9_trained", args.a9_checkpoint), ("a12_trained", args.a12_checkpoint)):
        prediction, direct_targets, direct_valid, direct_inputs, direct_metadata = _predict_dataset(
            torch, nn, checkpoint_path, train_dataset, device,
        )
        direct_predictions[state_name] = prediction
        source_error_distributions[state_name] = _distribution_report(
            prediction, direct_targets, direct_valid, direct_inputs, direct_metadata,
            source_ids, source_labels,
        )
        train_source_id = source_labels.index("3DPW") if "3DPW" in source_labels else None
        if train_source_id is not None:
            train_mask = source_ids == train_source_id
            coverage.setdefault("3dpw_train", {"dataset": _fingerprint(args.three_dpw_train), "states": {}})
            coverage["3dpw_train"]["states"][state_name] = _coverage_report(
                {state_name: prediction[train_mask]}, direct_targets[train_mask], direct_valid[train_mask],
                direct_inputs[train_mask], [meta for meta, keep in zip(direct_metadata, train_mask) if keep],
            )
        for split_name, split_path in coverage_datasets.items():
            split_dataset = load_dataset(split_path)
            split_prediction, split_targets, split_valid, split_inputs, split_metadata = _predict_dataset(
                torch, nn, checkpoint_path, split_dataset, device,
            )
            coverage.setdefault(split_name, {"dataset": _fingerprint(split_path), "states": {}})
            coverage[split_name]["states"][state_name] = _coverage_report(
                {state_name: split_prediction}, split_targets, split_valid, split_inputs, split_metadata,
            )

    report = {
        "schema": "animcv_a12_source_tail_aggregation_diagnosis_v1",
        "controlled_loss": {
            "name": "A12 Cartesian torso-tail",
            "coefficient": TAIL_WEIGHT,
            "tail_fraction": 0.05,
            "direction_only_loss_used": False,
            "angular_yaw_tail_used": False,
            "source_stratified_path": "diagnostic_only",
        },
        "datasets": {
            "train": _fingerprint(args.train_dataset),
            "three_dpw_train": _fingerprint(args.three_dpw_train),
            "three_dpw_validation": _fingerprint(args.three_dpw_validation),
            "three_dpw_holdout": _fingerprint(args.three_dpw_holdout),
            "source_labels": source_labels,
        },
        "input_source_balance": {
            key: value for key, value in balance.items()
            if key not in ("permutation", "batches")
        },
        "source_error_distributions": source_error_distributions,
        "coverage": coverage,
        "fixed_batch": {
            "batch_count": len(batches),
            "batch_size": args.batch_size,
            "seed": args.seed,
            "states": fixed_batch_report,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "batch_count": len(batches), "source_labels": source_labels}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
