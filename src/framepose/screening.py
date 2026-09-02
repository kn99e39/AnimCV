"""Pre-training loss screening.

A candidate loss contract does not automatically earn a full training run. This
stage replays fixed, real frame batches through one or more model states and
measures what the term would actually do to the optimizer.

No acceptance threshold is encoded. The measurements are evidence for a
researcher to interpret, exactly as docs/12-13 and docs/15 did for the Legacy
Temporal Pose Baseline. Synthetic contracts remain useful but cannot establish
architecture viability.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from framepose.backbones import resolve_backbone
from framepose.contract import FrameBank, JOINT_NAMES
from framepose.losses import BASELINE_GEOMETRY_V1, LossContract, compute_loss, loss_components
from framepose.model import ModelConfig, build_model


SCREENING_SCHEMA = "animcv_frame_pose_loss_screening_v1"


def fixed_batches(bank: FrameBank, *, split: str = "train", batch_count: int = 10,
                  batch_size: int = 256, seed: int = 1337) -> list[np.ndarray]:
    """Deterministic frame batches, identical for every screened contract."""
    positions = bank.indices(split)
    if not len(positions):
        raise ValueError(f"frame bank has no {split} split")
    generator = np.random.default_rng(seed)
    order = positions[generator.permutation(len(positions))]
    batches = []
    for index in range(batch_count):
        start = (index * batch_size) % max(len(order) - batch_size, 1)
        batches.append(order[start:start + batch_size])
    return batches


def screen_contracts(bank: FrameBank, contracts: Sequence[LossContract], *,
                     geometry: np.ndarray, features: np.ndarray | None = None,
                     backbone: str = "none", device: str = "cpu", seed: int = 1337,
                     batch_count: int = 10, batch_size: int = 256,
                     states: dict[str, Any] | None = None) -> dict[str, Any]:
    """Measure every contract on the same batches, in the same model states."""
    import torch

    spec = resolve_backbone(backbone)
    torch.manual_seed(seed)
    model_config = ModelConfig(
        visual_dim=spec.embed_dim if spec.kind != "none" else None,
        visual_tokens=spec.token_count if spec.kind != "none" else 0,
    )
    resolved = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
    model = build_model(model_config).to(resolved)
    batches = fixed_batches(bank, batch_count=batch_count, batch_size=batch_size, seed=seed)

    named_states = {"initialization": None}
    named_states.update(states or {})

    results: dict[str, Any] = {}
    for state_name, state_dict in named_states.items():
        if state_dict is not None:
            model.load_state_dict(state_dict)
        results[state_name] = _screen_state(
            torch, model, bank, contracts, batches, geometry, features, resolved)
    return {
        "schema": SCREENING_SCHEMA,
        "backbone": spec.to_dict(),
        "seed": seed,
        "batch_count": len(batches),
        "batch_size": batch_size,
        "batch_sample_ids": [[bank.samples[int(position)].sample_id for position in batch[:4]] for batch in batches],
        "base_contract": BASELINE_GEOMETRY_V1.to_dict(),
        "acceptance_thresholds": None,
        "acceptance_note": "measurements only; no automatic acceptance rule is encoded",
        "states": results,
    }


def _screen_state(torch, model, bank: FrameBank, contracts: Sequence[LossContract],
                  batches: list[np.ndarray], geometry: np.ndarray,
                  features: np.ndarray | None, device) -> dict[str, Any]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    accumulated: dict[str, list[dict[str, float]]] = {contract.name: [] for contract in contracts}
    component_accumulated: dict[str, list[dict[str, Any]]] = {}
    for batch in batches:
        geometry_batch = torch.as_tensor(geometry[batch], device=device)
        tokens = None if features is None else torch.as_tensor(
            np.asarray(features[batch], dtype=np.float32), device=device)
        target = torch.as_tensor(bank.arrays["target_3d"][batch], device=device)
        mask = torch.as_tensor(bank.arrays["target_valid"][batch].astype(np.float32)[..., None], device=device)

        prediction = model(geometry_batch, tokens)
        base_value = compute_loss(torch, prediction, target, mask, BASELINE_GEOMETRY_V1)
        # Reduced in float64: these gradients have ~1e6 terms, and a float32
        # dot product puts the base-vs-base cosine visibly outside [-1, 1].
        base_gradient = _flat_gradient(torch, base_value, parameters).double()
        base_norm = float(torch.linalg.vector_norm(base_gradient).item())

        for contract in contracts:
            prediction = model(geometry_batch, tokens)
            value = compute_loss(torch, prediction, target, mask, contract)
            gradient = _flat_gradient(torch, value, parameters).double()
            norm = float(torch.linalg.vector_norm(gradient).item())
            cosine = float(torch.nn.functional.cosine_similarity(
                gradient.unsqueeze(0), base_gradient.unsqueeze(0)).item())
            accumulated[contract.name].append({
                "raw_loss": float(value.item()),
                "gradient_norm": norm,
                "gradient_over_base_ratio": norm / max(base_norm, 1e-12),
                "gradient_cosine_with_base": cosine,
                "finite_loss": bool(torch.isfinite(value).item()),
                "finite_gradient": bool(torch.isfinite(gradient).all().item()),
            })

        prediction = model(geometry_batch, tokens)
        components = loss_components(torch, prediction, target, mask)
        for name, value in components.items():
            ownership = _joint_gradient_ownership(torch, value, prediction)
            component_accumulated.setdefault(name, []).append({
                "raw_loss": float(value.item()),
                "per_joint_gradient_ownership": ownership,
            })
        component_accumulated.setdefault("__frames__", []).append(
            _frame_association(torch, bank, batch, prediction, target, mask))
    return {
        "contracts": {name: _summarize(records) for name, records in accumulated.items()},
        "components": {name: _summarize_components(records)
                       for name, records in component_accumulated.items() if name != "__frames__"},
        "frame_association": _merge_association(component_accumulated.get("__frames__", [])),
    }


def _flat_gradient(torch, value, parameters):
    gradients = torch.autograd.grad(value, parameters, retain_graph=False, allow_unused=True)
    flat = [torch.zeros_like(parameter).reshape(-1) if gradient is None else gradient.reshape(-1)
            for parameter, gradient in zip(parameters, gradients)]
    return torch.cat(flat)


def _joint_gradient_ownership(torch, value, prediction) -> dict[str, float]:
    """Share of `d(loss)/d(prediction)` magnitude owned by each canonical joint."""
    gradient, = torch.autograd.grad(value, prediction, retain_graph=True, allow_unused=False)
    magnitude = torch.linalg.vector_norm(gradient, dim=-1).sum(dim=0)
    total = float(magnitude.sum().item())
    if total <= 0:
        return {name: 0.0 for name in JOINT_NAMES}
    return {name: float(magnitude[index].item() / total) for index, name in enumerate(JOINT_NAMES)}


def _frame_association(torch, bank: FrameBank, batch: np.ndarray, prediction, target, mask) -> dict[str, Any]:
    """Per-frame loss vs per-frame error, and per-source loss share."""
    with torch.no_grad():
        weight = mask.squeeze(-1)
        errors = torch.linalg.vector_norm(prediction.detach() - target, dim=-1)
        per_frame_error = (errors * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)
        per_frame_loss = (torch.nn.functional.smooth_l1_loss(
            prediction.detach(), target, reduction="none") * mask).sum(dim=(1, 2)) / \
            mask.sum(dim=(1, 2)).clamp_min(1.0)
    sources = [bank.samples[int(position)].source for position in batch]
    losses = per_frame_loss.cpu().numpy()
    contribution: dict[str, float] = {}
    for source, value in zip(sources, losses):
        contribution[source] = contribution.get(source, 0.0) + float(value)
    total = sum(contribution.values()) or 1.0
    return {
        "source_contribution": {key: value / total for key, value in contribution.items()},
        "loss_error_rank_correlation": _rank_correlation(losses, per_frame_error.cpu().numpy()),
        "hard_frame_loss_share": _hard_share(losses, per_frame_error.cpu().numpy()),
    }


def _rank_correlation(first: np.ndarray, second: np.ndarray) -> float | None:
    if first.size < 3:
        return None
    ranks = (np.argsort(np.argsort(first)).astype(np.float64),
             np.argsort(np.argsort(second)).astype(np.float64))
    if ranks[0].std() == 0 or ranks[1].std() == 0:
        return None
    return float(np.corrcoef(ranks[0], ranks[1])[0, 1])


def _hard_share(losses: np.ndarray, errors: np.ndarray) -> float | None:
    """Fraction of total loss carried by the worst-error decile of the batch."""
    if losses.size < 10:
        return None
    cutoff = np.quantile(errors, 0.9)
    hard = losses[errors >= cutoff]
    total = float(losses.sum())
    return float(hard.sum() / total) if total > 0 else None


def _summarize(records: list[dict[str, float]]) -> dict[str, Any]:
    keys = ("raw_loss", "gradient_norm", "gradient_over_base_ratio", "gradient_cosine_with_base")
    summary = {key: {
        "mean": float(np.mean([record[key] for record in records])),
        "min": float(np.min([record[key] for record in records])),
        "max": float(np.max([record[key] for record in records])),
    } for key in keys}
    summary["numerically_stable"] = all(record["finite_loss"] and record["finite_gradient"]
                                        for record in records)
    summary["batch_count"] = len(records)
    return summary


def _summarize_components(records: list[dict[str, Any]]) -> dict[str, Any]:
    ownership: dict[str, list[float]] = {}
    for record in records:
        for name, value in record["per_joint_gradient_ownership"].items():
            ownership.setdefault(name, []).append(value)
    return {
        "raw_loss_mean": float(np.mean([record["raw_loss"] for record in records])),
        "per_joint_gradient_ownership": {name: float(np.mean(values))
                                         for name, values in sorted(ownership.items())},
    }


def _merge_association(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {}
    sources: dict[str, list[float]] = {}
    for record in records:
        for name, value in record["source_contribution"].items():
            sources.setdefault(name, []).append(value)
    correlations = [record["loss_error_rank_correlation"] for record in records
                    if record["loss_error_rank_correlation"] is not None]
    shares = [record["hard_frame_loss_share"] for record in records
              if record["hard_frame_loss_share"] is not None]
    return {
        "source_contribution_mean": {name: float(np.mean(values)) for name, values in sorted(sources.items())},
        "loss_error_rank_correlation_mean": float(np.mean(correlations)) if correlations else None,
        "hard_frame_loss_share_mean": float(np.mean(shares)) if shares else None,
    }
