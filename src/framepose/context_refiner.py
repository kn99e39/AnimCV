"""A bounded, target-frame-only context-assisted Frame Pose refiner.

This module is deliberately separate from :mod:`framepose.model`.  The Frame
Pose Core and its frozen H0 lineage remain unchanged; this candidate consumes a
short same-sequence context and returns a residual for the centre frame only.

The runtime gate is part of the candidate contract, not a training label.  Its
thresholds are fitted from train observations and frozen before validation/test
evaluation.  No target 3D value is read by the window builder or the gate.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import numpy as np

from framepose.contract import FrameBank, JOINT_COUNT, JOINT_NAMES
from framepose.losses import compute_loss, resolve_contract


SCHEMA = "animcv_context_assisted_frame_refiner_v1"
WINDOW_OFFSETS = (-2, -1, 0, 1, 2)
RADIUS = 2
JOINT_FEATURES = 9  # H0 xyz/valid + 2D xy/confidence/valid/in-frame
SLOT_FEATURES = JOINT_COUNT * JOINT_FEATURES + 3  # present, frame offset, relative time
CONTEXT_DIM = len(WINDOW_OFFSETS) * SLOT_FEATURES
DEFAULT_HIDDEN = 128
EPSILON = 1e-12


def _sample_time(sample: Any) -> float:
    if sample.timestamp is not None and np.isfinite(sample.timestamp):
        return float(sample.timestamp)
    if sample.fps is not None and np.isfinite(sample.fps) and sample.fps > 0:
        return float(sample.frame_index) / float(sample.fps)
    return float(sample.frame_index)


def _validate_h0(bank: FrameBank, h0: np.ndarray) -> np.ndarray:
    values = np.asarray(h0, dtype=np.float32)
    expected = (len(bank), JOINT_COUNT, 3)
    if values.shape != expected:
        raise ValueError(f"H0 must have shape {expected}, got {values.shape}")
    return values


def _in_frame(bank: FrameBank) -> np.ndarray:
    observation = np.asarray(bank.arrays["input_2d"], dtype=np.float64)
    valid = np.asarray(bank.arrays["input_valid"], dtype=bool)
    xy = observation[..., :2]
    return (valid & np.isfinite(xy).all(axis=-1)
            & (xy[..., 0] >= 0.0) & (xy[..., 0] <= 1.0)
            & (xy[..., 1] >= 0.0) & (xy[..., 1] <= 1.0))


def sequence_window_positions(bank: FrameBank) -> np.ndarray:
    """Return same-sequence retained row positions for the fixed t-2...t+2 window.

    Missing slots are ``-1``.  Sorting is timestamp-aware and deterministic;
    no row from another sequence can enter a window.
    """
    grouped: dict[str, list[int]] = {}
    for position, sample in enumerate(bank.samples):
        grouped.setdefault(sample.sequence_id, []).append(position)
    windows = np.full((len(bank), len(WINDOW_OFFSETS)), -1, dtype=np.int64)
    for rows in grouped.values():
        ordered = sorted(rows, key=lambda row: (
            _sample_time(bank.samples[row]), bank.samples[row].frame_index,
            bank.samples[row].sample_id))
        for local, row in enumerate(ordered):
            for slot, offset in enumerate(WINDOW_OFFSETS):
                neighbour = local + offset
                if 0 <= neighbour < len(ordered):
                    windows[row, slot] = ordered[neighbour]
    return windows


def _nearest_support(signals: dict[str, np.ndarray], row: int, joint: int) -> tuple[int, int]:
    """Return nearest valid before/after rows already bounded by t+/-2."""
    before = signals["windows"][row, :2][::-1]
    after = signals["windows"][row, 3:]
    left = -1
    right = -1
    for candidate in before:
        if candidate >= 0 and signals["in_frame"][candidate, joint] and signals["h0_valid"][candidate, joint]:
            left = int(candidate)
            break
    for candidate in after:
        if candidate >= 0 and signals["in_frame"][candidate, joint] and signals["h0_valid"][candidate, joint]:
            right = int(candidate)
            break
    return left, right


def _interpolation_residual(values: np.ndarray, row: int, left: int, right: int,
                            timestamps: np.ndarray, joint: int, *, dimensions: int) -> float:
    if left < 0 or right < 0:
        return float("nan")
    span = timestamps[right] - timestamps[left]
    if not np.isfinite(span) or span <= 0.0:
        return float("nan")
    weight = (timestamps[row] - timestamps[left]) / span
    if not 0.0 < weight < 1.0:
        return float("nan")
    current = values[row, joint, :dimensions]
    before = values[left, joint, :dimensions]
    after = values[right, joint, :dimensions]
    if not (np.isfinite(current).all() and np.isfinite(before).all() and np.isfinite(after).all()):
        return float("nan")
    expected = (1.0 - weight) * before + weight * after
    return float(np.linalg.norm(current - expected))


def runtime_signals(bank: FrameBank, h0: np.ndarray) -> dict[str, np.ndarray]:
    """Compute target-free local signals used by the support gate."""
    h0 = _validate_h0(bank, h0)
    windows = sequence_window_positions(bank)
    timestamps = np.asarray([_sample_time(sample) for sample in bank.samples], dtype=np.float64)
    in_frame = _in_frame(bank)
    h0_valid = np.isfinite(h0).all(axis=-1)
    observation = bank.arrays["input_2d"].astype(np.float64)
    observation_residual = np.full((len(bank), JOINT_COUNT), np.nan, dtype=np.float64)
    h0_residual = np.full_like(observation_residual, np.nan)
    left = np.full((len(bank), JOINT_COUNT), -1, dtype=np.int64)
    right = np.full_like(left, -1)
    for row in range(len(bank)):
        local = {"windows": windows, "in_frame": in_frame, "h0_valid": h0_valid}
        for joint in range(JOINT_COUNT):
            before, after = _nearest_support(local, row, joint)
            left[row, joint], right[row, joint] = before, after
            if before < 0 or after < 0:
                continue
            observation_residual[row, joint] = _interpolation_residual(
                observation, row, before, after, timestamps, joint, dimensions=2)
            h0_residual[row, joint] = _interpolation_residual(
                h0, row, before, after, timestamps, joint, dimensions=3)
    current_observation_valid = in_frame
    current_h0_valid = h0_valid
    sequence_boundary = (windows[:, 0] < 0) | (windows[:, 3] < 0)
    return {
        "windows": windows,
        "timestamps": timestamps,
        "in_frame": in_frame,
        "h0_valid": h0_valid,
        "left": left,
        "right": right,
        "support": (left >= 0) & (right >= 0),
        "sequence_boundary": np.broadcast_to(sequence_boundary[:, None], left.shape).copy(),
        "observation_residual": observation_residual,
        "h0_residual": h0_residual,
        "current_observation_valid": current_observation_valid,
        "current_h0_valid": current_h0_valid,
    }


@dataclass(frozen=True)
class GateThresholds:
    """Train-only reference quantiles for the fixed runtime gate."""

    observation_p50: np.ndarray
    h0_p90: np.ndarray
    observation_count: np.ndarray
    h0_count: np.ndarray
    fit_split: str = "train"
    schema: str = "animcv_context_gate_thresholds_v1"

    def __post_init__(self) -> None:
        for name in ("observation_p50", "h0_p90", "observation_count", "h0_count"):
            value = np.asarray(getattr(self, name))
            if value.shape != (JOINT_COUNT,):
                raise ValueError(f"{name} must have shape ({JOINT_COUNT},), got {value.shape}")
        if self.fit_split != "train":
            raise ValueError("gate thresholds must be fitted on the train split")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "fit_split": self.fit_split,
            "observation_p50": self.observation_p50.tolist(),
            "h0_p90": self.h0_p90.tolist(),
            "observation_count": self.observation_count.astype(int).tolist(),
            "h0_count": self.h0_count.astype(int).tolist(),
            "joint_names": list(JOINT_NAMES),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GateThresholds":
        if payload.get("schema") != "animcv_context_gate_thresholds_v1":
            raise ValueError("unsupported context gate threshold schema")
        if payload.get("joint_names") != list(JOINT_NAMES):
            raise ValueError("gate threshold joint schema mismatch")
        return cls(
            observation_p50=np.asarray(payload["observation_p50"], dtype=np.float64),
            h0_p90=np.asarray(payload["h0_p90"], dtype=np.float64),
            observation_count=np.asarray(payload["observation_count"], dtype=np.int64),
            h0_count=np.asarray(payload["h0_count"], dtype=np.int64),
            fit_split=str(payload.get("fit_split", "train")),
        )


def fit_gate_thresholds(signals: dict[str, np.ndarray], train_positions: Sequence[int]) -> GateThresholds:
    """Fit exactly P50/P90 from runtime-observable train signals only."""
    positions = np.asarray(train_positions, dtype=np.int64)
    observation_p50 = np.full(JOINT_COUNT, np.nan, dtype=np.float64)
    h0_p90 = np.full(JOINT_COUNT, np.nan, dtype=np.float64)
    observation_count = np.zeros(JOINT_COUNT, dtype=np.int64)
    h0_count = np.zeros(JOINT_COUNT, dtype=np.int64)
    for joint in range(JOINT_COUNT):
        observed = signals["observation_residual"][positions, joint]
        h0 = signals["h0_residual"][positions, joint]
        observed = observed[np.isfinite(observed)]
        h0 = h0[np.isfinite(h0)]
        observation_count[joint] = len(observed)
        h0_count[joint] = len(h0)
        if len(observed):
            observation_p50[joint] = float(np.quantile(observed, 0.50))
        if len(h0):
            h0_p90[joint] = float(np.quantile(h0, 0.90))
    return GateThresholds(observation_p50, h0_p90, observation_count, h0_count)


@dataclass(frozen=True)
class GateDecision:
    gate_on: np.ndarray
    reason: np.ndarray
    observation_residual: np.ndarray
    h0_residual: np.ndarray
    left: np.ndarray
    right: np.ndarray


def apply_gate(signals: dict[str, np.ndarray], thresholds: GateThresholds) -> GateDecision:
    """Apply the fixed target-free gate and return auditable refusal reasons."""
    shape = signals["support"].shape
    gate_on = np.zeros(shape, dtype=bool)
    reason = np.full(shape, "threshold_unavailable", dtype=object)
    current_obs = signals["current_observation_valid"]
    current_h0 = signals["current_h0_valid"]
    reason[~current_obs] = "current_observation_invalid_or_out_of_frame"
    reason[current_obs & ~current_h0] = "current_h0_invalid"
    usable = current_obs & current_h0
    missing = usable & ~signals["support"]
    reason[missing & signals["sequence_boundary"]] = "sequence_boundary"
    reason[missing & ~signals["sequence_boundary"]] = "insufficient_valid_temporal_support"
    finite_signal = usable & signals["support"] & np.isfinite(signals["observation_residual"])
    reason[usable & signals["support"] & ~np.isfinite(signals["observation_residual"])] = "observation_signal_unavailable"
    unstable_2d = finite_signal & (
        signals["observation_residual"] > thresholds.observation_p50[None, :])
    reason[unstable_2d] = "local_2d_strongly_unstable"
    stable_h0 = finite_signal & np.isfinite(signals["h0_residual"])
    stable_h0 &= signals["h0_residual"] < thresholds.h0_p90[None, :]
    reason[stable_h0 & ~unstable_2d] = "h0_locally_stable"
    candidate = finite_signal & np.isfinite(signals["h0_residual"])
    candidate &= ~unstable_2d
    candidate &= signals["observation_residual"] <= thresholds.observation_p50[None, :]
    candidate &= signals["h0_residual"] >= thresholds.h0_p90[None, :]
    candidate &= np.isfinite(thresholds.observation_p50[None, :])
    candidate &= np.isfinite(thresholds.h0_p90[None, :])
    gate_on[candidate] = True
    reason[candidate] = "gate_on_stable_2d_unstable_h0_with_support"
    return GateDecision(gate_on, reason, signals["observation_residual"],
                        signals["h0_residual"], signals["left"], signals["right"])


def build_context_features(bank: FrameBank, h0: np.ndarray,
                           windows: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Flatten one fixed context window without reading target arrays."""
    h0 = _validate_h0(bank, h0)
    windows = sequence_window_positions(bank) if windows is None else np.asarray(windows, dtype=np.int64)
    if windows.shape != (len(bank), len(WINDOW_OFFSETS)):
        raise ValueError("context windows have an unexpected shape")
    in_frame = _in_frame(bank)
    observation = bank.arrays["input_2d"].astype(np.float64)
    valid = bank.arrays["input_valid"].astype(bool)
    timestamps = np.asarray([_sample_time(sample) for sample in bank.samples], dtype=np.float64)
    result = np.zeros((len(bank), CONTEXT_DIM), dtype=np.float32)
    for row in range(len(bank)):
        target_time = timestamps[row]
        cursor = 0
        for slot, offset in enumerate(WINDOW_OFFSETS):
            position = int(windows[row, slot])
            if position >= 0:
                result[row, cursor] = 1.0
                result[row, cursor + 1] = float(offset)
                result[row, cursor + 2] = float(timestamps[position] - target_time)
                feature_start = cursor + 3
                h0_row = h0[position]
                obs_row = observation[position]
                h0_ok = np.isfinite(h0_row).all(axis=-1)
                obs_xy = np.where(np.isfinite(obs_row[:, :2]), obs_row[:, :2], 0.0)
                confidence = np.where(np.isfinite(obs_row[:, 2]), obs_row[:, 2], 0.0)
                values = np.concatenate((
                    np.where(h0_ok[:, None], h0_row, 0.0), h0_ok[:, None].astype(np.float64),
                    obs_xy, confidence[:, None], valid[position, :, None].astype(np.float64),
                    in_frame[position, :, None].astype(np.float64)), axis=1)
                result[row, feature_start:feature_start + JOINT_COUNT * JOINT_FEATURES] = values.reshape(-1)
            cursor += 3 + JOINT_COUNT * JOINT_FEATURES
    return result, windows


def interpolate_gated_h0(h0: np.ndarray, signals: dict[str, np.ndarray], gate_on: np.ndarray) -> np.ndarray:
    """Parameter-free C1: interpolate only where the runtime gate is on."""
    result = np.asarray(h0, dtype=np.float32).copy()
    timestamps = signals["timestamps"]
    for row, joint in zip(*np.nonzero(gate_on)):
        left, right = int(signals["left"][row, joint]), int(signals["right"][row, joint])
        span = timestamps[right] - timestamps[left]
        weight = (timestamps[row] - timestamps[left]) / span
        result[row, joint] = ((1.0 - weight) * h0[left, joint] + weight * h0[right, joint]).astype(np.float32)
    return result


def _torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - environment guard
        raise ImportError("context refiner training requires torch") from exc
    return torch, nn


def build_refiner(context_dim: int = CONTEXT_DIM, hidden: int = DEFAULT_HIDDEN):
    """Build the one deliberately small temporal MLP used by this batch."""
    torch, nn = _torch()
    if context_dim != CONTEXT_DIM:
        raise ValueError("the candidate uses one fixed context dimension")

    class ContextAssistedFrameRefiner(nn.Module):
        schema = SCHEMA

        def __init__(self) -> None:
            super().__init__()
            self.context_dim = context_dim
            self.hidden = hidden
            self.encoder = nn.Sequential(
                nn.Linear(context_dim, hidden), nn.GELU(),
                nn.Linear(hidden, hidden), nn.GELU(),
                nn.Linear(hidden, JOINT_COUNT * 3),
            )

        def forward(self, context):
            if context.ndim == 3:
                context = context.reshape(context.shape[0], -1)
            if context.ndim != 2 or context.shape[-1] != self.context_dim:
                raise ValueError(f"context must be (B, {self.context_dim})")
            return self.encoder(context).reshape(-1, JOINT_COUNT, 3)

    return ContextAssistedFrameRefiner()


def parameter_report(model, *, context_dim: int = CONTEXT_DIM, hidden: int = DEFAULT_HIDDEN) -> dict[str, Any]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    multiply_adds = 2 * (context_dim * hidden + hidden * hidden + hidden * JOINT_COUNT * 3)
    return {
        "architecture": "small_temporal_mlp_v1",
        "context_dim": context_dim,
        "hidden": hidden,
        "parameter_count": int(total),
        "trainable_parameter_count": int(trainable),
        "estimated_inference_multiply_adds_per_frame": int(multiply_adds),
    }


@dataclass(frozen=True)
class RefinerConfig:
    name: str = "C2_support_gated_context_refiner"
    hidden: int = DEFAULT_HIDDEN
    loss_contract: str = "baseline_geometry_v1"
    epochs: int = 80
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 1337
    device: str = "cuda"
    mixed_precision: bool = True

    def __post_init__(self) -> None:
        if self.hidden != DEFAULT_HIDDEN:
            raise ValueError("hidden width is fixed for this architecture-validation batch")
        if min(self.epochs, self.batch_size) <= 0 or self.learning_rate <= 0:
            raise ValueError("epochs, batch_size and learning_rate must be positive")
        if self.loss_contract != "baseline_geometry_v1":
            raise ValueError("the candidate must use the existing baseline geometry contract")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _apply_residual(torch, h0, residual, gate):
    gate = gate.unsqueeze(-1).to(dtype=residual.dtype)
    clean_h0 = torch.nan_to_num(h0, nan=0.0, posinf=0.0, neginf=0.0)
    return clean_h0 + residual * gate


def _predict_torch(model, contexts, h0, gate, positions, torch, device, batch_size=512):
    outputs = []
    with torch.no_grad():
        for start in range(0, len(positions), batch_size):
            batch = np.asarray(positions[start:start + batch_size], dtype=np.int64)
            context = torch.as_tensor(contexts[batch], dtype=torch.float32, device=device)
            base = torch.as_tensor(h0[batch], dtype=torch.float32, device=device)
            mask = torch.as_tensor(gate[batch], dtype=torch.bool, device=device)
            outputs.append(_apply_residual(torch, base, model(context), mask).float().cpu().numpy())
    return np.concatenate(outputs, axis=0) if outputs else np.empty((0, JOINT_COUNT, 3), dtype=np.float32)


def train_refiner(bank: FrameBank, h0: np.ndarray, thresholds: GateThresholds,
                  config: RefinerConfig, *, contexts: np.ndarray | None = None,
                  gate: GateDecision | None = None,
                  checkpoint_path: str | Path | None = None) -> dict[str, Any]:
    """Train only a target-frame residual, selecting epochs on validation."""
    torch, _ = _torch()
    h0 = _validate_h0(bank, h0)
    if contexts is None:
        contexts, _ = build_context_features(bank, h0)
    if gate is None:
        gate = apply_gate(runtime_signals(bank, h0), thresholds)
    contract = resolve_contract(config.loss_contract)
    train_positions = bank.indices("train")
    validation_positions = bank.indices("validation")
    if not len(train_positions):
        raise ValueError("context refiner requires a train split")
    device = torch.device(config.device if (config.device != "cuda" or torch.cuda.is_available()) else "cpu")
    torch.manual_seed(config.seed)
    model = build_refiner(hidden=config.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    context_gpu = torch.as_tensor(contexts, dtype=torch.float32, device=device)
    h0_gpu = torch.as_tensor(np.nan_to_num(h0, nan=0.0, posinf=0.0, neginf=0.0), dtype=torch.float32, device=device)
    target_gpu = torch.as_tensor(bank.arrays["target_3d"], dtype=torch.float32, device=device)
    target_mask_gpu = torch.as_tensor(bank.arrays["target_valid"], dtype=torch.float32, device=device)[..., None]
    gate_gpu = torch.as_tensor(gate.gate_on, dtype=torch.bool, device=device)
    amp_enabled = bool(config.mixed_precision and device.type == "cuda")
    if hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    else:  # pragma: no cover - torch 2.1 compatibility on LabServer63
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    steps_per_epoch = math.ceil(len(train_positions) / config.batch_size)
    total_steps = steps_per_epoch * config.epochs
    best_state: dict[str, Any] | None = None
    best_validation = None
    telemetry = []
    started = perf_counter()
    step = 0
    for epoch in range(config.epochs):
        model.train()
        permutation = train_positions[torch.randperm(len(train_positions), generator=generator).numpy()]
        losses = []
        for start in range(0, len(permutation), config.batch_size):
            batch = permutation[start:start + config.batch_size]
            index = torch.as_tensor(batch, device=device)
            progress = min(step / max(total_steps - 1, 1), 1.0)
            lr = config.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                residual = model(context_gpu[index])
                refined = _apply_residual(torch, h0_gpu[index], residual, gate_gpu[index])
                loss = compute_loss(torch, refined, target_gpu[index], target_mask_gpu[index], contract)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().item()))
            step += 1
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)), "learning_rate": lr}
        if len(validation_positions):
            validation_prediction = _predict_torch(
                model, contexts, h0, gate.gate_on, validation_positions, torch, device)
            validation_target = bank.arrays["target_3d"][validation_positions]
            validation_valid = bank.arrays["target_valid"][validation_positions]
            error = np.linalg.norm(validation_prediction - validation_target, axis=-1)
            validation_mpjpe = float(error[validation_valid].mean() * 1000.0)
            record["validation_mpjpe_mm"] = validation_mpjpe
            if best_validation is None or validation_mpjpe < best_validation:
                best_validation = validation_mpjpe
                best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        telemetry.append(record)
    if best_state is not None:
        model.load_state_dict(best_state)
    elapsed = perf_counter() - started
    report = {
        "schema": "animcv_context_refiner_training_v1",
        "candidate": config.to_dict(),
        "architecture": parameter_report(model, hidden=config.hidden),
        "loss_contract": contract.to_dict(),
        "bank": {"content_digest": bank.content_digest(), "train_frames": int(len(train_positions)),
                 "validation_frames": int(len(validation_positions)), "test_ground_truth_used": False},
        "context": {"window_offsets": list(WINDOW_OFFSETS), "target_frame_only": True,
                     "context_dimension": CONTEXT_DIM, "sequence_boundary_masked": True},
        "gate": {"thresholds": thresholds.to_dict(), "definition": (
            "current observation in-frame AND both same-sequence valid H0/2D supports within t+/-2 "
            "AND local 2D residual <= train P50 AND local H0 residual >= train P90"),
                 "target_error_oracle_access": False,
                 "train_activation_rate": float(gate.gate_on[train_positions].mean()),
                 "validation_activation_rate": float(gate.gate_on[validation_positions].mean())
                 if len(validation_positions) else None},
        "selection": {"criterion": "validation_mpjpe_mm", "split": "validation",
                      "test_ground_truth_used": False, "best_validation_mpjpe_mm": best_validation},
        "execution": {"device": str(device), "mixed_precision": amp_enabled, "torch_version": torch.__version__},
        "performance": {"training_seconds": elapsed, "steps": step,
                         "frames_seen": int(step * config.batch_size),
                         "frames_per_second": float(step * config.batch_size / max(elapsed, 1e-9))},
        "epoch_telemetry": telemetry,
    }
    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"schema": "animcv_context_refiner_checkpoint_v1",
                    "model_config": {"context_dim": CONTEXT_DIM, "hidden": config.hidden},
                    "candidate": config.to_dict(), "bank_content_digest": bank.content_digest(),
                    "gate_thresholds": thresholds.to_dict(), "state_dict": model.state_dict()}, path)
        report["checkpoint_path"] = str(path)
    return report


def load_checkpoint(path: str | Path, device: str = "cpu"):
    torch, _ = _torch()
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("schema") != "animcv_context_refiner_checkpoint_v1":
        raise ValueError("unsupported context refiner checkpoint")
    model_config = payload.get("model_config", {})
    model = build_refiner(context_dim=int(model_config.get("context_dim", CONTEXT_DIM)),
                          hidden=int(model_config.get("hidden", DEFAULT_HIDDEN))).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def predict(model, contexts: np.ndarray, h0: np.ndarray, gate: np.ndarray,
            positions: Sequence[int], device: str = "cpu") -> np.ndarray:
    torch, _ = _torch()
    # Checkpoint replay is an explicit contract for this candidate.  The
    # context MLP has no stochastic layers at inference, but CUDA GEMM choices
    # (especially TF32) can otherwise differ by a few ulps after a reload.
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    return _predict_torch(model, contexts, h0, gate, positions, torch, torch.device(device))
