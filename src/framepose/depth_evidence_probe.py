"""DIAGNOSTIC (Worklog 65): D0_ZERO_DEPTH vs D1_DEPTH_ANYTHING on the Worklog 64 task.

One graph serves both candidates: the unchanged FramePoseEstimator with its
4-channel geometry projection replaced by a 5-channel one (4 geometry
channels + 1 depth-evidence channel).  Everything after the input projection
is the production estimator's own code path.  D0 and D1 differ ONLY in the
fifth channel's content (all zeros vs Depth Anything forward evidence).

Target, objective, output interpretation, optimizer, schedule, seed, batch
policy and selection are the Worklog 64 contract, imported unchanged from
``framepose.arm_depth_probe`` (which this module does not modify).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from framepose.arm_depth_probe import OBJECTIVE, OUTPUT_TOKEN, SEGMENT_NAMES, SELECTION, ProbeConfig, _cosine
from framepose.contract import JOINT_INDEX

DEPTH_CHANNELS = 1
CANDIDATES = ("D0_ZERO_DEPTH", "D1_DEPTH_ANYTHING")


def candidate_geometry(geometry: np.ndarray, forward_evidence: np.ndarray, candidate: str) -> np.ndarray:
    """`(N, 17, 5)`: identical first four channels; the fifth is the only variable."""
    geometry = np.asarray(geometry, dtype=np.float32)
    if geometry.shape[-1] != 4:
        raise ValueError("expected the 4-channel H0 geometry tensor")
    if candidate == "D0_ZERO_DEPTH":
        fifth = np.zeros(geometry.shape[:-1] + (1,), np.float32)
    elif candidate == "D1_DEPTH_ANYTHING":
        fifth = np.asarray(forward_evidence, dtype=np.float32)[..., None]
        if fifth.shape[:-1] != geometry.shape[:-1]:
            raise ValueError("depth evidence must align with geometry rows and joints")
    else:
        raise ValueError(f"unknown candidate {candidate!r}")
    return np.concatenate([geometry, fifth], axis=-1)


def build_depth_conditioned(model_config):
    """(model, interpret); model(geometry5, signs) -> per-joint head output."""
    import torch
    from torch import nn
    from framepose.model import build_model

    base = build_model(model_config)
    width = model_config.width

    class FiveChannelProjection(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(4 + DEPTH_CHANNELS, width)
            self.extra = None

        def forward(self, geometry4):
            if self.extra is None:
                raise RuntimeError("depth channel not supplied")
            return self.linear(torch.cat([geometry4, self.extra.to(geometry4.dtype)], dim=-1))

    class DepthConditionedEstimator(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.base = base
            self.base.geometry_projection = FiveChannelProjection()

        def forward(self, geometry5, sign_state):
            projection = self.base.geometry_projection
            projection.extra = geometry5[..., 4:]
            try:
                return self.base(geometry5[..., :4], None, sign_state)
            finally:
                projection.extra = None

    model = DepthConditionedEstimator()
    tokens = [JOINT_INDEX[OUTPUT_TOKEN[name]] for name in SEGMENT_NAMES]

    def interpret(output):
        return torch.tanh(output[:, tokens, 0].float())
    return model, interpret


def train_depth_candidate(geometry5: np.ndarray, signs: np.ndarray, targets: np.ndarray, masks: np.ndarray,
                          train_positions: np.ndarray, validation_positions: np.ndarray,
                          model_config, config: ProbeConfig) -> tuple[Any, Any, dict[str, Any]]:
    """Worklog 64 training loop, verbatim in policy, on the 5-channel graph."""
    import torch

    device = torch.device(config.device if (config.device != "cuda" or torch.cuda.is_available()) else "cpu")
    torch.manual_seed(config.seed)
    model, interpret = build_depth_conditioned(model_config)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    amp = bool(config.mixed_precision and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp) if hasattr(torch.amp, "GradScaler") \
        else torch.cuda.amp.GradScaler(enabled=amp)
    g = torch.as_tensor(np.asarray(geometry5, np.float32), device=device)
    s = torch.as_tensor(np.asarray(signs, dtype=np.int64), device=device)
    t = torch.as_tensor(np.asarray(targets, dtype=np.float32), device=device)
    m = torch.as_tensor(np.asarray(masks, dtype=np.float32), device=device)

    def loss_of(prediction, index):
        weight = m[index]
        return (torch.abs(prediction - t[index]) * weight).sum() / weight.sum().clamp_min(1.0)

    def validation_error():
        model.eval()
        total, count = 0.0, 0.0
        with torch.no_grad():
            for start in range(0, len(validation_positions), 512):
                index = torch.as_tensor(validation_positions[start:start + 512], device=device)
                with torch.amp.autocast("cuda", enabled=amp):
                    prediction = interpret(model(g[index], s[index]))
                weight = m[index]
                total += float((torch.abs(prediction - t[index]) * weight).sum())
                count += float(weight.sum())
        model.train()
        return total / max(count, 1.0)

    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    steps_per_epoch = math.ceil(len(train_positions) / config.batch_size)
    total_steps = steps_per_epoch * config.epochs
    best = {"epoch": None, "validation_mean_abs_forward_error": None}
    best_state, telemetry, step = None, [], 0
    for epoch in range(config.epochs):
        model.train()
        order = train_positions[torch.randperm(len(train_positions), generator=generator).numpy()]
        epoch_loss = 0.0
        for start in range(0, len(order), config.batch_size):
            batch = order[start:start + config.batch_size]
            index = torch.as_tensor(batch, device=device)
            for group in optimizer.param_groups:
                group["lr"] = _cosine(config, step, total_steps)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp):
                loss = loss_of(interpret(model(g[index], s[index])), index)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += float(loss.detach()) * len(batch)
            step += 1
        record = {"epoch": epoch, "train_loss": epoch_loss / len(order)}
        if epoch == config.epochs - 1 or (epoch + 1) % config.evaluate_every == 0:
            record["validation_mean_abs_forward_error"] = validation_error()
            if best["epoch"] is None or record["validation_mean_abs_forward_error"] < best["validation_mean_abs_forward_error"]:
                best = {"epoch": epoch, "validation_mean_abs_forward_error": record["validation_mean_abs_forward_error"]}
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        telemetry.append(record)
    model.load_state_dict(best_state)
    model.eval()
    return model, interpret, {"selection": {"criterion": SELECTION, **best}, "telemetry": telemetry,
                              "objective": OBJECTIVE, "config": config.to_dict(),
                              "device": str(device), "mixed_precision": amp}


def predict_depth_candidate(model, interpret, geometry5: np.ndarray, signs: np.ndarray,
                            positions: np.ndarray, device: str | None = None) -> np.ndarray:
    """Inference reads only the 5-channel input and the identical sign array."""
    import torch

    dev = next(model.parameters()).device if device is None else torch.device(device)
    out = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(positions), 512):
            index = np.asarray(positions[start:start + 512])
            prediction = interpret(model(torch.as_tensor(np.asarray(geometry5[index], np.float32), device=dev),
                                         torch.as_tensor(np.asarray(signs[index], dtype=np.int64), device=dev)))
            out.append(prediction.cpu().numpy().astype(np.float64))
    return np.concatenate(out, axis=0)
