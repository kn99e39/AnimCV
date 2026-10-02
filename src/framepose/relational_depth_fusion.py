"""DIAGNOSTIC (Worklog 66): late relational fusion of frozen H0 with Worklog 65 Depth Anything evidence.

Inputs per frame (all runtime observable):
    h0_upper_f, h0_lower_f, h0_chain_f   -- f = unit(v).Y of frozen H0 left-arm segments
    delta_upper, delta_lower, delta_chain -- Worklog 65 forward evidence differences
                                             (elbow-shoulder, wrist-elbow, wrist-shoulder)
Output: corrected (upper, lower, chain) forward fractions.

One fixed graph, 6 -> 32 -> GELU -> 32 -> GELU -> 3 -> tanh.  R0 feeds zeros
for the three relations, R1 the actual relations; nothing else differs.  The
Worklog 64/65 target, masked-MAE objective, optimizer, schedule, seed and
selection rule are reused.  GT never enters the inputs and there is no regime
gate or trust threshold.  Unavailable relations (an endpoint without depth
evidence) are 0, the same representation Worklog 65 used for unavailable depth.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from framepose.arm_depth_probe import (
    OBJECTIVE, SEGMENT_NAMES, SEGMENTS, SELECTION, ProbeConfig, _cosine, forward_fraction, segment_vectors,
)
from framepose.contract import JOINT_INDEX

CANDIDATES = ("R0_ZERO_RELATION", "R1_DEPTH_RELATION")
HIDDEN = 32
INPUTS = 6


def depth_relations(forward_evidence: np.ndarray, available: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`(N, 3)` distal-minus-proximal Worklog 65 evidence per segment and availability."""
    ev = np.asarray(forward_evidence, dtype=np.float64)
    ok = np.asarray(available, dtype=bool)
    deltas, masks = [], []
    for name in SEGMENT_NAMES:
        a, b = (JOINT_INDEX[j] for j in SEGMENTS[name])
        both = ok[:, a] & ok[:, b]
        deltas.append(np.where(both, ev[:, b] - ev[:, a], 0.0))
        masks.append(both)
    return np.stack(deltas, axis=1), np.stack(masks, axis=1)


def h0_forward(h0_points: np.ndarray) -> np.ndarray:
    """`(N, 3)` forward fractions of frozen H0 segments (0 for degenerate segments)."""
    vectors = segment_vectors(h0_points)
    return np.stack([forward_fraction(vectors[name])[0] for name in SEGMENT_NAMES], axis=1)


def fusion_inputs(h0_f: np.ndarray, relations: np.ndarray, candidate: str) -> np.ndarray:
    h0_f = np.asarray(h0_f, dtype=np.float32)
    if candidate == "R0_ZERO_RELATION":
        rel = np.zeros_like(h0_f)
    elif candidate == "R1_DEPTH_RELATION":
        rel = np.asarray(relations, dtype=np.float32)
        if rel.shape != h0_f.shape:
            raise ValueError("relations must align with H0 forward fractions")
    else:
        raise ValueError(f"unknown candidate {candidate!r}")
    return np.concatenate([h0_f, rel], axis=1)


def build_fusion():
    import torch
    from torch import nn

    return nn.Sequential(nn.Linear(INPUTS, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, HIDDEN), nn.GELU(),
                         nn.Linear(HIDDEN, 3), nn.Tanh())


def train_fusion(inputs: np.ndarray, targets: np.ndarray, masks: np.ndarray, train_positions: np.ndarray,
                 validation_positions: np.ndarray, config: ProbeConfig) -> tuple[Any, dict[str, Any]]:
    import torch

    device = torch.device(config.device if (config.device != "cuda" or torch.cuda.is_available()) else "cpu")
    torch.manual_seed(config.seed)
    model = build_fusion().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    amp = bool(config.mixed_precision and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp) if hasattr(torch.amp, "GradScaler") \
        else torch.cuda.amp.GradScaler(enabled=amp)
    x = torch.as_tensor(np.asarray(inputs, np.float32), device=device)
    t = torch.as_tensor(np.asarray(targets, np.float32), device=device)
    m = torch.as_tensor(np.asarray(masks, np.float32), device=device)

    def masked(prediction, index):
        weight = m[index]
        return (torch.abs(prediction.float() - t[index]) * weight).sum(), weight.sum()

    def validation_error():
        model.eval()
        with torch.no_grad():
            index = torch.as_tensor(validation_positions, device=device)
            with torch.amp.autocast("cuda", enabled=amp):
                total, count = masked(model(x[index]), index)
        model.train()
        return float(total) / max(float(count), 1.0)

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
            index = torch.as_tensor(order[start:start + config.batch_size], device=device)
            for group in optimizer.param_groups:
                group["lr"] = _cosine(config, step, total_steps)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp):
                total, count = masked(model(x[index]), index)
                loss = total / count.clamp_min(1.0)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += float(loss.detach()) * len(index)
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
    return model, {"selection": {"criterion": SELECTION, **best}, "telemetry": telemetry, "objective": OBJECTIVE,
                   "config": config.to_dict(), "device": str(device), "mixed_precision": amp}


def predict_fusion(model, inputs: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Inference reads only the six runtime inputs."""
    import torch

    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        out = model(torch.as_tensor(np.asarray(inputs[np.asarray(positions)], np.float32), device=device))
    return out.float().cpu().numpy().astype(np.float64)
