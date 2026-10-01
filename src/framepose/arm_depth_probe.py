"""DIAGNOSTIC: explicit 2.5D left-arm depth probe beside frozen H0 (Worklog 64).

Question: with the SAME single-frame evidence as H0 (crop-normalized 2D x/y,
confidence, validity, and the two O_BILATERAL oracle sign fields), can a model
that predicts the continuous forward component f = unit(v).Y of left-arm
directions recover depth better than H0's Cartesian XYZ output?

The probe reuses the historical FramePoseEstimator graph unchanged (same
ModelConfig, same parameter count); only the output interpretation and the
regression target differ.  The image-plane direction is NOT learned: it is
owned by the observed 2D segment (image x -> +X, image y -> -Z).  Nothing here
is used by production FramePose, AnimationSemantics, FK or IK.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from framepose.contract import JOINT_INDEX

SCHEMA = "animcv_framepose_arm_depth_probe_v1"
SEGMENTS: dict[str, tuple[str, str]] = {
    "upper": ("left_shoulder", "left_elbow"),
    "lower": ("left_elbow", "left_wrist"),
    "chain": ("left_shoulder", "left_wrist"),
}
SEGMENT_NAMES = tuple(SEGMENTS)
# Each forward fraction is read from channel 0 of the estimator's per-joint
# head at the segment's distal joint (chain: the shoulder token, which owns no
# other output here).  Fixed before training; never searched.
OUTPUT_TOKEN = {"upper": "left_elbow", "lower": "left_wrist", "chain": "left_shoulder"}
OBJECTIVE = "masked mean absolute error on tanh-bounded forward fractions (upper, lower, chain)"
SELECTION = "validation mean absolute forward-fraction error; test never used for selection"
_EPS = 1e-9


# ------------------------------------------------------------- targets


def segment_vectors(points: np.ndarray) -> dict[str, np.ndarray]:
    """`(N, 17, 3)` -> raw left-arm segment vectors `(N, 3)` per segment."""
    p = np.asarray(points, dtype=np.float64)
    return {name: p[:, JOINT_INDEX[b]] - p[:, JOINT_INDEX[a]] for name, (a, b) in SEGMENTS.items()}


def forward_fraction(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """unit(v).Y and a validity mask; scale-free and bounded in [-1, 1]."""
    v = np.asarray(vectors, dtype=np.float64)
    norm = np.linalg.norm(v, axis=-1)
    ok = np.isfinite(norm) & (norm > _EPS)
    f = np.where(ok, v[..., 1] / np.where(ok, norm, 1.0), 0.0)
    return np.clip(f, -1.0, 1.0), ok


def forward_targets(target_3d: np.ndarray, target_valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`(N, 3)` GT forward fractions (upper, lower, chain) and `(N, 3)` masks."""
    vectors = segment_vectors(target_3d)
    valid = np.asarray(target_valid, dtype=bool)
    values, masks = [], []
    for name, (a, b) in SEGMENTS.items():
        f, ok = forward_fraction(vectors[name])
        values.append(f)
        masks.append(ok & valid[:, JOINT_INDEX[a]] & valid[:, JOINT_INDEX[b]])
    return np.stack(values, axis=1), np.stack(masks, axis=1)


# ------------------------------------------------------ reconstruction


def observed_plane(input_2d: np.ndarray, image_sizes: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Observed image-plane unit direction per segment in canonical (X, Z).

    input_2d is the bank's `(N, 17, 3)` [x/width, y/height, conf]; pixels are
    restored with each frame's image size so the aspect ratio is exact.
    """
    x = np.asarray(input_2d, dtype=np.float64)
    size = np.asarray(image_sizes, dtype=np.float64)
    pixels = x[..., :2] * size[:, None, :]
    planes, oks = {}, {}
    for name, (a, b) in SEGMENTS.items():
        d = pixels[:, JOINT_INDEX[b]] - pixels[:, JOINT_INDEX[a]]
        xz = np.stack([d[:, 0], -d[:, 1]], axis=1)
        n = np.linalg.norm(xz, axis=1)
        oks[name] = n > _EPS
        planes[name] = xz / np.where(oks[name], n, 1.0)[:, None]
    return planes, oks


def reconstruct_direction(f: np.ndarray, plane_xz: np.ndarray) -> np.ndarray:
    """Unit 3D direction from a forward fraction and an image-plane direction.

    The plane direction is normalized here; only its orientation is used.
    |f| -> 1 gives the depth axis continuously (the plane weight vanishes).
    """
    f = np.clip(np.asarray(f, dtype=np.float64), -1.0, 1.0)
    p = np.asarray(plane_xz, dtype=np.float64)
    n = np.linalg.norm(p, axis=-1, keepdims=True)
    if np.any(n <= _EPS):
        raise ValueError("image-plane direction is degenerate")
    p = p / n
    r = np.sqrt(np.maximum(0.0, 1.0 - f * f))
    return np.stack([r * p[..., 0], f, r * p[..., 1]], axis=-1)


def angle_degrees(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise angle between 3D vectors, stable near 0 and 180 degrees."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    cross = np.linalg.norm(np.cross(a, b), axis=-1)
    return np.degrees(np.arctan2(cross, np.sum(a * b, axis=-1)))


def elevation_degrees(f: np.ndarray) -> np.ndarray:
    return np.degrees(np.arcsin(np.clip(np.asarray(f, dtype=np.float64), -1.0, 1.0)))


# ------------------------------------------------------------- model


@dataclass(frozen=True)
class ProbeConfig:
    """Training contract copied from the O_BILATERAL checkpoint's candidate."""

    epochs: int = 200
    batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    minimum_learning_rate: float = 1e-5
    seed: int = 1337
    mixed_precision: bool = True
    evaluate_every: int = 10
    device: str = "cuda"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _cosine(config: ProbeConfig, step: int, total: int) -> float:
    if total <= 1:
        return config.learning_rate
    progress = min(step / max(total - 1, 1), 1.0)
    span = config.learning_rate - config.minimum_learning_rate
    return config.minimum_learning_rate + span * 0.5 * (1.0 + math.cos(math.pi * progress))


def build_probe(model_config):
    """The unchanged FramePoseEstimator graph; returns (model, interpret)."""
    import torch
    from framepose.model import build_model

    model = build_model(model_config)
    tokens = [JOINT_INDEX[OUTPUT_TOKEN[name]] for name in SEGMENT_NAMES]

    def interpret(output):
        return torch.tanh(output[:, tokens, 0].float())
    return model, interpret


def train_probe(geometry: np.ndarray, signs: np.ndarray, targets: np.ndarray, masks: np.ndarray,
                train_positions: np.ndarray, validation_positions: np.ndarray,
                model_config, config: ProbeConfig) -> tuple[Any, Any, dict[str, Any]]:
    import torch

    device = torch.device(config.device if (config.device != "cuda" or torch.cuda.is_available()) else "cpu")
    torch.manual_seed(config.seed)
    model, interpret = build_probe(model_config)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    amp = bool(config.mixed_precision and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp) if hasattr(torch.amp, "GradScaler") \
        else torch.cuda.amp.GradScaler(enabled=amp)
    g = torch.as_tensor(geometry, device=device)
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
                    prediction = interpret(model(g[index], None, s[index]))
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
                loss = loss_of(interpret(model(g[index], None, s[index])), index)
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


def predict_probe(model, interpret, geometry: np.ndarray, signs: np.ndarray, positions: np.ndarray,
                  device: str = "cuda") -> np.ndarray:
    """Inference reads only the H0 geometry tensor and the identical sign array."""
    import torch

    dev = next(model.parameters()).device if device is None else torch.device(device)
    out = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(positions), 512):
            index = np.asarray(positions[start:start + 512])
            prediction = interpret(model(torch.as_tensor(geometry[index], device=dev), None,
                                         torch.as_tensor(np.asarray(signs[index], dtype=np.int64), device=dev)))
            out.append(prediction.cpu().numpy().astype(np.float64))
    return np.concatenate(out, axis=0)
