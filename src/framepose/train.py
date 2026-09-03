"""Frame-level training with strict candidate isolation.

One candidate = one (observation backend, loss contract) pair trained on the
whole shared frame batch stream. Across candidates the seed, the frame
identities, the sample order, the (absent) augmentation, the optimizer, the
fusion/head contract and the evaluator are all held fixed; only the declared
variable moves.

Model selection uses the validation split only. Test ground truth never
participates in choosing an epoch, a candidate or a threshold.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import numpy as np

from framepose.backbones import resolve_backbone
from framepose.contract import FrameBank, JOINT_COUNT
from framepose.crops import CROP_CONTRACT, crop_box, geometry_in_crop
from framepose.losses import LossContract, compute_loss, loss_components, resolve_contract
from framepose.model import ModelConfig, build_model, parameter_report
from framepose.observations import summarize as summarize_observations


CHECKPOINT_SCHEMA = "animcv_frame_pose_checkpoint_v1"
TRAINING_REPORT_SCHEMA = "animcv_frame_pose_training_v1"


@dataclass(frozen=True)
class CandidateConfig:
    """Everything that defines one controlled candidate run."""

    name: str
    backbone: str = "none"
    loss_contract: str = "baseline_geometry_v1"
    epochs: int = 120
    batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    minimum_learning_rate: float = 1e-5
    seed: int = 1337
    device: str = "cuda"
    mixed_precision: bool = True
    compile_training_graph: bool = False
    evaluate_every: int = 5
    # Reserved for Section 13's conditional parameter-efficient adaptation.
    # Frozen-first is the policy; this batch leaves it False.
    adapt_backbone: bool = False

    def __post_init__(self) -> None:
        if min(self.epochs, self.batch_size, self.evaluate_every) <= 0:
            raise ValueError("epochs, batch_size and evaluate_every must be positive")
        if self.learning_rate <= 0 or self.minimum_learning_rate < 0:
            raise ValueError("learning rate must be positive")
        if self.adapt_backbone:
            raise ValueError(
                "parameter-efficient backbone adaptation is gated on frozen-F2 evidence "
                "(Architecture_v3 section 8) and is not enabled in this batch")
        resolve_backbone(self.backbone)
        resolve_contract(self.loss_contract)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def geometry_tensor(bank: FrameBank) -> np.ndarray:
    """`(N, 17, 4)` crop-normalized geometry — identical for every candidate."""
    features = np.zeros((len(bank), JOINT_COUNT, 4), dtype=np.float32)
    for position, sample in enumerate(bank.samples):
        observation = bank.arrays["input_2d"][position]
        valid = bank.arrays["input_valid"][position]
        box = crop_box(observation, valid, sample.image_size)
        features[position] = geometry_in_crop(observation, valid, sample.image_size, box)
    return features


def train_candidate(bank: FrameBank, config: CandidateConfig, *,
                    features: np.ndarray | None = None,
                    geometry: np.ndarray | None = None,
                    checkpoint_path: str | Path | None = None) -> dict[str, Any]:
    """Train one candidate and return its full provenance-bearing report."""
    torch = _torch()
    spec = resolve_backbone(config.backbone)
    contract = resolve_contract(config.loss_contract)
    if spec.kind == "none" and features is not None:
        raise ValueError("the geometry-only candidate must not be given visual features")
    if spec.kind != "none" and features is None:
        raise ValueError(f"candidate {config.name} requires cached {config.backbone} features")

    geometry = geometry_tensor(bank) if geometry is None else geometry
    targets = bank.arrays["target_3d"]
    mask = bank.arrays["target_valid"].astype(np.float32)[..., None]

    train_positions = bank.indices("train")
    validation_positions = bank.indices("validation")
    if not len(train_positions):
        raise ValueError("frame bank has no train split")

    device = torch.device(config.device if (config.device != "cuda" or torch.cuda.is_available()) else "cpu")
    torch.manual_seed(config.seed)
    model_config = ModelConfig(
        visual_dim=spec.embed_dim if spec.kind != "none" else None,
        visual_tokens=spec.token_count if spec.kind != "none" else 0,
    )
    model = build_model(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)

    geometry_gpu = torch.as_tensor(geometry, device=device)
    target_gpu = torch.as_tensor(targets, device=device)
    mask_gpu = torch.as_tensor(mask, device=device)
    feature_source = None if features is None else np.ascontiguousarray(features)

    amp_enabled = bool(config.mixed_precision and device.type == "cuda")
    if hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    else:  # PyTorch 2.1 -- the training host's build -- keeps it under torch.cuda.amp.
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    def _forward_loss(geometry_batch, token_batch, target_batch, mask_batch):
        prediction = model(geometry_batch, token_batch)
        return prediction, compute_loss(torch, prediction, target_batch, mask_batch, contract)

    # docs/20 accepted torch.compile for the forward + loss graph; backward and
    # the optimizer step stay eager. Opt-in, and recorded in the report.
    execution_backend = "eager"
    forward_loss = _forward_loss
    if config.compile_training_graph:
        forward_loss = torch.compile(_forward_loss)
        execution_backend = "compiled"

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    steps_per_epoch = math.ceil(len(train_positions) / config.batch_size)
    total_steps = steps_per_epoch * config.epochs

    started = perf_counter()
    telemetry: list[dict[str, Any]] = []
    best = {"epoch": None, "validation_mpjpe_mm": None}
    best_state: dict[str, Any] | None = None
    step = 0
    frames_seen = 0
    for epoch in range(config.epochs):
        model.train()
        permutation = train_positions[torch.randperm(len(train_positions), generator=generator).numpy()]
        epoch_loss = 0.0
        for start in range(0, len(permutation), config.batch_size):
            batch = permutation[start:start + config.batch_size]
            index = torch.as_tensor(batch, device=device)
            tokens = _tokens(torch, feature_source, batch, device)
            for group in optimizer.param_groups:
                group["lr"] = _cosine_learning_rate(config, step, total_steps)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                _, loss = forward_loss(geometry_gpu[index], tokens, target_gpu[index], mask_gpu[index])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += float(loss.detach().item()) * len(batch)
            frames_seen += len(batch)
            step += 1
        record: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": epoch_loss / max(len(permutation), 1),
            "learning_rate": _cosine_learning_rate(config, step, total_steps),
        }
        final_epoch = epoch == config.epochs - 1
        if len(validation_positions) and (final_epoch or (epoch + 1) % config.evaluate_every == 0):
            record.update(_validation_snapshot(
                torch, model, geometry_gpu, target_gpu, mask_gpu, feature_source,
                validation_positions, device, amp_enabled, contract))
            if best["validation_mpjpe_mm"] is None or record["validation_mpjpe_mm"] < best["validation_mpjpe_mm"]:
                best = {"epoch": epoch, "validation_mpjpe_mm": record["validation_mpjpe_mm"]}
                best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        telemetry.append(record)
    elapsed = perf_counter() - started

    if best_state is not None:
        model.load_state_dict(best_state)
    report = {
        "schema": TRAINING_REPORT_SCHEMA,
        "candidate": config.to_dict(),
        "loss_contract": contract.to_dict(),
        "backbone": spec.to_dict(),
        "model": {**model_config.to_dict(), **parameter_report(model)},
        "crop_contract": CROP_CONTRACT,
        "bank": {"content_digest": bank.content_digest(),
                 "train_frames": int(len(train_positions)),
                 "validation_frames": int(len(validation_positions)),
                 "observation": summarize_observations([sample.observation for sample in bank.samples]),
                 "observation_regime": bank.regime()},
        "selection": {"criterion": "validation_mpjpe_mm", "split": "validation",
                      "test_ground_truth_used": False, **best},
        "augmentation": {"enabled": False,
                         "reason": "held constant across candidates; the manipulated variable is the observation backend"},
        "execution": {
            "device": str(device),
            "mixed_precision": amp_enabled,
            "execution_backend": execution_backend,
            "torch_version": torch.__version__,
        },
        "performance": {
            "training_seconds": elapsed,
            "frames_seen": frames_seen,
            "frames_per_second": frames_seen / max(elapsed, 1e-9),
            "steps": step,
            "steps_per_epoch": steps_per_epoch,
            "peak_gpu_memory_mb": (torch.cuda.max_memory_allocated(device) / (1024 ** 2))
                                  if device.type == "cuda" else None,
        },
        "epoch_telemetry": telemetry,
    }
    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "schema": CHECKPOINT_SCHEMA,
            "model_config": model_config.to_dict(),
            "candidate": config.to_dict(),
            "backbone": spec.to_dict(),
            "loss_contract": contract.to_dict(),
            "bank_content_digest": bank.content_digest(),
            "selection": report["selection"],
            "state_dict": model.state_dict(),
        }, path)
        report["checkpoint_path"] = str(path)
    return report


def predict(model, torch, geometry: np.ndarray, features: np.ndarray | None,
            positions: Sequence[int], device, *, batch_size: int = 512,
            amp_enabled: bool = False) -> np.ndarray:
    """Batched inference over an explicit list of bank positions."""
    positions = np.asarray(positions, dtype=np.int64)
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(positions), batch_size):
            batch = positions[start:start + batch_size]
            geometry_batch = torch.as_tensor(geometry[batch], device=device)
            tokens = _tokens(torch, features, batch, device)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                outputs.append(model(geometry_batch, tokens).float().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def load_checkpoint(path: str | Path, device: str = "cpu"):
    torch = _torch()
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError(f"unsupported frame pose checkpoint: {payload.get('schema')!r}")
    stored = dict(payload["model_config"])
    config = ModelConfig(visual_dim=stored["visual_dim"], visual_tokens=stored["visual_tokens"],
                         width=stored["width"], heads=stored["heads"],
                         fusion_depth=stored["fusion_depth"],
                         feedforward_multiplier=stored["feedforward_multiplier"])
    model = build_model(config).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def _tokens(torch, features: np.ndarray | None, batch: np.ndarray, device):
    if features is None:
        return None
    return torch.as_tensor(np.asarray(features[batch], dtype=np.float32), device=device)


def _cosine_learning_rate(config: CandidateConfig, step: int, total: int) -> float:
    if total <= 1:
        return config.learning_rate
    progress = min(step / max(total - 1, 1), 1.0)
    span = config.learning_rate - config.minimum_learning_rate
    return config.minimum_learning_rate + span * 0.5 * (1.0 + math.cos(math.pi * progress))


def _validation_snapshot(torch, model, geometry_gpu, target_gpu, mask_gpu, features,
                         positions, device, amp_enabled, contract: LossContract) -> dict[str, Any]:
    """Validation-split MPJPE and raw loss components; no test data involved."""
    was_training = model.training
    model.eval()
    total_error = 0.0
    total_count = 0.0
    components = {"coordinate": 0.0, "bone": 0.0, "torso": 0.0, "hinge": 0.0}
    batches = 0
    with torch.no_grad():
        for start in range(0, len(positions), 512):
            batch = positions[start:start + 512]
            index = torch.as_tensor(batch, device=device)
            tokens = _tokens(torch, features, batch, device)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                prediction = model(geometry_gpu[index], tokens)
            prediction = prediction.float()
            target = target_gpu[index]
            mask = mask_gpu[index]
            errors = torch.linalg.vector_norm(prediction - target, dim=-1)
            weight = mask.squeeze(-1)
            total_error += float((errors * weight).sum().item())
            total_count += float(weight.sum().item())
            for key, value in loss_components(torch, prediction, target, mask).items():
                components[key] += float(value.item())
            batches += 1
    if was_training:
        model.train()
    return {
        "validation_mpjpe_mm": total_error / max(total_count, 1.0) * 1000.0,
        "validation_loss_components": {key: value / max(batches, 1) for key, value in components.items()},
    }


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment guard
        raise ImportError("frame pose training requires torch; install the training extra") from exc
    return torch
