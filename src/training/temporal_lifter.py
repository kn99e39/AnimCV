"""Small, from-scratch temporal 2D-to-3D lifter and data contract.

This module contains no pretrained weights and accepts only data the project
owner is entitled to use.  Its deliberately compact TCN is a reproducible
baseline; it is not represented as a production-quality estimator until it
passes the calibrated holdout gates.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from common.serialization import read_json, write_json
from pose.pose_lifter import H36M_NAMES, LiftedPoseSequence
from pose.pose_lifter import LiftedPoseFrame, LiftedPosePoint
from pose.pose_types import PoseSequence


SCHEMA = "animcv_supervised_3d_lifter_dataset_v2"
LEGACY_SCHEMA = "animcv_supervised_3d_lifter_dataset_v1"


def build_dataset(
    pose: PoseSequence, target: LiftedPoseSequence, image_size: tuple[int, int], sequence_id: str,
) -> dict[str, Any]:
    """Pair licensed 2D observations with root-relative 3D ground truth."""
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    targets = {frame.frame_index: frame for frame in target.frames}
    frames = []
    for frame in pose.frames:
        target_frame = targets.get(frame.frame_index)
        if target_frame is None:
            continue
        inputs, outputs = [], []
        valid = []
        for name in H36M_NAMES:
            landmark_name = "neck" if name == "thorax" else name
            landmark = frame.landmarks.get(landmark_name)
            point = target_frame.points.get(name) or target_frame.points.get(landmark_name)
            input_valid = landmark is not None and landmark.visible
            target_valid = point is not None and point.observation_valid
            if landmark is None:
                inputs.append([0.0, 0.0, 0.0])
            else:
                inputs.append([landmark.x / width, landmark.y / height, landmark.confidence])
            outputs.append(list(point.position) if point is not None else [0.0, 0.0, 0.0])
            valid.append(bool(input_valid and target_valid))
        # A missing pelvis makes root-relative supervision meaningless.
        if not valid[0]:
            continue
        frames.append({"frame_index": frame.frame_index, "input_2d": inputs, "target_3d": outputs,
                       "target_valid": valid})
    if not frames:
        raise ValueError("no aligned 2D/3D frames for supervised dataset")
    return {
        "schema": SCHEMA, "joint_names": list(H36M_NAMES), "sequence_id": sequence_id,
        "source_fps": pose.source_fps, "image_size": [width, height], "frames": frames,
        "sequences": [{"sequence_id": sequence_id, "source_fps": pose.source_fps,
                       "image_size": [width, height], "frames": frames}],
    }


def save_dataset(dataset: dict[str, Any], path: str | Path) -> None:
    write_json(path, dataset)


def load_dataset(path: str | Path) -> dict[str, Any]:
    dataset = read_json(path)
    if dataset.get("schema") not in (SCHEMA, LEGACY_SCHEMA) or dataset.get("joint_names") != list(H36M_NAMES):
        raise ValueError("unsupported supervised 3D lifter dataset")
    if not dataset.get("frames"):
        raise ValueError("supervised dataset contains no frames")
    return dataset


def combine_datasets(datasets: list[dict[str, Any]], expected_split: str | None = None) -> dict[str, Any]:
    """Combine complete clips without allowing temporal windows to cross clips."""
    if not datasets:
        raise ValueError("at least one dataset is required")
    sequences = []
    identifiers = set()
    for dataset in datasets:
        if dataset.get("joint_names") != list(H36M_NAMES):
            raise ValueError("dataset joint schema mismatch")
        source = dataset.get("source", {})
        if expected_split is not None and source.get("split") != expected_split:
            raise ValueError(f"dataset split {source.get('split')!r} does not match expected {expected_split!r}")
        for sequence in dataset.get("sequences", [{"sequence_id": dataset.get("sequence_id"), "frames": dataset["frames"]}]):
            sequence_id = sequence.get("sequence_id")
            if not sequence_id or sequence_id in identifiers:
                raise ValueError("combined datasets require unique sequence_id values")
            identifiers.add(sequence_id)
            if not sequence.get("frames"):
                continue
            sequences.append(sequence)
    if not sequences:
        raise ValueError("combined datasets contain no frames")
    frames = [frame for sequence in sequences for frame in sequence["frames"]]
    return {"schema": SCHEMA, "joint_names": list(H36M_NAMES), "sequence_id": "combined",
            "source_fps": 0.0, "image_size": None, "frames": frames, "sequences": sequences}


@dataclass(frozen=True)
class TrainingConfig:
    window: int = 81
    channels: int = 256
    epochs: int = 30
    batch_size: int = 128
    learning_rate: float = 1e-3
    device: str = "cpu"
    mixed_precision: bool = True
    distributed: bool = False
    seed: int = 1337
    inference_batch_size: int = 1024

    def __post_init__(self) -> None:
        if self.window < 3 or self.window % 2 == 0:
            raise ValueError("window must be an odd value of at least 3")
        if min(self.channels, self.epochs, self.batch_size, self.inference_batch_size) <= 0 or self.learning_rate <= 0:
            raise ValueError("training dimensions, epochs, batch size, and learning rate must be positive")


def train(dataset: dict[str, Any], checkpoint_path: str | Path, config: TrainingConfig) -> dict[str, Any]:
    torch, nn = _torch()
    inputs, targets, valid, offsets = _arrays(dataset, config.window)
    context = _distributed_context(torch, config)
    device = context["device"]
    model = _model(nn, config.channels).to(device)
    if context["enabled"]:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[context["local_rank"]])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    # Pose supervision is compact; GPU-resident tensors avoid worker/PCIe overhead.
    x = torch.as_tensor(inputs, dtype=torch.float32, device=device)
    y = torch.as_tensor(targets, dtype=torch.float32, device=device)
    valid_tensor = torch.as_tensor(valid, dtype=torch.float32, device=device).unsqueeze(-1)
    offset_tensor = torch.as_tensor(offsets, dtype=torch.long, device=device)
    indices = torch.arange(len(inputs), device=device)
    amp_enabled = bool(config.mixed_precision and device.type == "cuda")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    started = perf_counter()
    local_samples_seen = 0
    for epoch in range(config.epochs):
        generator = torch.Generator(device=device).manual_seed(config.seed + epoch)
        permutation = indices[torch.randperm(len(indices), generator=generator, device=device)]
        permutation = _rank_shard(torch, permutation, context["rank"], context["world_size"])
        local_samples_seen += len(permutation)
        for batch in permutation.split(config.batch_size):
            # Advanced indexing builds every temporal window in one GPU operation.
            windows = x[offset_tensor[batch]]
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                prediction = model(windows)
                mask = valid_tensor[batch]
                loss = (torch.nn.functional.smooth_l1_loss(prediction, y[batch], reduction="none") * mask).sum() / mask.sum().clamp_min(1.0)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    model.eval()
    with torch.no_grad():
        # Evaluation has no gradient collectives, so ranks can use unequal shards
        # and together cover each sample exactly once.
        metric_indices = indices[context["rank"]::context["world_size"]]
        prediction = _predict_batched(model, x, offset_tensor[metric_indices], config.inference_batch_size, amp_enabled)
        errors = torch.linalg.vector_norm(prediction - y[metric_indices], dim=-1)
        metric_sum = (errors * valid_tensor[metric_indices].squeeze(-1)).sum()
        metric_count = valid_tensor[metric_indices].sum()
        if context["enabled"]:
            torch.distributed.all_reduce(metric_sum)
            torch.distributed.all_reduce(metric_count)
        mpjpe_mm = float((metric_sum / metric_count.clamp_min(1.0)).item() * 1000)
    if context["primary"]:
        payload = {"schema": "animcv_temporal_lifter_checkpoint_v2", "joint_names": list(H36M_NAMES),
                   "channels": config.channels, "window": config.window,
                   "state_dict": (model.module if context["enabled"] else model).state_dict()}
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, checkpoint_path)
    if context["enabled"]:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    elapsed_seconds = perf_counter() - started
    global_samples_seen = local_samples_seen * context["world_size"]
    performance = {
        "training_seconds": elapsed_seconds,
        "global_samples_seen": global_samples_seen,
        "global_samples_per_second": global_samples_seen / max(elapsed_seconds, 1e-9),
        "peak_gpu_memory_mb": (torch.cuda.max_memory_allocated(device) / (1024 ** 2)) if device.type == "cuda" else None,
    }
    return {"frame_count": len(inputs), "training_mpjpe_mm": mpjpe_mm,
            "valid_joint_count": int(valid.sum()), "config": config.__dict__,
            "parallelism": {"mode": "ddp" if config.distributed else "single_gpu", "world_size": context["world_size"],
                            "device": str(device), "mixed_precision": amp_enabled}, "performance": performance,
            "is_primary": context["primary"]}


def evaluate(dataset: dict[str, Any], checkpoint_path: str | Path, device: str = "cpu") -> dict[str, Any]:
    torch, nn = _torch()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint.get("schema") not in ("animcv_temporal_lifter_checkpoint_v1", "animcv_temporal_lifter_checkpoint_v2"):
        raise ValueError("unsupported temporal lifter checkpoint")
    inputs, targets, valid, offsets = _arrays(dataset, int(checkpoint["window"]))
    model = _model(nn, int(checkpoint["channels"])).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    x = torch.as_tensor(inputs, dtype=torch.float32, device=device)
    y = torch.as_tensor(targets, dtype=torch.float32, device=device)
    with torch.no_grad():
        prediction = _predict_batched(model, x, torch.as_tensor(offsets, dtype=torch.long, device=device), 1024, device.startswith("cuda"))
    errors = torch.linalg.vector_norm(prediction - y, dim=-1)[torch.as_tensor(valid, dtype=torch.bool, device=device)] * 1000
    return {"schema": "animcv_supervised_lifter_evaluation_v1", "frame_count": len(inputs),
            "mpjpe_mm": float(errors.mean().item()), "p95_joint_error_mm": float(torch.quantile(errors, 0.95).item()),
            "valid_joint_count": int(valid.sum()),
            "passed": False, "verdict": "informational: evaluate on a held-out sequence before defining a gate"}


def infer(pose: PoseSequence, checkpoint_path: str | Path, image_size: tuple[int, int], device: str = "cpu") -> LiftedPoseSequence:
    """Run a trained own-data baseline and emit AnimCV's standard 3D artifact."""
    torch, nn = _torch()
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint.get("schema") not in ("animcv_temporal_lifter_checkpoint_v1", "animcv_temporal_lifter_checkpoint_v2"):
        raise ValueError("unsupported temporal lifter checkpoint")
    model = _model(nn, int(checkpoint["channels"])).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    inputs = []
    for frame in pose.frames:
        joints = []
        for name in H36M_NAMES:
            landmark = frame.landmarks.get("neck" if name == "thorax" else name)
            if landmark is None:
                raise ValueError(f"frame {frame.frame_index} lacks {name} required by the supervised lifter")
            joints.append([landmark.x / width, landmark.y / height, landmark.confidence])
        inputs.append(joints)
    x = torch.as_tensor(np.asarray(inputs), dtype=torch.float32, device=device)
    offsets = _window_offsets(len(inputs), int(checkpoint["window"]))
    with torch.no_grad():
        prediction = _predict_batched(model, x, torch.as_tensor(offsets, dtype=torch.long, device=device), 1024, device.startswith("cuda")).cpu().numpy()
    frames = []
    for source, values in zip(pose.frames, prediction):
        points = {
            name: LiftedPosePoint(name, tuple(float(value) for value in position), 1.0, 0.0)
            for name, position in zip(H36M_NAMES, values)
        }
        frames.append(LiftedPoseFrame(source.frame_index, source.timestamp, points))
    return LiftedPoseSequence(
        frames=frames, source_fps=pose.source_fps, backend="animcv_supervised_temporal_lifter_v1",
        observation_confidence_threshold=pose.observation_confidence_threshold,
    )


def _arrays(dataset: dict[str, Any], window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sequences = dataset.get("sequences", [{"frames": dataset["frames"]}])
    frames = [frame for sequence in sequences for frame in sequence["frames"]]
    inputs = np.asarray([frame["input_2d"] for frame in frames], dtype=np.float32)
    targets = np.asarray([frame["target_3d"] for frame in frames], dtype=np.float32)
    valid = np.asarray([frame.get("target_valid", [True] * len(H36M_NAMES)) for frame in frames], dtype=bool)
    offset_groups, cursor = [], 0
    for sequence in sequences:
        length = len(sequence["frames"])
        offset_groups.append(_window_offsets(length, window) + cursor)
        cursor += length
    return inputs, targets, valid, np.concatenate(offset_groups)


def _window_offsets(length: int, window: int) -> np.ndarray:
    radius = window // 2
    return np.clip(np.arange(length)[:, None] + np.arange(-radius, radius + 1), 0, length - 1)


def _predict_batched(model, x, offsets, batch_size: int, amp_enabled: bool):
    """Vectorized evaluation/inference without materialising every window at once."""
    torch, _ = _torch()
    predictions = []
    for batch_offsets in offsets.split(batch_size):
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            predictions.append(model(x[batch_offsets]))
    return torch.cat(predictions, dim=0)


def _rank_shard(torch, indices, rank: int, world_size: int):
    """Give every DDP rank the same step count, padding only the final global shard."""
    if world_size == 1:
        return indices
    per_rank = (len(indices) + world_size - 1) // world_size
    total = per_rank * world_size
    if total != len(indices):
        indices = torch.cat((indices, indices[:total - len(indices)]))
    return indices.reshape(world_size, per_rank)[rank]


def _distributed_context(torch, config: TrainingConfig) -> dict[str, Any]:
    if not config.distributed:
        return {"enabled": False, "rank": 0, "local_rank": 0, "world_size": 1, "primary": True,
                "device": torch.device(config.device)}
    import os
    if not config.device.startswith("cuda"):
        raise ValueError("distributed training requires a CUDA device")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size < 2:
        raise ValueError("--distributed requires torchrun with at least two processes")
    if not torch.cuda.is_available() or local_rank >= torch.cuda.device_count():
        raise RuntimeError("torchrun rank does not map to an available CUDA device")
    torch.cuda.set_device(local_rank)
    # device_id is available in modern PyTorch and lets NCCL bind eagerly;
    # retain compatibility with older supported CUDA images.
    try:
        torch.distributed.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
    except TypeError:
        torch.distributed.init_process_group(backend="nccl")
    return {"enabled": True, "rank": rank, "local_rank": local_rank, "world_size": world_size,
            "primary": rank == 0, "device": torch.device(f"cuda:{local_rank}")}


def _model(nn, channels: int):
    class TemporalLifter(nn.Module):
        def __init__(self):
            super().__init__()
            self.network = nn.Sequential(nn.Conv1d(17 * 3, channels, 3, padding=1), nn.ReLU(),
                                         nn.Conv1d(channels, channels, 3, padding=1), nn.ReLU())
            self.head = nn.Linear(channels, 17 * 3)

        def forward(self, values):
            batch, frames, joints, features = values.shape
            encoded = self.network(values.reshape(batch, frames, joints * features).transpose(1, 2))
            return self.head(encoded[:, :, frames // 2]).reshape(batch, 17, 3)
    return TemporalLifter()


def _torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise ImportError("supervised 3D lifter training requires torch; install the training extra") from exc
    return torch, nn


def preflight(device: str = "cuda") -> dict[str, Any]:
    """Report whether the requested training device is genuinely usable."""
    torch, _ = _torch()
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; install a CUDA-enabled PyTorch build or use cpu")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return {
        "schema": "animcv_training_preflight_v1", "requested_device": device,
        "torch_version": torch.__version__, "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda, "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "passed": True,
    }
