"""Small, from-scratch temporal 2D-to-3D lifter and data contract.

This module contains no pretrained weights and accepts only data the project
owner is entitled to use.  Its deliberately compact TCN is a reproducible
baseline; it is not represented as a production-quality estimator until it
passes the calibrated holdout gates.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from common.serialization import read_json, write_json
from pose.pose_lifter import H36M_NAMES, LiftedPoseSequence
from pose.pose_lifter import LiftedPoseFrame, LiftedPosePoint
from pose.pose_types import PoseSequence


SCHEMA = "animcv_supervised_3d_lifter_dataset_v1"


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
        for name in H36M_NAMES:
            landmark_name = "neck" if name == "thorax" else name
            landmark = frame.landmarks.get(landmark_name)
            point = target_frame.points.get(name) or target_frame.points.get(landmark_name)
            if landmark is None or point is None:
                raise ValueError(f"frame {frame.frame_index} lacks {landmark_name} required by the supervised schema")
            inputs.append([landmark.x / width, landmark.y / height, landmark.confidence])
            outputs.append(list(point.position))
        frames.append({"frame_index": frame.frame_index, "input_2d": inputs, "target_3d": outputs})
    if not frames:
        raise ValueError("no aligned 2D/3D frames for supervised dataset")
    return {
        "schema": SCHEMA, "joint_names": list(H36M_NAMES), "sequence_id": sequence_id,
        "source_fps": pose.source_fps, "image_size": [width, height], "frames": frames,
    }


def save_dataset(dataset: dict[str, Any], path: str | Path) -> None:
    write_json(path, dataset)


def load_dataset(path: str | Path) -> dict[str, Any]:
    dataset = read_json(path)
    if dataset.get("schema") != SCHEMA or dataset.get("joint_names") != list(H36M_NAMES):
        raise ValueError("unsupported supervised 3D lifter dataset")
    if not dataset.get("frames"):
        raise ValueError("supervised dataset contains no frames")
    return dataset


@dataclass(frozen=True)
class TrainingConfig:
    window: int = 81
    channels: int = 256
    epochs: int = 30
    batch_size: int = 128
    learning_rate: float = 1e-3
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.window < 3 or self.window % 2 == 0:
            raise ValueError("window must be an odd value of at least 3")
        if min(self.channels, self.epochs, self.batch_size) <= 0 or self.learning_rate <= 0:
            raise ValueError("training dimensions, epochs, batch size, and learning rate must be positive")


def train(dataset: dict[str, Any], checkpoint_path: str | Path, config: TrainingConfig) -> dict[str, Any]:
    torch, nn = _torch()
    inputs, targets = _arrays(dataset)
    model = _model(nn, config.channels).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    x = torch.as_tensor(inputs, dtype=torch.float32, device=config.device)
    y = torch.as_tensor(targets, dtype=torch.float32, device=config.device)
    offsets = _window_offsets(len(inputs), config.window)
    indices = torch.arange(len(inputs), device=config.device)
    for _ in range(config.epochs):
        permutation = indices[torch.randperm(len(indices), device=config.device)]
        for batch in permutation.split(config.batch_size):
            windows = torch.stack([x[offsets[int(index)]] for index in batch])
            prediction = model(windows)
            loss = torch.nn.functional.smooth_l1_loss(prediction, y[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        prediction = torch.stack([model(x[offsets[index:index + 1]])[0] for index in range(len(inputs))])
        mpjpe_mm = float(torch.linalg.vector_norm(prediction - y, dim=-1).mean().item() * 1000)
    payload = {"schema": "animcv_temporal_lifter_checkpoint_v1", "joint_names": list(H36M_NAMES),
               "channels": config.channels, "window": config.window, "state_dict": model.state_dict()}
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint_path)
    return {"frame_count": len(inputs), "training_mpjpe_mm": mpjpe_mm, "config": config.__dict__}


def evaluate(dataset: dict[str, Any], checkpoint_path: str | Path, device: str = "cpu") -> dict[str, Any]:
    torch, nn = _torch()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint.get("schema") != "animcv_temporal_lifter_checkpoint_v1":
        raise ValueError("unsupported temporal lifter checkpoint")
    inputs, targets = _arrays(dataset)
    model = _model(nn, int(checkpoint["channels"])).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    x = torch.as_tensor(inputs, dtype=torch.float32, device=device)
    y = torch.as_tensor(targets, dtype=torch.float32, device=device)
    offsets = _window_offsets(len(inputs), int(checkpoint["window"]))
    with torch.no_grad():
        prediction = torch.stack([model(x[offsets[index:index + 1]])[0] for index in range(len(inputs))])
    errors = torch.linalg.vector_norm(prediction - y, dim=-1).flatten() * 1000
    return {"schema": "animcv_supervised_lifter_evaluation_v1", "frame_count": len(inputs),
            "mpjpe_mm": float(errors.mean().item()), "p95_joint_error_mm": float(torch.quantile(errors, 0.95).item()),
            "passed": False, "verdict": "informational: evaluate on a held-out sequence before defining a gate"}


def infer(pose: PoseSequence, checkpoint_path: str | Path, image_size: tuple[int, int], device: str = "cpu") -> LiftedPoseSequence:
    """Run a trained own-data baseline and emit AnimCV's standard 3D artifact."""
    torch, nn = _torch()
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint.get("schema") != "animcv_temporal_lifter_checkpoint_v1":
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
        prediction = torch.stack([model(x[offsets[index:index + 1]])[0] for index in range(len(inputs))]).cpu().numpy()
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


def _arrays(dataset: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    return (np.asarray([frame["input_2d"] for frame in dataset["frames"]], dtype=np.float32),
            np.asarray([frame["target_3d"] for frame in dataset["frames"]], dtype=np.float32))


def _window_offsets(length: int, window: int) -> np.ndarray:
    radius = window // 2
    return np.clip(np.arange(length)[:, None] + np.arange(-radius, radius + 1), 0, length - 1)


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
