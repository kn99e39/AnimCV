"""Shared synthetic fixtures for the frame-first pose core tests."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from common.serialization import write_json
from framepose.contract import JOINT_INDEX, JOINT_NAMES


IMAGE_SIZE = (192, 144)


def _frame(rng: np.random.Generator, frame_index: int, facing: float) -> dict:
    """One synthetic frame with a coherent standing pose at a known facing."""
    target = np.zeros((len(JOINT_NAMES), 3), dtype=float)
    right = np.asarray([math.cos(facing), -math.sin(facing), 0.0])
    up = np.asarray([0.0, 0.0, 1.0])
    layout = {
        "pelvis": (0.0, 0.0), "spine": (0.0, 0.2), "thorax": (0.0, 0.45), "neck": (0.0, 0.5),
        "head": (0.0, 0.7), "left_shoulder": (-0.18, 0.45), "right_shoulder": (0.18, 0.45),
        "left_elbow": (-0.28, 0.2), "right_elbow": (0.28, 0.2),
        "left_wrist": (-0.32, -0.05), "right_wrist": (0.32, -0.05),
        "left_hip": (-0.11, 0.0), "right_hip": (0.11, 0.0),
        "left_knee": (-0.12, -0.45), "right_knee": (0.12, -0.45),
        "left_ankle": (-0.12, -0.9), "right_ankle": (0.12, -0.9),
    }
    for name, (lateral, vertical) in layout.items():
        target[JOINT_INDEX[name]] = right * lateral + up * vertical
    target += rng.normal(scale=0.002, size=target.shape)
    # Targets are pelvis-relative by contract: centre *after* perturbation.
    target = target - target[JOINT_INDEX["pelvis"]]

    width, height = IMAGE_SIZE
    projected = np.zeros((len(JOINT_NAMES), 3), dtype=float)
    projected[:, 0] = 0.5 + target[:, 0] * 0.4
    projected[:, 1] = 0.5 - target[:, 2] * 0.4
    projected[:, 2] = 0.9
    valid = np.ones(len(JOINT_NAMES), dtype=bool)
    if frame_index % 7 == 3:
        # A partially visible frame, so visibility strata and masking are exercised.
        projected[JOINT_INDEX["left_wrist"]] = (0.0, 0.0, 0.0)
        valid[JOINT_INDEX["left_wrist"]] = False
    return {
        "frame_index": frame_index,
        "input_2d": projected.tolist(),
        "target_3d": target.tolist(),
        "target_valid": valid.tolist(),
    }


def prepared_dataset(path: Path, *, split: str, sequences: list[str], frames: int = 24,
                     dataset: str = "3DPW", seed: int = 0) -> Path:
    """Write a synthetic `animcv_supervised_3d_lifter_dataset_v2` artifact."""
    rng = np.random.default_rng(seed)
    payload_sequences = []
    for index, name in enumerate(sequences):
        payload_sequences.append({
            "sequence_id": name,
            "source_fps": 30.0,
            "image_size": list(IMAGE_SIZE),
            "source": {"dataset": dataset, "split": split},
            "frames": [_frame(rng, position, facing=(index + 1) * 0.7 + position * 0.05)
                       for position in range(frames)],
        })
    payload = {
        "schema": "animcv_supervised_3d_lifter_dataset_v2",
        "joint_names": list(JOINT_NAMES),
        "sequence_id": "combined",
        "source_fps": 30.0,
        "image_size": list(IMAGE_SIZE),
        "source": {"dataset": dataset, "split": split},
        "sequences": payload_sequences,
        "frames": [frame for sequence in payload_sequences for frame in sequence["frames"]],
    }
    write_json(path, payload)
    return path


def write_images(root: Path, sequences: list[str], frames: int = 24) -> Path:
    """Write deterministic JPEG frames matching the 3DPW image layout."""
    from PIL import Image

    rng = np.random.default_rng(7)
    width, height = IMAGE_SIZE
    for name in sequences:
        folder = root / name.split(":")[1]
        folder.mkdir(parents=True, exist_ok=True)
        for index in range(frames):
            pixels = (rng.random((height, width, 3)) * 255).astype(np.uint8)
            Image.fromarray(pixels).save(folder / f"image_{index:05d}.jpg", quality=95)
    return root
