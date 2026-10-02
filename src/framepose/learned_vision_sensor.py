"""DIAGNOSTIC (Worklog 68): learned left-arm near/far sensor over frozen Qwen3-VL visual tokens.

Feature source (fixed by inspection of transformers 4.57.6, not by accuracy):
``Qwen3VLModel.get_image_features`` -> ``self.visual(...)`` image embeddings,
i.e. the vision-tower patch-merger output that is placed into the language
model's image-token slots, before any autoregressive decoding.  The Qwen2-VL
image processor orders patches as (t, h/2, w/2, 2, 2) and the merger folds
each 2x2 group into one token, so the output is a regular row-major
(h/2) x (w/2) grid.  A 448 px crop gives 28x28 patches and a 14x14 token grid
(one token per 32x32 crop pixels), 4096 channels each.

The vision input is the FramePose person crop only (no S/E/W text).  Joint
identity reaches the learned head numerically: deterministic bilinear samples
of the frozen grid at the detector shoulder / elbow / wrist, one global mean
token, and pairwise image-plane geometry.  No H0, Depth Anything, GT or sign
input.  Not imported by any production module.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from framepose.contract import JOINT_INDEX
from framepose.crops import CROP_CONTRACT, crop_box, render_crop

MODEL_REPOSITORY = "Qwen/Qwen3-VL-8B-Instruct"
MODEL_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
MODEL_FINGERPRINT = "a0d72ded575eaa4460dad11dd1e313bc69f0403b2c97c3c1ba234af086952904"
FEATURE_SOURCE = "Qwen3VLModel.get_image_features -> visual patch-merger image embeddings (pre-LLM, no deepstack)"
RESOLUTION = 448
TOKEN_PIXELS = 32
GRID = RESOLUTION // TOKEN_PIXELS  # 14
CHANNELS = 4096
ARM = ("left_shoulder", "left_elbow", "left_wrist")
PAIRS = (("shoulder_elbow", 0, 1, "upper"), ("elbow_wrist", 1, 2, "lower"), ("shoulder_wrist", 0, 2, "chain"))
STABLE_SINE = math.sin(math.radians(10))
CLASSES = ("FIRST_CLOSER", "SECOND_CLOSER", "UNCLEAR")
VISUAL_PROJECTION = 64
HIDDEN = 64
GEOMETRY_FEATURES = 13


def crop_for_vision(image_rgb: np.ndarray, input_2d: np.ndarray, input_valid: np.ndarray,
                    image_size: tuple[int, int]) -> tuple[np.ndarray, Any]:
    """The FramePose crop region rendered at 448 px with no overlay."""
    if (image_rgb.shape[1], image_rgb.shape[0]) != tuple(image_size):
        raise ValueError("image does not match the sample's image size")
    box = crop_box(input_2d, input_valid, image_size)
    return render_crop(image_rgb, box, RESOLUTION), box


def token_coordinates(input_2d: np.ndarray, image_size: tuple[int, int], box) -> np.ndarray:
    """Detector joints -> continuous token-grid (gx, gy); token centres at integers."""
    width, height = image_size
    px = np.asarray(input_2d, float)[:, 0] * width
    py = np.asarray(input_2d, float)[:, 1] * height
    u = (px - box.x) / box.side * RESOLUTION
    v = (py - box.y) / box.side * RESOLUTION
    return np.stack([u / TOKEN_PIXELS - 0.5, v / TOKEN_PIXELS - 0.5], axis=-1)


def bilinear_sample(grid: np.ndarray, gx: float, gy: float) -> np.ndarray:
    """Sample a (GRID, GRID, C) token grid at continuous (gx, gy), clamped to the grid."""
    g = np.asarray(grid, dtype=np.float32)
    h, w = g.shape[:2]
    gx = float(np.clip(gx, 0.0, w - 1.0))
    gy = float(np.clip(gy, 0.0, h - 1.0))
    x0, y0 = int(math.floor(gx)), int(math.floor(gy))
    x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
    ax, ay = gx - x0, gy - y0
    return ((1 - ax) * (1 - ay) * g[y0, x0] + ax * (1 - ay) * g[y0, x1]
            + (1 - ax) * ay * g[y1, x0] + ax * ay * g[y1, x1])


def readout(tokens: np.ndarray, input_2d: np.ndarray, input_valid: np.ndarray, image_size, box) -> tuple[np.ndarray, np.ndarray]:
    """(4, C) = [shoulder, elbow, wrist, global mean] and (3,) joint validity.

    ``tokens`` is the merger output in its native row-major order (GRID*GRID, C).
    Invalid detector joints give a zero vector and validity 0.
    """
    t = np.asarray(tokens, dtype=np.float32)
    if t.shape[0] != GRID * GRID:
        raise ValueError(f"expected {GRID * GRID} visual tokens, got {t.shape[0]}")
    grid = t.reshape(GRID, GRID, -1)
    coords = token_coordinates(input_2d, image_size, box)
    out = np.zeros((4, t.shape[1]), np.float32)
    valid = np.zeros(3, bool)
    for k, joint in enumerate(ARM):
        index = JOINT_INDEX[joint]
        if bool(input_valid[index]):
            out[k] = bilinear_sample(grid, *coords[index])
            valid[k] = True
    out[3] = t.mean(axis=0)
    return out, valid


def pair_geometry(geometry: np.ndarray) -> np.ndarray:
    """`(N, 3, 13)` from the H0 crop-normalized geometry tensor `(N, 17, 4)`.

    Per pair: [xA, yA, cA, vA, xB, yB, cB, vB, dx, dy, length, cos, sin] in the
    crop frame (x right, y down as in the tensor).  Runtime observable only.
    """
    g = np.asarray(geometry, dtype=np.float32)
    arm = g[:, [JOINT_INDEX[j] for j in ARM]]
    out = np.zeros((g.shape[0], 3, GEOMETRY_FEATURES), np.float32)
    for k, (_, a, b, _) in enumerate(PAIRS):
        A, B = arm[:, a], arm[:, b]
        d = B[:, :2] - A[:, :2]
        length = np.linalg.norm(d, axis=1)
        safe = np.where(length > 1e-6, length, 1.0)
        both = (A[:, 3] > 0) & (B[:, 3] > 0)
        cos = np.where(both & (length > 1e-6), d[:, 0] / safe, 0.0)
        sin = np.where(both & (length > 1e-6), d[:, 1] / safe, 0.0)
        out[:, k] = np.concatenate([A, B, np.where(both[:, None], d, 0.0),
                                    np.where(both, length, 0.0)[:, None], cos[:, None], sin[:, None]], axis=1)
    return out


def pair_labels(f_gt: np.ndarray, masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Supervised target only: 0 FIRST_CLOSER (f > sin10), 1 SECOND_CLOSER (f < -sin10), 2 UNCLEAR."""
    f = np.asarray(f_gt, dtype=np.float64)
    labels = np.where(f > STABLE_SINE, 0, np.where(f < -STABLE_SINE, 1, 2)).astype(np.int64)
    return labels, np.asarray(masks, dtype=bool)


def sensor_inputs(readouts: np.ndarray, joint_valid: np.ndarray, candidate: str) -> np.ndarray:
    """`(N, 4, C)` visual inputs; G0 replaces every frozen feature with exact zero."""
    r = np.asarray(readouts)
    if candidate == "G0_ZERO_VISION":
        return np.zeros_like(r, dtype=np.float16)
    if candidate == "G1_QWEN_VISION":
        return r
    raise ValueError(f"unknown candidate {candidate!r}")


def build_sensor(channels: int = CHANNELS):
    import torch
    from torch import nn

    class NearFarSensor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.visual = nn.Linear(channels, VISUAL_PROJECTION)
            self.heads = nn.ModuleList(
                nn.Sequential(nn.Linear(GEOMETRY_FEATURES + 3 * VISUAL_PROJECTION, HIDDEN), nn.GELU(),
                              nn.Linear(HIDDEN, len(CLASSES)))
                for _ in PAIRS)

        def forward(self, visual, geometry):
            """visual (B, 4, C) [S, E, W, global]; geometry (B, 3, 13) -> logits (B, 3, 3)."""
            v = self.visual(visual.float())
            logits = []
            for k, (_, a, b, _) in enumerate(PAIRS):
                x = torch.cat([geometry[:, k], v[:, a], v[:, b], v[:, 3]], dim=-1)
                logits.append(self.heads[k](x))
            return torch.stack(logits, dim=1)

    return NearFarSensor()


def predicted_state(probabilities: np.ndarray) -> list[str]:
    return [CLASSES[i] for i in np.argmax(probabilities, axis=-1)]
