"""Deterministic person-centric image preprocessing.

Every RGB candidate sees exactly the same pixel region of exactly the same
frame. The crop is derived from the 2D observation the model is also given, so
no candidate receives extra localisation information, and no margin is tuned by
a holdout result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


# Fixed once, before any candidate was trained. Not swept, not tuned.
CROP_MARGIN = 0.25          # half-width padding added to the joint bounding box
CROP_MIN_PIXELS = 32.0      # floor so a degenerate box cannot collapse
CROP_RESOLUTION = 224       # square, matches both backbones' pretraining size
CROP_PAD_VALUE = 0          # constant black padding outside the source image
CROP_RESAMPLE = "bilinear"

CROP_CONTRACT_VERSION = "animcv_frame_pose_crop_contract_v1"

CROP_CONTRACT: dict[str, Any] = {
    "schema": CROP_CONTRACT_VERSION,
    "definition": "square box centred on the valid-2D-joint bounding box",
    "margin": CROP_MARGIN,
    "margin_rule": "side = max(box_width, box_height) * (1 + 2 * margin), clamped to >= 32 px",
    "min_pixels": CROP_MIN_PIXELS,
    "resolution": CROP_RESOLUTION,
    "resample": CROP_RESAMPLE,
    "padding": "constant 0 outside the source image; the box is never clamped inwards",
    "geometry_mapping": "joint pixel -> (pixel - origin) / side -> 2 * u - 1 in [-1, 1]",
    "fallback": "fewer than two valid joints -> centred square of side min(width, height)",
}


@dataclass(frozen=True)
class CropBox:
    """Square crop in source-image pixels; may extend outside the image."""

    x: float
    y: float
    side: float

    def to_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "side": self.side}


def crop_box(input_2d: np.ndarray, input_valid: np.ndarray,
             image_size: tuple[int, int]) -> CropBox:
    """Derive the deterministic crop from the observation, not from ground truth."""
    width, height = float(image_size[0]), float(image_size[1])
    pixels = np.asarray(input_2d, dtype=np.float64)[:, :2] * np.asarray([width, height])
    valid = np.asarray(input_valid, dtype=bool)
    if int(valid.sum()) < 2:
        side = min(width, height)
        return CropBox((width - side) / 2.0, (height - side) / 2.0, side)
    visible = pixels[valid]
    minimum = visible.min(axis=0)
    maximum = visible.max(axis=0)
    centre = (minimum + maximum) / 2.0
    side = float(max(float((maximum - minimum).max()) * (1.0 + 2.0 * CROP_MARGIN), CROP_MIN_PIXELS))
    return CropBox(float(centre[0] - side / 2.0), float(centre[1] - side / 2.0), side)


def geometry_in_crop(input_2d: np.ndarray, input_valid: np.ndarray,
                     image_size: tuple[int, int], box: CropBox) -> np.ndarray:
    """`(17, 4)` geometry token features: `x, y in [-1, 1]`, confidence, validity.

    The geometry input is held identical across candidates: F0 receives exactly
    the geometry F1 and F2 receive.

    That alone does **not** make F0 vs F1/F2 an information-only or
    capacity-matched comparison. F0 also has no image projection and no
    cross-attention sublayer, and so a different trainable parameter count. Only
    F1 vs F2 is architecture-matched; see Architecture_v3 section 9.1.
    """
    width, height = float(image_size[0]), float(image_size[1])
    pixels = np.asarray(input_2d, dtype=np.float64)[:, :2] * np.asarray([width, height])
    normalized = (pixels - np.asarray([box.x, box.y])) / max(box.side, 1e-6)
    normalized = 2.0 * normalized - 1.0
    valid = np.asarray(input_valid, dtype=bool)
    features = np.zeros((normalized.shape[0], 4), dtype=np.float32)
    features[:, :2] = np.where(valid[:, None], normalized, 0.0)
    features[:, 2] = np.where(valid, np.asarray(input_2d, dtype=np.float64)[:, 2], 0.0)
    features[:, 3] = valid.astype(np.float32)
    return features


def render_crop(image: np.ndarray, box: CropBox, resolution: int = CROP_RESOLUTION) -> np.ndarray:
    """Resample the crop to `resolution x resolution` uint8 RGB.

    Implemented on the array so the mapping stays identical whichever image
    reader produced `image`; padding outside the source is constant black.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be HxWx3 RGB")
    height, width = image.shape[:2]
    # Sample centres of the destination grid, mapped back into source pixels.
    steps = (np.arange(resolution, dtype=np.float64) + 0.5) / resolution
    xs = box.x + steps * box.side
    ys = box.y + steps * box.side
    grid_x, grid_y = np.meshgrid(xs, ys)
    return _bilinear_sample(image, grid_x, grid_y, width, height)


def _bilinear_sample(image: np.ndarray, grid_x: np.ndarray, grid_y: np.ndarray,
                     width: int, height: int) -> np.ndarray:
    x0 = np.floor(grid_x - 0.5).astype(np.int64)
    y0 = np.floor(grid_y - 0.5).astype(np.int64)
    fx = (grid_x - 0.5) - x0
    fy = (grid_y - 0.5) - y0
    accumulator = np.zeros(grid_x.shape + (3,), dtype=np.float64)
    for dy, wy in ((0, 1.0 - fy), (1, fy)):
        for dx, wx in ((0, 1.0 - fx), (1, fx)):
            xi = x0 + dx
            yi = y0 + dy
            inside = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
            safe_x = np.clip(xi, 0, width - 1)
            safe_y = np.clip(yi, 0, height - 1)
            weight = (wy * wx * inside)[..., None]
            accumulator += image[safe_y, safe_x].astype(np.float64) * weight
    # Outside the source image the accumulated weight is missing, which leaves
    # exactly the constant CROP_PAD_VALUE there.
    return np.clip(accumulator + CROP_PAD_VALUE * 0.0, 0, 255).astype(np.uint8)


def crop_contract_digest() -> str:
    """Stable identity of the crop contract in force.

    A frozen visual feature is a function of the crop that produced it, so a
    cache built under one crop contract must never be silently reused under
    another. This digest is what binds that into the visual-input identity.
    """
    import hashlib
    import json

    return hashlib.sha256(json.dumps(CROP_CONTRACT, sort_keys=True).encode("utf-8")).hexdigest()
