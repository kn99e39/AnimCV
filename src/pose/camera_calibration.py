"""Calibrated pinhole-camera contract and reprojection utilities.

The lifted pose is pelvis-relative in the documented camera axes: +X right,
+Y forward, +Z up.  A calibrated audit fits only the unknown per-frame pelvis
translation; it never treats that fitted translation as validated root motion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any

from common.serialization import read_json, write_json
from common.types import Vec3


@dataclass(frozen=True)
class CameraCalibration:
    image_width: int
    image_height: int
    fx: float
    fy: float
    cx: float
    cy: float
    radial_distortion: tuple[float, float, float] = (0.0, 0.0, 0.0)
    tangential_distortion: tuple[float, float] = (0.0, 0.0)
    source: str = "user_supplied"
    calibration_rms_pixels: float | None = None

    def __post_init__(self) -> None:
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("camera calibration image dimensions must be positive")
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("camera calibration focal lengths must be positive")
        if len(self.radial_distortion) != 3 or len(self.tangential_distortion) != 2:
            raise ValueError("expected radial [k1,k2,k3] and tangential [p1,p2] distortion")
        if self.calibration_rms_pixels is not None and self.calibration_rms_pixels < 0:
            raise ValueError("camera calibration RMS must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "animcv_camera_calibration_v1",
            "image_size": [self.image_width, self.image_height],
            "intrinsics": {"fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy},
            "distortion": {"radial": list(self.radial_distortion), "tangential": list(self.tangential_distortion)},
            "source": self.source,
            "calibration_rms_pixels": self.calibration_rms_pixels,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CameraCalibration":
        if data.get("schema") != "animcv_camera_calibration_v1":
            raise ValueError("unsupported camera calibration schema")
        width, height = data["image_size"]
        intrinsics = data["intrinsics"]
        distortion = data.get("distortion", {})
        return cls(
            image_width=int(width), image_height=int(height),
            fx=float(intrinsics["fx"]), fy=float(intrinsics["fy"]),
            cx=float(intrinsics["cx"]), cy=float(intrinsics["cy"]),
            radial_distortion=tuple(float(value) for value in distortion.get("radial", (0, 0, 0))),
            tangential_distortion=tuple(float(value) for value in distortion.get("tangential", (0, 0))),
            source=data.get("source", "user_supplied"),
            calibration_rms_pixels=data.get("calibration_rms_pixels"),
        )


def load_camera_calibration(path: str | Path) -> CameraCalibration:
    return CameraCalibration.from_dict(read_json(path))


def save_camera_calibration(calibration: CameraCalibration, path: str | Path) -> None:
    write_json(path, calibration.to_dict())


def project_camera_point(point: Vec3, calibration: CameraCalibration) -> tuple[float, float]:
    """Project a camera-space point using OpenCV-compatible distortion."""
    x, depth, z = point
    if depth <= 1e-6:
        raise ValueError("cannot project a point at or behind the camera")
    normalized_x, normalized_y = x / depth, -z / depth
    radius2 = normalized_x * normalized_x + normalized_y * normalized_y
    k1, k2, k3 = calibration.radial_distortion
    p1, p2 = calibration.tangential_distortion
    radial = 1.0 + k1 * radius2 + k2 * radius2 * radius2 + k3 * radius2 * radius2 * radius2
    distorted_x = normalized_x * radial + 2.0 * p1 * normalized_x * normalized_y + p2 * (radius2 + 2.0 * normalized_x * normalized_x)
    distorted_y = normalized_y * radial + p1 * (radius2 + 2.0 * normalized_y * normalized_y) + 2.0 * p2 * normalized_x * normalized_y
    return calibration.fx * distorted_x + calibration.cx, calibration.fy * distorted_y + calibration.cy


def fit_pelvis_translation(
    points: list[tuple[Vec3, tuple[float, float]]], calibration: CameraCalibration, iterations: int = 80
) -> Vec3:
    """Fit a translation to known intrinsics by deterministic 1D search.

    For each candidate forward distance, the lateral/up translation has a
    closed-form least-squares solution. Radial/tangential distortion is used
    in the final residual, so supplied lens data is never ignored.
    """
    if len(points) < 4:
        raise ValueError("calibrated reprojection needs at least four trusted joints per frame")
    body_span = max(math.dist(a, b) for a, _ in points for b, _ in points)
    lower, upper = max(0.1, body_span * 0.25), max(5.0, body_span * 100.0)

    def candidate(depth: float) -> tuple[float, Vec3]:
        tx_values = [((pixel[0] - calibration.cx) / calibration.fx) * (point[1] + depth) - point[0]
                     for point, pixel in points]
        tz_values = [(-(pixel[1] - calibration.cy) / calibration.fy) * (point[1] + depth) - point[2]
                     for point, pixel in points]
        translation = (sum(tx_values) / len(tx_values), depth, sum(tz_values) / len(tz_values))
        try:
            error = sum(math.dist(project_camera_point(_add(point, translation), calibration), pixel) ** 2
                        for point, pixel in points)
        except ValueError:
            return float("inf"), translation
        return error, translation

    # Golden-section search is stable, dependency-free, and sufficient because
    # this is only a validation fit, not a bundle-adjustment replacement.
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    c, d = upper - ratio * (upper - lower), lower + ratio * (upper - lower)
    for _ in range(iterations):
        if candidate(c)[0] <= candidate(d)[0]:
            upper, d = d, c
            c = upper - ratio * (upper - lower)
        else:
            lower, c = c, d
            d = lower + ratio * (upper - lower)
    return candidate((lower + upper) / 2.0)[1]


def _add(a: Vec3, b: Vec3) -> Vec3:
    return tuple(x + y for x, y in zip(a, b))  # type: ignore[return-value]
