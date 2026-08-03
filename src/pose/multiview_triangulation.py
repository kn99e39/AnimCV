"""Triangulate licensed, synchronised camera observations into 3D training GT.

The calibration file uses ordinary OpenCV camera coordinates: +X right, +Y
down, +Z forward.  The emitted ``LiftedPoseSequence`` uses AnimCV's camera
axes (+X right, +Y forward, +Z up) and is pelvis-root-relative in metres.
Keeping this conversion here, at the acquisition boundary, avoids silently
mixing the two conventions downstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from common.serialization import read_json
from pose.pose_lifter import LiftedPoseFrame, LiftedPosePoint, LiftedPoseSequence
from pose.pose_types import PoseSequence


SCHEMA = "animcv_multiview_calibration_v1"


@dataclass(frozen=True)
class MultiviewCamera:
    name: str
    matrix: np.ndarray
    projection: np.ndarray


def load_calibration(path: str | Path) -> dict[str, MultiviewCamera]:
    """Load a metric world-to-camera calibration, validating its geometry."""
    data = read_json(path)
    if data.get("schema") != SCHEMA or data.get("world_units") != "metres":
        raise ValueError(f"expected {SCHEMA} calibration in metres")
    cameras: dict[str, MultiviewCamera] = {}
    for name, value in data.get("cameras", {}).items():
        intrinsics = value.get("intrinsics", {})
        try:
            fx, fy, cx, cy = (float(intrinsics[key]) for key in ("fx", "fy", "cx", "cy"))
            matrix = np.asarray(value["world_to_camera"], dtype=float)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"camera {name} is missing valid intrinsics or world_to_camera") from exc
        if matrix.shape != (4, 4) or not np.allclose(matrix[3], (0, 0, 0, 1)):
            raise ValueError(f"camera {name} world_to_camera must be a 4x4 homogeneous matrix")
        rotation = matrix[:3, :3]
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
            raise ValueError(f"camera {name} world_to_camera rotation must be orthonormal")
        if fx <= 0 or fy <= 0:
            raise ValueError(f"camera {name} focal lengths must be positive")
        intrinsic_matrix = np.array(((fx, 0.0, cx), (0.0, fy, cy), (0.0, 0.0, 1.0)))
        cameras[name] = MultiviewCamera(name, matrix, intrinsic_matrix @ matrix[:3])
    if len(cameras) < 2:
        raise ValueError("multiview triangulation needs calibration for at least two cameras")
    return cameras


def triangulate(
    observations: dict[str, PoseSequence], calibration: dict[str, MultiviewCamera], reference_camera: str,
    min_confidence: float = 0.3, max_reprojection_error_pixels: float = 10.0,
) -> tuple[LiftedPoseSequence, dict[str, Any]]:
    """Triangulate frame-index-aligned canonical 2D observations.

    A joint is only retained when at least two confident visible cameras agree
    within the supplied reprojection threshold.  Invalid joints are retained
    with ``observation_valid=False`` so the data rejection is auditable.
    """
    if not 0 <= min_confidence <= 1 or max_reprojection_error_pixels <= 0:
        raise ValueError("invalid triangulation confidence or reprojection threshold")
    if reference_camera not in calibration or reference_camera not in observations:
        raise ValueError("reference camera must have both calibration and observations")
    unknown = set(observations) - set(calibration)
    if unknown:
        raise ValueError(f"observations lack calibration: {', '.join(sorted(unknown))}")
    indexed = {name: {frame.frame_index: frame for frame in sequence.frames} for name, sequence in observations.items()}
    shared_indices = sorted(set.intersection(*(set(frames) for frames in indexed.values())))
    if not shared_indices:
        raise ValueError("camera observations have no shared frame_index values")
    names = sorted(set.union(*(set(frame.landmarks) for frames in indexed.values() for frame in frames.values())))
    output, errors = [], []
    valid_count = 0
    insufficient_count = 0
    rejected_count = 0
    reference_frames = indexed[reference_camera]
    for frame_index in shared_indices:
        reconstructed: dict[str, tuple[np.ndarray, float, bool]] = {}
        for name in names:
            views = []
            for camera_name, frames in indexed.items():
                landmark = frames[frame_index].landmarks.get(name)
                if landmark and landmark.visible and landmark.confidence >= min_confidence:
                    views.append((calibration[camera_name], landmark.x, landmark.y, landmark.confidence))
            if len(views) < 2:
                reconstructed[name] = (np.zeros(3), 0.0, False)
                insufficient_count += 1
                continue
            world = _dlt(views)
            reprojection = [_reprojection_error(world, camera, x, y) for camera, x, y, _ in views]
            max_error = max(reprojection)
            if not np.all(np.isfinite(world)) or max_error > max_reprojection_error_pixels:
                reconstructed[name] = (np.zeros(3), 0.0, False)
                rejected_count += 1
                continue
            reference_standard = (calibration[reference_camera].matrix @ np.append(world, 1.0))[:3]
            # OpenCV camera (+x right,+y down,+z forward) -> AnimCV (+x right,+y forward,+z up).
            reconstructed[name] = (np.array((reference_standard[0], reference_standard[2], -reference_standard[1])),
                                   min(view[3] for view in views), True)
            errors.extend(reprojection)
            valid_count += 1
        pelvis, _, pelvis_valid = reconstructed.get("pelvis", (np.zeros(3), 0.0, False))
        points = {}
        for name, (position, confidence, valid) in reconstructed.items():
            points[name] = LiftedPosePoint(name, tuple((position - pelvis) if valid and pelvis_valid else np.zeros(3)),
                                            confidence, 0.0, valid)
        source = reference_frames[frame_index]
        output.append(LiftedPoseFrame(frame_index, source.timestamp, points))
    values = np.asarray(errors, dtype=float)
    report = {
        "schema": "animcv_multiview_triangulation_report_v1", "frame_count": len(output),
        "joint_sample_count": len(output) * len(names), "triangulated_joint_count": valid_count,
        "coverage": valid_count / max(1, len(output) * len(names)), "insufficient_view_joint_count": insufficient_count,
        "reprojection_rejected_joint_count": rejected_count,
        "mean_reprojection_error_pixels": float(values.mean()) if len(values) else None,
        "p95_reprojection_error_pixels": float(np.quantile(values, 0.95)) if len(values) else None,
        "passed": bool(len(values) and valid_count / max(1, len(output) * len(names)) >= 0.95
                       and float(np.quantile(values, 0.95)) <= max_reprojection_error_pixels),
    }
    return LiftedPoseSequence(output, observations[reference_camera].source_fps,
                              backend="multiview_triangulation_calibrated_v1",
                              observation_confidence_threshold=min_confidence), report


def _dlt(views: list[tuple[MultiviewCamera, float, float, float]]) -> np.ndarray:
    rows = []
    for camera, x, y, _ in views:
        rows.extend((x * camera.projection[2] - camera.projection[0], y * camera.projection[2] - camera.projection[1]))
    _, _, vectors = np.linalg.svd(np.asarray(rows))
    homogeneous = vectors[-1]
    return homogeneous[:3] / homogeneous[3]


def _reprojection_error(world: np.ndarray, camera: MultiviewCamera, x: float, y: float) -> float:
    homogeneous = camera.projection @ np.append(world, 1.0)
    return float(np.hypot(homogeneous[0] / homogeneous[2] - x, homogeneous[1] / homogeneous[2] - y))
