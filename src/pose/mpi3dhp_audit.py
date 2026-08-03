"""Metric 2D audit against locally imported MPI-INF-3DHP ground truth."""

from __future__ import annotations

import math

from pose.pose_types import PoseSequence


def audit_mpi3dhp_2d(predicted: PoseSequence, ground_truth: PoseSequence) -> dict:
    """Report pixel error and PCK@0.2, normalized by GT person bounding box."""
    truth_by_index = {frame.frame_index: frame for frame in ground_truth.frames}
    errors: list[float] = []
    normalized: list[float] = []
    frame_count = 0
    for frame in predicted.frames:
        truth = truth_by_index.get(frame.frame_index)
        if truth is None:
            continue
        common = sorted(set(frame.landmarks) & set(truth.landmarks))
        if not common:
            continue
        xs = [truth.landmarks[name].x for name in common]
        ys = [truth.landmarks[name].y for name in common]
        scale = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
        for name in common:
            a, b = frame.landmarks[name], truth.landmarks[name]
            if a.visible:
                error = math.hypot(a.x - b.x, a.y - b.y)
                errors.append(error)
                normalized.append(error / scale)
        frame_count += 1
    if not errors:
        raise ValueError("no overlapping visible predictions for MPI-INF-3DHP audit")
    pck = sum(value <= 0.2 for value in normalized) / len(normalized)
    return {
        "schema": "animcv_mpi3dhp_2d_audit_v1",
        "matched_frames": frame_count,
        "matched_joints": len(errors),
        "mean_pixel_error": sum(errors) / len(errors),
        "median_pixel_error": sorted(errors)[len(errors) // 2],
        "pck_at_0_2": pck,
        "passed": pck >= 0.90,
        "gate": {"pck_at_0_2_minimum": 0.90},
    }
