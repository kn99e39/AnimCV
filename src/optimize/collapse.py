"""Keyframe collapse / curve simplification (Architecture_v2.md section 8.3).

An RDP-style (Ramer-Douglas-Peucker) simplification: the first/last
sample of every track, any locked frame, and any frame whose
``importance.score_track_importance`` result is at least
``importance.MIN_IMPORTANCE_TO_KEEP`` are kept unconditionally. Between
two kept anchors, this recursively finds the point of maximum
interpolation error and keeps it too if the error exceeds the preset's
threshold, else drops everything in between.

Rotation-mapped bones (direction-mode mappings, section 7.2) are
simplified by quaternion-slerp angular error in degrees.
Location-mapped bones (landmark/point-mode mappings) are simplified by
linear-interpolation Euclidean distance. Milestone 5's fk_solver emits
raw pixel offsets for location-mapped bones, not normalized scene
units, so the position thresholds below are pixel-scale rather than the
doc's illustrative ~0.02 (which assumes an already-normalized
coordinate space this MVP doesn't have).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from common.types import Quaternion, Vec3
from optimize.importance import MIN_IMPORTANCE_TO_KEEP, score_track_importance
from optimize.report import KeyframeOptimizationReport
from retarget.solver import AnimationClip, AnimationTrack, BoneTransformSample

_PRESET_THRESHOLDS = {
    "light": {"max_rotation_error_deg": 1.0, "max_position_error_px": 2.0},
    "medium": {"max_rotation_error_deg": 3.0, "max_position_error_px": 8.0},
    "aggressive": {"max_rotation_error_deg": 8.0, "max_position_error_px": 20.0},
}


@dataclass
class CollapseThresholds:
    max_rotation_error_deg: float
    max_position_error_px: float


def thresholds_for_preset(
    preset: str, custom_threshold: float | None = None
) -> CollapseThresholds | None:
    """Returns None for preset == "none" (no collapse at all)."""
    if preset == "none":
        return None
    if preset == "custom":
        if custom_threshold is None:
            raise ValueError("collapse preset 'custom' requires an explicit threshold")
        return CollapseThresholds(
            max_rotation_error_deg=custom_threshold, max_position_error_px=custom_threshold
        )
    if preset not in _PRESET_THRESHOLDS:
        raise ValueError(
            f"unknown collapse preset {preset!r}; expected one of "
            f"{sorted(_PRESET_THRESHOLDS)} + ['none', 'custom']"
        )
    return CollapseThresholds(**_PRESET_THRESHOLDS[preset])


def _quaternion_angle_deg(a: Quaternion, b: Quaternion) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    dot = max(-1.0, min(1.0, abs(dot)))
    return math.degrees(2.0 * math.acos(dot))


def _slerp(a: Quaternion, b: Quaternion, t: float) -> Quaternion:
    dot = sum(x * y for x, y in zip(a, b))
    if dot < 0.0:
        b = tuple(-c for c in b)
        dot = -dot
    dot = max(-1.0, min(1.0, dot))
    if dot > 0.9995:
        result = tuple(x + t * (y - x) for x, y in zip(a, b))
        length = math.sqrt(sum(c * c for c in result)) or 1.0
        return tuple(c / length for c in result)
    theta_0 = math.acos(dot)
    theta = theta_0 * t
    sin_theta_0 = math.sin(theta_0)
    s0 = math.cos(theta) - dot * math.sin(theta) / sin_theta_0
    s1 = math.sin(theta) / sin_theta_0
    return tuple(s0 * x + s1 * y for x, y in zip(a, b))


def _lerp_location(a: Vec3, b: Vec3, t: float) -> Vec3:
    return tuple(x + t * (y - x) for x, y in zip(a, b))


def _track_uses_location(samples: list[BoneTransformSample]) -> bool:
    return samples[0].location is not None


def _interpolation_error(
    anchor_a: BoneTransformSample,
    anchor_b: BoneTransformSample,
    actual: BoneTransformSample,
    t: float,
    use_location: bool,
) -> float:
    if use_location:
        predicted = _lerp_location(anchor_a.location, anchor_b.location, t)
        return math.dist(predicted, actual.location)
    predicted = _slerp(anchor_a.rotation, anchor_b.rotation, t)
    return _quaternion_angle_deg(predicted, actual.rotation)


def _rdp(
    samples: list[BoneTransformSample],
    start: int,
    end: int,
    use_location: bool,
    threshold: float,
    keep: set[int],
) -> None:
    if end - start <= 1:
        return

    anchor_a, anchor_b = samples[start], samples[end]
    span = end - start

    max_error = -1.0
    max_error_index = -1
    for i in range(start + 1, end):
        t = (i - start) / span
        error = _interpolation_error(anchor_a, anchor_b, samples[i], t, use_location)
        if error > max_error:
            max_error = error
            max_error_index = i

    if max_error > threshold:
        keep.add(max_error_index)
        _rdp(samples, start, max_error_index, use_location, threshold, keep)
        _rdp(samples, max_error_index, end, use_location, threshold, keep)


def collapse_track(
    track: AnimationTrack,
    thresholds: CollapseThresholds | None,
    locked_frames: set[int] | None = None,
) -> tuple[AnimationTrack, KeyframeOptimizationReport]:
    samples = track.samples
    original_count = len(samples)
    locked_frames = locked_frames or set()

    if thresholds is None or original_count <= 2:
        report = KeyframeOptimizationReport(
            original_key_count=original_count,
            optimized_key_count=original_count,
            removed_key_count=0,
            max_error=0.0,
            threshold=0.0,
        )
        return AnimationTrack(bone_name=track.bone_name, samples=list(samples)), report

    use_location = _track_uses_location(samples)
    threshold = (
        thresholds.max_position_error_px if use_location else thresholds.max_rotation_error_deg
    )

    keep: set[int] = {0, original_count - 1}
    for i, sample in enumerate(samples):
        if sample.frame_index in locked_frames:
            keep.add(i)

    importance_scores = score_track_importance(track, locked_frames)
    for i, score in enumerate(importance_scores):
        if score >= MIN_IMPORTANCE_TO_KEEP:
            keep.add(i)

    kept_sorted = sorted(keep)
    for a, b in zip(kept_sorted, kept_sorted[1:]):
        _rdp(samples, a, b, use_location, threshold, keep)

    kept_indices = sorted(keep)
    max_observed_error = 0.0
    for a, b in zip(kept_indices, kept_indices[1:]):
        span = b - a
        if span <= 1:
            continue
        for i in range(a + 1, b):
            t = (i - a) / span
            error = _interpolation_error(samples[a], samples[b], samples[i], t, use_location)
            max_observed_error = max(max_observed_error, error)

    optimized_samples = [samples[i] for i in kept_indices]
    report = KeyframeOptimizationReport(
        original_key_count=original_count,
        optimized_key_count=len(optimized_samples),
        removed_key_count=original_count - len(optimized_samples),
        max_error=max_observed_error,
        threshold=threshold,
    )
    return AnimationTrack(bone_name=track.bone_name, samples=optimized_samples), report


def collapse_animation_clip(
    clip: AnimationClip,
    preset: str = "medium",
    custom_threshold: float | None = None,
    locked_frames: set[int] | None = None,
) -> tuple[AnimationClip, dict[str, KeyframeOptimizationReport]]:
    thresholds = thresholds_for_preset(preset, custom_threshold)

    tracks: dict[str, AnimationTrack] = {}
    reports: dict[str, KeyframeOptimizationReport] = {}
    for bone_name, track in clip.tracks.items():
        optimized_track, report = collapse_track(track, thresholds, locked_frames)
        tracks[bone_name] = optimized_track
        reports[bone_name] = report

    optimized_clip = AnimationClip(
        name=clip.name,
        fps=clip.fps,
        tracks=tracks,
        frame_start=clip.frame_start,
        frame_end=clip.frame_end,
    )
    return optimized_clip, reports
