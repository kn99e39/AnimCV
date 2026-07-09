import math

import pytest

from optimize.collapse import (
    CollapseThresholds,
    collapse_animation_clip,
    collapse_track,
    thresholds_for_preset,
)
from optimize.report import KeyframeCandidate, KeyframeOptimizationReport
from retarget.axis_utils import quaternion_from_axis_angle
from retarget.solver import AnimationClip, AnimationTrack, BoneTransformSample


def _sample_candidate() -> KeyframeCandidate:
    transform = BoneTransformSample(
        frame_index=12,
        bone_name="spine_01",
        location=None,
        rotation=(0.0, 0.0, 0.0, 1.0),
        scale=None,
        confidence=0.75,
    )
    return KeyframeCandidate(
        frame_index=12,
        bone_name="spine_01",
        transform=transform,
        importance=0.82,
        locked=False,
        reason="angular_velocity",
    )


def test_keyframe_candidate_roundtrip():
    candidate = _sample_candidate()

    restored = KeyframeCandidate.from_dict(candidate.to_dict())

    assert restored == candidate


def test_keyframe_optimization_report_roundtrip():
    report = KeyframeOptimizationReport(
        original_key_count=120,
        optimized_key_count=18,
        removed_key_count=102,
        max_error=2.4,
        threshold=3.0,
    )

    restored = KeyframeOptimizationReport.from_dict(report.to_dict())

    assert restored == report


def test_thresholds_for_preset_none_disables_collapse():
    assert thresholds_for_preset("none") is None


def test_thresholds_for_preset_known_presets():
    assert thresholds_for_preset("light") == CollapseThresholds(1.0, 2.0)
    assert thresholds_for_preset("medium") == CollapseThresholds(3.0, 8.0)
    assert thresholds_for_preset("aggressive") == CollapseThresholds(8.0, 20.0)


def test_thresholds_for_preset_custom_requires_a_value():
    with pytest.raises(ValueError):
        thresholds_for_preset("custom")

    assert thresholds_for_preset("custom", 5.0) == CollapseThresholds(5.0, 5.0)


def test_thresholds_for_preset_unknown_raises():
    with pytest.raises(ValueError):
        thresholds_for_preset("extreme")


def _rotation_sample(frame_index, angle_deg):
    rotation = quaternion_from_axis_angle((0.0, 0.0, 1.0), math.radians(angle_deg))
    return BoneTransformSample(
        frame_index=frame_index,
        bone_name="upper_arm.L",
        location=None,
        rotation=rotation,
        scale=None,
        confidence=0.9,
    )


def test_collapse_track_reduces_a_constant_velocity_rotation_sweep_to_two_keys():
    # Constant angular velocity around a single axis slerps back to an
    # exact match at every intermediate frame, so RDP should find zero
    # interpolation error everywhere and drop every frame but the ends.
    samples = [_rotation_sample(i, angle_deg=i * 9) for i in range(11)]
    track = AnimationTrack(bone_name="upper_arm.L", samples=samples)

    optimized_track, report = collapse_track(track, thresholds_for_preset("medium"))

    assert report.original_key_count == 11
    assert report.optimized_key_count == 2
    assert report.removed_key_count == 9
    assert [s.frame_index for s in optimized_track.samples] == [0, 10]


def test_collapse_track_keeps_a_sharp_spike_frame():
    angles = [0, 1, 2, 60, 3, 4, 5]
    samples = [_rotation_sample(i, angle_deg=a) for i, a in enumerate(angles)]
    track = AnimationTrack(bone_name="upper_arm.L", samples=samples)

    optimized_track, report = collapse_track(track, thresholds_for_preset("light"))

    kept_frames = {s.frame_index for s in optimized_track.samples}
    assert 3 in kept_frames
    assert report.optimized_key_count < report.original_key_count


def test_collapse_track_never_removes_locked_frames():
    samples = [_rotation_sample(i, angle_deg=i * 9) for i in range(11)]
    track = AnimationTrack(bone_name="upper_arm.L", samples=samples)

    optimized_track, report = collapse_track(
        track, thresholds_for_preset("medium"), locked_frames={5}
    )

    kept_frames = {s.frame_index for s in optimized_track.samples}
    assert 5 in kept_frames
    assert 0 in kept_frames and 10 in kept_frames


def test_collapse_track_with_none_threshold_is_a_no_op():
    samples = [_rotation_sample(i, angle_deg=i * 9) for i in range(11)]
    track = AnimationTrack(bone_name="upper_arm.L", samples=samples)

    optimized_track, report = collapse_track(track, None)

    assert len(optimized_track.samples) == 11
    assert report.removed_key_count == 0


def test_collapse_track_short_track_is_a_no_op():
    samples = [_rotation_sample(0, 0), _rotation_sample(1, 10)]
    track = AnimationTrack(bone_name="upper_arm.L", samples=samples)

    optimized_track, report = collapse_track(track, thresholds_for_preset("aggressive"))

    assert len(optimized_track.samples) == 2
    assert report.removed_key_count == 0


def _location_sample(frame_index, x, y):
    return BoneTransformSample(
        frame_index=frame_index,
        bone_name="hand.L",
        location=(x, y, 0.0),
        rotation=(0.0, 0.0, 0.0, 1.0),
        scale=None,
        confidence=0.9,
    )


def test_collapse_track_reduces_a_constant_velocity_location_sweep_to_two_keys():
    samples = [_location_sample(i, x=float(i), y=0.0) for i in range(8)]
    track = AnimationTrack(bone_name="hand.L", samples=samples)

    optimized_track, report = collapse_track(track, thresholds_for_preset("medium"))

    assert report.optimized_key_count == 2
    assert [s.frame_index for s in optimized_track.samples] == [0, 7]


def test_collapse_animation_clip_reduces_total_keys_and_reports_per_bone():
    rotation_track = AnimationTrack(
        bone_name="upper_arm.L", samples=[_rotation_sample(i, angle_deg=i * 9) for i in range(11)]
    )
    location_track = AnimationTrack(
        bone_name="hand.L", samples=[_location_sample(i, x=float(i), y=0.0) for i in range(8)]
    )
    clip = AnimationClip(
        name="Generated_Motion",
        fps=24.0,
        tracks={"upper_arm.L": rotation_track, "hand.L": location_track},
        frame_start=0,
        frame_end=10,
    )

    optimized_clip, reports = collapse_animation_clip(clip, preset="medium")

    assert set(reports) == {"upper_arm.L", "hand.L"}
    assert len(optimized_clip.tracks["upper_arm.L"].samples) < 11
    assert len(optimized_clip.tracks["hand.L"].samples) < 8


def test_collapse_animation_clip_none_preset_keeps_every_frame():
    rotation_track = AnimationTrack(
        bone_name="upper_arm.L", samples=[_rotation_sample(i, angle_deg=i * 9) for i in range(11)]
    )
    clip = AnimationClip(
        name="Generated_Motion", fps=24.0, tracks={"upper_arm.L": rotation_track},
        frame_start=0, frame_end=10,
    )

    optimized_clip, reports = collapse_animation_clip(clip, preset="none")

    assert len(optimized_clip.tracks["upper_arm.L"].samples) == 11
    assert reports["upper_arm.L"].removed_key_count == 0
