import math

import numpy as np
import pytest

from pose.contact import ContactState
from pose.oracle_world_reference import WorldReferenceFrame
from pose.root_translation_control import (
    rotation_angle_degrees, run_control_experiment, summarize,
)


def _identity() -> np.ndarray:
    return np.eye(3)


def _rotation_about_z(degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    return np.array([
        [math.cos(angle), -math.sin(angle), 0.0],
        [math.sin(angle), math.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])


def _planted_world_frames(rotations: list[np.ndarray]) -> list[WorldReferenceFrame]:
    """Pelvis translates +0.1m/frame along X; left ankle stays world-fixed."""
    frames = []
    for index, rotation in enumerate(rotations):
        frames.append(WorldReferenceFrame(
            frame_index=index,
            pelvis_world=(0.1 * index, 0.0, 0.0),
            left_ankle_world=(0.3, 0.0, 0.0),
            right_ankle_world=(-0.3, 0.0, 0.0),
            valid=True,
            camera_rotation=rotation,
        ))
    return frames


def test_planted_foot_recovers_root_displacement_exactly_in_world_oracle_condition():
    world_frames = _planted_world_frames([_identity()] * 3)
    # Static camera: naive camera-relative ankle equals world-relative ankle exactly.
    camera_relative_ankle = [(0.3 - 0.1 * i, 0.0, 0.0) for i in range(3)]
    planted = [ContactState.CONTACT, ContactState.CONTACT, ContactState.CONTACT]

    results = run_control_experiment(world_frames, camera_relative_ankle, "left", planted)

    assert all(r.resolved for r in results)
    for r in results:
        assert r.world_oracle_error_m == pytest.approx(0.0, abs=1e-9)
        assert r.world_true_delta_m == pytest.approx(0.1, abs=1e-9)


def test_static_camera_naive_condition_matches_oracle():
    world_frames = _planted_world_frames([_identity()] * 3)
    camera_relative_ankle = [(0.3 - 0.1 * i, 0.0, 0.0) for i in range(3)]
    planted = [ContactState.CONTACT] * 3

    results = run_control_experiment(world_frames, camera_relative_ankle, "left", planted)

    for r in results:
        assert r.naive_camera_error_m == pytest.approx(0.0, abs=1e-9)


def test_moving_camera_contaminates_the_naive_condition_but_not_world_oracle():
    # Camera rotates 30 degrees about Z between frame 1 and frame 2 only.
    world_frames = _planted_world_frames([_identity(), _identity(), _rotation_about_z(30.0)])
    world_rel = [(0.3 - 0.1 * i, 0.0, 0.0) for i in range(3)]
    # Frame 2's camera-relative ankle is world_rel[2] rotated by the camera's
    # own 30-degree turn -- exactly what a monocular pipeline anchored to the
    # (now-rotated) camera would actually report.
    rotated_frame2 = _rotation_about_z(30.0) @ np.asarray(world_rel[2])
    camera_relative_ankle = [world_rel[0], world_rel[1], tuple(rotated_frame2)]
    planted = [ContactState.CONTACT] * 3

    results = run_control_experiment(world_frames, camera_relative_ankle, "left", planted)

    # world_oracle is immune to camera rotation by construction.
    assert results[1].world_oracle_error_m == pytest.approx(0.0, abs=1e-9)
    # naive_camera degrades exactly where the camera rotated.
    assert results[0].naive_camera_error_m == pytest.approx(0.0, abs=1e-9)
    assert results[1].naive_camera_error_m > 1e-3
    assert results[1].camera_rotation_delta_degrees == pytest.approx(30.0, abs=1e-6)


def test_unplanted_or_invalid_frames_do_not_invent_a_translation():
    world_frames = _planted_world_frames([_identity()] * 3)
    camera_relative_ankle = [(0.3 - 0.1 * i, 0.0, 0.0) for i in range(3)]
    # Frame 1's reference says the foot was actually MOVING, not planted.
    planted = [ContactState.CONTACT, ContactState.MOVING, ContactState.CONTACT]

    results = run_control_experiment(world_frames, camera_relative_ankle, "left", planted)

    assert results[0].resolved is False  # needs frames 0 AND 1 both planted
    assert results[0].world_oracle_error_m is None
    assert results[0].world_true_delta_m is None


def test_invalid_world_frame_is_unresolved_not_guessed():
    world_frames = _planted_world_frames([_identity()] * 3)
    invalid = WorldReferenceFrame(1, None, None, None, valid=False, camera_rotation=_identity())
    world_frames[1] = invalid
    camera_relative_ankle = [(0.3, 0.0, 0.0)] * 3
    planted = [ContactState.CONTACT] * 3

    results = run_control_experiment(world_frames, camera_relative_ankle, "left", planted)

    assert results[0].resolved is False


def test_same_sequence_temporal_accounting_in_summary():
    world_frames = _planted_world_frames([_identity()] * 4)
    camera_relative_ankle = [(0.3 - 0.1 * i, 0.0, 0.0) for i in range(4)]
    planted = [ContactState.CONTACT, ContactState.CONTACT, ContactState.MOVING, ContactState.CONTACT]

    results = run_control_experiment(world_frames, camera_relative_ankle, "left", planted)
    report = summarize(results)

    assert report["frame_count"] == 3  # len(world_frames) - 1
    assert report["recoverable_segment_count"] + report["unresolved_segment_count"] == report["frame_count"]


def test_rotation_angle_degrees_is_zero_for_identical_rotations():
    assert rotation_angle_degrees(_identity(), _identity()) == pytest.approx(0.0, abs=1e-9)


def test_run_control_experiment_is_deterministic_on_replay():
    world_frames = _planted_world_frames([_identity()] * 3)
    camera_relative_ankle = [(0.3 - 0.1 * i, 0.0, 0.0) for i in range(3)]
    planted = [ContactState.CONTACT] * 3

    first = run_control_experiment(world_frames, camera_relative_ankle, "left", planted)
    second = run_control_experiment(world_frames, camera_relative_ankle, "left", planted)

    assert first == second


def test_mismatched_lengths_are_rejected():
    world_frames = _planted_world_frames([_identity()] * 3)
    with pytest.raises(ValueError):
        run_control_experiment(world_frames, [(0.0, 0.0, 0.0)] * 2, "left", [ContactState.CONTACT] * 3)
