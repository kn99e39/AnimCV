import json
import math

import numpy as np
import pytest

from motion.animation_semantics import AnimationSemantics, RootOrientation
from motion.animation_semantics_v2 import (
    AnimationSemanticsV2, CurrentRootOrientation, LEGACY_ROOT_POLICY_V2, ROOT_POLICY_V2,
    TimeAwareCalibration, load_animation_semantics_v2,
)
from motion.animation_semantics_v2_bridge import (
    build_animation_semantics_v2, calibrate_time_aware_contact_from_h0_train, current_root_orientation,
    valid_bilateral_sources,
)
from pose.contact import ContactState
from pose.contact_time_aware import (
    TimeAwareThresholds, classify_time_aware_contact, elapsed_times, kinematics,
)
from pose.pose_lifter import LiftedPoseFrame, LiftedPosePoint, LiftedPoseSequence
from pose.root_orientation_diagnostic import observe_yaw
from test_animation_semantics import TRAIN, VALID, _bank_and_h0, _identity, _rows


def _sequence(indices, positions, *, timestamps=None, fps=30.0):
    frames = [LiftedPoseFrame(index, None if timestamps is None else timestamps[i], {
        "left_ankle": LiftedPosePoint("left_ankle", position, 0.9, 0.1),
    }) for i, (index, position) in enumerate(zip(indices, positions))]
    return LiftedPoseSequence(frames=frames, source_fps=fps)


def test_elapsed_time_central_and_one_sided_with_unequal_spacing():
    sequence = _sequence([0, 2, 7], [(0, 0, 0), (0.2, 0, 0), (0.7, 0, 0)],
                         timestamps=[0, 0.1, 0.35])
    speeds, _ = kinematics(sequence, "left", 0.4)
    assert speeds == pytest.approx([2.0, 2.0, 2.0])
    assert elapsed_times(sequence) == [0, 0.1, 0.35]
    fallback = _sequence([0, 2, 7], [(0, 0, 0), (0.2, 0, 0), (0.7, 0, 0)])
    speeds, _ = kinematics(fallback, "left", 0.4)
    assert speeds == pytest.approx([3.0, 3.0, 3.0])
    with pytest.raises(ValueError):
        elapsed_times(_sequence([0, 2], [(0, 0, 0)] * 2, timestamps=[0.1, 0.1]))


def test_height_window_and_run_are_physical_duration():
    sequence = _sequence([0, 2, 5, 8], [(0, 0, 0), (0, 0, 1), (0, 0, 0), (0, 0, 0)])
    _, height = kinematics(sequence, "left", 4 / 30)
    assert height[0] == pytest.approx(0.5)  # frame 5 is outside the 4-frame horizon
    thresholds = TimeAwareThresholds(0, 1, 2, 4 / 30, 2 / 30)
    states = classify_time_aware_contact(sequence, "left", thresholds)
    assert states[0] is ContactState.UNKNOWN  # a singleton cannot satisfy the two-TRAIN-row run
    stationary = _sequence([0, 1, 2, 3], [(0, 0, 0)] * 4)
    assert classify_time_aware_contact(stationary, "left", thresholds) == [ContactState.CONTACT] * 4


def test_current_yaw_circular_wrap_invalid_geometry_and_no_carry():
    def torso(index, angle, valid=True):
        c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        points = {}
        for label in ("shoulder", "hip"):
            for side, sign in (("left", -1), ("right", 1)):
                points[f"{side}_{label}"] = LiftedPosePoint(
                    f"{side}_{label}", (sign * c, sign * s, 0), 0.9, 0.1,
                    observation_valid=valid)
        return LiftedPoseFrame(index, index / 30, points)
    sequence = LiftedPoseSequence(frames=[torso(0, 179), LiftedPoseFrame(3, 0.1, {}), torso(6, -179),
                                          torso(9, 0, valid=False)],
                                  source_fps=30)
    orientations = current_root_orientation(sequence)
    assert orientations[0].known and not orientations[1].known and orientations[2].known
    assert not orientations[3].known
    assert abs((math.degrees(orientations[2].yaw_radians - orientations[0].yaw_radians) + 180) % 360 - 180) == pytest.approx(2)
    assert CurrentRootOrientation.from_dict(orientations[1].to_dict()) == orientations[1]
    with pytest.raises(ValueError):
        CurrentRootOrientation.from_dict({"status": "known", "yaw_radians": 0.0, "yaw_held": False})


def _torso_frame(index, shoulder_angle, hip_angle, *, shoulders_valid=True, hips_valid=True,
                 invalid_confidence=0.9, invalid_width=0.2):
    points = {}
    for label, angle, valid, width in (("shoulder", shoulder_angle, shoulders_valid, 0.18),
                                       ("hip", hip_angle, hips_valid, invalid_width)):
        c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        for side, sign in (("left", -1), ("right", 1)):
            points[f"{side}_{label}"] = LiftedPosePoint(
                f"{side}_{label}", (sign * width * c, sign * width * s, 0),
                0.9 if valid else invalid_confidence, 0.1, observation_valid=valid)
    return LiftedPoseFrame(index, index / 30.0, points)


def test_valid_only_fusion_all_source_combinations_and_diagnostic_parity():
    frames = [_torso_frame(0, 170, -170),
              _torso_frame(1, 40, -100, hips_valid=False),
              _torso_frame(2, 40, -100, shoulders_valid=False),
              _torso_frame(3, 40, -100, shoulders_valid=False, hips_valid=False)]
    sequence = LiftedPoseSequence(frames=frames, source_fps=30)
    current = current_root_orientation(sequence)
    diagnostic = observe_yaw(sequence)
    assert valid_bilateral_sources(frames[0]) == ("shoulders", "hips")
    assert current[0].yaw_radians == diagnostic[0].yaw_radians
    assert abs(abs(math.degrees(current[0].yaw_radians)) - 180) < 2
    assert valid_bilateral_sources(frames[1]) == ("shoulders",)
    assert math.degrees(current[1].yaw_radians) == pytest.approx(40)
    assert valid_bilateral_sources(frames[2]) == ("hips",)
    assert math.degrees(current[2].yaw_radians) == pytest.approx(-100)
    assert valid_bilateral_sources(frames[3]) == ()
    assert current[3] == CurrentRootOrientation(False)  # no previous-frame carry


def test_invalid_source_geometry_and_confidence_cannot_change_yaw():
    first = _torso_frame(0, 25, -170, hips_valid=False, invalid_confidence=0.01, invalid_width=0.01)
    changed = _torso_frame(0, 25, 115, hips_valid=False, invalid_confidence=1000, invalid_width=100)
    a = current_root_orientation(LiftedPoseSequence(frames=[first], source_fps=30))[0]
    b = current_root_orientation(LiftedPoseSequence(frames=[changed], source_fps=30))[0]
    assert a == b
    assert math.degrees(a.yaw_radians) == pytest.approx(25)
    assert observe_yaw(LiftedPoseSequence(frames=[first], source_fps=30))[0].yaw_radians != (
        observe_yaw(LiftedPoseSequence(frames=[changed], source_fps=30))[0].yaw_radians)


def test_train_only_calibration_and_v2_bridge_contract(tmp_path):
    bank, h0 = _bank_and_h0()
    identity = _identity(bank)
    calibration = calibrate_time_aware_contact_from_h0_train(bank, h0, identity)
    assert calibration.source["split"] == "train"
    assert calibration.sampling["height_radius_seconds"] == pytest.approx(2 / 30)
    assert calibration.sampling["min_run_span_seconds"] == pytest.approx(1 / 30)
    assert TimeAwareCalibration.from_dict(calibration.to_dict()) == calibration
    before = build_animation_semantics_v2(bank, h0, identity, VALID[0], calibration)
    assert before.provenance.root_orientation["policy"] == ROOT_POLICY_V2
    assert before.to_dict()["schema"] == "animcv_animation_semantics_v2"
    assert [frame.timestamp for frame in before.frames] == [bank.samples[i].timestamp for i in _rows(bank, VALID[0])]
    assert all(frame.root_translation.to_dict() == {"status": "unavailable"} and
               frame.ground_height.to_dict() == {"status": "unavailable"} for frame in before.frames)
    assert AnimationSemanticsV2.from_dict(before.to_dict()) == before
    legacy_payload = json.loads(json.dumps(before.to_dict()))
    legacy_payload["provenance"]["root_orientation"]["policy"] = LEGACY_ROOT_POLICY_V2
    with pytest.raises(ValueError):
        AnimationSemanticsV2.from_dict(legacy_payload)
    legacy_path = tmp_path / "old_v2.semantics.json"
    legacy_path.write_text(json.dumps(legacy_payload))
    legacy = load_animation_semantics_v2(legacy_path, allow_legacy_policy=True)
    assert legacy.to_dict() == legacy_payload
    with pytest.raises(ValueError):
        load_animation_semantics_v2(legacy_path)
    with pytest.raises(ValueError):
        AnimationSemantics.from_dict(before.to_dict())
    old = RootOrientation(True, 0.0, 0.9, True)
    assert RootOrientation.from_dict(old.to_dict()) == old  # v1 held meaning remains intact
    bank.arrays["target_3d"][:] = 9e6
    bank.arrays["target_valid"][:] = False
    assert build_animation_semantics_v2(bank, h0, identity, VALID[0], calibration).to_dict() == before.to_dict()
    other = _rows(bank, VALID[1])
    h0[other] *= 100
    assert build_animation_semantics_v2(bank, h0, identity, VALID[0], calibration).frames == before.frames
    assert calibrate_time_aware_contact_from_h0_train(bank, h0, identity).thresholds == calibration.thresholds
    assert build_animation_semantics_v2(bank, h0, identity, VALID[0], calibration).content_digest() == before.content_digest()
