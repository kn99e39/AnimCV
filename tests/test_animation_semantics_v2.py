import math

import numpy as np
import pytest

from motion.animation_semantics import AnimationSemantics, RootOrientation
from motion.animation_semantics_v2 import (
    AnimationSemanticsV2, CurrentRootOrientation, ROOT_POLICY_V2, TimeAwareCalibration,
)
from motion.animation_semantics_v2_bridge import (
    build_animation_semantics_v2, calibrate_time_aware_contact_from_h0_train, current_root_orientation,
)
from pose.contact import ContactState
from pose.contact_time_aware import (
    TimeAwareThresholds, classify_time_aware_contact, elapsed_times, kinematics,
)
from pose.pose_lifter import LiftedPoseFrame, LiftedPosePoint, LiftedPoseSequence
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
