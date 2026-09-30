import math
import sys
from types import SimpleNamespace

import pytest

from blender.framepose_object_owner_adapter import ImportedArmatureBinding, apply_object_owned_frame
from motion.animation_semantics import FootMotion, LocalArticulation, ObservationReliability, SemanticsProvenance
from motion.animation_semantics_v2 import AnimationSemanticsV2, CurrentRootOrientation, ROOT_POLICY_V2, SemanticFrameV2
from pose.contact import ContactState
from pose.contact_time_aware import RULE_VERSION
from retarget.axis_utils import quaternion_from_axis_angle, rotate_vector_by_quaternion
from retarget.framepose_fk import RigFkCalibration, solve_framepose_fk
from retarget.framepose_object_owner import (
    ObjectOwnedFkResult, RigRotationSample, solve_object_owned_fk,
)
from retarget.framepose_target_rest import RestBone, TargetRigRestPose
from rig.bone_mapping import BoneMappingEntry, BoneMappingProfile

I = ((1., 0., 0., 0.), (0., 1., 0., 0.), (0., 0., 1., 0.), (0., 0., 0., 1.))


def _setup():
    def bone(name, parent, head, tail):
        return RestBone(name, parent, I, head, tail)
    rest = TargetRigRestPose("multi_root", {
        "body": bone("body", None, (0, 0, 0), (0, 0, 1)),
        "up": bone("up", "body", (0, 0, 2), (0, 0, 3)),
        "right": bone("right", "body", (1, 0, 1), (1, 1, 1)),
        "upper": bone("upper", "body", (-1, 0, 1), (-1, 1, 1)),
        "thigh": bone("thigh", None, (0, 0, -1), (0, 0, -2)),
        "calf": bone("calf", "thigh", (0, 0, -2), (0, 0, -3)),
    }, {"authority": "blender_imported_armature", "fixture": "two independent roots"})
    calibration = RigFkCalibration("multi_root", "body", "upper", "right", "up")
    mapping = BoneMappingProfile("multi_root", [
        BoneMappingEntry("upper", "landmark", ["shoulder", "elbow"], "direction"),
        BoneMappingEntry("calf", "landmark", ["hip", "knee"], "direction"),
    ])
    return rest, mapping, calibration


def _semantics(rows):
    frames = []
    for index, (yaw, arm, leg, valid) in enumerate(rows):
        q = quaternion_from_axis_angle((0, 0, 1), yaw or 0)
        arm_camera = rotate_vector_by_quaternion(arm, q)
        leg_camera = rotate_vector_by_quaternion(leg, q)
        frames.append(SemanticFrameV2(
            index * 3, index / 10,
            LocalArticulation(((0, 0, 0), arm_camera, (0, 0, 0), leg_camera)),
            CurrentRootOrientation(yaw is not None, yaw),
            FootMotion(ContactState.UNKNOWN, ContactState.UNKNOWN),
            ObservationReliability(valid, None)))
    return AnimationSemanticsV2(
        "two_roots", 30, "camera_root_relative",
        ("shoulder", "elbow", "hip", "knee"), tuple(frames),
        SemanticsProvenance({}, {"policy": ROOT_POLICY_V2},
                            {"rule_version": RULE_VERSION}, {}))


def _run(rows):
    rest, mapping, calibration = _setup()
    semantics = _semantics(rows)
    return solve_object_owned_fk(semantics, rest, mapping, calibration)


def _at(result, frame_index):
    return {sample.target_bone: sample for sample in result.pose_bone_samples
            if sample.frame_index == frame_index}


def test_two_independent_roots_inherit_one_object_yaw_without_local_leak():
    arm, leg = (0, 1, 0), (0.2, 0, -1)
    rows = [(yaw, arm, leg, (True,) * 4) for yaw in (0, math.pi / 4, math.pi / 2)]
    result = _run(rows)
    assert result.provenance["imported_top_level_bones"] == ["body", "thigh"]
    assert result.provenance["root_owner"] == "Blender Armature Object"
    assert {sample.target_bone for sample in result.pose_bone_samples} == {"upper", "calf"}
    for bone in ("upper", "calf"):
        first = _at(result, 0)[bone].rotation_local
        for frame_index in (3, 6):
            assert _at(result, frame_index)[bone].rotation_local == pytest.approx(first, abs=1e-8)
    object_quats = [s.rotation_armature_space for s in result.rig_rotations]
    assert object_quats[0] == pytest.approx((0, 0, 0, 1))
    for q, yaw in zip(object_quats, (0, math.pi / 4, math.pi / 2)):
        # Both skeletal roots inherit the same object transform; neither has
        # an independent yaw pose-bone sample.
        for _branch in ("body", "thigh"):
            assert rotate_vector_by_quaternion((1, 0, 0), q) == pytest.approx(
                (math.cos(yaw), math.sin(yaw), 0), abs=1e-8)
    assert result.provenance["root_translation"] == "unavailable"


def test_genuine_articulation_still_changes_each_branch_local_fk():
    rows = [(0, (0, 1, 0), (0, 0, -1), (True,) * 4),
            (0, (-1, 0, 0), (0.2, 0, -1), (True,) * 4)]
    result = _run(rows)
    assert _at(result, 0)["upper"].rotation_local != pytest.approx(_at(result, 3)["upper"].rotation_local)
    assert _at(result, 0)["calf"].rotation_local != pytest.approx(_at(result, 3)["calf"].rotation_local)
    assert result.rig_rotations[0].rotation_armature_space == result.rig_rotations[1].rotation_armature_space


def test_baserig_style_arm_fk_samples_remain_exactly_the_worklog59_values():
    rest, mapping, calibration = _setup()
    semantics = _semantics([(math.pi / 4, (0, 1, 0), (0, 0, -1), (True,) * 4)])
    bone_owned_arm_only = solve_framepose_fk(
        semantics, rest, BoneMappingProfile("multi_root", [mapping.entries[0]]), calibration)
    object_owned = solve_object_owned_fk(
        semantics, rest, BoneMappingProfile("multi_root", [mapping.entries[0]]), calibration)
    assert object_owned.pose_bone_samples[0].to_dict() == next(
        s.to_dict() for s in bone_owned_arm_only.samples if s.target_bone == "upper")
    assert all(s.target_bone != "body" for s in object_owned.pose_bone_samples)


def test_unknown_yaw_is_unavailable_and_review_rejects_before_mutation(monkeypatch):
    result = _run([(None, (0, 1, 0), (0, 0, -1), (True,) * 4)])
    assert result.rig_rotations[0].root_orientation_status == "unknown"
    assert result.rig_rotations[0].rotation_armature_space is None
    assert all(s.rotation_status == "unavailable" for s in result.pose_bone_samples)
    assert ObjectOwnedFkResult.from_dict(result.to_dict()) == result
    fake = SimpleNamespace(location=[0, 0, 0], scale=[1, 1, 1])
    monkeypatch.setitem(sys.modules, "bpy", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "mathutils", SimpleNamespace(Quaternion=object))
    binding = ImportedArmatureBinding(result.provenance["rest_snapshot_digest"],
                                      (0, 0, 0), (1, 1, 1), (1, 0, 0, 0), ("body", "thigh"))
    with pytest.raises(ValueError, match="Root Orientation unavailable"):
        apply_object_owned_frame(fake, binding, result, 0)
    assert fake.location == [0, 0, 0]


def test_invalid_leg_source_does_not_hold_a_previous_rotation():
    rows = [(0, (0, 1, 0), (0, 0, -1), (True,) * 4),
            (0, (0, 1, 0), (0, 0, -1), (True, True, True, False))]
    result = _run(rows)
    assert _at(result, 0)["calf"].rotation_status == "known"
    assert _at(result, 3)["calf"].rotation_status == "unavailable"
    assert _at(result, 3)["calf"].rotation_local is None
    assert result.rig_rotations[1].rotation_status == "known"
