import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from blender.framepose_rest_adapter import extract_target_rest_pose
from motion.animation_semantics import FootMotion, LocalArticulation, ObservationReliability, SemanticsProvenance
from motion.animation_semantics_v2 import AnimationSemanticsV2, CurrentRootOrientation, ROOT_POLICY_V2, SemanticFrameV2
from pose.contact import ContactState
from pose.contact_time_aware import RULE_VERSION
from retarget.axis_utils import quaternion_from_axis_angle, quaternion_multiply, rotate_vector_by_quaternion
from retarget.framepose_fk import FkResult, RigFkCalibration, derive_rig_alignment, solve_framepose_fk
from retarget.framepose_target_rest import RestBone, TargetRigRestPose
from retarget.framepose_target_rest import load_target_rest_pose
from retarget.framepose_fk import load_rig_fk_calibration
from rig.bone_mapping import BoneMappingEntry, BoneMappingProfile, load_bone_mapping_profile

I = ((1., 0., 0., 0.), (0., 1., 0., 0.), (0., 0., 1., 0.), (0., 0., 0., 1.))
RX90 = ((1., 0., 0., 0.), (0., 0., -1., 0.), (0., 1., 0., 0.), (0., 0., 0., 1.))
QI = (0., 0., 0., 1.)


def _rest(*, upper_direction=(0, 1, 0), upper_matrix=I):
    def bone(name, parent, head, tail, matrix=I):
        return RestBone(name, parent, matrix, head, tail)
    return TargetRigRestPose("rig", {
        "body": bone("body", None, (0, 0, 0), (0, 0, 1)),
        "up": bone("up", "body", (0, 0, 2), (0, 0, 3)),
        "right": bone("right", "body", (1, 0, 1), (1, 1, 1)),
        "upper": bone("upper", "body", (-1, 0, 1),
                      tuple((-1, 0, 1)[i] + upper_direction[i] for i in range(3)), upper_matrix),
        "lower": bone("lower", "upper", (-1, 1, 1), (-1, 2, 1)),
    }, {"authority": "blender_imported_armature", "fixture": "synthetic"})


def _calibration():
    return RigFkCalibration("rig", "body", "upper", "right", "up")


def _mapping():
    return BoneMappingProfile("rig", [
        BoneMappingEntry("upper", "landmark", ["shoulder", "elbow"], "direction"),
        BoneMappingEntry("lower", "landmark", ["elbow", "wrist"], "direction")])


def _semantics(rows):
    frames = []
    for index, (upper, lower, yaw, valid) in enumerate(rows):
        q = quaternion_from_axis_angle((0, 0, 1), yaw or 0)
        u = rotate_vector_by_quaternion(upper, q)
        l = rotate_vector_by_quaternion(lower, q)
        positions = ((0., 0., 0.), u, tuple(u[i] + l[i] for i in range(3)))
        frames.append(SemanticFrameV2(
            index * 3, index / 10, LocalArticulation(positions),
            CurrentRootOrientation(yaw is not None, yaw),
            FootMotion(ContactState.UNKNOWN, ContactState.UNKNOWN),
            ObservationReliability(valid, None)))
    return AnimationSemanticsV2(
        "synthetic", 30, "camera_root_relative", ("shoulder", "elbow", "wrist"),
        tuple(frames), SemanticsProvenance({}, {"policy": ROOT_POLICY_V2},
                                          {"rule_version": RULE_VERSION}, {}))


def _run(rows, *, rest=None):
    return solve_framepose_fk(_semantics(rows), rest or _rest(), _mapping(), _calibration())


def _frame(result, frame_index):
    return {s.target_bone: s for s in result.samples if s.frame_index == frame_index}


def test_blender_rest_snapshot_roundtrip_and_extraction(monkeypatch, tmp_path):
    from retarget.framepose_target_rest import load_target_rest_pose, save_target_rest_pose

    source = tmp_path / "rig.fbx"
    source.write_bytes(b"fbx")
    fake_bone = SimpleNamespace(name="body", parent=None, matrix_local=I,
                                head_local=(0, 0, 0), tail_local=(0, 0, 1))
    armature = SimpleNamespace(type="ARMATURE", name="Armature", data=SimpleNamespace(bones=[fake_bone]),
                               matrix_world=I)
    monkeypatch.setitem(sys.modules, "bpy", SimpleNamespace(app=SimpleNamespace(version_string="test")))
    snapshot = extract_target_rest_pose(armature, rig_id="rig", source_path=str(source))
    assert snapshot.bones["body"].parent is None
    assert snapshot.bones["body"].matrix_local == I
    assert snapshot.bones["body"].tail == (0, 0, 1)
    assert snapshot.provenance["authority"] == "blender_imported_armature"
    path = tmp_path / "rest.json"
    save_target_rest_pose(snapshot, path)
    assert load_target_rest_pose(path) == snapshot


def test_alignment_is_rig_determined_and_reusable_across_videos():
    rest = _rest()
    assert derive_rig_alignment(rest, _calibration()) == pytest.approx(QI)
    a = _run([((0, 1, 0), (0, 1, 0), 0, (True, True, True))], rest=rest)
    b = _run([((0, 1, 0), (0, 1, 0), math.pi / 2, (True, True, True))], rest=rest)
    assert a.provenance["canonical_heading_zero_to_armature_xyzw"] == b.provenance[
        "canonical_heading_zero_to_armature_xyzw"]
    assert a.provenance["rest_snapshot_digest"] == rest.digest()


def test_whole_body_yaw_changes_only_root_local_channels():
    rows = [((0, 1, 0), (0, 1, 0), yaw, (True, True, True))
            for yaw in (0, math.pi / 4, math.pi / 2)]
    result = _run(rows)
    frames = [_frame(result, index) for index in (0, 3, 6)]
    roots = [frame["body"].rotation_local for frame in frames]
    assert roots[0] == pytest.approx(QI)
    assert roots[1] != pytest.approx(roots[0]) and roots[2] != pytest.approx(roots[1])
    for bone in ("upper", "lower"):
        assert frames[0][bone].rotation_local == pytest.approx(QI)
        assert [frame[bone].rotation_local for frame in frames] == pytest.approx([QI] * 3)


def test_real_articulation_changes_upper_and_lower_locals_and_frame_zero_is_not_rest():
    rest = ((0, 1, 0), (0, 1, 0), 0, (True, True, True))
    arm_raised = ((-1, 0, 0), (0, 1, 0), 0, (True, True, True))
    elbow_bent = ((-1, 0, 0), (1, 0, 0), 0, (True, True, True))
    result = _run([arm_raised, rest, elbow_bent])
    frames = [_frame(result, index) for index in (0, 3, 6)]
    assert frames[0]["upper"].rotation_local != pytest.approx(QI)
    assert frames[1]["upper"].rotation_local == pytest.approx(QI)
    assert frames[0]["upper"].rotation_local == pytest.approx(frames[2]["upper"].rotation_local)
    assert frames[0]["lower"].rotation_local != pytest.approx(frames[2]["lower"].rotation_local)
    assert frames[0]["upper"].frame_index == 0 and frames[1]["upper"].timestamp == 0.1


def test_unavailable_evidence_and_unknown_yaw_are_explicit():
    result = _run([
        ((0, 1, 0), (0, 1, 0), 0, (True, True, True)),
        ((0, 1, 0), (0, 1, 0), 0, (True, False, True)),
        ((0, 1, 0), (0, 1, 0), None, (True, True, True)),
    ])
    first, invalid, unknown = (_frame(result, i) for i in (0, 3, 6))
    assert invalid["upper"].rotation_status == "unavailable"
    assert invalid["upper"].rotation_local is None
    assert invalid["lower"].rotation_status == "unavailable"
    assert first["upper"].rotation_local == QI
    assert unknown["body"].reason == "root_orientation_unknown"
    assert unknown["upper"].source_status == "known"
    assert unknown["upper"].raw_direction is not None
    assert unknown["upper"].rotation_status == "unavailable"
    assert unknown["upper"].heading_relative_direction is None


def test_parent_child_local_composition_and_rotated_rest_basis():
    result = _run([((-1, 0, 0), (0, 1, 0), 0, (True, True, True))])
    frame = _frame(result, 0)
    upper_world = rotate_vector_by_quaternion((0, 1, 0), frame["upper"].rotation_local)
    lower_world = rotate_vector_by_quaternion(
        rotate_vector_by_quaternion((0, 1, 0), frame["lower"].rotation_local),
        frame["upper"].rotation_local)
    assert upper_world == pytest.approx((-1, 0, 0), abs=1e-8)
    assert lower_world == pytest.approx((0, 1, 0), abs=1e-8)

    rotated = _rest(upper_direction=(0, 0, 1), upper_matrix=RX90)
    at_rest = _run([((0, 0, 1), (0, 1, 0), 0, (True, True, True))], rest=rotated)
    assert _frame(at_rest, 0)["upper"].rotation_local == pytest.approx(QI)
    moved = _run([((0, 1, 0), (0, 1, 0), 0, (True, True, True))], rest=rotated)
    q = _frame(moved, 0)["upper"].rotation_local
    in_world = rotate_vector_by_quaternion(
        rotate_vector_by_quaternion((0, 1, 0), q),
        quaternion_from_axis_angle((1, 0, 0), math.pi / 2))
    assert in_world == pytest.approx((0, 1, 0), abs=1e-8)


def test_arbitrary_rest_axis_reversal_normalization_determinism_and_zero_twist():
    rest = _rest(upper_direction=(1, 0, 0))
    row = ((-1, 0, 0), (0, 1, 0), 0, (True, True, True))
    a = _run([row], rest=rest)
    b = _run([row], rest=rest)
    assert a.to_dict() == b.to_dict()
    q = _frame(a, 0)["upper"].rotation_local
    assert sum(v * v for v in q) == pytest.approx(1)
    assert all(math.isfinite(v) for v in q)
    assert rotate_vector_by_quaternion((1, 0, 0), q) == pytest.approx((-1, 0, 0), abs=1e-8)
    # Swing's rotation axis is perpendicular to the bone's rest direction.
    assert q[0] == pytest.approx(0, abs=1e-8)
    assert a.provenance["rotation_convention"].endswith("zero axial twist")
    assert FkResult.from_dict(a.to_dict()) == a


def test_root_owner_must_be_declared_ancestor():
    calibration = RigFkCalibration("rig", "right", "upper", "right", "up")
    with pytest.raises(ValueError, match="not an ancestor"):
        solve_framepose_fk(_semantics([((0, 1, 0), (0, 1, 0), 0, (True, True, True))]),
                           _rest(), _mapping(), calibration)


def test_real_baserig_rest_yaw_invariance_and_bone_coverage():
    root = Path(__file__).resolve().parent.parent
    rest = load_target_rest_pose(root / "examples/e2e_demo/baserig_blender_rest.json")
    mapping = load_bone_mapping_profile(root / "examples/e2e_demo/mapping.json")
    calibration = load_rig_fk_calibration(root / "examples/e2e_demo/framepose_fk_calibration.json")
    assert len(rest.bones) == 51
    assert rest.bones["upperarm_l"].parent == "clavicle_l"
    assert rest.bones["lowerarm_l"].parent == "upperarm_l"
    assert rest.bones["thigh_l"].parent is None  # outside spine_01 yaw subtree
    frames = []
    for index, yaw in enumerate((0, math.pi / 4, math.pi / 2)):
        q = quaternion_from_axis_angle((0, 0, 1), yaw)
        shoulder = (0., 0., 0.)
        upper = rotate_vector_by_quaternion((0., 1., 0.), q)
        lower = rotate_vector_by_quaternion((0., 1., 0.), q)
        frames.append(SemanticFrameV2(
            index, index / 30, LocalArticulation((shoulder, upper,
                tuple(upper[i] + lower[i] for i in range(3)))),
            CurrentRootOrientation(True, yaw),
            FootMotion(ContactState.UNKNOWN, ContactState.UNKNOWN),
            ObservationReliability((True, True, True), None)))
    semantics = AnimationSemanticsV2(
        "baserig_yaw", 30, "camera_root_relative",
        ("left_shoulder", "left_elbow", "left_wrist"), tuple(frames),
        SemanticsProvenance({}, {"policy": ROOT_POLICY_V2}, {"rule_version": RULE_VERSION}, {}))
    result = solve_framepose_fk(semantics, rest, mapping, calibration)
    cases = [_frame(result, index) for index in range(3)]
    assert "thigh_l" in result.provenance["root_yaw_uncovered_bones"]
    assert cases[0]["spine_01"].rotation_local != pytest.approx(cases[1]["spine_01"].rotation_local)
    for bone in ("upperarm_l", "lowerarm_l"):
        for case in cases[1:]:
            assert case[bone].rotation_local == pytest.approx(
                cases[0][bone].rotation_local, abs=1e-8)
