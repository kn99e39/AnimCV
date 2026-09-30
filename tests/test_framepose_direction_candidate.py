import math

import pytest

from motion.animation_semantics import FootMotion, LocalArticulation, ObservationReliability, SemanticsProvenance
from motion.animation_semantics_v2 import AnimationSemanticsV2, CurrentRootOrientation, ROOT_POLICY_V2, SemanticFrameV2
from pose.contact import ContactState
from pose.contact_time_aware import RULE_VERSION
from retarget.axis_utils import quaternion_from_vectors, rotate_vector_by_quaternion
from retarget.framepose_direction_candidate import (
    DirectionCandidate, audit_direction_mapping, load_direction_candidate,
    save_direction_candidate, solve_framepose_direction_candidate, target_rest_axis,
)
from rig.bone_mapping import BoneMappingEntry, BoneMappingProfile
from rig.rig_profile import BoneInfo, RigProfile

IDENTITY = ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0))
ROT_X_90 = ((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, -1.0, 0.0),
            (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
QI = (0.0, 0.0, 0.0, 1.0)


def _semantics(directions, *, invalid=(), yaws=None):
    frames = []
    for i, direction in enumerate(directions):
        valid = (True, i not in invalid)
        frames.append(SemanticFrameV2(
            frame_index=i * 3, timestamp=i / 10,
            articulation=LocalArticulation(((0.0, 0.0, 0.0), tuple(direction))),
            root_orientation=CurrentRootOrientation(True, (yaws or [0.0] * len(directions))[i]),
            foot_motion=FootMotion(ContactState.UNKNOWN, ContactState.UNKNOWN),
            reliability=ObservationReliability(valid, None)))
    return AnimationSemanticsV2(
        "synthetic", 30.0, "camera_root_relative", ("shoulder", "elbow"), tuple(frames),
        SemanticsProvenance({}, {"policy": ROOT_POLICY_V2}, {"rule_version": RULE_VERSION}, {}))


def _rig(*, axis=None, rest=IDENTITY, child_offset=(0, 1, 0)):
    child_matrix = tuple(tuple(float(1 if r == c else child_offset[r] if c == 3 and r < 3 else 0)
                               for c in range(4)) for r in range(4))
    upper = BoneInfo("upper", None, ["end"], rest_local_matrix=rest,
                     local_axis_hint={"primary": axis} if axis is not None else None)
    end = BoneInfo("end", "upper", [], rest_local_matrix=child_matrix)
    return RigProfile("test_rig", "synthetic", {"upper": upper, "end": end}, "upper")


def _mapping(axis_hint="+Z"):
    return BoneMappingProfile("test_rig", [BoneMappingEntry(
        "upper", "landmark", ["shoulder", "elbow"], "direction", axis_hint=axis_hint)])


def _run(directions, *, rig=None, mapping=None, invalid=(), alignment=QI, yaws=None):
    return solve_framepose_direction_candidate(
        _semantics(directions, invalid=invalid, yaws=yaws), rig or _rig(), mapping or _mapping(),
        canonical_to_rig_rest=alignment)


def test_rest_direction_identity_and_first_frame_is_not_calibration():
    result = _run([(0, 1, 0), (-1, 0, 0)])
    assert result.samples[0].rotation_local == pytest.approx(QI)
    expected = (0, 0, math.sqrt(0.5), math.sqrt(0.5))
    assert result.samples[1].rotation_local == pytest.approx(expected)
    already_rotated = _run([(-1, 0, 0), (0, 1, 0)])
    assert already_rotated.samples[0].rotation_local == pytest.approx(expected)
    assert quaternion_from_vectors((-1, 0, 0), (-1, 0, 0)) == QI  # historical first-frame delta


def test_arbitrary_target_axis_and_mapping_axis_hint_is_not_rest_axis():
    rig = _rig(axis=(1, 0, 0), child_offset=(0, 1, 0))
    a = _run([(0, 1, 0)], rig=rig, mapping=_mapping("+X"))
    b = _run([(0, 1, 0)], rig=rig, mapping=_mapping("-Z"))
    assert a.samples[0].rotation_local == b.samples[0].rotation_local
    assert a.samples[0].rotation_local == pytest.approx((0, 0, math.sqrt(0.5), math.sqrt(0.5)))
    assert target_rest_axis(rig, "upper") == ((1.0, 0.0, 0.0), "bone_local_axis_hint_primary")
    assert audit_direction_mapping(rig, _mapping().entries[0])["rest_axis_source"] == "bone_local_axis_hint_primary"


def test_rotated_rest_basis_converts_world_alignment_into_local_quaternion():
    rig = _rig(rest=ROT_X_90)
    at_rest = _run([(0, 0, 1)], rig=rig).samples[0]
    assert at_rest.rotation_local == pytest.approx(QI)
    sample = _run([(0, 1, 0)], rig=rig).samples[0]
    rotated_local_axis = rotate_vector_by_quaternion((0, 1, 0), sample.rotation_local)
    world_axis = rotate_vector_by_quaternion(rotated_local_axis, (math.sqrt(0.5), 0, 0, math.sqrt(0.5)))
    assert world_axis == pytest.approx((0, 1, 0), abs=1e-8)


def test_reversal_is_finite_normalized_and_deterministic():
    first = _run([(0, -1, 0)])
    second = _run([(0, -1, 0)])
    q = first.samples[0].rotation_local
    assert first.to_dict() == second.to_dict()
    assert all(math.isfinite(v) for v in q)
    assert sum(v * v for v in q) == pytest.approx(1.0)
    assert rotate_vector_by_quaternion((0, 1, 0), q) == pytest.approx((0, -1, 0), abs=1e-8)


def test_missing_evidence_and_contract_are_unavailable_not_identity():
    invalid = _run([(0, 1, 0), (-1, 0, 0)], invalid={1})
    assert invalid.samples[0].rotation_status == "known"
    assert invalid.samples[1].source_status == "unknown"
    assert invalid.samples[1].rotation_status == "unavailable"
    assert invalid.samples[1].rotation_local is None
    missing_alignment = _run([(0, 1, 0)], alignment=None)
    assert missing_alignment.samples[0].source_status == "known"
    assert missing_alignment.samples[0].rotation_status == "unavailable"
    assert missing_alignment.samples[0].reason == "canonical_to_rig_rest_alignment_missing"
    missing_axis = _rig()
    missing_axis.bones["upper"].children = []
    assert _run([(0, 1, 0)], rig=missing_axis).samples[0].reason == "target_rest_axis_ambiguous"


def test_root_orientation_is_not_applied_to_limb_and_serialization_roundtrip(tmp_path):
    a = _run([(0, 1, 0), (-1, 0, 0)], yaws=[0, math.pi])
    b = _run([(0, 1, 0), (-1, 0, 0)], yaws=[math.pi, 0])
    assert [s.rotation_local for s in a.samples] == [s.rotation_local for s in b.samples]
    assert a.samples[0].frame_index == 0 and a.samples[1].frame_index == 3
    assert a.samples[1].timestamp == 0.1
    assert DirectionCandidate.from_dict(a.to_dict()) == a
    path = tmp_path / "candidate.json"
    save_direction_candidate(a, path)
    assert load_direction_candidate(path) == a


def test_mapped_child_rotation_is_local_to_animated_parent():
    frame = SemanticFrameV2(
        0, 0.0, LocalArticulation(((0, 0, 0), (1, 0, 0), (1, 1, 0))),
        CurrentRootOrientation(True, 0.0), FootMotion(ContactState.UNKNOWN, ContactState.UNKNOWN),
        ObservationReliability((True, True, True), None))
    semantics = AnimationSemanticsV2(
        "arm", 30, "camera_root_relative", ("shoulder", "elbow", "wrist"), (frame,),
        SemanticsProvenance({}, {"policy": ROOT_POLICY_V2}, {"rule_version": RULE_VERSION}, {}))
    rig = RigProfile("test_rig", "synthetic", {
        "upper": BoneInfo("upper", None, ["forearm"], rest_local_matrix=IDENTITY),
        "forearm": BoneInfo("forearm", "upper", ["hand"], rest_local_matrix=IDENTITY),
        "hand": BoneInfo("hand", "forearm", [], rest_local_matrix=(
            (1, 0, 0, 0), (0, 1, 0, 1), (0, 0, 1, 0), (0, 0, 0, 1))),
    }, "upper")
    # The forearm offset from upper is also +Y in the upper's local rest basis.
    rig.bones["forearm"].rest_local_matrix = (
        (1, 0, 0, 0), (0, 1, 0, 1), (0, 0, 1, 0), (0, 0, 0, 1))
    mapping = BoneMappingProfile("test_rig", [
        BoneMappingEntry("upper", "landmark", ["shoulder", "elbow"], "direction"),
        BoneMappingEntry("forearm", "landmark", ["elbow", "wrist"], "direction"),
    ])
    result = solve_framepose_direction_candidate(semantics, rig, mapping, canonical_to_rig_rest=QI)
    by_bone = {s.target_bone: s for s in result.samples}
    assert by_bone["upper"].rotation_local == pytest.approx((0, 0, -math.sqrt(0.5), math.sqrt(0.5)))
    assert by_bone["forearm"].rotation_local == pytest.approx((0, 0, math.sqrt(0.5), math.sqrt(0.5)))
