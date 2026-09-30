import math
from pathlib import Path

import pytest

from motion.animation_semantics import FootMotion, LocalArticulation, ObservationReliability, SemanticsProvenance
from motion.animation_semantics_v2 import AnimationSemanticsV2, CurrentRootOrientation, ROOT_POLICY_V2, SemanticFrameV2
from pose.contact import ContactState
from pose.contact_time_aware import RULE_VERSION
from retarget.axis_utils import quaternion_from_axis_angle, rotate_vector_by_quaternion
from retarget.framepose_fk import RigFkCalibration, load_rig_fk_calibration, solve_framepose_fk
from retarget.framepose_object_owner import solve_object_owned_fk
from retarget.framepose_target_rest import RestBone, TargetRigRestPose, load_target_rest_pose
from retarget.framepose_two_bone_ik import (
    TwoBoneIkChain, TwoBoneIkChainSet, TwoBoneIkResult, chain_forward_kinematics,
    compare_fk_ik, load_ik_chain_set, solve_framepose_two_bone_ik, solve_two_bone,
    target_chain_geometry,
)
from rig.bone_mapping import BoneMappingEntry, BoneMappingProfile, load_bone_mapping_profile

I = ((1., 0., 0., 0.), (0., 1., 0., 0.), (0., 0., 1., 0.), (0., 0., 0., 1.))
ROOT = Path(__file__).resolve().parent.parent
CHAIN = TwoBoneIkChain("arm", "shoulder", "elbow", "wrist", "upper", "lower", "hand")
CHAINS = TwoBoneIkChainSet("rig", (CHAIN,))
ALL_VALID = (True, True, True)


def _rest(l1=1.0, l2=1.0, *, straight=False):
    """Heading-zero rig: alignment is identity; the rest arm is bent 90 deg unless straight."""
    shoulder = (-1., 0., 1.)
    elbow = (shoulder[0], shoulder[1] + l1, shoulder[2])
    wrist = (elbow[0], elbow[1] + l2, elbow[2]) if straight else (elbow[0], elbow[1], elbow[2] - l2)
    hand = (wrist[0], wrist[1], wrist[2] - 0.1)

    def bone(name, parent, head, tail):
        return RestBone(name, parent, I, head, tail)
    return TargetRigRestPose("rig", {
        "body": bone("body", None, (0, 0, 0), (0, 0, 1)),
        "up": bone("up", "body", (0, 0, 2), (0, 0, 3)),
        "right": bone("right", "body", (1, 0, 1), (1, 1, 1)),
        "upper": bone("upper", "body", shoulder, elbow),
        "lower": bone("lower", "upper", elbow, wrist),
        "hand": bone("hand", "lower", wrist, hand),
    }, {"authority": "blender_imported_armature", "fixture": "synthetic two-bone"})


def _calibration():
    return RigFkCalibration("rig", "body", "upper", "right", "up")


def _mapping():
    return BoneMappingProfile("rig", [
        BoneMappingEntry("upper", "landmark", ["shoulder", "elbow"], "direction"),
        BoneMappingEntry("lower", "landmark", ["elbow", "wrist"], "direction")])


def _semantics(rows, *, scale=1.0):
    """rows: (upper vector, lower vector, yaw or None, validity) in heading-relative canonical."""
    frames = []
    for index, (upper, lower, yaw, valid) in enumerate(rows):
        q = quaternion_from_axis_angle((0, 0, 1), yaw or 0)
        u = tuple(scale * v for v in rotate_vector_by_quaternion(upper, q))
        l = tuple(scale * v for v in rotate_vector_by_quaternion(lower, q))
        origin = (0.3 * scale, 0.1 * scale, 1.4 * scale)
        elbow = tuple(origin[i] + u[i] for i in range(3))
        positions = (origin, elbow, tuple(elbow[i] + l[i] for i in range(3)))
        frames.append(SemanticFrameV2(
            index, index / 30, LocalArticulation(positions),
            CurrentRootOrientation(yaw is not None, yaw),
            FootMotion(ContactState.UNKNOWN, ContactState.UNKNOWN),
            ObservationReliability(valid, None)))
    return AnimationSemanticsV2(
        "synthetic_ik", 30, "camera_root_relative", ("shoulder", "elbow", "wrist"),
        tuple(frames), SemanticsProvenance({}, {"policy": ROOT_POLICY_V2},
                                          {"rule_version": RULE_VERSION}, {}))


def _solve(rows, rest=None, **kwargs):
    rest = rest or _rest()
    semantics = _semantics(rows, **kwargs)
    ik = solve_framepose_two_bone_ik(semantics, rest, _calibration(), CHAINS)
    fk = solve_object_owned_fk(semantics, rest, _mapping(), _calibration())
    return ik, fk, compare_fk_ik(fk.pose_bone_samples, ik, rest, CHAINS)


def _polar(length, degrees_from_forward, *, up=True):
    """Vector in the canonical Y/Z plane at an angle from +Y toward +Z (or -Z)."""
    a = math.radians(degrees_from_forward)
    return (0.0, length * math.cos(a), (1 if up else -1) * length * math.sin(a))


def _bent(l_upper, l_lower, elbow_degrees=60.0, *, flip=False):
    upper = _polar(l_upper, 0)
    lower = _polar(l_lower, elbow_degrees, up=not flip)
    return upper, lower


def _side(point, root, end):
    """Signed offset of point from the root->end line along canonical Z-ish normal."""
    e = tuple(end[i] - root[i] for i in range(3))
    n = math.sqrt(sum(v * v for v in e))
    e = tuple(v / n for v in e)
    p = tuple(point[i] - root[i] for i in range(3))
    along = sum(p[i] * e[i] for i in range(3))
    return tuple(p[i] - along * e[i] for i in range(3))


def test_target_rest_lengths_and_topology_come_from_blender_snapshot():
    geometry = target_chain_geometry(_rest(0.4, 0.6), CHAIN)
    assert (geometry.proximal_length, geometry.distal_length) == pytest.approx((0.4, 0.6))
    assert (geometry.min_reach, geometry.max_reach) == pytest.approx((0.2, 1.0))
    assert geometry.root_head == (-1, 0, 1)
    with pytest.raises(ValueError, match="not a child"):
        target_chain_geometry(_rest(), TwoBoneIkChain("x", "shoulder", "elbow", "wrist",
                                                     "lower", "upper"))
    broken = _rest()
    bones = dict(broken.bones)
    bones["lower"] = RestBone("lower", "upper", I, (-1, 1.5, 1), (-1, 2, 1))
    with pytest.raises(ValueError, match="not connected"):
        target_chain_geometry(TargetRigRestPose("rig", bones, broken.provenance), CHAIN)

    rest = load_target_rest_pose(ROOT / "examples/e2e_demo/baserig_blender_rest.json")
    chains = load_ik_chain_set(ROOT / "examples/e2e_demo/framepose_ik_chains.json")
    real = target_chain_geometry(rest, chains.chains[0])
    assert (real.proximal, real.distal) == ("upperarm_l", "lowerarm_l")
    assert real.proximal_length == pytest.approx(27.771168, abs=1e-5)
    assert real.distal_length == pytest.approx(27.251084, abs=1e-5)
    assert real.max_reach == pytest.approx(55.022252, abs=1e-5)
    assert real.min_reach == pytest.approx(0.520084, abs=1e-5)


def test_source_scale_invariance():
    rows = [(*_bent(0.6, 0.4, 70), 0.3, ALL_VALID)]
    small = _solve(rows, rest=_rest(0.4, 0.6), scale=0.5)[0]
    large = _solve(rows, rest=_rest(0.4, 0.6), scale=3.0)[0]
    a, b = small.samples[0], large.samples[0]
    assert a.proximal_status == "known"
    assert a.source_reach_fraction == pytest.approx(b.source_reach_fraction, abs=1e-12)
    assert a.requested_endpoint == pytest.approx(b.requested_endpoint, abs=1e-12)
    assert a.proximal_local == pytest.approx(b.proximal_local, abs=1e-12)
    assert a.distal_local == pytest.approx(b.distal_local, abs=1e-12)


def test_matched_proportions_ik_equals_direction_fk():
    rows = [(*_bent(0.6, 0.4, angle), 0.0, ALL_VALID) for angle in (20, 75, 130)]
    ik, fk, comparison = _solve(rows, rest=_rest(1.2, 0.8))
    for row in comparison:
        assert row.status == "both_known"
        assert row.fk_endpoint_error == pytest.approx(0, abs=1e-9)
        assert row.ik_endpoint_error == pytest.approx(0, abs=1e-9)
        assert row.fk_endpoint == pytest.approx(row.ik_endpoint, abs=1e-9)
        assert row.proximal_delta_degrees == pytest.approx(0, abs=1e-5)
        assert row.distal_delta_degrees == pytest.approx(0, abs=1e-5)


def test_mismatched_proportions_ik_corrects_endpoint_with_exact_lengths_and_bend_side():
    rest = _rest(0.4, 0.6)
    ik, fk, comparison = _solve([(*_bent(0.6, 0.4, 80), 0.0, ALL_VALID)], rest=rest)
    row, sample = comparison[0], ik.samples[0]
    assert row.fk_endpoint_error > 0.05
    assert row.ik_endpoint_error == pytest.approx(0, abs=1e-9)
    assert sample.clamp_status == "none" and sample.bend_plane_status == "observed"
    geometry = target_chain_geometry(rest, CHAIN)
    mid, end = chain_forward_kinematics(rest, geometry, sample.proximal_local, sample.distal_local)
    assert math.dist(geometry.root_head, mid) == pytest.approx(0.4, abs=1e-12)
    assert math.dist(mid, end) == pytest.approx(0.6, abs=1e-12)
    # Source elbow lies on the -Z side of the shoulder->wrist line (lower bends +Z).
    source_side = _side((0, 0.6, 0), (0, 0, 0), tuple(a + b for a, b in zip(*_bent(0.6, 0.4, 80))))
    target_side = _side(mid, geometry.root_head, end)
    assert sum(a * b for a, b in zip(source_side, target_side)) > 0
    assert row.proximal_delta_degrees > 1


def test_body_yaw_changes_object_rotation_but_not_ik_locals():
    pose = _bent(0.6, 0.4, 70)
    ik, fk, _ = _solve([(*pose, yaw, ALL_VALID) for yaw in (0, math.pi / 4, math.pi / 2)],
                       rest=_rest(0.4, 0.6))
    first = ik.samples[0]
    for sample in ik.samples[1:]:
        assert sample.proximal_local == pytest.approx(first.proximal_local, abs=1e-12)
        assert sample.distal_local == pytest.approx(first.distal_local, abs=1e-12)
        assert sample.requested_endpoint == pytest.approx(first.requested_endpoint, abs=1e-12)
    objects = [s.rotation_armature_space for s in fk.rig_rotations]
    assert objects[0] != pytest.approx(objects[1]) and objects[1] != pytest.approx(objects[2])
    assert ik.provenance["root_yaw_owner"].startswith("Blender Armature Object")


def test_articulation_sensitivity_with_fixed_yaw():
    base = _bent(0.6, 0.4, 70)
    moved_wrist = (base[0], _polar(0.4, 70, up=True)[:2] + (-0.1,))
    flipped = _bent(0.6, 0.4, 70, flip=True)
    straighter = _bent(0.6, 0.4, 30)
    ik, _, _ = _solve([(*pose, 0.5, ALL_VALID) for pose in (base, moved_wrist, flipped, straighter)],
                      rest=_rest(0.4, 0.6))
    s0, s1, s2, s3 = ik.samples
    for other in (s1, s2, s3):
        assert other.proximal_local != pytest.approx(s0.proximal_local, abs=1e-6)
    assert s3.source_reach_fraction > s0.source_reach_fraction
    geometry = target_chain_geometry(_rest(0.4, 0.6), CHAIN)
    side0 = _side(s0.solved_mid, geometry.root_head, s0.solved_endpoint)
    side2 = _side(s2.solved_mid, geometry.root_head, s2.solved_endpoint)
    assert sum(a * b for a, b in zip(side0, side2)) < 0


def test_reachable_boundaries_and_unreachable_clamps_are_explicit():
    root, e, side = (0., 0., 0.), (0., 1., 0.), (0., 0., 1.)
    at_max = solve_two_bone(root, e, 1.0, 0.4, 0.6, side)
    at_min = solve_two_bone(root, e, 0.2, 0.4, 0.6, side)
    over = solve_two_bone(root, e, 1.5, 0.4, 0.6, side)
    under = solve_two_bone(root, e, 0.05, 0.4, 0.6, side)
    assert at_max.clamp_status == at_min.clamp_status == "none"
    assert over.clamp_status == "clamped_to_max_reach" and over.reachable_distance == 1.0
    assert under.clamp_status == "clamped_to_min_reach" and under.reachable_distance == pytest.approx(0.2)
    for solve in (at_max, at_min, over, under):
        assert all(math.isfinite(v) for v in (*solve.mid, *solve.end))
        assert math.dist(root, solve.mid) == pytest.approx(0.4)
        assert math.dist(solve.mid, solve.end) == pytest.approx(0.6)
    # Pipeline: an equal-segment source folded to 170 deg asks for less than |L1-L2|.
    ik, _, comparison = _solve([(*_bent(0.5, 0.5, 170), 0.0, ALL_VALID)], rest=_rest(0.4, 0.6))
    sample = ik.samples[0]
    assert sample.clamp_status == "clamped_to_min_reach"
    assert sample.source_reach_fraction < 0.2
    assert sample.endpoint_error == pytest.approx(0.2 - sample.source_reach_fraction, abs=1e-9)
    assert comparison[0].ik_endpoint_error < comparison[0].fk_endpoint_error


def test_straight_chain_uses_declared_fallback_never_observed():
    ik, _, _ = _solve([(*_bent(0.6, 0.4, 0.5), 0.0, ALL_VALID),
                       (*_bent(0.6, 0.4, 0.0), 0.0, ALL_VALID)], rest=_rest(0.4, 0.6))
    for sample in ik.samples:
        assert sample.bend_plane_status == "degenerate_rest_plane_fallback"
        assert sample.proximal_status == "known"
        assert all(math.isfinite(v) for v in sample.proximal_local + sample.distal_local)
    assert ik.samples[1].source_reach_fraction == pytest.approx(1.0)
    assert ik.samples[1].clamp_status == "none"
    straight_rest, _, _ = _solve([(*_bent(0.6, 0.4, 0.5), 0.0, ALL_VALID)],
                                 rest=_rest(0.4, 0.6, straight=True))
    assert straight_rest.samples[0].reason == "bend_plane_undefined"
    assert straight_rest.samples[0].proximal_local is None


def test_invalid_source_joints_degenerate_segments_and_unknown_yaw_are_unavailable():
    pose = _bent(0.6, 0.4, 60)
    rows = [(*pose, 0.0, ALL_VALID),
            (*pose, 0.0, (False, True, True)),
            (*pose, 0.0, (True, False, True)),
            (*pose, 0.0, (True, True, False)),
            ((0, 0, 0), pose[1], 0.0, ALL_VALID),
            (*pose, None, ALL_VALID),
            (*pose, 0.0, ALL_VALID)]
    ik, fk, comparison = _solve(rows, rest=_rest(0.4, 0.6))
    reasons = [s.reason for s in ik.samples]
    assert reasons == [None, "source_root_invalid", "source_mid_invalid", "source_end_invalid",
                       "source_segment_degenerate", "root_orientation_unknown", None]
    unknown = ik.samples[5]
    assert unknown.source_status == "known" and unknown.root_orientation_status == "unknown"
    for sample in ik.samples[1:6]:
        assert sample.proximal_status == sample.distal_status == "unavailable"
        assert sample.proximal_local is None and sample.solved_endpoint is None
    # No implicit hold: the frame after the gap is solved from its own evidence only.
    assert ik.samples[6].proximal_local == pytest.approx(ik.samples[0].proximal_local)
    assert [row.status for row in comparison[1:6]] == ["neither"] * 5
    assert comparison[0].status == "both_known"


def test_determinism_serialization_and_quaternion_normalization():
    rows = [(*_bent(0.6, 0.4, angle), 0.2, ALL_VALID) for angle in (10, 60, 120)]
    a = _solve(rows, rest=_rest(0.4, 0.6))[0]
    b = _solve(rows, rest=_rest(0.4, 0.6))[0]
    assert a.to_dict() == b.to_dict()
    assert TwoBoneIkResult.from_dict(a.to_dict()) == a
    for sample in a.samples:
        for q in (sample.proximal_local, sample.distal_local):
            assert sum(v * v for v in q) == pytest.approx(1, abs=1e-12)
            assert q[3] >= 0
    assert "reach_fraction" in a.provenance["endpoint_formula"]


def test_fk_baseline_is_unchanged_by_ik():
    rows = [(*_bent(0.6, 0.4, 70), 0.4, ALL_VALID), (*_bent(0.6, 0.4, 70), 0.4, (True, False, True))]
    rest = _rest(0.4, 0.6)
    semantics = _semantics(rows)
    before = solve_object_owned_fk(semantics, rest, _mapping(), _calibration()).to_dict()
    solve_framepose_two_bone_ik(semantics, rest, _calibration(), CHAINS)
    after = solve_object_owned_fk(semantics, rest, _mapping(), _calibration())
    assert after.to_dict() == before
    direct = solve_framepose_fk(semantics, rest, _mapping(), _calibration(),
                                root_orientation_owner="armature_object")
    assert [s.to_dict() for s in after.pose_bone_samples] == [s.to_dict() for s in direct.samples]


def test_real_baserig_chain_exact_lengths_yaw_invariance_and_fk_contrast():
    rest = load_target_rest_pose(ROOT / "examples/e2e_demo/baserig_blender_rest.json")
    mapping = load_bone_mapping_profile(ROOT / "examples/e2e_demo/mapping.json")
    calibration = load_rig_fk_calibration(ROOT / "examples/e2e_demo/framepose_fk_calibration.json")
    chains = load_ik_chain_set(ROOT / "examples/e2e_demo/framepose_ik_chains.json")
    frames = []
    upper, lower = (0.30, 0.05, -0.1), (0.05, 0.2, 0.1)  # human-like 1.3:1 proportions
    for index, yaw in enumerate((0, math.pi / 4, math.pi / 2)):
        q = quaternion_from_axis_angle((0, 0, 1), yaw)
        u, l = rotate_vector_by_quaternion(upper, q), rotate_vector_by_quaternion(lower, q)
        frames.append(SemanticFrameV2(
            index, index / 30, LocalArticulation(((0., 0., 0.), u, tuple(u[i] + l[i] for i in range(3)))),
            CurrentRootOrientation(True, yaw), FootMotion(ContactState.UNKNOWN, ContactState.UNKNOWN),
            ObservationReliability(ALL_VALID, None)))
    semantics = AnimationSemanticsV2(
        "baserig_ik", 30, "camera_root_relative", ("left_shoulder", "left_elbow", "left_wrist"),
        tuple(frames), SemanticsProvenance({}, {"policy": ROOT_POLICY_V2}, {"rule_version": RULE_VERSION}, {}))
    ik = solve_framepose_two_bone_ik(semantics, rest, calibration, chains)
    fk = solve_object_owned_fk(semantics, rest, mapping, calibration)
    comparison = compare_fk_ik(fk.pose_bone_samples, ik, rest, chains)
    geometry = target_chain_geometry(rest, chains.chains[0])
    for sample, row in zip(ik.samples, comparison):
        mid, end = chain_forward_kinematics(rest, geometry, sample.proximal_local, sample.distal_local)
        assert math.dist(geometry.root_head, mid) == pytest.approx(geometry.proximal_length, abs=1e-9)
        assert math.dist(mid, end) == pytest.approx(geometry.distal_length, abs=1e-9)
        assert row.ik_endpoint_error == pytest.approx(0, abs=1e-9)
        assert row.fk_endpoint_error > 0.1
        assert sample.proximal_local == pytest.approx(ik.samples[0].proximal_local, abs=1e-9)
