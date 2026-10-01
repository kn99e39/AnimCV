import math
import re
from pathlib import Path

import pytest

from retarget.axis_utils import quaternion_from_axis_angle, rotate_vector_by_quaternion
from retarget.framepose_ik_gt_attribution import (
    VARIANTS, attribute_frame, dimensionless_endpoint, measure_chain,
)
from retarget.framepose_target_rest import RestBone, TargetRigRestPose
from retarget.framepose_two_bone_ik import TwoBoneIkChain, target_chain_geometry

I = ((1., 0., 0., 0.), (0., 1., 0., 0.), (0., 0., 1., 0.), (0., 0., 0., 1.))
QI = (0., 0., 0., 1.)
ROOT = Path(__file__).resolve().parent.parent


def _rest(l1=0.4, l2=0.6):
    shoulder = (-1., 0., 1.)
    elbow = (-1., l1, 1.)
    wrist = (-1., l1, 1. - l2)

    def bone(name, parent, head, tail):
        return RestBone(name, parent, I, head, tail)
    rest = TargetRigRestPose("rig", {
        "body": bone("body", None, (0, 0, 0), (0, 0, 1)),
        "upper": bone("upper", "body", shoulder, elbow),
        "lower": bone("lower", "upper", elbow, wrist),
        "hand": bone("hand", "lower", wrist, (wrist[0], wrist[1], wrist[2] - 0.1)),
    }, {"authority": "blender_imported_armature", "fixture": "synthetic attribution"})
    return rest, target_chain_geometry(rest, TwoBoneIkChain("arm", "s", "e", "w", "upper", "lower", "hand"))


def _chain(upper, lower, origin=(0.2, 0.1, 1.3)):
    mid = tuple(origin[i] + upper[i] for i in range(3))
    return origin, mid, tuple(mid[i] + lower[i] for i in range(3))


def _bent(lu, ll, degrees, *, flip=False):
    a = math.radians(degrees)
    return (0., lu, 0.), (0., ll * math.cos(a), (-1 if flip else 1) * ll * math.sin(a))


def _measure(upper, lower, yaw=0.0, scale=1.0):
    q = quaternion_from_axis_angle((0, 0, 1), yaw)
    u = tuple(scale * v for v in rotate_vector_by_quaternion(upper, q))
    l = tuple(scale * v for v in rotate_vector_by_quaternion(lower, q))
    return measure_chain(*_chain(u, l), yaw, QI)


def test_matched_geometry_measures_are_dimensionless_and_share_the_runtime_transform():
    m = _measure(*_bent(0.3, 0.2, 90))
    assert m.ratio == pytest.approx(1.5)
    assert m.reach_fraction == pytest.approx(math.hypot(0.3, 0.2) / 0.5)
    assert m.bend_angle_degrees == pytest.approx(90)
    assert m.direction == pytest.approx(tuple(v / math.hypot(0.3, 0.2) for v in (0, 0.3, 0.2)))
    assert m.side == pytest.approx((0, 0.2 / math.hypot(0.3, 0.2), -0.3 / math.hypot(0.3, 0.2)))
    # Source scale and the shared runtime yaw removal do not change the measures.
    other = _measure(*_bent(0.3, 0.2, 90), yaw=1.1, scale=4.0)
    assert other.direction == pytest.approx(m.direction, abs=1e-12)
    assert other.reach_fraction == pytest.approx(m.reach_fraction, abs=1e-12)


def test_gt_dimensionless_endpoint_construction():
    _, target = _rest()
    endpoint = dimensionless_endpoint(target, 0.8, (0., 0.6, 0.8))
    assert endpoint == pytest.approx((-1., 0.48, 1.64))


def test_identical_h0_and_gt_give_zero_ik_errors_and_fk_shows_only_true_adaptation():
    rest, target = _rest(0.4, 0.6)
    pose = _bent(0.3, 0.2, 70)
    row = attribute_frame(_measure(*pose), _measure(*pose), target, rest).to_dict()
    for name in ("ik_h0", "h0_direction_gt_reach", "gt_direction_h0_reach", "oracle_ik"):
        assert row[f"{name}_endpoint_error"] == pytest.approx(0, abs=1e-12)
        assert row[f"{name}_elbow_error"] == pytest.approx(0, abs=1e-12)
    assert row["fk_h0_endpoint_error"] == pytest.approx(row["fk_gt_endpoint_error"], abs=1e-12)
    assert row["fk_h0_endpoint_error"] > 0.05  # 1.5 source ratio vs 0.67 target ratio
    assert row["bend_side_agrees"] and row["bend_plane_error_degrees"] == pytest.approx(0, abs=1e-6)


def test_hybrids_isolate_direction_error_and_reach_error():
    rest, target = _rest(0.4, 0.6)
    gt_pose = _bent(0.3, 0.2, 70)
    gt = _measure(*gt_pose)
    # Direction-only H0 error: the same chain rigidly rotated 10 deg about the bend normal.
    q = quaternion_from_axis_angle((1, 0, 0), math.radians(10))
    rotated = _measure(rotate_vector_by_quaternion(gt_pose[0], q), rotate_vector_by_quaternion(gt_pose[1], q))
    row = attribute_frame(rotated, gt, target, rest).to_dict()
    chord = 2 * gt.reach_fraction * target.max_reach * math.sin(math.radians(5))
    assert row["endpoint_direction_error_degrees"] == pytest.approx(10)
    assert row["reach_fraction_error"] == pytest.approx(0, abs=1e-12)
    assert row["h0_direction_gt_reach_endpoint_error"] == pytest.approx(chord)
    assert row["gt_direction_h0_reach_endpoint_error"] == pytest.approx(0, abs=1e-12)
    # Reach-only H0 error: a distorted segment ratio with the same endpoint direction.
    h0 = _measure((0., 0.45, 0.), tuple(gt_pose[1][i] * 0.5 for i in range(3)))
    h0_aligned = measure_chain(*_chain(*(
        rotate_vector_by_quaternion(v, quaternion_from_axis_angle((1, 0, 0), math.radians(
            math.degrees(math.atan2(gt.direction[2], gt.direction[1]))
            - math.degrees(math.atan2(h0.direction[2], h0.direction[1])))))
        for v in (h0.upper, h0.lower))), 0.0, QI)
    row = attribute_frame(h0_aligned, gt, target, rest).to_dict()
    assert row["endpoint_direction_error_degrees"] == pytest.approx(0, abs=1e-6)
    assert row["gt_direction_h0_reach_endpoint_error"] == pytest.approx(
        abs(row["reach_fraction_error"]) * target.max_reach, abs=1e-9)
    assert row["h0_direction_gt_reach_endpoint_error"] == pytest.approx(0, abs=1e-6)
    assert row["oracle_ik_endpoint_error"] == pytest.approx(0, abs=1e-12)


def test_bend_side_comparison_detects_flips_on_the_same_gt_target():
    rest, target = _rest(0.4, 0.6)
    gt = _measure(*_bent(0.3, 0.2, 70))
    flipped = attribute_frame(_measure(*_bent(0.3, 0.2, 70, flip=True)), gt, target, rest).to_dict()
    assert flipped["bend_side_agrees"] is False
    assert flipped["bend_plane_error_degrees"] == pytest.approx(180, abs=1e-4)
    assert flipped["elbow_error_h0_side_on_gt_target"] > 0.1
    # Endpoint does not depend on bend side.
    assert flipped["gt_direction_h0_reach_endpoint_error"] == pytest.approx(0, abs=1e-9)
    straight = attribute_frame(_measure(*_bent(0.3, 0.2, 0.3)), gt, target, rest).to_dict()
    assert straight["h0_side_status"] == "degenerate_rest_plane_fallback"
    assert straight["h0_degenerate"] is True


def test_invalid_or_degenerate_chains_are_rejected():
    nan = float("nan")
    assert measure_chain((0, 0, 0), (nan, 0, 0), (0, 1, 0), 0.0, QI) is None
    assert measure_chain((0, 0, 0), (0, 0, 0), (0, 1, 0), 0.0, QI) is None
    assert measure_chain((0, 0, 0), (0, 1, 0), (0, 0, 0), 0.0, QI) is None  # root == end
    assert set(VARIANTS) >= {"fk_h0", "ik_h0", "oracle_ik"}


def test_no_production_target_leakage():
    production = [ROOT / "src/retarget/framepose_fk.py", ROOT / "src/retarget/framepose_object_owner.py",
                  ROOT / "src/retarget/framepose_two_bone_ik.py", ROOT / "src/retarget/framepose_target_rest.py",
                  ROOT / "src/blender/framepose_object_owner_adapter.py",
                  *sorted((ROOT / "src/motion").glob("animation_semantics*.py"))]
    pattern = re.compile(r"framepose_ik_gt_attribution|three_dpw_adapter|oracle_world_reference|jointPositions")
    for path in production:
        assert not pattern.search(path.read_text()), path
    attribution = (ROOT / "src/retarget/framepose_ik_gt_attribution.py").read_text()
    assert "EVALUATION ONLY" in attribution and "three_dpw" not in attribution
