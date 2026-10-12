import math
import re
from pathlib import Path

import numpy as np
import pytest

from retarget.axis_utils import quaternion_from_axis_angle, rotate_vector_by_quaternion
from retarget.framepose_ik_gt_attribution import measure_chain
from retarget.framepose_target_rest import RestBone, TargetRigRestPose
from retarget.framepose_two_bone_ik import (
    TwoBoneIkChain, chain_forward_kinematics, quaternion_angle_degrees, solve_two_bone, target_chain_geometry,
)
from retarget.framepose_visual_full_arm_ik import (
    adjacent_change, classify_change, compare, full_arm, gt_reference,
)
from retarget.framepose_visual_ik_endpoint import chain_geometry, endpoint_error, visual_direction
from retarget.framepose_visual_ik_handoff import oracle_reference, rig_direction, solve_with_direction

I = ((1., 0., 0., 0.), (0., 1., 0., 0.), (0., 0., 1., 0.), (0., 0., 0., 1.))
ROOT = Path(__file__).resolve().parent.parent
ALIGN = quaternion_from_axis_angle((0.1, 0.2, 0.97), 2.9)


def _target(l1=27.771168, l2=27.251084):
    s, e, w = (0., 0., 140.), (0., l1, 140.), (0., l1, 140. - l2)

    def bone(n, p, h, t):
        return RestBone(n, p, I, h, t)
    rest = TargetRigRestPose("rig", {"body": bone("body", None, (0, 0, 0), (0, 0, 1)), "upper": bone("upper", "body", s, e),
                                     "lower": bone("lower", "upper", e, w), "hand": bone("hand", "lower", w, (w[0], w[1], w[2] - 5))},
                             {"authority": "blender_imported_armature", "fixture": "w76"})
    return rest, target_chain_geometry(rest, TwoBoneIkChain("arm", "s", "e", "w", "upper", "lower", "hand"))


def _chain(theta, bend, upper=0.3, lower=0.25, origin=(0.1, 2.0, 1.4)):
    t, b = math.radians(theta), math.radians(bend)
    u = (upper * math.cos(t), upper * 0.5, -upper * math.sin(t))
    e = tuple(origin[i] + u[i] for i in range(3))
    l = (math.cos(t + b), -0.3, -math.sin(t + b))
    n = math.sqrt(sum(v * v for v in l))
    return origin, e, tuple(e[i] + lower * l[i] / n for i in range(3))


def test_full_arm_is_the_worklog74_solve_with_worklog61_rotations_and_fk():
    rest, target = _target()
    h0 = measure_chain(*_chain(30, 50), 0.7, ALIGN)
    d = rig_direction(visual_direction(0.4, (0.6, -0.8)), 0.7, ALIGN)
    arm, old = full_arm("f2", h0, d, target, rest), solve_with_direction("p2", h0, d, target, rest)
    assert arm.status == old.status == "known"
    assert arm.end == old.solved_endpoint and arm.mid == old.solved_mid and arm.clamp == old.clamp
    assert arm.side == old.side and arm.side_status == old.side_status == "observed"
    direct = solve_two_bone(target.root_head, arm.direction, h0.reach_fraction * target.max_reach,
                            target.proximal_length, target.distal_length, arm.side)
    assert arm.mid == direct.mid and arm.end == direct.end
    fk_mid, fk_end = chain_forward_kinematics(rest, target, arm.proximal_local, arm.distal_local)
    assert math.dist(fk_mid, arm.mid) < 1e-9 and math.dist(fk_end, arm.end) < 1e-9
    assert arm.integrity["ik_to_fk_residual"] < 1e-9 and arm.integrity["quaternion_norm_error"] < 1e-12
    assert arm.integrity["upper_length_residual"] < 1e-9 and arm.integrity["forearm_length_residual"] < 1e-9
    assert arm.integrity["endpoint_reconstruction_residual"] < 1e-9 and arm.integrity["finite"]
    # Deterministic reconstruction.
    assert full_arm("f2", h0, d, target, rest) == arm


def test_wrist_error_reproduces_worklog75_analytic_endpoint_and_reference_is_worklog62_oracle():
    rest, target = _target()
    h0_pts, gt_pts = _chain(30, 50), _chain(38, 62, 0.31, 0.24)
    h0c, gtc = chain_geometry(*h0_pts), chain_geometry(*gt_pts)
    for yaw in (0.0, 0.8, -2.5):
        h0, gt = measure_chain(*h0_pts, yaw, ALIGN), measure_chain(*gt_pts, yaw, ALIGN)
        e_gt, oracle_mid, gt_side, _ = oracle_reference(gt, target, rest)
        ref = gt_reference(gt, target, rest)
        assert math.dist(ref.mid, oracle_mid) < 1e-9 and math.dist(ref.end, e_gt) < 1e-9
        assert math.dist(ref.side, gt_side) < 1e-12
        for f, plane in ((0.3, (0.6, -0.8)), (-0.7, (1.0, 0.2))):
            cam = visual_direction(f, plane)
            c = compare(full_arm("f", h0, rig_direction(cam, yaw, ALIGN), target, rest), ref, e_gt)
            analytic = endpoint_error(target, h0c.reach_fraction, cam, gtc.reach_fraction, gtc.direction)[0]
            assert c["wrist_error"] == pytest.approx(analytic, abs=1e-9)
        f0 = compare(full_arm("f0", h0, h0.direction, target, rest), ref, e_gt)
        assert f0["wrist_error"] == pytest.approx(
            endpoint_error(target, h0c.reach_fraction, h0c.direction, gtc.reach_fraction, gtc.direction)[0], abs=1e-9)


def test_h0_reach_and_bend_ownership_are_fixed_and_gt_never_enters_candidates():
    rest, target = _target()
    h0 = measure_chain(*_chain(30, 50), 0.3, ALIGN)
    dirs = [rig_direction(visual_direction(f, (0.6, -0.8)), 0.3, ALIGN) for f in (-0.5, 0.0, 0.5)]
    arms = [full_arm("f", h0, d, target, rest) for d in dirs]
    for a, d in zip(arms, dirs):
        assert a.reach_fraction == h0.reach_fraction
        # Side = H0 observed side re-perpendicularized to the new direction (no new bend heuristic).
        s = np.asarray(h0.side) - np.dot(h0.side, d) * np.asarray(d)
        assert np.allclose(a.side, s / np.linalg.norm(s), atol=1e-12)
        assert a.root == tuple(target.root_head)
        assert np.linalg.norm(a.upper) == pytest.approx(target.proximal_length)
        assert np.linalg.norm(a.forearm) == pytest.approx(target.distal_length)
    # Candidate geometry depends only on H0 + direction: two different GT arms leave it unchanged.
    gt1, gt2 = measure_chain(*_chain(10, 20), 0.3, ALIGN), measure_chain(*_chain(60, 100), 0.3, ALIGN)
    a = full_arm("f", h0, dirs[0], target, rest)
    c1 = compare(a, gt_reference(gt1, target, rest), oracle_reference(gt1, target, rest)[0])
    c2 = compare(a, gt_reference(gt2, target, rest), oracle_reference(gt2, target, rest)[0])
    assert a == full_arm("f", h0, dirs[0], target, rest) and c1["wrist_error"] != c2["wrist_error"]


def test_unavailability_and_clamp_propagate_without_substitution():
    rest, target = _target()
    h0 = measure_chain(*_chain(30, 50), 0.0, ALIGN)
    parallel = full_arm("f", h0, h0.side, target, rest)
    assert parallel.status == "unavailable:bend_side" and parallel.end is None and parallel.side_status == "unavailable"
    assert compare(parallel, gt_reference(h0, target, rest), (0, 0, 0)) == {"status": "unavailable:bend_side"}
    assert full_arm("f", h0, (0, 0, 0), target, rest).status == "unavailable:degenerate_direction"
    # Same unavailability reasons as the Worklog 74 solve.
    assert solve_with_direction("p", h0, h0.side, target, rest).status == parallel.status
    assert solve_with_direction("p", h0, (0, 0, 0), target, rest).status == "unavailable:degenerate_direction"
    straight = measure_chain((0, 0, 0), (0.3, 0.1, -0.1), (0.54, 0.18, -0.18), 0.0, ALIGN)  # collinear -> rest-plane fallback
    fb = full_arm("f", straight, straight.direction, target, rest)
    assert fb.side_status == "degenerate_rest_plane_fallback" and fb.clamp == "none"
    folded = measure_chain((0, 0, 0), (0.3, 0, 0), (0.0001, 0.0001, 0), 0.0, ALIGN)
    clamped = full_arm("f", folded, folded.direction, target, rest)
    assert clamped.clamp == "clamped_to_min_reach"
    assert math.dist(clamped.end, target.root_head) == pytest.approx(abs(target.proximal_length - target.distal_length))


def test_temporal_metrics_on_synthetic_motion():
    rest, target = _target()
    pts = _chain(30, 50)
    arms = []
    for k in range(4):  # rigid rotation of the whole H0 arm about camera Z at 3 deg / frame
        q = quaternion_from_axis_angle((0, 0, 1), math.radians(3 * k))
        moved = [rotate_vector_by_quaternion(tuple(p[i] - pts[0][i] for i in range(3)), q) for p in pts]
        h0 = measure_chain(*moved, 0.0, ALIGN)
        arms.append(full_arm("f", h0, h0.direction, target, rest))
    changes = [adjacent_change(a, b) for a, b in zip(arms, arms[1:])]
    for (a, b), c in zip(zip(arms, arms[1:]), changes):
        # The quaternion distance bounds the upper-arm swing from above (zero-twist convention).
        swing = math.degrees(math.acos(min(1.0, float(np.dot(a.upper, b.upper) / (np.linalg.norm(a.upper) * np.linalg.norm(b.upper))))))
        assert swing - 1e-9 <= c["proximal_rotation_change_degrees"] < 6.0
        # Rigid rotation about an axis through the shoulder: constant joint displacements per frame.
        assert c["wrist_displacement"] == pytest.approx(changes[0]["wrist_displacement"], abs=1e-9)
        assert c["elbow_displacement"] == pytest.approx(changes[0]["elbow_displacement"], abs=1e-9)
        assert 0 < c["elbow_displacement"] < c["wrist_displacement"]
        # Rigid rotation keeps the bend: every segment/side swings by a constant amount, at most 3 degrees.
        for m in ("upper_direction_change_degrees", "forearm_direction_change_degrees", "bend_side_change_degrees",
                  "chain_direction_change_degrees"):
            assert 0 < c[m] <= 3.0 + 1e-9 and c[m] == pytest.approx(changes[0][m], abs=1e-9)
    assert adjacent_change(arms[0], arms[0])["wrist_displacement"] == 0
    assert quaternion_angle_degrees(arms[0].proximal_local, tuple(-v for v in arms[0].proximal_local)) == 0
    assert classify_change(5, 1, 3) == "candidate_discontinuity"
    assert classify_change(5, 4, 3) == "rapid_motion_candidate_jump"
    assert classify_change(1, 4, 3) == "rapid_motion_candidate_steady"
    assert classify_change(1, 1, 3) == "quiet"


def test_runner_contracts_static():
    run = (ROOT / "scripts/run_framepose_visual_full_arm_ik.py").read_text()
    for check in ("STOP: Worklog 74 full-IK parity failed", "STOP: test-path full-IK parity",
                  "STOP: Worklog 75 endpoint parity failed", "identity mismatch", "frozen H0 validation predictions changed",
                  "frozen H0 test predictions differ from Worklog 75", "Worklog 73 row does not match the FrameBank sample",
                  "duplicate Worklog 73 row", "build_h0_lifted_sequence", "current_root_orientation", "proximal_local",
                  "TOL_CM = 1e-9", "identity_set_equal"):
        assert check in run
    # Both gates run before any test metric.
    test_collect = run.index('collect(bank, h0, w73, target, rest, alignment, args.threedpw_raw, "test")')
    assert run.index("STOP: Worklog 74 full-IK parity failed") < run.index("STOP: test-path full-IK parity") < test_collect
    assert test_collect < run.index("STOP: Worklog 75 endpoint parity failed") < run.index("_block(test_rows, test_rows)")
    assert not re.search(r"\.backward\(|optimizer|train\(|checkpoint\.pt|smooth", run)
    module = (ROOT / "src/retarget/framepose_visual_full_arm_ik.py").read_text().split('"""', 2)[2]
    assert not re.search(r"3dpw|pkl|oracle_sequence|load_bank|rows_heldout|smooth", module)
    for path in sorted((ROOT / "src").rglob("*.py")):
        if path.name != "framepose_visual_full_arm_ik.py":
            assert "framepose_visual_full_arm_ik" not in path.read_text(), path
