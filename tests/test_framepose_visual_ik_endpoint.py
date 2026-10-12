import math
import re
from pathlib import Path

import numpy as np
import pytest

from retarget.axis_utils import quaternion_from_axis_angle
from retarget.framepose_ik_gt_attribution import attribute_frame, measure_chain
from retarget.framepose_target_rest import RestBone, TargetRigRestPose
from retarget.framepose_two_bone_ik import TwoBoneIkChain, target_chain_geometry
from retarget.framepose_visual_ik_endpoint import (
    analytic_formula, angle_degrees, chain_geometry, clamp_distance, endpoint_error, visual_direction,
)
from retarget.framepose_visual_ik_handoff import camera_chain_direction, rig_direction, solve_with_direction

I = ((1., 0., 0., 0.), (0., 1., 0., 0.), (0., 0., 1., 0.), (0., 0., 0., 1.))
ROOT = Path(__file__).resolve().parent.parent
ALIGN = quaternion_from_axis_angle((0.1, 0.2, 0.97), 2.9)


def _target(l1=27.771168, l2=27.251084):
    s, e, w = (0., 0., 140.), (0., l1, 140.), (0., l1, 140. - l2)

    def bone(n, p, h, t):
        return RestBone(n, p, I, h, t)
    rest = TargetRigRestPose("rig", {"body": bone("body", None, (0, 0, 0), (0, 0, 1)), "upper": bone("upper", "body", s, e),
                                     "lower": bone("lower", "upper", e, w), "hand": bone("hand", "lower", w, (w[0], w[1], w[2] - 5))},
                             {"authority": "blender_imported_armature", "fixture": "w75"})
    return rest, target_chain_geometry(rest, TwoBoneIkChain("arm", "s", "e", "w", "upper", "lower", "hand"))


def _chain(theta, bend, upper=0.3, lower=0.25, origin=(0.1, 2.0, 1.4)):
    t, b = math.radians(theta), math.radians(bend)
    u = (upper * math.cos(t), upper * 0.5, -upper * math.sin(t))
    e = tuple(origin[i] + u[i] for i in range(3))
    l = (math.cos(t + b), -0.3, -math.sin(t + b))
    n = math.sqrt(sum(v * v for v in l))
    return origin, e, tuple(e[i] + lower * l[i] / n for i in range(3))


def test_analytic_endpoint_equals_solved_worklog61_ik_for_any_direction_and_rotation():
    rest, target = _target()
    h0_pts, gt_pts = _chain(30, 50), _chain(38, 62, 0.31, 0.24)
    h0, gt = chain_geometry(*h0_pts), chain_geometry(*gt_pts)
    for yaw in (0.0, 0.8, -2.5):
        mh, mg = measure_chain(*h0_pts, yaw, ALIGN), measure_chain(*gt_pts, yaw, ALIGN)
        w62 = attribute_frame(mh, mg, target, rest).to_dict()
        for f, plane in ((0.3, (0.6, -0.8)), (-0.7, (1.0, 0.2)), (h0.direction[1], (h0.direction[0], h0.direction[2]))):
            cam = visual_direction(f, plane)
            solved = solve_with_direction("x", mh, rig_direction(cam, yaw, ALIGN), target, rest)
            analytic, requested, clamp = endpoint_error(target, h0.reach_fraction, cam, gt.reach_fraction, gt.direction)
            e_gt = w62["e_gt"]
            assert analytic == pytest.approx(math.dist(solved.solved_endpoint, e_gt), abs=1e-9)
            assert clamp == solved.clamp == "none" and requested == pytest.approx(analytic)
            cos = float(np.dot(cam, gt.direction))
            assert analytic == pytest.approx(analytic_formula(target.max_reach, h0.reach_fraction, gt.reach_fraction, cos), abs=1e-9)
        # T0 and T4 reproduce Worklog 62 ik_h0 and gt_direction_h0_reach exactly.
        assert endpoint_error(target, h0.reach_fraction, h0.direction, gt.reach_fraction, gt.direction)[0] == \
            pytest.approx(w62["ik_h0_endpoint_error"], abs=1e-9)
        assert endpoint_error(target, h0.reach_fraction, gt.direction, gt.reach_fraction, gt.direction)[0] == \
            pytest.approx(w62["gt_direction_h0_reach_endpoint_error"], abs=1e-9)


def test_reach_and_geometry_follow_worklog61_and_lengths_are_fixed():
    _, target = _target()
    g = chain_geometry(*_chain(20, 90))
    s, e, w = (np.array(p) for p in _chain(20, 90))
    assert g.reach_fraction == pytest.approx(min(1, np.linalg.norm(w - s) / (np.linalg.norm(e - s) + np.linalg.norm(w - e))))
    assert g.direction == pytest.approx(tuple((w - s) / np.linalg.norm(w - s)))
    assert g.bend_angle_degrees == pytest.approx(angle_degrees(e - s, w - e))
    assert target.max_reach == pytest.approx(27.771168 + 27.251084)
    assert chain_geometry((0, 0, 0), (0, 0, 0), (1, 0, 0)) is None
    assert chain_geometry((0, 0, 0), (float("nan"), 0, 0), (1, 0, 0)) is None


def test_clamp_semantics_match_solve_two_bone():
    _, target = _target()
    low = abs(target.proximal_length - target.distal_length)
    assert clamp_distance(0.1, target) == (low, "clamped_to_min_reach")
    assert clamp_distance(target.max_reach + 1, target) == (target.max_reach, "clamped_to_max_reach")
    assert clamp_distance(30.0, target) == (30.0, "none")
    solved, requested, clamp = endpoint_error(target, 0.001, (0, 1, 0), 0.5, (0, 1, 0))
    assert clamp == "clamped_to_min_reach"
    assert solved == pytest.approx(abs(low - 0.5 * target.max_reach)) and requested != pytest.approx(solved)


def test_visual_direction_is_the_worklog74_reconstruction():
    for f, plane in ((0.2, (3.0, -4.0)), (-0.95, (0.1, 1.0))):
        assert visual_direction(f, plane) == pytest.approx(camera_chain_direction(f, plane))


def test_runner_contracts_static():
    run = (ROOT / "scripts/run_framepose_visual_ik_endpoint_test.py").read_text()
    for check in ("STOP: Worklog 74 parity failed", "Worklog 73 row does not match the FrameBank sample", "duplicate Worklog 73 row",
                  "identity mismatch", "frozen H0 validation predictions changed", "H0 materialization was built from another FrameBank",
                  "Worklog 73 chain f identity differs", "build_h0_lifted_sequence", "current_root_orientation",
                  "PARITY_TOLERANCE_CM = 1e-9"):
        assert check in run
    # Parity gate runs before the test split is touched.
    assert run.index("STOP: Worklog 74 parity failed") < run.index('collect(bank, h0, w73, target, args.threedpw_raw, "test")')
    # GT enters only T3/T4 and the reference; no training/selection code.
    assert not re.search(r"\.backward\(|optimizer|train\(|checkpoint\.pt", run)
    module = (ROOT / "src/retarget/framepose_visual_ik_endpoint.py").read_text().split('"""', 2)[2]
    assert not re.search(r"3dpw|pkl|oracle_sequence|load_bank|rows_heldout", module)
    for path in sorted((ROOT / "src").rglob("*.py")):
        if path.name != "framepose_visual_ik_endpoint.py":
            assert "framepose_visual_ik_endpoint" not in path.read_text(), path
