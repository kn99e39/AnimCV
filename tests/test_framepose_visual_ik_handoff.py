import math
import re
from pathlib import Path

import numpy as np
import pytest

from framepose.arm_depth_probe import observed_plane, reconstruct_direction
from retarget.axis_utils import quaternion_from_axis_angle, rotate_vector_by_quaternion
from retarget.framepose_ik_gt_attribution import (
    _runtime_side, attribute_frame, dimensionless_endpoint, heading_relative, measure_chain,
)
from retarget.framepose_target_rest import RestBone, TargetRigRestPose
from retarget.framepose_two_bone_ik import (
    TwoBoneIkChain, observed_bend_side, solve_two_bone, source_chain_geometry, target_chain_geometry,
)
from retarget.framepose_visual_ik_handoff import (
    CANDIDATES, camera_chain_direction, oracle_reference, rig_direction, score, solve_with_direction,
)

I = ((1., 0., 0., 0.), (0., 1., 0., 0.), (0., 0., 1., 0.), (0., 0., 0., 1.))
ROOT = Path(__file__).resolve().parent.parent
ALIGN = quaternion_from_axis_angle((0.1, 0.2, 0.97), 2.9)  # a generic non-trivial rig alignment


def _rest(l1=0.4, l2=0.6):
    shoulder, elbow, wrist = (-1., 0., 1.), (-1., l1, 1.), (-1., l1, 1. - l2)

    def bone(name, parent, head, tail):
        return RestBone(name, parent, I, head, tail)
    rest = TargetRigRestPose("rig", {
        "body": bone("body", None, (0, 0, 0), (0, 0, 1)),
        "upper": bone("upper", "body", shoulder, elbow),
        "lower": bone("lower", "upper", elbow, wrist),
        "hand": bone("hand", "lower", wrist, (wrist[0], wrist[1], wrist[2] - 0.1)),
    }, {"authority": "blender_imported_armature", "fixture": "synthetic handoff"})
    return rest, target_chain_geometry(rest, TwoBoneIkChain("arm", "s", "e", "w", "upper", "lower", "hand"))


def _camera_chain(theta_deg=40.0, bend_deg=60.0, upper=0.3, lower=0.25):
    """A camera-frame shoulder/elbow/wrist chain with real depth."""
    t, b = math.radians(theta_deg), math.radians(bend_deg)
    s = (0.1, 2.0, 1.4)
    u = (upper * math.cos(t), upper * 0.6, -upper * math.sin(t))
    e = tuple(s[i] + u[i] for i in range(3))
    l_dir = (math.cos(t + b), -0.4, -math.sin(t + b))
    n = math.sqrt(sum(v * v for v in l_dir))
    w = tuple(e[i] + lower * l_dir[i] / n for i in range(3))
    return s, e, w


def _unit(v):
    n = math.sqrt(sum(x * x for x in v))
    return tuple(x / n for x in v)


def test_canonical_observed_plane_reconstruction():
    # f and an image-plane direction give the canonical camera unit vector (+X right, +Y depth, +Z up).
    d = camera_chain_direction(0.6, (3.0, 4.0))
    assert d == pytest.approx((0.8 * 0.6, 0.6, 0.8 * 0.8))
    assert d == pytest.approx(tuple(reconstruct_direction(np.array([0.6]), np.array([[3.0, 4.0]]))[0]))
    # observed_plane maps image (dx, -dy) to (X, Z): right and up in the image are +X and +Z.
    x = np.zeros((1, 17, 3))
    x[0, 11, :2] = (0.5, 0.5)
    x[0, 13, :2] = (0.6, 0.4)  # wrist right of and above the shoulder in the image
    planes, oks = observed_plane(x, np.array([[1000.0, 1000.0]]))
    assert oks["chain"][0] and planes["chain"][0] == pytest.approx((1 / math.sqrt(2), 1 / math.sqrt(2)))
    with pytest.raises(ValueError):
        camera_chain_direction(0.3, (0.0, 0.0))


def test_exact_h0_direction_rebuilt_through_the_visual_path_reproduces_worklog61():
    rest, target = _rest()
    s, e, w = _camera_chain()
    yaw = 0.7
    h0 = measure_chain(s, e, w, yaw, ALIGN)
    cam = _unit(tuple(b - a for a, b in zip(s, w)))
    rebuilt = camera_chain_direction(cam[1], (cam[0], cam[2]))
    assert rebuilt == pytest.approx(cam, abs=1e-14)
    direction = rig_direction(rebuilt, yaw, ALIGN)
    assert direction == pytest.approx(h0.direction, abs=1e-12)
    parity = solve_with_direction("parity", h0, direction, target, rest)
    # Worklog 61 runtime: observed side, closed form, H0 reach.
    geometry = source_chain_geometry(h0.upper, h0.lower)
    runtime = solve_two_bone(target.root_head, geometry.endpoint_direction, geometry.reach_fraction * target.max_reach,
                             target.proximal_length, target.distal_length, observed_bend_side(geometry))
    assert math.dist(parity.solved_endpoint, runtime.end) < 1e-12
    assert math.dist(parity.solved_mid, runtime.mid) < 1e-12
    assert parity.requested_endpoint == pytest.approx(dimensionless_endpoint(target, h0.reach_fraction, h0.direction))


def test_heading_relative_transform_parity():
    rng = np.random.default_rng(3)
    for _ in range(20):
        v, yaw = tuple(rng.normal(size=3)), float(rng.uniform(-math.pi, math.pi))
        manual = rotate_vector_by_quaternion(
            rotate_vector_by_quaternion(v, quaternion_from_axis_angle((0, 0, 1), -yaw)), ALIGN)
        assert rig_direction(v, yaw, ALIGN) == pytest.approx(heading_relative(v, yaw, ALIGN), abs=1e-15)
        assert rig_direction(v, yaw, ALIGN) == pytest.approx(manual, abs=1e-12)


def test_candidates_share_h0_reach_and_reprojected_h0_side_and_differ_only_by_direction():
    rest, target = _rest()
    s, e, w = _camera_chain()
    h0 = measure_chain(s, e, w, 0.4, ALIGN)
    plane = (0.6, -0.8)
    solves = {}
    for name, f in (("p1", 0.2), ("p2", -0.35), ("p2_again", -0.35)):
        d = rig_direction(camera_chain_direction(f, plane), 0.4, ALIGN)
        solves[name] = solve_with_direction(name, h0, d, target, rest)
    for c in solves.values():
        assert c.status == "known" and c.reach_fraction == h0.reach_fraction
        expected_side, status = _runtime_side(h0, target, rest, c.direction)
        assert c.side == pytest.approx(expected_side) and status == "observed"
        assert abs(sum(a * b for a, b in zip(c.side, c.direction))) < 1e-12  # side perpendicular to new direction
        # Fixed target rest lengths and the FK numerical contract.
        assert math.dist(target.root_head, c.solved_mid) == pytest.approx(target.proximal_length, abs=1e-12)
        assert math.dist(c.solved_mid, c.solved_endpoint) == pytest.approx(target.distal_length, abs=1e-12)
        assert c.fk_contract_error < 1e-9
        assert c.clamp == "none"
    assert solves["p2"].to_dict() == {**solves["p2_again"].to_dict(), "name": "p2"}  # same f -> same solve
    assert solves["p1"].direction != solves["p2"].direction


def test_scoring_against_the_worklog62_target_and_oracle_elbow():
    rest, target = _rest()
    s, e, w = _camera_chain(theta_deg=30, bend_deg=50)
    gs, ge, gw = _camera_chain(theta_deg=35, bend_deg=55, upper=0.31, lower=0.24)
    h0, gt = measure_chain(s, e, w, 0.2, ALIGN), measure_chain(gs, ge, gw, 0.2, ALIGN)
    e_gt, oracle_mid, gt_side, _ = oracle_reference(gt, target, rest)
    w62 = attribute_frame(h0, gt, target, rest).to_dict()
    assert list(e_gt) == pytest.approx(w62["e_gt"])
    p0 = solve_with_direction("p0", h0, h0.direction, target, rest)
    scored = score(p0, gt, e_gt, oracle_mid, gt_side)
    assert scored["solved_endpoint_error"] == pytest.approx(w62["ik_h0_endpoint_error"])
    assert scored["elbow_error"] == pytest.approx(w62["ik_h0_elbow_error"])  # same Worklog 62 oracle elbow
    ref = solve_with_direction("ref", h0, gt.direction, target, rest)
    assert math.dist(ref.solved_endpoint, e_gt) == pytest.approx(w62["gt_direction_h0_reach_endpoint_error"])
    # GT direction with GT reach would be exact; with H0 reach the residual is |reach error| * (L1 + L2).
    assert math.dist(ref.solved_endpoint, e_gt) == pytest.approx(
        abs(h0.reach_fraction - gt.reach_fraction) * target.max_reach)


def test_unavailable_behaviour_is_reported_not_repaired():
    rest, target = _rest()
    s, e, w = _camera_chain()
    h0 = measure_chain(s, e, w, 0.0, ALIGN)
    assert solve_with_direction("x", h0, (0.0, 0.0, 0.0), target, rest).status == "unavailable:degenerate_direction"
    # A new direction parallel to H0's observed side leaves no bend plane: unavailable, no fallback.
    c = solve_with_direction("x", h0, h0.side, target, rest)
    assert c.status == "unavailable:bend_side" and c.solved_endpoint is None and c.requested_endpoint is not None
    assert score(c, h0, (0, 0, 0), None, None)["solved_endpoint_error"] is None


def test_static_contracts_gt_only_in_p3_and_no_production_import():
    module = (ROOT / "src/retarget/framepose_visual_ik_handoff.py").read_text()
    code = module.split('"""', 2)[2]
    assert not re.search(r"3dpw|pkl|jointPositions|oracle_sequence|load_bank|rows_heldout", code)
    run = (ROOT / "scripts/run_framepose_visual_ik_handoff.py").read_text()
    for check in ("STOP: coordinate/alignment/identity contract not reproduced", "Worklog 73 row does not match the FrameBank sample",
                  "duplicate Worklog 73 row", "identity mismatch", "parity_vs_w61_solved_endpoint_max", "fk_contract_max"):
        assert check in run
    # Only the P3 oracle (and the evaluation target) consume GT forward fraction / direction.
    assert run.count('("p3_gt_forward_fraction", "gt")') == 1
    assert '"p1_e0_visual_direction", "e0"' in run and '"p2_e1_visual_direction", "e1"' in run
    assert CANDIDATES == ("p0_h0_ik", "p1_e0_visual_direction", "p2_e1_visual_direction", "p3_gt_forward_fraction")
    for path in sorted((ROOT / "src").rglob("*.py")):
        if path.name == "framepose_visual_ik_handoff.py":
            continue
        assert "framepose_visual_ik_handoff" not in path.read_text(), path
