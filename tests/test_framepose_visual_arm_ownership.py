import math
import re
from pathlib import Path

import numpy as np
import pytest

from retarget.axis_utils import quaternion_from_axis_angle, rotate_vector_by_quaternion
from retarget.framepose_ik_gt_attribution import measure_chain
from retarget.framepose_target_rest import RestBone, TargetRigRestPose
from retarget.framepose_two_bone_ik import TwoBoneIkChain, target_chain_geometry
from retarget.framepose_visual_arm_ownership import (
    CONTRACT, DIAGNOSTICS, FACTORIAL, RUNTIME, direction_penalty, motion_residual, owned_measure, ownership_effects,
    penalty_changes,
)
from retarget.framepose_visual_full_arm_ik import full_arm, gt_reference
from retarget.framepose_visual_ik_endpoint import visual_direction
from retarget.framepose_visual_ik_handoff import rig_direction

I = ((1., 0., 0., 0.), (0., 1., 0., 0.), (0., 0., 1., 0.), (0., 0., 0., 1.))
ROOT = Path(__file__).resolve().parent.parent
ALIGN = quaternion_from_axis_angle((0.1, 0.2, 0.97), 2.9)


def _target(l1=27.771168, l2=27.251084):
    s, e, w = (0., 0., 140.), (0., l1, 140.), (0., l1, 140. - l2)

    def bone(n, p, h, t):
        return RestBone(n, p, I, h, t)
    rest = TargetRigRestPose("rig", {"body": bone("body", None, (0, 0, 0), (0, 0, 1)), "upper": bone("upper", "body", s, e),
                                     "lower": bone("lower", "upper", e, w), "hand": bone("hand", "lower", w, (w[0], w[1], w[2] - 5))},
                             {"authority": "blender_imported_armature", "fixture": "w77"})
    return rest, target_chain_geometry(rest, TwoBoneIkChain("arm", "s", "e", "w", "upper", "lower", "hand"))


def _chain(theta, bend, upper=0.3, lower=0.25, origin=(0.1, 2.0, 1.4)):
    t, b = math.radians(theta), math.radians(bend)
    u = (upper * math.cos(t), upper * 0.5, -upper * math.sin(t))
    e = tuple(origin[i] + u[i] for i in range(3))
    l = (math.cos(t + b), -0.3, -math.sin(t + b))
    n = math.sqrt(sum(v * v for v in l))
    return origin, e, tuple(e[i] + lower * l[i] / n for i in range(3))


def _measures(yaw=0.4):
    return measure_chain(*_chain(30, 50), yaw, ALIGN), measure_chain(*_chain(38, 95, 0.31, 0.24), yaw, ALIGN)


def test_factor_assignment_is_exact_and_complete():
    assert set(FACTORIAL) == {f"c{d}{r}{s}" for d in "01" for r in "01" for s in "01"}
    for name, (d, r, s) in FACTORIAL.items():
        assert (d, r, s) == ({"0": "h0", "1": "e1"}[name[1]], {"0": "h0", "1": "gt"}[name[2]], {"0": "h0", "1": "gt"}[name[3]])
    assert DIAGNOSTICS == {"c_gtf_h0r_h0s": ("gt_f", "h0", "h0")}
    assert RUNTIME == ("c000", "c100") and all("gt" not in v for n in RUNTIME for v in FACTORIAL[n])
    h0, gt = _measures()
    assert owned_measure(h0, gt, "h0", "h0") is h0  # runtime candidates never see GT
    m = owned_measure(h0, gt, "gt", "h0")
    assert m.reach_fraction == gt.reach_fraction and m.side == h0.side and m.direction == h0.direction
    m = owned_measure(h0, gt, "h0", "gt")
    assert m.reach_fraction == h0.reach_fraction and m.side == gt.side and m.degenerate == gt.degenerate
    m = owned_measure(h0, gt, "gt", "gt")
    assert (m.reach_fraction, m.side, m.upper) == (gt.reach_fraction, gt.side, h0.upper)
    with pytest.raises(ValueError):
        owned_measure(h0, gt, "e1", "h0")


def test_factorial_arms_use_the_worklog76_solve_and_oracle_contract():
    rest, target = _target()
    h0, gt = _measures()
    e1 = rig_direction(visual_direction(0.3, (0.6, -0.8)), 0.4, ALIGN)
    # C000 / C100 are exactly the Worklog 76 F0 / F2 arms.
    assert full_arm("c000", owned_measure(h0, gt, "h0", "h0"), h0.direction, target, rest) == full_arm("c000", h0, h0.direction, target, rest)
    assert full_arm("c100", owned_measure(h0, gt, "h0", "h0"), e1, target, rest).end == full_arm("f2", h0, e1, target, rest).end
    # GT direction + GT reach + GT side is the Worklog 62 reference arm.
    ref = gt_reference(gt, target, rest)
    k = full_arm("k", owned_measure(h0, gt, "gt", "gt"), gt.direction, target, rest)
    assert math.dist(k.end, ref.end) < 1e-12 and math.dist(k.mid, ref.mid) < 1e-12 and k.proximal_local == ref.proximal_local
    for name, (d, r, s) in FACTORIAL.items():
        arm = full_arm(name, owned_measure(h0, gt, r, s), h0.direction if d == "h0" else e1, target, rest)
        assert arm.reach_fraction == (gt if r == "gt" else h0).reach_fraction
        src = np.asarray((gt if s == "gt" else h0).side)
        dd = np.asarray(arm.direction)
        perp = src - np.dot(src, dd) * dd
        assert np.allclose(arm.side, perp / np.linalg.norm(perp), atol=1e-12)  # Worklog 62 perpendicularization
        assert np.linalg.norm(arm.upper) == pytest.approx(target.proximal_length)
        assert np.linalg.norm(arm.forearm) == pytest.approx(target.distal_length)
        assert arm.integrity["ik_to_fk_residual"] < 1e-9
    # Deterministic.
    assert full_arm("c111", owned_measure(h0, gt, "gt", "gt"), e1, target, rest) == \
        full_arm("c111", owned_measure(h0, gt, "gt", "gt"), e1, target, rest)
    assert set(CONTRACT) == {"k_gtdir_h0r_h0s", "k_gtdir_gtr_gts"}


def test_degenerate_gt_side_and_clamp_propagate():
    rest, target = _target()
    h0 = measure_chain(*_chain(30, 50), 0.0, ALIGN)
    straight_gt = measure_chain((0, 0, 0), (0.3, 0.1, -0.1), (0.54, 0.18, -0.18), 0.0, ALIGN)
    arm = full_arm("c001", owned_measure(h0, straight_gt, "h0", "gt"), h0.direction, target, rest)
    assert arm.side_status == "degenerate_rest_plane_fallback"
    parallel = full_arm("c101", owned_measure(h0, straight_gt, "h0", "h0"), h0.side, target, rest)
    assert parallel.status == "unavailable:bend_side"
    folded = measure_chain((0, 0, 0), (0.3, 0, 0), (0.0001, 0.0001, 0), 0.0, ALIGN)
    assert full_arm("c010", owned_measure(h0, folded, "gt", "h0"), h0.direction, target, rest).clamp == "clamped_to_min_reach"


def test_motion_residual_on_synthetic_motion():
    rest, target = _target()
    pts = _chain(30, 50)

    def arm_at(k, offset=(0, 0, 0)):
        q = quaternion_from_axis_angle((0, 0, 1), math.radians(4 * k))
        moved = [tuple(a + o for a, o in zip(rotate_vector_by_quaternion(tuple(p[i] - pts[0][i] for i in range(3)), q), offset))
                 for p in pts]
        h0 = measure_chain(*moved, 0.0, ALIGN)
        return full_arm("a", h0, h0.direction, target, rest)
    r0, r1 = arm_at(0), arm_at(1)
    same = motion_residual(r0, r1, r0, r1)
    # acos near 1 limits the quaternion distance to ~1e-6 degrees in float64.
    assert all(abs(v) < (1e-5 if "rotation" in k else 1e-9) for k, v in same.items())
    static = motion_residual(r0, r0, r0, r1)  # candidate frozen while the reference moves
    assert static["elbow_motion_residual"] == pytest.approx(math.dist(r0.mid, r1.mid))
    assert static["wrist_motion_residual"] == pytest.approx(math.dist(r0.end, r1.end))
    assert static["upper_direction_residual_degrees"] > 0 and static["proximal_rotation_residual_degrees"] > 0
    # Candidate moving twice as far as the reference is penalized; moving like it is not.
    r2 = arm_at(2)
    double = motion_residual(r0, r2, r0, r1)
    assert double["chain_direction_residual_degrees"] > 0
    assert motion_residual(r1, r2, r1, r2)["bend_side_residual_degrees"] < 1e-9


def test_factorial_contrasts_and_interactions():
    additive = {f"c{d}{r}{s}": 1.0 + 2.0 * int(d) - 0.5 * int(r) - 0.25 * int(s) for d in "01" for r in "01" for s in "01"}
    assert set(direction_penalty(additive).values()) == {2.0}
    ch = penalty_changes(additive)
    assert ch == {"reach": 0.0, "side": 0.0, "both": 0.0, "three_way_interaction": 0.0}
    assert ownership_effects(additive)["e1"] == {"reach": -0.5, "side": -0.25, "both": -0.75, "reach_x_side": 0.0}
    # E1 penalty disappears only when reach AND side are GT-owned -> pure interaction.
    y = dict(additive)
    y["c111"] -= 2.0
    assert direction_penalty(y)["gtr_gts"] == 0.0
    assert penalty_changes(y) == {"reach": 0.0, "side": 0.0, "both": -2.0, "three_way_interaction": -2.0}


def test_sequence_aware_contrast_accounting():
    import run_framepose_visual_arm_ownership as run
    items = []
    for s in range(6):
        for k in range(30):
            y = {n: 0.0 for n in (*FACTORIAL, *DIAGNOSTICS)}
            y["c100"] = 1.0 + 0.1 * s
            y["c110"] = 0.5 + 0.1 * s
            items.append({"sequence_id": f"s{s}", "candidates": {n: {"m": v} for n, v in y.items()}})
    fac = run._factorial(items, "m")
    assert fac["direction_penalty"]["h0r_h0s"]["mean"] == pytest.approx(1.25)
    assert fac["penalty_change"]["reach"]["mean"] == pytest.approx(-0.5)
    assert fac["penalty_change"]["reach"]["sequences_negative"] == 6
    ci = fac["penalty_change"]["reach"]["bootstrap"]["sequence_cluster"]["frame_weighted_ci95"]
    assert ci[0] == pytest.approx(-0.5) and ci[1] == pytest.approx(-0.5)
    assert fac["gt_f_reduction_vs_e1"]["mean"] == pytest.approx(-1.25)
    assert run._factorial(items, "m") == fac  # deterministic bootstrap


def test_runner_contracts_static():
    run = (ROOT / "scripts/run_framepose_visual_arm_ownership.py").read_text()
    for check in ("STOP: Worklog 76 reproduction failed", "STOP: oracle-geometry contract failed", "identity mismatch",
                  "frozen H0 test predictions differ from Worklog 76", "pairs_identical_to_w76", "thresholds_gt_p99",
                  "build_h0_lifted_sequence", "current_root_orientation", "TOL_CM = 1e-9", "g1[\"pairs\"] == 5504"):
        assert check in run
    body = run.split("def main():", 1)[1]
    assert body.index("STOP: Worklog 76 reproduction failed") < body.index("STOP: oracle-geometry contract failed") \
        < body.index("_factorial(common")
    # Thresholds come from Worklog 76, never re-chosen; decisions are declared before main().
    assert 'thresholds = w76_temporal["thresholds_gt_p99"]' in body and run.index("DECISION = {") < run.index("def main():")
    assert not re.search(r"\.backward\(|optimizer|train\(|checkpoint\.pt|smooth", run)
    module = (ROOT / "src/retarget/framepose_visual_arm_ownership.py").read_text().split('"""', 2)[2]
    assert not re.search(r"3dpw|pkl|oracle_sequence|load_bank|rows_heldout|smooth", module)
    for path in sorted((ROOT / "src").rglob("*.py")):
        if path.name != "framepose_visual_arm_ownership.py":
            assert "framepose_visual_arm_ownership" not in path.read_text(), path
