import math
import re
from pathlib import Path

import numpy as np
import pytest

from retarget.axis_utils import quaternion_from_axis_angle
from retarget.framepose_ik_gt_attribution import measure_chain
from retarget.framepose_joint_factor_attribution import FACTORIAL, effects, flat_effects, owned
from retarget.framepose_target_rest import RestBone, TargetRigRestPose
from retarget.framepose_two_bone_ik import TwoBoneIkChain, target_chain_geometry
from retarget.framepose_visual_full_arm_ik import full_arm

ROOT = Path(__file__).resolve().parent.parent
I4 = ((1., 0., 0., 0.), (0., 1., 0., 0.), (0., 0., 1., 0.), (0., 0., 0., 1.))
ALIGN = quaternion_from_axis_angle((0.1, 0.2, 0.97), 2.9)


def _target():
    l1, l2 = 27.771168, 27.251084
    s, e, w = (0., 0., 140.), (0., l1, 140.), (0., l1, 140. - l2)

    def bone(n, p, h, t):
        return RestBone(n, p, I4, h, t)
    rest = TargetRigRestPose("rig", {"body": bone("body", None, (0, 0, 0), (0, 0, 1)), "upper": bone("upper", "body", s, e),
                                     "lower": bone("lower", "upper", e, w), "hand": bone("hand", "lower", w, (w[0], w[1], w[2] - 5))},
                             {"authority": "blender_imported_armature", "fixture": "w81"})
    return rest, target_chain_geometry(rest, TwoBoneIkChain("arm", "s", "e", "w", "upper", "lower", "hand"))


def _measure(theta, bend, forward, yaw=0.5):
    t, b = math.radians(theta), math.radians(bend)
    e = np.array([0.5 * math.cos(t), 0.5 * forward, -0.5 * math.sin(t)])
    l = np.array([math.cos(t + b), -forward, -math.sin(t + b)])
    w = e + 0.5 * l / np.linalg.norm(l)
    return measure_chain((0.0, 0.0, 0.0), tuple(e), tuple(w), yaw, ALIGN)


def test_factor_assignment_is_complete_and_corners_are_the_models():
    rest, target = _target()
    j0, j1 = _measure(30, 50, 0.3), _measure(40, 100, -0.5)
    assert set(FACTORIAL) == {f"v{d}{r}{s}" for d in "01" for r in "01" for s in "01"}
    m, d = owned(j0, j1, "0", "0", "0")
    assert m is j0 and d == j0.direction
    assert full_arm("v111", *owned(j0, j1, "1", "1", "1"), target, rest).end == full_arm("j1", j1, j1.direction, target, rest).end
    assert full_arm("v111", *owned(j0, j1, "1", "1", "1"), target, rest).mid == full_arm("j1", j1, j1.direction, target, rest).mid
    for name, (dd, r, s) in FACTORIAL.items():
        meas, direction = owned(j0, j1, dd, r, s)
        src = {"0": j0, "1": j1}
        assert meas.reach_fraction == src[r].reach_fraction and meas.side == src[s].side and direction == src[dd].direction
        arm = full_arm(name, meas, direction, target, rest)
        assert arm.status == "known" and arm.integrity["ik_to_fk_residual"] < 1e-9
        assert abs(float(np.dot(arm.side, arm.direction))) < 1e-12  # Worklog 62 perpendicularization
        assert np.linalg.norm(arm.upper) == pytest.approx(target.proximal_length)
    with pytest.raises(ValueError):
        owned(j0, j1, "2", "0", "0")


def test_degenerate_side_source_propagates_rest_plane_fallback():
    rest, target = _target()
    j0 = _measure(30, 50, 0.3)
    straight = measure_chain((0.0, 0.0, 0.0), (0.3, 0.1, -0.1), (0.54, 0.18, -0.18), 0.5, ALIGN)
    arm = full_arm("v001", *owned(j0, straight, "0", "0", "1"), target, rest)
    assert straight.degenerate and arm.side_status == "degenerate_rest_plane_fallback"
    arm = full_arm("v110", *owned(j0, straight, "1", "1", "0"), target, rest)
    assert arm.side_status == "observed" and arm.reach_fraction == straight.reach_fraction


def test_factorial_contrasts_additive_and_interaction_cases():
    add = {f"v{d}{r}{s}": 1.0 + 3.0 * int(d) - 1.0 * int(r) + 0.5 * int(s) for d in "01" for r in "01" for s in "01"}
    e = effects(add)
    assert e["total"] == pytest.approx(2.5)
    assert e["main"] == pytest.approx({"direction": 3.0, "reach": -1.0, "side": 0.5})
    assert e["swap_from_j0"] == e["revert_from_j1"] == pytest.approx(e["main"])
    assert all(abs(v) < 1e-12 for v in e["interaction"].values())
    y = dict(add)
    y["v111"] += 2.0  # pure three-way synergy
    e = effects(y)
    assert e["interaction"]["three_way"] == pytest.approx(2.0)
    assert e["swap_from_j0"]["direction"] == pytest.approx(3.0) and e["revert_from_j1"]["direction"] == pytest.approx(5.0)
    assert e["interaction"]["direction_x_reach"] == pytest.approx(1.0)
    flat = flat_effects(y)
    assert flat["total"] == pytest.approx(4.5) and flat["interaction:three_way"] == pytest.approx(2.0) and len(flat) == 14


def test_owner_rule_and_static_contracts():
    import run_framepose_joint_factor_attribution as run
    items = []
    for s in range(6):
        for k in range(20):
            y = {f"v{d}{r}{q}": 2.0 * int(d) + 0.1 * int(r) + 0.01 * s + 0.001 * k for d in "01" for r in "01" for q in "01"}
            items.append({"sequence_id": f"s{s}", "candidates": {v: {"m": val} for v, val in y.items()}})
    owner = run._owner(run._effect_block(items, "m"))
    assert owner["owner"] == "direction" and owner["total_significant"] and owner["main_share"]["direction"] == pytest.approx(2 / 2.1)
    src = (ROOT / "scripts/run_framepose_joint_factor_attribution.py").read_text()
    for check in ("STOP: Worklog 80 parity failed", "STOP: factor assignment / geometry contract failed", "identity mismatch",
                  'g1["pairs"] == 5504', "STOP: jump thresholds differ from Worklog 80"):
        assert check in src
    body = src.split("def main():", 1)[1]
    assert body.index("STOP: Worklog 80 parity failed") < body.index("STOP: factor assignment") < body.index("static_effects = ")
    assert src.index("DECISION = {") < src.index("def main():")
    assert not re.search(r"\.backward\(|optimizer|smooth|torch|load_state_dict", src)
    module = (ROOT / "src/retarget/framepose_joint_factor_attribution.py").read_text().split('"""', 2)[2]
    code = "\n".join(line for line in module.splitlines() if not line.startswith("from "))
    assert not re.search(r"\bgt\b|gt_|oracle|3dpw|load_bank", code, re.IGNORECASE)  # no GT source in the factors
    for path in sorted((ROOT / "src").rglob("*.py")):
        if path.name != "framepose_joint_factor_attribution.py":
            assert "framepose_joint_factor_attribution" not in path.read_text(), path
