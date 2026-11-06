"""W119 focused tests: per-request baseline isolation, family-shared contamination detection, A/B fairness (equal parent /
U / rig / pin / margin), endpoint world-space guide candidate Y (reach, flex floor, bend-plane ambiguity, nonconvergence,
missing DOF), hard-pin preservation, footprint/display reuse, decision and the predeclared Y-vs-X rule
(Python-only test rig; no Blender needed)."""
from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src"), str(ROOT / "tests")]
import w116_semantic_pin_base as S  # noqa: E402
import w118_transition_edit_base as XB  # noqa: E402
import w119_endpoint_transition_base as YB  # noqa: E402
from test_w118_transition_edit import (AUTH, ARMPTS, BREST, N, case, evals, fake_eval, full, rig, seq, wrist)  # noqa: E402,F401

FR = list(range(N))


def xy(holds=((10, 17),)):
    r, P, U, pin, base = case(holds)
    xv, xi = XB.transition_edit(r, U, base, pin, FR, AUTH, "lower")
    margins = YB.margins_of(xi)
    yv, yi = YB.endpoint_transition(r, U, base, pin, FR, AUTH, "upper", "lower", margins)
    return r, P, U, pin, base, xv, xi, yv, yi, margins


def elbow(r, v):
    return r.world(r.fk(v), "upper", "tail")


# ------------------------------------------------------------------------------------------ isolation ---
def test_isolated_baseline_passes_and_family_shared_state_is_rejected():
    r, P, U, pin, base = case(holds=((4, 7), (18, 21)))
    seg_a = [k for k in pin if k <= 7]
    Fr = full(U, {k: base[k] for k in seg_a})
    shared = full(U, base)                                  # family-shared: both segments' pins
    ok = YB.isolation_check_values(Fr, U, seg_a, FR)
    assert ok["ok"] and set(ok["changed_frames"]) <= set(seg_a)
    bad = YB.isolation_check_values(shared, U, seg_a, FR)
    assert not bad["ok"] and bad["reason"] == "BASELINE_NOT_ISOLATED" and set(bad["changed_outside_pin"]) == set(pin) - set(seg_a)
    fp_ok = XB.derive_footprint(evals(r, Fr), evals(r, U), FR, seg_a, ARMPTS, wrist=("lower", "tail"), elbow=("upper", "tail"))
    fp_bad = XB.derive_footprint(evals(r, shared), evals(r, U), FR, seg_a, ARMPTS, wrist=("lower", "tail"), elbow=("upper", "tail"))
    assert YB.isolation_check_footprint(fp_ok, seg_a)["ok"]
    b = YB.isolation_check_footprint(fp_bad, seg_a)
    assert not b["ok"] and b["runs_outside_pin"] == [[18, 21]]


def test_single_sibling_frame_change_is_detected():
    r, P, U, pin, base = case(holds=((4, 7), (18, 21)))
    seg_a = [k for k in pin if k <= 7]
    Fr = full(U, {k: base[k] for k in seg_a} | {19: base[19]})
    c = YB.isolation_check_values(Fr, U, seg_a, FR)
    assert not c["ok"] and c["changed_outside_pin"] == [19]


# ------------------------------------------------------------------------------------------- fairness ---
def _rec(**kw):
    r = {"parent_id": "p", "rig_id": "R", "U_state_id": "u", "constraint_digests": ["c"], "pin_frames": [1, 2], "auth": AUTH,
         "pin_values_digest": "v", "margins": {"1:2:entry": 3}, "budget_frames": [0, 1, 2, 3]}
    r.update(kw)
    return r


def test_ab_fairness_accepts_equal_and_rejects_each_mismatch():
    assert YB.ab_fairness({"F": _rec(), "X": _rec(), "Y": _rec()})["ok"]
    cases = {"PARENT_MISMATCH:Y": dict(parent_id="q"), "RIG_IDENTITY_MISMATCH:Y": dict(rig_id="R2"),
             "U_MISMATCH:Y": dict(U_state_id="u2"), "PIN_CONSTRAINT_MISMATCH:Y": dict(constraint_digests=["d"]),
             "AUTH_MISMATCH:Y": dict(auth=["upper"]), "MARGIN_BUDGET_MISMATCH:Y": dict(margins={"1:2:entry": 2}),
             "PIN_VALUES_DIFFER_FROM_F:Y": dict(pin_values_digest="w")}
    for want, kw in cases.items():
        res = YB.ab_fairness({"F": _rec(), "X": _rec(), "Y": _rec(**kw)})
        assert not res["ok"] and want in res["errors"], (want, res)


def test_margins_of_reads_frozen_x_sides():
    r, P, U, pin, base, xv, xi, yv, yi, margins = xy()
    assert set(margins) == {"10:17:entry", "10:17:release"}
    assert all(margins[f"10:17:{s['side']}"] == s["chosen_margin"] for s in xi["sides"])


# ------------------------------------------------------------------------------------------- candidate Y ---
def test_y_pins_untouched_matched_domain_authorised_only_and_deterministic():
    r, P, U, pin, base, xv, xi, yv, yi, margins = xy()
    yv2, yi2 = YB.endpoint_transition(r, U, base, pin, FR, AUTH, "upper", "lower", margins)
    assert all(all(np.array_equal(yv[k][c], yv2[k][c]) for c in yv[k]) for k in yv) and yi["frames"] == yi2["frames"]
    for k in pin:
        assert all(np.array_equal(yv[k][c], base[k][c]) for c in base[k])
    assert yi["planned_frames"] == xi["planned_frames"] and set(yv) == set(xv)       # identical edit domain to X
    for k in set(yv) - set(pin):
        assert np.array_equal(yv[k]["base"], U[k]["base"]) and np.array_equal(yv[k]["__object__"], U[k]["__object__"])
    assert yi["fallback"] is None and yi["request_status"] == "SOLVED"


def test_y_reaches_world_space_guide_and_compensation_follows_the_weight_profile():
    r, P, U, pin, base, xv, xi, yv, yi, margins = xy()
    for k, fi in yi["frames"].items():
        k = int(k)
        if fi["status"] == "GUIDE_REACHED":
            assert np.linalg.norm(wrist(r, yv[k]) - np.array(fi["guide"])) < 1e-6
    rel = next(s for s in yi["sides"] if s["side"] == "release" and s["margin"] > 0)
    b = rel["boundary"]
    cb = wrist(r, base[b]) - wrist(r, U[b])
    for k in rel["frames"]:
        fi = yi["frames"][str(k)]
        if fi["status"] == "GUIDE_REACHED":
            assert np.allclose(wrist(r, yv[k]) - wrist(r, U[k]), fi["weight"] * cb, atol=1e-6)


def test_y_compensation_steps_bounded_by_the_guide_profile_where_reached():
    r, P, U, pin, base, xv, xi, yv, yi, margins = xy()
    Y = full(U, yv)
    for s in yi["sides"]:
        if not s["margin"] or any(yi["frames"][str(k)]["status"] != "GUIDE_REACHED" for k in s["frames"]):
            continue
        m, b = s["margin"], s["boundary"]
        cb = float(np.linalg.norm(wrist(r, base[b]) - wrist(r, U[b])))
        seq_ = [b] + s["frames"] + [b + (m + 1) * (1 if s["side"] == "release" else -1)]
        c = [wrist(r, Y[k]) - wrist(r, U[k]) for k in seq_]
        steps = [float(np.linalg.norm(c[i] - c[i - 1])) for i in range(1, len(c))]
        assert max(steps) <= 1.5 / (m + 1) * cb + 1e-6


def test_flex_floor_prevents_straighter_than_both_endpoints():
    r, P, U, pin, base, xv, xi, yv, yi, margins = xy()
    for k, fi in yi["frames"].items():
        assert fi["flex_deg"] >= fi["flex_floor_deg"] - 1e-6


def test_unreachable_guide_is_clamped_disclosed_and_pins_still_untouched():
    r, P, U, pin, base, xv, xi, yv, yi, margins = xy()

    def push(k, Sh, G, Lsum):
        return Sh + (G - Sh) / np.linalg.norm(G - Sh) * (Lsum + 0.05)
    yv2, yi2 = YB.endpoint_transition(r, U, base, pin, FR, AUTH, "upper", "lower", margins, guide_override=push)
    assert yi2["status_counts"]["GUIDE_UNREACHABLE_OUTER"] == len(yi2["frames"]) > 0
    assert yi2["request_status"] == "SOLVED"
    for k in pin:
        assert all(np.array_equal(yv2[k][c], base[k][c]) for c in base[k])
    for k, fi in yi2["frames"].items():
        assert fi["guide_shortfall_fk_m"] >= 0.05 - 1e-9


def test_bend_plane_ambiguity_uses_continuity_pole_and_is_disclosed():
    r, P, U, pin, base, xv, xi, yv, yi, margins = xy()

    def on_axis(k, Sh, G, R):
        return Sh + 0.5 * (G - Sh)
    yv2, yi2 = YB.endpoint_transition(r, U, base, pin, FR, AUTH, "upper", "lower", margins, pole_override=on_axis)
    assert yi2["flag_counts"]["BEND_PLANE_AMBIGUOUS"] == len(yi2["frames"]) > 0
    assert yi2["flag_counts"]["BRANCH_FLIP_VS_PREVIOUS"] == 0 and yi2["request_status"] == "SOLVED"


def test_fault_injected_nonconvergence_is_reported_as_solver_failure():
    r, P, U, pin, base, xv, xi, yv, yi, margins = xy()
    yv2, yi2 = YB.endpoint_transition(r, U, base, pin, FR, AUTH, "upper", "lower", margins, fault="skip_lower_aim")
    assert yi2["request_status"] == "NON_CONVERGED"
    d = YB.decide_y([], ["SOLVED"] * len(pin), [], {"frames_changed": [], "channel_frames": {}, "outside_arm_points_changed": {},
                                                     "matrix_world_changed_frames": []}, yi2["budget_frames"], yi2["planned_frames"],
                    AUTH, yi2)
    assert d["decision"] == "REJECT" and d["failure_class"] == "SOLVER_FAILURE" and "NON_CONVERGED" in d["reasons"]


def test_missing_authorised_dof_is_rejected_not_fallen_back():
    r, P, U, pin, base, xv, xi, yv, yi, margins = xy()
    yv2, yi2 = YB.endpoint_transition(r, U, base, pin, FR, ["lower"], "upper", "lower", margins)
    assert yi2["request_status"] == "MISSING_AUTHORIZED_DOF" and set(yv2) == set(pin)
    d = YB.decide_y([], ["SOLVED"], [], None, [], [], ["lower"], yi2)
    assert d == {"decision": "REJECT", "failure_class": "INFEASIBLE_REQUEST", "reasons": ["Y_MISSING_AUTHORIZED_DOF"]}


def test_margin_exceeding_budget_is_not_silently_widened():
    r, P, U, pin, base, xv, xi, yv, yi, margins = xy()
    big = {k: v + 50 for k, v in margins.items()}
    _, yi2 = YB.endpoint_transition(r, U, base, pin, FR, AUTH, "upper", "lower", big)
    assert yi2["request_status"] == "MARGIN_EXCEEDS_AVAILABLE_BUDGET" and yi2["planned_frames"] == sorted(pin)


def test_zero_margins_equal_hard_pin_base():
    r, P, U, pin, base, xv, xi, yv, yi, margins = xy()
    yv2, yi2 = YB.endpoint_transition(r, U, base, pin, FR, AUTH, "upper", "lower", {k: 0 for k in margins})
    assert set(yv2) == set(pin) and yi2["planned_frames"] == sorted(pin)


# ---------------------------------------------------------------------------------- footprint / decision ---
def _decide_y(r, U, P, pin, yv, yi, T, display=None, fairness=None):
    Y, Ue, Pe = evals(r, full(U, yv)), evals(r, U), evals(r, P)
    fp = XB.derive_footprint(Y, Ue, FR, pin, ARMPTS, wrist=("lower", "tail"), elbow=("upper", "tail"))
    dom = S.domain_points(r, ["base", "upper", "lower"])
    ev = []
    for k in yi["planned_frames"]:
        pins = [{"bone": "lower", "point": "tail", "target": T, "semantic_joint": "w"}] if k in pin else []
        ev.append(S.evaluate_candidate(Y[k], Ue[k], Pe[k], pins, {"vs_parent": sorted(set(AUTH) | {"base"}), "vs_control": AUTH},
                                       dom, BREST, 1.0, chain=("upper", "lower")))
    return YB.decide_y([], ["SOLVED"] * len(pin), ev, fp, yi["budget_frames"], yi["planned_frames"], AUTH, yi,
                       display_check=display, fairness=fairness), fp


def test_y_accepts_with_truthful_footprint_and_rejects_tamper_and_unfair_comparison():
    r, P, U, pin, base, xv, xi, yv, yi, margins = xy()
    T = wrist(r, P[10])
    d, fp = _decide_y(r, U, P, pin, yv, yi, T)
    assert d["decision"] == "ACCEPT" and set(fp["frames_changed"]) <= set(yi["planned_frames"])
    assert set(fp["channel_frames"]) <= set(AUTH)
    bad = {k: dict(v) for k, v in yv.items()}
    bad[13] = dict(bad[13], lower=S.q_mul(bad[13]["lower"], S.q_axis_angle([0, 0, 1], math.radians(2))))
    assert _decide_y(r, U, P, pin, bad, yi, T)[0]["failure_class"] == "SOLVER_FAILURE"
    bad = {k: dict(v) for k, v in yv.items()}
    t = min(yi["planned_frames"])
    bad[t] = dict(bad[t], __object__=S.q_mul(U[t]["__object__"], S.q_axis_angle([0, 0, 1], 0.02)))
    d = _decide_y(r, U, P, pin, bad, yi, T)[0]
    assert d["decision"] == "REJECT" and "ROOT_MOVED" in d["reasons"]
    d = _decide_y(r, U, P, pin, yv, yi, T, fairness={"ok": False, "errors": ["U_MISMATCH:Y"]})[0]
    assert d["decision"] == "REJECT" and "AB_FAIRNESS:U_MISMATCH:Y" in d["reasons"]
    d = _decide_y(r, U, P, pin, yv, yi, T, display={"ok": False, "errors": ["CELL_MISMATCH"]})[0]
    assert d["decision"] == "REJECT" and "DISPLAY_FOOTPRINT_MISMATCH" in d["reasons"]


def test_display_truth_check_applies_to_y_and_flags_omitted_frame():
    r, P, U, pin, base, xv, xi, yv, yi, margins = xy()
    Y, Ue = evals(r, full(U, yv)), evals(r, U)
    fp = XB.derive_footprint(Y, Ue, FR, pin, ARMPTS, wrist=("lower", "tail"), elbow=("upper", "tail"))
    spec = XB.display_spec(fp, FR, pin, yi["budget_frames"], yi["planned_frames"], "sid", "pid", "PROPOSED", "t", "Y")
    assert XB.check_display(spec, Y, Ue, FR, pin, "sid")["ok"]
    s = copy.deepcopy(spec)
    k = next(f for f, c in s["cells"].items() if c == "AUTO_CHANGED")
    s["cells"][k] = "UNCHANGED"
    assert not XB.check_display(s, Y, Ue, FR, pin, "sid")["ok"]


def test_guide_residuals_are_computed_from_evaluated_frames():
    r, P, U, pin, base, xv, xi, yv, yi, margins = xy()
    Y = evals(r, full(U, yv))
    gr = YB.guide_residuals(Y, yi, "lower")
    assert set(gr) == set(yi["frames"])
    assert all(v["to_clamped_guide_m"] < 1e-6 for v in gr.values())


# --------------------------------------------------------------------------------------------- comparison ---
Q0 = {"B1_boundary_wrist_comp_step_peak_m": 0.10, "introduced_flip_transitions": [], "A4_introduced_near_straight_frames": [],
      "D2_elbow_comp_step_peak_m": 0.02, "D3_arm_rot_step_excess_peak_deg": 4.0, "boundary_raw_wrist_step_peak_m": 0.05}


def test_compare_y_predeclared_rule():
    c = YB.compare_y(Q0, dict(Q0, B1_boundary_wrist_comp_step_peak_m=0.079))
    assert c["meaningful_improvement"]
    assert not YB.compare_y(Q0, dict(Q0, B1_boundary_wrist_comp_step_peak_m=0.081))["meaningful_improvement"]   # < 20 %
    small = dict(Q0, B1_boundary_wrist_comp_step_peak_m=0.030)
    assert not YB.compare_y(small, dict(small, B1_boundary_wrist_comp_step_peak_m=0.021))["meaningful_improvement"]  # < 10 mm
    good = dict(Q0, B1_boundary_wrist_comp_step_peak_m=0.05)
    assert "NEW_BEND_FLIP" in YB.compare_y(Q0, dict(good, introduced_flip_transitions=[12]))["worse"]
    assert not YB.compare_y(dict(Q0, introduced_flip_transitions=[12]), dict(good, introduced_flip_transitions=[12]))["worse"]
    assert "NEW_NEAR_STRAIGHT" in YB.compare_y(Q0, dict(good, A4_introduced_near_straight_frames=[3]))["worse"]
    assert "WORSE_D2_elbow_comp_step_peak_m" in YB.compare_y(Q0, dict(good, D2_elbow_comp_step_peak_m=0.026))["worse"]
    assert not YB.compare_y(Q0, dict(good, D2_elbow_comp_step_peak_m=0.024))["worse"]
    assert "WORSE_D3_arm_rot_step_excess_peak_deg" in YB.compare_y(Q0, dict(good, D3_arm_rot_step_excess_peak_deg=6.5))["worse"]
    assert "WORSE_boundary_raw_wrist_step_peak_m" in YB.compare_y(Q0, dict(good, boundary_raw_wrist_step_peak_m=0.0626))["worse"]


def test_quality_attribution_separates_raw_and_compensation_steps():
    r, P, U, pin, base, xv, xi, yv, yi, margins = xy()
    Ue = evals(r, U)
    qY = XB.quality(evals(r, full(U, yv)), Ue, FR, pin, "upper", "lower", AUTH)
    qU = XB.quality(Ue, Ue, FR, pin, "upper", "lower", AUTH)
    assert qU["B1_boundary_wrist_comp_step_peak_m"] == 0.0
    assert qU["boundary_raw_wrist_step_peak_m"] == qU["boundary_control_raw_wrist_step_peak_m"]
    assert qY["boundary_control_raw_wrist_step_peak_m"] == qU["boundary_control_raw_wrist_step_peak_m"]
