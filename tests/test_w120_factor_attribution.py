"""W120 focused tests: matched four-factor parity (G0/G1 wrist target x F0/F1 flex floor), zero-budget contract, unequal budget,
frozen error-class precedence incl. combined errors (exhaustive vs the actual W119 decide_y), source/rig/U/pin/DOF equality
across variants, pin preservation, target reach / floor effect, factorial contrasts and interaction, footprint/display truth,
no fallback, negative-control expectation consistency (Python-only test rig; no Blender needed)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src"), str(ROOT / "tests")]
import w116_semantic_pin_base as S  # noqa: E402
import w118_transition_edit_base as XB  # noqa: E402
import w119_endpoint_transition_base as YB  # noqa: E402
import w120_factor_attribution_base as FB  # noqa: E402
from test_w118_transition_edit import (AUTH, ARMPTS, BREST, N, case, evals, full, wrist)  # noqa: E402

FR = list(range(N))


def setup(holds=((10, 17),), m_max=XB.M_MAX):
    r, P, U, pin, base = case(holds)
    xv, xi = XB.transition_edit(r, U, base, pin, FR, AUTH, "lower", m_max=m_max)
    margins = YB.margins_of(xi)
    xt = {k: xv[k] for k in xi["planned_frames"] if k not in set(pin)}
    out = {f: FB.factor_transition(r, U, base, pin, FR, AUTH, "upper", "lower", margins, xi["m_max"], f, x_vals=xt) for f in FB.FACTORS}
    return r, P, U, pin, base, xv, xi, margins, xt, out


def _eq(a, b):
    return set(a) == set(b) and all(all(np.array_equal(np.asarray(a[k][c]), np.asarray(b[k][c])) for c in a[k]) for k in a)


# ------------------------------------------------------------------------------------- factor parity ---
def test_g1f1_is_bit_identical_to_w119_y():
    r, P, U, pin, base, xv, xi, margins, xt, out = setup()
    yv, yi = YB.endpoint_transition(r, U, base, pin, FR, AUTH, "upper", "lower", margins)
    gv, gi = out["G1F1"]
    assert _eq(gv, yv) and gi["frames"] == yi["frames"] and gi["budget_frames"] == yi["budget_frames"]


def test_four_variants_share_domain_pins_authorisation_and_budget():
    r, P, U, pin, base, xv, xi, margins, xt, out = setup(holds=((4, 7), (18, 21)))
    for f, (v, i) in out.items():
        assert i["request_status"] == "SOLVED" and i["fallback"] is None
        assert i["planned_frames"] == xi["planned_frames"] and i["budget_frames"] == xi["budget_frames"] and i["margins"] == margins
        assert i["solver_params_digest"] == FB.SOLVER_PARAMS_DIGEST
        for k in pin:
            assert all(np.array_equal(v[k][c], base[k][c]) for c in base[k])
        for k in set(v) - set(pin):
            assert np.array_equal(v[k]["base"], U[k]["base"]) and np.array_equal(v[k]["__object__"], U[k]["__object__"])


def test_variants_are_deterministic():
    a = setup()[-1]
    b = setup()[-1]
    for f in FB.FACTORS:
        assert _eq(a[f][0], b[f][0]) and a[f][1]["frames"] == b[f][1]["frames"]


def test_g0_targets_the_x_wrist_and_g1_the_world_guide():
    r, P, U, pin, base, xv, xi, margins, xt, out = setup()
    v, i = out["G0F0"]
    for k, fi in i["frames"].items():
        k = int(k)
        assert np.linalg.norm(np.array(fi["guide"]) - wrist(r, xv[k])) < 1e-12
        if fi["status"] == "GUIDE_REACHED":
            assert np.linalg.norm(wrist(r, v[k]) - wrist(r, xv[k])) < 1e-6
    v, i = out["G1F0"]
    rel = next(s for s in i["sides"] if s["side"] == "release" and s["margin"] > 0)
    b = rel["boundary"]
    cb = wrist(r, base[b]) - wrist(r, U[b])
    for k in rel["frames"]:
        fi = i["frames"][str(k)]
        assert np.allclose(np.array(fi["guide"]), wrist(r, U[k]) + fi["weight"] * cb, atol=1e-12)


def test_floor_factor_only_changes_the_reach_clip():
    r, P, U, pin, base, xv, xi, margins, xt, out = setup()
    for f in ("G0F0", "G1F0"):
        for fi in out[f][1]["frames"].values():
            assert fi["flex_floor_deg"] == S.ELBOW_FLEX_LIMIT_DEG[0] and fi["status"] != "GUIDE_FLEX_FLOOR_CLAMPED"
    for f in ("G0F1", "G1F1"):
        for fi in out[f][1]["frames"].values():
            assert fi["flex_deg"] >= fi["flex_floor_deg"] - 1e-6
    # identical guides under the same G, regardless of F
    assert all(out["G0F0"][1]["frames"][k]["guide"] == out["G0F1"][1]["frames"][k]["guide"] for k in out["G0F0"][1]["frames"])
    assert all(out["G1F0"][1]["frames"][k]["guide"] == out["G1F1"][1]["frames"][k]["guide"] for k in out["G1F0"][1]["frames"])


def test_unreachable_target_without_floor_clamps_to_straight_and_is_disclosed():
    r, P, U, pin, base = case()
    xv, xi = XB.transition_edit(r, U, base, pin, FR, AUTH, "lower")

    def push(k, Sh, G, Lsum):
        return Sh + (G - Sh) / np.linalg.norm(G - Sh) * (Lsum + 0.05)
    v, i = FB.factor_transition(r, U, base, pin, FR, AUTH, "upper", "lower", YB.margins_of(xi), 8, "G1F0", guide_override=push)
    assert i["status_counts"]["GUIDE_UNREACHABLE_OUTER"] == len(i["frames"]) > 0 and i["request_status"] == "SOLVED"
    assert all(fi["guide_shortfall_fk_m"] >= 0.05 - 1e-9 and fi["flex_deg"] < 0.1 for fi in i["frames"].values())


def test_g0_reference_missing_is_a_status_not_a_fallback():
    r, P, U, pin, base, xv, xi, margins, xt, out = setup()
    v, i = FB.factor_transition(r, U, base, pin, FR, AUTH, "upper", "lower", margins, 8, "G0F0", x_vals={})
    assert i["request_status"] == "G0_REFERENCE_MISSING" and i["g0_reference_missing_frames"]


def test_missing_dof_rejects_every_variant():
    r, P, U, pin, base, xv, xi, margins, xt, out = setup()
    for f in FB.FACTORS:
        v, i = FB.factor_transition(r, U, base, pin, FR, ["lower"], "upper", "lower", margins, 8, f, x_vals=xt)
        assert i["request_status"] == "MISSING_AUTHORIZED_DOF" and set(v) == set(pin)


# ------------------------------------------------------------------------------- zero budget / fairness ---
def test_zero_budget_request_has_identical_empty_budget_in_every_variant():
    r, P, U, pin, base, xv, xi, margins, xt, out = setup(m_max=0)
    assert xi["budget_frames"] == sorted(pin)
    for f, (v, i) in out.items():
        assert i["budget_frames"] == xi["budget_frames"] == i["planned_frames"] and set(v) == set(pin)
    _, yi = YB.endpoint_transition(r, U, base, pin, FR, AUTH, "upper", "lower", margins)
    assert yi["budget_frames"] != xi["budget_frames"]        # the W119 Z_A defect, reproduced


def _rec(**kw):
    r = {"parent_id": "p", "rig_id": "R", "U_state_id": "u", "constraint_digests": ["c"], "pin_frames": [1, 2], "auth": AUTH,
         "pin_values_digest": "v", "margins": {"1:2:entry": 3}, "budget_frames": [0, 1, 2, 3], "solver_params_digest": FB.SOLVER_PARAMS_DIGEST}
    r.update(kw)
    return r


def test_fairness_rejects_each_mismatch_and_unequal_budget():
    F = _rec(margins=None, budget_frames=None)
    assert FB.fairness_factor(F, _rec(), _rec())["ok"]
    for want, kw in {"SEARCH_BUDGET_MISMATCH:Y": dict(budget_frames=[1, 2]), "MARGIN_BUDGET_MISMATCH:Y": dict(margins={"1:2:entry": 2}),
                     "U_MISMATCH:Y": dict(U_state_id="u2"), "RIG_IDENTITY_MISMATCH:Y": dict(rig_id="R2"),
                     "PIN_CONSTRAINT_MISMATCH:Y": dict(constraint_digests=["d"]), "PARENT_MISMATCH:Y": dict(parent_id="q"),
                     "AUTH_MISMATCH:Y": dict(auth=["upper"]), "PIN_VALUES_DIFFER_FROM_F:Y": dict(pin_values_digest="w")}.items():
        res = FB.fairness_factor(F, _rec(), _rec(**kw))
        assert not res["ok"] and want in res["errors"], (want, res)
    assert "SOLVER_PARAMS_MISMATCH:Y" in FB.fairness_factor(F, _rec(), _rec(), "other")["errors"]


def test_cross_factor_identity():
    assert FB.cross_factor_identity({f: _rec() for f in FB.FACTORS})["ok"]
    bad = {f: _rec() for f in FB.FACTORS} | {"G1F0": _rec(budget_frames=[0, 1, 2])}
    assert not FB.cross_factor_identity(bad)["ok"]
    assert not FB.cross_factor_identity({f: _rec() for f in FB.FACTORS[:3]})["ok"]


# ---------------------------------------------------------------------------------------- precedence ---
def test_precedence_table_matches_actual_decide_y_exhaustively():
    res = FB.precedence_exhaustive_check()
    assert res["n"] > 100000 and res["n_mismatches"] == 0, res["mismatches"][:3]


def test_combined_error_precedence_cases():
    P = lambda **c: FB.predict_decision(c)["failure_class"]  # noqa: E731
    assert P(fairness_bad=True, eval_pin=True) == "CONTRACT_VIOLATION"                 # W119 NY_PIN1: tamper + F mismatch
    assert P(nonconv=True) == "SOLVER_FAILURE"                                         # isolated nonconvergence
    assert P(nonconv=True, fairness_bad=True, eval_pin=True) == "SOLVER_FAILURE"
    assert P(nonconv=True, fairness_bad=True) == "CONTRACT_VIOLATION"
    assert P(nonconv=True, eval_scope=True, fp_channel=True) == "SOLVER_FAILURE"      # E == {NON_CONVERGED}
    assert P(pin_status="UNREACHABLE", fairness_bad=True) == "INFEASIBLE_REQUEST"
    assert P(eval_pin=True) == "SOLVER_FAILURE"                                        # corrupted pin, fairness valid
    assert P(display_bad=True) == "CONTRACT_VIOLATION"
    for c in ({"fairness_bad": True, "eval_pin": True}, {"nonconv": True}, {"eval_scope": True, "fp_channel": True}):
        got = YB.decide_y(**FB.synth_decide_inputs(c))
        assert got["failure_class"] == FB.predict_decision(c)["failure_class"]


def test_declared_negative_expectations_equal_the_precedence_prediction():
    import w120_factor_attribution as M
    res = M.neg_expectation_consistency()
    assert res["all_ok"], [k for k, v in res["controls"].items() if not v["ok"]]


# ----------------------------------------------------------------------------------- output / footprint ---
def _decide(r, U, P, pin, v, i, T, fairness=None, display=None):
    Y, Ue, Pe = evals(r, full(U, v)), evals(r, U), evals(r, P)
    fp = XB.derive_footprint(Y, Ue, FR, pin, ARMPTS, wrist=("lower", "tail"), elbow=("upper", "tail"))
    dom = S.domain_points(r, ["base", "upper", "lower"])
    ev = []
    for k in i["planned_frames"]:
        pins = [{"bone": "lower", "point": "tail", "target": T, "semantic_joint": "w"}] if k in pin else []
        ev.append(S.evaluate_candidate(Y[k], Ue[k], Pe[k], pins, {"vs_parent": sorted(set(AUTH) | {"base"}), "vs_control": AUTH},
                                       dom, BREST, 1.0, chain=("upper", "lower")))
    return YB.decide_y([], ["SOLVED"] * len(pin), ev, fp, i["budget_frames"], i["planned_frames"], AUTH, i, fairness=fairness,
                       display_check=display), fp


def test_every_variant_accepts_with_truthful_footprint_and_rejects_corruption():
    r, P, U, pin, base, xv, xi, margins, xt, out = setup()
    T = wrist(r, P[10])
    for f, (v, i) in out.items():
        d, fp = _decide(r, U, P, pin, v, i, T)
        assert d["decision"] == "ACCEPT", (f, d)
        assert set(fp["frames_changed"]) <= set(i["planned_frames"]) and set(fp["channel_frames"]) <= set(AUTH)
        spec = XB.display_spec(fp, FR, pin, i["budget_frames"], i["planned_frames"], "sid", "pid", "PROPOSED", "t", f)
        assert XB.check_display(spec, evals(r, full(U, v)), evals(r, U), FR, pin, "sid")["ok"]
    v, i = out["G0F1"]
    bad = {k: dict(x) for k, x in v.items()}
    bad[12] = dict(bad[12], lower=S.q_mul(bad[12]["lower"], S.q_axis_angle([0, 0, 1], math.radians(2))))
    d = _decide(r, U, P, pin, bad, i, T, fairness={"ok": False, "errors": ["PIN_VALUES_DIFFER_FROM_F:Y"]})[0]
    assert d["decision"] == "REJECT" and d["failure_class"] == "CONTRACT_VIOLATION" and "EVALUATED_PIN_UNSATISFIED" in d["reasons"]
    v, i = FB.factor_transition(r, U, base, pin, FR, AUTH, "upper", "lower", margins, 8, "G0F0", x_vals=xt, fault="skip_lower_aim")
    d = _decide(r, U, P, pin, v, i, T, fairness={"ok": True, "errors": []})[0]
    assert d == {"decision": "REJECT", "failure_class": "SOLVER_FAILURE", "reasons": ["NON_CONVERGED"]}


def test_forced_alternating_pole_is_flagged_as_branch_flip():
    r, P, U, pin, base, xv, xi, margins, xt, out = setup()

    def mirror(k, Sh, G, R):
        if k % 2 == 0:
            return R
        t = (G - Sh) / np.linalg.norm(G - Sh)
        rv = R - Sh
        return Sh + 2.0 * float(np.dot(rv, t)) * t - rv
    v, i = FB.factor_transition(r, U, base, pin, FR, AUTH, "upper", "lower", margins, 8, "G1F1", pole_override=mirror)
    assert i["flag_counts"]["BRANCH_FLIP_VS_PREVIOUS"] >= 1 and i["request_status"] == "SOLVED"


# ------------------------------------------------------------------------------------------ contrasts ---
def _q(b1, flips=(), straight=(), d2=0.01, d3=2.0):
    return {"B1_boundary_wrist_comp_step_peak_m": b1, "introduced_flip_transitions": list(flips), "A4_introduced_near_straight_frames": list(straight),
            "D2_elbow_comp_step_peak_m": d2, "D3_arm_rot_step_excess_peak_deg": d3, "boundary_raw_wrist_step_peak_m": 0.05,
            "cost_mean_elbow_dev_from_U_m": 0.01, "cost_peak_elbow_dev_from_U_m": 0.02, "min_flex_deg": 10.0}


def test_factorial_contrasts_and_interaction():
    q = {"G0F0": _q(0.10), "G0F1": _q(0.06), "G1F0": _q(0.09, flips=[7]), "G1F1": _q(0.03)}
    fx = FB.factorial(q)
    c = fx["contrasts"]
    assert abs(c["floor_at_G0"]["delta"]["B1_boundary_wrist_comp_step_peak_m"] + 0.04) < 1e-12
    assert c["floor_at_G0"]["improves"]["meaningful_improvement"] and fx["FLOOR_primary"]["improvement"]
    assert not c["guide_at_F0"]["improves"]["meaningful_improvement"] and c["guide_at_F0"]["new_flips"] == [7]
    assert fx["GUIDE_primary"]["new_articulation_defect"]
    assert abs(fx["interaction"]["delta"]["B1_boundary_wrist_comp_step_peak_m"] - (0.03 - 0.09 - 0.06 + 0.10)) < 1e-12
    assert fx["interaction"]["B1_material"] and fx["interaction"]["sign"].startswith("negative")
    assert c["guide_at_F1"]["improves"]["meaningful_improvement"]
    # reverse direction marks material worsening
    fx2 = FB.factorial({"G0F0": _q(0.03), "G0F1": _q(0.03), "G1F0": _q(0.08), "G1F1": _q(0.03)})
    assert fx2["GUIDE_primary"]["material_worsening"]


def test_endpoint_jumps_and_variant_diff_on_evaluated_fixture():
    r, P, U, pin, base, xv, xi, margins, xt, out = setup()
    Ue = evals(r, U)
    v = out["G1F1"][0]
    Ye = evals(r, full(U, v))
    q = XB.quality(Ye, Ue, FR, pin, "upper", "lower", AUTH)
    j = FB.endpoint_jumps(q, XB.pin_runs(pin), margins)["10:17"]
    assert j["entry_jump_comp_m"] == q["per_transition"]["10"]["wrist_comp_step_m"]
    assert j["release_jump_raw_U_m"] == q["per_transition"]["18"]["control_wrist_step_m"]
    tf = sorted(set(xi["planned_frames"]) - set(pin))
    same = FB.variant_diff(Ye, Ye, tf, "upper", "lower")
    assert same["max_wrist_m"] == 0.0 and same["opposite_branch_frames"] == []
    d = FB.variant_diff(evals(r, full(U, out["G0F0"][0])), evals(r, full(U, xv)), tf, "upper", "lower")
    assert d["max_wrist_m"] < 1e-6        # G0F0 reaches X's wrist; its elbow may differ (reported, not attributed to G)
