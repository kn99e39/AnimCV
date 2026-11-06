"""W118 focused tests: pin domain vs automatic budget, automatic transition editor, actual footprint, display truth check,
decision/disclosure rejections (Python-only test rig; no Blender needed)."""
from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import w116_semantic_pin_base as S  # noqa: E402
import w118_transition_edit_base as XB  # noqa: E402


def _mat(R, t):
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = t
    return M.tolist()


def rig():
    Rz_up = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)
    Rx = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], float)
    bones = {"base": {"parent": None, "matrix_local": _mat(Rz_up, [0, 0, 0]), "head": [0, 0, 0], "tail": [0, 0, 0.5]},
             "neck": {"parent": "base", "matrix_local": _mat(Rz_up, [0, 0, 0.5]), "head": [0, 0, 0.5], "tail": [0, 0, 0.65]},
             "upper": {"parent": "base", "matrix_local": _mat(Rx, [0.12, 0, 0.45]), "head": [0.12, 0, 0.45], "tail": [0.42, 0, 0.45]},
             "lower": {"parent": "upper", "matrix_local": _mat(Rx, [0.42, 0, 0.45]), "head": [0.42, 0, 0.45], "tail": [0.67, 0, 0.45]}}
    return S.Rig({"rig_id": "T118", "bones": bones, "object": {"location": [0, 0, 0], "scale": [1.0] * 3,
                                                               "rest_rotation_wxyz": [1, 0, 0, 0]}})


BREST = {"base": {"length": 0.5, "parent": None}, "neck": {"length": 0.15, "parent": "base"},
         "upper": {"length": 0.3, "parent": "base"}, "lower": {"length": 0.25, "parent": "upper"}}
AUTH = ["upper", "lower"]
N = 30


def seq(n=N):
    out = {}
    for f in range(n):
        tau = f / (n - 1)
        out[f] = {"__object__": np.array([0, 0, 0, 1.0]),
                  "base": S.q_axis_angle([0, 1, 0], math.radians(10 * math.sin(2 * math.pi * 0.7 * tau))),
                  "upper": S.q_mul(S.q_axis_angle([0, 0, 1], math.radians(-20 + 25 * tau)),
                                   S.q_axis_angle([1, 0, 0], math.radians(15 * math.sin(2 * math.pi * 0.5 * tau)))),
                  "lower": S.q_axis_angle([0, 0, 1], math.radians(50 + 20 * math.sin(2 * math.pi * 0.9 * tau)))}
    return out


def edited(r, vals, deg=8.0):
    e = {"kind": "local_axis_rotation", "channels": ["base"], "axis_local": [1, 0, 0], "sign": 1}
    return {k: S.apply_edit(r, v, e, {"angle_rad": math.radians(deg)}) for k, v in vals.items()}


def wrist(r, v):
    return r.world(r.fk(v), "lower", "tail")


def fake_eval(r, vals):
    fk = r.fk(vals)
    world = {b: {"head": r.world(fk, b, "head").tolist(), "tail": r.world(fk, b, "tail").tolist()} for b in r.bones}
    stored = {c: [float(v[3]), float(v[0]), float(v[1]), float(v[2])] for c, v in vals.items()}
    M = np.eye(4)
    M[:3, :3] = fk["Robj"]
    return {"world": world, "stored_wxyz": stored, "matrix_world": M.tolist()}


def case(holds=((10, 17),)):
    r = rig()
    P = seq()
    U = edited(r, P)
    pin, base = [], {}
    for lo, hi in holds:
        T = wrist(r, P[lo])
        for k in range(lo, hi + 1):
            pin.append(k)
            st, v, _ = S.resolve_frame(r, U[k], [{"bone": "lower", "point": "tail", "target": T}], AUTH)
            assert st in ("SOLVED", "NO_CHANGE_NEEDED")
            base[k] = v
    return r, P, U, pin, base


def run(r, U, pin, base, m_max=XB.M_MAX):
    return XB.transition_edit(r, U, base, pin, list(range(N)), AUTH, "lower", m_max=m_max)


def full(U, xv):
    return {k: (xv[k] if k in xv else U[k]) for k in U}


def evals(r, vals):
    return {k: fake_eval(r, v) for k, v in vals.items()}


ARMPTS = {"upper.tail", "lower.head", "lower.tail"}


# --------------------------------------------------------------------------------------------- domains ---
def test_pin_domain_and_budget_are_separate_and_clipped():
    plan, budget = XB.budget_plan(list(range(10, 18)), list(range(N)), 8)
    assert plan[0]["entry_available"] == 8 and plan[0]["release_available"] == 8
    assert budget == list(range(2, 26)) and set(range(10, 18)) < set(budget)
    plan, budget = XB.budget_plan(list(range(0, 5)), list(range(N)), 8)
    assert plan[0]["entry_available"] == 0 and plan[0]["entry_limit"] == "CLIP_START"
    plan, budget = XB.budget_plan(list(range(25, 30)), list(range(N)), 8)
    assert plan[0]["release_available"] == 0 and plan[0]["release_limit"] == "CLIP_END"
    plan, budget = XB.budget_plan(list(range(6, 11)) + list(range(20, 25)), list(range(N)), 8)
    assert [p["run"] for p in plan] == [[6, 10], [20, 24]]
    assert plan[0]["release_available"] == 4 and plan[1]["entry_available"] == 4 and plan[0]["release_limit"] == "GAP_SPLIT"
    assert 15 not in budget                       # middle gap frame outside the budget
    plan, budget = XB.budget_plan(list(range(10, 18)), list(range(N)), 0)
    assert budget == list(range(10, 18))


def test_pin_runs_disjoint_representation():
    assert XB.pin_runs([3, 4, 5, 9, 10, 20]) == [[3, 5], [9, 10], [20, 20]]


# ------------------------------------------------------------------------------------------- editor ---
def test_pin_frames_untouched_only_authorised_channels_within_budget_and_deterministic():
    r, P, U, pin, base = case()
    xv, info = run(r, U, pin, base)
    xv2, info2 = run(r, U, pin, base)
    assert info["sides"] == info2["sides"] and all(all(np.array_equal(xv[k][c], xv2[k][c]) for c in xv[k]) for k in xv)
    for k in pin:
        assert all(np.array_equal(xv[k][c], base[k][c]) for c in base[k])
    assert set(xv) == set(info["planned_frames"]) and set(info["planned_frames"]) <= set(info["budget_frames"])
    for k in set(xv) - set(pin):
        assert np.array_equal(xv[k]["base"], U[k]["base"]) and np.array_equal(xv[k]["__object__"], U[k]["__object__"])
        assert not np.array_equal(xv[k]["upper"], U[k]["upper"])
    for s in info["sides"]:
        assert 0 <= s["chosen_margin"] <= s["available"]
        if not s["saturated"] and s["chosen_margin"] > 0:
            assert s["predicted_peak_step_m"][str(s["chosen_margin"])] <= XB.SIGMA_STEP_M
        if s["saturated"]:
            assert s["chosen_margin"] == s["available"]
    assert info["fallback"] is None


def test_offset_layer_is_continuous_toward_the_boundary():
    r, P, U, pin, base = case()
    xv, info = run(r, U, pin, base)
    ent = next(s for s in info["sides"] if s["side"] == "entry")
    devs = [float(np.linalg.norm(wrist(r, xv[k]) - wrist(r, U[k]))) for k in sorted(ent["frames"])]
    assert devs == sorted(devs)                    # fades out monotonically away from the pin entry
    assert devs[-1] < float(np.linalg.norm(wrist(r, base[ent["boundary"]]) - wrist(r, U[ent["boundary"]])))


def test_zero_budget_equals_hard_pin_base_only():
    r, P, U, pin, base = case()
    xv, info = run(r, U, pin, base, m_max=0)
    assert set(xv) == set(pin) and info["planned_frames"] == sorted(pin)


def test_no_boundary_offset_means_no_transition_frames():
    r = rig()
    P = seq()
    pin = list(range(10, 18))
    base = {k: dict(P[k]) for k in pin}           # pins already satisfied (NO_CHANGE_NEEDED values = U)
    xv, info = XB.transition_edit(r, P, base, pin, list(range(N)), AUTH, "lower")
    assert set(xv) == set(pin) and all(s["reason"] in ("NO_BOUNDARY_OFFSET", "NO_FRAMES_AVAILABLE") for s in info["sides"])


def test_q_pow_endpoints():
    d = S.q_axis_angle([0.3, 0.2, 0.9], 0.7)
    assert np.allclose(XB.q_pow(d, 1.0), d) and np.allclose(XB.q_pow(d, 0.0), [0, 0, 0, 1])
    assert np.allclose(XB.q_pow(-d, 0.5), XB.q_pow(d, 0.5))


# ---------------------------------------------------------------------------------- footprint + display ---
def test_footprint_detects_actual_frames_channels_and_disjoint_runs():
    r, P, U, pin, base = case(holds=((3, 7), (19, 23)))
    xv, info = run(r, U, pin, base)
    X, Ue = evals(r, full(U, xv)), evals(r, U)
    fp = XB.derive_footprint(X, Ue, list(range(N)), pin, ARMPTS, wrist=("lower", "tail"), elbow=("upper", "tail"))
    assert fp["frames_changed"] == sorted(xv)
    assert len(fp["runs"]) >= 2 and fp["runs"] == XB.runs_of(fp["frames_changed"])
    assert set(fp["channel_frames"]) <= set(AUTH) and not fp["outside_arm_points_changed"]
    assert XB.derive_footprint(Ue, Ue, list(range(N)), pin, ARMPTS, wrist=("lower", "tail"), elbow=("upper", "tail"))["frames_changed"] == []


def _display(r, U, pin, xv, info, status="PROPOSED"):
    X, Ue = evals(r, full(U, xv)), evals(r, U)
    fp = XB.derive_footprint(X, Ue, list(range(N)), pin, ARMPTS, wrist=("lower", "tail"), elbow=("upper", "tail"))
    spec = XB.display_spec(fp, list(range(N)), pin, info["budget_frames"], info["planned_frames"], "sid", "pid", status, "t", "X")
    return spec, X, Ue, fp


def test_display_truth_check_passes_and_rejects_false_or_missing_disclosure():
    r, P, U, pin, base = case(holds=((3, 7), (19, 23)))
    xv, info = run(r, U, pin, base)
    spec, X, Ue, fp = _display(r, U, pin, xv, info)
    fr = list(range(N))
    assert XB.check_display(spec, X, Ue, fr, pin, "sid")["ok"]
    s = copy.deepcopy(spec)                       # missing changed frame
    k = next(f for f, c in s["cells"].items() if c == "AUTO_CHANGED")
    s["cells"][k] = "UNCHANGED"
    assert "CELL_MISMATCH" in XB.check_display(s, X, Ue, fr, pin, "sid")["errors"]
    s = copy.deepcopy(spec)                       # collapsed to the envelope
    s["changed_runs"] = [s["envelope"]]
    assert "RUNS_MISMATCH" in XB.check_display(s, X, Ue, fr, pin, "sid")["errors"]
    s = copy.deepcopy(spec)                       # incomplete channel coverage
    s["channel_rows"].pop("lower")
    assert "CHANNEL_COVERAGE_MISMATCH" in XB.check_display(s, X, Ue, fr, pin, "sid")["errors"]
    s = copy.deepcopy(spec)                       # timeline from the solver's budget instead of the actual output
    s["cells"] = {str(f): ("PIN_CHANGED" if f in pin else "AUTO_CHANGED") if f in info["budget_frames"] else "UNCHANGED" for f in fr}
    if set(info["budget_frames"]) != set(info["planned_frames"]):
        assert not XB.check_display(s, X, Ue, fr, pin, "sid")["ok"]
    assert "STATE_ID_MISMATCH" in XB.check_display(spec, X, Ue, fr, pin, "other")["errors"]
    s = dict(spec, status="ACCEPTED")             # a rejected candidate shown as applied
    assert "STATUS_LABEL_DISAGREES_WITH_DECISION" in XB.check_display(s, X, Ue, fr, pin, "sid", decision="REJECT")["errors"]
    s = copy.deepcopy(spec)
    s["magnitude"][k]["wrist_mm"] += 1.0
    assert "MAGNITUDE_MISMATCH" in XB.check_display(s, X, Ue, fr, pin, "sid")["errors"]


def test_sub_resolution_changes_are_displayed_not_hidden():
    r = rig()
    P = seq()
    U = dict(P)
    V = dict(P)
    V[12] = dict(P[12], lower=S.q_mul(P[12]["lower"], S.q_axis_angle([0, 0, 1], math.radians(0.0005))))
    X, Ue = evals(r, V), evals(r, U)
    fp = XB.derive_footprint(X, Ue, list(range(N)), [], ARMPTS, wrist=("lower", "tail"), elbow=("upper", "tail"))
    spec = XB.display_spec(fp, list(range(N)), [], [], [], "s", "p", "PROPOSED", "t", "X")
    assert spec["cells"]["12"] == "AUTO_CHANGED" and spec["magnitude"]["12"]["sub_resolution"] is True
    assert 12 in fp["numeric_only_frames"] or 12 in fp["material_frames"]


# ------------------------------------------------------------------------------------------- decision ---
def _decide(r, U, P, pin, xv, info, T, auth=AUTH, planned=None, display=None, statuses=None):
    X, Ue, Pe = evals(r, full(U, xv)), evals(r, U), evals(r, P)
    fp = XB.derive_footprint(X, Ue, list(range(N)), pin, ARMPTS, wrist=("lower", "tail"), elbow=("upper", "tail"))
    dom = S.domain_points(r, ["base", "upper", "lower"])
    ev = []
    planned = info["planned_frames"] if planned is None else planned
    for k in planned:
        pins = [{"bone": "lower", "point": "tail", "target": T, "semantic_joint": "w"}] if k in pin else []
        ev.append(S.evaluate_candidate(X[k], Ue[k], Pe[k], pins, {"vs_parent": sorted(set(auth) | {"base"}), "vs_control": auth},
                                       dom, BREST, 1.0, chain=("upper", "lower")))
    return XB.decide_x([], statuses or ["SOLVED"] * len(pin), ev, fp, info["budget_frames"], planned, auth, display_check=display)


def test_accept_and_rejections_outside_budget_planned_channel_display_and_false_solved():
    r, P, U, pin, base = case()
    T = wrist(r, P[10])
    xv, info = run(r, U, pin, base)
    assert _decide(r, U, P, pin, xv, info, T)["decision"] == "ACCEPT"
    bad = {k: dict(v) for k, v in xv.items()}
    bad[0] = dict(U[0], upper=S.q_mul(U[0]["upper"], S.q_axis_angle([0, 0, 1], 0.02)))
    d = _decide(r, U, P, pin, bad, info, T)
    assert d["failure_class"] == "CONTRACT_VIOLATION" and "CHANGE_OUTSIDE_AUTOMATIC_BUDGET" in d["reasons"]
    d = _decide(r, U, P, pin, xv, info, T, planned=[k for k in info["planned_frames"] if k != min(info["planned_frames"])])
    assert d["failure_class"] == "CONTRACT_VIOLATION" and "FOOTPRINT_EXCEEDS_DECLARED_DOMAIN" in d["reasons"]
    bad = {k: dict(v) for k, v in xv.items()}
    t = min(info["planned_frames"])
    bad[t] = dict(bad[t], base=S.q_mul(U[t]["base"], S.q_axis_angle([1, 0, 0], 0.02)))
    d = _decide(r, U, P, pin, bad, info, T)
    assert d["decision"] == "REJECT" and d["failure_class"] == "CONTRACT_VIOLATION"
    d = _decide(r, U, P, pin, xv, info, T, display={"ok": False, "errors": ["CELL_MISMATCH"]})
    assert d["decision"] == "REJECT" and "DISPLAY_FOOTPRINT_MISMATCH" in d["reasons"]
    lie = {k: (dict(U[k]) if k in pin else v) for k, v in xv.items()}
    d = _decide(r, U, P, pin, lie, info, T)
    assert d["failure_class"] == "SOLVER_FAILURE" and "EVALUATED_PIN_UNSATISFIED" in d["reasons"]
    d = _decide(r, U, P, pin, xv, info, T, statuses=["SOLVED"] * (len(pin) - 1) + ["UNREACHABLE"])
    assert d["failure_class"] == "INFEASIBLE_REQUEST"


def test_single_corrupted_pin_frame_rejects_whole_request():
    r, P, U, pin, base = case()
    T = wrist(r, P[10])
    xv, info = run(r, U, pin, base)
    bad = {k: dict(v) for k, v in xv.items()}
    bad[13] = dict(bad[13], lower=S.q_mul(bad[13]["lower"], S.q_axis_angle([0, 0, 1], math.radians(2))))
    d = _decide(r, U, P, pin, bad, info, T)
    assert d["decision"] == "REJECT" and d["failure_class"] == "SOLVER_FAILURE"


def test_atomic_commit_stale_parent_and_exact_revert(tmp_path):
    r, P, U, pin, base = case()
    xv, info = run(r, U, pin, base)
    ch = ["__object__", "base", "upper", "lower"]
    p0 = S.make_state(None, "T118", list(range(N)), ch, {c: [P[k][c].tolist() for k in range(N)] for c in ch}, {"kind": "ROOT"})
    xs = full(U, xv)
    c1 = S.make_state(p0["state_id"], "T118", list(range(N)), ch, {c: [np.asarray(xs[k][c]).tolist() for k in range(N)] for c in ch},
                      {"kind": "X_CANDIDATE"})
    st = S.Store(tmp_path)
    st.put(p0)
    st.init_ref("m", p0["state_id"])
    rej = st.commit("m", p0["state_id"], c1, {"decision": "REJECT", "reasons": ["X"]})
    assert rej["head_after"] == p0["state_id"]
    acc = st.commit("m", p0["state_id"], c1, {"decision": "ACCEPT", "reasons": []})
    assert acc["head_after"] == c1["state_id"]
    assert st.commit("m", p0["state_id"], c1, {"decision": "ACCEPT", "reasons": []})["reasons"] == ["STALE_PARENT"]
    rv, rec = st.revert("m")
    assert rec["values_equal_restored"] and S.canon(rv["values"]) == S.canon(p0["values"])


# --------------------------------------------------------------------------------------------- quality ---
def test_compare_rule_requires_boundary_gain_and_no_articulation_worsening():
    F = {"B1_boundary_wrist_comp_step_peak_m": 0.10, "D1_introduced_flips": 0, "D3_arm_rot_step_excess_peak_deg": 4.0,
         "D2_elbow_comp_step_peak_m": 0.02, "A5_elbow_comp_accel_peak_m": 0.02, "A4_introduced_near_straight_frames": [],
         "cost_mean_elbow_dev_from_U_m": 0.01, "cost_peak_elbow_dev_from_U_m": 0.02, "cost_peak_upper_comp_deg": 5.0,
         "cost_peak_flex_change_deg": 3.0}
    X = dict(F, B1_boundary_wrist_comp_step_peak_m=0.03, cost_mean_elbow_dev_from_U_m=0.05)
    c = XB.compare_x(F, X)
    assert c["meaningful_improvement"] and c["costs_delta"]["cost_mean_elbow_dev_from_U_m"] > 0   # cost reported, not gated
    assert not XB.compare_x(F, dict(X, D1_introduced_flips=1))["meaningful_improvement"]
    assert "A4_introduced_near_straight" in XB.compare_x(F, dict(X, A4_introduced_near_straight_frames=[3]))["worse"]
    assert "A2_rotation_step" in XB.compare_x(F, dict(X, D3_arm_rot_step_excess_peak_deg=12.0))["worse"]
    assert not XB.compare_x(F, dict(X, B1_boundary_wrist_comp_step_peak_m=0.06))["meaningful_improvement"]


def test_quality_metrics_on_fake_blender_frames_have_integer_frame_axis():
    r, P, U, pin, base = case()
    xv, info = run(r, U, pin, base)
    X, Ue = evals(r, full(U, xv)), evals(r, U)
    q = XB.quality(X, Ue, list(range(N)), pin, "upper", "lower", AUTH)
    assert q["window"] == [max(0, 10 - XB.M_MAX - 1), min(N - 1, 17 + XB.M_MAX + 1)]
    assert all(isinstance(b, int) for b in q["boundary_transitions"])
    Fq = XB.quality(evals(r, full(U, base)), Ue, list(range(N)), pin, "upper", "lower", AUTH)
    assert q["B1_boundary_wrist_comp_step_peak_m"] < Fq["B1_boundary_wrist_comp_step_peak_m"]
