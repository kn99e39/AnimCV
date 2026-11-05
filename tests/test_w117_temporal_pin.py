"""W117 focused tests: temporal WORLD_POSITION_PIN resolver, temporal metrics, interval atomicity (no Blender needed)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import w116_semantic_pin_base as S  # noqa: E402
import w117_temporal_pin_base as TB  # noqa: E402


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
    return S.Rig({"rig_id": "T117", "bones": bones, "object": {"location": [0, 0, 0], "scale": [1.0] * 3,
                                                               "rest_rotation_wxyz": [1, 0, 0, 0]}})


BREST = {"base": {"length": 0.5, "parent": None}, "neck": {"length": 0.15, "parent": "base"},
         "upper": {"length": 0.3, "parent": "base"}, "lower": {"length": 0.25, "parent": "upper"}}


def seq_vals(n=20, flex0=50.0, flexamp=20.0, base_amp=10.0, az=-20.0):
    out = {}
    for f in range(n):
        tau = f / (n - 1)
        out[f] = {"__object__": np.array([0, 0, 0, 1.0]),
                  "base": S.q_axis_angle([0, 1, 0], math.radians(base_amp * math.sin(2 * math.pi * 0.7 * tau))),
                  "upper": S.q_mul(S.q_axis_angle([0, 0, 1], math.radians(az + 25 * tau)),
                                   S.q_axis_angle([1, 0, 0], math.radians(15 * math.sin(2 * math.pi * 0.5 * tau)))),
                  "lower": S.q_axis_angle([0, 0, 1], math.radians(flex0 + flexamp * math.sin(2 * math.pi * 0.9 * tau)))}
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
    M[:3, :3] = fk["Robj"] * r.scale
    return {"world": world, "stored_wxyz": stored, "matrix_world": M.tolist()}


def hold_case(lo=6, hi=13, n=20):
    r = rig()
    P = seq_vals(n)
    U = edited(r, P)
    T = wrist(r, P[lo])
    frames = list(range(lo, hi + 1))
    return r, P, U, T, frames


def run(r, U, T, frames, lam=TB.LAMBDA, maxiter=TB.MAXITER, auth=("upper", "lower"), kinds=None, targets=None):
    kinds = kinds or {k: "PIN" for k in frames}
    targets = targets or {k: T for k in frames}
    struct = S._structure(r, [{"bone": "lower", "point": "tail", "target": T}], list(auth))
    return TB.temporal_two_bone(r, frames, U, targets, kinds, list(auth), "upper", "lower", lam=lam, maxiter=maxiter, pins_struct=struct)


# ----------------------------------------------------------------------------------------------- resolver ---
def test_lambda_zero_reduces_to_w116_framewise():
    r, P, U, T, frames = hold_case()
    st, vals, info = run(r, U, T, frames, lam=0.0)
    for k in frames:
        s116, v116, _ = S.resolve_frame(r, U[k], [{"bone": "lower", "point": "tail", "target": T}], ["upper", "lower"])
        assert st[k] == s116 == "SOLVED"
        for c in ("upper", "lower"):
            assert TB.qangle(vals[k][c], v116[c]) < 1e-5
        assert np.linalg.norm(wrist(r, vals[k]) - wrist(r, v116)) < 1e-9


def test_hard_pin_exact_and_only_authorised_channels_change():
    r, P, U, T, frames = hold_case()
    st, vals, info = run(r, U, T, frames)
    assert info["converged"] and set(st.values()) == {"SOLVED"}
    for k in frames:
        assert np.linalg.norm(wrist(r, vals[k]) - T) < 1e-9
        for c in ("__object__", "base"):
            assert np.array_equal(vals[k][c], U[k][c])
        L1 = np.linalg.norm(r.world(r.fk(vals[k]), "upper", "tail") - r.world(r.fk(vals[k]), "upper", "head"))
        assert L1 == pytest.approx(0.3, abs=1e-9)
    assert info["J_temporal"] <= info["J_framewise"] + 1e-12


def test_unreachable_frame_reported_not_relaxed():
    r, P, U, T, frames = hold_case()
    far = T + np.array([0.0, 0.0, 0.6])
    st, vals, info = run(r, U, far, frames)
    assert "UNREACHABLE" in st.values()
    d = S.decide([], list(st.values()), [])
    assert d["failure_class"] == "INFEASIBLE_REQUEST"


def test_unsupported_structure_and_insufficient_dof_are_distinct():
    r, P, U, T, frames = hold_case()
    st, _, _ = run(r, U, T, frames, auth=("lower",))
    assert set(st.values()) == {"RESOLVER_UNSUPPORTED"}
    assert S.decide([], list(st.values()), [])["failure_class"] == "SOLVER_FAILURE"
    st, _, _ = run(r, U, T, frames, auth=())
    assert set(st.values()) == {"INSUFFICIENT_DOF"}
    assert S.decide([], list(st.values()), [])["failure_class"] == "INFEASIBLE_REQUEST"


def test_bounded_iterations_report_non_convergence_without_fallback():
    r, P, U, T, frames = hold_case()
    st, vals, info = run(r, U, T, frames, maxiter=1)
    assert not info["converged"] and set(st.values()) == {"NON_CONVERGED"}
    assert S.decide([], list(st.values()), [])["failure_class"] == "SOLVER_FAILURE"


def test_already_satisfied_frames_are_anchored():
    r = rig()
    P = seq_vals()
    frames = list(range(5, 10))
    targets = {k: wrist(r, P[k]) for k in frames}
    st = TB.temporal_two_bone(r, frames, P, targets, {k: "PIN" for k in frames}, ["upper", "lower"], "upper", "lower",
                              pins_struct=("two_bone", None))
    assert set(st[0].values()) == {"NO_CHANGE_NEEDED"}
    for k in frames:
        assert all(np.array_equal(st[1][k][c], P[k][c]) for c in P[k])
    assert S.decide([], list(st[0].values()), [])["reasons"] == ["TRIVIALLY_SATISFIED_NOT_EVIDENCE"]


def test_guides_decay_to_unpinned_motion_and_are_not_pins():
    r, P, U, T, frames = hold_case(lo=6, hi=13)
    g = TB.guide_targets(r, U, "lower", 6, 13, T, TB.MARGIN, list(range(20)))
    assert sorted(g) == [2, 3, 4, 5, 14, 15, 16, 17]
    off = [np.linalg.norm(g[k] - wrist(r, U[k])) for k in (14, 15, 16, 17)]
    assert off == sorted(off, reverse=True) and off[-1] == pytest.approx((1 - TB.smoothstep(4 / 5)) * np.linalg.norm(T - wrist(r, U[13])), rel=1e-9)
    kinds = {k: "PIN" for k in frames} | {k: "GUIDE" for k in g}
    targets = {k: T for k in frames} | g
    run_frames = sorted(set(frames) | set(g))
    st, vals, info = TB.temporal_two_bone(r, run_frames, U, targets, kinds, ["upper", "lower"], "upper", "lower",
                                          pins_struct=("two_bone", None))
    assert {st[k] for k in g} == {"GUIDE"} and {st[k] for k in frames} == {"SOLVED"}


def test_near_singular_swivel_continuity():
    """Control elbow crossing the shoulder-target axis: framewise pole jumps around the swivel circle, temporal does not."""
    r = rig()
    n = 16
    U = {}
    for f in range(n):
        U[f] = {"__object__": np.array([0, 0, 0, 1.0]), "base": np.array([0, 0, 0, 1.0]),
                "upper": S.q_axis_angle([1, 0, 0], math.radians(-30 + 4 * f)), "lower": S.q_axis_angle([0, 0, 1], math.radians(3.0))}
    # target near max reach along the rest arm direction: the control elbow sweeps through the shoulder-target axis
    Sh = r.world(r.fk(U[0]), "upper", "head")
    T = Sh + np.array([0.5, 0.0, 0.0])
    frames = list(range(n))
    st, vals, info = run(r, U, T, frames)
    sw_t, sw_f = np.unwrap(np.radians(info["swivel_deg_temporal"])), np.unwrap(np.radians(info["swivel_deg_framewise"]))
    assert np.max(np.abs(np.diff(sw_t))) < np.max(np.abs(np.diff(sw_f)))
    for k in frames:
        assert np.linalg.norm(wrist(r, vals[k]) - T) < 1e-9


# ----------------------------------------------------------------------------------------------- metrics ---
def test_quaternion_metrics_handle_sign_ambiguity():
    q = S.q_axis_angle([0, 0, 1], 0.3)
    assert TB.qangle(q, -q) == pytest.approx(0.0, abs=1e-6)
    seq = [S.q_axis_angle([0, 0, 1], math.radians(5 * i)) for i in range(6)]
    seq_w = [[x[3], x[0], x[1], x[2]] for x in seq]
    flipped = [(-np.array(s)).tolist() if i % 2 else s for i, s in enumerate(seq_w)]
    a, b = TB.rot_first_order(seq_w), TB.rot_first_order(flipped)
    assert a == pytest.approx([5.0] * 5, abs=1e-6) and b == pytest.approx(a, abs=1e-6)
    assert TB.rot_second_order(flipped) == pytest.approx([0.0] * 4, abs=1e-5)


def test_bend_flip_and_compensation_metrics():
    r = rig()
    U = {k: {"__object__": np.array([0, 0, 0, 1.0]), "base": np.array([0, 0, 0, 1.0]), "upper": np.array([0, 0, 0, 1.0]),
             "lower": S.q_axis_angle([0, 0, 1], math.radians(40))} for k in range(6)}
    X = {k: dict(v) for k, v in U.items()}
    for k in (3, 4, 5):  # bend to the other side: flip at transition 3
        X[k] = dict(X[k], lower=S.q_axis_angle([0, 0, 1], math.radians(-40)))
    m = TB.temporal_metrics({k: fake_eval(r, v) for k, v in X.items()}, {k: fake_eval(r, v) for k, v in U.items()}, list(range(6)),
                            "upper", "lower", ["upper", "lower"])
    assert m["introduced_bend_flips"] == [3] and m["defects"]["D1"] == 1
    assert m["defects"]["D3"] == pytest.approx(80.0, abs=1e-4)


def test_compare_rules():
    F = {"defects": {"D1": 2, "D2": 0.02, "D3": 10.0, "D4": None}, "side": {"S1_mean_elbow_dev_m": 0.01, "S2_peak_upper_comp_deg": 10,
                                                                          "S3_peak_flex_change_deg": 5}}
    T = {"defects": {"D1": 0, "D2": 0.009, "D3": 9.0, "D4": None}, "side": {"S1_mean_elbow_dev_m": 0.012, "S2_peak_upper_comp_deg": 12,
                                                                          "S3_peak_flex_change_deg": 5}}
    c = TB.compare(F, T)
    assert c["improved"] == ["D1", "D2"] and not c["worse"] and c["material_improvement"]
    T2 = dict(T, side={"S1_mean_elbow_dev_m": 0.05, "S2_peak_upper_comp_deg": 12, "S3_peak_flex_change_deg": 5})
    assert TB.compare(F, T2)["worse"] == ["S1_mean_elbow_dev_m"] and not TB.compare(F, T2)["material_improvement"]
    T3 = dict(T, defects={"D1": 3, "D2": 0.02, "D3": 10.0, "D4": None})
    assert "D1" in TB.compare(F, T3)["worse"]


# ------------------------------------------------------------------------------------ evaluator / atomicity ---
def _request_eval(r, P, U, X, frames, margins=(), T=None):
    allf = sorted(P)
    pins_at = lambda k: [{"bone": "lower", "point": "tail", "target": T, "semantic_joint": "syn_wrist"}]  # noqa: E731
    Ue, Pe, Xe = ({k: fake_eval(r, v[k]) for k in allf} for v in (U, P, X))
    dom = S.domain_points(r, ["upper", "lower", "base"])
    arm = S.domain_points(r, ["upper", "lower"])
    return TB.evaluate_temporal_request(Xe, Ue, Pe, set(frames), set(margins), allf, pins_at, ["upper", "lower"], ["base"], dom, arm,
                                        BREST, 1.0, ("upper", "lower"))


def test_single_corrupted_frame_rejects_whole_interval():
    r, P, U, T, frames = hold_case()
    st, vals, _ = run(r, U, T, frames)
    X = {k: (vals[k] if k in vals else U[k]) for k in P}
    per, extra = _request_eval(r, P, U, X, frames, T=T)
    assert TB.temporal_decision([], list(st.values()), per, extra)["decision"] == "ACCEPT"
    Xb = dict(X)
    Xb[9] = dict(X[9], lower=S.q_mul(X[9]["lower"], S.q_axis_angle([0, 0, 1], math.radians(2))))
    per, extra = _request_eval(r, P, U, Xb, frames, T=T)
    d = TB.temporal_decision([], list(st.values()), per, extra)
    assert d["decision"] == "REJECT" and "EVALUATED_PIN_UNSATISFIED" in d["reasons"]
    assert sum(1 for k in frames if not per[k]["evaluation"]["pins_satisfied"]) == 1


def test_false_solved_claim_rejected_by_evaluator():
    r, P, U, T, frames = hold_case()
    per, extra = _request_eval(r, P, U, dict(U), frames, T=T)
    d = TB.temporal_decision([], ["SOLVED"] * len(frames), per, extra)
    assert d["decision"] == "REJECT" and d["failure_class"] == "SOLVER_FAILURE"


def test_change_outside_declared_temporal_domain_and_user_edit_tamper():
    r, P, U, T, frames = hold_case()
    st, vals, _ = run(r, U, T, frames)
    X = {k: (vals[k] if k in vals else U[k]) for k in P}
    X[2] = dict(U[2], upper=S.q_mul(U[2]["upper"], S.q_axis_angle([0, 0, 1], 0.01)))
    per, extra = _request_eval(r, P, U, X, frames, T=T)
    d = TB.temporal_decision([], list(st.values()), per, extra)
    assert d == {"decision": "REJECT", "failure_class": "CONTRACT_VIOLATION", "reasons": ["CHANGE_OUTSIDE_DECLARED_TEMPORAL_DOMAIN"]}
    X = {k: (vals[k] if k in vals else U[k]) for k in P}
    X[9] = dict(X[9], base=S.q_mul(X[9]["base"], S.q_axis_angle([1, 0, 0], math.radians(1))))
    per, extra = _request_eval(r, P, U, X, frames, T=T)
    d = TB.temporal_decision([], list(st.values()), per, extra)
    assert d["decision"] == "REJECT" and d["failure_class"] == "CONTRACT_VIOLATION"
    assert "UNAUTHORIZED_CHANNEL_CHANGE" in d["reasons"] and "POINT_OUTSIDE_ARM_DOMAIN_CHANGED_VS_CONTROL" in extra


def test_stale_parent_rejection_and_exact_revert(tmp_path):
    r, P, U, T, frames = hold_case()
    ch = ["__object__", "base", "upper", "lower"]
    tolist = lambda V: {c: [V[k][c].tolist() for k in sorted(V)] for c in ch}  # noqa: E731
    p = S.make_state(None, "T117", sorted(P), ch, tolist(P), {"kind": "ROOT"})
    st, vals, _ = run(r, U, T, frames)
    cand = S.make_state(p["state_id"], "T117", sorted(P), ch, tolist({k: vals.get(k, U[k]) for k in P}), {"kind": "PIN"})
    store = S.Store(tmp_path)
    store.put(p)
    store.init_ref("m", p["state_id"])
    acc = {"decision": "ACCEPT", "reasons": []}
    assert store.commit("m", p["state_id"], cand, acc)["head_after"] == cand["state_id"]
    rec = store.commit("m", p["state_id"], cand, acc)
    assert rec["reasons"] == ["STALE_PARENT"] and rec["head_after"] == cand["state_id"]
    rv, rr = store.revert("m")
    assert rr["values_equal_restored"] and S.canon(rv["values"]) == S.canon(p["values"])
    rej = store.commit("m", rv["state_id"], cand, {"decision": "REJECT", "reasons": ["X"]})
    assert rej["head_after"] == rv["state_id"]
