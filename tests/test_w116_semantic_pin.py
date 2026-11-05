"""W116 focused tests: scratch WORLD_POSITION_PIN contract, binding, resolver, evaluator, lineage (no Blender needed)."""
from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import w116_semantic_pin_base as S  # noqa: E402


def _mat(R, t):
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = t
    return M.tolist()


def syn_rig(obj_loc=(0.0, 0.0, 0.0), scale=1.0):
    """Two-bone chain on a vertical base; bone Y axis along the bone (Blender convention)."""
    Rz_up = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)  # local Y -> world +Z
    Rx = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], float)     # local Y -> world +X
    bones = {"base": {"parent": None, "matrix_local": _mat(Rz_up, [0, 0, 0]), "head": [0, 0, 0], "tail": [0, 0, 0.5]},
             "upper": {"parent": "base", "matrix_local": _mat(Rx, [0, 0, 0.5]), "head": [0, 0, 0.5], "tail": [0.3, 0, 0.5]},
             "lower": {"parent": "upper", "matrix_local": _mat(Rx, [0.3, 0, 0.5]), "head": [0.3, 0, 0.5], "tail": [0.55, 0, 0.5]}}
    return S.Rig({"rig_id": "T", "bones": bones, "object": {"location": list(obj_loc), "scale": [scale] * 3,
                                                           "rest_rotation_wxyz": [1, 0, 0, 0]}})


def bent_vals(bend_deg=60.0, base_deg=0.0):
    return {"__object__": np.array([0, 0, 0, 1.0]), "base": S.q_axis_angle([1, 0, 0], math.radians(base_deg)),
            "upper": np.array([0, 0, 0, 1.0]), "lower": S.q_axis_angle([0, 0, 1], math.radians(bend_deg))}


def fake_eval_frame(rig, vals):
    fk = rig.fk(vals)
    world = {b: {"head": rig.world(fk, b, "head").tolist(), "tail": rig.world(fk, b, "tail").tolist()} for b in rig.bones}
    stored = {c: [float(v[3]), float(v[0]), float(v[1]), float(v[2])] for c, v in vals.items()}
    M = np.eye(4)
    M[:3, :3] = fk["Robj"] * rig.scale
    M[:3, 3] = rig.loc
    return {"world": world, "stored_wxyz": stored, "matrix_world": M.tolist()}


BREST = {"base": {"length": 0.5, "parent": None}, "upper": {"length": 0.3, "parent": "base"}, "lower": {"length": 0.25, "parent": "upper"}}


# ------------------------------------------------------------------------------------- geometry/coordinates ---
def test_fk_lengths_scale_and_object_translation():
    r = syn_rig()
    fk = r.fk(bent_vals())
    assert np.linalg.norm(r.world(fk, "lower", "tail") - r.world(fk, "lower", "head")) == pytest.approx(0.25, abs=1e-12)
    r2 = syn_rig(obj_loc=(0.1, 0, 0))
    d = r2.world(r2.fk(bent_vals()), "lower", "tail") - r.world(fk, "lower", "tail")
    assert np.allclose(d, [0.1, 0, 0], atol=1e-12)
    r3 = syn_rig(scale=0.01)  # cm rig: world metres = 0.01 * armature units
    assert r3.world_len("upper") == pytest.approx(0.003)


def test_local_rotation_moves_only_descendants():
    r = syn_rig()
    v = bent_vals()
    a = r.all_points(r.fk(v))
    v2 = dict(v, lower=S.q_mul(v["lower"], S.q_axis_angle([1, 0, 0], 0.2)))
    b = r.all_points(r.fk(v2))
    moved = {f"{n}.{p}" for n in a for p in a[n] if np.linalg.norm(a[n][p] - b[n][p]) > 1e-12}
    assert moved == {"lower.tail"}
    assert S.domain_points(r, ["lower"]) == {"lower.tail"}
    assert S.domain_points(r, ["upper"]) == {"upper.tail", "lower.head", "lower.tail"}


def test_world_swing_is_pivoted_rotation():
    r = syn_rig()
    v = bent_vals()
    fk = r.fk(v)
    q = S.q_axis_angle([0, 0, 1], 0.3)
    v2 = r.apply_world_swing(v, fk, "upper", q)
    fk2 = r.fk(v2)
    S0 = r.world(fk, "upper", "head")
    W0, W1 = r.world(fk, "lower", "tail"), r.world(fk2, "lower", "tail")
    assert np.allclose(W1 - S0, S.q_to_m(q) @ (W0 - S0), atol=1e-12)


# --------------------------------------------------------------------------------------------- resolver ---
def test_two_bone_reachable_and_unreachable():
    r = syn_rig()
    v = bent_vals()
    fk = r.fk(v)
    W = r.world(fk, "lower", "tail")
    pin = [{"bone": "lower", "point": "tail", "target": W + np.array([0, 0, 0.05])}]
    st, pv, info = S.resolve_frame(r, v, pin, ["upper", "lower"])
    assert st == "SOLVED" and info["internal_residual_m"][0] < 1e-9
    assert np.allclose(pv["base"], v["base"])  # unauthorised channel untouched
    far = [{"bone": "lower", "point": "tail", "target": r.world(fk, "upper", "head") + np.array([0.6, 0, 0])}]
    assert S.resolve_frame(r, v, far, ["upper", "lower"])[0] == "UNREACHABLE"
    assert S.resolve_frame(r, v, pin, [])[0] == "INSUFFICIENT_DOF"
    assert S.resolve_frame(r, v, pin, ["base", "lower"])[0] == "RESOLVER_UNSUPPORTED"


def test_ccd_budget_and_convergence():
    r = syn_rig()
    v = bent_vals()
    W = r.world(r.fk(v), "lower", "tail")
    pin = [{"bone": "lower", "point": "tail", "target": W + np.array([0, 0, 0.05])}]
    assert S.resolve_frame(r, v, pin, ["upper", "lower"], method="ccd", max_iter=1)[0] == "NON_CONVERGED"
    st, _, info = S.resolve_frame(r, v, pin, ["upper", "lower"], method="ccd", max_iter=200)
    assert st == "SOLVED" and info["internal_residual_m"][0] <= S.CCD_TOL_M


def test_aim_distal_sphere_gap():
    r = syn_rig()
    v = bent_vals()
    fk = r.fk(v)
    E = r.world(fk, "lower", "head")
    ok = [{"bone": "lower", "point": "tail", "target": E + 0.25 * np.array([0, 0, 1.0])}]
    bad = [{"bone": "lower", "point": "tail", "target": E + 0.30 * np.array([0, 0, 1.0])}]
    assert S.resolve_frame(r, v, ok, ["lower"])[0] == "SOLVED"
    assert S.resolve_frame(r, v, bad, ["lower"])[0] == "UNREACHABLE"


# ---------------------------------------------------------------------------------- contract / validation ---
def ctx_for(state, rig, bid, target_pos):
    b = {"semantic_joint": "syn_wrist", "rig_id": state["rig_id"], "bone": "lower", "point": "tail"}
    return {"bindings": {bid: b}, "failed_bindings": {"syn_hand": {"status": "BINDING_AMBIGUOUS"}},
            "capture": {(state["state_id"], 0, bid): list(target_pos)}, "parent": state, "root_bones": {"base"}}


def mk_state():
    v = bent_vals()
    return S.make_state(None, "T", [0], ["__object__", "base", "upper", "lower"], {c: [v[c].tolist()] for c in v}, {"kind": "ROOT"})


def mk_pin(st, W, **kw):
    src = {"kind": "CAPTURED_FROM_EVALUATED_STATE", "state_id": st["state_id"], "frame_index": 0, "binding_id": "B"}
    return S.make_pin("c", kw.pop("joint", "syn_wrist"), kw.pop("bid", "B"), W, src, [0], kw.pop("auth", ["upper", "lower"]),
                      st["state_id"], **kw)


def reasons(req, ctx):
    return {r for _, r in S.validate_request(req, ctx)}


def test_constraint_roundtrip_and_validation_codes():
    r = syn_rig()
    st = mk_state()
    W = r.world(r.fk(S.frame_vals(st, 0)), "lower", "tail").tolist()
    ctx = ctx_for(st, r, "B", W)
    c = mk_pin(st, W)
    assert json.loads(json.dumps(c)) == c
    req = {"request_id": "x", "parent_state_id": st["state_id"], "edit": None, "constraints": [c]}
    assert S.validate_request(req, ctx) == []
    cases = {
        "COORDINATE_FRAME_UNSUPPORTED": mk_pin(st, W, frame="ARMATURE_LOCAL"),
        "UNIT_UNSUPPORTED": mk_pin(st, W, unit="cm"),
        "SOFT_CONSTRAINT_NOT_IMPLEMENTED": mk_pin(st, W, hard=False),
        "TYPE_NOT_IMPLEMENTED": mk_pin(st, W, ctype="ORIENTATION_PIN"),
        "ROOT_NOT_PERMITTED": mk_pin(st, W, auth=["base", "upper"]),
        "DOF_NOT_IN_STATE": mk_pin(st, W, auth=["upper", "lower_r"]),
        "TARGET_PROVENANCE_MISMATCH": mk_pin(st, [x * 100 for x in W]),
        "BINDING_AMBIGUOUS": mk_pin(st, W, joint="syn_hand", bid="UNBOUND"),
    }
    for want, cc in cases.items():
        assert want in reasons(dict(req, constraints=[cc]), ctx), want
    t = copy.deepcopy(c)
    t["target"][2] += 0.01
    assert {"TARGET_ALTERED_AFTER_BINDING", "CONSTRAINT_DIGEST_MISMATCH"} <= reasons(dict(req, constraints=[t]), ctx)
    c2 = mk_pin(st, W)
    c2["target"] = [W[0], W[1], W[2] + 0.05]
    c2["target_digest"] = S.digest({"target": c2["target"], "source": c2["target_source"], "binding_id": "B", "frames": [0]})
    c2["constraint_digest"] = S.digest({k: v for k, v in c2.items() if k != "constraint_digest"})
    assert ("INFEASIBLE_REQUEST", "CONFLICTING_HARD_CONSTRAINTS") in S.validate_request(dict(req, constraints=[c, c2]), ctx)
    edit = {"channels": ["upper"]}
    assert "AUTHORIZED_OVERLAPS_USER_EDIT" in reasons(dict(req, edit=edit), ctx)


# ---------------------------------------------------------------------------------- evaluator independence ---
def test_evaluator_uses_evaluated_output_not_solver_claim():
    r = syn_rig()
    v = bent_vals()
    W = r.world(r.fk(v), "lower", "tail")
    pins = [{"bone": "lower", "point": "tail", "target": W + np.array([0, 0, 0.05]), "semantic_joint": "syn_wrist"}]
    st, pv, _ = S.resolve_frame(r, v, pins, ["upper", "lower"])
    par = fake_eval_frame(r, v)
    allowed = {"vs_parent": ["lower", "upper"], "vs_control": ["lower", "upper"]}
    dom = S.domain_points(r, ["upper", "lower"])
    good = S.evaluate_candidate(fake_eval_frame(r, pv), par, par, pins, allowed, dom, BREST, 1.0, chain=("upper", "lower"))
    assert good["pins_satisfied"] and good["scope_ok"] and good["locality_ok"] and good["lengths_ok"]
    assert S.decide([], ["SOLVED"], [good])["decision"] == "ACCEPT"
    lie = S.evaluate_candidate(par, par, par, pins, allowed, dom, BREST, 1.0, chain=("upper", "lower"))
    d = S.decide([], ["SOLVED"], [lie])  # resolver claims SOLVED; evaluated output does not satisfy the pin
    assert d["decision"] == "REJECT" and "EVALUATED_PIN_UNSATISFIED" in d["reasons"] and d["failure_class"] == "SOLVER_FAILURE"
    tam = dict(pv, base=S.q_mul(pv["base"], S.q_axis_angle([1, 0, 0], 0.02)))
    ev = S.evaluate_candidate(fake_eval_frame(r, tam), par, par, pins, allowed, dom, BREST, 1.0, chain=("upper", "lower"))
    d = S.decide([], ["SOLVED"], [ev])
    assert "UNAUTHORIZED_CHANNEL_CHANGE" in d["reasons"] and d["failure_class"] == "CONTRACT_VIOLATION"


def test_gt_feasibility_independent_classes():
    r = syn_rig()
    v = bent_vals()
    fr = fake_eval_frame(r, v)
    S0 = np.array(fr["world"]["upper"]["head"])
    near = [{"bone": "lower", "point": "tail", "target": S0 + np.array([0.5, 0, 0])}]
    far = [{"bone": "lower", "point": "tail", "target": S0 + np.array([0.6, 0, 0])}]
    assert S.gt_feasibility("two_bone", fr, near, BREST, 1.0)[0] == "FEASIBLE"
    assert S.gt_feasibility("two_bone", fr, far, BREST, 1.0)[0] == "INFEASIBLE"


# ------------------------------------------------------------------------------------- lineage and source ---
def test_store_commit_stale_reject_and_revert(tmp_path):
    st = mk_state()
    store = S.Store(tmp_path)
    store.put(st)
    store.init_ref("m", st["state_id"])
    v = S.frame_vals(st, 0)
    v["upper"] = S.q_axis_angle([0, 0, 1], 0.1)
    child = S.make_state(st["state_id"], "T", [0], st["channels"], {c: [v[c].tolist()] for c in st["channels"]}, {"kind": "EDIT"})
    acc = {"decision": "ACCEPT", "reasons": []}
    assert store.commit("m", st["state_id"], child, acc)["head_after"] == child["state_id"]
    stale = store.commit("m", st["state_id"], child, acc)
    assert stale["reasons"] == ["STALE_PARENT"] and stale["head_after"] == child["state_id"]
    rej = store.commit("m", child["state_id"], child, {"decision": "REJECT", "reasons": ["X"]})
    assert rej["head_after"] == child["state_id"]
    rv, rec = store.revert("m")
    assert rec["values_equal_restored"] and S.canon(rv["values"]) == S.canon(st["values"])
    assert all(x["verifies"] for x in store.lineage(store.refs["m"]))
    bad = dict(child, values={**child["values"], "upper": [[0, 0, 0, 1.0]]})
    with pytest.raises(ValueError):
        store.put(bad)


def test_source_guard_detects_change(tmp_path):
    f = tmp_path / "src.bin"
    f.write_bytes(b"abc")
    g = S.SourceGuard({"s": f})
    assert g.check()["unchanged"]
    f.write_bytes(b"abd")
    assert not g.check()["unchanged"]


def test_binding_fail_safe():
    r = syn_rig()
    v = bent_vals()
    fr = fake_eval_frame(r, v)
    rest = {n: {"head_local": b["head"], "tail_local": b["tail"]} for n, b in r.spec["bones"].items()}
    ok = S.verify_binding("w", [("lower", "tail")], "T", "d", rest, [fr], ["upper", "lower"], r)
    assert ok["status"] == "BOUND" and ok["state_controllable_channels"] == ["upper", "lower"]
    assert S.verify_binding("h", [("lower", "tail"), ("lower", "head")], "T", "d", rest, [fr], [], r)["status"] == "BINDING_AMBIGUOUS"
    assert S.verify_binding("n", [], "T", "d", rest, [fr], [], r)["status"] == "BINDING_MISSING"
    assert S.verify_binding("f", [("finger", "tail")], "T", "d", rest, [fr], [], r)["status"] == "BINDING_MISSING"
