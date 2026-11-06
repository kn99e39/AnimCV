#!/usr/bin/env python3
"""W120 scratch matched two-factor mechanism-attribution instrument (NON-PRODUCTION, evidence only).

Separates the two compensation mechanisms that were confounded in the W119 candidate Y:

  Factor G (wrist target on every transition frame k):
    G0  W_X(k): the world wrist of the clean per-request X_r (Python FK of the stored X_r values).  A rotation-offset-derived
        target; re-solving it with the shared solver is NOT claimed to reproduce X_r's joint rotations.
    G1  the W119 world-space guide  G(k) = W_U(k) + w_j (W_F(b) - W_U(b)).
  Factor F (additional flex floor):
    F0  no extra floor: phi_floor = the W116 lower flex limit (0 deg), i.e. only the W116 physical reach / joint-limit band.
    F1  the W119 extra floor  phi_floor = min(flex_U(k), flex_F(b)), unchanged.

Everything else is SHARED by all four variants G0F0 / G0F1 / G1F0 / G1F1 and is the frozen W119 code, imported unchanged:
the analytic two-bone solve (`w119_endpoint_transition_base.solve_guide`), the elbow pole reference
R(k) = E_U(k) + w_j (E_F(b) - E_U(b)), the weight w_j = 1 - smoothstep(j / (m + 1)), the bend-plane ambiguity rule, the
nonconvergence tolerance, the branch-flip flag, the authorised DOF, the hard-pin base (F_r values) and the per-side margins
(the frozen X_r margins).  The automatic search budget is built with the SAME m_max as the request's X_r (0 for the
zero-budget control Z_A); W119 always used the global M_MAX = 8 for Y, which was its Z_A defect.

Also provides: four-way A/B fairness, the frozen W120 error-class precedence table and its independent predictor, factorial
contrasts, endpoint-jump accounting and G0F0-versus-X_r Blender differences.  Decisions reuse W119 `decide_y` unchanged.
"""
from __future__ import annotations

import math
import time

import numpy as np

import w116_semantic_pin_base as S
import w117_temporal_pin_base as TB
import w118_transition_edit_base as XB
import w119_endpoint_transition_base as YB

FACTORS = ("G0F0", "G0F1", "G1F0", "G1F1")
FACTOR_DEF = {
    "G": {"G0": "wrist target = world wrist of the clean per-request X_r at k (Python FK of stored X_r values)",
          "G1": "wrist target = W119 world-space guide W_U(k) + w_j (W_F(b) - W_U(b))"},
    "F": {"F0": "phi_floor = W116 lower flex limit (0 deg): reach clip only to the W116 band [r_min, r_max]",
          "F1": "phi_floor = min(flex_U(k), flex_F(b)) (W119 extra flex floor, unchanged)"},
}


def solver_params():
    """Everything that must be identical across the four variants (frozen; digested into every fairness record)."""
    return {"solver": "w119_endpoint_transition_base.solve_guide (analytic two-bone, minimal world swings)",
            "pole_rule": "R(k) = E_U(k) + w_j (E_F(b) - E_U(b)); elbow = circle point nearest R(k)",
            "weight": "w_j = 1 - smoothstep(j/(m+1))", "Y_AMBIG_DEG": YB.Y_AMBIG_DEG, "Y_INTERNAL_TOL_M": YB.Y_INTERNAL_TOL_M,
            "Y_FLIP_FLEX_FLOOR_DEG": YB.Y_FLIP_FLEX_FLOOR_DEG, "flex_limits_deg": list(S.ELBOW_FLEX_LIMIT_DEG),
            "margins": "frozen X_r per-side chosen margins", "budget_m_max": "= the request's X_r m_max",
            "fallback": None}


SOLVER_PARAMS_DIGEST = S.digest(solver_params())


def factor_transition(rig, ctrl, base, pin_frames, all_frames, auth, upper, lower, margins, m_max, factor, x_vals=None,
                      fault=None, guide_override=None, pole_override=None, params_digest=None):
    """ctrl: {k: U values}; base: {k: F_r values} (pin frames); margins: frozen X_r margins {"lo:hi:side": m};
    m_max: the X_r search-budget parameter of this request; factor in FACTORS; x_vals: {k: X_r values} (needed for G0).
    Returns (values{k} for pin + transition frames, info).  Every other frame stays U (not returned)."""
    if factor not in FACTORS:
        raise ValueError(factor)
    g, f = factor[:2], factor[2:]
    t0 = time.time()
    plan, budget = XB.budget_plan(pin_frames, all_frames, m_max)
    out = {k: {c: np.array(v, float) for c, v in base[k].items()} for k in pin_frames}
    info = {"method": "w120_matched_factor_transition", "factor": factor, "m_max": int(m_max), "plan": plan, "budget_frames": budget,
            "margins": dict(margins), "solver_params_digest": params_digest or SOLVER_PARAMS_DIGEST, "sides": [], "fallback": None,
            "frames": {}}
    if not ({upper, lower} <= set(auth)):
        info.update(request_status="MISSING_AUTHORIZED_DOF", planned_frames=sorted(pin_frames), wall_s=time.time() - t0)
        return out, info
    missing_ref = []
    for p in plan:
        lo, hi = p["run"]
        for side, b, sgn, avail in (("entry", lo, -1, p["entry_available"]), ("release", hi, +1, p["release_available"])):
            key = f"{lo}:{hi}:{side}"
            m = int(margins.get(key, 0))
            rec = {"run": [lo, hi], "side": side, "boundary": b, "available": avail, "margin": m, "frames": []}
            if m > avail:
                rec["error"] = "MARGIN_EXCEEDS_AVAILABLE_BUDGET"
                info["sides"].append(rec)
                continue
            if m == 0:
                info["sides"].append(rec)
                continue
            _, SB, EF, WF = YB._pts(rig, base[b], upper, lower)
            _, _, EU, WU = YB._pts(rig, ctrl[b], upper, lower)
            cW, cE = WF - WU, EF - EU
            phiB = YB._flex(SB, EF, WF)
            rec.update(boundary_wrist_offset_m=float(np.linalg.norm(cW)), boundary_elbow_offset_m=float(np.linalg.norm(cE)),
                       boundary_flex_deg=phiB)
            prev_E, prev_n = EF, YB._normal(SB, EF, WF)
            for j in range(1, m + 1):
                k = b + sgn * j
                w = 1.0 - TB.smoothstep(j / (m + 1))
                _, Sk, Ek, Wk = YB._pts(rig, ctrl[k], upper, lower)
                if g == "G1":
                    G = Wk + w * cW
                else:
                    if x_vals is None or k not in x_vals:
                        missing_ref.append(k)
                        G = Wk + w * cW
                    else:
                        G = YB._pts(rig, x_vals[k], upper, lower)[3]
                R = Ek + w * cE
                if guide_override is not None:
                    G = guide_override(k, Sk, G, rig.world_len(upper) + rig.world_len(lower))
                if pole_override is not None:
                    R = pole_override(k, Sk, G, R)
                phi_floor = min(YB._flex(Sk, Ek, Wk), phiB) if f == "F1" else S.ELBOW_FLEX_LIMIT_DEG[0]
                vals, fi = YB.solve_guide(rig, ctrl[k], G, R, prev_E, upper, lower, phi_floor, fault=fault)
                Sn, En, Wn = Sk, np.array(fi["elbow"]), np.array(fi["wrist"])
                n = YB._normal(Sn, En, Wn)
                if n is not None and prev_n is not None and float(np.dot(n, prev_n)) < 0:
                    fi["flags"].append("BRANCH_FLIP_VS_PREVIOUS")
                fi.update(weight=w, j=j, side=side, boundary=b, control_flex_deg=YB._flex(Sk, Ek, Wk))
                info["frames"][str(k)] = fi
                out[k] = vals
                rec["frames"].append(k)
                prev_E, prev_n = En, (n if n is not None else prev_n)
            info["sides"].append(rec)
    fr = info["frames"].values()
    bad_margin = [s for s in info["sides"] if s.get("error")]
    info["planned_frames"] = sorted(set(pin_frames) | {k for s in info["sides"] for k in s["frames"]})
    info["status_counts"] = {s: sum(1 for x in fr if x["status"] == s) for s in YB.Y_STATUSES}
    info["flag_counts"] = {fl: sum(1 for x in fr if fl in x["flags"]) for fl in
                           ("BEND_PLANE_AMBIGUOUS", "POLE_AXIS_FALLBACK", "BRANCH_FLIP_VS_PREVIOUS", "NON_CONVERGED")}
    info["g0_reference_missing_frames"] = sorted(missing_ref)
    if bad_margin:
        info["request_status"] = "MARGIN_EXCEEDS_AVAILABLE_BUDGET"
    elif missing_ref:
        info["request_status"] = "G0_REFERENCE_MISSING"
    elif info["flag_counts"]["NON_CONVERGED"]:
        info["request_status"] = "NON_CONVERGED"
    else:
        info["request_status"] = "SOLVED"
    info["wall_s"] = time.time() - t0
    return out, info


# ------------------------------------------------------------------------------------------------ fairness ---
def fairness_factor(F, X, V, params_digest_V=None):
    """Per-variant A/B fairness: the unchanged W119 `ab_fairness` with V in the Y slot, plus the shared solver-parameter
    digest.  F, X, V: fairness records (parent_id, rig_id, U_state_id, constraint_digests, pin_frames, auth,
    pin_values_digest, margins, budget_frames)."""
    r = YB.ab_fairness({"F": F, "X": X, "Y": V})
    errs = set(r["errors"])
    if (params_digest_V or SOLVER_PARAMS_DIGEST) != SOLVER_PARAMS_DIGEST:
        errs.add("SOLVER_PARAMS_MISMATCH:Y")
    return {"ok": not errs, "errors": sorted(errs)}


def cross_factor_identity(recs):
    """All four variant fairness records must be identical in every matched field (margins, budget, pin values, parent, U,
    rig, constraint digests, pin frames, authorisation, solver parameter digest)."""
    keys = ("parent_id", "rig_id", "U_state_id", "constraint_digests", "pin_frames", "auth", "pin_values_digest", "margins",
            "budget_frames", "solver_params_digest")
    ref = recs.get("G0F0")
    errs = []
    for v, r in recs.items():
        for k in keys:
            if ref is None or r.get(k) != ref.get(k):
                errs.append(f"{k}:{v}")
    return {"ok": not errs and set(recs) == set(FACTORS), "errors": sorted(set(errs)), "variants": sorted(recs)}


# ------------------------------------------------------------------------------- frozen precedence table ---
PRECEDENCE_TABLE = [
    "P1 request contract invalid (validation errors): REJECT with the class of the FIRST validation error (W116 decide); "
    "nothing else is consulted",
    "P2 request status MISSING_AUTHORIZED_DOF: REJECT INFEASIBLE_REQUEST [Y_MISSING_AUTHORIZED_DOF]; nothing else is consulted",
    "P3 base decision d = W118 decide_x: (a) pin resolver statuses not all SOLVED/NO_CHANGE_NEEDED -> REJECT, class "
    "SOLVER_FAILURE if the first of [RESOLVER_UNSUPPORTED, NON_CONVERGED, INSUFFICIENT_DOF, UNREACHABLE] present is "
    "RESOLVER_UNSUPPORTED/NON_CONVERGED else INFEASIBLE_REQUEST; (b) else evaluated-frame failures -> REJECT, "
    "CONTRACT_VIOLATION if any of {UNAUTHORIZED_CHANNEL_CHANGE, OUTPUT_OUTSIDE_DECLARED_DOMAIN, BONE_LENGTH_CHANGED, ROOT_MOVED} "
    "else SOLVER_FAILURE (EVALUATED_PIN_UNSATISFIED / JOINT_LIMIT only); (c) footprint/disclosure extras {CHANGE_OUTSIDE_"
    "AUTOMATIC_BUDGET, FOOTPRINT_EXCEEDS_DECLARED_DOMAIN, UNAUTHORIZED_CHANNEL_CHANGE, POINT_OUTSIDE_ARM_DOMAIN_CHANGED_VS_CONTROL,"
    " ROOT_MOVED, DISPLAY_FOOTPRINT_MISMATCH}: if d ACCEPT -> REJECT CONTRACT_VIOLATION; if d REJECT and every pin status "
    "SOLVED/NO_CHANGE_NEEDED -> CONTRACT_VIOLATION; otherwise d's class is kept (reasons united)",
    "P4 pre-check extras E = {NON_CONVERGED (request status), MARGIN_EXCEEDS_AVAILABLE_BUDGET, BASELINE_NOT_ISOLATED, "
    "AB_FAIRNESS:*}: if E is empty the result is d; else REJECT with class SOLVER_FAILURE iff E == {NON_CONVERGED} OR "
    "(d is REJECT/SOLVER_FAILURE AND NON_CONVERGED in E), otherwise CONTRACT_VIOLATION; finally, if d is REJECT with class "
    "INFEASIBLE_REQUEST or UNSUPPORTED_BINDING that class wins; reasons = d reasons U E",
    "P5 consequences that W119 mis-declared and W120 must honour: pin tamper (changes pin values vs F_r) adds "
    "AB_FAIRNESS:PIN_VALUES_DIFFER_FROM_F -> CONTRACT_VIOLATION unless NON_CONVERGED co-occurs with a SOLVER_FAILURE base; "
    "an isolated nonconvergent solve with valid fairness -> SOLVER_FAILURE; nonconvergence plus an output-scope violation "
    "(E == {NON_CONVERGED}) -> SOLVER_FAILURE; an unequal budget (zero-budget request with M_MAX=8 budget) -> CONTRACT_VIOLATION",
]

CONDITION_KEYS = ("validation", "missing_dof", "pin_status", "eval_pin", "eval_joint", "eval_scope", "eval_root", "fp_budget",
                  "fp_domain", "fp_channel", "fp_outside_arm", "fp_root", "display_bad", "nonconv", "margin_exceed",
                  "isolation_bad", "fairness_bad")


def predict_decision(c):
    """Independent implementation of PRECEDENCE_TABLE from a dict of injected conditions (keys CONDITION_KEYS).
    validation: None or a W116 class string; pin_status: None or a non-solved resolver status; others booleans.
    Returns {"decision", "failure_class"}."""
    if c.get("validation"):
        return {"decision": "REJECT", "failure_class": c["validation"]}
    if c.get("missing_dof"):
        return {"decision": "REJECT", "failure_class": "INFEASIBLE_REQUEST"}
    statuses_ok = not c.get("pin_status")
    if not statuses_ok:
        d = "SOLVER_FAILURE" if c["pin_status"] in ("RESOLVER_UNSUPPORTED", "NON_CONVERGED") else "INFEASIBLE_REQUEST"
    else:
        ev = {"pin": c.get("eval_pin"), "joint": c.get("eval_joint"), "scope": c.get("eval_scope"), "root": c.get("eval_root")}
        if ev["scope"] or ev["root"]:
            d = "CONTRACT_VIOLATION"
        elif ev["pin"] or ev["joint"]:
            d = "SOLVER_FAILURE"
        else:
            d = None    # ACCEPT so far
    fp_extra = any(c.get(k) for k in ("fp_budget", "fp_domain", "fp_channel", "fp_outside_arm", "fp_root", "display_bad"))
    if fp_extra:
        if d is None or statuses_ok:
            d = "CONTRACT_VIOLATION"
    E = set()
    if c.get("nonconv"):
        E.add("NON_CONVERGED")
    for k in ("margin_exceed", "isolation_bad", "fairness_bad"):
        if c.get(k):
            E.add(k)
    if not E:
        return {"decision": "ACCEPT", "failure_class": "NONE"} if d is None else {"decision": "REJECT", "failure_class": d}
    cls = "SOLVER_FAILURE" if E == {"NON_CONVERGED"} or (d == "SOLVER_FAILURE" and "NON_CONVERGED" in E) else "CONTRACT_VIOLATION"
    if d in ("INFEASIBLE_REQUEST", "UNSUPPORTED_BINDING"):
        cls = d
    return {"decision": "REJECT", "failure_class": cls}


def synth_decide_inputs(c):
    """Builds decide_y inputs that realise the injected conditions (used to test predict_decision against decide_y)."""
    verr = [(c["validation"], "INJECTED_" + c["validation"])] if c.get("validation") else []
    statuses = ["SOLVED", c["pin_status"]] if c.get("pin_status") else ["SOLVED", "SOLVED"]
    ev = {"pins_satisfied": not c.get("eval_pin"), "scope_ok": not c.get("eval_scope"), "locality_ok": True, "lengths_ok": True,
          "root_unchanged": not c.get("eval_root"), "joint_limit_ok": not c.get("eval_joint")}
    ok = {"pins_satisfied": True, "scope_ok": True, "locality_ok": True, "lengths_ok": True, "root_unchanged": True, "joint_limit_ok": True}
    budget, planned = [10, 11, 12, 13], [10, 11, 12]
    fc = [11, 12]
    if c.get("fp_budget"):
        fc = fc + [20]
    if c.get("fp_domain"):
        fc = fc + [13]
    fp = {"frames_changed": sorted(set(fc)), "channel_frames": {"upper": [11]} | ({"base": [11]} if c.get("fp_channel") else {}),
          "outside_arm_points_changed": {"11": ["base.tail"]} if c.get("fp_outside_arm") else {},
          "matrix_world_changed_frames": [11] if c.get("fp_root") else []}
    if c.get("fp_budget"):
        planned = planned + [20]     # isolate the budget extra from the planned-domain extra
    yi = {"request_status": "MISSING_AUTHORIZED_DOF" if c.get("missing_dof") else
          ("MARGIN_EXCEEDS_AVAILABLE_BUDGET" if c.get("margin_exceed") else ("NON_CONVERGED" if c.get("nonconv") else "SOLVED"))}
    fair = {"ok": not c.get("fairness_bad"), "errors": ["INJECTED_MISMATCH:Y"] if c.get("fairness_bad") else []}
    iso = {"ok": not c.get("isolation_bad")}
    disp = {"ok": not c.get("display_bad"), "errors": ["CELL_MISMATCH"] if c.get("display_bad") else []}
    return dict(validation_errors=verr, pin_statuses=statuses, evaluations=[ok, ev], fp=fp, budget_frames=budget, planned_frames=planned,
                auth=["upper", "lower"], y_info=yi, fairness=fair, display_check=disp, isolation=iso)


def precedence_exhaustive_check():
    """Every combination of the injected conditions (margin_exceed and nonconv are exclusive request statuses) through the
    actual W119 decide_y vs the frozen table predictor.  Returns {"n", "mismatches"}."""
    import itertools
    bools = ("missing_dof", "eval_pin", "eval_joint", "eval_scope", "eval_root", "fp_budget", "fp_domain", "fp_channel",
             "fp_outside_arm", "fp_root", "display_bad", "isolation_bad", "fairness_bad")
    mism, n = [], 0
    for validation in (None, "INFEASIBLE_REQUEST", "UNSUPPORTED_BINDING", "CONTRACT_VIOLATION"):
        for pin_status in (None, "UNREACHABLE", "NON_CONVERGED", "INSUFFICIENT_DOF"):
            for status in ("SOLVED", "NON_CONVERGED", "MARGIN_EXCEEDS_AVAILABLE_BUDGET"):
                for bits in itertools.product((False, True), repeat=len(bools)):
                    c = dict(zip(bools, bits), validation=validation, pin_status=pin_status,
                             nonconv=status == "NON_CONVERGED", margin_exceed=status == "MARGIN_EXCEEDS_AVAILABLE_BUDGET")
                    if c["missing_dof"] and status != "SOLVED":
                        continue
                    n += 1
                    got = YB.decide_y(**synth_decide_inputs(c))
                    want = predict_decision(c)
                    if [got["decision"], got["failure_class"]] != [want["decision"], want["failure_class"]]:
                        mism.append({"conditions": {k: v for k, v in c.items() if v}, "decide_y": got, "table": want})
    return {"n": n, "mismatches": mism[:50], "n_mismatches": len(mism)}


# --------------------------------------------------------------------------------------------- contrasts ---
METRICS = ("B1_boundary_wrist_comp_step_peak_m", "D2_elbow_comp_step_peak_m", "D3_arm_rot_step_excess_peak_deg",
           "boundary_raw_wrist_step_peak_m", "cost_mean_elbow_dev_from_U_m", "cost_peak_elbow_dev_from_U_m", "min_flex_deg")
CONTRASTS = {"guide_at_F0": ("G0F0", "G1F0"), "guide_at_F1": ("G0F1", "G1F1"), "floor_at_G0": ("G0F0", "G0F1"),
             "floor_at_G1": ("G1F0", "G1F1")}
PRIMARY_CONTRAST = {"GUIDE": "guide_at_F0", "FLOOR": "floor_at_G0"}
CONDITIONAL_CONTRAST = {"GUIDE": "guide_at_F1", "FLOOR": "floor_at_G1"}
INTERACTION_ABS_M = YB.IMPROVE_ABS_M     # |interaction on B1| >= 10 mm = material interaction (experiment-only diagnostic)


def _counts(q):
    return {"n_introduced_flips": len(q["introduced_flip_transitions"]), "n_introduced_near_straight": len(q["A4_introduced_near_straight_frames"])}


def factorial(q):
    """q: {factor: Blender quality dict} for the four variants of one request.  Raw differences (target - reference) per metric,
    the frozen W119 diagnostic comparator in both directions, and the interaction (G1F1 - G1F0 - G0F1 + G0F0)."""
    vals = {v: {m: q[v][m] for m in METRICS} | _counts(q[v]) for v in FACTORS}
    out = {"values": vals, "contrasts": {}}
    for name, (ref, tgt) in CONTRASTS.items():
        d = {m: (None if vals[tgt][m] is None or vals[ref][m] is None else vals[tgt][m] - vals[ref][m]) for m in vals[ref]}
        b_ref = vals[ref]["B1_boundary_wrist_comp_step_peak_m"]
        rel = (d["B1_boundary_wrist_comp_step_peak_m"] / b_ref) if (b_ref and d["B1_boundary_wrist_comp_step_peak_m"] is not None) else None
        out["contrasts"][name] = {"reference": ref, "target": tgt, "delta": d, "B1_relative_to_reference": rel,
                                  "improves": YB.compare_y(q[ref], q[tgt]), "reverse": YB.compare_y(q[tgt], q[ref]),
                                  "new_flips": sorted(set(q[tgt]["introduced_flip_transitions"]) - set(q[ref]["introduced_flip_transitions"])),
                                  "new_near_straight": sorted(set(q[tgt]["A4_introduced_near_straight_frames"])
                                                              - set(q[ref]["A4_introduced_near_straight_frames"]))}
    inter = {}
    for m in vals["G0F0"]:
        xs = [vals[v][m] for v in FACTORS]
        inter[m] = None if any(x is None for x in xs) else vals["G1F1"][m] - vals["G1F0"][m] - vals["G0F1"][m] + vals["G0F0"][m]
    ib = inter["B1_boundary_wrist_comp_step_peak_m"]
    out["interaction"] = {"delta": inter, "B1_material": ib is not None and abs(ib) >= INTERACTION_ABS_M,
                          "sign": None if ib is None else ("negative (joint B1 lower than additive)" if ib < 0 else
                                                           "positive (joint B1 higher than additive)" if ib > 0 else "zero")}
    for M, cn in PRIMARY_CONTRAST.items():
        c = out["contrasts"][cn]
        out[f"{M}_primary"] = {"improvement": c["improves"]["meaningful_improvement"], "material_worsening": c["reverse"]["meaningful_improvement"],
                               "new_articulation_defect": bool(c["new_flips"] or c["new_near_straight"])}
    return out


def endpoint_jumps(q, pin_runs, margins):
    """Compensation and raw wrist steps at pin entry/release and the peak inside each transition (Blender quality dict)."""
    pt = q["per_transition"]
    g = lambda k, key: pt.get(str(k), {}).get(key)  # noqa: E731
    out = {}
    for lo, hi in pin_runs:
        me, mr = int(margins.get(f"{lo}:{hi}:entry", 0)), int(margins.get(f"{lo}:{hi}:release", 0))
        ent_in = [k for k in range(lo - me, lo) if str(k) in pt]
        rel_in = [k for k in range(hi + 2, hi + mr + 2) if str(k) in pt]
        out[f"{lo}:{hi}"] = {
            "entry_jump_comp_m": g(lo, "wrist_comp_step_m"), "entry_jump_raw_m": g(lo, "wrist_step_m"),
            "entry_jump_raw_U_m": g(lo, "control_wrist_step_m"),
            "release_jump_comp_m": g(hi + 1, "wrist_comp_step_m"), "release_jump_raw_m": g(hi + 1, "wrist_step_m"),
            "release_jump_raw_U_m": g(hi + 1, "control_wrist_step_m"),
            "entry_inner_comp_peak_m": max([g(k, "wrist_comp_step_m") for k in ent_in] + [0.0]),
            "release_inner_comp_peak_m": max([g(k, "wrist_comp_step_m") for k in rel_in] + [0.0]),
            "entry_inner_raw_excess_peak_m": max([g(k, "wrist_step_m") - g(k, "control_wrist_step_m") for k in ent_in] + [0.0]),
            "release_inner_raw_excess_peak_m": max([g(k, "wrist_step_m") - g(k, "control_wrist_step_m") for k in rel_in] + [0.0])}
    return out


def _bn(fr, upper, lower):
    Sh, E, W = (np.array(fr["world"][upper]["head"]), np.array(fr["world"][upper]["tail"]), np.array(fr["world"][lower]["tail"]))
    return YB._normal(Sh, E, W), YB._flex(Sh, E, W), E, W


def variant_diff(A, B, frames, upper, lower):
    """Blender-evaluated difference of two variants on the given frames: wrist/elbow distance, stored rotation angle per arm
    channel, and bend-branch disagreement (opposite bend-plane normal where both normals are defined)."""
    per = {}
    for k in frames:
        na, fa, Ea, Wa = _bn(A[k], upper, lower)
        nb, fb, Eb, Wb = _bn(B[k], upper, lower)
        per[str(k)] = {"wrist_m": float(np.linalg.norm(Wa - Wb)), "elbow_m": float(np.linalg.norm(Ea - Eb)),
                       "rot_deg": {c: TB.qangle(A[k]["stored_wxyz"][c], B[k]["stored_wxyz"][c]) for c in (upper, lower)},
                       "flex_a_deg": fa, "flex_b_deg": fb,
                       "opposite_branch": bool(na is not None and nb is not None and float(np.dot(na, nb)) < 0)}
    v = list(per.values())
    return {"frames": frames, "max_wrist_m": max([x["wrist_m"] for x in v] + [0.0]), "max_elbow_m": max([x["elbow_m"] for x in v] + [0.0]),
            "max_rot_deg": {c: max([x["rot_deg"][c] for x in v] + [0.0]) for c in (upper, lower)},
            "opposite_branch_frames": [int(k) for k, x in per.items() if x["opposite_branch"]], "per_frame": per}
