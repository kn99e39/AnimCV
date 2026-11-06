#!/usr/bin/env python3
"""W119 scratch endpoint-space (wrist world-space) transition candidate Y + baseline-isolation / A-B fairness instruments
(NON-PRODUCTION, evidence only).

Reuses the frozen W116 rig/FK/resolver/evaluator/store, the frozen W117 temporal metrics and the frozen W118 domain,
footprint, display, decision and quality code UNCHANGED (imported).  Adds:

  A. Baseline isolation (`isolation_check_values`, `isolation_check_footprint`).  A per-request F_r is valid only if every
     frame whose stored values differ from U is one of the request's own pin frames (value level, before Blender) and
     its Blender-evaluated actual footprint vs U is a subset of the pin frames (evaluator level, after Blender).  A
     family-shared historical SEG_<family>__F fails both.

  B. A/B fairness (`ab_fairness`).  F_r, X_r and Y_r of one request must share parent state id, rig id, U state id,
     pin constraint digests, pin frames, authorised channels; X_r and Y_r must use identical per-side transition margins
     and identical pin-frame values (= F_r).

  C. Candidate Y (`endpoint_transition`) - SOLVER side.  Pin frames = the frozen W116 framewise values (= F_r), untouched.
     On the m frames beyond each pin-run boundary b (m = the frozen X_r margin of that side, never chosen by Y):
         w_j   = 1 - smoothstep(j / (m + 1))                              (same weight profile as X)
         G(k)  = W_U(k) + w_j (W_F(b) - W_U(b))                           wrist WORLD-SPACE guide (soft objective)
         R(k)  = E_U(k) + w_j (E_F(b) - E_U(b))                           elbow world-space pole reference
     Each guide is resolved on the unchanged rig with the authorised two-bone chain only, analytically:
       reach    d = |G - S|; reach band [r_min, r_max] from the W116 flex limits (0..150 deg);
       flex floor (near-straight degeneracy): the transition arm is never made straighter than BOTH poses it blends
                between: phi_floor = min(flex_U(k), flex_F(b)); reach used d_c = clip(d, r_min, min(r_max, r(phi_floor)));
       bend plane: elbow on the circle point nearest to R(k); if R(k) lies within Y_AMBIG_DEG of the shoulder->guide
                axis the plane is AMBIGUOUS and the previous (boundary-side) solved elbow direction is used (continuity
                pole); the event is disclosed;
       solve    upper: minimal world swing taking the elbow to the chosen circle point; lower: minimal swing aiming the
                wrist at S + d_c t;
       check    FK residual to the (clamped) guide <= Y_INTERNAL_TOL_M, else NON_CONVERGED (solver failure).
     Per-frame statuses: GUIDE_REACHED | GUIDE_FLEX_FLOOR_CLAMPED | GUIDE_UNREACHABLE_OUTER | GUIDE_UNREACHABLE_INNER,
     flags BEND_PLANE_AMBIGUOUS, BRANCH_FLIP_VS_PREVIOUS (reported), NON_CONVERGED.  Request: SOLVED | NON_CONVERGED |
     MISSING_AUTHORIZED_DOF.  No fallback to F or X exists.
"""
from __future__ import annotations

import math
import time

import numpy as np

import w116_semantic_pin_base as S
import w117_temporal_pin_base as TB
import w118_transition_edit_base as XB

# ------------------------------------------------------------------------------------------ declared W119 parameters ---
Y_AMBIG_DEG = 2.0           # pole reference within this angle of the shoulder->guide axis = bend plane ambiguous
Y_INTERNAL_TOL_M = 1e-6     # post-solve FK residual to the (clamped) guide; above = NON_CONVERGED
Y_FLIP_FLEX_FLOOR_DEG = TB.FLIP_FLEX_FLOOR_DEG   # bend normal defined only when flex >= this (same as W117 metric)

# declared Y-vs-X primary improvement test (experiment-only; frozen before any Y evaluation)
IMPROVE_RATIO = 0.8          # B1(Y) <= 0.8 B1(X)  (decrease >= 20 %)
IMPROVE_ABS_M = 0.010        # and B1(X) - B1(Y) >= 0.010 m
WORSE_REL = 0.25             # material worsening: Y - X > max(WORSE_REL * X, floor)
WORSE_FLOOR = {"D2_elbow_comp_step_peak_m": 0.005, "D3_arm_rot_step_excess_peak_deg": 2.0,
               "boundary_raw_wrist_step_peak_m": 0.005}

Y_STATUSES = ("GUIDE_REACHED", "GUIDE_FLEX_FLOOR_CLAMPED", "GUIDE_UNREACHABLE_OUTER", "GUIDE_UNREACHABLE_INNER")


# ------------------------------------------------------------------------------------------------ A. isolation ---
def _same(a, b):
    return all(np.array_equal(np.asarray(a[c], float), np.asarray(b[c], float)) for c in set(a) | set(b))


def isolation_check_values(F_vals, U_vals, pin_frames, frames):
    """Value level: frames where F_r differs from U must all be pin frames of THIS request."""
    pin = set(pin_frames)
    diff = [k for k in frames if not _same(F_vals[k], U_vals[k])]
    out = [k for k in diff if k not in pin]
    return {"ok": not out, "changed_frames": diff, "changed_outside_pin": out, "n_changed_outside_pin": len(out),
            "reason": None if not out else "BASELINE_NOT_ISOLATED"}


def isolation_check_footprint(fp, pin_frames):
    """Evaluator level (Blender): actual changed frames of F_r vs U must be a subset of the pin frames."""
    out = [k for k in fp["frames_changed"] if k not in set(pin_frames)]
    return {"ok": not out, "changed_outside_pin": out, "runs_outside_pin": XB.runs_of(out),
            "reason": None if not out else "BASELINE_NOT_ISOLATED"}


# ------------------------------------------------------------------------------------------------ B. fairness ---
def ab_fairness(recs):
    """recs: {variant: {parent_id, rig_id, U_state_id, constraint_digests, pin_frames, auth, margins?, pin_values?}}."""
    errs = []
    keys = (("parent_id", "PARENT_MISMATCH"), ("rig_id", "RIG_IDENTITY_MISMATCH"), ("U_state_id", "U_MISMATCH"),
            ("constraint_digests", "PIN_CONSTRAINT_MISMATCH"), ("pin_frames", "PIN_FRAMES_MISMATCH"), ("auth", "AUTH_MISMATCH"))
    ref = recs.get("F")
    for v, r in recs.items():
        if v == "F" or ref is None:
            continue
        for k, why in keys:
            if r.get(k) != ref.get(k):
                errs.append(f"{why}:{v}")
    if "X" in recs and "Y" in recs:
        if recs["X"].get("margins") != recs["Y"].get("margins"):
            errs.append("MARGIN_BUDGET_MISMATCH:Y")
        if recs["X"].get("budget_frames") != recs["Y"].get("budget_frames"):
            errs.append("SEARCH_BUDGET_MISMATCH:Y")
    for v in ("X", "Y"):
        if v in recs and ref is not None and recs[v].get("pin_values_digest") != ref.get("pin_values_digest"):
            errs.append(f"PIN_VALUES_DIFFER_FROM_F:{v}")
    return {"ok": not errs, "errors": sorted(set(errs))}


def margins_of(x_info):
    """Frozen per-side X margins {"lo:hi:side": m}."""
    return {f"{s['run'][0]}:{s['run'][1]}:{s['side']}": int(s["chosen_margin"]) for s in x_info.get("sides", [])}


# ------------------------------------------------------------------------------------------- C. candidate Y ---
def _pts(rig, vals, upper, lower):
    fk = rig.fk(vals)
    return fk, rig.world(fk, upper, "head"), rig.world(fk, upper, "tail"), rig.world(fk, lower, "tail")


def _flex(Sh, E, W):
    u, v = E - Sh, W - E
    return math.degrees(math.atan2(float(np.linalg.norm(np.cross(u, v))), float(np.dot(u, v))))


def _normal(Sh, E, W):
    n = np.cross(E - Sh, W - E)
    if _flex(Sh, E, W) < Y_FLIP_FLEX_FLOOR_DEG:
        return None
    return n / np.linalg.norm(n)


def _reach(L1, L2, phi_deg):
    return math.sqrt(L1 * L1 + L2 * L2 + 2 * L1 * L2 * math.cos(math.radians(phi_deg)))


def solve_guide(rig, ctrl_vals, G, Rref, prev_elbow, upper, lower, phi_floor, fault=None):
    """Resolve ONE transition frame toward wrist guide G with elbow pole reference Rref (authorised upper/lower only)."""
    vals = {c: np.array(v, float) for c, v in ctrl_vals.items()}
    fk, Sh, E, W = _pts(rig, vals, upper, lower)
    L1, L2 = rig.world_len(upper), rig.world_len(lower)
    rmin, rmax = S._limits_reach(L1, L2)
    rfloor = _reach(L1, L2, max(0.0, phi_floor))
    d = float(np.linalg.norm(G - Sh))
    status = "GUIDE_REACHED"
    if d > rmax:
        status = "GUIDE_UNREACHABLE_OUTER"
    elif d < rmin:
        status = "GUIDE_UNREACHABLE_INNER"
    elif d > rfloor:
        status = "GUIDE_FLEX_FLOOR_CLAMPED"
    dc = min(max(d, rmin), min(rmax, rfloor))
    t = (G - Sh) / d
    Gc = Sh + dc * t
    flags = []
    r = Rref - Sh
    bperp = r - np.dot(r, t) * t
    nr = float(np.linalg.norm(r))
    if nr < 1e-12 or float(np.linalg.norm(bperp)) < nr * math.sin(math.radians(Y_AMBIG_DEG)):
        flags.append("BEND_PLANE_AMBIGUOUS")
        q = prev_elbow - Sh
        bperp = q - np.dot(q, t) * t
        if float(np.linalg.norm(bperp)) < 1e-12:
            ax = rig.world_rot(fk, upper) @ np.array([1.0, 0.0, 0.0])
            bperp = ax - np.dot(ax, t) * t
            flags.append("POLE_AXIS_FALLBACK")
    bhat = bperp / np.linalg.norm(bperp)
    ca = max(-1.0, min(1.0, (L1 * L1 + dc * dc - L2 * L2) / (2 * L1 * dc)))
    Enew = Sh + L1 * (ca * t + math.sqrt(max(0.0, 1 - ca * ca)) * bhat)
    vals = rig.apply_world_swing(vals, fk, upper, S.q_between(E - Sh, Enew - Sh))
    fk = rig.fk(vals)
    E2, W2 = rig.world(fk, lower, "head"), rig.world(fk, lower, "tail")
    if fault != "skip_lower_aim":
        vals = rig.apply_world_swing(vals, fk, lower, S.q_between(W2 - E2, Gc - E2))
    fk, Sh3, E3, W3 = _pts(rig, vals, upper, lower)
    internal = float(np.linalg.norm(W3 - Gc))
    if internal > Y_INTERNAL_TOL_M:
        flags.append("NON_CONVERGED")
    info = {"status": status, "flags": flags, "guide": G.tolist(), "guide_clamped": Gc.tolist(), "pole_ref": Rref.tolist(),
            "shoulder_guide_m": d, "reach_used_m": dc, "reach_min_m": rmin, "reach_max_m": rmax, "reach_flex_floor_m": rfloor,
            "flex_floor_deg": phi_floor, "flex_deg": _flex(Sh3, E3, W3), "guide_shortfall_fk_m": float(np.linalg.norm(W3 - G)),
            "internal_residual_m": internal, "elbow": E3.tolist(), "wrist": W3.tolist()}
    return vals, info


def endpoint_transition(rig, ctrl, base, pin_frames, all_frames, auth, upper, lower, margins, fault=None,
                        guide_override=None, pole_override=None):
    """ctrl: {k: U values}; base: {k: F_r values} on pin frames; margins: frozen X margins {"lo:hi:side": m}.
    Returns (values{k} for pin + transition frames, info).  Every other frame stays U (not returned)."""
    t0 = time.time()
    plan, budget = XB.budget_plan(pin_frames, all_frames, XB.M_MAX)
    out = {k: {c: np.array(v, float) for c, v in base[k].items()} for k in pin_frames}
    info = {"method": "w119_endpoint_world_space_guide", "plan": plan, "budget_frames": budget, "margins": dict(margins),
            "sides": [], "fallback": None, "frames": {}}
    if not ({upper, lower} <= set(auth)):
        info.update(request_status="MISSING_AUTHORIZED_DOF", planned_frames=sorted(pin_frames), wall_s=time.time() - t0)
        return out, info
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
            _, SB, EF, WF = _pts(rig, base[b], upper, lower)
            _, _, EU, WU = _pts(rig, ctrl[b], upper, lower)
            cW, cE = WF - WU, EF - EU
            phiB = _flex(SB, EF, WF)
            rec.update(boundary_wrist_offset_m=float(np.linalg.norm(cW)), boundary_elbow_offset_m=float(np.linalg.norm(cE)),
                       boundary_flex_deg=phiB)
            prev_E, prev_n = EF, _normal(SB, EF, WF)
            for j in range(1, m + 1):
                k = b + sgn * j
                w = 1.0 - TB.smoothstep(j / (m + 1))
                _, Sk, Ek, Wk = _pts(rig, ctrl[k], upper, lower)
                G = Wk + w * cW
                R = Ek + w * cE
                if guide_override is not None:
                    G = guide_override(k, Sk, G, rig.world_len(upper) + rig.world_len(lower))
                if pole_override is not None:
                    R = pole_override(k, Sk, G, R)
                vals, fi = solve_guide(rig, ctrl[k], G, R, prev_E, upper, lower, min(_flex(Sk, Ek, Wk), phiB), fault=fault)
                Sn, En, Wn = Sk, np.array(fi["elbow"]), np.array(fi["wrist"])
                n = _normal(Sn, En, Wn)
                if n is not None and prev_n is not None and float(np.dot(n, prev_n)) < 0:
                    fi["flags"].append("BRANCH_FLIP_VS_PREVIOUS")
                fi.update(weight=w, j=j, side=side, boundary=b, control_flex_deg=_flex(Sk, Ek, Wk))
                info["frames"][str(k)] = fi
                out[k] = vals
                rec["frames"].append(k)
                prev_E, prev_n = En, (n if n is not None else prev_n)
            info["sides"].append(rec)
    fr = info["frames"].values()
    bad_margin = [s for s in info["sides"] if s.get("error")]
    info["planned_frames"] = sorted(set(pin_frames) | {k for s in info["sides"] for k in s["frames"]})
    info["status_counts"] = {s: sum(1 for f in fr if f["status"] == s) for s in Y_STATUSES}
    info["flag_counts"] = {fl: sum(1 for f in fr if fl in f["flags"]) for fl in
                           ("BEND_PLANE_AMBIGUOUS", "POLE_AXIS_FALLBACK", "BRANCH_FLIP_VS_PREVIOUS", "NON_CONVERGED")}
    if bad_margin:
        info["request_status"] = "MARGIN_EXCEEDS_AVAILABLE_BUDGET"
    elif info["flag_counts"]["NON_CONVERGED"]:
        info["request_status"] = "NON_CONVERGED"
    else:
        info["request_status"] = "SOLVED"
    info["wall_s"] = time.time() - t0
    return out, info


# ------------------------------------------------------------------------------------------------ decision ---
def decide_y(validation_errors, pin_statuses, evaluations, fp, budget_frames, planned_frames, auth, y_info, fairness=None,
             display_check=None, isolation=None):
    """Atomic.  Pre-checks (contract, Y request status, baseline isolation, A/B fairness) then W118 decide_x."""
    if validation_errors:
        return S.decide(validation_errors, pin_statuses, evaluations)
    st = (y_info or {}).get("request_status")
    if st == "MISSING_AUTHORIZED_DOF":
        return {"decision": "REJECT", "failure_class": "INFEASIBLE_REQUEST", "reasons": ["Y_MISSING_AUTHORIZED_DOF"]}
    d = XB.decide_x([], pin_statuses, evaluations, fp, budget_frames, planned_frames, auth, display_check=display_check)
    extra = set()
    if st == "NON_CONVERGED":
        extra.add("NON_CONVERGED")
    if st == "MARGIN_EXCEEDS_AVAILABLE_BUDGET":
        extra.add("MARGIN_EXCEEDS_AVAILABLE_BUDGET")
    if isolation is not None and not isolation["ok"]:
        extra.add("BASELINE_NOT_ISOLATED")
    if fairness is not None and not fairness["ok"]:
        extra |= {"AB_FAIRNESS:" + e for e in fairness["errors"]}
    if not extra:
        return d
    cls = "SOLVER_FAILURE" if extra == {"NON_CONVERGED"} or (d["decision"] == "REJECT" and d["failure_class"] == "SOLVER_FAILURE"
                                                            and "NON_CONVERGED" in extra) else "CONTRACT_VIOLATION"
    if d["decision"] == "REJECT" and d["failure_class"] in ("INFEASIBLE_REQUEST", "UNSUPPORTED_BINDING"):
        cls = d["failure_class"]
    return {"decision": "REJECT", "failure_class": cls,
            "reasons": sorted((set(d.get("reasons", [])) | extra) - {"TRIVIALLY_SATISFIED_NOT_EVIDENCE"})}


# ------------------------------------------------------------------------------------------------ comparison ---
def _worse(x, y, key):
    if x is None or y is None:
        return False
    return y - x > max(WORSE_REL * x, WORSE_FLOOR[key])


def compare_y(X, Y):
    """Declared W119 primary test: Y_r vs clean X_r on the same request and window (both Blender quality dicts)."""
    bX, bY = X["B1_boundary_wrist_comp_step_peak_m"], Y["B1_boundary_wrist_comp_step_peak_m"]
    rec = {"B1_X_m": bX, "B1_Y_m": bY}
    b1 = bX is not None and bY is not None and bY <= IMPROVE_RATIO * bX and bX - bY >= IMPROVE_ABS_M
    new_flip = sorted(set(Y["introduced_flip_transitions"]) - set(X["introduced_flip_transitions"]))
    new_straight = sorted(set(Y["A4_introduced_near_straight_frames"]) - set(X["A4_introduced_near_straight_frames"]))
    worse = []
    if new_flip:
        worse.append("NEW_BEND_FLIP")
    if new_straight:
        worse.append("NEW_NEAR_STRAIGHT")
    for k in ("D2_elbow_comp_step_peak_m", "D3_arm_rot_step_excess_peak_deg", "boundary_raw_wrist_step_peak_m"):
        if _worse(X[k], Y[k], k):
            worse.append("WORSE_" + k)
    rec.update(B1_improved=b1, new_flip_transitions=new_flip, new_near_straight_frames=new_straight, worse=worse,
               meaningful_improvement=bool(b1) and not worse,
               ratio=(bY / bX) if (bX and bY is not None) else None, delta_m=(bX - bY) if (bX is not None and bY is not None) else None)
    return rec


def guide_residuals(Yf, y_info, lower):
    """Evaluator side: Blender wrist of Y vs the declared guide on every transition frame."""
    out = {}
    for k, fi in y_info.get("frames", {}).items():
        w = np.array(Yf[int(k)]["world"][lower]["tail"], float)
        out[k] = {"to_guide_m": float(np.linalg.norm(w - np.array(fi["guide"]))),
                  "to_clamped_guide_m": float(np.linalg.norm(w - np.array(fi["guide_clamped"]))), "status": fi["status"],
                  "flags": fi["flags"]}
    return out
