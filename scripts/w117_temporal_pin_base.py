#!/usr/bin/env python3
"""W117 scratch temporally aware WORLD_POSITION_PIN resolver + temporal evaluator (NON-PRODUCTION, evidence only).

Reuses the frozen W116 contract, binding, FK model, store and per-frame evaluator UNCHANGED (`w116_semantic_pin_base`).
Adds two separated responsibilities:

  A. Temporal resolver (`temporal_two_bone`) — SOLVER side.  For a declared ordered run of frames it chooses, per frame,
     the elbow swivel angle of the authorised two-bone chain.  Every swivel angle reaches the SAME wrist position as the
     W116 framewise analytic solution (law of cosines; same reach band, same clamp rule), so the hard pin is never traded
     for smoothness.  The swivel sequence minimises a declared SOLVER QUALITY OBJECTIVE (not a semantic constraint):

         J(phi) = sum_k |c_k|^2 + LAMBDA * sum_k |c_k - c_{k-1}|^2,   c_k = A_k^T (E_k(phi_k) - E^U_k) / L1

     E^U = elbow of the unpinned user edit (control), A_k = world rotation of the upper bone's rest frame under its posed
     parent (body-attached frame).  With LAMBDA = 0 the per-frame minimiser is exactly the W116 pole (projection of the
     control elbow), so T reduces to F.  Global initialisation: Viterbi DP over a declared swivel grid; refinement:
     L-BFGS-B from both the DP and the F solution (declared bounded two-start search); the lower J is kept.
     Optional declared transition margins (variant T_E): frames outside the pin interval where the SAME authorised
     channels are re-solved toward an internal guide that decays the boundary wrist offset to the unpinned motion with
     a smoothstep.  Guide frames carry NO hard pin and are not constraint frames.

  B. Temporal evaluator (`temporal_metrics`, `evaluate_temporal_request`) — EVALUATOR side.  Reads ONLY Blender-evaluated
     frames (world head/tail, stored F-curve quaternions, matrix_world) and Blender rest lengths.  Never reads solver
     internals.  Quaternion steps are sign-safe (2*acos|<q1,q2>|).
"""
from __future__ import annotations

import math
import time

import numpy as np

import w116_semantic_pin_base as S

# ------------------------------------------------------------------------------------- declared solver constants ---
LAMBDA = 20.0            # compensation-smoothness weight (dimensionless, per Action frame); declared a priori, not tuned
GRID = 360               # DP swivel grid (1 degree)
MAXITER = 1000           # L-BFGS-B iteration cap per start
GTOL = 1e-7              # convergence: max |dJ/dphi| at the returned solution
RUNTIME_BUDGET_S = 120.0  # per request; exceeding it is reported as SOLVER_FAILURE (TIMEOUT)
MARGIN = 4               # T_E transition frames on each side of a hold interval

# ------------------------------------------------------------------------------------ declared metric constants ---
FLIP_FLEX_FLOOR_DEG = 2.0       # bend-plane normal is defined only when elbow flex >= this
NEAR_SINGULAR_SLACK_M = 0.005   # reach slack below this = near-singular frame (reported)
STRAIGHT_FLEX_DEG = 5.0         # pin frames with flex below this are reported as near-straight
FLOORS = {"D2": 0.005, "D3": 5.0, "D4": 0.005}    # absolute materiality floors (m, deg/frame, m)
WORSE_SIDE = {"S1_mean_elbow_dev_m": 0.010, "S2_peak_upper_comp_deg": 20.0, "S3_peak_flex_change_deg": 10.0}


# ------------------------------------------------------------------------------------------------ quaternions ---
def qangle(a, b):
    """Sign-safe geodesic angle (deg) between two unit quaternions (any consistent component order)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    d = abs(float(np.dot(a / np.linalg.norm(a), b / np.linalg.norm(b))))
    return math.degrees(2.0 * math.acos(min(1.0, d)))


def wxyz_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw])


def wxyz_inv(q):
    q = np.asarray(q, float)
    return np.array([q[0], -q[1], -q[2], -q[3]]) / float(np.dot(q, q))


def rot_first_order(seq):
    """seq: list of wxyz.  First-order steps (deg/frame), sign-safe."""
    return [qangle(seq[i], seq[i - 1]) for i in range(1, len(seq))]


def rot_second_order(seq):
    """Angle (deg/frame^2) between successive relative rotations r_k = q_k q_{k-1}^-1 (sign-safe)."""
    r = [wxyz_mul(np.asarray(seq[i], float), wxyz_inv(seq[i - 1])) for i in range(1, len(seq))]
    return [qangle(r[i], r[i - 1]) for i in range(1, len(r))]


def smoothstep(x):
    x = min(1.0, max(0.0, x))
    return 3 * x * x - 2 * x ** 3


# ------------------------------------------------------------------------------------------ temporal resolver ---
def _perp(v, t):
    p = v - np.dot(v, t) * t
    n = np.linalg.norm(p)
    return p / n if n > 1e-12 else None


def _frame_geometry(rig, vals, upper, lower, target, pinned):
    fk = rig.fk(vals)
    Sh, Eu, Wu = rig.world(fk, upper, "head"), rig.world(fk, upper, "tail"), rig.world(fk, lower, "tail")
    L1, L2 = rig.world_len(upper), rig.world_len(lower)
    rmin, rmax = S._limits_reach(L1, L2)
    dist = float(np.linalg.norm(target - Sh))
    unreachable = pinned and (dist > rmax + S.TOL_M or dist < rmin - S.TOL_M)
    dc = min(max(dist, rmin), rmax)
    t = (target - Sh) / dist
    ca = max(-1.0, min(1.0, (L1 * L1 + dc * dc - L2 * L2) / (2 * L1 * dc)))
    C = Sh + L1 * ca * t
    rho = L1 * math.sqrt(max(0.0, 1 - ca * ca))
    A = rig.frame_of_rest(fk, upper)
    return {"fk": fk, "S": Sh, "Eu": Eu, "Wu": Wu, "L1": L1, "L2": L2, "rmin": rmin, "rmax": rmax, "dist": dist,
            "unreachable": unreachable, "t": t, "C": C, "rho": rho, "A": A, "target": target,
            "upax": rig.world_rot(fk, upper) @ np.array([1.0, 0.0, 0.0])}


def _bases(geo):
    """Parallel-transported (u, v) swivel bases along the run (continuous parameterisation)."""
    prev = None
    for g in geo:
        u = None
        if prev is not None:
            u = _perp(prev, g["t"])
        if u is None:
            u = _perp(g["Eu"] - g["S"], g["t"])
        if u is None:
            u = _perp(g["upax"], g["t"])
        g["u"], g["v"] = u, np.cross(g["t"], u)
        prev = u


def _comp(g, phi):
    """Compensation vectors (body frame, / L1) for an array of swivel angles."""
    phi = np.atleast_1d(phi)
    E = g["C"][None, :] + g["rho"] * (np.cos(phi)[:, None] * g["u"][None, :] + np.sin(phi)[:, None] * g["v"][None, :])
    return ((E - g["Eu"][None, :]) @ g["A"]) / g["L1"]


def _dcomp(g, phi):
    d = g["rho"] * (-math.sin(phi) * g["u"] + math.cos(phi) * g["v"])
    return (d @ g["A"]) / g["L1"]


def _pack(geo, anchors):
    return {"C": np.array([g["C"] for g in geo]), "rho": np.array([g["rho"] for g in geo]), "u": np.array([g["u"] for g in geo]),
            "v": np.array([g["v"] for g in geo]), "Eu": np.array([g["Eu"] for g in geo]), "A": np.array([g["A"] for g in geo]),
            "L1": np.array([g["L1"] for g in geo]), "free": ~np.array(anchors, bool)}


def objective(geo, anchors, phi, lam, pk=None):
    """J and dJ/dphi (vectorised).  Anchored frames have c = 0 and zero gradient."""
    pk = pk or _pack(geo, anchors)
    phi = np.asarray(phi, float)
    cs, sn = np.cos(phi)[:, None], np.sin(phi)[:, None]
    E = pk["C"] + pk["rho"][:, None] * (cs * pk["u"] + sn * pk["v"])
    c = np.einsum("ki,kij->kj", E - pk["Eu"], pk["A"]) / pk["L1"][:, None]
    dE = pk["rho"][:, None] * (-sn * pk["u"] + cs * pk["v"])
    dc = np.einsum("ki,kij->kj", dE, pk["A"]) / pk["L1"][:, None]
    c = c * pk["free"][:, None]
    dc = dc * pk["free"][:, None]
    diff = c[1:] - c[:-1]
    J = float(np.sum(c * c)) + lam * float(np.sum(diff * diff))
    g = 2 * np.sum(c * dc, axis=1)
    g[1:] += 2 * lam * np.sum(diff * dc[1:], axis=1)
    g[:-1] -= 2 * lam * np.sum(diff * dc[:-1], axis=1)
    return J, g


def _dp(geo, anchors, lam, grid):
    ang = 2 * math.pi * np.arange(grid) / grid
    cs = [np.zeros((grid, 3)) if anchors[k] else _comp(g, ang) for k, g in enumerate(geo)]
    cost = np.sum(cs[0] ** 2, axis=1)
    back = []
    for k in range(1, len(geo)):
        pair = np.sum((cs[k][:, None, :] - cs[k - 1][None, :, :]) ** 2, axis=2)  # [j_now, i_prev]
        tot = cost[None, :] + lam * pair
        bi = np.argmin(tot, axis=1)
        cost = np.sum(cs[k] ** 2, axis=1) + tot[np.arange(grid), bi]
        back.append(bi)
    j = int(np.argmin(cost))
    idx = [j]
    for bi in reversed(back):
        j = int(bi[j])
        idx.append(j)
    return ang[np.array(idx[::-1])]


def _phi_f(g):
    """W116 framewise pole as a swivel angle (projection of the control elbow)."""
    d = g["Eu"] - g["C"]
    return math.atan2(float(np.dot(d, g["v"])), float(np.dot(d, g["u"])))


def _refine(geo, anchors, x0, lam, maxiter):
    from scipy.optimize import minimize
    free = [k for k in range(len(geo)) if not anchors[k]]
    if not free:
        return np.array(x0, float), 0, True

    pk = _pack(geo, anchors)

    def f(x):
        phi = np.array(x0, float)
        phi[free] = x
        J, g = objective(geo, anchors, phi, lam, pk)
        return J, g[free]

    r = minimize(f, np.array(x0, float)[free], jac=True, method="L-BFGS-B",
                 options={"maxiter": maxiter, "gtol": 1e-14, "ftol": 1e-16, "maxcor": 20})
    phi = np.array(x0, float)
    phi[free] = r.x
    return phi, int(r.nit), bool(r.success)


def temporal_two_bone(rig, frames, ctrl_vals, targets, kinds, auth, upper, lower, lam=LAMBDA, grid=GRID,
                      maxiter=MAXITER, pins_struct=None):
    """frames: ordered consecutive frame ids of the run.  ctrl_vals[k]: control (unpinned edit) values per frame.
    targets[k]: np.array world target (hard pin target for kind 'PIN'; internal guide for kind 'GUIDE').
    Returns (statuses{k}, values{k}, info).  statuses for PIN frames in SOLVED | NO_CHANGE_NEEDED | UNREACHABLE |
    NON_CONVERGED | INSUFFICIENT_DOF | RESOLVER_UNSUPPORTED; GUIDE frames get 'GUIDE'.  Never falls back to F."""
    t0 = time.time()
    info = {"method": "temporal_two_bone", "lambda": lam, "grid": grid, "maxiter": maxiter, "frames": list(frames)}
    pinned = [k for k in frames if kinds[k] == "PIN"]
    vals0 = {k: {c: np.array(v, float) for c, v in ctrl_vals[k].items()} for k in frames}
    if pins_struct is not None:
        mode, why = pins_struct
        if mode != "two_bone":
            st = "INSUFFICIENT_DOF" if mode == "none" else "RESOLVER_UNSUPPORTED"
            info.update(mode=mode, reason=why or "TEMPORAL_RESOLVER_SUPPORTS_TWO_BONE_ONLY")
            return {k: st for k in pinned} | {k: "GUIDE" for k in frames if kinds[k] != "PIN"}, vals0, info
    if not ({upper, lower} <= set(auth)):
        info.update(mode=None, reason="TEMPORAL_RESOLVER_SUPPORTS_TWO_BONE_ONLY")
        return {k: "RESOLVER_UNSUPPORTED" for k in pinned}, vals0, info
    geo = [_frame_geometry(rig, vals0[k], upper, lower, np.asarray(targets[k], float), kinds[k] == "PIN") for k in frames]
    anchors = []
    for k, g in zip(frames, geo):
        anchors.append(kinds[k] == "PIN" and float(np.linalg.norm(g["Wu"] - g["target"])) <= S.TOL_M)
    _bases(geo)
    phiF = np.array([_phi_f(g) for g in geo])
    phiDP = _dp(geo, anchors, lam, grid)
    starts = {}
    for name, x0 in (("dp", phiDP), ("framewise", phiF)):
        phi, nit, ok = _refine(geo, anchors, x0, lam, maxiter)
        J, gr = objective(geo, anchors, phi, lam)
        starts[name] = {"phi": phi, "J": J, "iterations": nit, "scipy_success": ok, "max_abs_grad": float(np.max(np.abs(gr)))}
    best = min(starts, key=lambda n: starts[n]["J"])
    phi = starts[best]["phi"]
    JF, _ = objective(geo, anchors, phiF, lam)
    JF0, _ = objective(geo, anchors, phiF, 0.0)
    JT0, _ = objective(geo, anchors, phi, 0.0)
    converged = starts[best]["max_abs_grad"] <= GTOL
    wall = time.time() - t0
    info.update(mode="two_bone", J_framewise=JF, J_temporal=starts[best]["J"], J_dp_grid=objective(geo, anchors, phiDP, lam)[0],
                fidelity_framewise=JF0, fidelity_temporal=JT0, best_start=best, converged=converged,
                starts={n: {k: v for k, v in s.items() if k != "phi"} for n, s in starts.items()},
                anchors=[k for k, a in zip(frames, anchors) if a], wall_s=wall,
                within_runtime_budget=wall <= RUNTIME_BUDGET_S,
                swivel_deg_temporal=[math.degrees(x) for x in phi], swivel_deg_framewise=[math.degrees(x) for x in phiF],
                rho_m=[g["rho"] for g in geo], reach_slack_m=[g["rmax"] - g["dist"] for g in geo])
    out, status = {}, {}
    for i, (k, g) in enumerate(zip(frames, geo)):
        v = vals0[k]
        if anchors[i]:
            out[k], status[k] = v, "NO_CHANGE_NEEDED"
            continue
        Enew = g["C"] + g["rho"] * (math.cos(phi[i]) * g["u"] + math.sin(phi[i]) * g["v"])
        v = rig.apply_world_swing(v, g["fk"], upper, S.q_between(g["Eu"] - g["S"], Enew - g["S"]))
        fk = rig.fk(v)
        E2, W2 = rig.world(fk, lower, "head"), rig.world(fk, lower, "tail")
        v = rig.apply_world_swing(v, fk, lower, S.q_between(W2 - E2, g["target"] - E2))
        out[k] = v
        if kinds[k] != "PIN":
            status[k] = "GUIDE"
        elif g["unreachable"]:
            status[k] = "UNREACHABLE"
        elif not converged or wall > RUNTIME_BUDGET_S:
            status[k] = "NON_CONVERGED"
        else:
            status[k] = "SOLVED"
    res = []
    for k in frames:
        fk = rig.fk(out[k])
        res.append(float(np.linalg.norm(rig.world(fk, lower, "tail") - np.asarray(targets[k], float))))
    info["internal_residual_m"] = res
    return status, out, info


def guide_targets(rig, ctrl_vals, lower, pin_lo, pin_hi, target, margin, frames_all):
    """T_E internal guides on declared margin frames: unpinned wrist + boundary offset * (1 - smoothstep)."""
    def wrist(k):
        fk = rig.fk({c: np.asarray(v, float) for c, v in ctrl_vals[k].items()})
        return rig.world(fk, lower, "tail")
    out = {}
    for side, b, sgn in (("entry", pin_lo, -1), ("release", pin_hi, +1)):
        D = np.asarray(target, float) - wrist(b)
        for j in range(1, margin + 1):
            k = b + sgn * j
            if k in frames_all:
                out[k] = wrist(k) + D * (1.0 - smoothstep(j / (margin + 1)))
    return out


# ---------------------------------------------------------------------------------------- temporal evaluator ---
def _pt(fr, b, p):
    return np.array(fr["world"][b][p], float)


def _flex_and_normal(fr, upper, lower):
    Sh, E, W = _pt(fr, upper, "head"), _pt(fr, upper, "tail"), _pt(fr, lower, "tail")
    u, v = E - Sh, W - E
    n = np.cross(u, v)
    flex = math.degrees(math.atan2(float(np.linalg.norm(n)), float(np.dot(u, v))))
    nn = n / np.linalg.norm(n) if flex >= FLIP_FLEX_FLOOR_DEG else None
    return flex, nn, Sh, E, W


def temporal_metrics(X, U, window, upper, lower, auth, boundary=None):
    """X, U: {frame: Blender-evaluated frame}.  window: ordered consecutive frames.  Evaluator side only."""
    m = {"window": [window[0], window[-1]], "n_frames": len(window)}
    flex, nrm, E, W = {}, {}, {}, {}
    flexU, nrmU, EU, WU = {}, {}, {}, {}
    for k in window:
        flex[k], nrm[k], _, E[k], W[k] = _flex_and_normal(X[k], upper, lower)
        flexU[k], nrmU[k], _, EU[k], WU[k] = _flex_and_normal(U[k], upper, lower)
    tr = list(zip(window[:-1], window[1:]))

    def flips(n):
        return {b for a, b in tr if n[a] is not None and n[b] is not None and float(np.dot(n[a], n[b])) < 0}
    fX, fU = flips(nrm), flips(nrmU)
    bend_step = [math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(nrm[a], nrm[b]))))))
                 for a, b in tr if nrm[a] is not None and nrm[b] is not None]
    cE = {k: E[k] - EU[k] for k in window}
    cW = {k: W[k] - WU[k] for k in window}
    comp_step_E = {b: float(np.linalg.norm(cE[b] - cE[a])) for a, b in tr}
    comp_step_W = {b: float(np.linalg.norm(cW[b] - cW[a])) for a, b in tr}
    wstep = {b: float(np.linalg.norm(W[b] - W[a])) for a, b in tr}
    estep = {b: float(np.linalg.norm(E[b] - E[a])) for a, b in tr}
    wstepU = {b: float(np.linalg.norm(WU[b] - WU[a])) for a, b in tr}
    rot = {}
    excess = 0.0
    for ch in auth:
        sx = [X[k]["stored_wxyz"][ch] for k in window]
        su = [U[k]["stored_wxyz"][ch] for k in window]
        w1, w1u = rot_first_order(sx), rot_first_order(su)
        w2, w2u = rot_second_order(sx), rot_second_order(su)
        comp = [qangle(a, b) for a, b in zip(sx, su)]
        ex = max([a - b for a, b in zip(w1, w1u)] + [0.0])
        excess = max(excess, ex)
        rot[ch] = {"first_order_peak_deg": max(w1 + [0.0]), "first_order_sum_deg": float(sum(w1)),
                   "control_first_order_peak_deg": max(w1u + [0.0]), "second_order_peak_deg": max(w2 + [0.0]),
                   "second_order_sum_deg": float(sum(w2)), "control_second_order_peak_deg": max(w2u + [0.0]),
                   "first_order_excess_over_control_peak_deg": ex,
                   "compensating_vs_control_peak_deg": max(comp), "compensating_vs_control_sum_deg": float(sum(comp))}
    m.update({
        "bend_flips": sorted(fX), "control_bend_flips": sorted(fU), "introduced_bend_flips": sorted(fX - fU),
        "bend_plane_step_peak_deg": max(bend_step + [0.0]),
        "elbow_comp_step_peak_m": max(comp_step_E.values(), default=0.0),
        "elbow_comp_step_sum_m": float(sum(comp_step_E.values())),
        "wrist_comp_step_peak_m": max(comp_step_W.values(), default=0.0),
        "wrist_step_peak_m": max(wstep.values(), default=0.0), "control_wrist_step_peak_m": max(wstepU.values(), default=0.0),
        "elbow_step_peak_m": max(estep.values(), default=0.0),
        "mean_elbow_dev_from_control_m": float(np.mean([np.linalg.norm(cE[k]) for k in window])),
        "peak_elbow_dev_from_control_m": float(max(np.linalg.norm(cE[k]) for k in window)),
        "flex_deg": {str(k): flex[k] for k in window}, "control_flex_deg": {str(k): flexU[k] for k in window},
        "peak_flex_change_deg": float(max(abs(flex[k] - flexU[k]) for k in window)),
        "min_flex_deg": float(min(flex.values())),
        "rotation": rot, "arm_rot_step_excess_peak_deg": excess,
        "per_transition": {str(b): {"wrist_step_m": wstep[b], "control_wrist_step_m": wstepU[b], "elbow_comp_step_m": comp_step_E[b],
                                    "wrist_comp_step_m": comp_step_W[b]} for _, b in tr},
        "wrist_path": {str(k): W[k].tolist() for k in window}, "elbow_path": {str(k): E[k].tolist() for k in window},
    })
    if boundary:
        lo, hi, M = boundary
        entry = [b for a, b in tr if lo - M <= b <= lo]
        rel = [b for a, b in tr if hi + 1 <= b <= hi + M + 1]
        m["entry_transitions"] = entry
        m["release_transitions"] = rel
        m["entry_jump_m"] = comp_step_W.get(lo, 0.0)
        m["release_jump_m"] = comp_step_W.get(hi + 1, 0.0)
        m["boundary_wrist_comp_step_peak_m"] = max([comp_step_W[b] for b in entry + rel] + [0.0])
        m["boundary_raw_wrist_step_peak_m"] = max([wstep[b] for b in entry + rel] + [0.0])
        m["boundary_control_raw_wrist_step_peak_m"] = max([wstepU[b] for b in entry + rel] + [0.0])
    m["defects"] = {"D1": len(fX - fU), "D2": m["elbow_comp_step_peak_m"], "D3": excess,
                    "D4": m.get("boundary_wrist_comp_step_peak_m")}
    up = rot.get(upper, {})
    m["side"] = {"S1_mean_elbow_dev_m": m["mean_elbow_dev_from_control_m"],
                 "S2_peak_upper_comp_deg": up.get("compensating_vs_control_peak_deg", 0.0),
                 "S3_peak_flex_change_deg": m["peak_flex_change_deg"]}
    return m


def compare(F, T):
    """Declared material improvement / worsening of T relative to F on ONE request (both metric dicts)."""
    imp, worse = [], []
    dF, dT = F["defects"], T["defects"]
    if dF["D1"] >= 1 and dT["D1"] < dF["D1"]:
        imp.append("D1")
    if dT["D1"] > dF["D1"]:
        worse.append("D1")
    for d in ("D2", "D3", "D4"):
        if dF[d] is None or dT[d] is None:
            continue
        if dT[d] <= 0.5 * dF[d] and dF[d] - dT[d] >= FLOORS[d]:
            imp.append(d)
        if dT[d] >= 2.0 * dF[d] and dT[d] - dF[d] >= FLOORS[d]:
            worse.append(d)
    for s, lim in WORSE_SIDE.items():
        if T["side"][s] > F["side"][s] + lim:
            worse.append(s)
    return {"improved": imp, "worse": worse, "material_improvement": bool(imp) and not worse}


def evaluate_temporal_request(cand, ctrl, parent, pin_frames, margin_frames, all_frames, pins_at, auth, edit_ch, domain_pts,
                              arm_domain_pts, rest, scale, chain):
    """Independent evaluation of a multi-frame request from Blender output only.
    - pin frames: W116 per-frame evaluator (hard pin, scope, locality, lengths, root, limit)
    - margin frames (T_E): same evaluator with no pin (scope/locality/lengths/root)
    - every other frame: stored values of every channel and every evaluated point bit-equal to the control
    - every frame: evaluated points outside the authorised arm's influence domain bit-equal to the control."""
    allowed = {"vs_parent": sorted(set(auth) | set(edit_ch)), "vs_control": sorted(auth)}
    per, extra = {}, set()
    for k in all_frames:
        c, u, p = cand[k], ctrl[k], parent[k]
        if k in pin_frames or k in margin_frames:
            ev = S.evaluate_candidate(c, u, p, pins_at(k) if k in pin_frames else [], allowed, domain_pts, rest, scale, chain=chain)
            per[k] = {"role": "PIN" if k in pin_frames else "MARGIN", "evaluation": ev}
        elif c["stored_wxyz"] != u["stored_wxyz"] or c["world"] != u["world"] or c["matrix_world"] != u["matrix_world"]:
            extra.add("CHANGE_OUTSIDE_DECLARED_TEMPORAL_DOMAIN")
            per[k] = {"role": "OUTSIDE", "changed": True}
        outside_arm = [f"{b}.{pt}" for b in c["world"] for pt in ("head", "tail")
                       if f"{b}.{pt}" not in arm_domain_pts and c["world"][b][pt] != u["world"][b][pt]]
        if outside_arm:
            extra.add("POINT_OUTSIDE_ARM_DOMAIN_CHANGED_VS_CONTROL")
            per.setdefault(k, {})["outside_arm_changed"] = outside_arm
    return per, sorted(extra)


def temporal_decision(validation_errors, statuses, per, extra):
    """Atomic: W116 `decide` over every pin and margin frame evaluation, plus temporal-domain violations."""
    evs = [r["evaluation"] for r in per.values() if r.get("evaluation") is not None]
    d = S.decide(validation_errors, statuses, evs)
    if d["decision"] == "ACCEPT" and extra:
        return {"decision": "REJECT", "failure_class": "CONTRACT_VIOLATION", "reasons": sorted(extra)}
    if d["decision"] == "REJECT" and extra and d["failure_class"] not in ("CONTRACT_VIOLATION",):
        if not validation_errors and set(statuses) <= {"SOLVED", "NO_CHANGE_NEEDED", "GUIDE"}:
            return {"decision": "REJECT", "failure_class": "CONTRACT_VIOLATION", "reasons": sorted(set(d["reasons"]) | set(extra))}
    return d
