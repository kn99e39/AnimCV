#!/usr/bin/env python3
"""W118 scratch automatic transition editor + actual edit-footprint instrument (NON-PRODUCTION, evidence only).

Reuses the frozen W116 contract/FK/resolver/evaluator/store and the frozen W117 temporal metrics UNCHANGED.  Adds four
separated responsibilities:

  A. Edit domains (`pin_runs`, `budget_plan`).  The requested HARD-PIN domain (the user's WORLD_POSITION_PIN frames) is
     kept separate from the predeclared automatic SEARCH BUDGET (at most M_MAX frames on each side of each pin run,
     clipped at the clip boundaries and split evenly in gaps between runs so two runs never share a transition frame).

  B. Automatic transition editor (`transition_edit`) - SOLVER side.  Hard-pin base = the frozen W116 framewise resolver
     values on the pin frames (no W117 optimiser, so no convergence criterion is involved).  Outside the pin domain each
     authorised channel receives a parent-frame rotation OFFSET LAYER that carries the boundary compensation
     d = q_F(b) * q_U(b)^-1 and fades it out:  q_X(k) = slerp(I, d, w_j) * q_U(k),  w_j = 1 - smoothstep(j / (m + 1)),
     j = distance to the boundary frame b.  The margin m per side is chosen AUTOMATICALLY by a predeclared rule: the
     smallest m in [0, available] whose predicted peak per-frame wrist compensation step (Python FK, solver side) is
     <= SIGMA_STEP_M; if none, m = available (reported BUDGET_SATURATED).  No fallback exists.  Never changes the pin frames
     or any non-authorised channel.

  C. Actual footprint (`derive_footprint`) - EVALUATOR side.  From Blender-evaluated output only (stored F-curve values,
     evaluated world points, matrix_world) of the candidate vs the unpinned user-edit control U.  Numerical F-curve
     differences (any bit) are separated from material evaluated pose differences (> MATERIAL_M).

  D. Display (`display_spec`) and its independent truth check (`check_display`).  The display spec is what the timeline
     renderer draws.  `check_display` re-derives the per-frame/per-channel changes from the raw Blender frames with its
     own loops and rejects any missing, extra, collapsed or mislabelled frame/channel/run/status.
"""
from __future__ import annotations

import math
import time

import numpy as np

import w116_semantic_pin_base as S
import w117_temporal_pin_base as TB

# ----------------------------------------------------------------------------------- declared W118 parameters ---
M_MAX = 8                 # automatic search budget: max transition frames per side of a pin run (EXPERIMENTAL parameter)
SIGMA_STEP_M = 0.010      # selection rule target: predicted peak per-frame wrist compensation step (m / Action frame)
MATERIAL_M = S.NUMERIC_FLOOR_M   # evaluated point displacement above this = material pose change (1e-5 m)
DISPLAY_RES_MM = 1.0      # display magnitude resolution (wrist/elbow displacement); smaller non-zero = "sub-resolution"
DISPLAY_RES_DEG = 0.1     # display resolution for channel rotation change

CELL_CLASSES = ("PIN_CHANGED", "PIN_UNCHANGED", "AUTO_CHANGED", "UNCHANGED")
STATUS_LABELS = ("PROPOSED", "ACCEPTED", "REJECTED")

# declared quality rule (W118 vs F on the same request; experiment-internal, not user policy)
Q_FLOORS = {"B1": 0.005, "D2": 0.005, "D3": 5.0, "A5": 0.005}


# ------------------------------------------------------------------------------------------------ A. domains ---
def pin_runs(pin_frames):
    fr = sorted(set(int(k) for k in pin_frames))
    runs, cur = [], []
    for k in fr:
        if cur and k == cur[-1] + 1:
            cur.append(k)
        else:
            if cur:
                runs.append(cur)
            cur = [k]
    if cur:
        runs.append(cur)
    return [[r[0], r[-1]] for r in runs]


def budget_plan(pin_frames, all_frames, m_max=M_MAX):
    """Per pin run: available transition frames on each side (clip-clipped, gap-split).  Pure function of the request."""
    first, last = min(all_frames), max(all_frames)
    runs = pin_runs(pin_frames)
    plan = []
    for i, (lo, hi) in enumerate(runs):
        if i == 0:
            e_av, e_lim = min(m_max, lo - first), ("CLIP_START" if lo - first < m_max else None)
        else:
            g = lo - runs[i - 1][1] - 1
            e_av, e_lim = min(m_max, g // 2), ("GAP_SPLIT" if g // 2 < m_max else None)
        if i == len(runs) - 1:
            r_av, r_lim = min(m_max, last - hi), ("CLIP_END" if last - hi < m_max else None)
        else:
            g = runs[i + 1][0] - hi - 1
            r_av, r_lim = min(m_max, g // 2), ("GAP_SPLIT" if g // 2 < m_max else None)
        plan.append({"run": [lo, hi], "entry_available": e_av, "entry_limit": e_lim, "release_available": r_av,
                     "release_limit": r_lim})
    budget = set(int(k) for k in pin_frames)
    for p in plan:
        lo, hi = p["run"]
        budget |= set(range(lo - p["entry_available"], lo)) | set(range(hi + 1, hi + 1 + p["release_available"]))
    return plan, sorted(budget)


def runs_of(frames):
    return pin_runs(frames)


# ---------------------------------------------------------------------------------------- B. transition editor ---
def _canon_q(q):
    q = np.asarray(q, float)
    return -q if q[3] < 0 else q


def q_pow(d, w):
    """slerp(identity, d, w) for unit quaternion d (xyzw), shortest path."""
    d = _canon_q(d / np.linalg.norm(d))
    ang = 2.0 * math.acos(min(1.0, max(-1.0, float(d[3]))))
    if ang < 1e-15:
        return np.array([0.0, 0.0, 0.0, 1.0])
    ax = d[:3] / math.sin(ang / 2)
    return S.q_axis_angle(ax, w * ang)


def _wrist(rig, vals, wrist_bone):
    return rig.world(rig.fk(vals), wrist_bone, "tail")


def transition_edit(rig, ctrl, base, pin_frames, all_frames, auth, wrist_bone, m_max=M_MAX, sigma=SIGMA_STEP_M):
    """ctrl: {k: U values}; base: {k: hard-pin base (F) values} for every pin frame.  Returns (values{k}, plan_info).
    values cover pin frames (= base, untouched) and chosen transition frames; every other frame is U (not returned)."""
    t0 = time.time()
    plan, budget = budget_plan(pin_frames, all_frames, m_max)
    out = {k: {c: np.array(v, float) for c, v in base[k].items()} for k in pin_frames}
    sides = []
    for p in plan:
        lo, hi = p["run"]
        for side, b, sgn, avail, lim in (("entry", lo, -1, p["entry_available"], p["entry_limit"]),
                                         ("release", hi, +1, p["release_available"], p["release_limit"])):
            rec = {"run": [lo, hi], "side": side, "boundary": b, "available": avail, "limit": lim}
            d = {}
            for ch in auth:
                if ch in base[b] and not np.array_equal(np.asarray(base[b][ch], float), np.asarray(ctrl[b][ch], float)):
                    d[ch] = S.q_mul(np.asarray(base[b][ch], float), S.q_conj(np.asarray(ctrl[b][ch], float)))
            rec["offset_deg"] = {ch: math.degrees(2 * math.acos(min(1.0, abs(float(q[3]))))) for ch, q in d.items()}
            if not d or avail == 0:
                rec.update(chosen_margin=0, reason="NO_BOUNDARY_OFFSET" if not d else "NO_FRAMES_AVAILABLE",
                           predicted_peak_step_m={}, saturated=False, frames=[])
                sides.append(rec)
                continue
            cb = _wrist(rig, base[b], wrist_bone) - _wrist(rig, ctrl[b], wrist_bone)
            preds, cand_vals, chosen = {}, {}, None
            for m in range(0, avail + 1):
                vals = {}
                seq = [cb]
                for j in range(1, m + 1):
                    k = b + sgn * j
                    w = 1.0 - TB.smoothstep(j / (m + 1))
                    v = {c: np.array(x, float) for c, x in ctrl[k].items()}
                    for ch, q in d.items():
                        v[ch] = S.q_mul(q_pow(q, w), np.asarray(ctrl[k][ch], float))
                    vals[k] = v
                    seq.append(_wrist(rig, v, wrist_bone) - _wrist(rig, ctrl[k], wrist_bone))
                beyond = b + sgn * (m + 1)
                if beyond in all_frames and beyond not in pin_frames:
                    seq.append(np.zeros(3))
                steps = [float(np.linalg.norm(seq[i] - seq[i - 1])) for i in range(1, len(seq))]
                preds[m] = max(steps) if steps else 0.0
                cand_vals[m] = vals
                if preds[m] <= sigma:
                    chosen = m
                    break
            saturated = chosen is None
            if saturated:
                chosen = avail
            rec.update(chosen_margin=chosen, reason="BUDGET_SATURATED" if saturated else "STEP_TARGET_MET",
                       predicted_peak_step_m={str(m): v for m, v in preds.items()}, saturated=saturated,
                       frames=sorted(cand_vals[chosen]))
            out.update(cand_vals[chosen])
            sides.append(rec)
    planned = sorted(set(pin_frames) | {k for s in sides for k in s["frames"]})
    info = {"method": "w118_offset_layer_transition", "m_max": m_max, "sigma_step_m": sigma, "plan": plan,
            "budget_frames": budget, "planned_frames": planned, "sides": sides, "fallback": None, "wall_s": time.time() - t0}
    return out, info


# ----------------------------------------------------------------------------------------- C. actual footprint ---
def _pt(fr, b, p):
    return np.array(fr["world"][b][p], float)


def derive_footprint(X, U, frames, pin_frames, arm_points, wrist=("lowerarm_l", "tail"), elbow=("upperarm_l", "tail")):
    """Evaluator side: actual changes of candidate X vs control U from Blender frames only."""
    pin = set(pin_frames)
    per, ch_frames, outside_arm = {}, {}, {}
    for k in frames:
        x, u = X[k], U[k]
        chs = sorted(c for c in x["stored_wxyz"] if x["stored_wxyz"][c] != u["stored_wxyz"].get(c))
        ang = {c: TB.qangle(x["stored_wxyz"][c], u["stored_wxyz"][c]) for c in chs}
        pts = {}
        for b in x["world"]:
            for p in ("head", "tail"):
                if x["world"][b][p] != u["world"][b][p]:
                    pts[f"{b}.{p}"] = float(np.linalg.norm(_pt(x, b, p) - _pt(u, b, p)))
        mw = x["matrix_world"] != u["matrix_world"]
        changed = bool(chs or pts or mw)
        oa = sorted(n for n in pts if n not in arm_points)
        if oa:
            outside_arm[k] = oa
        for c in chs:
            ch_frames.setdefault(c, []).append(k)
        wd = float(np.linalg.norm(_pt(x, *wrist) - _pt(u, *wrist)))
        ed = float(np.linalg.norm(_pt(x, *elbow) - _pt(u, *elbow)))
        per[k] = {"changed": changed, "channels": chs, "channel_angle_deg": ang, "max_point_disp_m": max(pts.values(), default=0.0),
                  "wrist_disp_m": wd, "elbow_disp_m": ed, "matrix_world_changed": mw,
                  "material": max(pts.values(), default=0.0) > MATERIAL_M, "pin": k in pin}
    changed = [k for k in frames if per[k]["changed"]]
    out_pin = [k for k in changed if k not in pin]
    return {"frames_changed": changed, "runs": runs_of(changed), "envelope": [changed[0], changed[-1]] if changed else None,
            "channel_frames": {c: sorted(v) for c, v in ch_frames.items()},
            "per_bone_changed_frames": {c: len(v) for c, v in ch_frames.items()},
            "out_of_pin_changed": out_pin, "out_of_pin_runs": runs_of(out_pin), "n_out_of_pin_changed": len(out_pin),
            "pin_frames_unchanged": [k for k in sorted(pin) if k in per and not per[k]["changed"]],
            "numeric_only_frames": [k for k in changed if not per[k]["material"]],
            "material_frames": [k for k in changed if per[k]["material"]],
            "outside_arm_points_changed": {str(k): v for k, v in outside_arm.items()},
            "matrix_world_changed_frames": [k for k in frames if per[k]["matrix_world_changed"]],
            "per_frame": {str(k): v for k, v in per.items()}}


# --------------------------------------------------------------------------------------------- D. display ---
def cell_class(changed, pin):
    if pin:
        return "PIN_CHANGED" if changed else "PIN_UNCHANGED"
    return "AUTO_CHANGED" if changed else "UNCHANGED"


def display_spec(fp, frames, pin_frames, budget_frames, planned_frames, state_id, parent_id, status, request_id, variant,
                 edit_channels=(), frame_axis="Action frame index"):
    """What the timeline renderer draws.  Built from the evaluator footprint record (never from the solver margin)."""
    pin = set(pin_frames)
    cells = {str(k): cell_class(k in set(fp["frames_changed"]), k in pin) for k in frames}
    mag = {}
    for k in frames:
        r = fp["per_frame"][str(k)]
        mag[str(k)] = {"wrist_mm": r["wrist_disp_m"] * 1e3, "elbow_mm": r["elbow_disp_m"] * 1e3,
                       "channel_deg": dict(r["channel_angle_deg"]),
                       "sub_resolution": bool(r["changed"] and r["wrist_disp_m"] * 1e3 < DISPLAY_RES_MM
                                              and r["elbow_disp_m"] * 1e3 < DISPLAY_RES_MM)}
    return {"schema": "w118_display_spec_v1", "request_id": request_id, "variant": variant, "state_id": state_id,
            "parent_state_id": parent_id, "status": status, "frame_axis": frame_axis, "frames": [int(k) for k in frames],
            "pin_frames": sorted(pin), "pin_runs": pin_runs(pin_frames), "budget_frames": list(budget_frames),
            "planned_frames": list(planned_frames), "cells": cells, "channel_rows": {c: list(v) for c, v in fp["channel_frames"].items()},
            "changed_runs": [list(r) for r in fp["runs"]], "envelope": fp["envelope"], "magnitude": mag,
            "display_resolution": {"mm": DISPLAY_RES_MM, "deg": DISPLAY_RES_DEG},
            "user_edit_channels_not_solver": list(edit_channels)}


def check_display(spec, X, U, frames, pin_frames, state_id, decision=None):
    """Independent truth check of a display spec against raw Blender frames (own loops; does not use derive_footprint)."""
    errs = []
    if spec.get("state_id") != state_id:
        errs.append("STATE_ID_MISMATCH")
    if spec.get("status") not in STATUS_LABELS:
        errs.append("STATUS_LABEL_INVALID")
    if decision is not None:
        want = "ACCEPTED" if decision == "ACCEPT" else "REJECTED"
        if spec.get("status") != want:
            errs.append("STATUS_LABEL_DISAGREES_WITH_DECISION")
    if sorted(spec.get("pin_frames", [])) != sorted(set(pin_frames)):
        errs.append("PIN_FRAMES_MISLABELLED")
    changed, chrows = [], {}
    for k in frames:
        x, u = X[k], U[k]
        ch = [c for c in x["stored_wxyz"] if x["stored_wxyz"][c] != u["stored_wxyz"].get(c)]
        moved = any(x["world"][b][p] != u["world"][b][p] for b in x["world"] for p in ("head", "tail"))
        is_changed = bool(ch) or moved or x["matrix_world"] != u["matrix_world"]
        if is_changed:
            changed.append(k)
        for c in ch:
            chrows.setdefault(c, []).append(k)
        want = ("PIN_" if k in set(pin_frames) else "") + ("CHANGED" if is_changed else "UNCHANGED")
        if k not in set(pin_frames):
            want = "AUTO_CHANGED" if is_changed else "UNCHANGED"
        if spec.get("cells", {}).get(str(k)) != want:
            errs.append(f"CELL_MISMATCH@{k}")
        w = float(np.linalg.norm(_pt(x, "lowerarm_l" if "lowerarm_l" in x["world"] else "lower", "tail")
                                 - _pt(u, "lowerarm_l" if "lowerarm_l" in u["world"] else "lower", "tail"))) * 1e3
        got = spec.get("magnitude", {}).get(str(k), {}).get("wrist_mm")
        if got is None or abs(got - w) > 1e-9:
            errs.append(f"MAGNITUDE_MISMATCH@{k}")
    rows = {c: sorted(v) for c, v in chrows.items()}
    if {c: sorted(v) for c, v in spec.get("channel_rows", {}).items()} != rows:
        errs.append("CHANNEL_COVERAGE_MISMATCH")
    runs, cur = [], None
    for k in changed:
        if cur and k == cur[1] + 1:
            cur[1] = k
        else:
            cur = [k, k]
            runs.append(cur)
    if [list(r) for r in spec.get("changed_runs", [])] != runs:
        errs.append("RUNS_MISMATCH")
    env = [changed[0], changed[-1]] if changed else None
    if spec.get("envelope") != env:
        errs.append("ENVELOPE_MISMATCH")
    collapsed = [m for m in errs if m.startswith("CELL_MISMATCH")]
    return {"ok": not errs, "errors": sorted(set(e.split("@")[0] for e in errs)), "detail": errs[:40],
            "n_cell_mismatch": len(collapsed)}


# ------------------------------------------------------------------------------------------- decision ---
def decide_x(validation_errors, pin_statuses, evaluations, fp, budget_frames, planned_frames, auth, display_check=None):
    """Atomic: W116 `decide` over pin + transition frame evaluations, plus footprint/budget/disclosure violations."""
    d = S.decide(validation_errors, pin_statuses, evaluations)
    extra = set()
    fc = set(fp["frames_changed"])
    if not fc <= set(budget_frames):
        extra.add("CHANGE_OUTSIDE_AUTOMATIC_BUDGET")
    if not fc <= set(planned_frames):
        extra.add("FOOTPRINT_EXCEEDS_DECLARED_DOMAIN")
    if not set(fp["channel_frames"]) <= set(auth):
        extra.add("UNAUTHORIZED_CHANNEL_CHANGE")
    if fp["outside_arm_points_changed"]:
        extra.add("POINT_OUTSIDE_ARM_DOMAIN_CHANGED_VS_CONTROL")
    if fp["matrix_world_changed_frames"]:
        extra.add("ROOT_MOVED")
    if display_check is not None and not display_check["ok"]:
        extra.add("DISPLAY_FOOTPRINT_MISMATCH")
    if d["decision"] == "ACCEPT" and extra:
        return {"decision": "REJECT", "failure_class": "CONTRACT_VIOLATION", "reasons": sorted(extra)}
    if d["decision"] == "REJECT" and extra and not validation_errors and set(pin_statuses) <= {"SOLVED", "NO_CHANGE_NEEDED"}:
        return {"decision": "REJECT", "failure_class": "CONTRACT_VIOLATION", "reasons": sorted(set(d["reasons"]) | extra)}
    if d["decision"] == "REJECT" and extra:
        return dict(d, reasons=sorted(set(d["reasons"]) | extra))
    return d


# --------------------------------------------------------------------------------------------- quality ---
def quality(X, U, frames_all, pin_frames, upper, lower, auth, m_window=M_MAX):
    """Blender-evaluated metrics on the union window [first run lo - m - 1, last run hi + m + 1] (clipped)."""
    runs = pin_runs(pin_frames)
    first, last = min(frames_all), max(frames_all)
    lo_w, hi_w = max(first, runs[0][0] - m_window - 1), min(last, runs[-1][1] + m_window + 1)
    w = list(range(lo_w, hi_w + 1))
    m = TB.temporal_metrics(X, U, w, upper, lower, auth)
    pt = m["per_transition"]
    bnd = []
    for lo, hi in runs:
        bnd += [b for b in range(lo - m_window, lo + 1) if str(b) in pt]
        bnd += [b for b in range(hi + 1, hi + m_window + 2) if str(b) in pt]
    bnd = sorted(set(bnd))
    has_out = any(k not in set(pin_frames) for k in w)
    B1 = max([pt[str(b)]["wrist_comp_step_m"] for b in bnd] + [0.0]) if has_out else None
    E = {k: np.array(m["elbow_path"][str(k)]) for k in w}
    EU = {k: _pt(U[k], upper, "tail") for k in w}
    acc = [float(np.linalg.norm((E[k + 1] - EU[k + 1]) - 2 * (E[k] - EU[k]) + (E[k - 1] - EU[k - 1]))) for k in w[1:-1]]
    accX = [float(np.linalg.norm(E[k + 1] - 2 * E[k] + E[k - 1])) for k in w[1:-1]]
    accU = [float(np.linalg.norm(EU[k + 1] - 2 * EU[k] + EU[k - 1])) for k in w[1:-1]]
    straight = [k for k in w if m["flex_deg"][str(k)] < TB.STRAIGHT_FLEX_DEG <= m["control_flex_deg"][str(k)]]
    return {"window": [lo_w, hi_w], "boundary_transitions": bnd, "B1_boundary_wrist_comp_step_peak_m": B1,
            "boundary_raw_wrist_step_peak_m": max([pt[str(b)]["wrist_step_m"] for b in bnd] + [0.0]),
            "boundary_control_raw_wrist_step_peak_m": max([pt[str(b)]["control_wrist_step_m"] for b in bnd] + [0.0]),
            "boundary_raw_wrist_step_excess_over_U_peak_m": max([pt[str(b)]["wrist_step_m"] - pt[str(b)]["control_wrist_step_m"]
                                                                for b in bnd] + [0.0]),
            "D1_introduced_flips": m["defects"]["D1"], "introduced_flip_transitions": m["introduced_bend_flips"],
            "D2_elbow_comp_step_peak_m": m["defects"]["D2"], "D3_arm_rot_step_excess_peak_deg": m["defects"]["D3"],
            "A4_introduced_near_straight_frames": straight,
            "A5_elbow_comp_accel_peak_m": max(acc + [0.0]), "elbow_accel_peak_m": max(accX + [0.0]),
            "control_elbow_accel_peak_m": max(accU + [0.0]),
            "elbow_step_peak_m": m["elbow_step_peak_m"], "wrist_step_peak_m": m["wrist_step_peak_m"],
            "cost_mean_elbow_dev_from_U_m": m["mean_elbow_dev_from_control_m"],
            "cost_peak_elbow_dev_from_U_m": m["peak_elbow_dev_from_control_m"],
            "cost_peak_upper_comp_deg": m["side"]["S2_peak_upper_comp_deg"], "cost_peak_flex_change_deg": m["peak_flex_change_deg"],
            "min_flex_deg": m["min_flex_deg"], "bend_flips": m["bend_flips"], "control_bend_flips": m["control_bend_flips"],
            "rotation": m["rotation"], "per_transition": pt, "wrist_path": m["wrist_path"], "elbow_path": m["elbow_path"],
            "flex_deg": m["flex_deg"], "control_flex_deg": m["control_flex_deg"]}


def compare_x(F, X):
    """Declared W118 rule: meaningful improvement of X over F on one request (both Blender-evaluated quality dicts)."""
    imp, worse = [], []
    bF, bX = F["B1_boundary_wrist_comp_step_peak_m"], X["B1_boundary_wrist_comp_step_peak_m"]
    if bF is not None and bX is not None:
        if bX <= 0.5 * bF and bF - bX >= Q_FLOORS["B1"]:
            imp.append("B1")
        if bX >= 2.0 * bF and bX - bF >= Q_FLOORS["B1"]:
            worse.append("B1")
    if X["D1_introduced_flips"] > F["D1_introduced_flips"]:
        worse.append("A1_introduced_flips")
    for key, fl, name in (("D3_arm_rot_step_excess_peak_deg", "D3", "A2_rotation_step"), ("D2_elbow_comp_step_peak_m", "D2", "A3_elbow_step"),
                          ("A5_elbow_comp_accel_peak_m", "A5", "A5_elbow_accel")):
        if X[key] >= 2.0 * F[key] and X[key] - F[key] >= Q_FLOORS[fl]:
            worse.append(name)
    if len(X["A4_introduced_near_straight_frames"]) > len(F["A4_introduced_near_straight_frames"]):
        worse.append("A4_introduced_near_straight")
    return {"improved": imp, "worse": worse, "meaningful_improvement": bool(imp) and not worse,
            "costs_delta": {k: X[k] - F[k] for k in ("cost_mean_elbow_dev_from_U_m", "cost_peak_elbow_dev_from_U_m",
                                                     "cost_peak_upper_comp_deg", "cost_peak_flex_change_deg")}}
