#!/usr/bin/env python3
"""W122 scratch helpers: prospective held-out WITHIN-SOURCE hold-window selection on the three W121 sources, the
source-dominated / edit-comparable stratification, the REPAIRED W122 validity instrument (V10 repeatability that excludes
ONLY enumerated telemetry paths; V12 display control DG5 that refuses to run unless the variant is genuinely REJECTed),
raw-motion attribution at the 30 fps real-time rate, the blind A/B label mapping, the blind rating schema and the frozen
W122 CASE rule (NON-PRODUCTION, evidence only).

Nothing here solves, evaluates or tunes the W116-W120 mechanism.  Window selection is a pure function of the UNEDITED
parent's Blender-evaluated wrist path, the W121 pin windows and the frozen W116 framewise resolver statuses (F feasibility);
it never sees a factor variant.  The W121 official CASE D is not touched by anything in this file.
"""
from __future__ import annotations

import copy
import fnmatch
import hashlib
import json
import math

import numpy as np

# ---------------------------------------------------------------------------------- held-out window design ---
HOLD_L = 9                        # = W121 round(0.30 s * 30 fps); unchanged user intent (9-frame WORLD_POSITION_PIN)
M_MAX = 8                         # frozen W118 automatic budget per side (unchanged)
HOLD_EDGE = M_MAX + 1             # W121: frames kept free at each clip end so the full budget is available
W121_EXCLUSION_PAD = M_MAX        # a W121 pin window [lo, hi] blocks [lo - 8, hi + 8] (its own full transition budget)
EDITS = ("none", "RBpos", "RBneg")
FEASIBLE_STATUSES = ("SOLVED", "NO_CHANGE_NEEDED")
SOURCE_DOMINATED_STEP_M = 0.050   # stratification only (NOT a source-quality threshold): unedited raw wrist step >= 50 mm
                                  # per frame (1.5 m/s at 30 fps) = 5x the frozen W119 10 mm meaningful-improvement floor
CONTEXT_FRAMES = 10               # real-time render context beyond the automatic budget on each side

WINDOW_RULE = (
    "Per source, candidate hold windows are every 9-frame run [s, s+8] of the W121 segment whose own automatic domain "
    "[s-8, s+16] lies inside the segment with HOLD_EDGE=9 frames free at each end (the W121 edge rule) and is DISJOINT from every "
    "W121 pin window extended by its +-8-frame budget.  D(window) = max distance of the UNEDITED parent's Blender-evaluated left "
    "wrist from its capture-frame (first-frame) position over the window (W121 definition).  A candidate is FEASIBLE iff the frozen "
    "W116 analytic framewise resolver returns SOLVED/NO_CHANGE_NEEDED on all 9 pin frames for ALL THREE edits none/RBpos/RBneg "
    "(a request-level F feasibility computed from U only; no transition or factor variant exists).  Candidates whose automatic "
    "domain contains a frozen 2D subject-switch proxy frame are excluded (TRACK_PROXY_FLAG).  Selection (deterministic, at most "
    "two per source): LOWER = the feasible candidate with minimum D (earliest capture on ties); HIGHER = the feasible candidate "
    "with maximum D among feasible candidates whose D is STRICTLY above the median D of ALL candidates of that source and whose "
    "automatic domain is separated from LOWER's by more than 2*HOLD_EDGE frames between pin windows (W121 non-overlap rule; "
    "earliest capture on ties).  HIGHER is NOT forced: if no such candidate exists it is reported as NOT_AVAILABLE with counts.")


def enumerate_windows(frames, wrist_world, w121_windows, feas, flag_frames=()):
    """frames: segment source-frame indices; wrist_world: {frame: xyz} of the UNEDITED parent (Blender); w121_windows: [[lo, hi]];
    feas: {capture: {edit: [status per pin frame]}} (may omit candidates that were not resolved because already excluded);
    flag_frames: frozen 2D subject-switch proxy frames.  Returns the full candidate list with reasons (never filtered)."""
    fr = list(frames)
    blocked = set()
    for lo, hi in w121_windows:
        blocked |= set(range(lo - W121_EXCLUSION_PAD, hi + W121_EXCLUSION_PAD + 1))
    flags = set(flag_frames)
    out = []
    for i in range(HOLD_EDGE, len(fr) - HOLD_EDGE - HOLD_L + 1):
        w = fr[i:i + HOLD_L]
        dom = list(range(w[0] - M_MAX, w[-1] + M_MAX + 1))
        p0 = np.array(wrist_world[w[0]], float)
        D = max(float(np.linalg.norm(np.array(wrist_world[k], float) - p0)) for k in w)
        why = []
        if any(k not in set(fr) for k in dom):
            why.append("DOMAIN_OUTSIDE_SEGMENT")
        if set(dom) & blocked:
            why.append("OVERLAPS_W121_WINDOW_OR_BUDGET")
        if set(dom) & flags:
            why.append("TRACK_PROXY_FLAG")
        st = feas.get(w[0]) if feas else None
        if not why:
            if st is None:
                why.append("F_NOT_RESOLVED")
            else:
                bad = {e: sorted(set(s) - set(FEASIBLE_STATUSES)) for e, s in st.items() if set(s) - set(FEASIBLE_STATUSES)}
                if bad or set(st) != set(EDITS):
                    why.append("F_INFEASIBLE:" + ",".join(f"{e}={'/'.join(v)}" for e, v in sorted(bad.items())))
        out.append({"capture": w[0], "frames": w, "domain": [dom[0], dom[-1]], "D_m": D, "reasons": why, "feasible": not why,
                    "F_statuses": st})
    return out


def select_windows(cands):
    """Deterministic declared rank rule (WINDOW_RULE).  Returns {"LOWER": cand|None, "HIGHER": cand|None, "median_D_m", ...}."""
    if not cands:
        return {"LOWER": None, "HIGHER": None, "median_D_m": None, "higher_reason": "NO_CANDIDATES"}
    med = float(np.median([c["D_m"] for c in cands]))
    feas = [c for c in cands if c["feasible"]]
    lower = min(feas, key=lambda c: (c["D_m"], c["capture"])) if feas else None

    def separated(c, o):
        return c["frames"][0] > o["frames"][-1] + 2 * HOLD_EDGE or c["frames"][-1] < o["frames"][0] - 2 * HOLD_EDGE
    hi_pool = [c for c in feas if c["D_m"] > med and (lower is None or separated(c, lower))]
    higher = max(hi_pool, key=lambda c: (c["D_m"], -c["capture"])) if hi_pool else None
    reason = None
    if higher is None:
        n_above = sum(1 for c in feas if c["D_m"] > med)
        reason = ("NO_FEASIBLE_CANDIDATE_ABOVE_MEDIAN_D" if n_above == 0 else "ABOVE_MEDIAN_FEASIBLE_CANDIDATES_OVERLAP_LOWER")
    return {"LOWER": lower, "HIGHER": higher, "median_D_m": med, "higher_reason": reason,
            "n_candidates": len(cands), "n_feasible": len(feas),
            "n_feasible_above_median": sum(1 for c in feas if c["D_m"] > med)}


def stratum(u_steps_m):
    """Pre-outcome diagnostic partition of a request's automatic domain from the UNEDITED (U) raw wrist steps (metres per frame)."""
    m = max(u_steps_m) if u_steps_m else 0.0
    return {"U_max_raw_wrist_step_m": m, "stratum": "SOURCE_DOMINATED" if m >= SOURCE_DOMINATED_STEP_M else "EDIT_COMPARABLE"}


# ------------------------------------------------------------------------------- V10 repaired repeatability ---
# The ONLY fields excluded from the W122 build-record repeatability comparison.  Each entry is an fnmatch path pattern over the
# dotted key path inside one source's build "records".  `wall_s` is the per-variant solver wall-clock time that
# w120_factor_attribution_base.factor_transition stores (the W121 V10 defect); W121's Src.control_record already strips the X_r
# wall_s before it reaches the record.  Nothing else is excluded.
NONDETERMINISTIC_RECORD_PATHS = ("*.factors.*.info.wall_s",)
# leaf keys that may never be excluded (statuses, frames, residual-bearing info, ids, margins, budgets, branch flags, quality)
PROTECTED_KEYS = {"request_status", "planned_frames", "budget_frames", "margins", "status_counts", "flag_counts", "frames", "sides",
                  "state", "state_id", "U_state_id", "parent_id", "F_statuses", "pin_frames", "auth", "decision", "failure_class",
                  "reasons", "solver_params_digest", "constraint_digests", "pin_values_digest", "fallback", "m_max", "plan", "fair",
                  "info", "factors", "validation_errors", "g0_reference_missing_frames", "x_info", "isolation_values_F_r", "status",
                  "flags", "guide", "elbow", "wrist", "weight", "residual_m", "quality"}


class ExclusionError(ValueError):
    pass


def validate_exclusions(patterns):
    """Refuse any exclusion that is not one of the declared telemetry paths, or that would hide a protected semantic field."""
    pats = list(patterns)
    for p in pats:
        if p not in NONDETERMINISTIC_RECORD_PATHS:
            raise ExclusionError(f"UNDECLARED_EXCLUSION:{p}")
        leaf = p.split(".")[-1]
        if leaf in PROTECTED_KEYS or "*" == leaf:
            raise ExclusionError(f"PROTECTED_FIELD_EXCLUDED:{p}")
    return pats


def _leaves(x, path=""):
    if isinstance(x, dict):
        for k in sorted(x):
            yield from _leaves(x[k], f"{path}.{k}" if path else str(k))
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from _leaves(v, f"{path}[{i}]")
    else:
        yield path, x


def _strip(x, patterns, path, removed):
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            p = f"{path}.{k}" if path else str(k)
            if any(fnmatch.fnmatchcase(p, pat) for pat in patterns):
                removed.append(p)
                continue
            out[k] = _strip(v, patterns, p, removed)
        return out
    if isinstance(x, list):
        return [_strip(v, patterns, path, removed) for v in x]
    return x


def strip_declared(records, patterns=NONDETERMINISTIC_RECORD_PATHS):
    validate_exclusions(patterns)
    removed = []
    return _strip(records, list(patterns), "", removed), removed


def canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def records_repeatable(rec1, rec2, patterns=NONDETERMINISTIC_RECORD_PATHS):
    """Compare two builds' records after removing ONLY the declared telemetry paths.  Non-vacuity: the compared leaf count must
    be positive and every removed path must match a declared pattern; differing leaves are listed."""
    s1, rm1 = strip_declared(rec1, patterns)
    s2, rm2 = strip_declared(rec2, patterns)
    l1, l2 = dict(_leaves(s1)), dict(_leaves(s2))
    diff = sorted(k for k in set(l1) | set(l2) if l1.get(k, "<missing>") != l2.get(k, "<missing>"))
    return {"equal": canon(s1) == canon(s2) and not diff, "n_compared_leaves": len(l1), "n_removed": len(rm1),
            "removed_paths_equal": sorted(rm1) == sorted(rm2), "differing_leaves": diff[:50], "n_differing": len(diff),
            "non_vacuous": len(l1) > 0 and len(l1) == len(l2)}


def semantic_corruption(records):
    """V10 live negative control: change ONE semantic solver field in a deep copy (the first factor info request_status found,
    else the first planned_frames list).  Returns (corrupted, path)."""
    r = copy.deepcopy(records)
    for rid in sorted(r):
        for f in sorted((r[rid].get("factors") or {})):
            info = r[rid]["factors"][f].get("info") or {}
            if "request_status" in info:
                info["request_status"] = "W122_INJECTED_SEMANTIC_CHANGE"
                return r, f"{rid}.factors.{f}.info.request_status"
    for rid in sorted(r):
        if r[rid].get("x_info", {}).get("planned_frames"):
            r[rid]["x_info"]["planned_frames"] = r[rid]["x_info"]["planned_frames"][:-1]
            return r, f"{rid}.x_info.planned_frames"
    raise ValueError("no semantic field to corrupt")


# ------------------------------------------------------------------------------- V12 repaired DG5 control ---
class PreconditionError(ValueError):
    pass


def dg5_inject(spec_result, true_decision, corrupt):
    """REJECT->ACCEPT display corruption with a VERIFIED precondition: the variant's committed decision must be REJECT and its
    result display must read REJECTED; the corruption must then change the displayed status.  `corrupt` is the frozen
    W119 corrupt_display.  Raises PreconditionError otherwise (no silent vacuous control)."""
    if true_decision != "REJECT":
        raise PreconditionError(f"DG5_PRECONDITION_FAILED: decision={true_decision} (must be REJECT)")
    if spec_result.get("status") != "REJECTED":
        raise PreconditionError(f"DG5_PRECONDITION_FAILED: displayed status={spec_result.get('status')} (must be REJECTED)")
    s = corrupt("DY5", spec_result)
    if s is None or s.get("status") == spec_result.get("status"):
        raise PreconditionError("DG5_CORRUPTION_DID_NOT_CHANGE_DISPLAYED_STATUS")
    return s


# ------------------------------------------------------------------------------------- motion attribution ---
def steps(P, frames):
    """Per-frame Euclidean step of a world point series (metres per frame), keyed by the later frame."""
    return {k: float(np.linalg.norm(np.asarray(P[k], float) - np.asarray(P[k - 1], float))) for k in frames if k - 1 in P and k in P}


def kinematics(P, frames, fps):
    """Speed (m/s) and acceleration magnitude (m/s^2) at the real-time rate from consecutive world positions (finite
    differences; no filtering)."""
    v = {k: (np.asarray(P[k], float) - np.asarray(P[k - 1], float)) * fps for k in frames if k - 1 in P and k in P}
    a = {k: float(np.linalg.norm((v[k] - v[k - 1]) * fps)) for k in frames if k in v and k - 1 in v}
    return {"speed_m_s": {k: float(np.linalg.norm(x)) for k, x in v.items()}, "accel_m_s2": a}


# ------------------------------------------------------------------------------------------ blind mapping ---
BLIND_SEED = "W122-blind-2026-11-07"


def blind_mapping(pair_ids, seed=BLIND_SEED):
    """Fixed, seed-determined assignment of G0F0 / G1F0 to labels A / B, and the review order."""
    m = {}
    for p in sorted(pair_ids):
        h = hashlib.sha256(f"{seed}:{p}:label".encode()).digest()
        m[p] = {"A": "G1F0", "B": "G0F0"} if h[0] % 2 else {"A": "G0F0", "B": "G1F0"}
    order = sorted(pair_ids, key=lambda p: hashlib.sha256(f"{seed}:{p}:order".encode()).hexdigest())
    return m, order


OVERALL = ("A_VISIBLY_BETTER", "B_VISIBLY_BETTER", "TIE", "UNJUDGEABLE")
DIMENSIONS = ("raw_motion_continuity", "elbow_plausibility", "pin_behavior", "temporal_transition", "source_fidelity", "user_intent")
DIM_VALUES = ("A_BETTER", "B_BETTER", "SAME", "CANNOT_JUDGE")


def validate_rating(r):
    errs = []
    if r.get("overall") not in OVERALL:
        errs.append("overall")
    for d in DIMENSIONS:
        if (r.get("dimensions") or {}).get(d) not in DIM_VALUES:
            errs.append(f"dim:{d}")
    if not str(r.get("evidence") or "").strip():
        errs.append("evidence_missing")
    if not r.get("frames_inspected"):
        errs.append("frames_inspected_missing")
    return errs


def unblind(rating, mapping):
    """Translate an A/B rating into G1F0-relative terms: overall G1_BETTER / G0_BETTER / TIE / UNJUDGEABLE, and per dimension
    G1_BETTER / G0_BETTER / SAME / CANNOT_JUDGE."""
    g1 = "A" if mapping["A"] == "G1F0" else "B"

    def tr(v, better_a, better_b):
        if v == better_a:
            return "G1_BETTER" if g1 == "A" else "G0_BETTER"
        if v == better_b:
            return "G1_BETTER" if g1 == "B" else "G0_BETTER"
        return v
    return {"overall": tr(rating["overall"], "A_VISIBLY_BETTER", "B_VISIBLY_BETTER"),
            "dimensions": {d: tr(v, "A_BETTER", "B_BETTER") for d, v in rating["dimensions"].items()}}


# ------------------------------------------------------------------------------------------ frozen CASE rule ---
RAW_HARM_REL, RAW_HARM_ABS_M = 0.25, 0.010
CASE_RULE = (
    "DATA_BLOCKED (no CASE) if an authenticated W121 original plus frozen usable rig input cannot be recovered for any source.  "
    "Otherwise D if any W122 validity gate V fails (authentication, frozen perception/rig reproduction, prospective window "
    "selection, baseline/anchor reproduction, isolation, factor fairness, pre-declaration/instrument, independent Blender "
    "evaluation + population completeness, REPAIRED repeatability, timing/30 fps render identity, REPAIRED negative/display "
    "controls, blinding, identity labelling, synthetic-control reproduction).  Otherwise C if any safety gate C fails (hard pin "
    "and scope, truthful footprint, atomicity/lineage/exact revert, source immutability, user edit and collateral, no fallback).  "
    "Otherwise, on the HELD-OUT within-source population only (W121 anchor windows and synthetic controls never count): a held-out "
    "window is UTILITY-POSITIVE iff at least one of its edit variants has (i) a meaningful primary GUIDE improvement guide_at_F0 "
    "under the frozen W119 comparator, (ii) a blind AGENT REVIEW overall rating that unblinds to G1_BETTER, (iii) no dimension "
    "raw_motion_continuity or elbow_plausibility rated G0_BETTER, and (iv) no new raw-motion harm: the G1F0 peak raw wrist step "
    "over the automatic domain does not exceed G0F0's by more than max(25 %, 10 mm); the window counts toward A only if its subject "
    "identity is VERIFIED.  A iff UTILITY-POSITIVE identity-VERIFIED held-out windows number >= 2 AND come from >= 2 distinct "
    "sources.  B if not A and the held-out population shows ANY meaningful GUIDE or FLOOR improvement (primary or conditional "
    "contrast) on at least one request OR any blind rating unblinding to G1_BETTER.  C otherwise (no meaningful utility on the "
    "held-out episodes).  Ties and UNJUDGEABLE ratings never count as visual support; numeric improvement with a tie / "
    "unjudgeable / G0_BETTER rating is numeric-only (at most B); a visual G1_BETTER without numeric improvement is visual-only "
    "(at most B).  Infeasible / rejected requests stay in every denominator.")


def raw_harm(g1_peak, g0_peak):
    if g1_peak is None or g0_peak is None:
        return False
    return g1_peak - g0_peak > max(RAW_HARM_REL * g0_peak, RAW_HARM_ABS_M)


def case_rule(V, Cg, heldout, data_blocked):
    """heldout: {rid: {"source", "window", "identity", "numeric": {"guide_primary", "guide_conditional", "floor_primary",
    "floor_conditional"}, "visual": unblinded rating or None, "raw_harm": bool}}."""
    if data_blocked:
        return {"case": "DATA_BLOCKED"}
    if not all(V.values()):
        return {"case": "D", "failed_V": sorted(k for k, v in V.items() if not v)}
    if not all(Cg.values()):
        return {"case": "C", "failed_C": sorted(k for k, v in Cg.items() if not v)}
    pos, per = set(), {}
    any_num = any_vis = False
    for rid, h in heldout.items():
        n = h["numeric"]
        vis = h.get("visual") or {}
        g1v = vis.get("overall") == "G1_BETTER"
        harm_dim = any((vis.get("dimensions") or {}).get(d) == "G0_BETTER" for d in ("raw_motion_continuity", "elbow_plausibility"))
        up = bool(n["guide_primary"] and g1v and not harm_dim and not h["raw_harm"])
        per[rid] = {"utility_positive": up, "visual_G1_better": g1v, "numeric_any": any(n.values()), "identity": h["identity"]}
        any_num = any_num or any(n.values())
        any_vis = any_vis or g1v
        if up and h["identity"] == "VERIFIED":
            pos.add((h["source"], h["window"]))
    srcs = {s for s, _ in pos}
    if len(pos) >= 2 and len(srcs) >= 2:
        return {"case": "A", "positive_windows": sorted(pos), "per_request": per}
    if any_num or any_vis:
        return {"case": "B", "positive_windows": sorted(pos), "per_request": per,
                "why": {"numeric_any": any_num, "visual_any": any_vis}}
    return {"case": "C", "per_request": per, "reason": "no meaningful held-out utility (numeric or visual)"}


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def finite(x):
    return x is not None and math.isfinite(x)
