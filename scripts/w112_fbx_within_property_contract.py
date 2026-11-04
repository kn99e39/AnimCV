#!/usr/bin/env python3
"""W112 evaluation contract: FBX within-property interpolation recoverability boundary (pure Python; evidence only).

Shared by the host stages, the Blender stages and the tests.  Nothing here touches Blender, a rig file or a
production module.  It defines

  * the fixed case matrix: three historical controls reused byte-for-byte from W110 (syn_step uniform held,
    syn_smooth uniform LINEAR, complete_control real replay) and a minimal factorial of scratch interpolation-only
    interventions on ONE item / ONE transform property (``upperarm_l``):
      R   synthetic rig, rotation (source quaternion W/X/Y/Z)        -- uniform / axis-mixed / interval-mixed
      L   synthetic rig, location (separately declared location keys) -- uniform / axis-mixed / interval-mixed
      RR  real complete_control rig, rotation (copied real Action)    -- SYNTHETIC INTERVENTION ON A REAL RIG
    every other channel stays as in its source (all CONSTANT),
  * the PRODUCER rule (source-informed, producer only): per selected-take FBX curve and per FBX key interval, the
    consensus of the interpolation in force on every source F-curve of the driving (item, transform group); a
    disagreement between source axes is a representation OBSTRUCTION (the case stops at the encoding layer),
  * the per-key KeyAttrFlags run-length encoding (keys, values and times untouched),
  * the FBX-level evaluation instrument: a scratch evaluation FBX whose selected-take curves carry extra keys at the
    probe times holding the FBX-level CONSTANT (hold) / LINEAR (lerp) value of the existing keys -- an evaluation
    device for the privileged FBX-representation bound, never a product or metadata candidate,
  * the two privileged SOURCE-KNOWN feasibility bounds (finite assignment spaces, exact per-interval minimum),
  * metrics, frozen tolerances and the CASE rule.

The W110 consumer adapter and the W112 source-free candidate never import this module.
"""
from __future__ import annotations

import itertools
import math

import numpy as np

import w108_fbx_subframe_contract as C8           # W108 reader, error fields (read-only reuse)
import w110_fbx_metadata_recovery_contract as C10  # W110 source-group parser, flags, tolerances (read-only reuse)

SCHEMA = "w112_fbx_within_property_v1"

TARGET_BONE = "upperarm_l"
QPATH = f'pose.bones["{TARGET_BONE}"].rotation_quaternion'
LPATH = f'pose.bones["{TARGET_BONE}"].location'
CONSTANT, LINEAR = "CONSTANT", "LINEAR"
SUPPORTED = (CONSTANT, LINEAR)
FLAG = {CONSTANT: C10.CONSTANT_FLAG, LINEAR: C10.SOURCE_FLAG}   # 0x6002 (W109-verified) / 0x6104 (exporter literal)
KTIME_PER_SECOND = C8.FBX_KTIME_PER_SECOND

# --- families (one target item, one target property each) ------------------------------------------------------------
FAMILIES = {
    "R": {"rig": "synthetic", "source_path": QPATH, "n": 4, "fbx_property": "Lcl Rotation", "group": "rotation",
          "held_axis": 1, "axes": "source quaternion W/X/Y/Z -> FBX Euler X/Y/Z -> imported quaternion W/X/Y/Z"},
    "L": {"rig": "synthetic", "source_path": LPATH, "n": 3, "fbx_property": "Lcl Translation", "group": "location",
          "held_axis": 0, "axes": "source bone-local location X/Y/Z -> FBX parent-frame translation X/Y/Z -> imported "
                                  "bone-local location X/Y/Z"},
    "RR": {"rig": "real", "source_path": QPATH, "n": 4, "fbx_property": "Lcl Rotation", "group": "rotation",
           "held_axis": 1, "axes": "source quaternion W/X/Y/Z -> FBX Euler X/Y/Z -> imported quaternion W/X/Y/Z"},
}
SPECS = {
    "UNI_C": "every target axis CONSTANT on every key interval",
    "UNI_L": "every target axis LINEAR on every key interval",
    "AXIS": "target axis `held_axis` CONSTANT, every other target axis LINEAR (axis-mixed, every interval)",
    "INT": "every target axis: key k CONSTANT if k even else LINEAR (interval-mixed, axis-uniform)",
    "INT1": "target axis `held_axis`: key k CONSTANT if k even else LINEAR; every other target axis LINEAR "
            "(interval-mixed on one axis; axis-mixed on even intervals)",
}
WITHIN_SPECS = ("AXIS", "INT", "INT1")

SYN_INPUT = "w106_synthetic_n0"           # W110 input copy of the W106 synthetic N0 file
SYN_BASE_ACTION = "SYN baseline"
REAL_ACTION = "W91 complete_control FIXED"

CASES = {
    # historical controls, reused byte-for-byte from W110 (source .blend + frozen V1 FBX)
    "syn_step": {"family": "R", "spec": "UNI_C", "origin": "w110_input", "action": "W108 SYN step CONSTANT",
                 "regime": "synthetic (W108 control)", "role": "uniform held control W110 recovers"},
    "syn_smooth": {"family": None, "spec": None, "origin": "w110_input", "action": "W108 SYN smooth LINEAR",
                   "regime": "synthetic (W108 control)", "role": "uniform smooth control blanket CONSTANT harms"},
    "complete_control": {"family": "RR", "spec": "UNI_C", "origin": "w110_input", "action": REAL_ACTION,
                         "regime": "benchmark_detector_observation (3DPW), unmodified real Action",
                         "role": "historical real replay; RR family uniform-held member"},
    # synthetic rotation family (keys = SYN baseline; interpolation-only interventions)
    "syn_R_uniL": {"family": "R", "spec": "UNI_L", "origin": "build", "action": "W112 SYN R uniL"},
    "syn_R_axis": {"family": "R", "spec": "AXIS", "origin": "build", "action": "W112 SYN R axis"},
    "syn_R_int": {"family": "R", "spec": "INT", "origin": "build", "action": "W112 SYN R int"},
    "syn_R_int1": {"family": "R", "spec": "INT1", "origin": "build", "action": "W112 SYN R int1"},
    # synthetic location family (keys = SYN baseline + the separately declared location keys)
    "syn_L_uniC": {"family": "L", "spec": "UNI_C", "origin": "build", "action": "W112 SYN L uniC"},
    "syn_L_uniL": {"family": "L", "spec": "UNI_L", "origin": "build", "action": "W112 SYN L uniL"},
    "syn_L_axis": {"family": "L", "spec": "AXIS", "origin": "build", "action": "W112 SYN L axis"},
    "syn_L_int": {"family": "L", "spec": "INT", "origin": "build", "action": "W112 SYN L int"},
    "syn_L_int1": {"family": "L", "spec": "INT1", "origin": "build", "action": "W112 SYN L int1"},
    # synthetic interventions on a copied real Action (complete_control rig); NOT real-world mixed evidence
    "rr_uniL": {"family": "RR", "spec": "UNI_L", "origin": "build", "action": "W112 RR uniL"},
    "rr_axis": {"family": "RR", "spec": "AXIS", "origin": "build", "action": "W112 RR axis"},
    "rr_int": {"family": "RR", "spec": "INT", "origin": "build", "action": "W112 RR int"},
}
for _c, _v in CASES.items():
    _v.setdefault("regime", "synthetic-on-real-rig (complete_control copy)" if _v["family"] == "RR"
                  else "synthetic (W112 construction)")
WITHIN_CASES = tuple(c for c, v in CASES.items() if v["spec"] in WITHIN_SPECS)
FAMILY_CASES = {f: tuple(c for c, v in CASES.items() if v["family"] == f) for f in FAMILIES}
UNIFORM_LINEAR_OF = {"R": "syn_R_uniL", "L": "syn_L_uniL", "RR": "rr_uniL"}
UNIFORM_HELD_OF = {"R": "syn_step", "L": "syn_L_uniC", "RR": "complete_control"}
NC_BASE_CASE = "syn_R_int"

# location keys of the L family (separately declared scratch key construction; bone-local metres, key k at 1 + 3k)
SYN_KEY_FRAMES = [1.0 + 3*k for k in range(9)]


def loc_value(k):
    return (0.06*math.sin(0.9*k + 0.3), 0.05*math.cos(1.1*k + 0.5), 0.04*math.sin(1.3*k + 1.0))


def target_interp(spec, family, axis, k):
    """Declared interpolation of target axis ``axis`` at its key index ``k``."""
    held = FAMILIES[family]["held_axis"]
    alt = CONSTANT if k % 2 == 0 else LINEAR
    if spec == "UNI_C":
        return CONSTANT
    if spec == "UNI_L":
        return LINEAR
    if spec == "AXIS":
        return CONSTANT if axis == held else LINEAR
    if spec == "INT":
        return alt
    if spec == "INT1":
        return alt if axis == held else LINEAR
    raise ValueError(spec)


# --- source semantics in force on an interval -------------------------------------------------------------------------
def in_force(frames, interps, f, eps=1e-6):
    """Interpolation governing (f, f + dt): that of the last key at or before f; before the first key the curve is
    extrapolated constant (held)."""
    i = int(np.searchsorted(np.asarray(frames, dtype=float), f + eps, side="right")) - 1
    return CONSTANT if i < 0 else interps[i]


def interval_class(curves, f):
    """'held' / 'smooth' / 'axis_mixed' / other of a group of source curves on (f, f+1)."""
    s = sorted({in_force(c["frames"], c["interpolation"], f) for c in curves})
    if s == [CONSTANT]:
        return "held"
    if s == [LINEAR]:
        return "smooth"
    if set(s) <= set(SUPPORTED):
        return "axis_mixed"
    return "unsupported:" + "/".join(s)


def census_groups(census):
    """{group key: [curve records]} using W110's (item, transform group) parser; non-transform curves -> None key."""
    out = {}
    for c in census["curves"]:
        g = C10.source_group_of(c["data_path"])
        key = None if g is None else (f"object|{g[2]}" if g[0] == "object" else f"bone:{g[1]}|{g[2]}")
        out.setdefault(key, []).append(c)
    return out


def fbx_curve_group_key(rec, census):
    if rec["model_type"] == "Null":
        return f"object|{rec['group']}" if rec["model_name"] == census["armature_object"] else None
    return f"bone:{rec['model_name']}|{rec['group']}" if rec["model_name"] in census["bones"] else None


def key_frames(ticks, frame_start, fps):
    """FBX KeyTime ticks -> source frames (V1 bakes every frame from frame_start).  Returns (frames, valid)."""
    t = np.asarray(ticks, dtype=np.int64)
    tpf = KTIME_PER_SECOND/float(fps)
    rel = (t - t[0])/tpf
    ok = bool(np.all(np.abs(rel - np.round(rel)) < 1e-6)) and bool(np.all(np.diff(t) > 0))
    return [float(frame_start + r) for r in np.round(rel)], ok


def producer_perkey(fmap, census, curve_ticks, frame_start, fps):
    """PRODUCER rule (source-informed).  Per selected-take FBX curve: ("UNDRIVEN", None) | ("UNMATCHED", None) |
    ("PERKEY", [CONSTANT|LINEAR|OBSTRUCTED|UNSUPPORTED per FBX key]) with the source group the curve belongs to."""
    groups = census_groups(census)
    out = {}
    for r in fmap["curves"]:
        if r["status"] != "OK":
            out[r["uid"]] = {"kind": "UNMATCHED", "per_key": None, "group_key": None}
            continue
        gk = fbx_curve_group_key(r, census)
        src = groups.get(gk, []) if gk is not None else None
        if src is None:
            out[r["uid"]] = {"kind": "UNMATCHED", "per_key": None, "group_key": None}
            continue
        if not src:
            out[r["uid"]] = {"kind": "UNDRIVEN", "per_key": None, "group_key": gk}
            continue
        frames, ok = key_frames(curve_ticks[r["uid"]], frame_start, fps)
        if not ok:
            out[r["uid"]] = {"kind": "BAD_KEY_GRID", "per_key": None, "group_key": gk}
            continue
        per = []
        for f in frames:
            s = sorted({in_force(c["frames"], c["interpolation"], f) for c in src})
            if len(s) == 1 and s[0] in SUPPORTED:
                per.append(s[0])
            elif set(s) <= set(SUPPORTED):
                per.append("OBSTRUCTED")
            else:
                per.append("UNSUPPORTED")
        out[r["uid"]] = {"kind": "PERKEY", "per_key": per, "group_key": gk}
    return out


def rle(seq):
    runs = []
    for x in seq:
        if runs and runs[-1][0] == x:
            runs[-1][1] += 1
        else:
            runs.append([x, 1])
    return [(a, b) for a, b in runs]


def expand_flags(flags, refs):
    """Per-key flags from KeyAttrFlags / KeyAttrRefCount runs."""
    out = []
    for f, r in zip(flags, refs):
        out += [int(f)]*int(r)
    return out


# --- FBX-level evaluation instrument (privileged bound; scratch evaluation files only) --------------------------------
PHASES = (0.0, 0.25, 0.5, 0.75, 0.99)   # every key interval: key time, quarter frames and the near-boundary f + 0.99
# Blender merges F-curve keys closer than 0.01 frame on import (construction debugging: f + 0.99 keys collided with the
# f + 1 keys), so the instrument writes TWO evaluation files per assignment: part "a" with keys at phases 0 / .25 / .5 /
# .75 and the final key, part "b" with keys at the .99 phase only; each part is sampled only at its own key times.
EPARTS = {"a": ((0.0, 0.25, 0.5, 0.75), True), "b": ((0.99,), False)}


def dense_ticks(ticks, phases=PHASES, include_end=True):
    t = np.asarray(ticks, dtype=np.int64)
    out = []
    for j in range(len(t) - 1):
        d = int(t[j + 1] - t[j])
        out += [int(t[j]) + int(round(p*d)) for p in phases]
    if include_end:
        out.append(int(t[-1]))
    return np.asarray(out, dtype=np.int64)


def densify(ticks, values, sem, phases=PHASES, include_end=True):
    """FBX-level evaluation of one curve at the probe times: interval j held (CONSTANT) or lerped (LINEAR) between the
    EXISTING keys j and j+1.  ``sem``: per-key semantic (len = keys; the last entry is unused)."""
    v = np.asarray(values, dtype=np.float64)
    out = []
    for j in range(len(v) - 1):
        for p in phases:
            out.append(v[j] if (sem[j] == CONSTANT or p == 0.0) else v[j] + p*(v[j + 1] - v[j]))
    if include_end:
        out.append(v[-1])
    return dense_ticks(ticks, phases, include_end), np.asarray(out, dtype=np.float32)


def part_times(times, part):
    """Probe times carried as keys by evaluation-file part ``part`` (part a also holds the final integer key)."""
    phases, _ = EPARTS[part]
    return [t for t in times if any(abs((t - math.floor(t)) - p) < 1e-9 for p in phases)]


def grid(frame_start, frame_end):
    """Probe grid: f + PHASES for every interval [f, f+1) of [frame_start, frame_end), plus frame_end."""
    return [float(f) + p for f in range(frame_start, frame_end) for p in PHASES] + [float(frame_end)]


COMBOS3 = ["".join(c) for c in itertools.product("CL", repeat=3)]
COMBOS4 = ["".join(c) for c in itertools.product("CL", repeat=4)]
LETTER = {"C": CONSTANT, "L": LINEAR}


def combos(n):
    return COMBOS3 if n == 3 else COMBOS4


# --- tolerances (historical, fixed a priori; metres) ------------------------------------------------------------------
TOL = {"recovered_m": C10.TOL["recovered_m"],     # 1 mm world-space recovered / preserved gate (W108)
       "identity_m": C10.TOL["deterministic_m"],  # 1e-9 m identity / repeat (W109)
       "harm_m": C10.TOL["harm_m"],               # 1 cm discriminable harm (W109)
       "time_match_frames": 1e-4}                 # candidate key-time correspondence (structural, not a motion gate)


# --- metrics ------------------------------------------------------------------------------------------------------------
def per_time_error(src, imp):
    """(T,) max |coordinate| world head/tail error over the common bones (W108 gate form)."""
    return C8.error_fields(src, imp)["coord"].max(axis=1)


def interval_index(times):
    """Per sample: None for integer samples, else the interval start f."""
    return [None if float(t) == math.floor(t) else int(math.floor(t)) for t in times]


def exact_bound(src, by_label, labels):
    """Exact minimum over the product space of per-interval assignments.

    F-curve (and FBX-level) evaluation inside (k_j, k_j+1) depends only on key j's interpolation of the target curves
    (CONSTANT / LINEAR ignore handles), every other curve is fixed, so the error at a probe in interval j depends only on
    that interval's choice: min over the product space = per-interval minimum.  Integer samples are key times and must
    not depend on the assignment (checked: ``integer_spread_m``)."""
    times = src["times"]
    iv = interval_index(times)
    err = {lab: per_time_error(src, by_label[lab]) for lab in labels}
    ints = [i for i, f in enumerate(iv) if f is None]
    int_err = {lab: float(err[lab][ints].max()) if ints else 0.0 for lab in labels}
    spread = 0.0
    for lab in labels:
        for k in ("world_head", "world_tail"):
            spread = max(spread, float(np.abs(by_label[lab][k][ints] - by_label[labels[0]][k][ints]).max()) if ints else 0.0)
    per = {}
    for f in sorted({f for f in iv if f is not None}):
        idx = [i for i, g in enumerate(iv) if g == f]
        vals = {lab: float(err[lab][idx].max()) for lab in labels}
        best = min(labels, key=lambda lab: (vals[lab], labels.index(lab)))
        per[f] = {"best": best, "best_m": vals[best], "errors_m": vals}
    worst = max([v["best_m"] for v in per.values()] + [max(int_err.values())])
    return {"bound_m": float(worst), "integer_m": float(max(int_err.values())), "integer_spread_m": spread,
            "per_interval": per}


def class_maxima(err, times, classes):
    """Max of a (T,) error over integer samples and over each interval class."""
    iv = interval_index(times)
    out = {"integer": None}
    acc = {}
    for i, f in enumerate(iv):
        k = "integer" if f is None else classes[f]
        acc.setdefault(k, []).append(float(err[i]))
    for k, v in acc.items():
        out[k] = max(v)
    out["all"] = float(np.max(err)) if len(err) else None
    return out


def target_local(src, imp, bone, integer_only=False):
    """Max sign-aligned |local quaternion| and |local location| difference of one bone (diagnostic: whether the imported
    bone frame and channels equal the source's)."""
    ns, ni = [str(b) for b in src["bones"]], [str(b) for b in imp["bones"]]
    if bone not in ns or bone not in ni:
        return None
    a, b = ns.index(bone), ni.index(bone)
    idx = [i for i, f in enumerate(interval_index(src["times"])) if f is None] if integer_only else         list(range(len(src["times"])))
    qa, qb = src["local_quat"][idx, a], imp["local_quat"][idx, b]
    dq = np.minimum(np.abs(qa - qb).max(-1), np.abs(qa + qb).max(-1))
    dl = np.abs(src["local_loc"][idx, a] - imp["local_loc"][idx, b]).max(-1)
    return {"quat_max": float(dq.max()), "loc_max": float(dl.max())}


def near_boundary_max(err, times, transition_intervals):
    sel = [i for i, t in enumerate(times) if abs((t - math.floor(t)) - 0.99) < 1e-9 and int(math.floor(t)) in
           transition_intervals]
    return float(np.max(err[sel])) if sel else None


# --- CASE mapping (frozen) ------------------------------------------------------------------------------------------------
D_CHECKS = (("preflight_ok", "preflight / provenance failed"),
            ("declaration_bound", "a bound script or input changed after the declaration"),
            ("sources_ok", "source / construction identity failed"),
            ("export_ok", "an export failed or changed its source"),
            ("encoding_integrity", "metadata encoding changed something other than KeyAttr flag runs"),
            ("instrument_valid", "the FBX-level evaluation instrument or the oracle harness is invalid"),
            ("isolation_ok", "source-free consumer isolation not verifiable"),
            ("integrity_ok", "an adapter changed something other than directed interpolation"),
            ("controls_ok", "a positive or negative control failed or was masked"),
            ("repeat_ok", "the independent repeat disagrees"),
            ("originals_unchanged", "an original artifact changed"))


def decide_case(pre):
    """``pre``: the D_CHECKS booleans and, per within-property case c (WITHIN_CASES):
      imp_ok[c]      privileged import-level bound (best CONSTANT/LINEAR per imported target F-curve per key interval)
                     within 1 mm at every probe,
      fbx_ok[c]      privileged FBX-level bound (best CONSTANT/LINEAR per FBX target curve per key interval, evaluated
                     by the instrument) within 1 mm,
      encoded[c]     the frozen producer rule encodes the case without obstruction (per-key flags, keys unchanged),
      sf_ok[c]       the conditional source-free candidate was eligible and recovered every probe within 1 mm with
                     0 unresolved / unmapped and clean integrity,
    and per family f: imp_ok_uniL[f], fbx_ok_uniL[f] (the same bounds on the family's uniform-LINEAR control).

    D: any D-check false.
    B: some within case fails a bound layer whose uniform-LINEAR family control passes that layer (the failure is then
       attributable to the within-property mixture, not to the family's uniform smooth representation).
    A: every within case passes both bounds, is encoded and is recovered by the source-free candidate.
    C: otherwise (feasible-but-unrecovered, rule-level obstruction where the FBX-level bound is feasible, or a bound
       failure that the uniform control shares)."""
    for k, why in D_CHECKS:
        if not pre[k]:
            return "D", why, []
    fam = {c: CASES[c]["family"] for c in WITHIN_CASES}
    refuted = []
    for c in WITHIN_CASES:
        if not pre["imp_ok"][c] and pre["imp_ok_uniL"][fam[c]]:
            refuted.append([c, "imported Action expressiveness (no CONSTANT/LINEAR assignment of the imported "
                               "F-curves reproduces the source)"])
        if not pre["fbx_ok"][c] and pre["fbx_ok_uniL"][fam[c]]:
            refuted.append([c, "FBX metadata encoding (no CONSTANT/LINEAR flag assignment of the existing FBX "
                               "curves reproduces the source)"])
    if refuted:
        return "B", "a within-property case is not representable by any declared interpolation-only assignment", refuted
    if all(pre["imp_ok"][c] and pre["fbx_ok"][c] and pre["encoded"][c] and pre["sf_ok"][c] for c in WITHIN_CASES):
        return "A", "both bounds feasible, every within case encoded and recovered source-free within 1 mm", []
    why = []
    for c in WITHIN_CASES:
        r = [k for k in ("imp_ok", "fbx_ok", "encoded", "sf_ok") if not pre[k][c]]
        if r:
            why.append([c, r])
    return "C", "bounds / encoding / source-free recovery incomplete without an attributable refutation", why


sha256_file = C8.sha256_file
load_npz = C8.load_npz
