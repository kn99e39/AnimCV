"""W104 analyze / video stages: paired outcome accounting under the hashed W104 declaration.

GT is used only for evaluation metrics, observation-quality context and the separately labelled privileged reference.
Nothing here creates a production input, a mask rule, a threshold, a crop or a correction.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from w103_observability import (DIRECT, JOINTS, KNEES, PARENT, SEGMENTS, SMPL_OF, angle_deg, distribution, dump,
                                load_context, load_predictions, raw_references, sha, torso_forward, video_bootstrap)
from w103_analysis import s0_metrics
from w104_missing_joint import (FACTORIAL, KNEE_OF_LIMB, LIMB_SEGMENTS, NATURAL_MIN_ROWS, PRIVILEGED, REGIME, RESTORE,
                                TEMPORAL_MAX_OFFSET, TEMPORAL_REPORT_OFFSETS, VIDEO_PREDICTIONS, factorial_row,
                                load_w103_labels, raw_landmarks, require_declaration, restore_row, restoration_inputs)

LIMBS = ("leg_l", "leg_r", "arm_l", "arm_r")
SEG_NAMES = list(SEGMENTS)
SEG_INDEX = {n: i for i, n in enumerate(SEG_NAMES)}
SPLITS = ("validation", "test")
CROSSCOUNTRY_NOTE = "W103 reference-validity defect: GT root behind camera (outdoors_crosscountry_00:actor0)"


# ------------------------------------------------------------------ per-row outcome summaries
def outcome(J, G):
    """W103 s0_metrics plus per-limb failure flags (segment > 45 deg, or a wrong knee branch for a leg)."""
    m = s0_metrics(J, G)
    lf = np.zeros((len(J), 4), bool)
    for k, limb in enumerate(LIMBS):
        seg = [SEG_INDEX[s] for s in LIMB_SEGMENTS[limb]]
        lf[:, k] = (m["dir"][:, seg] > 45.0).any(1)
        if limb in KNEE_OF_LIMB:
            lf[:, k] |= np.nan_to_num(m["knee_wrong"][:, KNEE_OF_LIMB[limb]]) > 0
    m["limb_fail"] = lf
    return m


def limb_mask(limbs):
    return np.array([l in limbs for l in LIMBS])


def induced(base, cond, lm):
    """Newly severe, or newly a limb failure on the limbs in mask lm, relative to base; and the reverse (recovered)."""
    new = (cond["severe"] & ~base["severe"]) | (cond["limb_fail"][:, lm] & ~base["limb_fail"][:, lm]).any(1)
    rec = (base["severe"] & ~cond["severe"]) | (base["limb_fail"][:, lm] & ~cond["limb_fail"][:, lm]).any(1)
    return new, rec


def knee_change(base, cond, legs):
    """Newly wrong / fixed knee branch among both-eligible knees in the listed legs."""
    new = fixed = elig = 0
    for leg in legs:
        k = KNEE_OF_LIMB[leg]
        a, b = base["knee_wrong"][:, k], cond["knee_wrong"][:, k]
        ok = np.isfinite(a) & np.isfinite(b)
        elig += int(ok.sum()); new += int(((a == 0) & (b == 1) & ok).sum()); fixed += int(((a == 1) & (b == 0) & ok).sum())
    return {"both_eligible": elig, "newly_wrong": new, "fixed": fixed}


def tails(x):
    x = np.asarray(x, float)
    return {k: float(np.percentile(x, q)) for k, q in (("p50", 50), ("p95", 95), ("p99", 99))} | {"max": float(x.max())} if len(x) else {}


def boot(vids, values):
    return video_bootstrap(list(vids), list(np.asarray(values, float)))


def per_row_mean(rows, values):
    """Average counterfactual copies per row (rows: population index per item) -> (unique rows, mean value)."""
    rows = np.asarray(rows); values = np.asarray(values, float)
    u, inv = np.unique(rows, return_inverse=True)
    s = np.bincount(inv, weights=values, minlength=len(u)); n = np.bincount(inv, minlength=len(u))
    return u, s/n


def pooled(rows, values, vid):
    u, m = per_row_mean(rows, values)
    return boot(vid[u], m)


# ------------------------------------------------------------------ factorial accounting
def factorial_analysis(ctx, keep):
    decl, fz, Mm, vid = ctx["decl"], ctx["fz"], ctx["Mm"], ctx["vid"]
    pats = decl["patterns"]
    row, pat = fz["rec_row"], fz["rec_pattern"]
    G = ctx["G"][row]
    met = {"F00": {k: v[row] for k, v in Mm.items()}}
    for c in ("F10", "F01", "F11"):
        met[c] = outcome(fz["pred_"+c].astype(float), G)
    quality = ctx["labels"]["quality_M"][row]
    inv = fz["crop_invariant"]; small = ~inv & (np.abs(fz["side_ratio"]-1) <= .01) & (fz["centre_shift"] < .01)
    stratum = np.where(inv, "invariant", np.where(small, "small", "large"))
    names = [p["name"] for p in pats]
    kind = np.array([p["kind"] for p in pats])[pat]
    lmask = np.stack([limb_mask(pats[p]["affected_limbs"]) for p in pat])
    ind, rec = {}, {}
    # vectorized induced flags (limb masks vary per item)
    for c in ("F10", "F01", "F11"):
        nl = (met[c]["limb_fail"] & ~met["F00"]["limb_fail"] & lmask).any(1)
        rl = (met["F00"]["limb_fail"] & ~met[c]["limb_fail"] & lmask).any(1)
        ind[c] = (met[c]["severe"] & ~met["F00"]["severe"]) | nl
        rec[c] = (met["F00"]["severe"] & ~met[c]["severe"]) | rl
    d = {c: met[c]["mpjpe"]-met["F00"]["mpjpe"] for c in ("F10", "F01", "F11")}
    mask_eff = .5*(d["F10"]+(met["F11"]["mpjpe"]-met["F01"]["mpjpe"]))
    crop_eff = .5*(d["F01"]+(met["F11"]["mpjpe"]-met["F10"]["mpjpe"]))
    inter = met["F11"]["mpjpe"]-met["F10"]["mpjpe"]-met["F01"]["mpjpe"]+met["F00"]["mpjpe"]
    # count-matched mirrored control evaluated on the target limbs
    lookup = {(int(r), int(p)): i for i, (r, p) in enumerate(zip(row, pat))}
    name_to_idx = {n: i for i, n in enumerate(names)}
    ctrl_ind = np.full(len(row), np.nan)
    for i, (r, p) in enumerate(zip(row, pat)):
        cname = decl["controls"].get(names[p])
        if not cname:
            continue
        j = lookup.get((int(r), name_to_idx[cname]))
        if j is None:
            continue
        nl = (met["F10"]["limb_fail"][j] & ~met["F00"]["limb_fail"][i] & lmask[i]).any()
        ctrl_ind[i] = float((met["F10"]["severe"][j] and not met["F00"]["severe"][i]) or nl)
    target = (kind != "control") & keep[row]
    out = {"populations": {}}
    for popname, popmask in (("primary_well", quality == "well"), ("secondary_mislocalized", quality == "mislocalized")):
        sel = target & popmask
        V = vid[row]
        block = {"rows": int(len(np.unique(row[sel]))), "videos": int(len(np.unique(V[sel]))), "row_pattern_pairs": int(sel.sum())}
        block["pooled"] = {
            "dMPJPE_F10": pooled(row[sel], d["F10"][sel], vid), "dMPJPE_F01": pooled(row[sel], d["F01"][sel], vid),
            "dMPJPE_F11": pooled(row[sel], d["F11"][sel], vid), "mask_main": pooled(row[sel], mask_eff[sel], vid),
            "crop_main": pooled(row[sel], crop_eff[sel], vid), "interaction": pooled(row[sel], inter[sel], vid),
            "induced_F10": pooled(row[sel], ind["F10"][sel], vid), "induced_F01": pooled(row[sel], ind["F01"][sel], vid),
            "induced_F11": pooled(row[sel], ind["F11"][sel], vid),
            "induced_F10_minus_F01": pooled(row[sel], ind["F10"][sel].astype(float)-ind["F01"][sel], vid)}
        cs = sel & np.isfinite(ctrl_ind)
        block["pooled"]["induced_F10_target_minus_control"] = pooled(row[cs], ind["F10"][cs].astype(float)-ctrl_ind[cs], vid)
        block["pooled"]["induced_F10_target_controlled_subset"] = pooled(row[cs], ind["F10"][cs], vid)
        block["pooled"]["induced_F10_control_on_target_limbs"] = pooled(row[cs], ctrl_ind[cs], vid)
        block["counts"] = {c: {"newly_failed": int(ind[c][sel].sum()), "recovered": int(rec[c][sel].sum()),
                               "newly_severe": int((met[c]["severe"] & ~met["F00"]["severe"])[sel].sum()),
                               "severe_recovered": int((met["F00"]["severe"] & ~met[c]["severe"])[sel].sum())}
                           for c in ("F10", "F01", "F11")}
        block["counts"]["F00_severe_pairs"] = int(met["F00"]["severe"][sel].sum())
        block["crop_strata_pairs"] = dict(Counter(stratum[sel]))
        block["f01_out_of_range_pairs"] = int(fz["f01_out_of_range"][sel].sum())
        # per pattern (all patterns, controls included)
        per = {}
        for p, name in enumerate(names):
            s = (pat == p) & popmask & keep[row]
            if not s.any():
                continue
            legs = [l for l in pats[p]["affected_limbs"] if l.startswith("leg")]
            e = {"kind": pats[p]["kind"], "pairs": int(s.sum()), "videos": int(len(np.unique(V[s]))),
                 "crop_strata": dict(Counter(stratum[s])), "f01_out_of_range": int(fz["f01_out_of_range"][s].sum()),
                 "median_side_ratio": float(np.median(fz["side_ratio"][s])),
                 "F00_severe": int(met["F00"]["severe"][s].sum())}
            for c in ("F10", "F01", "F11"):
                e[c] = {"dMPJPE": boot(V[s], d[c][s]), "induced_rate": float(ind[c][s].mean()),
                        "induced": int(ind[c][s].sum()), "recovered": int(rec[c][s].sum()),
                        "newly_severe": int((met[c]["severe"] & ~met["F00"]["severe"])[s].sum()),
                        "severe_recovered": int((met["F00"]["severe"] & ~met[c]["severe"])[s].sum()),
                        "knee_affected_legs": knee_change({k: v[s] for k, v in met["F00"].items()},
                                                          {k: v[s] for k, v in met[c].items()}, legs) if legs else None,
                        "depth_sign_new_errors_affected": int(sum(((met[c]["depth_err"][s][:, SEG_INDEX[g]] == 1) &
                                                                   (met["F00"]["depth_err"][s][:, SEG_INDEX[g]] == 0)).sum()
                                                                  for l in pats[p]["affected_limbs"] for g in LIMB_SEGMENTS[l])),
                        "mpjpe_tails": tails(met[c]["mpjpe"][s])}
            e["F00_mpjpe_tails"] = tails(met["F00"]["mpjpe"][s])
            e["mask_main"] = boot(V[s], mask_eff[s]); e["crop_main"] = boot(V[s], crop_eff[s]); e["interaction"] = boot(V[s], inter[s])
            removed = pats[p]["joints"]
            pj = met["F11"]["per_joint"][s]-met["F00"]["per_joint"][s]
            others = [j for j in range(17) if j not in pats[p]["removed_with_derived"] and j != 0]
            e["F11_per_joint_delta_mm"] = {"removed": float(pj[:, removed].mean()), "other_non_root": float(pj[:, others].mean())}
            cs = s & np.isfinite(ctrl_ind)
            if cs.any():
                e["control"] = {"name": decl["controls"].get(name), "F10_induced_on_target_limbs": float(ctrl_ind[cs].mean()),
                                "target_minus_control": boot(V[cs], ind["F10"][cs].astype(float)-ctrl_ind[cs])}
            per[name] = e
        block["per_pattern"] = per
        # strata (targets only, F11 and F10 induced / dMPJPE, per-row pooled)
        strata = {}
        G00 = ctx["G"][row]
        nrem = np.array([len(pats[p]["joints"]) for p in pat])
        flex = np.full(len(row), np.nan)
        for i, p in enumerate(pat):
            legs = [l for l in pats[p]["affected_limbs"] if l.startswith("leg")]
            if legs:
                h, k, a = KNEES["l" if legs[0] == "leg_l" else "r"]
                flex[i] = angle_deg(G00[i, k]-G00[i, h], G00[i, a]-G00[i, k])
        base_bin = np.digitize(met["F00"]["mpjpe"], [50, 100, 200])
        defs = {"split": ctx["split"][row], "posture": ctx["labels"]["posture"][row], "crop": stratum,
                "removed_count": nrem.astype(str),
                "baseline_mpjpe": np.array(["<50", "50-100", "100-200", ">200"])[base_bin],
                "affected_gt_knee_flexion": np.where(np.isnan(flex), "no_leg", np.where(flex < 20, "<20", np.where(flex < 60, "20-60", ">=60"))),
                "anatomy": np.array([("leg" if all(l.startswith("leg") for l in pats[p]["affected_limbs"]) else
                                      "arm" if all(l.startswith("arm") for l in pats[p]["affected_limbs"]) else "mixed") for p in pat])}
        for key, lab in defs.items():
            strata[key] = {}
            for value in sorted(set(lab[sel])):
                s2 = sel & (lab == value)
                strata[key][str(value)] = {"pairs": int(s2.sum()), "rows": int(len(np.unique(row[s2]))),
                                           "dMPJPE_F11": pooled(row[s2], d["F11"][s2], vid),
                                           "dMPJPE_F10": pooled(row[s2], d["F10"][s2], vid),
                                           "dMPJPE_F01": pooled(row[s2], d["F01"][s2], vid),
                                           "induced_F11": pooled(row[s2], ind["F11"][s2], vid),
                                           "induced_F10": pooled(row[s2], ind["F10"][s2], vid)}
        block["strata"] = strata
        # sequence-level failure clusters (F11 induced, targets)
        clus = Counter(); tot = Counter()
        for r_, v_, f_ in zip(row[sel], V[sel], ind["F11"][sel]):
            clus[str(ctx["seq"][r_])] += int(f_); tot[str(ctx["seq"][r_])] += 1
        block["sequence_clusters_F11_induced"] = sorted(([k, clus[k], tot[k], clus[k]/tot[k]] for k in tot), key=lambda x: -x[1])[:12]
        out["populations"][popname] = block
    out["temporal_events"] = temporal_events(ctx, met, row, pat, kind, keep)
    out["explained"] = explained(ctx, met, row, pat, keep)
    out["_items"] = {"d": d, "ind": ind, "met": met, "row": row, "pat": pat, "stratum": stratum}
    return out


def step_events(Pa, Pb, Ga, Gb):
    """Segment step > 45 deg while GT step < 15 deg; torso-forward step > 90 deg while GT step < 45 deg."""
    sa = np.stack([Pa[:, b]-Pa[:, a] for a, b in SEGMENTS.values()], 1)
    sb = np.stack([Pb[:, b]-Pb[:, a] for a, b in SEGMENTS.values()], 1)
    ga = np.stack([Ga[:, b]-Ga[:, a] for a, b in SEGMENTS.values()], 1)
    gb = np.stack([Gb[:, b]-Gb[:, a] for a, b in SEGMENTS.values()], 1)
    gstable = angle_deg(ga, gb) < 15.0
    seg = (angle_deg(sa, sb) > 45.0) & gstable
    tf = angle_deg(torso_forward(Pa), torso_forward(Pb)); tg = angle_deg(torso_forward(Ga), torso_forward(Gb))
    ok = np.isfinite(tf) & np.isfinite(tg) & (tg < 45.0)
    return seg, gstable, (np.nan_to_num(tf) > 90.0) & ok, ok


def temporal_events(ctx, met, row, pat, kind, keep):
    sid_row = ctx["row_of"]; quality = ctx["labels"]["quality_M"]
    lookup = {(int(r), int(p)): i for i, (r, p) in enumerate(zip(row, pat))}
    pairs = []
    for r in np.unique(row):
        nxt = ctx["bank"].samples[int(ctx["positions"][r])].neighbors.get("next")
        if nxt and nxt in sid_row:
            q = sid_row[nxt]
            if quality[q] == "well" and quality[r] == "well" and keep[r] and keep[q]:
                pairs.append((r, q))
    pairs = np.array(pairs)
    out = {"neighbour_pairs_primary": len(pairs)}
    if not len(pairs):
        return out
    G, M = ctx["G"], ctx["pred"]["M"]
    seg0, gs, tor0, tok = step_events(M[pairs[:, 0]], M[pairs[:, 1]], G[pairs[:, 0]], G[pairs[:, 1]])
    out["F00"] = {"segment_events": int(seg0.sum()), "segment_eligible_steps": int(gs.sum()),
                  "torso_events": int(tor0.sum()), "torso_eligible_steps": int(tok.sum())}
    fz = ctx["fz"]
    for c in ("F10", "F01", "F11"):
        P = fz["pred_"+c]
        tot = Counter()
        per_pair = []
        for p in range(len(ctx["decl"]["patterns"])):
            if ctx["decl"]["patterns"][p]["kind"] == "control":
                continue
            ia = [lookup.get((int(a), p)) for a, _ in pairs]; ib = [lookup.get((int(b), p)) for _, b in pairs]
            ok = np.array([x is not None and y is not None for x, y in zip(ia, ib)])
            if not ok.any():
                continue
            ia = np.array([x for x, o in zip(ia, ok) if o]); ib = np.array([y for y, o in zip(ib, ok) if o])
            s, g, t, tk = step_events(P[ia].astype(float), P[ib].astype(float), G[pairs[ok, 0]], G[pairs[ok, 1]])
            tot["segment_events"] += int(s.sum()); tot["segment_eligible_steps"] += int(g.sum())
            tot["torso_events"] += int(t.sum()); tot["torso_eligible_steps"] += int(tk.sum()); tot["pair_patterns"] += int(ok.sum())
            per_pair.append((pairs[ok, 0], s.sum(1)-seg0[ok].sum(1)))
        rows = np.concatenate([a for a, _ in per_pair]); vals = np.concatenate([b for _, b in per_pair])
        out[c] = dict(tot) | {"segment_events_minus_F00_per_pair": pooled(rows, vals, ctx["vid"])}
    return out


def explained(ctx, met, row, pat, keep):
    """Natural failure rate on exact-pattern natural rows vs F11 absolute failure on the primary substrate."""
    decl = ctx["decl"]; same = ctx["same"]; vm = ctx["obs"]["input_valid_M"]; Mm = ctx["Mm"]
    quality = ctx["labels"]["quality_M"][row]
    out = {"classes": {}, "min_rows": NATURAL_MIN_ROWS}
    num = den = 0.
    for p, pdef in enumerate(decl["patterns"]):
        if pdef["kind"] == "control":
            continue
        joints = set(pdef["joints"])
        nat = np.array([same[i] and keep[i] and set(j for j in DIRECT if not vm[i, j]) == joints for i in range(len(vm))])
        lm = limb_mask(pdef["affected_limbs"])
        nat_fail = Mm["severe"][nat] | (Mm["limb_fail"][nat][:, lm]).any(1)
        s = (pat == p) & (quality == "well") & keep[row]
        f11 = met["F11"]["severe"][s] | (met["F11"]["limb_fail"][s][:, lm]).any(1)
        f00 = met["F00"]["severe"][s] | (met["F00"]["limb_fail"][s][:, lm]).any(1)
        e = {"natural_rows": int(nat.sum()), "natural_videos": int(len(set(ctx["vid"][nat]))),
             "natural_failure_rate": float(nat_fail.mean()) if nat.any() else None,
             "natural_mpjpe_mean": float(Mm["mpjpe"][nat].mean()) if nat.any() else None,
             "F11_failure_rate": float(f11.mean()) if s.any() else None, "F00_failure_rate": float(f00.mean()) if s.any() else None,
             "F11_mpjpe_mean": float(met["F11"]["mpjpe"][s].mean()) if s.any() else None}
        if nat.sum() >= NATURAL_MIN_ROWS and s.any():
            num += nat.sum()*f11.mean(); den += nat.sum()*nat_fail.mean(); e["eligible"] = True
        out["classes"][pdef["name"]] = e
    out["weighted_F11_failure"] = num; out["weighted_natural_failure"] = den
    out["ratio"] = float(num/den) if den > 0 else None
    out["EXPLAINED"] = bool(den > 0 and num/den >= .5)
    return out


# ------------------------------------------------------------------ restoration accounting
def restoration_analysis(ctx, keep):
    rz, pz, Mm, vid = ctx["rz"], ctx["pz"], ctx["Mm"], ctx["vid"]
    rows, new = rz["rows"], rz["new_tokens"]
    G = ctx["G"][rows]
    met = {"M": {k: v[rows] for k, v in Mm.items()}}
    for c in RESTORE:
        met[c] = outcome(rz["pred_"+c].astype(float), G)
    partial = new[:, DIRECT].any(1)
    allm = np.ones(len(rows), bool)
    V = vid[rows]
    out = {"rows": int(len(rows)), "partial_rows_with_restorable_limb_token": int(partial.sum()),
           "head_or_derived_only_rows": int((~partial).sum())}
    def block(sel, cond, base="M"):
        a, b = met[base], met[cond]
        dm = b["mpjpe"]-a["mpjpe"]
        sev_new = b["severe"] & ~a["severe"]; sev_rec = a["severe"] & ~b["severe"]
        kn = knee_change({k: v[sel] for k, v in a.items()}, {k: v[sel] for k, v in b.items()}, ["leg_l", "leg_r"])
        lf_new = (b["limb_fail"] & ~a["limb_fail"]).any(1); lf_rec = (a["limb_fail"] & ~b["limb_fail"]).any(1)
        dep_new = int(((b["depth_err"] == 1) & (a["depth_err"] == 0))[sel].sum())
        dep_fix = int(((b["depth_err"] == 0) & (a["depth_err"] == 1))[sel].sum())
        return {"rows": int(sel.sum()), "videos": int(len(np.unique(V[sel]))),
                "dMPJPE": boot(V[sel], dm[sel]), "severe_rate_change": boot(V[sel], b["severe"][sel].astype(float)-a["severe"][sel]),
                "severe": [int(a["severe"][sel].sum()), int(b["severe"][sel].sum())],
                "newly_severe": int(sev_new[sel].sum()), "severe_recovered": int(sev_rec[sel].sum()),
                "knee": kn, "limb_fail_new": int(lf_new[sel].sum()), "limb_fail_recovered": int(lf_rec[sel].sum()),
                "depth_sign_new_errors": dep_new, "depth_sign_fixed": dep_fix,
                "torso_flip": [int(np.nan_to_num(a["torso_flip"][sel]).sum()), int(np.nan_to_num(b["torso_flip"][sel]).sum())],
                "harm_rows_gt_50mm": int((dm[sel] > 50).sum()), "improved_rows_gt_50mm": int((dm[sel] < -50).sum()),
                "mpjpe_tails": {base: tails(a["mpjpe"][sel]), cond: tails(b["mpjpe"][sel])},
                "mean_mpjpe": [float(a["mpjpe"][sel].mean()), float(b["mpjpe"][sel].mean())]}
    k = keep[rows]
    for name, sel in (("partial", partial & k), ("head_or_derived_only", ~partial & k)):
        out[name] = {c: block(sel, c) for c in RESTORE}
        out[name]["confidence_effect_R11_minus_Rc"] = block(sel, "R11", base="Rc")
        out[name]["coordinate_effect_R10_minus_M"] = out[name]["R10"]["dMPJPE"]
        out[name]["crop_effect_R01_minus_M"] = out[name]["R01"]["dMPJPE"]
        inter = (met["R11"]["mpjpe"]-met["R10"]["mpjpe"]-met["R01"]["mpjpe"]+met["M"]["mpjpe"])
        out[name]["interaction"] = boot(V[sel], inter[sel])
    sel = partial & k
    # strata on PARTIAL rows
    nlimb = new[:, DIRECT].sum(1)
    lm = ctx["lm"][rows]
    from eval_real_observation_estimator import CANON, SOURCE_OF
    src = [CANON.index(SOURCE_OF[n]) for n in JOINTS]
    conf = lm[:, src, 2]
    minconf = np.where(new[:, DIRECT], conf[:, DIRECT], np.inf).min(1)
    dm = met["R11"]["mpjpe"]-met["M"]["mpjpe"]
    strata = {}
    defs = {"split": ctx["split"][rows], "posture": ctx["labels"]["posture"][rows],
            "restored_limb_tokens": np.minimum(nlimb, 4).astype(str),
            "min_restored_conf": np.where(minconf < .1, "[0,.1)", np.where(minconf < .2, "[.1,.2)", "[.2,.3)")),
            "baseline_mpjpe": np.array(["<50", "50-100", "100-200", ">200"])[np.digitize(met["M"]["mpjpe"], [50, 100, 200])],
            "single_restored_joint": np.array([JOINTS[DIRECT[np.flatnonzero(r[DIRECT])[0]]] if r[DIRECT].sum() == 1 else "multiple"
                                               for r in new])}
    for key, lab in defs.items():
        strata[key] = {str(v): {"rows": int((sel & (lab == v)).sum()), "dMPJPE_R11": boot(V[sel & (lab == v)], dm[sel & (lab == v)]),
                                "severe": [int(met["M"]["severe"][sel & (lab == v)].sum()), int(met["R11"]["severe"][sel & (lab == v)].sum())]}
                       for v in sorted(set(lab[sel]))}
    out["partial_strata"] = strata
    # restored 2D accuracy (evaluation-only, after inference)
    uv = ctx["raw"]["uv"][rows]; sizes = ctx["sizes"][rows]
    gt12 = uv[:, [SMPL_OF[j] for j in DIRECT]]
    diag = np.linalg.norm(gt12.max(1)-gt12.min(1), axis=-1)
    pxr = lm[:, src, :2][:, DIRECT]
    err = np.linalg.norm(pxr-gt12, axis=-1)/diag[:, None]
    inside = (pxr[..., 0] >= 0) & (pxr[..., 0] < sizes[:, None, 0]) & (pxr[..., 1] >= 0) & (pxr[..., 1] < sizes[:, None, 1])
    nm = new[:, DIRECT] & sel[:, None] & (ctx["raw"]["root"][rows][:, 1] > 0)[:, None]
    cbin = np.digitize(conf[:, DIRECT], [.1, .2])
    acc = {}
    for b, lab in enumerate(("[0,.1)", "[.1,.2)", "[.2,.3)")):
        m = nm & (cbin == b)
        acc[lab] = {"tokens": int(m.sum()), "median_err_diag": float(np.median(err[m])) if m.any() else None,
                    "pck10": float((err[m] <= .1).mean()) if m.any() else None, "inside_image": float(inside[m].mean()) if m.any() else None}
    m = nm
    acc["all"] = {"tokens": int(m.sum()), "median_err_diag": float(np.median(err[m])), "pck10": float((err[m] <= .1).mean()),
                  "inside_image": float(inside[m].mean()),
                  "by_joint": {JOINTS[DIRECT[j]]: {"tokens": int(m[:, j].sum()), "pck10": float((err[m[:, j], j] <= .1).mean()) if m[:, j].any() else None}
                               for j in range(12)}}
    allgood = np.where(nm, err <= .1, True).all(1) & nm.any(1)
    anybad = (np.where(nm, err > .1, False)).any(1)
    acc["post_hoc_dMPJPE_by_restored_accuracy"] = {
        "all_restored_within_0.10": boot(V[allgood], dm[allgood]), "some_restored_beyond_0.10": boot(V[anybad], dm[anybad]),
        "note": "post-hoc, GT-referenced stratification; not a runtime-available rule"}
    out["restored_2d_accuracy_evaluation_only"] = acc
    # privileged references (separate; never pooled with real-source rates)
    priv = {}
    rowpos = {int(r): i for i, r in enumerate(rows)}
    for c in PRIVILEGED:
        pr = pz["rows_"+c]
        idx = np.array([rowpos[int(r)] for r in pr])
        sp = partial[idx] & k[idx]
        mc = outcome(pz["pred_"+c].astype(float), ctx["G"][pr])
        Vp = vid[pr]
        dM = mc["mpjpe"]-met["M"]["mpjpe"][idx]; dR = mc["mpjpe"]-met["R11"]["mpjpe"][idx]
        priv[c] = {"regime": REGIME[c], "rows_partial": int(sp.sum()),
                   "minus_M": boot(Vp[sp], dM[sp]), "minus_R11": boot(Vp[sp], dR[sp]),
                   "R11_minus_M_same_rows": boot(Vp[sp], (met["R11"]["mpjpe"][idx]-met["M"]["mpjpe"][idx])[sp]),
                   "severe_M_R11_this": [int(met["M"]["severe"][idx][sp].sum()), int(met["R11"]["severe"][idx][sp].sum()), int(mc["severe"][sp].sum())],
                   "knee_vs_M": knee_change({kk: v[idx][sp] for kk, v in met["M"].items()}, {kk: v[sp] for kk, v in mc.items()}, ["leg_l", "leg_r"])}
    out["privileged"] = priv
    out["_items"] = {"met": met, "rows": rows, "partial": partial}
    return out


def recovery_gates(rest):
    p = rest["partial"]["R11"]
    benefit = bool(p["dMPJPE"]["ci95"] and p["dMPJPE"]["ci95"][1] < 0 and p["severe_rate_change"]["ci95"]
                   and p["severe_rate_change"]["ci95"][1] < 0)
    harm = bool(p["newly_severe"] > .25*p["severe_recovered"] or p["knee"]["newly_wrong"] > .25*p["knee"]["fixed"]
                or p["mpjpe_tails"]["R11"]["p99"] > p["mpjpe_tails"]["M"]["p99"])
    return {"BENEFIT": benefit, "HARM": harm}


def causal_gates(fact):
    q = fact["populations"]["primary_well"]["pooled"]
    lo = lambda b: b["ci95"][0] if b["ci95"] else -np.inf
    causal = {"a_target_minus_control_lower": lo(q["induced_F10_target_minus_control"]),
              "b_F10_minus_F01_lower": lo(q["induced_F10_minus_F01"]),
              "c_dMPJPE_F11_lower": lo(q["dMPJPE_F11"])}
    causal["CAUSAL"] = bool(causal["a_target_minus_control_lower"] > 0 and causal["b_F10_minus_F01_lower"] > 0
                            and causal["c_dMPJPE_F11_lower"] >= 10.0)
    mask, crop, inter = q["mask_main"]["mean"], q["crop_main"]["mean"], q["interaction"]["mean"]
    crop_sec = bool(mask > 0 and abs(crop) <= .5*mask and abs(inter) <= .5*mask)
    return causal, {"mask_main": mask, "crop_main": crop, "interaction": inter, "CROP_SECONDARY": crop_sec}


def decide(repro, causal, crop, explained_, rec):
    if not repro:
        return "D"
    if causal["CAUSAL"] and crop["CROP_SECONDARY"] and explained_["EXPLAINED"]:
        if rec["BENEFIT"] and not rec["HARM"]:
            return "A"
        if not rec["BENEFIT"]:
            return "B"
        return "C"
    return "C"


# ------------------------------------------------------------------ temporal availability (measurement only)
def temporal_availability(ctx, keep):
    same, vm = ctx["same"], ctx["obs"]["input_valid_M"]
    bank, positions, row_of = ctx["bank"], ctx["positions"], ctx["row_of"]
    xm, sizes, uv, root = ctx["obs"]["input_2d_M"], ctx["sizes"], ctx["raw"]["uv"], ctx["raw"]["root"]
    lm = ctx["lm"]
    from eval_real_observation_estimator import CANON, SOURCE_OF
    src = [CANON.index(SOURCE_OF[n]) for n in JOINTS]

    def chain(i, direction):
        out, cur = [], i
        for _ in range(TEMPORAL_MAX_OFFSET):
            nid = bank.samples[int(positions[cur])].neighbors.get(direction)
            if not nid or nid not in row_of:
                break
            cur = row_of[nid]; out.append(cur)
        return out
    recs = []
    for i in np.flatnonzero(same & keep):
        miss = [j for j in DIRECT if not vm[i, j]]
        if not miss:
            continue
        back, fwd = chain(i, "previous"), chain(i, "next")
        for j in miss:
            def nearest(seq):
                inter = 0
                for o, r in enumerate(seq, 1):
                    if same[r] and vm[r, j]:
                        return o, r, inter
                    if not same[r]:
                        inter += 1
                return None, None, inter
            ob, rb, ib = nearest(back); of, rf, if_ = nearest(fwd)
            run = 1
            for r in back:
                if same[r] and not vm[r, j]:
                    run += 1
                else:
                    break
            for r in fwd:
                if same[r] and not vm[r, j]:
                    run += 1
                else:
                    break
            rec = {"row": int(i), "joint": JOINTS[j], "past": ob, "future": of, "inter_past": ib, "inter_future": if_, "run": run,
                   "dt_past": (bank.samples[int(positions[i])].timestamp-bank.samples[int(positions[rb])].timestamp) if rb is not None else None}
            W, H = sizes[i]
            # GT-free motion between bracketing observations, in units of the M valid-joint bbox diagonal at the target
            vv = vm[i]
            if vv.sum() >= 2:
                px = xm[i, vv, :2]*[W, H]; diag = float(np.linalg.norm(px.max(0)-px.min(0)))
            else:
                diag = np.nan
            if rb is not None and rf is not None and diag > 0:
                a = xm[rb, j, :2]*sizes[rb]; b = xm[rf, j, :2]*sizes[rf]
                rec["bracket_motion_diag"] = float(np.linalg.norm(a-b)/diag)
            # evaluation-only: GT projection error of the nearest past observation and of the raw current coordinate
            if root[i][1] > 0:
                gt12 = uv[i, [SMPL_OF[d] for d in DIRECT]]; gd = float(np.linalg.norm(gt12.max(0)-gt12.min(0)))
                g = uv[i, SMPL_OF[j]]
                if rb is not None:
                    rec["past_err_diag_eval_only"] = float(np.linalg.norm(xm[rb, j, :2]*sizes[rb]-g)/gd)
                r_ = lm[i, src[j]]
                if np.isfinite(r_[0]):
                    rec["raw_current_err_diag_eval_only"] = float(np.linalg.norm(r_[:2]-g)/gd)
            recs.append(rec)
    out = {"missing_limb_tokens": len(recs), "rows": int(len({r["row"] for r in recs}))}
    for o in TEMPORAL_REPORT_OFFSETS:
        out[f"observed_within_{o}"] = {
            "past": float(np.mean([r["past"] is not None and r["past"] <= o for r in recs])),
            "future": float(np.mean([r["future"] is not None and r["future"] <= o for r in recs])),
            "either": float(np.mean([(r["past"] is not None and r["past"] <= o) or (r["future"] is not None and r["future"] <= o) for r in recs])),
            "bracketed": float(np.mean([r["past"] is not None and r["past"] <= o and r["future"] is not None and r["future"] <= o for r in recs]))}
    out["missing_run_length"] = distribution([r["run"] for r in recs])
    out["dt_past_seconds"] = distribution([r["dt_past"] for r in recs if r["dt_past"] is not None])
    out["bracketed_with_intervening_non_same_person"] = int(sum(1 for r in recs if r["past"] and r["future"] and (r["inter_past"] or r["inter_future"])))
    out["bracket_motion_diag"] = distribution([r["bracket_motion_diag"] for r in recs if "bracket_motion_diag" in r])
    pe = [r["past_err_diag_eval_only"] for r in recs if "past_err_diag_eval_only" in r]
    ce = [r["raw_current_err_diag_eval_only"] for r in recs if "raw_current_err_diag_eval_only" in r]
    paired = [(r["past_err_diag_eval_only"], r["raw_current_err_diag_eval_only"]) for r in recs
              if "past_err_diag_eval_only" in r and "raw_current_err_diag_eval_only" in r]
    out["evaluation_only_2d_error"] = {
        "nearest_past_observation": distribution(pe) | {"pck10": float(np.mean(np.asarray(pe) <= .1)) if pe else None},
        "raw_current_low_conf": distribution(ce) | {"pck10": float(np.mean(np.asarray(ce) <= .1)) if ce else None},
        "paired_tokens": len(paired),
        "paired_past_better": float(np.mean([a < b for a, b in paired])) if paired else None,
        "note": "GT-projection error per GT limb-box diagonal; measurement of evidence content only, nothing is filled or fed to S0"}
    by_joint = defaultdict(list)
    for r in recs:
        by_joint[r["joint"]].append(r)
    out["by_joint"] = {j: {"tokens": len(v), "either_within_1": float(np.mean([(x["past"] == 1) or (x["future"] == 1) for x in v])),
                           "either_within_5": float(np.mean([x["past"] is not None or x["future"] is not None for x in v]))}
                       for j, v in sorted(by_joint.items())}
    return out


# ------------------------------------------------------------------ qualitative selections + figure
def selections(ctx, fact, rest):
    it = fact["_items"]; pats = ctx["decl"]["patterns"]; sid = ctx["sid"]
    row, pat, d = it["row"], it["pat"], it["d"]
    well = ctx["labels"]["quality_M"][row] == "well"
    kinds = np.array([pats[p]["kind"] for p in pat]); nj = np.array([len(pats[p]["joints"]) for p in pat])
    legs = np.array([all(l.startswith("leg") for l in pats[p]["affected_limbs"]) for p in pat])

    def pick(mask, values, largest=True):
        idx = np.flatnonzero(mask)
        if not len(idx):
            return None
        v = values[idx] if largest else -values[idx]
        order = np.lexsort((pat[idx], sid[row[idx]], -v))
        return int(idx[order[0]])
    out = {}
    i = pick(well & (kinds == "single") & legs, d["F10"])
    j = pick(well & (kinds == "single"), d["F01"])
    for key, k, c in (("1_single_leg_max_F10", i, "F10"), ("2_single_max_F01", j, "F01")):
        if k is not None:
            out[key] = {"sample_id": str(sid[row[k]]), "pattern": pats[pat[k]]["name"], "item": k,
                        "mpjpe": {cc: float(it["met"][cc]["mpjpe"][k]) for cc in FACTORIAL},
                        "crop_side_ratio": float(ctx["fz"]["side_ratio"][k])}
    rm = rest["_items"]; rows = rm["rows"]; part = rm["partial"]
    dm = rm["met"]["R11"]["mpjpe"]-rm["met"]["M"]["mpjpe"]
    for key, largest in (("3_partial_max_R11_improvement", False), ("4_partial_max_R11_harm", True)):
        idx = np.flatnonzero(part)
        v = dm[idx] if largest else -dm[idx]
        k = idx[np.lexsort((sid[rows[idx]], -v))[0]]
        out[key] = {"sample_id": str(sid[rows[k]]), "item": int(k), "restored": [JOINTS[t] for t in np.flatnonzero(ctx["rz"]["new_tokens"][k])],
                    "mpjpe": {"M": float(rm["met"]["M"]["mpjpe"][k]), "R11": float(rm["met"]["R11"]["mpjpe"][k])}}
    return out


def figure(ctx, sel, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    edges = [(c, PARENT[c]) for c in range(1, 17)]
    fig, axes = plt.subplots(len(sel), 2, figsize=(11, 4.2*len(sel)))
    fi = np.load(Path(ctx["infer_dir"])/"factorial_inputs.npz")
    ri = np.load(Path(ctx["infer_dir"])/"restoration_inputs.npz")
    colors = {"F00": "k", "F10": "tab:red", "F01": "tab:blue", "F11": "tab:purple", "M": "k", "R11": "tab:green", "GT": "tab:orange"}
    for ax, (key, s) in zip(axes, sel.items()):
        k = s["item"]
        if key.startswith(("1", "2")):
            r = ctx["fz"]["rec_row"][k]
            n = int(np.flatnonzero(ctx["fz"]["substrate"] == r)[0])
            tens = {"F00": fi["F00"][n], "F10": fi["F10"][k], "F01": fi["F01"][k], "F11": fi["F11"][k]}
            preds = {"F00": ctx["pred"]["M"][r], **{c: ctx["fz"]["pred_"+c][k] for c in ("F10", "F01", "F11")}}
        else:
            r = ctx["rz"]["rows"][k]
            tens = {"M": ctx["geoM"][r], "R11": ri["R11"][k]}
            preds = {"M": ctx["pred"]["M"][r], "R11": ctx["rz"]["pred_R11"][k]}
        for c, t in tens.items():
            v = t[:, 3] > .5
            for a, b in edges:
                if v[a] and v[b]:
                    ax[0].plot([t[a, 0], t[b, 0]], [-t[a, 1], -t[b, 1]], color=colors[c], lw=1.2, alpha=.7)
            ax[0].scatter(t[v, 0], -t[v, 1], s=8, color=colors[c], label=c)
        ax[0].set_title(f"{key}: model-visible tokens (crop units)\n{s['sample_id']} {s.get('pattern', '')}", fontsize=8)
        ax[0].legend(fontsize=7); ax[0].set_aspect("equal")
        preds["GT"] = ctx["G"][r]
        for c, P in preds.items():
            P = P-P[0]
            for a, b in edges:
                ax[1].plot([P[a, 1], P[b, 1]], [P[a, 2], P[b, 2]], color=colors[c], lw=1.2, alpha=.8)
            ax[1].plot([], [], color=colors[c], label=f"{c} {s['mpjpe'].get(c, float('nan')):.0f} mm" if c in s["mpjpe"] else
                       ("GT (EVALUATION ONLY, not available to S0)" if c == "GT" else c))
        ax[1].set_title("S0 output, side view (camera depth vs up)", fontsize=8); ax[1].legend(fontsize=7); ax[1].set_aspect("equal")
    fig.tight_layout(); fig.savefig(path, dpi=90); plt.close(fig)


# ------------------------------------------------------------------ stage: analyze
def cmd_analyze(args):
    decl = require_declaration(args)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=False)
    infer_dir = Path(args.infer_dir)
    inf = json.loads((infer_dir/"infer_report.json").read_text())
    if inf["declaration_sha256"] != args.declaration_sha:
        raise RuntimeError("STOP: inference ran under a different declaration")
    for f, digest in inf["outputs_sha256"].items():
        if sha(infer_dir/f) != digest:
            raise RuntimeError("STOP: inference output changed")
    bank, obs, side, positions = load_context(args.w102, args.w101_data)
    labels = load_w103_labels(args.w103, obs["sample_id"])
    lm, audit = raw_landmarks(args.w102, bank, positions, obs)
    pred, _ = load_predictions(args.w102, bank, positions)
    raw = raw_references(Path(args.w102)/"data/raw", bank, positions)
    G = bank.arrays["target_3d"][positions].astype(float); G = G-G[:, :1]
    if not raw["valid"].all() or np.abs(raw["G"]-G).max() > 1e-6:
        raise RuntimeError("STOP: valid GT alignment cannot be established")
    seq = np.array([bank.samples[int(p)].sequence_id for p in positions])
    from w103_observability import geometry_all
    ctx = {"decl": decl, "bank": bank, "obs": obs, "positions": positions, "labels": labels, "lm": lm, "pred": pred, "raw": raw,
           "G": G, "seq": seq, "vid": np.array([s.split(":")[1] for s in seq]), "sid": np.asarray(obs["sample_id"]),
           "split": np.array([bank.samples[int(p)].split for p in positions]), "same": np.asarray(side["identity"]) == "same_person",
           "sizes": np.array([bank.samples[int(p)].image_size for p in positions], float),
           "row_of": {str(s): i for i, s in enumerate(obs["sample_id"])}, "infer_dir": infer_dir,
           "fz": dict(np.load(infer_dir/"factorial.npz")), "rz": dict(np.load(infer_dir/"restoration.npz")),
           "pz": dict(np.load(infer_dir/"privileged.npz")), "geoM": geometry_all(bank, obs)["M"][positions]}
    ctx["Mm"] = outcome(pred["M"], G)
    # F00 recomputed vs stored: discrete decision parity
    sub = ctx["fz"]["substrate"]
    rec_m = outcome(ctx["fz"]["pred_F00"].astype(float), G[sub])
    parity = {k: int((np.nan_to_num(rec_m[k].astype(float), nan=-1) != np.nan_to_num(ctx["Mm"][k][sub].astype(float), nan=-1)).sum())
              for k in ("severe", "knee_wrong", "depth_err", "torso_flip", "limb_fail")}
    behind = raw["root"][:, 1] <= 0
    report = {"schema": "w104_analysis_v1", "declaration_sha256": args.declaration_sha,
              "infer_gate_reproduction_pass": inf["gate_reproduction_pass"], "F00_recomputed_decision_changes_vs_stored": parity,
              "reference_validity": {"rows_root_behind_camera": int(behind.sum()), "sequences": dict(Counter(seq[behind])),
                                     "same_person_rows": int((behind & ctx["same"]).sum()), "note": CROSSCOUNTRY_NOTE}}
    repro = bool(inf["gate_reproduction_pass"] and sum(parity.values()) == 0)
    report["REPRODUCTION"] = repro
    results = {}
    for run, keep in (("primary", np.ones(len(positions), bool)), ("reference_valid_sensitivity", ~behind)):
        fact = factorial_analysis(ctx, keep)
        rest = restoration_analysis(ctx, keep)
        causal, crop = causal_gates(fact)
        rec = recovery_gates(rest)
        case = decide(repro, causal, crop, fact["explained"], rec)
        if run == "primary":
            sel = selections(ctx, fact, rest)
            report["qualitative_selections"] = sel
            figure(ctx, sel, out/"qualitative_selections.png")
            report["temporal_availability"] = temporal_availability(ctx, keep)
        results[run] = {"factorial": {k: v for k, v in fact.items() if k != "_items"},
                        "restoration": {k: v for k, v in rest.items() if k != "_items"},
                        "gates": {"causal": causal, "crop": crop, "explained": {k: v for k, v in fact["explained"].items() if k != "classes"},
                                  "recovery": rec}, "CASE": case}
        print(run, "CASE", case, json.dumps(results[run]["gates"], default=str), flush=True)
    report["results"] = results
    report["CASE"] = results["primary"]["CASE"]
    dump(out/"analysis_report.json", report)
    print("analysis sha256", sha(out/"analysis_report.json"))


# ------------------------------------------------------------------ stage: video (GT-free)
def cmd_video(args):
    import torch
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    from framepose.crops import crop_box, geometry_in_crop
    from framepose.train import load_checkpoint
    from w101_reference_estimator import to_input
    from w103_observability import S0_CHECKPOINT_SHA, W102_MANIFEST_SHA
    from w104_missing_joint import s0_forward
    decl = require_declaration(args)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=False)
    w102 = Path(args.w102)
    if sha(w102/"evidence_manifest.json") != W102_MANIFEST_SHA:
        raise RuntimeError("STOP: W102 manifest identity failed")
    manifest = json.loads((w102/"evidence_manifest.json").read_text())["files"]
    if sha(w102/VIDEO_PREDICTIONS) != manifest[VIDEO_PREDICTIONS]["sha256"]:
        raise RuntimeError("STOP: historical clip observation identity failed")
    z = np.load(w102/VIDEO_PREDICTIONS)
    lm, x0, v0, geo0, p0 = z["landmarks"], z["input_2d"], z["input_valid"], z["geometry"], z["predictions"]
    k = np.flatnonzero(v0[:, 1] & (x0[:, 1, 0] > 0))[0]
    size = (int(round(lm[k, 10, 0]/x0[k, 1, 0])), int(round(lm[k, 10, 1]/x0[k, 1, 1])))  # left_hip pixel / normalized
    rows = [to_input(l if np.isfinite(l[:, 0]).any() else None, size) for l in lm]
    x = np.stack([r[0] for r in rows]); v = np.stack([r[1] for r in rows])
    geo = np.stack([geometry_in_crop(a, b, size, crop_box(a, b, size)) for a, b in rows])
    if sha(Path(args.w101_data)/"s0_checkpoint.pt") != S0_CHECKPOINT_SHA:
        raise RuntimeError("STOP: S0 checkpoint identity failed")
    model, _ = load_checkpoint(Path(args.w101_data)/"s0_checkpoint.pt", device="cuda")
    pM = s0_forward(model, torch, geo)
    report = {"schema": "w104_video_v1", "declaration_sha256": args.declaration_sha, "frames": len(lm), "image_size": size,
              "input_reproduced": bool(np.array_equal(x, x0) and np.array_equal(v, v0)), "geometry_byte_equal": bool(geo.tobytes() == geo0.tobytes()),
              "prediction_byte_equal": bool(pM.tobytes() == p0.tobytes()), "prediction_max_abs_mm": float(1000*np.abs(pM-p0).max()),
              "GT": "none (historical real clip; GT-free consequences only)", "conditions": {}}
    complete = v[:, DIRECT].all(1)
    conds = {}
    # R11 on frames with a restorable token
    R = geo.copy(); changed = np.zeros(len(lm), bool)
    for i in range(len(lm)):
        raw, have, new = restore_row(lm[i], size, x[i], v[i])
        if new.any():
            R[i] = restoration_inputs(x[i], v[i], raw, have, size)[0]["R11"]; changed[i] = True
    conds["R11"] = (R, changed)
    names = {"single:left_knee": (2,), "single:left_ankle": (3,), "single:right_knee": (5,), "single:right_ankle": (6,),
             "bilateral:left_ankle+right_ankle": (3, 6)}
    for name, joints in names.items():
        F = geo.copy()
        for i in np.flatnonzero(complete):
            F[i] = factorial_row(x[i], v[i], size, joints)[0]["F11"]
        conds["F11 "+name] = (F, complete.copy())
    def kinematics(P):
        flex = np.stack([angle_deg(P[:, k]-P[:, h], P[:, a]-P[:, k]) for h, k, a in KNEES.values()], 1)
        nrm = np.stack([np.cross(P[:, k]-P[:, h], P[:, a]-P[:, k]) for h, k, a in KNEES.values()], 1)
        steps = np.stack([angle_deg(P[1:, b]-P[1:, a], P[:-1, b]-P[:-1, a]) for a, b in SEGMENTS.values()], 1)
        return flex, nrm, steps
    fM, nM, sM = kinematics(pM.astype(float))
    report["M"] = {"valid_histogram": dict(Counter(map(int, v.sum(1)))), "complete_limb_frames": int(complete.sum()),
                   "segment_steps_over_45": int((sM > 45).sum()), "knee_flexion": distribution(fM)}
    for c, (geom, applied) in conds.items():
        P = s0_forward(model, torch, geom).astype(float)
        f, n, s = kinematics(P)
        ok = (f >= 20) & (fM >= 20)
        flip = ok & (angle_deg(n, nM) > 90)
        dev = 1000*np.linalg.norm((P-P[:, :1])-(pM-pM[:, :1]), axis=-1).mean(-1)
        report["conditions"][c] = {
            "frames_changed": int(applied.sum()), "regime": "GT-free counterfactual diagnostic" if c != "R11" else REGIME["R11"],
            "mean_joint_deviation_from_M_mm": distribution(dev[applied]),
            "knee_normal_flip_vs_M": int(flip[applied].sum()), "knee_both_flexed_frames": int(ok[applied].sum()),
            "knee_flexion_change_over_45deg": int((np.abs(f-fM)[applied] > 45).sum()),
            "segment_steps_over_45": int((s > 45).sum()), "segment_steps_over_45_M": int((sM > 45).sum())}
        print(c, report["conditions"][c], flush=True)
    dump(out/"video_report.json", report)


# ------------------------------------------------------------------ stage: posthoc (written AFTER the primary results; never decisive)
def cmd_posthoc(args):
    """Post-hoc descriptive views written after reading the primary results; they change no gate or CASE.

    (1) Non-tautological factorial outcomes: removing an end joint leaves its own segment unobserved, so the declared
        limb failure (segment > 45 deg) is close to definitional. Here only knee branch (legs) and SEVERE are used.
    (2) Where frozen S0 places a removed joint (distance to root / parent vs GT), and knee flexion under removal.
    (3) Historical clip: which tokens are invalid, and the direction of knee-flexion change under removal.
    """
    import torch
    from framepose.train import load_checkpoint
    from w104_missing_joint import VIDEO_PREDICTIONS, s0_forward
    decl = require_declaration(args)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=False)
    infer_dir = Path(args.infer_dir)
    inf = json.loads((infer_dir/"infer_report.json").read_text())
    for f, digest in inf["outputs_sha256"].items():
        if sha(infer_dir/f) != digest:
            raise RuntimeError("STOP: inference output changed")
    bank, obs, side, positions = load_context(args.w102, args.w101_data)
    labels = load_w103_labels(args.w103, obs["sample_id"])
    pred, _ = load_predictions(args.w102, bank, positions)
    G = bank.arrays["target_3d"][positions].astype(float); G = G-G[:, :1]
    vid = np.array([bank.samples[int(p)].sequence_id.split(":")[1] for p in positions])
    fz = dict(np.load(infer_dir/"factorial.npz"))
    pats = decl["patterns"]; names = [p["name"] for p in pats]
    row, pat = fz["rec_row"], fz["rec_pattern"]
    M = outcome(pred["M"], G)
    met = {c: outcome(fz["pred_"+c].astype(float), G[row]) for c in ("F10", "F01", "F11")}
    base = {k: v[row] for k, v in M.items()}
    lookup = {(int(r), int(p)): i for i, (r, p) in enumerate(zip(row, pat))}
    well = labels["quality_M"][row] == "well"
    report = {"schema": "w104_posthoc_v1", "status": "post-hoc, descriptive; written after the primary analysis; not part of any gate",
              "declaration_sha256": args.declaration_sha}
    # (1a) knee branch only: pure-leg target patterns vs their mirrored arm control, on the same knee
    kr = defaultdict(list)
    for p, pdef in enumerate(pats):
        limbs = pdef["affected_limbs"]
        if pdef["kind"] == "control" or not all(l.startswith("leg") for l in limbs):
            continue
        cname = decl["controls"].get(pdef["name"])
        s = np.flatnonzero((pat == p) & well)
        for leg in limbs:
            k = KNEE_OF_LIMB[leg]
            a = base["knee_wrong"][s, k]
            for c in ("F10", "F01", "F11"):
                b = met[c]["knee_wrong"][s, k]
                ok = a == 0
                kr[c].append((row[s][ok], np.nan_to_num(b[ok]) == 1))
            if cname:
                j = np.array([lookup[(int(r), names.index(cname))] for r in row[s]])
                b = met["F10"]["knee_wrong"][j, k]
                ok = a == 0
                kr["mirrored_arm_control_F10"].append((row[s][ok], np.nan_to_num(b[ok]) == 1))
    report["knee_newly_wrong_pure_leg_patterns_primary"] = {
        c: pooled(np.concatenate([r for r, _ in v]), np.concatenate([x for _, x in v]).astype(float), vid) for c, v in kr.items()}
    report["knee_newly_wrong_definition"] = ("affected knee correct and eligible under F00; share wrong under the condition "
                                             "(ineligible under the condition counts as not wrong); per-row pooled, video bootstrap")
    # (1b) newly severe only
    sev = {}
    targ = np.array([pats[p]["kind"] != "control" for p in pat]) & well
    for c in ("F10", "F01", "F11"):
        sev[c] = pooled(row[targ], (met[c]["severe"] & ~base["severe"])[targ].astype(float), vid)
    cs, cv = [], []
    for i in np.flatnonzero(targ):
        cname = decl["controls"].get(names[pat[i]])
        if cname:
            j = lookup[(int(row[i]), names.index(cname))]
            cs.append(row[i])
            cv.append(float(met["F10"]["severe"][i] and not base["severe"][i]) - float(met["F10"]["severe"][j] and not base["severe"][i]))
    sev["F10_target_minus_mirrored_control"] = pooled(np.array(cs), np.array(cv), vid)
    sev["F11_newly_severe_by_pattern"] = {names[p]: int((met["F11"]["severe"] & ~base["severe"])[(pat == p) & well].sum())
                                          for p in range(len(pats))}
    report["newly_severe_primary"] = sev
    # (2) placement of a removed joint (F10) and knee flexion under removal
    place = {}
    for p, pdef in enumerate(pats):
        if pdef["kind"] != "single":
            continue
        j = pdef["joints"][0]; par = PARENT[j]
        s = (pat == p) & well
        P = fz["pred_F10"][s].astype(float); P = P-P[:, :1]; Gs = G[row[s]]
        P0 = pred["M"][row[s]]; P0 = P0-P0[:, :1]
        place[pdef["name"]] = {
            "median_pred_dist_root_mm": float(np.median(1000*np.linalg.norm(P[:, j], axis=-1))),
            "median_gt_dist_root_mm": float(np.median(1000*np.linalg.norm(Gs[:, j], axis=-1))),
            "median_pred_bone_to_parent_mm": float(np.median(1000*np.linalg.norm(P[:, j]-P[:, par], axis=-1))),
            "median_gt_bone_to_parent_mm": float(np.median(1000*np.linalg.norm(Gs[:, j]-Gs[:, par], axis=-1))),
            "median_joint_error_F00_mm": float(np.median(1000*np.linalg.norm(P0[:, j]-Gs[:, j], axis=-1))),
            "median_joint_error_F10_mm": float(np.median(1000*np.linalg.norm(P[:, j]-Gs[:, j], axis=-1)))}
    report["removed_joint_placement_F10_primary"] = place
    flex = {}
    for nm in ("single:left_knee", "single:left_ankle", "single:right_knee", "single:right_ankle"):
        s = (pat == names.index(nm)) & well
        h, k, a = KNEES["l" if "left" in nm else "r"]
        f = lambda X: angle_deg(X[:, k]-X[:, h], X[:, a]-X[:, k])
        flex[nm] = {"GT": float(np.median(f(G[row[s]]))), "F00": float(np.median(f(pred["M"][row[s]]))),
                    "F10": float(np.median(f(fz["pred_F10"][s].astype(float)))), "F11": float(np.median(f(fz["pred_F11"][s].astype(float))))}
    report["median_knee_flexion_deg_under_removal_primary"] = flex
    # (3) historical clip
    z = np.load(Path(args.w102)/VIDEO_PREDICTIONS)
    vr = json.loads((Path(args.video_dir)/"video_report.json").read_text())
    size = tuple(vr["image_size"]); lm, v, x = z["landmarks"], z["input_valid"], z["input_2d"]
    from eval_real_observation_estimator import CANON, SOURCE_OF
    src = [CANON.index(SOURCE_OF[n]) for n in JOINTS]
    part = np.flatnonzero(~v[:, DIRECT].all(1))
    report["video_partial_frames"] = {
        "frames": int(len(part)),
        "invalid_limb_tokens": dict(Counter(JOINTS[DIRECT[j]] for i in part for j in np.flatnonzero(~v[i, DIRECT]))),
        "raw_conf_of_invalid": distribution([lm[i, src[DIRECT[j]], 2] for i in part for j in np.flatnonzero(~v[i, DIRECT])])}
    model, _ = load_checkpoint(Path(args.w101_data)/"s0_checkpoint.pt", device="cuda")
    complete = v[:, DIRECT].all(1)
    pM = z["predictions"].astype(float)
    vf = {}
    for nm, joints in (("single:left_knee", (2,)), ("single:left_ankle", (3,)), ("single:right_knee", (5,)),
                       ("single:right_ankle", (6,)), ("bilateral:left_ankle+right_ankle", (3, 6))):
        F = z["geometry"].copy()
        for i in np.flatnonzero(complete):
            F[i] = factorial_row(x[i], v[i], size, joints)[0]["F11"]
        P = s0_forward(model, torch, F).astype(float)
        e = {}
        for side_, (h, k, a) in KNEES.items():
            fm = angle_deg(pM[complete, k]-pM[complete, h], pM[complete, a]-pM[complete, k])
            fc = angle_deg(P[complete, k]-P[complete, h], P[complete, a]-P[complete, k])
            e["knee_"+side_] = {"median_M": float(np.median(fm)), "median_F11": float(np.median(fc)),
                                "share_flexion_increase_over_45": float(np.mean(fc-fm > 45))}
        vf[nm] = e
    report["video_knee_flexion_F11"] = vf
    dump(out/"posthoc_report.json", report)
    print(json.dumps(report, default=str)[:7000])
