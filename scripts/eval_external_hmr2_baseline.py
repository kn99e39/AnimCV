#!/usr/bin/env python3
"""EVALUATION (Worklog 97): frozen S0 (S) vs official HMR 2.0b (E) on identical rows; GT for scoring only.

Rows, GT joints, strata, SMPL pelvis, W95 legs and W96 measures come from the frozen W94/W95/W96 code paths.  E joints
are the 24 SMPL joints of the predicted parameters mapped to the AnimCV canonical 17 (``run_external_hmr2_baseline``).
Reconstruction metrics compare S and E against the same GT in the same canonical frame; the common-rig replay swaps
ONLY the leg joint source inside the W95 B_S0 control (SMPL pelvis Object, S0 thorax, J1X arm, zero twist ``Z``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_external_hmr2_baseline as hmr  # noqa: E402
import run_leg_knee_hinge_twist as w96  # noqa: E402
import run_pelvis_frame_leg_articulation as w95  # noqa: E402
import run_pelvis_torso_ownership as w94  # noqa: E402
from run_root_motion_contact_h0_replay import _matched_oracle_sequence  # noqa: E402
from run_runtime_heading_3dhp import shim  # noqa: E402
from run_torso_frame_rig_replay import NAMES, S0_SHA, conj, spine_local_xyzw  # noqa: E402
from run_visual_depth_motion_attribution import W87_PREDICTIONS, _brief  # noqa: E402

from framepose.bank import load_bank  # noqa: E402
from framepose.bedlam_supervision import ExternalBank  # noqa: E402
from framepose.connected_limb_transfer import chain_targets  # noqa: E402
from motion.torso_frame import geodesic_degrees, matrix_to_quaternion_xyzw, quaternion_to_matrix, torso_frames  # noqa: E402
from pose.framepose_bridge import assemble_h0, build_h0_lifted_sequence, sequence_ids_in_split  # noqa: E402
from pose.root_orientation_diagnostic import observe_yaw  # noqa: E402
from retarget.framepose_fk import derive_rig_alignment, load_rig_fk_calibration  # noqa: E402
from retarget.framepose_target_rest import load_target_rest_pose  # noqa: E402
from retarget.framepose_two_bone_ik import load_ik_chain_set, quaternion_angle_degrees, target_chain_geometry  # noqa: E402

LIMBS = {"upper_arm_l": (11, 12), "forearm_l": (12, 13), "upper_arm_r": (14, 15), "forearm_r": (15, 16),
         "thigh_l": (1, 2), "shin_l": (2, 3), "thigh_r": (4, 5), "shin_r": (5, 6)}
ENDS = {"wrist_l": 13, "wrist_r": 16, "ankle_l": 3, "ankle_r": 6}
LIMB_JOINTS = (2, 3, 5, 6, 12, 13, 15, 16)  # knees, ankles, elbows, wrists
KNEES = {"l": (1, 2, 3), "r": (4, 5, 6)}


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def ang(a, b):
    return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(_unit(a), _unit(b)))))))


def procrustes(P, G):
    """Similarity-aligned P (PA, secondary diagnostic)."""
    mp, mg = P.mean(0), G.mean(0)
    X, Y = P - mp, G - mg
    U, s, Vt = np.linalg.svd(X.T @ Y)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1, 1, d])
    R = U @ D @ Vt
    scale = (s * np.diag(D)).sum() / (X ** 2).sum()
    return scale * X @ R + mg


def source_metrics(J, G, R_smpl=None, R_rot=None):
    """Per-row reconstruction metrics of a root-relative canonical 17-joint source J against GT G."""
    J, G = J - J[0], G - G[0]
    out = {"mpjpe": 1000 * float(np.linalg.norm(J - G, axis=1).mean()),
           "pa_mpjpe": 1000 * float(np.linalg.norm(procrustes(J, G) - G, axis=1).mean()),
           "depth_err": 1000 * float(np.abs(J[list(LIMB_JOINTS), 1] - G[list(LIMB_JOINTS), 1]).mean())}
    for k, (a, b) in LIMBS.items():
        out[f"dir_{k}"] = ang(J[b] - J[a], G[b] - G[a])
    for k, j in ENDS.items():
        out[f"end_{k}"] = 1000 * float(np.linalg.norm(J[j] - G[j]))
    for s, (h, k, a) in KNEES.items():
        fj, fg = ang(J[k] - J[h], J[a] - J[k]), ang(G[k] - G[h], G[a] - G[k])
        out[f"knee_flex_gt_{s}"], out[f"knee_flex_{s}"] = fg, fj
        if fj >= 20 and fg >= 20:
            e = ang(np.cross(J[k] - J[h], J[a] - J[k]), np.cross(G[k] - G[h], G[a] - G[k]))
            out[f"knee_plane_{s}"], out[f"knee_wrong_{s}"] = e, float(e > 90)
    Rt, _ = torso_frames(J)
    Rg, _ = torso_frames(G)
    if Rt is not None and Rg is not None:
        out["torso_up"], out["torso_fwd"] = ang(Rt[:, 2], Rg[:, 2]), ang(Rt[:, 1], Rg[:, 1])
        out["torso_fwd_flip"] = float(out["torso_fwd"] > 90)
    if R_smpl is not None:
        low = w94.lower_trunk_frame(J)
        if low is not None:
            out["pelvis_joints_vs_smpl"] = geodesic_degrees(low, R_smpl)
        if R_rot is not None:
            out["pelvis_rot_vs_smpl"] = geodesic_degrees(R_rot, R_smpl)
    return out


def pelvis_from_global_orient(go):
    """HMR 2.0 global orient (OpenCV camera axes) -> AnimCV body frame, exactly as the W94 SMPL reference."""
    return w94.CV_TO_CANON @ np.asarray(go, float) @ w94.SMPL_BODY


def leg_measures_e(r, jE, ctx, smpl_legs):
    """W95 legs (zero twist) for a leg-joint source inside the B_S0 control frame + W96 Z measures."""
    if r["R_smpl"] is None or r["gt_legs"] is None:
        return None
    Rts = r["F"]["A_S0"][0]
    ev = w95.evaluate(jE, r["R_smpl"], Rts, ctx, r["gt_legs"])
    if ev is None:
        return None
    m = w96.process(r, ev, r["R_smpl"], ctx, smpl_legs, ("Z",))
    return ev, m


# ------------------------------------------------------------------ main

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bank", "s0_dir", "raw_3dpw", "w87_3dpw", "hp_bank", "hp_s0", "w87_3dhp", "w96_cases", "population", "predictions",
                 "rest", "calibration", "chains", "out_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--image-root-local", type=Path, default=None, help="local 3DPW imageFiles root for review crops (optional)")
    args = parser.parse_args()
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
    frozen = {args.w87_3dpw: W87_PREDICTIONS["3dpw"], args.w87_3dhp: W87_PREDICTIONS["3dhp"], args.hp_s0: S0_SHA["3dhp"],
              **{args.s0_dir / f"prediction_{s}.npy": S0_SHA[s] for s in ("validation", "test")}}
    for path, digest in frozen.items():
        if sha(path) != digest:
            raise SystemExit(f"STOP: frozen input changed: {path}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pop = json.loads(args.population.read_text())
    pred = np.load(args.predictions)
    E = {}
    for i, (pp, bp) in enumerate(zip(pred["population"], pred["index"][0])):
        if pred["valid"][i]:
            E[(str(pp), int(bp))] = {"j17": hmr.smpl24_to_canonical17(pred["joints24"][i]), "go": pred["global_orient"][i]}
    meta = json.loads(Path(str(args.predictions) + ".json").read_text()) if Path(str(args.predictions) + ".json").exists() else {}
    report = {"schema": "worklog97_external_hmr2_eval", "predeclared": hmr.PREDECLARED, "inference": meta,
              "predictions_sha256": sha(args.predictions), "population_sha256": sha(args.population), "results": {}}

    rest = load_target_rest_pose(args.rest)
    alignment = derive_rig_alignment(rest, load_rig_fk_calibration(args.calibration))
    Am = quaternion_to_matrix(alignment)
    target = target_chain_geometry(rest, load_ik_chain_set(args.chains).chains[0])
    sb = rest.bones["spine_01"]
    M = np.asarray(sb.matrix_local, float)[:3, :3]
    lf = w96.leg_frames(rest)
    ctx = {"alignment": alignment, "Am": Am, "rest": rest, "target": target, "spine_basis": M / np.linalg.norm(M, axis=0, keepdims=True),
           "spine_head": list(sb.head), "legs": w95.leg_targets(rest), "lf": lf, "psi0": w96.rest_offsets(lf, Am)}
    # ---- 3DPW rows exactly as W95/W96
    bank = load_bank(args.bank)
    s0, _ = assemble_h0(bank, {s: args.s0_dir / f"prediction_{s}.npy" for s in ("validation", "test")})
    smpl = w94.Smpl3dpw(args.raw_3dpw)
    sl = w96.SmplLegs(args.raw_3dpw)
    ctx3 = {**ctx, "chain_gt": chain_targets(bank.arrays["target_3d"], bank.arrays["target_valid"], w94.CHAIN),
            "time": lambda v, p: v.samples[p].frame_index / 30.0}
    joints3 = {}

    def g3(sid, pos):
        orc = _matched_oracle_sequence(sid, args.raw_3dpw, [bank.samples[p].frame_index for p in pos])
        js = [np.array([f.points[n].position for n in NAMES]) for f in orc.frames]
        for p, j in zip(pos, js):
            joints3[(sid, p)] = j
        return [o.yaw_radians for o in observe_yaw(orc)], js
    p3 = np.load(args.w87_3dpw)
    iv3 = bank.arrays["input_valid"]
    pw = {}
    for split in ("validation", "test"):
        pw[split] = w94.build_rows(bank, s0, sequence_ids_in_split(bank, split), g3, bank.arrays, p3, ctx3, smpl=smpl)
        w95.attach(pw[split], joints3, s0, lambda r: r["R_smpl"] is not None, lambda p: bool(iv3[p][[1, 2, 3, 4, 5, 6]].all()), ctx)
        w96.collect(pw[split], "S", "B_S0", ("Z",), ctx, sl)
        for rs in pw[split].values():
            for r in rs:
                G = joints3[(r["sid"], r["p"])]
                r["G17"], r["S17"] = G - G[0], s0[r["p"]] - s0[r["p"]][0]
                e = E.get((f"3dpw_{split}", r["p"]))
                r["E17"] = e["j17"] if e else None
                r["E_pelvis"] = pelvis_from_global_orient(e["go"]) if e else None
                r["m"] = {"S": source_metrics(r["S17"], r["G17"], r["R_smpl"]),
                          "E": source_metrics(r["E17"], r["G17"], r["R_smpl"], r["E_pelvis"]) if e else None}
                smpl_legs = sl.get(r["sid"], r["frame"]) if r["R_smpl"] is not None else None
                r["rigE"] = leg_measures_e(r, r["E17"], ctx, smpl_legs) if e else None
    # ---- 3DHP (no SMPL; limb metrics primary)
    mx = ExternalBank.load(args.hp_bank)
    hp = np.load(args.hp_s0)
    hrows = []
    for p, s in enumerate(mx.samples):
        if not bool(mx.arrays["target_valid"][p].all()):
            continue
        G = mx.arrays["target_3d"][p]
        e = E.get(("3dhp", p))
        r = {"sid": s.sequence_id, "p": p, "frame": int(s.frame_index), "G17": G - G[0], "S17": hp[p] - hp[p][0], "E17": e["j17"] if e else None}
        r["m"] = {"S": source_metrics(r["S17"], r["G17"]), "E": source_metrics(r["E17"], r["G17"]) if e else None}
        hrows.append(r)
    report["results"] = summarize_all(pw, hrows)
    report["temporal"] = {f"3dpw_{s}": temporal(pw[s]) for s in pw}
    report["rig_replay"] = {f"3dpw_{s}": rig_summary(pw[s], ctx) for s in pw}
    report["cases"] = export_cases(pw, ctx, joints3, args, rest, alignment, target)
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=1, default=str))
    review_cases(pw["test"], args, ctx)
    print(json.dumps({"coverage": report["results"]["coverage"]}, indent=1))


# ------------------------------------------------------------------ summaries

def challenge_masks(rows):
    def flex45(r):
        return any(r["m"]["S"].get(f"knee_flex_gt_{s}", 0) >= 45 for s in "lr")

    def straight(r):
        return all(r["m"]["S"].get(f"knee_flex_gt_{s}", 99) < 5 for s in "lr")

    def s0_wrong_plane(r):
        return any(r["m"]["S"].get(f"knee_plane_{s}", 0) > 45 for s in "lr")

    def out_of_plane(r):
        G = r["G17"]
        return any(abs(_unit(G[b] - G[a])[1]) > 0.5 for a, b in ((1, 2), (4, 5), (11, 12), (14, 15)))
    return {"waist_bend": lambda r: r.get("stratum") == "waist_bend", "seated_crouched_other": lambda r: r.get("stratum") == "other",
            "gt_knee_flex_ge_45": flex45, "gt_knee_near_straight": straight, "s0_wrong_knee_plane": s0_wrong_plane,
            "out_of_plane_limbs": out_of_plane}


METRIC_KEYS = ("mpjpe", "pa_mpjpe", "depth_err", "dir_upper_arm", "dir_forearm", "dir_thigh", "dir_shin", "end_wrist", "end_ankle",
               "knee_plane", "knee_wrong", "torso_up", "torso_fwd", "torso_fwd_flip", "pelvis_joints_vs_smpl", "pelvis_rot_vs_smpl")


def row_value(m, key):
    if m is None:
        return None
    if key in m:
        return m[key]
    vals = [m[k] for k in m if k.startswith(key + "_")]
    return float(np.mean(vals)) if vals else None


def summarize(rows, label):
    both = [r for r in rows if r["m"]["E"] is not None]
    out = {"rows": len(rows), "rows_with_E": len(both), "sequences": len({r["sid"] for r in both})}
    for key in METRIC_KEYS:
        pairs = [(r["sid"], row_value(r["m"]["S"], key), row_value(r["m"]["E"], key)) for r in both]
        if key == "pelvis_rot_vs_smpl":  # E only (S has no rotations); compare against S's joint proxy
            pairs = [(sd, row_value(r["m"]["S"], "pelvis_joints_vs_smpl"), e) for (sd, _, e), r in zip(pairs, both)]
        pairs = [p for p in pairs if p[1] is not None and p[2] is not None]
        if not pairs:
            continue
        sv, ev = np.array([p[1] for p in pairs]), np.array([p[2] for p in pairs])
        c = _brief(np.array([p[0] for p in pairs]), ev - sv)
        out[key] = {"n": len(pairs), "S_mean": float(sv.mean()), "E_mean": float(ev.mean()), "S_median": float(np.median(sv)),
                    "E_median": float(np.median(ev)), "S_p95": float(np.percentile(sv, 95)), "E_p95": float(np.percentile(ev, 95)),
                    "E_minus_S": c.get("mean"), "ci95": c.get("ci95"), "relative": float((ev.mean() - sv.mean()) / max(sv.mean(), 1e-9)),
                    "E_better_rows": float(np.mean(ev < sv))}
    return out


def material(o):
    res = {}
    for key in ("mpjpe", "dir_thigh", "dir_shin", "knee_plane"):
        x = o.get(key)
        if x is None or x.get("ci95") is None:
            res[key] = None
            continue
        res[key] = bool(x["ci95"][1] < 0 and x["relative"] <= -0.10)
    return res


def summarize_all(pw, hrows):
    out = {"coverage": {}}
    for split, seq_rows in pw.items():
        rows = [r for rs in seq_rows.values() for r in rs]
        out["coverage"][f"3dpw_{split}"] = {"rows": len(rows), "E_valid": sum(r["m"]["E"] is not None for r in rows)}
        out[f"3dpw_{split}"] = summarize(rows, split)
        out[f"3dpw_{split}"]["by_stratum"] = {s: summarize([r for r in rows if r.get("stratum") == s], s) for s in w94.STRATA}
        if split == "test":
            out["challenge_test"] = {k: summarize([r for r in rows if f(r)], k) for k, f in challenge_masks(rows).items()}
    out["coverage"]["3dhp"] = {"rows": len(hrows), "E_valid": sum(r["m"]["E"] is not None for r in hrows)}
    out["3dhp"] = summarize(hrows, "3dhp")
    out["material"] = {"3dpw_test": material(out["3dpw_test"]), "gt_knee_flex_ge_45": material(out["challenge_test"]["gt_knee_flex_ge_45"])}
    return out


def temporal(seq_rows):
    acc = defaultdict(lambda: defaultdict(list))
    for rs in seq_rows.values():
        for a, b in zip(rs[:-1], rs[1:]):
            if a["E17"] is None or b["E17"] is None:
                continue
            for k, (i, j) in LIMBS.items():
                gstep = ang(a["G17"][j] - a["G17"][i], b["G17"][j] - b["G17"][i])
                for src in ("S", "E"):
                    A, B = a[f"{src}17"], b[f"{src}17"]
                    st = ang(A[j] - A[i], B[j] - B[i])
                    acc[src]["step_residual"].append(abs(st - gstep))
                    if gstep < 2.0:
                        acc[src]["false_motion_when_gt_still"].append(st)
            for src in ("S", "E"):
                A, B = a[f"{src}17"], b[f"{src}17"]
                acc[src]["depth_motion_residual_mm"].append(1000 * float(np.abs((B[list(LIMB_JOINTS), 1] - A[list(LIMB_JOINTS), 1]) -
                                                                                 (b["G17"][list(LIMB_JOINTS), 1] - a["G17"][list(LIMB_JOINTS), 1])).mean()))
                ta, tb = torso_frames(A)[0], torso_frames(B)[0]
                ga, gb = torso_frames(a["G17"])[0], torso_frames(b["G17"])[0]
                if all(x is not None for x in (ta, tb, ga, gb)):
                    acc[src]["torso_flip"].append(float(geodesic_degrees(ta, tb) > 90 and geodesic_degrees(ga, gb) < 45))
                for s, (h, k, an) in KNEES.items():
                    na, nb = np.cross(A[k] - A[h], A[an] - A[k]), np.cross(B[k] - B[h], B[an] - B[k])
                    ga_, gb_ = np.cross(a["G17"][k] - a["G17"][h], a["G17"][an] - a["G17"][k]), np.cross(b["G17"][k] - b["G17"][h], b["G17"][an] - b["G17"][k])
                    if min(ang(A[k] - A[h], A[an] - A[k]), ang(B[k] - B[h], B[an] - B[k]), ang(a["G17"][k] - a["G17"][h], a["G17"][an] - a["G17"][k]),
                           ang(b["G17"][k] - b["G17"][h], b["G17"][an] - b["G17"][k])) >= 20:
                        acc[src]["knee_plane_flip"].append(float(ang(na, nb) > 90 and ang(ga_, gb_) < 45))
            if a["E_pelvis"] is not None and b["E_pelvis"] is not None and a["R_smpl"] is not None and b["R_smpl"] is not None:
                acc["E"]["pelvis_rot_spin"].append(float(geodesic_degrees(a["E_pelvis"], b["E_pelvis"]) > 90 and geodesic_degrees(a["R_smpl"], b["R_smpl"]) < 45))
    return {src: {k: {"n": len(v), "mean": float(np.mean(v)), "p95": float(np.percentile(v, 95)), "sum": float(np.sum(v))} for k, v in d.items()}
            for src, d in acc.items()}


def rig_records(seq_rows, ctx):
    """Per-row common-rig measures for S+Z and E+Z on the B_S0 control (plus E-only rows where S0 legs are unavailable)."""
    masks = challenge_masks([])
    recs = []
    for rs in seq_rows.values():
        for r in rs:
            if r.get("rigE") is None:
                continue
            have_s = ("S", "B_S0") in r.get("W96", {}) and r["S"].get("B_S0") is not None
            rec = {"sid": r["sid"], "s0_legs": have_s, "flags": {k: bool(f(r)) for k, f in masks.items()}, "stratum": r.get("stratum")}
            srcs = [("E", r["rigE"][0], r["rigE"][1], r["E17"])]
            if have_s:
                srcs.append(("S", r["S"]["B_S0"], r["W96"][("S", "B_S0")], r["S17"]))
            G = r["G17"]
            for src, ev, m, J in srcs:
                d = {"thigh_dir": np.mean([ev["legs"][x]["thigh_err"] for x in "lr"]),
                     "shin_dir": np.mean([ev["legs"][x]["shin_err"] for x in "lr"]),
                     "ankle_dir": np.mean([ev["legs"][x]["ankle_dir_err"] for x in "lr"]),
                     "recon_thigh_dir": np.mean([ang(J[b] - J[a], G[b] - G[a]) for a, b in ((1, 2), (4, 5))]),
                     "recon_shin_dir": np.mean([ang(J[b] - J[a], G[b] - G[a]) for a, b in ((2, 3), (5, 6))]),
                     "w95_any": float(ev["any"]), "w95_rom_only": float(any(v for k, v in ev["viol"].items() if k != "knee_hinge")),
                     "off_hinge": np.mean([abs(m[x]["Z"]["off_hinge"]) for x in "lr"])}
                re = [abs(m[x]["Z"]["smpl_roll_err"]) for x in "lr" if "smpl_roll_err" in m[x]["Z"]]
                d["roll_vs_smpl"] = np.mean(re) if re else np.nan
                kp = []
                for x, (h, k, a) in KNEES.items():
                    p = ev["legs"][x]["pose"]
                    if ang(G[k] - G[h], G[a] - G[k]) >= 20 and ev["legs"][x]["knee"] >= 20:
                        nw = r["R_smpl"] @ (ctx["Am"].T @ np.cross(p.upper, p.forearm))
                        kp.append(ang(nw, np.cross(G[k] - G[h], G[a] - G[k])))
                d["knee_plane_rig"] = np.mean(kp) if kp else np.nan
                d["q"] = {x: ev["legs"][x]["pose"].proximal_local for x in "lr"}
                rec[src] = d
            rec["gt_thigh"] = r["gt_legs"]
            recs.append(rec)
    return recs


RIG_KEYS = ("thigh_dir", "shin_dir", "ankle_dir", "recon_thigh_dir", "recon_shin_dir", "knee_plane_rig", "w95_any", "w95_rom_only",
            "off_hinge", "roll_vs_smpl")


def rig_agg(recs):
    paired = [r for r in recs if "S" in r]
    out = {"rows_paired": len(paired)}
    sid = np.array([r["sid"] for r in paired])
    for key in RIG_KEYS:
        s, e = np.array([r["S"][key] for r in paired], float), np.array([r["E"][key] for r in paired], float)
        ok = np.isfinite(s) & np.isfinite(e)
        if not ok.any():
            continue
        c = _brief(sid[ok], e[ok] - s[ok])
        out[key] = {"n": int(ok.sum()), "S_mean": float(s[ok].mean()), "E_mean": float(e[ok].mean()), "S_median": float(np.median(s[ok])),
                    "E_median": float(np.median(e[ok])), "E_minus_S": c.get("mean"), "ci95": c.get("ci95")}
    only = [r for r in recs if "S" not in r]
    out["E_only_rows_where_s0_legs_unavailable"] = {"rows": len(only), **{k: float(np.nanmean([r["E"][k] for r in only])) for k in ("thigh_dir", "shin_dir", "knee_plane_rig")}} if only else {"rows": 0}
    return out


def rig_summary(seq_rows, ctx):
    recs = rig_records(seq_rows, ctx)
    out = {"all": rig_agg(recs)}
    for k in challenge_masks([]):
        out[k] = rig_agg([r for r in recs if r["flags"][k]])
    for st in w94.STRATA:
        out[f"stratum_{st}"] = rig_agg([r for r in recs if r["stratum"] == st])
    for src in ("S", "E"):  # severe thigh-local jumps (W95 rule) on consecutive rows of the same sequence
        jumps, prev = 0, None
        for r in recs:
            if prev is not None and prev["sid"] == r["sid"] and src in prev and src in r:
                for x in "lr":
                    if quaternion_angle_degrees(prev[src]["q"][x], r[src]["q"][x]) > 90 and w95.angle(prev["gt_thigh"][x][0], r["gt_thigh"][x][0]) < 45:
                        jumps += 1
            prev = r
        out[f"severe_thigh_jumps_{src}"] = jumps
    return out


def export_rig_case(name, w, ctx, joints3, title):
    """S+Z vs E+Z vs GT-legs+Z (evaluation reference) on the B_S0 control frame (SMPL pelvis Object, S0 thorax, J1X arm)."""
    alignment, B, head, target = ctx["alignment"], ctx["spine_basis"], ctx["spine_head"], ctx["target"]
    slots = {"FIXED": "S0 legs + Z", "RUNTIME": "HMR 2.0b legs + Z", "GT": "GT legs + Z (eval only)"}
    frames, s_shin, s_plane = [], {v: [] for v in slots.values()}, {v: [] for v in slots.values()}
    for r in w:
        rec = {"bank_position": r["p"], "frame_index": r["frame"], "time_s": r["t"], "image": r["image"], "available": r["available"],
               "gt_valid": r["gt_arm_ok"], "runtime_status": "KNOWN", "runtime_reasons": [], "yaw_deg": {}, "modes": {}, "forward_cam": {},
               "mode_lines": {}, "object_rotation_xyzw": {}, "extra_bone_locals_xyzw": {}, "fk_points_armature": {}}
        if r["gt_legs"] is not None:
            rec["reference_legs_cam"] = {"left": [list(r["gt_legs"]["l"][0]), list(r["gt_legs"]["l"][1])],
                                         "right": [list(r["gt_legs"]["r"][0]), list(r["gt_legs"]["r"][1])]}
        evs = {"FIXED": r["S"].get("B_S0"), "RUNTIME": r["rigE"][0] if r.get("rigE") else None,
               "GT": w95.evaluate(joints3[(r["sid"], r["p"])], r["R_smpl"], r["F"]["A_S0"][0], ctx, r["gt_legs"]) if r["R_smpl"] is not None and r["gt_legs"] is not None else None}
        G = r["G17"]
        for slot, label in slots.items():
            if r["R_smpl"] is None:
                rec["yaw_deg"][slot] = None
                rec["mode_lines"][slot] = ["no SMPL pelvis for this row"]
                s_shin[label].append(None)
                s_plane[label].append(None)
                continue
            Rt, Ro = r["F"]["A_S0"][0], r["R_smpl"]
            y = w94.compat_yaw(Rt)
            rec["yaw_deg"][slot] = math.degrees(y) if y is not None else None
            rec["forward_cam"][slot] = list(Rt[:, 1])
            rec["object_rotation_xyzw"][slot] = list(matrix_to_quaternion_xyzw(conj(alignment, Ro)))
            bones = {"spine_01": list(spine_local_xyzw(alignment, Ro.T @ Rt, B))}
            ev = evs[slot]
            if ev is not None:
                for x in "lr":
                    p = ev["legs"][x]["pose"]
                    bones[f"thigh_{x}"], bones[f"calf_{x}"] = list(p.proximal_local), list(p.distal_local)
                    rec["fk_points_armature"].setdefault(slot, {}).update({f"thigh_{x}": list(p.mid), f"calf_{x}": list(p.end)})
            rec["extra_bone_locals_xyzw"][slot] = bones
            arm = r["arm"].get("S")
            if arm is not None and arm.status == "known":
                Q = conj(alignment, Ro.T @ Rt)
                sp = lambda pt: list(np.asarray(head) + Q @ (np.asarray(pt) - np.asarray(head)))  # noqa: E731
                rec["modes"][slot] = {"proximal_local_xyzw": list(arm.proximal_local), "distal_local_xyzw": list(arm.distal_local),
                                      "mid": sp(arm.mid), "end": sp(arm.end), "root": sp(target.root_head),
                                      "object_rotation_armature_xyzw": rec["object_rotation_xyzw"][slot], "extra_bone_locals_xyzw": {"spine_01": bones["spine_01"]}}
            if ev is None:
                rec["mode_lines"][slot] = ["legs unavailable (S0 2D leg joints invalid) -> previous keys held"]
                s_shin[label].append(None)
                s_plane[label].append(None)
                continue
            L, R_ = ev["legs"]["l"], ev["legs"]["r"]
            pl = []
            for x, (h, k, a) in KNEES.items():
                pp = ev["legs"][x]["pose"]
                if ang(G[k] - G[h], G[a] - G[k]) >= 20 and ev["legs"][x]["knee"] >= 20:
                    pl.append(ang(Ro @ (ctx["Am"].T @ np.cross(pp.upper, pp.forearm)), np.cross(G[k] - G[h], G[a] - G[k])))
                else:
                    pl.append(None)
            rec["mode_lines"][slot] = [f"thigh err L/R {L['thigh_err']:4.0f}/{R_['thigh_err']:4.0f} | shin err {L['shin_err']:4.0f}/{R_['shin_err']:4.0f}",
                                       f"knee flex L/R {L['knee']:4.0f}/{R_['knee']:4.0f} | knee plane err " +
                                       "/".join("-" if v is None else f"{v:.0f}" for v in pl)]
            s_shin[label].append(L["shin_err"])
            s_plane[label].append(pl[0])
        frames.append(rec)
    colors = {v: c for v, c in zip(slots.values(), ("#eb6834", "#2a78d6", "#1baf7a"))}
    return {"case": name, "split": "3dpw", "sequence_id": w[0]["sid"], "start_frame": w[0]["frame"], "fps_sampled": 1.0 / max(w[1]["t"] - w[0]["t"], 1e-6),
            "frames": frames, "labels": slots, "short_labels": slots, "title": f"W97 S0 vs HMR 2.0b legs, same W95 IK {title}",
            "header_note": "root = SMPL pelvis (eval control), S0 thorax, J1X arm, zero twist | thin green = GT legs / GT arm (evaluation only)",
            "plot_series": {"panels": [{"ylabel": "left shin direction error (deg)", "series": s_shin, "colors": colors},
                                       {"ylabel": "left knee plane error (deg)", "series": s_plane, "colors": colors}]}}


def export_cases(pw, ctx, joints3, args, rest, alignment, target):
    w96c = json.loads((args.w96_cases / "cases_3dpw.json").read_text())["cases"]
    dpw = {**pw["validation"], **pw["test"]}
    cases = {}
    for key in ("3dpw_walking_upright", "3dpw_pelvis_stable_bend", "3dpw_seated_or_crouched", "3dpw_worst_hinge"):
        c = w96c[key]
        cases[key] = export_rig_case(c["case"], w94.window_at(dpw, c["sequence_id"], c["start_frame"]), ctx, joints3, "(3DPW)")
    used = {(r["sid"], r["frame"]) for c in w96c.values() for r in w94.window_at(dpw, c["sequence_id"], c["start_frame"])}
    wrong = w94.best_window(dpw, lambda w: -1 if any((r["sid"], r["frame"]) in used for r in w) else
                            sum(any(r["m"]["S"].get(f"knee_plane_{x}", 0) > 45 for x in "lr") for r in w))
    cases["3dpw_s0_wrong_knee_plane"] = export_rig_case("s0_wrong_knee_plane", wrong, ctx, joints3, "(3DPW)")
    common = {"rest_snapshot_digest": rest.digest(), "canonical_to_armature_xyzw": list(alignment), "target_chain": target.to_dict()}
    (args.out_dir / "cases_3dpw.json").write_text(json.dumps({**common, "cases": cases}, indent=1))
    return {k: {"sequence_id": v["sequence_id"], "start_frame": v["start_frame"]} for k, v in cases.items()}


def review_cases(rows_by_seq, args, ctx):
    """Skeleton comparison figures for fixed-rule review rows (typical, best, worst, near-straight, deep flexion,
    S0 wrong plane, E worse than S)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for rs in rows_by_seq.values() for r in rs if r["m"]["E"] is not None]
    d = np.array([r["m"]["E"]["mpjpe"] - r["m"]["S"]["mpjpe"] for r in rows])
    order = np.argsort(d)
    pick = {"E_much_better": [rows[i] for i in order[:3]], "typical": [rows[i] for i in order[len(order) // 2 - 1:len(order) // 2 + 2]],
            "E_much_worse": [rows[i] for i in order[-3:]]}
    flex = [r for r in rows if any(r["m"]["S"].get(f"knee_flex_gt_{s}", 0) >= 90 for s in "lr")]
    pick["deep_knee_flexion"] = sorted(flex, key=lambda r: -max(r["m"]["S"].get(f"knee_flex_gt_{s}", 0) for s in "lr"))[:3]
    straight = [r for r in rows if all(r["m"]["S"].get(f"knee_flex_gt_{s}", 99) < 5 for s in "lr")]
    pick["near_straight"] = straight[:: max(1, len(straight) // 3)][:3]
    wrong = [r for r in rows if any(r["m"]["S"].get(f"knee_plane_{s}", 0) > 90 for s in "lr")]
    pick["s0_backward_knee_plane"] = wrong[:3]
    import cv2
    (args.out_dir / "review_rows.json").write_text(json.dumps({k: [{"sid": r["sid"], "frame": r["frame"], "p": r["p"]} for r in v] for k, v in pick.items()}, indent=1))
    pop = {(r["population"], r["bank_position"]): r for r in json.loads(args.population.read_text())["rows"]}
    for name, rs in pick.items():
        if not rs:
            continue
        fig, axes = plt.subplots(len(rs), 3, figsize=(10, 3.4 * len(rs)), dpi=90, squeeze=False)
        for ax_row, r in zip(axes, rs):
            pr = pop.get(("3dpw_test", r["p"]))
            img = cv2.imread(str(args.image_root_local / pr["image"].split("/", 1)[1])) if pr and args.image_root_local else None
            if img is not None and pr["box"] is not None:
                x1, y1, x2, y2 = (int(v) for v in pr["box"])
                h, w = img.shape[:2]
                crop = img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)][:, :, ::-1]
                ax_row[0].imshow(crop)
            ax_row[0].set_title(f"{r['sid'].split(':')[1]} f{r['frame']}", fontsize=7)
            for ax, (xi, title) in zip(ax_row[1:], ((1, "side view (depth Y x up Z)"), (0, "front view (X x up Z)"))):
                for src, col in (("G17", "#22aa22"), ("S17", "#eb6834"), ("E17", "#2a78d6")):
                    J = r[src]
                    for a, b in [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16)]:
                        ax.plot([J[a, xi], J[b, xi]], [J[a, 2], J[b, 2]], color=col, lw=2 if src == "G17" else 1.4)
                ax.set_aspect("equal")
                ax.set_title(f"{title} | MPJPE S {r['m']['S']['mpjpe']:.0f} E {r['m']['E']['mpjpe']:.0f} mm", fontsize=7)
                ax.tick_params(labelsize=6)
            ax_row[0].set_xticks([])
            ax_row[0].set_yticks([])
        fig.suptitle(f"W97 {name}: green GT (eval only), orange S0, blue HMR 2.0b", fontsize=9)
        fig.tight_layout()
        fig.savefig(args.out_dir / f"review_{name}.png")
        plt.close(fig)


if __name__ == "__main__":
    main()
