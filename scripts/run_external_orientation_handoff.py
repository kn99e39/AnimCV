#!/usr/bin/env python3
"""EXPERIMENT (Worklog 98): can HMR 2.0b's OWN predicted body orientation replace W97's evaluation-only SMPL pelvis?

GT-free adapter (``handoff_frames``; inputs = one frozen W97 HMR 2.0b prediction only):
  pelvis  R_p = C . G . S      G = predicted SMPL global_orient (OpenCV crop-camera axes), C = OpenCV -> AnimCV camera
                               axes (x, z, -y), S = SMPL template axes -> AnimCV body axes (x right, y forward, z up);
                               the same analytic construction as the W94 evaluation reference, no fitting.
  thorax  R_t = W93 thorax frame of HMR 2.0b's own 17 canonical joints (``torso_frames``), i.e. the W94 ownership convention
          (Object = R_p, spine_01 = R_p^T R_t, arm chain in R_t).
Conditions (identical W97 rows, frozen J1X arm, W95 two-bone legs with zero twist Z):
  O    HMR legs, Object = 3DPW SMPL pelvis (EVALUATION ONLY), S0 thorax          (= W97 E+Z rig replay)
  P    HMR legs, Object = HMR pelvis,                            S0 thorax
  T    HMR legs, Object = HMR pelvis,                            HMR thorax      (= T_H)
  T_S  S0 legs,  Object = HMR pelvis,                            HMR thorax      (mixed-owner source-isolation control)
GT (3DPW SMPL pelvis, GT joints, gravity strata) is used only to score.  ``PREDECLARED`` is fixed before results.
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
import eval_external_hmr2_baseline as w97e  # noqa: E402
import run_external_hmr2_baseline as hmr  # noqa: E402
import run_leg_knee_hinge_twist as w96  # noqa: E402
import run_pelvis_frame_leg_articulation as w95  # noqa: E402
import run_pelvis_torso_ownership as w94  # noqa: E402
from run_root_motion_contact_h0_replay import _find_3dpw_pkl, _matched_oracle_sequence, _parse_3dpw_sequence_id  # noqa: E402
from run_runtime_heading_3dhp import shim  # noqa: E402
from run_torso_frame_rig_replay import NAMES, S0_SHA, arm_pose, conj, spine_local_xyzw  # noqa: E402
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

W97_PREDICTIONS_SHA = "fc5d330565205f4792cd3641671633decbab4e24d70522d4dd85f045d87a7003"
W97_POPULATION_SHA = "533895c980a080985444b703f43e0f64046391c76909dea5a2f305a26b34cdb9"
PREDECLARED = {
    "adapter": "pelvis = CV_TO_CANON . global_orient . SMPL_BODY (analytic, identical to the W94 reference construction); thorax = "
               "W93 torso_frames thorax frame of the HMR 17 canonical joints; Object = pelvis, spine_01 = pelvis^T thorax (W94)",
    "invalid": "thorax frame None (degenerate HMR torso) -> condition T/T_S unavailable for the row (counted; never replaced by GT "
               "or by another source)",
    "conditions": ["O", "P", "T", "T_S"],
    "bounds": "W95 PREDECLARED ROM bounds (hip, knee, spine) and W95 severe-jump rule; Object spin = step > 90 deg while the SMPL "
              "reference step < 45 deg; false Object motion = mean Object step while the SMPL reference step < 2 deg",
    "references": "3DPW SMPL pelvis (W94) and GT W93 thorax frame; GT decomposition = W95 evaluate(GT joints, SMPL pelvis, GT "
                  "thorax); evaluation only",
    "negative_case": "upright-walking shin attribution: knee flexion error (rotation invariant), shin error after the best rigid "
                     "rotation of the whole 17-joint skeleton (Kabsch, no scale), share of the shin error in the camera-depth "
                     "plane, back-facing (GT chest normal pointing away from the camera) vs front-facing, crop-ray angle "
                     "(evaluation-only diagnostic from dataset intrinsics; not a candidate)",
}
COND_LABEL = {"O": "O: HMR legs, GT SMPL pelvis (eval), S0 thorax", "P": "P: HMR legs, HMR pelvis, S0 thorax",
              "T": "T: HMR legs, HMR pelvis, HMR thorax", "T_S": "T_S: S0 legs, HMR pelvis, HMR thorax (mixed)"}


# ------------------------------------------------------------------ GT-free adapter

def handoff_frames(global_orient, joints24):
    """(R_pelvis, R_thorax, joints17) in AnimCV canonical axes from ONE frozen HMR 2.0b prediction; R_thorax may be None."""
    R_p = hmr.CV_TO_CANON @ np.asarray(global_orient, float) @ w94.SMPL_BODY
    j17 = hmr.smpl24_to_canonical17(joints24)
    R_t, _ = torso_frames(j17)
    return R_p, R_t, j17


def smpl_chain_thorax(global_orient, body_pose):
    """Diagnostic: SMPL spine3 global frame (pelvis . spine1 . spine2 . spine3 local rotations) in AnimCV body axes."""
    G = np.asarray(global_orient, float) @ body_pose[2] @ body_pose[5] @ body_pose[8]
    return hmr.CV_TO_CANON @ G @ w94.SMPL_BODY


# ------------------------------------------------------------------ evaluation helpers

def axis_err(Ra, Rb, i):
    return w97e.ang(Ra[:, i], Rb[:, i])


def cond_eval(r, legs_j, Ro, Rt, ctx, smpl_legs, arm):
    ev = w95.evaluate(legs_j, Ro, Rt, ctx, r["gt_legs"])
    if ev is None:
        return None
    m = w96.process(r, ev, Ro, ctx, smpl_legs, ("Z",))
    out = {"Ro": Ro, "Rt": Rt, "ev": ev, "m": m, "arm": arm}
    if arm is not None and arm.status == "known":
        out["arm_world"] = conj(ctx["alignment"], Ro) @ conj(ctx["alignment"], Ro.T @ Rt) @ np.asarray(arm.upper)
    return out


def row_conditions(r, ctx, smpl_legs, j1x):
    """All four conditions for one 3DPW/3DHP row (None where unavailable)."""
    out = {}
    e = r.get("E")
    if e is None or r["gt_legs"] is None:
        return out
    Rts = r["F"]["A_S0"][0]
    a = ctx["alignment"]
    armS = r["arm"].get("S")
    if r.get("R_smpl") is not None:
        out["O"] = cond_eval(r, e["j17"], r["R_smpl"], Rts, ctx, smpl_legs, armS)
    out["P"] = cond_eval(r, e["j17"], e["R_p"], Rts, ctx, smpl_legs, armS)
    if e["R_t"] is not None:
        armT = arm_pose(j1x[0][r["p"]], j1x[1][r["p"]], e["R_t"], a, ctx["target"], ctx["rest"]) if r["available"] else None
        out["T"] = cond_eval(r, e["j17"], e["R_p"], e["R_t"], ctx, smpl_legs, armT)
        if r["s0_legs_available"]:
            out["T_S"] = cond_eval(r, r["S17"], e["R_p"], e["R_t"], ctx, smpl_legs, armT)
    return out


def metrics_row(r, c, gt_ev, Rt_gt):
    ev, m = c["ev"], c["m"]
    Ro, Rt = c["Ro"], c["Rt"]
    o = {"spine_flex": ev["spine_flex"], "spine_lat": ev["spine_lat"], "spine_twist": ev["spine_twist"],
         "spine_rom": float(any(ev["viol"][k] for k in ("spine_flexion", "spine_extension", "spine_lateral", "spine_axial"))),
         "hip_rom": float(any(ev["viol"][k] for k in ("hip_flexion", "hip_extension", "hip_abduction", "hip_adduction"))),
         "w95_rom_only": float(any(v for k, v in ev["viol"].items() if k != "knee_hinge")),
         "thigh_dir": np.mean([ev["legs"][x]["thigh_err"] for x in "lr"]), "shin_dir": np.mean([ev["legs"][x]["shin_err"] for x in "lr"]),
         "off_hinge": np.mean([abs(m[x]["Z"]["off_hinge"]) for x in "lr"]),
         "thorax_vs_gt": geodesic_degrees(Rt, Rt_gt) if Rt_gt is not None else None}
    re = [abs(m[x]["Z"]["smpl_roll_err"]) for x in "lr" if "smpl_roll_err" in m[x]["Z"]]
    o["roll_vs_smpl"] = float(np.mean(re)) if re else None
    if r.get("R_smpl") is not None:
        o["pelvis_geo"], o["pelvis_up"], o["pelvis_fwd"] = geodesic_degrees(Ro, r["R_smpl"]), axis_err(Ro, r["R_smpl"], 2), axis_err(Ro, r["R_smpl"], 1)
        if Rt_gt is not None:
            o["spine_rel_vs_gt"] = geodesic_degrees(Ro.T @ Rt, r["R_smpl"].T @ Rt_gt)
    if gt_ev is not None:
        o["hip_flex_err"] = np.mean([abs(ev["legs"][x]["hip_flex"] - gt_ev["legs"][x]["hip_flex"]) for x in "lr"])
        o["hip_abd_err"] = np.mean([abs(ev["legs"][x]["hip_abd"] - gt_ev["legs"][x]["hip_abd"]) for x in "lr"])
        o["spine_flex_err"] = abs(ev["spine_flex"] - gt_ev["spine_flex"])
        o["spine_twist_err"] = abs(ev["spine_twist"] - gt_ev["spine_twist"])
    G = r["G17"]
    kp = []
    for x, (h, k, a_) in w97e.KNEES.items():
        p = ev["legs"][x]["pose"]
        if w97e.ang(G[k] - G[h], G[a_] - G[k]) >= 20 and ev["legs"][x]["knee"] >= 20:
            kp.append(w97e.ang(Ro @ (r["Am"].T @ np.cross(p.upper, p.forearm)), np.cross(G[k] - G[h], G[a_] - G[k])))
    o["knee_plane_rig"] = float(np.mean(kp)) if kp else None
    return o


def summarize(rows, conds, key_pairs):
    out = {}
    for cnd in conds:
        vals = defaultdict(list)
        for r in rows:
            mm = r["H_m"].get(cnd)
            if mm is None:
                continue
            for k, v in mm.items():
                if v is not None:
                    vals[k].append(v)
        out[cnd] = {"rows": int(sum(r["H_m"].get(cnd) is not None for r in rows)),
                    **{k: {"mean": float(np.mean(v)), "median": float(np.median(v)), "p95": float(np.percentile(v, 95))} for k, v in vals.items()}}
    for a, b in key_pairs:
        both = [r for r in rows if r["H_m"].get(a) is not None and r["H_m"].get(b) is not None]
        res = {"rows": len(both)}
        for k in ("pelvis_geo", "pelvis_up", "pelvis_fwd", "thorax_vs_gt", "spine_rel_vs_gt", "spine_rom", "hip_rom", "w95_rom_only",
                  "hip_flex_err", "hip_abd_err", "spine_flex_err", "spine_twist_err", "thigh_dir", "shin_dir", "knee_plane_rig", "off_hinge", "roll_vs_smpl"):
            pr = [(r["sid"], r["H_m"][a][k], r["H_m"][b][k]) for r in both if r["H_m"][a].get(k) is not None and r["H_m"][b].get(k) is not None]
            if pr:
                c = _brief(np.array([p[0] for p in pr]), np.array([p[2] - p[1] for p in pr]))
                res[k] = {"n": len(pr), a: float(np.mean([p[1] for p in pr])), b: float(np.mean([p[2] for p in pr])),
                          "diff": c.get("mean"), "ci95": c.get("ci95")}
        out[f"{b}_minus_{a}"] = res
    return out


def invariance(rows, ctx):
    leg = arm = 0.0
    for r in rows:
        cs = [r["H"].get(k) for k in ("O", "P", "T") if r["H"].get(k) is not None]
        if len(cs) < 2:
            continue
        for x in "lr":
            ws = [c["ev"]["legs"][x]["thigh_w"] for c in cs]
            leg = max(leg, max(float(np.abs(w - ws[0]).max()) for w in ws))
        aw = [c["arm_world"] for c in cs if "arm_world" in c]
        if len(aw) > 1:
            arm = max(arm, max(float(np.abs(w - aw[0]).max()) for w in aw))
    return {"world_thigh_max_diff_O_P_T": leg, "world_upper_arm_max_diff_O_P_T": arm}


def temporal(seq_rows, conds):
    out = {}
    for cnd in conds:
        spins = steps = 0
        still = []
        spine_jumps = hip_jumps = 0
        for rs in seq_rows.values():
            for a, b in zip(rs[:-1], rs[1:]):
                ca, cb = a["H"].get(cnd), b["H"].get(cnd)
                if ca is None or cb is None:
                    continue
                steps += 1
                st = geodesic_degrees(ca["Ro"], cb["Ro"])
                if a.get("R_smpl") is not None and b.get("R_smpl") is not None:
                    gs = geodesic_degrees(a["R_smpl"], b["R_smpl"])
                    spins += int(st > 90 and gs < 45)
                    if gs < 2.0:
                        still.append(st)
                    sa, sb = ca["Ro"].T @ ca["Rt"], cb["Ro"].T @ cb["Rt"]
                    if a.get("Rt_gt") is not None and b.get("Rt_gt") is not None:
                        ga, gb = a["R_smpl"].T @ a["Rt_gt"], b["R_smpl"].T @ b["Rt_gt"]
                        spine_jumps += int(geodesic_degrees(sa, sb) > 90 and geodesic_degrees(ga, gb) < 45)
                for x in "lr":
                    if (quaternion_angle_degrees(ca["ev"]["legs"][x]["pose"].proximal_local, cb["ev"]["legs"][x]["pose"].proximal_local) > 90
                            and w95.angle(a["gt_legs"][x][0], b["gt_legs"][x][0]) < 45):
                        hip_jumps += 1
        out[cnd] = {"steps": steps, "object_spins": spins, "spine_jumps": spine_jumps, "severe_hip_jumps": hip_jumps,
                    "false_object_motion_when_smpl_still_deg": float(np.mean(still)) if still else None, "still_steps": len(still)}
    return out


# ------------------------------------------------------------------ negative case: upright shins

def kabsch_rot(P, G):
    U, _, Vt = np.linalg.svd((P - P.mean(0)).T @ (G - G.mean(0)))
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


def shin_attribution(rows, intrinsics_fn):
    out = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.get("E") is None or r.get("stratum") != "upright":
            continue
        G = r["G17"]
        Rt_gt = r.get("Rt_gt")
        facing = "back_facing" if (Rt_gt is not None and Rt_gt[1, 1] > 0.5) else ("front_facing" if (Rt_gt is not None and Rt_gt[1, 1] < -0.5) else "side")
        ray = intrinsics_fn(r)
        for src, J in (("S", r["S17"]), ("E", r["E"]["j17"])):
            R = kabsch_rot(J, G)
            Ja = (J - J.mean(0)) @ R + G.mean(0)
            for x, (h, k, a_) in w97e.KNEES.items():
                gs, js = G[a_] - G[k], J[a_] - J[k]
                e = w97e.ang(js, gs)
                fe = abs(w97e.ang(J[k] - J[h], js) - w97e.ang(G[k] - G[h], gs))
                ea = w97e.ang(Ja[a_] - Ja[k], gs)
                # share of the shin error in the camera depth (Y) direction: angle between projections onto the image plane (X,Z)
                ip = w97e.ang(js[[0, 2]].tolist() + [0.0], gs[[0, 2]].tolist() + [0.0]) if np.linalg.norm(js[[0, 2]]) > 1e-6 else 0.0
                for grp in ("all", facing):
                    out[(src, grp)]["shin_err"].append(e)
                    out[(src, grp)]["knee_flex_err"].append(fe)
                    out[(src, grp)]["shin_err_after_rigid_rotation"].append(ea)
                    out[(src, grp)]["shin_err_image_plane_only"].append(ip)
                    if ray is not None:
                        out[(src, grp)]["ray"].append(ray)
                        out[(src, grp)]["shin_err_for_ray"].append(e)
    res = {}
    for (src, grp), d in out.items():
        rr = {k: float(np.mean(v)) for k, v in d.items() if k not in ("ray", "shin_err_for_ray")}
        rr["n_legs"] = len(d["shin_err"])
        if len(d.get("ray", [])) > 10:
            rr["spearman_shin_err_vs_crop_ray_angle"] = w97e_spearman(d["ray"], d["shin_err_for_ray"])
        res[f"{src}_{grp}"] = rr
    return res


def w97e_spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))[0, 1])


# ------------------------------------------------------------------ main

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bank", "s0_dir", "raw_3dpw", "w87_3dpw", "hp_bank", "hp_s0", "w87_3dhp", "w93_cases", "w96_cases", "w97_eval", "population",
                 "predictions", "rest", "calibration", "chains", "out_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
    frozen = {args.w87_3dpw: W87_PREDICTIONS["3dpw"], args.w87_3dhp: W87_PREDICTIONS["3dhp"], args.hp_s0: S0_SHA["3dhp"],
              args.predictions: W97_PREDICTIONS_SHA, args.population: W97_POPULATION_SHA,
              **{args.s0_dir / f"prediction_{s}.npy": S0_SHA[s] for s in ("validation", "test")}}
    for path, digest in frozen.items():
        if sha(path) != digest:
            raise SystemExit(f"STOP: frozen input changed: {path}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pred = np.load(args.predictions)
    E = {}
    for i, (pp, bp) in enumerate(zip(pred["population"], pred["index"][0])):
        if pred["valid"][i]:
            R_p, R_t, j17 = handoff_frames(pred["global_orient"][i], pred["joints24"][i])
            E[(str(pp), int(bp))] = {"R_p": R_p, "R_t": R_t, "j17": j17, "R_chain": smpl_chain_thorax(pred["global_orient"][i], pred["body_pose"][i])}
    pop = {(r["population"], r["bank_position"]): r for r in json.loads(args.population.read_text())["rows"]}

    rest = load_target_rest_pose(args.rest)
    alignment = derive_rig_alignment(rest, load_rig_fk_calibration(args.calibration))
    Am = quaternion_to_matrix(alignment)
    target = target_chain_geometry(rest, load_ik_chain_set(args.chains).chains[0])
    sb = rest.bones["spine_01"]
    M = np.asarray(sb.matrix_local, float)[:3, :3]
    lf = w96.leg_frames(rest)
    ctx = {"alignment": alignment, "Am": Am, "rest": rest, "target": target, "spine_basis": M / np.linalg.norm(M, axis=0, keepdims=True),
           "spine_head": list(sb.head), "legs": w95.leg_targets(rest), "lf": lf, "psi0": w96.rest_offsets(lf, Am)}
    report = {"schema": "worklog98_external_orientation_handoff", "predeclared": PREDECLARED, "frozen_inputs": {str(k): v for k, v in frozen.items()}}
    # ---- 3DPW rows (W94/W95/W96 builders, W97 E attached)
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
    j1x3 = (p3["left_arm_g1_elbow"], p3["left_arm_g1_wrist"])
    iv3 = bank.arrays["input_valid"]
    intr = {}

    def ray_angle(r):  # evaluation-only diagnostic (dataset intrinsics); never used by a candidate
        name, _ = _parse_3dpw_sequence_id(r["sid"])
        if name not in intr:
            import pickle
            with _find_3dpw_pkl(args.raw_3dpw, name).open("rb") as h:
                intr[name] = np.asarray(pickle.load(h, encoding="latin1")["cam_intrinsics"], float)
        pr = pop.get((r["split_pop"], r["p"]))
        if pr is None or pr["box"] is None:
            return None
        x1, y1, x2, y2 = pr["box"]
        d = np.linalg.inv(intr[name]) @ np.array([(x1 + x2) / 2, (y1 + y2) / 2, 1.0])
        return w97e.ang(d, (0.0, 0.0, 1.0))
    pw, results = {}, {}
    for split in ("validation", "test"):
        pw[split] = w94.build_rows(bank, s0, sequence_ids_in_split(bank, split), g3, bank.arrays, p3, ctx3, smpl=smpl)
        w95.attach(pw[split], joints3, s0, lambda r: r["R_smpl"] is not None, lambda p: bool(iv3[p][[1, 2, 3, 4, 5, 6]].all()), ctx)
        rows = [r for rs in pw[split].values() for r in rs]
        for r in rows:
            G = joints3[(r["sid"], r["p"])]
            r["G17"], r["S17"], r["Am"], r["split_pop"] = G - G[0], s0[r["p"]] - s0[r["p"]][0], Am, f"3dpw_{split}"
            r["E"] = E.get((f"3dpw_{split}", r["p"]))
            r["Rt_gt"] = torso_frames(r["G17"])[0]
            r["m"] = {"S": w97e.source_metrics(r["S17"], r["G17"], r["R_smpl"])}  # W97 challenge-subset labels (evaluation only)
            smpl_legs = sl.get(r["sid"], r["frame"]) if r["R_smpl"] is not None else None
            r["H"] = row_conditions(r, ctx, smpl_legs, j1x3)
            gt_ev = w95.evaluate(G, r["R_smpl"], r["Rt_gt"], ctx, r["gt_legs"]) if (r["R_smpl"] is not None and r["Rt_gt"] is not None and r["gt_legs"] is not None) else None
            r["H_m"] = {k: (metrics_row(r, c, gt_ev, r["Rt_gt"]) if c is not None else None) for k, c in r["H"].items()}
            if r["E"] is not None:
                r["chain_vs_joints_thorax"] = geodesic_degrees(r["E"]["R_chain"], r["E"]["R_t"]) if r["E"]["R_t"] is not None else None
        masks = w97e.challenge_masks(rows)
        res = {"coverage": {"rows": len(rows), "E": sum(r["E"] is not None for r in rows),
                            **{c: sum(r["H"].get(c) is not None for r in rows) for c in ("O", "P", "T", "T_S")},
                            "hmr_thorax_invalid": sum(r["E"] is not None and r["E"]["R_t"] is None for r in rows)}}
        pairs = (("O", "P"), ("P", "T"), ("O", "T"), ("T", "T_S"))
        res["all"] = summarize(rows, ("O", "P", "T", "T_S"), pairs)
        res["by_stratum"] = {s: summarize([r for r in rows if r.get("stratum") == s], ("O", "P", "T", "T_S"), pairs) for s in w94.STRATA}
        res["challenge"] = {k: summarize([r for r in rows if f(r)], ("O", "P", "T", "T_S"), pairs) for k, f in masks.items()}
        res["invariance"] = invariance(rows, ctx)
        res["temporal"] = temporal(pw[split], ("O", "P", "T", "T_S"))
        cv = [r["chain_vs_joints_thorax"] for r in rows if r.get("chain_vs_joints_thorax") is not None]
        res["hmr_smpl_chain_thorax_vs_joint_thorax_deg"] = {"mean": float(np.mean(cv)), "median": float(np.median(cv)), "p95": float(np.percentile(cv, 95))}
        res["upright_shin_attribution"] = shin_attribution(rows, ray_angle)
        pel = [(ray_angle(r), r["H_m"]["P"]["pelvis_geo"]) for r in rows if r["H_m"].get("P") and r["H_m"]["P"].get("pelvis_geo") is not None]
        pel = [p for p in pel if p[0] is not None]
        res["pelvis_error_vs_crop_ray_angle_spearman"] = w97e_spearman([p[0] for p in pel], [p[1] for p in pel]) if pel else None
        results[f"3dpw_{split}"] = res
    # ---- 3DHP (no SMPL reference): coherence-only metrics and Blender cases
    mx = ExternalBank.load(args.hp_bank)
    rv = shim(mx, mx.arrays["input_2d"], mx.arrays["input_valid"])
    for s, ext in zip(rv.samples, mx.samples):
        s.sample_id, s.image_reference = ext.sample_id, ext.image_reference
    g2 = np.concatenate([np.zeros_like(mx.arrays["input_2d"][..., :2]), np.ones_like(mx.arrays["input_2d"][..., 2:])], axis=-1)
    gv = shim(mx, g2, mx.arrays["target_valid"])
    ctxh = {**ctx, "chain_gt": chain_targets(mx.arrays["target_3d"], mx.arrays["target_valid"], w94.CHAIN), "time": lambda v, p: v.samples[p].timestamp}
    jh = {}

    def gh(sid, pos):
        for p in pos:
            jh[(sid, p)] = mx.arrays["target_3d"][p]
        return [o.yaw_radians for o in observe_yaw(build_h0_lifted_sequence(gv, mx.arrays["target_3d"], sid))], [mx.arrays["target_3d"][p] for p in pos]
    hp = np.load(args.hp_s0)
    ph = np.load(args.w87_3dhp)
    hrows = w94.build_rows(rv, hp, sorted({s.sequence_id for s in mx.samples}), gh, mx.arrays, ph, ctxh,
                           gravity_fn=lambda sid: w94.hp_gravity(sid.split(":")[1]))
    tv, ivh = mx.arrays["target_valid"], mx.arrays["input_valid"]
    w95.attach(hrows, jh, hp, lambda r: bool(tv[r["p"]][[1, 2, 3, 4, 5, 6]].all()), lambda p: bool(ivh[p][[1, 2, 3, 4, 5, 6]].all()), ctx)
    hl = [r for rs in hrows.values() for r in rs]
    for r in hl:
        G = jh[(r["sid"], r["p"])]
        r["G17"], r["S17"], r["Am"] = G - G[0], hp[r["p"]] - hp[r["p"]][0], Am
        r["E"] = E.get(("3dhp", r["p"]))
        r["Rt_gt"] = torso_frames(r["G17"])[0]
        r["H"] = row_conditions(r, ctx, None, (ph["left_arm_g1_elbow"], ph["left_arm_g1_wrist"]))
        r["H_m"] = {k: (metrics_row(r, c, None, r["Rt_gt"]) if c is not None else None) for k, c in r["H"].items()}
        if r["E"] is not None and r.get("g") is not None:
            r["H_m_extra"] = {"pelvis_tilt_g": w97e.ang(r["E"]["R_p"][:, 2], r["g"]), "gt_trunk_tilt_g": r["trunk_g"]}
    results["3dhp"] = {"coverage": {"rows": len(hl), **{c: sum(r["H"].get(c) is not None for r in hl) for c in ("P", "T", "T_S")}},
                       "all": summarize(hl, ("P", "T", "T_S"), (("P", "T"), ("T", "T_S"))),
                       "by_stratum": {s: summarize([r for r in hl if r.get("stratum") == s], ("P", "T", "T_S"), (("P", "T"), ("T", "T_S"))) for s in w94.STRATA},
                       "temporal": temporal(hrows, ("P", "T", "T_S")), "invariance": invariance(hl, ctx)}
    lying = [r for r in hl if r.get("stratum") == "lying" and r.get("H_m_extra")]
    results["3dhp"]["lying_hmr_pelvis_tilt_vs_gt_trunk_tilt"] = {"rows": len(lying),
        "hmr_pelvis_tilt_g_mean": float(np.mean([r["H_m_extra"]["pelvis_tilt_g"] for r in lying])) if lying else None,
        "gt_trunk_tilt_g_mean": float(np.mean([r["H_m_extra"]["gt_trunk_tilt_g"] for r in lying])) if lying else None}
    report["results"] = results
    report["w97_oracle_reproduction"] = w97_reproduction(pw["test"], args)
    report["cases"] = export_cases(pw, hrows, ctx, args, rest, alignment, target)
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps(results["3dpw_test"]["coverage"]), json.dumps(report["w97_oracle_reproduction"]))


def w97_reproduction(seq_rows, args):
    """Condition O must equal W97's E+Z rig replay (same rows): compare the per-split means W97 reported."""
    w97 = json.loads((args.w97_eval / "report.json").read_text())["rig_replay"]["3dpw_test"]["all"]
    rows = [r for rs in seq_rows.values() for r in rs if r["H_m"].get("O") is not None and r["s0_legs_available"]]
    return {"rows": len(rows), "W97_E_thigh_dir": w97["thigh_dir"]["E_mean"], "O_thigh_dir": float(np.mean([r["H_m"]["O"]["thigh_dir"] for r in rows])),
            "W97_E_shin_dir": w97["shin_dir"]["E_mean"], "O_shin_dir": float(np.mean([r["H_m"]["O"]["shin_dir"] for r in rows])),
            "W97_E_roll_vs_smpl": w97["roll_vs_smpl"]["E_mean"], "O_roll_vs_smpl": float(np.mean([r["H_m"]["O"]["roll_vs_smpl"] for r in rows]))}


# ------------------------------------------------------------------ Blender cases

def export_case(name, w, slots, ctx, title, note):
    alignment, B, head, target = ctx["alignment"], ctx["spine_basis"], ctx["spine_head"], ctx["target"]
    frames = []
    s_pel, s_spine = {COND_LABEL[c]: [] for c in slots.values()}, {COND_LABEL[c]: [] for c in slots.values()}
    for r in w:
        rec = {"bank_position": r["p"], "frame_index": r["frame"], "time_s": r["t"], "image": r["image"], "available": r["available"],
               "gt_valid": r["gt_arm_ok"], "runtime_status": "KNOWN", "runtime_reasons": [], "yaw_deg": {}, "modes": {}, "forward_cam": {},
               "mode_lines": {}, "object_rotation_xyzw": {}, "extra_bone_locals_xyzw": {}, "fk_points_armature": {}}
        if r["gt_legs"] is not None:
            rec["reference_legs_cam"] = {"left": [list(r["gt_legs"]["l"][0]), list(r["gt_legs"]["l"][1])],
                                         "right": [list(r["gt_legs"]["r"][0]), list(r["gt_legs"]["r"][1])]}
        for slot, cnd in slots.items():
            lab = COND_LABEL[cnd]
            c = r["H"].get(cnd)
            if c is None:
                rec["yaw_deg"][slot] = None
                rec["mode_lines"][slot] = ["condition unavailable for this row (no GT/HMR fallback)"]
                s_pel[lab].append(None)
                s_spine[lab].append(None)
                continue
            Ro, Rt, ev = c["Ro"], c["Rt"], c["ev"]
            y = w94.compat_yaw(Rt)
            rec["yaw_deg"][slot] = math.degrees(y) if y is not None else None
            rec["forward_cam"][slot] = list(Rt[:, 1])
            rec["object_rotation_xyzw"][slot] = list(matrix_to_quaternion_xyzw(conj(alignment, Ro)))
            bones = {"spine_01": list(spine_local_xyzw(alignment, Ro.T @ Rt, B))}
            for x in "lr":
                p = ev["legs"][x]["pose"]
                bones[f"thigh_{x}"], bones[f"calf_{x}"] = list(p.proximal_local), list(p.distal_local)
                rec["fk_points_armature"].setdefault(slot, {}).update({f"thigh_{x}": list(p.mid), f"calf_{x}": list(p.end)})
            rec["extra_bone_locals_xyzw"][slot] = bones
            arm = c["arm"]
            if arm is not None and arm.status == "known":
                Q = conj(alignment, Ro.T @ Rt)
                sp = lambda pt: list(np.asarray(head) + Q @ (np.asarray(pt) - np.asarray(head)))  # noqa: E731
                rec["modes"][slot] = {"proximal_local_xyzw": list(arm.proximal_local), "distal_local_xyzw": list(arm.distal_local),
                                      "mid": sp(arm.mid), "end": sp(arm.end), "root": sp(target.root_head),
                                      "object_rotation_armature_xyzw": rec["object_rotation_xyzw"][slot], "extra_bone_locals_xyzw": {"spine_01": bones["spine_01"]}}
            mm = r["H_m"][cnd]
            lines = [f"spine flex/lat/twist {ev['spine_flex']:4.0f}/{ev['spine_lat']:4.0f}/{ev['spine_twist']:4.0f}",
                     f"hip flex L/R {ev['legs']['l']['hip_flex']:4.0f}/{ev['legs']['r']['hip_flex']:4.0f} | ROM " +
                     ("ok" if not mm["w95_rom_only"] else ",".join(sorted(k for k, v in ev["viol"].items() if v and k != "knee_hinge")))]
            if mm.get("pelvis_geo") is not None:
                lines.append(f"pelvis vs SMPL {mm['pelvis_geo']:4.0f} deg | spine rel vs GT {mm.get('spine_rel_vs_gt', float('nan')):4.0f}")
            rec["mode_lines"][slot] = lines
            s_pel[lab].append(mm.get("pelvis_geo") if mm.get("pelvis_geo") is not None else w97e.ang(Ro[:, 2], (0, 0, 1)))
            s_spine[lab].append(ev["spine_flex"])
        frames.append(rec)
    colors = {COND_LABEL[c]: col for c, col in zip(slots.values(), ("#1baf7a", "#eb6834", "#2a78d6"))}
    has_smpl = any(r.get("R_smpl") is not None for r in w)
    return {"case": name, "split": "3dpw" if has_smpl else "3dhp", "sequence_id": w[0]["sid"], "start_frame": w[0]["frame"],
            "fps_sampled": 1.0 / max(w[1]["t"] - w[0]["t"], 1e-6), "frames": frames, "labels": {s: COND_LABEL[c] for s, c in slots.items()},
            "short_labels": {s: c for s, c in slots.items()}, "title": f"W98 GT-free body orientation handoff {title}", "header_note": note,
            "plot_series": {"panels": [{"ylabel": "Object vs SMPL pelvis (deg)" if has_smpl else "Object tilt from camera up (deg)", "series": s_pel, "colors": colors},
                                       {"ylabel": "spine_01 flexion (deg)", "series": s_spine, "colors": colors}]}}


def export_cases(pw, hrows, ctx, args, rest, alignment, target):
    w96c = {**json.loads((args.w96_cases / "cases_3dpw.json").read_text())["cases"], **json.loads((args.w96_cases / "cases_3dhp.json").read_text())["cases"]}
    w97c = json.loads((args.w97_eval / "cases_3dpw.json").read_text())["cases"]
    dpw = {**pw["validation"], **pw["test"]}
    note = "J1X arm, W95 zero-twist legs, identical frames | thin green = GT legs / GT arm (evaluation only)"
    cases = {}
    slots = {"FIXED": "O", "RUNTIME": "P", "GT": "T"}
    for key, src in (("3dpw_walking_upright", w96c), ("3dpw_turn_180", w96c), ("3dpw_pelvis_stable_bend", w96c), ("3dpw_seated_or_crouched", w96c),
                     ("3dpw_s0_wrong_knee_plane", w97c)):
        c = src[key]
        cases[key] = export_case(c["case"], w94.window_at(dpw, c["sequence_id"], c["start_frame"]), slots, ctx, "(3DPW)", note)
    used = {(r["sid"], r["frame"]) for k in cases for r in w94.window_at(dpw, cases[k]["sequence_id"], cases[k]["start_frame"])}

    def worst_pelvis(w):
        if any((r["sid"], r["frame"]) in used for r in w):
            return -1
        return sum((r["H_m"].get("P") or {}).get("pelvis_geo", 0) or 0 for r in w)
    cases["3dpw_worst_hmr_pelvis"] = export_case("worst_hmr_pelvis", w94.best_window(dpw, worst_pelvis), slots, ctx, "(3DPW)", note)
    hslots = {"FIXED": "P", "RUNTIME": "T", "GT": "T_S"}
    for key in ("3dhp_ts2_supine", "3dhp_ts4_lying"):
        c = w96c[key]
        cases[key] = export_case(c["case"], w94.window_at(hrows, c["sequence_id"], c["start_frame"]), hslots, ctx, "(3DHP, no SMPL reference)", note)
    common = {"rest_snapshot_digest": rest.digest(), "canonical_to_armature_xyzw": list(alignment), "target_chain": target.to_dict()}
    for tag in ("3dpw", "3dhp"):
        (args.out_dir / f"cases_{tag}.json").write_text(json.dumps({**common, "cases": {k: v for k, v in cases.items() if k.startswith(tag)}}, indent=1))
    return {k: {"sequence_id": v["sequence_id"], "start_frame": v["start_frame"], "slots": v["labels"]} for k, v in cases.items()}


# ------------------------------------------------------------------ pelvis error structure (evaluation-only attribution)

def chordal_mean(Rs):
    U, _, Vt = np.linalg.svd(np.mean(Rs, axis=0))
    return U @ np.diag([1.0, 1.0, np.linalg.det(U @ Vt)]) @ Vt


def ray_rotation(d):
    """Minimal rotation taking the optical axis (0, 0, 1) onto the unit ray d (OpenCV camera axes)."""
    d = np.asarray(d, float) / np.linalg.norm(d)
    z = np.array([0.0, 0.0, 1.0])
    v, c = np.cross(z, d), float(z @ d)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1.0 + c)


def pelvis_error_structure(rows):
    """rows: (split, R_hmr, R_smpl, R_ray-corrected HMR).  Is the HMR pelvis error a constant basis offset or per-frame?

    Body-side offset  E_b = R_smpl^T R_hmr  (a wrong SMPL->body basis would make it constant);
    camera-side offset E_c = R_hmr R_smpl^T (a wrong camera conversion would make it constant).
    The mean offsets are estimated on VALIDATION and removed on TEST (diagnostic only; never a candidate).
    The crop-ray variant re-expresses the crop-camera orientation in the full camera with dataset intrinsics (diagnostic)."""
    out = {}
    by = defaultdict(list)
    for split, Rh, Rs, Rr in rows:
        by[split].append((Rh, Rs, Rr))
    val = by["3dpw_validation"]
    Eb = chordal_mean([Rs.T @ Rh for Rh, Rs, _ in val])
    Ec = chordal_mean([Rh @ Rs.T for Rh, Rs, _ in val])
    for split, rs in by.items():
        geo = [geodesic_degrees(Rh, Rs) for Rh, Rs, _ in rs]
        up = [w97e.ang(Rh[:, 2], Rs[:, 2]) for Rh, Rs, _ in rs]
        out[split] = {
            "rows": len(rs), "geodesic_mean": float(np.mean(geo)), "geodesic_median": float(np.median(geo)), "up_mean": float(np.mean(up)),
            "mean_body_offset_deg": geodesic_degrees(chordal_mean([Rs.T @ Rh for Rh, Rs, _ in rs]), np.eye(3)),
            "mean_camera_offset_deg": geodesic_degrees(chordal_mean([Rh @ Rs.T for Rh, Rs, _ in rs]), np.eye(3)),
            "geodesic_after_val_body_offset_removed": float(np.mean([geodesic_degrees(Rh @ Eb.T, Rs) for Rh, Rs, _ in rs])),
            "geodesic_after_val_camera_offset_removed": float(np.mean([geodesic_degrees(Ec.T @ Rh, Rs) for Rh, Rs, _ in rs])),
            "geodesic_crop_ray_corrected": float(np.mean([geodesic_degrees(Rr, Rs) for _, Rs, Rr in rs if Rr is not None])),
            "up_crop_ray_corrected": float(np.mean([w97e.ang(Rr[:, 2], Rs[:, 2]) for _, Rs, Rr in rs if Rr is not None])),
            "p95": float(np.percentile(geo, 95)), "share_gt_30deg": float(np.mean(np.asarray(geo) > 30))}
    out["validation_offsets_deg"] = {"body": geodesic_degrees(Eb, np.eye(3)), "camera": geodesic_degrees(Ec, np.eye(3))}
    return out


def main_structure(argv):
    import pickle
    parser = argparse.ArgumentParser(description="W98 pelvis error structure (evaluation only)")
    for name in ("population", "predictions", "raw_3dpw", "out"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args(argv)
    for path, digest in ((args.predictions, W97_PREDICTIONS_SHA), (args.population, W97_POPULATION_SHA)):
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise SystemExit(f"STOP: frozen input changed: {path}")
    pop = {(r["population"], r["bank_position"]): r for r in json.loads(args.population.read_text())["rows"]}
    pred = np.load(args.predictions)
    smpl, intr, rows = w94.Smpl3dpw(args.raw_3dpw), {}, []
    for i, (pp, bp) in enumerate(zip(pred["population"], pred["index"][0])):
        pp, bp = str(pp), int(bp)
        if not pred["valid"][i] or not pp.startswith("3dpw"):
            continue
        pr = pop[(pp, bp)]
        R_s, _ = smpl.get(pr["sequence_id"], pr["frame_index"])
        if R_s is None:
            continue
        go = np.asarray(pred["global_orient"][i], float)
        R_h, _, _ = handoff_frames(go, pred["joints24"][i])
        name, _ = _parse_3dpw_sequence_id(pr["sequence_id"])
        if name not in intr:
            with _find_3dpw_pkl(args.raw_3dpw, name).open("rb") as h:
                intr[name] = np.asarray(pickle.load(h, encoding="latin1")["cam_intrinsics"], float)
        R_r = None
        if pr["box"] is not None:
            x1, y1, x2, y2 = pr["box"]
            d = np.linalg.inv(intr[name]) @ np.array([(x1 + x2) / 2, (y1 + y2) / 2, 1.0])
            R_r = hmr.CV_TO_CANON @ ray_rotation(d) @ go @ w94.SMPL_BODY
        rows.append((pp, R_h, R_s, R_r))
    res = pelvis_error_structure(rows)
    args.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "pelvis-structure":
        main_structure(sys.argv[2:])
    else:
        main()
