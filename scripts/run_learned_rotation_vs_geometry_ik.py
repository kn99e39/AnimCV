#!/usr/bin/env python3
"""EXPERIMENT (Worklog 99): does HMR 2.0b's predicted SMPL hip rotation add thigh axial roll beyond its 3D joint centres?

Body frame = W98 condition T (GT-free): Object = HMR pelvis, spine_01 from the HMR joint thorax, J1X arm, W95 two-bone legs
on the HMR 17 joints.  Only thigh axial-roll ownership changes:
  Z   W95 zero-twist leg IK (thigh delta = shortest arc from the rest thigh direction in the Object frame)        = W98 T legs
  R   the same swing, rolled about the posed thigh axis u by phi_R so that the rig hinge h = delta n_rest matches the
      HMR-predicted SMPL thigh hinge axis a = C . global_orient . body_pose[hip] . e_x (template +x), minus the W96
      analytic rest offset psi0 (identity SMPL pose vs BaseRig rest hinge; no data fitting); the calf takes the shortest
      arc from its rolled rest direction to the same target direction (dependent transform; W96 ``rolled``).
  O   evaluation-only GT-roll oracle (W96 ``O``: the same construction with the 3DPW SMPL thigh axis), Blender reference.
World thigh/shin directions, bone lengths, endpoints, Object, spine and arm are identical in Z, R and O.  R reads only the
frozen HMR prediction and the rig; ``PREDECLARED`` was fixed before any R result was computed.

Subcommand ``fk-audit`` (server, SMPL v1 model): checks that HMR's global_orient/body_pose reproduce its own 24 joints by
SMPL forward kinematics and reports the template rest thigh/hinge geometry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

W97_PREDICTIONS_SHA = "fc5d330565205f4792cd3641671633decbab4e24d70522d4dd85f045d87a7003"
W97_POPULATION_SHA = "533895c980a080985444b703f43e0f64046391c76909dea5a2f305a26b34cdb9"
EX = np.array([1.0, 0.0, 0.0])
SMPL_SIDES = {"l": (1, 4, 7), "r": (2, 5, 8)}  # SMPL hip, knee, ankle joint indices; body_pose[j - 1] is joint j's local rotation
PREDECLARED = {
    "body_frame": "W98 condition T, unchanged (HMR pelvis Object, HMR joint thorax, J1X arm, W95 two-bone legs on HMR joints)",
    "adapter_R": {
        "hinge_source": "HMR thigh global frame G = CV_TO_CANON . global_orient . body_pose[hip-1]; predicted hinge axis a = G e_x "
                        "(SMPL template +x, the knee rotation axis of both legs)",
        "roll": "phi_R = wrap180(signed angle about the posed world thigh axis u from the Z rig hinge h_w to a) - psi0[side]; "
                "applied with W96 rolled(): thigh rolled about u, calf = shortest arc from its rolled rest direction to the target",
        "rest_offset": "psi0 = W96 rest_offsets (identity SMPL pose vs BaseRig rest hinge, signed about the rig rest thigh "
                       "direction in body axes; +7.8 left / -7.5 right); not refit",
        "no_weighting_no_flip": "no flexion weighting, no sign flip, no clamp: the predicted rotation is used as is (a roll "
                                "beyond 90 deg is reported, not corrected)",
        "unavailable": "no HMR prediction, no W98 T body frame, or |a perpendicular to u| < 0.1 -> R unavailable for that leg "
                       "(counted; never replaced by Z, GT or another source)",
    },
    "evaluation": {
        "reference": "3DPW SMPL thigh hinge axis (poses hip rotation through cam_poses, W96 SmplLegs) with the same psi0; "
                     "evaluation only; shares the SMPL body-model family with HMR (not an independent anatomical sensor)",
        "flexion_bins_deg": [0, 5, 20, 45, 90, 180],
        "bounds": {"hip_axial_rotation_max_deg": 45.0, "knee_off_hinge_max_deg": 10.0, "calf_twist_max_deg": 30.0},
        "severe_jump": "thigh-local step > 90 deg, or |roll step| > 90 deg, while the GT thigh direction step < 45 deg (W96)",
        "roll_jitter": "mean |adjacent-frame change of the signed roll error vs SMPL| (subject's own roll motion removed)",
        "rules": {
            "R1_deep_roll": "Z knee flexion >= 45: mean(|roll err R| - |roll err Z|) < 0 with the sequence-cluster 95% interval "
                            "below 0, on TEST and on VALIDATION",
            "R2_low_flex_roll": "flexion < 20: mean(|roll err R| - |roll err Z|) <= +2 deg on TEST and VALIDATION",
            "R3_deep_off_hinge": "flexion >= 45: mean(|off-hinge R| - |off-hinge Z|) < 0 with the interval below 0, TEST and VALIDATION",
            "R4_low_flex_off_hinge": "flexion < 20: mean(|off-hinge R| - |off-hinge Z|) <= +2 deg, TEST and VALIDATION",
            "R5_temporal": "severe thigh jumps and severe roll steps R <= Z, and roll jitter R <= Z + 1 deg, on TEST and VALIDATION",
            "R6_hip_axial": "share of R legs with |roll| > 45 deg <= 2% on TEST and VALIDATION",
            "blender": "skinned-rig review of the fixed cases: no visible new twisting of near-straight legs or feet, knee hinge "
                       "direction not worse in seated/deep cases",
        },
        "case_mapping": "A: R1-R6 pass on both splits and Blender shows no visible regression; B: deep-flexion benefit (R1 and "
                        "R3) in a bounded subset but another rule fails or Blender regresses; C: R1 and R3 fail, or only the "
                        "SMPL-family roll metric improves while rig-internal/visual quality does not; D: Gate 1-2 blocked",
    },
    "gates": {
        "G1_fk_consistency": "SMPL FK of HMR global_orient/body_pose/betas reproduces HMR joints24 (root-relative) to < 1e-4 m",
        "G1_swing": "the HMR thigh frame's template thigh direction equals the HMR hip->knee direction (follows from G1)",
        "G2_extra_dof": "axial roll is not fixed by joint centres: report the angle between the HMR hinge axis and HMR's own "
                        "hip-knee-ankle plane normal by flexion (a pure hinge would make them equal for flexed knees)",
    },
    "cases": {
        "fixed": "W96 walking_upright, pelvis_stable_bend, seated_or_crouched, turn_180, worst_hinge; W97 s0_wrong_knee_plane; "
                 "W98 worst_hmr_pelvis; 3DHP W96 ts2_supine, ts4_lying",
        "new_rules": "hmr_upright_shin_regression = window with the largest summed HMR shin error on upright rows; "
                     "near_straight_max_roll = window with the largest summed |phi_R| on legs with flexion < 20 (a failure probe); "
                     "worst_R_vs_Z = window with the largest summed (|roll err R| - |roll err Z|) (a failure probe); windows "
                     "never overlap already used frames",
        "closeups": "seated_or_crouched, worst_hinge, walking_upright, near_straight_max_roll, worst_R_vs_Z",
    },
}
METHOD_LABEL = {"Z": "Z: zero-twist legs (W98 T)", "R": "R: HMR SMPL hip roll", "O": "O: GT SMPL roll oracle (eval only)",
                "Zrep": "Z (repeat; 3DHP has no SMPL roll reference)"}


def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def ang(a, b):
    return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(unit(a), unit(b)))))))


# ------------------------------------------------------------------ GT-free rotation readout and adapter

def hmr_leg_frames(global_orient, body_pose, cv_to_canon):
    """Per leg: HMR thigh global frame (SMPL template axes -> canonical), its hinge axis and the knee local rotation."""
    out = {}
    for s, (hj, kj, _) in SMPL_SIDES.items():
        G = cv_to_canon @ np.asarray(global_orient, float) @ np.asarray(body_pose[hj - 1], float)
        out[s] = {"G": G, "axis": G @ EX, "knee_local": np.asarray(body_pose[kj - 1], float)}
    return out


def r_phi(upper, h_arm, Ro, Am, axis_w, psi0, min_perp=0.1):
    """Roll (deg) about the posed thigh axis that brings the rig hinge onto the HMR hinge axis; (None, perp) if degenerate."""
    uw = Ro @ (Am.T @ unit(upper))
    hw = Ro @ (Am.T @ np.asarray(h_arm, float))
    a = np.asarray(axis_w, float)
    perp = float(np.linalg.norm(a - np.dot(a, uw) * uw))
    if perp < min_perp:
        return None, perp
    p, q = hw - np.dot(hw, uw) * uw, a - np.dot(a, uw) * uw
    return wrap(math.degrees(math.atan2(float(np.dot(uw, np.cross(p, q))), float(np.dot(p, q)))) - psi0), perp


def knee_components(K, rot_axis_fn, twist_fn):
    """HMR knee local rotation: total angle, signed hinge twist about template x, off-hinge swing angle (swing = K twist^-1)."""
    tw = twist_fn(K, EX)
    sw = K @ rot_axis_fn(EX, -tw)
    return {"angle": math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(K) - 1) / 2)))), "hinge_twist": tw,
            "off_hinge_swing": math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(sw) - 1) / 2))))}


# ------------------------------------------------------------------ per-row legs (Z / R / O)

def leg_conditions(r, ctx, smpl_legs, w96):
    """r['W99'] for one row under the W98 T body frame; None when T or the HMR prediction is unavailable."""
    c, e = r["H"].get("T"), r.get("E")
    if c is None or e is None:
        return None
    ev, Ro, Am = c["ev"], c["Ro"], ctx["Am"]
    out = {"Z": {}, "R": {}, "O": {}, "hmr": {}}
    for s in ("l", "r"):
        pose, f = ev["legs"][s]["pose"], ctx["lf"][s]
        sm = smpl_legs[s] if smpl_legs is not None else None
        args = (Ro, Am, sm, ctx["psi0"][s], ctx["rest"], ctx["legs"][s]["target"])

        def meas(dp, dd, phi):
            m = w96.leg_measures(pose, f, dp, dd, *args, phi)
            m["thigh_world_dev"] = ang(dp @ f["t0"], pose.upper)
            m["shin_world_dev"] = ang(dd @ f["c0"], pose.forearm)
            return m
        dp0, dd0 = w96.deltas(pose, f)
        out["Z"][s] = meas(dp0, dd0, 0.0)
        hl = e["legs"][s]
        phi, perp = r_phi(pose.upper, dp0 @ f["n_rest"], Ro, Am, hl["axis"], ctx["psi0"][s])
        if phi is None:
            out["R"][s] = None
        else:
            dp, dd, _, _ = w96.rolled(pose, f, phi)
            out["R"][s] = meas(dp, dd, phi)
            out["R"][s]["axis_perp"] = perp
        if "smpl_roll_err" in out["Z"][s]:
            po = out["Z"][s]["smpl_roll_err"]
            dp, dd, _, _ = w96.rolled(pose, f, po)
            out["O"][s] = meas(dp, dd, po)
        else:
            out["O"][s] = None
        n = np.cross(pose.upper, pose.forearm)
        kc = knee_components(hl["knee_local"], w96.rot_axis, w96.twist_deg)
        d = {"joint_flex": pose.bend_angle_degrees(), "knee_bp_angle": kc["angle"], "knee_bp_hinge_twist": kc["hinge_twist"],
             "knee_bp_off_hinge": kc["off_hinge_swing"], "axis_perp": perp}
        if pose.bend_angle_degrees() >= 20 and np.linalg.norm(n) > 1e-9:
            nw = Ro @ (Am.T @ unit(n))
            uw = Ro @ (Am.T @ unit(pose.upper))
            a = hl["axis"] - np.dot(hl["axis"], uw) * uw
            d["hinge_vs_own_plane"] = math.degrees(math.acos(min(1.0, abs(float(np.dot(unit(a), nw))))))
        out["hmr"][s] = d
    co = r["H"].get("O")
    if co is not None:
        out["Z_O"] = {s: co["m"][s]["Z"] for s in ("l", "r")}
    return out


# ------------------------------------------------------------------ summaries

def flex_bin(x):
    e = PREDECLARED["evaluation"]["flexion_bins_deg"]
    for lo, hi in zip(e[:-1], e[1:]):
        if lo <= x < hi:
            return f"{lo}-{hi}"
    return f"{e[-1]}+"


def legs_of(rows):
    return [(r, s) for r in rows if r.get("W99") for s in ("l", "r")]


def group_stats(legs, brief, methods=("Z", "R", "O", "Z_O")):
    B = PREDECLARED["evaluation"]["bounds"]
    out = {"legs": len(legs)}
    for m in methods:
        pairs = [((r, s), r["W99"][m][s]) for r, s in legs if r["W99"].get(m) and r["W99"][m].get(s) is not None]
        xs = [x for _, x in pairs]
        if not xs:
            continue
        bad = [((r["sid"], r["frame"], s), x["fk_parity"]) for (r, s), x in pairs if x.get("fk_parity", 0) > 1e-6]
        d = {"n": len(xs), "off_hinge_abs": float(np.mean([abs(x["off_hinge"]) for x in xs])),
             "off_hinge_over": float(np.mean([abs(x["off_hinge"]) > B["knee_off_hinge_max_deg"] for x in xs])),
             "calf_twist_abs": float(np.mean([abs(x["calf_twist"]) for x in xs])),
             "calf_twist_over": float(np.mean([abs(x["calf_twist"]) > B["calf_twist_max_deg"] for x in xs])),
             "hip_axial_abs": float(np.mean([abs(x["roll"]) for x in xs])),
             "hip_axial_over_45": float(np.mean([abs(x["roll"]) > B["hip_axial_rotation_max_deg"] for x in xs])),
             "hip_axial_over_90": float(np.mean([abs(x["roll"]) > 90 for x in xs])),
             "fk_parity_max": float(max(x["fk_parity"] for x in xs)) if "fk_parity" in xs[0] else None,
             "fk_parity_over_1e-6": len(bad), "fk_parity_over_1e-6_legs": [[*k, v] for k, v in bad[:10]]}
        if "thigh_world_dev" in xs[0]:
            d["thigh_world_dev_max"] = float(max(x["thigh_world_dev"] for x in xs))
            d["shin_world_dev_max"] = float(max(x["shin_world_dev"] for x in xs))
        re = [abs(x["smpl_roll_err"]) for x in xs if "smpl_roll_err" in x]
        if re:
            d["roll_err_abs"], d["roll_err_median"], d["roll_err_p95"] = float(np.mean(re)), float(np.median(re)), float(np.percentile(re, 95))
        ke = [x["smpl_knee_axis_err"] for x in xs if "smpl_knee_axis_err" in x]
        if ke:
            d["knee_axis_err"] = float(np.mean(ke))
        out[m] = d
    for k, f in (("roll_err", lambda x: abs(x["smpl_roll_err"]) if "smpl_roll_err" in x else None),
                 ("off_hinge", lambda x: abs(x["off_hinge"])), ("calf_twist", lambda x: abs(x["calf_twist"]))):
        for a, b in (("Z", "R"), ("Z_O", "R"), ("Z", "O")):
            pr = []
            for r, s in legs:
                xa, xb = (r["W99"].get(a) or {}).get(s), (r["W99"].get(b) or {}).get(s)
                if xa is None or xb is None or f(xa) is None or f(xb) is None:
                    continue
                pr.append((r["sid"], f(xb) - f(xa)))
            if len(pr) > 1:
                c = brief(np.array([p[0] for p in pr]), np.array([p[1] for p in pr]))
                out[f"{k}_{b}_minus_{a}"] = {"n": len(pr), "diff": c.get("mean"), "ci95": c.get("ci95")}
    hm = [r["W99"]["hmr"][s] for r, s in legs]
    out["hmr"] = {"knee_bp_off_hinge": float(np.mean([h["knee_bp_off_hinge"] for h in hm])),
                  "knee_bp_angle_minus_joint_flex": float(np.mean([abs(h["knee_bp_angle"] - h["joint_flex"]) for h in hm]))}
    hv = [h["hinge_vs_own_plane"] for h in hm if "hinge_vs_own_plane" in h]
    if hv:
        out["hmr"]["hinge_vs_own_plane_mean"], out["hmr"]["hinge_vs_own_plane_median"] = float(np.mean(hv)), float(np.median(hv))
    return out


def summarize(rows, brief, masks=None):
    legs = legs_of(rows)
    out = {"all": group_stats(legs, brief), "by_flexion": {}, "by_stratum": {}, "challenge": {}, "by_side": {}}
    g = defaultdict(list)
    for r, s in legs:
        g[flex_bin(r["W99"]["Z"][s]["flex"])].append((r, s))
    out["by_flexion"] = {k: group_stats(v, brief) for k, v in sorted(g.items())}
    for band, test in (("flex_ge_45", lambda x: x >= 45), ("flex_20_45", lambda x: 20 <= x < 45), ("flex_lt_20", lambda x: x < 20)):
        out["by_flexion"][band] = group_stats([(r, s) for r, s in legs if test(r["W99"]["Z"][s]["flex"])], brief)
    st = defaultdict(list)
    for r, s in legs:
        st[str(r.get("stratum"))].append((r, s))
    out["by_stratum"] = {k: group_stats(v, brief) for k, v in sorted(st.items())}
    for k, f in (masks or {}).items():
        out["challenge"][k] = group_stats([(r, s) for r, s in legs if f(r)], brief)
    for side in ("l", "r"):
        sl = [(r, s) for r, s in legs if s == side]
        d = group_stats(sl, brief)
        for m in ("Z", "R", "O"):
            xs = [r["W99"][m][s] for r, s in sl if r["W99"].get(m) and r["W99"][m].get(s) is not None]
            if xs:
                d[m]["roll_signed_mean"] = float(np.mean([x["roll"] for x in xs]))
                se = [x["smpl_roll_err"] for x in xs if "smpl_roll_err" in x]
                if se:
                    d[m]["roll_err_signed_mean"] = float(np.mean(se))
        out["by_side"][side] = d
    return out


def temporal(seq_rows, ang_q, wangle, methods=("Z", "R", "O")):
    E = PREDECLARED["evaluation"]
    out = {}
    for m in methods:
        jumps = rolls = steps = 0
        jit, obj = [], []
        for rs in seq_rows.values():
            for a, b in zip(rs[:-1], rs[1:]):
                wa, wb = a.get("W99"), b.get("W99")
                if not wa or not wb or a.get("gt_legs") is None or b.get("gt_legs") is None:
                    continue
                for s in ("l", "r"):
                    xa, xb = (wa.get(m) or {}).get(s), (wb.get(m) or {}).get(s)
                    if xa is None or xb is None:
                        continue
                    steps += 1
                    if wangle(a["gt_legs"][s][0], b["gt_legs"][s][0]) < 45:
                        jumps += ang_q(xa["q_p"], xb["q_p"]) > 90
                        rolls += abs(wrap(xb["roll"] - xa["roll"])) > 90
                    if "smpl_roll_err" in xa and "smpl_roll_err" in xb:
                        jit.append(abs(wrap(xb["smpl_roll_err"] - xa["smpl_roll_err"])))
        out[m] = {"leg_steps": steps, "severe_thigh_jumps": int(jumps), "severe_roll_steps": int(rolls),
                  "roll_jitter_mean": float(np.mean(jit)) if jit else None, "roll_jitter_p95": float(np.percentile(jit, 95)) if jit else None}
    # subject's own roll motion (GT thigh hinge step about the thigh axis) for scale, from the oracle's zero-error definition
    return out


def rules(res):
    """Evaluate the predeclared R1-R6 on TEST and VALIDATION (Blender judged separately)."""
    out = {}
    for split in ("3dpw_test", "3dpw_validation"):
        f = res[split]["summary"]["by_flexion"]
        t = res[split]["temporal"]
        a = res[split]["summary"]["all"]
        deep, low = f["flex_ge_45"], f["flex_lt_20"]
        def below(c):
            return c is not None and c.get("ci95") is not None and c["diff"] < 0 and c["ci95"][1] < 0
        out[split] = {
            "R1_deep_roll": below(deep.get("roll_err_R_minus_Z")),
            "R2_low_flex_roll": low.get("roll_err_R_minus_Z", {}).get("diff", 99) <= 2.0,
            "R3_deep_off_hinge": below(deep.get("off_hinge_R_minus_Z")),
            "R4_low_flex_off_hinge": low.get("off_hinge_R_minus_Z", {}).get("diff", 99) <= 2.0,
            "R5_temporal": (t["R"]["severe_thigh_jumps"] <= t["Z"]["severe_thigh_jumps"] and t["R"]["severe_roll_steps"] <= t["Z"]["severe_roll_steps"]
                            and (t["R"]["roll_jitter_mean"] or 0) <= (t["Z"]["roll_jitter_mean"] or 0) + 1.0),
            "R6_hip_axial": a["R"]["hip_axial_over_45"] <= 0.02,
        }
    return out


# ------------------------------------------------------------------ fk-audit (server; SMPL v1 model)

def fk_positions(go, bp, J, parents):
    G = [np.asarray(go, float)]
    P = [J[0]]
    for k in range(1, len(J)):
        p = parents[k]
        G.append(G[p] @ bp[k - 1])
        P.append(P[p] + G[p] @ (J[k] - J[p]))
    return np.array(P), G


def main_fk_audit(argv):
    import scipy.sparse as sp
    ap = argparse.ArgumentParser(description="W99 SMPL FK consistency audit of the frozen HMR predictions")
    for name in ("predictions", "smpl_model", "out"):
        ap.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    a = ap.parse_args(argv)
    if hashlib.sha256(a.predictions.read_bytes()).hexdigest() != W97_PREDICTIONS_SHA:
        raise SystemExit("STOP: frozen input changed: predictions")
    d = pickle.load(a.smpl_model.open("rb"), encoding="latin1")
    arr = lambda k: d[k].toarray() if sp.issparse(d[k]) else np.asarray(d[k], float)  # noqa: E731
    vt, sd, Jr = arr("v_template"), arr("shapedirs")[:, :, :10], arr("J_regressor")
    parents = np.asarray(d["kintree_table"], int)[0]
    parents[0] = -1
    pred = np.load(a.predictions)
    err, err_thigh, err_shin = [], [], []
    for i in np.flatnonzero(pred["valid"]):
        J = Jr @ (vt + sd @ pred["betas"][i])
        P, _ = fk_positions(pred["global_orient"][i], pred["body_pose"][i], J, parents)
        j = pred["joints24"][i]
        err.append(float(np.abs((P - P[0]) - (j - j[0])).max()))
        for hj, kj, aj in SMPL_SIDES.values():
            err_thigh.append(ang(P[kj] - P[hj], j[kj] - j[hj]))
            err_shin.append(ang(P[aj] - P[kj], j[aj] - j[kj]))
    J0 = Jr @ vt
    rest = {s: {"thigh_dir_template": list(unit(J0[kj] - J0[hj])), "shin_dir_template": list(unit(J0[aj] - J0[kj])),
                "thigh_vs_x_deg": ang(J0[kj] - J0[hj], EX), "rest_knee_bend_deg": ang(J0[kj] - J0[hj], J0[aj] - J0[kj]),
                "x_vs_rest_plane_normal_deg": ang(np.cross(J0[kj] - J0[hj], J0[aj] - J0[kj]), EX)}
            for s, (hj, kj, aj) in SMPL_SIDES.items()}
    res = {"rows": len(err), "joint_pos_err_max_m": float(max(err)), "joint_pos_err_mean_m": float(np.mean(err)),
           "thigh_dir_err_max_deg": float(max(err_thigh)), "shin_dir_err_max_deg": float(max(err_shin)),
           "parents": parents.tolist()[:9], "template_rest_zero_betas": rest,
           "betas_effect": {"thigh_dir_spread_deg": float(np.max([ang(unit((Jr @ (vt + sd @ b))[4] - (Jr @ (vt + sd @ b))[1]), J0[4] - J0[1])
                                                                    for b in pred["betas"][np.flatnonzero(pred["valid"])[::50]]]))}}
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


# ------------------------------------------------------------------ main replay

def main():
    import eval_external_hmr2_baseline as w97e
    import run_external_hmr2_baseline as hmr
    import run_external_orientation_handoff as w98
    import run_leg_knee_hinge_twist as w96
    import run_pelvis_frame_leg_articulation as w95
    import run_pelvis_torso_ownership as w94
    from run_root_motion_contact_h0_replay import _matched_oracle_sequence
    from run_runtime_heading_3dhp import shim
    from run_torso_frame_rig_replay import NAMES, S0_SHA
    from run_visual_depth_motion_attribution import W87_PREDICTIONS, _brief

    from framepose.bank import load_bank
    from framepose.bedlam_supervision import ExternalBank
    from framepose.connected_limb_transfer import chain_targets
    from motion.torso_frame import quaternion_to_matrix, torso_frames
    from pose.framepose_bridge import assemble_h0, build_h0_lifted_sequence, sequence_ids_in_split
    from pose.root_orientation_diagnostic import observe_yaw
    from retarget.framepose_fk import derive_rig_alignment, load_rig_fk_calibration
    from retarget.framepose_target_rest import load_target_rest_pose
    from retarget.framepose_two_bone_ik import load_ik_chain_set, quaternion_angle_degrees, target_chain_geometry

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bank", "s0_dir", "raw_3dpw", "w87_3dpw", "hp_bank", "hp_s0", "w87_3dhp", "w96_cases", "w97_eval", "w98_report", "population",
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
            R_p, R_t, j17 = w98.handoff_frames(pred["global_orient"][i], pred["joints24"][i])
            E[(str(pp), int(bp))] = {"R_p": R_p, "R_t": R_t, "j17": j17, "R_chain": w98.smpl_chain_thorax(pred["global_orient"][i], pred["body_pose"][i]),
                                     "legs": hmr_leg_frames(pred["global_orient"][i], pred["body_pose"][i], hmr.CV_TO_CANON),
                                     "joints24": pred["joints24"][i]}
    rest = load_target_rest_pose(args.rest)
    alignment = derive_rig_alignment(rest, load_rig_fk_calibration(args.calibration))
    Am = quaternion_to_matrix(alignment)
    target = target_chain_geometry(rest, load_ik_chain_set(args.chains).chains[0])
    sb = rest.bones["spine_01"]
    M = np.asarray(sb.matrix_local, float)[:3, :3]
    lf = w96.leg_frames(rest)
    ctx = {"alignment": alignment, "Am": Am, "rest": rest, "target": target, "spine_basis": M / np.linalg.norm(M, axis=0, keepdims=True),
           "spine_head": list(sb.head), "legs": w95.leg_targets(rest), "lf": lf, "psi0": w96.rest_offsets(lf, Am)}
    report = {"schema": "worklog99_learned_rotation_vs_geometry_ik", "predeclared": PREDECLARED,
              "frozen_inputs": {str(k): v for k, v in frozen.items()}, "w98_report_sha256": sha(args.w98_report),
              "psi0_deg": ctx["psi0"], "rest_audit": rest_audit(ctx, w94, w96)}
    # ---- 3DPW (identical to the W98 row construction)
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
    w98r = json.loads(args.w98_report.read_text())["results"]
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
            r["H"] = w98.row_conditions(r, ctx, smpl_legs, j1x3)
            r["W99"] = leg_conditions(r, ctx, smpl_legs, w96)
            if r["E"] is not None:  # HMR shin direction error vs GT (canonical 17 joints; evaluation-only case selection)
                J = r["E"]["j17"]
                r["hmr_shin_err"] = float(np.mean([ang(J[a] - J[k], G[a] - G[k]) for k, a in ((2, 3), (5, 6))]))
        masks = w97e.challenge_masks(rows)
        res = {"coverage": coverage(rows), "summary": summarize(rows, _brief, masks), "temporal": temporal(pw[split], quaternion_angle_degrees, w95.angle),
               "w98_reproduction": w98_reproduction(rows, w98r[f"3dpw_{split}"])}
        results[f"3dpw_{split}"] = res
        print(split, json.dumps(res["coverage"]), json.dumps(res["w98_reproduction"]), flush=True)
    # ---- 3DHP (no SMPL reference; rig-internal only)
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
        r["H"] = w98.row_conditions(r, ctx, None, (ph["left_arm_g1_elbow"], ph["left_arm_g1_wrist"]))
        r["W99"] = leg_conditions(r, ctx, None, w96)
    results["3dhp"] = {"coverage": coverage(hl), "summary": summarize(hl, _brief), "temporal": temporal(hrows, quaternion_angle_degrees, w95.angle, ("Z", "R"))}
    report["results"] = results
    report["rules"] = rules(results)
    report["cases"] = export_cases(pw, hrows, ctx, args, rest, alignment, target, w94, w98, w96)
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps(report["rules"]))


def coverage(rows):
    w = [r for r in rows if r.get("W99")]
    return {"rows": len(rows), "T_rows": sum(r["H"].get("T") is not None for r in rows), "W99_rows": len(w),
            "R_legs": sum(r["W99"]["R"][s] is not None for r in w for s in "lr"),
            "R_unavailable_legs": sum(r["W99"]["R"][s] is None for r in w for s in "lr"),
            "O_legs": sum(r["W99"]["O"][s] is not None for r in w for s in "lr")}


def w98_reproduction(rows, w98res):
    """Z under T must reproduce W98 condition T's leg metrics (roll vs SMPL, off-hinge), row means of the two legs."""
    w = [r for r in rows if r.get("W99")]
    rz = [np.mean([abs(r["W99"]["Z"][s]["smpl_roll_err"]) for s in "lr"]) for r in w if all("smpl_roll_err" in r["W99"]["Z"][s] for s in "lr")]
    oz = [np.mean([abs(r["W99"]["Z"][s]["off_hinge"]) for s in "lr"]) for r in w]
    t = w98res["all"]["T"]
    return {"rows": len(w), "W98_T_rows": t["rows"], "W98_T_roll_vs_smpl": t["roll_vs_smpl"]["mean"], "Z_roll_vs_smpl": float(np.mean(rz)),
            "W98_T_off_hinge": t["off_hinge"]["mean"], "Z_off_hinge": float(np.mean(oz))}


def rest_audit(ctx, w94, w96):
    """BaseRig rest legs in body axes vs the SMPL template conventions (no data)."""
    Am, lf = ctx["Am"], ctx["lf"]
    out = {}
    for s in ("l", "r"):
        f = lf[s]
        t0b, n0b = Am.T @ f["t0"], Am.T @ f["n_rest"]
        bz = Am.T @ f["n_bone_z"]
        lat = w94.SMPL_BODY.T @ EX
        out[s] = {"rig_rest_thigh_body": list(t0b), "rig_rest_knee_bend_deg": ang(f["t0"], f["c0"]),
                  "rig_hinge_vs_bone_z_deg": ang(f["n_rest"], f["n_bone_z"]),
                  "rig_hinge_vs_smpl_x_deg": ang(n0b, lat), "rig_bone_z_vs_smpl_x_deg": ang(bz, lat),
                  "psi0_deg": ctx["psi0"][s],
                  "psi0_if_bone_z_deg": w96.signed_angle(bz, lat, t0b)}
    return out


# ------------------------------------------------------------------ Blender cases

def export_case(name, w, slots, ctx, title, note, w94, w98):
    from motion.torso_frame import matrix_to_quaternion_xyzw
    from run_torso_frame_rig_replay import conj, spine_local_xyzw
    alignment, B, head, target = ctx["alignment"], ctx["spine_basis"], ctx["spine_head"], ctx["target"]
    # the side shown in the plots: larger mean Z knee flexion over the window
    fl = {s: np.mean([r["W99"]["Z"][s]["flex"] for r in w if r.get("W99")] or [0]) for s in ("l", "r")}
    side = max(fl, key=fl.get)
    frames = []
    s_roll = {METHOD_LABEL[m]: [] for m in slots.values()}
    s_off = {METHOD_LABEL[m]: [] for m in slots.values()}
    for r in w:
        rec = {"bank_position": r["p"], "frame_index": r["frame"], "time_s": r["t"], "image": r["image"], "available": r["available"],
               "gt_valid": r["gt_arm_ok"], "runtime_status": "KNOWN", "runtime_reasons": [], "yaw_deg": {}, "modes": {}, "forward_cam": {},
               "mode_lines": {}, "object_rotation_xyzw": {}, "extra_bone_locals_xyzw": {}, "fk_points_armature": {}}
        if r["gt_legs"] is not None:
            rec["reference_legs_cam"] = {"left": [list(r["gt_legs"]["l"][0]), list(r["gt_legs"]["l"][1])],
                                         "right": [list(r["gt_legs"]["r"][0]), list(r["gt_legs"]["r"][1])]}
        c, W = r["H"].get("T"), r.get("W99")
        for slot, m in slots.items():
            lab = METHOD_LABEL[m]
            mm = "Z" if m == "Zrep" else m
            legs = (W or {}).get(mm) or {}
            if c is None or W is None or any(legs.get(s) is None for s in ("l", "r")):
                rec["yaw_deg"][slot] = None
                rec["mode_lines"][slot] = ["condition unavailable for this row (no GT/Z fallback)"]
                s_roll[lab].append(None)
                s_off[lab].append(None)
                continue
            Ro, Rt, ev = c["Ro"], c["Rt"], c["ev"]
            y = w94.compat_yaw(Rt)
            rec["yaw_deg"][slot] = math.degrees(y) if y is not None else None
            rec["forward_cam"][slot] = list(Rt[:, 1])
            rec["object_rotation_xyzw"][slot] = list(matrix_to_quaternion_xyzw(conj(alignment, Ro)))
            bones = {"spine_01": list(spine_local_xyzw(alignment, Ro.T @ Rt, B))}
            for x in "lr":
                p = ev["legs"][x]["pose"]
                bones[f"thigh_{x}"], bones[f"calf_{x}"] = list(legs[x]["q_p"]), list(legs[x]["q_d"])
                rec["fk_points_armature"].setdefault(slot, {}).update({f"thigh_{x}": list(p.mid), f"calf_{x}": list(p.end)})
            rec["extra_bone_locals_xyzw"][slot] = bones
            arm = c["arm"]
            if arm is not None and arm.status == "known":
                Q = conj(alignment, Ro.T @ Rt)
                sp = lambda pt: list(np.asarray(head) + Q @ (np.asarray(pt) - np.asarray(head)))  # noqa: E731
                rec["modes"][slot] = {"proximal_local_xyzw": list(arm.proximal_local), "distal_local_xyzw": list(arm.distal_local),
                                      "mid": sp(arm.mid), "end": sp(arm.end), "root": sp(target.root_head),
                                      "object_rotation_armature_xyzw": rec["object_rotation_xyzw"][slot], "extra_bone_locals_xyzw": {"spine_01": bones["spine_01"]}}
            L, Rr = legs["l"], legs["r"]
            lines = [f"knee flex L/R {L['flex']:4.0f}/{Rr['flex']:4.0f} | hip axial roll L/R {L['roll']:4.0f}/{Rr['roll']:4.0f}",
                     f"knee off-hinge L/R {L['off_hinge']:4.0f}/{Rr['off_hinge']:4.0f} | calf twist L/R {L['calf_twist']:4.0f}/{Rr['calf_twist']:4.0f}"]
            if "smpl_roll_err" in L and "smpl_roll_err" in Rr:
                lines.append(f"roll vs SMPL L/R {L['smpl_roll_err']:4.0f}/{Rr['smpl_roll_err']:4.0f} (eval only)")
            rec["mode_lines"][slot] = lines
            s_roll[lab].append(legs[side]["roll"])
            s_off[lab].append(legs[side]["off_hinge"])
        frames.append(rec)
    colors = {METHOD_LABEL[m]: col for m, col in zip(slots.values(), ("#eb6834", "#2a78d6", "#1baf7a"))}
    sname = {"l": "left", "r": "right"}[side]
    return {"case": name, "split": "3dpw" if any(r.get("R_smpl") is not None for r in w) else "3dhp", "sequence_id": w[0]["sid"],
            "start_frame": w[0]["frame"], "fps_sampled": 1.0 / max(w[1]["t"] - w[0]["t"], 1e-6), "frames": frames,
            "labels": {s: METHOD_LABEL[m] for s, m in slots.items()}, "short_labels": {s: m for s, m in slots.items()},
            "title": f"W99 learned hip roll vs zero-twist IK {title}", "header_note": note,
            "plot_series": {"panels": [{"ylabel": f"{sname} hip axial roll vs zero-twist (deg)", "series": s_roll, "colors": colors},
                                       {"ylabel": f"{sname} knee off-hinge (deg)", "series": s_off, "colors": colors}]}}


def export_cases(pw, hrows, ctx, args, rest, alignment, target, w94, w98, w96):
    w96c = {**json.loads((args.w96_cases / "cases_3dpw.json").read_text())["cases"], **json.loads((args.w96_cases / "cases_3dhp.json").read_text())["cases"]}
    w97c = json.loads((args.w97_eval / "cases_3dpw.json").read_text())["cases"]
    dpw = {**pw["validation"], **pw["test"]}
    note = "W98 T body frame (HMR pelvis + thorax), HMR joints, J1X arm, identical frames | thin green = GT legs (evaluation only)"
    slots = {"FIXED": "Z", "RUNTIME": "R", "GT": "O"}
    cases = {}
    for key, src in (("3dpw_walking_upright", w96c), ("3dpw_pelvis_stable_bend", w96c), ("3dpw_seated_or_crouched", w96c),
                     ("3dpw_turn_180", w96c), ("3dpw_worst_hinge", w96c), ("3dpw_s0_wrong_knee_plane", w97c)):
        c = src[key]
        cases[key] = (c["case"], w94.window_at(dpw, c["sequence_id"], c["start_frame"]))
    cases["3dpw_worst_hmr_pelvis"] = ("worst_hmr_pelvis", w94.window_at(dpw, "3dpw:downtown_runForBus_01:actor0", 451))
    used = {(r["sid"], r["frame"]) for _, w in cases.values() for r in w}

    def rule(score):
        def f(w):
            if any((r["sid"], r["frame"]) in used for r in w) or any(not r.get("W99") for r in w):
                return -1e9
            return score(w)
        return f
    for key, name, score in (
            ("3dpw_hmr_upright_shin_regression", "hmr_upright_shin_regression",
             lambda w: sum(r.get("hmr_shin_err", 0) for r in w if r.get("stratum") == "upright")),
            ("3dpw_near_straight_max_roll", "near_straight_max_roll",
             lambda w: sum(abs(r["W99"]["R"][s]["roll"]) for r in w for s in "lr" if r["W99"]["R"][s] is not None and r["W99"]["Z"][s]["flex"] < 20)),
            ("3dpw_worst_R_vs_Z", "worst_R_vs_Z",
             lambda w: sum(abs(r["W99"]["R"][s]["smpl_roll_err"]) - abs(r["W99"]["Z"][s]["smpl_roll_err"]) for r in w for s in "lr"
                           if r["W99"]["R"][s] is not None and "smpl_roll_err" in r["W99"]["R"][s] and "smpl_roll_err" in r["W99"]["Z"][s]))):
        win = w94.best_window(dpw, rule(score))
        cases[key] = (name, win)
        used |= {(r["sid"], r["frame"]) for r in win}
    out = {k: export_case(n, w, slots, ctx, "(3DPW)", note, w94, w98) for k, (n, w) in cases.items()}
    hslots = {"FIXED": "Z", "RUNTIME": "R", "GT": "Zrep"}
    for key in ("3dhp_ts2_supine", "3dhp_ts4_lying"):
        c = w96c[key]
        out[key] = export_case(c["case"], w94.window_at(hrows, c["sequence_id"], c["start_frame"]), hslots, ctx, "(3DHP, no SMPL reference)", note, w94, w98)
    common = {"rest_snapshot_digest": rest.digest(), "canonical_to_armature_xyzw": list(alignment), "target_chain": target.to_dict()}
    close = [c.strip() for c in PREDECLARED["cases"]["closeups"].split(",")]
    for tag in ("3dpw", "3dhp"):
        (args.out_dir / f"cases_{tag}.json").write_text(json.dumps({**common, "cases": {k: v for k, v in out.items() if k.startswith(tag)}}, indent=1))
    cl = {f"{k}_closeup": {**v, "case": v["case"] + "_closeup"} for k, v in out.items() if k.startswith("3dpw") and any(k.endswith(c) for c in close)}
    (args.out_dir / "cases_3dpw_closeup.json").write_text(json.dumps({**common, "cases": cl}, indent=1))
    return {k: {"sequence_id": v["sequence_id"], "start_frame": v["start_frame"], "slots": v["labels"]} for k, v in out.items()}


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "fk-audit":
        main_fk_audit(sys.argv[2:])
    else:
        main()
