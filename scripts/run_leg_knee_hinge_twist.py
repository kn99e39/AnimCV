#!/usr/bin/env python3
"""EXPERIMENT (Worklog 96): is W95's knee-hinge deviation a missing thigh-roll representation?

Leg geometry, rows, root policies and the zero-twist leg poses are W95's, unchanged (``Z``).  For one leg:
  u = H->K (posed thigh direction), c = K->A (posed calf direction), n_obs = unit(u x c) (observed knee-plane normal),
  delta_p = B_t L_p B_t^T (armature-space thigh delta; zero twist = shortest arc from the rest thigh direction),
  rig hinge h = delta_p n_rest, n_rest = unit(rest thigh dir x rest calf dir) (the existing IK contract's rest bend
  normal, also used by ``rest_plane_side``).  W95's hinge metric is angle(h, n_obs) for knee flexion >= 20 deg.

Candidate ``T`` (opt-in, experimental): roll the thigh about u by phi so that h is aligned with s*n_obs, s = +/-1 chosen
so |phi| <= 90 deg (no 180-degree flips; a bend opposite to the rig hinge stays a hyperextension), weighted by
w = clip((flex - 5) / 15, 0, 1) for near-straight knees; the calf takes the shortest arc from its rolled rest direction
to c (a pure rotation about h when w = 1).  World thigh/shin directions, lengths and the parent Object are unchanged.

Evaluation-only oracle ``O`` (3DPW): the same roll construction with phi chosen so the rig thigh's lateral axis matches
the SMPL thigh lateral axis (``poses`` hip rotation through ``cam_poses``), rest offset calibrated analytically.
Independent validation uses that SMPL thigh frame; GT is never a runtime input.  ``PREDECLARED`` is fixed before any
candidate result was read.
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
import run_pelvis_frame_leg_articulation as w95  # noqa: E402
import run_pelvis_torso_ownership as w94  # noqa: E402
from run_root_motion_contact_h0_replay import _find_3dpw_pkl, _matched_oracle_sequence, _parse_3dpw_sequence_id  # noqa: E402
from run_runtime_heading_3dhp import shim  # noqa: E402
from run_torso_frame_rig_replay import NAMES, S0_SHA, conj, spine_local_xyzw  # noqa: E402
from run_visual_depth_motion_attribution import W87_PREDICTIONS, _brief  # noqa: E402

from framepose.bank import load_bank  # noqa: E402
from framepose.bedlam_supervision import ExternalBank  # noqa: E402
from framepose.connected_limb_transfer import chain_targets  # noqa: E402
from motion.torso_frame import matrix_to_quaternion_xyzw, quaternion_to_matrix  # noqa: E402
from pose.framepose_bridge import assemble_h0, build_h0_lifted_sequence, sequence_ids_in_split  # noqa: E402
from pose.root_orientation_diagnostic import observe_yaw  # noqa: E402
from retarget.framepose_fk import _rest_rotation, derive_rig_alignment, load_rig_fk_calibration  # noqa: E402
from retarget.framepose_target_rest import load_target_rest_pose  # noqa: E402
from retarget.framepose_two_bone_ik import chain_forward_kinematics, load_ik_chain_set, quaternion_angle_degrees, target_chain_geometry  # noqa: E402

PREDECLARED = {
    "candidate_T": {
        "hinge_axis": "existing contract rest bend normal n_rest (alternative bone-local Z convention reported as sensitivity only)",
        "roll": "phi = signed angle about u from perp_u(delta_Z n_rest) to perp_u(s n_obs), s = +/-1 with |phi| <= 90 deg",
        "degeneracy_weight": {"zero_below_flexion_deg": 5.0, "full_at_flexion_deg": 20.0},
        "calf": "shortest arc from the rolled rest calf direction to the target calf direction",
    },
    "evaluation": {
        "hip_axial_rotation_max_deg": 45.0, "knee_off_hinge_max_deg": 10.0, "calf_twist_max_deg": 30.0,
        "severe_jump": {"thigh_local_step_deg": 90.0, "roll_step_deg": 90.0, "while_gt_thigh_step_below_deg": 45.0},
        "flexion_bins_deg": [0, 5, 20, 45, 90, 180],
        "smpl_reference": "3DPW SMPL thigh frame (global orient x hip local rotation) lateral axis = template +x; rest offset "
                          "calibrated analytically from the identity SMPL pose vs the BaseRig rest hinge (no data fitting)",
        "independent_rule": "3DPW root B, GT legs, rows with knee flexion >= 20 deg in TEST and in VALIDATION: mean(|roll error "
                            "vs SMPL| T - Z) < 0 with the sequence-cluster interval below 0; flexion < 20 deg: |mean change| "
                            "<= 2 deg; T adds no severe jumps; plus Blender review (visible knee behaviour, hip twist)",
    },
    "w95_metrics": "W95 any-violation / ROM-only rules unchanged (reported for comparability, not as validation of T)",
}
SIDES = {"l": (1, 4), "r": (2, 5)}  # SMPL hip, knee joint indices
STRATA = w94.STRATA


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def rot_axis(u, deg):
    a = _unit(u)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    t = math.radians(deg)
    return np.eye(3) + math.sin(t) * K + (1 - math.cos(t)) * K @ K


def signed_angle(a, b, u):
    u = _unit(u)
    a = np.asarray(a, float) - np.dot(a, u) * u
    b = np.asarray(b, float) - np.dot(b, u) * u
    return math.degrees(math.atan2(float(np.dot(u, np.cross(a, b))), float(np.dot(a, b))))


def shortest_arc(a, b):
    a, b = _unit(a), _unit(b)
    v, c = np.cross(a, b), float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-12:
        if c > 0:
            return np.eye(3)
        p = np.cross(a, (1.0, 0.0, 0.0) if abs(a[0]) < 0.9 else (0.0, 1.0, 0.0))
        return rot_axis(p, 180.0)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1.0 + c)


def twist_deg(R, axis):
    x, y, z, w = matrix_to_quaternion_xyzw(R)
    a = _unit(axis)
    t = math.degrees(2.0 * math.atan2(float(np.dot((x, y, z), a)), w))
    return (t + 180.0) % 360.0 - 180.0


def leg_frames(rest):
    out = {}
    for s in ("l", "r"):
        t, c = rest.bones[f"thigh_{s}"], rest.bones[f"calf_{s}"]
        # exactly the IK's rest rotation (quaternion_from_matrix of matrix_local), so Z round-trips bit-for-bit
        Bt = quaternion_to_matrix(_rest_rotation(rest, f"thigh_{s}"))
        Bc = quaternion_to_matrix(_rest_rotation(rest, f"calf_{s}"))
        t0, c0 = _unit(np.subtract(t.tail, t.head)), _unit(np.subtract(c.tail, c.head))
        out[s] = {"Bt": Bt, "Bc": Bc, "t0": t0, "c0": c0, "n_rest": _unit(np.cross(t0, c0))}
        out[s]["n_bone_z"] = out[s]["Bt"][:, 2] * np.sign(np.dot(out[s]["Bt"][:, 2], out[s]["n_rest"]))
    return out


def deltas(pose, f):
    dp = f["Bt"] @ quaternion_to_matrix(pose.proximal_local) @ f["Bt"].T
    dd = dp @ (f["Bc"] @ quaternion_to_matrix(pose.distal_local) @ f["Bc"].T)
    return dp, dd


def rolled(pose, f, phi):
    """Thigh rolled by phi about the posed thigh direction; calf by shortest arc to the same target direction."""
    dp0, _ = deltas(pose, f)
    u, c = _unit(pose.upper), _unit(pose.forearm)
    dp = rot_axis(u, phi) @ dp0
    dd = shortest_arc(dp @ f["c0"], c) @ dp
    Lp = f["Bt"].T @ dp @ f["Bt"]
    Ld = f["Bc"].T @ (dp.T @ dd) @ f["Bc"]
    return dp, dd, matrix_to_quaternion_xyzw(Lp), matrix_to_quaternion_xyzw(Ld)


def candidate_phi(pose, f, hinge="n_rest"):
    P = PREDECLARED["candidate_T"]["degeneracy_weight"]
    flex = pose.bend_angle_degrees()
    n = np.cross(pose.upper, pose.forearm)
    if np.linalg.norm(n) < 1e-9:
        return 0.0, 0.0, 0, None
    dp0, _ = deltas(pose, f)
    u = _unit(pose.upper)
    h = dp0 @ f[hinge]
    phi, s = signed_angle(h, n, u), 1
    if abs(phi) > 90.0:
        phi, s = signed_angle(h, -n, u), -1
    w = min(1.0, max(0.0, (flex - P["zero_below_flexion_deg"]) / (P["full_at_flexion_deg"] - P["zero_below_flexion_deg"])))
    return w * phi, w, s, phi


class SmplLegs:
    """3DPW SMPL hip/knee frames in AnimCV camera axes (EVALUATION ONLY)."""

    def __init__(self, raw_root):
        self.raw_root, self.cache = raw_root, {}

    def get(self, sid, fi):
        from scipy.spatial.transform import Rotation

        if sid not in self.cache:
            name, actor = _parse_3dpw_sequence_id(sid)
            with _find_3dpw_pkl(self.raw_root, name).open("rb") as h:
                raw = pickle.load(h, encoding="latin1")
            self.cache[sid] = (np.asarray(raw["poses"][actor], float), np.asarray(raw["cam_poses"], float))
        poses, cams = self.cache[sid]
        C = w94.CV_TO_CANON @ cams[fi, :3, :3]
        Rgo = Rotation.from_rotvec(poses[fi, :3]).as_matrix()
        out = {}
        for s, (hj, kj) in SIDES.items():
            Rh = Rotation.from_rotvec(poses[fi, 3 * hj:3 * hj + 3]).as_matrix()
            kv = poses[fi, 3 * kj:3 * kj + 3]
            G = C @ Rgo @ Rh
            ax = G @ (kv / np.linalg.norm(kv)) if np.linalg.norm(kv) > 1e-9 else None
            out[s] = {"lateral": G @ np.array([1.0, 0.0, 0.0]), "knee_axis": ax, "knee_angle": math.degrees(float(np.linalg.norm(kv)))}
        return out


def rest_offsets(f, Am):
    """Analytic rest roll offset between the BaseRig rest hinge and the SMPL template lateral axis (body frame)."""
    lat = w94.SMPL_BODY.T @ np.array([1.0, 0.0, 0.0])
    return {s: signed_angle(Am.T @ f[s]["n_rest"], lat, Am.T @ f[s]["t0"]) for s in ("l", "r")}


def leg_measures(pose, f, dp, dd, Ro, Am, smpl_side, psi0, rest, target, phi):
    u, c = _unit(pose.upper), _unit(pose.forearm)
    flex = pose.bend_angle_degrees()
    h = _unit(dp @ f["n_rest"])
    n = np.cross(pose.upper, pose.forearm)
    out = {"flex": flex, "roll": phi,
           "hinge_w95": (math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(h, _unit(n))))))) if flex >= 20 and np.linalg.norm(n) > 1e-9 else None),
           "off_hinge": math.degrees(math.asin(max(-1.0, min(1.0, float(np.dot(c, h)))))),
           "calf_twist": twist_deg(dp.T @ dd, f["c0"])}
    qp, qd = matrix_to_quaternion_xyzw(f["Bt"].T @ dp @ f["Bt"]), matrix_to_quaternion_xyzw(f["Bc"].T @ (dp.T @ dd) @ f["Bc"])
    mid, end = chain_forward_kinematics(rest, target, qp, qd)
    out["fk_parity"] = max(float(np.linalg.norm(np.subtract(mid, pose.mid))), float(np.linalg.norm(np.subtract(end, pose.end))))
    out["q_p"], out["q_d"] = qp, qd
    if smpl_side is not None:
        hw, uw = Ro @ (Am.T @ h), Ro @ (Am.T @ u)
        e = signed_angle(hw, smpl_side["lateral"], uw) - psi0
        out["smpl_roll_err"] = (e + 180.0) % 360.0 - 180.0
        if smpl_side["knee_axis"] is not None and smpl_side["knee_angle"] >= 20:
            a = math.degrees(math.acos(min(1.0, abs(float(np.dot(_unit(hw), _unit(smpl_side["knee_axis"])))))))
            out["smpl_knee_axis_err"] = a
    return out


def process(r, ev, Ro, ctx, smpl_legs, methods):
    """Per-leg measures for the requested methods (Z, T, T_boneZ, O)."""
    out = {}
    for s in ("l", "r"):
        pose, f = ev["legs"][s]["pose"], ctx["lf"][s]
        sm = smpl_legs[s] if smpl_legs is not None else None
        res = {}
        for m in methods:
            if m == "Z":
                phi = 0.0
                dp, dd = deltas(pose, f)
            elif m in ("T", "T_boneZ"):
                phi, w, sign, raw = candidate_phi(pose, f, "n_rest" if m == "T" else "n_bone_z")
                dp, dd, _, _ = rolled(pose, f, phi)
            elif m == "O":
                if sm is None:
                    continue
                z = leg_measures(pose, f, *deltas(pose, f), Ro, ctx["Am"], sm, ctx["psi0"][s], ctx["rest"], ctx["legs"][s]["target"], 0.0)
                phi = z["smpl_roll_err"]  # roll that nulls the calibrated SMPL roll error
                dp, dd, _, _ = rolled(pose, f, phi)
            res[m] = leg_measures(pose, f, dp, dd, Ro, ctx["Am"], sm, ctx["psi0"][s], ctx["rest"], ctx["legs"][s]["target"], phi)
            if m == "T":
                res[m]["weight"], res[m]["sign"], res[m]["raw_phi"] = w, sign, raw
        out[s] = res
    return out


def w95_any(ev, legs_m):
    """W95 any-violation with the knee-hinge term recomputed for this leg method (other terms are roll-independent)."""
    rom = any(v for k, v in ev["viol"].items() if k != "knee_hinge")
    hinge = any(legs_m[s]["hinge_w95"] is not None and legs_m[s]["hinge_w95"] > w95.PREDECLARED["knee_hinge"]["deviation_max_deg"] for s in ("l", "r"))
    return rom or hinge, rom


def collect(seq_rows, phase, root, methods, ctx, smpl):
    """Attach W96 measures to every evaluable row: r['W96'][(phase, root)] = {side: {method: measures}}."""
    for rs in seq_rows.values():
        for r in rs:
            ev = r[phase].get(root)
            if ev is None:
                continue
            if phase == "G":
                Ro = r["F"][root][1]
            else:
                Ro = r["R_smpl"] if root == "B_S0" else r["F"][root][1]
            sl = smpl.get(r["sid"], r["frame"]) if (smpl is not None and r["R_smpl"] is not None) else None
            r.setdefault("W96", {})[(phase, root)] = process(r, ev, Ro, ctx, sl, methods)


def bins_of(x):
    e = PREDECLARED["evaluation"]["flexion_bins_deg"]
    for lo, hi in zip(e[:-1], e[1:]):
        if lo <= x < hi:
            return f"{lo}-{hi}"
    return f"{e[-1]}+"


def summarize(seq_rows, phase, root, methods, gt_dirs_key="gt_legs"):
    E = PREDECLARED["evaluation"]
    rows = [r for rs in seq_rows.values() for r in rs if (phase, root) in r.get("W96", {})]
    out = {"rows": len(rows)}
    legs = [(r, s) for r in rows for s in ("l", "r")]
    for m in methods:
        if not all(m in r["W96"][(phase, root)][s] for r, s in legs):
            continue
        L = [r["W96"][(phase, root)][s][m] for r, s in legs]
        k = int(np.argmax([x["fk_parity"] for x in L]))
        o = {"fk_parity_max": float(L[k]["fk_parity"]), "fk_parity_argmax": [legs[k][0]["sid"], legs[k][0]["frame"], legs[k][1], L[k]["flex"]],
             "fk_parity_over_1e-6": int(sum(x["fk_parity"] > 1e-6 for x in L)), "by_flexion": {}, "by_stratum": {}}
        groups = defaultdict(list)
        for (r, s), x in zip(legs, L):
            groups[("flex", bins_of(x["flex"]))].append(x)
            groups[("stratum", str(r["stratum"]))].append(x)
        for (kind, key), xs in sorted(groups.items()):
            hw = [x["hinge_w95"] for x in xs if x["hinge_w95"] is not None]
            d = {"legs": len(xs), "off_hinge_abs_mean": float(np.mean([abs(x["off_hinge"]) for x in xs])),
                 "off_hinge_over_bound": float(np.mean([abs(x["off_hinge"]) > E["knee_off_hinge_max_deg"] for x in xs])),
                 "calf_twist_abs_mean": float(np.mean([abs(x["calf_twist"]) for x in xs])),
                 "calf_twist_over_bound": float(np.mean([abs(x["calf_twist"]) > E["calf_twist_max_deg"] for x in xs])),
                 "roll_abs_mean": float(np.mean([abs(x["roll"]) for x in xs])),
                 "roll_over_hip_axial_bound": float(np.mean([abs(x["roll"]) > E["hip_axial_rotation_max_deg"] for x in xs])),
                 "hinge_w95_mean": float(np.mean(hw)) if hw else None, "hinge_w95_p95": float(np.percentile(hw, 95)) if hw else None,
                 "hinge_w95_over_15": float(np.mean([h > 15 for h in hw])) if hw else None}
            pe = [abs(x["smpl_roll_err"]) for x in xs if "smpl_roll_err" in x]
            if pe:
                d["smpl_roll_err_abs_mean"], d["smpl_roll_err_abs_median"] = float(np.mean(pe)), float(np.median(pe))
            ke = [x["smpl_knee_axis_err"] for x in xs if "smpl_knee_axis_err" in x]
            if ke:
                d["smpl_knee_axis_err_mean"] = float(np.mean(ke))
            o["by_flexion" if kind == "flex" else "by_stratum"][key] = d
        # W95 any-violation (Z reproduces W95; others recompute only the hinge term)
        anys = [w95_any(r[phase][root], {sd: r["W96"][(phase, root)][sd][m] for sd in ("l", "r")}) for r in rows]
        o["w95_any_violation_rate"] = float(np.mean([a for a, _ in anys]))
        o["w95_rom_only_rate"] = float(np.mean([b for _, b in anys]))
        o["w95_any_by_stratum"] = {st: float(np.mean([a for (a, _), r in zip(anys, rows) if str(r["stratum"]) == st]))
                                   for st in STRATA if any(str(r["stratum"]) == st for r in rows)}
        # temporal
        jumps = rolls = 0
        for rs in seq_rows.values():
            for a, b in zip(rs[:-1], rs[1:]):
                if (phase, root) not in a.get("W96", {}) or (phase, root) not in b.get("W96", {}):
                    continue
                for s in ("l", "r"):
                    xa, xb = a["W96"][(phase, root)][s].get(m), b["W96"][(phase, root)][s].get(m)
                    if xa is None or xb is None:
                        continue
                    gt = w95.angle(a[gt_dirs_key][s][0], b[gt_dirs_key][s][0])
                    if gt < E["severe_jump"]["while_gt_thigh_step_below_deg"]:
                        jumps += quaternion_angle_degrees(xa["q_p"], xb["q_p"]) > E["severe_jump"]["thigh_local_step_deg"]
                        rolls += abs(xb["roll"] - xa["roll"]) > E["severe_jump"]["roll_step_deg"]
        o["severe_thigh_jumps"], o["severe_roll_steps"] = int(jumps), int(rolls)
        out[m] = o
    # paired contrasts vs Z on the SMPL roll error (leg-level, sequence clusters)
    for m in methods:
        if m == "Z" or m not in out:
            continue
        for band, test in (("flex_ge_20", lambda x: x >= 20), ("flex_lt_20", lambda x: x < 20)):
            pairs = [(r["sid"], abs(r["W96"][(phase, root)][s][m]["smpl_roll_err"]) - abs(r["W96"][(phase, root)][s]["Z"]["smpl_roll_err"]))
                     for r, s in legs if "smpl_roll_err" in r["W96"][(phase, root)][s]["Z"] and test(r["W96"][(phase, root)][s]["Z"]["flex"])]
            if pairs:
                out[f"smpl_roll_err_{m}_minus_Z_{band}"] = _brief(np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs]))
    return out


def s0_plane_diagnostics(seq_rows, root, ctx):
    """S0 knee plane vs GT knee plane (both flexed >= 20 deg), backward-knee rate, T hinge vs GT plane."""
    rows = [r for rs in seq_rows.values() for r in rs if ("S", root) in r.get("W96", {})]
    vals = defaultdict(list)
    for r in rows:
        ev = r["S"][root]
        for s, (h, k, a) in w95.LEGS.items():
            g = r["gt_legs"][s]
            ng = np.cross(g[0], g[1])
            fg = w95.angle(g[0], g[1])
            pose = ev["legs"][s]["pose"]
            if fg < 20 or pose.bend_angle_degrees() < 20 or np.linalg.norm(ng) < 1e-9:
                vals["excluded_straight"].append(1)
                continue
            Ro = r["R_smpl"] if root == "B_S0" else r["F"][root][1]
            ns = Ro @ (ctx["Am"].T @ np.cross(pose.upper, pose.forearm))
            ang = w95.angle(ns, ng)
            vals["plane_err"].append(ang)
            vals["backward"].append(ang > 90)
            for m in ("Z", "T"):
                x = r["W96"][("S", root)][s][m]
                hw = Ro @ (ctx["Am"].T @ (deltas(pose, ctx["lf"][s])[0] @ ctx["lf"][s]["n_rest"])) if m == "Z" else None
                if m == "T":
                    dp, _, _, _ = rolled(pose, ctx["lf"][s], x["roll"])
                    hw = Ro @ (ctx["Am"].T @ (dp @ ctx["lf"][s]["n_rest"]))
                vals[f"{m}_hinge_vs_gt_plane"].append(w95.angle(hw, ng))
    return {"legs_compared": len(vals["plane_err"]), "legs_excluded_straight": len(vals["excluded_straight"]),
            "s0_plane_vs_gt_plane_mean": float(np.mean(vals["plane_err"])), "s0_plane_vs_gt_plane_median": float(np.median(vals["plane_err"])),
            "backward_knee_rate": float(np.mean(vals["backward"])),
            "Z_hinge_vs_gt_plane_mean": float(np.mean(vals["Z_hinge_vs_gt_plane"])), "T_hinge_vs_gt_plane_mean": float(np.mean(vals["T_hinge_vs_gt_plane"])),
            "T_hinge_vs_gt_plane_median": float(np.median(vals["T_hinge_vs_gt_plane"])),
            "Z_hinge_vs_gt_plane_median": float(np.median(vals["Z_hinge_vs_gt_plane"]))}


# ------------------------------------------------------------------ export

def export_case(name, w, slots, ctx, title, note):
    """slots: {slot: (phase, root, method)}; the root frame and arm come from W95, the legs from the method."""
    alignment, B, head = ctx["alignment"], ctx["spine_basis"], ctx["spine_head"]
    target = ctx["target"]
    lab = lambda ph, root, m: f"{m} {'GT legs' if ph == 'G' else 'S0 legs'}, root {root}"  # noqa: E731
    frames = []
    s_off = {lab(*v): [] for v in slots.values()}
    s_roll = {lab(*v): [] for v in slots.values()}
    has_smpl = any(r["R_smpl"] is not None for r in w)
    for r in w:
        rec = {"bank_position": r["p"], "frame_index": r["frame"], "time_s": r["t"], "image": r["image"], "available": r["available"],
               "gt_valid": r["gt_arm_ok"], "runtime_status": "KNOWN" if r["s0_valid"] else "S0_FRAME_INVALID_YAW_FALLBACK", "runtime_reasons": [],
               "yaw_deg": {}, "modes": {}, "forward_cam": {}, "mode_lines": {}, "object_rotation_xyzw": {}, "extra_bone_locals_xyzw": {},
               "fk_points_armature": {}}
        if r["gt_legs"] is not None:
            rec["reference_legs_cam"] = {"left": [list(r["gt_legs"]["l"][0]), list(r["gt_legs"]["l"][1])],
                                         "right": [list(r["gt_legs"]["r"][0]), list(r["gt_legs"]["r"][1])]}
        for slot, (ph, root, m) in slots.items():
            key = lab(ph, root, m)
            if ph == "G":
                fr = r["F"].get(root)
            else:
                fr = (r["F"]["A_S0"][0], r["R_smpl"]) if root == "B_S0" else r["F"].get(root)
                if root == "B_S0" and r["R_smpl"] is None:
                    fr = None
            if fr is None:
                rec["yaw_deg"][slot] = None
                rec["mode_lines"][slot] = ["not evaluable"]
                s_off[key].append(None)
                s_roll[key].append(None)
                continue
            Rt, Ro = fr
            y = w94.compat_yaw(Rt)
            rec["yaw_deg"][slot] = math.degrees(y) if y is not None else None
            rec["forward_cam"][slot] = list(Rt[:, 1])
            rec["object_rotation_xyzw"][slot] = list(matrix_to_quaternion_xyzw(conj(alignment, Ro)))
            bones = {"spine_01": list(spine_local_xyzw(alignment, Ro.T @ Rt, B))}
            meas = r.get("W96", {}).get((ph, root))
            ev = r[ph].get(root)
            if meas is not None and all(m in meas[s] for s in ("l", "r")):
                for s in ("l", "r"):
                    bones[f"thigh_{s}"], bones[f"calf_{s}"] = list(meas[s][m]["q_p"]), list(meas[s][m]["q_d"])
                    p = ev["legs"][s]["pose"]
                    rec["fk_points_armature"].setdefault(slot, {}).update({f"thigh_{s}": list(p.mid), f"calf_{s}": list(p.end)})
            rec["extra_bone_locals_xyzw"][slot] = bones
            arm = r["arm"].get("S" if ph == "S" else "T")
            if arm is not None and arm.status == "known":
                Q = conj(alignment, Ro.T @ Rt)
                sp = lambda pt: list(np.asarray(head) + Q @ (np.asarray(pt) - np.asarray(head)))  # noqa: E731
                rec["modes"][slot] = {"proximal_local_xyzw": list(arm.proximal_local), "distal_local_xyzw": list(arm.distal_local),
                                      "mid": sp(arm.mid), "end": sp(arm.end), "root": sp(target.root_head),
                                      "object_rotation_armature_xyzw": rec["object_rotation_xyzw"][slot], "extra_bone_locals_xyzw": {"spine_01": bones["spine_01"]}}
            if meas is None or not all(m in meas[s] for s in ("l", "r")):
                rec["mode_lines"][slot] = ["legs unavailable -> previous leg keys held"]
                s_off[key].append(None)
                s_roll[key].append(None)
                continue
            Lm, Rm = meas["l"][m], meas["r"][m]
            lines = [f"knee flex L/R {Lm['flex']:4.0f}/{Rm['flex']:4.0f} | thigh roll {Lm['roll']:4.0f}/{Rm['roll']:4.0f}",
                     f"knee off-hinge L/R {Lm['off_hinge']:4.0f}/{Rm['off_hinge']:4.0f} | calf twist {Lm['calf_twist']:4.0f}/{Rm['calf_twist']:4.0f}"]
            if "smpl_roll_err" in Lm:
                lines.append(f"roll err vs SMPL L/R {Lm['smpl_roll_err']:4.0f}/{Rm['smpl_roll_err']:4.0f}")
            rec["mode_lines"][slot] = lines
            s_off[key].append(Lm["off_hinge"])
            s_roll[key].append(Lm.get("smpl_roll_err") if has_smpl else Lm["roll"])
        frames.append(rec)
    colors = {lab(*v): c for v, c in zip(slots.values(), ("#2a78d6", "#eb6834", "#1baf7a"))}
    return {"case": name, "split": ctx["split"], "sequence_id": w[0]["sid"], "start_frame": w[0]["frame"],
            "fps_sampled": 1.0 / max(w[1]["t"] - w[0]["t"], 1e-6), "frames": frames,
            "labels": {slot: lab(*v) for slot, v in slots.items()}, "short_labels": {slot: lab(*v) for slot, v in slots.items()},
            "title": f"W96 knee hinge / thigh roll {title}", "header_note": note + " | thin green = GT legs / GT arm (evaluation only)",
            "plot_series": {"panels": [{"ylabel": "left knee off-hinge angle (deg)", "series": s_off, "colors": colors},
                                       {"ylabel": "left thigh roll error vs SMPL (deg)" if has_smpl else "left thigh roll (deg)",
                                        "series": s_roll, "colors": colors}]}}


def figure(rows_test, out_path):
    """Distribution figure: W95 hinge deviation vs knee flexion (Z) and |roll error vs SMPL| by flexion (Z, T)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    legs = [r["W96"][("G", "B")][s] for r in rows_test if ("G", "B") in r.get("W96", {}) for s in ("l", "r")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=110)
    fx = np.array([x["Z"]["flex"] for x in legs])
    hz = np.array([x["Z"]["hinge_w95"] if x["Z"]["hinge_w95"] is not None else np.nan for x in legs])
    oz = np.array([abs(x["Z"]["off_hinge"]) for x in legs])
    a1.scatter(fx, hz, s=6, color="#2a78d6", alpha=0.35, linewidths=0, label="Z: W95 hinge deviation (flex >= 20)")
    a1.scatter(fx, oz, s=6, color="#eb6834", alpha=0.25, linewidths=0, label="Z: |knee off-hinge| (calf out of rig hinge plane)")
    a1.axhline(15, color="#888888", lw=1, ls="--")
    a1.text(150, 16, "W95 15 deg bound", fontsize=7, color="#555555", ha="right")
    a1.set_xlabel("knee flexion (deg)", fontsize=8)
    a1.set_ylabel("deg", fontsize=8)
    a1.set_title("3DPW TEST, GT legs, SMPL pelvis root, zero-twist (Z)", fontsize=9)
    a1.legend(fontsize=7, loc="upper right")
    edges = PREDECLARED["evaluation"]["flexion_bins_deg"]
    cx = [(lo + hi) / 2 for lo, hi in zip(edges[:-1], edges[1:])]
    for m, col in (("Z", "#2a78d6"), ("T", "#eb6834")):
        med, q1, q3 = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            v = [abs(x[m]["smpl_roll_err"]) for x in legs if lo <= x["Z"]["flex"] < hi and "smpl_roll_err" in x[m]]
            med.append(np.median(v) if v else np.nan)
            q1.append(np.percentile(v, 25) if v else np.nan)
            q3.append(np.percentile(v, 75) if v else np.nan)
        a2.plot(cx, med, color=col, lw=2, marker="o", ms=5, label=f"{m}: median")
        a2.fill_between(cx, q1, q3, color=col, alpha=0.15, linewidth=0)
        a2.annotate(m, (cx[-1], med[-1]), textcoords="offset points", xytext=(6, 0), fontsize=8, color="#333333")
    a2.set_xlabel("knee flexion bin centre (deg)", fontsize=8)
    a2.set_ylabel("|thigh roll error vs SMPL| (deg), IQR band", fontsize=8)
    a2.set_title("Independent check: rig thigh lateral axis vs SMPL thigh", fontsize=9)
    a2.legend(fontsize=7)
    for a in (a1, a2):
        a.tick_params(labelsize=7)
        a.grid(color="#e6e6e6", lw=0.6)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bank", "s0_dir", "raw_3dpw", "w87_3dpw", "hp_bank", "hp_s0", "w87_3dhp", "w93_cases", "w94_cases", "w95_cases", "rest", "calibration", "chains", "out_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
    frozen = {args.w87_3dpw: W87_PREDICTIONS["3dpw"], args.w87_3dhp: W87_PREDICTIONS["3dhp"], args.hp_s0: S0_SHA["3dhp"],
              **{args.s0_dir / f"prediction_{s}.npy": S0_SHA[s] for s in ("validation", "test")}}
    for path, digest in frozen.items():
        if sha(path) != digest:
            raise SystemExit(f"STOP: frozen input changed: {path}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rest = load_target_rest_pose(args.rest)
    alignment = derive_rig_alignment(rest, load_rig_fk_calibration(args.calibration))
    Am = quaternion_to_matrix(alignment)
    target = target_chain_geometry(rest, load_ik_chain_set(args.chains).chains[0])
    sb = rest.bones["spine_01"]
    M = np.asarray(sb.matrix_local, float)[:3, :3]
    lf = leg_frames(rest)
    ctx = {"alignment": alignment, "Am": Am, "rest": rest, "target": target, "spine_basis": M / np.linalg.norm(M, axis=0, keepdims=True),
           "spine_head": list(sb.head), "legs": w95.leg_targets(rest), "lf": lf, "psi0": rest_offsets(lf, Am)}
    report = {"schema": "worklog96_leg_knee_hinge_twist", "predeclared": PREDECLARED, "frozen_inputs": {str(k): v for k, v in frozen.items()},
              "rest_audit": {s: {"rest_knee_bend_deg": math.degrees(math.asin(float(np.linalg.norm(np.cross(lf[s]["t0"], lf[s]["c0"]))))),
                                 "n_rest_vs_bone_local_z_deg": w95.angle(lf[s]["n_rest"], lf[s]["n_bone_z"]),
                                 "smpl_rest_roll_offset_deg": ctx["psi0"][s]} for s in ("l", "r")}, "results": {}}
    # ---- rows exactly as W95 (3DPW)
    bank = load_bank(args.bank)
    s0, _ = assemble_h0(bank, {s: args.s0_dir / f"prediction_{s}.npy" for s in ("validation", "test")})
    smpl = w94.Smpl3dpw(args.raw_3dpw)
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
    sl = SmplLegs(args.raw_3dpw)
    pw = {}
    for split in ("validation", "test"):
        pw[split] = w94.build_rows(bank, s0, sequence_ids_in_split(bank, split), g3, bank.arrays, p3, ctx3, smpl=smpl)
        w95.attach(pw[split], joints3, s0, lambda r: r["R_smpl"] is not None, lambda p: bool(iv3[p][[1, 2, 3, 4, 5, 6]].all()), ctx)
        res = report["results"][f"3dpw_{split}"] = {}
        collect(pw[split], "G", "B", ("Z", "T", "T_boneZ", "O"), ctx, sl)
        res["primary_G_B"] = summarize(pw[split], "G", "B", ("Z", "T", "T_boneZ", "O"))
        for root in ("U", "A", "L"):
            collect(pw[split], "G", root, ("Z", "T"), ctx, sl)
            res[f"policy_G_{root}"] = summarize(pw[split], "G", root, ("Z", "T"))
        for root in ("B_S0", "A_S0", "U_S0", "L_S0"):
            collect(pw[split], "S", root, ("Z", "T"), ctx, sl)
            res[f"S0_{root}"] = summarize(pw[split], "S", root, ("Z", "T"))
        res["S0_plane_B_S0"] = s0_plane_diagnostics(pw[split], "B_S0", ctx)
    # ---- 3DHP (no SMPL; roots A/U/L; S0 roots)
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
    hrows = w94.build_rows(rv, hp, sorted({s.sequence_id for s in mx.samples}), gh, mx.arrays, np.load(args.w87_3dhp), ctxh,
                           gravity_fn=lambda sid: w94.hp_gravity(sid.split(":")[1]))
    tv, ivh = mx.arrays["target_valid"], mx.arrays["input_valid"]
    w95.attach(hrows, jh, hp, lambda r: bool(tv[r["p"]][[1, 2, 3, 4, 5, 6]].all()), lambda p: bool(ivh[p][[1, 2, 3, 4, 5, 6]].all()), ctx)
    res = report["results"]["3dhp"] = {}
    for root in ("A", "U", "L"):
        collect(hrows, "G", root, ("Z", "T"), ctx, None)
        res[f"policy_G_{root}"] = summarize(hrows, "G", root, ("Z", "T"))
    for root in ("A_S0", "U_S0", "L_S0"):
        collect(hrows, "S", root, ("Z", "T"), ctx, None)
        res[f"S0_{root}"] = summarize(hrows, "S", root, ("Z", "T"))
    res["S0_plane_A_S0"] = s0_plane_diagnostics(hrows, "A_S0", ctx)
    figure([r for rs in pw["test"].values() for r in rs], args.out_dir / "w96_hinge_distributions.png")
    # ---- cases (W95 windows; identical rows/timestamps) + fixed-rule worst-hinge window
    w95c = {**json.loads((args.w95_cases / "cases_3dpw.json").read_text())["cases"], **json.loads((args.w95_cases / "cases_3dhp.json").read_text())["cases"]}
    dpw = {**pw["validation"], **pw["test"]}
    cases = {}
    zt = {"FIXED": ("G", "B", "Z"), "RUNTIME": ("G", "B", "T"), "GT": ("G", "B", "O")}
    for key in ("3dpw_walking_upright", "3dpw_pelvis_stable_bend", "3dpw_seated_or_crouched", "3dpw_turn_180"):
        c = w95c[key]
        cases[key] = export_case(c["case"], w94.window_at(dpw, c["sequence_id"], c["start_frame"]), zt, {**ctx, "split": "3dpw"},
                                 "(3DPW, root B)", "Z zero-twist | T hinge-aware roll | O SMPL-roll oracle (eval only)")
    used = {(r["sid"], r["frame"]) for c in w95c.values() if c["sequence_id"].startswith("3dpw") for r in w94.window_at(dpw, c["sequence_id"], c["start_frame"])}

    def worst(w):
        if any((r["sid"], r["frame"]) in used for r in w):
            return -1
        return sum(x["Z"]["hinge_w95"] or 0.0 for r in w if ("G", "B") in r.get("W96", {}) for x in r["W96"][("G", "B")].values())
    cases["3dpw_worst_hinge"] = export_case("worst_hinge", w94.best_window(dpw, worst), zt, {**ctx, "split": "3dpw"}, "(3DPW, root B)",
                                            "Z zero-twist | T hinge-aware roll | O SMPL-roll oracle (eval only)")
    c = w95c["3dpw_bent_or_seated_s0"]
    cases["3dpw_bent_or_seated_s0"] = export_case("bent_or_seated_s0", w94.window_at(dpw, c["sequence_id"], c["start_frame"]),
                                                  {"FIXED": ("S", "B_S0", "Z"), "RUNTIME": ("S", "B_S0", "T"), "GT": ("G", "B", "T")},
                                                  {**ctx, "split": "3dpw"}, "(3DPW, S0 legs)", "S0 legs Z | S0 legs T | GT legs T (eval)")
    hz = {"FIXED": ("G", "A", "Z"), "RUNTIME": ("G", "A", "T"), "GT": ("G", "L", "T")}
    for key in ("3dhp_ts2_supine", "3dhp_ts4_lying"):
        c = w95c[key]
        cases[key] = export_case(c["case"], w94.window_at(hrows, c["sequence_id"], c["start_frame"]), hz, {**ctx, "split": "3dhp"},
                                 "(3DHP, no SMPL)", "root A: Z | root A: T | root L: T")
    c = w95c["3dhp_s0_spins_s0"]
    cases["3dhp_s0_backward_knees"] = export_case("s0_backward_knees", w94.window_at(hrows, c["sequence_id"], c["start_frame"]),
                                                  {"FIXED": ("S", "A_S0", "Z"), "RUNTIME": ("S", "A_S0", "T"), "GT": ("G", "A", "T")},
                                                  {**ctx, "split": "3dhp"}, "(3DHP, S0 legs)", "S0 legs Z | S0 legs T | GT legs T (eval)")
    report["cases"] = {k: {"sequence_id": v["sequence_id"], "start_frame": v["start_frame"], "slots": v["labels"]} for k, v in cases.items()}
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=1, default=str))
    common = {"rest_snapshot_digest": rest.digest(), "canonical_to_armature_xyzw": list(alignment), "target_chain": target.to_dict()}
    for tag in ("3dpw", "3dhp"):
        (args.out_dir / f"cases_{tag}.json").write_text(json.dumps({**common, "cases": {k: v for k, v in cases.items() if k.startswith(tag)}}, indent=1))
    print(json.dumps(report["rest_audit"], indent=1))
    print(report["cases"])


if __name__ == "__main__":
    main()
