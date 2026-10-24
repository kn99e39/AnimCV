#!/usr/bin/env python3
"""EXPERIMENT (Worklog 95): is an anatomical pelvis orientation still needed once both legs are articulated?

Experimental leg path (no production change): each leg is a two-bone chain thigh_{l,r} -> calf_{l,r} built with the
existing ``TwoBoneIkChain`` / ``target_chain_geometry`` contract (no end anchor: foot_{l,r} sits 0.04-0.06 rig units
off the calf tail).  The thighs are top-level bones, so their parent frame is the Armature Object R_o and the chain is
expressed as A R_o^T v (W93 ``measure_chain_full``) and solved with the unchanged W61/W76 IK (``full_arm``: observed
bend side, rest-plane fallback, zero-twist shortest-arc locals).  World thigh/shin directions are therefore the same for
every R_o; what depends on R_o is the hip local rotation, the thigh roll (zero-twist convention) and with it the knee
hinge orientation, the spine_01 rotation R_o^T R_t, and the Object (pelvis mesh) orientation.

Root policies (Object R_o; spine_01 carries R_o^T R_t; J1X left arm unchanged in the thorax frame):
  B  SMPL pelvis (EVALUATION ONLY, 3DPW)     U  hip-axis yaw       A  W93 trunk-axis frame      L  hips + lower trunk
Phase G (controlled): GT thorax, GT legs, policies B / U / A / L.
Phase S (GT-free source): frozen S0 thorax (W93 fallback) and S0 legs, policies A_S0 / U_S0 / L_S0 exactly as W94,
plus the control B_S0 = SMPL pelvis with the same S0 thorax and legs (separates sensor from factorization failures).
GT legs and gravity are used only for scoring, posture labels and evaluation overlays.

The evaluation bounds below are declared before any candidate result was read (``PREDECLARED``).
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
import run_pelvis_torso_ownership as w94  # noqa: E402
from run_root_motion_contact_h0_replay import _matched_oracle_sequence  # noqa: E402
from run_runtime_heading_3dhp import shim  # noqa: E402
from run_torso_frame_rig_replay import NAMES, S0_SHA, arm_pose, conj, spine_local_xyzw  # noqa: E402
from run_visual_depth_motion_attribution import W87_PREDICTIONS, _brief  # noqa: E402

from framepose.bank import load_bank  # noqa: E402
from framepose.bedlam_supervision import ExternalBank  # noqa: E402
from framepose.connected_limb_transfer import chain_targets  # noqa: E402
from motion.torso_frame import geodesic_degrees, matrix_to_quaternion_xyzw, quaternion_to_matrix, torso_frames, yaw_frame  # noqa: E402
from pose.framepose_bridge import assemble_h0, build_h0_lifted_sequence, sequence_ids_in_split  # noqa: E402
from pose.root_orientation_diagnostic import observe_yaw  # noqa: E402
from retarget.framepose_fk import derive_rig_alignment, load_rig_fk_calibration  # noqa: E402
from retarget.framepose_target_rest import load_target_rest_pose  # noqa: E402
from retarget.framepose_two_bone_ik import TwoBoneIkChain, load_ik_chain_set, quaternion_angle_degrees, target_chain_geometry  # noqa: E402

PREDECLARED = {
    "source": "commonly cited normal adult active range of motion (AAOS 1965; AMA Guides); approximate, EVALUATION ONLY - "
              "BaseRig and the AnimCV IK define no joint limits or twist constraints",
    "hip_deg": {"flexion_max": 120.0, "extension_max": 30.0, "abduction_max": 45.0, "adduction_max": 30.0},
    "knee_deg": {"flexion_max": 150.0},
    "spine_deg": {"flexion_max": 90.0, "extension_max": 30.0, "lateral_max": 40.0, "axial_max": 45.0},
    "knee_hinge": {"deviation_max_deg": 15.0, "evaluated_when_knee_flexion_at_least_deg": 20.0,
                   "definition": "angle between the rig knee hinge (rest bend normal carried by the posed thigh) and the "
                                 "posed knee plane normal; 0 = knee bends about the rig hinge, 180 = backwards"},
    "severe_jump": {"local_thigh_step_deg": 90.0, "while_gt_thigh_step_below_deg": 45.0},
    "acceptance": "per posture stratum: a GT-free-compatible policy is acceptable on 3DPW if its any-violation rate "
                  "exceeds the B (SMPL pelvis) rate by at most 5 percentage points and it adds no severe jumps beyond B; "
                  "on 3DHP (no pelvis reference) if its any-violation rate is at most 10%",
    "hip_angle_definition": "thigh direction d in the Object frame (x right, y forward, z up): flexion = atan2(d_y, -d_z); "
                            "abduction = asin(-d_x) left, asin(d_x) right",
    "spine_angle_definition": "R = R_o^T R_t; flexion = atan2(z_y, z_z), lateral = asin(z_x) of its z column; axial = "
                              "twist of R about z (swing-twist)",
}
LEGS = {"l": (1, 2, 3), "r": (4, 5, 6)}
G_POLICIES = ("B", "U", "A", "L")
S_POLICIES = ("B_S0", "A_S0", "U_S0", "L_S0")
LABELS = {"B": "B: SMPL pelvis Object (eval only)", "U": "U: hip-yaw Object", "A": "A: W93 trunk-axis Object",
          "L": "L: hips + lower-trunk Object", "B_S0": "B_S0: SMPL pelvis + S0 legs (eval control)",
          "A_S0": "A_S0: W93 frame from S0", "U_S0": "U_S0: hip yaw from S0", "L_S0": "L_S0: lower trunk from S0"}
SHORT = {"B": "B SMPL", "U": "U hip-yaw", "A": "A W93", "L": "L lower-trunk", "B_S0": "B_S0", "A_S0": "A_S0", "U_S0": "U_S0",
         "L_S0": "L_S0"}


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def angle(a, b):
    return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(_unit(a), _unit(b)))))))


def leg_targets(rest):
    out = {}
    for s in ("l", "r"):
        chain = TwoBoneIkChain(f"{s}_leg_w95", "hip", "knee", "ankle", f"thigh_{s}", f"calf_{s}", None)
        t = target_chain_geometry(rest, chain)
        M = np.asarray(rest.bones[f"thigh_{s}"].matrix_local, float)[:3, :3]
        out[s] = {"target": t, "basis": M / np.linalg.norm(M, axis=0, keepdims=True), "hinge": np.asarray(t.rest_bend_normal, float)}
    return out


def solve_leg(j, s, Ro, ctx):
    """Two-bone leg pose in the Object frame; None when the source chain is degenerate."""
    h, k, a = LEGS[s]
    lt = ctx["legs"][s]
    return arm_pose(np.asarray(j[k]) - j[h], np.asarray(j[a]) - j[h], Ro, ctx["alignment"], lt["target"], ctx["rest"])


def hip_angles(pose, s, Am):
    d = _unit(Am.T @ pose.upper)  # Object/body coordinates
    flex = math.degrees(math.atan2(d[1], -d[2]))
    abd = math.degrees(math.asin(max(-1.0, min(1.0, -d[0] if s == "l" else d[0]))))
    return flex, abd


def knee_hinge(pose, s, ctx):
    lt = ctx["legs"][s]
    B = lt["basis"]
    delta = B @ quaternion_to_matrix(pose.proximal_local) @ B.T  # armature-space delta of the top-level thigh
    n_rig = delta @ lt["hinge"]
    n = np.cross(pose.upper, pose.forearm)
    if np.linalg.norm(n) < 1e-9:
        return None
    return angle(n_rig, n)


def spine_angles(Ro, Rt):
    R = Ro.T @ Rt
    z = R[:, 2]
    flex = math.degrees(math.atan2(z[1], z[2]))
    lat = math.degrees(math.asin(max(-1.0, min(1.0, z[0]))))
    x, y, zq, w = matrix_to_quaternion_xyzw(R)
    tw = math.degrees(2.0 * math.atan2(zq, w))
    tw = (tw + 180.0) % 360.0 - 180.0
    return flex, lat, tw


def evaluate(j_leg, Ro, Rt, ctx, gt_legs=None):
    """Per-row articulation diagnostics for one policy (both legs required)."""
    P = PREDECLARED
    Am = ctx["Am"]
    out = {"legs": {}, "viol": defaultdict(bool)}
    for s in ("l", "r"):
        pose = solve_leg(j_leg, s, Ro, ctx)
        if pose is None or pose.status != "known":
            return None
        flex, abd = hip_angles(pose, s, Am)
        knee = pose.bend_angle_degrees()
        hinge = knee_hinge(pose, s, ctx) if knee >= P["knee_hinge"]["evaluated_when_knee_flexion_at_least_deg"] else None
        thigh_w, shin_w = Ro @ (Am.T @ pose.upper), Ro @ (Am.T @ pose.forearm)
        leg = {"pose": pose, "hip_flex": flex, "hip_abd": abd, "knee": knee, "hinge": hinge, "thigh_w": _unit(thigh_w), "shin_w": _unit(shin_w),
               "ankle_dir_w": _unit(Ro @ (Am.T @ (np.asarray(pose.end) - np.asarray(pose.root))))}
        if gt_legs is not None:
            leg["thigh_err"] = angle(leg["thigh_w"], gt_legs[s][0])
            leg["shin_err"] = angle(leg["shin_w"], gt_legs[s][1])
            leg["ankle_dir_err"] = angle(leg["ankle_dir_w"], gt_legs[s][2])
        out["legs"][s] = leg
        v = out["viol"]
        v["hip_flexion"] |= flex > P["hip_deg"]["flexion_max"]
        v["hip_extension"] |= flex < -P["hip_deg"]["extension_max"]
        v["hip_abduction"] |= abd > P["hip_deg"]["abduction_max"]
        v["hip_adduction"] |= abd < -P["hip_deg"]["adduction_max"]
        v["knee_flexion"] |= knee > P["knee_deg"]["flexion_max"]
        v["knee_hinge"] |= hinge is not None and hinge > P["knee_hinge"]["deviation_max_deg"]
    sf, sl, st = spine_angles(Ro, Rt)
    out.update({"spine_flex": sf, "spine_lat": sl, "spine_twist": st})
    v = out["viol"]
    v["spine_flexion"] = sf > P["spine_deg"]["flexion_max"]
    v["spine_extension"] = sf < -P["spine_deg"]["extension_max"]
    v["spine_lateral"] = abs(sl) > P["spine_deg"]["lateral_max"]
    v["spine_axial"] = abs(st) > P["spine_deg"]["axial_max"]
    out["any"] = any(v.values())
    out["viol"] = dict(v)
    return out


def gt_leg_dirs(j):
    return {s: (_unit(j[k] - j[h]), _unit(j[a] - j[k]), _unit(j[a] - j[h])) for s, (h, k, a) in LEGS.items()}


def attach(seq_rows, joints_gt, joints_s0, leg_valid_gt, leg_avail_s0, ctx):
    """Phase G and phase S diagnostics on the W94 rows (W94 frames reused unchanged)."""
    for sid, rows in seq_rows.items():
        for r in rows:
            j, js = joints_gt[(sid, r["p"])], joints_s0[r["p"]]
            r["gt_legs"] = gt_leg_dirs(j) if leg_valid_gt(r) else None
            r["G"], r["S"] = {}, {}
            if r["gt_legs"] is not None:
                for m in G_POLICIES:
                    if r["F"].get(m) is not None:
                        r["G"][m] = evaluate(j, r["F"][m][1], r["F"][m][0], ctx, r["gt_legs"])
            r["s0_legs_available"] = leg_avail_s0(r["p"])
            if r["gt_legs"] is not None and r["s0_legs_available"]:
                Rts = r["F"]["A_S0"][0]  # S0 thorax (W93 fallback), identical for every S0 policy
                for m in S_POLICIES:
                    if m == "B_S0":
                        if r["R_smpl"] is not None:
                            r["S"][m] = evaluate(js, r["R_smpl"], Rts, ctx, r["gt_legs"])
                    else:
                        r["S"][m] = evaluate(js, r["F"][m][1], r["F"][m][0], ctx, r["gt_legs"])


def summarize(seq_rows, phase, policies, has_b):
    rows = [r for rs in seq_rows.values() for r in rs]
    use = [r for r in rows if all(r[phase].get(m) is not None for m in policies)]
    seq = np.array([r["sid"] for r in use])
    strata = np.array([str(r["stratum"]) for r in use])
    out = {"eligible_rows": len(rows), "evaluated_rows": len(use)}
    for s in ("all",) + w94.STRATA:
        mask = np.ones(len(use), bool) if s == "all" else strata == s
        if not mask.any():
            continue
        sub = [r for r, k in zip(use, mask) if k]
        o = {"rows": len(sub)}
        for m in policies:
            ev = [r[phase][m] for r in sub]
            legs = [e["legs"][side] for e in ev for side in ("l", "r")]
            hinge = [x["hinge"] for x in legs if x["hinge"] is not None]
            o[m] = {"any_violation_rate": float(np.mean([e["any"] for e in ev])),
                    # POST-HOC (added after reading the predeclared result): ROM bounds only, knee hinge excluded
                    "rom_only_violation_rate_posthoc": float(np.mean([any(v for k, v in e["viol"].items() if k != "knee_hinge") for e in ev])),
                    "violation_rates": {k: float(np.mean([e["viol"][k] for e in ev])) for k in ev[0]["viol"]},
                    "hip_flexion_mean": float(np.mean([x["hip_flex"] for x in legs])),
                    "hip_flexion_p05_p95": [float(np.percentile([x["hip_flex"] for x in legs], q)) for q in (5, 95)],
                    "hip_abduction_mean": float(np.mean([x["hip_abd"] for x in legs])),
                    "knee_flexion_mean": float(np.mean([x["knee"] for x in legs])),
                    "knee_hinge_evaluated_legs": len(hinge),
                    "knee_hinge_deviation_mean": float(np.mean(hinge)) if hinge else None,
                    "knee_hinge_deviation_p95": float(np.percentile(hinge, 95)) if hinge else None,
                    "spine_flexion_mean": float(np.mean([e["spine_flex"] for e in ev])),
                    "spine_lateral_abs_mean": float(np.mean([abs(e["spine_lat"]) for e in ev])),
                    "spine_axial_abs_mean": float(np.mean([abs(e["spine_twist"]) for e in ev])),
                    "thigh_dir_err_mean": float(np.mean([x["thigh_err"] for x in legs])),
                    "shin_dir_err_mean": float(np.mean([x["shin_err"] for x in legs])),
                    "ankle_dir_err_mean": float(np.mean([x["ankle_dir_err"] for x in legs])),
                    "ik_residual_max": float(max(max(x["pose"].integrity["endpoint_reconstruction_residual"], x["pose"].integrity["ik_to_fk_residual"]) for x in legs))}
            if has_b and not m.startswith("B"):
                o[m]["pelvis_vs_smpl_mean"] = float(np.mean([geodesic_degrees(r["F"][m][1], r["R_smpl"]) for r in sub]))
        ref = "B" if phase == "G" else "B_S0"
        if has_b:
            for m in policies:
                if m != ref:
                    o[f"any_violation_{m}_minus_{ref}"] = _brief(seq[mask], np.array([float(r[phase][m]["any"]) - float(r[phase][ref]["any"]) for r in sub]))
                    rom = lambda e: float(any(v for k, v in e["viol"].items() if k != "knee_hinge"))  # noqa: E731
                    o[f"rom_only_{m}_minus_{ref}_posthoc"] = _brief(seq[mask], np.array([rom(r[phase][m]) - rom(r[phase][ref]) for r in sub]))
        # world leg geometry is policy-invariant (thigh is a child of the Object)
        o["world_leg_direction_max_diff_across_policies"] = float(max(
            max(float(np.abs(r[phase][m]["legs"][side]["thigh_w"] - r[phase][policies[0]]["legs"][side]["thigh_w"]).max()) for m in policies for side in ("l", "r"))
            for r in sub))
        out[s] = o
    # severe hip jumps and Object spins
    jumps = {}
    for m in policies:
        n_jump = n_spin = steps = 0
        by = defaultdict(int)
        for rs in seq_rows.values():
            for a, b in zip(rs[:-1], rs[1:]):
                if a[phase].get(m) is None or b[phase].get(m) is None:
                    continue
                steps += 1
                for side in ("l", "r"):
                    lo = quaternion_angle_degrees(a[phase][m]["legs"][side]["pose"].proximal_local, b[phase][m]["legs"][side]["pose"].proximal_local)
                    gt = angle(a["gt_legs"][side][0], b["gt_legs"][side][0])
                    if lo > PREDECLARED["severe_jump"]["local_thigh_step_deg"] and gt < PREDECLARED["severe_jump"]["while_gt_thigh_step_below_deg"]:
                        n_jump += 1
                        by[str(a["stratum"])] += 1
        jumps[m] = {"steps": steps, "severe_hip_jumps": n_jump, "by_stratum": dict(by)}
    out["temporal"] = jumps
    return out


def acceptance(summary, phase, policies, has_b):
    ref = "B" if phase == "G" else "B_S0"
    out = {}
    for s in w94.STRATA:
        if s not in summary:
            continue
        o = summary[s]
        out[s] = {}
        for m in policies:
            if m.startswith("B"):
                continue
            if has_b:
                excess = o[m]["any_violation_rate"] - o[ref]["any_violation_rate"]
                jm = summary["temporal"][m]["by_stratum"].get(s, 0)
                jr = summary["temporal"][ref]["by_stratum"].get(s, 0)
                ok = excess <= 0.05 and jm <= jr
                out[s][m] = {"excess_violation_rate": excess, "severe_hip_jumps": jm, "reference_jumps": jr, "acceptable": bool(ok)}
            else:
                out[s][m] = {"violation_rate": o[m]["any_violation_rate"], "acceptable": bool(o[m]["any_violation_rate"] <= 0.10)}
    return out


# ------------------------------------------------------------------ fixtures

def _rx(a):
    return w94._rx(a)


def fixture_skeleton(Pp, Pt, thigh, shin):
    j = np.zeros((17, 3))
    j[1], j[4] = Pp @ (-0.1, 0.0, 0.0), Pp @ (0.1, 0.0, 0.0)
    j[7] = Pp @ (0.0, -0.02, 0.10)
    j[8] = j[7] + Pt @ (0.0, 0.0, 0.35)
    j[11], j[14] = j[8] + Pt @ (-0.18, 0.0, 0.0), j[8] + Pt @ (0.18, 0.0, 0.0)
    for h, k, a in LEGS.values():
        j[k] = j[h] + 0.45 * _unit(thigh)
        j[a] = j[k] + 0.42 * _unit(shin)
    return j


def fixture_set():
    down = (0.0, 0.0, -1.0)
    F = {"upright_neutral": (np.eye(3), np.eye(3), down, down),
         "stable_pelvis_waist_bend_60": (np.eye(3), _rx(-60), down, down),
         "bend_with_flexed_hips_40_plus_30": (_rx(-40), _rx(-70), down, down),
         "lying_supine": (_rx(90), _rx(90), (0.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
         "lying_prone": (_rx(-90), _rx(-90), (0.0, -1.0, 0.0), (0.0, -1.0, 0.0)),
         "crouch": (_rx(-30), _rx(-45), (0.0, 0.95, -0.3), (0.0, -0.45, -0.9)),
         "body_turn_90": (yaw_frame(math.radians(90)), yaw_frame(math.radians(90)), down, down)}
    for t in range(0, 91, 15):  # upright -> lying transition (whole body tilts back)
        F[f"transition_{t:02d}"] = (_rx(t), _rx(t), _rx(t) @ np.array(down), _rx(t) @ np.array(down))
    return F


def fixtures(ctx):
    out = {}
    for name, (Pp, Pt, thigh, shin) in fixture_set().items():
        j = fixture_skeleton(Pp, Pt, thigh, shin)
        Rt, Rp = torso_frames(j)
        pol = {"B": Pp, "U": w94.upright_root(Rt, j), "A": Rp, "L": w94.lower_trunk_frame(j)}
        row, ankles = {}, []
        for m, Ro in pol.items():
            e = evaluate(j, Ro, Rt, ctx, gt_leg_dirs(j))
            ankles.append(np.concatenate([e["legs"][s]["ankle_dir_w"] for s in "lr"]))
            row[m] = {"object_vs_true_pelvis": geodesic_degrees(Ro, Pp), "hip_flex_l": e["legs"]["l"]["hip_flex"], "hip_abd_l": e["legs"]["l"]["hip_abd"],
                      "knee_l": e["legs"]["l"]["knee"], "knee_hinge_l": e["legs"]["l"]["hinge"], "spine_flex": e["spine_flex"],
                      "spine_lat": e["spine_lat"], "spine_twist": e["spine_twist"], "violations": sorted(k for k, v in e["viol"].items() if v)}
        row["world_ankle_direction_max_diff_across_policies"] = float(max(np.abs(a - ankles[0]).max() for a in ankles))
        out[name] = row
    return out


# ------------------------------------------------------------------ export

def export_case(name, w, phase, slots, ctx, title):
    alignment, B, head, Am = ctx["alignment"], ctx["spine_basis"], ctx["spine_head"], ctx["Am"]
    target = ctx["target"]
    frames = []
    hip_series = {SHORT[m]: [] for m in slots.values()}
    spine_series = {SHORT[m]: [] for m in slots.values()}

    def frame_of(r, m):
        if phase == "G":
            return r["F"].get(m)
        if m == "B_S0":
            return (r["F"]["A_S0"][0], r["R_smpl"]) if r["R_smpl"] is not None else None
        return r["F"].get(m)

    for r in w:
        rec = {"bank_position": r["p"], "frame_index": r["frame"], "time_s": r["t"], "image": r["image"], "available": r["available"],
               "gt_valid": r["gt_arm_ok"], "runtime_status": "KNOWN" if r["s0_valid"] else "S0_FRAME_INVALID_YAW_FALLBACK", "runtime_reasons": [],
               "yaw_deg": {}, "modes": {}, "forward_cam": {}, "mode_lines": {}, "object_rotation_xyzw": {}, "extra_bone_locals_xyzw": {},
               "fk_points_armature": {}}
        if r["gt_legs"] is not None:
            rec["reference_legs_cam"] = {"left": [list(r["gt_legs"]["l"][0]), list(r["gt_legs"]["l"][1])],
                                         "right": [list(r["gt_legs"]["r"][0]), list(r["gt_legs"]["r"][1])]}
        for slot, m in slots.items():
            fr = frame_of(r, m)
            ev = r[phase].get(m)
            if fr is None:
                rec["yaw_deg"][slot] = None
                rec["mode_lines"][slot] = ["not evaluable for this row"]
                hip_series[SHORT[m]].append(None)
                spine_series[SHORT[m]].append(None)
                continue
            Rt, Ro = fr
            y = w94.compat_yaw(Rt)
            rec["yaw_deg"][slot] = math.degrees(y) if y is not None else None
            rec["forward_cam"][slot] = list(Rt[:, 1])
            rec["object_rotation_xyzw"][slot] = list(matrix_to_quaternion_xyzw(conj(alignment, Ro)))
            bones = {"spine_01": list(spine_local_xyzw(alignment, Ro.T @ Rt, B))}
            if ev is not None:
                for side in ("l", "r"):
                    p = ev["legs"][side]["pose"]
                    bones[f"thigh_{side}"], bones[f"calf_{side}"] = list(p.proximal_local), list(p.distal_local)
                    rec["fk_points_armature"].setdefault(slot, {}).update({f"thigh_{side}": list(p.mid), f"calf_{side}": list(p.end)})
            rec["extra_bone_locals_xyzw"][slot] = bones
            arm = r["arm"].get("S" if phase == "S" else "T")
            if arm is not None and arm.status == "known":
                Q = conj(alignment, Ro.T @ Rt)
                sp = lambda pt: list(np.asarray(head) + Q @ (np.asarray(pt) - np.asarray(head)))  # noqa: E731
                rec["modes"][slot] = {"proximal_local_xyzw": list(arm.proximal_local), "distal_local_xyzw": list(arm.distal_local),
                                      "mid": sp(arm.mid), "end": sp(arm.end), "root": sp(target.root_head),
                                      "object_rotation_armature_xyzw": rec["object_rotation_xyzw"][slot],
                                      "extra_bone_locals_xyzw": {"spine_01": bones["spine_01"]}}
            if ev is None:
                rec["mode_lines"][slot] = ["legs unavailable -> previous leg keys held"]
                hip_series[SHORT[m]].append(None)
                spine_series[SHORT[m]].append(spine_angles(Ro, Rt)[0])
                continue
            L, Rr = ev["legs"]["l"], ev["legs"]["r"]
            hinge = [x["hinge"] for x in (L, Rr) if x["hinge"] is not None]
            viol = sorted(k for k, v in ev["viol"].items() if v)
            lines = [f"hip flex L/R {L['hip_flex']:4.0f}/{Rr['hip_flex']:4.0f} | spine flex {ev['spine_flex']:4.0f}",
                     f"knee hinge dev {max(hinge):4.0f} deg" if hinge else "knee hinge: straight legs",
                     ("ROM: " + ",".join(viol)) if viol else "ROM: ok"]
            if r["R_smpl"] is not None and not m.startswith("B"):
                lines.append(f"pelvis vs SMPL {geodesic_degrees(Ro, r['R_smpl']):4.0f} deg")
            rec["mode_lines"][slot] = lines
            hip_series[SHORT[m]].append(L["hip_flex"])
            spine_series[SHORT[m]].append(ev["spine_flex"])
        frames.append(rec)
    colors = {SHORT[m]: c for m, c in zip(slots.values(), ("#ff8c1a", "#d61ad6", "#19b4d2"))}
    n = len(frames)
    P = PREDECLARED
    return {"case": name, "split": ctx["split"], "sequence_id": w[0]["sid"], "start_frame": w[0]["frame"],
            "fps_sampled": 1.0 / max(w[1]["t"] - w[0]["t"], 1e-6), "frames": frames,
            "labels": {slot: LABELS[m] for slot, m in slots.items()}, "short_labels": {slot: SHORT[m] for slot, m in slots.items()},
            "title": f"W95 articulated legs {title}",
            "header_note": ("GT thorax + GT legs (controlled)" if phase == "G" else "S0 thorax + S0 legs (GT-free source)") +
                           " | J1X arm | thin green = GT legs / GT arm (evaluation only)",
            "plot_series": {"panels": [{"ylabel": "left hip flexion vs Object (deg)", "colors": {**colors, "ROM ext limit": "#999999", "ROM flex limit": "#999999"},
                                        "series": {**hip_series, "ROM ext limit": [-P["hip_deg"]["extension_max"]] * n, "ROM flex limit": [P["hip_deg"]["flexion_max"]] * n}},
                                       {"ylabel": "spine_01 flexion (deg)", "colors": {**colors, "ROM ext limit": "#999999"},
                                        "series": {**spine_series, "ROM ext limit": [-P["spine_deg"]["extension_max"]] * n}}]}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bank", "s0_dir", "raw_3dpw", "w87_3dpw", "hp_bank", "hp_s0", "w87_3dhp", "w93_cases", "w94_cases", "rest", "calibration", "chains", "out_dir"):
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
    ctx = {"alignment": alignment, "Am": Am, "rest": rest, "target": target, "spine_basis": M / np.linalg.norm(M, axis=0, keepdims=True),
           "spine_head": list(sb.head), "legs": leg_targets(rest)}
    ctx["legs_rest"] = w94.rest_leg_dirs(rest, Am)
    report = {"schema": "worklog95_pelvis_frame_leg_articulation", "predeclared": PREDECLARED,
              "frozen_inputs": {str(k): v for k, v in frozen.items()},
              "leg_chains": {s: v["target"].to_dict() for s, v in ctx["legs"].items()}, "fixtures": fixtures(ctx), "results": {}}
    # ---- 3DPW
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
    pw = {}
    for split in ("validation", "test"):
        pw[split] = w94.build_rows(bank, s0, sequence_ids_in_split(bank, split), g3, bank.arrays, p3, ctx3, smpl=smpl)
        attach(pw[split], joints3, s0, lambda r: r["R_smpl"] is not None, lambda p: bool(iv3[p][[1, 2, 3, 4, 5, 6]].all()), ctx)
        report["results"][f"3dpw_{split}"] = {ph: summarize(pw[split], ph, pol, True) for ph, pol in (("G", G_POLICIES), ("S", S_POLICIES))}
        for ph, pol in (("G", G_POLICIES), ("S", S_POLICIES)):
            report["results"][f"3dpw_{split}"][f"acceptance_{ph}"] = acceptance(report["results"][f"3dpw_{split}"][ph], ph, pol, True)
    # ---- 3DHP
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
    attach(hrows, jh, hp, lambda r: bool(tv[r["p"]][[1, 2, 3, 4, 5, 6]].all()), lambda p: bool(ivh[p][[1, 2, 3, 4, 5, 6]].all()), ctx)
    gpol, spol = ("U", "A", "L"), ("A_S0", "U_S0", "L_S0")
    report["results"]["3dhp"] = {"G": summarize(hrows, "G", gpol, False), "S": summarize(hrows, "S", spol, False)}
    report["results"]["3dhp"]["acceptance_G"] = acceptance(report["results"]["3dhp"]["G"], "G", gpol, False)
    report["results"]["3dhp"]["acceptance_S"] = acceptance(report["results"]["3dhp"]["S"], "S", spol, False)
    # upper body invariance across the GT policies (thorax + J1X arm unchanged)
    worst = 0.0
    for r in [r for rs in pw["test"].values() for r in rs]:
        pose = r["arm"].get("T")
        if pose is None or pose.status != "known":
            continue
        ws = [conj(alignment, r["F"][m][1]) @ conj(alignment, r["F"][m][1].T @ r["F"][m][0]) @ np.asarray(pose.upper) for m in G_POLICIES if r["F"].get(m) is not None]
        worst = max(worst, max(float(np.abs(x - ws[0]).max()) for x in ws))
    report["upper_body_world_arm_max_diff_3dpw_test"] = worst
    # ---- review cases (W93/W94 windows, identical rows and timestamps) + one fixed-rule seated/crouched window
    w93c = json.loads(args.w93_cases.read_text())["cases"]
    w94c = {**json.loads((args.w94_cases / "cases_3dpw.json").read_text())["cases"], **json.loads((args.w94_cases / "cases_3dhp.json").read_text())["cases"]}
    dpw = {**pw["validation"], **pw["test"]}
    cases = {}
    gslots3 = {"FIXED": "B", "RUNTIME": "U", "GT": "A"}
    for key in ("3dpw_bent_or_seated", "3dpw_pelvis_stable_bend", "3dpw_walking_upright", "3dpw_turn_180"):
        c = w94c[key]
        cases[key] = export_case(c["case"], w94.window_at(dpw, c["sequence_id"], c["start_frame"]), "G", gslots3, {**ctx, "split": "3dpw"}, "(3DPW, GT legs)")
    used = {(r["sid"], r["frame"]) for c in w94c.values() if c["sequence_id"].startswith("3dpw")
            for r in w94.window_at(dpw, c["sequence_id"], c["start_frame"])}
    seated = w94.best_window(dpw, lambda w: -1 if any((r["sid"], r["frame"]) in used for r in w) else
                             sum(r["stratum"] == "other" and r["G"].get("B") is not None and r["G"]["B"]["legs"]["l"]["knee"] >= 60 for r in w))
    cases["3dpw_seated_or_crouched"] = export_case("seated_or_crouched", seated, "G", gslots3, {**ctx, "split": "3dpw"}, "(3DPW, GT legs)")
    c = w94c["3dpw_bent_or_seated"]
    cases["3dpw_bent_or_seated_s0"] = export_case("bent_or_seated_s0", w94.window_at(dpw, c["sequence_id"], c["start_frame"]), "S",
                                                  {"FIXED": "B_S0", "RUNTIME": "U_S0", "GT": "A_S0"}, {**ctx, "split": "3dpw"}, "(3DPW, S0 legs)")
    gslotsh = {"FIXED": "A", "RUNTIME": "U", "GT": "L"}
    for key in ("3dhp_ts2_supine", "3dhp_ts4_lying", "3dhp_upright_lying_transition"):
        c = w93c[key]
        cases[key] = export_case(c["case"], w94.window_at(hrows, c["sequence_id"], c["start_frame"]), "G", gslotsh, {**ctx, "split": "3dhp"},
                                 "(3DHP, GT legs; no pelvis reference)")
    for key, src in (("3dhp_ts2_supine_s0", w93c["3dhp_ts2_supine"]), ("3dhp_s0_spins_s0", w94c["3dhp_s0_spins"])):
        cases[key] = export_case(key[5:], w94.window_at(hrows, src["sequence_id"], src["start_frame"]), "S",
                                 {"FIXED": "A_S0", "RUNTIME": "U_S0", "GT": "L_S0"}, {**ctx, "split": "3dhp"}, "(3DHP, S0 legs)")
    report["cases"] = {k: {"sequence_id": v["sequence_id"], "start_frame": v["start_frame"], "slots": v["labels"]} for k, v in cases.items()}
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=1, default=str))
    common = {"rest_snapshot_digest": rest.digest(), "canonical_to_armature_xyzw": list(alignment), "target_chain": target.to_dict()}
    for tag in ("3dpw", "3dhp"):
        (args.out_dir / f"cases_{tag}.json").write_text(json.dumps({**common, "cases": {k: v for k, v in cases.items() if k.startswith(tag)}}, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k not in ("results", "fixtures", "leg_chains", "predeclared")}, indent=1, default=str))


if __name__ == "__main__":
    main()
