#!/usr/bin/env python3
"""EXPERIMENT (Worklog 94): root / pelvis / thorax orientation ownership and its observability.

W93 (frozen baseline) builds the thorax frame R_t from the shoulders and the pelvis frame R_p from the hips around the
SAME trunk axis u = normalize(thorax - pelvis); R_p owns the Armature Object (and therefore both top-level thighs) and
R_p^T R_t (a pure rotation about u, i.e. axial twist only) goes to ``spine_01``.  This script measures what that
allocation does to the skeleton, against independent references, with the thorax frame and the frozen J1X arm held
identical:

Ownership (Object frame R_o; ``spine_01`` local carries R_o^T R_t; arm chain A R_t^T v through the unchanged W76 IK):
  A  W93: R_o = R_p (hips around the trunk axis)                                     [frozen W93 F_GT]
  B  independent pelvis: R_o = SMPL pelvis (3DPW ``poses[:3]`` + ``cam_poses``; EVALUATION ONLY, 3DPW only)
  L  joint proxy: R_o = hips around the lower-trunk axis (pelvis -> SMPL spine1 / 3DHP spine joint)
  U  upright root: R_o = Rz(yaw of the hip axis); all trunk flexion and twist on ``spine_01``, legs stay camera-vertical
  C  yaw-only control: R_t = R_o = Rz(GT heading)                                    [frozen W93 Y_GT]
GT-free counterparts A_S0 (= W93 F_S0), L_S0, U_S0 use the frozen S0 joints with the W93 invalid-frame fallback.

Evaluation-only gravity references (never runtime inputs): 3DPW world +Y through ``cam_poses``; MPI-INF-3DHP test
camera calibration ``up`` / ``right`` vectors (TS1-4 studio camera cam_8, TS5-6 outdoor camera; world +Y up).
Gravity strata: upright (trunk < 30 deg, thighs < 30 deg from down), waist_bend (trunk >= 30, thighs < 30), lying
(trunk >= 60, thighs >= 60), other.  Legs are the rig's REST legs under the Object (a rigid rest-leg diagnostic, not a
leg reconstruction); GT hip->knee directions are the evaluation-only reference.
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
from run_root_motion_contact_h0_replay import _find_3dpw_pkl, _matched_oracle_sequence, _parse_3dpw_sequence_id  # noqa: E402
from run_runtime_heading_3dhp import shim  # noqa: E402
from run_torso_frame_rig_replay import NAMES, S0_SHA, arm_pose, conj, spine_local_xyzw  # noqa: E402
from run_visual_depth_motion_attribution import W87_PREDICTIONS, _brief  # noqa: E402

from framepose.bank import load_bank  # noqa: E402
from framepose.bedlam_supervision import ExternalBank  # noqa: E402
from framepose.connected_limb_transfer import chain_indices, chain_targets  # noqa: E402
from motion.runtime_heading import runtime_heading  # noqa: E402
from motion.torso_frame import (  # noqa: E402
    PELVIS, THORAX, axis_angle_degrees, compat_yaw, geodesic_degrees, matrix_to_quaternion_xyzw, quaternion_to_matrix, torso_frames,
    yaw_frame,
)
from pose.framepose_bridge import assemble_h0, build_h0_lifted_sequence, sequence_frame_positions, sequence_ids_in_split  # noqa: E402
from pose.root_orientation_diagnostic import observe_yaw  # noqa: E402
from retarget.framepose_fk import derive_rig_alignment, load_rig_fk_calibration  # noqa: E402
from retarget.framepose_target_rest import load_target_rest_pose  # noqa: E402
from retarget.framepose_two_bone_ik import load_ik_chain_set, target_chain_geometry  # noqa: E402

CHAIN = "left_arm"
SPINE, L_HIP, L_KNEE, L_ANKLE, R_HIP, R_KNEE, R_ANKLE = 7, 1, 2, 3, 4, 5, 6
CV_TO_CANON = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])  # OpenCV camera -> AnimCV camera axes
# columns: AnimCV body axes (x subject-right, y forward, z up) in SMPL template coordinates (x subject-left, y up, z forward)
SMPL_BODY = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]).T
# MPI-INF-3DHP test_util/camera_calibration (camera up / right in the capture world, world +Y up); evaluation only
HP_CALIB = {"TS1-4": {"up": (-0.208215, 0.976233, 0.06014), "right": (0.000575281, 0.0616098, -0.9981)},
            "TS5-6": {"up": (0.025272345, 0.995038509, 0.096227370), "right": (-0.939647257, -0.009210289, 0.342020929)}}
GT_OWNERS = ("A", "B", "L", "U", "C")
S0_OWNERS = ("A_S0", "L_S0", "U_S0")
STRATA = ("upright", "waist_bend", "lying", "other")
LABELS = {"A": "A: W93 ownership (hip frame on Object)", "B": "B: independent SMPL pelvis on Object (eval only)",
          "L": "L: joint proxy pelvis (hips + lower trunk)", "U": "U: upright root, trunk on spine_01",
          "C": "C: yaw-only control (GT heading)", "A_S0": "A_S0: W93 GT-free (F_S0)", "L_S0": "L_S0: joint proxy, S0 joints",
          "U_S0": "U_S0: upright root, S0 joints", "B_GRAV": "B gravity-levelled (eval only, GT camera pose)",
          "A_GRAV": "A gravity-levelled (eval only, GT camera pose)"}
SHORT = {"A": "A W93", "B": "B SMPL pelvis", "L": "L proxy", "U": "U upright root", "C": "C yaw-only", "A_S0": "A_S0",
         "L_S0": "L_S0", "U_S0": "U_S0", "B_GRAV": "B grav", "A_GRAV": "A grav"}


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def angle(a, b):
    return axis_angle_degrees(np.asarray(a, float), np.asarray(b, float))


def lower_trunk_frame(j):
    """Pelvis frame from the hips around the lower-trunk axis (pelvis -> spine joint); same validity rules as W93."""
    m = np.array(j, float)
    m[THORAX] = m[SPINE]
    return torso_frames(m)[1]


def leg_frame(j):
    """Pelvis frame from the hips around the reversed mean thigh direction ("pelvis aligned with the thighs")."""
    m = np.array(j, float)
    m[THORAX] = m[PELVIS] - 0.5 * ((m[L_KNEE] - m[L_HIP]) + (m[R_KNEE] - m[R_HIP]))
    return torso_frames(m)[1]


def upright_root(Rt, j):
    """Rz(yaw) of the hip axis projected on the camera horizontal plane (thorax right axis if the hip axis is vertical)."""
    h = np.asarray(j[R_HIP], float) - np.asarray(j[L_HIP], float)
    if math.hypot(h[0], h[1]) > 1e-9:
        return yaw_frame(math.atan2(h[1], h[0]))
    y = compat_yaw(Rt)
    return yaw_frame(y if y is not None else 0.0)


def level(g):
    """Minimal rotation taking the camera-frame gravity up g to +Z (EVALUATION ONLY)."""
    z = np.array([0.0, 0.0, 1.0])
    v, c = np.cross(g, z), float(np.dot(g, z))
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1.0 + c)


def hp_gravity(ts):
    cal = HP_CALIB["TS1-4" if ts in ("TS1", "TS2", "TS3", "TS4") else "TS5-6"]
    right, up = np.asarray(cal["right"], float), np.asarray(cal["up"], float)
    down = -up
    fwd = np.cross(right, down)
    world_up = np.array([0.0, 1.0, 0.0])
    return _unit(CV_TO_CANON @ np.array([right @ world_up, down @ world_up, fwd @ world_up]))


def stratum(trunk_g, thigh_g):
    if trunk_g is None or thigh_g is None:
        return None
    if trunk_g >= 60 and thigh_g >= 60:
        return "lying"
    if trunk_g >= 30 and thigh_g < 30:
        return "waist_bend"
    if trunk_g < 30 and thigh_g < 30:
        return "upright"
    return "other"


def rest_leg_dirs(rest, Am):
    out = {}
    for side in ("l", "r"):
        for bone in (f"thigh_{side}", f"calf_{side}"):
            b = rest.bones[bone]
            out[bone] = _unit(Am.T @ (np.asarray(b.tail, float) - np.asarray(b.head, float)))
    return out


def chordal_mean(Rs):
    U, _, Vt = np.linalg.svd(np.mean(Rs, axis=0))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


class Smpl3dpw:
    """Per-sequence 3DPW SMPL pelvis frame and gravity up in AnimCV camera axes (EVALUATION ONLY)."""

    def __init__(self, raw_root):
        self.raw_root, self.cache = raw_root, {}

    def get(self, sid, frame_index):
        if sid not in self.cache:
            name, actor = _parse_3dpw_sequence_id(sid)
            with _find_3dpw_pkl(self.raw_root, name).open("rb") as h:
                raw = pickle.load(h, encoding="latin1")
            self.cache[sid] = (np.asarray(raw["poses"][actor], float), np.asarray(raw["cam_poses"], float),
                               np.asarray(raw["campose_valid"][actor], bool))
        poses, cams, valid = self.cache[sid]
        Rwc = cams[frame_index, :3, :3]
        g = _unit(CV_TO_CANON @ Rwc @ np.array([0.0, 1.0, 0.0]))
        if not valid[frame_index]:
            return None, g
        from scipy.spatial.transform import Rotation

        Rgo = Rotation.from_rotvec(poses[frame_index, :3]).as_matrix()
        return CV_TO_CANON @ Rwc @ Rgo @ SMPL_BODY, g


def build_rows(view, preds_s0, seqs, gt_fn, arrays, pred_arm, ctx, smpl=None, gravity_fn=None):
    """Rows and filtering identical to the W93 replay (so its fixed-rule windows are reproduced exactly)."""
    alignment, target, rest = ctx["alignment"], ctx["target"], ctx["rest"]
    e_t, w_t, gt_ok = ctx["chain_gt"]
    e_j, w_j = pred_arm[f"{CHAIN}_g1_elbow"], pred_arm[f"{CHAIN}_g1_wrist"]
    out = {}
    for sid in seqs:
        pos = sequence_frame_positions(view, sid)
        heads = runtime_heading(build_h0_lifted_sequence(view, preds_s0, sid))
        gt_yaw, gt_j = gt_fn(sid, pos)
        rows = []
        for i, p in enumerate(pos):
            if gt_yaw[i] is None or heads[i].yaw_radians is None:
                continue
            j = np.asarray(gt_j[i], float)
            Rt, Rp = torso_frames(j)
            if Rt is None or Rp is None:
                continue
            js = np.asarray(preds_s0[p], float)
            Rts, Rps = torso_frames(js)
            s0_valid = Rts is not None and Rps is not None and bool(arrays["input_valid"][p][[1, 4, 11, 14]].all())
            Ys, Yg = yaw_frame(heads[i].yaw_radians), yaw_frame(gt_yaw[i])
            fi = int(view.samples[p].frame_index)
            R_smpl, g = smpl.get(sid, fi) if smpl is not None else (None, gravity_fn(sid))
            low, low_s = lower_trunk_frame(j), lower_trunk_frame(js) if s0_valid else None
            F = {"A": (Rt, Rp), "B": (Rt, R_smpl) if R_smpl is not None else None, "L": (Rt, low) if low is not None else None,
                 "U": (Rt, upright_root(Rt, j)), "C": (Yg, Yg),
                 "A_S0": (Rts, Rps) if s0_valid else (Ys, Ys),
                 "L_S0": (Rts, low_s) if s0_valid and low_s is not None else (Ys, Ys),
                 "U_S0": (Rts, upright_root(Rts, js)) if s0_valid else (Ys, Ys)}
            if g is not None:
                G = level(g)
                F["A_GRAV"] = (G @ Rt, G @ Rp)
                if R_smpl is not None:
                    F["B_GRAV"] = (G @ Rt, G @ R_smpl)
            thighs = {s: _unit(j[k] - j[h]) for s, h, k in (("l", L_HIP, L_KNEE), ("r", R_HIP, R_KNEE))}
            calves = {s: _unit(j[a] - j[k]) for s, k, a in (("l", L_KNEE, L_ANKLE), ("r", R_KNEE, R_ANKLE))}
            trunk_g = angle(Rt[:, 2], g) if g is not None else None
            thigh_g = 0.5 * (angle(thighs["l"], -g) + angle(thighs["r"], -g)) if g is not None else None
            rec = {"sid": sid, "p": int(p), "frame": fi, "t": ctx["time"](view, p), "image": view.samples[p].image_reference.relative_path,
                   "available": bool(arrays["input_valid"][p][chain_indices(CHAIN)].all()), "gt_arm_ok": bool(gt_ok[p]),
                   "s0_valid": s0_valid, "F": F, "g": g, "R_smpl": R_smpl, "low_proxy": low, "leg_proxy": leg_frame(j),
                   "low_proxy_s0": low_s, "w93_pelvis_s0": Rps if s0_valid else None,
                   "thighs": thighs, "calves": calves, "trunk_g": trunk_g, "thigh_g": thigh_g, "stratum": stratum(trunk_g, thigh_g),
                   "incl_cam": math.degrees(math.acos(max(-1.0, min(1.0, Rt[2, 2])))), "arm": {}}
            if rec["available"]:
                rec["arm"]["T"] = arm_pose(e_j[p], w_j[p], Rt, alignment, target, rest)  # shared by A, B, L, U (same thorax)
                rec["arm"]["C"] = arm_pose(e_j[p], w_j[p], Yg, alignment, target, rest)
                rec["arm"]["S"] = arm_pose(e_j[p], w_j[p], F["A_S0"][0], alignment, target, rest)  # shared by the S0 owners
                if rec["gt_arm_ok"]:
                    rec["arm"]["GT_REF"] = arm_pose(e_t[p], w_t[p], Rt, alignment, target, rest)
            rows.append(rec)
        out[sid] = rows
    return out


def arm_key(mode):
    return "C" if mode == "C" else ("S" if mode.endswith("_S0") else "T")


def thigh_error(r, Ro, legs):
    return 0.5 * sum(angle(Ro @ legs[f"thigh_{s}"], r["thighs"][s]) for s in ("l", "r"))


def metrics(seq_rows, ctx):
    legs, alignment, rest = ctx["legs"], ctx["alignment"], ctx["rest"]
    rows = [r for rs in seq_rows.values() for r in rs]
    has_b = any(r["F"]["B"] is not None for r in rows)
    out = {"rows": len(rows), "smpl_pelvis_rows": int(sum(r["F"]["B"] is not None for r in rows)),
           "gravity_rows": int(sum(r["g"] is not None for r in rows)),
           "strata_rows": {s: int(sum(r["stratum"] == s for r in rows)) for s in STRATA}}
    owners = [m for m in GT_OWNERS if m != "B" or has_b] + list(S0_OWNERS)
    # per-row quantities on rows where every compared owner is defined
    use = [r for r in rows if all(r["F"].get(m) is not None for m in owners)]
    out["compared_rows"] = len(use)
    seq = np.array([r["sid"] for r in use])
    strata = np.array([r["stratum"] for r in use])
    q = {}
    for m in owners:
        q[m] = {"thigh": np.array([thigh_error(r, r["F"][m][1], legs) for r in use]),
                "obj_tilt_g": np.array([angle(r["F"][m][1][:, 2], r["g"]) for r in use]),
                "spine_swing": np.array([math.degrees(math.acos(max(-1.0, min(1.0, (r["F"][m][1].T @ r["F"][m][0])[2, 2])))) for r in use]),
                "spine_total": np.array([geodesic_degrees(r["F"][m][1], r["F"][m][0]) for r in use])}
        if has_b:
            q[m]["pelvis_geo"] = np.array([geodesic_degrees(r["F"][m][1], r["R_smpl"]) for r in use])
            q[m]["pelvis_tilt"] = np.array([angle(r["F"][m][1][:, 2], r["R_smpl"][:, 2]) for r in use])
    ref = {"trunk_g": np.array([r["trunk_g"] for r in use]), "thigh_g": np.array([r["thigh_g"] for r in use])}
    if has_b:
        ref["smpl_tilt_g"] = np.array([angle(r["R_smpl"][:, 2], r["g"]) for r in use])
        ref["smpl_flexion"] = np.array([math.degrees(math.acos(max(-1.0, min(1.0, (r["R_smpl"].T @ r["F"]["A"][0])[2, 2])))) for r in use])
    for s in ("all",) + STRATA:
        mask = np.ones(len(use), bool) if s == "all" else strata == s
        if not mask.any():
            continue
        o = {"rows": int(mask.sum()), "reference": {k: float(v[mask].mean()) for k, v in ref.items()}}
        for m in owners:
            o[m] = {k: float(v[mask].mean()) for k, v in q[m].items()}
            o[m]["thigh_p95"] = float(np.percentile(q[m]["thigh"][mask], 95))
        base = "B" if has_b else "A"
        for m in owners:
            if m != base and not m.endswith("_S0"):
                o[f"thigh_{m}_minus_{base}"] = _brief(seq[mask], q[m]["thigh"][mask] - q[base]["thigh"][mask])
        if has_b:
            for m in ("A", "L", "U", "C"):
                o[f"pelvis_geo_{m}_minus_B"] = _brief(seq[mask], q[m]["pelvis_geo"][mask] - q["B"]["pelvis_geo"][mask])
        out[s] = o
    # upper body: thorax-carried quantities are identical across the GT owners A/B/L/U
    diff_world, diff_shoulder = 0.0, 0.0
    head = np.asarray(rest.bones["spine_01"].head, float)
    sh = np.asarray(rest.bones["upperarm_l"].head, float)
    for r in use:
        pose = r["arm"].get("T")
        if pose is None or pose.status != "known":
            continue
        worlds, shoulders = [], []
        for m in ("A", "B", "L", "U"):
            if r["F"].get(m) is None:
                continue
            Rt, Ro = r["F"][m]
            O, Qr = conj(alignment, Ro), conj(alignment, Ro.T @ Rt)
            worlds.append(O @ Qr @ np.asarray(pose.upper))
            shoulders.append(O @ (Qr @ (sh - head)))  # shoulder relative to the pelvis (spine_01 head), world
        diff_world = max(diff_world, max(float(np.abs(w - worlds[0]).max()) for w in worlds))
        diff_shoulder = max(diff_shoulder, max(float(np.abs(w - shoulders[0]).max()) for w in shoulders))
    out["upper_body_identity"] = {"world_upper_arm_max_diff": diff_world, "pelvis_relative_shoulder_max_diff_rig_units": diff_shoulder}
    # temporal: Object discontinuities (> 90 deg step while the owner's GT counterpart steps < 45 deg)
    disc = {}
    for m in owners:
        ref_m = {"A_S0": "A", "L_S0": "L", "U_S0": "U"}.get(m, "B" if has_b else "A")
        spins = fb = steps = 0
        by = defaultdict(int)
        for rs in seq_rows.values():
            for a, b in zip(rs[:-1], rs[1:]):
                if any(x["F"].get(k) is None for x in (a, b) for k in (m, ref_m)):
                    continue
                steps += 1
                if geodesic_degrees(a["F"][m][1], b["F"][m][1]) > 90 and geodesic_degrees(a["F"][ref_m][1], b["F"][ref_m][1]) < 45:
                    spins += 1
                    by[str(a["stratum"])] += 1
                    fb += int(m.endswith("_S0") and a["s0_valid"] != b["s0_valid"])
        disc[m] = {"steps": steps, "object_spins": spins, "by_stratum": dict(by), "at_fallback_switch": fb, "reference": ref_m}
    out["object_discontinuities"] = disc
    cross = defaultdict(lambda: defaultdict(int))  # W93 camera-relative posture strata vs gravity strata
    for r in rows:
        cam = "upright" if r["incl_cam"] < 45 else ("lying" if r["incl_cam"] >= 60 else "bent")
        cross[cam][str(r["stratum"])] += 1
    out["w93_camera_strata_vs_gravity_strata"] = {k: dict(v) for k, v in cross.items()}
    return out, use


def observability(rows_val, rows_test):
    """Pelvis proxies against the independent SMPL pelvis (3DPW); constant offsets fitted on VALIDATION upright rows."""
    proxies = {"W93_trunk_axis": lambda r: r["F"]["A"][1], "lower_trunk": lambda r: r["low_proxy"], "thigh_aligned": lambda r: r["leg_proxy"],
               "upright_root": lambda r: r["F"]["U"][1],
               "W93_trunk_axis_S0_joints": lambda r: r["w93_pelvis_s0"], "lower_trunk_S0_joints": lambda r: r["low_proxy_s0"]}
    out = {}
    fit_rows = [r for r in rows_val if r["R_smpl"] is not None and r["stratum"] == "upright"]
    for name, fn in proxies.items():
        offs = [fn(r).T @ r["R_smpl"] for r in fit_rows if fn(r) is not None]
        off = chordal_mean(offs)
        o = {"offset_fitted_on_val_upright_deg": geodesic_degrees(off, np.eye(3)), "fit_rows": len(offs)}
        for split, rows in (("validation", rows_val), ("test", rows_test)):
            for s in ("all",) + STRATA:
                rs = [r for r in rows if r["R_smpl"] is not None and fn(r) is not None and (s == "all" or r["stratum"] == s)]
                if not rs:
                    continue
                raw = np.array([geodesic_degrees(fn(r), r["R_smpl"]) for r in rs])
                cor = np.array([geodesic_degrees(fn(r) @ off, r["R_smpl"]) for r in rs])
                tilt = np.array([angle((fn(r) @ off)[:, 2], r["R_smpl"][:, 2]) for r in rs])
                o[f"{split}_{s}"] = {"rows": len(rs), "geodesic_raw": float(raw.mean()), "geodesic_offset_corrected": float(cor.mean()),
                                     "geodesic_offset_corrected_p95": float(np.percentile(cor, 95)), "tilt_error_corrected": float(tilt.mean()),
                                     "invalid_rows": int(sum(fn(r) is None for r in rows if r["R_smpl"] is not None))}
        out[name] = o
    # how much of the GT trunk inclination is pelvis tilt vs pelvis->thorax flexion (gravity-relative, SMPL)
    split_rows = {}
    for s in STRATA:
        rs = [r for r in rows_val + rows_test if r["R_smpl"] is not None and r["stratum"] == s]
        if rs:
            split_rows[s] = {"rows": len(rs), "trunk_tilt_g": float(np.mean([r["trunk_g"] for r in rs])),
                             "smpl_pelvis_tilt_g": float(np.mean([angle(r["R_smpl"][:, 2], r["g"]) for r in rs])),
                             "smpl_pelvis_to_thorax_flexion": float(np.mean([math.degrees(math.acos(max(-1, min(1, (r["R_smpl"].T @ r["F"]["A"][0])[2, 2])))) for r in rs])),
                             "smpl_x_vs_hip_axis": float(np.mean([angle(r["R_smpl"][:, 0], r["F"]["A"][1][:, 0]) for r in rs]))}
    out["trunk_inclination_split"] = split_rows
    return out


def camera_gravity(seq_rows):
    rows = [r for rs in seq_rows.values() for r in rs if r["g"] is not None]
    up = [r for r in rows if r["stratum"] == "upright"]
    tilt = np.array([angle(r["g"], (0, 0, 1)) for r in rows])
    o = {"rows": len(rows), "camera_tilt_vs_gravity_mean": float(tilt.mean()), "camera_tilt_p95": float(np.percentile(tilt, 95)),
         "upright_rows": len(up)}
    if up:
        o["upright_trunk_camera_relative_inclination"] = float(np.mean([r["incl_cam"] for r in up]))
        o["upright_trunk_gravity_relative_inclination"] = float(np.mean([r["trunk_g"] for r in up]))
        o["upright_camera_tilt"] = float(np.mean([angle(r["g"], (0, 0, 1)) for r in up]))
        for m in ("A", "B", "U", "C"):
            vals = [angle(r["F"][m][1][:, 2], (0, 0, 1)) for r in up if r["F"].get(m) is not None]
            if vals:
                o[f"upright_object_lean_in_camera_{m}"] = float(np.mean(vals))
            vals = [angle(r["F"][m][1][:, 2], r["g"]) for r in up if r["F"].get(m) is not None]
            if vals:
                o[f"upright_object_lean_vs_gravity_{m}"] = float(np.mean(vals))
    per_seq = defaultdict(list)
    for r in rows:
        per_seq[r["sid"]].append(angle(r["g"], (0, 0, 1)))
    o["camera_tilt_by_sequence"] = {k: float(np.mean(v)) for k, v in sorted(per_seq.items())}
    return o


def export_case(name, w, slots, ctx, title, extra=""):
    alignment, B, head, legs = ctx["alignment"], ctx["spine_basis"], ctx["spine_head"], ctx["legs"]
    target = ctx["target"]
    frames = []
    series_tilt = {SHORT[m]: [] for m in slots.values()}
    series_thigh = {SHORT[m]: [] for m in slots.values()}
    ref_tilt = {"GT trunk": [], "GT SMPL pelvis": []}

    def spun(pose, Rt, Ro):
        Q = conj(alignment, Ro.T @ Rt)
        sp = lambda pt: list(np.asarray(head) + Q @ (np.asarray(pt) - np.asarray(head)))  # noqa: E731
        return {"proximal_local_xyzw": list(pose.proximal_local), "distal_local_xyzw": list(pose.distal_local), "mid": sp(pose.mid),
                "end": sp(pose.end), "root": sp(target.root_head), "object_rotation_armature_xyzw": list(matrix_to_quaternion_xyzw(conj(alignment, Ro))),
                "extra_bone_locals_xyzw": {"spine_01": list(spine_local_xyzw(alignment, Ro.T @ Rt, B))}}

    for r in w:
        g = r["g"]
        rec = {"bank_position": r["p"], "frame_index": r["frame"], "time_s": r["t"], "image": r["image"], "available": r["available"],
               "gt_valid": r["gt_arm_ok"], "runtime_status": "KNOWN" if r["s0_valid"] else "S0_FRAME_INVALID_YAW_FALLBACK", "runtime_reasons": [],
               "yaw_deg": {}, "modes": {}, "forward_cam": {}, "mode_lines": {}, "object_rotation_xyzw": {}, "extra_bone_locals_xyzw": {},
               "reference_legs_cam": {"left": [list(r["thighs"]["l"]), list(r["calves"]["l"])], "right": [list(r["thighs"]["r"]), list(r["calves"]["r"])]}}
        for slot, m in slots.items():
            fr = r["F"].get(m)
            if fr is None:
                rec["yaw_deg"][slot] = None
                rec["mode_lines"][slot] = ["not evaluable (no reference for this row)"]
                series_tilt[SHORT[m]].append(None)
                series_thigh[SHORT[m]].append(None)
                continue
            Rt, Ro = fr
            y = compat_yaw(Rt)
            rec["yaw_deg"][slot] = math.degrees(y) if y is not None else None
            rec["forward_cam"][slot] = list(Rt[:, 1])
            rec["object_rotation_xyzw"][slot] = list(matrix_to_quaternion_xyzw(conj(alignment, Ro)))
            rec["extra_bone_locals_xyzw"][slot] = {"spine_01": list(spine_local_xyzw(alignment, Ro.T @ Rt, B))}
            pose = r["arm"].get("C" if m == "C" else ("S" if m.endswith("_S0") else "T"))
            if pose is not None and pose.status == "known":
                rec["modes"][slot] = spun(pose, Rt, Ro)
            Rm = Ro
            if m.endswith("_GRAV"):  # rendered gravity-levelled: measure in camera axes, level the GT overlays for this slot
                G = level(g)
                Rm = G.T @ Ro
                rec["reference_legs_cam"][slot] = {side: [list(G @ np.asarray(v)) for v in dirs]
                                                  for side, dirs in rec["reference_legs_cam"].items() if side in ("left", "right")}
                gp = r["arm"].get("GT_REF")
                ref_m = "B" if r["F"].get("B") is not None else "A"
                if gp is not None and gp.status == "known":
                    rec["modes"][f"GT_REF_{slot}"] = spun(gp, G @ r["F"][ref_m][0], G @ r["F"][ref_m][1])
            th = thigh_error(r, Rm, legs)
            tilt = angle(Rm[:, 2], g) if g is not None else None
            swing = math.degrees(math.acos(max(-1.0, min(1.0, (Ro.T @ Rt)[2, 2]))))
            lines = [f"object tilt (gravity) {tilt:4.0f} deg" if tilt is not None else "object tilt: no gravity ref",
                     f"spine_01 swing {swing:4.0f} deg | rest-thigh err {th:4.0f} deg"]
            if r["R_smpl"] is not None:
                lines.append(f"pelvis vs SMPL pelvis {geodesic_degrees(Rm, r['R_smpl']):5.1f} deg")
            if m.endswith("_GRAV"):
                lines.append(f"levelled by the GT camera tilt {angle(g, (0, 0, 1)):4.1f} deg")
            rec["mode_lines"][slot] = lines
            series_tilt[SHORT[m]].append(tilt)
            series_thigh[SHORT[m]].append(th)
        ref_m = "B" if r["F"].get("B") is not None else "A"
        gp = r["arm"].get("GT_REF")
        if gp is not None and gp.status == "known":
            rec["modes"]["GT_REF"] = spun(gp, *r["F"][ref_m])
        ref_tilt["GT trunk"].append(r["trunk_g"])
        ref_tilt["GT SMPL pelvis"].append(angle(r["R_smpl"][:, 2], g) if r["R_smpl"] is not None else None)
        frames.append(rec)
    colors = {SHORT[m]: c for m, c in zip(slots.values(), ("#ff8c1a", "#d61ad6", "#19b4d2"))}
    colors.update({"GT trunk": "#22aa22", "GT SMPL pelvis": "#0b5d0b"})
    if all(v is None for v in ref_tilt["GT SMPL pelvis"]):
        ref_tilt.pop("GT SMPL pelvis")
    return {"case": name, "split": ctx["split"], "sequence_id": w[0]["sid"], "start_frame": w[0]["frame"],
            "fps_sampled": 1.0 / max(w[1]["t"] - w[0]["t"], 1e-6), "frames": frames,
            "labels": {slot: LABELS[m] for slot, m in slots.items()}, "short_labels": {slot: SHORT[m] for slot, m in slots.items()},
            "title": f"W94 pelvis/torso ownership {title}",
            "header_note": "same thorax + J1X arm in A/B/L/U | REST legs | thin green = GT legs / GT arm (evaluation only)" + extra,
            "plot_series": {"panels": [{"ylabel": "Object / trunk tilt from gravity (deg)", "series": {**ref_tilt, **series_tilt}, "colors": colors},
                                       {"ylabel": "rest-thigh direction error vs GT (deg)", "series": series_thigh, "colors": colors}]}}


def _rx(a):
    c, s = math.cos(math.radians(a)), math.sin(math.radians(a))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


FIXTURES = {  # (true pelvis frame, true thorax frame, thigh direction) in camera axes; body facing +Y, forward = +Y
    "erect": (np.eye(3), np.eye(3), (0.0, 0.0, -1.0)),
    "waist_flexion_60": (np.eye(3), _rx(-60), (0.0, 0.0, -1.0)),  # pelvis upright, trunk pitched forward
    "lying_supine": (_rx(90), _rx(90), (0.0, 1.0, 0.0)),  # whole body horizontal, chest up, legs along the body
    "pelvis_tilt_20_torso_compensating": (_rx(-20), np.eye(3), (0.0, 0.0, -1.0)),
    "axial_twist_40": (np.eye(3), yaw_frame(math.radians(40)), (0.0, 0.0, -1.0)),
}


def fixture_joints(Rp, Rt, thigh):
    """17 joints from explicit frames: hips and the spine joint ride on the pelvis, shoulders on the thorax."""
    j = np.zeros((17, 3))
    j[L_HIP], j[R_HIP] = Rp @ (-0.1, 0.0, 0.0), Rp @ (0.1, 0.0, 0.0)
    j[SPINE] = Rp @ (0.0, -0.02, 0.10)
    j[THORAX] = j[SPINE] + Rt @ (0.0, 0.0, 0.35)
    j[11], j[14] = j[THORAX] + Rt @ (-0.18, 0.0, 0.0), j[THORAX] + Rt @ (0.18, 0.0, 0.0)
    for h, k, a in ((L_HIP, L_KNEE, L_ANKLE), (R_HIP, R_KNEE, R_ANKLE)):
        j[k] = j[h] + 0.45 * np.asarray(thigh)
        j[a] = j[k] + 0.42 * np.asarray(thigh)
    return j


def fixtures(legs):
    """What each ownership represents on controlled skeletons (true pelvis/thorax known by construction)."""
    out = {}
    for name, (Pp, Pt, thigh) in FIXTURES.items():
        j = fixture_joints(Pp, Pt, thigh)
        Rt, Rp = torso_frames(j)
        owners = {"A_W93": Rp, "L_lower_trunk": lower_trunk_frame(j), "U_upright_root": upright_root(Rt, j), "B_true_pelvis": Pp}
        row = {}
        for k, Ro in owners.items():
            rel = Ro.T @ Rt
            row[k] = {"object_vs_true_pelvis": geodesic_degrees(Ro, Pp), "object_tilt": angle(Ro[:, 2], (0, 0, 1)),
                      "spine_swing": math.degrees(math.acos(max(-1.0, min(1.0, rel[2, 2])))),
                      "spine_vs_true_relative": geodesic_degrees(rel, Pp.T @ Pt),
                      "rest_thigh_error": 0.5 * sum(angle(Ro @ legs[f"thigh_{s}"], thigh) for s in ("l", "r"))}
        row["thorax_frame_vs_true"] = geodesic_degrees(Rt, Pt)
        out[name] = row
    return out


def window_at(seq_rows, sid, start_frame, size=36):
    rows = seq_rows[sid]
    i = next(k for k, r in enumerate(rows) if r["frame"] == start_frame)
    return rows[i:i + size]


def best_window(seq_rows, score, size=36):
    best = None
    for sid, rows in sorted(seq_rows.items()):
        for s in range(0, max(len(rows) - size, 0) + 1, 4):
            w = rows[s:s + size]
            if len(w) == size:
                k = (score(w), sid, w[0]["frame"])
                if best is None or k > best[0]:
                    best = (k, w)
    return best[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bank", "s0_dir", "raw_3dpw", "w87_3dpw", "hp_bank", "hp_s0", "w87_3dhp", "w93_cases", "rest", "calibration", "chains", "out_dir"):
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
    ctx = {"alignment": alignment, "rest": rest, "target": target, "spine_basis": M / np.linalg.norm(M, axis=0, keepdims=True),
           "spine_head": list(sb.head), "legs": rest_leg_dirs(rest, Am)}
    report = {"schema": "worklog94_pelvis_torso_ownership", "frozen_inputs": {str(k): v for k, v in frozen.items()}, "results": {},
              "fixtures": fixtures(ctx["legs"])}
    print(json.dumps(report["fixtures"], indent=1))
    # ---- 3DPW (independent SMPL pelvis + world gravity, evaluation only)
    bank = load_bank(args.bank)
    s0, _ = assemble_h0(bank, {s: args.s0_dir / f"prediction_{s}.npy" for s in ("validation", "test")})
    smpl = Smpl3dpw(args.raw_3dpw)
    ctx3 = {**ctx, "chain_gt": chain_targets(bank.arrays["target_3d"], bank.arrays["target_valid"], CHAIN),
            "time": lambda v, p: v.samples[p].frame_index / 30.0}

    def g3(sid, pos):
        orc = _matched_oracle_sequence(sid, args.raw_3dpw, [bank.samples[p].frame_index for p in pos])
        return [o.yaw_radians for o in observe_yaw(orc)], [np.array([f.points[n].position for n in NAMES]) for f in orc.frames]
    p3 = np.load(args.w87_3dpw)
    pw = {s: build_rows(bank, s0, sequence_ids_in_split(bank, s), g3, bank.arrays, p3, ctx3, smpl=smpl) for s in ("validation", "test")}
    for s in ("validation", "test"):
        report["results"][f"3dpw_{s}"], _ = metrics(pw[s], ctx)
        report["results"][f"3dpw_{s}"]["camera_gravity"] = camera_gravity(pw[s])
    report["observability_3dpw"] = observability([r for rs in pw["validation"].values() for r in rs], [r for rs in pw["test"].values() for r in rs])
    # ---- 3DHP (no pelvis reference; calibration gravity, evaluation only)
    mx = ExternalBank.load(args.hp_bank)
    rv = shim(mx, mx.arrays["input_2d"], mx.arrays["input_valid"])
    for s, ext in zip(rv.samples, mx.samples):
        s.sample_id, s.image_reference = ext.sample_id, ext.image_reference
    g2 = np.concatenate([np.zeros_like(mx.arrays["input_2d"][..., :2]), np.ones_like(mx.arrays["input_2d"][..., 2:])], axis=-1)
    gv = shim(mx, g2, mx.arrays["target_valid"])
    ctxh = {**ctx, "chain_gt": chain_targets(mx.arrays["target_3d"], mx.arrays["target_valid"], CHAIN), "time": lambda v, p: v.samples[p].timestamp}

    def gh(sid, pos):
        return [o.yaw_radians for o in observe_yaw(build_h0_lifted_sequence(gv, mx.arrays["target_3d"], sid))], [mx.arrays["target_3d"][p] for p in pos]
    hrows = build_rows(rv, np.load(args.hp_s0), sorted({s.sequence_id for s in mx.samples}), gh, mx.arrays, np.load(args.w87_3dhp), ctxh,
                       gravity_fn=lambda sid: hp_gravity(sid.split(":")[1]))
    report["results"]["3dhp"], _ = metrics(hrows, ctx)
    report["results"]["3dhp"]["camera_gravity"] = camera_gravity(hrows)
    # ---- review cases: W93 windows (same rows/timestamps) + fixed-rule W94 windows
    w93 = json.loads(args.w93_cases.read_text())["cases"]
    cases = {}
    dpw = {**pw["validation"], **pw["test"]}
    gt3 = {"FIXED": "A", "RUNTIME": "B", "GT": "C"}
    for key in ("3dpw_bent_or_seated", "3dpw_turn_180", "3dpw_walking_upright"):
        c = w93[key]
        cases[key] = export_case(c["case"], window_at(dpw, c["sequence_id"], c["start_frame"]), gt3, {**ctx, "split": "3dpw_test"}, "(3DPW TEST)")
    c = w93["3dpw_walking_upright"]
    cases["3dpw_walking_gravity"] = export_case("walking_gravity", window_at(dpw, c["sequence_id"], c["start_frame"]),
                                                {"FIXED": "A", "RUNTIME": "B", "GT": "B_GRAV"}, {**ctx, "split": "3dpw_test"}, "(3DPW TEST)",
                                                " | B grav uses the GT camera pose")
    used = {(r["sid"], r["frame"]) for key in ("3dpw_bent_or_seated", "3dpw_turn_180", "3dpw_walking_upright")
            for r in window_at(dpw, w93[key]["sequence_id"], w93[key]["start_frame"])}
    bend = best_window(dpw, lambda w: -1 if any((r["sid"], r["frame"]) in used for r in w) else
                       sum(r["stratum"] == "waist_bend" and r["F"]["B"] is not None for r in w))
    cases["3dpw_pelvis_stable_bend"] = export_case("pelvis_stable_bend", bend, {"FIXED": "A", "RUNTIME": "B", "GT": "U"}, {**ctx, "split": "3dpw"}, "(3DPW)")
    hp3 = {"FIXED": "A", "RUNTIME": "L", "GT": "U"}
    for key in ("3dhp_ts2_supine", "3dhp_ts4_lying", "3dhp_upright_lying_transition"):
        c = w93[key]
        cases[key] = export_case(c["case"], window_at(hrows, c["sequence_id"], c["start_frame"]), hp3, {**ctx, "split": "3dhp"},
                                 "(3DHP; no independent pelvis reference)")

    def s0_spins(w):
        return sum(geodesic_degrees(a["F"]["A_S0"][1], b["F"]["A_S0"][1]) > 90 and geodesic_degrees(a["F"]["A"][1], b["F"]["A"][1]) < 45
                   for a, b in zip(w[:-1], w[1:]))
    cases["3dhp_s0_spins"] = export_case("s0_spins", best_window(hrows, s0_spins), {"FIXED": "A_S0", "RUNTIME": "U_S0", "GT": "A"},
                                         {**ctx, "split": "3dhp"}, "(3DHP, GT-free S0 owners)")
    report["cases"] = {k: {"sequence_id": v["sequence_id"], "start_frame": v["start_frame"], "slots": v["labels"]} for k, v in cases.items()}
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=1, default=str))
    common = {"rest_snapshot_digest": rest.digest(), "canonical_to_armature_xyzw": list(alignment), "target_chain": target.to_dict()}
    for tag in ("3dpw", "3dhp"):
        (args.out_dir / f"cases_{tag}.json").write_text(json.dumps({**common, "cases": {k: v for k, v in cases.items() if k.startswith(tag)}}, indent=1))
    for name, res in report["results"].items():
        print("==", name, {k: res[k] for k in ("rows", "smpl_pelvis_rows", "gravity_rows", "strata_rows", "compared_rows")})
        for s in ("all",) + STRATA:
            if s in res:
                o = res[s]
                print(f"  {s:10s} n={o['rows']:5d} ref", {k: round(v, 1) for k, v in o["reference"].items()})
                for m in GT_OWNERS + S0_OWNERS:
                    if m in o:
                        print(f"     {m:5s}", {k: round(v, 1) for k, v in o[m].items()})
                for k, v in o.items():
                    if k.startswith(("thigh_", "pelvis_geo_")) and isinstance(v, dict) and "mean" in v:
                        print(f"     {k}: {v['mean']:+.1f} {[round(x, 1) for x in v['ci95']] if v.get('ci95') else None}")
        print("  upper", res["upper_body_identity"])
        print("  disc", {m: (d["object_spins"], d["at_fallback_switch"]) for m, d in res["object_discontinuities"].items()})
        print("  camera", {k: v for k, v in res["camera_gravity"].items() if k != "camera_tilt_by_sequence"})
    print(json.dumps(report["observability_3dpw"], indent=1))
    print(report["cases"])


if __name__ == "__main__":
    main()
