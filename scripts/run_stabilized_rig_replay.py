#!/usr/bin/env python3
"""EXPERIMENT (Worklog 90): end-to-end rig replay of the Worklog 89 depth stabilizer on the left arm.

Opt-in hybrid replay, nothing in production changes:
  RAW         frozen J1X unit chain                       -> Worklog 61/76 IK (``measure_chain`` + ``full_arm``)
  STABILIZED  the same J1X chain -> Worklog 89 offline stabilizer (frozen config, unchanged code) -> the same IK
  G0X         frozen matched G0X chain                     -> the same IK (geometry-only reference)
  GT_REF      FrameBank GT chain                           -> the same IK (EVALUATION ONLY rotational reference)

Rig contract (unchanged Worklog 61/76): BaseRig ``upperarm_l`` rest head is the anchor, rig rest lengths L1/L2,
reach = |D| x (L1 + L2), runtime bend side, closed-form two-bone solve, zero-twist parent-relative local quaternions.
Root yaw: one constant psi for every mode (``--yaw-deg``; primary 180 = the rest torso faces the camera, so the
video-camera view of the rig is directly comparable with the RGB; 0 = facing away, reported as sensitivity).  The
arm's world (camera-frame) direction is the same for any constant psi; psi only changes the torso the local
rotations are expressed against.  The historical runtime yaw comes from H0, whose materialization uses two oracle
torso sign fields (Worklog 75), so it is not a GT-free runtime input; a constant keeps every mode identical and
GT-free.  Frames whose detector arm joints are invalid are UNAVAILABLE in every mode.

Outputs: per-row rig poses for 3DPW TEST and VALIDATION (0.1 s rows), rig-level metrics with the predeclared
``DECISION`` rule, and per-case JSON exports for the Blender replay (``render_stabilized_rig_replay_blender.py``).
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
from review_depth_motion_stabilization import select, window_stats  # noqa: E402
from run_depth_motion_stabilization import _brief, best_shift, build_populations, run_stabilizer  # noqa: E402

from framepose.connected_limb_transfer import chain_indices, chain_targets  # noqa: E402
from framepose.depth_motion_stabilizer import StabilizerConfig  # noqa: E402
from framepose.visual_depth_motion_attribution import gt_motion_stratum  # noqa: E402
from retarget.axis_utils import quaternion_conjugate, quaternion_multiply, rotate_vector_by_quaternion  # noqa: E402
from retarget.framepose_fk import derive_rig_alignment, load_rig_fk_calibration  # noqa: E402
from retarget.framepose_ik_gt_attribution import measure_chain  # noqa: E402
from retarget.framepose_target_rest import load_target_rest_pose  # noqa: E402
from retarget.framepose_two_bone_ik import load_ik_chain_set, target_chain_geometry  # noqa: E402
from retarget.framepose_visual_full_arm_ik import full_arm  # noqa: E402

CHAIN = "left_arm"
MODES = ("RAW", "STABILIZED", "G0X")
FAST_DEG_S = 120.0  # GT-reference relative-rotation speed of a "fast" pair
HOLD_DEG_S = 30.0  # a fast pair whose projected predicted speed stays below this is a false hold
DISCONTINUITY_DEG = 30.0  # predicted per-pair rotation change > 30 deg while the GT reference changes < 10 deg
DISCONTINUITY_GT_DEG = 10.0
POLE_BEND_DEG = 20.0
NEAR_STRAIGHT_DEG = 15.0
SEVERE_WRIST_CM = 5.0
DECISION = {
    "rotation_stability": "STABILIZED - RAW relative-rotation residual |rv_pred - rv_GT| (per pair, deg) < 0 with the "
                          "sequence-cluster 95% interval below 0, for upper arm AND forearm",
    "artifacts": "spurious pole-flip rate and rotation-discontinuity rate: STABILIZED - RAW <= +0.5 percentage points",
    "motion": "fast-pair projected rotation amplitude retention (S - G0X) / (RAW - G0X) >= 0.8 for upper arm and forearm, "
              "false-hold rate S - RAW <= +2 percentage points, best wrist-depth lag 0 rows",
    "position": "elbow and wrist mean error S - RAW <= +0.5 cm and severe wrist regressions (> 5 cm worse) <= 1% of frames",
    "A": "all criteria on 3DPW TEST and VALIDATION 0.1 s",
    "B": "rotation_stability on TEST and VALIDATION but another criterion fails on either",
    "C": "rotation_stability fails on TEST or VALIDATION, or an artifact rate is significantly higher (interval above 0)",
    "D": "W89 parity, rig identity, frame identity or GT-free runtime contract fails",
    "review_cases": "Worklog 89 TEST review windows (stationary_limb, historical_false_motion, walking_arm_swing, "
                    "toward_away_camera, fast_arm_movement, worst_static_regression, worst_regression_left_arm) and the same "
                    "Worklog 89 rules on VALIDATION windows of 60 rows (6 s at 0.1 s) for stationary_limb, "
                    "historical_false_motion, walking_arm_swing, toward_away_camera, fast_arm_movement, worst_regression_left_arm"}
TEST_CASES = ("stationary_limb", "historical_false_motion", "walking_arm_swing", "toward_away_camera", "fast_arm_movement",
              "worst_static_regression", "worst_regression_left_arm")
VAL_CASES = ("stationary_limb", "historical_false_motion", "walking_arm_swing", "toward_away_camera", "fast_arm_movement",
             "worst_regression_left_arm")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rotvec(q_from, q_to):
    """Rotation vector (deg) of q_to * conj(q_from) (xyzw), shortest arc."""
    r = quaternion_multiply(q_to, quaternion_conjugate(q_from))
    x, y, z, w = r
    if w < 0:
        x, y, z, w = -x, -y, -z, -w
    s = math.sqrt(x * x + y * y + z * z)
    if s < 1e-12:
        return np.zeros(3)
    return np.degrees(2 * math.atan2(s, w)) * np.array([x, y, z]) / s


def solve_rows(pop, rows, chains, target, rest, alignment, yaw):
    """Rig pose per row and mode; None when unmeasurable."""
    out = {}
    for name, (e, w) in chains.items():
        poses = {}
        for p in rows:
            m = measure_chain((0.0, 0.0, 0.0), tuple(float(v) for v in e[p]), tuple(float(v) for v in w[p]), yaw, alignment)
            poses[p] = None if m is None else full_arm(name, m, m.direction, target, rest)
        out[name] = poses
    return out


def plane_normal(pose):
    return pose.plane_normal() if pose is not None and pose.status == "known" else None


def metrics(pop, rows, poses, available, gt_ok, fwd, step):
    """Rig-level metrics for one population; STABILIZED/G0X contrasts against RAW."""
    seq_of = {p: pop.bank.samples[p].sequence_id for p in rows}
    fr = {p: int(pop.bank.samples[p].frame_index) for p in rows}
    by = defaultdict(dict)
    for p in rows:
        by[seq_of[p]][fr[p]] = p
    usable = lambda p: available[p] and gt_ok[p] and all(poses[k][p] is not None and poses[k][p].status == "known" for k in (*MODES, "GT_REF"))  # noqa: E731
    pairs = [(s, f, p, by[s][f + step]) for s, frames in by.items() for f, p in sorted(frames.items())
             if f + step in frames and usable(p) and usable(frames[f + step])]
    frames = [p for p in rows if usable(p)]
    root = np.asarray(poses["GT_REF"][frames[0]].root)
    out = {"frames": len(frames), "pairs": len(pairs), "unavailable_rows": int(sum(not available[p] for p in rows)),
           "sequences": len({seq_of[p] for p in frames}), "models": {}}
    gt = poses["GT_REF"]
    dt = np.array([pop.times[q] - pop.times[p] for _, _, p, q in pairs])
    seq = np.array([s for s, _, _, _ in pairs])
    fseq = np.array([seq_of[p] for p in frames])
    rv_gt = {b: np.array([rotvec(getattr(gt[p], f"{b}_local"), getattr(gt[q], f"{b}_local")) for _, _, p, q in pairs]) for b in ("proximal", "distal")}
    sp_gt = {b: np.linalg.norm(rv_gt[b], axis=1) / dt for b in rv_gt}
    e_t, w_t, _ = chain_targets(pop.bank.arrays["target_3d"], pop.bank.arrays["target_valid"], CHAIN)
    strata = gt_motion_stratum(w_t[[p for _, _, p, _ in pairs]], w_t[[q for _, _, _, q in pairs]])
    gt_bend = np.array([min(gt[p].bend_angle_degrees(), gt[q].bend_angle_degrees()) for _, _, p, q in pairs])
    depth = lambda pose: float(np.dot(np.asarray(pose.end) - root, fwd))  # noqa: E731
    d_gt = np.array([depth(gt[q]) - depth(gt[p]) for _, _, p, q in pairs])
    per_pair, per_frame = {}, {}
    for k in MODES:
        P = poses[k]
        o, pp, pf = {}, {}, {}
        for b, label in (("proximal", "upper_arm"), ("distal", "forearm")):
            rv = np.array([rotvec(getattr(P[p], f"{b}_local"), getattr(P[q], f"{b}_local")) for _, _, p, q in pairs])
            res = np.linalg.norm(rv - rv_gt[b], axis=1)
            sp = np.linalg.norm(rv, axis=1) / dt
            fast = sp_gt[b] >= FAST_DEG_S
            unit = rv_gt[b] / np.maximum(np.linalg.norm(rv_gt[b], axis=1, keepdims=True), 1e-9)
            proj = (rv * unit).sum(1) / dt
            disc = (np.linalg.norm(rv, axis=1) > DISCONTINUITY_DEG) & (np.linalg.norm(rv_gt[b], axis=1) < DISCONTINUITY_GT_DEG)
            o[label] = {"relative_rotation_residual_deg": float(res.mean()), "residual_p95": float(np.percentile(res, 95)),
                        "angular_speed_deg_s": float(sp.mean()), "gt_angular_speed_deg_s": float(sp_gt[b].mean()),
                        "low_motion_angular_speed_deg_s": float(sp[strata == "low"].mean()),
                        "gt_low_motion_angular_speed_deg_s": float(sp_gt[b][strata == "low"].mean()),
                        "fast_pairs": int(fast.sum()), "fast_projected_amplitude_ratio": float(proj[fast].mean() / sp_gt[b][fast].mean()) if fast.any() else None,
                        "false_hold_rate": float((proj[fast] < HOLD_DEG_S).mean()) if fast.any() else None,
                        "discontinuity_rate": float(disc.mean()),
                        "near_straight_residual_deg": float(res[gt_bend < NEAR_STRAIGHT_DEG].mean()) if (gt_bend < NEAR_STRAIGHT_DEG).any() else None}
            pp[f"{label}_residual"] = res
            pp[f"{label}_disc"] = disc.astype(float)
            pp[f"{label}_hold"] = np.where(fast, (proj < HOLD_DEG_S).astype(float), np.nan)
        # angular acceleration on consecutive pairs (deg/s^2)
        nxt = {(s, f): i for i, (s, f, _, _) in enumerate(pairs)}
        acc, acc_gt = [], []
        for i, (s, f, p, q) in enumerate(pairs):
            j = nxt.get((s, f + step))
            if j is None:
                continue
            r1 = rotvec(P[p].proximal_local, P[q].proximal_local) / dt[i]
            r2 = rotvec(P[pairs[j][2]].proximal_local, P[pairs[j][3]].proximal_local) / dt[j]
            acc.append(np.linalg.norm(r2 - r1) / dt[j])
            acc_gt.append(np.linalg.norm(rv_gt["proximal"][j] / dt[j] - rv_gt["proximal"][i] / dt[i]) / dt[j])
        o["upper_arm"]["angular_acceleration_deg_s2"] = float(np.mean(acc)) if acc else None
        o["upper_arm"]["gt_angular_acceleration_deg_s2"] = float(np.mean(acc_gt)) if acc_gt else None
        # elbow flexion and pole
        flex = np.array([abs((P[q].bend_angle_degrees() - P[p].bend_angle_degrees()) - (gt[q].bend_angle_degrees() - gt[p].bend_angle_degrees())) for _, _, p, q in pairs])
        flips = []
        for _, _, p, q in pairs:
            na, nb, ga, gb = plane_normal(P[p]), plane_normal(P[q]), plane_normal(gt[p]), plane_normal(gt[q])
            bend_ok = P[p].bend_angle_degrees() >= POLE_BEND_DEG and P[q].bend_angle_degrees() >= POLE_BEND_DEG
            gt_keeps = ga is not None and gb is not None and float(np.dot(ga, gb)) > 0
            flips.append(float(na is not None and nb is not None and bend_ok and gt_keeps and float(np.dot(na, nb)) < 0))
        o["elbow_flexion_change_residual_deg"] = float(flex.mean())
        o["spurious_pole_flip_rate"] = float(np.mean(flips))
        pp["pole_flip"] = np.array(flips)
        # depth motion of the wrist in rig space (cm)
        d = np.array([depth(P[q]) - depth(P[p]) for _, _, p, q in pairs])
        dres = np.abs(d - d_gt)
        hi = strata == "high"
        o["wrist_depth_motion_residual_cm"] = float(dres.mean())
        o["wrist_depth_motion_residual_low_cm"] = float(dres[strata == "low"].mean())
        o["wrist_depth_slope_high"] = float(np.polyfit(d_gt[hi], d[hi], 1)[0]) if hi.sum() > 10 else None
        pp["depth_residual"] = dres
        # static rig accuracy (cm, deg) on usable frames
        ew = np.array([math.dist(P[p].end, gt[p].end) for p in frames])
        ee = np.array([math.dist(P[p].mid, gt[p].mid) for p in frames])
        ua = np.array([_angle(P[p].upper, gt[p].upper) for p in frames])
        fa = np.array([_angle(P[p].forearm, gt[p].forearm) for p in frames])
        be = np.array([abs(P[p].bend_angle_degrees() - gt[p].bend_angle_degrees()) for p in frames])
        o["static"] = {"wrist_cm": float(ew.mean()), "wrist_p95_cm": float(np.percentile(ew, 95)), "elbow_cm": float(ee.mean()),
                       "upper_arm_dir_deg": float(ua.mean()), "forearm_dir_deg": float(fa.mean()), "bend_deg": float(be.mean()),
                       "ik_endpoint_residual_max": float(max(P[p].integrity["endpoint_reconstruction_residual"] for p in frames)),
                       "ik_to_fk_residual_max": float(max(P[p].integrity["ik_to_fk_residual"] for p in frames)),
                       "length_residual_max": float(max(max(P[p].integrity["upper_length_residual"], P[p].integrity["forearm_length_residual"]) for p in frames))}
        pf["wrist"], pf["elbow"] = ew, ee
        out["models"][k] = o
        per_pair[k], per_frame[k] = pp, pf
    # lag of the wrist depth trajectory (rows)
    lag = {}
    fidx = np.array(frames)
    for k in MODES:
        yk = np.full((len(pop.bank), 3), np.nan)
        yg = np.full((len(pop.bank), 3), np.nan)
        for p in frames:
            yk[p, 1] = depth(poses[k][p])
            yg[p, 1] = depth(gt[p])
        lag[k] = best_shift(pop, fidx, yk, yg)[0]
    out["lag_rows"] = lag
    con = {}
    for k in ("STABILIZED", "G0X"):
        c = {}
        for key in ("upper_arm_residual", "forearm_residual", "depth_residual"):
            c[key] = _brief(seq, per_pair[k][key] - per_pair["RAW"][key])
            c[key + "_low"] = _brief(seq[strata == "low"], (per_pair[k][key] - per_pair["RAW"][key])[strata == "low"])
            c[key + "_high"] = _brief(seq[strata == "high"], (per_pair[k][key] - per_pair["RAW"][key])[strata == "high"])
        for key in ("upper_arm_disc", "forearm_disc", "pole_flip"):
            c[key] = _brief(seq, per_pair[k][key] - per_pair["RAW"][key])
        for key in ("wrist", "elbow"):
            c[key] = _brief(fseq, per_frame[k][key] - per_frame["RAW"][key])
        c["severe_wrist_regression_rate"] = float(np.mean(per_frame[k]["wrist"] - per_frame["RAW"]["wrist"] > SEVERE_WRIST_CM))
        con[k] = c
    out["contrast_vs_RAW"] = con
    out["strata"] = {s: int((strata == s).sum()) for s in ("low", "mid", "high")}
    return out


def _angle(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))))))


def criteria(r):
    M, c = r["models"], r["contrast_vs_RAW"]["STABILIZED"]
    stab = all(c[f"{b}_residual"].get("ci95") is not None and c[f"{b}_residual"]["ci95"][1] < 0 for b in ("upper_arm", "forearm"))
    art = all(c[k]["mean"] <= 0.005 for k in ("upper_arm_disc", "forearm_disc", "pole_flip"))
    art_worse = any(c[k].get("ci95") is not None and c[k]["ci95"][0] > 0 for k in ("upper_arm_disc", "forearm_disc", "pole_flip"))
    ret = {}
    for b in ("upper_arm", "forearm"):
        a_r, a_s, a_g = (M[k][b]["fast_projected_amplitude_ratio"] for k in ("RAW", "STABILIZED", "G0X"))
        ret[b] = (a_s - a_g) / (a_r - a_g) if None not in (a_r, a_s, a_g) and abs(a_r - a_g) > 1e-9 else None
    hold = all((M["STABILIZED"][b]["false_hold_rate"] or 0) - (M["RAW"][b]["false_hold_rate"] or 0) <= 0.02 for b in ("upper_arm", "forearm"))
    motion = all(v is not None and v >= 0.8 for v in ret.values()) and hold and r["lag_rows"]["STABILIZED"] == 0
    pos = c["wrist"]["mean"] <= 0.5 and c["elbow"]["mean"] <= 0.5 and c["severe_wrist_regression_rate"] <= 0.01
    return {"rotation_stability": bool(stab), "artifacts": bool(art), "artifacts_significantly_worse": bool(art_worse),
            "amplitude_retention": ret, "false_hold_ok": bool(hold), "motion": bool(motion), "position": bool(pos),
            "all": bool(stab and art and motion and pos)}


def track_windows(pop, step, size):
    rows = sorted(pop.rows, key=lambda p: (pop.bank.samples[p].sequence_id, pop.bank.samples[p].frame_index))
    out, track = [], [rows[0]]
    for p in rows[1:]:
        a, b = pop.bank.samples[track[-1]], pop.bank.samples[p]
        if a.sequence_id == b.sequence_id and b.frame_index - a.frame_index == step:
            track.append(p)
        else:
            out.append(track)
            track = [p]
    out.append(track)
    return [np.array(t[s:s + size]) for t in out for s in range(0, max(len(t) - size, 0) + 1, 6) if len(t) >= size]


def export_case(pop, w, poses, available, gt_ok, fwd, name, split, step):
    frames = []
    root = None
    for p in w:
        s = pop.bank.samples[p]
        rec = {"bank_position": int(p), "frame_index": int(s.frame_index), "time_s": float(pop.times[p]),
               "image": s.image_reference.relative_path, "available": bool(available[p]), "gt_valid": bool(gt_ok[p]),
               "detector_min_confidence": float(pop.bank.arrays["input_2d"][p][chain_indices(CHAIN), 2].min()), "modes": {}}
        for k in (*MODES, "GT_REF"):
            pose = poses[k][p]
            ok = pose is not None and pose.status == "known" and (available[p] if k != "GT_REF" else gt_ok[p])
            if ok:
                root = pose.root
                rec["modes"][k] = {"proximal_local_xyzw": list(pose.proximal_local), "distal_local_xyzw": list(pose.distal_local),
                                   "mid": list(pose.mid), "end": list(pose.end), "bend_deg": pose.bend_angle_degrees(),
                                   "wrist_depth_cm": float(np.dot(np.asarray(pose.end) - np.asarray(pose.root), fwd))}
        frames.append(rec)
    return {"case": name, "split": split, "sequence_id": pop.bank.samples[w[0]].sequence_id, "step_frames": step,
            "fps_sampled": 30.0 / step, "root": list(root) if root is not None else None, "frames": frames}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bank", "w83_dir", "w87_dir", "w89_dir", "rig", "rest", "calibration", "chains", "out_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--yaw-deg", type=float, default=180.0)
    args = parser.parse_args()
    yaw = math.radians(args.yaw_deg)
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise SystemExit("STOP: output directory is not empty; never overwrite experimental artifacts")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    w89_val = json.loads((args.w89_dir / "validation" / "report.json").read_text())
    if _sha(args.w89_dir / "config.json") != w89_val["config_sha256"]:
        raise SystemExit("STOP: Worklog 89 config changed")
    w89_rep = json.loads((args.w89_dir / "replay" / "report.json").read_text())
    rest = load_target_rest_pose(args.rest)
    if rest.provenance["source_fbx_sha256"] != _sha(args.rig):
        raise SystemExit("STOP: Blender rest snapshot differs from current FBX bytes")
    alignment = derive_rig_alignment(rest, load_rig_fk_calibration(args.calibration))
    target = target_chain_geometry(rest, load_ik_chain_set(args.chains).chains[0])
    if target.proximal != "upperarm_l" or target.distal != "lowerarm_l":
        raise SystemExit("STOP: unexpected left-arm chain mapping")
    # camera depth axis in armature space under the constant yaw: A Rz(-psi) (0, 1, 0)
    fwd = np.asarray(rotate_vector_by_quaternion((-math.sin(-yaw), math.cos(-yaw), 0.0), alignment))
    pops = build_populations(args)
    cfg = StabilizerConfig(**json.loads((args.w89_dir / "config.json").read_text())["chains"][CHAIN]["offline"])
    report = {"schema": "worklog90_stabilized_rig_replay", "decision_contract": DECISION, "yaw_deg": args.yaw_deg,
              "identities": {"w89_config_sha256": w89_val["config_sha256"], "rig_fbx_sha256": _sha(args.rig),
                             "rest_snapshot_digest": rest.digest(), "target_chain": target.to_dict(),
                             "canonical_to_armature_xyzw": list(alignment)},
              "results": {}, "parity": {}}
    exports = {}
    for split, name, step, w89_key, size, cases in (("test", "3dpw_test", 5, ("replay", "3dpw_test"), 36, TEST_CASES),
                                                    ("validation", "3dpw_validation_0.1s", 3, ("validation", "3dpw_validation_0.1s"), 60, VAL_CASES)):
        pop = pops[name]
        e_c, w_c, timing = run_stabilizer(pop, CHAIN, cfg)
        # parity: the stabilized chains reproduce the Worklog 89 distal static error exactly
        e_t, w_t, gt_ok = chain_targets(pop.bank.arrays["target_3d"], pop.bank.arrays["target_valid"], CHAIN)
        rows_gt = np.array([p for p in pop.rows if gt_ok[p]])
        static_c = float(np.mean(100 * np.linalg.norm(w_c[rows_gt] - w_t[rows_gt], axis=1)))
        want = (w89_rep if w89_key[0] == "replay" else w89_val)["results"][CHAIN][w89_key[1]]["static"]["C"]["distal"]
        report["parity"][name] = abs(static_c - want)
        if abs(static_c - want) > 1e-9:
            raise SystemExit(f"STOP: Worklog 89 stabilized output not reproduced on {name} ({static_c} vs {want})")
        p87 = pop.pred
        chains = {"RAW": (p87[f"{CHAIN}_g1_elbow"], p87[f"{CHAIN}_g1_wrist"]), "STABILIZED": (e_c, w_c),
                  "G0X": (p87[f"{CHAIN}_g0_elbow"], p87[f"{CHAIN}_g0_wrist"]), "GT_REF": (e_t, w_t)}
        poses = solve_rows(pop, pop.rows, chains, target, rest, alignment, yaw)
        available = {p: bool(pop.bank.arrays["input_valid"][p][chain_indices(CHAIN)].all()) for p in pop.rows}
        gt_ok_d = {p: bool(gt_ok[p]) for p in pop.rows}
        r = metrics(pop, pop.rows, poses, available, gt_ok_d, fwd, step)
        r["criteria"] = criteria(r)
        r["stabilizer_timing"] = timing
        report["results"][name] = r
        print(json.dumps({"population": name, "frames": r["frames"], "pairs": r["pairs"], "criteria": r["criteria"]}), flush=True)
        # review windows (Worklog 89 rules)
        if split == "test":
            idx = json.loads((args.w89_dir / "review" / "review_index.json").read_text())
            by_key = {(pop.bank.samples[w[0]].sequence_id, int(pop.bank.samples[w[0]].frame_index)): w for w in track_windows(pop, step, size)}
            chosen = {c: by_key[(idx[c]["sequence_id"], idx[c]["start_frame"])] for c in cases}
        else:
            wins = track_windows(pop, step, size)
            mods = {"A": (p87[f"{CHAIN}_g1_elbow"], p87[f"{CHAIN}_g1_wrist"]), "B": (p87[f"{CHAIN}_g0_elbow"], p87[f"{CHAIN}_g0_wrist"]), "C": (e_c, w_c)}
            gt = (e_t, w_t, gt_ok)
            stats = {CHAIN: {(pop.bank.samples[w[0]].sequence_id, int(pop.bank.samples[w[0]].frame_index)): window_stats(pop, CHAIN, w, mods, gt) for w in wins},
                     "right_leg": {}}
            by_key = {(pop.bank.samples[w[0]].sequence_id, int(pop.bank.samples[w[0]].frame_index)): w for w in wins}
            sel = select(stats)
            chosen = {c: by_key[sel[c][1]] for c in cases if sel[c][1] is not None}
        for c, w in chosen.items():
            exports[f"{split}_{c}"] = export_case(pop, w, poses, available, gt_ok_d, fwd, c, split, step)
    report["decision"] = classify(report["results"])
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=1, default=str))
    (args.out_dir / "cases.json").write_text(json.dumps({"rest_snapshot_digest": rest.digest(), "canonical_to_armature_xyzw": list(alignment),
                                                         "yaw_deg": args.yaw_deg,
                                                         "target_chain": target.to_dict(), "cases": exports}, indent=1))
    print(json.dumps(report["decision"], indent=1))


def classify(res):
    t, v = res["3dpw_test"]["criteria"], res["3dpw_validation_0.1s"]["criteria"]
    if t["all"] and v["all"]:
        case = "A"
    elif not (t["rotation_stability"] and v["rotation_stability"]) or t["artifacts_significantly_worse"] or v["artifacts_significantly_worse"]:
        case = "C"
    else:
        case = "B"
    return {"test": t, "validation": v, "engineering_case": case}


if __name__ == "__main__":
    main()
