#!/usr/bin/env python3
"""EXPERIMENT (Worklog 93): does a full 3D torso frame improve rig animation that yaw-only orientation cannot express?

Identical frozen inputs in every mode -- the J1X left-arm chain (Worklog 87 predictions, camera frame) through the
unchanged W61/W76 two-bone IK -- and the orientation representation is the only change:
  Y_S0   yaw-only, psi = W91 GT-free runtime heading (S0 + current_root_orientation)       [A: current runtime]
  Y_GT   yaw-only, psi = GT heading (EVALUATION ONLY; isolates the representation)
  F_GT   full torso frame from GT joints (EVALUATION ONLY)                                 [B: oracle full frame]
  F_S0   full torso frame from frozen S0 joints (GT-free; invalid frame -> Y_S0 fallback)  [C: runtime candidate]
Frames: ``motion.torso_frame`` (thorax frame R_t carries the arms; pelvis frame R_p owns the Armature Object; the
trunk twist R_p^T R_t goes to the top-level ``spine_01`` bone).  For yaw-only modes R_t = R_p = Rz(psi).
Arm chain in the torso frame: A R_t^T v (the yaw-only contract is the special case R_t = Rz(psi)).

Reference (EVALUATION ONLY): the GT left-arm chain expressed in the GT thorax frame -> anatomical local rotations.
Postures (evaluation-only strata) by the GT trunk inclination: upright < 45, bent 45-60, lying >= 60 deg.
Populations: 3DPW VALIDATION / TEST and MPI-INF-3DHP TEST.  Review windows are exported for the Worklog 91 renderer.
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
from run_root_motion_contact_h0_replay import _matched_oracle_sequence  # noqa: E402
from run_runtime_heading_3dhp import shim  # noqa: E402
from run_visual_depth_motion_attribution import W87_PREDICTIONS, _brief  # noqa: E402

from framepose.bank import load_bank  # noqa: E402
from framepose.bedlam_supervision import ExternalBank  # noqa: E402
from framepose.connected_limb_transfer import chain_indices, chain_targets  # noqa: E402
from motion.runtime_heading import runtime_heading  # noqa: E402
from motion.torso_frame import (  # noqa: E402
    axis_angle_degrees, compat_yaw, geodesic_degrees, inclination_degrees, matrix_to_quaternion_xyzw, quaternion_to_matrix,
    torso_frames, yaw_frame,
)
from motion.vlm_heading_sign_advisor import parse_answer  # noqa: E402
from pose.framepose_bridge import assemble_h0, build_h0_lifted_sequence, sequence_frame_positions, sequence_ids_in_split  # noqa: E402
from pose.root_orientation_diagnostic import observe_yaw  # noqa: E402
from retarget.axis_utils import rotate_vector_by_quaternion  # noqa: E402
from retarget.framepose_fk import derive_rig_alignment, load_rig_fk_calibration  # noqa: E402
from retarget.framepose_ik_gt_attribution import ChainMeasure, _norm  # noqa: E402
from retarget.framepose_target_rest import load_target_rest_pose  # noqa: E402
from retarget.framepose_two_bone_ik import (  # noqa: E402
    DEGENERATE_BEND_SINE, load_ik_chain_set, observed_bend_side, quaternion_angle_degrees, source_chain_geometry, target_chain_geometry,
)
from retarget.framepose_visual_full_arm_ik import full_arm  # noqa: E402

CHAIN = "left_arm"
MODES = ("Y_S0", "Y_GT", "F_GT", "F_S0")
NAMES = ("pelvis", "left_hip", "left_knee", "left_ankle", "right_hip", "right_knee", "right_ankle", "spine", "thorax", "neck",
         "head", "left_shoulder", "left_elbow", "left_wrist", "right_shoulder", "right_elbow", "right_wrist")
S0_SHA = {"validation": "402d70a2bed79ae31919ff86905e19c8972108ea988d6cfa5deac514562132b9",
          "test": "4357c7bdb46d81e7712269b08c45f39452d7a46e697a0c26fdb70d65fe1f4996",
          "3dhp": "a78adeb0ed0391b26d4c378e0ea15ce7d1e1eaf71dfcd9a384bb6dd6e3c6e420"}
GATE = {"material": "F_GT minus Y_GT upper-arm local rotation error on lying + bent rows <= -10 deg mean with the sequence-cluster "
                    "interval below 0, and upright rows not worse by more than +1 deg",
        "stop": "otherwise stop the GT-free branch (report the dominant remaining owner)"}


def measure_chain_full(mid, end, R, alignment):
    """W62 ``measure_chain`` with the yaw-only transform A Rz(-psi) generalized to A R^T (root at the origin)."""
    aq = alignment
    upper = rotate_vector_by_quaternion(tuple(R.T @ np.asarray(mid, float)), aq)
    lower = rotate_vector_by_quaternion(tuple(R.T @ (np.asarray(end, float) - np.asarray(mid, float))), aq)
    geometry = source_chain_geometry(upper, lower)
    if geometry is None or geometry.endpoint_direction is None:
        return None
    side = observed_bend_side(geometry)
    return ChainMeasure(upper, lower, _norm(upper), _norm(lower), _norm(upper) / _norm(lower), geometry.reach_fraction,
                        geometry.endpoint_direction, geometry.bend_angle_degrees, geometry.bend_sine < DEGENERATE_BEND_SINE, side)


def arm_pose(mid, end, R, alignment, target, rest):
    m = measure_chain_full(mid, end, R, alignment)
    return None if m is None else full_arm("arm", m, m.direction, target, rest)


def conj(alignment, R):
    """Armature-space rotation A R A^-1 as a 3x3 matrix."""
    A = quaternion_to_matrix(alignment)
    return A @ R @ A.T


def spine_local_xyzw(alignment, R_rel, spine_rest_basis):
    """Pose-bone local rotation of top-level spine_01 that rotates the armature by A R_rel A^-1 about its head."""
    Q = conj(alignment, R_rel)
    B = spine_rest_basis
    return matrix_to_quaternion_xyzw(B.T @ Q @ B)


def gt_label_from_frame(R):
    """BACK when the chest normal (frame +Y) points up the camera Z axis."""
    return "BACK" if R[2, 1] > 0 else "STOMACH"


def build_rows(name, view, preds_s0, seqs, gt_joints_fn, arrays, pred_arm, ctx):
    alignment, target, rest = ctx["alignment"], ctx["target"], ctx["rest"]
    e_t, w_t, gt_ok = chain_targets(arrays["target_3d"], arrays["target_valid"], CHAIN) if name.startswith("3dhp") else ctx["chain_gt"]
    e_j, w_j = pred_arm[f"{CHAIN}_g1_elbow"], pred_arm[f"{CHAIN}_g1_wrist"]
    seq_rows = {}
    for sid in seqs:
        pos = sequence_frame_positions(view, sid)
        lifted = build_h0_lifted_sequence(view, preds_s0, sid)
        heads = runtime_heading(lifted)
        gt_yaw, gt_j = gt_joints_fn(sid, pos)
        rows = []
        for i, p in enumerate(pos):
            if gt_yaw[i] is None or heads[i].yaw_radians is None:
                continue
            Rt_g, Rp_g = torso_frames(gt_j[i])
            if Rt_g is None or Rp_g is None:
                continue
            Rt_s, Rp_s = torso_frames(preds_s0[p])
            s0_valid = Rt_s is not None and Rp_s is not None and bool(arrays["input_valid"][p][[1, 4, 11, 14]].all())
            Y_s0, Y_gt = yaw_frame(heads[i].yaw_radians), yaw_frame(gt_yaw[i])
            frames = {"Y_S0": (Y_s0, Y_s0), "Y_GT": (Y_gt, Y_gt), "F_GT": (Rt_g, Rp_g),
                      "F_S0": (Rt_s, Rp_s) if s0_valid else (Y_s0, Y_s0)}
            rec = {"sid": sid, "p": int(p), "frame": int(view.samples[p].frame_index), "t": ctx["time"](view, p),
                   "available": bool(arrays["input_valid"][p][chain_indices(CHAIN)].all()), "gt_arm_ok": bool(gt_ok[p]),
                   "incl_gt": inclination_degrees(Rt_g), "f_s0_valid": s0_valid, "frames": frames, "pose": {},
                   "sample_id": view.samples[p].sample_id, "image": view.samples[p].image_reference.relative_path}
            if rec["available"]:
                for k, (Rt, _) in frames.items():
                    rec["pose"][k] = arm_pose(e_j[p], w_j[p], Rt, alignment, target, rest)
                if rec["gt_arm_ok"]:
                    rec["pose"]["GT_REF"] = arm_pose(e_t[p], w_t[p], Rt_g, alignment, target, rest)
            rows.append(rec)
        seq_rows[sid] = rows
    return seq_rows


def posture(incl):
    return "upright" if incl < 45 else ("lying" if incl >= 60 else "bent")


def metrics(seq_rows, alignment):
    rows = [r for rs in seq_rows.values() for r in rs]
    use = [r for r in rows if r["available"] and r["gt_arm_ok"] and all(r["pose"].get(k) is not None and r["pose"][k].status == "known" for k in (*MODES, "GT_REF"))]
    out = {"rows": len(rows), "arm_rows": len(use), "f_s0_invalid_rows": int(sum(not r["f_s0_valid"] for r in rows)),
           "posture_rows": {s: int(sum(posture(r["incl_gt"]) == s for r in rows)) for s in ("upright", "bent", "lying")}}
    per = {}
    world_diff = 0.0
    for k in MODES:
        Rt_g = [r["frames"]["F_GT"][0] for r in use]
        geo_t = np.array([geodesic_degrees(r["frames"][k][0], g) for r, g in zip(use, Rt_g)])
        geo_p = np.array([geodesic_degrees(r["frames"][k][1], r["frames"]["F_GT"][1]) for r in use])
        up_ax = np.array([axis_angle_degrees(r["frames"][k][0][:, 2], g[:, 2]) for r, g in zip(use, Rt_g)])
        fw_ax = np.array([axis_angle_degrees(r["frames"][k][0][:, 1], g[:, 1]) for r, g in zip(use, Rt_g)])
        incl = np.array([abs(inclination_degrees(r["frames"][k][0]) - r["incl_gt"]) for r in use])
        upa = np.array([quaternion_angle_degrees(r["pose"][k].proximal_local, r["pose"]["GT_REF"].proximal_local) for r in use])
        fra = np.array([quaternion_angle_degrees(r["pose"][k].distal_local, r["pose"]["GT_REF"].distal_local) for r in use])
        ik = max(max(r["pose"][k].integrity["endpoint_reconstruction_residual"], r["pose"][k].integrity["ik_to_fk_residual"]) for r in use)
        per[k] = {"geo_t": geo_t, "geo_p": geo_p, "up": up_ax, "fw": fw_ax, "incl": incl, "upper": upa, "fore": fra, "ik": ik}
    for r in use:  # world-frame arm direction: object(R_p) * spine(R_p^T R_t) * arm(R_t^T v)  ==  A v  for every mode
        ws = []
        for k in MODES:
            Rt, _ = r["frames"][k]
            ws.append(conj(alignment, Rt) @ np.asarray(r["pose"][k].upper))
        world_diff = max(world_diff, max(float(np.abs(w - ws[0]).max()) for w in ws))
    out["world_arm_direction_max_diff_across_modes"] = world_diff
    seq = np.array([r["sid"] for r in use])
    strata = np.array([posture(r["incl_gt"]) for r in use])
    for stratum in ("all", "upright", "bent", "lying"):
        mask = np.ones(len(use), bool) if stratum == "all" else strata == stratum
        if not mask.any():
            continue
        o = {"rows": int(mask.sum())}
        for k in MODES:
            d = per[k]
            o[k] = {"thorax_geodesic_mean": float(d["geo_t"][mask].mean()), "thorax_geodesic_p95": float(np.percentile(d["geo_t"][mask], 95)),
                    "thorax_geodesic_gt135": float((d["geo_t"][mask] > 135).mean()), "pelvis_geodesic_mean": float(d["geo_p"][mask].mean()),
                    "up_axis_mean": float(d["up"][mask].mean()), "forward_axis_mean": float(d["fw"][mask].mean()),
                    "inclination_error_mean": float(d["incl"][mask].mean()), "upper_arm_local_mean": float(d["upper"][mask].mean()),
                    "upper_arm_local_p95": float(np.percentile(d["upper"][mask], 95)), "forearm_local_mean": float(d["fore"][mask].mean()),
                    "upper_arm_local_gt90": float((d["upper"][mask] > 90).mean())}
        for a, b in (("F_GT", "Y_GT"), ("F_S0", "Y_S0"), ("F_S0", "F_GT")):
            o[f"{a}_minus_{b}_upper_arm_local"] = _brief(seq[mask], per[a]["upper"][mask] - per[b]["upper"][mask])
        out[stratum] = o
    out["ik_residual_max"] = {k: per[k]["ik"] for k in MODES}
    # temporal: root (pelvis frame) and thorax step discontinuities
    disc = {}
    for k in MODES:
        steps = spins = 0
        by_posture = defaultdict(int)
        fallback = 0
        for rs in seq_rows.values():
            for a, b in zip(rs[:-1], rs[1:]):
                d = geodesic_degrees(a["frames"][k][1], b["frames"][k][1])
                g = geodesic_degrees(a["frames"]["F_GT"][1], b["frames"]["F_GT"][1])
                steps += 1
                if d > 90 and g < 45:
                    spins += 1
                    by_posture[posture(max(a["incl_gt"], b["incl_gt"]))] += 1
                    fallback += int(k == "F_S0" and (a["f_s0_valid"] != b["f_s0_valid"]))
        disc[k] = {"steps": steps, "root_spins_gt90_while_gt_lt45": spins, "spins_by_posture": dict(by_posture),
                   "spins_at_fallback_switch": fallback}
    out["root_discontinuities"] = disc
    return out


def export_case(name, w, ctx, extra_title):
    alignment, rest, target = ctx["alignment"], ctx["rest"], ctx["target"]
    B = ctx["spine_basis"]
    head = ctx["spine_head"]
    slots = {"FIXED": "Y_S0", "RUNTIME": "F_S0", "GT": "F_GT"}
    frames = []
    for r in w:
        rec = {"bank_position": r["p"], "frame_index": r["frame"], "time_s": r["t"], "image": r["image"], "available": r["available"],
               "gt_valid": r["gt_arm_ok"], "runtime_status": "KNOWN" if r["f_s0_valid"] else "F_S0_FALLBACK_YAW", "runtime_reasons": [],
               "yaw_deg": {}, "modes": {}, "forward_cam": {}, "mode_lines": {}}
        Rt_g = r["frames"]["F_GT"][0]
        for slot, k in (*slots.items(), ("GT_REF", "F_GT")):
            Rt, Rp = r["frames"][k]
            y = compat_yaw(Rt)
            if slot != "GT_REF":
                rec["yaw_deg"][slot] = math.degrees(y) if y is not None else None
                rec["forward_cam"][slot] = list(Rt[:, 1])
                # root and trunk rotations do not depend on the arm being available
                rec.setdefault("object_rotation_xyzw", {})[slot] = list(matrix_to_quaternion_xyzw(conj(alignment, Rp)))
                rec.setdefault("extra_bone_locals_xyzw", {})[slot] = {"spine_01": list(spine_local_xyzw(alignment, Rp.T @ Rt, B))}
            pose = r["pose"].get("GT_REF" if slot == "GT_REF" else k)
            if pose is None or pose.status != "known":
                continue
            R_rel = Rp.T @ Rt
            Q = conj(alignment, R_rel)
            spin = lambda pt: list(np.asarray(head) + Q @ (np.asarray(pt) - np.asarray(head)))  # noqa: E731
            rec["modes"][slot] = {"proximal_local_xyzw": list(pose.proximal_local), "distal_local_xyzw": list(pose.distal_local),
                                  "mid": spin(pose.mid), "end": spin(pose.end), "root": spin(target.root_head),
                                  "object_rotation_armature_xyzw": list(matrix_to_quaternion_xyzw(conj(alignment, Rp))),
                                  "extra_bone_locals_xyzw": {"spine_01": list(spine_local_xyzw(alignment, R_rel, B))}}
            if slot != "GT_REF" and r["pose"].get("GT_REF") is not None:
                rec["mode_lines"][slot] = [f"torso err {geodesic_degrees(Rt, Rt_g):5.1f} deg  incl {inclination_degrees(Rt):4.0f} (GT {r['incl_gt']:.0f})",
                                           f"upper-arm local err {quaternion_angle_degrees(pose.proximal_local, r['pose']['GT_REF'].proximal_local):5.1f} deg"]
        frames.append(rec)
    return {"case": name, "split": ctx["split"], "sequence_id": w[0]["sid"], "start_frame": w[0]["frame"],
            "fps_sampled": 1.0 / max(w[1]["t"] - w[0]["t"], 1e-6), "frames": frames,
            "labels": {"FIXED": "A: yaw-only runtime (W91 S0 heading)", "RUNTIME": "C: GT-free S0 full torso frame (W93)",
                       "GT": "B: GT full torso frame (evaluation only)"},
            "short_labels": {"FIXED": "A yaw-only", "RUNTIME": "C S0 frame", "GT": "B GT frame"},
            "title": f"W93 full torso frame rig replay {extra_title}"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bank", "s0_dir", "raw_3dpw", "w87_3dpw", "hp_bank", "hp_s0", "w87_3dhp", "advisor_dir", "rest", "calibration", "chains", "out_dir"):
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
    target = target_chain_geometry(rest, load_ik_chain_set(args.chains).chains[0])
    sb = rest.bones["spine_01"]
    M = np.asarray(sb.matrix_local, float)[:3, :3]
    spine_basis = M / np.linalg.norm(M, axis=0, keepdims=True)
    ctx = {"alignment": alignment, "rest": rest, "target": target, "spine_basis": spine_basis, "spine_head": list(sb.head)}
    report = {"schema": "worklog93_torso_frame_rig_replay", "gate_contract": GATE, "results": {}}
    cases = {}
    # ---- 3DPW
    bank = load_bank(args.bank)
    s0, _ = assemble_h0(bank, {s: args.s0_dir / f"prediction_{s}.npy" for s in ("validation", "test")})
    p3 = np.load(args.w87_3dpw)
    ctx3 = {**ctx, "chain_gt": chain_targets(bank.arrays["target_3d"], bank.arrays["target_valid"], CHAIN),
            "time": lambda v, p: v.samples[p].frame_index / 30.0}

    def g3(sid, pos):
        orc = _matched_oracle_sequence(sid, args.raw_3dpw, [bank.samples[p].frame_index for p in pos])
        return [o.yaw_radians for o in observe_yaw(orc)], [np.array([f.points[n].position for n in NAMES]) for f in orc.frames]
    pw = {}
    for split in ("validation", "test"):
        pw[split] = build_rows(f"3dpw_{split}", bank, s0, sequence_ids_in_split(bank, split), g3, bank.arrays, p3, ctx3)
        report["results"][f"3dpw_{split}"] = metrics(pw[split], alignment)
    # ---- 3DHP
    mx = ExternalBank.load(args.hp_bank)
    rv = shim(mx, mx.arrays["input_2d"], mx.arrays["input_valid"])
    for s, ext in zip(rv.samples, mx.samples):
        s.sample_id, s.image_reference = ext.sample_id, ext.image_reference
    g2 = np.concatenate([np.zeros_like(mx.arrays["input_2d"][..., :2]), np.ones_like(mx.arrays["input_2d"][..., 2:])], axis=-1)
    gv = shim(mx, g2, mx.arrays["target_valid"])
    hp = np.load(args.hp_s0)
    ctxh = {**ctx, "time": lambda v, p: v.samples[p].timestamp}

    def gh(sid, pos):
        return [o.yaw_radians for o in observe_yaw(build_h0_lifted_sequence(gv, mx.arrays["target_3d"], sid))], [mx.arrays["target_3d"][p] for p in pos]
    hrows = build_rows("3dhp", rv, hp, sorted({s.sequence_id for s in mx.samples}), gh, mx.arrays, np.load(args.w87_3dhp), ctxh)
    report["results"]["3dhp"] = metrics(hrows, alignment)
    # ---- W92 advisor: diagnostic compatibility with the full frame (no integration)
    pop = {r["sample_id"]: r for r in json.loads((args.advisor_dir / "population.json").read_text())["rows"]}
    ans = defaultdict(dict)
    for line in (args.advisor_dir / "responses.jsonl").read_text().splitlines():
        it = json.loads(line)
        if it["kind"] == "real":
            ans[it["sample_id"]][it["prompt"]] = parse_answer(it["raw_text"])
    diag = defaultdict(int)
    for rs in hrows.values():
        for r in rs:
            if r["sample_id"] not in pop or not r["f_s0_valid"]:
                continue
            a = ans[r["sample_id"]]
            if a.get("back_first") != a.get("stomach_first") or a.get("back_first") not in ("BACK", "STOMACH"):
                diag["abstain"] += 1
                continue
            lab_s0 = gt_label_from_frame(r["frames"]["F_S0"][0])
            lab_gt = gt_label_from_frame(r["frames"]["F_GT"][0])
            diag["answered"] += 1
            diag["s0_frame_label_matches_gt"] += int(lab_s0 == lab_gt)
            diag["vlm_matches_gt"] += int(a["back_first"] == lab_gt)
            diag["vlm_disagrees_with_s0_frame"] += int(a["back_first"] != lab_s0)
            diag["vlm_disagrees_and_vlm_right"] += int(a["back_first"] != lab_s0 and a["back_first"] == lab_gt)
            if posture(r["incl_gt"]) == "lying":
                diag["lying_answered"] += 1
                diag["lying_s0_frame_label_matches_gt"] += int(lab_s0 == lab_gt)
    report["w92_advisor_full_frame_diagnostic"] = dict(diag)
    # ---- review windows (fixed rules)
    def windows(seq_rows, size):
        out = []
        for sid, rows in seq_rows.items():
            for s in range(0, max(len(rows) - size, 0) + 1, 4):
                w = rows[s:s + size]
                if len(w) == size:
                    incl = [r["incl_gt"] for r in w]
                    out.append({"sid": sid, "rows": w, "lying": np.mean([x >= 60 for x in incl]), "bent": np.mean([45 <= x < 60 for x in incl]),
                                "range": max(incl) - min(incl), "start": w[0]["frame"]})
        return out
    key = lambda w: (w["sid"], w["start"])  # noqa: E731
    hw = windows(hrows, 36)
    pick_h = {"ts2_supine": max([w for w in hw if w["sid"].endswith("TS2") and w["start"] <= 700], key=lambda w: (w["lying"], key(w))),
              "ts4_lying": max([w for w in hw if w["sid"].endswith("TS4")], key=lambda w: (w["lying"], key(w))),
              "w92_introduced": min([w for w in hw if w["sid"].endswith("TS2") and w["start"] >= 1080], key=lambda w: (w["start"], key(w))),
              "upright_lying_transition": max(hw, key=lambda w: (w["range"], key(w)))}
    tw = windows(pw["test"], 36)
    pick_p = {"walking_upright": [w for w in tw if w["sid"] == "3dpw:downtown_walkBridge_01:actor0" and w["start"] == 333],
              "turn_180": [w for w in tw if w["sid"] == "3dpw:downtown_cafe_00:actor1" and w["start"] == 1021],
              "bent_or_seated": [max(tw, key=lambda w: (w["bent"] + w["lying"], key(w)))]}
    for name, w in pick_h.items():
        cases[f"3dhp_{name}"] = export_case(name, w["rows"], {**ctx, "split": "3dhp"}, "(3DHP)")
    for name, ws in pick_p.items():
        if ws:
            cases[f"3dpw_{name}"] = export_case(name, ws[0]["rows"], {**ctx, "split": "3dpw_test"}, "(3DPW TEST)")
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=1, default=str))
    (args.out_dir / "cases.json").write_text(json.dumps({"rest_snapshot_digest": rest.digest(), "canonical_to_armature_xyzw": list(alignment),
                                                         "target_chain": target.to_dict(), "cases": cases}, indent=1))
    for pop_name, r in report["results"].items():
        print("==", pop_name, {k: r[k] for k in ("rows", "arm_rows", "f_s0_invalid_rows", "posture_rows", "world_arm_direction_max_diff_across_modes")})
        for s in ("upright", "bent", "lying"):
            if s in r:
                o = r[s]
                print(f"  {s:8s} n={o['rows']}", " | ".join(f"{k}: torso {o[k]['thorax_geodesic_mean']:.1f} up {o[k]['up_axis_mean']:.1f} incl {o[k]['inclination_error_mean']:.1f} "
                                                          f"arm {o[k]['upper_arm_local_mean']:.1f}/{o[k]['forearm_local_mean']:.1f}" for k in MODES))
                for c in ("F_GT_minus_Y_GT_upper_arm_local", "F_S0_minus_Y_S0_upper_arm_local", "F_S0_minus_F_GT_upper_arm_local"):
                    z = o[c]
                    print(f"      {c}: {z['mean']:+.2f} {z.get('ci95')}")
        print("  root discontinuities", r["root_discontinuities"], "ik", r["ik_residual_max"])
    print("advisor diagnostic", report["w92_advisor_full_frame_diagnostic"])
    print({k: (v["sequence_id"], v["start_frame"]) for k, v in cases.items()})


if __name__ == "__main__":
    main()
