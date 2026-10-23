#!/usr/bin/env python3
"""EXPERIMENT (Worklog 92): rig replay of the VLM sign advisor on MPI-INF-3DHP (torso + J1X left arm on BaseRig).

Identical inputs in every mode -- the frozen J1X left-arm chain (Worklog 87 ``predictions_3dhp.npz``) through the
unchanged W61/W76 IK and the W60 object-owned root rotation -- with three root headings (exported in the Worklog 91
renderer's mode slots):
  FIXED slot   -> S0      the W91 GT-free runtime heading (no advisor)
  RUNTIME slot -> S0_VLM  the same heading after ``motion.vlm_heading_sign_advisor.decide`` (Worklog 92 responses)
  GT slot      -> GT      the 3DHP GT heading (EVALUATION ONLY)
The GT-reference arm is the GT chain under the GT heading.  Metrics are computed on every 3DHP row; review windows are
chosen by fixed rules: the TS2 window with the most advisor MIRROR corrections, the window with the advisor's
largest introduced error, and a TS4 window containing the head-turn suspects (rows the rejected W91A rule would flip).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_runtime_heading_3dhp import shim  # noqa: E402
from run_runtime_heading_rig_replay import circ, object_rotation, solve  # noqa: E402
from run_visual_depth_motion_attribution import _brief  # noqa: E402

from framepose.bedlam_supervision import ExternalBank  # noqa: E402
from framepose.connected_limb_transfer import chain_indices, chain_targets  # noqa: E402
from motion.runtime_heading import runtime_heading  # noqa: E402
from motion.runtime_heading_side_check import nose_offset, side_view_check  # noqa: E402
from motion.vlm_heading_sign_advisor import decide, parse_answer  # noqa: E402
from pose.framepose_bridge import build_h0_lifted_sequence, sequence_frame_positions  # noqa: E402
from pose.root_orientation_diagnostic import observe_yaw  # noqa: E402
from retarget.framepose_fk import derive_rig_alignment, load_rig_fk_calibration  # noqa: E402
from retarget.framepose_target_rest import load_target_rest_pose  # noqa: E402
from retarget.framepose_two_bone_ik import load_ik_chain_set, quaternion_angle_degrees, target_chain_geometry  # noqa: E402

CHAIN = "left_arm"
SLOTS = {"FIXED": "S0", "RUNTIME": "S0_VLM", "GT": "GT"}
WINDOW = 36  # rows (0.4 s apart in TS1-TS4: 14 s)
LABELS = {"FIXED": "S0 runtime heading (W91, no advisor)", "RUNTIME": "S0 + VLM sign advisor (W92)", "GT": "GT heading (evaluation only)"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("hp_bank", "hp_s0", "w87_predictions", "advisor_dir", "rest", "calibration", "chains", "out_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    mx = ExternalBank.load(args.hp_bank)
    hp = np.load(args.hp_s0)
    pred = np.load(args.w87_predictions)
    pop = {r["sample_id"]: r for r in json.loads((args.advisor_dir / "population.json").read_text())["rows"]}
    answers = defaultdict(dict)
    for line in (args.advisor_dir / "responses.jsonl").read_text().splitlines():
        it = json.loads(line)
        if it["kind"] == "real":
            answers[it["sample_id"]][it["prompt"]] = parse_answer(it["raw_text"])
    rest = load_target_rest_pose(args.rest)
    alignment = derive_rig_alignment(rest, load_rig_fk_calibration(args.calibration))
    target = target_chain_geometry(rest, load_ik_chain_set(args.chains).chains[0])
    e_t, w_t, gt_ok = chain_targets(mx.arrays["target_3d"], mx.arrays["target_valid"], CHAIN)
    e_j, w_j = pred[f"{CHAIN}_g1_elbow"], pred[f"{CHAIN}_g1_wrist"]
    rv = shim(mx, mx.arrays["input_2d"], mx.arrays["input_valid"])
    g2 = np.concatenate([np.zeros_like(mx.arrays["input_2d"][..., :2]), np.ones_like(mx.arrays["input_2d"][..., 2:])], axis=-1)
    gv = shim(mx, g2, mx.arrays["target_valid"])
    seq_rows = {}
    for sid in sorted({s.sequence_id for s in mx.samples}):
        pos = sequence_frame_positions(rv, sid)
        heads = runtime_heading(build_h0_lifted_sequence(rv, hp, sid))
        gt = observe_yaw(build_h0_lifted_sequence(gv, mx.arrays["target_3d"], sid))
        rows = []
        for i, p in enumerate(pos):
            s = mx.samples[p]
            y = heads[i].yaw_radians
            act = "NOT_INVOKED"
            y_vlm = y
            if s.sample_id in pop:
                y_vlm, act, _ = decide(y, answers[s.sample_id], pop[s.sample_id]["axis_x"])
            off, conf = nose_offset(mx.arrays["input_2d"][p], mx.arrays["input_valid"][p], s.image_size)
            yaws = {"S0": y, "S0_VLM": y_vlm, "GT": gt[i].yaw_radians}
            avail = bool(mx.arrays["input_valid"][p][chain_indices(CHAIN)].all())
            rec = {"sid": sid, "p": int(p), "frame": int(s.frame_index), "t": rv.samples[p].timestamp, "image": s.image_reference.relative_path,
                   "available": avail, "gt_ok": bool(gt_ok[p]) and yaws["GT"] is not None, "action": act,
                   "head_turn_suspect": side_view_check(y, off, conf)[1] == "flipped", "yaw": yaws, "pose": {}}
            if avail:
                for k, yy in yaws.items():
                    if yy is not None:
                        rec["pose"][k] = solve(e_j[p], w_j[p], yy, alignment, target, rest)
                if rec["gt_ok"]:
                    rec["pose"]["GT_REF"] = solve(e_t[p], w_t[p], yaws["GT"], alignment, target, rest)
            rows.append(rec)
        seq_rows[sid] = rows
    allrows = [r for rs in seq_rows.values() for r in rs]
    use = [r for r in allrows if r["available"] and r["gt_ok"] and all(r["pose"].get(k) is not None and r["pose"][k].status == "known" for k in ("S0", "S0_VLM", "GT", "GT_REF"))]
    seq = np.array([r["sid"] for r in use])
    metrics = {"frames": len(use)}
    up = {}
    for k in ("S0", "S0_VLM"):
        err = np.array([abs(circ(r["yaw"][k], r["yaw"]["GT"])) for r in use])
        up[k] = np.array([quaternion_angle_degrees(r["pose"][k].proximal_local, r["pose"]["GT_REF"].proximal_local) for r in use])
        steps, spins = [], 0
        for rs in seq_rows.values():
            for a, b in zip(rs[:-1], rs[1:]):
                if None in (a["yaw"][k], b["yaw"][k], a["yaw"]["GT"], b["yaw"]["GT"]):
                    continue
                d = abs(circ(b["yaw"][k], a["yaw"][k]))
                steps.append(d)
                spins += int(d > 90 and abs(circ(b["yaw"]["GT"], a["yaw"]["GT"])) < 45)
        metrics[k] = {"torso_heading_mean": float(err.mean()), "torso_heading_p95": float(np.percentile(err, 95)),
                      "torso_gt135": float((err > 135).mean()), "upper_arm_local_vs_gt_ref_deg": float(up[k].mean()),
                      "upper_arm_local_p95": float(np.percentile(up[k], 95)), "sudden_spins": spins, "steps": len(steps)}
    metrics["upper_arm_local_contrast_vlm_minus_s0"] = _brief(seq, up["S0_VLM"] - up["S0"])
    (args.out_dir / "report.json").write_text(json.dumps(metrics, indent=1, default=str))
    print(json.dumps({k: v for k, v in metrics.items() if k != "upper_arm_local_contrast_vlm_minus_s0"}, indent=1))
    print("contrast", {k: metrics["upper_arm_local_contrast_vlm_minus_s0"].get(k) for k in ("mean", "ci95", "n")})
    # review windows (fixed rules)
    wins = []
    for sid, rows in seq_rows.items():
        for s in range(0, max(len(rows) - WINDOW, 0) + 1, 4):
            w = rows[s:s + WINDOW]
            if len(w) < WINDOW:
                continue
            gain = sum(1 for r in w if r["action"] == "MIRROR")
            worst = max((abs(circ(r["yaw"]["S0_VLM"], r["yaw"]["GT"])) - abs(circ(r["yaw"]["S0"], r["yaw"]["GT"]))
                         for r in w if r["action"] == "MIRROR" and r["yaw"]["GT"] is not None), default=-1e9)
            wins.append({"sid": sid, "rows": w, "mirrors": gain, "introduced": worst, "head_turn": sum(r["head_turn_suspect"] for r in w)})
    key = lambda w: (w["sid"], w["rows"][0]["frame"])  # noqa: E731
    picks = {"ts2_corrections": max([w for w in wins if w["sid"].endswith("TS2")], key=lambda w: (w["mirrors"], key(w))),
             "worst_introduced": max(wins, key=lambda w: (w["introduced"], key(w))),
             "ts4_head_turn": max([w for w in wins if w["sid"].endswith("TS4")], key=lambda w: (w["head_turn"], key(w)))}
    cases = {}
    for name, w in picks.items():
        frames = []
        for r in w["rows"]:
            rec = {"bank_position": r["p"], "frame_index": r["frame"], "time_s": r["t"], "image": r["image"], "available": r["available"],
                   "gt_valid": r["gt_ok"], "runtime_status": "KNOWN" if r["action"] in ("NOT_INVOKED", "KEEP") else f"ADVISOR_{r['action']}",
                   "runtime_reasons": [], "yaw_deg": {slot: (math.degrees(r["yaw"][k]) if r["yaw"][k] is not None else None) for slot, k in SLOTS.items()},
                   "modes": {}}
            for slot, k in (*SLOTS.items(), ("GT_REF", "GT_REF")):
                pose = r["pose"].get(k)
                if pose is not None and pose.status == "known":
                    yy = r["yaw"]["GT" if k == "GT_REF" else k]
                    rec["modes"][slot] = {"proximal_local_xyzw": list(pose.proximal_local), "distal_local_xyzw": list(pose.distal_local),
                                          "mid": list(pose.mid), "end": list(pose.end), "object_rotation_armature_xyzw": list(object_rotation(alignment, yy))}
            frames.append(rec)
        cases[f"3dhp_{name}"] = {"case": name, "split": "3dhp", "sequence_id": w["sid"], "start_frame": w["rows"][0]["frame"],
                                 "fps_sampled": 1.0 / max(w["rows"][1]["t"] - w["rows"][0]["t"], 1e-6), "frames": frames,
                                 "labels": LABELS, "short_labels": {"FIXED": "S0", "RUNTIME": "S0+VLM", "GT": "GT"},
                                 "title": "W92 VLM heading-sign advisor rig replay"}
    (args.out_dir / "cases.json").write_text(json.dumps({"rest_snapshot_digest": rest.digest(), "canonical_to_armature_xyzw": list(alignment),
                                                         "target_chain": target.to_dict(), "cases": cases}, indent=1))
    print({k: (v["sequence_id"], v["start_frame"]) for k, v in cases.items()})


if __name__ == "__main__":
    main()
