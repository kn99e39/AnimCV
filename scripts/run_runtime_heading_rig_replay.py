#!/usr/bin/env python3
"""EXPERIMENT (Worklog 91): rig-level replay of GT-free runtime heading on BaseRig (left arm + torso).

Identical frozen inputs in every mode -- the J1X left-arm unit chain (Worklog 87 predictions) through the unchanged
W61/W76 IK (``measure_chain`` + ``full_arm``) and the W60 object-owned root rotation (rest * A Rz(psi) A^-1):
  FIXED    psi = 180 deg (torso faces the camera; the Worklog 90 control)
  RUNTIME  psi = GT-free runtime heading: frozen S0 + unchanged ``current_root_orientation`` (``motion.runtime_heading``
           also reports its KNOWN / UNRELIABLE state; the yaw is identical)
  GT       psi = GT heading (``observe_yaw`` of the matched 3DPW GT joints) -- EVALUATION-ONLY visualization/reference
The GT-reference arm (GT chain + GT heading) gives the reference local rotations.  Because the heading-relative arm
measurement and the root rotation cancel in the world-frame arm direction (Worklog 90), heading quality is judged on
the torso orientation and on the arm's local (torso-relative) rotations, not on wrist world position.
Frames whose detector arm joints are invalid are UNAVAILABLE in every mode.
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
from run_visual_depth_motion_attribution import _brief  # noqa: E402

from framepose.bank import load_bank  # noqa: E402
from framepose.connected_limb_transfer import chain_indices, chain_targets  # noqa: E402
from motion.runtime_heading import runtime_heading  # noqa: E402
from pose.framepose_bridge import assemble_h0, build_h0_lifted_sequence, sequence_frame_positions, sequence_ids_in_split  # noqa: E402
from pose.root_orientation_diagnostic import observe_yaw  # noqa: E402
from retarget.axis_utils import quaternion_conjugate, quaternion_from_axis_angle, quaternion_multiply, rotate_vector_by_quaternion  # noqa: E402
from retarget.framepose_fk import derive_rig_alignment, load_rig_fk_calibration  # noqa: E402
from retarget.framepose_ik_gt_attribution import measure_chain  # noqa: E402
from retarget.framepose_target_rest import load_target_rest_pose  # noqa: E402
from retarget.framepose_two_bone_ik import load_ik_chain_set, quaternion_angle_degrees, target_chain_geometry  # noqa: E402
from retarget.framepose_visual_full_arm_ik import full_arm  # noqa: E402

CHAIN = "left_arm"
MODES = ("FIXED", "RUNTIME", "GT")
W87_SHA = "cb9dc51e3f84ddc9b45c5ada5e941ceaaad5ee73440a6e387ee0c3f2b689c46f"
S0_SHA = {"validation": "402d70a2bed79ae31919ff86905e19c8972108ea988d6cfa5deac514562132b9",
          "test": "4357c7bdb46d81e7712269b08c45f39452d7a46e697a0c26fdb70d65fe1f4996"}
WINDOW = {"test": 36, "validation": 60}  # 6 s of bank rows
CASE_RULES = {
    "turning": "max |GT heading change| between the first and last row of the window (circular, unwrapped sum)",
    "walking_away": "max share of GT back-view rows (GT heading within 30 deg of 0)",
    "side_on": "max share of GT side-view rows (within 30 deg of +-90)",
    "stationary_facing": "min total |GT heading step| among windows with >= 80% front-view rows",
    "worst_runtime": "max mean RUNTIME heading error",
    "two_actors": "TEST only: the window of downtown_arguing_00 actor1 with max GT heading range (a second actor in the same "
                  "video as actor0; heading is stateless per track)",
    "observation_loss": "max count of rows with an observation problem: bank frame gap before the row (missing detections), "
                        "partial bilateral torso evidence, or an unavailable detector arm",
    "toward_camera": "among windows with >= 60% GT front-view rows, max growth of the detector 2D torso length (mean of the last "
                     "quarter over the first quarter; the subject approaches the camera)",
}


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def circ(a, b):
    return (math.degrees(a - b) + 180.0) % 360.0 - 180.0


def object_rotation(alignment, yaw):
    """Armature-space root rotation A Rz(psi) A^-1 (W60 contract), xyzw."""
    return quaternion_multiply(quaternion_multiply(alignment, quaternion_from_axis_angle((0, 0, 1), yaw)), quaternion_conjugate(alignment))


def solve(e, w, yaw, alignment, target, rest):
    m = measure_chain((0.0, 0.0, 0.0), tuple(float(v) for v in e), tuple(float(v) for v in w), yaw, alignment)
    return None if m is None else full_arm("arm", m, m.direction, target, rest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bank", "s0_dir", "w87_predictions", "raw_3dpw", "rest", "calibration", "chains", "out_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    if _sha(args.w87_predictions) != W87_SHA:
        raise SystemExit("STOP: Worklog 87 predictions changed")
    for split in ("validation", "test"):
        if _sha(args.s0_dir / f"prediction_{split}.npy") != S0_SHA[split]:
            raise SystemExit("STOP: S0 predictions changed")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    bank = load_bank(args.bank)
    s0, _ = assemble_h0(bank, {s: args.s0_dir / f"prediction_{s}.npy" for s in ("validation", "test")})
    pred = np.load(args.w87_predictions)
    rest = load_target_rest_pose(args.rest)
    alignment = derive_rig_alignment(rest, load_rig_fk_calibration(args.calibration))
    target = target_chain_geometry(rest, load_ik_chain_set(args.chains).chains[0])
    e_t, w_t, gt_ok = chain_targets(bank.arrays["target_3d"], bank.arrays["target_valid"], CHAIN)
    e_j, w_j = pred[f"{CHAIN}_g1_elbow"], pred[f"{CHAIN}_g1_wrist"]
    report = {"schema": "worklog91_runtime_heading_rig_replay", "case_rules": CASE_RULES, "results": {}, "invariance": {}}
    cases = {}
    for split in ("validation", "test"):
        per = []
        seq_rows = {}
        for sid in sequence_ids_in_split(bank, split):
            pos = sequence_frame_positions(bank, sid)
            heads = runtime_heading(build_h0_lifted_sequence(bank, s0, sid))
            gt = observe_yaw(_matched_oracle_sequence(sid, args.raw_3dpw, [bank.samples[p].frame_index for p in pos]))
            rows = []
            for i, p in enumerate(pos):
                avail = bool(bank.arrays["input_valid"][p][chain_indices(CHAIN)].all())
                g = gt[i].yaw_radians
                h = heads[i]
                yaws = {"FIXED": math.pi, "RUNTIME": h.yaw_radians, "GT": g}
                ij, iv = bank.arrays["input_2d"][p], bank.arrays["input_valid"][p]
                size = np.asarray(bank.samples[p].image_size, float)
                torso_px = float(np.linalg.norm(((ij[11, :2] + ij[14, :2]) - (ij[1, :2] + ij[4, :2])) / 2 * size))
                rec = {"sid": sid, "p": int(p), "frame": int(bank.samples[p].frame_index), "t": bank.samples[p].frame_index / 30.0,
                       "partial_torso": not bool(iv[1] and iv[4] and iv[11] and iv[14]), "torso_px": torso_px,
                       "available": avail, "gt_ok": bool(gt_ok[p]) and g is not None, "status": h.status, "reasons": list(h.reasons),
                       "yaw": {k: v for k, v in yaws.items()}, "pose": {}}
                if avail:
                    for k, y in yaws.items():
                        if y is not None:
                            rec["pose"][k] = solve(e_j[p], w_j[p], y, alignment, target, rest)
                    if rec["gt_ok"]:
                        rec["pose"]["GT_REF"] = solve(e_t[p], w_t[p], g, alignment, target, rest)
                rows.append(rec)
            seq_rows[sid] = rows
            per.extend(rows)
        report["results"][split] = metrics(per, alignment)
        cases.update(select_cases(split, seq_rows, alignment, rest, target))
        print(split, json.dumps({k: v for k, v in report["results"][split].items() if k in ("frames", "torso_heading", "local_rotation_vs_gt_ref")}), flush=True)
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=1, default=str))
    (args.out_dir / "cases.json").write_text(json.dumps({"rest_snapshot_digest": rest.digest(), "canonical_to_armature_xyzw": list(alignment),
                                                         "target_chain": target.to_dict(), "cases": cases}, indent=1))


def metrics(rows, alignment):
    use = [r for r in rows if r["available"] and r["gt_ok"] and all(r["pose"].get(k) is not None and r["pose"][k].status == "known" for k in (*MODES, "GT_REF"))]
    seq = np.array([r["sid"] for r in use])
    out = {"frames": len(use), "unavailable_rows": int(sum(not r["available"] for r in rows)),
           "runtime_status": {s: int(sum(r["status"] == s for r in rows)) for s in ("KNOWN", "UNRELIABLE", "UNKNOWN")}}
    th, lr, inv = {}, {}, 0.0
    for k in ("FIXED", "RUNTIME"):
        err = np.array([abs(circ(r["yaw"][k], r["yaw"]["GT"])) for r in use])
        th[k] = {"mean": float(err.mean()), "p95": float(np.percentile(err, 95)), "gt90": float((err > 90).mean())}
        up = np.array([quaternion_angle_degrees(r["pose"][k].proximal_local, r["pose"]["GT_REF"].proximal_local) for r in use])
        fo = np.array([quaternion_angle_degrees(r["pose"][k].distal_local, r["pose"]["GT_REF"].distal_local) for r in use])
        # the same chain under the GT heading isolates the heading's own contribution
        up_h = np.array([quaternion_angle_degrees(r["pose"][k].proximal_local, r["pose"]["GT"].proximal_local) for r in use])
        lr[k] = {"upper_arm_vs_gt_ref_deg": float(up.mean()), "upper_arm_p95": float(np.percentile(up, 95)),
                 "forearm_vs_gt_ref_deg": float(fo.mean()), "upper_arm_vs_same_chain_gt_heading_deg": float(up_h.mean()),
                 "upper_arm_vs_same_chain_gt_heading_p95": float(np.percentile(up_h, 95)), "_up": up}
        ik = max(max(r["pose"][k].integrity["endpoint_reconstruction_residual"], r["pose"][k].integrity["ik_to_fk_residual"]) for r in use)
        lr[k]["ik_residual_max"] = float(ik)
    for r in use:
        # world-frame arm direction is heading-invariant: object rotation applied to the heading-relative arm
        ws = []
        for k in MODES:
            q = object_rotation(alignment, r["yaw"][k])
            ws.append(np.asarray(rotate_vector_by_quaternion(tuple(r["pose"][k].upper), q)))
        inv = max(inv, float(np.abs(ws[0] - ws[1]).max()), float(np.abs(ws[0] - ws[2]).max()))
    out["torso_heading"] = th
    out["local_rotation_vs_gt_ref"] = {k: {a: b for a, b in v.items() if not a.startswith("_")} for k, v in lr.items()}
    out["local_rotation_contrast_runtime_minus_fixed"] = _brief(seq, lr["RUNTIME"]["_up"] - lr["FIXED"]["_up"])
    out["world_arm_direction_max_diff_across_modes"] = inv
    # sudden armature rotations (consecutive available rows)
    by = defaultdict(list)
    for r in rows:
        by[r["sid"]].append(r)
    steps, sudden = [], 0
    for rs in by.values():
        for a, b in zip(rs[:-1], rs[1:]):
            if a["yaw"]["RUNTIME"] is None or b["yaw"]["RUNTIME"] is None or a["yaw"]["GT"] is None or b["yaw"]["GT"] is None:
                continue
            d = abs(circ(b["yaw"]["RUNTIME"], a["yaw"]["RUNTIME"]))
            steps.append(d)
            sudden += int(d > 90 and abs(circ(b["yaw"]["GT"], a["yaw"]["GT"])) < 45)
    out["runtime_object_rotation_step_deg"] = {"mean": float(np.mean(steps)), "p95": float(np.percentile(steps, 95)),
                                               "max": float(np.max(steps)), "sudden_gt90_while_gt_lt45": sudden, "steps": len(steps)}
    return out


def select_cases(split, seq_rows, alignment, rest, target):
    size = WINDOW[split]
    wins = []
    for sid, rows in seq_rows.items():
        for s in range(0, max(len(rows) - size, 0) + 1, 6):
            w = rows[s:s + size]
            if len(w) < size or sum(r["gt_ok"] for r in w) < 0.8 * size:
                continue
            g = [r["yaw"]["GT"] for r in w if r["yaw"]["GT"] is not None]
            steps = [abs(circ(b, a)) for a, b in zip(g[:-1], g[1:])]
            signed = sum(circ(b, a) for a, b in zip(g[:-1], g[1:]))
            view = lambda y: abs(circ(y, math.pi))  # noqa: E731
            err = [abs(circ(r["yaw"]["RUNTIME"], r["yaw"]["GT"])) for r in w if r["yaw"]["RUNTIME"] is not None and r["yaw"]["GT"] is not None]
            stride = min((b["frame"] - a["frame"] for a, b in zip(rows[:-1], rows[1:])), default=1)
            loss = sum(1 for k, r in enumerate(w) if (k and r["frame"] - w[k - 1]["frame"] > stride) or r["partial_torso"] or not r["available"])
            q = max(1, size // 4)
            growth = float(np.mean([r["torso_px"] for r in w[-q:]]) / max(np.mean([r["torso_px"] for r in w[:q]]), 1e-6))
            wins.append({"sid": sid, "start": w[0]["frame"], "rows": w, "turn": abs(signed), "back": np.mean([view(y) >= 150 for y in g]),
                         "side": np.mean([abs(view(y) - 90) <= 30 for y in g]), "front": np.mean([view(y) <= 30 for y in g]),
                         "steps": sum(steps), "err": float(np.mean(err)) if err else 0.0,
                         "range": float(max(g) - min(g)) if g else 0.0, "loss": loss, "growth": growth})
    key = lambda w: (w["sid"], w["start"])  # noqa: E731
    pick = {"turning": max(wins, key=lambda w: (w["turn"], key(w))),
            "walking_away": max(wins, key=lambda w: (w["back"], key(w))),
            "side_on": max(wins, key=lambda w: (w["side"], key(w))),
            "stationary_facing": min([w for w in wins if w["front"] >= 0.8] or wins, key=lambda w: (w["steps"], key(w))),
            "worst_runtime": max(wins, key=lambda w: (w["err"], key(w))),
            "observation_loss": max(wins, key=lambda w: (w["loss"], key(w))),
            "toward_camera": max([w for w in wins if w["front"] >= 0.6] or wins, key=lambda w: (w["growth"], key(w)))}
    if split == "test":
        cand = [w for w in wins if w["sid"] == "3dpw:downtown_arguing_00:actor1"]
        if cand:
            pick["two_actors"] = max(cand, key=lambda w: (w["range"], key(w)))
    return {f"{split}_{name}": export(w, name, split, alignment) for name, w in pick.items()}


def export(w, name, split, alignment):
    frames = []
    for r in w["rows"]:
        rec = {"bank_position": r["p"], "frame_index": r["frame"], "time_s": r["t"], "available": r["available"], "gt_valid": r["gt_ok"],
               "runtime_status": r["status"], "runtime_reasons": r["reasons"],
               "yaw_deg": {k: (math.degrees(v) if v is not None else None) for k, v in r["yaw"].items()}, "modes": {}}
        for k in (*MODES, "GT_REF"):
            pose = r["pose"].get(k)
            if pose is not None and pose.status == "known":
                y = r["yaw"]["GT" if k == "GT_REF" else k]
                rec["modes"][k] = {"proximal_local_xyzw": list(pose.proximal_local), "distal_local_xyzw": list(pose.distal_local),
                                   "mid": list(pose.mid), "end": list(pose.end), "object_rotation_armature_xyzw": list(object_rotation(alignment, y))}
        frames.append(rec)
    return {"case": name, "split": split, "sequence_id": w["sid"], "start_frame": w["start"], "fps_sampled": 30.0 / (5 if split == "test" else 3),
            "frames": frames, "rule": CASE_RULES[name]}


if __name__ == "__main__":
    main()
