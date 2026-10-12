#!/usr/bin/env python3
"""EVALUATION ONLY (Worklog 76): full-arm Worklog 61 IK pose viability of visual chain directions.

F0_H0_FULL_IK / F1_E0_FULL_IK / F2_E1_FULL_IK (``framepose_visual_full_arm_ik``)
change only the chain endpoint direction; H0 reach, H0 observed bend side,
alignment, yaw_H0, rest lengths, clamps and zero-twist rotations are the
Worklog 61 contracts.  The GT-anchored reference arm is the Worklog 62 oracle
(GT direction + GT reach + GT side) on the same root/rest/transform.

Order of work (each gate stops the run on failure):
  1. frozen artifact identities (Worklog 61/62/73/74/75, H0 materialization);
  2. Gate 1 — historical full-IK parity on the exact 395 Worklog 74 rows, through
     the Worklog 74 path (AnimationSemantics v2 + persisted Worklog 61 IK): wrist,
     elbow, local rotations, reach, side status, clamp, availability and the
     Worklog 62 reference; then the same rows through the test path
     (``build_h0_lifted_sequence`` + ``current_root_orientation``) must give the
     identical full arm;
  3. Gate 2 — the 6,003-row Worklog 75 test population must be reproduced with
     F0/F1/F2 wrist errors equal to Worklog 75 T0/T1/T2 row by row;
  4. only then test full-arm metrics.  No model is trained, gated or selected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_framepose_ik_gt_attribution import SEQUENCES, _stats  # noqa: E402
from run_root_motion_contact_h0_replay import _matched_oracle_sequence  # noqa: E402

from framepose.arm_depth_probe import observed_plane  # noqa: E402
from framepose.bank import load_bank  # noqa: E402
from motion.animation_semantics_v2 import load_animation_semantics_v2  # noqa: E402
from motion.animation_semantics_v2_bridge import current_root_orientation  # noqa: E402
from pose.framepose_bridge import assemble_h0, build_h0_lifted_sequence, sequence_frame_positions, sequence_ids_in_split  # noqa: E402
from retarget.framepose_fk import derive_rig_alignment, load_rig_fk_calibration  # noqa: E402
from retarget.framepose_ik_gt_attribution import measure_chain  # noqa: E402
from retarget.framepose_target_rest import load_target_rest_pose  # noqa: E402
from retarget.framepose_two_bone_ik import (  # noqa: E402
    load_ik_chain_set, load_two_bone_ik, local_rotations_for_directions, target_chain_geometry,
)
from retarget.framepose_visual_full_arm_ik import (  # noqa: E402
    ATTRIBUTION_TEMPORAL_METRICS, CANDIDATES, TEMPORAL_METRICS, adjacent_change, classify_change, compare, full_arm, gt_reference,
)
from retarget.framepose_visual_ik_handoff import camera_chain_direction, oracle_reference, rig_direction  # noqa: E402

PINNED = {"w62_report": "f7a100cde398e3117717520247d35a61b992c401fcf30d48ea2f4d86c5c9d523",
          "w73_report": "1c05d50a5e6577329f0573786c6927b07268612ec150fbd079a1aa2bc4b03934",
          "w73_rows_heldout": "4e17bd77399e4c016098f70592d0cfd68f508245ae60914869f756605d9897fc",
          "w74_report": "8158e8b2381bf12ef23eb42698384903e70766ca8e05ad988a831910540c7795",
          "w74_rows": "d3183e657ffc9316093eea2546690982c2ec2daf6dba81b5079ba8c78e99c027",
          "w75_report": "89218babea89033537895a851a013e2b9615ed97314de8945a84b1168424dea7",
          "w75_rows_test": "dc39b22b9489689dc523d5348adda003d3a86b233bd33e061adeecab721de78e"}
H0_VALIDATION_PREFIX = "88a007fc"
# Historical numerical tolerances: Worklog 61/74 solved-geometry contract (cm) and Worklog 75 direction (deg).
TOL_CM = 1e-9
TOL_DEG = 1e-6
TOL_QUAT = 1e-9
NAMES = ("left_shoulder", "left_elbow", "left_wrist")
F_TO_P = {"f0_h0_full_ik": "p0_h0_ik", "f1_e0_full_ik": "p1_e0_visual_direction", "f2_e1_full_ik": "p2_e1_visual_direction"}
F_TO_T = {"f0_h0_full_ik": "t0_h0", "f1_e0_full_ik": "t1_e0_visual", "f2_e1_full_ik": "t2_e1_visual"}
SHORT = {"f0_h0_full_ik": "F0", "f1_e0_full_ik": "F1", "f2_e1_full_ik": "F2"}
PAIRS = (("f2_e1_full_ik", "f1_e0_full_ik"), ("f2_e1_full_ik", "f0_h0_full_ik"), ("f1_e0_full_ik", "f0_h0_full_ik"))
PAIRED_METRICS = ("wrist_error", "elbow_error", "upper_direction_error_degrees", "forearm_direction_error_degrees")
POSE_METRICS = (*PAIRED_METRICS, "chain_direction_error_degrees", "abs_bend_angle_error_degrees",
                "bend_angle_error_degrees", "bend_plane_error_degrees",
                "proximal_rotation_vs_gt_degrees", "distal_rotation_vs_gt_degrees")
INTEGRITY = ("integrity_upper_length_residual", "integrity_forearm_length_residual",
             "integrity_endpoint_reconstruction_residual", "integrity_ik_to_fk_residual",
             "integrity_quaternion_norm_error")
# Worklog 74/75 GT bend bins (degrees of the GT source chain bend), named per the Worklog 76 direction.
BEND_REGIMES = {"near_straight [0,15)": (0, 15), "moderate [15,45)": (15, 45), "strong [45,90)": (45, 90),
                "near_folded [90,180]": (90, 181)}
STABLE = math.sin(math.radians(10))
BANK_STRIDE = 5  # FrameBank 3DPW sampling: every 5th source frame (1/6 s at 30 fps)
BOOTSTRAP = {"replicates": 2000, "seed": 1337, "moving_block_lengths": (10, 25),
             "procedures": ("sequence_cluster", "moving_block_within_sequence")}
# Predeclared before any test candidate statistic was computed; none depends on F2.
TEMPORAL_CONTRACT = {
    "adjacency": "consecutive FrameBank samples of one sequence exactly one bank stride apart (5 source frames = 1/6 s at "
                 "30 fps), both common-valid; consecutive samples across a bank hole are dropped and counted",
    "jump_threshold": "per metric, the 99th percentile of the GT-anchored reference arm's own adjacent change on the same pairs",
    "why": "the reference arm follows true 3DPW motion through the same rig/IK, so its tail defines plausible fast motion "
           "independently of any candidate; a candidate jump without a reference jump is a candidate-specific discontinuity"}
DECISION = {"material_elbow_p95_regression_cm": 1.0,
            "meaningful_improvement": "paired mean < 0 with every predeclared 95% interval excluding 0",
            "broad": "improves in >= 2/3 of sequences and the largest single sequence supplies < 50% of the net gain",
            "instability": "F2 candidate-specific discontinuity rate exceeds F0's by > 25% relative on any temporal metric",
            "invalidity": "F2 unavailable count > F0 unavailable count"}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _unit(v):
    n = math.sqrt(sum(x * x for x in v))
    return tuple(x / n for x in v)


def _qdiff(a, b) -> float:
    """Max component difference after the q/-q sign alignment."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if float(np.dot(a, b)) < 0:
        b = -b
    return float(np.max(np.abs(a - b)))


def plane_for(bank, position):
    planes, oks = observed_plane(bank.arrays["input_2d"][position][None], np.asarray([bank.samples[position].image_size], float))
    return (float(planes["chain"][0, 0]), float(planes["chain"][0, 1])) if bool(oks["chain"][0]) else None


def evaluate(h0_pos, gt_pos, yaw, alignment, target, rest, f_e0, f_e1, plane):
    """F0/F1/F2 full arms and the Worklog 62 reference for one frame (rig space)."""
    h0, gt = measure_chain(*h0_pos, yaw, alignment), measure_chain(*gt_pos, yaw, alignment)
    if h0 is None or gt is None:
        return None
    e_gt, oracle_mid, gt_side, gt_side_status = oracle_reference(gt, target, rest)
    ref = gt_reference(gt, target, rest)
    cams = {"f1_e0_full_ik": camera_chain_direction(f_e0, plane), "f2_e1_full_ik": camera_chain_direction(f_e1, plane)}
    poses = {"f0_h0_full_ik": full_arm("f0_h0_full_ik", h0, h0.direction, target, rest)}
    for name, cam in cams.items():
        poses[name] = full_arm(name, h0, rig_direction(cam, yaw, alignment), target, rest)
    f_gt = _unit(tuple(b - a for a, b in zip(gt_pos[0], gt_pos[2])))[1]
    f_h0 = _unit(tuple(b - a for a, b in zip(h0_pos[0], h0_pos[2])))[1]
    return {"h0": h0, "gt": gt, "f_gt": f_gt, "f": {"h0": f_h0, "e0": f_e0, "e1": f_e1}, "e_gt": e_gt, "oracle_mid": oracle_mid, "gt_side_status": gt_side_status,
            "ref": ref, "poses": poses, "cams": cams}


def pose_record(p):
    return {"status": p.status, "reach_fraction": p.reach_fraction, "side_status": p.side_status, "clamp": p.clamp,
            "direction": list(p.direction) if p.direction else None, "side": list(p.side) if p.side else None,
            "root": list(p.root) if p.root else None, "mid": list(p.mid) if p.mid else None,
            "end": list(p.end) if p.end else None,
            "proximal_local": list(p.proximal_local) if p.proximal_local else None,
            "distal_local": list(p.distal_local) if p.distal_local else None}


def row_record(sid, frame_index, position, order, sample_id, conf, yaw, ev, split):
    ref = ev["ref"]
    cands = {}
    for n, p in ev["poses"].items():
        c = compare(p, ref, ev["e_gt"])
        if c["status"] == "known":
            c["abs_bend_angle_error_degrees"] = abs(c["bend_angle_error_degrees"])
        cands[n] = c
    f2, f0 = ev["poses"]["f2_e1_full_ik"], ev["poses"]["f0_h0_full_ik"]
    change = {}
    if f2.status == "known" and f0.status == "known":
        n2, n0 = f2.plane_normal(), f0.plane_normal()
        change = {"f2_vs_f0_bend_plane_change_degrees": (math.degrees(math.acos(max(-1., min(1., float(np.dot(n2, n0))))))
                                                         if n2 is not None and n0 is not None else None),
                  "f2_vs_f0_side_flip": float(np.dot(f2.side, f0.side)) <= 0}
    return {"split": split, "sequence_id": sid, "frame_index": frame_index, "position": position, "bank_order": order,
            "sample_id": sample_id, "min_arm_confidence": conf, "yaw_h0_radians": yaw,
            "h0_reach": ev["h0"].reach_fraction, "gt_reach": ev["gt"].reach_fraction,
            "h0_bend_degrees": ev["h0"].bend_angle_degrees, "gt_bend_degrees": ev["gt"].bend_angle_degrees,
            "f_gt": ev["f_gt"], "f_h0": ev["f"]["h0"], "f_e0": ev["f"]["e0"], "f_e1": ev["f"]["e1"],
            "h0_side_status": "degenerate" if ev["h0"].degenerate else "observed", "gt_side_status": ev["gt_side_status"],
            "e_gt": list(ev["e_gt"]), "camera_directions": {k: list(v) for k, v in ev["cams"].items()},
            "reference": {**pose_record(ref), "bend_angle_degrees": ref.bend_angle_degrees() if ref.status == "known" else None},
            "geometry": {n: pose_record(p) for n, p in ev["poses"].items()},
            "candidates": cands, **change}


def collect(bank, h0, w73, target, rest, alignment, raw, split, keep=None):
    """Worklog 75 validity and identity join, Worklog 61 rig-space full arms."""
    rows, accounting = [], {}
    for sid in sequence_ids_in_split(bank, split):
        if keep is not None and sid not in {k[0] for k in keep}:
            continue
        lifted = build_h0_lifted_sequence(bank, h0, sid)
        yaw = current_root_orientation(lifted)
        positions = sequence_frame_positions(bank, sid)
        oracle = _matched_oracle_sequence(sid, raw, [bank.samples[p].frame_index for p in positions])
        acc = Counter()
        for order, (frame, orient, gt_frame, p) in enumerate(zip(lifted.frames, yaw, oracle.frames, positions)):
            if gt_frame.frame_index != frame.frame_index or bank.samples[p].frame_index != frame.frame_index:
                raise SystemExit(f"{sid}: frame alignment broken")
            key = (sid, frame.frame_index)
            if keep is not None and key not in keep:
                continue
            acc["bank_rows"] += 1
            pred = w73.get(key)
            flags = {"no_w73_prediction": pred is None,
                     **{f"h0_source_invalid:{n}": not frame.points[n].observation_valid for n in NAMES},
                     "root_orientation_unknown": not orient.known,
                     "gt_invalid": not all(gt_frame.points[n].observation_valid for n in NAMES)}
            for k, v in flags.items():
                acc[f"flag:{k}"] += int(v)
            first = next((k for k, v in flags.items() if v), None)
            if first:
                acc[f"excluded:{first}"] += 1
                continue
            if pred["position"] != p or pred["sample_id"] != bank.samples[p].sample_id:
                raise SystemExit(f"{key}: Worklog 73 row does not match the FrameBank sample")
            h0_pos = [frame.points[n].position for n in NAMES]
            gt_pos = [gt_frame.points[n].position for n in NAMES]
            h0_cam, gt_cam = (_unit(tuple(b - a for a, b in zip(q[0], q[2]))) for q in (h0_pos, gt_pos))
            if abs(pred["chain_f_h0"] - h0_cam[1]) > 1e-6 or abs(pred["chain_f_gt"] - gt_cam[1]) > 1e-5:
                raise SystemExit(f"{key}: Worklog 73 chain f identity differs")
            plane = plane_for(bank, p)
            if plane is None:
                acc["excluded:degenerate_observed_plane"] += 1
                continue
            ev = evaluate(h0_pos, gt_pos, orient.yaw_radians, alignment, target, rest,
                          pred["chain_f_e0"], pred["chain_f_e1"], plane)
            if ev is None:
                acc["excluded:degenerate_chain_geometry"] += 1
                continue
            conf = min(float(bank.arrays["input_2d"][p][j, 2]) for j in (11, 12, 13))
            acc["w75_common_eligible"] += 1
            r = row_record(sid, frame.frame_index, p, order, bank.samples[p].sample_id, conf, orient.yaw_radians, ev, split)
            for n in CANDIDATES:
                acc[f"{SHORT[n]}:{r['geometry'][n]['status']}"] += 1
            acc[f"reference:{r['reference']['status']}"] += 1
            rows.append(r)
        accounting[sid] = dict(acc)
    return rows, accounting


# ----------------------------------------------------------------------------- Gate 1
def gate1_worklog74(args, bank, w73, w74, target, rest, alignment):
    """The exact 395 Worklog 74 rows through the Worklog 74 path (semantics + Worklog 61 IK)."""
    source = json.loads((args.semantics_dir / "report.json").read_text())
    files = {r["sequence_id"]: r for r in source["per_sequence"]}
    ik61 = json.loads((args.ik61_dir / "report.json").read_text())
    position = {(s.sequence_id, s.frame_index): p for p, s in enumerate(bank.samples)}
    checks, counts, poses_by_key = defaultdict(float), Counter(), {}

    def worst(k, v):
        checks[k] = max(checks[k], v)

    for sid in SEQUENCES:
        sem = load_animation_semantics_v2(args.semantics_dir / files[sid]["file"])
        if sem.content_digest() != files[sid]["content_digest"]:
            raise SystemExit(f"{sid}: semantics digest changed")
        ik = load_two_bone_ik(args.ik61_dir / ik61["sequences"][sid]["ik_file"])
        if ik.provenance["semantics_digest"] != sem.content_digest():
            raise SystemExit(f"{sid}: Worklog 61 IK was built from other semantics")
        oracle = _matched_oracle_sequence(sid, args.threedpw_raw, [f.frame_index for f in sem.frames])
        lookup = {n: i for i, n in enumerate(sem.joint_names)}
        ik_by = {s.frame_index: s for s in ik.samples}
        for frame, gt_frame in zip(sem.frames, oracle.frames):
            key = (sid, frame.frame_index)
            if key not in w74:
                continue
            old = w74[key]
            sample = ik_by[frame.frame_index]
            p, pred = position[key], w73[key]
            if pred["position"] != p or pred["sample_id"] != bank.samples[p].sample_id or old["bank_position"] != p:
                raise SystemExit(f"{key}: identity join broken")
            plane = plane_for(bank, p)
            worst("observed_plane_vs_w74", max(abs(a - b) for a, b in zip(plane, old["observed_chain_plane_xz"])))
            yaw = frame.root_orientation.yaw_radians
            worst("yaw_vs_w74", abs(yaw - old["yaw_h0_radians"]))
            ev = evaluate([frame.articulation.joint_positions[lookup[n]] for n in NAMES],
                          [gt_frame.points[n].position for n in NAMES], yaw, alignment, target, rest,
                          pred["chain_f_e0"], pred["chain_f_e1"], plane)
            counts["rows"] += 1
            poses_by_key[key] = ev
            for n, pname in F_TO_P.items():
                f, g, s = ev["poses"][n], old["geometry"][pname], old["candidates"][pname]
                counts[f"status_mismatch:{n}"] += f.status != g["status"]
                counts[f"side_status_mismatch:{n}"] += f.side_status != g["side_status"]
                counts[f"clamp_mismatch:{n}"] += f.clamp != g["clamp"]
                if f.status != "known" or g["status"] != "known":
                    counts[f"unavailable:{n}"] += 1
                    continue
                worst(f"{n}:wrist_vs_w74_cm", math.dist(f.end, g["solved_endpoint"]))
                worst(f"{n}:elbow_vs_w74_cm", math.dist(f.mid, g["solved_mid"]))
                worst(f"{n}:reach_vs_w74", abs(f.reach_fraction - g["reach_fraction"]))
                worst(f"{n}:direction_vs_w74", math.dist(f.direction, g["direction"]))
                worst(f"{n}:side_vs_w74", math.dist(f.side, g["side"]))
                c = compare(f, ev["ref"], ev["e_gt"])
                worst(f"{n}:wrist_error_vs_w74_score_cm", abs(c["wrist_error"] - s["solved_endpoint_error"]))
                worst(f"{n}:elbow_error_vs_w74_score_cm", abs(c["elbow_error"] - s["elbow_error"]))
                counts[f"side_agreement_mismatch:{n}"] += c["side_agrees_with_gt"] != s["side_agrees_with_gt"]
                # Historical local rotations implied by the persisted Worklog 74 joints (same Worklog 61 function).
                q1, q2 = local_rotations_for_directions(
                    rest, target.proximal, target.distal,
                    tuple(m - r for m, r in zip(g["solved_mid"], target.root_head)),
                    tuple(e - m for e, m in zip(g["solved_endpoint"], g["solved_mid"])))
                worst(f"{n}:rotation_vs_w74_joints", max(_qdiff(f.proximal_local, q1), _qdiff(f.distal_local, q2)))
                worst(f"{n}:ik_to_fk_residual", f.integrity["ik_to_fk_residual"])
            # F0 against the persisted Worklog 61 runtime sample, including its local rotations.
            f0 = ev["poses"]["f0_h0_full_ik"]
            counts["f0_vs_w61_status_mismatch"] += (sample.proximal_status != "known") != (f0.status != "known")
            counts["f0_vs_w61_bend_status_mismatch"] += sample.bend_plane_status != f0.side_status
            counts["f0_vs_w61_clamp_mismatch"] += sample.clamp_status != f0.clamp
            worst("f0:wrist_vs_w61_cm", math.dist(f0.end, sample.solved_endpoint))
            worst("f0:elbow_vs_w61_cm", math.dist(f0.mid, sample.solved_mid))
            worst("f0:reach_vs_w61", abs(f0.reach_fraction - sample.source_reach_fraction))
            worst("f0:proximal_local_vs_w61", _qdiff(f0.proximal_local, sample.proximal_local))
            worst("f0:distal_local_vs_w61", _qdiff(f0.distal_local, sample.distal_local))
            # Worklog 62 GT reference.
            ref = ev["ref"]
            counts["gt_side_status_mismatch"] += ev["gt_side_status"] != old["gt_side_status"]
            worst("reference:e_gt_vs_w74_cm", math.dist(ev["e_gt"], old["e_gt"]))
            worst("reference:oracle_elbow_vs_w74_cm", math.dist(ref.mid, old["oracle_mid"]))
            worst("reference:oracle_elbow_vs_oracle_reference_cm", math.dist(ref.mid, ev["oracle_mid"]))
            worst("reference:wrist_vs_e_gt_cm", math.dist(ref.end, ev["e_gt"]))
    limits = {k: (TOL_QUAT if ("rotation" in k or "local" in k) else TOL_CM) for k in checks}
    limits["yaw_vs_w74"] = 0.0
    limits["observed_plane_vs_w74"] = 0.0
    mismatches = {k: v for k, v in counts.items() if k != "rows" and v}
    failed = {k: v for k, v in checks.items() if v > limits[k]}
    result = {"rows": counts["rows"], "expected": len(w74), "max_abs_diff": dict(checks), "limits": limits,
              "status_mismatches": mismatches, "passed": counts["rows"] == len(w74) and not failed and not mismatches}
    return result, poses_by_key


def gate1_test_path(val_rows, poses_by_key):
    """The same 395 rows through the test path must give the identical full arm."""
    out, n = defaultdict(float), 0
    for r in val_rows:
        key = (r["sequence_id"], r["frame_index"])
        if key not in poses_by_key:
            continue
        n += 1
        ev = poses_by_key[key]
        for name in CANDIDATES:
            a, b = r["geometry"][name], ev["poses"][name]
            if a["status"] != b.status or a["side_status"] != b.side_status or a["clamp"] != b.clamp:
                out["status_mismatch"] += 1
                continue
            if b.status != "known":
                continue
            out["joints_cm"] = max(out["joints_cm"], math.dist(a["end"], b.end), math.dist(a["mid"], b.mid))
            out["rotations"] = max(out["rotations"], _qdiff(a["proximal_local"], b.proximal_local),
                                   _qdiff(a["distal_local"], b.distal_local))
        out["reference_cm"] = max(out["reference_cm"], math.dist(r["reference"]["mid"], ev["ref"].mid),
                                  math.dist(r["e_gt"], ev["e_gt"]))
    passed = (n == len(poses_by_key) and not out["status_mismatch"] and out["joints_cm"] <= TOL_CM
              and out["rotations"] <= TOL_QUAT and out["reference_cm"] <= TOL_CM)
    return {"rows": n, "expected": len(poses_by_key), "max_abs_diff": dict(out), "passed": passed}


# ----------------------------------------------------------------------------- Gate 2
def gate2_worklog75(rows, w75):
    keys = {(r["sequence_id"], r["frame_index"]) for r in rows}
    out = {"rows": len(rows), "expected": len(w75), "identity_set_equal": keys == set(w75),
           "max_abs_diff": defaultdict(float), "unavailable": Counter(), "mismatch": Counter()}
    m = out["max_abs_diff"]
    for r in rows:
        old = w75.get((r["sequence_id"], r["frame_index"]))
        if old is None:
            continue
        if old["position"] != r["position"] or old["sample_id"] != r["sample_id"]:
            out["mismatch"]["sample_identity"] += 1
        m["h0_reach"] = max(m["h0_reach"], abs(old["h0_reach"] - r["h0_reach"]))
        m["gt_reach"] = max(m["gt_reach"], abs(old["gt_reach"] - r["gt_reach"]))
        m["min_arm_confidence"] = max(m["min_arm_confidence"], abs(old["min_arm_confidence"] - r["min_arm_confidence"]))
        for n, t in F_TO_T.items():
            c, o = r["candidates"][n], old["candidates"][t]
            if c["status"] != "known":
                out["unavailable"][n] += 1
                continue
            m[f"{n}:wrist_error_cm"] = max(m[f"{n}:wrist_error_cm"], abs(c["wrist_error"] - o["endpoint_error"]))
            m[f"{n}:chain_direction_deg"] = max(m[f"{n}:chain_direction_deg"],
                                                abs(c["chain_direction_error_degrees"] - o["direction_error_degrees"]))
            out["mismatch"][f"{n}:clamp"] += c["clamp"] != o["clamp"]
            if n in r["camera_directions"]:
                m[f"{n}:camera_direction"] = max(m[f"{n}:camera_direction"], math.dist(r["camera_directions"][n], o["direction"]))
    limits = {k: (TOL_DEG if k.endswith("_deg") else TOL_CM) for k in m}
    failed = {k: v for k, v in m.items() if v > limits[k]}
    out["limits"] = limits
    out["passed"] = (out["identity_set_equal"] and out["rows"] == out["expected"] and not failed
                     and not any(out["mismatch"].values()))
    out["max_abs_diff"], out["unavailable"], out["mismatch"] = dict(m), dict(out["unavailable"]), dict(out["mismatch"])
    return out


# ----------------------------------------------------------------------------- statistics
def _common(rows):
    return [r for r in rows if r["reference"]["status"] == "known"
            and all(r["candidates"][n]["status"] == "known" for n in CANDIDATES)]


def _seq_values(rows, value):
    by = defaultdict(list)
    for r in rows:
        v = value(r)
        if v is not None:
            by[r["sequence_id"]].append(v)
    return by


def _paired(rows, a, b, metric):
    d = [r["candidates"][a][metric] - r["candidates"][b][metric] for r in rows]
    by_seq = _seq_values(rows, lambda r: r["candidates"][a][metric] - r["candidates"][b][metric])
    seq_means = {s: sum(v) / len(v) for s, v in by_seq.items()}
    sums = {s: sum(v) for s, v in by_seq.items()}
    net = sum(sums.values())
    ranked = sorted(sums.items(), key=lambda kv: kv[1])
    return {**_stats(d), "improved_share": sum(x < 0 for x in d) / len(d) if d else None,
            "regressed_share": sum(x > 0 for x in d) / len(d) if d else None,
            "frame_weighted_mean": sum(d) / len(d) if d else None,
            "sequence_weighted_mean": sum(seq_means.values()) / len(seq_means) if seq_means else None,
            "sequences": len(seq_means), "sequences_improved": sum(v < 0 for v in seq_means.values()),
            "sequences_regressed": sum(v > 0 for v in seq_means.values()),
            "largest_single_sequence_contribution": {
                "sequence_id": ranked[0][0], "sum": ranked[0][1], "net_sum": net,
                "share_of_net": ranked[0][1] / net if net < 0 else None} if ranked else None,
            "mean_without_largest_contributor": ((net - ranked[0][1]) / (len(d) - len(by_seq[ranked[0][0]]))
                                                 if ranked and len(d) > len(by_seq[ranked[0][0]]) else None),
            "worst_regressions": [{"sequence_id": s, "sum": v, "mean": seq_means[s]} for s, v in ranked[::-1][:3] if v > 0],
            "worst_row_regressions": [{"sequence_id": r["sequence_id"], "frame_index": r["frame_index"], "delta": x}
                                      for x, r in sorted(zip(d, rows), key=lambda t: -t[0])[:3]]}


def _bootstrap(by_seq):
    """Predeclared Worklog 75 sequence-aware 95% intervals; ``by_seq`` values must be in frame order."""
    rng = random.Random(BOOTSTRAP["seed"])
    seqs = sorted(by_seq)
    q = lambda v: [sorted(v)[int(0.025 * len(v))], sorted(v)[int(0.975 * len(v)) - 1]]  # noqa: E731
    frame_means, seq_means = [], []
    for _ in range(BOOTSTRAP["replicates"]):
        pick = [rng.choice(seqs) for _ in seqs]
        allv = [x for s in pick for x in by_seq[s]]
        frame_means.append(sum(allv) / len(allv))
        seq_means.append(sum(sum(by_seq[s]) / len(by_seq[s]) for s in pick) / len(pick))
    out = {"sequence_cluster": {"frame_weighted_ci95": q(frame_means), "sequence_weighted_ci95": q(seq_means)}}
    for L in BOOTSTRAP["moving_block_lengths"]:
        means = []
        for _ in range(BOOTSTRAP["replicates"]):
            tot = []
            for s in seqs:
                d, n = by_seq[s], len(by_seq[s])
                if n <= L:
                    tot += [d[rng.randrange(n)] for _ in range(n)]
                    continue
                chunk = []
                while len(chunk) < n:
                    i = rng.randrange(0, n - L + 1)
                    chunk += d[i:i + L]
                tot += chunk[:n]
            means.append(sum(tot) / len(tot))
        out[f"moving_block_{L}"] = {"frame_weighted_ci95": q(means)}
    return out


def _excludes_zero_negative(boot):
    return all(v["frame_weighted_ci95"][1] < 0 for v in boot.values()) and boot["sequence_cluster"]["sequence_weighted_ci95"][1] < 0


def _candidate_summary(rows, all_rows, n):
    c = [r["candidates"][n] for r in rows]
    agree = [x["side_agrees_with_gt"] for x in c if x["side_agrees_with_gt"] is not None]
    every = [r["candidates"][n] for r in all_rows]
    return {**{m: _stats(x[m] for x in c) for m in POSE_METRICS},
            "side_agrees_with_gt": sum(agree), "side_flips_vs_gt": len(agree) - sum(agree),
            "bend_plane_undefined": sum(x["bend_plane_error_degrees"] is None for x in c),
            "integrity_max": {k: max((x[k] for x in c), default=None) for k in INTEGRITY},
            "integrity_nonfinite": sum(not x["integrity_finite"] for x in c),
            "clamps": dict(Counter(str(x["clamp"]) for x in c)),
            "side_status": dict(Counter(str(x["side_status"]) for x in c)),
            "unavailable_all_eligible": dict(Counter(x["status"] for x in every if x["status"] != "known"))}


def _coupling(rows, a, b):
    cells = defaultdict(list)
    for r in rows:
        dw = r["candidates"][a]["wrist_error"] - r["candidates"][b]["wrist_error"]
        de = r["candidates"][a]["elbow_error"] - r["candidates"][b]["elbow_error"]
        cells[f"wrist_{'improves' if dw < 0 else 'worsens'}/elbow_{'improves' if de < 0 else 'worsens'}"].append((dw, de))
    out = {}
    for k in ("wrist_improves/elbow_improves", "wrist_improves/elbow_worsens", "wrist_worsens/elbow_improves",
              "wrist_worsens/elbow_worsens"):
        v = cells.get(k, [])
        out[k] = {"n": len(v), "share": len(v) / len(rows) if rows else None,
                  "mean_wrist_delta": sum(x for x, _ in v) / len(v) if v else None,
                  "mean_elbow_delta": sum(y for _, y in v) / len(v) if v else None,
                  "sum_wrist_delta": sum(x for x, _ in v), "sum_elbow_delta": sum(y for _, y in v)}
    dw = [r["candidates"][a]["wrist_error"] - r["candidates"][b]["wrist_error"] for r in rows]
    de = [r["candidates"][a]["elbow_error"] - r["candidates"][b]["elbow_error"] for r in rows]
    out["row_correlation_wrist_vs_elbow_delta"] = float(np.corrcoef(dw, de)[0, 1]) if len(rows) > 2 else None
    return out


def _adjacent_pairs(rows):
    """Consecutive FrameBank samples of one sequence one bank stride apart, both common-valid.

    Returns the pairs and the count of consecutive samples dropped for a larger source-frame gap
    (bank holes), which are not adjacent frames.
    """
    by = defaultdict(list)
    for r in rows:
        by[r["sequence_id"]].append(r)
    pairs, wide = [], Counter()
    for sid in sorted(by):
        g = sorted(by[sid], key=lambda r: r["bank_order"])
        for a, b in zip(g, g[1:]):
            if b["bank_order"] != a["bank_order"] + 1:
                continue
            if b["frame_index"] - a["frame_index"] == BANK_STRIDE:
                pairs.append((a, b))
            else:
                wide[b["frame_index"] - a["frame_index"]] += 1
    return pairs, wide


class _Arm:
    """Light view of a persisted arm for ``adjacent_change``."""

    def __init__(self, g):
        self.proximal_local, self.distal_local = g["proximal_local"], g["distal_local"]
        self.end, self.mid, self.direction = g["end"], g["mid"], g["direction"]
        self.root, self.side = g["root"], g["side"]


def _temporal(rows, thresholds=None):
    pairs, wide = _adjacent_pairs(rows)
    if not pairs:
        return {"pairs": 0}, thresholds
    ch = {"reference": [adjacent_change(_Arm(a["reference"]), _Arm(b["reference"])) for a, b in pairs]}
    for n in CANDIDATES:
        ch[n] = [adjacent_change(_Arm(a["geometry"][n]), _Arm(b["geometry"][n])) for a, b in pairs]
    if thresholds is None:
        thresholds = {m: float(np.percentile([c[m] for c in ch["reference"]], 99)) for m in TEMPORAL_METRICS}
    out = {"pairs": len(pairs), "dropped_consecutive_samples_by_source_gap": dict(sorted(wide.items())),
           "thresholds_gt_p99": thresholds,
           "reference": {m: _stats(c[m] for c in ch["reference"]) for m in (*TEMPORAL_METRICS, *ATTRIBUTION_TEMPORAL_METRICS)}}
    for n in CANDIDATES:
        block = {}
        for m in TEMPORAL_METRICS:
            cand = [c[m] for c in ch[n]]
            ref = [c[m] for c in ch["reference"]]
            cls = Counter(classify_change(x, y, thresholds[m]) for x, y in zip(cand, ref))
            block[m] = {"change": _stats(cand), "excess_over_reference": _stats(x - y for x, y in zip(cand, ref)),
                        "abs_difference_from_reference": _stats(abs(x - y) for x, y in zip(cand, ref)),
                        "classes": dict(cls), "candidate_discontinuity_rate": cls["candidate_discontinuity"] / len(pairs),
                        "jump_rate": (cls["candidate_discontinuity"] + cls["rapid_motion_candidate_jump"]) / len(pairs)}
        for m in ATTRIBUTION_TEMPORAL_METRICS:
            cand, ref = [c[m] for c in ch[n]], [c[m] for c in ch["reference"]]
            block[m] = {"change": _stats(cand), "excess_over_reference": _stats(x - y for x, y in zip(cand, ref))}
        # Does extra joint motion follow the extra chain-direction motion? (attribution, row correlation)
        dd = [c["chain_direction_change_degrees"] - r["chain_direction_change_degrees"] for c, r in zip(ch[n], ch["reference"])]
        dp = [c["proximal_rotation_change_degrees"] - r["proximal_rotation_change_degrees"] for c, r in zip(ch[n], ch["reference"])]
        block["corr_excess_chain_direction_vs_excess_proximal"] = float(np.corrcoef(dd, dp)[0, 1])
        out[n] = block
    # Frame-to-frame change of the chain forward fraction that each candidate's direction is built from.
    out["abs_forward_fraction_change"] = {k: _stats(abs(b[k] - a[k]) for a, b in pairs)
                                          for k in ("f_gt", "f_h0", "f_e0", "f_e1")}
    out["paired_change"] = {}
    for a, b in PAIRS:
        out["paired_change"][f"{a}-{b}"] = {}
        for m in (*TEMPORAL_METRICS, *ATTRIBUTION_TEMPORAL_METRICS):
            by = defaultdict(list)
            for (ra, _), x, y in zip(pairs, ch[a], ch[b]):
                by[ra["sequence_id"]].append(x[m] - y[m])
            d = [v for s in by.values() for v in s]
            out["paired_change"][f"{a}-{b}"][m] = {**_stats(d), "mean": sum(d) / len(d),
                                                   "bootstrap": _bootstrap(by) if len(by) > 1 else None}
    worst = defaultdict(list)
    for (ra, rb), c2, cr in zip(pairs, ch["f2_e1_full_ik"], ch["reference"]):
        worst[ra["sequence_id"]].append(int(classify_change(c2["proximal_rotation_change_degrees"], cr["proximal_rotation_change_degrees"],
                                                            thresholds["proximal_rotation_change_degrees"]) == "candidate_discontinuity"))
    out["f2_proximal_discontinuities_by_sequence_top"] = sorted(((s, sum(v), len(v)) for s, v in worst.items()), key=lambda t: -t[1])[:5]
    return out, thresholds


def _strata(rows, key, bins):
    out = {}
    for label, (lo, hi) in bins.items():
        g = [r for r in rows if key(r) is not None and lo <= key(r) < hi]
        block = {"n": len(g), "sequences": len({r["sequence_id"] for r in g})}
        if g:
            for n in CANDIDATES:
                block[n] = {m: sum(r["candidates"][n][m] for r in g) / len(g) for m in PAIRED_METRICS}
                bp = [r["candidates"][n]["bend_plane_error_degrees"] for r in g if r["candidates"][n]["bend_plane_error_degrees"] is not None]
                block[n]["bend_plane_error_degrees_mean"] = sum(bp) / len(bp) if bp else None
                agree = [r["candidates"][n]["side_agrees_with_gt"] for r in g]
                block[n]["side_agreement_share"] = sum(agree) / len(agree)
                block[n]["abs_bend_angle_error_degrees_mean"] = sum(r["candidates"][n]["abs_bend_angle_error_degrees"] for r in g) / len(g)
            for a, b in PAIRS[:2]:
                block[f"{a}-{b}"] = {m: {k: v for k, v in _paired(g, a, b, m).items()
                                         if k in ("mean", "p50", "p95", "improved_share", "sequences", "sequences_improved")}
                                     for m in PAIRED_METRICS}
                block[f"{a}-{b}"]["coupling"] = {k: v["n"] for k, v in _coupling(g, a, b).items() if isinstance(v, dict)}
            pc = [r["f2_vs_f0_bend_plane_change_degrees"] for r in g if r.get("f2_vs_f0_bend_plane_change_degrees") is not None]
            block["f2_vs_f0_bend_plane_change_degrees"] = _stats(pc)
            block["f2_vs_f0_side_flips"] = sum(bool(r.get("f2_vs_f0_side_flip")) for r in g)
        out[label] = block
    return out


def _by_sequence(rows):
    out = {}
    for sid in sorted({r["sequence_id"] for r in rows}):
        g = [r for r in rows if r["sequence_id"] == sid]
        out[sid] = {"n": len(g), **{f"{SHORT[n]}_{m}": sum(r["candidates"][n][m] for r in g) / len(g)
                                    for n in CANDIDATES for m in PAIRED_METRICS},
                    "coupling_f2_vs_f0": {k: v["n"] for k, v in _coupling(g, "f2_e1_full_ik", "f0_h0_full_ik").items()
                                          if isinstance(v, dict)}}
    return out


def _block(rows, all_rows, bootstrap=True, thresholds=None):
    common = _common(rows)
    out = {"rows_all_eligible": len(all_rows), "common_valid_rows": len(common),
           "sequences": len({r["sequence_id"] for r in common}),
           "reference_unavailable": sum(r["reference"]["status"] != "known" for r in all_rows),
           "candidates": {n: _candidate_summary(common, all_rows, n) for n in CANDIDATES},
           "paired": {f"{a}-{b}": {m: _paired(common, a, b, m) for m in PAIRED_METRICS} for a, b in PAIRS},
           "coupling": {f"{a}-{b}": _coupling(common, a, b) for a, b in PAIRS[:2]}}
    if bootstrap and common:
        out["bootstrap"] = {f"{a}-{b}": {m: _bootstrap(_seq_values(common, lambda r: r["candidates"][a][m] - r["candidates"][b][m]))
                                         for m in PAIRED_METRICS} for a, b in PAIRS}
    out["temporal"], thresholds = _temporal(common, thresholds)
    return out, thresholds


def _decision(block):
    """Mechanical evaluation of the predeclared CASE criteria (classification is written up by hand)."""
    f2, f0 = block["candidates"]["f2_e1_full_ik"], block["candidates"]["f0_h0_full_ik"]
    boot = block["bootstrap"]["f2_e1_full_ik-f0_h0_full_ik"]
    paired = block["paired"]["f2_e1_full_ik-f0_h0_full_ik"]
    tmp = block["temporal"]
    out = {m: {"mean": paired[m]["mean"], "all_intervals_exclude_zero_negative": _excludes_zero_negative(boot[m]),
               "sequences_improved": paired[m]["sequences_improved"], "sequences": paired[m]["sequences"],
               "largest_sequence_share_of_net": paired[m]["largest_single_sequence_contribution"]["share_of_net"]}
           for m in PAIRED_METRICS}
    out["elbow_p95_change_cm"] = f2["elbow_error"]["p95"] - f0["elbow_error"]["p95"]
    out["material_elbow_p95_regression"] = out["elbow_p95_change_cm"] > DECISION["material_elbow_p95_regression_cm"]
    out["discontinuity_rate"] = {m: {SHORT[n]: tmp[n][m]["candidate_discontinuity_rate"] for n in CANDIDATES} for m in TEMPORAL_METRICS}
    out["instability_flag"] = any(
        tmp["f2_e1_full_ik"][m]["candidate_discontinuity_rate"] > 1.25 * tmp["f0_h0_full_ik"][m]["candidate_discontinuity_rate"]
        for m in TEMPORAL_METRICS)
    out["unavailable"] = {SHORT[n]: sum(block["candidates"][n]["unavailable_all_eligible"].values()) for n in CANDIDATES}
    out["invalidity_flag"] = out["unavailable"]["F2"] > out["unavailable"]["F0"]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("rig", "rest", "calibration", "chains", "bank", "h0_dir", "threedpw_raw", "semantics_dir", "ik61_dir",
                 "w62_dir", "w73_dir", "w74_dir", "w75_dir", "out_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()

    identities = {"w62_report": _sha(args.w62_dir / "report.json"), "w73_report": _sha(args.w73_dir / "report.json"),
                  "w73_rows_heldout": _sha(args.w73_dir / "rows_heldout.json"),
                  "w74_report": _sha(args.w74_dir / "report.json"), "w74_rows": _sha(args.w74_dir / "rows.json"),
                  "w75_report": _sha(args.w75_dir / "report.json"), "w75_rows_test": _sha(args.w75_dir / "rows_test.json")}
    for k, exp in PINNED.items():
        if identities[k] != exp:
            raise SystemExit(f"{k} identity mismatch: {identities[k]}")
    identities.update({"w61_report": _sha(args.ik61_dir / "report.json"),
                       "h0_validation": _sha(args.h0_dir / "prediction_validation.npy"),
                       "h0_test": _sha(args.h0_dir / "prediction_test.npy")})
    if not identities["h0_validation"].startswith(H0_VALIDATION_PREFIX):
        raise SystemExit("frozen H0 validation predictions changed")
    w75_report = json.loads((args.w75_dir / "report.json").read_text())
    if w75_report["identities"]["h0_test"] != identities["h0_test"]:
        raise SystemExit("frozen H0 test predictions differ from Worklog 75")
    rest = load_target_rest_pose(args.rest)
    if rest.provenance["source_fbx_sha256"] != _sha(args.rig):
        raise SystemExit("Blender rest snapshot differs from current FBX bytes")
    alignment = derive_rig_alignment(rest, load_rig_fk_calibration(args.calibration))
    target = target_chain_geometry(rest, load_ik_chain_set(args.chains).chains[0])
    bank = load_bank(args.bank)
    materialization = json.loads((args.h0_dir / "materialization_report.json").read_text())
    if materialization["bank_content_digest"] != bank.content_digest():
        raise SystemExit("H0 materialization was built from another FrameBank")
    identities["bank_content_digest"] = bank.content_digest()
    h0, _ = assemble_h0(bank, {"validation": args.h0_dir / "prediction_validation.npy",
                               "test": args.h0_dir / "prediction_test.npy"})
    w73 = {}
    for r in json.loads((args.w73_dir / "rows_heldout.json").read_text()):
        key = (r["sequence_id"], r["frame_index"])
        if key in w73:
            raise SystemExit(f"duplicate Worklog 73 row {key}")
        w73[key] = r
    w74 = {(r["sequence_id"], r["frame_index"]): r for r in json.loads((args.w74_dir / "rows.json").read_text())}

    # --- Gate 1: historical full-IK parity (Worklog 74 path, then the test path on the same rows) ---
    gate1, poses_by_key = gate1_worklog74(args, bank, w73, w74, target, rest, alignment)
    if not gate1["passed"]:
        raise SystemExit(f"STOP: Worklog 74 full-IK parity failed: {json.dumps(gate1, default=str)}")
    val_rows, val_acc = collect(bank, h0, w73, target, rest, alignment, args.threedpw_raw, "validation")
    gate1b = gate1_test_path(val_rows, poses_by_key)
    if not gate1b["passed"]:
        raise SystemExit(f"STOP: test-path full-IK parity on the Worklog 74 rows failed: {gate1b}")

    # --- Gate 2: Worklog 75 endpoint parity on the full test population ---
    w75 = {(r["sequence_id"], r["frame_index"]): r for r in json.loads((args.w75_dir / "rows_test.json").read_text())}
    test_rows, test_acc = collect(bank, h0, w73, target, rest, alignment, args.threedpw_raw, "test")
    gate2 = gate2_worklog75(test_rows, w75)
    if not gate2["passed"]:
        raise SystemExit(f"STOP: Worklog 75 endpoint parity failed: {json.dumps(gate2, default=str)}")

    # --- Test full-arm evaluation ---
    totals = Counter()
    for acc in test_acc.values():
        totals.update(acc)
    pooled, thresholds = _block(test_rows, test_rows)
    common = _common(test_rows)
    w74_rows = [r for r in val_rows if (r["sequence_id"], r["frame_index"]) in w74]
    report = {
        "schema": "animcv_framepose_visual_full_arm_ik_v1", "identities": identities,
        "labels": {"runtime": "F0/F1/F2: frozen H0 joints + yaw_H0, Worklog 61 IK, FrameBank detector plane, Worklog 73 E0/E1 chain f",
                   "evaluation": "Worklog 62 GT-anchored reference arm (GT direction + GT reach + GT side); GT never enters F0/F1/F2"},
        "target_chain": target.to_dict(), "tolerances": {"cm": TOL_CM, "deg": TOL_DEG, "quaternion": TOL_QUAT},
        "gate1_worklog74_full_ik_parity": gate1, "gate1_test_path_parity": gate1b, "gate2_worklog75_endpoint_parity": gate2,
        "bootstrap_contract": BOOTSTRAP, "temporal_contract": TEMPORAL_CONTRACT, "decision_contract": DECISION,
        "test": {"accounting": test_acc, "accounting_totals": dict(totals),
                 "bank_test_rows": len(bank.indices("test")),
                 "pooled": pooled, "decision_inputs": _decision(pooled),
                 "by_sequence": _by_sequence(common),
                 "bend_regimes_gt": _strata(common, lambda r: r["gt_bend_degrees"], BEND_REGIMES),
                 "bend_regimes_h0": _strata(common, lambda r: r["h0_bend_degrees"], BEND_REGIMES),
                 "side_status": {s: _strata([r for r in common if r["candidates"]["f0_h0_full_ik"]["side_status"] == s],
                                            lambda r: 0.0, {"all": (0, 1)})["all"]
                                 for s in sorted({r["candidates"]["f0_h0_full_ik"]["side_status"] for r in common})},
                 "abs_gt_chain_f": _strata(common, lambda r: abs(r["f_gt"]),
                                           {"near_plane": (0, STABLE), "intermediate": (STABLE, 0.5), "out_of_plane": (0.5, 1.01)}),
                 "temporal_by_gt_bend_regime": {label: _temporal([r for r in common if lo <= r["gt_bend_degrees"] < hi], thresholds)[0]
                                                for label, (lo, hi) in BEND_REGIMES.items()}},
        "validation": {"note": "selection-contaminated split (chose E0/E1 and H0 epochs); never merged with test",
                       "accounting": val_acc,
                       "worklog74_population": _block(w74_rows, w74_rows, bootstrap=False)[0],
                       "all_eligible": _block(val_rows, val_rows, bootstrap=False)[0]},
        "owner_rows": {},
    }
    by_key = {(r["sequence_id"], r["frame_index"]): r for r in test_rows}
    for sid, fi in (("3dpw:downtown_bar_00:actor0", 775), ("3dpw:downtown_sitOnStairs_00:actor0", 325),
                    ("3dpw:downtown_runForBus_01:actor0", 185), ("3dpw:flat_packBags_00:actor0", 446)):
        r = by_key.get((sid, fi))
        report["owner_rows"][f"{sid}#{fi}"] = None if r is None else {
            n: {m: r["candidates"][n].get(m) for m in (*POSE_METRICS, "side_agrees_with_gt", "status")} for n in CANDIDATES}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "rows_test.json").write_text(json.dumps(test_rows))
    (args.out_dir / "rows_validation.json").write_text(json.dumps(val_rows))
    path = args.out_dir / "report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    p = report["test"]["pooled"]
    print(json.dumps({"sha256": _sha(path), "gate1": {k: gate1[k] for k in ("rows", "passed", "max_abs_diff")},
                      "gate1b": gate1b, "gate2": {k: gate2[k] for k in ("rows", "passed", "max_abs_diff", "unavailable")},
                      "test_common": p["common_valid_rows"],
                      "means": {SHORT[n]: {m: p["candidates"][n][m]["mean"] for m in PAIRED_METRICS} for n in CANDIDATES},
                      "decision_inputs": report["test"]["decision_inputs"]}, indent=1, default=str))


if __name__ == "__main__":
    main()
