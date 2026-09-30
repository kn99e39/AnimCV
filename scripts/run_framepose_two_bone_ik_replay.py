#!/usr/bin/env python3
"""Replay current-policy v2: unchanged Worklog 60 FK vs two-bone IK on the left arm."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from motion.animation_semantics_v2 import load_animation_semantics_v2
from retarget.framepose_fk import load_rig_fk_calibration
from retarget.framepose_object_owner import load_object_owned_fk, solve_object_owned_fk
from retarget.framepose_target_rest import load_target_rest_pose
from retarget.framepose_two_bone_ik import (
    compare_fk_ik, load_ik_chain_set, save_two_bone_ik, solve_framepose_two_bone_ik,
    target_chain_geometry,
)
from rig.bone_mapping import load_bone_mapping_profile

# Worklog 60 owner cases, unchanged; bent/straight cases are added by a fixed rule below.
CASES = {
    "known_good": ("3dpw:courtyard_dancing_00:actor0", 424),
    "turning_invalid": ("3dpw:outdoors_crosscountry_00:actor0", 202),
    "turning_review": ("3dpw:outdoors_crosscountry_00:actor0", 211),
    "dynamic": ("3dpw:outdoors_crosscountry_00:actor0", 462),
    "tracking_loss": ("3dpw:courtyard_hug_00:actor1", 312),
    "invalid_endpoint": ("3dpw:courtyard_hug_00:actor1", 406),
}


def _stats(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return {"n": 0}

    def pct(p):
        k = (len(values) - 1) * p / 100
        lo, hi = math.floor(k), math.ceil(k)
        return values[lo] + (values[hi] - values[lo]) * (k - lo)
    return {"n": len(values), "mean": sum(values) / len(values), "min": values[0],
            "p05": pct(5), "p50": pct(50), "p95": pct(95), "max": values[-1]}


def _source_proportion(semantics, frame, chain):
    lookup = {name: i for i, name in enumerate(semantics.joint_names)}
    r, m, e = (frame.articulation.joint_positions[lookup[n]]
               for n in (chain.source_root, chain.source_mid, chain.source_end))
    lower = math.dist(m, e)
    return math.dist(r, m) / lower if lower > 1e-8 else None


def _summary(rows, samples, proportions, max_reach):
    ik_known = [s for s in samples if s.proximal_status == "known"]
    both = [r for r in rows if r.status == "both_known"]
    observed = {s.frame_index for s in ik_known if s.bend_plane_status == "observed"}
    groups = {"fully_valid_observed_bend": [r for r in both if r.frame_index in observed],
              "degenerate_bend_fallback": [r for r in both if r.frame_index not in observed]}
    out = {
        "frames": len(samples),
        "status": dict(Counter(r.status for r in rows)),
        "ik_known": len(ik_known),
        "bend_plane_status": dict(Counter(s.bend_plane_status for s in samples)),
        "unavailable_reasons": dict(Counter(s.reason for s in samples if s.reason)),
        "clamp_status": dict(Counter(s.clamp_status for s in ik_known)),
        "clamp_rate": (sum(s.clamp_status != "none" for s in ik_known) / len(ik_known)
                       if ik_known else None),
        "source_reach_fraction": _stats(s.source_reach_fraction for s in ik_known),
        "source_bend_angle_degrees": _stats(s.source_bend_angle_degrees for s in ik_known),
        "source_upper_over_lower": _stats(proportions[s.frame_index] for s in ik_known),
        "groups": {},
    }
    for name, group in groups.items():
        fk_err = [r.fk_endpoint_error for r in group]
        ik_err = [r.ik_endpoint_error for r in group]
        out["groups"][name] = {
            "n": len(group),
            "fk_endpoint_error_cm": _stats(fk_err),
            "ik_endpoint_error_cm": _stats(ik_err),
            "fk_endpoint_error_over_chain": _stats(v / max_reach for v in fk_err),
            "ik_endpoint_error_over_chain": _stats(v / max_reach for v in ik_err),
            "ik_better_or_equal_frames": sum(i <= f + 1e-9 for f, i in zip(fk_err, ik_err)),
            "proximal_delta_degrees": _stats(r.proximal_delta_degrees for r in group),
            "distal_delta_degrees": _stats(r.distal_delta_degrees for r in group),
        }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("rig", "rest", "mapping", "calibration", "chains",
                 "semantics_dir", "fk60_dir", "out_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()

    rest = load_target_rest_pose(args.rest)
    if rest.provenance["source_fbx_sha256"] != hashlib.sha256(args.rig.read_bytes()).hexdigest():
        raise ValueError("Blender rest snapshot differs from current FBX bytes")
    mapping = load_bone_mapping_profile(args.mapping)
    calibration = load_rig_fk_calibration(args.calibration)
    chains = load_ik_chain_set(args.chains)
    chain = chains.chains[0]
    geometry = target_chain_geometry(rest, chain)
    source_report = json.loads((args.semantics_dir / "report.json").read_text())
    fk60_report = json.loads((args.fk60_dir / "report.json").read_text())
    source_files = {row["sequence_id"]: row for row in source_report["per_sequence"]}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = {"schema": "animcv_framepose_two_bone_ik_replay_v1",
              "regime": source_report["regime"], "h0_identity": source_report["h0_identity"],
              "rest_snapshot_digest": rest.digest(), "target_chain": geometry.to_dict(),
              "sequences": {}, "pooled": None, "cases": {}}
    pooled_rows, pooled_samples, pooled_props = [], [], {}
    per = {}
    for sid in sorted({sid for sid, _ in CASES.values()}):
        source = source_files[sid]
        semantics = load_animation_semantics_v2(args.semantics_dir / source["file"])
        if semantics.content_digest() != source["content_digest"]:
            raise ValueError(f"{sid}: persisted semantics digest changed")
        fk = solve_object_owned_fk(semantics, rest, mapping, calibration)
        old = load_object_owned_fk(args.fk60_dir / fk60_report["sequences"][sid]["file"])
        if fk.to_dict() != old.to_dict():
            raise ValueError(f"{sid}: object-owned FK differs from Worklog 60")
        ik = solve_framepose_two_bone_ik(semantics, rest, calibration, chains)
        rows = compare_fk_ik(fk.pose_bone_samples, ik, rest, chains)
        proportions = {f.frame_index: _source_proportion(semantics, f, chain) for f in semantics.frames}
        base = sid.replace(":", "_")
        save_two_bone_ik(ik, args.out_dir / "sequences" / f"{base}.ik.json")
        (args.out_dir / "sequences" / f"{base}.fk_vs_ik.json").write_text(
            json.dumps([r.to_dict() for r in rows], indent=1))
        report["sequences"][sid] = {
            "ik_file": f"sequences/{base}.ik.json",
            "comparison_file": f"sequences/{base}.fk_vs_ik.json",
            "source_semantics_digest": semantics.content_digest(),
            "worklog60_object_fk_exact_match": True,
            **_summary(rows, ik.samples, proportions, geometry.max_reach)}
        per[sid] = (ik, rows, fk)
        pooled_rows += rows
        pooled_samples += list(ik.samples)
        pooled_props.update({(sid, k): v for k, v in proportions.items()})
    # _summary keys proportions by frame_index; pool with sequence-qualified keys.
    report["pooled"] = _summary(
        pooled_rows, pooled_samples,
        {s.frame_index: None for s in pooled_samples}, geometry.max_reach)
    report["pooled"]["source_upper_over_lower"] = _stats(
        pooled_props[(sid, s.frame_index)] for sid, (ik, _, _) in per.items()
        for s in ik.samples if s.proximal_status == "known")

    # Fixed-rule extra owner cases over all IK-known observed-bend frames.
    known = [(sid, s) for sid, (ik, _, _) in per.items() for s in ik.samples
             if s.proximal_status == "known" and s.bend_plane_status == "observed"]
    bent = max(known, key=lambda x: (x[1].source_bend_angle_degrees, x[0], x[1].frame_index))
    straight = min(known, key=lambda x: (x[1].source_bend_angle_degrees, x[0], x[1].frame_index))
    cases = dict(CASES)
    cases["strongly_bent"] = (bent[0], bent[1].frame_index)
    cases["near_straight"] = (straight[0], straight[1].frame_index)
    report["case_rule"] = ("Worklog 60 cases unchanged; strongly_bent / near_straight = max / min "
                           "source bend angle over IK-known observed-bend frames of all three sequences")
    for label, (sid, frame_index) in cases.items():
        ik, rows, fk = per[sid]
        sample = next((s for s in ik.samples if s.frame_index == frame_index), None)
        row = next((r for r in rows if r.frame_index == frame_index), None)
        root = next((s for s in fk.rig_rotations if s.frame_index == frame_index), None)
        if sample is None or row is None or root is None:
            raise ValueError(f"{label}: case frame missing")
        report["cases"][label] = {"sequence_id": sid, "frame_index": frame_index,
                                  "ik": sample.to_dict(), "fk_vs_ik": row.to_dict(),
                                  "rig_rotation": root.to_dict()}
    output = args.out_dir / "report.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({"report": str(output),
                      "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                      "pooled": report["pooled"],
                      "sequences": {k: {kk: v[kk] for kk in ("status", "unavailable_reasons",
                                                           "bend_plane_status", "clamp_status", "groups")}
                                    for k, v in report["sequences"].items()},
                      "cases": {k: {"sid": v["sequence_id"], "frame": v["frame_index"],
                                    "reason": v["ik"]["reason"], "bend": v["ik"]["bend_plane_status"],
                                    "fk_err": v["fk_vs_ik"]["fk_endpoint_error"],
                                    "ik_err": v["fk_vs_ik"]["ik_endpoint_error"],
                                    "angle": v["ik"]["source_bend_angle_degrees"]}
                              for k, v in report["cases"].items()}}, indent=2))


if __name__ == "__main__":
    main()
