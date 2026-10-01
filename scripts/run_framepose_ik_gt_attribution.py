#!/usr/bin/env python3
"""EVALUATION ONLY: GT-anchored attribution of Worklog 61 left-arm IK vs Worklog 60 FK.

H0 / runtime observable: AnimationSemantics v2 sidecars, Worklog 60 FK, Worklog 61 IK.
GT / evaluation oracle: 3DPW jointPositions on the same frame rows (never fed back).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_root_motion_contact_h0_replay import _matched_oracle_sequence  # noqa: E402

from motion.animation_semantics_v2 import load_animation_semantics_v2  # noqa: E402
from retarget.framepose_fk import derive_rig_alignment, load_rig_fk_calibration  # noqa: E402
from retarget.framepose_ik_gt_attribution import VARIANTS, attribute_frame, measure_chain  # noqa: E402
from retarget.framepose_object_owner import load_object_owned_fk  # noqa: E402
from retarget.framepose_target_rest import load_target_rest_pose  # noqa: E402
from retarget.framepose_two_bone_ik import (  # noqa: E402
    chain_forward_kinematics, load_ik_chain_set, load_two_bone_ik, target_chain_geometry,
)

SEQUENCES = ("3dpw:courtyard_dancing_00:actor0", "3dpw:outdoors_crosscountry_00:actor0",
             "3dpw:courtyard_hug_00:actor1")


def _stats(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return {"n": 0}

    def pct(p):
        k = (len(values) - 1) * p / 100
        lo, hi = math.floor(k), math.ceil(k)
        return values[lo] + (values[hi] - values[lo]) * (k - lo)
    mean = sum(values) / len(values)
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
    return {"n": len(values), "mean": mean, "cv": sd / mean if mean else None, "min": values[0],
            "p05": pct(5), "p50": pct(50), "p95": pct(95), "max": values[-1]}


def _rank(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2
        i = j + 1
    return ranks


def _pearson(x, y):
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    if len(pairs) < 3:
        return None
    xs, ys = zip(*pairs)
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    den = math.sqrt(sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys))
    return sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / den if den else None


def _spearman(x, y):
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    if len(pairs) < 3:
        return None
    xs, ys = zip(*pairs)
    return _pearson(_rank(list(xs)), _rank(list(ys)))


def _variant_summary(rows, chain_length):
    out = {}
    for name in VARIANTS:
        errors = [r[f"{name}_endpoint_error"] for r in rows]
        out[name] = {"endpoint_error_cm": _stats(errors),
                     "endpoint_error_over_chain": _stats(e / chain_length for e in errors),
                     "elbow_error_cm": _stats(r[f"{name}_elbow_error"] for r in rows),
                     "clamps": dict(Counter(r[f"{name}_clamp"] for r in rows))}
    paired = [r["ik_h0_endpoint_error"] - r["fk_h0_endpoint_error"] for r in rows]
    out["ik_h0_minus_fk_h0_cm"] = _stats(paired)
    out["ik_h0_better_than_fk_h0_frames"] = sum(d < 0 for d in paired)
    return out


def _geometry_summary(rows):
    agree = [r["bend_side_agrees"] for r in rows if r["bend_side_agrees"] is not None]
    return {
        "endpoint_direction_error_degrees": _stats(r["endpoint_direction_error_degrees"] for r in rows),
        "reach_fraction_error": _stats(r["reach_fraction_error"] for r in rows),
        "abs_reach_fraction_error": _stats(abs(r["reach_fraction_error"]) for r in rows),
        "ratio_log_error": _stats(r["ratio_log_error"] for r in rows),
        "abs_ratio_log_error": _stats(abs(r["ratio_log_error"]) for r in rows),
        "bend_angle_error_degrees": _stats(r["bend_angle_error_degrees"] for r in rows),
        "abs_bend_angle_error_degrees": _stats(abs(r["bend_angle_error_degrees"]) for r in rows),
        "bend_plane_error_degrees": _stats(r["bend_plane_error_degrees"] for r in rows),
        "bend_side_agree": sum(agree), "bend_side_flip": len(agree) - sum(agree),
        "bend_side_compared": len(agree),
        "h0_side_status": dict(Counter(r["h0_side_status"] for r in rows)),
        "gt_side_status": dict(Counter(r["gt_side_status"] for r in rows)),
    }


def _bend_by(rows, key, bins):
    out = {}
    for lo, hi in bins:
        group = [r for r in rows if r[key] is not None and lo <= r[key] < hi]
        agree = [r["bend_side_agrees"] for r in group if r["bend_side_agrees"] is not None]
        out[f"{lo}-{hi}"] = {"n": len(group), "side_flip": len(agree) - sum(agree),
                             "bend_plane_error_degrees": _stats(r["bend_plane_error_degrees"] for r in group),
                             "elbow_error_h0_side_cm": _stats(r["elbow_error_h0_side_on_gt_target"] for r in group)}
    return out


def _strata(rows, key, transform=abs):
    values = sorted(transform(r[key]) for r in rows)
    edges = [values[int(len(values) * q / 4)] for q in range(1, 4)]
    out = {}
    bounds = [-math.inf, *edges, math.inf]
    for i in range(4):
        group = [r for r in rows if bounds[i] <= transform(r[key]) < bounds[i + 1]]
        out[f"Q{i + 1}"] = {"range": [min(transform(r[key]) for r in group), max(transform(r[key]) for r in group)]
                            if group else None, "n": len(group),
                            **{f"{v}_mean_cm": (sum(r[f"{v}_endpoint_error"] for r in group) / len(group)
                                                if group else None) for v in VARIANTS}}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("rig", "rest", "calibration", "chains", "semantics_dir", "fk60_dir",
                 "ik61_dir", "threedpw_raw", "out_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()

    rest = load_target_rest_pose(args.rest)
    if rest.provenance["source_fbx_sha256"] != hashlib.sha256(args.rig.read_bytes()).hexdigest():
        raise ValueError("Blender rest snapshot differs from current FBX bytes")
    calibration = load_rig_fk_calibration(args.calibration)
    chains = load_ik_chain_set(args.chains)
    chain = chains.chains[0]
    target = target_chain_geometry(rest, chain)
    alignment = derive_rig_alignment(rest, calibration)
    source_report = json.loads((args.semantics_dir / "report.json").read_text())
    fk60_report = json.loads((args.fk60_dir / "report.json").read_text())
    ik61_report = json.loads((args.ik61_dir / "report.json").read_text())
    source_files = {row["sequence_id"]: row for row in source_report["per_sequence"]}
    names = (chain.source_root, chain.source_mid, chain.source_end)

    rows, stability, exclusions, consistency = [], {}, {}, {"ik_endpoint_max": 0.0, "fk_endpoint_max": 0.0}
    for sid in SEQUENCES:
        semantics = load_animation_semantics_v2(args.semantics_dir / source_files[sid]["file"])
        if semantics.content_digest() != source_files[sid]["content_digest"]:
            raise ValueError(f"{sid}: semantics digest changed")
        ik = load_two_bone_ik(args.ik61_dir / ik61_report["sequences"][sid]["ik_file"])
        if ik.provenance["semantics_digest"] != semantics.content_digest():
            raise ValueError(f"{sid}: Worklog 61 IK was built from other semantics")
        comparison = {r["frame_index"]: r for r in json.loads(
            (args.ik61_dir / ik61_report["sequences"][sid]["comparison_file"]).read_text())}
        fk = load_object_owned_fk(args.fk60_dir / fk60_report["sequences"][sid]["file"])
        fk_samples = {(s.frame_index, s.target_bone): s for s in fk.pose_bone_samples}
        indices = [f.frame_index for f in semantics.frames]
        oracle = _matched_oracle_sequence(sid, args.threedpw_raw, indices)  # GT: evaluation only
        lookup = {n: i for i, n in enumerate(semantics.joint_names)}
        ik_by = {s.frame_index: s for s in ik.samples}
        gt_lengths, h0_lengths_all = {"upper": [], "lower": [], "ratio": []}, {"upper": [], "lower": [], "ratio": []}
        excluded = Counter()
        for frame, gt_frame in zip(semantics.frames, oracle.frames):
            if gt_frame.frame_index != frame.frame_index:
                raise ValueError("GT/H0 frame alignment broken")
            gt_points = [gt_frame.points[n] for n in names]
            gt_valid = all(p.observation_valid for p in gt_points)
            yaw = frame.root_orientation.yaw_radians
            if gt_valid:
                g = [p.position for p in gt_points]
                u, l = math.dist(g[0], g[1]), math.dist(g[1], g[2])
                gt_lengths["upper"].append(u)
                gt_lengths["lower"].append(l)
                gt_lengths["ratio"].append(u / l)
            sample = ik_by[frame.frame_index]
            if sample.proximal_status != "known":
                excluded["h0_ik_unavailable:" + sample.reason] += 1
                continue
            if not gt_valid:
                excluded["gt_invalid"] += 1
                continue
            h0_pos = [frame.articulation.joint_positions[lookup[n]] for n in names]
            h0 = measure_chain(*h0_pos, yaw, alignment)
            gt = measure_chain(*[p.position for p in gt_points], yaw, alignment)  # same runtime transform
            if h0 is None or gt is None:
                excluded["degenerate_measure"] += 1
                continue
            p = fk_samples[(frame.frame_index, target.proximal)]
            d = fk_samples[(frame.frame_index, target.distal)]
            fk_mid, fk_end = chain_forward_kinematics(rest, target, p.rotation_local, d.rotation_local)
            row = attribute_frame(h0, gt, target, rest, fk_h0_endpoint=fk_end, fk_h0_mid=fk_mid).to_dict()
            # Consistency with the persisted Worklog 61 IK and Worklog 60 FK.
            e_h0 = [target.root_head[i] + h0.direction[i] * h0.reach_fraction * target.max_reach for i in range(3)]
            consistency["ik_endpoint_max"] = max(consistency["ik_endpoint_max"],
                                                 math.dist(e_h0, sample.solved_endpoint))
            consistency["fk_endpoint_max"] = max(consistency["fk_endpoint_max"], math.dist(
                fk_end, comparison[frame.frame_index]["fk_endpoint"]))
            for key in ("upper", "lower", "ratio"):
                h0_lengths_all[key].append({"upper": h0.upper_length, "lower": h0.lower_length,
                                            "ratio": h0.ratio}[key])
            rows.append({"sequence_id": sid, "frame_index": frame.frame_index,
                         "w61_proximal_delta_degrees": comparison[frame.frame_index]["proximal_delta_degrees"],
                         "w61_distal_delta_degrees": comparison[frame.frame_index]["distal_delta_degrees"],
                         "gt_upper_m": gt.upper_length, "gt_lower_m": gt.lower_length,
                         "h0_upper_m": h0.upper_length, "h0_lower_m": h0.lower_length, **row})
        matched = [r for r in rows if r["sequence_id"] == sid]
        stability[sid] = {
            "gt_all_valid_rows": {k: _stats(v) for k, v in gt_lengths.items()},
            "gt_matched_rows": {"upper": _stats(r["gt_upper_m"] for r in matched),
                                "lower": _stats(r["gt_lower_m"] for r in matched),
                                "ratio": _stats(r["gt_ratio"] for r in matched)},
            "h0_matched_rows": {"upper": _stats(r["h0_upper_m"] for r in matched),
                                "lower": _stats(r["h0_lower_m"] for r in matched),
                                "ratio": _stats(r["h0_ratio"] for r in matched)},
        }
        exclusions[sid] = {"semantics_frames": len(semantics.frames), "matched_rows": len(matched),
                           **dict(excluded)}

    if consistency["ik_endpoint_max"] > 1e-9 or consistency["fk_endpoint_max"] > 1e-9:
        raise ValueError(f"reconstruction differs from persisted Worklog 60/61: {consistency}")
    L = target.max_reach
    by_seq = {sid: [r for r in rows if r["sequence_id"] == sid] for sid in SEQUENCES}
    large_delta = [r for r in rows if max(r["w61_proximal_delta_degrees"], r["w61_distal_delta_degrees"]) >= 10]
    report = {
        "schema": "animcv_framepose_ik_gt_attribution_v1",
        "labels": {"H0": "runtime observable (AnimationSemantics v2 / Worklog 60 FK / Worklog 61 IK)",
                   "GT": "3DPW jointPositions, evaluation oracle only; never enters runtime"},
        "regime": source_report["regime"], "h0_identity": source_report["h0_identity"],
        "shared_transform": "A*Rz(-yaw_H0) applied to H0 and GT alike; GT yaw unused",
        "gt_endpoint": "E_GT = target_root + GT_reach_fraction*(L1+L2)*GT_endpoint_direction",
        "target_chain": target.to_dict(),
        "reconstruction_consistency_cm": consistency,
        "exclusions": exclusions, "stability": stability,
        "variants": {"pooled": _variant_summary(rows, L),
                     **{sid: _variant_summary(v, L) for sid, v in by_seq.items()}},
        "geometry_error": {"pooled": _geometry_summary(rows),
                           **{sid: _geometry_summary(v) for sid, v in by_seq.items()}},
        "fk_ik_disagreement_cm": {
            "h0": _stats(r["fk_ik_h0_disagreement"] for r in rows),
            "gt_true_proportion_adaptation": _stats(r["fk_ik_gt_disagreement"] for r in rows),
            "pearson_h0_vs_gt": _pearson([r["fk_ik_h0_disagreement"] for r in rows],
                                         [r["fk_ik_gt_disagreement"] for r in rows])},
        "bend_by_gt_angle": _bend_by(rows, "gt_bend_angle_degrees",
                                     [(0, 15), (15, 45), (45, 90), (90, 181)]),
        "bend_on_large_w61_delta": {"definition": "max(W61 proximal, distal FK-vs-IK delta) >= 10 deg",
                                    **_geometry_summary(large_delta), "n": len(large_delta),
                                    "variants": _variant_summary(large_delta, L) if large_delta else None},
        "correlation": {}, "strata": {},
    }
    factors = {"abs_ratio_log_error": [abs(r["ratio_log_error"]) for r in rows],
               "abs_reach_fraction_error": [abs(r["reach_fraction_error"]) for r in rows],
               "endpoint_direction_error_degrees": [r["endpoint_direction_error_degrees"] for r in rows],
               "gt_bend_angle_degrees": [r["gt_bend_angle_degrees"] for r in rows]}
    for variant in ("fk_h0", "ik_h0", "h0_direction_gt_reach", "gt_direction_h0_reach"):
        errors = [r[f"{variant}_endpoint_error"] for r in rows]
        report["correlation"][variant] = {k: {"pearson": _pearson(v, errors), "spearman": _spearman(v, errors)}
                                          for k, v in factors.items()}
    for key in ("ratio_log_error", "reach_fraction_error", "endpoint_direction_error_degrees"):
        report["strata"][key] = _strata(rows, key)
    report["strata"]["gt_bend_angle_degrees"] = _strata(rows, "gt_bend_angle_degrees", lambda v: v)

    cases = {label: (c["sequence_id"], c["frame_index"]) for label, c in ik61_report["cases"].items()}
    extremes = {"largest_ratio_error": lambda r: abs(r["ratio_log_error"]),
                "largest_reach_error": lambda r: abs(r["reach_fraction_error"]),
                "largest_direction_error": lambda r: r["endpoint_direction_error_degrees"]}
    for label, key in extremes.items():
        best = max(rows, key=lambda r: (key(r), r["sequence_id"], r["frame_index"]))
        cases[label] = (best["sequence_id"], best["frame_index"])
    report["owner_cases"] = {}
    row_by = {(r["sequence_id"], r["frame_index"]): r for r in rows}
    for label, key in cases.items():
        overlap = sorted(other for other, k in cases.items() if k == key and other != label)
        report["owner_cases"][label] = {"sequence_id": key[0], "frame_index": key[1],
                                        "overlaps": overlap, "row": row_by.get(key),
                                        "excluded": None if key in row_by else "not a matched H0-IK-known & GT-valid row"}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "rows.json").write_text(json.dumps(rows, indent=1))
    columns = ["sequence_id", "frame_index", "gt_ratio", "h0_ratio", "gt_reach_fraction", "h0_reach_fraction",
               "endpoint_direction_error_degrees", "bend_plane_error_degrees", "bend_side_agrees",
               "gt_bend_angle_degrees", "h0_bend_angle_degrees",
               *[f"{v}_endpoint_error" for v in VARIANTS],
               "w61_proximal_delta_degrees", "w61_distal_delta_degrees"]
    with (args.out_dir / "attribution_table.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    output = args.out_dir / "report.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({"sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "rows": len(rows),
                      "exclusions": exclusions}, indent=2))


if __name__ == "__main__":
    main()
