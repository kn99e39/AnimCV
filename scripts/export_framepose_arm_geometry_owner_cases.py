#!/usr/bin/env python3
"""EVALUATION ONLY (Worklog 78): Blender review cases for the learned reach / bend-side owner.

Fixed selection rules on the Worklog 78 test rows/pairs (render with the unchanged Worklog 77
``review_framepose_visual_arm_ownership_blender.py``; same rig/camera/object-rotation contract):
  * Worklog 76/77 owner frames: downtown_bar_00 a0 #514, #775; downtown_sitOnStairs_00 a0 #325;
    flat_packBags_00 a0 #446;
  * largest G1-B1 elbow improvement / regression;
  * near-folded wrist/elbow conflict: GT bend >= 90, B1 wrist better and elbow worse than B0,
    largest min(wrist gain, elbow loss) (the Worklog 77 rule);
  * largest G1-B1 upper-arm motion-residual improvement / regression (adjacent-frame strips);
  * most stable / least stable sequences for G1 vs B1 (mean upper-arm residual change, >= 100
    pairs): strips around the median / worst G1 pair.
Drawn: B0 blue, B1 magenta, G0 orange, G1 cyan, GT reference green, E_GT yellow.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_framepose_visual_full_arm_cases import object_delta  # noqa: E402

from retarget.framepose_fk import derive_rig_alignment, load_rig_fk_calibration  # noqa: E402
from retarget.framepose_target_rest import load_target_rest_pose  # noqa: E402
from retarget.framepose_two_bone_ik import chain_forward_kinematics, load_ik_chain_set, target_chain_geometry  # noqa: E402

DRAW = {"B0": ("b0", (0.15, 0.45, 1.0, 1.0)), "B1": ("b1", (0.9, 0.15, 0.85, 1.0)),
        "G0": ("g0", (1.0, 0.55, 0.1, 1.0)), "G1": ("g1", (0.1, 0.85, 0.95, 1.0))}
SHOW = ("b0", "b1", "g0", "g1", "c111")
OWNERS = (("3dpw:downtown_bar_00:actor0", 514), ("3dpw:downtown_bar_00:actor0", 775),
          ("3dpw:downtown_sitOnStairs_00:actor0", 325), ("3dpw:flat_packBags_00:actor0", 446))


def _caption(label, r, pair=None):
    c, lr = r["candidates"], r["learned"]
    lines = [f"W78 learned reach/bend-side owner (3DPW test) | {label}",
             f"{r['sequence_id']}  frame #{r['frame_index']}  bank {r['position']}   GT bend {r['gt_bend_degrees']:.0f} deg",
             (f"reach: GT {r['gt_reach']:.2f}  H0 {r['h0_reach']:.2f}  G0 {lr['G0']['reach']:.2f}  G1 {lr['G1']['reach']:.2f}"
              f"   chain f: E1 {r['f_e1']:+.2f}  GT {r['f_gt']:+.2f}"),
             "blue B0 = H0 dir+H0 reach/side   magenta B1 = E1 dir+H0 reach/side",
             "orange G0 = E1 dir+learned (2D only)   cyan G1 = E1 dir+learned (2D+Qwen)   green = GT ref   yellow = E_GT",
             "elbow cm: " + "  ".join(f"{n.upper()} {c[n]['elbow_error']:.1f}" for n in SHOW) + "  (C111 = E1+GT reach/side oracle)",
             "wrist cm: " + "  ".join(f"{n.upper()} {c[n]['wrist_error']:.1f}" for n in SHOW),
             "side err deg: " + "  ".join(f"{n.upper()} {c[n]['side_error_degrees']:.0f}" for n in ("b1", "g0", "g1") if c[n].get("side_error_degrees") is not None)]
    if pair is not None:
        p = pair["candidates"]
        lines.append(f"next pair #{pair['frame_index']}->#{pair['next_frame_index']} upper-arm motion residual (deg): "
                     + "  ".join(f"{n.upper()} {p[n]['upper_direction_residual_degrees']:.1f}" for n in ("b0", "b1", "g0", "g1")))
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("rig", "rest", "calibration", "chains", "eval_dir", "out"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    rest = load_target_rest_pose(args.rest)
    if rest.provenance["source_fbx_sha256"] != hashlib.sha256(args.rig.read_bytes()).hexdigest():
        raise ValueError("Blender rest snapshot differs from current FBX bytes")
    alignment = derive_rig_alignment(rest, load_rig_fk_calibration(args.calibration))
    target = target_chain_geometry(rest, load_ik_chain_set(args.chains).chains[0])
    rows = json.loads((args.eval_dir / "rows_test.json").read_text())
    pairs = json.loads((args.eval_dir / "pairs_test.json").read_text())
    rows = [r for r in rows if all(r["candidates"][n]["status"] == "known" for n in SHOW)]
    by_key = {(r["sequence_id"], r["frame_index"]): r for r in rows}
    by_order = {(r["sequence_id"], r["bank_order"]): r for r in rows}
    pair_by = {(p["sequence_id"], p["frame_index"]): p for p in pairs}
    key = lambda r: (r["sequence_id"], r["frame_index"])  # noqa: E731
    e = lambda r, n, m: r["candidates"][n][m]  # noqa: E731
    de = lambda r: e(r, "g1", "elbow_error") - e(r, "b1", "elbow_error")  # noqa: E731

    stills = {f"w76_owner_{s.split(':')[1]}_{f}": by_key[(s, f)] for s, f in OWNERS}
    stills["largest_g1_elbow_improvement"] = min(rows, key=lambda r: (de(r), key(r)))
    stills["largest_g1_elbow_regression"] = max(rows, key=lambda r: (de(r), key(r)))
    conflict = [r for r in rows if r["gt_bend_degrees"] >= 90 and e(r, "b1", "wrist_error") < e(r, "b0", "wrist_error")
                and e(r, "b1", "elbow_error") > e(r, "b0", "elbow_error")]
    stills["near_folded_wrist_elbow_conflict"] = max(conflict, key=lambda r: (min(e(r, "b0", "wrist_error") - e(r, "b1", "wrist_error"),
                                                                                    e(r, "b1", "elbow_error") - e(r, "b0", "elbow_error")), key(r)))

    def window(p):
        a = by_key.get((p["sequence_id"], p["frame_index"]))
        frames = [by_order.get((a["sequence_id"], a["bank_order"] + k)) for k in (-1, 0, 1, 2)] if a else []
        return [f for f in frames if f is not None]
    ur = lambda p: p["candidates"]["g1"]["upper_direction_residual_degrees"] - p["candidates"]["b1"]["upper_direction_residual_degrees"]  # noqa: E731
    strips = {"largest_g1_temporal_improvement": window(min(pairs, key=lambda p: (ur(p), p["sequence_id"], p["frame_index"]))),
              "largest_g1_temporal_regression": window(max(pairs, key=lambda p: (ur(p), p["sequence_id"], p["frame_index"])))}
    by_seq = defaultdict(list)
    for p in pairs:
        by_seq[p["sequence_id"]].append(p)
    pen = {s: sum(ur(p) for p in g) / len(g) for s, g in by_seq.items() if len(g) >= 100}
    best, worst = min(pen, key=lambda s: (pen[s], s)), max(pen, key=lambda s: (pen[s], s))
    g = sorted(by_seq[best], key=lambda p: (p["candidates"]["g1"]["upper_direction_residual_degrees"], p["frame_index"]))
    strips[f"g1_stable_sequence_{best.split(':')[1]}"] = window(g[len(g) // 2])
    strips[f"g1_unstable_sequence_{worst.split(':')[1]}"] = window(
        max(by_seq[worst], key=lambda p: (p["candidates"]["g1"]["upper_direction_residual_degrees"], p["frame_index"])))

    fk_mid, fk_end = chain_forward_kinematics(rest, target, (0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0))

    def case(label, r, strip=None, index=None):
        gm = r["geometry"]
        return {"sequence_id": r["sequence_id"], "frame_index": r["frame_index"], "bank_position": r["position"],
                "root": list(target.root_head), "fk_mid": list(fk_mid), "fk_end": list(fk_end), "e_gt": r["e_gt"],
                "oracle_mid": r["reference"]["mid"],
                "draw": {tag: {"mid": gm[n]["mid"], "end": gm[n]["end"], "color": col} for tag, (n, col) in DRAW.items()},
                "caption": _caption(label, r, pair_by.get(key(r))),
                "numbers": {n: {m: r["candidates"][n].get(m) for m in ("wrist_error", "elbow_error", "upper_direction_error_degrees",
                                                                      "forearm_direction_error_degrees", "side_error_degrees")} for n in SHOW},
                "rig_rotation_armature_space_xyzw": list(object_delta(alignment, r["yaw_h0_radians"])),
                "pose_bone_local_xyzw": {}, "strip": strip, "strip_index": index}
    out = {"schema": "animcv_framepose_arm_geometry_owner_review_cases_v1", "rest_snapshot_digest": rest.digest(),
           "target_chain": target.to_dict(), "canonical_heading_zero_to_armature_xyzw": list(alignment),
           "sanity_checks_cm": {}, "refused": {}, "cases": {}}
    for label, r in stills.items():
        out["cases"][label] = case(label, r)
    for name, frames in strips.items():
        for i, r in enumerate(frames):
            out["cases"][f"{name}__{i}_f{r['frame_index']}"] = case(f"{name} [{i + 1}/{len(frames)}]", r, name, i)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(json.dumps({"stills": {k: key(v) for k, v in stills.items()}, "strips": {k: [key(r) for r in v] for k, v in strips.items()},
                      "stable": [best, pen[best]], "unstable": [worst, pen[worst]], "conflict_rows": len(conflict)}, indent=1, default=str))


if __name__ == "__main__":
    main()
