#!/usr/bin/env python3
"""EVALUATION ONLY (Worklog 82): Blender review cases for the anchored visual correction (R0 / R1).

Drawn: J0 orange, J1 magenta, R0 blue, R1 cyan, GT reference green, E_GT yellow (B0/B1 numbers in the
caption).  Fixed rules on the Worklog 82 test rows/pairs:
  * owner frames downtown_bar_00 a0 #514, #775; downtown_sitOnStairs_00 a0 #325; flat_packBags_00 a0 #446;
    downtown_runForBus_01 a0 #185;
  * largest R1 static improvement / regression vs J0 (elbow + wrist error change);
  * largest R1 temporal improvement / regression vs J1 (upper-arm motion residual; adjacent-frame strips);
  * strongly out-of-plane example: |f_GT| >= 0.5, largest R1-J0 wrist gain among unchosen rows;
  * near-plane failure: |f_GT| < sin 10 deg with R1 wrist worse than J0, median of that set;
  * moderate-bend failure: GT bend in [15, 45) with R1 elbow worse than B1, median of that set;
  * near-folded example: GT bend >= 90, median R1 elbow error.
Render with the unchanged Worklog 77 renderer (authoritative rest, Worklog 76-81 camera contract).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_framepose_visual_full_arm_cases import object_delta  # noqa: E402

from retarget.framepose_fk import derive_rig_alignment, load_rig_fk_calibration  # noqa: E402
from retarget.framepose_target_rest import load_target_rest_pose  # noqa: E402
from retarget.framepose_two_bone_ik import chain_forward_kinematics, load_ik_chain_set, target_chain_geometry  # noqa: E402

DRAW = {"J0": ("j0", (1.0, 0.55, 0.1, 1.0)), "J1": ("j1", (0.9, 0.15, 0.85, 1.0)),
        "R0": ("r0", (0.15, 0.45, 1.0, 1.0)), "R1": ("r1", (0.1, 0.85, 0.95, 1.0))}
SHOW = ("b1", "j0", "j1", "r0", "r1")
OWNERS = (("3dpw:downtown_bar_00:actor0", 514), ("3dpw:downtown_bar_00:actor0", 775),
          ("3dpw:downtown_sitOnStairs_00:actor0", 325), ("3dpw:flat_packBags_00:actor0", 446),
          ("3dpw:downtown_runForBus_01:actor0", 185))


def _caption(label, r, pair=None):
    c, k = r["candidates"], r["correction"]
    lines = [f"W82 anchored visual correction (3DPW test) | {label}",
             f"{r['sequence_id']}  frame #{r['frame_index']}  bank {r['position']}   GT bend {r['gt_bend_degrees']:.0f} deg   GT f {r['f_gt']:+.2f}",
             f"sample {r['sample_id']}",
             "orange J0 (frozen 2D anchor)   magenta J1 (direct 2D+Qwen)   blue R0 (2D correction)   cyan R1 (Qwen correction)",
             "green = GT ref   yellow = E_GT",
             "elbow cm: " + "  ".join(f"{n.upper()} {c[n]['elbow_error']:.1f}" for n in SHOW),
             "wrist cm: " + "  ".join(f"{n.upper()} {c[n]['wrist_error']:.1f}" for n in SHOW),
             "forearm deg: " + "  ".join(f"{n.upper()} {c[n]['forearm_direction_error_degrees']:.0f}" for n in SHOW),
             "correction: " + "  ".join(f"{g.upper()} upper {k[g]['upper_rotation_degrees']:.0f} deg / forearm {k[g]['forearm_rotation_degrees']:.0f} deg"
                                        for g in ("r0", "r1"))]
    if pair is not None:
        p = pair["candidates"]
        lines.append(f"next pair #{pair['frame_index']}->#{pair['next_frame_index']} upper residual deg: "
                     + "  ".join(f"{n.upper()} {p[n]['upper_direction_residual_degrees']:.1f}" for n in SHOW))
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
    by_key = {(r["sequence_id"], r["frame_index"]): r for r in rows}
    by_order = {(r["sequence_id"], r["bank_order"]): r for r in rows}
    pair_by = {(p["sequence_id"], p["frame_index"]): p for p in pairs}
    key = lambda r: (r["sequence_id"], r["frame_index"])  # noqa: E731
    e = lambda r, n, m: r["candidates"][n][m]  # noqa: E731
    ds = lambda r: (e(r, "r1", "elbow_error") + e(r, "r1", "wrist_error")) - (e(r, "j0", "elbow_error") + e(r, "j0", "wrist_error"))  # noqa: E731

    stills = {f"owner_{s.split(':')[1]}_{f}": by_key[(s, f)] for s, f in OWNERS}
    stills["largest_r1_static_improvement"] = min(rows, key=lambda r: (ds(r), key(r)))
    stills["largest_r1_static_regression"] = max(rows, key=lambda r: (ds(r), key(r)))
    chosen = {key(r) for r in stills.values()}
    stills["strongly_out_of_plane"] = min((r for r in rows if abs(r["f_gt"]) >= 0.5 and key(r) not in chosen),
                                          key=lambda r: (e(r, "r1", "wrist_error") - e(r, "j0", "wrist_error"), key(r)))

    def median_of(group, fn):
        g = sorted(group, key=lambda r: (fn(r), key(r)))
        return g[len(g) // 2]
    stills["near_plane_failure"] = median_of([r for r in rows if abs(r["f_gt"]) < math.sin(math.radians(10))
                                              and e(r, "r1", "wrist_error") > e(r, "j0", "wrist_error")],
                                             lambda r: e(r, "r1", "wrist_error") - e(r, "j0", "wrist_error"))
    stills["moderate_bend_failure"] = median_of([r for r in rows if 15 <= r["gt_bend_degrees"] < 45
                                                 and e(r, "r1", "elbow_error") > e(r, "b1", "elbow_error")],
                                                lambda r: e(r, "r1", "elbow_error") - e(r, "b1", "elbow_error"))
    stills["near_folded"] = median_of([r for r in rows if r["gt_bend_degrees"] >= 90], lambda r: e(r, "r1", "elbow_error"))

    def window(p):
        a = by_key[(p["sequence_id"], p["frame_index"])]
        return [f for f in (by_order.get((a["sequence_id"], a["bank_order"] + k)) for k in (-1, 0, 1, 2)) if f is not None]
    ur = lambda p: p["candidates"]["r1"]["upper_direction_residual_degrees"] - p["candidates"]["j1"]["upper_direction_residual_degrees"]  # noqa: E731
    strips = {"largest_r1_temporal_improvement": window(min(pairs, key=lambda p: (ur(p), p["sequence_id"], p["frame_index"]))),
              "largest_r1_temporal_regression": window(max(pairs, key=lambda p: (ur(p), p["sequence_id"], p["frame_index"])))}
    fk_mid, fk_end = chain_forward_kinematics(rest, target, (0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0))

    def case(label, r, strip=None, index=None):
        gm = r["geometry"]
        return {"sequence_id": r["sequence_id"], "frame_index": r["frame_index"], "sample_id": r["sample_id"], "bank_position": r["position"],
                "root": list(target.root_head), "fk_mid": list(fk_mid), "fk_end": list(fk_end), "e_gt": r["e_gt"], "oracle_mid": r["reference"]["mid"],
                "draw": {tag: {"mid": gm[n]["mid"], "end": gm[n]["end"], "color": col} for tag, (n, col) in DRAW.items()},
                "caption": _caption(label, r, pair_by.get(key(r))),
                "numbers": {n: {m: r["candidates"][n].get(m) for m in ("wrist_error", "elbow_error", "forearm_direction_error_degrees",
                                                                      "chain_direction_error_degrees", "reach_error", "side_error_degrees")}
                            for n in ("b0", *SHOW)} | {"correction": r["correction"]},
                "rig_rotation_armature_space_xyzw": list(object_delta(alignment, r["yaw_h0_radians"])),
                "pose_bone_local_xyzw": {}, "strip": strip, "strip_index": index}
    out = {"schema": "animcv_framepose_anchored_arm_review_cases_v1", "rest_snapshot_digest": rest.digest(),
           "target_chain": target.to_dict(), "canonical_heading_zero_to_armature_xyzw": list(alignment),
           "sanity_checks_cm": {}, "refused": {}, "cases": {}}
    for label, r in stills.items():
        out["cases"][label] = case(label, r)
    for name, frames in strips.items():
        for i, r in enumerate(frames):
            out["cases"][f"{name}__{i}_f{r['frame_index']}"] = case(f"{name} [{i + 1}/{len(frames)}]", r, name, i)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(json.dumps({"stills": {k: key(v) for k, v in stills.items()}, "strips": {k: [key(r) for r in v] for k, v in strips.items()}}, indent=1, default=str))


if __name__ == "__main__":
    main()
