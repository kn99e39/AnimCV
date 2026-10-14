#!/usr/bin/env python3
"""EVALUATION ONLY (Worklog 81): compact Blender review of the J0/J1 factorial attribution.

Drawn (compact, readable contrast set): J0 = V000 orange, J1 = V111 cyan, V100 (J1 direction with J0
reach/side) magenta, V011 (J0 direction with J1 reach/side) blue, GT reference green, E_GT yellow.
V100 isolates the direction swap and V011 the reach+side swap; all eight candidates' numbers are in the
caption and the JSON.  Mixed candidates are attribution tools, not runtime architectures.
Fixed selection rules (test rows/pairs of the Worklog 81 run):
  * owner frames downtown_bar_00 a0 #514, #775; downtown_sitOnStairs_00 a0 #325; flat_packBags_00 a0 #446;
    downtown_runForBus_01 a0 #185;
  * largest J1-J0 wrist improvement; largest J1-J0 elbow regression;
  * largest J1-J0 upper-arm motion-residual regression (adjacent-frame strip);
  * strong near-folded example: GT bend >= 90, largest J0-B1 elbow gain;
  * strongly out-of-plane example: |f_GT| >= 0.5, largest J1-J0 wrist gain among rows not already chosen;
  * representative moderate-bend failure: GT bend in [15, 45) with J1 elbow worse than B1, the median of that set.
Render with the unchanged Worklog 77 renderer (authoritative rest, Worklog 76-80 camera contract).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_framepose_visual_full_arm_cases import object_delta  # noqa: E402

from retarget.framepose_fk import derive_rig_alignment, load_rig_fk_calibration  # noqa: E402
from retarget.framepose_target_rest import load_target_rest_pose  # noqa: E402
from retarget.framepose_two_bone_ik import chain_forward_kinematics, load_ik_chain_set, target_chain_geometry  # noqa: E402

DRAW = {"J0": ("v000", (1.0, 0.55, 0.1, 1.0)), "J1": ("v111", (0.1, 0.85, 0.95, 1.0)),
        "V100": ("v100", (0.9, 0.15, 0.85, 1.0)), "V011": ("v011", (0.15, 0.45, 1.0, 1.0))}
ORDER = ("v000", "v100", "v010", "v001", "v110", "v101", "v011", "v111")
OWNERS = (("3dpw:downtown_bar_00:actor0", 514), ("3dpw:downtown_bar_00:actor0", 775),
          ("3dpw:downtown_sitOnStairs_00:actor0", 325), ("3dpw:flat_packBags_00:actor0", 446),
          ("3dpw:downtown_runForBus_01:actor0", 185))


def _caption(label, r, pair=None):
    c = r["candidates"]
    lines = [f"W81 J0/J1 factor attribution (3DPW test) | {label}",
             f"{r['sequence_id']}  frame #{r['frame_index']}  bank {r['position']}   GT bend {r['gt_bend_degrees']:.0f} deg   GT f {r['f_gt']:+.2f}",
             f"sample {r['sample_id']}",
             "orange J0 (V000)   cyan J1 (V111)   magenta V100 = J1 dir + J0 reach/side   blue V011 = J0 dir + J1 reach/side",
             "green = GT ref   yellow = E_GT   (Vdrs: d/r/s = 0 J0, 1 J1; mixed V are attribution tools only)",
             "elbow cm: " + " ".join(f"{n[1:]} {c[n]['elbow_error']:.1f}" for n in ORDER) + f"  B1 {c['b1']['elbow_error']:.1f}",
             "wrist cm: " + " ".join(f"{n[1:]} {c[n]['wrist_error']:.1f}" for n in ORDER) + f"  B1 {c['b1']['wrist_error']:.1f}",
             "forearm deg: " + " ".join(f"{n[1:]} {c[n]['forearm_direction_error_degrees']:.0f}" for n in ORDER)]
    if pair is not None:
        p = pair["candidates"]
        lines.append(f"next pair #{pair['frame_index']}->#{pair['next_frame_index']} upper residual deg: "
                     + " ".join(f"{n[1:]} {p[n]['upper_direction_residual_degrees']:.1f}" for n in ORDER))
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("rig", "rest", "calibration", "chains", "attribution_dir", "out"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    rest = load_target_rest_pose(args.rest)
    if rest.provenance["source_fbx_sha256"] != hashlib.sha256(args.rig.read_bytes()).hexdigest():
        raise ValueError("Blender rest snapshot differs from current FBX bytes")
    alignment = derive_rig_alignment(rest, load_rig_fk_calibration(args.calibration))
    target = target_chain_geometry(rest, load_ik_chain_set(args.chains).chains[0])
    rows = json.loads((args.attribution_dir / "rows_test.json").read_text())
    pairs = json.loads((args.attribution_dir / "pairs_test.json").read_text())
    by_key = {(r["sequence_id"], r["frame_index"]): r for r in rows}
    by_order = {(r["sequence_id"], r["bank_order"]): r for r in rows}
    pair_by = {(p["sequence_id"], p["frame_index"]): p for p in pairs}
    key = lambda r: (r["sequence_id"], r["frame_index"])  # noqa: E731
    e = lambda r, n, m: r["candidates"][n][m]  # noqa: E731

    stills = {f"owner_{s.split(':')[1]}_{f}": by_key[(s, f)] for s, f in OWNERS}
    stills["largest_j1_vs_j0_wrist_improvement"] = min(rows, key=lambda r: (e(r, "v111", "wrist_error") - e(r, "v000", "wrist_error"), key(r)))
    stills["largest_j1_vs_j0_elbow_regression"] = max(rows, key=lambda r: (e(r, "v111", "elbow_error") - e(r, "v000", "elbow_error"), key(r)))
    folded = [r for r in rows if r["gt_bend_degrees"] >= 90]
    stills["strong_near_folded"] = min(folded, key=lambda r: (e(r, "v000", "elbow_error") - e(r, "b1", "elbow_error"), key(r)))
    chosen = {key(r) for r in stills.values()}
    oop = [r for r in rows if abs(r["f_gt"]) >= 0.5 and key(r) not in chosen]
    stills["strongly_out_of_plane"] = min(oop, key=lambda r: (e(r, "v111", "wrist_error") - e(r, "v000", "wrist_error"), key(r)))
    moderate = sorted((r for r in rows if 15 <= r["gt_bend_degrees"] < 45 and e(r, "v111", "elbow_error") > e(r, "b1", "elbow_error")),
                      key=lambda r: (e(r, "v111", "elbow_error") - e(r, "b1", "elbow_error"), key(r)))
    stills["moderate_bend_failure"] = moderate[len(moderate) // 2]
    worst = max(pairs, key=lambda p: (p["candidates"]["v111"]["upper_direction_residual_degrees"]
                                      - p["candidates"]["v000"]["upper_direction_residual_degrees"], p["sequence_id"], p["frame_index"]))
    a = by_key[(worst["sequence_id"], worst["frame_index"])]
    strip = [f for f in (by_order.get((a["sequence_id"], a["bank_order"] + k)) for k in (-1, 0, 1, 2)) if f is not None]

    fk_mid, fk_end = chain_forward_kinematics(rest, target, (0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0))

    def case(label, r, strip_name=None, index=None):
        gm = r["geometry"]
        return {"sequence_id": r["sequence_id"], "frame_index": r["frame_index"], "sample_id": r["sample_id"], "bank_position": r["position"],
                "root": list(target.root_head), "fk_mid": list(fk_mid), "fk_end": list(fk_end), "e_gt": r["e_gt"], "oracle_mid": r["reference"]["mid"],
                "draw": {tag: {"mid": gm[n]["mid"], "end": gm[n]["end"], "color": col} for tag, (n, col) in DRAW.items()},
                "caption": _caption(label, r, pair_by.get(key(r))),
                "numbers": {n: {m: r["candidates"][n].get(m) for m in ("wrist_error", "elbow_error", "forearm_direction_error_degrees",
                                                                      "chain_direction_error_degrees", "reach_error", "side_error_degrees")}
                            for n in (*ORDER, "b1")},
                "rig_rotation_armature_space_xyzw": list(object_delta(alignment, r["yaw_h0_radians"])),
                "pose_bone_local_xyzw": {}, "strip": strip_name, "strip_index": index}
    out = {"schema": "animcv_framepose_joint_factor_review_cases_v1", "rest_snapshot_digest": rest.digest(),
           "target_chain": target.to_dict(), "canonical_heading_zero_to_armature_xyzw": list(alignment),
           "sanity_checks_cm": {}, "refused": {}, "cases": {}}
    for label, r in stills.items():
        out["cases"][label] = case(label, r)
    for i, r in enumerate(strip):
        out["cases"][f"largest_j1_vs_j0_temporal_regression__{i}_f{r['frame_index']}"] = case(
            f"largest_j1_vs_j0_temporal_regression [{i + 1}/{len(strip)}]", r, "largest_j1_vs_j0_temporal_regression", i)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(json.dumps({"stills": {k: key(v) for k, v in stills.items()}, "strip": [key(r) for r in strip]}, indent=1, default=str))


if __name__ == "__main__":
    main()
