#!/usr/bin/env python3
"""EVALUATION ONLY (Worklog 77): Blender review cases for the arm-geometry ownership counterfactuals.

Fixed selection rules on the Worklog 76/77 common-valid test rows (C100 = E1 runtime arm,
C000 = H0 runtime arm):
  * near-folded (GT bend >= 90 deg) row where the wrist improves and the elbow regresses most
    (largest min(wrist gain, elbow loss), C100 vs C000);
  * largest C100-C000 elbow regression;
  * largest C100 candidate-specific upper-arm discontinuity (largest excess proximal change on
    a pair flagged by the Worklog 76 contract) - adjacent-frame strip;
  * E1-stable / E1-unstable sequences: lowest / highest mean C100-C000 upper-arm motion residual
    penalty among sequences with >= 100 pairs - strips around the median / worst C100 pair;
  * Worklog 76 owner frames (downtown_bar_00 a0 #514, #775; sitOnStairs_00 a0 #325;
    runForBus_01 a0 #185; flat_packBags_00 a0 #446).
Drawn arms: C000 blue, C100 magenta, C111 (E1 + GT reach + GT side) cyan, C_GT_F (GT f, H0
reach/side) orange, GT-anchored reference green.  Object rotation A*Rz(yaw_H0)*A^-1 as in the
Worklog 76 review (formula proven there against the persisted Worklog 60 rotations).
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

DRAW = {"C000": ("c000", (0.15, 0.45, 1.0, 1.0)), "C100": ("c100", (0.9, 0.15, 0.85, 1.0)),
        "C111": ("c111", (0.1, 0.85, 0.95, 1.0)), "CGTF": ("c_gtf_h0r_h0s", (1.0, 0.55, 0.1, 1.0))}
ORDER = ("c000", "c100", "c010", "c110", "c001", "c101", "c011", "c111", "c_gtf_h0r_h0s")
SHORT = {n: n.upper().replace("C_GTF_H0R_H0S", "CGTF") for n in ORDER}
OWNERS = (("3dpw:downtown_bar_00:actor0", 514), ("3dpw:downtown_bar_00:actor0", 775),
          ("3dpw:downtown_sitOnStairs_00:actor0", 325), ("3dpw:downtown_runForBus_01:actor0", 185),
          ("3dpw:flat_packBags_00:actor0", 446))


def _caption(label, r, pair=None):
    c = r["candidates"]
    lines = [f"W77 arm-geometry ownership (3DPW test) | {label}",
             f"{r['sequence_id']}  frame #{r['frame_index']}  bank {r['position']}   GT bend {r['gt_bend_degrees']:.0f} deg",
             (f"reach fraction H0 {r['h0_reach']:.2f} / GT {r['gt_reach']:.2f}   chain f: H0 {r['f_h0']:+.2f}  E1 {r['f_e1']:+.2f}"
              f"  GT {r['f_gt']:+.2f}"),
             "blue C000 = H0 dir + H0 reach/side   magenta C100 = E1 dir + H0 reach/side",
             "cyan C111 = E1 dir + GT reach/side   orange CGTF = GT f + H0 reach/side   green = GT ref   yellow = E_GT",
             "elbow cm: " + "  ".join(f"{SHORT[n]} {c[n]['elbow_error']:.1f}" for n in ORDER),
             "wrist cm: " + "  ".join(f"{SHORT[n]} {c[n]['wrist_error']:.1f}" for n in ORDER)]
    if pair is not None:
        p = pair["candidates"]
        lines.append(f"next pair #{pair['frame_index']}->#{pair['next_frame_index']} upper-arm motion residual (deg):")
        lines.append("  " + "  ".join(f"{SHORT[n]} {p[n]['upper_direction_residual_degrees']:.1f}" for n in ORDER))
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("rig", "rest", "calibration", "chains", "ownership_dir", "out"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    rest = load_target_rest_pose(args.rest)
    if rest.provenance["source_fbx_sha256"] != hashlib.sha256(args.rig.read_bytes()).hexdigest():
        raise ValueError("Blender rest snapshot differs from current FBX bytes")
    alignment = derive_rig_alignment(rest, load_rig_fk_calibration(args.calibration))
    target = target_chain_geometry(rest, load_ik_chain_set(args.chains).chains[0])
    rows = json.loads((args.ownership_dir / "rows_test.json").read_text())
    pairs = json.loads((args.ownership_dir / "pairs_test.json").read_text())
    by_key = {(r["sequence_id"], r["frame_index"]): r for r in rows}
    by_seq_order = {(r["sequence_id"], r["bank_order"]): r for r in rows}
    key = lambda r: (r["sequence_id"], r["frame_index"])  # noqa: E731
    e = lambda r, n, m: r["candidates"][n][m]  # noqa: E731
    dw = lambda r: e(r, "c100", "wrist_error") - e(r, "c000", "wrist_error")  # noqa: E731
    de = lambda r: e(r, "c100", "elbow_error") - e(r, "c000", "elbow_error")  # noqa: E731

    stills = {}
    folded = [r for r in rows if r["gt_bend_degrees"] >= 90 and dw(r) < 0 < de(r)]
    stills["near_folded_wrist_improves_elbow_regresses"] = max(folded, key=lambda r: (min(-dw(r), de(r)), key(r)))
    stills["largest_elbow_regression"] = max(rows, key=lambda r: (de(r), key(r)))
    for sid, fi in OWNERS:
        stills[f"w76_owner_{sid.split(':')[1]}_{fi}"] = by_key[(sid, fi)]

    def window(pair, n=4):
        a = by_key[(pair["sequence_id"], pair["frame_index"])]
        frames = [by_seq_order.get((a["sequence_id"], a["bank_order"] + k)) for k in (-1, 0, 1, 2)]
        frames = [f for f in frames if f is not None]
        return frames[:n]
    pair_by = {(p["sequence_id"], p["frame_index"]): p for p in pairs}
    strips = {}
    flagged = [p for p in pairs if p["candidates"]["c100"]["disc:proximal_rotation_change_degrees"]]
    worst = max(flagged, key=lambda p: (p["candidates"]["c100"]["excess_proximal_change"], p["sequence_id"], p["frame_index"]))
    strips["largest_upper_arm_discontinuity"] = window(worst)
    by_seq = defaultdict(list)
    for p in pairs:
        by_seq[p["sequence_id"]].append(p)
    pen = {s: sum(p["candidates"]["c100"]["upper_direction_residual_degrees"] - p["candidates"]["c000"]["upper_direction_residual_degrees"]
                  for p in g) / len(g) for s, g in by_seq.items() if len(g) >= 100}
    stable, unstable = min(pen, key=lambda s: (pen[s], s)), max(pen, key=lambda s: (pen[s], s))
    g = sorted(by_seq[stable], key=lambda p: (p["candidates"]["c100"]["upper_direction_residual_degrees"], p["frame_index"]))
    strips[f"e1_stable_sequence_{stable.split(':')[1]}"] = window(g[len(g) // 2])
    strips[f"e1_unstable_sequence_{unstable.split(':')[1]}"] = window(
        max(by_seq[unstable], key=lambda p: (p["candidates"]["c100"]["upper_direction_residual_degrees"], p["frame_index"])))

    fk_mid, fk_end = chain_forward_kinematics(rest, target, (0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0))

    def case(label, r, strip=None, index=None):
        g = r["geometry"]
        return {"sequence_id": r["sequence_id"], "frame_index": r["frame_index"], "bank_position": r["position"],
                "root": list(target.root_head), "fk_mid": list(fk_mid), "fk_end": list(fk_end),
                "e_gt": r["e_gt"], "oracle_mid": r["reference"]["mid"],
                "draw": {tag: {"mid": g[n]["mid"], "end": g[n]["end"], "color": col} for tag, (n, col) in DRAW.items()},
                "caption": _caption(label, r, pair_by.get(key(r))),
                "numbers": {SHORT[n]: {m: r["candidates"][n][m] for m in ("wrist_error", "elbow_error",
                                                                         "upper_direction_error_degrees", "forearm_direction_error_degrees")}
                            for n in ORDER},
                "rig_rotation_armature_space_xyzw": list(object_delta(alignment, r["yaw_h0_radians"])),
                "pose_bone_local_xyzw": {}, "strip": strip, "strip_index": index}
    out = {"schema": "animcv_framepose_visual_arm_ownership_review_cases_v1", "rest_snapshot_digest": rest.digest(),
           "target_chain": target.to_dict(), "canonical_heading_zero_to_armature_xyzw": list(alignment),
           "sanity_checks_cm": {}, "refused": {}, "cases": {}}
    for label, r in stills.items():
        out["cases"][label] = case(label, r)
    for name, frames in strips.items():
        for i, r in enumerate(frames):
            out["cases"][f"{name}__{i}_f{r['frame_index']}"] = case(f"{name} [{i + 1}/{len(frames)}]", r, name, i)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(json.dumps({"stills": {k: key(v) for k, v in stills.items()},
                      "strips": {k: [key(r) for r in v] for k, v in strips.items()},
                      "stable_penalty": pen[stable], "unstable_penalty": pen[unstable]}, indent=1, default=str))


if __name__ == "__main__":
    main()
