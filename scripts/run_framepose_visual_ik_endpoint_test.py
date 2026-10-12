#!/usr/bin/env python3
"""EVALUATION ONLY (Worklog 75): selection-clean 3DPW test endpoint attribution of visual chain directions.

T0 H0 direction, T1 E0 / T2 E1 visual direction (Worklog 73 chain f + FrameBank
detector plane), T3 GT f + the same plane (oracle), T4 GT 3D direction
(reference); every candidate keeps the frozen H0 reach and is scored with the
analytic Worklog 61/62 endpoint contract (``framepose_visual_ik_endpoint``).

Order of work (each gate stops the run on failure):
  1. frozen artifact identities (H0 materialization, Worklog 73/74 reports);
  2. Worklog 74 parity: the analytic metric on the exact 395 Worklog 74 rows must
     reproduce the persisted solved-IK endpoint errors of P0/P1/P2/P3/reference;
  3. only then the test split (and, separately labelled, all validation rows).
Row validity mirrors Worklog 61: shoulder/elbow/wrist ``observation_valid``
(detector input_valid AND finite H0, via the production ``build_h0_lifted_sequence``)
and a known current-frame yaw (production ``current_root_orientation``); GT
validity is 3DPW ``campose_valid``.  No model is trained or selected.
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
from run_framepose_ik_gt_attribution import _stats  # noqa: E402
from run_root_motion_contact_h0_replay import _matched_oracle_sequence  # noqa: E402

from framepose.arm_depth_probe import observed_plane  # noqa: E402
from framepose.bank import load_bank  # noqa: E402
from motion.animation_semantics_v2_bridge import current_root_orientation  # noqa: E402
from pose.framepose_bridge import assemble_h0, build_h0_lifted_sequence, sequence_ids_in_split  # noqa: E402
from retarget.framepose_target_rest import load_target_rest_pose  # noqa: E402
from retarget.framepose_two_bone_ik import load_ik_chain_set, target_chain_geometry  # noqa: E402
from retarget.framepose_visual_ik_endpoint import (  # noqa: E402
    CANDIDATES, angle_degrees, chain_geometry, endpoint_error, visual_direction,
)

W73_REPORT_SHA256 = "1c05d50a5e6577329f0573786c6927b07268612ec150fbd079a1aa2bc4b03934"
W73_ROWS_SHA256 = "4e17bd77399e4c016098f70592d0cfd68f508245ae60914869f756605d9897fc"
W74_REPORT_SHA256 = "8158e8b2381bf12ef23eb42698384903e70766ca8e05ad988a831910540c7795"
H0_SHA256 = {"validation": "88a007fc", "test": None}  # validation prefix pinned by Worklogs 52-74; test recorded
PARITY_TOLERANCE_CM = 1e-9      # the Worklog 61/74 solved-endpoint contract limit
PARITY_TOLERANCE_DEG = 1e-6
STABLE = math.sin(math.radians(10))
NAMES = ("left_shoulder", "left_elbow", "left_wrist")
W74_MAP = {"t0_h0": "p0_h0_ik", "t1_e0_visual": "p1_e0_visual_direction", "t2_e1_visual": "p2_e1_visual_direction",
           "t3_gt_f_oracle": "p3_gt_forward_fraction", "t4_gt_direction_reference": "r_gt_direction_h0_reach"}
PAIRS = (("t2_e1_visual", "t1_e0_visual"), ("t2_e1_visual", "t0_h0"), ("t1_e0_visual", "t0_h0"),
         ("t3_gt_f_oracle", "t2_e1_visual"), ("t4_gt_direction_reference", "t3_gt_f_oracle"))
BOOTSTRAP = {"replicates": 2000, "seed": 1337, "moving_block_lengths": (10, 25),
             "procedures": ("sequence_cluster", "moving_block_within_sequence")}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_row(target, h0_pts, gt_pts, f_e0, f_e1, plane):
    """All candidates for one row; returns None-valued entries when the plane is degenerate."""
    h0, gt = chain_geometry(*h0_pts), chain_geometry(*gt_pts)
    if h0 is None or gt is None:
        return None
    f_gt, f_h0 = gt.direction[1], h0.direction[1]
    dirs = {"t0_h0": h0.direction, "t4_gt_direction_reference": gt.direction}
    fs = {"t0_h0": f_h0, "t4_gt_direction_reference": f_gt}
    if plane is not None:
        for name, f in (("t1_e0_visual", f_e0), ("t2_e1_visual", f_e1), ("t3_gt_f_oracle", f_gt)):
            dirs[name] = visual_direction(f, plane)
            fs[name] = dirs[name][1]
    out = {}
    for name in CANDIDATES:
        if name not in dirs:
            out[name] = None
            continue
        solved, requested, clamp = endpoint_error(target, h0.reach_fraction, dirs[name], gt.reach_fraction, gt.direction)
        out[name] = {"endpoint_error": solved, "requested_endpoint_error": requested, "clamp": clamp,
                     "direction_error_degrees": angle_degrees(dirs[name], gt.direction),
                     "f": fs[name], "abs_f_error": abs(fs[name] - f_gt), "direction": list(dirs[name])}
    return {"h0_reach": h0.reach_fraction, "gt_reach": gt.reach_fraction, "f_gt": f_gt, "f_h0": f_h0,
            "abs_reach_error": abs(h0.reach_fraction - gt.reach_fraction), "gt_bend_degrees": gt.bend_angle_degrees,
            "h0_bend_degrees": h0.bend_angle_degrees, "candidates": out}


def plane_for(bank, position):
    planes, oks = observed_plane(bank.arrays["input_2d"][position][None], np.asarray([bank.samples[position].image_size], float))
    return (float(planes["chain"][0, 0]), float(planes["chain"][0, 1])) if bool(oks["chain"][0]) else None


def collect(bank, h0, w73, target, raw, split, sequences=None):
    """Rows of one split with Worklog 61 validity, exact identity join and exclusion accounting."""
    rows, accounting = [], {}
    for sid in (sequences or sequence_ids_in_split(bank, split)):
        lifted = build_h0_lifted_sequence(bank, h0, sid)
        yaw = current_root_orientation(lifted)
        positions = sorted((p for p, s in enumerate(bank.samples) if s.sequence_id == sid), key=lambda p: bank.samples[p].frame_index)
        oracle = _matched_oracle_sequence(sid, raw, [bank.samples[p].frame_index for p in positions])
        acc = Counter()
        for frame, orient, gt_frame, p in zip(lifted.frames, yaw, oracle.frames, positions):
            if gt_frame.frame_index != frame.frame_index or bank.samples[p].frame_index != frame.frame_index:
                raise SystemExit(f"{sid}: frame alignment broken")
            acc["bank_rows"] += 1
            key = (sid, frame.frame_index)
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
            plane = plane_for(bank, p)
            r = evaluate_row(target, [frame.points[n].position for n in NAMES], [gt_frame.points[n].position for n in NAMES],
                             pred["chain_f_e0"], pred["chain_f_e1"], plane)
            if r is None:
                acc["excluded:degenerate_chain_geometry"] += 1
                continue
            if abs(pred["chain_f_h0"] - r["f_h0"]) > 1e-6 or abs(pred["chain_f_gt"] - r["f_gt"]) > 1e-5:
                raise SystemExit(f"{key}: Worklog 73 chain f identity differs (H0 {pred['chain_f_h0']} vs {r['f_h0']})")
            if plane is None:
                acc["excluded:degenerate_observed_plane"] += 1
                continue
            conf = [float(bank.arrays["input_2d"][p][j, 2]) for j in (11, 12, 13)]
            acc["common_eligible"] += 1
            rows.append({"sequence_id": sid, "frame_index": frame.frame_index, "position": p, "split": split,
                         "sample_id": bank.samples[p].sample_id, "min_arm_confidence": min(conf), **r})
        accounting[sid] = dict(acc)
    return rows, accounting


def _paired(rows, a, b, metric="endpoint_error"):
    d = [r["candidates"][a][metric] - r["candidates"][b][metric] for r in rows]
    by_seq = defaultdict(list)
    for r, x in zip(rows, d):
        by_seq[r["sequence_id"]].append(x)
    seq_means = {s: sum(v) / len(v) for s, v in by_seq.items()}
    sums = {s: sum(v) for s, v in by_seq.items()}
    net = sum(sums.values())
    ranked = sorted(sums.items(), key=lambda kv: kv[1])
    return {**_stats(d), "improved_share": sum(x < 0 for x in d) / len(d) if d else None,
            "frame_weighted_mean": sum(d) / len(d) if d else None,
            "sequence_weighted_mean": sum(seq_means.values()) / len(seq_means) if seq_means else None,
            "sequences": len(seq_means), "sequences_improved": sum(v < 0 for v in seq_means.values()),
            "sequences_regressed": sum(v > 0 for v in seq_means.values()),
            "largest_single_sequence_contribution": {"sequence_id": ranked[0][0], "sum": ranked[0][1],
                                                     "share_of_net": ranked[0][1] / net if net < 0 else None} if ranked else None,
            "worst_regressions": [{"sequence_id": s, "sum": v, "mean": seq_means[s]} for s, v in ranked[::-1][:3] if v > 0]}


def _bootstrap(rows, a, b):
    """Predeclared sequence-aware 95% intervals for the frame- and sequence-weighted mean difference."""
    rng = random.Random(BOOTSTRAP["seed"])
    by_seq = defaultdict(list)
    for r in rows:
        by_seq[r["sequence_id"]].append(r["candidates"][a]["endpoint_error"] - r["candidates"][b]["endpoint_error"])
    seqs = sorted(by_seq)
    out = {}
    frame_means, seq_means = [], []
    for _ in range(BOOTSTRAP["replicates"]):
        pick = [rng.choice(seqs) for _ in seqs]
        allv = [x for s in pick for x in by_seq[s]]
        frame_means.append(sum(allv) / len(allv))
        seq_means.append(sum(sum(by_seq[s]) / len(by_seq[s]) for s in pick) / len(pick))
    q = lambda v: [sorted(v)[int(0.025 * len(v))], sorted(v)[int(0.975 * len(v)) - 1]]  # noqa: E731
    out["sequence_cluster"] = {"frame_weighted_ci95": q(frame_means), "sequence_weighted_ci95": q(seq_means)}
    for L in BOOTSTRAP["moving_block_lengths"]:
        means = []
        for _ in range(BOOTSTRAP["replicates"]):
            tot = []
            for s in seqs:
                d = by_seq[s]
                n = len(d)
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


def _block(rows, bootstrap=True):
    out = {"rows": len(rows), "sequences": len({r["sequence_id"] for r in rows})}
    out["candidates"] = {n: {m: _stats(r["candidates"][n][m] for r in rows)
                             for m in ("abs_f_error", "direction_error_degrees", "endpoint_error", "requested_endpoint_error")}
                         | {"clamps": dict(Counter(r["candidates"][n]["clamp"] for r in rows))} for n in CANDIDATES}
    out["paired"] = {f"{a}-{b}": _paired(rows, a, b) for a, b in PAIRS}
    out["paired_direction"] = {f"{a}-{b}": _stats(r["candidates"][a]["direction_error_degrees"]
                                                  - r["candidates"][b]["direction_error_degrees"] for r in rows)
                               for a, b in PAIRS[:3]}
    if bootstrap and rows:
        out["bootstrap"] = {f"{a}-{b}": _bootstrap(rows, a, b) for a, b in PAIRS[:3]}
    return out


def _strata(rows, key, bins):
    out = {}
    for label, (lo, hi) in bins.items():
        g = [r for r in rows if lo <= key(r) < hi]
        out[label] = {"n": len(g), **{f"{n}_endpoint_mean": (sum(r["candidates"][n]["endpoint_error"] for r in g) / len(g)
                                                             if g else None) for n in CANDIDATES}}
        if g:
            for a, b in PAIRS[:3]:
                out[label][f"{a}-{b}"] = {k: v for k, v in _paired(g, a, b).items()
                                          if k in ("mean", "improved_share", "sequences", "sequences_improved")}
    return out


def _quartiles(values):
    v = sorted(values)
    e = [v[int(len(v) * q / 4)] for q in (1, 2, 3)]
    b = [-math.inf, *e, math.inf]
    return {f"Q{i + 1} [{b[i]:.3g}, {b[i + 1]:.3g})": (b[i], b[i + 1]) for i in range(4)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("rest", "chains", "bank", "h0_dir", "threedpw_raw", "w73_dir", "w74_dir", "out_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()

    identities = {"w73_report": _sha(args.w73_dir / "report.json"), "w73_rows_heldout": _sha(args.w73_dir / "rows_heldout.json"),
                  "w74_report": _sha(args.w74_dir / "report.json"), "w74_rows": _sha(args.w74_dir / "rows.json"),
                  "h0_validation": _sha(args.h0_dir / "prediction_validation.npy"),
                  "h0_test": _sha(args.h0_dir / "prediction_test.npy")}
    for k, exp in (("w73_report", W73_REPORT_SHA256), ("w73_rows_heldout", W73_ROWS_SHA256), ("w74_report", W74_REPORT_SHA256)):
        if identities[k] != exp:
            raise SystemExit(f"{k} identity mismatch: {identities[k]}")
    if not identities["h0_validation"].startswith(H0_SHA256["validation"]):
        raise SystemExit("frozen H0 validation predictions changed")
    materialization = json.loads((args.h0_dir / "materialization_report.json").read_text())
    bank = load_bank(args.bank)
    if materialization["bank_content_digest"] != bank.content_digest():
        raise SystemExit("H0 materialization was built from another FrameBank")
    if materialization["sign_source"] != "oracle" or materialization["predictions"]["test"]["shape"] != [7076, 17, 3]:
        raise SystemExit("unexpected H0 materialization contract")
    identities["bank_content_digest"] = bank.content_digest()
    h0, h0_identity = assemble_h0(bank, {"validation": args.h0_dir / "prediction_validation.npy",
                                         "test": args.h0_dir / "prediction_test.npy"})
    target = target_chain_geometry(load_target_rest_pose(args.rest), load_ik_chain_set(args.chains).chains[0])
    w73 = {}
    for r in json.loads((args.w73_dir / "rows_heldout.json").read_text()):
        key = (r["sequence_id"], r["frame_index"])
        if key in w73:
            raise SystemExit(f"duplicate Worklog 73 row {key}")
        w73[key] = r

    # --- Gate 2: Worklog 74 parity on the exact 395 rows ---
    w74_rows = json.loads((args.w74_dir / "rows.json").read_text())
    w74_keys = {(r["sequence_id"], r["frame_index"]): r for r in w74_rows}
    w74_sequences = sorted({k[0] for k in w74_keys})
    val_rows, val_acc = collect(bank, h0, w73, target, args.threedpw_raw, "validation")
    parity_rows = [r for r in val_rows if (r["sequence_id"], r["frame_index"]) in w74_keys]
    parity = {"rows": len(parity_rows), "expected": len(w74_rows), "endpoint_max_abs_diff_cm": {}, "direction_max_abs_diff_deg": {}}
    for t, p in W74_MAP.items():
        de = [abs(r["candidates"][t]["endpoint_error"] - w74_keys[(r["sequence_id"], r["frame_index"])]["candidates"][p]["solved_endpoint_error"])
              for r in parity_rows]
        dd = [abs(r["candidates"][t]["direction_error_degrees"] - w74_keys[(r["sequence_id"], r["frame_index"])]["candidates"][p]["direction_error_degrees"])
              for r in parity_rows]
        parity["endpoint_max_abs_diff_cm"][t], parity["direction_max_abs_diff_deg"][t] = max(de), max(dd)
    if parity["rows"] != parity["expected"] or any(v > PARITY_TOLERANCE_CM for v in parity["endpoint_max_abs_diff_cm"].values()) \
            or any(v > PARITY_TOLERANCE_DEG for v in parity["direction_max_abs_diff_deg"].values()):
        raise SystemExit(f"STOP: Worklog 74 parity failed: {parity}")

    # --- Gate 3: test split ---
    test_rows, test_acc = collect(bank, h0, w73, target, args.threedpw_raw, "test")
    totals = Counter()
    for acc in test_acc.values():
        totals.update(acc)
    family = lambda r: r["sequence_id"].split(":")[1].split("_")[0]  # noqa: E731
    report = {
        "schema": "animcv_framepose_visual_ik_endpoint_test_v1", "identities": identities,
        "h0_identity": {"bank_content_digest": h0_identity.bank_content_digest, "materialization": materialization},
        "target_chain": target.to_dict(), "parity_tolerance": {"cm": PARITY_TOLERANCE_CM, "deg": PARITY_TOLERANCE_DEG},
        "worklog74_parity": parity, "bootstrap_contract": BOOTSTRAP,
        "test": {"accounting": test_acc, "accounting_totals": dict(totals),
                 "w73_test_rows": sum(1 for k, r in w73.items() if bank.samples[r["position"]].split == "test"),
                 "bank_test_rows": len(bank.indices("test")),
                 "pooled": _block(test_rows),
                 "by_family": {fam: _block([r for r in test_rows if family(r) == fam], bootstrap=False)
                               for fam in sorted({family(r) for r in test_rows})},
                 "by_sequence": {sid: {"n": len(g), **{n: sum(r["candidates"][n]["endpoint_error"] for r in g) / len(g)
                                                       for n in CANDIDATES},
                                       "h0_abs_f_error": sum(r["candidates"]["t0_h0"]["abs_f_error"] for r in g) / len(g),
                                       "e1_abs_f_error": sum(r["candidates"]["t2_e1_visual"]["abs_f_error"] for r in g) / len(g)}
                                 for sid in sorted({r["sequence_id"] for r in test_rows})
                                 for g in [[r for r in test_rows if r["sequence_id"] == sid]]},
                 "strata": {
                     "abs_gt_chain_f": _strata(test_rows, lambda r: abs(r["f_gt"]),
                                               {"near_plane": (0, STABLE), "intermediate": (STABLE, 0.5), "out_of_plane": (0.5, 1.01)}),
                     "abs_h0_reach_error": _strata(test_rows, lambda r: r["abs_reach_error"], _quartiles([r["abs_reach_error"] for r in test_rows])),
                     "gt_bend_degrees": _strata(test_rows, lambda r: r["gt_bend_degrees"],
                                                {"0-15": (0, 15), "15-45": (15, 45), "45-90": (45, 90), ">=90": (90, 181)}),
                     "h0_bend_degrees": _strata(test_rows, lambda r: r["h0_bend_degrees"],
                                                {"0-15": (0, 15), "15-45": (15, 45), "45-90": (45, 90), ">=90": (90, 181)}),
                     "min_arm_detector_confidence": _strata(test_rows, lambda r: r["min_arm_confidence"],
                                                            _quartiles([r["min_arm_confidence"] for r in test_rows]))}},
        "validation": {"note": "selection-contaminated: the split that chose E0/E1 and H0 epochs; reported separately",
                       "accounting": val_acc,
                       "worklog74_population": _block(parity_rows),
                       "all_eligible": _block(val_rows),
                       "worklog74_sequences_only": _block([r for r in val_rows if r["sequence_id"] in w74_sequences]),
                       "other_validation_sequences": _block([r for r in val_rows if r["sequence_id"] not in w74_sequences]),
                       "by_sequence": {sid: {"n": len(g), **{n: sum(r["candidates"][n]["endpoint_error"] for r in g) / len(g)
                                                             for n in CANDIDATES},
                                             "h0_abs_f_error": sum(r["candidates"]["t0_h0"]["abs_f_error"] for r in g) / len(g),
                                             "e1_abs_f_error": sum(r["candidates"]["t2_e1_visual"]["abs_f_error"] for r in g) / len(g)}
                                       for sid in sorted({r["sequence_id"] for r in val_rows})
                                       for g in [[r for r in val_rows if r["sequence_id"] == sid]]}},
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "rows_test.json").write_text(json.dumps(test_rows))
    (args.out_dir / "rows_validation.json").write_text(json.dumps(val_rows))
    path = args.out_dir / "report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    t = report["test"]["pooled"]
    print(json.dumps({"sha256": _sha(path), "parity": parity, "test_rows": len(test_rows), "test_totals": dict(totals),
                      "endpoint_mean": {n: t["candidates"][n]["endpoint_error"]["mean"] for n in CANDIDATES},
                      "paired_mean": {k: (v["mean"], v["improved_share"], v["sequence_weighted_mean"],
                                          v["sequences_improved"], v["sequences_regressed"]) for k, v in t["paired"].items()}},
                     indent=1, default=str))


if __name__ == "__main__":
    main()
