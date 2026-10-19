#!/usr/bin/env python3
"""EVALUATION (Worklog 91): GT-free runtime Root Orientation (body heading) providers on 3DPW.

Heading convention (unchanged AnimationSemantics v2 / docs/57): yaw = atan2(dy, dx) of the right-minus-left
shoulder (and hip) axis in AnimCV camera coordinates (X right, Y forward depth, Z up); a subject facing the camera
has yaw = 180 deg.  Providers (identical frames, detector observations and coordinate conventions):

  A_h0_oracle  frozen O_BILATERAL H0 (context_h0_v2) + ``current_root_orientation``.  Its network INPUT includes two
               torso depth signs computed from 3DPW GT (``prepare_context_h0.py --sign-source oracle``): an
               EVALUATION-ONLY reference, never a runtime candidate.
  B_s0_runtime frozen S0 neutral-sign H0 (sign_v1/S0: same graph and seed, every sign field UNKNOWN at training and
               inference, checkpoint selected on validation MPJPE) + the same unchanged ``current_root_orientation``.
               The simplest existing GT-free heading policy.
  F_fixed_180  constant yaw 180 deg (torso faces the camera; the Worklog 90 rig-replay control).
  C_s0_reliable  ``motion.runtime_heading`` (experimental): B's identical yaw, reported only when its reliability state
               is KNOWN (UNRELIABLE and UNKNOWN frames count as not covered).  Added after B's failure analysis on
               VALIDATION; its two thresholds are fixed in the module from physical meaning before the TEST replay.

GT heading (offline evaluation only): ``observe_yaw`` of the 3DPW GT camera-frame joints matched to the bank frame
indices (``_matched_oracle_sequence``), exactly as docs/55-57.  GT never selects a runtime branch.
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
from motion.animation_semantics_v2_bridge import current_root_orientation  # noqa: E402
from motion.runtime_heading import runtime_heading  # noqa: E402
from pose.framepose_bridge import assemble_h0, build_h0_lifted_sequence, sequence_frame_positions, sequence_ids_in_split  # noqa: E402
from pose.root_orientation_diagnostic import observe_yaw  # noqa: E402

H0_SHA = {"validation": "88a007fca42eee08833ec8ece453aa3160a26232738be01cd849cf5016524937",
          "test": "6df04c2031b2bde6a4f0af37fdae0a7f0228e9fc23401e15e1aa2b3bd35695a5"}
S0_SHA = {"validation": "402d70a2bed79ae31919ff86905e19c8972108ea988d6cfa5deac514562132b9",
          "test": "4357c7bdb46d81e7712269b08c45f39452d7a46e697a0c26fdb70d65fe1f4996"}
BANK_DIGEST = "75519e6394a764e3749ddaa30555b58b73a01db582ccee14f661374b9a0ed536"
W57_VALIDATION_MEAN_ERROR = (6.0, 6.2)  # docs/57: ~6.10 deg over 3,407 rows (both-valid 5.894)
FLIP_DEG, FLIP_GT_DEG = 90.0, 45.0
TURN_DEG, TURN_WINDOW_S = 45.0, 1.0
WRONG_DEG = 45.0
VIEW_HALF_WIDTH = 30.0


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def circ(a, b):
    """Signed circular difference a - b in degrees, in (-180, 180]."""
    return (math.degrees(a - b) + 180.0) % 360.0 - 180.0


def view_class(gt_yaw):
    """Evaluation-only GT view stratum from the GT heading (facing camera = 180)."""
    d = abs(circ(gt_yaw, math.pi))
    if d <= VIEW_HALF_WIDTH:
        return "front"
    if d >= 180 - VIEW_HALF_WIDTH:
        return "back"
    if abs(d - 90) <= VIEW_HALF_WIDTH:
        return "side"
    return "oblique"


def provider_lifted(bank, arrays, name):
    def run(sid):
        lifted = build_h0_lifted_sequence(bank, arrays[name], sid)
        return [(o.known, o.yaw_radians) for o in current_root_orientation(lifted)], lifted
    return run


def provider_reliable(bank, arrays, name):
    def run(sid):
        lifted = build_h0_lifted_sequence(bank, arrays[name], sid)
        return [(h.status == "KNOWN", h.yaw_radians if h.status == "KNOWN" else None, h.min_torso_confidence)
                for h in runtime_heading(lifted)], lifted
    return run


def evaluate(rows):
    """rows: dicts with sequence_id, frame, t, gt, pred(known, yaw), conf, weight; ordered per sequence."""
    known = [r for r in rows if r["known"]]
    err = np.array([abs(circ(r["yaw"], r["gt"])) for r in known])
    out = {"rows": len(rows), "known": len(known), "coverage": len(known) / max(len(rows), 1)}
    if not len(known):
        return out
    out["error"] = {"mean": float(err.mean()), "median": float(np.median(err)), "p90": float(np.percentile(err, 90)),
                    "p95": float(np.percentile(err, 95)), "gt45": float((err > 45).mean()), "gt90": float((err > 90).mean()),
                    "front_back_gt135": float((err > 135).mean())}
    seqs = np.array([r["sequence_id"] for r in known])
    out["error_sequence_cluster"] = _brief(seqs, err)
    views = defaultdict(list)
    for r, e in zip(known, err):
        views[view_class(r["gt"])].append(e)
    total_by_view = defaultdict(int)
    for r in rows:
        total_by_view[view_class(r["gt"])] += 1
    out["by_view"] = {v: {"rows": total_by_view[v], "coverage": len(views[v]) / total_by_view[v], "mean": float(np.mean(views[v])),
                          "p95": float(np.percentile(views[v], 95)), "gt90": float(np.mean(np.array(views[v]) > 90))}
                      for v in ("front", "oblique", "side", "back") if views[v]}
    # temporal: flips, true turns, wrong runs, lag (per sequence, consecutive rows)
    by = defaultdict(list)
    for r in rows:
        by[r["sequence_id"]].append(r)
    flips = pairs = 0
    turn_hit = turn_n = 0
    runs, wrong_frames_in_long = [], 0
    per_seq = {}
    lags = []
    for sid, rs in by.items():
        rs.sort(key=lambda r: r["t"])
        for a, b in zip(rs[:-1], rs[1:]):
            if a["known"] and b["known"]:
                pairs += 1
                if abs(circ(b["yaw"], a["yaw"])) > FLIP_DEG and abs(circ(b["gt"], a["gt"])) < FLIP_GT_DEG:
                    flips += 1
        # true turns over ~1 s windows (GT turns > 45 deg)
        for i, a in enumerate(rs):
            j = next((k for k in range(i + 1, len(rs)) if rs[k]["t"] - a["t"] >= TURN_WINDOW_S - 1e-6), None)
            if j is None:
                break
            b = rs[j]
            g = circ(b["gt"], a["gt"])
            if abs(g) >= TURN_DEG and a["known"] and b["known"]:
                turn_n += 1
                p = circ(b["yaw"], a["yaw"])
                turn_hit += int(np.sign(p) == np.sign(g) and abs(p) >= 0.5 * abs(g))
        # long incorrect runs (consecutive known rows with error > 45 deg)
        cur = []
        for r in rs + [None]:
            bad = r is not None and r["known"] and abs(circ(r["yaw"], r["gt"])) > WRONG_DEG
            if bad:
                cur.append(r)
            elif cur:
                runs.append(cur[-1]["t"] - cur[0]["t"] + (cur[1]["t"] - cur[0]["t"] if len(cur) > 1 else 0.0))
                if runs[-1] >= 1.0:
                    wrong_frames_in_long += len(cur)
                cur = []
        e_s = [abs(circ(r["yaw"], r["gt"])) for r in rs if r["known"]]
        per_seq[sid] = {"rows": len(rs), "known": len(e_s), "mean": float(np.mean(e_s)) if e_s else None,
                        "gt90": float(np.mean(np.array(e_s) > 90)) if e_s else None}
        # lag: integer row shift minimizing mean circular error (positive = prediction lags GT)
        cost = {}
        for s in range(-5, 6):
            vals = [abs(circ(rs[i + s]["yaw"], rs[i]["gt"])) for i in range(max(0, -s), len(rs) - max(0, s))
                    if rs[i + s]["known"]]
            if vals:
                cost[s] = np.mean(vals)
        if cost:
            lags.append(min(cost, key=cost.get))
    gaps = []
    for sid, rs in by.items():
        cur = []
        for r in rs + [None]:
            if r is not None and not r["known"]:
                cur.append(r)
            elif cur:
                gaps.append(len(cur))
                cur = []
    out["unknown_runs"] = {"runs": len(gaps), "longest_rows": max(gaps) if gaps else 0,
                           "runs_ge_6_rows": int(sum(g >= 6 for g in gaps))}
    out["temporal"] = {"known_pairs": pairs, "sudden_flip_rate": flips / max(pairs, 1), "sudden_flips": flips,
                       "true_turn_windows": turn_n, "true_turn_tracked_rate": turn_hit / max(turn_n, 1),
                       "wrong_runs": len(runs), "wrong_runs_ge_1s": int(sum(r >= 1.0 for r in runs)),
                       "longest_wrong_run_s": float(max(runs)) if runs else 0.0,
                       "share_known_rows_in_wrong_runs_ge_1s": wrong_frames_in_long / max(len(known), 1),
                       "best_lag_rows": {str(k): int(sum(1 for x in lags if x == k)) for k in sorted(set(lags))}}
    worst = sorted(((s, v) for s, v in per_seq.items() if v["mean"] is not None), key=lambda t: -t[1]["mean"])
    out["worst_sequences"] = [{"sequence_id": s, **v} for s, v in worst[:6]]
    # confidence / detector quality vs error
    conf = np.array([r["conf"] for r in known])
    q = np.quantile(conf, [0.25, 0.5, 0.75])
    bins = np.digitize(conf, q)
    out["error_by_torso_confidence_quartile"] = [{"quartile": int(b + 1), "mean": float(err[bins == b].mean()),
                                                  "gt90": float((err[bins == b] > 90).mean())} for b in range(4) if (bins == b).any()]
    if all(r.get("reliability") is not None for r in known):
        rel = np.array([r["reliability"] for r in known])
        qr = np.quantile(rel, [0.25, 0.5, 0.75])
        br = np.digitize(rel, qr)
        out["error_by_reliability_quartile"] = [{"quartile": int(b + 1), "mean": float(err[br == b].mean()),
                                                 "gt90": float((err[br == b] > 90).mean())} for b in range(4) if (br == b).any()]
    return out


def collect(bank, split, providers, raw_3dpw):
    rows = {name: [] for name in providers}
    for sid in sequence_ids_in_split(bank, split):
        positions = sequence_frame_positions(bank, sid)
        outputs = {name: fn(sid) for name, fn in providers.items()}
        frames = [bank.samples[p].frame_index for p in positions]
        oracle = _matched_oracle_sequence(sid, raw_3dpw, frames)
        gt = observe_yaw(oracle)
        for i, p in enumerate(positions):
            g = gt[i]
            if g.yaw_radians is None:
                continue
            ij = bank.arrays["input_2d"][p]
            conf = float(min(ij[j, 2] for j in (1, 4, 11, 14)))
            t = bank.samples[p].frame_index / 30.0
            for name in providers:
                o = outputs[name][0][i]
                rows[name].append({"sequence_id": sid, "frame": frames[i], "t": t, "gt": g.yaw_radians,
                                   "known": bool(o[0]), "yaw": o[1], "conf": conf,
                                   "reliability": o[2] if len(o) > 2 else None})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bank", "h0_dir", "s0_dir", "raw_3dpw", "out"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    bank = load_bank(args.bank)
    if bank.content_digest() != BANK_DIGEST:
        raise SystemExit("STOP: FrameBank differs")
    for split in ("validation", "test"):
        if _sha(args.h0_dir / f"prediction_{split}.npy") != H0_SHA[split] or _sha(args.s0_dir / f"prediction_{split}.npy") != S0_SHA[split]:
            raise SystemExit(f"STOP: frozen {split} predictions differ")
    h0, _ = assemble_h0(bank, {s: args.h0_dir / f"prediction_{s}.npy" for s in ("validation", "test")})
    s0, _ = assemble_h0(bank, {s: args.s0_dir / f"prediction_{s}.npy" for s in ("validation", "test")})
    arrays = {"A_h0_oracle": h0, "B_s0_runtime": s0}
    providers = {"A_h0_oracle": provider_lifted(bank, arrays, "A_h0_oracle"),
                 "B_s0_runtime": provider_lifted(bank, arrays, "B_s0_runtime"),
                 "C_s0_reliable": provider_reliable(bank, arrays, "B_s0_runtime"),
                 "F_fixed_180": lambda sid: ([(True, math.pi)] * len(sequence_frame_positions(bank, sid)), None)}
    report = {"schema": "worklog91_runtime_heading_eval", "providers": list(providers), "results": {}}
    for split in ("validation", "test"):
        rows = collect(bank, split, providers, args.raw_3dpw)
        report["results"][split] = {name: evaluate(r) for name, r in rows.items()}
        if split == "validation":
            m = report["results"][split]["A_h0_oracle"]["error"]["mean"]
            report["w57_parity_mean_error"] = m
            if not (W57_VALIDATION_MEAN_ERROR[0] <= m <= W57_VALIDATION_MEAN_ERROR[1]):
                raise SystemExit(f"STOP: docs/57 validation heading error not reproduced ({m})")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps({s: {n: {k: v for k, v in r.items() if k in ("coverage", "error")} for n, r in x.items()}
                      for s, x in report["results"].items()}, indent=1))


if __name__ == "__main__":
    main()
