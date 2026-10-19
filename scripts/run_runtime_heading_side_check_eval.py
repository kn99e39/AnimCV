#!/usr/bin/env python3
"""EVALUATION (Worklog 91A): runtime side-view facing check on top of the GT-free Worklog 91 heading.

Providers on identical frames and detector observations:
  B        S0 + unchanged ``current_root_orientation`` (Worklog 91 B)
  B_side   B + ``motion.runtime_heading_side_check`` (2D nose profile check; flips psi -> -psi on disagreement)
  C        Worklog 91 C (B's yaw on KNOWN rows only)
  C_side   C + the same check
Populations: 3DPW VALIDATION / TEST (frozen S0 predictions; GT = matched 3DPW joints, docs/55-57) and MPI-INF-3DHP TEST
(Worklog 91 part 2 frozen S0 predictions ``a78adeb0...``; GT = bank target_3d).  The check's thresholds were fixed from
geometry and only its sign convention was confirmed on 3DPW VALIDATION before TEST / 3DHP were evaluated.
GT is used for evaluation only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_root_motion_contact_h0_replay import _matched_oracle_sequence  # noqa: E402
from run_runtime_heading_3dhp import shim  # noqa: E402
from run_runtime_heading_eval import circ, evaluate  # noqa: E402

from framepose.bank import load_bank  # noqa: E402
from framepose.bedlam_supervision import ExternalBank  # noqa: E402
from motion.runtime_heading import runtime_heading  # noqa: E402
from motion.runtime_heading_side_check import nose_offset, side_view_check  # noqa: E402
from pose.framepose_bridge import assemble_h0, build_h0_lifted_sequence, sequence_frame_positions, sequence_ids_in_split  # noqa: E402
from pose.root_orientation_diagnostic import observe_yaw  # noqa: E402

S0_SHA = {"validation": "402d70a2bed79ae31919ff86905e19c8972108ea988d6cfa5deac514562132b9",
          "test": "4357c7bdb46d81e7712269b08c45f39452d7a46e697a0c26fdb70d65fe1f4996",
          "3dhp": "a78adeb0ed0391b26d4c378e0ea15ce7d1e1eaf71dfcd9a384bb6dd6e3c6e420"}


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rows_for_sequence(view, preds, sid, gt_yaws, arrays, sizes, times):
    pos = sequence_frame_positions(view, sid)
    heads = runtime_heading(build_h0_lifted_sequence(view, preds, sid))
    out = {k: [] for k in ("B", "B_side", "C", "C_side")}
    actions = Counter()
    for i, p in enumerate(pos):
        g = gt_yaws[i]
        if g is None:
            continue
        h = heads[i]
        off, conf = nose_offset(arrays["input_2d"][p], arrays["input_valid"][p], sizes[p])
        y_side, act = side_view_check(h.yaw_radians, off, conf)
        actions[act] += 1
        if act == "flipped":
            actions["flip_improved"] += int(abs(circ(y_side, g)) < abs(circ(h.yaw_radians, g)))
        base = {"sequence_id": sid, "frame": int(view.samples[p].frame_index), "t": times[p], "gt": g,
                "conf": float(min(arrays["input_2d"][p][j, 2] for j in (1, 4, 11, 14))), "reliability": None}
        known_c = h.status == "KNOWN"
        out["B"].append({**base, "known": h.yaw_radians is not None, "yaw": h.yaw_radians})
        out["B_side"].append({**base, "known": y_side is not None, "yaw": y_side})
        out["C"].append({**base, "known": known_c, "yaw": h.yaw_radians if known_c else None})
        out["C_side"].append({**base, "known": known_c, "yaw": y_side if known_c else None})
    return out, actions


def summarize(rows, actions):
    res = {k: evaluate(v) for k, v in rows.items()}
    return {"actions": dict(actions), "providers": res}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bank", "s0_dir", "raw_3dpw", "hp_bank", "hp_s0", "out"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    for split in ("validation", "test"):
        if _sha(args.s0_dir / f"prediction_{split}.npy") != S0_SHA[split]:
            raise SystemExit("STOP: S0 3DPW predictions changed")
    if _sha(args.hp_s0) != S0_SHA["3dhp"]:
        raise SystemExit("STOP: S0 3DHP predictions changed")
    report = {"schema": "worklog91a_side_view_check", "results": {}}
    bank = load_bank(args.bank)
    s0, _ = assemble_h0(bank, {s: args.s0_dir / f"prediction_{s}.npy" for s in ("validation", "test")})
    sizes = [s.image_size for s in bank.samples]
    times = [s.frame_index / 30.0 for s in bank.samples]
    for split in ("validation", "test"):
        rows, acts = {k: [] for k in ("B", "B_side", "C", "C_side")}, Counter()
        for sid in sequence_ids_in_split(bank, split):
            pos = sequence_frame_positions(bank, sid)
            gt = [o.yaw_radians for o in observe_yaw(_matched_oracle_sequence(sid, args.raw_3dpw, [bank.samples[p].frame_index for p in pos]))]
            r, a = rows_for_sequence(bank, s0, sid, gt, bank.arrays, sizes, times)
            for k in rows:
                rows[k].extend(r[k])
            acts.update(a)
        report["results"][f"3dpw_{split}"] = summarize(rows, acts)
    mx = ExternalBank.load(args.hp_bank)
    hp = np.load(args.hp_s0)
    rv = shim(mx, mx.arrays["input_2d"], mx.arrays["input_valid"])
    g2 = np.concatenate([np.zeros_like(mx.arrays["input_2d"][..., :2]), np.ones_like(mx.arrays["input_2d"][..., 2:])], axis=-1)
    gv = shim(mx, g2, mx.arrays["target_valid"])
    hsizes = [s.image_size for s in mx.samples]
    htimes = [s.timestamp for s in rv.samples]
    groups = {"3dhp_all": None, "3dhp_studio_TS1_TS4": ("TS1", "TS2", "TS3", "TS4"), "3dhp_outdoor_TS5_TS6": ("TS5", "TS6"), "3dhp_TS2": ("TS2",)}
    per_seq = {}
    for sid in sorted({s.sequence_id for s in mx.samples}):
        gt = [o.yaw_radians for o in observe_yaw(build_h0_lifted_sequence(gv, mx.arrays["target_3d"], sid))]
        per_seq[sid] = rows_for_sequence(rv, hp, sid, gt, mx.arrays, hsizes, htimes)
    for gname, keep in groups.items():
        rows, acts = {k: [] for k in ("B", "B_side", "C", "C_side")}, Counter()
        for sid, (r, a) in per_seq.items():
            if keep is None or sid.split(":")[1] in keep:
                for k in rows:
                    rows[k].extend(r[k])
                acts.update(a)
        report["results"][gname] = summarize(rows, acts)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str))
    for g, x in report["results"].items():
        print(g, x["actions"])
        for k, m in x["providers"].items():
            e, t = m["error"], m["temporal"]
            print(f"   {k:7s} cov {m['coverage']:.3f} mean {e['mean']:.2f} p95 {e['p95']:.1f} >90 {e['gt90']:.4f} >135 {e['front_back_gt135']:.4f} "
                  f"flips {t['sudden_flips']} wrong>=1s {t['wrong_runs_ge_1s']} longest {t['longest_wrong_run_s']:.1f}s")


if __name__ == "__main__":
    main()
