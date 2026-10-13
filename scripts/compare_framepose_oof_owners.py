#!/usr/bin/env python3
"""Worklog 79 evaluation-only comparisons using unchanged W78 metrics/bootstrap.

Reads completed historical and OOF W78-compatible evaluations. No inference,
training, smoothing, selector, or new production behavior.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from audit_framepose_oof_conditioning import sha, stats, write_json
from run_framepose_arm_geometry_owner_eval import STATIC, TEMPORAL, _cmp, _table
from run_framepose_visual_full_arm_ik import BOOTSTRAP

COMPARISONS = (("g0_oof", "g0_in"), ("g1_oof", "g1_in"), ("g1_oof", "g0_oof"),
               ("g0_oof", "b1"), ("g1_oof", "b1"), ("g0_oof", "b0"), ("g1_oof", "b0"))


def key(r):
    return r["sequence_id"], r["frame_index"], r.get("next_frame_index")


def merge(old, new):
    oldmap = {key(r): r for r in old}
    if len(oldmap) != len(old) or set(oldmap) != {key(r) for r in new}:
        raise ValueError("historical/OOF populations differ")
    merged = []
    for r in new:
        a = oldmap[key(r)]
        for field in ("reference", "f_e1", "f_gt", "yaw_h0_radians"):
            if field in r and r[field] != a[field]:
                raise ValueError(f"frozen reference/inference changed: {field}")
        for b in ("b0", "b1", "c111"):
            if r["candidates"][b] != a["candidates"][b]:
                raise ValueError("historical baseline or threshold changed")
        cand = dict(r["candidates"])
        cand.update({f"{g}_in": a["candidates"][g] for g in ("g0", "g1")})
        cand.update({f"{g}_oof": r["candidates"][g] for g in ("g0", "g1")})
        merged.append({**r, "candidates": cand})
    return merged


def comparison_table(items, metrics, existing):
    out = {}
    for a, b in COMPARISONS:
        normalized = f"{a.replace('_oof', '')}-{b.replace('_oof', '')}"
        cached = existing.get(normalized, {})
        # These contrasts already used identical ordered rows, thresholds and
        # bootstrap code in the unchanged W78 evaluation. Reuse exact results.
        out[f"{a}-{b}"] = {m: cached[m] if m in cached else _cmp(items, a, b, m) for m in metrics}
        print(f"compared {a}-{b}", flush=True)
    return out


def matching(accounting):
    """Describe whether OOF approaches the untouched inference distribution.

    MAE differences are source-specific; validation and selection-clean test
    remain separate. Raw per-sequence/regime accounting is in the linked JSON.
    """
    out = {}
    original, oof = accounting["original_scopes"], accounting["scopes"]
    for source in ("3dpw", "bedlam"):
        tr, cross = original[source + "_train"], oof[source + "_train"]
        for split in (("validation", "test") if source == "3dpw" else ("validation",)):
            held = original[source + "_" + split]
            ma = lambda v: v["absolute_error"]["mean"]
            sign = lambda v: v["sign_magnitude"]["e1"]["stable_sign_accuracy"]
            regime = {k: {"in_sample_mae": tr["regimes"][k]["absolute_error"]["mean"],
                          "oof_mae": cross["regimes"][k]["absolute_error"]["mean"],
                          "inference_mae": held["regimes"][k]["absolute_error"]["mean"]} for k in tr["regimes"]}
            out[source + "_" + split] = {
                "in_sample_mae": ma(tr), "oof_mae": ma(cross), "inference_mae": ma(held),
                "in_sample_mae_gap": abs(ma(tr) - ma(held)), "oof_mae_gap": abs(ma(cross) - ma(held)),
                "oof_to_inference_mae_ratio": ma(cross) / ma(held),
                "in_sample_sign_accuracy": sign(tr), "oof_sign_accuracy": sign(cross), "inference_sign_accuracy": sign(held),
                "in_sample_prediction_magnitude": tr["prediction_magnitude"],
                "oof_prediction_magnitude": cross["prediction_magnitude"],
                "inference_prediction_magnitude": held["prediction_magnitude"], "regimes": regime}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("historical_eval", "oof_eval", "crossfit_dir", "out_dir"):
        p.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    old = json.loads((args.historical_eval / "report.json").read_text())
    new = json.loads((args.oof_eval / "report.json").read_text())
    if not old["gate1_historical_parity"]["passed"] or not new["gate1_historical_parity"]["passed"]:
        raise SystemExit("STOP: historical parity failed")
    rows = merge(json.loads((args.historical_eval / "rows_test.json").read_text()),
                 json.loads((args.oof_eval / "rows_test.json").read_text()))
    pairs = merge(json.loads((args.historical_eval / "pairs_test.json").read_text()),
                  json.loads((args.oof_eval / "pairs_test.json").read_text()))
    if len(rows) != 6003 or len(pairs) != 5504:
        raise ValueError("test population changed")
    names = ("b0", "b1", "g0_in", "g1_in", "g0_oof", "g1_oof")
    static = comparison_table(rows, STATIC, new["static"]["comparisons"])
    temporal = comparison_table(pairs, TEMPORAL, new["temporal"]["comparisons"])
    per_seq = {}
    for sid in sorted({r["sequence_id"] for r in rows}):
        rr, pp = [r for r in rows if r["sequence_id"] == sid], [r for r in pairs if r["sequence_id"] == sid]
        per_seq[sid] = {"rows": len(rr), "pairs": len(pp), "static": _table(rr, names, STATIC),
                        "temporal": _table(pp, names, TEMPORAL)}
    accounting = json.loads((args.crossfit_dir / "oof_distribution.json").read_text())
    report = {"schema": "animcv_w79_oof_owner_comparison_v1", "regime": "benchmark_detector_observation",
              "rows": len(rows), "pairs": len(pairs), "frozen_geometry_and_inference_exact": True,
              "identities": {"historical_report": sha(args.historical_eval / "report.json"),
                             "oof_report": sha(args.oof_eval / "report.json"),
                             "fold_manifest": sha(args.crossfit_dir / "fold_manifest.json"),
                             "oof_predictions": sha(args.crossfit_dir / "oof_predictions.npz")},
              "static": {"candidates": _table(rows, names, STATIC), "comparisons": static},
              "temporal": {"candidates": _table(pairs, names, TEMPORAL), "comparisons": temporal},
              "per_sequence": per_seq, "distribution_matching": matching(accounting),
              "bootstrap_contract": BOOTSTRAP,
              "oof_attribution": new["static"]["attribution"],
              "historical_attribution": old["static"]["attribution"],
              "oof_regimes": new["regimes"], "historical_regimes": old["regimes"],
              "coverage": {"oof": new["gate2_learned_geometry"], "historical": old["gate2_learned_geometry"]},
              "validation_separate": {"oof": new["validation"], "historical": old["validation"]},
              "case_note": "W79 cases differ from W78 mechanical cases; classify only after distribution and owner accounting."}
    write_json(args.out_dir / "report.json", report)


if __name__ == "__main__":
    main()
