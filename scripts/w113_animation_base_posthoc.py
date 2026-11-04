#!/usr/bin/env python3
"""W113 POST-HOC diagnostic, written AFTER the frozen instrument returned CASE D. It never assigns a CASE.

The frozen declaration (c3f7485c...) required detector confidence in [0, 1]. On authentic inputs the frozen build
stops at `validate_view(real)` with CONFIDENCE_AMBIGUOUS (identically in two builds). This script:

  frozen-failure  records that D decision and the confidence-domain evidence (counts, joints, frames, consumers
                  that assume a unit interval), using the frozen validator unchanged;
  build           re-runs the frozen build with ONE labelled override, V.CONFIDENCE_UPPER = inf, so the remaining
                  declared checks (controls, directives, coverage, existing paths, rig association) are preserved
                  as partial diagnostics;
  gates           reports the frozen decision inputs on that post-hoc build WITHOUT a CASE letter.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import w113_animation_base_contract as C  # noqa: E402
import w113_animation_base_view as V  # noqa: E402

LABEL = "POSTHOC_DIAGNOSTIC_NOT_A_VERDICT (confidence domain override after frozen CASE D)"


def frozen_code(view, expected):
    try:
        V.validate_view(view, expected=expected)
        return "PASS"
    except V.ContractError as e:
        return e.code


def cmd_frozen_failure(a):
    decl = C.load_declaration(a)
    out = C.new_dir(a.out_dir)
    if V.CONFIDENCE_UPPER != 1.0:
        raise RuntimeError("frozen validator altered")
    syn = C.synthetic_view(out)
    z, wrep, insp = C.real_inputs(decl)
    real, lifted, _ = C.real_view(decl, z, wrep, insp)
    rexp = C.expected_of(real, C.source(decl, "w102_observations_predictions", "w102_video_report", "w102_video_inspection",
                                        "repo_production_demo_clip", "s0_checkpoint"), frame_count=int(wrep["frames"]))
    bview, binfo = C.bench_view(decl)
    bexp = C.expected_of(bview, C.source(decl, "bench_bank_index", "bench_bank_arrays", "bench_frozen_prediction_validation"))
    a_ = real["arrays"]
    over = a_["obs2d_present"] & (np.nan_to_num(a_["obs2d_conf"]) > 1)
    du = np.array([[f.points[n].depth_uncertainty for n in V.JOINTS] for f in lifted.frames])
    bank = np.load(decl["inputs"]["bench_bank_arrays"]["path"])
    bc, bv = bank["input_2d"][..., 2], bank["input_valid"]
    bo = bview["arrays"]["obs2d_present"] & (np.nan_to_num(bview["arrays"]["obs2d_conf"]) > 1)
    rep = {
        "schema": "w113_frozen_decision_v1", "declaration_sha256": a.declaration_sha, "case": "D",
        "reason": "frozen positive control failed on authentic data: validate_view(real) -> CONFIDENCE_AMBIGUOUS "
                  "(declared confidence domain [0,1]; authentic detector scores exceed 1). Identical in build_r1 and "
                  "build_r2. No viability verdict.",
        "frozen_validation": {"synthetic": frozen_code(syn, C.expected_of(syn, {"synthetic_source": syn["meta"]["sources"]["synthetic_source"]})),
                              "real": frozen_code(real, rexp), "benchmark": frozen_code(bview, bexp)},
        "real_confidence_domain": {
            "present_cells": int(a_["obs2d_present"].sum()), "cells_above_1": int(over.sum()),
            "max": float(np.nanmax(a_["obs2d_conf"])), "min_present": float(np.nanmin(a_["obs2d_conf"])),
            "frames_with_any_above_1": int(over.any(1).sum()),
            "per_joint_above_1": {n: int(over[:, i].sum()) for i, n in enumerate(V.JOINTS) if over[:, i].any()},
            "valid_cells_above_1": int((over & a_["input_valid"]).sum()),
            "visibility_threshold_recorded": wrep["M1_identity"]["config"]["visibility_threshold"]},
        "benchmark_confidence_domain": {
            "bank_valid_cells": int(bv.sum()), "bank_valid_cells_above_1": int((bc[bv] > 1).sum()),
            "bank_max": float(bc[bv].max()), "sequence": binfo["sequence_id"], "sequence_cells_above_1": int(bo.sum())},
        "consumers_assuming_unit_interval": {
            "pose.framepose_bridge depth_uncertainty=1-confidence (real clip)": {
                "negative_values": int((du < 0).sum()), "min": float(du.min())},
            "retarget.quality min_mean_confidence=0.30 (CLI default)": "scale-dependent; semantics undeclared",
            "MotionGraph/AnimationClip confidence": "raw score passed through; >1 values appear (see post-hoc build)"},
    }
    C.dump(out / "decision_frozen.json", rep)
    print(json.dumps({"case": "D", "frozen_validation": rep["frozen_validation"],
                      "real_above_1": rep["real_confidence_domain"]["cells_above_1"]}))
    return 0


def cmd_build(a):
    V.CONFIDENCE_UPPER = math.inf
    a.tag = f"POSTHOC:{a.tag}"
    rc = C.cmd_build(a)
    p = Path(a.out_dir) / "build_report.json"
    r = json.loads(p.read_text())
    r["posthoc_label"] = LABEL
    C.dump(p, r)
    (Path(a.out_dir) / "POSTHOC_NOT_A_VERDICT.txt").write_text(LABEL + "\n")
    return rc


def cmd_gates(a):
    out = Path(a.out_dir)
    sub = out / "_decide_inputs"
    C.cmd_decide(argparse.Namespace(declaration=a.declaration, declaration_sha=a.declaration_sha, out_dir=str(sub),
                                    preflight=a.preflight, build_dir=a.build_dir, repeat=a.repeat))
    d = json.loads((sub / "decision.json").read_text())
    d.pop("case")
    d["label"] = LABEL
    d["note"] = "Frozen decision-input logic evaluated on the post-hoc build. No CASE is assigned; the W113 CASE is D."
    clip = json.loads((Path(a.build_dir) / "legacy_animation.json").read_text())
    confs = [s["confidence"] for t in clip["tracks"].values() for s in t["samples"]]
    d["legacy_clip_confidence_above_1"] = int(sum(c > 1 for c in confs))
    C.dump(out / "posthoc_gate_inputs.json", d)
    (sub / "decision.json").unlink()
    sub.rmdir()
    print(json.dumps({"A_unmet": d["A_unmet"], "B_hits": d["B_hits"], "invalid": d["invalid"]}))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("frozen-failure", "build", "gates"):
        p = sub.add_parser(name)
        p.add_argument("--declaration", required=True)
        p.add_argument("--declaration-sha", required=True)
        p.add_argument("--out-dir", required=True)
    sub.choices["build"].add_argument("--tag", required=True)
    sub.choices["build"].add_argument("--rig-audit", default=None)
    for k in ("preflight", "build_dir", "repeat"):
        sub.choices["gates"].add_argument("--" + k.replace("_", "-"), required=True)
    a = ap.parse_args(argv)
    return {"frozen-failure": cmd_frozen_failure, "build": cmd_build, "gates": cmd_gates}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
