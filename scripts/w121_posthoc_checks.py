#!/usr/bin/env python3
"""W121 POST-HOC, READ-ONLY checks written AFTER the frozen decision (no CASE authority; the official W121 CASE is the frozen
decision.json).  It quantifies the two instrument-specification errors that made V10 and V12 fail, and states the mechanical
CASE the frozen rule would output if exactly those two defects were set aside.  Nothing is rebuilt, re-evaluated or re-decided.

  1. V10: build records compared with the non-deterministic factor-info timing field `wall_s` included.
  2. V12 / DG5: the "REJECTed displayed as ACCEPTED" display corruption was declared on an ACCEPTed variant (no-op).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import w121_independent_transfer_base as AB  # noqa: E402

R = Path("C:/Users/dna10/animcv-w121/results")


def strip(x):
    if isinstance(x, dict):
        return {k: strip(v) for k, v in x.items() if k != "wall_s"}
    if isinstance(x, list):
        return [strip(v) for v in x]
    return x


def main():
    import w118_transition_edit_base as XB
    import w119_endpoint_transition as W9
    import w116_semantic_pin_contract as C
    d = json.loads((R / "decision/decision.json").read_text())
    b1, b2 = (json.loads((R / f"build_{t}/build_report.json").read_text()) for t in ("r1", "r2"))
    A = json.loads((R / "assess/assess_report.json").read_text())
    out = {"label": "POST-HOC, read-only, no CASE authority", "official_case": d["case"]}
    diff_keys = set()

    def walk(x, y):
        if isinstance(x, dict):
            for k in x:
                if k not in y or not isinstance(y, dict):
                    diff_keys.add(k)
                    continue
                if isinstance(x[k], (dict, list)):
                    walk(x[k], y[k])
                elif x[k] != y[k]:
                    diff_keys.add(k)
        elif isinstance(x, list):
            for u, v in zip(x, y):
                walk(u, v)
    for sid in b1["sources"]:
        walk(b1["sources"][sid]["records"], b2["sources"][sid]["records"])
    out["V10"] = {"differing_record_keys": sorted(diff_keys),
                  "records_equal_without_wall_s": {sid: strip(b1["sources"][sid]["records"]) == strip(b2["sources"][sid]["records"])
                                                   for sid in b1["sources"]},
                  "states_and_evals_byte_identical": d["detail"]["V10"]}
    dg5 = {}
    for rid in ("S1_REACH", "S1_HIGH_RBpos"):
        y = A["requests"][rid]["variants"]["G1F0"]
        s = W9.corrupt_display("DY5", y["display_spec_result"])
        ev = json.loads((R / "eval_e1/S1_eval.json").read_text())
        rec = b1["sources"]["S1"]["records"][rid]
        name = rec["factors"]["G1F0"]["state"]
        chk = XB.check_display(s, C.ev_frames(ev, name), C.ev_frames(ev, rec["U"]), ev["states"]["v0"] and
                               [fr["frame_index"] for fr in ev["states"]["v0"]["frames"]], rec["pin_frames"],
                               ev["states"][name]["state_id"], decision=y["commit"]["decision"])
        dg5[rid] = {"decision": y["commit"]["decision"], "corrupted_label": s["status"], "truth_check_fails": not chk["ok"], "errors": chk["errors"]}
    out["V12_DG5_on_rejected_variant"] = dg5
    V = {k: v for k, v in d["gates"].items() if k.startswith("V")}
    Cg = {k: v for k, v in d["gates"].items() if k.startswith("C")}
    V_alt = dict(V, V10_repeatability=all(out["V10"]["records_equal_without_wall_s"].values()),
                 V12_negative_controls=all(x["truth_check_fails"] for x in dg5.values())
                 and all(x["ok"] for x in d["detail"]["V12"]["negatives"].values()))
    mech = {M: dict(m, primary_improved=[tuple(x) for x in m["primary_improved"]], conditional_improved=[tuple(x) for x in m["conditional_improved"]])
            for M, m in d["mechanisms"].items()}
    out["mechanical_alternative"] = {"V_set_aside": V_alt, "result": AB.case_rule(V_alt, Cg, mech, len(d["population"]["sources"]), False),
                                     "assigned": False}
    (R / "posthoc").mkdir(exist_ok=True)
    (R / "posthoc/posthoc_checks.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str)[:4000])


if __name__ == "__main__":
    main()
