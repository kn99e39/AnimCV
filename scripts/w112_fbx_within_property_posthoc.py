#!/usr/bin/env python3
"""W112 POST-HOC characterisation (written after the frozen analysis; NOT a gate, does not change the CASE).

  repeat     the frozen repeat compared report lists whose order follows exporter UIDs (``obstructed_curves`` and the
             candidate's ``group_index``); re-compare both runs with those lists sorted by (model, property, channel)
             and the plan index dropped
  L family   the frozen point set includes the ``base`` bone tail, which the armature-only round trip relocates (no bind
             pose: the importer rebuilds rest geometry from the exported posed Lcl values); recompute the L-family
             diagnostics with ONLY that point removed, and show the removed error is constant in time and identical
             across conditions
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT/"scripts"))
import w112_fbx_within_property_contract as C  # noqa: E402
import w112_fbx_within_property as H  # noqa: E402
from w108_fbx_subframe import wjson  # noqa: E402

EXCLUDED = ("base", "tail")


def _canonical_lists(res):
    r = copy.deepcopy(res)
    for v in r["cases"].values():
        if v.get("obstructed_curves"):
            v["obstructed_curves"] = sorted(v["obstructed_curves"], key=lambda x: (x["model"], x["property"],
                                                                                   x["channel"]))
        for p in (v.get("candidate") or {}).get("perkey", []):
            p.pop("group_index", None)
    return r


def _err_excl(src, imp):
    """Per-time max |coordinate| error over all heads and tails except the base tail."""
    common, ia, ib = H.C8.common_index(src, imp)
    dh = np.abs(src["world_head"][:, ia] - imp["world_head"][:, ib]).max(-1)
    dt = np.abs(src["world_tail"][:, ia] - imp["world_tail"][:, ib]).max(-1)
    dt[:, common.index(EXCLUDED[0])] = 0.0
    return np.maximum(dh, dt).max(1), np.abs(src["world_tail"][:, ia[common.index("base")]] -
                                             imp["world_tail"][:, ib[common.index("base")]]).max(-1)


def _bound(src, by, labels, classes, case, tc):
    iv = C.interval_index(src["times"])
    err = {lab: _err_excl(src, by[lab])[0] for lab in labels}
    ints = [i for i, f in enumerate(iv) if f is None]
    per = {}
    for f in sorted({f for f in iv if f is not None}):
        idx = [i for i, g in enumerate(iv) if g == f]
        vals = {lab: float(err[lab][idx].max()) for lab in labels}
        best = min(labels, key=lambda lab: (vals[lab], labels.index(lab)))
        per[f] = (best, vals)
    integ = max(float(err[lab][ints].max()) for lab in labels)
    n = len(labels[0])
    out = {"bound_m": max([v[1][v[0]] for v in per.values()] + [integ]), "integer_m": integ,
           "group_uniform_bound_m": max([min(v[1]["C"*n], v[1]["L"*n]) for v in per.values()] + [integ])}
    by_class = {}
    for f, (b, vals) in per.items():
        by_class.setdefault(classes[f], []).append(vals[b])
    out["by_class_max_m"] = {k: max(v) for k, v in by_class.items()}
    out["intervals_over_1mm"] = sum(1 for b, vals in per.values() if vals[b] > C.TOL["recovered_m"])
    if len(tc) == n:
        out["source_aligned_bound_m"] = max(
            [vals["".join("C" if C.in_force(c["frames"], c["interpolation"], f) == C.CONSTANT else "L" for c in tc)]
             for f, (b, vals) in per.items()] + [integ])
    best = {}
    for b, _ in per.values():
        best[b] = best.get(b, 0) + 1
    out["best_label_counts"] = best
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out-dir", required=True)
    o = ap.parse_args()
    R = Path(o.results)
    out = Path(o.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    a = json.loads((R/"analysis/analysis_report.json").read_text())
    decl = json.loads((R/"declare/declaration.json").read_text())
    names = list(a["runs"])
    diffs = H.compare_runs(_canonical_lists(a["runs"][names[0]]), _canonical_lists(a["runs"][names[1]]))
    rep = {"schema": C.SCHEMA + ":posthoc", "note": "post-hoc, not a gate; the frozen CASE is unchanged",
           "repeat": {"frozen_differences": a["repeat"]["n_result_differences"],
                      "differences_after_sorting_uid_ordered_lists": len(diffs), "remaining": diffs[:20]},
           "L_family_excluding_base_tail": {}}
    for run in names:
        rd = R/run
        prod = json.loads((rd/"produce/produce_report.json").read_text())
        oi = json.loads((rd/"encode/oracle_input.json").read_text())
        res = {}
        for cid in C.FAMILY_CASES["L"]:
            case = decl["cases"][cid]
            src = C.load_npz(rd/"produce"/cid/"source.npz")
            census = prod["cases"][cid]["perkey_census"]
            tc = sorted([c for c in census["curves"] if c["data_path"] == case["target"]["source_path"]],
                        key=lambda c: c["index"])
            classes = {f: C.interval_class(tc, f) for f in range(case["frame_start"], case["frame_end"])}
            r = {"conditions": {}}
            base_tail = {}
            for k in ("control_U", "control_K", "meta_A"):
                p = rd/"consume"/cid/f"{k}.npz"
                if p.exists():
                    e, bt = _err_excl(src, C.load_npz(p))
                    r["conditions"][k] = C.class_maxima(e, src["times"], classes)
                    base_tail[k] = {"min": float(bt.min()), "max": float(bt.max())}
            labs = oi["cases"][cid]["imp_labels"]
            imp = {lab: C.load_npz(rd/"oracle"/cid/f"imp_{lab}.npz") for lab in labs}
            r["imp"] = _bound(src, imp, labs, classes, case, tc)
            elabs = list(oi["cases"][cid]["efiles"])
            ef = {lab: C.load_npz(rd/"oracle"/cid/f"fbx_{lab}.npz") for lab in elabs}
            r["fbx"] = _bound(src, ef, elabs, classes, case, [])
            for lab in labs[:1] + elabs[:1]:
                _, bt = _err_excl(src, imp.get(lab, ef.get(lab)))
                base_tail[f"oracle:{lab}"] = {"min": float(bt.min()), "max": float(bt.max())}
            r["excluded_base_tail_error_m"] = base_tail
            res[cid] = r
        rep["L_family_excluding_base_tail"][run] = res
    s = wjson(out/"posthoc_report.json", rep)
    print(s, rep["repeat"]["differences_after_sorting_uid_ordered_lists"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
