#!/usr/bin/env python3
"""W111 post-hoc diagnostics (written after the frozen analysis; NOT a gate; does not change the CASE).

  1. Real-data discrimination: per case, the W111 canonical comparison of the reference variant's unmodified
     control.fbx with its W110-producer meta.fbx.  The only semantic difference between them is the CONSTANT flag on
     the producer's CONSTANT curves, so a discriminating canonicaliser must report exactly those curves.
  2. Same-cell byte localisation: per case and run, the byte positions where the control.fbx of s1_pA and of its
     independent same-cell repeat s1_pA_r differ, and whether all of them lie inside FBXHeaderExtension/
     CreationTimeStamp.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT/"scripts"))
import w108_fbx_subframe_contract as C8  # noqa: E402
import w109_fbx_constant_import_contract as C9  # noqa: E402
import w111_fbx_semantic_identity as S  # noqa: E402
import w111_fbx_semantic_repeatability_contract as C  # noqa: E402
from w108_fbx_subframe import wjson  # noqa: E402


def _timestamp_span(data):
    root, _ = C9.fbx_index(data)
    hdr = [e for e in root.elems if e.id == b"FBXHeaderExtension"][0]
    ts = [e for e in hdr.elems if e.id == b"CreationTimeStamp"][0]
    offs = [p["value_offset"] for c in ts.elems for p in c.props]
    return min(offs), max(offs) + 4


def posthoc(o):
    res = Path(o.results)
    out = Path(o.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    enc = json.loads((res/"run1/encode/encode_report.json").read_text())
    rep = {"schema": C.SCHEMA + ":posthoc", "note": __doc__.strip(), "control_vs_meta": {}, "same_cell_bytes": {}}
    for cid in C.CASES:
        d = res/"run1/encode"/C.REF_CELL/cid
        a, b = (S.canonicalize(C8.fbx_read(d/f"{t}.fbx")[0]) for t in ("control", "meta"))
        cmp = S.compare(a, b, limit=1000)
        inv = {u: k for k, u in a["uid_of"].items()}
        want = {inv[u] for u in enc["cells"][C.REF_CELL]["cases"][cid]["constant_uids"]}
        got = cmp["_sets"]["differ"]
        rep["control_vs_meta"][cid] = {
            "differences": cmp["differences"], "objects_differ": len(got), "producer_constant_curves": len(want),
            "differ_equals_producer_constant_set": got == want, "only_curves": all(k.startswith("AnimationCurve[")
                                                                               for k in got),
            "header_or_connection_differences": len(cmp["header_sections_differ"]) + cmp["n_connections_only_a"]
            + cmp["n_connections_only_b"]}
        for run in C.RUNS:
            x = (res/run/"encode/s1_pA"/cid/"control.fbx").read_bytes()
            y = (res/run/"encode/s1_pA_r"/cid/"control.fbx").read_bytes()
            r = {"same_length": len(x) == len(y)}
            if r["same_length"]:
                pos = np.nonzero(np.frombuffer(x, np.uint8) != np.frombuffer(y, np.uint8))[0]
                lo, hi = _timestamp_span(x)
                r.update({"bytes_differ": int(pos.size), "timestamp_span": [lo, hi],
                          "all_in_timestamp": bool(((pos >= lo) & (pos < hi)).all())})
            rep["same_cell_bytes"][f"{run}:{cid}"] = r
    s = wjson(out/"posthoc_report.json", rep)
    print(json.dumps({"sha256": s, **rep["control_vs_meta"]}, indent=1))
    print(json.dumps(rep["same_cell_bytes"]))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out-dir", required=True)
    return posthoc(ap.parse_args())


if __name__ == "__main__":
    sys.exit(main())
