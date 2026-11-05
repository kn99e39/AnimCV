#!/usr/bin/env python3
"""W115 POST-HOC diagnostic, written AFTER the frozen W115 decision (CASE D via V8) was inspected. It changes no gate,
perturbation, parameter or expected outcome, and assigns no CASE.

  c1-stored-mechanism  For the C1 float64 exact-direction diagnostic and the C1 float32 candidate, list every stored
                       (float32) channel-frame change with the component's value, its float64 pre-cast delta and the float32
                       spacing at that value, and account, over every L1-changed component, how many deltas exceed half the
                       local float32 spacing. Mechanism only: explains why V8's zero-stored-change prediction failed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import w114_rig_influence as R  # noqa: E402
import w114_source_preserving_base as B  # noqa: E402
import w114_source_preserving_contract as C  # noqa: E402
import w115_validity_base as W  # noqa: E402
import w115_validity_contract as K  # noqa: E402

LABEL = "POSTHOC_DIAGNOSTIC_NOT_A_VERDICT (written after the frozen W115 decision was inspected)"


def account(L1b, L1a, L2b, L2a, rows):
    changes, n_l1, n_above_half, n_stored = [], 0, 0, 0
    for c in W.CH:
        for r in rows:
            a, b = L1b[c][r], L1a[c][r]
            if np.isnan(a).any() or np.array_equal(a, b):
                continue
            for k in range(4):
                if a[k] == b[k]:
                    continue
                n_l1 += 1
                sp = float(np.spacing(np.float32(abs(a[k]))))
                n_above_half += int(abs(b[k] - a[k]) > sp / 2)
                if L2b[c][r][k] != L2a[c][r][k]:
                    n_stored += 1
                    changes.append({"channel": c, "row": int(r), "component_wxyz": int(k), "case_value_before": float(a[k]),
                                    "case_value_after": float(b[k]), "float64_delta": float(b[k] - a[k]),
                                    "float32_spacing_at_value": sp, "stored_before": float(L2b[c][r][k]),
                                    "stored_after": float(L2a[c][r][k])})
    return {"l1_changed_components": n_l1, "l1_delta_above_half_local_float32_spacing": n_above_half,
            "stored_changed_components": n_stored, "stored_changes": changes}


def cmd_c1(a):
    decl = K.load_declaration(a)
    out = C.new_dir(a.out_dir)
    rq = decl["blender_rest_rotation_wxyz"]
    z, wrep, insp = C.real_inputs(decl)
    ev, _ = C.real_evidence(decl, z, wrep, insp)
    rc = R.rig_context()
    fi, ts = ev["arrays"]["frame_index"], ev["arrays"]["timestamp"]
    rows = np.flatnonzero(fi % 2 == 0)
    base = np.array(ev["arrays"]["est3d_xyz_hypothesis"])
    u = base[:, B.JI["thorax"]].astype(np.float64) - base[:, B.JI["pelvis"]].astype(np.float64)
    u /= np.linalg.norm(u, axis=-1, keepdims=True)
    rs = B.JI["right_shoulder"]
    p32 = base.copy()
    p32[rows, rs] = p32[rows, rs] + (0.01 * u[rows]).astype(np.float32)
    b64 = base.astype(np.float64)
    p64 = b64.copy()
    p64[rows, rs] += 0.01 * u[rows]
    L1b, L2b = K.layers(R.real_case_frames(base, ts, rc, "b"), "SLOT", rq)
    L1a64, L2a64 = K.layers(R.real_case_frames(p64, ts, rc, "p64"), "SLOT", rq)
    L1a32, L2a32 = K.layers(R.real_case_frames(p32, ts, rc, "p32"), "SLOT", rq)
    rep = {"label": LABEL, "float64_exact_diagnostic": account(L1b, L1a64, L2b, L2a64, rows),
           "float32_candidate": account(L1b, L1a32, L2b, L2a32, rows)}
    rep["float32_candidate"]["stored_changes"] = rep["float32_candidate"]["stored_changes"][:12] + (
        [{"truncated": len(rep["float32_candidate"]["stored_changes"]) - 12}] if len(rep["float32_candidate"]["stored_changes"]) > 12 else [])
    C.dump(out / "c1_stored_mechanism.json", rep)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "stored_changes"} for k, v in rep.items() if isinstance(v, dict)}))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("c1-stored-mechanism")
    for k in ("declaration", "declaration-sha", "out-dir"):
        p.add_argument(f"--{k}", required=True)
    a = ap.parse_args(argv)
    return {"c1-stored-mechanism": cmd_c1}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
