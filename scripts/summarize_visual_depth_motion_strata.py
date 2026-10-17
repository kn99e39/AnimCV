#!/usr/bin/env python3
"""POST HOC (Worklog 88): stratum-wise removal effects and connected-chain kinematics of the depth-motion excess.

Added after the predeclared run (``run_visual_depth_motion_attribution.py``) once the aggregate 3DPW VALIDATION
(0.2 s) excess turned out to be near zero while its low-motion stratum was not.  Reads ``attribution_v2/pairs.json``
only; no model is evaluated.  Per pair: excess = res1 - res0, removal_X = res1 - |dP1 - phi_X - dG| (%L).  Intervals
are the Worklog 75/76 sequence-cluster bootstrap.  Tails: 95th / 99th percentile of the per-pair residual and the rate
above 20 %L, per model.

Chain kinematics (3DPW populations, frozen Worklog 87 G0 / G1 predictions vs GT, unit chains, L = 1): per pair the
temporal change of u_y, v_y (depth components of the proximal / distal directions), rho, middle depth, distal depth,
full-chain direction depth c_y = (w / |w|)_y and bend angle acos(u.v); the residual of each is |d pred - d GT| (x100)
and the excess is G1 minus G0.  Bend-side flip = bend-plane normals (u x v) of a and b point against each other while
both frames bend >= 20 degrees.  This summary does not enter the CASE classification.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_visual_depth_motion_attribution import _brief  # noqa: E402

from framepose.bank import load_bank  # noqa: E402

BEND_SIDE_MIN = np.deg2rad(20.0)
SEVERE = 20.0  # %L distal depth-motion residual


def kinematics(e, w):
    rho = np.linalg.norm(e, axis=1)
    u = e / rho[:, None]
    seg = w - e
    v = seg / np.linalg.norm(seg, axis=1, keepdims=True)
    c = w / np.linalg.norm(w, axis=1, keepdims=True)
    bend = np.arccos(np.clip((u * v).sum(1), -1, 1))
    n = np.cross(u, v)
    return {"u_y": u[:, 1], "v_y": v[:, 1], "rho": rho, "middle_y": e[:, 1], "distal_y": w[:, 1], "chain_y": c[:, 1],
            "bend_deg": np.rad2deg(bend) / 100, "_bend": bend, "_n": n}


def chain_kinematics(rows, pred, chain, pos):
    a = np.array([pos[(r["sequence_id"], r["frame"])][0] for r in rows])
    b = np.array([pos[(r["sequence_id"], r["frame"])][1] for r in rows])
    k = {who: [kinematics(pred[f"{chain}_{name}_elbow"][i], pred[f"{chain}_{name}_wrist"][i]) for i in (a, b)]
         for who, name in (("gt", "target"), ("g0", "g0"), ("g1", "g1"))}
    seq = [r["sequence_id"] for r in rows]
    out = {}
    for q in ("u_y", "v_y", "rho", "middle_y", "distal_y", "chain_y", "bend_deg"):
        dg = k["gt"][1][q] - k["gt"][0][q]
        res = {g: 100 * np.abs(k[g][1][q] - k[g][0][q] - dg) for g in ("g0", "g1")}
        out[q] = {"residual_g0": float(res["g0"].mean()), "residual_g1": float(res["g1"].mean()),
                  "excess": _brief(seq, res["g1"] - res["g0"])}
    for g in ("gt", "g0", "g1"):
        ok = (k[g][0]["_bend"] >= BEND_SIDE_MIN) & (k[g][1]["_bend"] >= BEND_SIDE_MIN)
        flip = (k[g][0]["_n"] * k[g][1]["_n"]).sum(1) < 0
        out[f"bend_side_flip_rate_{g}"] = float(flip[ok].mean()) if ok.any() else None
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attribution-dir", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True, help="Worklog 87 predictions_3dpw.npz")
    parser.add_argument("--bank", type=Path, required=True)
    args = parser.parse_args()
    pairs = json.loads((args.attribution_dir / "pairs.json").read_text())
    out = {}
    for key, rows in pairs.items():
        r0, r1 = np.array([x["res0"] for x in rows]), np.array([x["res1"] for x in rows])
        out[key] = {"tails": {g: {"p95": float(np.percentile(v, 95)), "p99": float(np.percentile(v, 99)),
                                  f"rate_above_{SEVERE:g}": float((v > SEVERE).mean())} for g, v in (("g0", r0), ("g1", r1))}}
        for stratum in ("low", "mid", "high"):
            r = [x for x in rows if x["stratum"] == stratum]
            if not r:
                continue
            seq = [x["sequence_id"] for x in r]
            out[key][stratum] = {
                "pairs": len(r),
                "excess": _brief(seq, [x["res1"] - x["res0"] for x in r]),
                **{f"removal_{k}": _brief(seq, [x["res1"] - abs(x["dP1"] - x["phi"][k] - x["dG"]) for x in r])
                   for k in ("visual", "content", "position", "geometry")}}
    pred, bank = np.load(args.predictions), load_bank(args.bank)
    kin = {}
    for split, pop, step in (("test", "3dpw_test", 5), ("validation", "3dpw_validation", 6), ("validation", "3dpw_validation_0.1s", 3)):
        idx = {(bank.samples[p].sequence_id, int(bank.samples[p].frame_index)): p for p in bank.indices(split)}
        for chain in ("left_arm", "right_arm", "right_leg"):
            rows = pairs[f"{chain}_{pop}"]
            pos = {(r["sequence_id"], r["frame"]): (idx[(r["sequence_id"], r["frame"])], idx[(r["sequence_id"], r["frame"] + step)]) for r in rows}
            kin[f"{chain}_{pop}"] = chain_kinematics(rows, pred, chain, pos)
    for key, v in out.items():
        if key in kin:
            v["chain_kinematics"] = kin[key]
    (args.attribution_dir / "strata_posthoc.json").write_text(json.dumps(out, indent=1, default=str))
    for key, v in out.items():
        cells = []
        for s, x in v.items():
            if s in ("chain_kinematics", "tails"):
                continue
            ci = lambda z: f"{z['mean']:+.2f} [{z['ci95'][0]:+.2f},{z['ci95'][1]:+.2f}]" if z.get("ci95") else f"{z.get('mean', float('nan')):+.2f}"  # noqa: E731
            cells.append(f"{s} n={x['pairs']} E {ci(x['excess'])} rmVis {ci(x['removal_visual'])} rmCon {ci(x['removal_content'])}")
        t = v["tails"]
        print(key, " | ".join(cells), f" | tails p95 {t['g0']['p95']:.1f}->{t['g1']['p95']:.1f} p99 {t['g0']['p99']:.1f}->{t['g1']['p99']:.1f}"
              f" >20 {t['g0']['rate_above_20']:.3f}->{t['g1']['rate_above_20']:.3f}")
    for key, v in kin.items():
        print(key, "  ".join(f"{q}: {x['residual_g0']:.2f}->{x['residual_g1']:.2f}" for q, x in v.items() if isinstance(x, dict)),
              " side-flip gt/g0/g1", [None if v[f"bend_side_flip_rate_{g}"] is None else round(v[f"bend_side_flip_rate_{g}"], 3) for g in ("gt", "g0", "g1")])


if __name__ == "__main__":
    main()
