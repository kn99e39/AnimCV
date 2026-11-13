#!/usr/bin/env python3
"""W143 Phase 1 mechanism investigation (PRE-DECLARATION; no new GT, no held-out data).

(a) Synthetic correspondence controls: does a homography preference make the rotation-only measurement unusable?  Nuisance
    factors on a pure or near-pure rotation: intrinsic error, residual radial distortion, rolling shutter, millimetre
    translation with near depth, and the planar-translation trap.  Reports the GRIC fit model, the W143 rotation-validation
    decision and reasons, the true rotation error, the bias bound, rotation-vs-homography agreement and the translation edge
    state.  (Cycle consistency needs a sequence and is exercised in the tests.)
(b) GT-free statistics of the HISTORICAL W142 walking_static and synthetic edges (GRIC margins, inlier support,
    consensus residuals): how many W142 'planar-ambiguous' edges carry rotation support by the W143 criteria.

    PY -X utf8 scripts/w143_mechanism.py --out <dir>
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]
import numpy as np                                                    # noqa: E402

import app.experimental_camera_pose as CP                             # noqa: E402
import app.experimental_decoupled_camera_state as DC                  # noqa: E402

K = np.array([[535.4, 0, 320.1], [0, 539.2, 247.6], [0, 0, 1.0]])     # fr3 intrinsics


def rot(axis, deg):
    from scipy.spatial.transform import Rotation
    a = np.asarray(axis, float)
    return Rotation.from_rotvec(np.radians(deg) * a / np.linalg.norm(a)).as_matrix()


def scene(n=1200, near=1.0, far=5.0, planar=False, seed=0):
    rng = np.random.default_rng(seed)
    if planar:
        return np.column_stack([rng.uniform(-2, 2, n), rng.uniform(-1.5, 1.5, n), np.full(n, 3.0)])
    z = rng.uniform(near, far, n)
    return np.column_stack([rng.uniform(-0.8, 0.8, n) * z, rng.uniform(-0.6, 0.6, n) * z, z])


def distort(x, k1, Kt=K):
    u = (x - Kt[:2, 2]) / np.array([Kt[0, 0], Kt[1, 1]])
    r2 = np.sum(u * u, 1, keepdims=True)
    return u * (1 + k1 * r2) * np.array([Kt[0, 0], Kt[1, 1]]) + Kt[:2, 2]


def case(name, R_rel, t=(0, 0, 0), planar=False, near=1.0, far=5.0, k_obs=None, k1=0.0, rs_deg_per_frame=0.0, noise=0.4):
    X = scene(planar=planar, near=near, far=far)
    T1 = CP.make_T(np.eye(3), [0, 0, 0])
    T2 = CP.make_T(R_rel, t)
    Ko = K if k_obs is None else k_obs
    x1, z1 = CP.project(Ko, T1, X)
    if rs_deg_per_frame:
        # rolling shutter: each row of frame 2 is exposed at a rotation proportional to its row position
        x2 = np.zeros_like(x1)
        z2 = np.zeros(len(X))
        for k in range(len(X)):
            row0, _ = CP.project(Ko, T2, X[k:k + 1])
            frac = float(np.clip(row0[0, 1] / 480.0, 0, 1))
            Rk = R_rel @ rot([0, 1, 0], rs_deg_per_frame * frac)
            p, zz = CP.project(Ko, CP.make_T(Rk, t), X[k:k + 1])
            x2[k], z2[k] = p[0], zz[0]
    else:
        x2, z2 = CP.project(Ko, T2, X)
    if k1:
        x1, x2 = distort(x1, k1, Ko), distort(x2, k1, Ko)
    ok = (z1 > 0) & (z2 > 0) & np.all((x1 > 0) & (x1 < [640, 480]), 1) & np.all((x2 > 0) & (x2 < [640, 480]), 1)
    rng = np.random.default_rng(7)
    x1, x2 = x1[ok] + rng.normal(0, noise, (ok.sum(), 2)), x2[ok] + rng.normal(0, noise, (ok.sum(), 2))
    e = DC.analyse_edge(K, x1, x2, seed=1)
    e.pop("_inl")
    true_M = R_rel.T                                       # R_C2C1 with R_WC1 = I
    return {"case": name, "fit_model": e["fit_model"], "rotation_candidate": e["rotation_candidate"], "reject": e["reject"],
            "rot_err_true_deg": float(CP.geodesic_deg(np.asarray(e["M"]), true_M)), "bias_bound_deg": e["rotation_bias_bound_deg"],
            "support": e["rot_support_ratio"], "resid_p50": e["rot_consensus_resid_p50_px"], "resid_p90": e["rot_consensus_resid_p90_px"],
            "agree_RH_deg": e["agree_rot_vs_homography_deg"], "agree_RE_deg": e["agree_rot_vs_essential_deg"],
            "gric_R_minus_H": e["gric"]["R"] - e["gric"]["H"], "translation_edge": e["translation_edge"], "parallax_px": e.get("parallax_px")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    R05 = rot([0.3, 1, 0.1], 0.5)
    Kf1, Kf2, Kc = K.copy(), K.copy(), K.copy()
    Kf1[0, 0] *= 1.01
    Kf1[1, 1] *= 1.01
    Kf2[0, 0] *= 1.02
    Kf2[1, 1] *= 1.02
    Kc[0, 2] += 5
    Kc[1, 2] -= 5
    cases = [case("pure_rotation_0.5deg", R05),
             case("pure_rotation_3deg", rot([0.3, 1, 0.1], 3.0)),
             case("focal_error_1pct", R05, k_obs=Kf1), case("focal_error_2pct_3deg", rot([0.3, 1, 0.1], 3.0), k_obs=Kf2),
             case("principal_point_error_5px_3deg", rot([0.3, 1, 0.1], 3.0), k_obs=Kc),
             case("radial_k1_0.02", R05, k1=0.02), case("radial_k1_0.05_3deg", rot([0.3, 1, 0.1], 3.0), k1=0.05),
             case("rolling_shutter_0.3deg_per_frame", R05, rs_deg_per_frame=0.3),
             case("translation_2mm_near_1to5m", R05, t=(0.002, 0, 0)), case("translation_10mm_near_1to5m", R05, t=(0.01, 0, 0)),
             case("translation_30mm_near_1to5m", R05, t=(0.03, 0, 0)), case("translation_10mm_near_0.6to2m", R05, t=(0.01, 0, 0), near=0.6, far=2.0),
             case("planar_translation_200mm", np.eye(3), t=(0.2, 0, 0), planar=True),
             case("planar_translation_30mm", R05, t=(0.03, 0, 0), planar=True),
             case("general_translation_300mm", rot([0, 1, 0], 2.0), t=(0.3, 0, 0.05))]
    res = {"label": "PRE-DECLARATION mechanism investigation (synthetic correspondences + GT-free historical W142 statistics)", "cases": cases}
    # ---- (b) historical W142 GT-free statistics ----
    hist = {}
    for run in ("walking_static_rgbmask", "walking_xyz_rgbmask", "syn_A3_rgbmask", "syn_A1_rgbmask"):
        est = json.load(open(f"C:/Users/dna10/animcv-w142/results/estimate/{run}/estimate.json"))
        rows = []
        for e in est["edges"]:
            if "n_inliers" not in e or e["rot_residual_p90_px"] is None:
                continue
            nb = max(e["n_inliers"][m] for m in ("R", "H", "E"))
            rows.append({"state": e["state"], "support": e["n_inliers"]["R"] / max(nb, 1), "p90": e["rot_residual_p90_px"],
                         "gric_RmH": e["gric"]["R"] - e["gric"]["H"], "span": e["span"]})
        by = {}
        for r in rows:
            b = by.setdefault(r["state"], {"n": 0, "pass_support_and_resid": 0, "support": [], "p90": [], "gric_RmH": []})
            b["n"] += 1
            b["pass_support_and_resid"] += int(r["support"] >= DC.SUPPORT_RATIO_MIN and r["p90"] <= DC.CONSENSUS_P90_MAX_PX)
            for k in ("support", "p90", "gric_RmH"):
                b[k].append(r[k])
        hist[run] = {s: {"n": b["n"], "frac_pass_support_and_resid": b["pass_support_and_resid"] / b["n"],
                         "support_median": float(np.median(b["support"])), "p90_resid_median_px": float(np.median(b["p90"])),
                         "gric_R_minus_H_median": float(np.median(b["gric_RmH"]))} for s, b in by.items()}
    res["w142_historical_gt_free"] = hist
    res["note"] = ("W142 stored the rotation residual p90 over the BEST-model inliers, i.e. the same consensus statistic W143 uses; the "
                   "homography-agreement and cycle checks were not stored by W142 and are not previewed here.")
    (out / "mechanism.json").write_text(json.dumps(res, indent=1))
    for c in cases:
        print(f"{c['case']:34s} fit={c['fit_model']} cand={c['rotation_candidate']!s:5s} err={c['rot_err_true_deg']:.3f} bound={c['bias_bound_deg']:.3f} "
              f"sup={c['support']:.2f} p90={c['resid_p90']:.2f} RH={c['agree_RH_deg']} tr={c['translation_edge']} par={c['parallax_px']} rej={c['reject']}")
    print(json.dumps(hist, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
