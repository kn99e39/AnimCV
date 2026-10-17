"""DIAGNOSTIC (Worklog 87): near-plane regime accounting and exact error decompositions for connected chains.

Chains use the Worklog 80/86 normalized convention: P = 0, M = rho u, D = M + (1 - rho) v (unit total length),
canonical camera axes X right, Y depth (forward), Z up.  Everything here is evaluation-only: GT-derived regimes
and counterfactual factor hybrids are never runtime inputs or deployable models.

Regimes (``REGIMES``, declared before any Worklog 87 diagnostic was computed; historical boundaries):
  D_degenerate   -- GT reach |D| < 0.25: the P->D direction (and its depth sign) is not reliably defined;
  C_out_of_plane -- |f_chain| >= 0.5 (Worklog 76-86 out-of-plane boundary);
  A_true_near    -- |f_proximal| < sin 10 deg AND |f_distal| < sin 10 deg (both segments near the image plane);
  B_cancellation -- |f_chain| < sin 10 deg but not A (the chain is near-plane only because segment depths cancel
                    or one segment leaves the plane);
  I_intermediate -- everything else (sin 10 deg <= |f_chain| < 0.5, not A).
f = the depth (Y) component of the unit segment / chain direction.
"""

from __future__ import annotations

import math

import numpy as np

STABLE = math.sin(math.radians(10))
OUT_OF_PLANE = 0.5
DEGENERATE_REACH = 0.25
REGIMES = ("A_true_near", "B_cancellation", "I_intermediate", "C_out_of_plane", "D_degenerate")


def _unit(x):
    x = np.asarray(x, np.float64)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-12), n[..., 0]


def factors(elbow, wrist):
    """(u, v, rho) of a unit connected chain: M = rho u, D = M + (1 - rho) v."""
    e, w = np.asarray(elbow, np.float64), np.asarray(wrist, np.float64)
    u, rho = _unit(e)
    v, _ = _unit(w - e)
    return u, v, rho


def compose(u, v, rho):
    m = np.asarray(rho)[..., None] * u
    return m, m + (1 - np.asarray(rho))[..., None] * v


def depth_components(elbow, wrist):
    """Depth (Y) components of the unit proximal, distal and chain directions, and the reach |D|."""
    u, v, _ = factors(elbow, wrist)
    c, reach = _unit(wrist)
    return u[..., 1], v[..., 1], c[..., 1], reach


def classify(elbow, wrist) -> np.ndarray:
    fp, fd, fc, reach = depth_components(elbow, wrist)
    out = np.full(len(fp), "I_intermediate", dtype=object)
    out[np.abs(fc) >= OUT_OF_PLANE] = "C_out_of_plane"
    true_near = (np.abs(fp) < STABLE) & (np.abs(fd) < STABLE)
    out[true_near] = "A_true_near"
    out[(np.abs(fc) < STABLE) & ~true_near] = "B_cancellation"
    out[reach < DEGENERATE_REACH] = "D_degenerate"
    return out.astype(str)


def axis_squared_errors(pred, gt) -> dict:
    """Exact per-axis squared-error decomposition: |p - g|^2 = dX^2 + dY^2 + dZ^2 (normalized units)."""
    d = np.asarray(pred, np.float64) - np.asarray(gt, np.float64)
    sq = d ** 2
    return {"total": sq.sum(-1), "x": sq[..., 0], "depth": sq[..., 1], "z": sq[..., 2], "signed_depth": d[..., 1]}


def distal_decomposition(elbow, wrist, gt_elbow, gt_wrist) -> dict:
    """Exact split of the distal error: D_err = M_err + S_err with S = D - M, |D_err|^2 = |M_err|^2 + |S_err|^2 + 2 M_err.S_err."""
    me = np.asarray(elbow, np.float64) - np.asarray(gt_elbow, np.float64)
    se = (np.asarray(wrist, np.float64) - np.asarray(elbow, np.float64)) - (np.asarray(gt_wrist, np.float64) - np.asarray(gt_elbow, np.float64))
    return {"middle_part": (me ** 2).sum(-1), "segment_part": (se ** 2).sum(-1), "cross": 2 * (me * se).sum(-1)}


def factorial_distal_errors(e0, w0, e1, w1, ge, gw) -> dict:
    """EVALUATION-ONLY 2^3 hybrids: each of (u, v, rho) taken from model 0 or model 1; distal squared error per hybrid.

    Keys 'uvr' with digits 0/1 (e.g. '100' = u from model 1, v and rho from model 0).  The hybrids are geometrically
    valid unit chains, never deployable models.
    """
    f0, f1 = factors(e0, w0), factors(e1, w1)
    out = {}
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                u = (f0, f1)[a][0]
                v = (f0, f1)[b][1]
                r = (f0, f1)[c][2]
                _, d = compose(u, v, r)
                out[f"{a}{b}{c}"] = ((d - np.asarray(gw, np.float64)) ** 2).sum(-1)
    return out


def factorial_effects(y: dict) -> dict:
    """Main effects and interactions of switching each factor from model 0 to model 1 (means over the given rows)."""
    m = {k: float(np.mean(v)) for k, v in y.items()}
    eff = {}
    for name, pos in (("u", 0), ("v", 1), ("rho", 2)):
        hi = [k for k in m if k[pos] == "1"]
        lo = [k for k in m if k[pos] == "0"]
        eff[name] = (sum(m[k] for k in hi) - sum(m[k] for k in lo)) / 4
    total = m["111"] - m["000"]
    eff["interaction_residual"] = total - (m["100"] - m["000"]) - (m["010"] - m["000"]) - (m["001"] - m["000"])
    eff["total"] = total
    eff["single_switch"] = {"u": m["100"] - m["000"], "v": m["010"] - m["000"], "rho": m["001"] - m["000"]}
    return eff
