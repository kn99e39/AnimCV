"""DIAGNOSTIC (Worklog 88): exact temporal depth-motion quantities and Shapley accounting for frozen connected limbs.

Conventions: unit connected chains (P = 0, M = rho u, D = M + (1 - rho) v), canonical camera axes X right,
Y depth, Z up; all motion quantities in %L (normalized x 100).  For a chronological pair (a, b):
  predicted depth displacement  dP = Y_hat(b) - Y_hat(a);  GT depth displacement  dG = Y(b) - Y(a);
  signed motion residual  dP - dG;  absolute motion residual  |dP - dG|  (the Worklog 87 depth-motion residual);
  amplitude  |dP|  and  |dG|;  static-error change  |e(b)| - |e(a)| with e = Y_hat - Y.
Shapley attribution: the frozen model's output change between frame a and frame b is split exactly over the
inputs (players) switched from a to b; hybrid inputs are EVALUATION-ONLY sensitivity probes, never observations.
"""

from __future__ import annotations

import itertools
import math

import numpy as np

# GT-motion strata (declared before any Worklog 88 diagnostic; evaluation-only grouping):
# per-pair GT 3D displacement of the distal joint in %L.
LOW_GT_MOTION = 4.0
HIGH_GT_MOTION = 15.0
FALSE_MOTION = 5.0  # a low-GT-motion pair counts as false depth motion when |dP| exceeds 5 %L


def motion_terms(pred_a, pred_b, gt_a, gt_b, axis=1) -> dict:
    """Per-pair depth-motion quantities (inputs in normalized units; outputs in %L)."""
    pa, pb, ga, gb = (100 * np.asarray(x, np.float64)[..., axis] for x in (pred_a, pred_b, gt_a, gt_b))
    dp, dg = pb - pa, gb - ga
    return {"pred_disp": dp, "gt_disp": dg, "signed_residual": dp - dg, "abs_residual": np.abs(dp - dg),
            "pred_amp": np.abs(dp), "gt_amp": np.abs(dg), "static_err_change": np.abs(pb - gb) - np.abs(pa - ga),
            "over_response": np.abs(dp) - np.abs(dg)}


def gt_motion_stratum(gt_a, gt_b) -> np.ndarray:
    d = 100 * np.linalg.norm(np.asarray(gt_b, np.float64) - np.asarray(gt_a, np.float64), axis=-1)
    out = np.full(len(d), "mid", dtype=object)
    out[d < LOW_GT_MOTION] = "low"
    out[d >= HIGH_GT_MOTION] = "high"
    return out.astype(str)


def shapley_weights(n: int) -> dict:
    """Shapley coefficient |S|! (n - |S| - 1)! / n! per coalition size."""
    return {k: math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n) for k in range(n)}


def shapley(values: dict, players: tuple) -> dict:
    """Exact Shapley values from v(S) for every coalition S (keys = frozensets of players switched to frame b)."""
    n = len(players)
    w = shapley_weights(n)
    phi = {}
    for i in players:
        others = [p for p in players if p != i]
        total = 0.0
        for k in range(n):
            for s in itertools.combinations(others, k):
                s = frozenset(s)
                total = total + w[k] * (values[s | {i}] - values[s])
        phi[i] = total
    return phi


def depth_change_terms(e_a, w_a, e_b, w_b) -> dict:
    """Exact split of the distal depth change of a unit chain into proximal-direction, distal-direction and ratio terms.

    D_y = rho u_y + (1 - rho) v_y; with midpoint averages (bar) the product rule is exact:
    dD_y = rho_bar du_y + u_bar_y drho + (1 - rho_bar) dv_y - v_bar_y drho.
    """
    def f(e, w):
        e, w = np.asarray(e, np.float64), np.asarray(w, np.float64)
        rho = np.linalg.norm(e, axis=-1)
        u = e / np.maximum(rho[..., None], 1e-12)
        s = w - e
        v = s / np.maximum(np.linalg.norm(s, axis=-1)[..., None], 1e-12)
        return u, v, rho
    ua, va, ra = f(e_a, w_a)
    ub, vb, rb = f(e_b, w_b)
    rbar, ubar, vbar = (ra + rb) / 2, (ua + ub) / 2, (va + vb) / 2
    return {"proximal_direction": 100 * rbar * (ub[..., 1] - ua[..., 1]),
            "distal_direction": 100 * (1 - rbar) * (vb[..., 1] - va[..., 1]),
            "ratio": 100 * (rb - ra) * (ubar[..., 1] - vbar[..., 1])}
