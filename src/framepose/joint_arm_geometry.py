"""DIAGNOSTIC (Worklog 80): one-stage joint estimation of one connected, scale-free left arm.

ONE network predicts one canonical-camera-frame arm from runtime observations only
(detector pair geometry in the person crop and, for J1, the frozen Worklog 68 Qwen3-VL
readouts; no E1, H0, oracle sign, GT or temporal input):

* u  - upper-arm unit direction (free 3-vector, normalized),
* v  - forearm unit direction (free 3-vector, normalized),
* rho = sigmoid(z) in (0, 1) - source upper-arm share of the total arm length.

The arm is built by construction as one connected chain with a single shoulder anchor and
unit total length:  S = 0,  E = rho u,  W = rho u + (1 - rho) v.  Chain direction, reach
fraction (= |W| <= 1 by the triangle inequality), bend side and both segment directions are
therefore all derived from the same configuration - exactly the Worklog 61
``source_chain_geometry`` / ``observed_bend_side`` of (E - S, W - E) - and at evaluation the
three points enter the unchanged Worklog 62 ``measure_chain`` (A * Rz(-yaw_H0)) and Worklog 76
``full_arm`` like H0's joints.  Degenerate (near-straight, bend sine < sin 1 deg) arms keep a
valid direction and reach and use the historical rest-plane side fallback; nothing is
rejected.  Source proportions (rho) are learned per frame and never imposed on the BaseRig,
whose L1/L2 come only from the rest snapshot.  Not imported by any production module.
"""

from __future__ import annotations

import numpy as np

from framepose.contract import JOINT_INDEX

CANDIDATES = ("J0_GEOMETRY_ONLY", "J1_GEOMETRY_PLUS_VISION")
PAIR_FEATURES = 13 * 3
VISUAL_PROJECTION = 64
HIDDEN = 128
OUTPUTS = 7                      # u (3), v (3), rho logit (1)
LENGTH_EPS = 1e-8                # Worklog 61 ``_EPS`` for segment lengths
ARM = ("left_shoulder", "left_elbow", "left_wrist")
REPRESENTATION = {
    "prediction": "u (upper-arm unit direction), v (forearm unit direction), rho = sigmoid(z) upper-arm length share; "
                  "canonical camera frame (+X right, +Y forward depth, +Z up)",
    "connected_arm": "S = 0, E = rho*u, W = rho*u + (1 - rho)*v; total source length 1; one shoulder anchor",
    "derived": "chain direction = W/|W|, reach = |W| (<= 1 by construction), bend side = Worklog 61 observed_bend_side of "
               "(E, W - E); upper/forearm directions = u, v. At evaluation: Worklog 62 measure_chain(S, E, W, yaw_H0, A) "
               "-> Worklog 76 full_arm(measure, measure.direction) - the same path as H0's joints",
    "degeneracy": "near-straight arms (bend sine < sin 1 deg) keep direction and reach and take the Worklog 61 rest-plane "
                  "side fallback, as for any degenerate source; |W| = 0 (exactly folded) would be unavailable",
    "why_not_sequential": "direction, reach and side are not separately predicted or conditioned on a committed direction; "
                          "all come from one predicted connected configuration trained with one joint-position objective",
    "constraints_by_construction": "connectivity, single anchor, unit total length, reach in [0, 1], unit segment directions, "
                                   "scale-free (no metric or BaseRig length)",
    "network": "shared Linear(4096->64) per visual token (S, E, W, global) -> 256; concat with 39 pair-geometry features -> "
               "Linear(295->128) GELU Linear(128->128) GELU Linear(128->7)",
    "loss": "mean over GT-valid rows of |E_pred - E_GT| + |W_pred - W_GT| (Euclidean), GT positions relative to the GT "
            "shoulder divided by the GT total arm length |E-S| + |W-E|; weights 1 and 1",
    "validity": "a row trains when its three GT arm joints are valid, finite and both GT segments are longer than 1e-8; "
                "near-straight and folded arms are kept (positions are always defined)",
    "selection_metric": "3DPW validation mean of the same connected-arm objective (elbow + wrist normalized error)",
    "J0_vs_J1": "identical graph, seed, initialization, samples, optimizer and selection; J0 visual tensors are exact zeros",
}


def candidate_visual(readouts: np.ndarray, candidate: str) -> np.ndarray:
    r = np.asarray(readouts)
    if candidate == "J0_GEOMETRY_ONLY":
        return np.zeros_like(r, dtype=np.float16)
    if candidate == "J1_GEOMETRY_PLUS_VISION":
        return r
    raise ValueError(f"unknown candidate {candidate!r}")


def normalized_arm_targets(target_3d: np.ndarray, target_valid: np.ndarray):
    """GT elbow and wrist relative to the shoulder in units of the GT total arm length (labels only)."""
    p = np.asarray(target_3d, np.float64)
    s, e, w = (p[:, JOINT_INDEX[j]] for j in ARM)
    valid = np.asarray(target_valid, bool)[:, [JOINT_INDEX[j] for j in ARM]].all(1)
    finite = np.isfinite(np.stack([s, e, w], 1)).all((1, 2))
    lu, ll = np.linalg.norm(e - s, axis=1), np.linalg.norm(w - e, axis=1)
    ok = valid & finite & (lu > LENGTH_EPS) & (ll > LENGTH_EPS)
    total = np.where(ok, lu + ll, 1.0)[:, None]
    elbow = np.where(ok[:, None], (e - s) / total, 0.0)
    wrist = np.where(ok[:, None], (w - s) / total, 0.0)
    return elbow, wrist, ok


def decode_numpy(raw: np.ndarray):
    """(N, 7) raw outputs -> shoulder-relative unit-length arm (E (N, 3), W (N, 3)), rho (N,)."""
    raw = np.asarray(raw, np.float64)
    u = raw[:, 0:3] / np.linalg.norm(raw[:, 0:3], axis=1, keepdims=True)
    v = raw[:, 3:6] / np.linalg.norm(raw[:, 3:6], axis=1, keepdims=True)
    rho = 1.0 / (1.0 + np.exp(-raw[:, 6]))
    elbow = rho[:, None] * u
    return elbow, elbow + (1.0 - rho)[:, None] * v, rho


def build_joint_arm(channels: int):
    import torch
    from torch import nn

    class JointArmGeometry(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.visual = nn.Linear(channels, VISUAL_PROJECTION)
            self.trunk = nn.Sequential(nn.Linear(PAIR_FEATURES + 4 * VISUAL_PROJECTION, HIDDEN), nn.GELU(),
                                       nn.Linear(HIDDEN, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, OUTPUTS))

        def forward(self, visual, geometry):
            """visual (B, 4, C); geometry (B, 39) -> raw (B, 7)."""
            v = self.visual(visual.float()).flatten(1)
            return self.trunk(torch.cat([geometry.float(), v], dim=-1))

    return JointArmGeometry()


def decode_torch(raw, eps: float = 1e-6):
    import torch

    u = raw[:, 0:3] / raw[:, 0:3].norm(dim=-1, keepdim=True).clamp_min(eps)
    v = raw[:, 3:6] / raw[:, 3:6].norm(dim=-1, keepdim=True).clamp_min(eps)
    rho = torch.sigmoid(raw[:, 6:7])
    elbow = rho * u
    return elbow, elbow + (1.0 - rho) * v


def joint_loss(raw, elbow_t, wrist_t, mask):
    """Declared objective: masked mean of normalized elbow + wrist Euclidean errors (weights 1 and 1)."""
    elbow, wrist = decode_torch(raw)
    e = (elbow - elbow_t).norm(dim=-1)
    w = (wrist - wrist_t).norm(dim=-1)
    n = mask.sum().clamp_min(1.0)
    return ((e + w) * mask).sum() / n, (e * mask).sum() / n, (w * mask).sum() / n


def predict_joint(model, visual: np.ndarray, geometry: np.ndarray, positions: np.ndarray, device) -> np.ndarray:
    """Inference from runtime inputs only: raw (N, 7)."""
    import torch

    out = []
    model.eval()
    with torch.no_grad():
        for s in range(0, len(positions), 2048):
            idx = np.asarray(positions[s:s + 2048])
            out.append(model(torch.as_tensor(np.asarray(visual[idx]), device=device),
                             torch.as_tensor(geometry[idx], device=device)).cpu().numpy().astype(np.float64))
    return np.concatenate(out)
