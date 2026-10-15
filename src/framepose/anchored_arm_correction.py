"""DIAGNOSTIC (Worklog 82): RGB-conditioned correction of a frozen 2D-derived connected arm.

Anchor: the frozen Worklog 80 J0 model (2D only) gives one connected unit-length arm per frame,
u0 (upper-arm unit direction), v0 (forearm unit direction), rho0 (upper share), with
S = 0, E0 = rho0 u0, W0 = E0 + (1 - rho0) v0.  Its weights are never updated and it never sees RGB.

Correction: ONE network predicts c = (w_u, w_v, delta) in R^7 from the runtime pair geometry, the
frozen anchor arm and (R1 only) the frozen Qwen3-VL readouts:

    u = Rot(w_u) u0,   v = Rot(w_v) v0,   rho = sigmoid(logit(rho0) + delta)
    S = 0,  E = rho u,  W = E + (1 - rho) v

Rot is the Rodrigues rotation by the rotation vector w (axis w/|w|, angle |w|).  The corrected arm
is therefore again one connected unit-length arm with a single shoulder anchor; chain direction,
reach (= |W| <= 1), bend side and segment directions are derived from it exactly as for J0 / J1
(Worklog 62 measure_chain -> Worklog 76 full_arm at evaluation).  Rotations of unit vectors are
defined everywhere, so the correction itself has no degenerate state; a near-straight corrected arm
takes the historical rest-plane side fallback and |W| = 0 would be unavailable.  The correction
head's last layer is zero-initialized, so at initialization c = 0 and the corrected arm equals the
anchor exactly.  No cap, gate, blend coefficient, temporal input or E1/H0 quantity is used.
Not imported by any production module.
"""

from __future__ import annotations

import numpy as np

CANDIDATES = ("R0_GEOMETRY_CORRECTION", "R1_VISUAL_CORRECTION")
PAIR_FEATURES = 13 * 3
ANCHOR_FEATURES = 13              # u0 (3), v0 (3), rho0 (1), E0 (3), W0 (3)
VISUAL_PROJECTION = 64
HIDDEN = 128
OUTPUTS = 7                       # w_u (3), w_v (3), delta (1)
RHO_EPS = 1e-6
REPRESENTATION = {
    "anchor": "frozen Worklog 80 J0 (2D only, weights frozen, no RGB): u0, v0, rho0 -> S=0, E0=rho0*u0, W0=E0+(1-rho0)*v0",
    "correction": "c = (w_u, w_v, delta): u = Rodrigues(w_u) u0, v = Rodrigues(w_v) v0, rho = sigmoid(logit(rho0) + delta); "
                  "S=0, E=rho*u, W=E+(1-rho)*v",
    "coherence": "the corrected output is one connected unit-length arm with one shoulder anchor; direction, reach (|W| <= 1), "
                 "bend side and segment directions are all derived from it (Worklog 62 measure_chain -> Worklog 76 full_arm)",
    "identity": "last correction layer zero-initialized -> c = 0 -> corrected arm == anchor arm exactly at initialization",
    "degeneracy": "rotations of unit vectors are defined everywhere; near-straight corrected arms take the Worklog 61 "
                  "rest-plane side fallback; |W| = 0 would be unavailable (reported, never replaced)",
    "network": "shared Linear(4096->64) per visual token -> 256; concat with 39 pair-geometry + 13 anchor features -> "
               "Linear(308->128) GELU Linear(128->128) GELU Linear(128->7, zero-initialized)",
    "loss": "Worklog 80 objective: mean over GT-valid rows of |E - E_GT| + |W - W_GT| (Euclidean), GT relative to the GT "
            "shoulder and divided by the GT total arm length; weights 1 and 1",
    "selection": "lowest 3DPW validation objective, every 10 epochs and the last epoch",
    "not_sequential": "the correction acts on one whole arm (both segment directions and the proportion together) under one "
                      "joint-position loss; it never takes or commits a separate chain direction, reach or side, and no E1 or "
                      "H0 quantity enters (unlike the Worklog 78/79 direction-conditioned reach/side owner)",
    "R0_vs_R1": "identical graph, seed, initialization, frozen anchor, samples, optimizer and selection; R0 visual tensors are "
                "exact zeros",
}


def candidate_visual(readouts: np.ndarray, candidate: str) -> np.ndarray:
    r = np.asarray(readouts)
    if candidate == "R0_GEOMETRY_CORRECTION":
        return np.zeros_like(r, dtype=np.float16)
    if candidate == "R1_VISUAL_CORRECTION":
        return r
    raise ValueError(f"unknown candidate {candidate!r}")


def anchor_from_raw(raw: np.ndarray):
    """Frozen anchor (u0, v0, rho0) from the anchor model's raw 7-vector output (Worklog 80 decoding)."""
    raw = np.asarray(raw, np.float64)
    u = raw[:, 0:3] / np.linalg.norm(raw[:, 0:3], axis=1, keepdims=True)
    v = raw[:, 3:6] / np.linalg.norm(raw[:, 3:6], axis=1, keepdims=True)
    rho = 1.0 / (1.0 + np.exp(-raw[:, 6]))
    return u, v, rho


def anchor_features(u0: np.ndarray, v0: np.ndarray, rho0: np.ndarray) -> np.ndarray:
    elbow = rho0[:, None] * u0
    wrist = elbow + (1.0 - rho0)[:, None] * v0
    return np.concatenate([u0, v0, rho0[:, None], elbow, wrist], axis=1).astype(np.float32)


def rodrigues_numpy(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Rotate rows of x by rotation vectors w (float64)."""
    w, x = np.asarray(w, np.float64), np.asarray(x, np.float64)
    theta = np.linalg.norm(w, axis=1, keepdims=True)
    small = theta < 1e-8
    k = np.where(small, 0.0, w / np.where(small, 1.0, theta))
    cos, sin = np.cos(theta), np.sin(theta)
    rot = x * cos + np.cross(k, x) * sin + k * (k * x).sum(1, keepdims=True) * (1.0 - cos)
    return np.where(small, x + np.cross(w, x), rot)


def correct_numpy(c: np.ndarray, u0: np.ndarray, v0: np.ndarray, rho0: np.ndarray):
    """Corrected arm (E, W, rho, u, v) in float64 from correction c (N, 7) and the anchor."""
    c = np.asarray(c, np.float64)
    u, v = rodrigues_numpy(c[:, 0:3], u0), rodrigues_numpy(c[:, 3:6], v0)
    r0 = np.clip(rho0, RHO_EPS, 1 - RHO_EPS)
    rho = 1.0 / (1.0 + np.exp(-(np.log(r0 / (1 - r0)) + c[:, 6])))
    elbow = rho[:, None] * u
    return elbow, elbow + (1.0 - rho)[:, None] * v, rho, u, v


def rodrigues_torch(w, x):
    import torch

    theta = w.norm(dim=-1, keepdim=True)
    t2 = theta * theta
    small = theta < 1e-4
    safe = torch.where(small, torch.ones_like(theta), theta)
    a = torch.where(small, 1 - t2 / 6, torch.sin(safe) / safe)                 # sin(t)/t
    b = torch.where(small, 0.5 - t2 / 24, (1 - torch.cos(safe)) / (safe * safe))  # (1-cos t)/t^2
    cos = torch.where(small, 1 - t2 / 2, torch.cos(safe))
    return x * cos + torch.cross(w, x, dim=-1) * a + w * (w * x).sum(-1, keepdim=True) * b


def build_correction(channels: int):
    import torch
    from torch import nn

    class AnchoredArmCorrection(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.visual = nn.Linear(channels, VISUAL_PROJECTION)
            self.trunk = nn.Sequential(nn.Linear(PAIR_FEATURES + ANCHOR_FEATURES + 4 * VISUAL_PROJECTION, HIDDEN), nn.GELU(),
                                       nn.Linear(HIDDEN, HIDDEN), nn.GELU())
            self.head = nn.Linear(HIDDEN, OUTPUTS)
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

        def forward(self, visual, geometry, anchor):
            """visual (B, 4, C); geometry (B, 39); anchor (B, 13) -> correction c (B, 7)."""
            v = self.visual(visual.float()).flatten(1)
            return self.head(self.trunk(torch.cat([geometry.float(), anchor.float(), v], dim=-1)))

    return AnchoredArmCorrection()


def correct_torch(c, anchor):
    """Corrected (E, W) from correction c (B, 7) and anchor features (B, 13)."""
    import torch

    u0, v0, rho0 = anchor[:, 0:3], anchor[:, 3:6], anchor[:, 6:7].clamp(RHO_EPS, 1 - RHO_EPS)
    u, v = rodrigues_torch(c[:, 0:3], u0), rodrigues_torch(c[:, 3:6], v0)
    rho = torch.sigmoid(torch.log(rho0 / (1 - rho0)) + c[:, 6:7])
    elbow = rho * u
    return elbow, elbow + (1 - rho) * v


def correction_loss(c, anchor, elbow_t, wrist_t, mask):
    """Declared objective (Worklog 80): masked mean normalized elbow + wrist Euclidean error, weights 1 and 1."""
    elbow, wrist = correct_torch(c, anchor)
    e, w = (elbow - elbow_t).norm(dim=-1), (wrist - wrist_t).norm(dim=-1)
    n = mask.sum().clamp_min(1.0)
    return ((e + w) * mask).sum() / n, (e * mask).sum() / n, (w * mask).sum() / n


def predict_correction(model, visual, geometry, anchor, positions, device) -> np.ndarray:
    """Inference from runtime inputs and the frozen anchor only: c (N, 7) float64."""
    import torch

    out = []
    model.eval()
    with torch.no_grad():
        for s in range(0, len(positions), 2048):
            idx = np.asarray(positions[s:s + 2048])
            out.append(model(torch.as_tensor(np.asarray(visual[idx]), device=device), torch.as_tensor(geometry[idx], device=device),
                             torch.as_tensor(anchor[idx], device=device)).cpu().numpy().astype(np.float64))
    return np.concatenate(out)
