"""DIAGNOSTIC (Worklog 78): a single learned reach / bend-side owner conditioned on the frozen E1 direction.

The Worklog 73 E1 chain forward fraction is frozen; its canonical camera-frame
chain direction d is rebuilt with the Worklog 64-73 ``reconstruct_direction``
from the detector-observed chain plane.  ONE network predicts, from the same
runtime observations, the two remaining Worklog 61 IK inputs:

* reach fraction  r = sigmoid(z_r) in (0, 1)  (the Worklog 61 definition
  |S - W| / (|S - E| + |E - W|); ``solve_two_bone`` applies its own clamp);
* bend-side evidence  s = a free camera-frame 3-vector; only its component
  perpendicular to d is used, s_perp = unit(s - (s.d) d), which is exactly the
  Worklog 62 ``_runtime_side`` rule applied downstream (unavailable when the
  perpendicular part vanishes - no fallback).

Inputs: the Worklog 68/69 pair geometry (3 x 13 detector features in the person
crop), the frozen E1 f and d (4), and the frozen Qwen3-VL readouts (4 tokens x
4096) through a shared 64-d projection.  G0 feeds exact zeros for every visual
feature, G1 the cache; nothing else differs.  No direction, segment depth, bone
length, scale, root orientation, temporal state or H0 quantity is predicted or
consumed.  Not imported by any production module.
"""

from __future__ import annotations

import math

import numpy as np

from framepose.arm_depth_probe import observed_plane, reconstruct_direction
from framepose.contract import JOINT_INDEX

CANDIDATES = ("G0_GEOMETRY_ONLY", "G1_GEOMETRY_PLUS_VISION")
PAIR_FEATURES = 13 * 3
E1_FEATURES = 4                    # f_E1 and the camera-frame unit direction d_E1
VISUAL_PROJECTION = 64
HIDDEN = 128
# Worklog 61 DEGENERATE_BEND_SINE (sin 1 deg), restated so this module does not import retarget code.
DEGENERATE_BEND_SINE = math.sin(math.radians(1.0))
UNIT_EPS = 1e-12                   # Worklog 62 ``_unit`` threshold for the re-perpendicularized side
LENGTH_EPS = 1e-8                  # Worklog 61 ``_EPS`` for segment lengths
ARCHITECTURE = {
    "reach": "r = sigmoid(z_r), dimensionless Worklog 61 reach fraction; Worklog 61 clamp applied by solve_two_bone",
    "bend_side": "free camera-frame 3-vector s; used as unit(s - (s.d_E1) d_E1); downstream the Worklog 62 _runtime_side "
                 "re-perpendicularization in rig space (identical: rotation preserves the projection); unavailable if it vanishes",
    "ik_reconstruction": "Worklog 76 full_arm with direction = E1 rig direction (A*Rz(-yaw_H0)), reach = r, side = "
                         "heading_relative(s, yaw_H0, A); BaseRig root and rest lengths unchanged",
    "network": "shared Linear(4096->64) per visual token (S, E, W, global) -> 256; concat with 39 pair-geometry + 4 E1 "
               "features -> Linear(299->128) GELU Linear(128->128) GELU -> reach Linear(128->1) + side Linear(128->3)",
    "loss": "masked mean |r - r_GT| (reach-valid rows) + masked mean (1 - cos(s_perp, s_GT_perp)) (side-valid rows); weights 1 and 1",
    "reach_target": "GT 3D (bank target_3d, canonical camera frame) via the Worklog 61 source_chain_geometry formula",
    "side_target": "Worklog 61 observed_bend_side of the GT arm (valid only when GT bend sine >= sin 1 deg), then "
                   "re-perpendicularized to d_E1 with the Worklog 62 rule (valid when the result is non-degenerate)",
    "degeneracy": "rows whose GT arm is near-straight (bend sine < sin 1 deg) train reach only; no inference fallback",
    "G0_vs_G1": "identical graph, seed, initialization, samples, optimizer and selection; G0 visual tensors are exact zeros",
}


def candidate_visual(readouts: np.ndarray, candidate: str) -> np.ndarray:
    r = np.asarray(readouts)
    if candidate == "G0_GEOMETRY_ONLY":
        return np.zeros_like(r, dtype=np.float16)
    if candidate == "G1_GEOMETRY_PLUS_VISION":
        return r
    raise ValueError(f"unknown candidate {candidate!r}")


def e1_direction(f_e1: np.ndarray, input_2d: np.ndarray, image_sizes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Camera-frame E1 chain direction from the frozen f and the observed chain plane; (N, 3), ok (N,)."""
    planes, oks = observed_plane(input_2d, image_sizes)
    ok = np.asarray(oks["chain"], bool)
    d = np.zeros((len(ok), 3))
    if ok.any():
        d[ok] = reconstruct_direction(np.asarray(f_e1)[ok], planes["chain"][ok])
    return d, ok


def _unit_rows(v: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray]:
    n = np.linalg.norm(v, axis=-1)
    ok = np.isfinite(n) & (n > eps)
    return np.where(ok[:, None], v / np.where(ok, n, 1.0)[:, None], 0.0), ok


def chain_targets(target_3d: np.ndarray, target_valid: np.ndarray, direction: np.ndarray, direction_ok: np.ndarray):
    """GT reach fraction and GT bend side re-perpendicularized to ``direction`` (training labels only)."""
    p = np.asarray(target_3d, np.float64)
    s, e, w = (p[:, JOINT_INDEX[j]] for j in ("left_shoulder", "left_elbow", "left_wrist"))
    valid = np.asarray(target_valid, bool)[:, [JOINT_INDEX[j] for j in ("left_shoulder", "left_elbow", "left_wrist")]].all(1)
    upper, lower = e - s, w - e
    lu, ll = np.linalg.norm(upper, axis=1), np.linalg.norm(lower, axis=1)
    finite = np.isfinite(p[:, [JOINT_INDEX["left_shoulder"], JOINT_INDEX["left_elbow"], JOINT_INDEX["left_wrist"]]]).all((1, 2))
    seg_ok = valid & finite & (lu > LENGTH_EPS) & (ll > LENGTH_EPS)
    chain = upper + lower
    reach = np.where(seg_ok, np.minimum(1.0, np.linalg.norm(chain, axis=1) / np.where(seg_ok, lu + ll, 1.0)), 0.0)
    e_dir, chain_ok = _unit_rows(chain, LENGTH_EPS)
    reach_ok = seg_ok & chain_ok
    uu, _ = _unit_rows(upper, LENGTH_EPS)
    ul, _ = _unit_rows(lower, LENGTH_EPS)
    sine = np.linalg.norm(np.cross(uu, ul), axis=1)
    side_gt, side_gt_ok = _unit_rows(upper - (upper * e_dir).sum(1, keepdims=True) * e_dir, LENGTH_EPS)
    d = np.asarray(direction, np.float64)
    side, side_ok = _unit_rows(side_gt - (side_gt * d).sum(1, keepdims=True) * d, UNIT_EPS)
    side_ok = side_ok & side_gt_ok & reach_ok & (sine >= DEGENERATE_BEND_SINE) & np.asarray(direction_ok, bool)
    return reach, reach_ok, np.where(side_ok[:, None], side, 0.0), side_ok


def geometry_inputs(pair_geometry: np.ndarray, f_e1: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """(N, 43) runtime inputs: flattened pair geometry, frozen f_E1, camera-frame d_E1."""
    g = np.asarray(pair_geometry, np.float32).reshape(len(pair_geometry), PAIR_FEATURES)
    return np.concatenate([g, np.asarray(f_e1, np.float32)[:, None], np.asarray(direction, np.float32)], axis=1)


def build_owner(channels: int):
    import torch
    from torch import nn

    class ArmGeometryOwner(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.visual = nn.Linear(channels, VISUAL_PROJECTION)
            self.trunk = nn.Sequential(nn.Linear(PAIR_FEATURES + E1_FEATURES + 4 * VISUAL_PROJECTION, HIDDEN), nn.GELU(),
                                       nn.Linear(HIDDEN, HIDDEN), nn.GELU())
            self.reach = nn.Linear(HIDDEN, 1)
            self.side = nn.Linear(HIDDEN, 3)

        def forward(self, visual, geometry):
            """visual (B, 4, C) [S, E, W, global]; geometry (B, 43) -> reach (B,), side (B, 3)."""
            v = self.visual(visual.float()).flatten(1)
            h = self.trunk(torch.cat([geometry.float(), v], dim=-1))
            return torch.sigmoid(self.reach(h))[:, 0], self.side(h)

    return ArmGeometryOwner()


def perpendicular_side(side, direction, eps: float = 1e-6):
    """Torch: unit(s - (s.d) d) with a numerical floor (training only; inference uses the Worklog 62 rule)."""
    p = side - (side * direction).sum(-1, keepdim=True) * direction
    return p / p.norm(dim=-1, keepdim=True).clamp_min(eps)


def owner_loss(reach, side, direction, reach_t, reach_m, side_t, side_m):
    """Declared objective: masked reach MAE + masked (1 - cosine) of the perpendicular side; weights 1 and 1."""
    reach_term = ((reach - reach_t).abs() * reach_m).sum() / reach_m.sum().clamp_min(1.0)
    cos = (perpendicular_side(side, direction) * side_t).sum(-1)
    side_term = ((1.0 - cos) * side_m).sum() / side_m.sum().clamp_min(1.0)
    return reach_term + side_term, reach_term, side_term


def predict_owner(model, visual: np.ndarray, geometry: np.ndarray, positions: np.ndarray, device) -> tuple[np.ndarray, np.ndarray]:
    """Inference from runtime inputs only: (reach (N,), raw camera-frame side (N, 3))."""
    import torch

    reach, side = [], []
    model.eval()
    with torch.no_grad():
        for s in range(0, len(positions), 2048):
            idx = np.asarray(positions[s:s + 2048])
            r, v = model(torch.as_tensor(np.asarray(visual[idx]), device=device), torch.as_tensor(geometry[idx], device=device))
            reach.append(r.cpu().numpy().astype(np.float64))
            side.append(v.cpu().numpy().astype(np.float64))
    return np.concatenate(reach), np.concatenate(side)
