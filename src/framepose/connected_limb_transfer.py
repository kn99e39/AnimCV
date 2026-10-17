"""DIAGNOSTIC (Worklog 86): two-segment connected-limb contract for chains beyond the left arm.

A chain is (proximal P, middle M, distal D) in the FrameBank 17-joint order.  Everything here restates the
left-arm definitions of Worklogs 69 / 80 with the three joint indices as a parameter, so the left arm
reproduces the historical arrays exactly (tested against the originals):
  * target: P = 0, M = (GT_M - GT_P) / L, D = (GT_D - GT_P) / L, L = |M - P| + |D - M| (canonical camera
    axes +X right, +Y depth, +Z up); valid iff target_valid (GT valid AND detector-visible) on P, M, D;
  * 2D geometry input: the Worklog 69 pair features (P-M, M-D, P-D; 13 values each) from the H0 crop-normalized
    detector geometry tensor;
  * visual input: the Worklog 68 frozen Qwen token grid read bilinearly at the DETECTOR's P, M, D positions
    (token centres at 32 * (i + 0.5) px of the 448 px crop; invalid detector joint -> zero vector), plus the
    stored global mean.  The grid is the cached fp16 merger output, so joint readouts are taken from
    fp16-rounded tokens (the historical left-arm readouts were taken from float32 tokens, then rounded).
The representation makes no bend-direction assumption (rho u, (1 - rho) v), so an elbow and a knee use it
unchanged; no foot, contact, pelvis or root quantity is represented.
"""

from __future__ import annotations

import math

import numpy as np

from framepose.contract import JOINT_INDEX

CHAINS = {
    "left_arm": ("left_shoulder", "left_elbow", "left_wrist"),
    "right_arm": ("right_shoulder", "right_elbow", "right_wrist"),
    "left_leg": ("left_hip", "left_knee", "left_ankle"),
    "right_leg": ("right_hip", "right_knee", "right_ankle"),
}
PAIRS = ((0, 1), (1, 2), (0, 2))  # proximal-middle, middle-distal, proximal-distal (Worklog 69 order)
GEOMETRY_FEATURES = 13
LENGTH_EPS = 1e-8
RESOLUTION, TOKEN_PIXELS = 448, 32
GRID = RESOLUTION // TOKEN_PIXELS


def chain_indices(chain: str) -> list[int]:
    return [JOINT_INDEX[j] for j in CHAINS[chain]]


def chain_targets(target_3d: np.ndarray, target_valid: np.ndarray, chain: str):
    """Normalized (M, D) relative to P in units of the GT total chain length, and the validity mask."""
    p = np.asarray(target_3d, np.float64)
    i = chain_indices(chain)
    s, e, w = p[:, i[0]], p[:, i[1]], p[:, i[2]]
    valid = np.asarray(target_valid, bool)[:, i].all(1)
    finite = np.isfinite(np.stack([s, e, w], 1)).all((1, 2))
    lu, ll = np.linalg.norm(e - s, axis=1), np.linalg.norm(w - e, axis=1)
    ok = valid & finite & (lu > LENGTH_EPS) & (ll > LENGTH_EPS)
    total = np.where(ok, lu + ll, 1.0)[:, None]
    return np.where(ok[:, None], (e - s) / total, 0.0), np.where(ok[:, None], (w - s) / total, 0.0), ok


def chain_pair_geometry(geometry: np.ndarray, chain: str) -> np.ndarray:
    """(N, 3, 13) Worklog 69 pair features of the chain from the (N, 17, 4) crop-normalized detector geometry."""
    g = np.asarray(geometry, dtype=np.float32)
    arm = g[:, chain_indices(chain)]
    out = np.zeros((g.shape[0], 3, GEOMETRY_FEATURES), np.float32)
    for k, (a, b) in enumerate(PAIRS):
        A, B = arm[:, a], arm[:, b]
        d = B[:, :2] - A[:, :2]
        length = np.linalg.norm(d, axis=1)
        safe = np.where(length > 1e-6, length, 1.0)
        both = (A[:, 3] > 0) & (B[:, 3] > 0)
        cos = np.where(both & (length > 1e-6), d[:, 0] / safe, 0.0)
        sin = np.where(both & (length > 1e-6), d[:, 1] / safe, 0.0)
        out[:, k] = np.concatenate([A, B, np.where(both[:, None], d, 0.0),
                                    np.where(both, length, 0.0)[:, None], cos[:, None], sin[:, None]], axis=1)
    return out


def token_coordinates(input_2d: np.ndarray, image_size, box_xy_side) -> np.ndarray:
    """Detector joints -> continuous token-grid (gx, gy); token centres at integers (Worklog 68 contract)."""
    width, height = image_size
    bx, by, side = box_xy_side
    px = np.asarray(input_2d, float)[:, 0] * width
    py = np.asarray(input_2d, float)[:, 1] * height
    u = (px - bx) / side * RESOLUTION
    v = (py - by) / side * RESOLUTION
    return np.stack([u / TOKEN_PIXELS - 0.5, v / TOKEN_PIXELS - 0.5], axis=-1)


def bilinear_taps(gx: float, gy: float):
    """The four (row, col, weight) taps of the Worklog 68 clamped bilinear sample."""
    gx = float(np.clip(gx, 0.0, GRID - 1.0))
    gy = float(np.clip(gy, 0.0, GRID - 1.0))
    x0, y0 = int(math.floor(gx)), int(math.floor(gy))
    x1, y1 = min(x0 + 1, GRID - 1), min(y0 + 1, GRID - 1)
    ax, ay = gx - x0, gy - y0
    return ((y0, x0, (1 - ax) * (1 - ay)), (y0, x1, ax * (1 - ay)), (y1, x0, (1 - ax) * ay), (y1, x1, ax * ay))


def chain_readout(token_rows, input_2d, input_valid, image_size, box_xy_side, chain: str, global_mean):
    """(4, C) = [P, M, D, stored global mean] from a token-row accessor ``token_rows(list_of_flat_indices)``."""
    coords = token_coordinates(input_2d, image_size, box_xy_side)
    out = np.zeros((4, len(global_mean)), np.float32)
    for k, index in enumerate(chain_indices(chain)):
        if bool(input_valid[index]):
            taps = bilinear_taps(*coords[index])
            rows = token_rows([r * GRID + c for r, c, _ in taps]).astype(np.float32)
            out[k] = sum(w * rows[j] for j, (_, _, w) in enumerate(taps))
    out[3] = np.asarray(global_mean, np.float32)
    return out
