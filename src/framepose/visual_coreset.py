"""DIAGNOSTIC (Worklog 72): deterministic per-sequence visual coreset in frozen arm-readout space.

Feature: L2-normalized concatenation of the Worklog 68 shoulder, elbow and
wrist readouts (the global token is excluded; Worklog 71 showed it is nearly a
scene signature).  Per TRAIN sequence: start at the frame closest (cosine) to
the sequence feature centroid, then repeatedly add the frame maximizing its
minimum cosine distance to the selected set; ties -> smallest FrameBank
position.  K = minimum eligible TRAIN frames over sequences.  No GT, target,
H0, Depth Anything, error, threshold or randomness.  Not imported by production.
"""

from __future__ import annotations

import numpy as np


def arm_features(readout: np.ndarray) -> np.ndarray:
    """(N, 3*C) L2-normalized [shoulder, elbow, wrist] readouts from (N, 4, C)."""
    r = np.asarray(readout, dtype=np.float32)[:, :3]
    flat = r.reshape(r.shape[0], -1).astype(np.float64)
    norm = np.linalg.norm(flat, axis=1, keepdims=True)
    return flat / np.maximum(norm, 1e-12)


def coreset_size(groups: dict[str, np.ndarray]) -> int:
    return int(min(len(v) for v in groups.values()))


def farthest_point_coreset(features: np.ndarray, positions: np.ndarray, k: int) -> np.ndarray:
    """Deterministic farthest-point selection (cosine distance) of k positions."""
    positions = np.asarray(positions, dtype=np.int64)
    order = np.argsort(positions, kind="stable")
    positions = positions[order]
    x = np.asarray(features[positions], dtype=np.float64)
    if k >= len(positions):
        return positions.copy()
    centroid = x.mean(axis=0)
    centroid /= max(np.linalg.norm(centroid), 1e-12)
    start = int(np.argmin(1.0 - x @ centroid))  # argmin -> smallest position on ties
    selected = [start]
    min_dist = 1.0 - x @ x[start]
    min_dist[start] = -np.inf
    while len(selected) < k:
        nxt = int(np.argmax(min_dist))  # argmax -> smallest position on ties
        selected.append(nxt)
        min_dist = np.minimum(min_dist, 1.0 - x @ x[nxt])
        min_dist[selected] = -np.inf
    return positions[np.sort(np.asarray(selected))]


def redundancy(features: np.ndarray, positions: np.ndarray) -> dict[str, float]:
    """Mean pairwise cosine and mean nearest-neighbour cosine within a pool."""
    x = np.asarray(features[np.asarray(positions)], dtype=np.float64)
    s = x @ x.T
    n = len(x)
    off = s[~np.eye(n, dtype=bool)]
    np.fill_diagonal(s, -np.inf)
    return {"frames": n, "mean_pairwise_cosine": float(off.mean()),
            "mean_nearest_neighbour_cosine": float(s.max(axis=1).mean())}
