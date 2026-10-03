"""DIAGNOSTIC (Worklog 70): skeleton-geometry-biased attention over the frozen Qwen3-VL token grid.

Reads the unchanged Worklog 68 full grid (14x14x4096 per sample).  Per arm
segment a single-head attention query built from the 13 runtime pair-geometry
features attends over all 196 projected tokens; A1 adds a deterministic,
parameter-free logit bias of minus the token-centre distance (in token-grid
units) to the observed 2D detector segment, A0 adds exactly zero.  The bias is
derived from the same 2D geometry the model already receives, so A1 > A0 would
mean the geometry is useful as a spatial attention prior, not that new
geometric information was added.  Not an EGformer port: no equirectangular
coordinates, ERPE, panorama windows or cyclicity.  Not imported by production.
"""

from __future__ import annotations

import math

import numpy as np

# Worklog 68 constants, restated (not imported) so that the Worklog 68 module
# keeps having no importer under src/; a test pins the equality.
GRID = 14
RESOLUTION = 448
TOKEN_PIXELS = 32
GEOMETRY_FEATURES = 13
VISUAL_PROJECTION = 64
HIDDEN = 64
PAIRS = (("shoulder_elbow", 0, 1, "upper"), ("elbow_wrist", 1, 2, "lower"), ("shoulder_wrist", 0, 2, "chain"))
ARM_JOINT_INDEX = (11, 12, 13)  # H36M left_shoulder, left_elbow, left_wrist (pinned by test)
CANDIDATES = ("A0_UNBIASED_ATTENTION", "A1_SKELETON_BIASED_ATTENTION")

_GY, _GX = np.meshgrid(np.arange(GRID, dtype=np.float64), np.arange(GRID, dtype=np.float64), indexing="ij")
TOKEN_CENTRES = np.stack([_GX.ravel(), _GY.ravel()], axis=1)  # (196, 2) as (gx, gy), row-major


def joint_token_coordinates(input_2d: np.ndarray, image_size, box_xy_side) -> np.ndarray:
    """Detector joints -> continuous token-grid coordinates (token centres at integers).

    Identical mapping to Worklog 68 ``token_coordinates`` (not clamped here).
    """
    width, height = image_size
    x, y, side = box_xy_side
    p = np.asarray(input_2d, float)
    u = (p[:, 0] * width - x) / side * RESOLUTION
    v = (p[:, 1] * height - y) / side * RESOLUTION
    return np.stack([u / TOKEN_PIXELS - 0.5, v / TOKEN_PIXELS - 0.5], axis=-1)


def point_segment_distance(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Euclidean distance from each point to the closed segment a-b."""
    points = np.asarray(points, float)
    a, b = np.asarray(a, float), np.asarray(b, float)
    ab = b - a
    denom = float(ab @ ab)
    if denom <= 1e-12:
        return np.linalg.norm(points - a, axis=1)
    t = np.clip(((points - a) @ ab) / denom, 0.0, 1.0)
    return np.linalg.norm(points - (a + t[:, None] * ab), axis=1)


def skeleton_bias(input_2d: np.ndarray, input_valid: np.ndarray, image_size, box_xy_side) -> np.ndarray:
    """(3, 196) logit bias = -distance(token centre, observed 2D segment) per pair.

    A pair with an invalid endpoint is masked out of training/evaluation; its
    bias is set to 0 (never computed from a substitute point).
    """
    coords = joint_token_coordinates(input_2d, image_size, box_xy_side)
    out = np.zeros((len(PAIRS), GRID * GRID), np.float32)
    for k, (_, a, b, _) in enumerate(PAIRS):
        ja, jb = ARM_JOINT_INDEX[a], ARM_JOINT_INDEX[b]
        if bool(input_valid[ja]) and bool(input_valid[jb]):
            out[k] = -point_segment_distance(TOKEN_CENTRES, coords[ja], coords[jb])
    return out


def candidate_bias(bias: np.ndarray, candidate: str) -> np.ndarray:
    if candidate == "A0_UNBIASED_ATTENTION":
        return np.zeros_like(bias, dtype=np.float32)
    if candidate == "A1_SKELETON_BIASED_ATTENTION":
        return np.asarray(bias, dtype=np.float32)
    raise ValueError(f"unknown candidate {candidate!r}")


def build_attention_model(channels: int):
    import torch
    from torch import nn

    class SkeletonAttentionDepth(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.visual = nn.Linear(channels, VISUAL_PROJECTION)
            self.queries = nn.ModuleList(nn.Linear(GEOMETRY_FEATURES, VISUAL_PROJECTION) for _ in PAIRS)
            self.heads = nn.ModuleList(
                nn.Sequential(nn.Linear(GEOMETRY_FEATURES + 4 * VISUAL_PROJECTION, HIDDEN), nn.GELU(),
                              nn.Linear(HIDDEN, 1), nn.Tanh())
                for _ in PAIRS)

        def forward(self, grid, readout, geometry, bias, return_attention: bool = False):
            """grid (B,196,C), readout (B,4,C) [S,E,W,global], geometry (B,3,13), bias (B,3,196)."""
            tokens = self.visual(grid.float())          # (B,196,64): keys = values
            r = self.visual(readout.float())            # (B,4,64)
            out, attention = [], []
            for k, (_, a, b, _) in enumerate(PAIRS):
                q = self.queries[k](geometry[:, k])     # (B,64)
                score = torch.einsum("bd,bnd->bn", q, tokens) / math.sqrt(VISUAL_PROJECTION) + bias[:, k]
                weights = torch.softmax(score, dim=-1)
                context = torch.einsum("bn,bnd->bd", weights, tokens)
                x = torch.cat([geometry[:, k], r[:, a], r[:, b], r[:, 3], context], dim=-1)
                out.append(self.heads[k](x)[:, 0])
                attention.append(weights)
            f = torch.stack(out, dim=1)
            return (f, torch.stack(attention, dim=1)) if return_attention else f

    return SkeletonAttentionDepth()
