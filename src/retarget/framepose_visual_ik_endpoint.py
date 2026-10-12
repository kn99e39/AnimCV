"""EVALUATION ONLY (Worklog 75): analytic Worklog 61/62 endpoint error for chain-direction candidates.

The Worklog 61 IK places the solved wrist at
``target_root + clamp(reach_fraction * (L1 + L2)) * chain_direction`` and the
Worklog 62 reference at ``E_GT = target_root + GT_reach * (L1 + L2) * GT_direction``.
Candidate and reference pass through the SAME rigid runtime transform
(``A * Rz(-yaw_H0)``), so the endpoint distance can be computed in the
canonical camera frame:

    |clamp(r_c L) d_c - r_GT L d_GT|
      = L * sqrt((r_c - r_GT)^2 + 2 r_c r_GT (1 - d_c . d_GT))   (unclamped rows)

Reach fraction, chain direction and bend angle come from the Worklog 61
``source_chain_geometry``; the clamp interval and labels are those of
``solve_two_bone``.  Rebuilt visual directions use the Worklog 64-73
``reconstruct_direction(f, observed_plane(detector 2D))`` contract.  No elbow is
reconstructed here.  Not imported by production code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from framepose.arm_depth_probe import reconstruct_direction
from retarget.framepose_two_bone_ik import TargetChainGeometry, source_chain_geometry

CANDIDATES = ("t0_h0", "t1_e0_visual", "t2_e1_visual", "t3_gt_f_oracle", "t4_gt_direction_reference")


@dataclass(frozen=True)
class ChainGeometry:
    direction: tuple[float, float, float]
    reach_fraction: float
    bend_angle_degrees: float


def chain_geometry(shoulder, elbow, wrist) -> ChainGeometry | None:
    """Worklog 61 source geometry of one camera-frame chain (None when degenerate)."""
    s, e, w = (tuple(float(v) for v in p) for p in (shoulder, elbow, wrist))
    if not all(math.isfinite(v) for v in (*s, *e, *w)):
        return None
    geometry = source_chain_geometry(tuple(b - a for a, b in zip(s, e)), tuple(b - a for a, b in zip(e, w)))
    if geometry is None or geometry.endpoint_direction is None:
        return None
    return ChainGeometry(tuple(geometry.endpoint_direction), geometry.reach_fraction, geometry.bend_angle_degrees)


def visual_direction(f: float, plane_xz) -> tuple[float, float, float]:
    d = reconstruct_direction(np.array([f], float), np.array([plane_xz], float))[0]
    return (float(d[0]), float(d[1]), float(d[2]))


def clamp_distance(distance: float, target: TargetChainGeometry) -> tuple[float, str]:
    """Exactly ``solve_two_bone``'s reach clamp."""
    low, high = abs(target.proximal_length - target.distal_length), target.proximal_length + target.distal_length
    if distance > high:
        return high, "clamped_to_max_reach"
    if distance < low:
        return low, "clamped_to_min_reach"
    return distance, "none"


def endpoint_error(target: TargetChainGeometry, reach: float, direction, gt_reach: float, gt_direction):
    """(solved endpoint error, requested endpoint error, clamp status) in armature units (cm)."""
    total = target.max_reach
    d = np.asarray(direction, float) / np.linalg.norm(direction)
    g = np.asarray(gt_direction, float) / np.linalg.norm(gt_direction)
    e_gt = gt_reach * total * g
    solved_distance, clamp = clamp_distance(reach * total, target)
    return (float(np.linalg.norm(solved_distance * d - e_gt)), float(np.linalg.norm(reach * total * d - e_gt)),
            clamp)


def analytic_formula(total: float, reach: float, gt_reach: float, cosine: float) -> float:
    """The DIRECTION's closed form for unclamped rows (used to cross-check ``endpoint_error``)."""
    return total * math.sqrt(max(0.0, (reach - gt_reach) ** 2 + 2 * reach * gt_reach * (1 - cosine)))


def angle_degrees(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    c = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))
