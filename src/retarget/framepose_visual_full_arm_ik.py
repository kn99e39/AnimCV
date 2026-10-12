"""EVALUATION ONLY (Worklog 76): the connected left-arm Worklog 61 IK pose for visual chain directions.

F0/F1/F2 are the Worklog 74 candidates (H0 reach, H0 observed bend side
re-perpendicularized to the new direction with the Worklog 62 ``_runtime_side``,
the Worklog 61 closed form ``solve_two_bone``) extended to the full pose: elbow, wrist, upper-arm and forearm
directions, Worklog 61 zero-twist local rotations (``local_rotations_for_directions``)
and FK.  The GT reference is the Worklog 62 oracle arm (GT direction, GT
reach, GT side) on the same BaseRig root and rest lengths, built through the
same function.  Only the chain endpoint direction differs between F0/F1/F2;
upper/forearm forward fractions are never used and no skeleton is stitched.
Not imported by production code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from common.types import Quaternion, Vec3
from retarget.framepose_ik_gt_attribution import ChainMeasure, _runtime_side
from retarget.framepose_target_rest import TargetRigRestPose
from retarget.framepose_two_bone_ik import (
    DEGENERATE_BEND_SINE, TargetChainGeometry, chain_forward_kinematics, local_rotations_for_directions,
    quaternion_angle_degrees, solve_two_bone,
)

CANDIDATES = ("f0_h0_full_ik", "f1_e0_full_ik", "f2_e1_full_ik")


def _v(a) -> np.ndarray:
    return np.asarray(a, dtype=np.float64)


def _angle(a, b) -> float:
    a, b = _v(a), _v(b)
    c = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


@dataclass(frozen=True)
class FullArmPose:
    name: str
    status: str
    reach_fraction: float | None = None
    direction: Vec3 | None = None
    side: Vec3 | None = None
    side_status: str | None = None
    clamp: str | None = None
    root: Vec3 | None = None
    mid: Vec3 | None = None
    end: Vec3 | None = None
    proximal_local: Quaternion | None = None
    distal_local: Quaternion | None = None
    integrity: dict = field(default_factory=dict)

    @property
    def upper(self) -> np.ndarray:
        return _v(self.mid) - _v(self.root)

    @property
    def forearm(self) -> np.ndarray:
        return _v(self.end) - _v(self.mid)

    def bend_angle_degrees(self) -> float:
        return _angle(self.upper, self.forearm)

    def plane_normal(self):
        n = np.cross(self.upper / np.linalg.norm(self.upper), self.forearm / np.linalg.norm(self.forearm))
        s = float(np.linalg.norm(n))
        return n / s if s >= DEGENERATE_BEND_SINE else None


def full_arm(name: str, chain: ChainMeasure, direction: Vec3, target: TargetChainGeometry,
             rest: TargetRigRestPose) -> FullArmPose:
    """Worklog 61 IK with ``chain``'s reach and bend side, endpoint direction ``direction``.

    Step for step the Worklog 74 ``solve_with_direction`` (proven equal by Gate 1 and the tests).
    """
    norm = math.sqrt(sum(v * v for v in direction))
    if not math.isfinite(norm) or norm < 1e-9:
        return FullArmPose(name, "unavailable:degenerate_direction")
    direction = tuple(v / norm for v in direction)
    side, side_status = _runtime_side(chain, target, rest, direction)
    if side is None:
        return FullArmPose(name, "unavailable:bend_side", chain.reach_fraction, direction, None, side_status)
    root = target.root_head
    solve = solve_two_bone(root, direction, chain.reach_fraction * target.max_reach,
                           target.proximal_length, target.distal_length, side)
    upper = tuple(m - r for m, r in zip(solve.mid, root))
    lower = tuple(e - m for e, m in zip(solve.end, solve.mid))
    q1, q2 = local_rotations_for_directions(rest, target.proximal, target.distal, upper, lower)
    fk_mid, fk_end = chain_forward_kinematics(rest, target, q1, q2)
    low, high = abs(target.proximal_length - target.distal_length), target.proximal_length + target.distal_length
    distance = min(max(chain.reach_fraction * target.max_reach, low), high)
    expected_end = _v(root) + distance * _v(direction)
    integrity = {
        "upper_length_residual": abs(float(np.linalg.norm(upper)) - target.proximal_length),
        "forearm_length_residual": abs(float(np.linalg.norm(lower)) - target.distal_length),
        "endpoint_reconstruction_residual": float(np.linalg.norm(_v(solve.end) - expected_end)),
        "ik_to_fk_residual": max(math.dist(fk_mid, solve.mid), math.dist(fk_end, solve.end)),
        "quaternion_norm_error": max(abs(float(np.linalg.norm(q1)) - 1.0), abs(float(np.linalg.norm(q2)) - 1.0)),
        "finite": bool(np.isfinite([*q1, *q2, *solve.mid, *solve.end]).all()),
    }
    return FullArmPose(name, "known", chain.reach_fraction, direction, side, side_status,
                       solve.clamp_status, tuple(root), solve.mid, solve.end, q1, q2, integrity)


def gt_reference(gt: ChainMeasure, target: TargetChainGeometry, rest: TargetRigRestPose) -> FullArmPose:
    """Worklog 62 oracle arm: GT direction, GT reach, GT side (evaluation only)."""
    return full_arm("gt_reference", gt, gt.direction, target, rest)


def compare(c: FullArmPose, ref: FullArmPose, e_gt: Vec3) -> dict:
    """Pose errors of a candidate against the SAME GT-anchored reference (armature units = cm).

    The wrist is scored against the Worklog 62 target E_GT (= root + r_GT L d_GT, the
    Worklog 74/75 endpoint contract); the elbow against the Worklog 62 oracle elbow.
    """
    if c.status != "known" or ref.status != "known":
        return {"status": c.status}
    nc, nr = c.plane_normal(), ref.plane_normal()
    side_cos = float(np.dot(_v(c.side), _v(ref.side))) if c.side is not None and ref.side is not None else None
    return {
        "status": "known", "clamp": c.clamp, "side_status": c.side_status,
        "wrist_error": math.dist(c.end, e_gt),
        "elbow_error": math.dist(c.mid, ref.mid),
        "upper_direction_error_degrees": _angle(c.upper, ref.upper),
        "forearm_direction_error_degrees": _angle(c.forearm, ref.forearm),
        "chain_direction_error_degrees": _angle(c.direction, ref.direction),
        "bend_angle_error_degrees": c.bend_angle_degrees() - ref.bend_angle_degrees(),
        "bend_plane_error_degrees": _angle(nc, nr) if nc is not None and nr is not None else None,
        "side_agrees_with_gt": side_cos > 0 if side_cos is not None else None,
        "proximal_rotation_vs_gt_degrees": quaternion_angle_degrees(c.proximal_local, ref.proximal_local),
        "distal_rotation_vs_gt_degrees": quaternion_angle_degrees(c.distal_local, ref.distal_local),
        **{f"integrity_{k}": v for k, v in c.integrity.items()},
    }


TEMPORAL_METRICS = ("proximal_rotation_change_degrees", "distal_rotation_change_degrees",
                    "wrist_displacement", "elbow_displacement")
# Attribution only (not a decision metric): the one input that differs between F0/F1/F2.
ATTRIBUTION_TEMPORAL_METRICS = ("chain_direction_change_degrees", "upper_direction_change_degrees",
                                "forearm_direction_change_degrees", "bend_side_change_degrees")


def adjacent_change(a: FullArmPose, b: FullArmPose) -> dict:
    """Frame-to-frame change of one arm (Worklog 61 sign-invariant quaternion distance, cm)."""
    return {"proximal_rotation_change_degrees": quaternion_angle_degrees(a.proximal_local, b.proximal_local),
            "distal_rotation_change_degrees": quaternion_angle_degrees(a.distal_local, b.distal_local),
            "wrist_displacement": math.dist(a.end, b.end), "elbow_displacement": math.dist(a.mid, b.mid),
            "chain_direction_change_degrees": _angle(a.direction, b.direction),
            "upper_direction_change_degrees": _angle(_v(a.mid) - _v(a.root), _v(b.mid) - _v(b.root)),
            "forearm_direction_change_degrees": _angle(_v(a.end) - _v(a.mid), _v(b.end) - _v(b.mid)),
            "bend_side_change_degrees": _angle(a.side, b.side)}


def classify_change(candidate: float, reference: float, threshold: float) -> str:
    """Separate true rapid motion (the GT reference itself exceeds the threshold) from
    candidate-specific discontinuities (only the candidate exceeds it)."""
    if reference > threshold:
        return "rapid_motion_candidate_jump" if candidate > threshold else "rapid_motion_candidate_steady"
    return "candidate_discontinuity" if candidate > threshold else "quiet"
