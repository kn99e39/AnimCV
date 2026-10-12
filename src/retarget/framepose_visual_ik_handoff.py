"""EVALUATION ONLY (Worklog 74): hand a visual shoulder->wrist direction to the frozen Worklog 61 IK.

Candidates (left arm, one frame):
  P0_H0_IK               the Worklog 61 runtime chain exactly (H0 direction, H0 reach, H0 side)
  P1_E0_VISUAL_DIRECTION H0 reach + H0 side, direction rebuilt from Worklog 73 E0 chain f
  P2_E1_VISUAL_DIRECTION same, from Worklog 73 E1 chain f
  P3_GT_FORWARD_FRACTION same, from GT chain f (evaluation-only oracle)
All rebuilt directions use the Worklog 64-73 contract
``reconstruct_direction(f, observed_plane(detector 2D))`` in the canonical
camera frame, then the SAME runtime transform as Worklog 61/62,
``A * Rz(-yaw_H0)`` (``heading_relative``).  Only the endpoint direction
changes: the reach fraction is H0's, and H0's observed bend side is
re-perpendicularized to the new direction with the Worklog 62 convention
(``_runtime_side``); a side that becomes undefined stays UNAVAILABLE.  The
connected joints are built only by the Worklog 61 solver (``solve_two_bone``);
upper/forearm forward fractions are never stitched into a skeleton.

Nothing here is imported by production FramePose, FK, IK, AnimationSemantics
or the Blender writer.  GT enters only through P3 and the evaluation target.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from common.types import Quaternion, Vec3
from framepose.arm_depth_probe import reconstruct_direction
from retarget.framepose_ik_gt_attribution import (
    ChainMeasure, _ik, _runtime_side, dimensionless_endpoint, heading_relative,
)
from retarget.framepose_target_rest import TargetRigRestPose
from retarget.framepose_two_bone_ik import (
    TargetChainGeometry, chain_forward_kinematics, local_rotations_for_directions,
)

CANDIDATES = ("p0_h0_ik", "p1_e0_visual_direction", "p2_e1_visual_direction", "p3_gt_forward_fraction")
# Attribution controls (evaluation only, not runtime candidates).
CONTROLS = ("a1_h0_f_detector_plane",)
FK_CONTRACT_TOLERANCE = 1e-9


def _angle(a: Vec3, b: Vec3) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    c = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def camera_chain_direction(f: float, plane_xz: tuple[float, float]) -> Vec3:
    """Canonical camera-frame unit direction from a forward fraction and an image-plane direction."""
    d = reconstruct_direction(np.array([f], float), np.array([plane_xz], float))[0]
    return (float(d[0]), float(d[1]), float(d[2]))


def rig_direction(camera_direction: Vec3, yaw: float, alignment: Quaternion) -> Vec3:
    """The runtime transform A * Rz(-yaw_H0), identical to Worklog 61/62."""
    return heading_relative(camera_direction, yaw, alignment)


@dataclass(frozen=True)
class CandidateSolve:
    name: str
    status: str               # "known" or "unavailable:<reason>"
    direction: Vec3 | None
    reach_fraction: float | None
    requested_endpoint: Vec3 | None
    solved_endpoint: Vec3 | None
    solved_mid: Vec3 | None
    clamp: str | None
    side: Vec3 | None
    side_status: str | None
    fk_contract_error: float | None  # max |FK(local rotations) - solved| over mid/end

    def to_dict(self) -> dict[str, Any]:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}


def unavailable(name: str, reason: str) -> CandidateSolve:
    return CandidateSolve(name, f"unavailable:{reason}", None, None, None, None, None, None, None, None, None)


def solve_with_direction(name: str, h0: ChainMeasure, direction: Vec3, target: TargetChainGeometry,
                         rest: TargetRigRestPose) -> CandidateSolve:
    """Worklog 61 IK with H0 reach and H0 bend evidence, endpoint direction replaced by ``direction``."""
    norm = math.sqrt(sum(v * v for v in direction))
    if not math.isfinite(norm) or norm < 1e-9:
        return unavailable(name, "degenerate_direction")
    direction = tuple(v / norm for v in direction)
    side, side_status = _runtime_side(h0, target, rest, direction)
    requested = dimensionless_endpoint(target, h0.reach_fraction, direction)
    if side is None:
        return CandidateSolve(name, "unavailable:bend_side", direction, h0.reach_fraction, requested,
                              None, None, None, None, side_status, None)
    mid, end, clamp = _ik(target, h0.reach_fraction, direction, side)
    proximal_q, distal_q = local_rotations_for_directions(
        rest, target.proximal, target.distal,
        tuple(m - r for m, r in zip(mid, target.root_head)), tuple(e - m for e, m in zip(end, mid)))
    fk_mid, fk_end = chain_forward_kinematics(rest, target, proximal_q, distal_q)
    fk_error = max(math.dist(fk_mid, mid), math.dist(fk_end, end))
    return CandidateSolve(name, "known", direction, h0.reach_fraction, requested, end, mid, clamp, side,
                          side_status, fk_error)


def score(candidate: CandidateSolve, gt: ChainMeasure, e_gt: Vec3, oracle_mid: Vec3 | None,
          gt_side: Vec3 | None) -> dict[str, Any]:
    """Errors against the SAME Worklog 62 GT-anchored target (armature units = cm at scale 0.01)."""
    out: dict[str, Any] = {"status": candidate.status, "side_status": candidate.side_status,
                           "clamp": candidate.clamp}
    out["direction_error_degrees"] = _angle(candidate.direction, gt.direction) if candidate.direction else None
    out["requested_endpoint_error"] = (math.dist(candidate.requested_endpoint, e_gt)
                                       if candidate.requested_endpoint is not None else None)
    out["solved_endpoint_error"] = (math.dist(candidate.solved_endpoint, e_gt)
                                    if candidate.solved_endpoint is not None else None)
    out["elbow_error"] = (math.dist(candidate.solved_mid, oracle_mid)
                          if candidate.solved_mid is not None and oracle_mid is not None else None)
    if candidate.side is not None and gt_side is not None:
        c = sum(a * b for a, b in zip(candidate.side, gt_side))
        out["side_vs_gt_degrees"] = math.degrees(math.acos(max(-1.0, min(1.0, c))))
        out["side_agrees_with_gt"] = c > 0
    else:
        out["side_vs_gt_degrees"], out["side_agrees_with_gt"] = None, None
    out["fk_contract_error"] = candidate.fk_contract_error
    return out


def oracle_reference(gt: ChainMeasure, target: TargetChainGeometry, rest: TargetRigRestPose):
    """Worklog 62 oracle IK: GT direction + GT reach + GT side (the elbow reference)."""
    gt_side, gt_side_status = _runtime_side(gt, target, rest, gt.direction)
    e_gt = dimensionless_endpoint(target, gt.reach_fraction, gt.direction)
    if gt_side is None:
        return e_gt, None, None, gt_side_status
    mid, end, _ = _ik(target, gt.reach_fraction, gt.direction, gt_side)
    return e_gt, mid, gt_side, gt_side_status
