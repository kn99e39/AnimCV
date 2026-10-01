"""EVALUATION ONLY: attribute Worklog 61 IK/FK endpoint error with 3DPW oracle geometry.

Nothing here is imported by production FK, IK, AnimationSemantics or
reliability code.  GT joints enter only as evaluation evidence.  The runtime
H0 heading transform ``A * Rz(-yaw_H0)`` is applied identically to H0 and GT
chains, so Root Orientation error is not a variable under test and GT yaw is
never used.

Every endpoint below is in heading-relative imported-armature space on the
same BaseRig chain (rest root, Blender rest L1/L2) as Worklog 61.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from common.types import Quaternion, Vec3
from retarget.axis_utils import quaternion_from_axis_angle, rotate_vector_by_quaternion
from retarget.framepose_target_rest import TargetRigRestPose
from retarget.framepose_two_bone_ik import (
    DEGENERATE_BEND_SINE, TargetChainGeometry, observed_bend_side, rest_plane_side,
    solve_two_bone, source_chain_geometry,
)

VARIANTS = ("fk_h0", "ik_h0", "h0_direction_gt_reach", "gt_direction_h0_reach",
            "oracle_ik", "fk_gt")


def _sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def _add(a, b):
    return tuple(x + y for x, y in zip(a, b))


def _scale(a, s):
    return tuple(x * s for x in a)


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _norm(a):
    return math.sqrt(_dot(a, a))


def _unit(a):
    n = _norm(a)
    return _scale(a, 1.0 / n) if n > 1e-12 else None


def _angle(a, b) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, _dot(_unit(a), _unit(b))))))


def heading_relative(vector: Vec3, yaw: float, alignment: Quaternion) -> Vec3:
    """Same operation as FK/IK: remove runtime yaw, then map through rig alignment."""
    unyaw = quaternion_from_axis_angle((0, 0, 1), -yaw)
    return rotate_vector_by_quaternion(rotate_vector_by_quaternion(vector, unyaw), alignment)


@dataclass(frozen=True)
class ChainMeasure:
    """Dimensionless chain geometry of one source (H0 or GT) after the shared transform."""

    upper: Vec3
    lower: Vec3
    upper_length: float
    lower_length: float
    ratio: float
    reach_fraction: float
    direction: Vec3
    bend_angle_degrees: float
    degenerate: bool
    side: Vec3 | None  # observed perpendicular elbow offset; None when degenerate

    def to_dict(self) -> dict[str, Any]:
        return {"upper_length": self.upper_length, "lower_length": self.lower_length,
                "ratio": self.ratio, "reach_fraction": self.reach_fraction,
                "direction": list(self.direction), "bend_angle_degrees": self.bend_angle_degrees,
                "degenerate": self.degenerate,
                "side": list(self.side) if self.side is not None else None}


def measure_chain(root: Vec3, mid: Vec3, end: Vec3, yaw: float,
                  alignment: Quaternion) -> ChainMeasure | None:
    """None for a degenerate segment or undefined root->end direction."""
    if not all(math.isfinite(v) for v in (*root, *mid, *end)):
        return None
    upper = heading_relative(_sub(mid, root), yaw, alignment)
    lower = heading_relative(_sub(end, mid), yaw, alignment)
    geometry = source_chain_geometry(upper, lower)
    if geometry is None or geometry.endpoint_direction is None:
        return None
    side = observed_bend_side(geometry)
    return ChainMeasure(upper, lower, _norm(upper), _norm(lower), _norm(upper) / _norm(lower),
                        geometry.reach_fraction, geometry.endpoint_direction,
                        geometry.bend_angle_degrees, geometry.bend_sine < DEGENERATE_BEND_SINE, side)


def dimensionless_endpoint(target: TargetChainGeometry, reach_fraction: float,
                           direction: Vec3) -> Vec3:
    """Worklog 61 contract: root + fraction * (L1 + L2) * direction."""
    return _add(target.root_head, _scale(direction, reach_fraction * target.max_reach))


def _perpendicular(vector: Vec3, axis: Vec3) -> Vec3 | None:
    return _unit(_sub(vector, _scale(axis, _dot(vector, axis))))


def _runtime_side(measure: ChainMeasure, target: TargetChainGeometry,
                  rest: TargetRigRestPose, direction: Vec3) -> tuple[Vec3 | None, str]:
    """Bend side exactly as the runtime would choose it, re-perpendicularized to ``direction``."""
    if measure.side is not None:
        side = _perpendicular(measure.side, direction)
        return side, "observed" if side is not None else "unavailable"
    side = rest_plane_side(target, rest, direction)
    return side, "degenerate_rest_plane_fallback" if side is not None else "unavailable"


def _ik(target, fraction, direction, side):
    solve = solve_two_bone(target.root_head, direction, fraction * target.max_reach,
                           target.proximal_length, target.distal_length, side)
    return solve.mid, solve.end, solve.clamp_status


def _fk(target, measure):
    mid = _add(target.root_head, _scale(_unit(measure.upper), target.proximal_length))
    return mid, _add(mid, _scale(_unit(measure.lower), target.distal_length))


@dataclass(frozen=True)
class AttributionRow:
    values: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return dict(self.values)


def attribute_frame(h0: ChainMeasure, gt: ChainMeasure, target: TargetChainGeometry,
                    rest: TargetRigRestPose, *, fk_h0_endpoint: Vec3 | None = None,
                    fk_h0_mid: Vec3 | None = None) -> AttributionRow:
    """All variants against the SAME E_GT, with endpoint and elbow reported separately.

    ``fk_h0_endpoint`` / ``fk_h0_mid`` may be supplied from the actual Worklog 60
    FK quaternions; otherwise the direction-FK chain is reconstructed.
    """
    e_gt = dimensionless_endpoint(target, gt.reach_fraction, gt.direction)
    gt_side, gt_side_status = _runtime_side(gt, target, rest, gt.direction)
    if fk_h0_endpoint is None or fk_h0_mid is None:
        fk_h0_mid, fk_h0_endpoint = _fk(target, h0)
    fk_gt_mid, fk_gt_end = _fk(target, gt)
    h0_side_own, h0_side_status = _runtime_side(h0, target, rest, h0.direction)
    endpoints, clamps, mids = {}, {}, {}
    endpoints["fk_h0"], mids["fk_h0"] = fk_h0_endpoint, fk_h0_mid
    endpoints["fk_gt"], mids["fk_gt"] = fk_gt_end, fk_gt_mid
    for name, fraction, direction in (
        ("ik_h0", h0.reach_fraction, h0.direction),
        ("h0_direction_gt_reach", gt.reach_fraction, h0.direction),
        ("gt_direction_h0_reach", h0.reach_fraction, gt.direction),
        ("oracle_ik", gt.reach_fraction, gt.direction),
    ):
        side = h0_side_own if name in ("ik_h0", "h0_direction_gt_reach") else gt_side
        if side is None:
            endpoints[name] = dimensionless_endpoint(target, fraction, direction)
            mids[name], clamps[name] = None, None
            continue
        if name == "h0_direction_gt_reach":
            side = _perpendicular(side, direction) or side
        mids[name], endpoints[name], clamps[name] = _ik(target, fraction, direction, side)

    # Bend attribution: same GT endpoint target, H0 bend evidence vs GT bend evidence.
    h0_side_on_gt, h0_on_gt_status = _runtime_side(h0, target, rest, gt.direction)
    elbow_h0_side = (_ik(target, gt.reach_fraction, gt.direction, h0_side_on_gt)[0]
                     if h0_side_on_gt is not None else None)
    oracle_elbow = mids["oracle_ik"]
    side_cosine = (_dot(h0_side_on_gt, gt_side)
                   if h0_side_on_gt is not None and gt_side is not None else None)

    values: dict[str, Any] = {
        "gt_ratio": gt.ratio, "h0_ratio": h0.ratio,
        "ratio_log_error": math.log(h0.ratio / gt.ratio),
        "gt_reach_fraction": gt.reach_fraction, "h0_reach_fraction": h0.reach_fraction,
        "reach_fraction_error": h0.reach_fraction - gt.reach_fraction,
        "endpoint_direction_error_degrees": _angle(h0.direction, gt.direction),
        "gt_bend_angle_degrees": gt.bend_angle_degrees,
        "h0_bend_angle_degrees": h0.bend_angle_degrees,
        "bend_angle_error_degrees": h0.bend_angle_degrees - gt.bend_angle_degrees,
        "gt_degenerate": gt.degenerate, "h0_degenerate": h0.degenerate,
        "gt_side_status": gt_side_status, "h0_side_status": h0_side_status,
        "h0_side_on_gt_target_status": h0_on_gt_status,
        "bend_plane_error_degrees": (
            math.degrees(math.acos(max(-1.0, min(1.0, side_cosine))))
            if side_cosine is not None else None),
        "bend_side_agrees": side_cosine > 0 if side_cosine is not None else None,
        "raw_side_angle_degrees": (_angle(h0.side, gt.side)
                                   if h0.side is not None and gt.side is not None else None),
        "elbow_error_h0_side_on_gt_target": (math.dist(elbow_h0_side, oracle_elbow)
                                              if elbow_h0_side is not None and oracle_elbow is not None
                                              else None),
        "e_gt": list(e_gt),
    }
    for name in VARIANTS:
        values[f"{name}_endpoint_error"] = math.dist(endpoints[name], e_gt)
        values[f"{name}_elbow_error"] = (math.dist(mids[name], oracle_elbow)
                                         if mids.get(name) is not None and oracle_elbow is not None
                                         else None)
        values[f"{name}_clamp"] = clamps.get(name)
    values["fk_ik_h0_disagreement"] = math.dist(endpoints["fk_h0"], endpoints["ik_h0"])
    values["fk_ik_gt_disagreement"] = math.dist(endpoints["fk_gt"], endpoints["oracle_ik"])
    return AttributionRow(values)
