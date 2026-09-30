"""Target-length-aware FramePose two-bone IK, separate from Worklog 59/60 FK.

The source chain contributes only dimensionless geometry: the root->end
direction, the reach fraction |root-end| / (|root-mid| + |mid-end|), and the
side of the root->end line the mid joint lies on.  The target chain keeps its
Blender rest lengths L1/L2.  Root yaw is removed exactly as in FK and stays
owned by the Armature Object; no frame is a calibration frame and no previous
frame's pole or rotation is used.

This is not the historical MotionGraph ``retarget.ik_solver``: that solver is
2D, calibrates bone lengths on the first valid frame and holds rotations.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common.serialization import read_json, write_json
from common.types import Quaternion, Vec3
from framepose.contract import COORDINATE_FRAME
from motion.animation_semantics_v2 import AnimationSemanticsV2, ROOT_POLICY_V2
from retarget.axis_utils import (
    quaternion_conjugate, quaternion_from_axis_angle, quaternion_from_vectors,
    quaternion_multiply, rotate_vector_by_quaternion,
)
from retarget.framepose_fk import (
    RigFkCalibration, _q_unit, _rest_direction, _rest_rotation, derive_rig_alignment,
)
from retarget.framepose_target_rest import TargetRigRestPose

CHAINS_SCHEMA = "animcv_framepose_two_bone_ik_chains_v1"
SCHEMA = "animcv_framepose_two_bone_ik_v1"
POLICY = "dimensionless_reach_fraction_observed_bend_side_two_bone_ik_v1"
ENDPOINT_FORMULA = (
    "s_u = A*Rz(-yaw)*(mid-root); s_l = A*Rz(-yaw)*(end-mid); "
    "reach_fraction = |s_u+s_l| / (|s_u|+|s_l|); e = (s_u+s_l)/|s_u+s_l|; "
    "requested_endpoint = proximal_rest_head + reach_fraction*(L1+L2)*e"
)
# A fixed geometric definition of "straight or folded", declared before any
# replay and never fitted to data: the source upper/lower segment directions
# are within 1 degree of parallel or anti-parallel.
DEGENERATE_BEND_SINE = math.sin(math.radians(1.0))
# Relative tolerance used only to verify the rest snapshot's chain is connected.
CONNECTED_TOLERANCE = 1e-6
_EPS = 1e-8


def _sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _scale(a: Vec3, s: float) -> Vec3:
    return (a[0] * s, a[1] * s, a[2] * s)


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _norm(a: Vec3) -> float:
    return math.sqrt(_dot(a, a))


def _unit_or_none(a: Vec3) -> Vec3 | None:
    length = _norm(a)
    if not math.isfinite(length) or length <= _EPS:
        return None
    return _scale(a, 1.0 / length)


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def quaternion_angle_degrees(a: Quaternion, b: Quaternion) -> float:
    dot = abs(sum(x * y for x, y in zip(a, b)))
    return math.degrees(2.0 * math.acos(min(1.0, dot)))


# ---------------------------------------------------------------- contract


@dataclass(frozen=True)
class TwoBoneIkChain:
    """One explicit rig-specific chain; reused unchanged across videos."""

    name: str
    source_root: str
    source_mid: str
    source_end: str
    target_proximal: str
    target_distal: str
    target_end_anchor: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return vars(self).copy()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TwoBoneIkChain":
        return cls(**value)


@dataclass(frozen=True)
class TwoBoneIkChainSet:
    rig_id: str
    chains: tuple[TwoBoneIkChain, ...]

    def __post_init__(self) -> None:
        names = [chain.name for chain in self.chains]
        if not self.chains or len(set(names)) != len(names):
            raise ValueError("IK chain set needs unique chain names")

    def to_dict(self) -> dict[str, Any]:
        return {"schema": CHAINS_SCHEMA, "rig_id": self.rig_id,
                "chains": [chain.to_dict() for chain in self.chains]}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TwoBoneIkChainSet":
        if value.get("schema") != CHAINS_SCHEMA:
            raise ValueError("not a FramePose two-bone IK chain contract")
        return cls(value["rig_id"], tuple(TwoBoneIkChain.from_dict(c) for c in value["chains"]))


def load_ik_chain_set(path: str | Path) -> TwoBoneIkChainSet:
    return TwoBoneIkChainSet.from_dict(read_json(path))


@dataclass(frozen=True)
class TargetChainGeometry:
    """Rest facts of a target chain, read only from the Blender rest snapshot."""

    chain: str
    proximal: str
    distal: str
    root_head: Vec3
    proximal_length: float
    distal_length: float
    rest_bend_normal: Vec3 | None

    @property
    def max_reach(self) -> float:
        return self.proximal_length + self.distal_length

    @property
    def min_reach(self) -> float:
        return abs(self.proximal_length - self.distal_length)

    def to_dict(self) -> dict[str, Any]:
        return {"chain": self.chain, "proximal": self.proximal, "distal": self.distal,
                "root_head": list(self.root_head),
                "proximal_length": self.proximal_length,
                "distal_length": self.distal_length,
                "proximal_over_distal": self.proximal_length / self.distal_length,
                "min_reach": self.min_reach, "max_reach": self.max_reach,
                "rest_bend_normal": (list(self.rest_bend_normal)
                                     if self.rest_bend_normal is not None else None),
                "length_authority": "blender_imported_armature head/tail"}


def target_chain_geometry(rest: TargetRigRestPose, chain: TwoBoneIkChain) -> TargetChainGeometry:
    """Validate topology and extract L1/L2; nothing is taken from Assimp or video."""
    bones = rest.bones
    for name in (chain.target_proximal, chain.target_distal, chain.target_end_anchor):
        if name is not None and name not in bones:
            raise ValueError(f"{chain.name}: target bone missing: {name}")
    proximal, distal = bones[chain.target_proximal], bones[chain.target_distal]
    if distal.parent != proximal.name:
        raise ValueError(f"{chain.name}: distal bone is not a child of the proximal bone")
    l1 = math.dist(proximal.head, proximal.tail)
    l2 = math.dist(distal.head, distal.tail)
    if math.dist(proximal.tail, distal.head) > CONNECTED_TOLERANCE * max(l1, 1.0):
        raise ValueError(f"{chain.name}: proximal tail and distal head are not connected")
    if chain.target_end_anchor is not None:
        anchor = bones[chain.target_end_anchor]
        if anchor.parent != distal.name or math.dist(distal.tail, anchor.head) > (
                CONNECTED_TOLERANCE * max(l2, 1.0)):
            raise ValueError(f"{chain.name}: end anchor is not the distal tail child")
    normal = _unit_or_none(_cross(_rest_direction(rest, proximal.name),
                                  _rest_direction(rest, distal.name)))
    return TargetChainGeometry(chain.name, proximal.name, distal.name,
                               tuple(proximal.head), l1, l2, normal)


# ------------------------------------------------------------- geometry


@dataclass(frozen=True)
class SourceChainGeometry:
    """Heading-relative source chain mapped into imported armature rest axes."""

    upper: Vec3
    lower: Vec3
    reach_fraction: float
    endpoint_direction: Vec3 | None
    bend_sine: float
    bend_angle_degrees: float  # 0 = straight, 180 = folded


def source_chain_geometry(upper: Vec3, lower: Vec3) -> SourceChainGeometry | None:
    """Return dimensionless chain geometry, or None for a degenerate segment."""
    lu, ll = _norm(upper), _norm(lower)
    if not (math.isfinite(lu) and math.isfinite(ll)) or lu <= _EPS or ll <= _EPS:
        return None
    reach = _add(upper, lower)
    fraction = min(1.0, _norm(reach) / (lu + ll))
    uu, ul = _scale(upper, 1 / lu), _scale(lower, 1 / ll)
    cosine = max(-1.0, min(1.0, _dot(uu, ul)))
    return SourceChainGeometry(upper, lower, fraction, _unit_or_none(reach),
                               _norm(_cross(uu, ul)), math.degrees(math.acos(cosine)))


def observed_bend_side(source: SourceChainGeometry) -> Vec3 | None:
    """Unit offset of the source mid joint from the root->end line, or None."""
    if source.endpoint_direction is None or source.bend_sine < DEGENERATE_BEND_SINE:
        return None
    e = source.endpoint_direction
    return _unit_or_none(_sub(source.upper, _scale(e, _dot(source.upper, e))))


def rest_plane_side(target: TargetChainGeometry, rest: TargetRigRestPose,
                    endpoint_direction: Vec3) -> Vec3 | None:
    """Deterministic degeneracy fallback: the target rest bend plane.

    The mid joint is placed on the side ``sign * (e x n_rest)``, where the sign
    reproduces the rest elbow side for the rest endpoint direction.  This is a
    rig fact, not observed bend evidence.
    """
    if target.rest_bend_normal is None:
        return None
    rest_u = _rest_direction(rest, target.proximal)
    rest_e = _unit_or_none(_add(_scale(rest_u, target.proximal_length),
                                _scale(_rest_direction(rest, target.distal), target.distal_length)))
    side = _unit_or_none(_cross(endpoint_direction, target.rest_bend_normal))
    if rest_e is None or side is None:
        return None
    rest_side = _sub(rest_u, _scale(rest_e, _dot(rest_u, rest_e)))
    sign = 1.0 if _dot(_cross(rest_e, target.rest_bend_normal), rest_side) >= 0 else -1.0
    return _scale(side, sign)


@dataclass(frozen=True)
class TwoBoneSolve:
    requested_distance: float
    reachable_distance: float
    clamp_status: str  # none / clamped_to_max_reach / clamped_to_min_reach
    mid: Vec3
    end: Vec3


def solve_two_bone(root: Vec3, direction: Vec3, distance: float, l1: float, l2: float,
                   side: Vec3) -> TwoBoneSolve:
    """Closed-form two-bone solve in the plane spanned by ``direction`` and ``side``."""
    if not (math.isfinite(distance) and distance >= 0):
        raise ValueError("requested reach must be finite and non-negative")
    low, high = abs(l1 - l2), l1 + l2
    if distance > high:
        reach, clamp = high, "clamped_to_max_reach"
    elif distance < low:
        reach, clamp = low, "clamped_to_min_reach"
    else:
        reach, clamp = distance, "none"
    if reach <= _EPS:
        cosine = 1.0
    else:
        cosine = max(-1.0, min(1.0, (l1 * l1 + reach * reach - l2 * l2) / (2 * l1 * reach)))
    sine = math.sqrt(max(0.0, 1.0 - cosine * cosine))
    mid = _add(root, _add(_scale(direction, l1 * cosine), _scale(side, l1 * sine)))
    end = _add(root, _scale(direction, reach))
    return TwoBoneSolve(distance, reach, clamp, mid, end)


def local_rotations_for_directions(
    rest: TargetRigRestPose, proximal: str, distal: str,
    proximal_direction: Vec3, distal_direction: Vec3,
) -> tuple[Quaternion, Quaternion]:
    """Same shortest-arc, zero-twist, parent-relative convention as FramePose FK."""
    proximal_direction = _unit_or_none(proximal_direction)
    distal_direction = _unit_or_none(distal_direction)
    if proximal_direction is None or distal_direction is None:
        raise ValueError("IK bone direction is degenerate")
    delta_p = _q_unit(quaternion_from_vectors(_rest_direction(rest, proximal), proximal_direction))
    delta_d = _q_unit(quaternion_from_vectors(_rest_direction(rest, distal), distal_direction))
    rp, rd = _rest_rotation(rest, proximal), _rest_rotation(rest, distal)
    local_p = _q_unit(quaternion_multiply(quaternion_multiply(quaternion_conjugate(rp), delta_p), rp))
    relative = quaternion_multiply(quaternion_conjugate(delta_p), delta_d)
    local_d = _q_unit(quaternion_multiply(quaternion_multiply(quaternion_conjugate(rd), relative), rd))
    return local_p, local_d


def chain_forward_kinematics(rest: TargetRigRestPose, target: TargetChainGeometry,
                             local_p: Quaternion, local_d: Quaternion) -> tuple[Vec3, Vec3]:
    """Mid/end positions from local pose quaternions and Blender rest geometry.

    Unmapped ancestors of the proximal bone stay at rest, so the chain root is
    the proximal rest head in heading-relative armature space.
    """
    rp, rd = _rest_rotation(rest, target.proximal), _rest_rotation(rest, target.distal)
    delta_p = quaternion_multiply(quaternion_multiply(rp, local_p), quaternion_conjugate(rp))
    delta_d = quaternion_multiply(delta_p, quaternion_multiply(
        quaternion_multiply(rd, local_d), quaternion_conjugate(rd)))
    p, d = rest.bones[target.proximal], rest.bones[target.distal]
    mid = _add(target.root_head, rotate_vector_by_quaternion(_sub(p.tail, p.head), delta_p))
    end = _add(mid, rotate_vector_by_quaternion(_sub(d.tail, d.head), delta_d))
    return mid, end


# ---------------------------------------------------------------- output


def _vec(value):
    return list(value) if value is not None else None


def _tup(value):
    return tuple(value) if value is not None else None


@dataclass(frozen=True)
class TwoBoneIkSample:
    frame_index: int
    timestamp: float
    chain: str
    source_status: str  # known / unknown
    root_orientation_status: str  # known / unknown
    source_reach_fraction: float | None
    source_bend_angle_degrees: float | None
    bend_plane_status: str  # observed / degenerate_rest_plane_fallback / unavailable
    requested_endpoint: Vec3 | None
    reachable_endpoint: Vec3 | None
    solved_mid: Vec3 | None
    solved_endpoint: Vec3 | None
    endpoint_error: float | None
    clamp_status: str | None
    proximal_status: str
    proximal_local: Quaternion | None
    distal_status: str
    distal_local: Quaternion | None
    reason: str | None

    def __post_init__(self) -> None:
        if self.bend_plane_status not in ("observed", "degenerate_rest_plane_fallback", "unavailable"):
            raise ValueError("invalid bend-plane status")
        if self.proximal_status != self.distal_status or self.proximal_status not in ("known", "unavailable"):
            raise ValueError("two-bone IK bones are solved together or not at all")
        if self.proximal_status == "known":
            if (self.source_status != "known" or self.root_orientation_status != "known"
                    or self.reason is not None or self.bend_plane_status == "unavailable"
                    or self.clamp_status is None):
                raise ValueError("known IK requires known source, yaw and bend plane")
            for q in (self.proximal_local, self.distal_local):
                if q is None or abs(sum(v * v for v in q) - 1.0) > 1e-8:
                    raise ValueError("IK rotations must be unit quaternions")
            values = (*self.requested_endpoint, *self.solved_endpoint, self.endpoint_error)
            if not all(math.isfinite(v) for v in values):
                raise ValueError("IK output must be finite")
        elif (self.reason is None or self.proximal_local is not None or self.distal_local is not None
              or self.solved_endpoint is not None):
            raise ValueError("unavailable IK carries a reason and no pose")

    def to_dict(self) -> dict[str, Any]:
        return {"frame_index": self.frame_index, "timestamp": self.timestamp, "chain": self.chain,
                "source_status": self.source_status,
                "root_orientation_status": self.root_orientation_status,
                "source_reach_fraction": self.source_reach_fraction,
                "source_bend_angle_degrees": self.source_bend_angle_degrees,
                "bend_plane_status": self.bend_plane_status,
                "requested_endpoint": _vec(self.requested_endpoint),
                "reachable_endpoint": _vec(self.reachable_endpoint),
                "solved_mid": _vec(self.solved_mid),
                "solved_endpoint": _vec(self.solved_endpoint),
                "endpoint_error": self.endpoint_error, "clamp_status": self.clamp_status,
                "proximal_status": self.proximal_status, "proximal_local": _vec(self.proximal_local),
                "distal_status": self.distal_status, "distal_local": _vec(self.distal_local),
                "reason": self.reason}

    @classmethod
    def from_dict(cls, v: dict[str, Any]) -> "TwoBoneIkSample":
        return cls(v["frame_index"], v["timestamp"], v["chain"], v["source_status"],
                   v["root_orientation_status"], v["source_reach_fraction"],
                   v["source_bend_angle_degrees"], v["bend_plane_status"],
                   _tup(v["requested_endpoint"]), _tup(v["reachable_endpoint"]),
                   _tup(v["solved_mid"]), _tup(v["solved_endpoint"]), v["endpoint_error"],
                   v["clamp_status"], v["proximal_status"], _tup(v["proximal_local"]),
                   v["distal_status"], _tup(v["distal_local"]), v["reason"])


@dataclass(frozen=True)
class TwoBoneIkResult:
    sequence_id: str
    rig_id: str
    provenance: dict[str, Any]
    samples: tuple[TwoBoneIkSample, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"schema": SCHEMA, "sequence_id": self.sequence_id, "rig_id": self.rig_id,
                "provenance": self.provenance, "samples": [s.to_dict() for s in self.samples]}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TwoBoneIkResult":
        if value.get("schema") != SCHEMA or value.get("provenance", {}).get("policy") != POLICY:
            raise ValueError("not a FramePose two-bone IK result")
        return cls(value["sequence_id"], value["rig_id"], value["provenance"],
                   tuple(TwoBoneIkSample.from_dict(s) for s in value["samples"]))


def save_two_bone_ik(value: TwoBoneIkResult, path: str | Path) -> None:
    write_json(path, value.to_dict())


def load_two_bone_ik(path: str | Path) -> TwoBoneIkResult:
    return TwoBoneIkResult.from_dict(read_json(path))


def _unavailable(frame, chain, source_status, yaw_status, reason, geometry=None, requested=None):
    return TwoBoneIkSample(
        frame.frame_index, frame.timestamp, chain.name, source_status, yaw_status,
        geometry.reach_fraction if geometry else None,
        geometry.bend_angle_degrees if geometry else None, "unavailable",
        requested, None, None, None, None, None, "unavailable", None, "unavailable", None, reason)


def heading_relative_chain(semantics: AnimationSemanticsV2, frame, chain: TwoBoneIkChain,
                           alignment: Quaternion) -> tuple[str, Vec3 | None, Vec3 | None, str | None]:
    """(source_status, upper, lower, reason) with yaw removed exactly as FK does."""
    lookup = {name: index for index, name in enumerate(semantics.joint_names)}
    indices = [lookup[name] for name in (chain.source_root, chain.source_mid, chain.source_end)]
    valid = frame.reliability.joint_observation_valid
    for role, index in zip(("root", "mid", "end"), indices):
        if not valid[index]:
            return "unknown", None, None, f"source_{role}_invalid"
    root, mid, end = (frame.articulation.joint_positions[i] for i in indices)
    if not all(math.isfinite(v) for v in (*root, *mid, *end)):
        return "unknown", None, None, "source_non_finite"
    yaw = frame.root_orientation.yaw_radians
    if yaw is None:
        return "known", None, None, "root_orientation_unknown"
    unyaw = quaternion_from_axis_angle((0, 0, 1), -yaw)

    def mapped(v):
        return rotate_vector_by_quaternion(rotate_vector_by_quaternion(v, unyaw), alignment)
    return "known", mapped(_sub(mid, root)), mapped(_sub(end, mid)), None


def solve_framepose_two_bone_ik(
    semantics: AnimationSemanticsV2, rest: TargetRigRestPose,
    calibration: RigFkCalibration, chains: TwoBoneIkChainSet,
) -> TwoBoneIkResult:
    if semantics.coordinate_frame != COORDINATE_FRAME or semantics.provenance.root_orientation.get("policy") != ROOT_POLICY_V2:
        raise ValueError("IK requires current-policy canonical AnimationSemantics v2")
    if not (rest.rig_id == calibration.rig_id == chains.rig_id):
        raise ValueError("rig, calibration and IK chain IDs must match")
    alignment = derive_rig_alignment(rest, calibration)
    geometry = {chain.name: target_chain_geometry(rest, chain) for chain in chains.chains}
    for chain in chains.chains:
        missing = [n for n in (chain.source_root, chain.source_mid, chain.source_end)
                   if n not in semantics.joint_names]
        if missing:
            raise ValueError(f"{chain.name}: source joint absent: {missing}")
    samples = []
    for frame in semantics.frames:
        yaw_status = "known" if frame.root_orientation.yaw_radians is not None else "unknown"
        for chain in chains.chains:
            target = geometry[chain.name]
            status, upper, lower, reason = heading_relative_chain(semantics, frame, chain, alignment)
            if reason is not None:
                samples.append(_unavailable(frame, chain, status, yaw_status, reason))
                continue
            source = source_chain_geometry(upper, lower)
            if source is None:
                samples.append(_unavailable(frame, chain, "unknown", yaw_status,
                                            "source_segment_degenerate"))
                continue
            if source.endpoint_direction is None:
                samples.append(_unavailable(frame, chain, "known", yaw_status,
                                            "source_endpoint_direction_undefined", source))
                continue
            e = source.endpoint_direction
            distance = source.reach_fraction * target.max_reach
            requested = _add(target.root_head, _scale(e, distance))
            side = observed_bend_side(source)
            bend_status = "observed"
            if side is None:
                side = rest_plane_side(target, rest, e)
                bend_status = "degenerate_rest_plane_fallback"
            if side is None:
                samples.append(_unavailable(frame, chain, "known", yaw_status,
                                            "bend_plane_undefined", source, requested))
                continue
            solve = solve_two_bone(target.root_head, e, distance,
                                   target.proximal_length, target.distal_length, side)
            local_p, local_d = local_rotations_for_directions(
                rest, target.proximal, target.distal,
                _sub(solve.mid, target.root_head), _sub(solve.end, solve.mid))
            mid, end = chain_forward_kinematics(rest, target, local_p, local_d)
            samples.append(TwoBoneIkSample(
                frame.frame_index, frame.timestamp, chain.name, "known", yaw_status,
                source.reach_fraction, source.bend_angle_degrees, bend_status,
                requested, solve.end, mid, end, math.dist(end, requested), solve.clamp_status,
                "known", local_p, "known", local_d, None))
    return TwoBoneIkResult(semantics.sequence_id, rest.rig_id, {
        "policy": POLICY, "semantics_digest": semantics.content_digest(),
        "rest_snapshot_digest": rest.digest(), "rest_provenance": rest.provenance,
        "rig_calibration_digest": _digest(calibration.to_dict()),
        "chain_contract": chains.to_dict(), "chain_contract_digest": _digest(chains.to_dict()),
        "target_chains": {name: g.to_dict() for name, g in geometry.items()},
        "canonical_heading_zero_to_armature_xyzw": list(alignment),
        "endpoint_formula": ENDPOINT_FORMULA,
        "degenerate_bend_rule": "source |u x l| < sin(1 deg): rest-plane fallback, never observed",
        "reach_clamp": "deterministic to [|L1-L2|, L1+L2]; every clamp recorded",
        "root_yaw_owner": "Blender Armature Object (not applied to IK chains)",
        "rotation_convention": "xyzw; Blender bone-local pose delta; shortest-arc swing; zero axial twist",
        "temporal_policy": "per-frame; no previous pole, rotation or length",
        "root_translation": "unavailable", "ground_placement": "unavailable",
    }, tuple(samples))


# ----------------------------------------------------------- FK control


@dataclass(frozen=True)
class FkIkComparison:
    frame_index: int
    chain: str
    status: str  # both_known / fk_only / ik_only / neither
    requested_endpoint: Vec3 | None
    fk_endpoint: Vec3 | None
    ik_endpoint: Vec3 | None
    fk_endpoint_error: float | None
    ik_endpoint_error: float | None
    proximal_delta_degrees: float | None
    distal_delta_degrees: float | None

    def to_dict(self) -> dict[str, Any]:
        return {"frame_index": self.frame_index, "chain": self.chain, "status": self.status,
                "requested_endpoint": _vec(self.requested_endpoint),
                "fk_endpoint": _vec(self.fk_endpoint), "ik_endpoint": _vec(self.ik_endpoint),
                "fk_endpoint_error": self.fk_endpoint_error,
                "ik_endpoint_error": self.ik_endpoint_error,
                "proximal_delta_degrees": self.proximal_delta_degrees,
                "distal_delta_degrees": self.distal_delta_degrees}


def compare_fk_ik(fk_samples, ik: TwoBoneIkResult, rest: TargetRigRestPose,
                  chains: TwoBoneIkChainSet) -> tuple[FkIkComparison, ...]:
    """Evaluate unchanged FK locals and IK locals against the same endpoint contract."""
    if ik.provenance["rest_snapshot_digest"] != rest.digest():
        raise ValueError("IK result/rest mismatch")
    fk = {(s.frame_index, s.target_bone): s for s in fk_samples}
    geometry = {chain.name: target_chain_geometry(rest, chain) for chain in chains.chains}
    rows = []
    for sample in ik.samples:
        target = geometry[sample.chain]
        p = fk.get((sample.frame_index, target.proximal))
        d = fk.get((sample.frame_index, target.distal))
        if p is None or d is None:
            raise ValueError(f"FK baseline lacks {sample.chain} frame {sample.frame_index}")
        fk_known = p.rotation_status == "known" and d.rotation_status == "known"
        ik_known = sample.proximal_status == "known"
        fk_end = (chain_forward_kinematics(rest, target, p.rotation_local, d.rotation_local)[1]
                  if fk_known else None)
        requested = sample.requested_endpoint
        rows.append(FkIkComparison(
            sample.frame_index, sample.chain,
            {(True, True): "both_known", (True, False): "fk_only",
             (False, True): "ik_only", (False, False): "neither"}[(fk_known, ik_known)],
            requested, fk_end, sample.solved_endpoint,
            math.dist(fk_end, requested) if fk_end is not None and requested is not None else None,
            sample.endpoint_error,
            quaternion_angle_degrees(p.rotation_local, sample.proximal_local)
            if fk_known and ik_known else None,
            quaternion_angle_degrees(d.rotation_local, sample.distal_local)
            if fk_known and ik_known else None))
    return tuple(rows)
