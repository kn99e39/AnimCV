"""Direction-only FramePose retarget contract control, separate from MotionGraph.

An explicit canonical-to-rig rest-world orientation is required. RigProfile
does not currently own that video/session alignment. A missing declaration
produces UNAVAILABLE rotations; this module never guesses one from frame 0.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from common.serialization import read_json, write_json
from common.types import Matrix4, Quaternion, Vec3
from motion.animation_semantics_v2 import AnimationSemanticsV2, ROOT_POLICY_V2
from retarget.axis_utils import (
    quaternion_conjugate, quaternion_from_matrix, quaternion_from_vectors,
    quaternion_multiply, rotate_vector_by_quaternion,
)
from rig.bone_mapping import BoneMappingEntry, BoneMappingProfile
from rig.rig_profile import BoneInfo, RigProfile

POLICY = "framepose_canonical_direction_to_rig_rest_v1"
SCHEMA = "animcv_framepose_direction_candidate_v1"
_IDENTITY: Quaternion = (0.0, 0.0, 0.0, 1.0)


class ContractGap(ValueError):
    """An explicit rig or alignment fact needed for this mapping is absent."""


def _unit(v: Vec3) -> Vec3:
    if not all(math.isfinite(x) for x in v):
        raise ContractGap("non_finite_direction")
    length = math.sqrt(sum(x * x for x in v))
    if length <= 1e-8:
        raise ContractGap("zero_length_direction")
    return tuple(x / length for x in v)


def _unit_quaternion(q: Quaternion) -> Quaternion:
    if len(q) != 4 or not all(math.isfinite(v) for v in q):
        raise ContractGap("invalid_coordinate_alignment")
    length = math.sqrt(sum(v * v for v in q))
    if length <= 1e-8:
        raise ContractGap("invalid_coordinate_alignment")
    unit = tuple(v / length for v in q)
    # q and -q encode the same rotation. Use a stable representation without
    # temporal hold or sign propagation from another frame.
    if unit[3] < -1e-12 or (abs(unit[3]) <= 1e-12 and next((v for v in unit[:3] if abs(v) > 1e-12), 0) < 0):
        unit = tuple(-v for v in unit)
    return unit


def _matrix(bone: BoneInfo, *, world: bool) -> np.ndarray:
    raw = bone.rest_world_matrix if world else bone.rest_local_matrix
    if raw is None:
        raise ContractGap("rest_world_matrix_missing" if world else "rest_local_matrix_missing")
    matrix = np.asarray(raw, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ContractGap("invalid_rest_matrix")
    if abs(float(np.linalg.det(matrix[:3, :3]))) <= 1e-10:
        raise ContractGap("singular_rest_basis")
    return matrix


def _rest_world_matrix(rig: RigProfile, name: str, visiting: frozenset[str] = frozenset()) -> np.ndarray:
    bone = rig.bones.get(name)
    if bone is None:
        raise ContractGap("target_bone_missing")
    if bone.rest_world_matrix is not None:
        return _matrix(bone, world=True)
    if name in visiting:
        raise ContractGap("cyclic_target_hierarchy")
    local = _matrix(bone, world=False)
    if bone.parent is None:
        return local
    return _rest_world_matrix(rig, bone.parent, visiting | {name}) @ local


def target_rest_axis(rig: RigProfile, name: str) -> tuple[Vec3, str]:
    """Bone-local head-to-tail axis: explicit primary hint, else unique child offset.

    BoneMappingEntry.axis_hint is the historical 2D *rotation axis*, never
    reinterpreted here as the bone's longitudinal rest axis.
    """
    bone = rig.bones.get(name)
    if bone is None:
        raise ContractGap("target_bone_missing")
    if bone.local_axis_hint is not None and "primary" in bone.local_axis_hint:
        return _unit(tuple(bone.local_axis_hint["primary"])), "bone_local_axis_hint_primary"
    if len(bone.children) != 1:
        raise ContractGap("target_rest_axis_ambiguous")
    child = rig.bones.get(bone.children[0])
    if child is None or child.parent != name:
        raise ContractGap("target_child_hierarchy_missing")
    local = _matrix(child, world=False)
    return _unit(tuple(float(local[i, 3]) for i in range(3))), "unique_child_rest_offset"


def _rest_quaternion(rig: RigProfile, name: str) -> Quaternion:
    matrix = _rest_world_matrix(rig, name)
    return _unit_quaternion(quaternion_from_matrix(tuple(tuple(float(v) for v in row) for row in matrix)))


def source_direction(semantics: AnimationSemanticsV2, frame, source_a: str, source_b: str) -> Vec3 | None:
    lookup = {name: index for index, name in enumerate(semantics.joint_names)}
    if source_a not in lookup or source_b not in lookup:
        raise ContractGap("source_joint_not_in_semantics")
    a, b = lookup[source_a], lookup[source_b]
    if not (frame.reliability.joint_observation_valid[a] and frame.reliability.joint_observation_valid[b]):
        return None
    pa, pb = frame.articulation.joint_positions[a], frame.articulation.joint_positions[b]
    try:
        return _unit(tuple(pb[i] - pa[i] for i in range(3)))
    except ContractGap:
        return None


@dataclass(frozen=True)
class DirectionSample:
    frame_index: int
    timestamp: float
    target_bone: str
    source_pair: tuple[str, str]
    source_status: str  # known or unknown
    source_direction_canonical: Vec3 | None
    rotation_status: str  # known or unavailable
    rotation_local: Quaternion | None
    reason: str | None

    def __post_init__(self) -> None:
        if self.source_status not in ("known", "unknown") or self.rotation_status not in ("known", "unavailable"):
            raise ValueError("invalid direction/rotation status")
        if (self.source_status == "known") != (self.source_direction_canonical is not None):
            raise ValueError("source direction/status mismatch")
        if self.rotation_status == "known":
            if self.source_status != "known" or self.rotation_local is None or self.reason is not None:
                raise ValueError("known rotation requires known source and quaternion")
            if abs(sum(v * v for v in self.rotation_local) - 1.0) > 1e-8:
                raise ValueError("rotation must be a unit quaternion")
        elif self.rotation_local is not None or self.reason is None:
            raise ValueError("unavailable rotation must carry a reason, never a quaternion")

    def to_dict(self) -> dict[str, Any]:
        return {"frame_index": self.frame_index, "timestamp": self.timestamp,
                "target_bone": self.target_bone, "source_pair": list(self.source_pair),
                "source_status": self.source_status,
                "source_direction_canonical": (list(self.source_direction_canonical)
                                               if self.source_direction_canonical is not None else None),
                "rotation_status": self.rotation_status,
                "rotation_local": list(self.rotation_local) if self.rotation_local is not None else None,
                "reason": self.reason}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DirectionSample":
        direction = data["source_direction_canonical"]
        rotation = data["rotation_local"]
        return cls(int(data["frame_index"]), float(data["timestamp"]), data["target_bone"],
                   tuple(data["source_pair"]), data["source_status"],
                   tuple(float(v) for v in direction) if direction is not None else None,
                   data["rotation_status"], tuple(float(v) for v in rotation) if rotation is not None else None,
                   data["reason"])


@dataclass(frozen=True)
class DirectionCandidate:
    sequence_id: str
    rig_id: str
    provenance: dict[str, Any]
    samples: tuple[DirectionSample, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"schema": SCHEMA, "sequence_id": self.sequence_id, "rig_id": self.rig_id,
                "provenance": self.provenance, "samples": [sample.to_dict() for sample in self.samples]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DirectionCandidate":
        if data.get("schema") != SCHEMA or data.get("provenance", {}).get("policy") != POLICY:
            raise ValueError("not a FramePose direction candidate")
        return cls(data["sequence_id"], data["rig_id"], data["provenance"],
                   tuple(DirectionSample.from_dict(sample) for sample in data["samples"]))


def save_direction_candidate(value: DirectionCandidate, path: str | Path) -> None:
    write_json(path, value.to_dict())


def load_direction_candidate(path: str | Path) -> DirectionCandidate:
    return DirectionCandidate.from_dict(read_json(path))


def _digest(data: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _mapped_ancestor(rig: RigProfile, name: str, mapping_names: set[str]) -> str | None:
    parent = rig.bones[name].parent
    seen = {name}
    while parent is not None:
        if parent in seen or parent not in rig.bones:
            raise ContractGap("invalid_target_hierarchy")
        if parent in mapping_names:
            return parent
        seen.add(parent)
        parent = rig.bones[parent].parent
    return None


def _depth(rig: RigProfile, name: str) -> int:
    depth, seen = 0, set()
    while name in rig.bones and rig.bones[name].parent is not None:
        if name in seen:
            raise ContractGap("cyclic_target_hierarchy")
        seen.add(name)
        name = rig.bones[name].parent
        depth += 1
    return depth


def solve_framepose_direction_candidate(
    semantics: AnimationSemanticsV2,
    rig: RigProfile,
    mapping: BoneMappingProfile,
    *,
    canonical_to_rig_rest: Quaternion | None,
) -> DirectionCandidate:
    """Return target-local direction quaternions, or explicit UNAVAILABLE rows.

    The explicit coordinate alignment is a session contract. None is a
    legitimate audit input and never falls back to identity or first frame.
    """
    if semantics.provenance.root_orientation.get("policy") != ROOT_POLICY_V2:
        raise ValueError("requires current valid-only AnimationSemantics v2")
    if mapping.rig_id != rig.rig_id:
        raise ValueError("mapping rig_id differs from RigProfile")
    entries = [entry for entry in mapping.entries if entry.mapping_mode == "direction"]
    if len({entry.target_bone for entry in entries}) != len(entries):
        raise ValueError("duplicate direction target bone")
    for entry in entries:
        if len(entry.source_names) != 2:
            raise ValueError(f"{entry.target_bone}: direction needs exactly two source joints")
    entries.sort(key=lambda entry: (_depth(rig, entry.target_bone), entry.target_bone))
    alignment = _unit_quaternion(canonical_to_rig_rest) if canonical_to_rig_rest is not None else None
    mapping_names = {entry.target_bone for entry in entries}
    samples = []
    for frame in semantics.frames:
        world_deltas: dict[str, Quaternion | None] = {}
        for entry in entries:
            pair = tuple(entry.source_names)
            try:
                direction = source_direction(semantics, frame, pair[0], pair[1])
                source_reason = "source_observation_invalid_or_degenerate" if direction is None else None
            except ContractGap as exc:
                direction, source_reason = None, str(exc)
            rotation, reason = None, source_reason
            if direction is not None:
                try:
                    axis_local, _ = target_rest_axis(rig, entry.target_bone)
                    rest = _rest_quaternion(rig, entry.target_bone)
                    if alignment is None:
                        raise ContractGap("canonical_to_rig_rest_alignment_missing")
                    ancestor = _mapped_ancestor(rig, entry.target_bone, mapping_names)
                    if ancestor is not None and world_deltas.get(ancestor) is None:
                        raise ContractGap("mapped_ancestor_rotation_unavailable")
                    rest_world_axis = _unit(rotate_vector_by_quaternion(axis_local, rest))
                    target_world_axis = _unit(rotate_vector_by_quaternion(direction, alignment))
                    world_delta = _unit_quaternion(quaternion_from_vectors(rest_world_axis, target_world_axis))
                    parent_delta = world_deltas[ancestor] if ancestor is not None else _IDENTITY
                    relative_delta = quaternion_multiply(quaternion_conjugate(parent_delta), world_delta)
                    rotation = _unit_quaternion(quaternion_multiply(
                        quaternion_multiply(quaternion_conjugate(rest), relative_delta), rest))
                    world_deltas[entry.target_bone] = world_delta
                    reason = None
                except ContractGap as exc:
                    reason = str(exc)
            if rotation is None:
                world_deltas[entry.target_bone] = None
            samples.append(DirectionSample(
                frame_index=frame.frame_index, timestamp=frame.timestamp,
                target_bone=entry.target_bone, source_pair=pair,
                source_status="known" if direction is not None else "unknown",
                source_direction_canonical=direction,
                rotation_status="known" if rotation is not None else "unavailable",
                rotation_local=rotation, reason=reason))
    return DirectionCandidate(semantics.sequence_id, rig.rig_id,
                              {"policy": POLICY, "semantics_schema": "animcv_animation_semantics_v2",
                               "semantics_policy": ROOT_POLICY_V2,
                               "semantics_content_digest": semantics.content_digest(),
                               "rig_profile_digest": _digest(rig.to_dict()),
                               "mapping_profile_digest": _digest(mapping.to_dict()),
                               "canonical_to_rig_rest": list(alignment) if alignment is not None else None,
                               "rotation_convention": "xyzw; shortest arc; zero twist; target bone local"},
                              tuple(samples))


def audit_direction_mapping(rig: RigProfile, entry: BoneMappingEntry) -> dict[str, Any]:
    bone = rig.bones.get(entry.target_bone)
    result = {"target_bone": entry.target_bone, "target_exists": bone is not None,
              "source_pair": list(entry.source_names), "mapping_mode": entry.mapping_mode,
              "mapping_axis_hint": entry.axis_hint,
              "mapping_axis_hint_meaning": "historical 2D rotation axis; not a target rest direction",
              "parent": bone.parent if bone else None, "children": list(bone.children) if bone else None,
              "rest_local_available": bone.rest_local_matrix is not None if bone else False,
              "rest_world_available": bone.rest_world_matrix is not None if bone else False,
              "rest_axis_source": None, "rest_axis_local": None, "rest_basis_available": False,
              "contract_gap": None}
    try:
        axis, source = target_rest_axis(rig, entry.target_bone)
        _rest_quaternion(rig, entry.target_bone)
        result["rest_axis_source"], result["rest_axis_local"] = source, list(axis)
        result["rest_basis_available"] = True
    except ContractGap as exc:
        result["contract_gap"] = str(exc)
    return result
