"""Direction-only FramePose FK using the Blender-imported target rest pose.

The rig calibration is fixed for a rest snapshot and semantic mapping.  Source
frame zero never participates in calibration.  Root yaw is removed from each
camera-relative limb direction, then applied once at the declared root owner.
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
    quaternion_conjugate, quaternion_from_axis_angle, quaternion_from_matrix,
    quaternion_from_vectors, quaternion_multiply, rotate_vector_by_quaternion,
)
from retarget.framepose_direction_candidate import source_direction
from retarget.framepose_target_rest import TargetRigRestPose
from rig.bone_mapping import BoneMappingProfile

SCHEMA = "animcv_framepose_fk_v1"
POLICY = "blender_rest_rig_calibrated_heading_separated_fk_v1"
_IDENTITY: Quaternion = (0.0, 0.0, 0.0, 1.0)


def _unit(vector: Vec3) -> Vec3:
    length = math.sqrt(sum(v * v for v in vector))
    if not math.isfinite(length) or length <= 1e-8:
        raise ValueError("degenerate rig calibration/rest direction")
    return tuple(v / length for v in vector)


def _q_unit(quaternion: Quaternion) -> Quaternion:
    length = math.sqrt(sum(v * v for v in quaternion))
    if not math.isfinite(length) or length <= 1e-8:
        raise ValueError("invalid FK quaternion")
    q = tuple(v / length for v in quaternion)
    if q[3] < -1e-12 or (abs(q[3]) <= 1e-12 and next(
        (v for v in q[:3] if abs(v) > 1e-12), 0.0) < 0
    ):
        q = tuple(-v for v in q)
    return q


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _subtract(a: Vec3, b: Vec3) -> Vec3:
    return tuple(x - y for x, y in zip(a, b))


def _digest(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class RigFkCalibration:
    rig_id: str
    root_orientation_target_bone: str
    left_shoulder_target_bone: str
    right_shoulder_target_bone: str
    up_target_bone: str

    def to_dict(self) -> dict[str, str]:
        return vars(self).copy()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RigFkCalibration":
        return cls(**value)


def load_rig_fk_calibration(path: str | Path) -> RigFkCalibration:
    return RigFkCalibration.from_dict(read_json(path))


def derive_rig_alignment(rest: TargetRigRestPose, calibration: RigFkCalibration) -> Quaternion:
    """Map canonical heading-zero (+X right, +Y forward, +Z up) to armature space.

    The semantic left/right shoulder anchors and root-to-upper-body vector are
    fixed rig facts.  The pair resolves heading; the up vector resolves roll.
    Degenerate landmarks are rejected, never completed from video frames.
    """
    if rest.rig_id != calibration.rig_id:
        raise ValueError("calibration rig_id differs from Blender rest snapshot")
    try:
        root = rest.bones[calibration.root_orientation_target_bone]
        left = rest.bones[calibration.left_shoulder_target_bone]
        right = rest.bones[calibration.right_shoulder_target_bone]
        up = rest.bones[calibration.up_target_bone]
    except KeyError as exc:
        raise ValueError(f"calibration target bone missing: {exc.args[0]}") from exc
    x = _unit(_subtract(right.head, left.head))
    up_raw = _subtract(up.head, root.head)
    up_projected = tuple(up_raw[i] - sum(up_raw[j] * x[j] for j in range(3)) * x[i]
                         for i in range(3))
    z = _unit(up_projected)
    y = _unit(_cross(z, x))
    z = _unit(_cross(x, y))
    matrix = tuple(tuple((x, y, z)[column][row] if column < 3 else 0.0
                         for column in range(4)) for row in range(3)) + ((0.0, 0.0, 0.0, 1.0),)
    return _q_unit(quaternion_from_matrix(matrix))


def _is_ancestor(rest: TargetRigRestPose, ancestor: str, child: str) -> bool:
    seen = set()
    parent = rest.bones[child].parent
    while parent is not None:
        if parent in seen:
            raise ValueError("cyclic target hierarchy")
        if parent == ancestor:
            return True
        seen.add(parent)
        parent = rest.bones[parent].parent
    return False


def _depth(rest: TargetRigRestPose, name: str) -> int:
    depth = 0
    while rest.bones[name].parent is not None:
        name = rest.bones[name].parent
        depth += 1
    return depth


def _rest_direction(rest: TargetRigRestPose, name: str) -> Vec3:
    bone = rest.bones[name]
    return _unit(_subtract(bone.tail, bone.head))


def _rest_rotation(rest: TargetRigRestPose, name: str) -> Quaternion:
    return _q_unit(quaternion_from_matrix(rest.bones[name].matrix_local))


@dataclass(frozen=True)
class FkSample:
    frame_index: int
    timestamp: float
    target_bone: str
    source_pair: tuple[str, str] | None
    source_status: str
    raw_direction: Vec3 | None
    root_yaw_radians: float | None
    heading_relative_direction: Vec3 | None
    target_rest_direction: Vec3 | None
    rotation_status: str
    rotation_local: Quaternion | None
    reason: str | None

    def __post_init__(self) -> None:
        if self.source_status not in ("known", "unknown") or self.rotation_status not in ("known", "unavailable"):
            raise ValueError("invalid FK status")
        if self.rotation_status == "known":
            if self.rotation_local is None or self.reason is not None or self.source_status != "known":
                raise ValueError("known FK rotation requires valid source and quaternion")
            if abs(sum(v * v for v in self.rotation_local) - 1.0) > 1e-8:
                raise ValueError("FK rotation is not normalized")
        elif self.rotation_local is not None or self.reason is None:
            raise ValueError("unavailable FK rotation must have reason and no quaternion")

    def to_dict(self) -> dict[str, Any]:
        return {"frame_index": self.frame_index, "timestamp": self.timestamp,
                "target_bone": self.target_bone,
                "source_pair": list(self.source_pair) if self.source_pair else None,
                "source_status": self.source_status,
                "raw_direction": list(self.raw_direction) if self.raw_direction else None,
                "root_yaw_radians": self.root_yaw_radians,
                "heading_relative_direction": (list(self.heading_relative_direction)
                                               if self.heading_relative_direction else None),
                "target_rest_direction": (list(self.target_rest_direction)
                                          if self.target_rest_direction else None),
                "rotation_status": self.rotation_status,
                "rotation_local": list(self.rotation_local) if self.rotation_local else None,
                "reason": self.reason}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FkSample":
        def vec(key):
            return tuple(value[key]) if value[key] is not None else None
        return cls(value["frame_index"], value["timestamp"], value["target_bone"],
                   vec("source_pair"), value["source_status"], vec("raw_direction"),
                   value["root_yaw_radians"], vec("heading_relative_direction"),
                   vec("target_rest_direction"), value["rotation_status"],
                   vec("rotation_local"), value["reason"])


@dataclass(frozen=True)
class FkResult:
    sequence_id: str
    rig_id: str
    provenance: dict[str, Any]
    samples: tuple[FkSample, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"schema": SCHEMA, "sequence_id": self.sequence_id,
                "rig_id": self.rig_id, "provenance": self.provenance,
                "samples": [sample.to_dict() for sample in self.samples]}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FkResult":
        if value.get("schema") != SCHEMA or value.get("provenance", {}).get("policy") != POLICY:
            raise ValueError("not a FramePose FK result")
        return cls(value["sequence_id"], value["rig_id"], value["provenance"],
                   tuple(FkSample.from_dict(sample) for sample in value["samples"]))


def save_fk_result(value: FkResult, path: str | Path) -> None:
    write_json(path, value.to_dict())


def load_fk_result(path: str | Path) -> FkResult:
    return FkResult.from_dict(read_json(path))


def solve_framepose_fk(
    semantics: AnimationSemanticsV2, rest: TargetRigRestPose,
    mapping: BoneMappingProfile, calibration: RigFkCalibration,
) -> FkResult:
    if semantics.coordinate_frame != COORDINATE_FRAME or semantics.provenance.root_orientation.get("policy") != ROOT_POLICY_V2:
        raise ValueError("FK requires current-policy canonical AnimationSemantics v2")
    if rest.rig_id != mapping.rig_id or rest.rig_id != calibration.rig_id:
        raise ValueError("rig, mapping and calibration IDs must match")
    alignment = derive_rig_alignment(rest, calibration)
    owner = calibration.root_orientation_target_bone
    entries = [entry for entry in mapping.entries if entry.mapping_mode == "direction"]
    if len(entries) != len(mapping.entries) or not entries:
        raise ValueError("first FramePose FK supports direction mappings only")
    if len({entry.target_bone for entry in entries}) != len(entries):
        raise ValueError("duplicate FK target bone")
    for entry in entries:
        if len(entry.source_names) != 2 or entry.target_bone not in rest.bones:
            raise ValueError(f"invalid direction mapping: {entry.target_bone}")
        if entry.target_bone == owner or not _is_ancestor(rest, owner, entry.target_bone):
            raise ValueError(f"root orientation owner is not an ancestor of {entry.target_bone}")
    entries.sort(key=lambda entry: (_depth(rest, entry.target_bone), entry.target_bone))
    names = {entry.target_bone for entry in entries}
    for entry in entries:
        if any(name not in semantics.joint_names for name in entry.source_names):
            raise ValueError(f"source joint absent: {entry.source_names}")

    samples = []
    owner_rest_rotation = _rest_rotation(rest, owner)
    for frame in semantics.frames:
        yaw = frame.root_orientation.yaw_radians
        root_local = None
        if yaw is not None:
            source_yaw = quaternion_from_axis_angle((0, 0, 1), yaw)
            rig_yaw = quaternion_multiply(quaternion_multiply(alignment, source_yaw),
                                          quaternion_conjugate(alignment))
            root_local = _q_unit(quaternion_multiply(quaternion_multiply(
                quaternion_conjugate(owner_rest_rotation), rig_yaw), owner_rest_rotation))
        samples.append(FkSample(
            frame.frame_index, frame.timestamp, owner, None,
            "known" if yaw is not None else "unknown", None, yaw, None, None,
            "known" if root_local is not None else "unavailable", root_local,
            None if root_local is not None else "root_orientation_unknown"))

        world_deltas: dict[str, Quaternion | None] = {}
        for entry in entries:
            target = entry.target_bone
            pair = tuple(entry.source_names)
            raw = source_direction(semantics, frame, *pair)
            rest_direction = _rest_direction(rest, target)
            body_direction = None
            local = None
            reason = None
            if raw is None:
                reason = "source_observation_invalid_or_degenerate"
            elif yaw is None:
                reason = "root_orientation_unknown"
            else:
                body_direction = _unit(rotate_vector_by_quaternion(
                    raw, quaternion_from_axis_angle((0, 0, 1), -yaw)))
                target_direction = _unit(rotate_vector_by_quaternion(body_direction, alignment))
                delta_world = _q_unit(quaternion_from_vectors(rest_direction, target_direction))
                parent = rest.bones[target].parent
                while parent is not None and parent not in names:
                    parent = rest.bones[parent].parent
                parent_delta = world_deltas.get(parent, _IDENTITY) if parent else _IDENTITY
                if parent_delta is None:
                    reason = "mapped_ancestor_rotation_unavailable"
                else:
                    relative = quaternion_multiply(quaternion_conjugate(parent_delta), delta_world)
                    rest_rotation = _rest_rotation(rest, target)
                    local = _q_unit(quaternion_multiply(quaternion_multiply(
                        quaternion_conjugate(rest_rotation), relative), rest_rotation))
                    world_deltas[target] = delta_world
            if local is None:
                world_deltas[target] = None
            samples.append(FkSample(
                frame.frame_index, frame.timestamp, target, pair,
                "known" if raw is not None else "unknown", raw, yaw, body_direction,
                rest_direction, "known" if local is not None else "unavailable",
                local, reason))

    return FkResult(semantics.sequence_id, rest.rig_id, {
        "policy": POLICY, "semantics_digest": semantics.content_digest(),
        "rest_snapshot_digest": rest.digest(), "rest_provenance": rest.provenance,
        "mapping_digest": _digest(mapping.to_dict()),
        "rig_calibration": calibration.to_dict(),
        "rig_calibration_digest": _digest(calibration.to_dict()),
        "canonical_heading_zero_to_armature_xyzw": list(alignment),
        "root_yaw_owner": owner,
        "root_yaw_uncovered_bones": sorted(name for name in rest.bones
                                           if name != owner and not _is_ancestor(rest, owner, name)),
        "rotation_convention": "xyzw; Blender bone-local pose delta; shortest-arc swing; zero axial twist",
        "root_translation": "unavailable", "ground_placement": "unavailable",
    }, tuple(samples))
