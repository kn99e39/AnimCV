"""Whole-rig Root Orientation output above unchanged FramePose pose-bone FK.

The returned object rotation is a delta in imported armature object space.
Blender composes it with the imported object's rest rotation; pose-bone local
FK channels contain no duplicate yaw.  No object translation is produced.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common.serialization import read_json, write_json
from common.types import Quaternion
from motion.animation_semantics_v2 import AnimationSemanticsV2
from retarget.axis_utils import (
    quaternion_conjugate, quaternion_from_axis_angle, quaternion_multiply,
)
from retarget.framepose_fk import (
    FkSample, RigFkCalibration, _q_unit, derive_rig_alignment, solve_framepose_fk,
)
from retarget.framepose_target_rest import TargetRigRestPose
from rig.bone_mapping import BoneMappingProfile

SCHEMA = "animcv_framepose_object_owned_fk_v1"
POLICY = "rig_object_root_yaw_with_heading_relative_pose_bone_fk_v1"


@dataclass(frozen=True)
class RigRotationSample:
    frame_index: int
    timestamp: float
    root_orientation_status: str  # known / unknown
    yaw_radians: float | None
    rotation_status: str  # known / unavailable
    rotation_armature_space: Quaternion | None
    reason: str | None

    def __post_init__(self) -> None:
        if self.root_orientation_status not in ("known", "unknown") or self.rotation_status not in ("known", "unavailable"):
            raise ValueError("invalid rig rotation status")
        if self.root_orientation_status == "known":
            if self.yaw_radians is None or not math.isfinite(self.yaw_radians):
                raise ValueError("known Root Orientation requires finite yaw")
            if self.rotation_status != "known" or self.rotation_armature_space is None or self.reason is not None:
                raise ValueError("known yaw requires rig rotation")
            if abs(sum(v * v for v in self.rotation_armature_space) - 1) > 1e-8:
                raise ValueError("rig rotation must be normalized")
        elif (self.yaw_radians is not None or self.rotation_status != "unavailable"
              or self.rotation_armature_space is not None or self.reason != "root_orientation_unknown"):
            raise ValueError("unknown Root Orientation must remain unavailable")

    def to_dict(self) -> dict[str, Any]:
        return {"frame_index": self.frame_index, "timestamp": self.timestamp,
                "root_orientation_status": self.root_orientation_status,
                "yaw_radians": self.yaw_radians, "rotation_status": self.rotation_status,
                "rotation_armature_space": (list(self.rotation_armature_space)
                                            if self.rotation_armature_space is not None else None),
                "reason": self.reason}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RigRotationSample":
        q = value["rotation_armature_space"]
        return cls(int(value["frame_index"]), float(value["timestamp"]),
                   value["root_orientation_status"], value["yaw_radians"],
                   value["rotation_status"], tuple(q) if q is not None else None,
                   value["reason"])


@dataclass(frozen=True)
class ObjectOwnedFkResult:
    sequence_id: str
    rig_id: str
    provenance: dict[str, Any]
    rig_rotations: tuple[RigRotationSample, ...]
    pose_bone_samples: tuple[FkSample, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"schema": SCHEMA, "sequence_id": self.sequence_id,
                "rig_id": self.rig_id, "provenance": self.provenance,
                "rig_rotations": [s.to_dict() for s in self.rig_rotations],
                "pose_bone_samples": [s.to_dict() for s in self.pose_bone_samples]}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ObjectOwnedFkResult":
        if value.get("schema") != SCHEMA or value.get("provenance", {}).get("policy") != POLICY:
            raise ValueError("not an object-owned FramePose FK result")
        return cls(value["sequence_id"], value["rig_id"], value["provenance"],
                   tuple(RigRotationSample.from_dict(s) for s in value["rig_rotations"]),
                   tuple(FkSample.from_dict(s) for s in value["pose_bone_samples"]))


def save_object_owned_fk(value: ObjectOwnedFkResult, path: str | Path) -> None:
    write_json(path, value.to_dict())


def load_object_owned_fk(path: str | Path) -> ObjectOwnedFkResult:
    return ObjectOwnedFkResult.from_dict(read_json(path))


def solve_object_owned_fk(
    semantics: AnimationSemanticsV2, rest: TargetRigRestPose,
    mapping: BoneMappingProfile, calibration: RigFkCalibration,
) -> ObjectOwnedFkResult:
    """Keep limb FK from Worklog 59 verbatim, changing only yaw ownership."""
    base = solve_framepose_fk(semantics, rest, mapping, calibration,
                              root_orientation_owner="armature_object")
    alignment = derive_rig_alignment(rest, calibration)
    rotations = []
    for frame in semantics.frames:
        yaw = frame.root_orientation.yaw_radians
        if yaw is None:
            rotations.append(RigRotationSample(
                frame.frame_index, frame.timestamp, "unknown", None,
                "unavailable", None, "root_orientation_unknown"))
        else:
            source_yaw = quaternion_from_axis_angle((0, 0, 1), yaw)
            object_delta = _q_unit(quaternion_multiply(
                quaternion_multiply(alignment, source_yaw), quaternion_conjugate(alignment)))
            rotations.append(RigRotationSample(
                frame.frame_index, frame.timestamp, "known", yaw,
                "known", object_delta, None))
    limbs = base.samples
    if len(limbs) != len(semantics.frames) * len(mapping.entries):
        raise ValueError("pose-bone FK row count mismatch")
    baseline_bytes = json.dumps([s.to_dict() for s in limbs], sort_keys=True,
                                separators=(",", ":")).encode()
    return ObjectOwnedFkResult(semantics.sequence_id, rest.rig_id, {
        "policy": POLICY,
        "semantics_digest": semantics.content_digest(),
        "rest_snapshot_digest": rest.digest(),
        "rest_provenance": rest.provenance,
        "mapping_digest": base.provenance["mapping_digest"],
        "rig_calibration_digest": base.provenance["rig_calibration_digest"],
        "canonical_heading_zero_to_armature_xyzw": list(alignment),
        "pose_bone_fk_policy": base.provenance["policy"],
        "pose_bone_samples_sha256": hashlib.sha256(baseline_bytes).hexdigest(),
        "root_owner": "Blender Armature Object",
        "imported_top_level_bones": sorted(name for name, bone in rest.bones.items()
                                           if bone.parent is None),
        "rotation_convention": "xyzw; object-space delta; imported object rest rotation composed by Blender adapter",
        "root_translation": "unavailable",
        "ground_placement": "unavailable",
    }, tuple(rotations), limbs)
