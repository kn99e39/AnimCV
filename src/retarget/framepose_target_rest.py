"""Blender-imported armature rest pose, serialized without a bpy dependency.

Matrices, heads and tails are in armature object space.  This is deliberately
separate from Assimp RigProfile: an FBX importer may change bone bases.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common.serialization import read_json, write_json

SCHEMA = "animcv_blender_target_rest_pose_v1"


@dataclass(frozen=True)
class RestBone:
    name: str
    parent: str | None
    matrix_local: tuple[tuple[float, ...], ...]
    head: tuple[float, float, float]
    tail: tuple[float, float, float]

    def __post_init__(self) -> None:
        if len(self.matrix_local) != 4 or any(len(row) != 4 for row in self.matrix_local):
            raise ValueError("rest matrix must be 4x4")
        if not all(math.isfinite(v) for row in self.matrix_local for v in row):
            raise ValueError("rest matrix must be finite")
        if len(self.head) != 3 or len(self.tail) != 3 or not all(
            math.isfinite(v) for v in (*self.head, *self.tail)
        ):
            raise ValueError("head/tail must be finite 3D points")
        if math.dist(self.head, self.tail) <= 1e-8:
            raise ValueError("rest bone must have a nonzero head-to-tail direction")

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "parent": self.parent,
                "matrix_local": [list(row) for row in self.matrix_local],
                "head": list(self.head), "tail": list(self.tail)}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RestBone":
        return cls(value["name"], value["parent"],
                   tuple(tuple(float(x) for x in row) for row in value["matrix_local"]),
                   tuple(float(x) for x in value["head"]),
                   tuple(float(x) for x in value["tail"]))


@dataclass(frozen=True)
class TargetRigRestPose:
    rig_id: str
    bones: dict[str, RestBone]
    provenance: dict[str, Any]

    def __post_init__(self) -> None:
        if self.provenance.get("authority") != "blender_imported_armature":
            raise ValueError("FK rest authority must be a Blender-imported armature")
        for name, bone in self.bones.items():
            if name != bone.name or (bone.parent is not None and bone.parent not in self.bones):
                raise ValueError("invalid Blender rest hierarchy")
            seen = {name}
            parent = bone.parent
            while parent is not None:
                if parent in seen:
                    raise ValueError("cyclic Blender rest hierarchy")
                seen.add(parent)
                parent = self.bones[parent].parent

    def to_dict(self) -> dict[str, Any]:
        return {"schema": SCHEMA, "rig_id": self.rig_id,
                "coordinate_convention": "Blender armature object space; matrix_local and head/tail at rest",
                "provenance": self.provenance,
                "bones": {name: bone.to_dict() for name, bone in sorted(self.bones.items())}}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TargetRigRestPose":
        if value.get("schema") != SCHEMA or value.get("coordinate_convention") != (
            "Blender armature object space; matrix_local and head/tail at rest"
        ):
            raise ValueError("not a Blender target rest snapshot")
        return cls(value["rig_id"], {name: RestBone.from_dict(bone)
                                     for name, bone in value["bones"].items()}, value["provenance"])

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True,
                                         separators=(",", ":")).encode()).hexdigest()


def save_target_rest_pose(value: TargetRigRestPose, path: str | Path) -> None:
    write_json(path, value.to_dict())


def load_target_rest_pose(path: str | Path) -> TargetRigRestPose:
    return TargetRigRestPose.from_dict(read_json(path))
