"""RigProfile data model (Architecture_v2.md section 3.3).

Downstream modules must consume ``RigProfile``, never raw FBX SDK or
Assimp structures directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from common.serialization import read_json, write_json
from common.types import Matrix4, Vec3


@dataclass
class BoneInfo:
    name: str
    parent: str | None
    children: list[str] = field(default_factory=list)
    rest_local_matrix: Matrix4 | None = None
    rest_world_matrix: Matrix4 | None = None
    local_axis_hint: dict[str, Vec3] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "parent": self.parent,
            "children": list(self.children),
            "rest_local_matrix": _matrix_to_list(self.rest_local_matrix),
            "rest_world_matrix": _matrix_to_list(self.rest_world_matrix),
            "local_axis_hint": (
                {axis: list(vec) for axis, vec in self.local_axis_hint.items()}
                if self.local_axis_hint is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BoneInfo":
        axis_hint = data.get("local_axis_hint")
        return cls(
            name=data["name"],
            parent=data["parent"],
            children=list(data["children"]),
            rest_local_matrix=_matrix_from_list(data.get("rest_local_matrix")),
            rest_world_matrix=_matrix_from_list(data.get("rest_world_matrix")),
            local_axis_hint=(
                {axis: tuple(vec) for axis, vec in axis_hint.items()}
                if axis_hint is not None
                else None
            ),
        )


@dataclass
class RigProfile:
    rig_id: str
    source_path: str
    bones: dict[str, BoneInfo] = field(default_factory=dict)
    root_bone: str | None = None
    scale: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rig_id": self.rig_id,
            "source_path": self.source_path,
            "bones": {name: bone.to_dict() for name, bone in self.bones.items()},
            "root_bone": self.root_bone,
            "scale": self.scale,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RigProfile":
        return cls(
            rig_id=data["rig_id"],
            source_path=data["source_path"],
            bones={
                name: BoneInfo.from_dict(bone)
                for name, bone in data["bones"].items()
            },
            root_bone=data["root_bone"],
            scale=data["scale"],
            metadata=dict(data["metadata"]),
        )


def save_rig_profile(profile: RigProfile, path: str | Path) -> None:
    write_json(path, profile.to_dict())


def load_rig_profile(path: str | Path) -> RigProfile:
    return RigProfile.from_dict(read_json(path))


def _matrix_to_list(matrix: Matrix4 | None) -> list[list[float]] | None:
    if matrix is None:
        return None
    return [list(row) for row in matrix]


def _matrix_from_list(data: list[list[float]] | None) -> Matrix4 | None:
    if data is None:
        return None
    return tuple(tuple(row) for row in data)
