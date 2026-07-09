"""BoneMappingProfile data model (Architecture_v2.md section 6.4).

This is the user-assisted binding between target rig bones and observed
video landmarks / custom points. It is intentionally separate from
``RigProfile`` (which only describes the rig's own bone hierarchy).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from common.serialization import read_json, write_json


@dataclass
class BoneMappingEntry:
    target_bone: str
    source_type: str
    source_names: list[str]
    mapping_mode: str
    weight: float = 1.0
    axis_hint: str | None = None
    locked: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_bone": self.target_bone,
            "source_type": self.source_type,
            "source_names": list(self.source_names),
            "mapping_mode": self.mapping_mode,
            "weight": self.weight,
            "axis_hint": self.axis_hint,
            "locked": self.locked,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BoneMappingEntry":
        return cls(
            target_bone=data["target_bone"],
            source_type=data["source_type"],
            source_names=list(data["source_names"]),
            mapping_mode=data["mapping_mode"],
            weight=data["weight"],
            axis_hint=data["axis_hint"],
            locked=data["locked"],
        )


@dataclass
class BoneMappingProfile:
    rig_id: str
    entries: list[BoneMappingEntry] = field(default_factory=list)
    created_from_frame: int = 0
    user_notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "rig_id": self.rig_id,
            "entries": [entry.to_dict() for entry in self.entries],
            "created_from_frame": self.created_from_frame,
            "user_notes": self.user_notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BoneMappingProfile":
        return cls(
            rig_id=data["rig_id"],
            entries=[BoneMappingEntry.from_dict(e) for e in data["entries"]],
            created_from_frame=data["created_from_frame"],
            user_notes=data["user_notes"],
        )


def save_bone_mapping_profile(profile: BoneMappingProfile, path: str | Path) -> None:
    write_json(path, profile.to_dict())


def load_bone_mapping_profile(path: str | Path) -> BoneMappingProfile:
    return BoneMappingProfile.from_dict(read_json(path))
