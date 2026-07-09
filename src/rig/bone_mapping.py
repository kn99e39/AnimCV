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
class IKChainEntry:
    """A 2-bone IK chain (root -> mid -> end), e.g. shoulder->elbow->wrist
    or hip->knee->ankle. Not part of the original Architecture_v2.md
    BoneMappingEntry schema (section 6.4) -- added to support inverse
    kinematics (section 7's FK solver has no IK counterpart in v2).

    root_source/mid_source/end_source are the tracked landmark names
    for the chain's three joints; bone lengths for the IK solve are
    calibrated from their observed distances at the first valid frame,
    not from the rig's rest pose, since there's no camera calibration
    to relate rig-space lengths to 2D image-pixel distances (see
    retarget/ik_solver.py).
    """

    name: str
    root_bone: str
    mid_bone: str
    end_bone: str
    root_source: str
    mid_source: str
    end_source: str
    root_axis_hint: str | None = None
    mid_axis_hint: str | None = None
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "root_bone": self.root_bone,
            "mid_bone": self.mid_bone,
            "end_bone": self.end_bone,
            "root_source": self.root_source,
            "mid_source": self.mid_source,
            "end_source": self.end_source,
            "root_axis_hint": self.root_axis_hint,
            "mid_axis_hint": self.mid_axis_hint,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IKChainEntry":
        return cls(
            name=data["name"],
            root_bone=data["root_bone"],
            mid_bone=data["mid_bone"],
            end_bone=data["end_bone"],
            root_source=data["root_source"],
            mid_source=data["mid_source"],
            end_source=data["end_source"],
            root_axis_hint=data["root_axis_hint"],
            mid_axis_hint=data["mid_axis_hint"],
            enabled=data["enabled"],
        )


@dataclass
class BoneMappingProfile:
    rig_id: str
    entries: list[BoneMappingEntry] = field(default_factory=list)
    ik_chains: list[IKChainEntry] = field(default_factory=list)
    created_from_frame: int = 0
    user_notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "rig_id": self.rig_id,
            "entries": [entry.to_dict() for entry in self.entries],
            "ik_chains": [chain.to_dict() for chain in self.ik_chains],
            "created_from_frame": self.created_from_frame,
            "user_notes": self.user_notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BoneMappingProfile":
        return cls(
            rig_id=data["rig_id"],
            entries=[BoneMappingEntry.from_dict(e) for e in data["entries"]],
            ik_chains=[IKChainEntry.from_dict(c) for c in data.get("ik_chains", [])],
            created_from_frame=data["created_from_frame"],
            user_notes=data["user_notes"],
        )


def save_bone_mapping_profile(profile: BoneMappingProfile, path: str | Path) -> None:
    write_json(path, profile.to_dict())


def load_bone_mapping_profile(path: str | Path) -> BoneMappingProfile:
    return BoneMappingProfile.from_dict(read_json(path))
