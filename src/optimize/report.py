"""Keyframe optimization data model (Architecture_v2.md section 8.4)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from retarget.solver import BoneTransformSample


@dataclass
class KeyframeCandidate:
    frame_index: int
    bone_name: str
    transform: BoneTransformSample
    importance: float
    locked: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "bone_name": self.bone_name,
            "transform": self.transform.to_dict(),
            "importance": self.importance,
            "locked": self.locked,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KeyframeCandidate":
        return cls(
            frame_index=data["frame_index"],
            bone_name=data["bone_name"],
            transform=BoneTransformSample.from_dict(data["transform"]),
            importance=data["importance"],
            locked=data["locked"],
            reason=data["reason"],
        )


@dataclass
class KeyframeOptimizationReport:
    original_key_count: int
    optimized_key_count: int
    removed_key_count: int
    max_error: float
    threshold: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_key_count": self.original_key_count,
            "optimized_key_count": self.optimized_key_count,
            "removed_key_count": self.removed_key_count,
            "max_error": self.max_error,
            "threshold": self.threshold,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KeyframeOptimizationReport":
        return cls(
            original_key_count=data["original_key_count"],
            optimized_key_count=data["optimized_key_count"],
            removed_key_count=data["removed_key_count"],
            max_error=data["max_error"],
            threshold=data["threshold"],
        )
