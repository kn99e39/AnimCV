"""MotionGraph data model (Architecture_v2.md section 5).

The Motion Graph is the central intermediate representation: it stores
interpreted motion independently from Blender, FBX, or any other output
software. It is not itself the final animation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from common.types import Vec2, Vec3


@dataclass
class MotionPoint:
    semantic_name: str
    frame_index: int
    position_2d: Vec2
    position_3d: Vec3 | None
    confidence: float
    visible: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "semantic_name": self.semantic_name,
            "frame_index": self.frame_index,
            "position_2d": list(self.position_2d),
            "position_3d": list(self.position_3d) if self.position_3d else None,
            "confidence": self.confidence,
            "visible": self.visible,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MotionPoint":
        pos_3d = data["position_3d"]
        return cls(
            semantic_name=data["semantic_name"],
            frame_index=data["frame_index"],
            position_2d=tuple(data["position_2d"]),
            position_3d=tuple(pos_3d) if pos_3d is not None else None,
            confidence=data["confidence"],
            visible=data["visible"],
        )


@dataclass
class MotionTrack:
    semantic_name: str
    points: list[MotionPoint] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "semantic_name": self.semantic_name,
            "points": [point.to_dict() for point in self.points],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MotionTrack":
        return cls(
            semantic_name=data["semantic_name"],
            points=[MotionPoint.from_dict(p) for p in data["points"]],
        )


@dataclass
class MotionFrame:
    frame_index: int
    timestamp: float
    points: dict[str, MotionPoint] = field(default_factory=dict)
    importance: float = 0.0
    locked: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "points": {name: p.to_dict() for name, p in self.points.items()},
            "importance": self.importance,
            "locked": self.locked,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MotionFrame":
        return cls(
            frame_index=data["frame_index"],
            timestamp=data["timestamp"],
            points={
                name: MotionPoint.from_dict(p) for name, p in data["points"].items()
            },
            importance=data["importance"],
            locked=data["locked"],
        )


@dataclass
class MotionGraph:
    frames: list[MotionFrame] = field(default_factory=list)
    tracks: dict[str, MotionTrack] = field(default_factory=dict)
    fps: float = 0.0
    source_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fps": self.fps,
            "source_metadata": dict(self.source_metadata),
            "frames": [frame.to_dict() for frame in self.frames],
            "tracks": {name: track.to_dict() for name, track in self.tracks.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MotionGraph":
        return cls(
            frames=[MotionFrame.from_dict(f) for f in data["frames"]],
            tracks={
                name: MotionTrack.from_dict(t) for name, t in data["tracks"].items()
            },
            fps=data["fps"],
            source_metadata=dict(data["source_metadata"]),
        )
