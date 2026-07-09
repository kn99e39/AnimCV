"""PoseSequence data model (Architecture_v2.md section 3.2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PoseLandmark:
    name: str
    x: float
    y: float
    confidence: float
    visible: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "x": self.x,
            "y": self.y,
            "confidence": self.confidence,
            "visible": self.visible,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PoseLandmark":
        return cls(
            name=data["name"],
            x=data["x"],
            y=data["y"],
            confidence=data["confidence"],
            visible=data["visible"],
        )


@dataclass
class PoseFrame:
    frame_index: int
    timestamp: float
    landmarks: dict[str, PoseLandmark] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "landmarks": {name: lm.to_dict() for name, lm in self.landmarks.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PoseFrame":
        return cls(
            frame_index=data["frame_index"],
            timestamp=data["timestamp"],
            landmarks={
                name: PoseLandmark.from_dict(lm)
                for name, lm in data["landmarks"].items()
            },
        )


@dataclass
class PoseSequence:
    frames: list[PoseFrame] = field(default_factory=list)
    source_fps: float = 0.0
    landmark_schema: str = "canonical_v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_fps": self.source_fps,
            "landmark_schema": self.landmark_schema,
            "frames": [frame.to_dict() for frame in self.frames],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PoseSequence":
        return cls(
            frames=[PoseFrame.from_dict(f) for f in data["frames"]],
            source_fps=data["source_fps"],
            landmark_schema=data["landmark_schema"],
        )
