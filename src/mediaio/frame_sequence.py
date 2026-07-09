"""Frame and FrameSequence types (Architecture_v2.md section 3.1).

``Frame.image`` holds raw pixel data and is never written to JSON. Only
``FrameSequenceMetadata`` (fps, dimensions, per-frame timestamps, source
path) is serializable, so a project can be inspected/replayed without
re-decoding the source video.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Frame:
    index: int
    timestamp: float
    image: np.ndarray
    width: int
    height: int


@dataclass
class FrameSequence:
    frames: list[Frame]
    fps: float
    width: int
    height: int
    source_path: str


@dataclass
class FrameSequenceMetadata:
    fps: float
    width: int
    height: int
    source_path: str
    frame_count: int
    frame_timestamps: list[float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "source_path": self.source_path,
            "frame_count": self.frame_count,
            "frame_timestamps": list(self.frame_timestamps),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FrameSequenceMetadata":
        return cls(
            fps=data["fps"],
            width=data["width"],
            height=data["height"],
            source_path=data["source_path"],
            frame_count=data["frame_count"],
            frame_timestamps=list(data["frame_timestamps"]),
        )

    @classmethod
    def from_sequence(cls, sequence: FrameSequence) -> "FrameSequenceMetadata":
        return cls(
            fps=sequence.fps,
            width=sequence.width,
            height=sequence.height,
            source_path=sequence.source_path,
            frame_count=len(sequence.frames),
            frame_timestamps=[frame.timestamp for frame in sequence.frames],
        )
