"""Project-level configuration (Architecture_v2.md section 10)."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

COLLAPSE_PRESETS = ("none", "light", "medium", "aggressive", "custom")


@dataclass
class ProjectConfig:
    video_path: str | None = None
    rig_path: str | None = None
    mapping_path: str | None = None
    output_path: str | None = None
    fps_override: float | None = None
    frame_start: int | None = None
    frame_end: int | None = None
    collapse_preset: str = "medium"
    collapse_threshold: float | None = None
    pose_backend: str = "mmpose"

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProjectConfig":
        return cls(**{**dataclasses.asdict(cls()), **data})

    @classmethod
    def load(cls, path: str | Path) -> "ProjectConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.from_dict(raw)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")
