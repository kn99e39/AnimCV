"""Save/load helpers for MotionGraph (``.motion.json``)."""

from __future__ import annotations

from pathlib import Path

from common.serialization import read_json, write_json
from motion.motion_graph import MotionGraph


def save_motion_graph(graph: MotionGraph, path: str | Path) -> None:
    write_json(path, graph.to_dict())


def load_motion_graph(path: str | Path) -> MotionGraph:
    return MotionGraph.from_dict(read_json(path))
