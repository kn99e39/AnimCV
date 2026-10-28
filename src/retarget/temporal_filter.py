"""Small, deterministic temporal filter for retarget input landmarks."""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from motion.motion_graph import MotionFrame, MotionGraph, MotionPoint


def median_filter_motion_graph(motion_graph: MotionGraph, window_size: int = 3) -> MotionGraph:
    """Return a copy with visible landmark positions median-filtered over time.

    Only complete, odd-sized neighborhoods are filtered.  Endpoints and frames
    next to an occlusion are retained exactly, so the filter cannot fabricate a
    pose across missing observations. Confidence and visibility remain raw;
    quality checks therefore still describe the original detector evidence.
    """
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError("smoothing window must be an odd positive integer")
    if window_size == 1:
        return motion_graph

    radius = window_size // 2
    frames = motion_graph.frames
    filtered = _filtered_positions(frames, window_size, radius)
    filtered_frames: list[MotionFrame] = []
    for index, frame in enumerate(frames):
        points: dict[str, MotionPoint] = {}
        for name, point in frame.points.items():
            position_2d, position_3d = filtered.get((index, name), (point.position_2d, point.position_3d))
            points[name] = MotionPoint(
                semantic_name=point.semantic_name,
                frame_index=point.frame_index,
                position_2d=position_2d,
                position_3d=position_3d,
                confidence=point.confidence,
                visible=point.visible,
            )
        filtered_frames.append(
            MotionFrame(
                frame_index=frame.frame_index,
                timestamp=frame.timestamp,
                points=points,
                importance=frame.importance,
                locked=frame.locked,
            )
        )
    return MotionGraph(
        frames=filtered_frames,
        tracks=motion_graph.tracks,
        fps=motion_graph.fps,
        source_metadata=dict(motion_graph.source_metadata),
    )


def _filtered_positions(frames: list[MotionFrame], window_size: int, radius: int) -> dict:
    """Vectorised per-landmark window medians: {(frame index, name): (position_2d, position_3d)} for filtered points.

    A point is filtered only when its whole window is present and visible (endpoints and occlusion-adjacent frames are never filtered);
    the 3D median additionally needs 3D on every neighbour, otherwise the point keeps its own 3D position.
    """
    n = len(frames)
    if n < window_size:
        return {}
    names = list(dict.fromkeys(name for frame in frames for name in frame.points))
    out = {}
    for name in names:
        pts = [frame.points.get(name) for frame in frames]
        ok = np.array([p is not None and p.visible for p in pts])
        has3 = np.array([p is not None and p.position_3d is not None for p in pts])
        p2 = np.array([p.position_2d if p is not None else (np.nan, np.nan) for p in pts], float)
        p3 = np.array([p.position_3d if (p is not None and p.position_3d is not None) else (np.nan,) * 3 for p in pts], float)
        win_ok = sliding_window_view(ok, window_size).all(axis=1)
        idx = np.flatnonzero(win_ok) + radius
        if idx.size == 0:
            continue
        med2 = np.median(sliding_window_view(p2, window_size, axis=0)[win_ok], axis=-1)
        win3 = sliding_window_view(has3, window_size).all(axis=1)[win_ok]
        med3 = np.median(sliding_window_view(p3, window_size, axis=0)[win_ok], axis=-1)
        for k, i in enumerate(idx.tolist()):
            position_3d = tuple(med3[k].tolist()) if win3[k] else pts[i].position_3d
            out[(i, name)] = (tuple(med2[k].tolist()), position_3d)
    return out
