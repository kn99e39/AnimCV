"""Single-subject selection over per-frame person detections."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    bbox: tuple[float, float, float, float]
    confidence: float


class SubjectTracker:
    """Keeps the detection most consistent with an initial user selection."""

    def __init__(self, initial_box: tuple[float, float, float, float] | None = None):
        self._initial_box = initial_box
        self._previous: Detection | None = None

    def select(self, detections: list[Detection]) -> Detection | None:
        if not detections:
            return None
        reference = self._previous.bbox if self._previous else self._initial_box
        if reference is None:
            selected = max(detections, key=lambda item: item.confidence * _area(item.bbox))
        else:
            selected = max(
                detections,
                key=lambda item: 0.75 * _iou(reference, item.bbox) + 0.25 * item.confidence,
            )
        self._previous = selected
        return selected


def _area(box: tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = _area(a) + _area(b) - intersection
    return intersection / union if union else 0.0
