"""Input-quality gate for video-to-rig retargeting.

Retargeting can technically emit a track even when every required landmark is
invisible: the FK solver holds its last transform to keep curves continuous.
That is useful for brief occlusion, but producing an apparently successful,
static animation from an unusable input is misleading.  This module evaluates
every applicable mapping before the solver is allowed to emit an AnimationClip.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from motion.motion_graph import MotionGraph
from rig.bone_mapping import BoneMappingEntry, BoneMappingProfile
from rig.rig_profile import RigProfile


@dataclass(frozen=True)
class RetargetQualityConfig:
    """Quality limits for mappings that are present on the target rig.

    The direction-step limit is intentionally generous: it catches impossible
    one-frame landmark swaps without rejecting ordinary fast gestures.  Callers
    may relax any limit for specialised footage.
    """

    min_visibility_rate: float = 0.60
    min_mean_confidence: float = 0.30
    max_direction_step_degrees: float = 120.0


@dataclass(frozen=True)
class MappingQuality:
    target_bone: str
    source_names: tuple[str, ...]
    mapping_mode: str
    visibility_rate: float
    mean_confidence: float
    max_direction_step_degrees: float | None
    failures: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "target_bone": self.target_bone,
            "source_names": list(self.source_names),
            "mapping_mode": self.mapping_mode,
            "visibility_rate": self.visibility_rate,
            "mean_confidence": self.mean_confidence,
            "max_direction_step_degrees": self.max_direction_step_degrees,
            "failures": list(self.failures),
        }


@dataclass(frozen=True)
class RetargetQualityReport:
    mappings: tuple[MappingQuality, ...] = ()

    @property
    def passed(self) -> bool:
        return all(not item.failures for item in self.mappings)

    def summary(self) -> str:
        if self.passed:
            return "retarget input quality passed"
        details = "; ".join(
            f"{item.target_bone}: {', '.join(item.failures)}"
            for item in self.mappings
            if item.failures
        )
        return f"retarget input quality check failed ({details})"

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "summary": self.summary(),
            "mappings": [mapping.to_dict() for mapping in self.mappings],
        }


class RetargetQualityError(ValueError):
    def __init__(self, report: RetargetQualityReport):
        self.report = report
        super().__init__(report.summary())


def assess_retarget_quality(
    motion_graph: MotionGraph,
    rig_profile: RigProfile,
    mapping_profile: BoneMappingProfile,
    config: RetargetQualityConfig | None = None,
) -> RetargetQualityReport:
    """Assess every supported mapping that will be solved for this rig."""
    config = config or RetargetQualityConfig()
    _validate_config(config)

    mappings = []
    for entry in mapping_profile.entries:
        if entry.target_bone not in rig_profile.bones:
            continue
        if entry.mapping_mode not in ("direction", "landmark", "point"):
            continue
        mappings.append(_assess_entry(motion_graph, entry, config))
    return RetargetQualityReport(mappings=tuple(mappings))


def require_retarget_quality(
    motion_graph: MotionGraph,
    rig_profile: RigProfile,
    mapping_profile: BoneMappingProfile,
    config: RetargetQualityConfig | None = None,
) -> RetargetQualityReport:
    report = assess_retarget_quality(motion_graph, rig_profile, mapping_profile, config)
    if not report.passed:
        raise RetargetQualityError(report)
    return report


def _assess_entry(
    motion_graph: MotionGraph,
    entry: BoneMappingEntry,
    config: RetargetQualityConfig,
) -> MappingQuality:
    if entry.mapping_mode == "direction":
        expected_sources = 2
    else:
        expected_sources = 1
    sources = tuple(entry.source_names)
    if len(sources) != expected_sources:
        return MappingQuality(
            target_bone=entry.target_bone,
            source_names=sources,
            mapping_mode=entry.mapping_mode,
            visibility_rate=0.0,
            mean_confidence=0.0,
            max_direction_step_degrees=None,
            failures=(f"expected {expected_sources} source landmark(s)",),
        )

    observations = []
    directions: list[tuple[int, tuple[float, float]]] = []
    for frame in motion_graph.frames:
        points = [frame.points.get(name) for name in sources]
        if any(point is None for point in points):
            observations.append((False, 0.0))
            continue
        confidence = min(point.confidence for point in points if point is not None)
        visible = all(point.visible for point in points if point is not None)
        observations.append((visible, confidence))
        if visible and entry.mapping_mode == "direction":
            direction = _unit_direction(points[0].position_2d, points[1].position_2d)  # type: ignore[union-attr]
            if direction is not None:
                directions.append((frame.frame_index, direction))

    frame_count = len(motion_graph.frames)
    visibility_rate = (
        sum(visible for visible, _ in observations) / frame_count if frame_count else 0.0
    )
    mean_confidence = (
        sum(confidence for _, confidence in observations) / frame_count if frame_count else 0.0
    )
    max_step = _max_consecutive_direction_step(directions)
    failures = []
    if visibility_rate < config.min_visibility_rate:
        failures.append(
            f"visibility {visibility_rate:.0%} is below {config.min_visibility_rate:.0%}"
        )
    if mean_confidence < config.min_mean_confidence:
        failures.append(
            f"mean confidence {mean_confidence:.2f} is below {config.min_mean_confidence:.2f}"
        )
    if max_step is not None and max_step > config.max_direction_step_degrees:
        failures.append(
            f"one-frame direction change {max_step:.1f}° exceeds "
            f"{config.max_direction_step_degrees:.1f}°"
        )
    return MappingQuality(
        target_bone=entry.target_bone,
        source_names=sources,
        mapping_mode=entry.mapping_mode,
        visibility_rate=visibility_rate,
        mean_confidence=mean_confidence,
        max_direction_step_degrees=max_step,
        failures=tuple(failures),
    )


def _unit_direction(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float] | None:
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dy)
    if length == 0.0:
        return None
    return (dx / length, dy / length)


def _max_consecutive_direction_step(
    directions: list[tuple[int, tuple[float, float]]],
) -> float | None:
    steps = []
    for (previous_index, previous), (index, current) in zip(directions, directions[1:]):
        if index != previous_index + 1:
            continue
        dot = max(-1.0, min(1.0, previous[0] * current[0] + previous[1] * current[1]))
        steps.append(math.degrees(math.acos(dot)))
    return max(steps) if steps else None


def _validate_config(config: RetargetQualityConfig) -> None:
    if not 0.0 <= config.min_visibility_rate <= 1.0:
        raise ValueError("min_visibility_rate must be between 0 and 1")
    if not 0.0 <= config.min_mean_confidence <= 1.0:
        raise ValueError("min_mean_confidence must be between 0 and 1")
    if not 0.0 <= config.max_direction_step_degrees <= 180.0:
        raise ValueError("max_direction_step_degrees must be between 0 and 180")
