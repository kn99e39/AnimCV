"""Attribution accounting for the historical Root Orientation hold (docs/55).

Compares two already-existing yaw signals on identical rows, and nothing else:

    H0-CURRENT   root_orientation_diagnostic.observe_yaw on the H0 sequence —
                 the current-frame fused shoulder+hip bilateral heading
    LEGACY-HOLD  AnimationSemantics root orientation — the historical
                 pose.root_motion.estimate_root_motion output (median + hold)

This module designs no estimator and introduces no threshold. The only
angles it names are reporting bins: 20 degrees is root_motion's own
max_yaw_step_degrees, 90 degrees is root_orientation_diagnostic's flip
definition, and 45 degrees is a requested midpoint bin. None of them feeds
back into any signal.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np

HISTORICAL_STEP_DEGREES = 20.0   # pose.root_motion default max_yaw_step_degrees
DIAGNOSTIC_FLIP_DEGREES = 90.0   # root_orientation_diagnostic flip definition
REPORT_BINS_DEGREES = (20.0, 45.0, 90.0)
AGE_BUCKETS = ("first", "2-5", "6-20", ">20")


def circular_delta_degrees(a: float, b: float) -> float:
    """Smallest absolute angle between two headings, in [0, 180]. Radians in."""
    return abs(math.degrees((a - b + math.pi) % (2.0 * math.pi) - math.pi))


def held_run_ages(held: Sequence[bool]) -> list[int | None]:
    """1-based position of each row inside its held run; None for non-held rows."""
    ages: list[int | None] = []
    age = 0
    for flag in held:
        age = age + 1 if flag else 0
        ages.append(age if flag else None)
    return ages


def age_bucket(age: int) -> str:
    if age < 1:
        raise ValueError("held-run age starts at 1")
    if age == 1:
        return "first"
    if age <= 5:
        return "2-5"
    if age <= 20:
        return "6-20"
    return ">20"


def held_runs(held: Sequence[bool]) -> list[tuple[int, int]]:
    """Maximal held runs as half-open (start_row, end_row)."""
    runs, start = [], None
    for row, flag in enumerate(held):
        if flag and start is None:
            start = row
        elif not flag and start is not None:
            runs.append((start, row))
            start = None
    if start is not None:
        runs.append((start, len(held)))
    return runs


@dataclass(frozen=True)
class YawRow:
    sequence_id: str
    frame_index: int
    current_yaw: float | None      # H0-CURRENT, radians
    current_reliable: bool         # every contributing pair observation_valid
    legacy_yaw: float | None       # LEGACY-HOLD, radians (None = UNKNOWN)
    legacy_held: bool
    held_age: int | None
    oracle_yaw: float | None       # matched oracle fused yaw, radians
    oracle_reliable: bool | None = None  # oracle pairs observation_valid (3DPW campose_valid)

    def _delta(self, a: float | None, b: float | None) -> float | None:
        return None if a is None or b is None else circular_delta_degrees(a, b)

    @property
    def held_current_gap(self) -> float | None:
        return self._delta(self.legacy_yaw, self.current_yaw)

    @property
    def current_error(self) -> float | None:
        return self._delta(self.current_yaw, self.oracle_yaw)

    @property
    def legacy_error(self) -> float | None:
        return self._delta(self.legacy_yaw, self.oracle_yaw)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "held_current_gap": self.held_current_gap,
                "current_error": self.current_error, "legacy_error": self.legacy_error}


def align_rows(
    sequence_id: str,
    current: Sequence[tuple[int, float | None, bool]],
    legacy: Sequence[tuple[int, float | None, bool]],
    oracle: Sequence[tuple[int, float | None, bool]] | None,
) -> list[YawRow]:
    """Join the three signals row by row; refuses any frame_index mismatch.

    current: (frame_index, yaw, reliable); legacy: (frame_index, yaw, held);
    oracle: (frame_index, yaw, reliable) or None when no matched oracle exists.
    """
    indices = [row[0] for row in current]
    if [row[0] for row in legacy] != indices:
        raise ValueError(f"{sequence_id}: H0-CURRENT and LEGACY-HOLD rows are not the same frames")
    if oracle is not None and [row[0] for row in oracle] != indices:
        raise ValueError(f"{sequence_id}: oracle rows are not the same frames")
    ages = held_run_ages([row[2] for row in legacy])
    return [
        YawRow(sequence_id, indices[i], current[i][1], bool(current[i][2]), legacy[i][1], bool(legacy[i][2]),
               ages[i], oracle[i][1] if oracle is not None else None,
               bool(oracle[i][2]) if oracle is not None else None)
        for i in range(len(indices))
    ]


def distribution(values: Sequence[float]) -> dict[str, Any]:
    values = [float(v) for v in values if v is not None]
    if not values:
        return {"count": 0}
    array = np.asarray(values)
    report: dict[str, Any] = {
        "count": len(values), "mean": float(array.mean()), "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)), "p95": float(np.percentile(array, 95)), "max": float(array.max()),
    }
    for bound in REPORT_BINS_DEGREES:
        report[f"frac_gt_{int(bound)}"] = float((array > bound).mean())
    return report


def run_length_distribution(lengths: Sequence[int]) -> dict[str, Any]:
    if not lengths:
        return {"count": 0}
    array = np.asarray(lengths)
    return {"count": int(len(array)), "total_rows": int(array.sum()), "median": float(np.median(array)),
            "p90": float(np.percentile(array, 90)), "max": int(array.max()),
            "histogram": {("1" if bucket == "first" else bucket): int(sum(1 for n in lengths if age_bucket(n) == bucket))
                          for bucket in AGE_BUCKETS}}


def attribute_run(rows: Sequence[YawRow], start: int, end: int) -> dict[str, Any]:
    """Run-level accounting for one held run rows[start:end] (needs oracle).

    Compares the two signals' accumulated oracle error over the whole run —
    not a single transition — and records how far the oracle body heading
    actually moved between the last accepted row and the run's end.
    """
    run = rows[start:end]
    legacy = [r.legacy_error for r in run]
    current = [r.current_error for r in run]
    if any(v is None for v in legacy + current):
        return {"start_frame": run[0].frame_index, "length": len(run), "classification": "no_oracle"}
    anchor = rows[start - 1] if start > 0 else None
    oracle_net = (circular_delta_degrees(run[-1].oracle_yaw, anchor.oracle_yaw)
                  if anchor is not None and anchor.oracle_yaw is not None else None)
    oracle_max_excursion = (max(circular_delta_degrees(r.oracle_yaw, anchor.oracle_yaw) for r in run)
                            if anchor is not None and anchor.oracle_yaw is not None else None)
    legacy_mean, current_mean = float(np.mean(legacy)), float(np.mean(current))
    if legacy_mean <= current_mean:
        classification = "hold_better_than_current"
    elif legacy[0] <= current[0]:
        classification = "stale_after_reasonable_start"
    else:
        classification = "hold_worse_from_start"
    return {
        "sequence_id": run[0].sequence_id, "start_frame": run[0].frame_index, "end_frame": run[-1].frame_index,
        "length": len(run),
        "legacy_mean_error": legacy_mean, "current_mean_error": current_mean,
        "legacy_error_first": legacy[0], "current_error_first": current[0],
        "legacy_error_last": legacy[-1], "current_error_last": current[-1],
        "legacy_error_max": float(max(legacy)),
        "oracle_net_rotation_from_anchor": oracle_net,
        "oracle_max_excursion_from_anchor": oracle_max_excursion,
        "oracle_turned_beyond_historical_step": (oracle_max_excursion is not None
                                                  and oracle_max_excursion > HISTORICAL_STEP_DEGREES),
        "classification": classification,
    }


def large_current_changes(rows: Sequence[YawRow]) -> list[dict[str, Any]]:
    """Consecutive-row H0-CURRENT changes beyond the historical step, with oracle context."""
    events = []
    for previous, row in zip(rows, rows[1:]):
        if previous.current_yaw is None or row.current_yaw is None:
            continue
        current_delta = circular_delta_degrees(row.current_yaw, previous.current_yaw)
        if current_delta <= HISTORICAL_STEP_DEGREES:
            continue
        oracle_delta = (circular_delta_degrees(row.oracle_yaw, previous.oracle_yaw)
                        if row.oracle_yaw is not None and previous.oracle_yaw is not None else None)
        events.append({
            "sequence_id": row.sequence_id, "frame_index": row.frame_index,
            "current_delta": current_delta, "oracle_delta": oracle_delta,
            "oracle_also_changed": (oracle_delta is not None and oracle_delta > HISTORICAL_STEP_DEGREES),
            "legacy_held": row.legacy_held,
            "current_error": row.current_error, "legacy_error": row.legacy_error,
        })
    return events
