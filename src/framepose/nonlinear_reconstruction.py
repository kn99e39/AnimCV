"""Bounded, deterministic nonlinear reconstruction research module.

This is an experiment-side consumer of frozen framewise poses.  It is not a
temporal neural network and it is not imported by the production Frame Pose
path.  The only temporary representation is canonical parent position,
parent-to-child direction, and a trusted local bone length.  Quaternions carry
the deterministic swing from the canonical +Z axis to a bone direction; they
never carry roll, twist, or target-rig local axes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from common.canonical_pose import BONES, JOINT_INDEX, JOINT_NAMES


RECONSTRUCTION_SCHEMA = "animcv_bounded_nonlinear_reconstruction_v1"
CANONICAL_AXIS = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
EPSILON = 1e-10


@dataclass(frozen=True)
class Gap:
    """One contiguous unreliable interval within exactly one sequence."""

    sequence_id: str
    joint: int
    rows: tuple[int, ...]
    left_anchor: int
    right_anchor: int
    left_support: tuple[int, ...]
    right_support: tuple[int, ...]

    @property
    def length(self) -> int:
        return len(self.rows)

    @property
    def length_bin(self) -> str:
        if self.length <= 2:
            return "1-2"
        if self.length <= 5:
            return "3-5"
        return ">5"

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence_id": self.sequence_id,
            "joint": JOINT_NAMES[self.joint],
            "rows": list(self.rows),
            "left_anchor": self.left_anchor,
            "right_anchor": self.right_anchor,
            "left_support": list(self.left_support),
            "right_support": list(self.right_support),
            "length": self.length,
            "length_bin": self.length_bin,
        }


@dataclass
class ReconstructionResult:
    recovered: np.ndarray
    linear_control: np.ndarray
    changed: np.ndarray
    resolved: np.ndarray
    gaps: list[Gap]
    resolved_gap_keys: set[tuple[str, int, tuple[int, ...]]]
    unresolved: list[dict[str, Any]]
    bone_records: list[dict[str, Any]]


def in_frame_mask(observation: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Exact structural in-frame mask; this does not infer occlusion."""
    xy = np.asarray(observation, dtype=np.float64)[..., :2]
    return (np.asarray(valid, dtype=bool) & np.isfinite(xy).all(axis=-1)
            & (xy[..., 0] >= 0.0) & (xy[..., 0] <= 1.0)
            & (xy[..., 1] >= 0.0) & (xy[..., 1] <= 1.0))


def sequence_order(sequence_ids: Sequence[str], timestamps: Sequence[float] | None = None,
                   frame_indices: Sequence[int] | None = None) -> dict[str, list[int]]:
    """Return deterministic sequence-local order; no row can cross a sequence."""
    timestamps = timestamps if timestamps is not None else [float(i) for i in range(len(sequence_ids))]
    frame_indices = frame_indices if frame_indices is not None else list(range(len(sequence_ids)))
    grouped: dict[str, list[int]] = {}
    for row, sequence in enumerate(sequence_ids):
        grouped.setdefault(str(sequence), []).append(row)
    for sequence, rows in grouped.items():
        grouped[sequence] = sorted(rows, key=lambda row: (
            float(timestamps[row]), int(frame_indices[row]), row))
    return grouped


def find_gaps(usable: np.ndarray, sequence_ids: Sequence[str], *,
              timestamps: Sequence[float] | None = None,
              frame_indices: Sequence[int] | None = None) -> list[Gap]:
    """Find all false runs and retain their adjacent/support rows.

    A gap is reported even when an anchor or derivative support row is absent;
    the reconstruction driver then records it as unresolved.  This makes lack
    of support visible instead of silently turning it into an extrapolation.
    """
    mask = np.asarray(usable, dtype=bool)
    if mask.ndim != 2 or mask.shape[1] != len(JOINT_NAMES):
        raise ValueError("usable must have shape (frames, canonical joints)")
    grouped = sequence_order(sequence_ids, timestamps, frame_indices)
    gaps: list[Gap] = []
    for sequence, rows in grouped.items():
        for joint in range(mask.shape[1]):
            local = 0
            while local < len(rows):
                if mask[rows[local], joint]:
                    local += 1
                    continue
                start = local
                while local < len(rows) and not mask[rows[local], joint]:
                    local += 1
                end = local
                left_anchor = rows[start - 1] if start > 0 else -1
                right_anchor = rows[end] if end < len(rows) else -1
                left_support = tuple(rows[max(0, start - 3):start - 1][::-1]) if start > 1 else tuple()
                right_support = tuple(rows[end + 1:min(len(rows), end + 3)]) if end + 1 < len(rows) else tuple()
                gaps.append(Gap(sequence, joint, tuple(rows[start:end]), left_anchor,
                                right_anchor, left_support, right_support))
    return gaps


def _lagrange_derivative(times: np.ndarray, values: np.ndarray, at: int) -> np.ndarray | None:
    if len(times) < 2 or not np.isfinite(times).all() or not np.isfinite(values).all():
        return None
    if len(times) == 2:
        delta = float(times[1] - times[0])
        return (values[1] - values[0]) / delta if delta > EPSILON else None
    result = np.zeros(values.shape[1:], dtype=np.float64)
    for i in range(len(times)):
        denominator = 1.0
        numerator = 0.0
        for j in range(len(times)):
            if i == j:
                continue
            denominator *= float(times[i] - times[j])
            product = 1.0
            for m in range(len(times)):
                if m != i and m != j:
                    product *= float(times[at] - times[m])
            numerator += product
        if abs(denominator) <= EPSILON:
            return None
        result += values[i] * (numerator / denominator)
    return result


def timestamp_aware_tangent(values: np.ndarray, times: np.ndarray, anchor: int,
                            support: Sequence[int], *, side: str) -> np.ndarray | None:
    """Fixed endpoint derivative using up to two same-side reliable samples."""
    if side not in {"left", "right"}:
        raise ValueError("side must be 'left' or 'right'")
    support = list(support)
    if side == "left":
        rows = list(reversed(support[-2:])) + [anchor]
    else:
        rows = [anchor] + support[:2]
    rows = [row for row in rows if row >= 0]
    if len(rows) < 2:
        return None
    # For the left side rows are chronological after the reversal; for the
    # right side they already are.  The derivative is evaluated at the anchor.
    local_times = np.asarray([times[row] for row in rows], dtype=np.float64)
    local_values = np.asarray([values[row] for row in rows], dtype=np.float64)
    if len(rows) == 2:
        delta = float(local_times[1] - local_times[0])
        if side == "left":
            delta = float(times[anchor] - times[rows[0]])
            return (values[anchor] - values[rows[0]]) / delta if delta > EPSILON else None
        return (values[rows[1]] - values[anchor]) / delta if delta > EPSILON else None
    return _lagrange_derivative(local_times, local_values, len(rows) - 1 if side == "left" else 0)


def cubic_hermite(p0: np.ndarray, p1: np.ndarray, m0: np.ndarray, m1: np.ndarray,
                  t0: float, t1: float, t: float) -> np.ndarray:
    """One fixed timestamp-aware cubic Hermite segment."""
    duration = float(t1 - t0)
    if not np.isfinite(duration) or duration <= EPSILON:
        raise ValueError("Hermite endpoints must have strictly increasing timestamps")
    u = float((t - t0) / duration)
    h00 = 2.0 * u ** 3 - 3.0 * u ** 2 + 1.0
    h10 = u ** 3 - 2.0 * u ** 2 + u
    h01 = -2.0 * u ** 3 + 3.0 * u ** 2
    h11 = u ** 3 - u ** 2
    return h00 * p0 + h10 * duration * m0 + h01 * p1 + h11 * duration * m1


# ------------------------------ quaternion swing / SQUAD -----------------

def quaternion_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if norm <= EPSILON or not np.isfinite(norm):
        raise ValueError("zero or non-finite quaternion")
    return q / norm


def quaternion_hemisphere(q: np.ndarray, reference: np.ndarray) -> np.ndarray:
    q = quaternion_normalize(q)
    return -q if float(np.dot(q, reference)) < 0.0 else q


def swing_quaternion(direction: np.ndarray, axis: np.ndarray = CANONICAL_AXIS) -> np.ndarray:
    """Deterministic shortest-arc swing from ``axis`` to ``direction``."""
    source = np.asarray(axis, dtype=np.float64)
    target = np.asarray(direction, dtype=np.float64)
    source /= max(float(np.linalg.norm(source)), EPSILON)
    target /= max(float(np.linalg.norm(target)), EPSILON)
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if dot > -1.0 + 1e-8:
        return quaternion_normalize(np.concatenate(([1.0 + dot], np.cross(source, target))))
    # Anti-parallel vectors have infinitely many shortest arcs.  Choose the
    # basis vector least aligned with source, then cross in a fixed order.
    basis = np.eye(3)[int(np.argmin(np.abs(source)))]
    rotation_axis = np.cross(source, basis)
    rotation_axis /= max(float(np.linalg.norm(rotation_axis)), EPSILON)
    return np.asarray([0.0, *rotation_axis], dtype=np.float64)


def quaternion_multiply(q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
    w0, x0, y0, z0 = q0
    w1, x1, y1, z1 = q1
    return np.asarray([w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
                       w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
                       w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
                       w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1], dtype=np.float64)


def quaternion_conjugate(q: np.ndarray) -> np.ndarray:
    return np.asarray([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quaternion_rotate(q: np.ndarray, vector: np.ndarray) -> np.ndarray:
    unit = quaternion_normalize(q)
    pure = np.asarray([0.0, *np.asarray(vector, dtype=np.float64)], dtype=np.float64)
    return quaternion_multiply(quaternion_multiply(unit, pure), quaternion_conjugate(unit))[1:]


def quaternion_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    a = quaternion_normalize(q0)
    b = quaternion_hemisphere(q1, a)
    if t <= 0.0:
        return a
    if t >= 1.0:
        return b
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if dot > 1.0 - 1e-8:
        return quaternion_normalize((1.0 - t) * a + t * b)
    angle = math.acos(dot)
    sine = math.sin(angle)
    return (math.sin((1.0 - t) * angle) / sine) * a + (math.sin(t * angle) / sine) * b


def quaternion_log(q: np.ndarray) -> np.ndarray:
    unit = quaternion_normalize(q)
    vector = unit[1:]
    norm = float(np.linalg.norm(vector))
    if norm <= EPSILON:
        return np.zeros(3, dtype=np.float64)
    angle = math.atan2(norm, float(unit[0]))
    return vector * (angle / norm)


def quaternion_exp(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    angle = float(np.linalg.norm(vector))
    if angle <= EPSILON:
        return quaternion_normalize(np.asarray([1.0, *vector], dtype=np.float64))
    return np.asarray([math.cos(angle), *(math.sin(angle) * vector / angle)], dtype=np.float64)


def squad_control(q_prev: np.ndarray, q: np.ndarray, q_next: np.ndarray) -> np.ndarray:
    center = quaternion_normalize(q)
    previous = quaternion_hemisphere(q_prev, center)
    following = quaternion_hemisphere(q_next, center)
    inverse = quaternion_conjugate(center)
    tangent = -0.25 * (quaternion_log(quaternion_multiply(inverse, following))
                       + quaternion_log(quaternion_multiply(inverse, previous)))
    return quaternion_normalize(quaternion_multiply(center, quaternion_exp(tangent)))


def quaternion_squad(q0: np.ndarray, q1: np.ndarray, a0: np.ndarray, a1: np.ndarray,
                     t: float) -> np.ndarray:
    if t <= 0.0:
        return quaternion_normalize(q0)
    if t >= 1.0:
        return quaternion_hemisphere(q1, q0)
    first = quaternion_slerp(q0, q1, t)
    second = quaternion_slerp(a0, a1, t)
    return quaternion_slerp(first, second, 2.0 * t * (1.0 - t))


def _unit_direction(vector: np.ndarray) -> np.ndarray | None:
    vector = np.asarray(vector, dtype=np.float64)
    length = float(np.linalg.norm(vector))
    if length <= EPSILON or not np.isfinite(length):
        return None
    return vector / length


def robust_local_bone_length(poses: np.ndarray, usable: np.ndarray, parent: int, child: int,
                             rows: Iterable[int]) -> float | None:
    """Median of trusted positive local lengths; rule fixed before evaluation."""
    lengths = []
    for row in rows:
        if not (usable[row, parent] and usable[row, child]):
            continue
        length = float(np.linalg.norm(poses[row, child] - poses[row, parent]))
        if np.isfinite(length) and length > EPSILON:
            lengths.append(length)
    if len(lengths) < 2:
        return None
    return float(np.median(np.asarray(lengths, dtype=np.float64)))


def _linear_segment(p0: np.ndarray, p1: np.ndarray, t0: float, t1: float, t: float) -> np.ndarray:
    duration = float(t1 - t0)
    if duration <= EPSILON:
        raise ValueError("linear endpoints must have increasing timestamps")
    return p0 + ((float(t) - t0) / duration) * (p1 - p0)


def _record_unresolved(unresolved: list[dict[str, Any]], gap: Gap, reason: str) -> None:
    unresolved.append({**gap.to_dict(), "reason": reason})


def _gap_key(gap: Gap) -> tuple[str, int, tuple[int, ...]]:
    return gap.sequence_id, gap.joint, gap.rows


def reconstruct_nonlinear(poses: np.ndarray, usable: np.ndarray, timestamps: np.ndarray,
                          sequence_ids: Sequence[str], *, frame_indices: Sequence[int] | None = None,
                          joint_names: Sequence[str] = JOINT_NAMES) -> ReconstructionResult:
    """Reconstruct structural gaps with Hermite positions + SQUAD directions.

    The input pose is read-only in spirit: all changes are written to a copy,
    and ``changed`` is false outside a structural gap.  A gap is unresolved
    when either side, the required derivative/swing support, or a local robust
    bone length is unavailable.  There is no extrapolation or linear fallback.
    """
    base = np.asarray(poses, dtype=np.float64)
    mask = np.asarray(usable, dtype=bool)
    times = np.asarray(timestamps, dtype=np.float64)
    if base.ndim != 3 or base.shape[1:] != (len(JOINT_NAMES), 3):
        raise ValueError("poses must have shape (frames, canonical joints, 3)")
    if mask.shape != base.shape[:2] or times.shape != (len(base),):
        raise ValueError("usable/timestamps shape mismatch")
    if len(joint_names) != len(JOINT_NAMES) or tuple(joint_names) != tuple(JOINT_NAMES):
        raise ValueError("only the canonical joint order is supported")
    gaps = find_gaps(mask, sequence_ids, timestamps=times, frame_indices=frame_indices)
    recovered = base.copy()
    linear = base.copy()
    changed = np.zeros(mask.shape, dtype=bool)
    resolved = mask.copy()
    unresolved: list[dict[str, Any]] = []
    resolved_keys: set[tuple[str, int, tuple[int, ...]]] = set()
    bone_records: list[dict[str, Any]] = []

    by_joint: dict[int, list[Gap]] = {}
    for gap in gaps:
        by_joint.setdefault(gap.joint, []).append(gap)

    # Historical control only: materialize the direct timestamp-aware linear
    # segment for every two-sided finite gap before checking nonlinear support.
    # It is never copied into ``recovered`` when Hermite/SQUAD is unresolved.
    for gap in gaps:
        if gap.left_anchor < 0 or gap.right_anchor < 0:
            continue
        left, right = gap.left_anchor, gap.right_anchor
        if not (np.isfinite(base[[left, right], gap.joint]).all()
                and np.isfinite(times[[left, right]]).all()
                and times[right] > times[left]):
            continue
        for row in gap.rows:
            linear[row, gap.joint] = _linear_segment(base[left, gap.joint], base[right, gap.joint],
                                                      times[left], times[right], times[row])

    # Pelvis has no parent bone and is the only joint reconstructed directly by
    # Hermite position.  All descendants are then processed in BONES order so
    # a recovered parent can support a descendant's position.
    root = JOINT_INDEX["pelvis"]
    for gap in by_joint.get(root, []):
        if gap.left_anchor < 0 or gap.right_anchor < 0:
            _record_unresolved(unresolved, gap, "missing_two_sided_anchor")
            continue
        left_tangent = timestamp_aware_tangent(base[:, root], times, gap.left_anchor,
                                               gap.left_support, side="left")
        right_tangent = timestamp_aware_tangent(base[:, root], times, gap.right_anchor,
                                                gap.right_support, side="right")
        if left_tangent is None or right_tangent is None:
            _record_unresolved(unresolved, gap, "insufficient_hermite_support")
            continue
        left, right = gap.left_anchor, gap.right_anchor
        if not (np.isfinite(base[[left, right], root]).all() and times[right] > times[left]):
            _record_unresolved(unresolved, gap, "invalid_hermite_endpoint")
            continue
        for row in gap.rows:
            recovered[row, root] = cubic_hermite(base[left, root], base[right, root],
                                                 left_tangent, right_tangent,
                                                 times[left], times[right], times[row])
            changed[row, root] = True
            resolved[row, root] = True
        resolved_keys.add(_gap_key(gap))

    for parent_name, child_name in BONES:
        parent, child = JOINT_INDEX[parent_name], JOINT_INDEX[child_name]
        for gap in by_joint.get(child, []):
            if gap.left_anchor < 0 or gap.right_anchor < 0:
                _record_unresolved(unresolved, gap, "missing_two_sided_anchor")
                continue
            left, right = gap.left_anchor, gap.right_anchor
            if not (mask[left, parent] and mask[right, parent]
                    and mask[left, child] and mask[right, child]):
                _record_unresolved(unresolved, gap, "anchor_parent_or_child_unreliable")
                continue
            if not all(resolved[row, parent] for row in gap.rows):
                _record_unresolved(unresolved, gap, "parent_gap_unresolved")
                continue
            left_vector = _unit_direction(base[left, child] - base[left, parent])
            right_vector = _unit_direction(base[right, child] - base[right, parent])
            if left_vector is None or right_vector is None:
                _record_unresolved(unresolved, gap, "degenerate_anchor_bone")
                continue
            prev_row = gap.left_support[0] if gap.left_support else -1
            next_row = gap.right_support[0] if gap.right_support else -1
            if prev_row < 0 or next_row < 0:
                _record_unresolved(unresolved, gap, "insufficient_squad_support")
                continue
            prev_vector = _unit_direction(base[prev_row, child] - base[prev_row, parent])
            next_vector = _unit_direction(base[next_row, child] - base[next_row, parent])
            if prev_vector is None or next_vector is None:
                _record_unresolved(unresolved, gap, "degenerate_squad_support_bone")
                continue
            length_rows = (gap.left_support[-2:] + (left,) + (right,) + gap.right_support[:2])
            length = robust_local_bone_length(base, mask, parent, child, length_rows)
            if length is None:
                _record_unresolved(unresolved, gap, "insufficient_trusted_bone_lengths")
                continue
            q0 = swing_quaternion(left_vector)
            q1 = quaternion_hemisphere(swing_quaternion(right_vector), q0)
            qprev = quaternion_hemisphere(swing_quaternion(prev_vector), q0)
            qnext = quaternion_hemisphere(swing_quaternion(next_vector), q1)
            a0 = squad_control(qprev, q0, q1)
            a1 = squad_control(q0, q1, qnext)
            if not times[right] > times[left]:
                _record_unresolved(unresolved, gap, "non_increasing_anchor_timestamps")
                continue
            for row in gap.rows:
                fraction = float((times[row] - times[left]) / (times[right] - times[left]))
                orientation = quaternion_squad(q0, q1, a0, a1, fraction)
                direction = quaternion_rotate(orientation, CANONICAL_AXIS)
                direction = direction / max(float(np.linalg.norm(direction)), EPSILON)
                recovered[row, child] = recovered[row, parent] + direction * length
                changed[row, child] = True
                resolved[row, child] = True
            resolved_keys.add(_gap_key(gap))
            bone_records.append({
                **gap.to_dict(), "parent": parent_name, "child": child_name,
                "trusted_length_m": length, "direction_interpolator": "SQUAD_swing_only",
                "position_interpolator": "parent_position_plus_direction_times_length",
            })
    if np.any(changed & mask):
        raise AssertionError("reconstruction changed a structurally usable joint")
    return ReconstructionResult(recovered, linear, changed, resolved, gaps, resolved_keys,
                                unresolved, bone_records)


def _summary(values: Sequence[float]) -> dict[str, Any]:
    data = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if not len(data):
        return {"count": 0, "mean": None, "median": None, "p95": None}
    return {"count": int(len(data)), "mean": float(np.mean(data)),
            "median": float(np.median(data)), "p95": float(np.quantile(data, 0.95))}


def _error_mm(pose: np.ndarray, target: np.ndarray, valid: np.ndarray) -> np.ndarray:
    errors = np.linalg.norm(pose - target, axis=-1) * 1000.0
    return np.where(valid & np.isfinite(errors), errors, np.nan)


def _limb_for_joint(name: str) -> str:
    if name.startswith("left_"):
        return "left_limb" if any(token in name for token in ("hip", "knee", "ankle")) else "left_arm"
    if name.startswith("right_"):
        return "right_limb" if any(token in name for token in ("hip", "knee", "ankle")) else "right_arm"
    return "torso"


def evaluate_reconstruction(original: np.ndarray, recovered: np.ndarray, linear_control: np.ndarray,
                            target: np.ndarray, target_valid: np.ndarray, changed: np.ndarray,
                            gaps: Sequence[Gap], timestamps: np.ndarray,
                            *, failure_labels: np.ndarray | None = None,
                            bone_records: Sequence[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Report H0/nonlinear/linear errors and bounded continuity evidence."""
    original = np.asarray(original, dtype=np.float64)
    recovered = np.asarray(recovered, dtype=np.float64)
    linear_control = np.asarray(linear_control, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    valid = np.asarray(target_valid, dtype=bool)
    changed = np.asarray(changed, dtype=bool)
    h0_error = _error_mm(original, target, valid)
    nonlinear_error = _error_mm(recovered, target, valid)
    linear_error = _error_mm(linear_control, target, valid)
    changed_scored = changed & valid
    delta = nonlinear_error - h0_error
    linear_delta = linear_error - h0_error

    def stats_for(mask: np.ndarray) -> dict[str, Any]:
        before = h0_error[mask]
        after = nonlinear_error[mask]
        difference = after - before
        return {
            "count": int(np.isfinite(before).sum()),
            "h0_mm": _summary(before.ravel().tolist()),
            "nonlinear_mm": _summary(after.ravel().tolist()),
            "linear_control_mm": _summary(linear_error[mask].ravel().tolist()),
            "delta_nonlinear_minus_h0_mm": _summary(difference.ravel().tolist()),
            "delta_linear_minus_h0_mm": _summary(linear_delta[mask].ravel().tolist()),
            "improved_count": int(np.count_nonzero(np.isfinite(difference) & (difference < 0.0))),
            "worsened_count": int(np.count_nonzero(np.isfinite(difference) & (difference > 0.0))),
            "unchanged_count": int(np.count_nonzero(np.isfinite(difference) & (difference == 0.0))),
            "p95_damage_mm": float(np.quantile(difference[np.isfinite(difference)], 0.95)) if np.isfinite(difference).any() else None,
        }

    by_joint = {name: stats_for(changed_scored[:, index]) for index, name in enumerate(JOINT_NAMES)}
    by_limb = {}
    for limb in sorted({_limb_for_joint(name) for name in JOINT_NAMES}):
        joint_mask = np.asarray([_limb_for_joint(name) == limb for name in JOINT_NAMES])
        by_limb[limb] = stats_for(changed_scored & joint_mask[None, :])

    gap_metrics = []
    for gap in gaps:
        joint = gap.joint
        rows = np.asarray(gap.rows, dtype=np.int64)
        scores = valid[rows, joint]
        before = h0_error[rows, joint]
        after = nonlinear_error[rows, joint]
        gap_metrics.append({
            **gap.to_dict(),
            "resolved": bool(np.all(changed[rows, joint])),
            "scored_count": int(np.isfinite(before[scores]).sum()),
            "h0_mm": _summary(before[scores].tolist()),
            "nonlinear_mm": _summary(after[scores].tolist()),
            "delta_mm": _summary((after[scores] - before[scores]).tolist()),
        })

    endpoint_position: list[float] = []
    velocity_jumps: list[float] = []
    continuity_records: list[dict[str, Any]] = []
    for gap in gaps:
        if gap.left_anchor < 0 or gap.right_anchor < 0 or not gap.rows:
            continue
        joint = gap.joint
        first, last = gap.rows[0], gap.rows[-1]
        left_error = float(np.linalg.norm(recovered[gap.left_anchor, joint] - original[gap.left_anchor, joint]) * 1000.0)
        right_error = float(np.linalg.norm(recovered[gap.right_anchor, joint] - original[gap.right_anchor, joint]) * 1000.0)
        endpoint_value = max(left_error, right_error)
        endpoint_position.append(endpoint_value)
        failure_label = "UNLABELED"
        if failure_labels is not None:
            failure_label = str(np.asarray(failure_labels, dtype=object)[first, joint] or "UNLABELED")
        continuity_record = {
            "joint": JOINT_NAMES[joint], "limb": _limb_for_joint(JOINT_NAMES[joint]),
            "gap_length_bin": gap.length_bin, "failure_label": failure_label,
            "endpoint_position_error_mm": endpoint_value, "velocity_jump_mm_per_s": [],
        }
        if gap.left_support:
            before_row = gap.left_support[0]
            dt_a = float(timestamps[gap.left_anchor] - timestamps[before_row])
            dt_b = float(timestamps[first] - timestamps[gap.left_anchor])
            if dt_a > EPSILON and dt_b > EPSILON:
                jump_vector = ((recovered[first, joint] - recovered[gap.left_anchor, joint]) / dt_b
                               - (original[gap.left_anchor, joint] - original[before_row, joint]) / dt_a)
                jump = float(np.linalg.norm(jump_vector) * 1000.0)
                velocity_jumps.append(jump)
                continuity_record["velocity_jump_mm_per_s"].append(jump)
        if gap.right_support:
            after_row = gap.right_support[0]
            dt_a = float(timestamps[gap.right_anchor] - timestamps[last])
            dt_b = float(timestamps[after_row] - timestamps[gap.right_anchor])
            if dt_a > EPSILON and dt_b > EPSILON:
                jump_vector = ((original[after_row, joint] - original[gap.right_anchor, joint]) / dt_b
                               - (recovered[gap.right_anchor, joint] - recovered[last, joint]) / dt_a)
                jump = float(np.linalg.norm(jump_vector) * 1000.0)
                velocity_jumps.append(jump)
                continuity_record["velocity_jump_mm_per_s"].append(jump)
        continuity_records.append(continuity_record)

    def continuity_group(key: str, value_key: str) -> dict[str, Any]:
        grouped: dict[str, list[float]] = {}
        for record in continuity_records:
            values = record[value_key]
            if not isinstance(values, list):
                values = [values]
            grouped.setdefault(str(record[key]), []).extend(values)
        return {name: _summary(values) for name, values in sorted(grouped.items())}

    gap_bin_masks: dict[str, np.ndarray] = {
        name: np.zeros_like(changed, dtype=bool) for name in ("1-2", "3-5", ">5")}
    for gap in gaps:
        gap_bin_masks[gap.length_bin][np.asarray(gap.rows), gap.joint] = True

    bone_length_errors: list[float] = []
    bone_length_by_child: dict[str, list[float]] = {}
    for record in bone_records or []:
        parent = JOINT_INDEX[str(record["parent"])]
        child = JOINT_INDEX[str(record["child"])]
        expected = float(record["trusted_length_m"])
        rows = np.asarray(record["rows"], dtype=np.int64)
        values = (np.linalg.norm(recovered[rows, child] - recovered[rows, parent], axis=-1)
                  - expected) * 1000.0
        values = values[np.isfinite(values)]
        bone_length_errors.extend(values.tolist())
        bone_length_by_child.setdefault(str(record["child"]), []).extend(values.tolist())

    by_failure = {}
    if failure_labels is not None:
        labels = np.asarray(failure_labels, dtype=object)
        if labels.shape != changed.shape:
            raise ValueError("failure_labels must have shape (frames, joints)")
        for label in sorted({str(value) for value in labels.ravel() if str(value)}):
            by_failure[label] = stats_for(changed_scored & (labels == label))

    return {
        "schema": "animcv_nonlinear_reconstruction_metrics_v1",
        "scored_regime": "benchmark_detector_observation",
        "all_target_valid": stats_for(valid),
        "reconstructable_target_valid": stats_for(changed_scored),
        "by_joint": by_joint,
        "by_limb": by_limb,
        "gaps": gap_metrics,
        "gap_length_bins": {name: stats_for(changed_scored & gap_bin_masks[name])
                            for name in ("1-2", "3-5", ">5")},
        "by_failure_label": by_failure,
        "continuity": {
            "endpoint_position_error_mm": _summary(endpoint_position),
            "velocity_jump_mm_per_s": _summary(velocity_jumps),
            "position_by_joint": continuity_group("joint", "endpoint_position_error_mm"),
            "position_by_limb": continuity_group("limb", "endpoint_position_error_mm"),
            "position_by_gap_length_bin": continuity_group("gap_length_bin", "endpoint_position_error_mm"),
            "position_by_failure_label": continuity_group("failure_label", "endpoint_position_error_mm"),
            "velocity_by_joint": continuity_group("joint", "velocity_jump_mm_per_s"),
            "velocity_by_limb": continuity_group("limb", "velocity_jump_mm_per_s"),
            "velocity_by_gap_length_bin": continuity_group("gap_length_bin", "velocity_jump_mm_per_s"),
            "velocity_by_failure_label": continuity_group("failure_label", "velocity_jump_mm_per_s"),
        },
        "no_change_outside_changed": bool(np.array_equal(original[~changed], recovered[~changed])),
        "improved_count": int(np.count_nonzero(np.isfinite(delta) & changed & (delta < 0.0))),
        "worsened_count": int(np.count_nonzero(np.isfinite(delta) & changed & (delta > 0.0))),
        "median_delta_mm": float(np.nanmedian(delta[changed_scored])) if np.any(changed_scored) else None,
        "p95_damage_mm": float(np.nanquantile(delta[changed_scored], 0.95)) if np.any(changed_scored) else None,
        "bone_length": {
            "rule": "median of >=2 finite positive trusted-anchor lengths from A-2,A-1,A,B,B+1,B+2",
            "error_mm": _summary(bone_length_errors),
            "by_child": {name: _summary(values) for name, values in sorted(bone_length_by_child.items())},
        },
    }
