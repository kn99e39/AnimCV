#!/usr/bin/env python3
"""Export a compact, labelled real-scene review for temporal-evidence replay.

These videos are evidence displays, not a temporal production output.  Every
frame corresponds to one retained FrameBank timestamp in the fixed t±2 context
used by ``diagnose_temporal_context_evidence.py``; no display interpolation is
performed here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from common.canonical_pose import JOINT_INDEX, JOINT_NAMES
from common.serialization import write_json
from framepose.bank import load_bank
from framepose.replay_provenance import verify_source_identity
from diagnose_temporal_context_evidence import (
    FIXED_BANK_DIGEST,
    FIXED_EVALUATION_SHA256,
    FIXED_PREDICTION_SHA256,
    build_cohorts,
    failure_mask,
    interpolate_recovery,
    local_neighbor_rows,
    nearest_temporal_support,
    sequence_rows,
    _sample_time,
)


PANEL_W, PANEL_H = 480, 270
VIDEO_W, VIDEO_H = PANEL_W * 3, PANEL_H * 2
SKELETON_BONES = (
    ("pelvis", "left_hip"), ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("pelvis", "right_hip"), ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ("pelvis", "spine"), ("spine", "thorax"), ("thorax", "neck"), ("neck", "head"),
    ("thorax", "left_shoulder"), ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"), ("thorax", "right_shoulder"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
)
DEFAULT_SELECTION = (
    ("owner_good_case", "3dpw:downtown_arguing_00:actor0#000550", "left_elbow"),
    ("owner_jitter_note", "3dpw:downtown_sitOnStairs_00:actor0#000045", "left_knee"),
    ("owner_tracking_collapse_note", "3dpw:downtown_bar_00:actor0#000985", "right_elbow"),
    ("owner_implausible_articulation_note", "3dpw:downtown_warmWelcome_00:actor0#000330", "right_knee"),
    ("owner_subject_absence_note", "3dpw:downtown_car_00:actor1#000507", "right_knee"),
    ("owner_ankle_foot_note", "3dpw:downtown_windowShopping_00:actor0#000588", "right_ankle"),
    ("deterministic_T4_observation_loss", "3dpw:downtown_enterShop_00:actor0#001128", "left_ankle"),
    ("deterministic_T1_h0_jitter", "3dpw:downtown_runForBus_01:actor1#000150", "neck"),
    ("deterministic_T2_observation_degradation", "3dpw:downtown_stairs_00:actor0#001088", "right_ankle"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fit_image(image: np.ndarray) -> tuple[np.ndarray, float, int, int]:
    h, w = image.shape[:2]
    scale = min(PANEL_W / w, (PANEL_H - 30) / h)
    resized = cv2.resize(image, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    result = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
    left, top = (PANEL_W - resized.shape[1]) // 2, 30 + (PANEL_H - 30 - resized.shape[0]) // 2
    result[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
    return result, scale, left, top


def _panel(canvas: np.ndarray, index: int, title: str) -> np.ndarray:
    x, y = (index % 3) * PANEL_W, (index // 3) * PANEL_H
    tile = canvas[y:y + PANEL_H, x:x + PANEL_W]
    cv2.rectangle(tile, (0, 0), (PANEL_W - 1, PANEL_H - 1), (90, 90, 90), 1)
    cv2.rectangle(tile, (0, 0), (PANEL_W - 1, 30), (20, 20, 20), -1)
    cv2.putText(tile, title[:60], (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (245, 245, 245), 1, cv2.LINE_AA)
    return tile


def _draw_skeleton(tile: np.ndarray, pose: np.ndarray | None, target_joint: int,
                   bounds: tuple[float, float, float, float], color: tuple[int, int, int]) -> None:
    if pose is None or not np.isfinite(pose).all():
        cv2.putText(tile, "Unavailable at this timestamp", (20, 145), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (140, 140, 140), 1, cv2.LINE_AA)
        return
    points = np.column_stack((pose[:, 0], -pose[:, 2]))
    x0, x1, y0, y1 = bounds
    scale = min((PANEL_W - 70) / max(x1 - x0, 1e-9), (PANEL_H - 75) / max(y1 - y0, 1e-9))
    base_x, base_y = 35, 45
    def pixel(point):
        return (int(round(base_x + (point[0] - x0) * scale)),
                int(round(base_y + (y1 - point[1]) * scale)))
    for first, second in SKELETON_BONES:
        a, b = JOINT_INDEX[first], JOINT_INDEX[second]
        cv2.line(tile, pixel(points[a]), pixel(points[b]), (120, 120, 120), 2, cv2.LINE_AA)
    for index, point in enumerate(points):
        active = index == target_joint
        cv2.circle(tile, pixel(point), 7 if active else 3, color if active else (185, 185, 185), -1, cv2.LINE_AA)
    cv2.putText(tile, "+X right / +Z up", (8, PANEL_H - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.36, (140, 140, 140), 1, cv2.LINE_AA)


def _bounds(poses: list[np.ndarray]) -> tuple[float, float, float, float]:
    values = np.concatenate([np.column_stack((pose[:, 0], -pose[:, 2])) for pose in poses if np.isfinite(pose).all()])
    mins, maxs = values.min(axis=0), values.max(axis=0)
    center = (mins + maxs) / 2.0
    span = max(float((maxs - mins).max()), 0.1) * 1.3
    return (float(center[0] - span / 2), float(center[0] + span / 2),
            float(center[1] - span / 2), float(center[1] + span / 2))


def _draw_trajectory(tile: np.ndarray, rows: list[int], current: int, joint: int,
                     observation: np.ndarray, in_frame: np.ndarray) -> None:
    cv2.putText(tile, "observed 2D target-joint trajectory", (12, 51), cv2.FONT_HERSHEY_SIMPLEX,
                0.46, (230, 230, 230), 1, cv2.LINE_AA)
    points = []
    for row in rows:
        if in_frame[row, joint]:
            xy = observation[row, joint, :2]
            points.append((int(35 + xy[0] * (PANEL_W - 70)), int(65 + xy[1] * (PANEL_H - 105)), row))
    for first, second in zip(points, points[1:]):
        cv2.line(tile, first[:2], second[:2], (190, 190, 190), 2, cv2.LINE_AA)
    for x, y, row in points:
        color = (0, 180, 255) if row == current else (220, 220, 220)
        cv2.circle(tile, (x, y), 7 if row == current else 4, color, -1, cv2.LINE_AA)
        cv2.putText(tile, "t" if row == current else "n", (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, color, 1, cv2.LINE_AA)
    cv2.rectangle(tile, (35, 65), (PANEL_W - 35, PANEL_H - 40), (110, 110, 110), 1)
    cv2.putText(tile, "orange=t; grey=retained t+/-2 observations", (12, PANEL_H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, (150, 150, 150), 1, cv2.LINE_AA)


def _render_event(destination: Path, label: str, row: int, joint: int, context_rows: list[int],
                  bank, positions: np.ndarray, image_root: Path, observation: np.ndarray,
                  in_frame: np.ndarray, h0: np.ndarray, recovered: np.ndarray, target: np.ndarray,
                  target_available: np.ndarray, eligible: np.ndarray, temporal_class: np.ndarray) -> dict[str, Any]:
    poses = [h0[index] for index in context_rows] + [recovered[index] for index in context_rows]
    poses.extend(target[index] for index in context_rows if target_available[index, joint])
    bounds = _bounds(poses)
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"), 2.0, (VIDEO_W, VIDEO_H))
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not create compact review MP4")
    for display_row in context_rows:
        sample = bank.samples[int(positions[display_row])]
        image_path = sample.image_reference.resolve({"3dpw_images": image_root}) if sample.image_reference else None
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR) if image_path and image_path.is_file() else None
        canvas = np.zeros((VIDEO_H, VIDEO_W, 3), dtype=np.uint8)
        rgb = _panel(canvas, 0, "Original RGB + observed target joint")
        if image is not None:
            fitted, scale, left, top = _fit_image(image)
            rgb[:] = fitted
            cv2.rectangle(rgb, (0, 0), (PANEL_W - 1, 30), (20, 20, 20), -1)
            cv2.putText(rgb, "Original RGB + observed target joint", (8, 21), cv2.FONT_HERSHEY_SIMPLEX,
                        0.47, (245, 245, 245), 1, cv2.LINE_AA)
            if in_frame[display_row, joint]:
                xy = observation[display_row, joint, :2]
                point = (int(left + xy[0] * sample.image_size[0] * scale), int(top + xy[1] * sample.image_size[1] * scale))
                cv2.circle(rgb, point, 6, (0, 180, 255), -1, cv2.LINE_AA)
        else:
            cv2.putText(rgb, "RGB unavailable", (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (140, 140, 140), 1, cv2.LINE_AA)
        trajectory = _panel(canvas, 1, "2D trajectory")
        _draw_trajectory(trajectory, context_rows, display_row, joint, observation, in_frame)
        h0_tile = _panel(canvas, 2, "H0 3D (frozen framewise prediction)")
        _draw_skeleton(h0_tile, h0[display_row], joint, bounds, (255, 160, 0))
        recovery_tile = _panel(canvas, 3, "Temporal interpolation probe (display only)")
        _draw_skeleton(recovery_tile, recovered[display_row], joint, bounds, (60, 220, 60))
        target_tile = _panel(canvas, 4, "Oracle target 3D (evaluation only)")
        _draw_skeleton(target_tile, target[display_row] if target_available[display_row, joint] else None,
                       joint, bounds, (255, 0, 220))
        info = _panel(canvas, 5, "Attribution state")
        lines = [f"selection: {label}", f"joint: {JOINT_NAMES[joint]}",
                 f"FrameBank frame: {sample.frame_index}", f"class: {temporal_class[display_row, joint] or 'not probed'}",
                 f"probe applied here: {bool(eligible[display_row, joint])}",
                 "t+/-2 only; no model training or production change"]
        for index, text in enumerate(lines):
            cv2.putText(info, text[:62], (12, 58 + index * 29), cv2.FONT_HERSHEY_SIMPLEX,
                        0.43, (235, 235, 235), 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (0, VIDEO_H - 26), (VIDEO_W, VIDEO_H), (20, 20, 20), -1)
        cv2.putText(canvas, "TEMPORAL EVIDENCE PROBE ONLY - frozen H0 and production behavior unchanged",
                    (12, VIDEO_H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (245, 245, 245), 1, cv2.LINE_AA)
        writer.write(canvas)
    writer.release()
    return {"label": label, "sample_id": bank.samples[int(positions[row])].sample_id,
            "joint": JOINT_NAMES[joint], "context_test_rows": context_rows, "path": destination.name,
            "sha256": _sha256(destination)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--prediction", required=True, type=Path)
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite compact review package: {args.out}")
    bank = load_bank(args.bank)
    positions = bank.indices("test")
    if len(positions) != 7076 or bank.content_digest() != FIXED_BANK_DIGEST:
        raise ValueError("fixed docs/44-46 test FrameBank was not recovered")
    identity = verify_source_identity(prediction=args.prediction, evaluation=args.evaluation, bank=bank,
                                      split="test", candidate="O_BILATERAL_oracle_forward_depth_only",
                                      frames=len(positions), joints=len(JOINT_NAMES))
    if identity["prediction"]["sha256"] != FIXED_PREDICTION_SHA256 or \
            identity["evaluation"]["sha256"] != FIXED_EVALUATION_SHA256:
        raise ValueError("prediction/evaluation identity differs from frozen baseline")
    h0 = np.load(args.prediction).astype(np.float64)
    observation = bank.arrays["input_2d"][positions].astype(np.float64)
    observation_valid = bank.arrays["input_valid"][positions]
    target = bank.arrays["target_3d"][positions].astype(np.float64)
    target_valid = bank.arrays["target_valid"][positions]
    grouped = sequence_rows(bank.samples, positions)
    before, after = local_neighbor_rows(grouped, len(positions))
    timestamps = np.asarray([_sample_time(bank.samples[int(position)]) for position in positions], dtype=np.float64)
    cohorts, quantities = build_cohorts(observation, observation_valid, h0, target, target_valid,
                                        timestamps, before, after)
    left, right, support = nearest_temporal_support(quantities["in_frame"], h0, timestamps, before, after)
    eligible = failure_mask(cohorts) & support
    recovered = interpolate_recovery(h0, timestamps, left, right, eligible)
    before_error = quantities["h0_error"]
    after_error = np.linalg.norm(recovered - target, axis=-1)
    after_error[~quantities["finite_target"]] = np.nan
    from diagnose_temporal_context_evidence import classify_temporal_rows
    stable_observation = quantities["in_frame"] & np.isfinite(quantities["observation_residual"]) & (
        quantities["observation_residual"] <= quantities["observation_residual_p50"][None, :])
    labels = classify_temporal_rows(failure_mask(cohorts), quantities["in_frame"], stable_observation,
                                    eligible, quantities["finite_target"], before_error, after_error)
    row_by_id = {bank.samples[int(position)].sample_id: row for row, position in enumerate(positions)}
    order_by_row = {row: local for rows in grouped.values() for local, row in enumerate(rows.tolist())}
    rows_by_sequence = {row: rows for rows in grouped.values() for row in rows.tolist()}
    args.out.mkdir(parents=True, exist_ok=False)
    rendered = []
    for label, sample_id, joint_name in DEFAULT_SELECTION:
        if sample_id not in row_by_id:
            raise KeyError(f"selected review sample is absent: {sample_id}")
        row, joint = row_by_id[sample_id], JOINT_INDEX[joint_name]
        sequence_rows_here, local = rows_by_sequence[row], order_by_row[row]
        context = sequence_rows_here[max(0, local - 2):local + 3].astype(int).tolist()
        filename = f"{label}.mp4"
        rendered.append(_render_event(args.out / filename, label, row, joint, context, bank, positions,
                                      args.image_root, observation, quantities["in_frame"], h0, recovered,
                                      target, quantities["finite_target"], eligible, labels))
    manifest = {"schema": "animcv_temporal_context_compact_review_v1", "purpose": "owner verification of temporal-evidence attribution; not a blind contest",
                "frozen_boundaries": ["H0 unchanged", "no model training", "no production temporal behavior", "no Pose Reconciliation change"],
                "context": "retained same-sequence FrameBank t±2 rows only; 2 FPS display", "source": {"bank_content_digest": bank.content_digest(), **identity},
                "events": rendered}
    write_json(args.out / "review_manifest.json", manifest)
    print(json.dumps({"events": len(rendered), "output": str(args.out)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
