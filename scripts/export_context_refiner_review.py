#!/usr/bin/env python3
"""Export a compact RGB/2D/3D review for the context-refiner candidate."""

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
from framepose.context_refiner import (
    GateDecision, apply_gate, runtime_signals,
)
from diagnose_temporal_context_evidence import build_cohorts, local_neighbor_rows, sequence_rows


PANEL_W, PANEL_H = 480, 270
VIDEO_W, VIDEO_H = PANEL_W * 3, PANEL_H * 2
BONES = (
    ("pelvis", "left_hip"), ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("pelvis", "right_hip"), ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ("pelvis", "spine"), ("spine", "thorax"), ("thorax", "neck"), ("neck", "head"),
    ("thorax", "left_shoulder"), ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"), ("thorax", "right_shoulder"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
)
OWNER_SEEDS = (
    ("owner_arguing_550", "3dpw:downtown_arguing_00:actor0#000550", "left_elbow"),
    ("owner_sitOnStairs_45", "3dpw:downtown_sitOnStairs_00:actor0#000045", "left_knee"),
    ("owner_bar_985", "3dpw:downtown_bar_00:actor0#000985", "right_elbow"),
    ("owner_warmWelcome_330", "3dpw:downtown_warmWelcome_00:actor0#000330", "right_knee"),
    ("owner_car_507", "3dpw:downtown_car_00:actor1#000507", "right_knee"),
    ("owner_windowShopping_588", "3dpw:downtown_windowShopping_00:actor0#000588", "right_ankle"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fit_image(image: np.ndarray) -> tuple[np.ndarray, float, int, int]:
    height, width = image.shape[:2]
    scale = min(PANEL_W / width, (PANEL_H - 30) / height)
    resized = cv2.resize(image, (int(round(width * scale)), int(round(height * scale))), interpolation=cv2.INTER_AREA)
    result = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
    left = (PANEL_W - resized.shape[1]) // 2
    top = 30 + (PANEL_H - 30 - resized.shape[0]) // 2
    result[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
    return result, scale, left, top


def _title(tile: np.ndarray, title: str) -> None:
    cv2.rectangle(tile, (0, 0), (PANEL_W - 1, 30), (20, 20, 20), -1)
    cv2.putText(tile, title[:60], (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.47,
                (245, 245, 245), 1, cv2.LINE_AA)


def _skeleton(tile: np.ndarray, pose: np.ndarray | None, target_joint: int,
              bounds: tuple[float, float, float, float], color: tuple[int, int, int]) -> None:
    if pose is None or not np.isfinite(pose).all():
        cv2.putText(tile, "Unavailable", (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (140, 140, 140), 1, cv2.LINE_AA)
        return
    points = np.column_stack((pose[:, 0], -pose[:, 2]))
    x0, x1, y0, y1 = bounds
    scale = min((PANEL_W - 70) / max(x1 - x0, 1e-9), (PANEL_H - 75) / max(y1 - y0, 1e-9))
    def pixel(point):
        return (int(round(35 + (point[0] - x0) * scale)),
                int(round(45 + (y1 - point[1]) * scale)))
    for first, second in BONES:
        cv2.line(tile, pixel(points[JOINT_INDEX[first]]), pixel(points[JOINT_INDEX[second]]),
                 (120, 120, 120), 2, cv2.LINE_AA)
    for index, point in enumerate(points):
        cv2.circle(tile, pixel(point), 7 if index == target_joint else 3,
                   color if index == target_joint else (185, 185, 185), -1, cv2.LINE_AA)
    cv2.putText(tile, "+X right / +Z up", (8, PANEL_H - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.36, (140, 140, 140), 1, cv2.LINE_AA)


def _bounds(poses: list[np.ndarray]) -> tuple[float, float, float, float]:
    values = np.concatenate([np.column_stack((pose[:, 0], -pose[:, 2])) for pose in poses if np.isfinite(pose).all()])
    mins, maxs = values.min(axis=0), values.max(axis=0)
    centre = (mins + maxs) / 2.0
    span = max(float((maxs - mins).max()), 0.1) * 1.3
    return (float(centre[0] - span / 2), float(centre[0] + span / 2),
            float(centre[1] - span / 2), float(centre[1] + span / 2))


def _trajectory(tile: np.ndarray, rows: list[int], current: int, joint: int,
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
    cv2.rectangle(tile, (35, 65), (PANEL_W - 35, PANEL_H - 40), (110, 110, 110), 1)
    cv2.putText(tile, "orange=t; grey=t+/-2", (12, PANEL_H - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.34, (150, 150, 150), 1, cv2.LINE_AA)


def _render(path: Path, label: str, row: int, joint: int, context: list[int], bank,
            positions: np.ndarray, image_root: Path, observation: np.ndarray, in_frame: np.ndarray,
            h0: np.ndarray, c2: np.ndarray, target: np.ndarray, target_valid: np.ndarray,
            decision, writer_fps: float = 2.0) -> dict[str, Any]:
    poses = [h0[index] for index in context] + [c2[index] for index in context]
    poses.extend(target[index] for index in context if np.isfinite(target[index]).all())
    bounds = _bounds(poses)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), writer_fps, (VIDEO_W, VIDEO_H))
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not create context review MP4")
    for display_row in context:
        sample = bank.samples[int(positions[display_row])]
        canvas = np.zeros((VIDEO_H, VIDEO_W, 3), dtype=np.uint8)
        rgb = canvas[0:PANEL_H, 0:PANEL_W]
        _title(rgb, "RGB + observed target joint")
        if sample.image_reference is not None:
            image_path = sample.image_reference.resolve({"3dpw_images": image_root})
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR) if image_path.is_file() else None
        else:
            image = None
        if image is not None:
            fitted, scale, left, top = _fit_image(image)
            rgb[:] = fitted
            _title(rgb, "RGB + observed target joint")
            if in_frame[display_row, joint]:
                xy = observation[display_row, joint, :2]
                point = (int(left + xy[0] * sample.image_size[0] * scale), int(top + xy[1] * sample.image_size[1] * scale))
                cv2.circle(rgb, point, 6, (0, 180, 255), -1, cv2.LINE_AA)
        else:
            cv2.putText(rgb, "RGB unavailable", (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (140, 140, 140), 1, cv2.LINE_AA)
        trajectory = canvas[0:PANEL_H, PANEL_W:2 * PANEL_W]
        _title(trajectory, "2D observation context")
        _trajectory(trajectory, context, display_row, joint, observation, in_frame)
        h0_tile = canvas[0:PANEL_H, 2 * PANEL_W:3 * PANEL_W]
        _title(h0_tile, "H0 frozen baseline")
        _skeleton(h0_tile, h0[display_row], joint, bounds, (255, 160, 0))
        c2_tile = canvas[PANEL_H:2 * PANEL_H, 0:PANEL_W]
        _title(c2_tile, "C2 refined target frame")
        _skeleton(c2_tile, c2[display_row], joint, bounds, (60, 220, 60))
        target_tile = canvas[PANEL_H:2 * PANEL_H, PANEL_W:2 * PANEL_W]
        _title(target_tile, "Oracle target (evaluation only)")
        _skeleton(target_tile, target[display_row] if target_valid[display_row, joint] else None,
                  joint, bounds, (255, 0, 220))
        info = canvas[PANEL_H:2 * PANEL_H, 2 * PANEL_W:3 * PANEL_W]
        _title(info, "Gate state")
        lines = [f"{label}", f"joint: {JOINT_NAMES[joint]}", f"frame: {sample.frame_index}",
                 f"gate ON: {bool(decision.gate_on[display_row, joint])}",
                 f"reason: {str(decision.reason[display_row, joint])}",
                 "same-sequence t-2...t+2; target frame only"]
        for index, line in enumerate(lines):
            cv2.putText(info, line[:62], (12, 58 + index * 29), cv2.FONT_HERSHEY_SIMPLEX,
                        0.43, (235, 235, 235), 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (0, VIDEO_H - 26), (VIDEO_W, VIDEO_H), (20, 20, 20), -1)
        cv2.putText(canvas, "CONTEXT REFINER REVIEW - no sequence output / no smoothing objective",
                    (12, VIDEO_H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (245, 245, 245), 1, cv2.LINE_AA)
        writer.write(canvas)
    writer.release()
    return {"label": label, "sample_id": bank.samples[int(positions[row])].sample_id,
            "joint": JOINT_NAMES[joint], "context_test_rows": context, "path": path.name,
            "sha256": _sha256(path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--h0-train", required=True, type=Path)
    parser.add_argument("--h0-validation", required=True, type=Path)
    parser.add_argument("--h0-test", required=True, type=Path)
    parser.add_argument("--prediction-test-c2", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"refusing to overwrite review output: {args.out}")
    bank = load_bank(args.bank)
    h0_parts = {
        "train": np.asarray(np.load(args.h0_train), dtype=np.float32),
        "validation": np.asarray(np.load(args.h0_validation), dtype=np.float32),
        "test": np.asarray(np.load(args.h0_test), dtype=np.float32),
    }
    h0 = np.full((len(bank), len(JOINT_NAMES), 3), np.nan, dtype=np.float32)
    for split, values in h0_parts.items():
        positions = bank.indices(split)
        if values.shape != (len(positions), len(JOINT_NAMES), 3):
            raise ValueError(f"unexpected {split} H0 shape: {values.shape}")
        h0[positions] = values
    c2_test = np.asarray(np.load(args.prediction_test_c2), dtype=np.float32)
    test_positions = bank.indices("test")
    if c2_test.shape != (len(test_positions), len(JOINT_NAMES), 3):
        raise ValueError(f"unexpected C2 test shape: {c2_test.shape}")
    signals = runtime_signals(bank, h0)
    thresholds = json.loads((args.prediction_test_c2.parent / "gate_thresholds.json").read_text(encoding="utf-8"))
    from framepose.context_refiner import GateThresholds
    decision = apply_gate(signals, GateThresholds.from_dict(thresholds))
    observation = bank.arrays["input_2d"][test_positions].astype(np.float64)
    observation_valid = bank.arrays["input_valid"][test_positions]
    target = bank.arrays["target_3d"][test_positions].astype(np.float64)
    target_valid = bank.arrays["target_valid"][test_positions]
    grouped = sequence_rows(bank.samples, test_positions)
    before, after = local_neighbor_rows(grouped, len(test_positions))
    cohorts, quantities = build_cohorts(observation, observation_valid, h0[test_positions], target,
                                        target_valid,
                                        np.asarray([signals["timestamps"][position] for position in test_positions]),
                                        before, after)
    row_by_id = {bank.samples[int(position)].sample_id: row for row, position in enumerate(test_positions)}
    sequence_for_row = {row: rows for rows in grouped.values() for row in rows.tolist()}
    order_for_row = {row: local for rows in grouped.values() for local, row in enumerate(rows.tolist())}
    selections = list(OWNER_SEEDS)
    for label, cohort_name, joint_name in (
        ("stable_2d_h0_jitter", "h0_jitter_stable_observation", "neck"),
        ("high_2d_instability", "observation_degradation", "right_ankle"),
        ("distal_ankle", "distal_joint_failure", "right_ankle"),
        ("observation_loss", "current_frame_observation_loss", "left_ankle"),
    ):
        candidates = np.argwhere(cohorts[cohort_name][:, JOINT_INDEX[joint_name]])
        if len(candidates):
            row = int(candidates[0, 0])
            selections.append((label, bank.samples[int(test_positions[row])].sample_id, joint_name))
    args.out.mkdir(parents=True, exist_ok=True)
    rendered = []
    for label, sample_id, joint_name in selections:
        if sample_id not in row_by_id:
            continue
        row = row_by_id[sample_id]
        local = order_for_row[row]
        rows = sequence_for_row[row]
        context = rows[max(0, local - 2):local + 3].astype(int).tolist()
        rendered.append(_render(args.out / f"{label}.mp4", label, row, JOINT_INDEX[joint_name], context,
                                 bank, test_positions, args.image_root, observation,
                                 quantities["in_frame"], h0[test_positions], c2_test, target,
                                 target_valid, GateDecision(
                                     decision.gate_on[test_positions], decision.reason[test_positions],
                                     decision.observation_residual[test_positions], decision.h0_residual[test_positions],
                                     decision.left[test_positions], decision.right[test_positions])))
    manifest = {"schema": "animcv_context_refiner_compact_review_v1",
                "purpose": "owner-seed and deterministic-slice review of C2 target-frame behavior",
                "frozen_boundaries": ["H0 unchanged", "C2 outputs target frame only", "no temporal smoothness loss",
                                      "no Pose Reconciliation or SignState change"],
                "events": rendered, "bank_content_digest": bank.content_digest(),
                "c2_prediction_sha256": _sha256(args.prediction_test_c2)}
    write_json(args.out / "review_manifest.json", manifest)
    print(json.dumps({"events": len(rendered), "output": str(args.out)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
