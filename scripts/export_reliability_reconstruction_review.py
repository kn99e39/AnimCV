#!/usr/bin/env python3
"""Export a compact qualitative package for Track A/B owner replay.

The package is an evidence display, not a production animation.  Each event
shows a short same-sequence RGB strip plus structural state, sampled VLM state,
frozen H0, Track B output, oracle target, and the selected two-sided anchors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from common.canonical_pose import BONES, JOINT_INDEX, JOINT_NAMES
from common.serialization import write_json
from framepose.bank import load_bank
from framepose.nonlinear_reconstruction import in_frame_mask


PANEL_W, PANEL_H = 480, 270
VIDEO_W, VIDEO_H = PANEL_W * 3, PANEL_H * 2
DEFAULT_EVENTS = (
    ("owner_tracking_loss", "3dpw:downtown_bar_00:actor0#000985", "right_elbow"),
    ("owner_out_of_observation", "3dpw:downtown_car_00:actor1#000507", "right_knee"),
    ("owner_ankle_foot_loss", "3dpw:downtown_windowShopping_00:actor0#000588", "right_ankle"),
    ("owner_good_case", "3dpw:downtown_arguing_00:actor0#000550", "left_elbow"),
    ("owner_jitter_note", "3dpw:downtown_sitOnStairs_00:actor0#000045", "left_knee"),
    ("owner_implausible_articulation", "3dpw:downtown_warmWelcome_00:actor0#000330", "right_knee"),
    ("deterministic_observation_loss", "3dpw:downtown_enterShop_00:actor0#001128", "left_ankle"),
    ("deterministic_observation_degradation", "3dpw:downtown_stairs_00:actor0#001088", "right_ankle"),
)


def _event(value: str) -> tuple[str, str, str]:
    parts = value.split("|")
    if len(parts) != 3:
        raise ValueError("--event expects label|sample_id|joint")
    if parts[2] not in JOINT_INDEX:
        raise ValueError(f"unknown joint {parts[2]!r}")
    return parts[0], parts[1], parts[2]


def _fit_image(image: np.ndarray) -> tuple[np.ndarray, float, int, int]:
    height, width = image.shape[:2]
    scale = min(PANEL_W / width, (PANEL_H - 30) / height)
    resized = cv2.resize(image, (int(round(width * scale)), int(round(height * scale))), interpolation=cv2.INTER_AREA)
    result = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
    left = (PANEL_W - resized.shape[1]) // 2
    top = 30 + (PANEL_H - 30 - resized.shape[0]) // 2
    result[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
    return result, scale, left, top


def _title(tile: np.ndarray, value: str) -> None:
    cv2.rectangle(tile, (0, 0), (PANEL_W - 1, 30), (20, 20, 20), -1)
    cv2.putText(tile, value[:62], (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.47,
                (245, 245, 245), 1, cv2.LINE_AA)


def _draw_pose(tile: np.ndarray, pose: np.ndarray | None, target_joint: int,
               bounds: tuple[float, float, float, float], color: tuple[int, int, int]) -> None:
    if pose is None or not np.isfinite(pose).all():
        cv2.putText(tile, "unavailable", (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (150, 150, 150), 1, cv2.LINE_AA)
        return
    points = np.column_stack((pose[:, 0], -pose[:, 2]))
    x0, x1, y0, y1 = bounds
    scale = min((PANEL_W - 70) / max(x1 - x0, 1e-9), (PANEL_H - 75) / max(y1 - y0, 1e-9))
    def pixel(point: np.ndarray) -> tuple[int, int]:
        return int(round(35 + (point[0] - x0) * scale)), int(round(45 + (y1 - point[1]) * scale))
    for first_name, second_name in BONES:
        cv2.line(tile, pixel(points[JOINT_INDEX[first_name]]), pixel(points[JOINT_INDEX[second_name]]),
                 (125, 125, 125), 2, cv2.LINE_AA)
    for index, point in enumerate(points):
        cv2.circle(tile, pixel(point), 7 if index == target_joint else 3,
                   color if index == target_joint else (190, 190, 190), -1, cv2.LINE_AA)
    cv2.putText(tile, "+X right / +Z up", (8, PANEL_H - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.36, (150, 150, 150), 1, cv2.LINE_AA)


def _bounds(poses: list[np.ndarray]) -> tuple[float, float, float, float]:
    values = np.concatenate([np.column_stack((pose[:, 0], -pose[:, 2]))
                             for pose in poses if pose is not None and np.isfinite(pose).all()])
    minimum, maximum = values.min(axis=0), values.max(axis=0)
    center = (minimum + maximum) / 2.0
    span = max(float((maximum - minimum).max()), 0.1) * 1.3
    return (float(center[0] - span / 2.0), float(center[0] + span / 2.0),
            float(center[1] - span / 2.0), float(center[1] + span / 2.0))


def _text_panel(tile: np.ndarray, title: str, lines: list[str]) -> None:
    _title(tile, title)
    for index, line in enumerate(lines):
        cv2.putText(tile, line[:62], (12, 58 + 29 * index), cv2.FONT_HERSHEY_SIMPLEX,
                    0.43, (235, 235, 235), 1, cv2.LINE_AA)


def _row_context(rows: list[int], row: int) -> list[int]:
    local = rows.index(row)
    return rows[max(0, local - 2):local + 3]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--reconstruction", required=True, type=Path)
    parser.add_argument("--vlm-records", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--event", action="append", default=[])
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite review package: {args.out}")
    events = [_event(value) for value in args.event] if args.event else list(DEFAULT_EVENTS)
    bank = load_bank(args.bank)
    reconstruction = np.load(args.reconstruction / "reconstruction.npz")
    positions = reconstruction["positions"].astype(np.int64)
    h0 = reconstruction["h0"].astype(np.float64)
    nonlinear = reconstruction["nonlinear"].astype(np.float64)
    changed = reconstruction["changed"].astype(bool)
    usable = reconstruction["usable"].astype(bool)
    target = bank.arrays["target_3d"][positions].astype(np.float64)
    target_valid = bank.arrays["target_valid"][positions]
    records = json.loads((args.vlm_records / "diagnostic_records.json").read_text(encoding="utf-8"))["records"]
    vlm_by_id = {record["sample_id"]: record for record in records}
    row_by_id = {bank.samples[int(position)].sample_id: row for row, position in enumerate(positions)}
    grouped: dict[str, list[int]] = {}
    for row, position in enumerate(positions):
        grouped.setdefault(bank.samples[int(position)].sequence_id, []).append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: (bank.samples[int(positions[row])].timestamp or 0.0,
                                   bank.samples[int(positions[row])].frame_index, row))
    structural = in_frame_mask(bank.arrays["input_2d"][positions], bank.arrays["input_valid"][positions])
    args.out.mkdir(parents=True, exist_ok=False)
    rendered: list[dict[str, Any]] = []
    for label, sample_id, joint_name in events:
        if sample_id not in row_by_id:
            continue
        row = row_by_id[sample_id]
        joint = JOINT_INDEX[joint_name]
        sequence = bank.samples[int(positions[row])].sequence_id
        context = _row_context(grouped[sequence], row)
        pose_list = [h0[index] for index in context] + [nonlinear[index] for index in context]
        pose_list += [target[index] for index in context if target_valid[index].all()]
        bounds = _bounds(pose_list)
        destination = args.out / f"{label}.mp4"
        writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"), 2.0, (VIDEO_W, VIDEO_H))
        if not writer.isOpened():
            raise RuntimeError("OpenCV could not create review MP4")
        gap = None
        report = json.loads((args.reconstruction / "reconstruction.json").read_text(encoding="utf-8"))
        for candidate in report["gaps"]:
            if candidate["joint"] == joint_name and row in candidate["rows"]:
                gap = candidate
                break
        for display_row in context:
            sample = bank.samples[int(positions[display_row])]
            image_path = sample.image_reference.resolve({"3dpw_images": args.image_root})
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            canvas = np.zeros((VIDEO_H, VIDEO_W, 3), dtype=np.uint8)
            rgb = canvas[0:PANEL_H, 0:PANEL_W]
            _title(rgb, "RGB + structural observation")
            if image is not None:
                fitted, scale, left, top = _fit_image(image)
                rgb[:] = fitted
                _title(rgb, "RGB + structural observation")
                if structural[display_row, joint]:
                    xy = bank.arrays["input_2d"][int(positions[display_row]), joint, :2]
                    point = (int(left + xy[0] * sample.image_size[0] * scale),
                             int(top + xy[1] * sample.image_size[1] * scale))
                    cv2.circle(rgb, point, 6, (0, 180, 255), -1, cv2.LINE_AA)
            state = "IN_FRAME" if structural[display_row, joint] else (
                "MISSING_OR_OUT_OF_FRAME" if not bank.arrays["input_valid"][int(positions[display_row]), joint]
                else "OUT_OF_FRAME")
            info = canvas[0:PANEL_H, PANEL_W:2 * PANEL_W]
            record = vlm_by_id.get(sample.sample_id)
            vlm_state = "NOT_SAMPLED"
            if record:
                answer = record["answers"]["real"]
                vlm_state = answer["state"].get(joint_name, "UNKNOWN") if answer["valid"] else "UNKNOWN(REJECTED)"
            _text_panel(info, "VLM / structural state", [f"sample: {sample.sample_id}", f"joint: {joint_name}",
                f"structural: {state}", f"VLM: {vlm_state}", "VLM state is not GT occlusion"])
            h0_tile = canvas[0:PANEL_H, 2 * PANEL_W:3 * PANEL_W]
            _text_panel(h0_tile, "Frozen H0", [sample.sample_id, f"target joint: {joint_name}"])
            _draw_pose(h0_tile, h0[display_row], joint, bounds, (255, 160, 0))
            rec_tile = canvas[PANEL_H:2 * PANEL_H, 0:PANEL_W]
            _text_panel(rec_tile, "Track B nonlinear", ["changed here: " + str(bool(changed[display_row, joint])),
                "no production fallback"])
            _draw_pose(rec_tile, nonlinear[display_row], joint, bounds, (60, 220, 60))
            target_tile = canvas[PANEL_H:2 * PANEL_H, PANEL_W:2 * PANEL_W]
            _text_panel(target_tile, "Oracle target (evaluation only)", ["target_valid: " + str(bool(target_valid[display_row, joint]))])
            _draw_pose(target_tile, target[display_row] if target_valid[display_row].all() else None,
                       joint, bounds, (255, 0, 220))
            anchor_tile = canvas[PANEL_H:2 * PANEL_H, 2 * PANEL_W:3 * PANEL_W]
            if gap:
                lines = [f"gap length: {gap['length']} ({gap['length_bin']})",
                         f"A row: {gap['left_anchor']}; B row: {gap['right_anchor']}",
                         f"context row: {display_row}", "same sequence only"]
            else:
                lines = ["no structural gap at this row", f"context row: {display_row}",
                         "owner replay / qualitative only"]
            _text_panel(anchor_tile, "Anchors / attribution", lines)
            cv2.rectangle(canvas, (0, VIDEO_H - 26), (VIDEO_W, VIDEO_H), (20, 20, 20), -1)
            cv2.putText(canvas, "TRACK A/B EVIDENCE PACKAGE - not production output", (12, VIDEO_H - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (245, 245, 245), 1, cv2.LINE_AA)
            writer.write(canvas)
        writer.release()
        rendered.append({"label": label, "sample_id": sample_id, "joint": joint_name,
                         "context_rows": context, "path": destination.name,
                         "vlm_sampled": sample_id in vlm_by_id,
                         "structural_state_at_target": "IN_FRAME" if structural[row, joint] else "INVALID_OR_OUT_OF_FRAME"})
    write_json(args.out / "review_manifest.json", {
        "schema": "animcv_reliability_reconstruction_review_v1",
        "purpose": "compact owner replay for Track A/B architecture decision",
        "boundaries": ["production Frame Pose path unchanged", "no target used by Track A/B selection",
                        "oracle target is displayed for evaluation only", "OCCLUDED has no invented GT"],
        "events": rendered,
    })
    print(json.dumps({"events": len(rendered), "output": str(args.out)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
