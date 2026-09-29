#!/usr/bin/env python3
"""Owner review: does the legacy yaw hold stabilize bad H0, or freeze an old heading? (docs/55)

Per owner case from run_root_orientation_hold_attribution.py: an MP4 (bank
rows only, 10 fps real time, no interpolation) and a still of the case frame.

- RGB: 2D detector input + H0 skeleton, projected with GT root anchoring
  (display only, as in docs/46/54).
- TOP view: three headings from the pelvis — H0-CURRENT (cyan), LEGACY-HOLD
  (red when held, white otherwise), oracle (green) — with held-run age.
- Timeline: heading error vs oracle for H0-CURRENT and LEGACY-HOLD (0-180 deg)
  over the clip, with held rows shaded.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from render_animation_semantics_review import (  # noqa: E402
    _clip_crop, _draw_skeleton, _put, _rgb_panel,
)
from replay_pose_reconciliation import _load_camera_state  # noqa: E402

from common.canonical_pose import JOINT_INDEX  # noqa: E402
from framepose.bank import load_bank  # noqa: E402
from motion.animation_semantics import load_animation_semantics  # noqa: E402

CURRENT, HELD, LEGACY, ORACLE = (230, 230, 0), (60, 60, 230), (235, 235, 235), (80, 210, 80)  # BGR
PANEL_W, BODY_H, HEADER_H, TIMELINE_H, MIN_WIDTH = 560, 640, 110, 210, 1280
FPS_OUT = 10


def _fmt(value, unit="deg"):
    return "n/a" if value is None else f"{value:.0f}{unit}"


def _heading(canvas, origin, yaw, color, length, thickness):
    if yaw is None:
        return
    tip = origin + np.array([-math.sin(yaw), -math.cos(yaw)]) * length  # forward=(-sin, cos); image y is down
    cv2.arrowedLine(canvas, tuple(int(v) for v in origin), tuple(int(v) for v in tip), color, thickness,
                    cv2.LINE_AA, tipLength=0.18)


def _top_panel(frame, row, bounds):
    canvas = np.full((BODY_H, PANEL_W, 3), 28, np.uint8)
    points = np.asarray(frame.articulation.joint_positions, dtype=np.float64)
    lo, hi = bounds
    span = max(hi[0] - lo[0], hi[1] - lo[1]) * 1.3
    scale = (PANEL_W - 80) / max(span, 1e-6) * 0.6
    center = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2])
    pixels = np.stack([PANEL_W / 2 + (points[:, 0] - center[0]) * scale,
                       BODY_H * 0.42 - (points[:, 1] - center[1]) * scale], -1)
    _put(canvas, "TOP view (up = away from camera)", (10, 24), 0.55)
    _draw_skeleton(canvas, pixels, np.asarray(frame.reliability.joint_observation_valid), 2)
    origin = pixels[JOINT_INDEX["pelvis"]]
    _heading(canvas, origin, row["oracle_yaw"], ORACLE, 150, 5)
    _heading(canvas, origin, row["legacy_yaw"], HELD if row["legacy_held"] else LEGACY, 130, 3)
    _heading(canvas, origin, row["current_yaw"], CURRENT, 110, 2)
    y = BODY_H - 150
    _put(canvas, "green  = oracle heading (GT, evaluation only)", (10, y), 0.52, ORACLE)
    _put(canvas, f"cyan   = H0-CURRENT   error {_fmt(row['current_error'])}", (10, y + 26), 0.52, CURRENT)
    held = f"HELD, run age {row['held_age']} rows" if row["legacy_held"] else "not held"
    _put(canvas, f"{'red' if row['legacy_held'] else 'white'}    = LEGACY-HOLD  error {_fmt(row['legacy_error'])}  ({held})",
         (10, y + 52), 0.52, HELD if row["legacy_held"] else LEGACY)
    _put(canvas, f"held-vs-current gap {_fmt(row['held_current_gap'])}", (10, y + 78), 0.52)
    return canvas


def _timeline(rows, cursor, width):
    canvas = np.full((TIMELINE_H, width, 3), 18, np.uint8)
    left, right, top, bottom = 70, width - 20, 16, TIMELINE_H - 46
    n = len(rows)
    x = lambda i: int(left + (right - left) * i / max(1, n - 1))  # noqa: E731
    y = lambda v: int(bottom - (bottom - top) * min(v, 180.0) / 180.0)  # noqa: E731
    for i, row in enumerate(rows):
        if row["legacy_held"]:
            cv2.rectangle(canvas, (x(i) - 2, top), (x(i) + 2, bottom), (40, 30, 90), -1)
    for level in (0, 45, 90, 180):
        cv2.line(canvas, (left, y(level)), (right, y(level)), (60, 60, 60), 1)
        _put(canvas, f"{level}", (20, y(level) + 5), 0.42)
    for key, color in (("legacy_error", HELD), ("current_error", CURRENT)):
        points = [(x(i), y(r[key])) for i, r in enumerate(rows) if r[key] is not None]
        for a, b in zip(points, points[1:]):
            cv2.line(canvas, a, b, color, 2, cv2.LINE_AA)
    cv2.line(canvas, (x(cursor), top - 4), (x(cursor), bottom + 4), (255, 255, 255), 2)
    _put(canvas, "error vs oracle (deg):  cyan = H0-CURRENT   red = LEGACY-HOLD   purple band = held rows   "
                 "white = current frame", (10, TIMELINE_H - 16), 0.46)
    return canvas


def _header(case_name, case, row, frame, width):
    canvas = np.full((HEADER_H, width, 3), 10, np.uint8)
    _put(canvas, f"{case_name} | {row['sequence_id']} | frame {row['frame_index']} | t={frame.timestamp:.2f}s",
         (10, 26), 0.62)
    note = {k: v for k, v in case.items() if k in ("run_rows", "gap", "oracle_delta_per_row")}
    _put(canvas, "case: " + ", ".join(f"{k} {v:.0f}" if isinstance(v, float) else f"{k} {v}" for k, v in note.items()),
         (10, 54), 0.52, (200, 230, 255))
    _put(canvas, "Question: is LEGACY-HOLD (red) stabilizing a bad H0 observation, or freezing an old heading?",
         (10, 80), 0.52, (200, 200, 200))
    _put(canvas, "RGB: cyan = 2D input, thick = H0 skeleton (GT-root anchoring, display only)", (10, 102), 0.48,
         (150, 150, 150))
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--attribution-dir", required=True, type=Path)
    parser.add_argument("--semantics-dir", required=True, type=Path)
    parser.add_argument("--raw-3dpw", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--radius-rows", type=int, default=25)
    args = parser.parse_args()

    bank = load_bank(args.bank)
    report = json.loads((args.attribution_dir / "report.json").read_text())
    rows_by_sequence: dict[str, list[dict]] = {}
    for line in (args.attribution_dir / "rows.jsonl").read_text().splitlines():
        row = json.loads(line)
        rows_by_sequence.setdefault(row["sequence_id"], []).append(row)
    semantics_files = {e["sequence_id"]: e["file"] for e in
                       json.loads((args.semantics_dir / "report.json").read_text())["per_sequence"]}
    by_key = {(s.sequence_id, s.frame_index): i for i, s in enumerate(bank.samples)}
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for case_name, case in report["owner_cases"].items():
        sequence_id = case["sequence_id"]
        rows = rows_by_sequence[sequence_id]
        semantics = load_animation_semantics(args.semantics_dir / semantics_files[sequence_id])
        if [f.frame_index for f in semantics.frames] != [r["frame_index"] for r in rows]:
            raise SystemExit(f"{sequence_id}: semantics and attribution rows differ")
        center = next(i for i, r in enumerate(rows) if r["frame_index"] == case["frame_index"])
        radius = max(args.radius_rows, case.get("run_rows", 0) // 2 + 5)
        lo, hi = max(0, center - radius), min(len(rows), center + radius + 1)
        clip_rows, clip_frames = rows[lo:hi], semantics.frames[lo:hi]
        positions = np.asarray([by_key[(sequence_id, f.frame_index)] for f in clip_frames])
        *_, contexts, _, _, _ = _load_camera_state(bank, positions, args.raw_3dpw / "sequenceFiles", "validation")
        all_points = np.concatenate([np.asarray(f.articulation.joint_positions) for f in clip_frames])
        bounds, crop = (all_points.min(0), all_points.max(0)), _clip_crop(bank, positions)
        name = sequence_id.split(":")[1]
        writer = None
        for order, (row, frame) in enumerate(zip(clip_rows, clip_frames)):
            position = int(positions[order])
            image = cv2.imread(str(args.raw_3dpw / "imageFiles" / name / f"image_{frame.frame_index:05d}.jpg"))
            rgb = _rgb_panel(image, frame, bank.samples[position], bank.arrays["input_2d"][position],
                             bank.arrays["input_valid"][position], contexts[order], BODY_H, crop)
            body = np.hstack([rgb, _top_panel(frame, row, bounds)])
            if body.shape[1] < MIN_WIDTH:
                body = np.hstack([body, np.full((BODY_H, MIN_WIDTH - body.shape[1], 3), 28, np.uint8)])
            canvas = np.vstack([_header(case_name, case, row, frame, body.shape[1]), body,
                                _timeline(clip_rows, order, body.shape[1])])
            if writer is None:
                writer = cv2.VideoWriter(str(args.out / f"{case_name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                         FPS_OUT, (canvas.shape[1], canvas.shape[0]))
            writer.write(canvas)
            if lo + order == center:
                cv2.imwrite(str(args.out / f"{case_name}.png"), canvas)
        writer.release()
        manifest.append({"case": case_name, **case, "clip_frame_range": [clip_rows[0]["frame_index"],
                                                                          clip_rows[-1]["frame_index"]],
                         "clip_rows": len(clip_rows), "video": f"{case_name}.mp4", "still": f"{case_name}.png"})
        print(f"rendered {case_name}: {len(clip_rows)} rows")
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
