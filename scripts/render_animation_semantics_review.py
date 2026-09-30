#!/usr/bin/env python3
"""Owner-facing bone-overlay review of AnimationSemantics owner cases (docs/53/54).

Reads the persisted AnimationSemantics files written by
run_animation_semantics_replay.py and renders, per owner case, an MP4 clip of
+/- N bank rows plus a still of the case frame:

- RGB panel (one fixed person-centred crop per clip): the 2D detector input skeleton (what FramePose saw) and the H0
  skeleton projected onto the image. The projection anchors the root-relative
  H0 pose with the 3DPW ground-truth pelvis position and intrinsics
  (`research_oracle_absolute_root_placement`, as in docs/46). That anchoring is
  DISPLAY ONLY: AnimationSemantics carries no root translation.
- 3D panels: H0 top view (camera x/depth) with the semantic heading arrow and
  the current shoulder axis, and a front view with ankle state markers.
- Timeline: per-row left/right foot state and yaw-held state, with a cursor.

Only bank rows are drawn (validation stride 3 -> 10 fps real time). No frame
is interpolated.
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

from replay_pose_reconciliation import _load_camera_state  # noqa: E402

from common.canonical_pose import BONES, JOINT_INDEX  # noqa: E402
from framepose.bank import load_bank  # noqa: E402
from motion.animation_semantics import load_animation_semantics  # noqa: E402
from motion.animation_semantics_v2 import load_animation_semantics_v2  # noqa: E402
from pose.contact import ContactState  # noqa: E402

LEFT, RIGHT, CENTER = (255, 160, 60), (60, 140, 255), (220, 220, 220)  # BGR
STATE_COLOR = {ContactState.CONTACT: (80, 200, 80), ContactState.MOVING: (0, 140, 255),
               ContactState.UNKNOWN: (130, 130, 130)}
HELD_COLOR, KNOWN_COLOR, UNKNOWN_YAW_COLOR = (60, 60, 230), (200, 200, 200), (90, 90, 90)
INPUT_2D_COLOR = (230, 230, 0)
PANEL_W, TOTAL_H, TIMELINE_H, HEADER_H = 520, 960, 170, 106
MIN_WIDTH = 1280
FPS_OUT = 10


def _side_color(first: str, second: str):
    names = first + second
    return LEFT if "left" in names else RIGHT if "right" in names else CENTER


def _put(canvas, text, org, scale=0.6, color=(255, 255, 255), thickness=1):
    cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _draw_skeleton(canvas, pixels, valid, thickness, color_override=None):
    for first, second in BONES:
        a, b = pixels[JOINT_INDEX[first]], pixels[JOINT_INDEX[second]]
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            continue
        color = color_override or _side_color(first, second)
        ok = valid[JOINT_INDEX[first]] and valid[JOINT_INDEX[second]]
        pa, pb = tuple(int(v) for v in a), tuple(int(v) for v in b)
        if ok:
            cv2.line(canvas, pa, pb, color, thickness, cv2.LINE_AA)
        else:  # invalid observation: thin dashed-looking red segment
            for t in np.linspace(0, 1, 9)[::2]:
                p = (int(a[0] + (b[0] - a[0]) * t), int(a[1] + (b[1] - a[1]) * t))
                q = (int(a[0] + (b[0] - a[0]) * min(1, t + 0.06)), int(a[1] + (b[1] - a[1]) * min(1, t + 0.06)))
                cv2.line(canvas, p, q, (0, 0, 255), max(1, thickness - 1), cv2.LINE_AA)


def _ankle_markers(canvas, pixels, frame, radius=9):
    for side in ("left", "right"):
        point = pixels[JOINT_INDEX[f"{side}_ankle"]]
        if not np.isfinite(point).all():
            continue
        state = getattr(frame.foot_motion, side)
        center = tuple(int(v) for v in point)
        if state is ContactState.CONTACT:
            cv2.rectangle(canvas, (center[0] - radius, center[1] - radius),
                          (center[0] + radius, center[1] + radius), STATE_COLOR[state], -1)
        else:
            cv2.circle(canvas, center, radius, STATE_COLOR[state], 2 if state is ContactState.UNKNOWN else -1)
        _put(canvas, side[0].upper(), (center[0] + radius + 2, center[1] + 5), 0.5, STATE_COLOR[state])


def _clip_crop(bank, positions, aspect=0.9, margin=0.35):
    """One fixed person-centred crop (x0, y0, x1, y1) in pixels for the whole clip."""
    points = []
    for position in positions:
        size = np.asarray(bank.samples[int(position)].image_size, dtype=np.float64)
        valid = bank.arrays["input_valid"][int(position)]
        points.append(bank.arrays["input_2d"][int(position)][valid, :2] * size)
    points = np.concatenate(points)
    width, height = bank.samples[int(positions[0])].image_size
    (x0, y0), (x1, y1) = points.min(0), points.max(0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    h = max(y1 - y0, (x1 - x0) / aspect) * (1 + 2 * margin)
    h = min(h, height, width / aspect)
    w = h * aspect
    x0 = int(np.clip(cx - w / 2, 0, width - w)); y0 = int(np.clip(cy - h / 2, 0, height - h))
    return x0, y0, int(x0 + w), int(y0 + h)


def _rgb_panel(image, frame, sample, input_2d, input_valid, context, height, crop):
    if image is None:
        image = np.zeros((sample.image_size[1], sample.image_size[0], 3), np.uint8)
        _put(image, "RGB frame missing", (40, 80), 1.2, (0, 0, 255), 2)
    x0, y0, x1, y1 = crop
    offset = np.array([x0, y0], dtype=np.float64)
    image = image[y0:y1, x0:x1]
    scale = height / image.shape[0]
    canvas = cv2.resize(image, (int(image.shape[1] * scale), height))
    size = np.asarray(sample.image_size, dtype=np.float64)
    observed = (input_2d[:, :2] * size - offset) * scale
    observed[~input_valid] = np.nan
    valid = np.asarray(frame.reliability.joint_observation_valid)
    if context is not None:
        projected = np.full((len(valid), 2), np.nan)
        for j, point in enumerate(frame.articulation.joint_positions):
            try:
                projected[j] = context.project_root_relative(point)
            except ValueError:
                pass
        projected = (projected - offset) * scale
        if np.isfinite(projected[JOINT_INDEX["pelvis"]]).all():
            _draw_skeleton(canvas, projected, valid, 3)
            _ankle_markers(canvas, projected, frame)
        else:
            _put(canvas, "GT root anchoring unusable on this frame (pelvis not projectable): H0 not drawn on RGB",
                 (10, height - 40), 0.5, (0, 0, 255))
    else:
        _put(canvas, "no GT root/camera for display anchoring", (10, height - 40), 0.6, (0, 0, 255))
    _draw_skeleton(canvas, observed, np.ones(len(observed), bool), 1, INPUT_2D_COLOR)
    for point in observed:
        if np.isfinite(point).all():
            cv2.circle(canvas, tuple(int(v) for v in point), 3, INPUT_2D_COLOR, -1, cv2.LINE_AA)
    return canvas


def _view_panel(frame, width, height, bounds):
    canvas = np.full((height, width, 3), 28, np.uint8)
    points = np.asarray(frame.articulation.joint_positions, dtype=np.float64)
    valid = np.asarray(frame.reliability.joint_observation_valid)
    half = height // 2
    lo, hi = bounds

    def to_px(u, v, top, box_h, span_u, span_v):
        s = min((width - 40) / span_u[2], (box_h - 50) / span_v[2])
        return np.stack([width / 2 + (u - span_u[0]) * s, top + box_h / 2 + 10 - (v - span_v[0]) * s], -1)

    center_u = ((lo[0] + hi[0]) / 2, 0, max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) * 1.2)
    # Top view: x right, depth (y) up = away from camera.
    top = to_px(points[:, 0], points[:, 1], 0, half, center_u, ((lo[1] + hi[1]) / 2, 0, center_u[2]), )
    _put(canvas, "TOP view (camera x / depth; up = away from camera)", (8, 20), 0.5)
    _draw_skeleton(canvas, top, valid, 2)
    orientation = frame.root_orientation
    pelvis = top[JOINT_INDEX["pelvis"]]
    ls, rs = points[JOINT_INDEX["left_shoulder"]], points[JOINT_INDEX["right_shoulder"]]
    raw = math.atan2(rs[1] - ls[1], rs[0] - ls[0])
    raw_fwd = np.array([-math.sin(raw), math.cos(raw)])
    tip = pelvis + np.array([raw_fwd[0], -raw_fwd[1]]) * 60
    cv2.arrowedLine(canvas, tuple(int(v) for v in pelvis), tuple(int(v) for v in tip), (160, 160, 160), 1,
                    cv2.LINE_AA, tipLength=0.2)
    if orientation.known:
        yaw = orientation.yaw_radians
        fwd = np.array([-math.sin(yaw), math.cos(yaw)])
        tip = pelvis + np.array([fwd[0], -fwd[1]]) * 90
        held = getattr(orientation, "yaw_held", False)
        color = HELD_COLOR if held else KNOWN_COLOR
        cv2.arrowedLine(canvas, tuple(int(v) for v in pelvis), tuple(int(v) for v in tip), color, 3,
                        cv2.LINE_AA, tipLength=0.2)
        gap = abs((math.degrees(yaw - raw) + 180) % 360 - 180)
        label = f"semantic yaw {math.degrees(yaw):.1f} deg" + ("  HELD (last accepted value)" if held else "  CURRENT" if not hasattr(orientation, "yaw_held") else "")
        _put(canvas, label, (8, half - 30), 0.5, color)
        _put(canvas, f"thin grey = this frame's shoulder heading; gap {gap:.0f} deg", (8, half - 10), 0.45)
    else:
        _put(canvas, "root orientation UNKNOWN", (8, half - 10), 0.5, UNKNOWN_YAW_COLOR)
    # Front view: x right, z up.
    front = to_px(points[:, 0], points[:, 2], half, half, center_u, ((lo[2] + hi[2]) / 2, 0, center_u[2]))
    cv2.line(canvas, (0, half), (width, half), (70, 70, 70), 1)
    _put(canvas, "FRONT view (camera x / up)", (8, half + 20), 0.5)
    _draw_skeleton(canvas, front, valid, 2)
    _ankle_markers(canvas, front, frame, radius=7)
    return canvas


def _timeline(frames, cursor, width):
    canvas = np.full((TIMELINE_H, width, 3), 18, np.uint8)
    n = len(frames)
    x = lambda i: int(90 + (width - 100) * i / max(1, n))  # noqa: E731
    rows = (("yaw", lambda f: HELD_COLOR if f.root_orientation.known and getattr(f.root_orientation, "yaw_held", False)
             else KNOWN_COLOR if f.root_orientation.known else UNKNOWN_YAW_COLOR),
            ("L foot", lambda f: STATE_COLOR[f.foot_motion.left]),
            ("R foot", lambda f: STATE_COLOR[f.foot_motion.right]),
            ("valid", lambda f: (80, 200, 80) if f.reliability.all_joints_valid else (0, 0, 200)))
    for r, (label, color_of) in enumerate(rows):
        y0 = 12 + r * 26
        _put(canvas, label, (8, y0 + 16), 0.45)
        for i, f in enumerate(frames):
            cv2.rectangle(canvas, (x(i), y0), (max(x(i) + 1, x(i + 1) - 1), y0 + 20), color_of(f), -1)
    cv2.rectangle(canvas, (x(cursor) - 1, 6), (x(cursor + 1), 12 + 4 * 26), (255, 255, 255), 2)
    yaw_legend = ("yaw: red=HELD  grey=known  dark=unknown" if hasattr(frames[0].root_orientation, "yaw_held")
                  else "yaw: grey=CURRENT  dark=unknown")
    _put(canvas, yaw_legend + "   |   valid: red = some joint invalid", (8, TIMELINE_H - 34), 0.45)
    _put(canvas, "foot: green square=CONTACT-like  orange=MOVING  grey ring=UNKNOWN   |   white box = current frame",
         (8, TIMELINE_H - 12), 0.45)
    return canvas


def _header(frame, category, sequence_id, width):
    canvas = np.full((HEADER_H, width, 3), 10, np.uint8)
    o = frame.root_orientation
    yaw = ("UNKNOWN" if not o.known else f"{math.degrees(o.yaw_radians):.1f}deg" +
           (" HELD" if getattr(o, "yaw_held", False) else " CURRENT" if not hasattr(o, "yaw_held") else ""))
    _put(canvas, f"{category} | {sequence_id} | frame {frame.frame_index} | t={frame.timestamp:.2f}s", (10, 24), 0.6)
    _put(canvas, f"yaw {yaw} | L {frame.foot_motion.left.value} | R {frame.foot_motion.right.value} | "
                 f"valid joints {sum(frame.reliability.joint_observation_valid)}/17", (10, 50), 0.55, (200, 230, 255))
    _put(canvas, "root_translation UNAVAILABLE | ground_height UNAVAILABLE | RGB overlay anchored with GT root: display only",
         (10, 74), 0.5, (150, 150, 150))
    _put(canvas, "cyan dots/thin = 2D detector input | thick = H0 3D projected (blue=left, orange=right, "
                 "red dashes = invalid joint)", (10, 96), 0.5, INPUT_2D_COLOR)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--semantics-dir", required=True, type=Path, help="run_animation_semantics_replay out dir")
    parser.add_argument("--raw-3dpw", required=True, type=Path, help=".../DATASET_Motion")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--radius-rows", type=int, default=20)
    args = parser.parse_args()

    bank = load_bank(args.bank)
    report = json.loads((args.semantics_dir / "report.json").read_text())
    files = {entry["sequence_id"]: entry["file"] for entry in report["per_sequence"]}
    by_key = {(s.sequence_id, s.frame_index): i for i, s in enumerate(bank.samples)}
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for category, case in report["owner_cases"].items():
        loader = load_animation_semantics_v2 if report["schema"] == "animcv_animation_semantics_v2_replay_v1" else load_animation_semantics
        semantics = loader(args.semantics_dir / files[case["sequence_id"]])
        frames = list(semantics.frames)
        center = next(i for i, f in enumerate(frames) if f.frame_index == case["frame_index"])
        lo_row, hi_row = max(0, center - args.radius_rows), min(len(frames), center + args.radius_rows + 1)
        clip = frames[lo_row:hi_row]
        positions = np.asarray([by_key[(semantics.sequence_id, f.frame_index)] for f in clip])
        *_, contexts, _, _, _ = _load_camera_state(bank, positions, args.raw_3dpw / "sequenceFiles", "validation")
        all_points = np.concatenate([np.asarray(f.articulation.joint_positions) for f in clip])
        bounds = (all_points.min(0), all_points.max(0))
        crop = _clip_crop(bank, positions)
        name = semantics.sequence_id.split(":")[1]
        video_path = args.out / f"{category}.mp4"
        writer = None
        for order, frame in enumerate(clip):
            position = int(positions[order])
            sample = bank.samples[position]
            image = cv2.imread(str(args.raw_3dpw / "imageFiles" / name / f"image_{frame.frame_index:05d}.jpg"))
            body_h = TOTAL_H - HEADER_H - TIMELINE_H
            rgb = _rgb_panel(image, frame, sample, bank.arrays["input_2d"][position],
                             bank.arrays["input_valid"][position], contexts[order], body_h, crop)
            views = _view_panel(frame, PANEL_W, body_h, bounds)
            body = np.hstack([rgb, views])
            if body.shape[1] < MIN_WIDTH:
                body = np.hstack([body, np.full((body.shape[0], MIN_WIDTH - body.shape[1], 3), 28, np.uint8)])
            width = body.shape[1]
            canvas = np.vstack([_header(frame, category, semantics.sequence_id, width), body,
                                _timeline(clip, order, width)])
            if writer is None:
                writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS_OUT,
                                         (canvas.shape[1], canvas.shape[0]))
            writer.write(canvas)
            if lo_row + order == center:
                cv2.imwrite(str(args.out / f"{category}.png"), canvas)
        writer.release()
        manifest.append({"category": category, "sequence_id": semantics.sequence_id,
                         "case_frame_index": case["frame_index"], "clip_rows": len(clip),
                         "clip_frame_range": [clip[0].frame_index, clip[-1].frame_index],
                         "video": video_path.name, "still": f"{category}.png",
                         "display_anchoring": "research_oracle_absolute_root_placement (display only)"})
        print(f"rendered {category}: {len(clip)} rows")
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
