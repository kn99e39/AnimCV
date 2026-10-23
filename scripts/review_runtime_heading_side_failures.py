#!/usr/bin/env python3
"""REVIEW (Worklog 91A follow-up): RGB contact sheets of the 3DHP side-view heading failures.

Question: in the TS2 rows where both S0 and the 2D nose cue point to the profile opposite to the 3DHP GT, is the head
turned against the torso, or do the image observation and the 3D annotation disagree?

Fixed selection (3DHP TEST, frozen S0 ``a78adeb0...``, GT = bank target_3d):
  ts2_failure   every 3rd TS2 row with runtime heading error > 135 deg
  ts2_correct   the first 4 TS2 rows (by time) with error < 20 deg and |sin(GT yaw)| >= 0.7 (side-on, correct)
  ts4_flip      the first 4 TS4 rows where the 2D nose check would flip a correct S0 heading (head-turn suspects)
  ts1_correct   the first 4 TS1 side-on rows with error < 20 deg (reference)
Each tile: a square crop around the detector torso, detector shoulders/hips/elbows (subject LEFT = cyan, RIGHT =
orange), the nose (yellow), arrows of the body-forward image-x component for GT (green) and S0 (magenta)
(forward f = (-sin psi, cos psi): image-x component -sin psi), and the numbers.  GT is shown for evaluation only.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_runtime_heading_3dhp import shim  # noqa: E402
from run_runtime_heading_eval import circ  # noqa: E402

from framepose.bedlam_supervision import ExternalBank  # noqa: E402
from motion.runtime_heading import runtime_heading  # noqa: E402
from motion.runtime_heading_side_check import nose_offset, side_view_check  # noqa: E402
from pose.framepose_bridge import build_h0_lifted_sequence, sequence_frame_positions  # noqa: E402
from pose.root_orientation_diagnostic import observe_yaw  # noqa: E402

LEFT, RIGHT, NOSE = (230, 220, 30), (30, 140, 255), (40, 230, 240)  # BGR
T = 360


def tile(img, ij, size, gt, s0, off, conf, label):
    w, h = size
    pts = {j: (ij[j, 0] * w, ij[j, 1] * h) for j in range(17)}
    torso = [pts[j] for j in (1, 4, 11, 14)]
    cx, cy = np.mean([p[0] for p in torso]), np.mean([p[1] for p in torso])
    span = max(np.linalg.norm(np.subtract(pts[11], pts[1])), np.linalg.norm(np.subtract(pts[14], pts[4])), 40) * 3.2
    x0, y0 = int(max(0, cx - span / 2)), int(max(0, cy - span * 0.6))
    crop = img[y0:int(min(h, y0 + span)), x0:int(min(w, x0 + span))].copy()
    s = T / max(crop.shape[:2])
    crop = cv2.resize(crop, (int(crop.shape[1] * s), int(crop.shape[0] * s)))
    canvas = np.zeros((T + 70, T, 3), np.uint8)
    canvas[:crop.shape[0], :crop.shape[1]] = crop
    p = lambda j: (int((pts[j][0] - x0) * s), int((pts[j][1] - y0) * s))  # noqa: E731
    for a, b, c in ((11, 12, LEFT), (14, 15, RIGHT), (11, 1, LEFT), (14, 4, RIGHT), (11, 14, (200, 200, 200)), (1, 4, (200, 200, 200))):
        if ij[a, 2] > 0 and ij[b, 2] > 0:
            cv2.line(canvas, p(a), p(b), c, 2)
    for j, c in ((11, LEFT), (1, LEFT), (14, RIGHT), (4, RIGHT)):
        cv2.circle(canvas, p(j), 5, c, -1)
    if ij[10, 2] > 0:
        cv2.circle(canvas, p(10), 6, NOSE, -1)
    mid = ((p(11)[0] + p(14)[0]) // 2, (p(11)[1] + p(14)[1]) // 2)
    for yaw, c, dy in ((gt, (60, 200, 60), -8), (s0, (214, 26, 214), 8)):
        dx = int(-math.sin(yaw) * 60)
        cv2.arrowedLine(canvas, (mid[0], mid[1] + dy), (mid[0] + dx, mid[1] + dy), c, 3, tipLength=0.3)
    lines = [label, f"GT yaw {math.degrees(gt):+.0f}  S0 {math.degrees(s0):+.0f}  err {abs(circ(s0, gt)):.0f}",
             f"nose offset {off if off is None else round(off, 2)}  conf {conf:.2f}"]
    for i, t in enumerate(lines):
        cv2.putText(canvas, t, (6, T + 18 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1, cv2.LINE_AA)
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("w83_dir", "s0_predictions", "image_root", "out_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    mx = ExternalBank.load(args.w83_dir / "bank" / "mpi3dhp_test")
    hp = np.load(args.s0_predictions)
    rv = shim(mx, mx.arrays["input_2d"], mx.arrays["input_valid"])
    g2 = np.concatenate([np.zeros_like(mx.arrays["input_2d"][..., :2]), np.ones_like(mx.arrays["input_2d"][..., 2:])], axis=-1)
    gv = shim(mx, g2, mx.arrays["target_valid"])
    rows = {}
    for ts in ("TS1", "TS2", "TS4"):
        sid = f"3dhp:{ts}"
        pos = sequence_frame_positions(rv, sid)
        gt = observe_yaw(build_h0_lifted_sequence(gv, mx.arrays["target_3d"], sid))
        hs = runtime_heading(build_h0_lifted_sequence(rv, hp, sid))
        out = []
        for i, p in enumerate(pos):
            g, y = gt[i].yaw_radians, hs[i].yaw_radians
            if g is None or y is None:
                continue
            off, conf = nose_offset(mx.arrays["input_2d"][p], mx.arrays["input_valid"][p], mx.samples[p].image_size)
            out.append((p, rv.samples[p].timestamp, g, y, off, conf, side_view_check(y, off, conf)[1]))
        rows[ts] = out
    sel = {"ts2_failure": [r for r in rows["TS2"] if abs(circ(r[3], r[2])) > 135][::3],
           "ts2_correct": [r for r in rows["TS2"] if abs(circ(r[3], r[2])) < 20 and abs(math.sin(r[2])) >= 0.7][:4],
           "ts4_flip": [r for r in rows["TS4"] if r[6] == "flipped" and abs(circ(r[3], r[2])) < 45][:4],
           "ts1_correct": [r for r in rows["TS1"] if abs(circ(r[3], r[2])) < 20 and abs(math.sin(r[2])) >= 0.7][:4]}
    for name, rs in sel.items():
        tiles = []
        for p, t, g, y, off, conf, _ in rs:
            s = mx.samples[p]
            img = cv2.imread(str(args.image_root / s.image_reference.relative_path))
            tiles.append(tile(img, mx.arrays["input_2d"][p], s.image_size, g, y, off, conf, f"{name} {s.sequence_id.split(':')[1]} t={t:.1f}s"))
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
        cv2.imwrite(str(args.out_dir / f"{name}.png"), sheet)
        print(name, len(rs), [round(r[1], 1) for r in rs])


if __name__ == "__main__":
    main()
