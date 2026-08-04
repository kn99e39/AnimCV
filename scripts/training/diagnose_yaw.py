#!/usr/bin/env python3
"""Root-cause analysis for a failed root-yaw holdout gate.

Recomputes the same shoulder/hip-based yaw signal root_motion.py uses,
directly from gt.json / prediction.json (bypassing the smoothing/hold logic),
to see whether errors come from the depth (Y) axis, the left-right (X) axis,
or specific joints, and whether they cluster near a 180-degree flip.

Reads training_cache/<run_name>/holdout/*/{gt.json,prediction.json} — these
are large, fully regenerable intermediates that are gitignored, not
committed. Re-run scripts/training/run_mpi_inf_3dhp_pipeline.sh first if
training_cache/ is missing.
"""
import glob
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RUN_NAME = sys.argv[1] if len(sys.argv) > 1 else "mpi_s1s6_cam0"
HOLDOUT_DIR = os.path.join(REPO_ROOT, "training_cache", RUN_NAME, "holdout")

PAIRS = (("left_shoulder", "right_shoulder"), ("left_hip", "right_hip"))


def fused_yaw(points):
    vectors = []
    for left, right in PAIRS:
        if left not in points or right not in points:
            continue
        a, b = points[left]["position"], points[right]["position"]
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dx, dy)
        if length > 1e-6:
            vectors.append((math.atan2(dy, dx), length, dx, dy))
    if not vectors:
        return None
    x = sum(math.cos(a) * w for a, w, *_ in vectors)
    y = sum(math.sin(a) * w for a, w, *_ in vectors)
    if math.hypot(x, y) < 1e-6:
        return None
    return math.atan2(y, x), vectors


def angle_delta_deg(a, b):
    return ((a - b + math.pi) % (2 * math.pi) - math.pi) * 180.0 / math.pi


if not os.path.isdir(HOLDOUT_DIR):
    raise SystemExit(f"{HOLDOUT_DIR} not found — run scripts/training/run_mpi_inf_3dhp_pipeline.sh first "
                      "(training_cache/ holds large regenerable intermediates and is gitignored).")

rows = []
joint_depth_err = defaultdict(list)
joint_x_err = defaultdict(list)

for gt_path in sorted(glob.glob(os.path.join(HOLDOUT_DIR, "*", "gt.json"))):
    clip = os.path.basename(os.path.dirname(gt_path))
    pred_path = gt_path.replace("gt.json", "prediction.json")
    gt = {f["frame_index"]: f["points"] for f in json.load(open(gt_path))["frames"]}
    pred = {f["frame_index"]: f["points"] for f in json.load(open(pred_path))["frames"]}
    for idx, gt_points in gt.items():
        pred_points = pred.get(idx)
        if pred_points is None:
            continue
        gy = fused_yaw(gt_points)
        py = fused_yaw(pred_points)
        if gy is None or py is None:
            continue
        gt_yaw, gt_vecs = gy
        pred_yaw, pred_vecs = py
        err = angle_delta_deg(pred_yaw, gt_yaw)
        for name in ("left_shoulder", "right_shoulder", "left_hip", "right_hip"):
            if name in gt_points and name in pred_points:
                joint_depth_err[name].append(pred_points[name]["position"][1] - gt_points[name]["position"][1])
                joint_x_err[name].append(pred_points[name]["position"][0] - gt_points[name]["position"][0])
        rows.append({
            "clip": clip, "frame": idx, "gt_yaw_deg": math.degrees(gt_yaw),
            "pred_yaw_deg": math.degrees(pred_yaw), "err_deg": err, "abs_err_deg": abs(err),
            "gt_shoulder_dy": gt_vecs[0][3] if gt_vecs else None,
            "gt_shoulder_dx": gt_vecs[0][2] if gt_vecs else None,
            "gt_shoulder_len": gt_vecs[0][1] if gt_vecs else None,
        })

errs = np.array([r["abs_err_deg"] for r in rows])
print(f"total frames analyzed: {len(rows)}")
print(f"raw per-frame |yaw error|: mean={errs.mean():.1f} median={np.median(errs):.1f} "
      f"p95={np.percentile(errs,95):.1f}")

bins = [(0, 15), (15, 45), (45, 90), (90, 135), (135, 165), (165, 180.01)]
print("\nerror distribution buckets:")
for lo, hi in bins:
    frac = np.mean((errs >= lo) & (errs < hi))
    print(f"  [{lo:>5.0f},{hi:<5.0f}): {frac*100:5.1f}%")

gt_dy = np.array([abs(r["gt_shoulder_dy"]) for r in rows if r["gt_shoulder_dy"] is not None])
matched_err = np.array([r["abs_err_deg"] for r in rows if r["gt_shoulder_dy"] is not None])
order = np.argsort(gt_dy)
gt_dy_sorted, err_sorted = gt_dy[order], matched_err[order]
n = len(gt_dy_sorted)
print("\nmean |yaw error| by true |shoulder depth-diff| (GT) quintile (small -> degenerate case):")
for i in range(5):
    lo, hi = n * i // 5, n * (i + 1) // 5
    print(f"  quintile {i+1} (|dy| in [{gt_dy_sorted[lo]*1000:.1f}, {gt_dy_sorted[hi-1]*1000:.1f}] mm): "
          f"mean_err={err_sorted[lo:hi].mean():.1f} deg  n={hi-lo}")

print("\nper-joint prediction error (mean +/- std, mm), depth(Y) axis vs left-right(X) axis:")
for name in ("left_shoulder", "right_shoulder", "left_hip", "right_hip"):
    dy = np.array(joint_depth_err[name]) * 1000
    dx = np.array(joint_x_err[name]) * 1000
    print(f"  {name:16s} Y(depth): mean={dy.mean():7.1f} std={dy.std():6.1f}  "
          f"X(left-right): mean={dx.mean():7.1f} std={dx.std():6.1f}")

all_dy = np.concatenate([np.array(v) for v in joint_depth_err.values()]) * 1000
all_dx = np.concatenate([np.array(v) for v in joint_x_err.values()]) * 1000
print(f"\ncombined shoulder/hip joints: mean|Y error|={np.abs(all_dy).mean():.1f}mm  "
      f"mean|X error|={np.abs(all_dx).mean():.1f}mm  ratio={np.abs(all_dy).mean()/np.abs(all_dx).mean():.2f}x")
