#!/usr/bin/env python3
"""Worklog 75 real-scene review (images only): 3DPW test rows, chain directions in side and front view.

Rows are chosen by fixed rules from the common-valid test population:
largest T2 gain vs T0, largest T2 regression vs T0, largest T2 gain vs T1,
typical (T2 endpoint error closest to its median), strongly out-of-plane
(|GT f| >= 0.5, median T0 error), near-plane false depth (|GT f| < sin 10 deg,
largest |E1 f|), and lowest detector arm confidence.  Each panel shows the
FramePose person crop with the detector shoulder/elbow/wrist (drawn only
here), and unit shoulder->wrist directions (canonical camera frame) for GT,
H0, E0 and E1 in a side view (forward depth to the right, up) and a front view
(right, up).  These are direction plots, not Blender/IK renders; the analytic
endpoint numbers are the quantitative evidence.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from framepose.bank import load_bank  # noqa: E402
from framepose.crops import crop_box, render_crop  # noqa: E402

COLORS = {"GT": "#1a9850", "H0 (T0)": "#2166ac", "E0 (T1)": "#f46d43", "E1 (T2)": "#c51b7d"}
KEYS = {"GT": "t4_gt_direction_reference", "H0 (T0)": "t0_h0", "E0 (T1)": "t1_e0_visual", "E1 (T2)": "t2_e1_visual"}
STABLE = math.sin(math.radians(10))


def pick(rows):
    e = lambda r, n: r["candidates"][n]["endpoint_error"]  # noqa: E731
    key = lambda r: (r["sequence_id"], r["frame_index"])  # noqa: E731
    out = {"largest_t2_gain_vs_t0": min(rows, key=lambda r: (e(r, "t2_e1_visual") - e(r, "t0_h0"), key(r))),
           "largest_t2_regression_vs_t0": max(rows, key=lambda r: (e(r, "t2_e1_visual") - e(r, "t0_h0"), key(r))),
           "largest_t2_gain_vs_t1": min(rows, key=lambda r: (e(r, "t2_e1_visual") - e(r, "t1_e0_visual"), key(r)))}
    med = float(np.median([e(r, "t2_e1_visual") for r in rows]))
    out["typical"] = min(rows, key=lambda r: (abs(e(r, "t2_e1_visual") - med), key(r)))
    oop = sorted((r for r in rows if abs(r["f_gt"]) >= 0.5), key=lambda r: (e(r, "t0_h0"), key(r)))
    out["strongly_out_of_plane"] = oop[len(oop) // 2]
    near = [r for r in rows if abs(r["f_gt"]) < STABLE]
    out["near_plane_false_depth"] = max(near, key=lambda r: (abs(r["candidates"]["t2_e1_visual"]["f"]), key(r)))
    out["low_detector_confidence"] = min(rows, key=lambda r: (r["min_arm_confidence"], key(r)))
    return out


def panel(label, row, bank, image_root, out_dir):
    p = row["position"]
    s = bank.samples[p]
    img = cv2.cvtColor(cv2.imread(str(image_root / s.image_reference.relative_path)), cv2.COLOR_BGR2RGB)
    inp, valid = bank.arrays["input_2d"][p], bank.arrays["input_valid"][p]
    box = crop_box(inp, valid, s.image_size)
    crop = render_crop(img, box, 448)
    fig = plt.figure(figsize=(15, 6.6))
    ax0 = fig.add_axes([0.01, 0.2, 0.32, 0.75])
    ax0.imshow(crop)
    pts = []
    for j in (11, 12, 13):
        if valid[j]:
            u = (inp[j, 0] * s.image_size[0] - box.x) / box.side * 448
            v = (inp[j, 1] * s.image_size[1] - box.y) / box.side * 448
            pts.append((u, v))
    if pts:
        xs, ys = zip(*pts)
        ax0.plot(xs, ys, "-o", color="red", lw=2, ms=7, mfc="none", mew=2)
    ax0.set_axis_off()
    ax0.set_title("person crop; red = detector L shoulder/elbow/wrist", fontsize=10)
    for ax_pos, (title, i, j, xl) in zip(([0.37, 0.2, 0.29, 0.72], [0.69, 0.2, 0.29, 0.72]),
                                         (("side view (camera depth Y → farther, Z up)", 1, 2, "depth Y"),
                                          ("front view (X right, Z up)", 0, 2, "right X"))):
        ax = fig.add_axes(ax_pos)
        for name, k in KEYS.items():
            d = row["candidates"][k]["direction"]
            ax.annotate("", xy=(d[i], d[j]), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>", color=COLORS[name], lw=3 if name == "GT" else 2))
            ax.plot([], [], color=COLORS[name], lw=3, label=name)
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect("equal")
        ax.axhline(0, color="#ccc", lw=0.8)
        ax.axvline(0, color="#ccc", lw=0.8)
        ax.set_xlabel(xl)
        ax.set_title(title, fontsize=10)
        ax.legend(loc="lower right", fontsize=8)
    c = row["candidates"]
    lines = [
        f"W75 selection-clean test | {label} | {row['sequence_id']}  frame #{row['frame_index']}  (bank position {p})",
        (f"chain f: GT {row['f_gt']:+.2f}   H0 {c['t0_h0']['f']:+.2f}   E0 {c['t1_e0_visual']['f']:+.2f}   E1 {c['t2_e1_visual']['f']:+.2f}"
         f"   | min detector arm confidence {row['min_arm_confidence']:.2f}   | H0 reach {row['h0_reach']:.3f} vs GT {row['gt_reach']:.3f}"),
        ("analytic W61/62 endpoint error (cm): " + "   ".join(
            f"{n} {c[k]['endpoint_error']:.2f}" for n, k in (("T0 H0", "t0_h0"), ("T1 E0", "t1_e0_visual"), ("T2 E1", "t2_e1_visual"),
                                                              ("T3 GT-f", "t3_gt_f_oracle"), ("T4 GT-dir", "t4_gt_direction_reference")))),
        ("chain direction error (deg): " + "   ".join(
            f"{n} {c[k]['direction_error_degrees']:.1f}" for n, k in (("H0", "t0_h0"), ("E0", "t1_e0_visual"), ("E1", "t2_e1_visual"),
                                                                       ("GT-f", "t3_gt_f_oracle")))
         + "   (endpoint-only evidence: no elbow, no Blender/IK pose)"),
    ]
    for k, line in enumerate(lines):
        fig.text(0.01, 0.15 - 0.037 * k, line, fontsize=10.5, family="monospace")
    path = out_dir / f"{label}.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path.name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bank", "image_root", "rows", "out_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    bank = load_bank(args.bank)
    rows = json.loads(args.rows.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    chosen = pick(rows)
    index = {}
    for label, row in chosen.items():
        index[label] = {"sequence_id": row["sequence_id"], "frame_index": row["frame_index"],
                        "png": panel(label, row, bank, args.image_root, args.out_dir)}
    (args.out_dir / "review_index.json").write_text(json.dumps(index, indent=1))
    print(json.dumps(index, indent=1))


if __name__ == "__main__":
    main()
