#!/usr/bin/env python3
"""EVALUATION ONLY (Worklog 85): canonical connected-arm review of historical vs BEDLAM-expanded J1/R1 on 3DHP.

No rig, no IK, no Blender (canonical geometry only).  Panels: the exact source image with the detector left
arm (runtime input, white), a front view (X right x Z up) and a side view (depth x up) of the unit arm for
GT (green), frozen J0 anchor (orange), J1 (magenta), J1X (purple, dashed), R1 (cyan), R1X (blue, dashed),
with per-candidate numbers.  Fixed case rules on the evaluation rows / pairs:
  * successful / failed transfer: median of the rows where R1X - R1 (elbow + wrist) < 0 / > 0;
  * large frozen-anchor error: median R1X row inside J0's worst-wrist quartile;
  * successful RGB correction: largest R1X wrist reduction relative to the anchor;
  * correction-induced false depth: near-plane rows (abs(f_GT) < sin 10 deg) with R1X wrist worse than J0, median;
  * out-of-plane geometry: abs(f_GT) >= 0.5, median R1X wrist;
  * moderate-bend error: GT bend in [15, 45), median R1X elbow;
  * largest temporal regression / improvement: max / min R1X - R1 upper-arm residual pair (4-frame strip).
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

STYLE = {"gt": ("#22aa22", "-", 3.0), "j0": ("#ff8c1a", "-", 1.6), "j1": ("#e626d9", "-", 1.6), "j1x": ("#7a1fa2", "--", 1.8),
         "r1": ("#1ad9f2", "-", 1.6), "r1x": ("#1f4fd9", "--", 1.8)}
SHOW = ("j0", "j1", "j1x", "r1", "r1x")
STABLE = math.sin(math.radians(10))
ARM = (11, 12, 13)


def _crop(img, inp, size):
    w, h = size
    pts = np.array([[x * w, y * h] for x, y, c in inp if c > 0])
    lo, hi = pts.min(0), pts.max(0)
    c, side = (lo + hi) / 2, max(hi - lo) * 1.25 + 40
    x0, y0 = int(max(0, c[0] - side / 2)), int(max(0, c[1] - side / 2))
    return img[y0:int(min(h, c[1] + side / 2)), x0:int(min(w, c[0] + side / 2))], (x0, y0)


def _panel(axes, row, image, inp, size, title):
    a_img, a_front, a_side = axes
    crop, (x0, y0) = _crop(image, inp, size)
    a_img.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    vis = [(inp[j][0] * size[0] - x0, inp[j][1] * size[1] - y0) for j in ARM if inp[j][2] > 0]
    a_img.plot([p[0] for p in vis], [p[1] for p in vis], "-o", color="white", lw=2, ms=4)
    a_img.set_title(title, fontsize=7)
    a_img.axis("off")
    for ax, (i, j, xl) in ((a_front, (0, 2, "X right")), (a_side, (1, 2, "Y depth"))):
        for name in ("gt", *SHOW):
            g = row["geometry"][name]
            color, ls, lw = STYLE[name]
            pts = np.array([[0, 0, 0], g["elbow"], g["wrist"]])
            ax.plot(pts[:, i], pts[:, j], ls, marker="o", color=color, lw=lw, ms=3, label=name.upper())
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_aspect("equal")
        ax.axhline(0, color="#ccc", lw=0.5)
        ax.axvline(0, color="#ccc", lw=0.5)
        ax.set_xlabel(xl, fontsize=7)
        ax.set_ylabel("Z up", fontsize=7)
        ax.tick_params(labelsize=6)
    a_front.set_title("front (X x Z)", fontsize=7)
    a_side.set_title("side (depth x up)", fontsize=7)
    a_side.legend(fontsize=6, loc="lower right")


def _caption(row, pair=None):
    c = row["candidates"]
    lines = [f"{row['sequence_id']}  annotation row {row['frame_index']}  GT bend {row['gt_bend_degrees']:.0f} deg  GT f {row['f_gt']:+.2f}  ({row['sample_id']})",
             "wrist %L: " + "  ".join(f"{n.upper()} {c[n]['wrist_error']:.1f}" for n in SHOW)
             + "   elbow %L: " + "  ".join(f"{n.upper()} {c[n]['elbow_error']:.1f}" for n in SHOW),
             "forearm deg: " + "  ".join(f"{n.upper()} {c[n]['forearm_direction_error_degrees']:.0f}" for n in SHOW)
             + "   correction upper/forearm deg: " + "  ".join(
                 f"{g.upper()} {row['correction'][g]['upper_rotation_degrees']:.0f}/{row['correction'][g]['forearm_rotation_degrees']:.0f}" for g in ("r1", "r1x"))]
    if pair is not None:
        p = pair["candidates"]
        lines.append(f"pair row {pair['frame_index']}->{pair['next_frame_index']} upper residual deg: "
                     + "  ".join(f"{n.upper()} {p[n]['upper_direction_residual_degrees']:.1f}" for n in SHOW)
                     + f"   (GT upper change {pair['gt']['upper_direction_change_degrees']:.1f})")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("eval_dir", "bank_json", "image_root", "out_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads((args.eval_dir / "rows_3dhp.json").read_text())
    pairs = json.loads((args.eval_dir / "pairs_3dhp.json").read_text())
    bank = json.loads(args.bank_json.with_suffix(".json").read_text())
    arrays = np.load(args.bank_json.with_suffix(".npz"))
    by_key = {(r["sequence_id"], r["frame_index"]): r for r in rows}
    pair_by = {(p["sequence_id"], p["frame_index"]): p for p in pairs}
    key = lambda r: (r["sequence_id"], r["frame_index"])  # noqa: E731
    e = lambda r, n, m: r["candidates"][n][m]  # noqa: E731
    both = lambda r, n: e(r, n, "elbow_error") + e(r, n, "wrist_error")  # noqa: E731

    def median_of(group, fn):
        g = sorted(group, key=lambda r: (fn(r), key(r)))
        return g[len(g) // 2] if g else None
    d = lambda r: both(r, "r1x") - both(r, "r1")  # noqa: E731
    q3 = sorted(e(r, "j0", "wrist_error") for r in rows)[int(0.75 * (len(rows) - 1))]
    stills = {"successful_transfer_r1x_vs_r1": median_of([r for r in rows if d(r) < 0], d),
              "failed_transfer_r1x_vs_r1": median_of([r for r in rows if d(r) > 0], d),
              "large_frozen_anchor_error": median_of([r for r in rows if e(r, "j0", "wrist_error") >= q3], lambda r: e(r, "r1x", "wrist_error")),
              "successful_rgb_correction": min(rows, key=lambda r: (r["correction"]["r1x"]["wrist_error_change_vs_anchor"], key(r))),
              "correction_induced_false_depth": median_of([r for r in rows if r["f_gt"] is not None and abs(r["f_gt"]) < STABLE
                                                           and e(r, "r1x", "wrist_error") > e(r, "j0", "wrist_error")],
                                                          lambda r: e(r, "r1x", "wrist_error") - e(r, "j0", "wrist_error")),
              "out_of_plane_geometry": median_of([r for r in rows if r["f_gt"] is not None and abs(r["f_gt"]) >= 0.5], lambda r: e(r, "r1x", "wrist_error")),
              "moderate_bend_error": median_of([r for r in rows if r["gt_bend_degrees"] is not None and 15 <= r["gt_bend_degrees"] < 45],
                                               lambda r: e(r, "r1x", "elbow_error"))}

    def strip_from(p):
        frames, cur = [by_key[(p["sequence_id"], p["frame_index"])]], p
        while cur is not None and len(frames) < 4:
            nxt = by_key.get((cur["sequence_id"], cur["next_frame_index"]))
            if nxt is None:
                break
            frames.append(nxt)
            cur = pair_by.get((cur["sequence_id"], cur["next_frame_index"]))
        return frames
    ur = lambda p: p["candidates"]["r1x"]["upper_direction_residual_degrees"] - p["candidates"]["r1"]["upper_direction_residual_degrees"]  # noqa: E731
    strips = {"largest_temporal_regression_r1x_vs_r1": strip_from(max(pairs, key=lambda p: (ur(p), p["sequence_id"], p["frame_index"]))),
              "largest_temporal_improvement_r1x_vs_r1": strip_from(min(pairs, key=lambda p: (ur(p), p["sequence_id"], p["frame_index"])))}

    def load(r):
        s = bank["samples"][r["position"]]
        return cv2.imread(str(args.image_root / s["image_reference"]["relative_path"])), arrays["input_2d"][r["position"]], tuple(s["image_size"])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    index = {}
    for name, r in stills.items():
        if r is None:
            index[name] = None
            continue
        fig, axes = plt.subplots(1, 3, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1.1, 1, 1]})
        _panel(axes, r, *load(r), name)
        fig.suptitle(_caption(r, pair_by.get(key(r))), fontsize=7, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.82))
        fig.savefig(args.out_dir / f"{name}.png", dpi=130)
        plt.close(fig)
        index[name] = list(key(r))
    for name, frames in strips.items():
        fig, axes = plt.subplots(len(frames), 3, figsize=(11, 4.2 * len(frames)), gridspec_kw={"width_ratios": [1.1, 1, 1]}, squeeze=False)
        for i, r in enumerate(frames):
            _panel(axes[i], r, *load(r), f"{name} [{i + 1}/{len(frames)}]")
            axes[i][0].text(0, -0.08, _caption(r, pair_by.get(key(r))), transform=axes[i][0].transAxes, fontsize=6, va="top")
        fig.tight_layout(h_pad=5.0)
        fig.savefig(args.out_dir / f"{name}.png", dpi=110)
        plt.close(fig)
        index[name] = [list(key(r)) for r in frames]
    (args.out_dir / "review_index.json").write_text(json.dumps(index, indent=1))
    print(json.dumps(index, indent=1))


if __name__ == "__main__":
    main()
