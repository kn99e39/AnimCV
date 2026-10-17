#!/usr/bin/env python3
"""EVALUATION ONLY (Worklog 86): canonical two-segment review of right-arm and right-leg G0 / G1 on 3DHP.

No rig, IK, foot contact or Blender claim.  Panels: the exact source image with the detector chain (runtime
input, white), front (X right x Z up) and side (depth x up) views of the unit chain for GT (green), G0
(orange, 2D only) and G1 (cyan, 2D + frozen Qwen), with per-candidate numbers.  Fixed rules per chain:
  * correct reconstruction: median G1 row among the rows whose G1 distal error is in the lowest decile;
  * large error: the G1 distal-error 95th-percentile row;
  * strongly out-of-plane: abs(f_GT) >= 0.5, median G1 distal error;  near-plane: abs(f_GT) < sin 10 deg, median;
  * near-straight: GT bend < 15 deg, median;  strongly bent: GT bend >= 45 deg, median;
  * low-confidence observation: lowest-decile minimum detector confidence over P, M, D, median G1 distal error;
  * temporal improvement / regression: min / max G1 - G0 proximal-segment residual pair (4-frame strip).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from framepose.connected_limb_transfer import CHAINS, chain_indices  # noqa: E402

STYLE = {"gt": ("#22aa22", "-", 3.0), "g0": ("#ff8c1a", "-", 1.8), "g1": ("#1ad9f2", "-", 1.8)}
NAMES = {"right_arm": ("shoulder", "elbow", "wrist"), "right_leg": ("hip", "knee", "ankle")}


def _crop(img, pts, size):
    w, h = size
    lo, hi = pts.min(0), pts.max(0)
    c, side = (lo + hi) / 2, max(max(hi - lo) * 1.6, 160)
    x0, y0 = int(max(0, c[0] - side / 2)), int(max(0, c[1] - side / 2))
    return img[y0:int(min(h, c[1] + side / 2)), x0:int(min(w, c[0] + side / 2))], (x0, y0)


def _panel(axes, row, image, inp, size, chain, title):
    a_img, a_front, a_side = axes
    idx = chain_indices(chain)
    vis = [(inp[j][0] * size[0], inp[j][1] * size[1]) for j in idx if inp[j][2] > 0]
    allpts = np.array([[inp[j][0] * size[0], inp[j][1] * size[1]] for j in range(17) if inp[j][2] > 0])
    crop, (x0, y0) = _crop(image, allpts, size)
    a_img.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    a_img.plot([p[0] - x0 for p in vis], [p[1] - y0 for p in vis], "-o", color="white", lw=2, ms=4)
    a_img.set_title(title, fontsize=7)
    a_img.axis("off")
    for ax, (i, j, xl) in ((a_front, (0, 2, "X right")), (a_side, (1, 2, "Y depth"))):
        for name in ("gt", "g0", "g1"):
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


def _caption(row, chain, conf, pair=None):
    c = row["candidates"]
    p, m, d = NAMES[chain]
    lines = [f"{chain}  {row['sequence_id']}  annotation row {row['frame_index']}  GT bend {row['gt_bend_degrees']:.0f} deg  GT f {row['f_gt']:+.2f}"
             f"  min detector conf {conf:.2f}  ({row['sample_id']})",
             f"{d} %L: G0 {c['g0']['wrist_error']:.1f}  G1 {c['g1']['wrist_error']:.1f}   {m} %L: G0 {c['g0']['elbow_error']:.1f}  G1 {c['g1']['elbow_error']:.1f}"
             f"   {m}-{d} segment deg: G0 {c['g0']['forearm_direction_error_degrees']:.0f}  G1 {c['g1']['forearm_direction_error_degrees']:.0f}"
             f"   bend err deg: G0 {c['g0']['abs_bend_angle_error_degrees']:.0f}  G1 {c['g1']['abs_bend_angle_error_degrees']:.0f}"]
    if pair is not None:
        q = pair["candidates"]
        lines.append(f"pair row {pair['frame_index']}->{pair['next_frame_index']} {p}-{m} segment residual deg: G0 "
                     f"{q['g0']['upper_direction_residual_degrees']:.1f}  G1 {q['g1']['upper_direction_residual_degrees']:.1f}"
                     f"  (GT change {pair['gt']['upper_direction_change_degrees']:.1f})")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("eval_dir", "bank_json", "image_root", "out_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    bank = json.loads(args.bank_json.with_suffix(".json").read_text())
    arrays = np.load(args.bank_json.with_suffix(".npz"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    index = {}
    for chain in ("right_arm", "right_leg"):
        rows = json.loads((args.eval_dir / f"{chain}_rows_3dhp.json").read_text())
        pairs = json.loads((args.eval_dir / f"{chain}_pairs_3dhp.json").read_text())
        idx = chain_indices(chain)
        conf = {r["position"]: float(arrays["input_2d"][r["position"]][idx, 2].min()) for r in rows}
        by_key = {(r["sequence_id"], r["frame_index"]): r for r in rows}
        pair_by = {(p["sequence_id"], p["frame_index"]): p for p in pairs}
        key = lambda r: (r["sequence_id"], r["frame_index"])  # noqa: E731
        d1 = lambda r: r["candidates"]["g1"]["wrist_error"]  # noqa: E731

        def median_of(group, fn=d1):
            g = sorted(group, key=lambda r: (fn(r), key(r)))
            return g[len(g) // 2] if g else None
        order = sorted(rows, key=lambda r: (d1(r), key(r)))
        cq = sorted(conf.values())[int(0.1 * (len(conf) - 1))]
        stills = {"correct_reconstruction": median_of(order[:max(1, len(order) // 10)]),
                  "large_error": order[int(0.95 * (len(order) - 1))],
                  "strongly_out_of_plane": median_of([r for r in rows if r["f_gt"] is not None and abs(r["f_gt"]) >= 0.5]),
                  "near_plane": median_of([r for r in rows if r["f_gt"] is not None and abs(r["f_gt"]) < math.sin(math.radians(10))]),
                  "near_straight": median_of([r for r in rows if r["gt_bend_degrees"] is not None and r["gt_bend_degrees"] < 15]),
                  "strongly_bent": median_of([r for r in rows if r["gt_bend_degrees"] is not None and r["gt_bend_degrees"] >= 45]),
                  "low_confidence_observation": median_of([r for r in rows if conf[r["position"]] <= cq])}

        def strip_from(p):
            frames, cur = [by_key[(p["sequence_id"], p["frame_index"])]], p
            while cur is not None and len(frames) < 4:
                nxt = by_key.get((cur["sequence_id"], cur["next_frame_index"]))
                if nxt is None:
                    break
                frames.append(nxt)
                cur = pair_by.get((cur["sequence_id"], cur["next_frame_index"]))
            return frames
        ur = lambda p: p["candidates"]["g1"]["upper_direction_residual_degrees"] - p["candidates"]["g0"]["upper_direction_residual_degrees"]  # noqa: E731
        strips = {"temporal_improvement": strip_from(min(pairs, key=lambda p: (ur(p), p["sequence_id"], p["frame_index"]))),
                  "temporal_regression": strip_from(max(pairs, key=lambda p: (ur(p), p["sequence_id"], p["frame_index"])))}

        def load(r):
            s = bank["samples"][r["position"]]
            return cv2.imread(str(args.image_root / s["image_reference"]["relative_path"])), arrays["input_2d"][r["position"]], tuple(s["image_size"])
        for name, r in stills.items():
            if r is None:
                index[f"{chain}_{name}"] = None
                continue
            fig, axes = plt.subplots(1, 3, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1.1, 1, 1]})
            _panel(axes, r, *load(r), chain, f"{chain} {name}")
            fig.suptitle(_caption(r, chain, conf[r["position"]], pair_by.get(key(r))), fontsize=7, x=0.01, ha="left")
            fig.tight_layout(rect=(0, 0, 1, 0.84))
            fig.savefig(args.out_dir / f"{chain}_{name}.png", dpi=130)
            plt.close(fig)
            index[f"{chain}_{name}"] = list(key(r))
        for name, frames in strips.items():
            fig, axes = plt.subplots(len(frames), 3, figsize=(11, 4.2 * len(frames)), gridspec_kw={"width_ratios": [1.1, 1, 1]}, squeeze=False)
            for i, r in enumerate(frames):
                _panel(axes[i], r, *load(r), chain, f"{chain} {name} [{i + 1}/{len(frames)}]")
                axes[i][0].text(0, -0.08, _caption(r, chain, conf[r["position"]], pair_by.get(key(r))), transform=axes[i][0].transAxes, fontsize=6, va="top")
            fig.tight_layout(h_pad=5.0)
            fig.savefig(args.out_dir / f"{chain}_{name}.png", dpi=110)
            plt.close(fig)
            index[f"{chain}_{name}"] = [list(key(r)) for r in frames]
    (args.out_dir / "review_index.json").write_text(json.dumps(index, indent=1))
    print(json.dumps(index, indent=1))


if __name__ == "__main__":
    main()
