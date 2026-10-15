#!/usr/bin/env python3
"""EVALUATION ONLY (Worklog 83): canonical connected-arm review of J0/J1/R0/R1 on MPI-INF-3DHP test frames.

No rig, no IK, no Blender: the claim of Worklog 83 is canonical connected-arm geometry only.  Each panel
shows the exact source image (detector left shoulder/elbow/wrist, the runtime input, in white), a front
view (X right x Z up) and a side view (Y depth x Z up) of the shoulder-anchored unit arm for GT (green),
J0 (orange), J1 (magenta), R0 (blue) and R1 (cyan), with the per-candidate numbers.  Case rules (fixed,
evaluation rows/pairs only):
  * typical static improvement / regression: median of the rows where R1-J1 (elbow + wrist error) < 0 / > 0;
  * large out-of-plane correction: |f_GT| >= 0.5, largest R1 correction wrist displacement;
  * near-plane failure: |f_GT| < sin 10 deg with R1 wrist worse than J0, median of that set;
  * moderate-bend example: GT bend in [15, 45), median R1 elbow error;
  * largest temporal improvement / regression: min / max R1-J1 upper-arm residual pair (4-frame strip);
  * representative stable / unstable sequence: lowest / highest mean R1 upper-arm residual sequence,
    4-frame strip starting at that sequence's middle pair.
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

COLORS = {"gt": "#22aa22", "j0": "#ff8c1a", "j1": "#e626d9", "r0": "#2673ff", "r1": "#1ad9f2"}
SHOW = ("j0", "j1", "r0", "r1")
STABLE = math.sin(math.radians(10))
ARM = (11, 12, 13)  # FrameBank left shoulder, elbow, wrist


def _crop(img, inp, size):
    w, h = size
    pts = np.array([[x * w, y * h] for x, y, c in inp if c > 0])
    lo, hi = pts.min(0), pts.max(0)
    c, side = (lo + hi) / 2, max(hi - lo) * 1.25 + 40
    x0, y0 = int(max(0, c[0] - side / 2)), int(max(0, c[1] - side / 2))
    x1, y1 = int(min(w, c[0] + side / 2)), int(min(h, c[1] + side / 2))
    return img[y0:y1, x0:x1], (x0, y0)


def _panel(axes, row, image, inp, size, title):
    a_img, a_front, a_side = axes
    crop, (x0, y0) = _crop(image, inp, size)
    a_img.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    arm = [(inp[j][0] * size[0] - x0, inp[j][1] * size[1] - y0, inp[j][2]) for j in ARM]
    vis = [p for p in arm if p[2] > 0]
    a_img.plot([p[0] for p in vis], [p[1] for p in vis], "-o", color="white", lw=2, ms=4)
    a_img.set_title(title, fontsize=7)
    a_img.axis("off")
    for ax, (i, j, xl) in ((a_front, (0, 2, "X right")), (a_side, (1, 2, "Y depth"))):
        for name in ("gt", *SHOW):
            g = row["geometry"][name]
            pts = np.array([[0, 0, 0], g["elbow"], g["wrist"]])
            ax.plot(pts[:, i], pts[:, j], "-o", color=COLORS[name], lw=3 if name == "gt" else 1.6, ms=3,
                    label=name.upper(), alpha=0.9)
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
    lines = [f"{row['sequence_id']}  annotation row {row['frame_index']}  GT bend {row['gt_bend_degrees']:.0f} deg  "
             f"GT f {row['f_gt']:+.2f}  ({row['sample_id']})",
             "elbow %L: " + "  ".join(f"{n.upper()} {c[n]['elbow_error']:.1f}" for n in SHOW)
             + "   wrist %L: " + "  ".join(f"{n.upper()} {c[n]['wrist_error']:.1f}" for n in SHOW),
             "forearm deg: " + "  ".join(f"{n.upper()} {c[n]['forearm_direction_error_degrees']:.0f}" for n in SHOW)
             + "   R1 correction: upper %.0f deg / forearm %.0f deg" % (row["correction"]["r1"]["upper_rotation_degrees"],
                                                                     row["correction"]["r1"]["forearm_rotation_degrees"])]
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
    samples = bank["samples"]
    by_key = {(r["sequence_id"], r["frame_index"]): r for r in rows}
    pair_by = {(p["sequence_id"], p["frame_index"]): p for p in pairs}
    key = lambda r: (r["sequence_id"], r["frame_index"])  # noqa: E731
    e = lambda r, n, m: r["candidates"][n][m]  # noqa: E731
    both = lambda r, n: e(r, n, "elbow_error") + e(r, n, "wrist_error")  # noqa: E731

    def median_of(group, fn):
        g = sorted(group, key=lambda r: (fn(r), key(r)))
        return g[len(g) // 2] if g else None
    d = lambda r: both(r, "r1") - both(r, "j1")  # noqa: E731
    stills = {"typical_static_improvement_vs_j1": median_of([r for r in rows if d(r) < 0], d),
              "typical_static_regression_vs_j1": median_of([r for r in rows if d(r) > 0], d),
              "large_out_of_plane_correction": max((r for r in rows if r["f_gt"] is not None and abs(r["f_gt"]) >= 0.5),
                                                   key=lambda r: (r["correction"]["r1"]["wrist_displacement"], key(r)), default=None),
              "near_plane_failure": median_of([r for r in rows if r["f_gt"] is not None and abs(r["f_gt"]) < STABLE
                                               and e(r, "r1", "wrist_error") > e(r, "j0", "wrist_error")],
                                              lambda r: e(r, "r1", "wrist_error") - e(r, "j0", "wrist_error")),
              "moderate_bend_example": median_of([r for r in rows if r["gt_bend_degrees"] is not None and 15 <= r["gt_bend_degrees"] < 45],
                                                 lambda r: e(r, "r1", "elbow_error"))}

    def strip_from(p):
        frames, cur = [], p
        start = by_key[(p["sequence_id"], p["frame_index"])]
        frames.append(start)
        while cur is not None and len(frames) < 4:
            nxt = by_key.get((cur["sequence_id"], cur["next_frame_index"]))
            frames.append(nxt)
            cur = pair_by.get((cur["sequence_id"], cur["next_frame_index"]))
        return [f for f in frames if f is not None]
    ur = lambda p: p["candidates"]["r1"]["upper_direction_residual_degrees"] - p["candidates"]["j1"]["upper_direction_residual_degrees"]  # noqa: E731
    seq_res = {}
    for p in pairs:
        seq_res.setdefault(p["sequence_id"], []).append(p["candidates"]["r1"]["upper_direction_residual_degrees"])
    mean_res = {s: sum(v) / len(v) for s, v in seq_res.items()}

    def middle_pair(sid):
        ps = sorted((p for p in pairs if p["sequence_id"] == sid), key=lambda p: p["frame_index"])
        return ps[len(ps) // 2]
    strips = {"largest_r1_temporal_improvement_vs_j1": strip_from(min(pairs, key=lambda p: (ur(p), p["sequence_id"], p["frame_index"]))),
              "largest_r1_temporal_regression_vs_j1": strip_from(max(pairs, key=lambda p: (ur(p), p["sequence_id"], p["frame_index"]))),
              "stable_sequence_" + min(mean_res, key=mean_res.get).split(":")[1]: strip_from(middle_pair(min(mean_res, key=mean_res.get))),
              "unstable_sequence_" + max(mean_res, key=mean_res.get).split(":")[1]: strip_from(middle_pair(max(mean_res, key=mean_res.get)))}

    def load(r):
        s = samples[r["position"]]
        img = cv2.imread(str(args.image_root / s["image_reference"]["relative_path"]))
        return img, arrays["input_2d"][r["position"]], tuple(s["image_size"])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    index = {}
    for name, r in stills.items():
        if r is None:
            index[name] = None
            continue
        fig, axes = plt.subplots(1, 3, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.1, 1, 1]})
        img, inp, size = load(r)
        _panel(axes, r, img, inp, size, name)
        fig.suptitle(_caption(r, pair_by.get(key(r))), fontsize=7, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.84))
        fig.savefig(args.out_dir / f"{name}.png", dpi=130)
        plt.close(fig)
        index[name] = list(key(r))
    for name, frames in strips.items():
        fig, axes = plt.subplots(len(frames), 3, figsize=(11, 4.0 * len(frames)), gridspec_kw={"width_ratios": [1.1, 1, 1]})
        for i, r in enumerate(frames):
            img, inp, size = load(r)
            _panel(axes[i], r, img, inp, size, f"{name} [{i + 1}/{len(frames)}]")
            axes[i][0].text(0, -0.08, _caption(r, pair_by.get(key(r))), transform=axes[i][0].transAxes, fontsize=6, va="top")
        fig.tight_layout(h_pad=4.5)
        fig.savefig(args.out_dir / f"{name}.png", dpi=110)
        plt.close(fig)
        index[name] = [list(key(r)) for r in frames]
    (args.out_dir / "review_index.json").write_text(json.dumps(index, indent=1))
    print(json.dumps(index, indent=1))


if __name__ == "__main__":
    main()
