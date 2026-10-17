#!/usr/bin/env python3
"""DIAGNOSTIC (Worklog 88): fixed-rule consecutive-frame review of short-interval depth motion (frozen models only).

3DPW TEST pairs (1/6 s) from ``attribution_v2/pairs.json``.  Every case is a strip of four consecutive TEST rows
(previous, pair frame a, pair frame b, next) showing the exact source image with the detector chain (runtime input,
white), side views (depth x up) of the unit chain at a and b for GT (green), G0 (orange) and G1 (cyan; J1X for the
left arm), and the distal-joint depth trace of the surrounding +-12 rows with the per-pair joint readout change.
Selection rules (declared before rendering; ties broken by (sequence, frame)):
  * large_error_low_gt_motion: left arm, low GT-motion stratum, 95th percentile of G1 |dP - dG|;
  * large_legitimate_gt_motion: left arm, 95th percentile of |dG|;
  * correct_g1_depth_tracking: left arm, high stratum, pairs where G1 beats G0, median improvement (res0 - res1);
  * g1_false_depth_oscillation: first left-arm run of three consecutive pairs with |dG| < 4 %L, |dP1| >= 5 %L and
    alternating dP1 sign;
  * detector_instability: left arm, 99th percentile of detector-vs-GT 2D jitter;
  * crop_readout_change: left arm, 99th percentile of crop scale + shift change;
  * high_feature_small_output: left arm, joint-readout change >= 90th percentile, 10th percentile of |dP1| among them;
  * small_feature_large_output: left arm, joint-readout change <= 10th percentile, 90th percentile of |dP1| among them;
  * similar_error_both_arms: pairs present for both arms with G1 excess (res1 - res0) >= 5 %L on both, median of the
    smaller excess;
  * right_leg_contrast: pairs where the left arm has G1 excess >= 5 %L and the right leg has excess <= 0, median
    left-arm excess (rendered for the right leg);
  * contradicts_visual_hypothesis: left arm pairs with G1 excess >= 5 %L where |phi_geometry| > |phi_visual|, median
    excess (the excess is not carried by the visual inputs).
Rendering only: no model is evaluated here; G0 / G1 outputs are the frozen Worklog 87 predictions.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from framepose.bank import load_bank  # noqa: E402
from framepose.connected_limb_transfer import chain_indices  # noqa: E402
from framepose.visual_depth_motion_attribution import FALSE_MOTION, LOW_GT_MOTION  # noqa: E402

STYLE = {"gt": ("#22aa22", 3.0), "g0": ("#ff8c1a", 1.8), "g1": ("#1ad9f2", 1.8)}
STRIDE = 5
WINDOW = 12


def _key(r):
    return (r["sequence_id"], r["frame"])


def _sorted(rows, fn):
    return sorted(rows, key=lambda r: (fn(r), _key(r)))


def pct(rows, fn, q):
    g = _sorted(rows, fn)
    return g[int(q * (len(g) - 1))] if g else None


def excess(r):
    return r["res1"] - r["res0"]


def select(pairs):
    L, R, Lg = pairs["left_arm_3dpw_test"], pairs["right_arm_3dpw_test"], pairs["right_leg_3dpw_test"]
    absdp = lambda r: abs(r["dP1"])  # noqa: E731
    fj = sorted(r["feat_joints"] for r in L)
    hi_f, lo_f = fj[int(0.9 * (len(fj) - 1))], fj[int(0.1 * (len(fj) - 1))]
    cases = {
        "large_error_low_gt_motion": ("left_arm", pct([r for r in L if r["stratum"] == "low"], lambda r: r["res1"], 0.95)),
        "large_legitimate_gt_motion": ("left_arm", pct(L, lambda r: abs(r["dG"]), 0.95)),
        "correct_g1_depth_tracking": ("left_arm", pct([r for r in L if r["stratum"] == "high" and r["res1"] < r["res0"]],
                                                      lambda r: r["res0"] - r["res1"], 0.5)),
        "detector_instability": ("left_arm", pct(L, lambda r: r["jitter"], 0.99)),
        "crop_readout_change": ("left_arm", pct(L, lambda r: r["crop_scale"] + r["crop_shift"], 0.99)),
        "high_feature_small_output": ("left_arm", pct([r for r in L if r["feat_joints"] >= hi_f], absdp, 0.1)),
        "small_feature_large_output": ("left_arm", pct([r for r in L if r["feat_joints"] <= lo_f], absdp, 0.9)),
        "contradicts_visual_hypothesis": ("left_arm", pct([r for r in L if excess(r) >= 5 and abs(r["phi"]["geometry"]) > abs(r["phi"]["visual"])],
                                                          excess, 0.5)),
    }
    by_r, by_lg = {_key(r): r for r in R}, {_key(r): r for r in Lg}
    both = [(r, by_r[_key(r)]) for r in L if _key(r) in by_r and excess(r) >= 5 and excess(by_r[_key(r)]) >= 5]
    both = sorted(both, key=lambda t: (min(excess(t[0]), excess(t[1])), _key(t[0])))
    cases["similar_error_both_arms"] = ("left_arm+right_arm", both[len(both) // 2][0] if both else None)
    con = [(r, by_lg[_key(r)]) for r in L if _key(r) in by_lg and excess(r) >= 5 and excess(by_lg[_key(r)]) <= 0]
    con = sorted(con, key=lambda t: (excess(t[0]), _key(t[0])))
    cases["right_leg_contrast"] = ("right_leg", con[len(con) // 2][1] if con else None)
    at = {_key(r): r for r in L}
    osc = None
    for r in sorted(L, key=_key):
        run = [r, at.get((r["sequence_id"], r["frame"] + STRIDE)), at.get((r["sequence_id"], r["frame"] + 2 * STRIDE))]
        if all(x is not None and abs(x["dG"]) < LOW_GT_MOTION and abs(x["dP1"]) >= FALSE_MOTION for x in run) \
                and np.sign(run[0]["dP1"]) != np.sign(run[1]["dP1"]) != np.sign(run[2]["dP1"]):
            osc = r
            break
    cases["g1_false_depth_oscillation"] = ("left_arm", osc)
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("attribution_dir", "predictions", "bank", "image_root", "out_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    pairs = json.loads((args.attribution_dir / "pairs.json").read_text())
    pred = np.load(args.predictions)
    bank = load_bank(args.bank)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pos = {}
    for p in bank.indices("test"):
        s = bank.samples[p]
        pos[(s.sequence_id, int(s.frame_index))] = p
    by_pair = {k: {_key(r): r for r in v} for k, v in pairs.items()}

    def chain_xyz(chain, p, who):
        name = "target" if who == "gt" else who
        return np.array([[0, 0, 0], pred[f"{chain}_{name}_elbow"][p], pred[f"{chain}_{name}_wrist"][p]])

    def image_panel(ax, chain, p, title):
        s = bank.samples[p]
        img, inp, size = cv2.imread(str(args.image_root / s.image_reference.relative_path)), bank.arrays["input_2d"][p], s.image_size
        pts = np.array([[inp[j][0] * size[0], inp[j][1] * size[1]] for j in range(17) if inp[j][2] > 0])
        lo, hi = pts.min(0), pts.max(0)
        c, side = (lo + hi) / 2, max(max(hi - lo) * 1.3, 200)
        x0, y0 = int(max(0, c[0] - side / 2)), int(max(0, c[1] - side / 2))
        ax.imshow(cv2.cvtColor(img[y0:int(min(size[1], c[1] + side / 2)), x0:int(min(size[0], c[0] + side / 2))], cv2.COLOR_BGR2RGB))
        vis = [(inp[j][0] * size[0] - x0, inp[j][1] * size[1] - y0) for j in chain_indices(chain) if inp[j][2] > 0]
        ax.plot([q[0] for q in vis], [q[1] for q in vis], "-o", color="white", lw=2, ms=4)
        ax.set_title(title, fontsize=7)
        ax.axis("off")

    def side_panel(ax, chain, p, title):
        for who in ("gt", "g0", "g1"):
            q = chain_xyz(chain, p, who)
            ax.plot(q[:, 1], q[:, 2], "-o", color=STYLE[who][0], lw=STYLE[who][1], ms=3, label=who.upper())
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_aspect("equal")
        ax.axhline(0, color="#ccc", lw=0.5)
        ax.axvline(0, color="#ccc", lw=0.5)
        ax.set_xlabel("Y depth", fontsize=7)
        ax.set_ylabel("Z up", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_title(title, fontsize=7)
        ax.legend(fontsize=6, loc="lower right")

    def trace_panel(ax, chain, r):
        sid, f = _key(r)
        frames = [f + k * STRIDE for k in range(-WINDOW, WINDOW + 1) if (sid, f + k * STRIDE) in pos]
        for who in ("gt", "g0", "g1"):
            name = "target" if who == "gt" else who
            # GT is drawn only where the target is valid (invalid targets are stored as zeros)
            ys = [100 * pred[f"{chain}_{name}_wrist"][pos[(sid, x)]][1]
                  if who != "gt" or pred[f"{chain}_target_valid"][pos[(sid, x)]] else np.nan for x in frames]
            ax.plot(frames, ys, "-o",
                    color=STYLE[who][0], lw=STYLE[who][1], ms=2, label=f"{who.upper()} distal depth")
        ax.axvspan(f, f + STRIDE, color="#ffe08a", alpha=0.5)
        ax.set_xlabel("frame (30 fps; TEST rows 5 frames apart)", fontsize=7)
        ax.set_ylabel("distal depth Y (%L)", fontsize=7)
        ax.tick_params(labelsize=6)
        rows = by_pair[f"{chain}_3dpw_test"]
        tw = ax.twinx()
        fx = [x for x in frames if (sid, x) in rows]
        tw.bar([x + STRIDE / 2 for x in fx], [rows[(sid, x)]["feat_joints"] for x in fx], width=STRIDE * 0.6, color="#888", alpha=0.3)
        tw.set_ylabel("joint readout rel. change (bars)", fontsize=6)
        tw.tick_params(labelsize=6)
        ax.legend(fontsize=6, loc="upper left")

    def caption(chain, r):
        ph = r["phi"]
        return (f"{chain}  {r['sequence_id']}  frames {r['frame']}->{r['frame'] + STRIDE}  stratum {r['stratum']}   "
                f"dG {r['dG']:+.1f}  dP0 {r['dP0']:+.1f}  dP1 {r['dP1']:+.1f}  |res| G0 {r['res0']:.1f}  G1 {r['res1']:.1f}  (%L)\n"
                f"Shapley of dP1: geometry {ph['geometry']:+.1f}  content {ph['content']:+.1f}  position {ph['position']:+.1f}  "
                f"global {ph['global']:+.1f}   detector disp {r['det_disp_px']:.1f}px  jitter {r['jitter']:.3f}  dconf {r['dconf']:+.2f}  "
                f"crop scale {r['crop_scale']:.3f} shift {r['crop_shift']:.3f}  readout change joints {r['feat_joints']:.2f} global {r['feat_global']:.2f}")

    def render(name, chain, r):
        sid, f = _key(r)
        strip = [(sid, f - STRIDE), (sid, f), (sid, f + STRIDE), (sid, f + 2 * STRIDE)]
        labels = ["previous", "pair frame a", "pair frame b", "next"]
        fig = plt.figure(figsize=(13, 7.6))
        gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1])
        for i, (k, lab) in enumerate(zip(strip, labels)):
            ax = fig.add_subplot(gs[0, i])
            if k in pos:
                image_panel(ax, chain, pos[k], f"{lab}  frame {k[1]}")
            else:
                ax.axis("off")
                ax.set_title(f"{lab}: not a TEST row", fontsize=7)
        side_panel(fig.add_subplot(gs[1, 0]), chain, pos[strip[1]], "side, frame a")
        side_panel(fig.add_subplot(gs[1, 1]), chain, pos[strip[2]], "side, frame b")
        trace_panel(fig.add_subplot(gs[1, 2:]), chain, r)
        fig.suptitle(f"{name}\n" + caption(chain, r), fontsize=7, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(args.out_dir / f"{name}.png", dpi=120)
        plt.close(fig)

    index = {}
    for name, (chain, r) in select(pairs).items():
        if r is None:
            index[name] = None
            continue
        if chain == "left_arm+right_arm":
            render(name + "_left_arm", "left_arm", r)
            render(name + "_right_arm", "right_arm", by_pair["right_arm_3dpw_test"][_key(r)])
        else:
            render(name, chain, r)
        index[name] = [chain, *_key(r)]
    (args.out_dir / "review_index.json").write_text(json.dumps(index, indent=1))
    print(json.dumps(index, indent=1))


if __name__ == "__main__":
    main()
