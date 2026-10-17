#!/usr/bin/env python3
"""DIAGNOSTIC (Worklog 87): fixed-rule real-scene review of near-plane false depth (frozen G0 / G1 only).

3DPW TEST (the primary diagnostic population).  Panels: the exact source image with the detector chain (runtime
input, white), front (X right x Z up) and side (depth x up) views of the unit chain for GT (green), G0 (orange,
2D only) and G1 (cyan, 2D + frozen Qwen; J1X for the left arm), with regime, depth components and errors.
Fixed selection rules (declared before reviewing; frozen baselines only):
  * truly near-plane arm / leg: left arm / right leg rows in A_true_near, median G1 distal error;
  * chain-near-plane cancellation: left arm B_cancellation, median G1 distal error;
  * strongly out-of-plane arm / leg: C_out_of_plane, median G1 distal error;
  * large historical false depth: left arm A_true_near, 95th percentile of G1 (|pred depth| - |GT depth|);
  * correct historical visual depth: left arm C_out_of_plane rows where G1 beats G0, median improvement;
  * severe regression: left arm, 99th percentile of G1 - G0 distal error (all regimes);
  * temporal depth-sign instability: first left-arm pair (sequence, frame order) where G1's distal depth sign flips
    while GT keeps a stable sign (|f_chain| >= sin 10 deg in both frames) -- 2-frame strip;
  * high / low confidence: left arm rows in the top / bottom decile of min detector confidence, median G1 distal error.
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
from framepose.connected_limb_transfer import chain_indices  # noqa: E402
from framepose.near_plane_attribution import STABLE, depth_components  # noqa: E402

STYLE = {"gt": ("#22aa22", 3.0), "g0": ("#ff8c1a", 1.8), "g1": ("#1ad9f2", 1.8)}


def _row(d, i):
    return {k: (np.asarray(v[i]) if isinstance(v[i], list) else v[i]) for k, v in d.items()}


def _err(a, b):
    return 100 * float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def panel(axes, r, chain, image, inp, size, title):
    a_img, a_front, a_side = axes
    idx = chain_indices(chain)
    pts = np.array([[inp[j][0] * size[0], inp[j][1] * size[1]] for j in range(17) if inp[j][2] > 0])
    lo, hi = pts.min(0), pts.max(0)
    c, side = (lo + hi) / 2, max(max(hi - lo) * 1.3, 200)
    x0, y0 = int(max(0, c[0] - side / 2)), int(max(0, c[1] - side / 2))
    crop = image[y0:int(min(size[1], c[1] + side / 2)), x0:int(min(size[0], c[0] + side / 2))]
    a_img.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    vis = [(inp[j][0] * size[0] - x0, inp[j][1] * size[1] - y0) for j in idx if inp[j][2] > 0]
    a_img.plot([p[0] for p in vis], [p[1] for p in vis], "-o", color="white", lw=2, ms=4)
    a_img.set_title(title, fontsize=7)
    a_img.axis("off")
    for ax, (i, j, xl) in ((a_front, (0, 2, "X right")), (a_side, (1, 2, "Y depth"))):
        for name, (e, w) in (("gt", (r["ge"], r["gw"])), ("g0", (r["g0_e"], r["g0_w"])), ("g1", (r["g1_e"], r["g1_w"]))):
            p = np.array([[0, 0, 0], e, w])
            ax.plot(p[:, i], p[:, j], "-o", color=STYLE[name][0], lw=STYLE[name][1], ms=3, label=name.upper())
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


def caption(r, chain):
    fp, fd, fc, reach = depth_components(np.asarray([r["ge"]]), np.asarray([r["gw"]]))
    return (f"{chain}  {r['sequence_id']}  frame {r['frame_index']}  regime {r['regime']}  GT f prox {fp[0]:+.2f} dist {fd[0]:+.2f} chain {fc[0]:+.2f}"
            f"  min conf {r['conf']:.2f}  det-vs-GT disc {r['disc']:.3f}\n"
            f"distal %L: G0 {_err(r['g0_w'], r['gw']):.1f}  G1 {_err(r['g1_w'], r['gw']):.1f}   middle %L: G0 {_err(r['g0_e'], r['ge']):.1f}  G1 {_err(r['g1_e'], r['ge']):.1f}"
            f"   distal depth (Y) x100: GT {100 * r['gw'][1]:+.1f}  G0 {100 * r['g0_w'][1]:+.1f}  G1 {100 * r['g1_w'][1]:+.1f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("attribution_dir", "bank", "image_root", "out_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    dumps = json.loads((args.attribution_dir / "rows.json").read_text())
    bank = load_bank(args.bank)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    def rows(chain):
        d = dumps[f"{chain}_3dpw_test"]
        return [_row(d, i) for i in range(len(d["position"]))]
    L, Lg = rows("left_arm"), rows("right_leg")
    key = lambda r: (r["sequence_id"], r["frame_index"])  # noqa: E731
    d1 = lambda r: _err(r["g1_w"], r["gw"])  # noqa: E731
    dd = lambda r: _err(r["g1_w"], r["gw"]) - _err(r["g0_w"], r["gw"])  # noqa: E731
    od = lambda r: abs(r["g1_w"][1]) - abs(r["gw"][1])  # noqa: E731

    def median_of(group, fn=d1):
        g = sorted(group, key=lambda r: (fn(r), key(r)))
        return g[len(g) // 2] if g else None

    def pct(group, fn, q):
        g = sorted(group, key=lambda r: (fn(r), key(r)))
        return g[int(q * (len(g) - 1))] if g else None
    reg = lambda rs, name: [r for r in rs if r["regime"] == name]  # noqa: E731
    confs = sorted(r["conf"] for r in L)
    stills = {"truly_near_plane_arm": ("left_arm", median_of(reg(L, "A_true_near"))),
              "truly_near_plane_leg": ("right_leg", median_of(reg(Lg, "A_true_near"))),
              "chain_near_plane_cancellation": ("left_arm", median_of(reg(L, "B_cancellation"))),
              "strongly_out_of_plane_arm": ("left_arm", median_of(reg(L, "C_out_of_plane"))),
              "strongly_out_of_plane_leg": ("right_leg", median_of(reg(Lg, "C_out_of_plane"))),
              "large_historical_false_depth": ("left_arm", pct(reg(L, "A_true_near"), od, 0.95)),
              "correct_historical_visual_depth": ("left_arm", median_of([r for r in reg(L, "C_out_of_plane") if dd(r) < 0], dd)),
              "severe_regression": ("left_arm", pct(L, dd, 0.99)),
              "high_confidence_observation": ("left_arm", median_of([r for r in L if r["conf"] >= confs[int(0.9 * (len(confs) - 1))]])),
              "low_confidence_observation": ("left_arm", median_of([r for r in L if r["conf"] <= confs[int(0.1 * (len(confs) - 1))]]))}
    at = {key(r): r for r in L}
    flip = None
    for r in sorted(L, key=key):
        nxt = at.get((r["sequence_id"], r["frame_index"] + 5))
        if nxt is None:
            continue
        fa = depth_components(np.asarray([r["ge"]]), np.asarray([r["gw"]]))[2][0]
        fb = depth_components(np.asarray([nxt["ge"]]), np.asarray([nxt["gw"]]))[2][0]
        if abs(fa) >= STABLE and abs(fb) >= STABLE and np.sign(r["gw"][1]) == np.sign(nxt["gw"][1]) and np.sign(r["g1_w"][1]) != np.sign(nxt["g1_w"][1]):
            flip = [r, nxt]
            break

    def load(r):
        s = bank.samples[r["position"]]
        return cv2.imread(str(args.image_root / s.image_reference.relative_path)), bank.arrays["input_2d"][r["position"]], tuple(s.image_size)
    index = {}
    for name, (chain, r) in stills.items():
        if r is None:
            index[name] = None
            continue
        fig, axes = plt.subplots(1, 3, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1.1, 1, 1]})
        panel(axes, r, chain, *load(r), f"{name} ({chain})")
        fig.suptitle(caption(r, chain), fontsize=7, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.85))
        fig.savefig(args.out_dir / f"{name}.png", dpi=130)
        plt.close(fig)
        index[name] = [chain, *key(r)]
    if flip:
        fig, axes = plt.subplots(2, 3, figsize=(11, 8.6), gridspec_kw={"width_ratios": [1.1, 1, 1]})
        for i, r in enumerate(flip):
            panel(axes[i], r, "left_arm", *load(r), f"temporal_depth_sign_instability [{i + 1}/2]")
            axes[i][0].text(0, -0.08, caption(r, "left_arm"), transform=axes[i][0].transAxes, fontsize=6, va="top")
        fig.tight_layout(h_pad=5.0)
        fig.savefig(args.out_dir / "temporal_depth_sign_instability.png", dpi=110)
        plt.close(fig)
        index["temporal_depth_sign_instability"] = [["left_arm", *key(r)] for r in flip]
    (args.out_dir / "review_index.json").write_text(json.dumps(index, indent=1))
    print(json.dumps(index, indent=1))


if __name__ == "__main__":
    main()
