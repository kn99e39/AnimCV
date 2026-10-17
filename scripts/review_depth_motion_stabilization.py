#!/usr/bin/env python3
"""REVIEW (Worklog 89): consecutive-frame A/B video clips of raw G1, G0 and the stabilized candidate (3DPW TEST).

Each clip covers ``WINDOW`` consecutive TEST rows (5 frames = 1/6 s apart, played in real time at 6 fps) of one track
and shows: the source RGB with the detector chain (runtime input, white), a side view (depth x up) of GT (green),
G0 (orange), raw G1 (cyan) and stabilized C (magenta), and the distal camera-depth trajectories with a moving cursor
and per-pair annotations (false-motion / stale-hold flags evaluated against GT).  A summary PNG (the last frame) is
written next to each clip.  The stabilizer is rerun from the frozen ``config.json`` (deterministic).

Fixed selection rules over windows (start every 6 rows, >= 30 GT-valid rows; ties by (sequence, start)):
  stationary_limb            left arm, min mean GT distal 3D displacement per pair
  walking_arm_swing          left arm, max GT distal-depth velocity sign changes among windows with GT depth range >= 30 %L
  fast_arm_movement          left arm, max 95th percentile GT distal 3D displacement per pair
  toward_away_camera         left arm, max sum |dG_y| among windows whose mean detector 2D distal displacement is below the
                             median window
  strong_out_of_plane        left arm, max mean |GT full-chain depth direction|
  depth_sign_change          left arm, max GT distal-depth sign changes (|Y| >= 0.15 on both sides)
  occlusion                  left arm, max share of rows with an invalid detector chain joint
  confidence_degradation     left arm, min mean detector chain confidence among windows with >= 80% valid rows
  historical_false_motion    left arm, max sum over low-GT-motion pairs of (|dP_G1| - 5)+
  historical_correct_g1      left arm, max sum over high-GT-motion pairs of (res_G0 - res_G1)
  worst_regression_left_arm  left arm, max sum over pairs of (res_C - res_G1)
  worst_regression_right_leg right leg, max sum over pairs of (res_C - res_G1)
  worst_static_regression    left arm, max mean (distal error C - distal error G1)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from framepose.connected_limb_transfer import chain_indices, chain_targets  # noqa: E402
from framepose.depth_motion_stabilizer import StabilizerConfig  # noqa: E402
from framepose.visual_depth_motion_attribution import FALSE_MOTION, HIGH_GT_MOTION, LOW_GT_MOTION  # noqa: E402
from run_depth_motion_stabilization import STALE, build_populations, run_stabilizer  # noqa: E402

STYLE = {"gt": ("#22aa22", 3.0, "GT"), "B": ("#ff8c1a", 1.6, "G0"), "A": ("#1ad9f2", 1.6, "G1 raw"), "C": ("#d61ad6", 2.2, "G1 stabilized")}
WINDOW, START_EVERY, MIN_VALID = 36, 6, 30


def windows(pop, rows_sorted):
    """Track-contiguous windows of TEST rows (consecutive rows exactly 5 frames apart)."""
    out, track = [], [rows_sorted[0]]
    for p in rows_sorted[1:]:
        a, b = pop.bank.samples[track[-1]], pop.bank.samples[p]
        if a.sequence_id == b.sequence_id and b.frame_index - a.frame_index == 5:
            track.append(p)
        else:
            out.append(track)
            track = [p]
    out.append(track)
    return [np.array(t[s:s + WINDOW]) for t in out for s in range(0, max(len(t) - WINDOW, 0) + 1, START_EVERY) if len(t) >= WINDOW]


def window_stats(pop, chain, w, mods, gt):
    e_t, w_t, ok = gt
    v = ok[w]
    if v.sum() < MIN_VALID:
        return None
    d3 = 100 * np.linalg.norm(np.diff(w_t[w], axis=0), axis=1)
    pair_ok = v[1:] & v[:-1]
    dy = 100 * np.diff(w_t[w, 1])
    det = pop.bank.arrays["input_2d"][w][:, chain_indices(chain)]
    size = np.array([pop.bank.samples[p].image_size for p in w], float)
    d2 = np.linalg.norm(np.diff(det[:, 2, :2] * size, axis=0), axis=1)
    res = {k: np.abs(100 * np.diff(mods[k][1][w, 1]) - dy) for k in ("A", "B", "C")}
    dpa = 100 * np.diff(mods["A"][1][w, 1])
    low, high = pair_ok & (d3 < LOW_GT_MOTION), pair_ok & (d3 >= HIGH_GT_MOTION)
    yv = 100 * w_t[w, 1][v]
    vel = np.sign(dy[pair_ok])
    sig = w_t[w, 1][v]
    sig = np.sign(sig[np.abs(sig) >= 0.15])
    cy = w_t[w][v] / np.linalg.norm(w_t[w][v], axis=1, keepdims=True)
    errA = 100 * np.linalg.norm(mods["A"][1][w][v] - w_t[w][v], axis=1)
    errC = 100 * np.linalg.norm(mods["C"][1][w][v] - w_t[w][v], axis=1)
    obs = pop.bank.arrays["input_valid"][w][:, chain_indices(chain)].all(1)
    return {"motion": float(d3[pair_ok].mean()), "range": float(yv.max() - yv.min()), "vel_changes": int((np.diff(vel[vel != 0]) != 0).sum()),
            "p95_motion": float(np.percentile(d3[pair_ok], 95)), "sum_dy": float(np.abs(dy[pair_ok]).sum()), "det2d": float(d2[pair_ok].mean()),
            "out_of_plane": float(np.abs(cy[:, 1]).mean()), "sign_changes": int((np.diff(sig) != 0).sum()) if len(sig) > 1 else 0,
            "invalid": float(1 - obs.mean()), "conf": float(det[:, :, 2].min(1).mean()),
            "false_motion": float(np.clip(np.abs(dpa[low]) - FALSE_MOTION, 0, None).sum()),
            "correct_g1": float((res["B"][high] - res["A"][high]).sum()), "regression": float((res["C"][pair_ok] - res["A"][pair_ok]).sum()),
            "static_regression": float((errC - errA).mean())}


def select(stats):
    def best(chain, key, sign=1, where=lambda s: True):
        c = [(k, s) for k, s in stats[chain].items() if s is not None and where(s)]
        return min(c, key=lambda t: (-sign * t[1][key], t[0]))[0] if c else None
    det_med = np.median([s["det2d"] for s in stats["left_arm"].values() if s is not None])
    return {"stationary_limb": ("left_arm", best("left_arm", "motion", -1)),
            "walking_arm_swing": ("left_arm", best("left_arm", "vel_changes", 1, lambda s: s["range"] >= 30)),
            "fast_arm_movement": ("left_arm", best("left_arm", "p95_motion")),
            "toward_away_camera": ("left_arm", best("left_arm", "sum_dy", 1, lambda s: s["det2d"] < det_med)),
            "strong_out_of_plane": ("left_arm", best("left_arm", "out_of_plane")),
            "depth_sign_change": ("left_arm", best("left_arm", "sign_changes")),
            "occlusion": ("left_arm", best("left_arm", "invalid")),
            "confidence_degradation": ("left_arm", best("left_arm", "conf", -1, lambda s: s["invalid"] <= 0.2)),
            "historical_false_motion": ("left_arm", best("left_arm", "false_motion")),
            "historical_correct_g1": ("left_arm", best("left_arm", "correct_g1")),
            "worst_regression_left_arm": ("left_arm", best("left_arm", "regression")),
            "worst_regression_right_leg": ("right_leg", best("right_leg", "regression")),
            "worst_static_regression": ("left_arm", best("left_arm", "static_regression"))}


def render(path, pop, chain, w, mods, gt, image_root, title):
    e_t, w_t, ok = gt
    times = pop.times[w] - pop.times[w[0]]
    ys = {k: 100 * mods[k][1][w, 1] for k in ("A", "B", "C")}
    ys["gt"] = np.where(ok[w], 100 * w_t[w, 1], np.nan)
    idx = chain_indices(chain)
    writer, last = None, None
    for i, p in enumerate(w):
        s = pop.bank.samples[p]
        fig = plt.figure(figsize=(12.8, 6.4), dpi=80)
        gs = fig.add_gridspec(2, 3, width_ratios=[1.2, 0.9, 1.6], height_ratios=[1, 0.18])
        ax_img, ax_side, ax_tr = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])
        img = cv2.imread(str(image_root / s.image_reference.relative_path))
        inp, size = pop.bank.arrays["input_2d"][p], s.image_size
        pts = np.array([[inp[j][0] * size[0], inp[j][1] * size[1]] for j in range(17) if inp[j][2] > 0])
        if len(pts):
            lo, hi = pts.min(0), pts.max(0)
            c, side = (lo + hi) / 2, max(max(hi - lo) * 1.3, 200)
        else:
            c, side = np.array(size) / 2, max(size)
        x0, y0 = int(max(0, c[0] - side / 2)), int(max(0, c[1] - side / 2))
        ax_img.imshow(cv2.cvtColor(img[y0:int(min(size[1], c[1] + side / 2)), x0:int(min(size[0], c[0] + side / 2))], cv2.COLOR_BGR2RGB))
        vis = [(inp[j][0] * size[0] - x0, inp[j][1] * size[1] - y0) for j in idx if inp[j][2] > 0 and pop.bank.arrays["input_valid"][p][j]]
        if vis:
            ax_img.plot([q[0] for q in vis], [q[1] for q in vis], "-o", color="white", lw=2, ms=4)
        conf = float(inp[idx, 2].min())
        ax_img.set_title(f"{s.sequence_id.split(':', 1)[1]}  frame {s.frame_index}  t={times[i]:.2f}s  det conf {conf:.2f}", fontsize=8)
        ax_img.axis("off")
        chains = {"B": mods["B"], "A": mods["A"], "C": mods["C"]}
        if ok[p]:
            chains = {"gt": (e_t, w_t), **chains}
        for k, (e, ww) in chains.items():
            q = np.array([[0, 0, 0], e[p], ww[p]])
            ax_side.plot(q[:, 1], q[:, 2], "-o", color=STYLE[k][0], lw=STYLE[k][1], ms=3, label=STYLE[k][2])
        ax_side.set_xlim(-1.05, 1.05)
        ax_side.set_ylim(-1.05, 1.05)
        ax_side.set_aspect("equal")
        ax_side.axhline(0, color="#ccc", lw=0.5)
        ax_side.axvline(0, color="#ccc", lw=0.5)
        ax_side.set_xlabel("Y depth", fontsize=7)
        ax_side.set_ylabel("Z up", fontsize=7)
        ax_side.set_title(f"side view ({chain})", fontsize=8)
        ax_side.legend(fontsize=6, loc="lower right")
        ax_side.tick_params(labelsize=6)
        for k in ("gt", "B", "A", "C"):
            ax_tr.plot(times, ys[k], "-", color=STYLE[k][0], lw=STYLE[k][1], label=STYLE[k][2])
        ax_tr.axvline(times[i], color="k", lw=0.8)
        ax_tr.set_xlabel("time (s)", fontsize=7)
        ax_tr.set_ylabel("distal depth Y (%L)", fontsize=7)
        ax_tr.tick_params(labelsize=6)
        ax_tr.legend(fontsize=6, loc="upper left", ncol=4)
        ax_tr.set_title(title, fontsize=8)
        ax_txt = fig.add_subplot(gs[1, :])
        ax_txt.axis("off")
        if i > 0 and ok[p] and ok[w[i - 1]]:
            dg = ys["gt"][i] - ys["gt"][i - 1]
            d3 = 100 * np.linalg.norm(w_t[p] - w_t[w[i - 1]])
            parts = [f"pair {s.frame_index - 5}->{s.frame_index}: GT 3D motion {d3:.1f} %L, GT dY {dg:+.1f}"]
            for k in ("B", "A", "C"):
                dp = ys[k][i] - ys[k][i - 1]
                flag = " FALSE-MOTION" if d3 < LOW_GT_MOTION and abs(dp) > FALSE_MOTION else (
                    " STALE" if d3 >= HIGH_GT_MOTION and abs(dp) < STALE else "")
                parts.append(f"{STYLE[k][2]} dY {dp:+.1f} (|res| {abs(dp - dg):.1f}){flag}")
            ax_txt.text(0, 0.5, "   ".join(parts), fontsize=8, va="center")
        else:
            ax_txt.text(0, 0.5, "no GT pair for this step (GT invalid or first row)", fontsize=8, va="center")
        fig.tight_layout()
        fig.canvas.draw()
        frame = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba())[:, :, :3], cv2.COLOR_RGB2BGR)
        if writer is None:
            writer = cv2.VideoWriter(str(path.with_suffix(".mp4")), cv2.VideoWriter_fourcc(*"mp4v"), 6, (frame.shape[1], frame.shape[0]))
        writer.write(frame)
        last = frame
        plt.close(fig)
    writer.release()
    cv2.imwrite(str(path.with_suffix(".png")), last)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bank", "w83_dir", "w87_dir", "stabilization_dir", "image_root", "out_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pop = build_populations(args)["3dpw_test"]
    cfg = json.loads((args.stabilization_dir / "config.json").read_text())["chains"]
    rows = np.array(sorted(pop.rows, key=lambda p: (pop.bank.samples[p].sequence_id, pop.bank.samples[p].frame_index)))
    wins = windows(pop, rows)
    mods, gts, stats = {}, {}, {}
    for chain in ("left_arm", "right_leg"):
        e_c, w_c, _ = run_stabilizer(pop, chain, StabilizerConfig(**cfg[chain]["offline"]))
        p = pop.pred
        mods[chain] = {"A": (p[f"{chain}_g1_elbow"], p[f"{chain}_g1_wrist"]), "B": (p[f"{chain}_g0_elbow"], p[f"{chain}_g0_wrist"]), "C": (e_c, w_c)}
        gts[chain] = chain_targets(pop.bank.arrays["target_3d"], pop.bank.arrays["target_valid"], chain)
        stats[chain] = {(pop.bank.samples[w[0]].sequence_id, int(pop.bank.samples[w[0]].frame_index)): window_stats(pop, chain, w, mods[chain], gts[chain])
                        for w in wins}
    by_key = {(pop.bank.samples[w[0]].sequence_id, int(pop.bank.samples[w[0]].frame_index)): w for w in wins}
    index = {}
    for name, (chain, key) in select(stats).items():
        if key is None:
            index[name] = None
            continue
        render(args.out_dir / name, pop, chain, by_key[key], mods[chain], gts[chain], args.image_root, name)
        index[name] = {"chain": chain, "sequence_id": key[0], "start_frame": key[1], "rows": WINDOW, "stats": stats[chain][key]}
        print(name, index[name], flush=True)
    (args.out_dir / "review_index.json").write_text(json.dumps(index, indent=1))


if __name__ == "__main__":
    main()
