#!/usr/bin/env python3
"""W111 figures (presentation only; not outcome-determining and not bound by the declaration's script SHAs).

  mixed_variants.png    mixed synthetic Action, frames 8.5-17.5: held hand_l tail z and smooth sibling tail x for the
                        source, the untouched import, blanket CONSTANT and the adapter on all ten producer variants
  real_pose_variants.png  complete_control at the worst untouched held-transition sample: source vs untouched vs the
                        adapter on two variants that differ in seed, path and run
  mixed_error_time.png  mixed held / smooth component world error per sample time (log scale), including the f+0.99
                        near-boundary probes: the sub-millimetre smooth residual and the untouched / blanket failures
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT/"scripts"))
import w110_fbx_metadata_recovery_contract as C10  # noqa: E402
import w111_fbx_semantic_repeatability_contract as C  # noqa: E402
from w108_fbx_subframe import wjson  # noqa: E402

_ld = C10.load_npz
INK, MUTED, GRID, BG = "#1f2328", "#59636e", "#d0d7de", "#fcfcfb"
STYLE = {"source": (INK, "-", 2.6), "U": ("#eb6834", "--", 1.6), "K": ("#1baf7a", ":", 2.0), "A": ("#2a78d6", "-", 1.0)}
LABEL = {"source": "source Action", "U": "untouched import (all LINEAR)", "K": "blanket CONSTANT",
         "A": "FBX-metadata adapter (10 variants)"}
VARIANTS = [(r, c) for r in C.RUNS for c, _, _ in C.CELLS]


def _style(ax):
    ax.tick_params(colors=MUTED, labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.6)


def _series(res, run, cell, cid, cond=None):
    d = res/run/("cells" if cond is None else "consume")/cell/cid
    stem = "source" if cond is None else f"meta_{cond}"
    main, nb = _ld(d/f"{stem}.npz"), _ld(d/f"{stem}_nb.npz")
    return main, nb


def _cat(main, nb, field, bone, coord):
    j = [str(b) for b in main["bones"]].index(bone)
    t = np.concatenate([main["times"], nb["times"]])
    y = np.concatenate([main[field][:, j, coord], nb[field][:, j, coord]])
    o = np.argsort(t)
    return t[o], y[o]


def figures(o):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res, out = Path(o.results), Path(o.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    rec = {}
    cid = C.MIXED
    # 1) mixed trajectories, all ten adapter variants overlaid
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True, facecolor=BG)
    spread = 0.0
    for ax, (bone, coord, title) in zip(axes, (("hand_l", 2, "held chain: hand_l tail, world z (m)"),
                                               ("sibling", 0, "smooth bone: sibling tail, world x (m)"))):
        s = _series(res, "run1", C.REF_CELL, cid)
        ax.plot(*_cat(*s, "world_tail", bone, coord), color=STYLE["source"][0], ls="-", lw=2.6, label=LABEL["source"])
        for k in ("U", "K"):
            ax.plot(*_cat(*_series(res, "run1", C.REF_CELL, cid, k), "world_tail", bone, coord), color=STYLE[k][0],
                    ls=STYLE[k][1], lw=STYLE[k][2], label=LABEL[k])
        ys = []
        for i, (r, c) in enumerate(VARIANTS):
            t, y = _cat(*_series(res, r, c, cid, "A"), "world_tail", bone, coord)
            ys.append(y)
            ax.plot(t, y, color=STYLE["A"][0], lw=1.0, alpha=0.6, label=LABEL["A"] if i == 0 else None)
        spread = max(spread, float(np.ptp(np.stack(ys), axis=0).max()))
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.set_xlim(8.5, 17.5)
        _style(ax)
    axes[0].legend(frameon=False, fontsize=8, loc="best")
    axes[1].set_xlabel("frame (quarter-frame grid + f+0.99 probes)", color=MUTED, fontsize=9)
    fig.suptitle("W111 mixed synthetic Action: the adapter result of all 10 seed/path/run variants coincides",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out/"mixed_variants.png", dpi=130)
    plt.close(fig)
    rec["mixed_adapter_variant_spread_m"] = spread
    # 2) real-rig pose at the worst untouched transition sample, two maximally different variants
    cid = "complete_control"
    src = _ld(res/"run1/cells"/C.REF_CELL/cid/"source.npz")
    u = _ld(res/"run1/consume"/C.REF_CELL/cid/"meta_U.npz")
    a1 = _ld(res/"run1/consume"/C.REF_CELL/cid/"meta_A.npz")
    a2 = _ld(res/"run2/consume/s2_pB"/cid/"meta_A.npz")
    names = [str(b) for b in src["bones"]]
    ib = [[str(b) for b in u["bones"]].index(n) for n in names]
    err = np.abs(src["world_tail"] - u["world_tail"][:, ib]).max(axis=(1, 2))
    ti = int(np.argmax(err))
    fig, axes = plt.subplots(1, 4, figsize=(12, 4.4), sharey=True, facecolor=BG)
    panels = (("source Action", src, INK, list(range(len(names)))), ("untouched import", u, STYLE["U"][0], ib),
              ("adapter: run1 s1_pA", a1, STYLE["A"][0], ib), ("adapter: run2 s2_pB", a2, STYLE["A"][0], ib))
    for ax, (title, d, col, jj) in zip(axes, panels):
        if d is not src:
            for j in range(len(names)):
                h, tl = src["world_head"][ti, j], src["world_tail"][ti, j]
                ax.plot([h[0], tl[0]], [h[2], tl[2]], color="#8c959f", lw=0.8, alpha=0.6)
        for j in jj:
            h, tl = d["world_head"][ti, j], d["world_tail"][ti, j]
            ax.plot([h[0], tl[0]], [h[2], tl[2]], color=col, lw=1.6)
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        ax.set_aspect("equal")
        _style(ax)
        ax.grid(False)
    fig.suptitle(f"complete_control at t = {float(src['times'][ti]):.2f} (worst untouched sample; grey = source)",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out/"real_pose_variants.png", dpi=130)
    plt.close(fig)
    rec.update({"real_pose_time": float(src["times"][ti]), "real_pose_untouched_err_m": float(err[ti]),
                "real_pose_adapter_err_m": [float(np.abs(src["world_tail"][ti] - a["world_tail"][ti][ib]).max())
                                            for a in (a1, a2)],
                "real_pose_adapter_variants_max_diff_m": float(np.abs(a1["world_tail"] - a2["world_tail"]).max())})
    # 3) mixed component error over time (log scale)
    cid = C.MIXED
    s_main, s_nb = _series(res, "run1", C.REF_CELL, cid)
    comp = C10.component_points(C10.MIXED_SPEC)
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.0), sharex=True, facecolor=BG)
    worst = {}
    for ax, part in zip(axes, ("held", "smooth")):
        p = comp[part]
        for k in ("U", "K", "A"):
            m, nb = _series(res, "run1", C.REF_CELL, cid, k)
            e = []
            t = np.concatenate([m["times"], nb["times"]])
            for a, b in ((s_main, m), (s_nb, nb)):
                e.append(np.abs(C10._points(a, p["heads"], p["tails"]) - C10._points(b, p["heads"], p["tails"]))
                         .max(axis=(1, 2)))
            e = np.maximum(np.concatenate(e), 1e-9)
            o = np.argsort(t)
            ax.plot(t[o], e[o], color=STYLE[k][0], ls=STYLE[k][1], lw=1.4, label=LABEL[k].replace(" (10 variants)", ""))
            if k == "A":
                worst[part] = {"max_m": float(e.max()), "at": float(t[int(np.argmax(e))])}
        ax.axhline(1e-3, color=MUTED, lw=0.8, ls="--")
        ax.text(1.0, 1.15e-3, "1 mm gate", color=MUTED, fontsize=7)
        ax.set_yscale("log")
        ax.set_title(f"mixed {part} component: max world error (m)", loc="left", fontsize=10, color=INK)
        _style(ax)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    axes[1].set_xlabel("frame (quarter-frame grid + f+0.99 probes)", color=MUTED, fontsize=9)
    fig.tight_layout()
    fig.savefig(out/"mixed_error_time.png", dpi=130)
    plt.close(fig)
    rec["mixed_adapter_worst"] = worst
    for f in ("mixed_variants.png", "real_pose_variants.png", "mixed_error_time.png"):
        rec[f] = C10.sha256_file(out/f)
    wjson(out/"figures.json", rec)
    print(json.dumps(rec))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out-dir", required=True)
    return figures(ap.parse_args())


if __name__ == "__main__":
    sys.exit(main())
