#!/usr/bin/env python3
"""W110 figures (presentation only; not outcome-determining and not bound by the declaration's script SHAs).

  one mixed-curve trajectory plot (held hand_l tail z and smooth sibling tail x: source / untouched / blanket
  CONSTANT / FBX-metadata adapter) and one real-rig transition pose plot (complete_control at the worst untouched
  sample: source vs untouched vs adapter), for qualitative review of the stored W110 samples.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT/"scripts"))
import w110_fbx_metadata_recovery_contract as C  # noqa: E402
from w108_fbx_subframe import wjson  # noqa: E402

sha256 = C.sha256_file
_ld = C.load_npz


def figures(o):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    decl = json.loads(Path(o.declaration).read_bytes())
    out = Path(o.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    pd, cd = Path(o.produce_dir), Path(o.consume_dir)
    ink, muted = "#1f2328", "#59636e"
    style = {"source": ("#1f2328", "-", 2.6), "meta_U": ("#eb6834", "--", 1.6), "meta_K": ("#1baf7a", ":", 2.0),
             "meta_A": ("#2a78d6", "-", 1.2)}
    label = {"source": "source Action", "meta_U": "untouched import (all LINEAR)",
             "meta_K": "blanket CONSTANT", "meta_A": "FBX-metadata adapter"}
    # 1) mixed-curve example: one held point and one smooth point over time
    cid = C.MIXED
    series = {"source": (_ld(pd/cid/"source.npz"), _ld(pd/cid/"source_nb.npz"))}
    for k in ("meta_U", "meta_K", "meta_A"):
        series[k] = (_ld(cd/cid/f"{k}.npz"), _ld(cd/cid/f"{k}_nb.npz"))
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True, facecolor="#fcfcfb")
    for ax, (bone, title) in zip(axes, (("hand_l", "held chain: hand_l tail, world z (m)"),
                                        ("sibling", "smooth bone: sibling tail, world x (m)"))):
        coord = 2 if bone == "hand_l" else 0
        for k, (main, nb) in series.items():
            t = np.concatenate([main["times"], nb["times"]])
            j = [str(b) for b in main["bones"]].index(bone)
            y = np.concatenate([main["world_tail"][:, j, coord], nb["world_tail"][:, j, coord]])
            order = np.argsort(t)
            col, ls, lw = style[k]
            ax.plot(t[order], y[order], color=col, linestyle=ls, linewidth=lw, label=label[k])
        ax.set_title(title, loc="left", fontsize=10, color=ink)
        ax.tick_params(colors=muted, labelsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", color="#d0d7de", linewidth=0.6)
        ax.set_xlim(8.5, 17.5)
    axes[0].legend(frameon=False, fontsize=8, loc="best")
    axes[1].set_xlabel("frame (quarter-frame grid + f+0.99 probes)", color=muted, fontsize=9)
    fig.suptitle("W110 mixed synthetic Action, frames 8.5-17.5 (sibling's key at 13 is missing by construction)",
                 fontsize=10, color=ink, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out/"mixed_curve_example.png", dpi=130)
    plt.close(fig)
    # 2) real-rig pose at the worst untouched transition subframe (complete_control), front view x-z
    cid = "complete_control"
    src = _ld(pd/cid/"source.npz")
    imps = {k: _ld(cd/cid/f"{k}.npz") for k in ("meta_U", "meta_A")}
    names = [str(b) for b in src["bones"]]
    ib = [[str(b) for b in imps["meta_U"]["bones"]].index(n) for n in names]
    err = np.abs(src["world_tail"] - imps["meta_U"]["world_tail"][:, ib]).max(axis=(1, 2))
    ti = int(np.argmax(err))
    fig, axes = plt.subplots(1, 3, figsize=(10, 4.4), sharey=True, facecolor="#fcfcfb")
    for ax, (k, d) in zip(axes, (("source", src), ("meta_U", imps["meta_U"]), ("meta_A", imps["meta_A"]))):
        jj = list(range(len(names))) if k == "source" else ib
        for j in jj:
            h, tl = d["world_head"][ti, j], d["world_tail"][ti, j]
            ax.plot([h[0], tl[0]], [h[2], tl[2]], color=style[k][0], linewidth=1.6)
        if k != "source":
            for j in range(len(names)):
                h, tl = src["world_head"][ti, j], src["world_tail"][ti, j]
                ax.plot([h[0], tl[0]], [h[2], tl[2]], color="#8c959f", linewidth=0.8, alpha=0.6)
        ax.set_title(label[k], fontsize=9, color=ink, loc="left")
        ax.set_aspect("equal")
        ax.tick_params(colors=muted, labelsize=7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle(f"complete_control at t = {float(src['times'][ti]):.2f} (worst untouched sample; grey = source)",
                 fontsize=10, color=ink, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out/"real_pose_transition.png", dpi=130)
    plt.close(fig)
    rec = {"mixed_curve_example.png": sha256(out/"mixed_curve_example.png"),
           "real_pose_transition.png": sha256(out/"real_pose_transition.png"),
           "real_pose_time": float(src["times"][ti]), "real_pose_untouched_err_m": float(err[ti]),
           "real_pose_adapter_err_m": float(np.abs(src["world_tail"][ti] - imps["meta_A"]["world_tail"][ti][ib]).max())}
    wjson(out/"figures.json", rec)
    print(json.dumps(rec))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("declaration", "produce_dir", "consume_dir", "out_dir"):
        ap.add_argument("--"+name.replace("_", "-"), required=True)
    return figures(ap.parse_args())


if __name__ == "__main__":
    sys.exit(main())
