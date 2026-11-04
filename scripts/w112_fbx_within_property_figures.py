#!/usr/bin/env python3
"""W112 presentation-only figures (read the frozen analysis and samples; no gate, no decision input).

  bounds_summary.png      per case: untouched import, privileged import-level and FBX-level bounds, source-free
                          candidate (log metres, 1 mm line)
  held_to_smooth.png      syn_R_int: a held -> smooth -> held transition of the target bone's tail
  axis_mixed.png          per-interval best error of the axis-mixed cases at both layers vs group-uniform
  real_rig_pose.png       rr_axis at its worst import-level interval (near-boundary probe): source, untouched,
                          import-level best stick figures (X-Z projection)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT/"scripts"))
import w112_fbx_within_property_contract as C  # noqa: E402

# reference categorical slots 1-5 (fixed order), identity also carried by marker / line style
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, MUTED = "#2b2b2a", "#8a8987"


def _best_series(rd, cid, prefix, per):
    """Compose the per-interval best assignment's samples (integer samples from the first label)."""
    labels = sorted({v["best"] for v in per.values()})
    arr = {lab: C.load_npz(Path(rd)/"oracle"/cid/f"{prefix}_{lab}.npz") for lab in labels}
    first = arr[labels[0]]
    out = {k: np.array(first[k], copy=True) for k in ("world_head", "world_tail")}
    for i, f in enumerate(C.interval_index(first["times"])):
        if f is not None:
            lab = per[str(f)]["best"]
            for k in out:
                out[k][i] = arr[lab][k][i]
    out["times"], out["bones"] = first["times"], first["bones"]
    return out


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=8)
    ax.grid(True, color="#e6e5e1", linewidth=0.6)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--analysis", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out-dir", required=True)
    o = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rep = json.loads(Path(o.analysis).read_text())
    res = rep["runs"][Path(o.run).name]["cases"]
    out = Path(o.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    meta = {"schema": C.SCHEMA + ":figures", "analysis": o.analysis, "run": o.run, "figures": {}}

    # 1 bounds summary (dot plot, one log axis)
    cases = list(res)
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    series = [("untouched import (control_U)", lambda r: r["conditions"].get("control_U", {}).get("all"), "o"),
              ("import-level bound (privileged)", lambda r: r.get("bounds", {}).get("imp", {}).get("bound_m"), "s"),
              ("FBX-level bound (privileged)", lambda r: r.get("bounds", {}).get("fbx", {}).get("bound_m"), "D"),
              ("source-free candidate (S)", lambda r: (r.get("candidate") or {}).get("all"), "^")]
    rows = {}
    for k, (name, fn, mk) in enumerate(series):
        y, x = [], []
        for i, c in enumerate(cases):
            v = fn(res[c])
            if v is not None:
                y.append(len(cases) - 1 - i + (k - 1.5)*0.16)
                x.append(max(v, 1e-8))
        rows[name] = {c: fn(res[c]) for c in cases}
        ax.scatter(x, y, s=34, marker=mk, color=COL[k], edgecolors="white", linewidths=1.2, label=name, zorder=3)
    ax.axvline(1e-3, color=INK, linewidth=1, linestyle="--")
    ax.text(1.1e-3, len(cases) - 0.4, "1 mm gate", fontsize=8, color=INK)
    ax.set_xscale("log")
    ax.set_yticks(range(len(cases)))
    ax.set_yticklabels([f"{c}{'  *' if c in C.WITHIN_CASES else ''}" for c in reversed(cases)], fontsize=8)
    ax.set_xlabel("max world |coordinate| error over all probes (m, log)", fontsize=9, color=INK)
    ax.set_title("W112: per-case bounds and conditions  (* = within-property case)", fontsize=10, color=INK)
    _style(ax)
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(out/"bounds_summary.png", dpi=130)
    plt.close(fig)
    meta["figures"]["bounds_summary.png"] = rows

    # 2 held -> smooth transition (syn_R_int), target tail coordinate with the largest source range
    cid = "syn_R_int"
    rd = Path(o.run)
    src = C.load_npz(rd/"produce"/cid/"source.npz")
    names = [str(b) for b in src["bones"]]
    j = names.index(C.TARGET_BONE)
    ax_i = int(np.argmax(np.ptp(src["world_tail"][:, j], axis=0)))
    t = np.asarray(src["times"])
    sel = (t >= 1) & (t <= 13)
    curves = [("source", src), ("untouched import", C.load_npz(rd/"consume"/cid/"control_U.npz"))]
    if (rd/"consume"/cid/"meta_A.npz").exists():
        curves.append(("W110 adapter (meta_A)", C.load_npz(rd/"consume"/cid/"meta_A.npz")))
    if (rd/"candidate/cases"/cid/"S.npz").exists():
        curves.append(("source-free candidate (S)", C.load_npz(rd/"candidate/cases"/cid/"S.npz")))
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    styles = ["-", "--", ":", "-."]
    for k, (name, a) in enumerate(curves):
        ax.plot(t[sel], a["world_tail"][sel, j, ax_i], styles[k], color=COL[k], linewidth=2, label=name,
                marker="o" if k == 0 else None, markersize=3)
    for f in range(1, 13):
        cls = res[cid]["bounds"]["imp"]["per_interval"][str(f)]["class"]
        if cls == "held":
            ax.axvspan(f, f + 1, color="#f2f1ed", zorder=0)
    ax.set_xlabel("frame (shaded = held interval of the target)", fontsize=9, color=INK)
    ax.set_ylabel(f"{C.TARGET_BONE} tail, world axis {'xyz'[ax_i]} (m)", fontsize=9, color=INK)
    ax.set_title("syn_R_int: held -> smooth -> held transitions on one rotation property", fontsize=10, color=INK)
    _style(ax)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out/"held_to_smooth.png", dpi=130)
    plt.close(fig)
    meta["figures"]["held_to_smooth.png"] = {"case": cid, "axis": int(ax_i), "series": [n for n, _ in curves]}

    # 3 axis-mixed per-interval best error, both layers + group-uniform
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    info = {}
    for ax, cid in zip(axs, ("syn_R_axis", "syn_L_axis", "rr_axis")):
        b = res[cid].get("bounds", {})
        f = sorted(int(k) for k in b["imp"]["per_interval"])
        for k, (lay, name, mk) in enumerate((("imp", "import-level best", "s"), ("fbx", "FBX-level best", "D"))):
            if lay in b:
                y = [max(b[lay]["per_interval"][str(x)]["best_m"], 1e-8) for x in f]
                ax.plot(f, y, color=COL[k + 1], marker=mk, markersize=4, linewidth=1.5, label=name)
        ax.axhline(1e-3, color=INK, linewidth=1, linestyle="--")
        ax.set_yscale("log")
        ax.set_title(cid, fontsize=9, color=INK)
        ax.set_xlabel("interval start frame", fontsize=8, color=INK)
        _style(ax)
        info[cid] = {lay: b.get(lay, {}).get("bound_m") for lay in ("imp", "fbx")}
    axs[0].set_ylabel("best per-interval error (m, log)", fontsize=8, color=INK)
    axs[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Axis-mixed cases: privileged per-interval minimum at each layer (1 mm dashed)", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out/"axis_mixed.png", dpi=130)
    plt.close(fig)
    meta["figures"]["axis_mixed.png"] = info

    # 4 real-rig pose at the worst import-level interval of rr_axis (near-boundary probe)
    cid = "rr_axis"
    per = res[cid]["bounds"]["imp"]["per_interval"]
    worst = max(per, key=lambda k: per[k]["best_m"])
    src = C.load_npz(rd/"produce"/cid/"source.npz")
    t = np.asarray(src["times"])
    i = int(np.argmin(np.abs(t - (int(worst) + 0.99))))
    poses = [("source", src), ("untouched import", C.load_npz(rd/"consume"/cid/"control_U.npz")),
             ("import-level best (privileged)", _best_series(rd, cid, "imp", per))]
    fig, ax = plt.subplots(figsize=(5.2, 7))
    for k, (name, a) in enumerate(poses):
        nm = [str(b) for b in a["bones"]]
        for b in range(len(nm)):
            if nm[b].endswith("_end"):
                continue
            h, tl = a["world_head"][i, b], a["world_tail"][i, b]
            ax.plot([h[0], tl[0]], [h[2], tl[2]], ["-", "--", ":"][k], color=COL[k], linewidth=2,
                    label=name if b == 0 else None)
    ax.set_aspect("equal")
    ax.set_xlabel("world X (m)", fontsize=8, color=INK)
    ax.set_ylabel("world Z (m)", fontsize=8, color=INK)
    ax.set_title(f"rr_axis (synthetic-on-real-rig), t = {t[i]:.2f}", fontsize=9, color=INK)
    _style(ax)
    ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=1)
    fig.tight_layout()
    fig.savefig(out/"real_rig_pose.png", dpi=130)
    plt.close(fig)
    meta["figures"]["real_rig_pose.png"] = {"case": cid, "time": float(t[i]), "worst_interval": int(worst),
                                            "imp_best_m": per[worst]["best_m"]}
    (out/"figures.json").write_text(json.dumps(meta, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
