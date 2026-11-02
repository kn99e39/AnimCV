#!/usr/bin/env python3
"""W103 qualitative panels for the declared deterministic selections (evaluation-only figures).

Panel order per row: RGB with the M1 model-visible joints | M model-visible tensor | B model-visible tensor |
GT evaluation geometry (NOT available to S0) | frozen S0 on M | frozen S0 on B | frozen HMR2 (contextual).
3D panels: camera-frame front view (x right, z up) and side view (y depth, z up), root-relative metres.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"), str(ROOT/"scripts")]
from w103_observability import (BASE, CONDITIONS, JOINTS, PARENT, dump, geometry_all, load_context, load_hmr,
                                load_predictions, sha)

BONES = [(c, p) for c, p in PARENT.items()]
LEFT = {1, 2, 3, 11, 12, 13}


def draw3d(ax, J, title, color):
    J = J - J[0]
    for c, p in BONES:
        col = color if c not in LEFT else "tab:orange"
        ax.plot([J[p, 0], J[c, 0]], [J[p, 2], J[c, 2]], "-", color=col, lw=2)
        ax.plot([J[p, 1]+1.3, J[c, 1]+1.3], [J[p, 2], J[c, 2]], "-", color=col, lw=2, alpha=.8)
    ax.axvline(.65, color="0.8", lw=.5)
    ax.text(0, -1.15, "front (x,z)", ha="center", fontsize=6); ax.text(1.3, -1.15, "side (depth y,z)", ha="center", fontsize=6)
    ax.set_xlim(-.7, 2.0); ax.set_ylim(-1.2, 1.0); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=7)


def draw_tensor(ax, F, title):
    v = F[:, 3] > .5
    for c, p in BONES:
        if v[c] and v[p]:
            ax.plot([F[p, 0], F[c, 0]], [-F[p, 1], -F[c, 1]], "-", color="tab:orange" if c in LEFT else "tab:blue", lw=1.5)
    ax.scatter(F[v, 0], -F[v, 1], s=6+30*np.clip(F[v, 2], 0, 1), c="k", zorder=3)
    missing = [JOINTS[j] for j in BASE if not v[j]]
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title + ("\ninvalid: " + ", ".join(missing) if missing else "\nall 13 base points valid"), fontsize=6)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--w102", required=True); ap.add_argument("--w101-data", required=True)
    ap.add_argument("--analysis", required=True, nargs="+", help="analysis_report.json files whose selections are rendered")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=False)
    bank, obs, side, positions = load_context(args.w102, args.w101_data)
    geometry = geometry_all(bank, obs)
    pred, _ = load_predictions(args.w102, bank, positions)
    hmr = load_hmr(args.w102, positions)
    image_sha = json.loads((Path(args.w102)/"data/w101_transfer_image_sha256.json").read_text())
    G = bank.arrays["target_3d"][positions].astype(float)
    manifest = []
    for path in args.analysis:
        report = json.loads(Path(path).read_text())
        label = Path(path).parent.name
        for name, sel in report["selections"].items():
            rows = sel["rows"]
            fig, axes = plt.subplots(len(rows), 7, figsize=(17, 2.9*len(rows)), squeeze=False)
            entries = []
            for k, r in enumerate(rows):
                p = int(positions[r]); s = bank.samples[p]
                rel = s.image_reference.relative_path
                img_path = Path(args.w102)/"data/images"/rel
                digest = sha(img_path)
                if digest != image_sha[rel]:
                    raise RuntimeError(f"STOP: image identity failed for {rel}")
                img = np.asarray(Image.open(img_path).convert("RGB"))
                W, H = s.image_size
                xm, vm = obs["input_2d_M"][r], obs["input_valid_M"][r]
                px = xm[:, :2]*[W, H]
                ok = px[vm]
                lo, hi = ok.min(0) if len(ok) else (0, 0), ok.max(0) if len(ok) else (W, H)
                c, half = (np.asarray(lo)+hi)/2, max(*(np.asarray(hi)-lo))*.75+20
                ax = axes[k, 0]; ax.imshow(img)
                for cc, pp in BONES:
                    if vm[cc] and vm[pp]:
                        ax.plot([px[pp, 0], px[cc, 0]], [px[pp, 1], px[cc, 1]], "-", color="tab:orange" if cc in LEFT else "cyan", lw=1.2)
                ax.set_xlim(c[0]-half, c[0]+half); ax.set_ylim(c[1]+half, c[1]-half); ax.set_xticks([]); ax.set_yticks([])
                ident = str(side["identity"][r])
                ax.set_title(f"{s.sample_id}\n{s.split} t={s.timestamp:.3f}s  M1 identity={ident}", fontsize=6)
                draw_tensor(axes[k, 1], geometry["M"][p], "M model-visible (17,4) tensor")
                draw_tensor(axes[k, 2], geometry["B"][p], "B model-visible (17,4) tensor")
                g = G[r]-G[r, 0]
                draw3d(axes[k, 3], g, "GT EVALUATION ONLY\n(not available to S0)", "0.3")
                mpj = {}
                for col, (alias, title, color) in enumerate((("M", "frozen S0 on M_Main", "tab:green"),
                                                            ("B", "frozen S0 on B_Main", "tab:purple")), start=4):
                    j = pred[alias][r]-pred[alias][r][0]
                    mpj[alias] = float(1000*np.linalg.norm(j-g, axis=-1).mean())
                    draw3d(axes[k, col], j, f"{title}\nMPJPE {mpj[alias]:.0f} mm", color)
                if np.isfinite(hmr[r]).all():
                    mpj["E"] = float(1000*np.linalg.norm(hmr[r]-g, axis=-1).mean())
                    draw3d(axes[k, 6], hmr[r], f"frozen HMR2 (RGB, contextual)\nMPJPE {mpj['E']:.0f} mm", "tab:red")
                else:
                    axes[k, 6].axis("off"); axes[k, 6].set_title("HMR2 unavailable", fontsize=7)
                entries.append({"row": int(r), "sample_id": s.sample_id, "image": rel, "image_sha256": digest, "split": s.split,
                                "timestamp": s.timestamp, "m1_identity": ident, "mpjpe_mm": mpj})
            fig.suptitle(f"W103 {name}  [{label}]  " + ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else "" for k, v in sel.items() if k in ("d", "s0_step_deg", "gt_step_deg")), fontsize=9)
            fig.tight_layout()
            fname = f"{label}__{name}.png"
            fig.savefig(out/fname, dpi=110); plt.close(fig)
            manifest.append({"figure": fname, "selection": name, "analysis": label, "rows": entries, "sha256": sha(out/fname)})
            print(fname, flush=True)
    dump(out/"qualitative_manifest.json", manifest)


if __name__ == "__main__":
    main()
