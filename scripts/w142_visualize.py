"""W142 visualisation (called by ``w142_harness.py visualize``): reads only FROZEN W142 outputs, the W141 inputs/masks and GT.

Never changes an estimate, metric, gate or classification.  Not reviewed by a human unless stated.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np

import app.experimental_background_rotation as BR
import app.experimental_camera_pose as CP

STATE_COL = {"NO_DETECTABLE_MOTION": (0.6, 0.6, 0.6), "ROTATION_ONLY_CONSISTENT": (0.2, 0.5, 0.9), "TRANSLATION_PRESENT_PLANAR_AMBIGUOUS": (0.95, 0.6, 0.1),
             "TRANSLATION_OBSERVED": (0.85, 0.15, 0.15), "INSUFFICIENT_EVIDENCE": (0.0, 0.0, 0.0), None: (1, 1, 1)}
REG_COL = {"NEAR_STATIC_TRANSLATION_UNRESOLVED": (0.6, 0.6, 0.6), "ROTATING_TRANSLATION_UNRESOLVED": (0.2, 0.5, 0.9), "TRANSLATING": (0.85, 0.15, 0.15),
           "INSUFFICIENT_EVIDENCE": (0, 0, 0)}
GTL_COL = {"NEAR_STATIC": (0.6, 0.6, 0.6), "ROTATING": (0.2, 0.5, 0.9), "TRANSLATING": (0.85, 0.15, 0.15)}


def load(p):
    return json.load(open(p, encoding="utf-8"))


def label(img, txt, org, col=(255, 255, 255), sc=0.45, th=1):
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, sc, (0, 0, 0), th + 2, cv2.LINE_AA)
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, sc, col, th, cv2.LINE_AA)


def rotvec_deg(R):
    from scipy.spatial.transform import Rotation
    return np.degrees(Rotation.from_matrix(R).as_rotvec())


def visualize(a, ROOT, W141, W142, RUNS):
    import importlib.util
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    spec = importlib.util.spec_from_file_location("w142h", ROOT / "scripts" / "w142_harness.py")
    H = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(H)
    ev = load(Path(a.eval_dir) / "evaluation.json")
    pf = load(a.preflight)
    E = Path(a.estimate_dir)
    out = Path(a.out_dir)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    made, notes = [], {}

    def timeline(ax, xs, vals, cmap, y, h=0.8):
        for x0, x1, v in zip(xs[:-1], xs[1:], vals):
            ax.axvspan(x0, x1, ymin=y, ymax=y + h * 0.3, color=cmap.get(v, (1, 1, 1)), lw=0)

    # ---------------------------------------------------------------- orientation + regime figures (GT datasets)
    for ds in ("syn_A1", "syn_A2", "syn_A3", "syn_A4", "walking_static", "walking_xyz"):
        gt, kind = H.gt_for(ds, pf)
        fig, axs = plt.subplots(4, 1, figsize=(15, 13), gridspec_kw={"height_ratios": [3, 2, 2, 1.4]})
        est_p = load(E / f"{ds}_rgbmask" / "estimate.json")
        times = np.asarray([f["timestamp"] for f in est_p["frames"]])
        t0 = times[0]
        Tg = [H.associate(gt, kind, t) for t in times]
        gi = [k for k in range(len(times)) if Tg[k] is not None]
        Rg0 = Tg[gi[0]][:3, :3]
        gv = np.asarray([rotvec_deg(Rg0.T @ Tg[k][:3, :3]) for k in gi])
        for c, nm in enumerate(("x (pitch)", "y (yaw)", "z (roll)")):
            axs[0].plot(times[gi] - t0, gv[:, c], "-", color=f"C{c}", lw=2.2, alpha=0.45, label=f"GT {nm}")
        for mk, ls in (("rgbmask", "-"), ("none", ":")):
            p = E / f"{ds}_{mk}" / "estimate.json"
            if not p.is_file():
                continue
            est = load(p)
            ok = [k for k in gi if est["frames"][k]["orientation"]["status"] == "ESTIMATED"]
            if len(ok) < 3:
                continue
            Rg = np.asarray([Tg[k][:3, :3] for k in ok])
            Re = np.asarray([est["frames"][k]["orientation"]["R_WC"] for k in ok])
            Ra = BR.chordal_align(Rg, Re)
            ev_ = np.asarray([rotvec_deg(Rg0.T @ Ra @ R) for R in Re])
            for c in range(3):
                axs[0].plot(times[ok] - t0, ev_[:, c], ls, color=f"C{c}", lw=1, label=f"EST[{mk}] {'xyz'[c]}" if c == 0 else None)
            err = CP.geodesic_deg(np.einsum("ij,njk->nik", Ra, Re), Rg)
            axs[1].plot(times[ok] - t0, err, ls, color="tab:red", lw=1, label=f"W142 [{mk}] aligned orientation error")
        _, berr = BR.orientation_errors(np.asarray([Tg[k][:3, :3] for k in gi]), np.broadcast_to(np.eye(3), (len(gi), 3, 3)))
        axs[1].plot(times[gi] - t0, berr, "k--", lw=1, label="constant-camera baseline")
        hs = {"walking_static": ["walking_static_A", "walking_static_B"], "walking_xyz": ["walking_xyz_B"]}.get(ds, [])
        for hn in hs:
            e = load(W141 / "results/estimate" / hn / "estimate.json")
            ok = [k for k, f in enumerate(e["frames"]) if f["status"] == "ESTIMATED" and H.associate(gt, kind, f["timestamp"]) is not None]
            if len(ok) >= 3:
                Rg = np.asarray([H.associate(gt, kind, e["frames"][k]["timestamp"])[:3, :3] for k in ok])
                Re = np.asarray([np.asarray(e["frames"][k]["T_WC"])[:3, :3] for k in ok])
                _, err = BR.orientation_errors(Rg, Re)
                axs[1].plot([e["frames"][k]["timestamp"] - t0 for k in ok], err, ".", ms=3, label=f"W141 SfM {hn[-1]} (same alignment)")
        axs[1].set_ylabel("deg")
        axs[1].legend(fontsize=7)
        axs[0].set_ylabel("rotation vector rel. to first GT frame [deg]")
        axs[0].legend(fontsize=6, ncol=3)
        # timelines
        ax = axs[2]
        r = ev["runs"][f"{ds}_rgbmask"]
        regs = [f["regime_window"]["regime"] for f in est_p["frames"]]
        st = [f["motion_state_from_previous"] for f in est_p["frames"]]
        xs = list(times - t0) + [times[-1] - t0 + float(np.median(np.diff(times)))]
        timeline(ax, xs, regs, REG_COL, 0.66)
        timeline(ax, xs, st, STATE_COL, 0.33)
        lab = [r["gt_labels"].get(str(f["frame_id"]), r["gt_labels"].get(f["frame_id"])) for f in est_p["frames"]]
        timeline(ax, xs, lab, GTL_COL, 0.0)
        ax.set_yticks([0.15, 0.48, 0.81])
        ax.set_yticklabels(["GT label", "edge state (prev)", "W142 regime"])
        ax.set_xlim(xs[0], xs[-1])
        handles = [plt.Rectangle((0, 0), 1, 1, color=v) for v in REG_COL.values()] + [plt.Rectangle((0, 0), 1, 1, color=STATE_COL[k]) for k in BR.STATES]
        ax.legend(handles, [f"regime {k}" for k in REG_COL] + [f"edge {k}" for k in BR.STATES], fontsize=6, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.15))
        axs[3].axis("off")
        txt = [f"{ds}: W142 sequence regime {r['sequence_regime']['regime']}; edge states {r['edge_state_counts']}",
               f"orientation status {r['orientation_status_counts']}; coverage {r.get('coverage', 0):.3f}"]
        if "aligned" in r:
            txt.append(f"aligned orientation error median {r['aligned']['median_deg']:.3f} deg (p90 {r['aligned']['p90_deg']:.3f}); constant baseline "
                       f"{r['baseline_aligned']['median_deg']:.3f}; 1 s relative {r['relative_1s']['median_deg']} vs baseline {r['relative_1s']['baseline_median_deg']}")
        txt.append(f"GT motion: {r.get('gt_motion')}; translation is NEVER output (value null)")
        axs[3].text(0, 1, "\n".join(txt), va="top", family="monospace", fontsize=7.5)
        axs[2].set_xlabel("time [s]")
        fig.suptitle(f"W142 {ds}: GT vs W142 background-rotation orientation (evaluation-only chordal alignment), motion states and regimes")
        fig.tight_layout()
        p = out / f"{ds}_orientation_regimes.png"
        fig.savefig(p, dpi=100)
        plt.close(fig)
        made.append(p)

    # ---------------------------------------------------------------- feature flow before / after rotation compensation
    def flow_sheet(ds, mk, pairs, tag):
        est = load(E / f"{ds}_{mk}" / "estimate.json")
        F = np.load(E / f"{ds}_{mk}" / "features.npz")
        Eg = np.load(E / f"{ds}_{mk}" / "edges.npz")
        ind = Path(pf["inputs"][ds]["dir"])
        K = np.array([[est["camera"]["fx"], 0, est["camera"]["cx"]], [0, est["camera"]["fy"], est["camera"]["cy"]], [0, 0, 1.0]])
        mdir = Path(pf["masks"][ds]["dirs"]["rgbmask"]).parent / "raw"
        rows = []
        for i, j in pairs:
            e = next((x for x in est["edges"] if x["i"] == i and x["j"] == j), None)
            if e is None:
                continue
            f = est["frames"][i]
            img = cv2.imread(str(ind / "images" / f["image"]))
            raw = cv2.imread(str(mdir / f["image"]), cv2.IMREAD_UNCHANGED) > 127 if (mdir / f["image"]).is_file() else np.zeros(img.shape[:2], bool)
            base = img.copy()
            base[raw] = (0.5 * base[raw] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
            mm = Eg[f"e{i}_{j}"]
            x1, x2 = F[f"f{i}"][mm[:, 0]], F[f"f{j}"][mm[:, 1]]
            a_, b_ = base.copy(), base.copy()
            M = np.asarray(e["M"]) if e.get("M") is not None else np.eye(3)
            pred, _ = BR.project_bearing(K, (M @ BR.bearings(K, x1).T).T)
            for p, q, r_ in zip(x1[::2], x2[::2], pred[::2]):
                cv2.arrowedLine(a_, tuple(int(v) for v in p), tuple(int(v) for v in p + 5 * (q - p)), (0, 0, 255), 1, tipLength=0.2)
                cv2.arrowedLine(b_, tuple(int(v) for v in p), tuple(int(v) for v in p + 5 * (q - r_)), (0, 200, 0), 1, tipLength=0.2)
            label(a_, f"{ds} frames {f['frame_id']}->{est['frames'][j]['frame_id']} (span {e['span']}, dt {e['dt_s']:.2f}s): RAW background flow x5", (5, 16))
            label(a_, f"state {e['state']}  inliers {len(mm)}  red tint = RGB person mask", (5, 34))
            label(b_, f"RESIDUAL after rotation compensation x5 (rotation {e.get('rotation_deg', float('nan')):.3f} deg)", (5, 16))
            label(b_, f"rot-model residual median {e.get('rot_residual_median_px')} px", (5, 34))
            rows.append(np.hstack([a_, b_]))
        if rows:
            p = out / f"{tag}_flow_compensation.png"
            cv2.imwrite(str(p), np.vstack(rows))
            made.append(p)
    n_ws = len(load(Path(pf["inputs"]["walking_static"]["dir"]) / "frames.json"))
    flow_sheet("walking_static", "rgbmask", [(10, 18), (120, 128), (200, 208), (280, 288)], "walking_static")
    flow_sheet("walking_xyz", "rgbmask", [(10, 12), (150, 152)], "walking_xyz")
    flow_sheet("s1", "rgbmask", [(10, 18), (100, 108), (200, 208)], "s1")
    flow_sheet("syn_A1", "rgbmask", [(10, 18)], "syn_A1")
    flow_sheet("syn_A2", "rgbmask", [(10, 18)], "syn_A2")

    # ---------------------------------------------------------------- S1 diagnostic timeline
    est = load(E / "s1_rgbmask" / "estimate.json")
    t = np.asarray([f["timestamp"] for f in est["frames"]])
    fig, ax = plt.subplots(3, 1, figsize=(14, 8))
    e1 = [e for e in est["edges"] if e["span"] == 1]
    ax[0].plot([t[e["i"]] for e in e1], [e.get("flow_raw_median_px") or np.nan for e in e1], "r-", label="raw background flow median (span 1) [px]")
    ax[0].plot([t[e["i"]] for e in e1], [e.get("rot_residual_median_px") or np.nan for e in e1], "g-", label="residual after rotation model [px]")
    ax[0].plot([t[e["i"]] for e in e1], [e["n_matches"] / 100 for e in e1], "k:", label="matches / 100")
    ax[0].legend(fontsize=7)
    for s_, axx in ((1, ax[1]), (8, ax[1])):
        es = [e for e in est["edges"] if e["span"] == s_ and "best_kfree_model" in e]
        axx.plot([t[e["i"]] for e in es], [1 if e["best_kfree_model"] == "F" else 0 for e in es], ".", label=f"span {s_}: K-free best model (1 = fundamental / parallax, 0 = homography)")
    ax[1].legend(fontsize=7)
    xs = list(t - t[0]) + [t[-1] - t[0] + 1 / 30]
    timeline(ax[2], xs, [f["regime_window"]["regime"] for f in est["frames"]], REG_COL, 0.5)
    timeline(ax[2], xs, [f["motion_state_from_previous"] for f in est["frames"]], STATE_COL, 0.1)
    ax[2].set_yticks([0.25, 0.65])
    ax[2].set_yticklabels(["edge state (NOMINAL K)", "regime (NOMINAL K)"])
    ax[2].set_xlim(xs[0], xs[-1])
    fig.suptitle("W142 S1 (authentic, UNCALIBRATED: nominal focal) background-motion diagnostics; no GT; not a camera trajectory")
    fig.tight_layout()
    p = out / "s1_background_diagnostics.png"
    fig.savefig(p, dpi=100)
    plt.close(fig)
    made.append(p)

    # ---------------------------------------------------------------- playback videos (real-time pacing)
    for ds in ("walking_static", "s1"):
        est = load(E / f"{ds}_rgbmask" / "estimate.json")
        F = np.load(E / f"{ds}_rgbmask" / "features.npz")
        ind = Path(pf["inputs"][ds]["dir"])
        mdir = Path(pf["masks"][ds]["dirs"]["rgbmask"])
        ts = np.asarray([f["timestamp"] for f in est["frames"]])
        vid = out / f"{ds}_playback_realtime.mp4"
        wr = None
        acc, n_out = 0.0, 0
        for k, f in enumerate(est["frames"]):
            img = cv2.imread(str(ind / "images" / f["image"]))
            ex = cv2.imread(str(mdir / (f["image"] + ".png")), cv2.IMREAD_UNCHANGED) == 0
            img[ex] = (0.55 * img[ex] + 0.45 * np.array([0, 0, 255])).astype(np.uint8)
            for u, v in F[f"f{k}"][::2]:
                cv2.circle(img, (int(round(u)), int(round(v))), 2, (0, 255, 0), -1)
            panel = np.full((img.shape[0], 420, 3), 255, np.uint8)
            o = f["orientation"]
            lines = [f"W142 {ds}  t={f['timestamp'] - ts[0]:.2f}s  frame {f['frame_id']}",
                     f"orientation: {o['status']}", f"motion state (prev): {f['motion_state_from_previous']}",
                     f"regime window: {f['regime_window']['regime']}", f"translation: value NULL",
                     f"  {f['translation']['status'][:40]}", f"background keypoints: {f['evidence']['n_kp']}"]
            if ds == "s1":
                lines.append("S1: NOMINAL focal, uncalibrated")
            for n_, s_ in enumerate(lines):
                cv2.putText(panel, s_, (8, 24 + 22 * n_), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(panel, "red = excluded (person/letterbox); green = background kp", (8, img.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1)
            cv2.putText(panel, "real-time pacing (selected frames held)", (8, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (90, 90, 90), 1)
            fr = np.hstack([img, panel])
            if wr is None:
                wr = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (fr.shape[1], fr.shape[0]))
            dt = (ts[k + 1] - ts[k]) if k + 1 < len(ts) else float(np.median(np.diff(ts)))
            acc += dt * 30.0
            for _ in range(max(int(round(acc)) - n_out, 1 if k == 0 else 0)):
                wr.write(fr)
                n_out += 1
        wr.release()
        made.append(vid)
        notes[vid.name] = f"{n_out} frames at 30 fps = {n_out / 30:.2f} s for a {ts[-1] - ts[0]:.2f} s span (real-time pacing)"

    # ---------------------------------------------------------------- synthetic masks / retained features
    rows = []
    for sc in ("A1", "A2", "A3"):
        ds = f"syn_{sc}"
        ind = Path(pf["inputs"][ds]["dir"])
        est = load(E / f"{ds}_rgbmask" / "estimate.json")
        F = np.load(E / f"{ds}_rgbmask" / "features.npz")
        k = 60
        f = est["frames"][k]
        img = cv2.imread(str(ind / "images" / f["image"]))
        ex = cv2.imread(str(Path(pf["masks"][ds]["dirs"]["rgbmask"]) / (f["image"] + ".png")), cv2.IMREAD_UNCHANGED) == 0
        exo = cv2.imread(str(Path(pf["masks"][ds]["dirs"]["oracle"]) / (f["image"] + ".png")), cv2.IMREAD_UNCHANGED) == 0
        a_ = img.copy()
        a_[ex] = (0.5 * a_[ex] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
        for u, v in F[f"f{k}"][::2]:
            cv2.circle(a_, (int(round(u)), int(round(v))), 1, (0, 255, 0), -1)
        b_ = img.copy()
        b_[exo] = (0.5 * b_[exo] + 0.5 * np.array([255, 0, 255])).astype(np.uint8)
        label(a_, f"{ds} frame {k}: RGB person mask (red), kept background kp (green)", (5, 16))
        label(b_, f"ORACLE foreground mask (magenta, diagnostic only)", (5, 16))
        rows.append(np.hstack([a_, b_]))
    p = out / "synthetic_masks_features.png"
    cv2.imwrite(str(p), np.vstack(rows))
    made.append(p)

    ue = ROOT / "user_eval" / "W142_near_static_camera"
    ue.mkdir(parents=True, exist_ok=True)
    for p in made:
        shutil.copyfile(p, ue / p.name)
    CP.dump(out / "visualize.json", {"files": {p.name: CP.sha256_file(p) for p in made}, "copied_to": str(ue), "notes": notes,
                                     "review": "AGENT-generated; not reviewed by a human"})
    print(json.dumps({"files": [p.name for p in made], "notes": notes}, indent=1))
    return 0
