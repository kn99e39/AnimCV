"""W143 visualisation (called by ``w143_harness.py visualize``): reads only FROZEN W143/W142 outputs, inputs, masks and GT.

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

CAM_COL = {"FULL_6DOF_AVAILABLE": (0.1, 0.6, 0.1), "ORIENTATION_ONLY_AVAILABLE": (0.2, 0.5, 0.9), "CAMERA_MOTION_UNRESOLVED": (0, 0, 0)}
TR_COL = {"TRANSLATION_DIRECTION_ONLY": (0.85, 0.15, 0.15), "TRANSLATION_UNRESOLVED": (0.65, 0.65, 0.65), "TRANSLATION_INSUFFICIENT_EVIDENCE": (0, 0, 0),
          "TRANSLATION_ESTIMATED_SCALE_AMBIGUOUS": (0.1, 0.6, 0.1)}
GTL_COL = {"NEAR_STATIC": (0.6, 0.6, 0.6), "ROTATING": (0.2, 0.5, 0.9), "TRANSLATING": (0.85, 0.15, 0.15)}
REJ_COL = {"accepted": (0.1, 0.65, 0.1), "rotation_support": (0.9, 0.5, 0.1), "rotation_residual": (0.8, 0.1, 0.1), "rotation_vs_homography": (0.6, 0.2, 0.7),
           "cycle_inconsistent": (0.1, 0.1, 0.1), "rotation_inliers": (0.5, 0.5, 0.5), "too_few_matches": (0.8, 0.8, 0.8)}


def load(p):
    return json.load(open(p, encoding="utf-8"))


def label(img, txt, org, col=(255, 255, 255), sc=0.45, th=1):
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, sc, (0, 0, 0), th + 2, cv2.LINE_AA)
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, sc, col, th, cv2.LINE_AA)


def rotvec_deg(R):
    from scipy.spatial.transform import Rotation
    return np.degrees(Rotation.from_matrix(R).as_rotvec())


def visualize(a, ROOT, W141, W142, W143, RUNS, gt_for, h42):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ev = load(Path(a.eval_dir) / "evaluation.json")
    pf = load(a.preflight)
    E = Path(a.estimate_dir)
    out = Path(a.out_dir)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    made, notes = [], {}

    def band(ax, xs, vals, cmap, y0, h=0.28):
        for x0, x1, v in zip(xs[:-1], xs[1:], vals):
            ax.axvspan(x0, x1, ymin=y0, ymax=y0 + h, color=cmap.get(v, (1, 1, 1)), lw=0)

    for name, dsn in RUNS:
        if dsn == "s1":
            continue
        est = load(E / name / "estimate.json")
        gt, kind = gt_for(dsn, pf)
        frames = est["frames"]
        t = np.asarray([f["timestamp"] for f in frames])
        t0 = t[0]
        Tg = [h42.associate(gt, kind, x) for x in t]
        gi = [k for k in range(len(t)) if Tg[k] is not None]
        Rg0 = Tg[gi[0]][:3, :3]
        fig, axs = plt.subplots(5, 1, figsize=(15, 16), gridspec_kw={"height_ratios": [3, 2, 1.3, 2.2, 1.2]})
        gv = np.asarray([rotvec_deg(Rg0.T @ Tg[k][:3, :3]) for k in gi])
        for c in range(3):
            axs[0].plot(t[gi] - t0, gv[:, c], "-", color=f"C{c}", lw=2.2, alpha=0.4, label=f"GT {'xyz'[c]}")
        srcs = [("W143", est, lambda f: f["orientation"]["status"] == "ORIENTATION_ESTIMATED", "-")]
        w142p = W142 / "results/estimate" / f"{dsn}_rgbmask" / "estimate.json"
        if not w142p.is_file():
            w142p = E / "w142_sitting_static_rgbmask" / "estimate.json"
        w142 = load(w142p)
        srcs.append(("W142", w142, lambda f: f["orientation"]["status"] == "ESTIMATED", ":"))
        for nm, e_, ok_fn, ls in srcs:
            ok = [k for k in gi if ok_fn(e_["frames"][k])]
            if len(ok) < 3:
                continue
            Rg = np.asarray([Tg[k][:3, :3] for k in ok])
            Re = np.asarray([e_["frames"][k]["orientation"]["R_WC"] for k in ok])
            Ra = BR.chordal_align(Rg, Re)
            v = np.asarray([rotvec_deg(Rg0.T @ Ra @ R) for R in Re])
            for c in range(3):
                axs[0].plot(t[ok] - t0, v[:, c], ls, color=f"C{c}", lw=1, label=f"{nm} {'xyz'[c]}" if c == 0 else None)
            err = CP.geodesic_deg(np.einsum("ij,njk->nik", Ra, Re), Rg)
            axs[1].plot(t[ok] - t0, err, ls, color="tab:red" if nm == "W143" else "tab:purple", lw=1, label=f"{nm} aligned orientation error ({len(ok)} frames)")
        _, berr = BR.orientation_errors(np.asarray([Tg[k][:3, :3] for k in gi]), np.broadcast_to(np.eye(3), (len(gi), 3, 3)))
        axs[1].plot(t[gi] - t0, berr, "k--", lw=1, label="constant-camera baseline")
        axs[0].legend(fontsize=6, ncol=4)
        axs[0].set_ylabel("rotation vector rel. first GT frame [deg]")
        axs[1].legend(fontsize=7)
        axs[1].set_ylabel("deg")
        xs = list(t - t0) + [t[-1] - t0 + float(np.median(np.diff(t)))]
        band(axs[2], xs, [f["camera_state"] for f in frames], CAM_COL, 0.68)
        band(axs[2], xs, [f["translation_window"]["translation"] for f in frames], TR_COL, 0.36)
        r = ev["runs"][name]
        lab_frames = BR.gt_regime_labels([t[k] for k in gi], [Tg[k][:3, 3] for k in gi], [Tg[k][:3, :3] for k in gi])
        labs = [None] * len(t)
        for k, l in zip(gi, lab_frames):
            labs[k] = l
        band(axs[2], xs, labs, GTL_COL, 0.04)
        axs[2].set_yticks([0.18, 0.5, 0.82])
        axs[2].set_yticklabels(["GT label", "translation obs.", "camera state"])
        axs[2].set_xlim(xs[0], xs[-1])
        # edge acceptance map (pair-graph connectivity)
        ax = axs[3]
        for e in est["edges"]:
            key = "accepted" if e["rotation_accepted"] else (e["reject"][-1] if e["reject"] else "rotation_inliers")
            ax.plot([t[e["i"]] - t0], [np.log2(e["span"])], "s", ms=2.5, color=REJ_COL.get(key, (1, 0, 1)))
        comp = set(est["rotation_averaging"].get("component", []))
        for k in range(len(t)):
            ax.plot([t[k] - t0], [-0.8], "|", color="tab:green" if (k in comp and frames[k]["orientation"]["status"] == "ORIENTATION_ESTIMATED") else "tab:red", ms=6)
        ax.set_yticks([-0.8] + list(range(6)))
        ax.set_yticklabels(["frame oriented"] + [f"span {2 ** s}" for s in range(6)])
        ax.legend([plt.Line2D([], [], marker="s", ls="", color=c) for c in REJ_COL.values()], list(REJ_COL), fontsize=6, ncol=4, loc="upper center",
                  bbox_to_anchor=(0.5, -0.12))
        ax.set_xlabel("time [s]")
        axs[4].axis("off")
        txt = [f"{name}: coverage {r.get('coverage', 0):.3f}; camera state {r['camera_state_counts']}; translation {r['counts']['translation']}"]
        if "aligned" in r:
            txt.append(f"aligned median {r['aligned']['median_deg']:.3f} deg (p90 {r['aligned']['p90_deg']:.3f}, max {r['aligned']['max_deg']:.3f}); "
                       f"constant baseline {r['baseline_aligned']['median_deg']:.3f}; 1 s relative {r['relative_1s']['median_deg']} vs {r['relative_1s']['baseline_median_deg']}")
        txt.append(f"edges accepted {r['counts']['edges_rotation_accepted']}/{r['counts']['edges_total']}; rejections {r['counts']['edge_reject_reasons']}")
        txt.append(f"SfM validity {r['sfm_validity']}")
        cmp_ = ev["w142_comparison"].get(dsn, {})
        txt.append(f"W142 coverage {cmp_.get('w142_coverage')}; common-frame comparison {cmp_.get('common')}")
        axs[4].text(0, 1, "\n".join(txt), va="top", family="monospace", fontsize=7)
        fig.suptitle(f"W143 {dsn}: decoupled orientation (W143) vs W142 vs GT, camera state, translation observability, edge validation")
        fig.tight_layout()
        p = out / f"{dsn}_w143_orientation_state.png"
        fig.savefig(p, dpi=95)
        plt.close(fig)
        made.append(p)

    # ---------------------------------------------------------------- representative correspondences (accepted / rejected)
    for dsn, picks in (("walking_static", None), ("sitting_static", None), ("s1", None)):
        est = load(E / f"w143_{dsn}" / "estimate.json")
        F = np.load(E / f"w143_{dsn}" / "features.npz")
        Eg = np.load(E / f"w143_{dsn}" / "edges.npz")
        ind = Path(pf["inputs"][dsn]["dir"])
        mdir = Path(pf["masks"][dsn]["dir"])
        K = np.array([[est["camera"]["fx"], 0, est["camera"]["cx"]], [0, est["camera"]["fy"], est["camera"]["cy"]], [0, 0, 1.0]])
        acc = [e for e in est["edges"] if e["rotation_accepted"] and e["span"] == 8]
        rej = [e for e in est["edges"] if not e["rotation_accepted"] and e["span"] == 8 and e.get("M") is not None]
        sel = ([acc[len(acc) // 4], acc[3 * len(acc) // 4]] if len(acc) >= 4 else acc[:2]) + rej[:2]
        rows = []
        for e in sel:
            f = est["frames"][e["i"]]
            img = cv2.imread(str(ind / "images" / f["image"]))
            ex = cv2.imread(str(mdir / (f["image"] + ".png")), cv2.IMREAD_UNCHANGED) == 0
            img[ex] = (0.55 * img[ex] + 0.45 * np.array([0, 0, 255])).astype(np.uint8)
            mm = Eg[f"e{e['i']}_{e['j']}"]
            x1, x2 = F[f"f{e['i']}"][mm[:, 0]], F[f"f{e['j']}"][mm[:, 1]]
            pred, _ = BR.project_bearing(K, (np.asarray(e["M"]) @ BR.bearings(K, x1).T).T)
            a_, b_ = img.copy(), img.copy()
            for p, q, r_ in zip(x1[::2], x2[::2], pred[::2]):
                cv2.arrowedLine(a_, tuple(int(v) for v in p), tuple(int(v) for v in p + 5 * (q - p)), (0, 0, 255), 1, tipLength=0.2)
                cv2.arrowedLine(b_, tuple(int(v) for v in p), tuple(int(v) for v in p + 10 * (q - r_)), (0, 220, 0), 1, tipLength=0.2)
            st = "ACCEPTED" if e["rotation_accepted"] else f"REJECTED {e['reject']}"
            label(a_, f"{dsn} {e['frame_ids'][0]}->{e['frame_ids'][1]} span {e['span']} fit={e['fit_model']} {st}"[:95], (5, 16))
            label(a_, f"raw flow of rotation inliers x5; support {e.get('rot_support_ratio', 0):.2f}, RH {e.get('agree_rot_vs_homography_deg')}", (5, 34))
            label(b_, f"residual after rotation x10: p50 {e.get('rot_consensus_resid_p50_px', 0):.2f} p90 {e.get('rot_consensus_resid_p90_px', 0):.2f} px; "
                      f"bias ind {e.get('rotation_bias_bound_deg', 0):.3f} deg", (5, 16))
            label(b_, f"translation edge: {e['translation_edge']}; cycle n={e.get('cycle_n')} med={e.get('cycle_median_deg')}", (5, 34))
            rows.append(np.hstack([a_, b_]))
        if rows:
            p = out / f"{dsn}_w143_correspondences.png"
            cv2.imwrite(str(p), np.vstack(rows))
            made.append(p)

    # ---------------------------------------------------------------- S1 timeline
    est = load(E / "w143_s1" / "estimate.json")
    t = np.asarray([f["timestamp"] for f in est["frames"]])
    fig, ax = plt.subplots(2, 1, figsize=(14, 6))
    for e in est["edges"]:
        key = "accepted" if e["rotation_accepted"] else (e["reject"][-1] if e["reject"] else "rotation_inliers")
        ax[0].plot([t[e["i"]] - t[0]], [np.log2(e["span"])], "s", ms=3, color=REJ_COL.get(key, (1, 0, 1)))
    ax[0].set_yticks(range(6))
    ax[0].set_yticklabels([f"span {2 ** s}" for s in range(6)])
    ax[0].legend([plt.Line2D([], [], marker="s", ls="", color=c) for c in REJ_COL.values()], list(REJ_COL), fontsize=6, ncol=4)
    xs = list(t - t[0]) + [t[-1] - t[0] + 1 / 30]
    band(ax[1], xs, [f["camera_state"] for f in est["frames"]], CAM_COL, 0.55, 0.4)
    band(ax[1], xs, [f["translation_window"]["translation"] for f in est["frames"]], TR_COL, 0.05, 0.4)
    ax[1].set_yticks([0.25, 0.75])
    ax[1].set_yticklabels(["translation obs.", "camera state"])
    ax[1].set_xlim(xs[0], xs[-1])
    fig.suptitle(f"W143 S1 (authentic, NOMINAL_UNCALIBRATED): edge validation and camera state; counts {est['counts']['camera_state']}")
    fig.tight_layout()
    p = out / "s1_w143_state.png"
    fig.savefig(p, dpi=100)
    plt.close(fig)
    made.append(p)

    # ---------------------------------------------------------------- playback videos (real-time pacing)
    for dsn in ("sitting_static", "walking_static"):
        est = load(E / f"w143_{dsn}" / "estimate.json")
        F = np.load(E / f"w143_{dsn}" / "features.npz")
        ind = Path(pf["inputs"][dsn]["dir"])
        mdir = Path(pf["masks"][dsn]["dir"])
        ts = np.asarray([f["timestamp"] for f in est["frames"]])
        vid = out / f"{dsn}_w143_playback_realtime.mp4"
        wr, acc_, n_out = None, 0.0, 0
        for k, f in enumerate(est["frames"]):
            img = cv2.imread(str(ind / "images" / f["image"]))
            ex = cv2.imread(str(mdir / (f["image"] + ".png")), cv2.IMREAD_UNCHANGED) == 0
            img[ex] = (0.55 * img[ex] + 0.45 * np.array([0, 0, 255])).astype(np.uint8)
            for u, v in F[f"f{k}"][::2]:
                cv2.circle(img, (int(round(u)), int(round(v))), 2, (0, 255, 0), -1)
            panel = np.full((480, 440, 3), 255, np.uint8)
            o = f["orientation"]
            lines = [f"W143 {dsn}  t={f['timestamp'] - ts[0]:.2f}s  frame {f['frame_id']}", f"camera: {f['camera_state']}", f"orientation: {o['status']}",
                     f"  confidence: {o['evidence']['confidence']}  accepted edges {o['evidence']['accepted_edges']}",
                     f"translation: {f['translation']['status']}", f"  value: {'null' if f['translation']['value'] is None else 'SfM, SCALE_AMBIGUOUS'}",
                     f"SfM: {f['sfm']['model_status']}", f"background keypoints: {f['evidence']['n_kp']}"]
            for n_, s_ in enumerate(lines):
                cv2.putText(panel, s_, (8, 24 + 22 * n_), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(panel, "red = excluded person mask; green = background kp", (8, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1)
            cv2.putText(panel, "real-time pacing (selected frames held)", (8, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (90, 90, 90), 1)
            fr = np.hstack([img, panel])
            if wr is None:
                wr = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (fr.shape[1], fr.shape[0]))
            dt = (ts[k + 1] - ts[k]) if k + 1 < len(ts) else float(np.median(np.diff(ts)))
            acc_ += dt * 30.0
            for _ in range(max(int(round(acc_)) - n_out, 1 if k == 0 else 0)):
                wr.write(fr)
                n_out += 1
        wr.release()
        made.append(vid)
        notes[vid.name] = f"{n_out} frames at 30 fps = {n_out / 30:.2f} s for a {ts[-1] - ts[0]:.2f} s span (real-time pacing)"
    ue = ROOT / "user_eval" / "W143_decoupled_camera"
    ue.mkdir(parents=True, exist_ok=True)
    for p in made:
        shutil.copyfile(p, ue / p.name)
    CP.dump(out / "visualize.json", {"files": {p.name: CP.sha256_file(p) for p in made}, "copied_to": str(ue), "notes": notes,
                                     "review": "AGENT-generated; not reviewed by a human"})
    print(json.dumps({"files": [p.name for p in made], "notes": notes}, indent=1))
    return 0
