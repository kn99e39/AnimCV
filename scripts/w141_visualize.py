"""W141 visualisation (called by ``w141_dynamic_camera_harness.py visualize``): reads only FROZEN W141 outputs.

Produces mask / retained-vs-excluded feature overlays, high-occlusion frames, GT vs aligned trajectories per condition,
error and occupancy curves, synthetic clean/dynamic/masked comparisons, playback videos (real-time pacing: each selected
frame held for its true inter-frame interval) and third-person 3D viewport videos (scripts/w140_viewport_blender.py,
unchanged).  Never changes any estimate, metric, gate or classification.  Not reviewed by a human unless stated.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

import app.experimental_camera_pose as CP
import app.experimental_dynamic_camera_pose as DP

COL = {"A": (0, 0, 230), "B": (0, 170, 0), "C": (230, 120, 0), "GT": (0, 0, 0)}           # BGR
MCOL = {"A": "tab:red", "B": "tab:green", "C": "tab:blue", "GT": "k"}
SYNC = {"syn_clean": "tab:gray", "syn_unmasked": "tab:red", "syn_rgbmask": "tab:green", "syn_oracle": "tab:purple", "syn_oracle_areactrl": "tab:blue"}


def label(img, txt, org, col=(255, 255, 255), sc=0.45, th=1):
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, sc, (0, 0, 0), th + 2, cv2.LINE_AA)
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, sc, col, th, cv2.LINE_AA)


def tint(img, m, bgr, alpha=0.45):
    out = img.copy()
    out[m] = (alpha * np.asarray(bgr) + (1 - alpha) * out[m]).astype(np.uint8)
    cnt, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, cnt, -1, bgr, 1)
    return out


def load(p):
    return json.load(open(p, encoding="utf-8"))


def visualize(a, ROOT, W141, SYN_DYN, BLENDER, SEQS, RUNS):
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
    R = ev["runs"]

    def kp(name):
        return DP.read_db_keypoints(E / name / "database.db")

    def model_obs(name):
        p = E / name / "primary_model_txt" / "images.txt"
        return DP.read_images_txt(p) if p.is_file() else {}

    def status_map(name):
        return {f["frame_id"]: f["status"] for f in load(E / name / "estimate.json")["frames"]}

    # ------------------------------------------------------------------ real sequences: masks / features / occlusion
    for seq in SEQS:
        ind = Path(pf["inputs"][seq]["dir"])
        md = Path(pf["masks"][seq]["dir"])
        frames = load(ind / "frames.json")
        ms = load(md / "mask_summary.json")["per_frame"]
        occ = np.asarray([r["raw_frac"] for r in ms])
        kA = kp(f"{seq}_A")
        oA = model_obs(f"{seq}_A")
        oB = model_obs(f"{seq}_B")
        st = {c: status_map(f"{seq}_{c}") for c in "ABC"}
        uni = list(np.linspace(0, len(frames) - 1, 4).astype(int))
        top = [int(i) for i in np.argsort(-occ, kind="stable")[:2] if int(i) not in uni]
        rows = []
        for i in uni + top:
            f = frames[i]
            img = cv2.imread(str(ind / "images" / f["image"]))
            raw = cv2.imread(str(md / "raw" / f["image"]), 0) > 127
            Bx = cv2.imread(str(md / "keep_rgbmask" / (f["image"] + ".png")), 0) == 0
            Cx = cv2.imread(str(md / "keep_areactrl" / (f["image"] + ".png")), 0) == 0
            xy = kA.get(f["image"], np.zeros((0, 2)))
            inB, inC = DP.keypoints_in_mask(xy, Bx), DP.keypoints_in_mask(xy, Cx)
            pB = tint(tint(img, Bx & ~raw, (0, 0, 160), 0.25), raw, (0, 0, 255), 0.45)
            pC = tint(img, Cx, (255, 120, 0), 0.40)
            for (u, v), b in zip(xy - 0.5, inB):
                cv2.circle(pB, (int(round(u)), int(round(v))), 1, (0, 0, 255) if b else (0, 255, 0), -1)
            for (u, v), c in zip(xy - 0.5, inC):
                cv2.circle(pC, (int(round(u)), int(round(v))), 1, (255, 0, 0) if c else (0, 255, 0), -1)
            tag = "HIGH-OCCUPANCY " if i in top else ""
            label(pB, f"{tag}{seq} frame {f['frame_id']} t={f['timestamp']:.3f}  person {occ[i]:.1%}", (5, 16))
            label(pB, f"B mask (RGB person + {DP.DILATE_PX}px): kept {int((~inB).sum())} / excluded {int(inB.sum())} kp", (5, 34))
            label(pC, f"C area control: kept {int((~inC).sum())} / excluded {int(inC.sum())} kp", (5, 16))
            label(pC, "status A/B/C: " + "/".join(st[c].get(f["frame_id"], "?") for c in "ABC"), (5, 34))
            rows.append(np.hstack([pB, pC]))
        p = out / f"{seq}_masks_features.png"
        cv2.imwrite(str(p), np.vstack(rows))
        made.append(p)
        # high occlusion: model-observed features of A (yellow inside person mask = contamination) vs B
        rows = []
        for i in [int(j) for j in np.argsort(-occ, kind="stable")[:4]]:
            f = frames[i]
            img = cv2.imread(str(ind / "images" / f["image"]))
            raw = cv2.imread(str(md / "raw" / f["image"]), 0) > 127
            tiles = []
            for c, obs in (("A", oA), ("B", oB)):
                pimg = tint(img, raw, (0, 0, 255), 0.35)
                o = obs.get(f["image"])
                n_in = 0
                if o is not None:
                    inn = DP.keypoints_in_mask(o["xy"], raw)
                    n_in = int(inn.sum())
                    for (u, v), b in zip(o["xy"] - 0.5, inn):
                        cv2.circle(pimg, (int(round(u)), int(round(v))), 2, (0, 255, 255) if b else (0, 255, 0), -1)
                label(pimg, f"{c}: {seq} frame {f['frame_id']} t={f['timestamp']:.3f} person {occ[i]:.1%} [{st[c].get(f['frame_id'])}]", (5, 16))
                label(pimg, f"model-observed features: {0 if o is None else len(o['xy'])}, on person (yellow) {n_in}", (5, 34))
                tiles.append(pimg)
            rows.append(np.hstack(tiles))
        p = out / f"{seq}_high_occlusion_model_features.png"
        cv2.imwrite(str(p), np.vstack(rows))
        made.append(p)

        # ------------------------------------------------------------------ trajectories + errors
        fig, ax = plt.subplots(2, 3, figsize=(17, 9))
        t0 = frames[0]["timestamp"]
        gt_done = False
        for c in "ABC":
            e = R.get(f"{seq}_{c}", {})
            if "full_sim3" not in e:
                continue
            Pa, Pg = np.asarray(e["aligned_positions"]), np.asarray(e["gt_positions"])
            t = np.asarray(e["times"]) - t0
            if not gt_done:
                ax[0, 0].plot(Pg[:, 0], Pg[:, 1], "k-", lw=2, label="GT (motion capture)")
                ax[0, 1].plot(Pg[:, 0], Pg[:, 2], "k-", lw=2, label="GT")
                ax[1, 2].plot(t, np.linalg.norm(Pg - Pg.mean(0), axis=1), "k-", lw=2, label="GT |c - mean|")
                gt_done = True
            lab = f"{c} ALIGNED EST (cov {e['coverage_estimated']:.0%}, ATE {e['full_sim3']['ate']['rmse'] * 1000:.1f} mm, rot {e['full_sim3']['rot']['median_deg']:.2f}°)"
            ax[0, 0].plot(Pa[:, 0], Pa[:, 1], "-", color=MCOL[c], lw=1, label=lab)
            ax[0, 1].plot(Pa[:, 0], Pa[:, 2], "-", color=MCOL[c], lw=1)
            ax[1, 0].plot(t, np.asarray(e["full_sim3"]["ate"]["per_frame"]) * 1000, "-", color=MCOL[c], lw=1, label=f"{c} position error")
            ax[1, 1].plot(t, e["full_sim3"]["rot"]["per_frame"], "-", color=MCOL[c], lw=1, label=f"{c} rotation error")
            ax[1, 2].plot(t, np.linalg.norm(Pa - Pg.mean(0) - (Pa.mean(0) - Pg.mean(0)), axis=1), "-", color=MCOL[c], lw=1, label=f"{c} |c - mean|")
            for s_ in e["spikes_gt_referenced"]["spikes"]:
                ax[1, 0].axvline(s_["t"] - t0, color=MCOL[c], alpha=0.25, lw=0.8)
        tt = np.asarray([f["timestamp"] for f in frames]) - t0
        ax2 = ax[1, 0].twinx()
        ax2.fill_between(tt, occ * 100, color="orange", alpha=0.2, step="mid")
        ax2.set_ylabel("RGB person occupancy [%] (orange)")
        for k, (x_, y_) in enumerate(("xy", "xz")):
            ax[0, k].set_xlabel(f"{x_} [m, GT world]")
            ax[0, k].set_ylabel(f"{y_} [m]")
            ax[0, k].axis("equal")
        ax[0, 0].legend(fontsize=7)
        ax[1, 0].set_ylabel("position error [mm] (full Sim3)")
        ax[1, 1].set_ylabel("rotation error [deg]")
        ax[1, 2].set_ylabel("distance from trajectory mean [m]")
        for axx in (ax[1, 0], ax[1, 1], ax[1, 2]):
            axx.set_xlabel("time [s]")
            axx.legend(fontsize=7, loc="upper left")
        comp = ev["comparisons"][seq]
        ax[0, 2].axis("off")
        txt = [f"{seq}: masking benefit verdict = {comp['masking_benefit']['verdict']}", f"common registered frames: {comp['n_common']}"]
        for c in "ABC":
            cm = comp["common"][c]
            g = comp["gates"][c]
            txt.append(f"{c}: coverage {comp['coverage'][c]:.3f}  common ATE {1000 * cm.get('ate_rmse', float('nan')):.1f} mm  rot {cm.get('rot_median_deg', float('nan')):.2f}°  gate {'PASS' if g.get('pass') else 'fail'}")
        e0 = R.get(f"{seq}_A", {})
        if "baseline_stationary" in e0:
            txt.append(f"stationary baseline: ATE {1000 * e0['baseline_stationary']['ate_rmse']:.1f} mm, rot {e0['baseline_stationary']['rot_median_deg']:.2f}°")
            txt.append(f"GT path {e0['path_length']['gt_m']:.3f} m over selected frames")
        txt.append("Sim(3) per condition is evaluation-only; scale is not recovered (SCALE_AMBIGUOUS).")
        ax[0, 2].text(0, 1, "\n".join(txt), va="top", family="monospace", fontsize=8)
        fig.suptitle(f"W141 {seq}: GT vs ALIGNED ESTIMATE (A unmasked / B RGB person exclusion / C area control)")
        fig.tight_layout()
        p = out / f"{seq}_trajectories_errors.png"
        fig.savefig(p, dpi=105)
        plt.close(fig)
        made.append(p)

        # ------------------------------------------------------------------ playback video (real-time pacing)
        pos = {}
        allp = []
        for c in ("A", "B", "C"):
            e = R.get(f"{seq}_{c}", {})
            if "full_sim3" in e:
                pos[c] = dict(zip(e["frame_ids"], np.asarray(e["aligned_positions"])))
                allp.append(np.asarray(e["aligned_positions"]))
                if "GT" not in pos:
                    pos["GT"] = dict(zip(e["frame_ids"], np.asarray(e["gt_positions"])))
                    allp.append(np.asarray(e["gt_positions"]))
        if allp:
            P_ = np.vstack(allp)
            lo, hi = np.percentile(P_, 1, axis=0), np.percentile(P_, 99, axis=0)
            span = max(hi[:2] - lo[:2]) * 1.3 + 1e-6
            ctr = (hi + lo) / 2
            vid = out / f"{seq}_playback_realtime.mp4"
            wr = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (1120, 480))
            ts = np.asarray([f["timestamp"] for f in frames])
            acc, n_out = 0.0, 0

            def px(p):
                return int(240 + (p[0] - ctr[0]) / span * 440), int(240 - (p[1] - ctr[1]) / span * 440)
            for i, f in enumerate(frames):
                img = cv2.imread(str(ind / "images" / f["image"]))
                raw = cv2.imread(str(md / "raw" / f["image"]), 0) > 127
                img = tint(img, raw, (0, 0, 255), 0.35)
                o = oB.get(f["image"])
                if o is not None:
                    for u, v in o["xy"][::2] - 0.5:
                        cv2.circle(img, (int(round(u)), int(round(v))), 2, (0, 255, 0), -1)
                label(img, f"{seq} t={f['timestamp'] - t0:6.2f}s frame {f['frame_id']}  person {occ[i]:.0%}", (6, 18), sc=0.5)
                label(img, "red = RGB person mask; green = B model features", (6, 470), sc=0.45)
                panel = np.full((480, 480, 3), 255, np.uint8)
                for c in ("GT", "A", "B", "C"):
                    if c not in pos:
                        continue
                    pts = [pos[c][fr["frame_id"]] for fr in frames[:i + 1] if fr["frame_id"] in pos[c]]
                    for k in range(len(pts) - 1):
                        cv2.line(panel, px(pts[k]), px(pts[k + 1]), COL[c], 2 if c == "GT" else 1)
                    if f["frame_id"] in pos[c]:
                        cv2.circle(panel, px(pos[c][f["frame_id"]]), 5 if c == "GT" else 4, COL[c], -1)
                label(panel, "top view x-y (GT world, m): GT black", (6, 16), (0, 0, 0), sc=0.42, th=1)
                label(panel, "ALIGNED EST: A red / B green / C blue", (6, 34), (0, 0, 0), sc=0.42, th=1)
                yy = 56
                for c in "ABC":
                    s_ = st[c].get(f["frame_id"], "?")
                    label(panel, f"{c}: {'ESTIMATED' if s_ == 'ESTIMATED' else 'TRACKING LOST' if s_ == 'LOST' else s_}", (6, yy),
                          (0, 120, 0) if s_ == "ESTIMATED" else (0, 0, 200), sc=0.45)
                    yy += 18
                cv2.line(panel, (380, 460), (380 + int(0.1 / span * 440), 460), (0, 0, 0), 2)
                label(panel, "0.1 m", (380, 452), (0, 0, 0), sc=0.4)
                label(panel, "real-time pacing (selected frames held)", (190, 476), (90, 90, 90), sc=0.38)
                fr = np.hstack([img, panel])
                dt = (ts[i + 1] - ts[i]) if i + 1 < len(ts) else float(np.median(np.diff(ts)))
                acc += dt * 30.0
                for _ in range(max(int(round(acc)) - n_out, 1 if i == 0 else 0)):
                    wr.write(fr)
                    n_out += 1
            wr.release()
            made.append(vid)
            notes[vid.name] = f"{n_out} frames at 30 fps = {n_out / 30:.2f} s for a {ts[-1] - ts[0]:.2f} s sequence (real-time pacing)"

    # ------------------------------------------------------------------ synthetic comparisons
    syn_names = ["syn_clean", "syn_unmasked", "syn_rgbmask", "syn_oracle", "syn_oracle_areactrl"]
    fig, ax = plt.subplots(2, 3, figsize=(17, 9))
    gt_done = False
    for n in syn_names:
        e = R.get(n, {})
        if "full_sim3" not in e:
            continue
        Pa, Pg = np.asarray(e["aligned_positions"]), np.asarray(e["gt_positions"])
        fid = np.asarray(e["frame_ids"])
        if not gt_done:
            ax[0, 0].plot(Pg[:, 0], Pg[:, 1], "k-", lw=2.5, label="GT (exact)")
            gt_done = True
        lab = f"{n}: cov {e['coverage_estimated']:.0%} ATE {e['full_sim3']['ate']['rmse'] * 1000:.2f} mm rot {e['full_sim3']['rot']['median_deg']:.3f}°"
        ax[0, 0].plot(Pa[:, 0], Pa[:, 1], "-", color=SYNC[n], lw=1, label=lab)
        ax[1, 0].semilogy(fid, np.maximum(np.asarray(e["full_sim3"]["ate"]["per_frame"]) * 1000, 1e-3), "-", color=SYNC[n], lw=1, label=n)
        ax[1, 1].semilogy(fid, np.maximum(e["full_sim3"]["rot"]["per_frame"], 1e-4), "-", color=SYNC[n], lw=1, label=n)
    seg = load(SYN_DYN / "gt" / "segments.json")
    for axx in (ax[1, 0], ax[1, 1]):
        for s_, (lo_, hi_) in seg.items():
            axx.axvspan(lo_, hi_, alpha=0.06 if s_ in ("stationary", "rotation") else 0.0, color="gray")
            axx.text((lo_ + hi_) / 2, axx.get_ylim()[1] if axx.get_ylim()[1] > 0 else 1, s_, ha="center", va="top", fontsize=7)
        axx.set_xlabel("frame (30 fps nominal)")
        axx.legend(fontsize=7)
    ax[0, 0].axis("equal")
    ax[0, 0].legend(fontsize=7)
    ax[1, 0].set_ylabel("position error [mm] (log)")
    ax[1, 1].set_ylabel("rotation error [deg] (log)")
    dm = load(Path(pf["masks"]["syn_dynamic"]["dir"]) / "mask_summary.json")["per_frame"]
    ax[0, 1].plot([r["frame_id"] for r in dm], [100 * r["oracle_frac"] for r in dm], "k-", label="oracle foreground %")
    ax[0, 1].plot([r["frame_id"] for r in dm], [100 * r["raw_frac"] for r in dm], "g-", label="RGB person mask %")
    ax[0, 1].plot([r["frame_id"] for r in dm], [100 * (r["rgb_vs_oracle_raw"]["iou"] or 0) for r in dm], "m:", label="RGB vs oracle IoU %")
    ax[0, 1].set_xlabel("frame")
    ax[0, 1].legend(fontsize=7)
    ax[0, 1].set_title("synthetic foreground occupancy and RGB-mask quality")
    ax[0, 2].axis("off")
    sy = ev["comparisons"].get("synthetic", {})
    txt = [f"clean reproduces W140 gates: {sy.get('clean_reproduces_w140_gates')}",
           f"dynamic foreground disturbs (unmasked): {sy.get('dynamic_foreground_disturbs')}"]
    for n in syn_names[1:]:
        d_ = sy.get(n, {})
        txt.append(f"{n}: disturbed={d_.get('disturbed')} {'; '.join(d_.get('why', []))[:70]}")
    for n in syn_names:
        e = R.get(n, {})
        sg = e.get("segments", {}).get("stationary", {})
        lc = e.get("landmark_consistency", {})
        if sg:
            txt.append(f"{n}: stationary path {1000 * sg.get('est_path_length', 0):.2f} mm (GT 0); landmarks<2cm {lc.get('frac_within_2cm', float('nan')):.3f}")
    ax[0, 2].text(0, 1, "\n".join(txt), va="top", family="monospace", fontsize=7.5)
    ax[1, 2].axis("off")
    fig.suptitle("W141 synthetic: clean (W140 frames) vs dynamic unmasked vs RGB-mask vs ORACLE-mask vs oracle area control")
    fig.tight_layout()
    p = out / "synthetic_comparison.png"
    fig.savefig(p, dpi=105)
    plt.close(fig)
    made.append(p)
    # synthetic mask overlay frames
    ind = Path(pf["inputs"]["syn_dynamic"]["dir"])
    md = Path(pf["masks"]["syn_dynamic"]["dir"])
    kU = kp("syn_unmasked")
    rows = []
    for fi in (5, 40, 95, 130):
        nm = f"f{fi:04d}.png"
        img = cv2.imread(str(ind / "images" / nm))
        orc = cv2.imread(str(SYN_DYN / "gt" / "fg_masks" / nm), 0) > 127
        raw = cv2.imread(str(md / "raw" / nm), 0) > 127
        xy = kU.get(nm, np.zeros((0, 2)))
        a_ = tint(img, orc, (255, 0, 255), 0.4)
        b_ = tint(img, raw, (0, 0, 255), 0.4)
        for (u, v), bb in zip(xy - 0.5, DP.keypoints_in_mask(xy, orc)):
            cv2.circle(a_, (int(round(u)), int(round(v))), 1, (0, 255, 255) if bb else (0, 255, 0), -1)
        q = DP.mask_quality(raw, orc)
        label(a_, f"synthetic frame {fi}: ORACLE foreground (magenta); yellow = unmasked kp on movers", (5, 16))
        label(b_, f"RGB Mask R-CNN person mask (red): IoU {q['iou'] if q['iou'] is None else round(q['iou'], 2)} recall {q['recall'] if q['recall'] is None else round(q['recall'], 2)}", (5, 16))
        rows.append(np.hstack([a_, b_]))
    p = out / "synthetic_masks_features.png"
    cv2.imwrite(str(p), np.vstack(rows))
    made.append(p)

    # ------------------------------------------------------------------ 3D viewport videos (third-person, W140 Blender renderer unchanged)
    vp = {}
    if getattr(a, "skip_viewport", False) is False:
        for seq in SEQS:
            for c in ("A", "B"):
                name = f"{seq}_{c}"
                e = R.get(name, {})
                if "full_sim3" not in e:
                    continue
                vp[name] = viewport(name, seq, e, pf, E, out, ROOT, BLENDER, W141)
                made += [Path(vp[name]["video"]), Path(vp[name]["stills"])]
    ue = ROOT / "user_eval" / "W141_dynamic_camera_pose"
    ue.mkdir(parents=True, exist_ok=True)
    for p in made:
        shutil.copyfile(p, ue / p.name)
    CP.dump(out / "visualize.json", {"files": {p.name: CP.sha256_file(p) for p in made}, "copied_to": str(ue), "notes": notes, "viewport": vp,
                                     "review": "AGENT-generated; not reviewed by a human"})
    print(json.dumps({"files": [p.name for p in made], "user_eval": str(ue), "notes": notes}, indent=1))
    return 0


def viewport(name, seq, e, pf, E, out, ROOT, BLENDER, W141):
    data = W141 / "data" / "extracted" / f"rgbd_dataset_freiburg3_{seq}" / "groundtruth.txt"
    gt = CP.read_groundtruth(data)
    est = load(E / name / "estimate.json")
    cam = load(Path(pf["inputs"][seq]["dir"]) / "camera.json")
    sim = e["full_sim3"]["sim3"]
    s, Rs, ts = sim["scale"], np.asarray(sim["R"]), np.asarray(sim["t"])
    frames = []
    for f in est["frames"]:
        Tg, _ = CP.interpolate_gt(gt, f["timestamp"])
        Te = CP.apply_sim3((s, Rs, ts), [np.asarray(f["T_WC"])])[0] if f["status"] == "ESTIMATED" else None
        frames.append({"frame_id": f["frame_id"], "timestamp": f["timestamp"], "image": f["image"], "status": f["status"],
                       "gt_T": Tg.tolist() if Tg is not None else None, "est_T": Te.tolist() if Te is not None else None})
    pts = np.load(E / name / "points3D.npy")
    P = (s * (Rs @ pts[:, :3].T)).T + ts if len(pts) else np.zeros((0, 3))
    c = np.mean([np.asarray(f["gt_T"])[:3, 3] for f in frames if f["gt_T"]], axis=0)
    P = P[np.linalg.norm(P - c, axis=1) < 3.0]
    P = P[np.random.default_rng(0).permutation(len(P))[:4000]] if len(P) > 4000 else P
    spec = {"dataset": name, "K": {k: cam[k] for k in ("fx", "fy", "cx", "cy", "width", "height")}, "frames": frames, "points": P.tolist(),
            "sim3": sim, "view_azimuth_deg": 30.0, "input_dir": pf["inputs"][seq]["dir"], "grid_at_floor": False}
    sp = out / f"{name}_viewport_spec.json"
    sp.write_text(json.dumps(spec))
    fdir = out / f"viewport_frames_{name}"
    r = subprocess.run([BLENDER, "-b", "--factory-startup", "--python-exit-code", "1", "-P", str(ROOT / "scripts" / "w140_viewport_blender.py"),
                        "--", "--spec", str(sp), "--out", str(fdir)], capture_output=True, text=True, encoding="utf-8", errors="replace")
    (out / f"{name}_viewport_blender.log").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr, encoding="utf-8")
    if r.returncode != 0:
        return {"error": r.stderr[-500:], "video": None, "stills": None}
    gz = load(fdir / "view.json")["grid_z_m"]
    t = np.asarray([f["timestamp"] for f in frames])
    vid = out / f"{name}_viewport3d_realtime.mp4"
    wr = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (1280, 720))
    n_out, acc = 0, 0.0
    cond = {"A": "A unmasked", "B": "B RGB person exclusion"}[name[-1]]
    for i, f in enumerate(frames):
        v = cv2.imread(str(fdir / f"v{i:04d}.png"))
        rgb = cv2.imread(str(Path(spec["input_dir"]) / "images" / f["image"]))
        v[720 - 290:720 - 50, 1280 - 336:1280 - 16] = cv2.resize(rgb, (320, 240))
        label(v, "camera view (RGB input)", (1280 - 334, 720 - 298), sc=0.5)
        label(v, f"W141 {seq} [{cond}]  t = {f['timestamp'] - t[0]:6.2f} s   frame {f['frame_id']}", (16, 34), sc=0.7, th=2)
        label(v, "GT camera / path (motion capture)", (16, 600), (240, 240, 240), sc=0.55)
        label(v, "ALIGNED ESTIMATE camera / path (RGB-only COLMAP; Sim(3) for evaluation only)", (16, 624), (13, 115, 255), sc=0.55)
        label(v, f"grid: horizontal plane z = {gz:.1f} m (parallel to floor, z up); 0.1/1 m; red X, green Y", (16, 648), (200, 200, 200), sc=0.48)
        stt = "ESTIMATED" if f["status"] == "ESTIMATED" else ("TRACKING LOST" if f["status"] == "LOST" else f["status"])
        label(v, stt, (16, 700), (80, 230, 80) if stt == "ESTIMATED" else (60, 60, 255), sc=0.9, th=2)
        label(v, "external 3rd-person view (fixed); real-time pacing", (820, 700), (200, 200, 200), sc=0.5)
        dt = (t[i + 1] - t[i]) if i + 1 < len(t) else float(np.median(np.diff(t)))
        acc += dt * 30.0
        for _ in range(max(int(round(acc)) - n_out, 1 if i == 0 else 0)):
            wr.write(v)
            n_out += 1
    wr.release()
    stills = [0, len(frames) // 3, 2 * len(frames) // 3, len(frames) - 1]
    tiles = [cv2.resize(cv2.imread(str(fdir / f"v{i:04d}.png")), (640, 360)) for i in stills]
    sheet = out / f"{name}_viewport3d_stills.png"
    cv2.imwrite(str(sheet), np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])]))
    return {"video": str(vid), "stills": str(sheet), "frames_written": n_out, "duration_s": n_out / 30.0, "sequence_span_s": float(t[-1] - t[0])}
