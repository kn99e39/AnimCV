#!/usr/bin/env python3
"""W140 (EXPERIMENTAL, opt-in, scratch only): Scene-Anchored Camera Pose Reconstruction harness.

Datasets: A = deterministic Blender static scene (scripts/w140_synthetic_scene.py); B = TUM RGB-D fr1_xyz; C = TUM fr1_rpy.
Estimator: COLMAP incremental SfM via pycolmap 4.2.1 (scripts/w140_colmap_estimate.py), RGB + timestamps + published
intrinsics ONLY.  GT is read only by `evaluate`, after `estimate` has frozen every trajectory (sha recorded).

Stages (each refuses an existing output directory):
  preflight  repo/branch/worktree, dataset identities (archive SHA-256, rgb.txt/groundtruth.txt SHA-256), licence and
             calibration text from the official pages, deterministic frame selection (300 uniform by rgb.txt index),
             image decode check, estimator input directories (TUM frames undistorted with the published 5-parameter model,
             PINHOLE camera.json, frames.json with original timestamps; synthetic copied), input manifests, oracle
             projection contract (Control C, synthetic GT + marker landmarks; never an estimator input).
  declare    prospective declaration (method, versions, config, selections, conventions, alignment, metrics, gates).
  estimate   run the frozen estimator on synthetic, fr1_xyz, fr1_rpy; record runtime; freeze outputs (SHA-256).
  evaluate   GT association/interpolation, Sim(3) full and early-segment alignment, ATE/RPE/rotation/drift/temporal,
             stationary baseline (Control A), synthetic segment analysis, parallax diagnostics, gates.
             The prospective classification (SUPPORTED / PARTIALLY / NOT YET / BLOCKED) is computed here.
  visualize  plots, feature-track overlays, failure examples, synchronized trajectory video; copy to user_eval/.

    PY=C:/Projects/AnimCV/.venv_build_full/Scripts/python.exe; export PYTHONPATH=src
    $PY -X utf8 scripts/w140_camera_harness.py <stage> --out-dir <dir> [--preflight P --declaration D --declaration-sha S
        --estimate-dir E --eval-dir V]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]
import numpy as np                                                    # noqa: E402

import app.experimental_camera_pose as CP                             # noqa: E402

W140 = Path("C:/Users/dna10/animcv-w140")
DATA = W140 / "data"
PC_PY = W140 / "env" / "pc311" / "Scripts" / "python.exe"
EST = ROOT / "scripts" / "w140_colmap_estimate.py"
BRANCH = "Dual_Stage_Generator"
TUM = {"xyz": {"archive": "rgbd_dataset_freiburg1_xyz.tgz", "sha256": "a0236d97b8c30cd93b653656d2b6c293ff7c982a4130ef2a1a8beecdb124ef98",
               "url": "https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz"},
       "rpy": {"archive": "rgbd_dataset_freiburg1_rpy.tgz", "sha256": "78103722a25873dbbb4de027eaa8c810a6382100691f02b6cf95c3adc91c4ac1",
               "url": "https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_rpy.tgz"}}
FR1 = {"fx": 517.3, "fy": 516.5, "cx": 318.6, "cy": 255.3, "dist": [0.2624, -0.9531, -0.0054, 0.0026, 1.1633], "width": 640, "height": 480}
CALIB_ROW = "Freiburg 1 RGB 517.3 516.5 318.6 255.3 0.2624 -0.9531 -0.0054 0.0026 1.1633"
LICENCE_CLAUSE = "licensed under a Creative Commons 4.0 Attribution License (CC BY 4.0)"
GT_CLAUSE = "give the position of the optical center of the color camera with respect to the world origin as defined by the motion capture system"
SYN_DIR = DATA / "synthetic_v2"
INSTRUMENTS = ["src/app/experimental_camera_pose.py", "scripts/w140_colmap_estimate.py", "scripts/w140_synthetic_scene.py",
               "scripts/w140_camera_harness.py", "tests/test_experimental_camera_pose.py"]
DATASETS = ("synthetic", "xyz", "rpy")


def sha(p):
    return CP.sha256_file(p)


def lf_sha(rel):
    return hashlib.sha256((ROOT / rel).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def git(*a):
    return subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True).stdout.strip()


def new_dir(p):
    p = Path(p)
    if p.exists():
        raise SystemExit(f"refusing to overwrite {p}")
    p.mkdir(parents=True)
    return p


def load(p):
    return json.load(open(p, encoding="utf-8"))


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def page_text(p):
    import html
    t = html.unescape(re.sub(r"<[^>]+>", " ", Path(p).read_text(encoding="utf-8", errors="replace")))
    return re.sub(r"\s+", " ", t)


def check_declaration(a):
    if M_sha(a.declaration) != a.declaration_sha:
        raise SystemExit("STOP INTEGRITY_FAILED: declaration changed")
    d = load(a.declaration)
    bad = [f for f, h in d["instrument_sha256_lf"].items() if lf_sha(f) != h]
    if bad:
        raise SystemExit(f"STOP DECLARATION_INVALID: instruments changed {bad}")
    return d


def M_sha(p):
    return sha(p)


# ------------------------------------------------------------------------------------------------ oracle (Control C) ---
def marker_centroids(img, colour):
    """Centroid of pure-colour marker pixels (FLAT Workbench render)."""
    import cv2  # noqa: F401
    b, g, r = img[..., 0].astype(int), img[..., 1].astype(int), img[..., 2].astype(int)
    targets = {"red": (r > 200) & (g < 60) & (b < 60), "green": (g > 200) & (r < 60) & (b < 60), "blue": (b > 200) & (r < 60) & (g < 60),
               "yellow": (r > 200) & (g > 200) & (b < 60)}
    m = targets[colour]
    ys, xs = np.nonzero(m)
    if len(xs) < 5:
        return None
    return np.array([xs.mean(), ys.mean()]), int(len(xs))


def oracle_contract(syn_dir):
    """Project the marker landmarks with the GT poses and the declared K; compare with rendered marker centroids.
    Negative variants (inverted transform, xyzw->wxyz misread, focal x1.1, flipped y axis, +1 frame shift) must be worse."""
    import cv2
    cam = load(syn_dir / "input" / "camera.json")
    K = np.array([[cam["fx"], 0, cam["cx"]], [0, cam["fy"], cam["cy"]], [0, 0, 1.0]])
    CP.check_intrinsics(K, cam["width"], cam["height"])
    gt = CP.read_groundtruth(syn_dir / "gt" / "camera_gt.txt")
    lm = load(syn_dir / "gt" / "oracle_landmarks.json")
    obs = []
    for i in range(0, len(gt), 3):
        img = cv2.imread(str(syn_dir / "images" / f"f{i:04d}.png"))
        for nm, X in lm.items():
            c = marker_centroids(img, nm)
            if c is not None and c[1] >= 20:
                obs.append((i, nm, np.asarray(X, float), c[0]))

    def err(variant):
        e = []
        for i, nm, X, uv in obs:
            j = i + 1 if variant == "shift_plus1" and i + 1 < len(gt) else i
            T = CP.T_WC_from_tum(gt[j, 1:4], gt[j, 4:8])
            KK = K.copy()
            if variant == "inverted":
                T = CP.inv_T(T)
            elif variant == "quat_wxyz_misread":
                q = gt[j, 4:8]
                T = CP.T_WC_from_tum(gt[j, 1:4], [q[1], q[2], q[3], q[0]])
            elif variant == "focal_x1.1":
                KK[0, 0] *= 1.1
                KK[1, 1] *= 1.1
            elif variant == "flip_y_axis":
                T = T @ CP.make_T(np.diag([1.0, -1.0, -1.0]), [0, 0, 0])
            p, z = CP.project(KK, T, X[None])
            e.append(float(np.linalg.norm(p[0] - uv)) if z[0] > 0 else 1e6)
        return {"median_px": float(np.median(e)), "max_px": float(np.max(e))}
    res = {"n_observations": len(obs), "correct": err("correct")}
    for v in ("inverted", "quat_wxyz_misread", "focal_x1.1", "flip_y_axis", "shift_plus1"):
        res[v] = err(v)
    res["ok"] = bool(res["n_observations"] >= 20 and res["correct"]["max_px"] <= 1.5 and
                     all(res[v]["median_px"] > 5 * max(res["correct"]["median_px"], 0.2) for v in ("inverted", "quat_wxyz_misread", "flip_y_axis"))
                     and res["focal_x1.1"]["median_px"] > res["correct"]["median_px"] and res["shift_plus1"]["max_px"] > res["correct"]["max_px"])
    return res


# ------------------------------------------------------------------------------------------------ preflight ---
def build_tum_input(name, out):
    import cv2
    src = DATA / "extracted" / f"rgbd_dataset_freiburg1_{name}"
    ts, files = CP.read_rgb_list(src / "rgb.txt")
    sel = CP.select_uniform(len(ts))
    ind = new_dir(out / f"input_{name}")
    (ind / "images").mkdir()
    K = np.array([[FR1["fx"], 0, FR1["cx"]], [0, FR1["fy"], FR1["cy"]], [0, 0, 1.0]])
    dist = np.array(FR1["dist"])
    frames, bad = [], []
    for k in sel:
        im = cv2.imread(str(src / files[k]), cv2.IMREAD_COLOR)
        if im is None or im.shape[:2] != (FR1["height"], FR1["width"]):
            bad.append({"frame_id": k, "file": files[k], "reason": "unreadable or wrong size"})
            continue
        und = cv2.undistort(im, K, dist, None, K)
        nm = f"f{k:04d}_{ts[k]:.6f}.png"
        cv2.imwrite(str(ind / "images" / nm), und)
        frames.append({"frame_id": k, "timestamp": float(ts[k]), "image": nm, "source": files[k]})
    json.dump(frames, open(ind / "frames.json", "w"), indent=1)
    json.dump({"model": "PINHOLE", "width": FR1["width"], "height": FR1["height"], "fx": FR1["fx"], "fy": FR1["fy"], "cx": FR1["cx"], "cy": FR1["cy"],
               "pixel_convention": "OpenCV (integer = pixel centre); COLMAP cx = cx + 0.5",
               "undistortion": "cv2.undistort with the published fr1 RGB d0..d4 (OpenCV k1 k2 p1 p2 k3), new camera matrix = K",
               "source": "TUM file_formats page, 'Freiburg 1 RGB' row"}, open(ind / "camera.json", "w"), indent=1)
    man = {"selected_frame_ids": sel, "n_total_rgb": len(ts), "unreadable": bad,
           "rule": "uniform: round(linspace(0, N-1, 300)) over rgb.txt order; an unreadable/corrupt selected frame is kept in the "
                   "denominator as INVALID and never replaced",
           "files_sha256": {f["image"]: sha(ind / "images" / f["image"]) for f in frames},
           "source_files_sha256": {f["source"]: sha(src / f["source"]) for f in frames}}
    json.dump(man, open(ind / "input_manifest.json", "w"), indent=1)
    return ind, man


def stage_preflight(a):
    out = new_dir(a.out_dir)
    rep = {"stage": "preflight", "time": now(), "branch": git("rev-parse", "--abbrev-ref", "HEAD"), "head": git("rev-parse", "HEAD"),
           "status_porcelain": git("status", "--porcelain").splitlines(), "steps": []}
    rep["branch_ok"] = rep["branch"] == BRANCH
    ds = {}
    for nm, d in TUM.items():
        arc = DATA / "archives" / d["archive"]
        src = DATA / "extracted" / f"rgbd_dataset_freiburg1_{nm}"
        ds[nm] = {"url": d["url"], "archive_sha256": sha(arc), "archive_bytes": arc.stat().st_size, "expected_sha256": d["sha256"],
                  "rgb_txt_sha256": sha(src / "rgb.txt"), "groundtruth_txt_sha256": sha(src / "groundtruth.txt"),
                  "n_rgb": len(CP.read_rgb_list(src / "rgb.txt")[0]), "modalities_present": sorted(p.name for p in src.iterdir())}
        ds[nm]["ok"] = ds[nm]["archive_sha256"] == d["sha256"]
    rep["datasets"] = ds
    ff, ov = page_text(DATA / "docs" / "rgbd-dataset_file_formats.html"), page_text(DATA / "docs" / "rgbd-dataset.html")
    rep["docs_sha256"] = {p.name: sha(p) for p in (DATA / "docs").iterdir()}
    rep["calibration"] = {"published_row_present": CALIB_ROW in ff, "used": FR1, "gt_semantics_clause_present": GT_CLAUSE in ff,
                          "note": "fr1 RGB images are NOT pre-undistorted (only fr3 is, per the page)"}
    rep["rights"] = {"licence_clause_present": LICENCE_CLAUSE in ov, "licence": "CC BY 4.0 (TUM RGB-D benchmark data)",
                     "use": "local research evaluation; datasets kept in scratch, never committed; attribution in the Worklog"}
    for nm in TUM:
        ind, man = build_tum_input(nm, out)
        rep[f"input_{nm}"] = {"dir": str(ind), "n_selected": len(man["selected_frame_ids"]), "unreadable": man["unreadable"],
                              "manifest_sha256": sha(ind / "input_manifest.json"), "frames_sha256": sha(ind / "frames.json")}
    ind = new_dir(out / "input_synthetic")
    shutil.copytree(SYN_DIR / "images", ind / "images")
    shutil.copyfile(SYN_DIR / "input" / "camera.json", ind / "camera.json")
    shutil.copyfile(SYN_DIR / "input" / "frames.json", ind / "frames.json")
    json.dump({"files_sha256": {p.name: sha(p) for p in sorted((ind / "images").iterdir())}, "source": str(SYN_DIR),
               "selected_frame_ids": list(range(len(load(ind / "frames.json"))))}, open(ind / "input_manifest.json", "w"), indent=1)
    rep["input_synthetic"] = {"dir": str(ind), "n_selected": len(load(ind / "frames.json")), "manifest_sha256": sha(ind / "input_manifest.json"),
                              "scene_blend_sha256": sha(SYN_DIR / "scene.blend"), "gt_sha256": sha(SYN_DIR / "gt" / "camera_gt.txt"),
                              "generator_script_lf_sha256": lf_sha("scripts/w140_synthetic_scene.py")}
    rep["oracle_contract"] = oracle_contract(SYN_DIR)
    envr = subprocess.run([str(PC_PY), "-I", "-c", "import pycolmap,numpy,cv2,sys;print(pycolmap.__version__,numpy.__version__,cv2.__version__,sys.version.split()[0])"],
                          capture_output=True, text=True).stdout.split()
    rep["estimator_env"] = {"python_exe": str(PC_PY), "pycolmap": envr[0] if envr else None, "numpy": envr[1] if len(envr) > 1 else None,
                            "opencv": envr[2] if len(envr) > 2 else None, "python": envr[3] if len(envr) > 3 else None,
                            "freeze_sha256": sha(W140 / "env" / "freeze.txt")}
    du = shutil.disk_usage(str(W140))
    rep["resources"] = {"disk_free_gb": round(du.free / 1e9, 1)}
    ok = (rep["branch_ok"] and all(v["ok"] for v in ds.values()) and rep["calibration"]["published_row_present"] and rep["calibration"]["gt_semantics_clause_present"]
          and rep["rights"]["licence_clause_present"] and rep["oracle_contract"]["ok"] and rep["estimator_env"]["pycolmap"] == "4.2.1")
    rep["status"] = "OK" if ok else "BLOCKED"
    CP.dump(out / "preflight.json", rep)
    print(json.dumps({k: rep[k] for k in ("status", "branch", "head", "branch_ok", "calibration", "rights", "oracle_contract", "estimator_env")}
                     | {"datasets_ok": {k: v["ok"] for k, v in ds.items()}, "selected": {k: rep[f"input_{k}"]["n_selected"] for k in DATASETS}}, indent=1, default=str))
    return 0 if ok else 3


# ------------------------------------------------------------------------------------------------ declare ---
def outputs_exist():
    return sorted(str(p) for p in (W140 / "results").rglob("estimate.json")) if (W140 / "results").is_dir() else []


def stage_declare(a):
    out = new_dir(a.out_dir)
    pf = load(a.preflight)
    if pf["status"] != "OK":
        raise SystemExit("preflight not OK")
    if outputs_exist():
        raise SystemExit(f"STOP: estimator output already exists {outputs_exist()}")
    syn_seg = load(SYN_DIR / "gt" / "segments.json")
    d = {"schema": "animcv_exp_w140_declaration_v1", "time": now(), "estimates_exist": False, "gt_scores_seen": False,
         "branch": pf["branch"], "repo_head_at_declaration": git("rev-parse", "HEAD"), "preflight_sha256": sha(a.preflight),
         "datasets": pf["datasets"], "inputs": {k: pf[f"input_{k}"] for k in DATASETS},
         "estimator": {"method": "COLMAP incremental Structure-from-Motion (pycolmap 4.2.1, CPU)", "env": pf["estimator_env"],
                       "config": "SIFT max 8192 features; sequential pairing overlap 10 quadratic, no loop detection; single PINHOLE camera with "
                                 "the given intrinsics, NOT refined; mapper defaults incl. COLMAP's own init relaxation; random_seed 0; 8 threads; "
                                 "multiple models allowed; primary = largest model",
                       "inputs_only": ["RGB PNG frames (TUM: undistorted with the published fr1 parameters)", "frame timestamps (original)",
                                       "published intrinsics"],
                       "never_inputs": ["groundtruth.txt / synthetic camera_gt.txt", "motion capture", "depth", "accelerometer/IMU",
                                        "human pose", "GT-derived initial poses, scale or alignment"],
                       "leak_guard": "the runner refuses any input entry other than images/*.png, frames.json, camera.json, input_manifest.json",
                       "determinism": "random_seed 0 but multi-threaded mapping is not guaranteed bit-reproducible; exactly one run per dataset",
                       "status_semantics": {"ESTIMATED": "registered in the primary (largest) model", "INVALID": "registered only in a secondary "
                                            "model (other gauge) or unreadable input", "LOST": "never registered",
                                            "PREDICTED_INTERPOLATED": "never produced (no interpolation)", "UNOBSERVABLE": "not used by this estimator"}},
         "conventions": CP.__doc__,
         "selection": {"rule": "TUM: round(linspace(0, N-1, 300)) over rgb.txt order, chosen without GT or estimator output; synthetic: all 140 frames",
                       "xyz_ids": load(Path(pf["input_xyz"]["dir"]) / "input_manifest.json")["selected_frame_ids"],
                       "rpy_ids": load(Path(pf["input_rpy"]["dir"]) / "input_manifest.json")["selected_frame_ids"],
                       "missing_or_corrupt_rule": "kept in the denominator as INVALID, never replaced"},
         "gt_association": {"rule": f"GT pose at each estimated frame's timestamp: linear position + SLERP rotation between the two bracketing GT "
                                    f"samples, only if their gap <= {CP.GT_MAX_BRACKET_GAP_S} s; otherwise unassociated (reported); no index matching",
                            "gt_access": "evaluate stage only, after estimate.json files are frozen by SHA-256"},
         "alignment": {"primary": "full-trajectory Umeyama Sim(3) on camera centres (evaluation only; reported scale/R/t)",
                       "secondary": f"Sim(3) fitted on the first {int(CP.EARLY_FRACTION * 100)} % (by time) of associated estimated frames, "
                                    "applied unchanged to the remaining frames (drift)",
                       "statement": "neither alignment is estimation or online localisation; ATE after Sim(3) is not evidence of metric scale"},
         "metrics": ["coverage (ESTIMATED / selected)", "ATE RMSE/median/max after Sim(3)", "RPE consecutive and ~1 s (translation scaled by the "
                     "Sim(3) scale; rotation)", "rotation geodesic error median/p90/max", "early-alignment ATE/rotation on the rest (drift)",
                     "error vs time", "temporal increments + spikes (> 5x median)", "best time lag (expect 0)", "stationary baseline (Control A: "
                     "best constant pose)", "feature/inlier/reprojection/track diagnostics", "parallax proxy per frame (baseline/median depth)",
                     "synthetic per-segment: stationary spurious motion, translation-interval rotation error, rotation-interval spurious translation"],
         "controls": {"A": "stationary best-constant-pose baseline (GT mean centre + chordal-mean GT rotation: most favourable to the baseline)",
                      "B": "the RGB-only estimator", "C": f"synthetic oracle projection contract (preflight): {pf['oracle_contract']}"},
         "gates": {"synthetic": {"contracts": "oracle contract ok + focused tests pass", "stationary": f"aligned max centre spread <= {CP.SYN_STATIONARY_MAX_REL:.0%} "
                                 f"of GT extent and max rotation spread <= {CP.SYN_STATIONARY_MAX_ROT_DEG} deg on frames {syn_seg['stationary']}",
                                 "recovery": f"aligned ATE RMSE <= {CP.SYN_ATE_MAX_REL:.0%} of GT extent, median rotation error <= {CP.SYN_MED_ROT_DEG} deg, "
                                 f"coverage >= {CP.SYN_COVERAGE:.0%}, best time lag 0"},
                   "xyz": f"coverage >= {CP.GATE_COVERAGE:.0%} (denominator = all 300 selected) AND Sim(3) ATE RMSE < {CP.GATE_XYZ_ATE_M} m AND ATE < "
                          "the stationary baseline's",
                   "rpy": f"coverage >= {CP.GATE_COVERAGE:.0%} AND median aligned rotation error < {CP.GATE_RPY_MED_ROT_DEG} deg AND < the stationary baseline's",
                   "classification": "SUPPORTED: synthetic + xyz + rpy pass; PARTIALLY_SUPPORTED: some pass; NOT_YET_SUPPORTED: none; BLOCKED: a "
                                     "prerequisite fails. Pilot thresholds, not Production tolerances; never changed after results"},
         "synthetic_segments": syn_seg,
         "budget": {"estimator_wall_h_max": 2.0, "scratch_gb_max": 8.0, "real_frames_max_per_sequence": 300, "training": "none"},
         "pre_declaration_disclosures": [
             "synthetic scene v1 was re-rendered as v2 before any estimation because its UV mapping tiled each texture ~12x on walls "
             "(repetitive pattern); v2 maps one texture copy per face; v1 kept on disk",
             "dev smoke runs of the estimator on the first 25 and first 70 synthetic frames (C:/Users/dna10/animcv-w140/dev) to exercise "
             "code paths: 25 frames (stationary + tiny translation) gave no initial pair (all LOST, expected degeneracy); 70 frames registered "
             "70/70 with 0.16 px mean reprojection error; NO GT-based score was computed or seen; configuration unchanged afterwards"],
         "instrument_sha256_lf": {f: lf_sha(f) for f in INSTRUMENTS}}
    h = CP.dump(out / "declaration.json", d)
    print(json.dumps({"declaration_sha256": sha(out / "declaration.json"), "content_sha": h}, indent=1))
    return 0


# ------------------------------------------------------------------------------------------------ estimate ---
def stage_estimate(a):
    d = check_declaration(a)
    pf = load(a.preflight)
    if sha(a.preflight) != d["preflight_sha256"]:
        raise SystemExit("STOP preflight differs from the declared one")
    out = new_dir(a.out_dir)
    res = {"stage": "estimate", "time": now(), "head": git("rev-parse", "HEAD"), "runs": {}}
    for nm in DATASETS:
        ind = Path(pf[f"input_{nm}"]["dir"])
        if sha(ind / "input_manifest.json") != d["inputs"][nm]["manifest_sha256"]:
            raise SystemExit(f"STOP input {nm} changed")
        t0 = time.perf_counter()
        r = subprocess.run([str(PC_PY), "-I", str(EST), "--input", str(ind), "--out", str(out / nm)], capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        (out / f"{nm}.log").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr, encoding="utf-8")
        res["runs"][nm] = {"rc": r.returncode, "wall_s": time.perf_counter() - t0,
                           "estimate_sha256": sha(out / nm / "estimate.json") if (out / nm / "estimate.json").is_file() else None}
    CP.dump(out / "estimate_result.json", res)
    print(json.dumps(res, indent=1))
    return 0


# ------------------------------------------------------------------------------------------------ evaluate ---
def parallax_proxy(est_json, pts):
    """Per consecutive estimated pair: baseline / median landmark depth (scale-free); low values = low parallax."""
    fr = [f for f in est_json["frames"] if f["status"] == "ESTIMATED"]
    if len(fr) < 2 or len(pts) == 0:
        return []
    P = pts[:, :3]
    out = []
    for f0, f1 in zip(fr[:-1], fr[1:]):
        T0, T1 = np.asarray(f0["T_WC"]), np.asarray(f1["T_WC"])
        b = float(np.linalg.norm(T1[:3, 3] - T0[:3, 3]))
        z = (CP.inv_T(T0)[:3, :3] @ P.T).T + CP.inv_T(T0)[:3, 3]
        zz = z[:, 2][z[:, 2] > 0]
        out.append({"pair": [f0["frame_id"], f1["frame_id"]], "baseline_over_median_depth": b / float(np.median(zz)) if len(zz) else None})
    return out


def stage_evaluate(a):
    d = check_declaration(a)
    er = load(Path(a.estimate_dir) / "estimate_result.json")
    out = new_dir(a.out_dir)
    res = {"stage": "evaluate", "time": now(), "datasets": {}}
    gts = {"synthetic": SYN_DIR / "gt" / "camera_gt.txt", "xyz": DATA / "extracted" / "rgbd_dataset_freiburg1_xyz" / "groundtruth.txt",
           "rpy": DATA / "extracted" / "rgbd_dataset_freiburg1_rpy" / "groundtruth.txt"}
    for nm in DATASETS:
        ep = Path(a.estimate_dir) / nm / "estimate.json"
        if not ep.is_file() or sha(ep) != er["runs"][nm]["estimate_sha256"]:
            res["datasets"][nm] = {"uncomputable": "estimate missing or changed after freezing"}
            continue
        e = load(ep)
        gt_sha = sha(gts[nm])
        want = d["inputs"]["synthetic"]["gt_sha256"] if nm == "synthetic" else d["datasets"][nm]["groundtruth_txt_sha256"]
        if gt_sha != want:
            raise SystemExit(f"STOP GT identity {nm}")
        gt = CP.read_groundtruth(gts[nm])
        ev = CP.evaluate(e["frames"], gt)
        pts = np.load(Path(a.estimate_dir) / nm / "points3D.npy")
        ev["estimator_diagnostics"] = {"models": e["models"], "primary_model_id": e["primary_model_id"], "timing_s": e["timing_s"],
                                       "n_points3D": int(len(pts)), "mean_track_length": float(pts[:, 7].mean()) if len(pts) else None,
                                       "mean_point_error_px": float(pts[:, 6].mean()) if len(pts) else None,
                                       "keypoints_median": float(np.median([f["n_keypoints"] for f in e["frames"]])),
                                       "observed_points_median": float(np.median([f.get("n_points3D_observed") or 0 for f in e["frames"] if f["status"] == "ESTIMATED"]))
                                       if ev["status_counts"]["ESTIMATED"] else None,
                                       "consecutive_inlier_ratio_median": float(np.median([p["inlier_ratio"] for p in e["pair_stats_consecutive"] if p["inlier_ratio"] is not None]))
                                       if any(p["inlier_ratio"] is not None for p in e["pair_stats_consecutive"]) else None,
                                       "consecutive_inliers_min": min((p["inliers"] for p in e["pair_stats_consecutive"]), default=None),
                                       "lost_frame_ids": [f["frame_id"] for f in e["frames"] if f["status"] == "LOST"],
                                       "invalid_frame_ids": [f["frame_id"] for f in e["frames"] if f["status"] == "INVALID"]}
        ev["parallax"] = parallax_proxy(e, pts)
        if nm == "synthetic":
            seg = d["synthetic_segments"]
            ids = ev.get("frame_ids", [])
            segs = {}
            if "full_sim3" in ev:
                Pa, Pg = np.asarray(ev["aligned_positions"]), np.asarray(ev["gt_positions"])
                Ra, Rg = np.asarray(ev["aligned_rotations"]), np.asarray(ev["gt_rotations"])
                for s, (lo, hi) in seg.items():
                    ix = [i for i, f in enumerate(ids) if lo <= f <= hi]
                    if len(ix) < 2:
                        segs[s] = {"n": len(ix)}
                        continue
                    est_dt = float(np.linalg.norm(Pa[ix[-1]] - Pa[ix[0]]))
                    gt_dt = float(np.linalg.norm(Pg[ix[-1]] - Pg[ix[0]]))
                    segs[s] = {"n": len(ix), "ate_rmse": float(np.sqrt(np.mean(np.sum((Pa[ix] - Pg[ix]) ** 2, 1)))),
                               "rot_err_median_deg": float(np.median(CP.geodesic_deg(Ra[ix], Rg[ix]))),
                               "est_net_translation": est_dt, "gt_net_translation": gt_dt,
                               "est_path_length": float(np.sum(np.linalg.norm(np.diff(Pa[ix], axis=0), axis=1))),
                               "gt_path_length": float(np.sum(np.linalg.norm(np.diff(Pg[ix], axis=0), axis=1))),
                               "est_rotation_range_deg": float(CP.geodesic_deg(Ra[ix][:, None], Ra[ix][None]).max()),
                               "gt_rotation_range_deg": float(CP.geodesic_deg(Rg[ix][:, None], Rg[ix][None]).max())}
            ev["segments"] = segs
            st = list(range(seg["stationary"][0], seg["stationary"][1] + 1))
            ev["gates"] = CP.synthetic_gates(ev, st, contracts_ok=load(a.preflight)["oracle_contract"]["ok"])
        else:
            ev["gates"] = CP.real_gates(nm, ev)
        res["datasets"][nm] = ev
    res["judgment"] = CP.judgment(res["datasets"].get("synthetic", {}).get("gates", {}).get("pass", False),
                                  res["datasets"].get("xyz", {}).get("gates", {}).get("pass", False),
                                  res["datasets"].get("rpy", {}).get("gates", {}).get("pass", False))
    CP.dump(out / "evaluation.json", res)
    summ = {nm: {k: v for k, v in (ev.get("gates") or {}).items()} for nm, ev in res["datasets"].items()}
    for nm, ev in res["datasets"].items():
        if "full_sim3" in ev:
            f = ev["full_sim3"]
            summ[nm].update({"sim3_scale": f["sim3"]["scale"], "ate_rmse": f["ate"]["rmse"], "ate_median": f["ate"]["median"],
                             "rot_median": f["rot"]["median_deg"], "rot_p90": f["rot"]["p90_deg"], "rpe": f["rpe"],
                             "early_ate_rest": (ev["early_sim3"].get("ate_rest") or {}).get("rmse"),
                             "early_rot_rest_median": (ev["early_sim3"].get("rot_rest") or {}).get("median_deg"),
                             "baseline": {k: ev["baseline_stationary"][k] for k in ("ate_rmse", "rot_median_deg")},
                             "gt_extent_m": ev["gt_extent_m"], "gt_rot_range_deg": ev["gt_rotation_range_deg"], "status": ev["status_counts"],
                             "time_lag": f["time_lag"], "spikes": {"t": len(ev["temporal_est_scaled"]["trans_spikes"]), "r": len(ev["temporal_est_scaled"]["rot_spikes"])}})
    print(json.dumps({"summary": summ, "judgment": res["judgment"]}, indent=1, default=str))
    return 0


# ------------------------------------------------------------------------------------------------ visualize ---
def stage_visualize(a):
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    check_declaration(a)
    ev = load(Path(a.eval_dir) / "evaluation.json")
    pf = load(a.preflight)
    out = new_dir(a.out_dir)
    made = []
    for nm, e in ev["datasets"].items():
        if "full_sim3" not in e:
            continue
        Pa, Pg = np.asarray(e["aligned_positions"]), np.asarray(e["gt_positions"])
        t = np.asarray(e["times"]) - e["times"][0]
        fig = plt.figure(figsize=(14, 9))
        ax = fig.add_subplot(2, 3, 1, projection="3d")
        ax.plot(*Pg.T, "k-", label="GT")
        ax.plot(*Pa.T, "r-", label="ALIGNED ESTIMATE (Sim3, eval only)")
        ax.set_title(f"{nm}: 3D camera centres")
        ax.legend(fontsize=7)
        for k, (i, j) in enumerate(((0, 1), (0, 2))):
            axx = fig.add_subplot(2, 3, 2 + k)
            axx.plot(Pg[:, i], Pg[:, j], "k-", label="GT")
            axx.plot(Pa[:, i], Pa[:, j], "r-", label="ALIGNED ESTIMATE")
            axx.set_xlabel("xyz"[i])
            axx.set_ylabel("xyz"[j])
            axx.axis("equal")
            axx.legend(fontsize=7)
        ax4 = fig.add_subplot(2, 3, 4)
        ax4.plot(t, e["full_sim3"]["ate"]["per_frame"], "r-", label="position error (full Sim3)")
        if e["early_sim3"].get("ate_rest"):
            k = e["early_sim3"]["n_fit"]
            ax4.plot(t[k:], e["early_sim3"]["ate_rest"]["per_frame"], "b-", label="position error (early-segment Sim3)")
        ax4.set_xlabel("time [s]")
        ax4.set_ylabel("error [GT units]")
        ax4.legend(fontsize=7)
        ax5 = fig.add_subplot(2, 3, 5)
        ax5.plot(t, e["full_sim3"]["rot"]["per_frame"], "r-", label="rotation error (full Sim3)")
        if e["early_sim3"].get("rot_rest"):
            ax5.plot(t[k:], e["early_sim3"]["rot_rest"]["per_frame"], "b-", label="rotation error (early-segment Sim3)")
        ax5.set_xlabel("time [s]")
        ax5.set_ylabel("deg")
        ax5.legend(fontsize=7)
        ax6 = fig.add_subplot(2, 3, 6)
        te = e["temporal_est_scaled"]
        ax6.plot(te["trans_steps"], "r-", label="ESTIMATE step (scaled)")
        ax6.plot(e["temporal_gt"]["trans_steps"], "k-", label="GT step")
        ax6.set_ylabel("frame-to-frame translation")
        ax6.legend(fontsize=7)
        sc = e["status_counts"]
        fig.suptitle(f"W140 {nm}: ESTIMATED {sc['ESTIMATED']} / LOST {sc['LOST']} / INVALID {sc['INVALID']} / INTERPOLATED {sc['PREDICTED_INTERPOLATED']}"
                     f" | Sim3 scale {e['full_sim3']['sim3']['scale']:.4f} (eval only; SCALE_AMBIGUOUS)")
        fig.tight_layout()
        p = out / f"{nm}_trajectory_errors.png"
        fig.savefig(p, dpi=110)
        plt.close(fig)
        made.append(p)
    # feature-track overlays + failure examples + synchronized video (real sequences)
    for nm in ("xyz", "rpy", "synthetic"):
        e = ev["datasets"].get(nm, {})
        ind = Path(pf[f"input_{nm}"]["dir"])
        est = load(Path(a.estimate_dir) / nm / "estimate.json")
        txt = Path(a.estimate_dir) / nm / "primary_model_txt" / "images.txt"
        obs = {}
        if txt.is_file():
            lines = [l for l in txt.read_text().splitlines() if not l.startswith("#")]
            for i in range(0, len(lines) - 1, 2):
                hdr, pts = lines[i].split(), lines[i + 1].split()
                arr = np.asarray(pts, float).reshape(-1, 3) if pts else np.zeros((0, 3))
                obs[hdr[9]] = arr[arr[:, 2] >= 0][:, :2] - 0.5                       # COLMAP corner origin -> OpenCV centres
        frames = est["frames"]
        st = {f["image"]: f["status"] for f in frames}
        pick = [frames[i] for i in np.linspace(0, len(frames) - 1, 4).astype(int)]
        lost = [f for f in frames if f["status"] != "ESTIMATED"][:4]
        for tag, sel in (("tracks", pick), ("failures", lost)):
            tiles = []
            for f in sel:
                im = cv2.imread(str(ind / "images" / f["image"]))
                for u, v in obs.get(f["image"], [])[::3]:
                    cv2.circle(im, (int(round(u)), int(round(v))), 2, (0, 255, 0), -1)
                lab = st[f["image"]] if st[f["image"]] != "LOST" else "TRACKING LOST"
                cv2.putText(im, f"{nm} frame {f['frame_id']} t={f['timestamp']:.3f} {lab}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
                cv2.putText(im, f"{nm} frame {f['frame_id']} t={f['timestamp']:.3f} {lab}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                tiles.append(im)
            if tiles:
                while len(tiles) % 2:
                    tiles.append(np.zeros_like(tiles[0]))
                grid = np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)])
                p = out / f"{nm}_{tag}.png"
                cv2.imwrite(str(p), grid)
                made.append(p)
        if "full_sim3" not in e:
            continue
        Pa, Pg = np.asarray(e["aligned_positions"]), np.asarray(e["gt_positions"])
        ids = e["frame_ids"]
        lo, hi = np.minimum(Pa.min(0), Pg.min(0)), np.maximum(Pa.max(0), Pg.max(0))
        span = max(hi - lo) * 1.1 + 1e-9
        vid = out / f"{nm}_synchronized_30fps.mp4"
        wr = None
        idx = {fid: i for i, fid in enumerate(ids)}
        for f in frames:
            im = cv2.imread(str(ind / "images" / f["image"]))
            for u, v in obs.get(f["image"], [])[::3]:
                cv2.circle(im, (int(round(u)), int(round(v))), 2, (0, 255, 0), -1)
            panel = np.full((480, 480, 3), 255, np.uint8)

            def to_px(p, a0, a1):
                return int(20 + (p[a0] - lo[a0]) / span * 440), int(460 - (p[a1] - lo[a1]) / span * 440)
            for P, col in ((Pg, (0, 0, 0)), (Pa, (0, 0, 255))):
                for k in range(len(P) - 1):
                    cv2.line(panel, to_px(P[k], 0, 1), to_px(P[k + 1], 0, 1), col, 1)
            if f["frame_id"] in idx:
                i = idx[f["frame_id"]]
                cv2.circle(panel, to_px(Pg[i], 0, 1), 6, (0, 0, 0), 2)
                cv2.circle(panel, to_px(Pa[i], 0, 1), 5, (0, 0, 255), -1)
                lab = f"ESTIMATED  err {e['full_sim3']['ate']['per_frame'][i]:.3f} / {e['full_sim3']['rot']['per_frame'][i]:.1f} deg"
            else:
                lab = "TRACKING LOST" if f["status"] == "LOST" else f["status"]
            cv2.putText(panel, "top view x-y: GT (black) / ALIGNED ESTIMATE (red)", (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
            cv2.putText(panel, lab, (8, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 200) if f["frame_id"] in idx else (0, 120, 255), 1)
            cv2.putText(im, f"{nm} t={f['timestamp']:.3f}s frame {f['frame_id']} (green = model-observed features)", (6, 470),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
            fr = np.hstack([im, panel])
            if wr is None:
                wr = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (fr.shape[1], fr.shape[0]))
            wr.write(fr)
        wr.release()
        made.append(vid)
    ue = ROOT / "user_eval" / "W140_camera_pose"
    ue.mkdir(parents=True, exist_ok=True)
    for p in made:
        shutil.copyfile(p, ue / p.name)
    CP.dump(out / "visualize.json", {"files": {p.name: sha(p) for p in made}, "copied_to": str(ue),
                                     "note": "videos play the selected frames at 30 fps for inspection (not the original timing)"})
    print(json.dumps({"files": [p.name for p in made], "user_eval": str(ue)}, indent=1))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=("preflight", "declare", "estimate", "evaluate", "visualize"))
    ap.add_argument("--out-dir", required=True)
    for k in ("preflight", "declaration", "declaration-sha", "estimate-dir", "eval-dir"):
        ap.add_argument("--" + k)
    a = ap.parse_args()
    return {"preflight": stage_preflight, "declare": stage_declare, "estimate": stage_estimate, "evaluate": stage_evaluate,
            "visualize": stage_visualize}[a.stage](a)


if __name__ == "__main__":
    sys.exit(main())
