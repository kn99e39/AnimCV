#!/usr/bin/env python3
"""W143 (EXPERIMENTAL, opt-in, scratch only): Decoupled Camera Orientation and Translation Observability harness.

Candidate: scripts/w143_decoupled_estimate.py (app.experimental_decoupled_camera_state).  Controls: frozen W142 estimator
(historical outputs; one new frozen-W142 run on the held-out sequence), frozen W140/W141 SfM, constant-camera baseline.

Datasets: synthetic A1-A3 (W142 renders/masks) and A4 (W141 synthetic dynamic), walking_static (known) and walking_xyz
(moving control) with the exact W141 inputs/masks, sitting_static (HELD-OUT: new download, W141 selection rule, frozen W141
segmentation), authentic S1 (W142 input/mask, NOMINAL_UNCALIBRATED).

Stages: preflight, declare, estimate, evaluate, visualize (each refuses an existing output directory).

    PY=C:/Projects/AnimCV/.venv_build_full/Scripts/python.exe; export PYTHONPATH=src
    $PY -X utf8 scripts/w143_harness.py <stage> --out-dir <dir> [--preflight P --declaration D --declaration-sha S --estimate-dir E --eval-dir V]
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]
import numpy as np                                                    # noqa: E402

import app.experimental_background_rotation as BR                     # noqa: E402
import app.experimental_camera_pose as CP                             # noqa: E402
import app.experimental_decoupled_camera_state as DC                  # noqa: E402
import app.experimental_dynamic_camera_pose as DP                     # noqa: E402

W143 = Path("C:/Users/dna10/animcv-w143")
W142 = Path("C:/Users/dna10/animcv-w142")
W141 = Path("C:/Users/dna10/animcv-w141")
W140 = Path("C:/Users/dna10/animcv-w140")
PC_PY = W140 / "env" / "pc311" / "Scripts" / "python.exe"
EST143 = ROOT / "scripts" / "w143_decoupled_estimate.py"
EST142 = ROOT / "scripts" / "w142_background_rotation_estimate.py"
EST141 = ROOT / "scripts" / "w141_colmap_masked_estimate.py"
SEG141 = ROOT / "scripts" / "w141_segment_people.py"
BRANCH = "Dual_Stage_Generator"
SIT = {"archive": "rgbd_dataset_freiburg3_sitting_static.tgz", "bytes": 443506009,
       "url": "https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_sitting_static.tgz",
       "last_modified": "Tue, 07 Aug 2012 18:50:04 GMT",
       "clauses": ["Two persons sit at a desk, talk, and gesticulate a little bit. The Asus Xtion sensor has been kept in place manually.",
                   "Ground-truth trajectory length: 0.259m"]}
FR3 = {"fx": 535.4, "fy": 539.2, "cx": 320.1, "cy": 247.6, "width": 640, "height": 480}
CALIB_ROW = "Freiburg 3 RGB 535.4 539.2 320.1 247.6 0 0 0 0 0"
LICENCE_CLAUSE = "licensed under a Creative Commons 4.0 Attribution License (CC BY 4.0)"
DOCS = W141 / "data" / "docs"
W142_DECL = ROOT / "examples" / "experimental_camera_pose" / "w142_declaration.json"
W142_DECL_SHA = "fb0a4d282f06b915f0547d7351517df77dab47e2a1fcf8f95e05e2caae113817"
W142_IDS = {"evaluation.json": ("results/evaluate/evaluation.json", "a38215714d05c2cc0fdd27465efb388c3bc1700c4efcb77802165933daff2f80"),
            "estimate_result.json": ("results/estimate/estimate_result.json", "0f438f5b0c57960aca6a046b401141f89ae857cec5ff88dd1af318f35a783080"),
            "routing.json": ("results/estimate/routing.json", "b2c94ddf316815f200e912b3056c101292001cad5c831a1df50eed5b55c9fb46"),
            "evidence_manifest.json": ("results/evidence_manifest.json", "bba4ee3eaed233a57e8d1fa31af1f0b218a270186fab673b0e4f79132115da84"),
            "posthoc.json": ("results/posthoc/posthoc.json", "c83ec3ed224a568d4b1a2c2f520a37455146d643b314725daad0b97a370147a2")}
W141_EVAL_SHA = "c370522e244bd0d0af7081becdeeb811c27b8c9620e40003ed2edb6767d294e2"
W140_EVAL_SHA = "3bb32caa90e76059e19a2cd8f95eb9ccb994e059754b07963a56f30d7e6fb953"
INSTRUMENTS = ["src/app/experimental_decoupled_camera_state.py", "scripts/w143_decoupled_estimate.py", "scripts/w143_harness.py",
               "scripts/w143_mechanism.py", "tests/test_w143_decoupled_camera_state.py", "src/app/experimental_background_rotation.py",
               "scripts/w142_background_rotation_estimate.py", "scripts/w142_harness.py", "src/app/experimental_camera_pose.py",
               "src/app/experimental_dynamic_camera_pose.py", "scripts/w140_colmap_estimate.py", "scripts/w141_colmap_masked_estimate.py",
               "scripts/w141_segment_people.py"]
DATASETS = ("syn_A1", "syn_A2", "syn_A3", "syn_A4", "walking_static", "walking_xyz", "sitting_static", "s1")
GT_DATASETS = DATASETS[:-1]
# SfM source per dataset for the 6DoF availability decision (frozen estimator outputs; never GT)
SFM_SRC = {"syn_A1": W142 / "results/estimate/sfm_syn_A1_rgbmask/estimate.json", "syn_A2": W142 / "results/estimate/sfm_syn_A2_rgbmask/estimate.json",
           "syn_A3": W142 / "results/estimate/sfm_syn_A3_rgbmask/estimate.json", "syn_A4": W141 / "results/estimate/syn_rgbmask/estimate.json",
           "walking_static": W141 / "results/estimate/walking_static_B/estimate.json", "walking_xyz": W141 / "results/estimate/walking_xyz_B/estimate.json",
           "sitting_static": "NEW:sfm_sitting_static_rgbmask", "s1": "NEW:sfm_s1_rgbmask"}
W142_HIST = {d: W142 / f"results/estimate/{d}_rgbmask/estimate.json" for d in ("syn_A1", "syn_A2", "syn_A3", "syn_A4", "walking_static", "walking_xyz", "s1")}
# official executions: 8 W143 candidate + 1 frozen W142 (held-out) + 2 frozen SfM
RUNS143 = [(f"w143_{d}", d) for d in DATASETS]
G = {"syn_cov": 0.80, "syn_A1_max_deg": 0.5, "syn_A2_med_deg": 1.0, "syn_A3_med_deg": 1.0, "syn_A4_med_deg": 1.0, "syn_false_dir_frac": 0.01,
     "syn_A3_dir_err_deg": 30.0, "syn_A4_seg_dir_frac": 0.5, "real_cov": 0.80, "real_med_deg": 3.0, "nondeterioration_deg": 0.25,
     "sit_meaningful_rot_deg": 1.0, "sit_false_dir_disp_m": 0.01, "sit_false_dir_frac": 0.10, "xyz_full6dof_frac": 0.95,
     "xyz_near_static_frac": 0.20, "notyet_cov_gain": 0.20, "spike_deg": 2.0}
PARALLEL = 3


def _h42():
    spec = importlib.util.spec_from_file_location("w142h", ROOT / "scripts" / "w142_harness.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


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


def run(cmd, log):
    t0 = time.perf_counter()
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, encoding="utf-8", errors="replace")
    Path(log).write_text(r.stdout + "\n--- stderr ---\n" + r.stderr, encoding="utf-8")
    return r.returncode, time.perf_counter() - t0


def dir_sha(d):
    return hashlib.sha256("".join(sha(p) for p in sorted(Path(d).iterdir())).encode()).hexdigest()


def check_declaration(a):
    if sha(a.declaration) != a.declaration_sha:
        raise SystemExit("STOP INTEGRITY_FAILED: declaration changed")
    d = load(a.declaration)
    bad = [f for f, h in d["instrument_sha256_lf"].items() if lf_sha(f) != h]
    if bad:
        raise SystemExit(f"STOP DECLARATION_INVALID: instruments changed {bad}")
    return d


# ------------------------------------------------------------------------------------------------ preflight ---
def stage_preflight(a):
    import cv2
    out = new_dir(a.out_dir)
    rep = {"stage": "preflight", "time": now(), "branch": git("rev-parse", "--abbrev-ref", "HEAD"), "head": git("rev-parse", "HEAD"),
           "status_porcelain": git("status", "--porcelain").splitlines()}
    rep["branch_ok"] = rep["branch"] == BRANCH
    d142 = load(W142_DECL)
    hist = {"w142_declaration_ok": sha(W142_DECL) == W142_DECL_SHA,
            "w142": {k: sha(W142 / p) == h for k, (p, h) in W142_IDS.items()},
            "w142_instruments_unchanged": {f: lf_sha(f) == h for f, h in d142["instrument_sha256_lf"].items()},
            "w141_evaluation_ok": sha(W141 / "results/evaluate/evaluation.json") == W141_EVAL_SHA,
            "w140_evaluation_ok": sha(W140 / "results/evaluate/evaluation.json") == W140_EVAL_SHA}
    er142 = load(W142 / "results/estimate/estimate_result.json")
    hist["w142_estimates_frozen"] = {n: sha(W142 / "results/estimate" / n / "estimate.json") == v["estimate_sha256"]
                                     for n, v in {**er142["runs"], **er142["sfm_runs"]}.items()}
    hist["ok"] = bool(hist["w142_declaration_ok"] and all(hist["w142"].values()) and all(hist["w142_instruments_unchanged"].values())
                      and hist["w141_evaluation_ok"] and hist["w140_evaluation_ok"] and all(hist["w142_estimates_frozen"].values()))
    rep["historical"] = hist
    pf142 = load(W142 / "results/preflight/preflight.json")
    inputs, masks = {}, {}
    d141 = load(ROOT / "examples" / "experimental_camera_pose" / "w141_declaration.json")
    for dsn in ("syn_A1", "syn_A2", "syn_A3", "syn_A4", "walking_static", "walking_xyz", "s1"):
        ind = Path(pf142["inputs"][dsn]["dir"])
        mk = Path(pf142["masks"][dsn]["dirs"]["rgbmask"])
        if dsn in ("walking_static", "walking_xyz", "syn_A4"):
            want_mk = d141["masks"]["syn_dynamic" if dsn == "syn_A4" else dsn]["keep_dirs_sha256"]["rgbmask"]
        else:
            want_mk = d142["masks"][dsn]["summary"]["keep_dirs_sha256"]["rgbmask"]
        inputs[dsn] = {"dir": str(ind), "manifest_sha256": sha(ind / "input_manifest.json"),
                       "ok": sha(ind / "input_manifest.json") == d142["inputs"][dsn]["manifest_sha256"], "gt": pf142["inputs"][dsn].get("gt"),
                       "source": "W142 preflight record (W141 inputs for real data and A4)"}
        masks[dsn] = {"dir": str(mk), "dir_sha256": dir_sha(mk), "ok": dir_sha(mk) == want_mk, "declared_sha256": want_mk}
    # ---- held-out sitting_static: identity, rights, calibration, selection, inputs, masks (GT only hashed) ----
    src = W143 / "data" / "extracted" / "rgbd_dataset_freiburg3_sitting_static"
    arc = W143 / "data" / "archives" / SIT["archive"]
    import html
    import re

    def page(n):
        t = html.unescape(re.sub(r"<[^>]+>", " ", (DOCS / n).read_text(encoding="utf-8", errors="replace")))
        return re.sub(r"\s+", " ", t)
    dl, ff, ov = page("rgbd-dataset_download.html"), page("rgbd-dataset_file_formats.html"), page("rgbd-dataset.html")
    sit = {"url": SIT["url"], "archive_sha256": sha(arc), "archive_bytes": arc.stat().st_size, "expected_bytes": SIT["bytes"],
           "server_last_modified": SIT["last_modified"], "rgb_txt_sha256": sha(src / "rgb.txt"), "groundtruth_txt_sha256": sha(src / "groundtruth.txt"),
           "extracted_entries": sorted(p.name for p in src.iterdir()), "clauses_present": {c: c in dl for c in SIT["clauses"]},
           "calibration_row_present": CALIB_ROW in ff, "licence_clause_present": LICENCE_CLAUSE in ov,
           "docs": "official TUM pages saved in W141 scratch (sha in the W141 preflight)",
           "authentication": "official HTTPS URL; byte size equals the server Content-Length; downloaded SHA-256 recorded as identity",
           "gt_access": "only hashed here; read exclusively by the evaluate stage after all W143 outputs are frozen"}
    sit["ok"] = bool(sit["archive_bytes"] == SIT["bytes"] and all(sit["clauses_present"].values()) and sit["calibration_row_present"]
                     and sit["licence_clause_present"] and "depth" not in sit["extracted_entries"])
    rep["sitting_static"] = sit
    ts, files = CP.read_rgb_list(src / "rgb.txt")
    sel = CP.select_uniform(len(ts))
    ind = new_dir(out / "input_sitting_static")
    (ind / "images").mkdir()
    frames, bad = [], []
    for k in sel:
        im = cv2.imread(str(src / files[k]), cv2.IMREAD_COLOR)
        if im is None or im.shape[:2] != (480, 640):
            bad.append(k)
            continue
        nm = f"f{k:04d}_{ts[k]:.6f}.png"
        shutil.copyfile(src / files[k], ind / "images" / nm)
        frames.append({"frame_id": k, "timestamp": float(ts[k]), "image": nm, "source": files[k]})
    json.dump(frames, open(ind / "frames.json", "w"), indent=1)
    json.dump({"model": "PINHOLE", **FR3, "pixel_convention": "OpenCV (integer = pixel centre); COLMAP cx = cx + 0.5",
               "undistortion": "none: fr3 colour images are already undistorted (official page)", "source": "TUM 'Freiburg 3 RGB' row"},
              open(ind / "camera.json", "w"), indent=1)
    json.dump({"selected_frame_ids": sel, "n_total_rgb": len(ts), "unreadable": bad, "rule": "round(linspace(0, N-1, 300)) over rgb.txt order (W140-W142)",
               "files_sha256": {f["image"]: sha(ind / "images" / f["image"]) for f in frames}}, open(ind / "input_manifest.json", "w"), indent=1)
    inputs["sitting_static"] = {"dir": str(ind), "manifest_sha256": sha(ind / "input_manifest.json"), "ok": not bad, "n": len(frames),
                                "n_total_rgb": len(ts), "gt": str(src / "groundtruth.txt")}
    md = new_dir(out / "masks" / "sitting_static")
    rc, dt = run([sys.executable, "-I", SEG141, "--images", ind / "images", "--frames", ind / "frames.json", "--out", md / "raw",
                  "--torch-home", W141 / "models"], md / "segment.log")
    if rc != 0:
        raise SystemExit("STOP segmentation")
    (md / "keep_rgbmask").mkdir()
    occ = []
    for f in frames:
        raw = cv2.imread(str(md / "raw" / f["image"]), cv2.IMREAD_UNCHANGED) > 127
        cv2.imwrite(str(md / "keep_rgbmask" / (f["image"] + ".png")), DP.keep_from_exclusion(DP.disk_dilate(raw)))
        occ.append(float(raw.mean()))
    masks["sitting_static"] = {"dir": str(md / "keep_rgbmask"), "dir_sha256": dir_sha(md / "keep_rgbmask"), "ok": True, "segment_s": dt,
                               "raw_frac_mean": float(np.mean(occ)), "raw_frac_median": float(np.median(occ)), "raw_frac_p90": float(np.percentile(occ, 90)),
                               "frames_with_person": int(sum(o > 0 for o in occ)),
                               "mechanism": "frozen W141 segmentation (Mask R-CNN COCO_V1, score>=0.5, mask>=0.5) + 15 px disk dilation"}
    rep["inputs"], rep["masks"] = inputs, masks
    # extraction-only mask contract on the held-out input
    xc = {"kp": 0, "kp_on_excluded": 0}
    for f in frames[:5]:
        img = cv2.imread(str(ind / "images" / f["image"]))
        keep = cv2.imread(str(md / "keep_rgbmask" / (f["image"] + ".png")), cv2.IMREAD_UNCHANGED) > 0
        xy, _, _ = BR.sift_features(img, keep)
        xc["kp"] += len(xy)
        xc["kp_on_excluded"] += int((~BR.on_keep(xy, keep)).sum())
    xc["ok"] = xc["kp_on_excluded"] == 0 and xc["kp"] > 0
    rep["extract_check_sitting_static"] = xc
    mech = W143 / "results" / "mechanism" / "mechanism.json"
    rep["mechanism_sha256"] = sha(mech) if mech.is_file() else None
    rep["resources"] = {"disk_free_gb": round(shutil.disk_usage(str(W143)).free / 1e9, 1),
                        "w143_scratch_gb": round(sum(p.stat().st_size for p in W143.rglob("*") if p.is_file()) / 1e9, 2)}
    ok = (rep["branch_ok"] and hist["ok"] and sit["ok"] and all(v["ok"] for v in inputs.values()) and all(v["ok"] for v in masks.values())
          and xc["ok"] and rep["mechanism_sha256"] is not None)
    rep["status"] = "OK" if ok else "BLOCKED"
    CP.dump(out / "preflight.json", rep)
    print(json.dumps({"status": rep["status"], "historical_ok": hist["ok"], "sitting_static": {k: v for k, v in sit.items() if k not in ("docs",)},
                      "inputs_ok": {k: v["ok"] for k, v in inputs.items()}, "masks_ok": {k: v["ok"] for k, v in masks.items()},
                      "sitting_masks": {k: v for k, v in masks["sitting_static"].items() if k != "dir"}, "extract_check": xc,
                      "n_sitting": len(frames), "resources": rep["resources"]}, indent=1, default=str))
    return 0 if ok else 3


H42 = None


# ------------------------------------------------------------------------------------------------ declare ---
def stage_declare(a):
    out = new_dir(a.out_dir)
    pf = load(a.preflight)
    if pf["status"] != "OK":
        raise SystemExit("preflight not OK")
    if (W143 / "results").is_dir() and [p for p in (W143 / "results").rglob("estimate.json")]:
        raise SystemExit("STOP: W143 estimator output already exists")
    d = {"schema": "animcv_exp_w143_declaration_v1", "time": now(), "estimates_exist": False, "gt_scores_seen_w143": False,
         "branch": pf["branch"], "repo_head_at_declaration": git("rev-parse", "HEAD"), "preflight_sha256": sha(a.preflight),
         "historical": pf["historical"], "inputs": pf["inputs"], "masks": pf["masks"], "sitting_static": pf["sitting_static"],
         "mechanism_investigation_sha256": pf["mechanism_sha256"],
         "candidate": {"module_doc": DC.__doc__, "params": {k: getattr(DC, k) for k in dir(DC) if k.isupper() and not k.endswith("STATES") and k not in ("SCHEMA", "BR", "CP")},
                       "w142_functions_reused_unchanged": "SIFT features, mutual ratio matching, spans, rotation-only RANSAC + Kabsch, homography, "
                                                          "essential + refinement, GRIC, robust chordal rotation averaging, +-0.5 s windows",
                       "may_consume": ["RGB frames", "timestamps", "approved intrinsics", "frozen W141 RGB person keep masks (held-out: same "
                                       "frozen mechanism)", "frozen SfM estimate.json for the 6DoF availability decision"],
                       "never": ["GT", "depth", "IMU", "oracle masks", "W142 post-hoc results"]},
         "relationship_to_w142_posthoc": "The W142 POST-HOC forced rotation-only diagnostic (all rotation-only edges with >= 30 inliers, no "
                                         "validation) is NOT a validated method and is not the W143 candidate. W143 accepts a rotation-only "
                                         "measurement only if it passes support, consensus-residual, homography-agreement and cycle checks, and "
                                         "reports a residual-based bias indicator; translation and 6DoF decisions are separate",
         "mechanism_findings_used_for_design": [
             "synthetic correspondences: GRIC prefers a homography over pure rotation under residual radial distortion, rolling shutter and "
             "planar translation, while intrinsic errors of 1-2 % / 5 px did not flip it; under distortion the rotation-only estimate stays "
             "accurate (0.04 deg) -> homography preference alone does not make a rotation measurement unusable",
             "the W142 cheirality failures at millimetre baselines are genuine essential-decomposition ill-conditioning (2-6 points in front at "
             "5-10 mm; recoverPose distance cut-off ruled out); essential-derived rotation is therefore not used for orientation in W143",
             "translation of 10-30 mm at 1-5 m depth biases the rotation-only estimate by 0.16-0.42 deg and is rejected by the support / "
             "residual criteria; 2 mm is accepted with 0.04 deg error; planar 30 mm translation is rejected (residual) although its bias "
             "indicator (0.17 deg) under-states its error (0.52 deg): the indicator is not a guaranteed bound",
             "GT-free W142 walking_static statistics: 71 % of W142 'planar-ambiguous' edges and 92 % of rotation-only edges pass the support "
             "and residual criteria (W142 did not store the homography-agreement or cycle quantities)"],
         "datasets": {"synthetic": "W142 A1-A3 renders and RGB masks, W141 A4", "known": "walking_static (W141 inputs/masks); W142 outcomes "
                      "already known -> not evidence of generalisation", "moving_control": "walking_xyz", "held_out": "sitting_static: never used "
                      "for design; GT read only after freezing", "s1": "GT-free observation, NOMINAL_UNCALIBRATED"},
         "runs": {"w143": [r[0] for r in RUNS143], "frozen_w142_on_held_out": "w142_sitting_static_rgbmask",
                  "frozen_sfm": ["sfm_sitting_static_rgbmask", "sfm_s1_rgbmask"], "n_official_executions": len(RUNS143) + 3},
         "sfm_sources": {k: str(v) for k, v in SFM_SRC.items()},
         "evaluation": {"orientation": "W142 protocol unchanged: one evaluation-only chordal rotation alignment over the condition's valid "
                                       "associated frames; relative consecutive and ~1 s errors without alignment; constant-camera baseline on the "
                                       "same frames; coverage over all selected frames",
                        "w142_comparison": "W142 (historical; frozen run on sitting_static) vs W143 on the common valid frames, each re-aligned there",
                        "edge_accounting": "W143 accepted / rejected edges cross-tabulated with the W142 state of the same pair; relative-rotation "
                                           "error vs GT per category; bias-indicator calibration (fraction of accepted edges with GT error <= "
                                           "indicator + 0.05 deg)",
                        "translation": "edge direction claims vs GT direction; frames with TRANSLATION_DIRECTION_ONLY vs GT window displacement; "
                                       "no pseudo-ATE",
                        "gt_association": "real: W141 rule incl. duplicate-timestamp removal; synthetic: exact timestamp match"},
         "gates": {"synthetic": f"A1: coverage >= {G['syn_cov']}, max aligned < {G['syn_A1_max_deg']} deg; A2: coverage, median < {G['syn_A2_med_deg']}; "
                                f"A1/A2 DIRECTION_OBSERVED edge fraction <= {G['syn_false_dir_frac']}; A3: coverage, median < {G['syn_A3_med_deg']}, "
                                f"direction claims (if any) median error < {G['syn_A3_dir_err_deg']} deg; A4: coverage, median < {G['syn_A4_med_deg']}, "
                                f"frames 20-69 TRANSLATION_DIRECTION_ONLY >= {G['syn_A4_seg_dir_frac']}; no numeric translation without valid SfM; "
                                "0 keypoints on excluded pixels",
                   "walking_static": f"coverage >= {G['real_cov']}; aligned median < {G['real_med_deg']}; aligned median and 1 s relative median below "
                                     f"the constant baseline; W143 median on W142-common frames <= W142 + {G['nondeterioration_deg']} deg; no numeric "
                                     "translation without valid SfM",
                   "sitting_static (held-out)": f"coverage >= {G['real_cov']}; aligned median < {G['real_med_deg']}; IF GT rotation range on valid frames "
                                                f">= {G['sit_meaningful_rot_deg']} deg then aligned and 1 s medians below the constant baseline; "
                                                f"frames with TRANSLATION_DIRECTION_ONLY whose GT +-0.5 s displacement < {G['sit_false_dir_disp_m']} m "
                                                f"<= {G['sit_false_dir_frac']} of frames; no numeric translation without valid SfM",
                   "walking_xyz": f"FULL_6DOF_AVAILABLE >= {G['xyz_full6dof_frac']} of frames; NEAR_STATIC frames <= {G['xyz_near_static_frac']}; "
                                  "0 SfM-valid frames downgraded",
                   "classification": "SUPPORTED: all four gates pass. NOT_YET_SUPPORTED: walking_static and sitting_static both fail AND W143 "
                                     f"walking_static coverage gain over W142 < {G['notyet_cov_gain']} (or accuracy deteriorates). "
                                     "PARTIALLY_SUPPORTED otherwise. BLOCKED: prerequisite failure. S1 reported, not gated."},
         "gate_values": G,
         "budget": {"primary_estimation_h_max": 2.0, "scratch_gb_max": 10.0, "parallel": PARALLEL, "training": "none",
                    "downloads": "sitting_static archive only"},
         "pre_declaration_disclosures": [
             "Cycle validation design change from a synthetic unit test (no data output): naive median triangle scoring also rejected "
             "innocent partners of one corrupted edge, so cycle validation became greedy worst-first removal with a tie-break preferring the "
             "edge with more triangle evidence.",
             "Phase 1 mechanism script (synthetic correspondences + GT-free historical W142 statistics) was run before this declaration; its "
             "output is bound by SHA-256. No W143 estimator was run on any real or synthetic image set before this declaration.",
             "The known-case (walking_static) W142 GT outcomes, including the POST-HOC forced rotation-only result, were known when the "
             "thresholds were chosen; thresholds were set from the RANSAC threshold (1.5 px), a majority-support rule (0.8) and synthetic "
             "controls, not by searching on walking_static GT. The held-out sitting_static frames, masks and GT were not inspected "
             "(GT hashed only)."],
         "instrument_sha256_lf": {f: lf_sha(f) for f in INSTRUMENTS}}
    CP.dump(out / "declaration.json", d)
    print(json.dumps({"declaration_sha256": sha(out / "declaration.json")}, indent=1))
    return 0


# ------------------------------------------------------------------------------------------------ estimate ---
def stage_estimate(a):
    d = check_declaration(a)
    pf = load(a.preflight)
    if sha(a.preflight) != d["preflight_sha256"]:
        raise SystemExit("STOP preflight differs")
    out = new_dir(a.out_dir)
    res = {"stage": "estimate", "time": now(), "head": git("rev-parse", "HEAD"), "runs": {}}
    t0 = time.perf_counter()
    # 1) frozen SfM on held-out and S1 (W141 masked wrapper; needed by the 6DoF availability decision)
    for dsn in ("sitting_static", "s1"):
        name = f"sfm_{dsn}_rgbmask"
        rc, dt = run([PC_PY, "-I", EST141, "--input", pf["inputs"][dsn]["dir"], "--out", out / name, "--keep-masks", pf["masks"][dsn]["dir"]],
                     out / f"{name}.log")
        ep = out / name / "estimate.json"
        res["runs"][name] = {"rc": rc, "wall_s": dt, "estimate_sha256": sha(ep) if ep.is_file() else None}
        print(name, rc, round(dt, 1), flush=True)

    def sfm_path(dsn):
        s = SFM_SRC[dsn]
        return str(out / s.split(":")[1] / "estimate.json") if isinstance(s, str) and s.startswith("NEW:") else str(s)

    def one(item):
        name, dsn = item
        sp = sfm_path(dsn)
        cmd = [sys.executable, "-I", EST143, "--input", pf["inputs"][dsn]["dir"], "--out", out / name, "--keep-masks", pf["masks"][dsn]["dir"],
               "--mask-label", "rgbmask"] + (["--sfm", sp] if Path(sp).is_file() else [])
        rc, dt = run(cmd, out / f"{name}.log")
        ep = out / name / "estimate.json"
        return name, {"rc": rc, "wall_s": dt, "estimate_sha256": sha(ep) if ep.is_file() else None, "sfm": sp if Path(sp).is_file() else None}

    def w142(item):
        name, dsn = item
        cmd = [sys.executable, "-I", EST142, "--input", pf["inputs"][dsn]["dir"], "--out", out / name, "--keep-masks", pf["masks"][dsn]["dir"],
               "--mask-label", "rgbmask"]
        rc, dt = run(cmd, out / f"{name}.log")
        ep = out / name / "estimate.json"
        return name, {"rc": rc, "wall_s": dt, "estimate_sha256": sha(ep) if ep.is_file() else None, "estimator": "frozen W142"}
    with ThreadPoolExecutor(PARALLEL) as ex:
        futs = [ex.submit(one, r) for r in RUNS143] + [ex.submit(w142, ("w142_sitting_static_rgbmask", "sitting_static"))]
        for fu in futs:
            name, v = fu.result()
            res["runs"][name] = v
            print(name, v["rc"], round(v["wall_s"], 1), flush=True)
            CP.dump(out / "estimate_result.json", res)
    res["wall_s"] = time.perf_counter() - t0
    CP.dump(out / "estimate_result.json", res)
    print(json.dumps({"wall_s": res["wall_s"], "rc": {k: v["rc"] for k, v in res["runs"].items()}}, indent=1))
    return 0


# ------------------------------------------------------------------------------------------------ evaluate ---
def gt_for(dsn, pf):
    if dsn in ("walking_static", "walking_xyz"):
        return DP.read_groundtruth_dedup(W141 / "data" / "extracted" / f"rgbd_dataset_freiburg3_{dsn}" / "groundtruth.txt")[0], "real"
    if dsn == "sitting_static":
        return DP.read_groundtruth_dedup(pf["inputs"]["sitting_static"]["gt"])[0], "real"
    return CP.read_groundtruth(pf["inputs"][dsn]["gt"]), "synthetic"


def evaluate_w143(est, gt, kind, h42):
    frames = est["frames"]
    ids = [f["frame_id"] for f in frames]
    times = [f["timestamp"] for f in frames]
    Tg = [h42.associate(gt, kind, t) for t in times]
    st = [f["orientation"]["status"] == "ORIENTATION_ESTIMATED" for f in frames]
    Re = [np.asarray(f["orientation"]["R_WC"]) if f["orientation"]["R_WC"] is not None else None for f in frames]
    rep, ok = h42.orient_eval(st, Re, Tg, times, ids)
    ed = []
    for e in est["edges"]:
        Ti, Tj = Tg[e["i"]], Tg[e["j"]]
        if Ti is None or Tj is None or "M" not in e:
            continue
        Grel = Ti[:3, :3].T @ Tj[:3, :3]
        tg = Ti[:3, :3].T @ (Tj[:3, 3] - Ti[:3, 3])
        r = {"i": e["i"], "j": e["j"], "span": e["span"], "accepted": e["rotation_accepted"], "reject": e["reject"], "fit": e["fit_model"],
             "rot_err_deg": float(CP.geodesic_deg(np.asarray(e["M"]).T, Grel)), "gt_rot_deg": BR.angle_deg(Grel), "gt_trans_m": float(np.linalg.norm(tg)),
             "bias_ind": e["rotation_bias_bound_deg"], "translation_edge": e["translation_edge"]}
        if e.get("translation_direction_in_cam_i") and np.linalg.norm(tg) > 0.005:
            r["dir_err_deg"] = float(np.degrees(np.arccos(np.clip(np.dot(e["translation_direction_in_cam_i"], tg / np.linalg.norm(tg)), -1, 1))))
        ed.append(r)
    accd = [r for r in ed if r["accepted"]]
    rej = [r for r in ed if not r["accepted"]]
    rep["edges"] = {"n_with_gt": len(ed), "accepted": len(accd), "accepted_rot_err_median_deg": float(np.median([r["rot_err_deg"] for r in accd])) if accd else None,
                    "accepted_rot_err_p90_deg": float(np.percentile([r["rot_err_deg"] for r in accd], 90)) if accd else None,
                    "rejected_rot_err_median_deg": float(np.median([r["rot_err_deg"] for r in rej])) if rej else None,
                    "bias_indicator_calibration": float(np.mean([r["rot_err_deg"] <= r["bias_ind"] + 0.05 for r in accd])) if accd else None,
                    "by_reject_reason": {}}
    for reason in ("rotation_inliers", "rotation_support", "rotation_residual", "rotation_vs_homography", "cycle_inconsistent", "too_few_matches"):
        rr = [r for r in ed if reason in r["reject"]]
        rep["edges"]["by_reject_reason"][reason] = {"n": len(rr), "rot_err_median_deg": float(np.median([r["rot_err_deg"] for r in rr])) if rr else None}
    dirc = [r for r in ed if r["translation_edge"] == "DIRECTION_OBSERVED"]
    zero = [r for r in ed if r["gt_trans_m"] < 1e-9]
    rep["translation_edges"] = {"direction_observed": len(dirc), "dir_err_median_deg": float(np.median([r["dir_err_deg"] for r in dirc if "dir_err_deg" in r]))
                                if any("dir_err_deg" in r for r in dirc) else None,
                                "false_dir_on_zero_translation": sum(r["translation_edge"] == "DIRECTION_OBSERVED" for r in zero),
                                "n_zero_translation_edges": len(zero),
                                "direction_observed_gt_trans_median_m": float(np.median([r["gt_trans_m"] for r in dirc])) if dirc else None}
    rep["translation_edges"]["false_dir_frac"] = (rep["translation_edges"]["false_dir_on_zero_translation"] / len(zero)) if zero else None
    # frame-level translation claims vs GT window displacement
    gi = [k for k in range(len(frames)) if Tg[k] is not None]
    disp = {}
    for k in gi:
        w = [q for q in gi if abs(times[q] - times[k]) <= BR.WINDOW_S + 1e-9]
        P = np.asarray([Tg[q][:3, 3] for q in w])
        disp[k] = float(np.linalg.norm(P[:, None] - P[None], axis=-1).max()) if len(P) > 1 else 0.0
    dir_frames = [k for k in gi if frames[k]["translation"]["status"] == "TRANSLATION_DIRECTION_ONLY"]
    rep["translation_frames"] = {"counts": est["counts"]["translation"], "direction_only_frames": len(dir_frames),
                                 "direction_only_with_gt_disp_lt_1cm": sum(disp[k] < G["sit_false_dir_disp_m"] for k in dir_frames),
                                 "frac_frames_false_direction": sum(disp[k] < G["sit_false_dir_disp_m"] for k in dir_frames) / max(len(gi), 1),
                                 "gt_window_disp_median_m": float(np.median(list(disp.values()))) if disp else None}
    lab = BR.gt_regime_labels([times[k] for k in gi], [Tg[k][:3, 3] for k in gi], [Tg[k][:3, :3] for k in gi])
    conf = {}
    for k, l in zip(gi, lab):
        key = f"{l}->{frames[k]['translation_window'].get('regime', 'INSUFFICIENT_EVIDENCE')}"
        conf[key] = conf.get(key, 0) + 1
    rep["regime_confusion"] = conf
    rep["regime_frames"] = [f["translation_window"].get("regime", "INSUFFICIENT_EVIDENCE") for f in frames]
    rep["camera_state_counts"] = est["counts"]["camera_state"]
    rep["numeric_translation_without_valid_sfm"] = est["counts"]["numeric_translation_without_valid_sfm"]
    rep["counts"] = est["counts"]
    rep["sfm_validity"] = est["sfm_validity"]
    rep["kp_on_excluded_total"] = est["kp_on_excluded_total"]
    rep["timing_s"] = est["timing_s"]
    rep["rotation_averaging"] = est["rotation_averaging"]
    if gi:
        Pg = np.asarray([Tg[k][:3, 3] for k in gi])
        Rg = np.asarray([Tg[k][:3, :3] for k in gi])
        rep["gt_motion"] = {"max_dev_from_mean_m": float(np.linalg.norm(Pg - Pg.mean(0), axis=1).max()),
                            "rot_range_deg": float(CP.geodesic_deg(Rg[:, None], Rg[None]).max()), "path_m": float(np.sum(np.linalg.norm(np.diff(Pg, axis=0), axis=1)))}
    if "valid_ids" in rep and rep["valid_ids"]:
        Rv = np.asarray([Tg[ids.index(i)][:3, :3] for i in rep["valid_ids"]])
        rep["gt_rot_range_valid_deg"] = float(CP.geodesic_deg(Rv[:, None], Rv[None]).max())
    return rep, Tg


def common_compare(est_a, est_b, key_a, key_b, gt, kind, h42):
    """Orientation accuracy of two estimates on their common valid frames, each re-aligned on that set."""
    va = {f["frame_id"]: np.asarray(f["orientation"]["R_WC"]) for f in est_a["frames"] if key_a(f)}
    vb = {f["frame_id"]: np.asarray(f["orientation"]["R_WC"]) for f in est_b["frames"] if key_b(f)}
    tmap = {f["frame_id"]: f["timestamp"] for f in est_a["frames"]}
    ids = sorted(set(va) & set(vb))
    Tg = {i: h42.associate(gt, kind, tmap[i]) for i in ids}
    ids = [i for i in ids if Tg[i] is not None]
    if len(ids) < 3:
        return {"n": len(ids)}
    Rg = np.asarray([Tg[i][:3, :3] for i in ids])
    out = {"n": len(ids)}
    for nm, v in (("a", va), ("b", vb)):
        _, err = BR.orientation_errors(Rg, np.asarray([v[i] for i in ids]))
        out[f"{nm}_median_deg"] = float(np.median(err))
        out[f"{nm}_p90_deg"] = float(np.percentile(err, 90))
    _, berr = BR.orientation_errors(Rg, np.broadcast_to(np.eye(3), Rg.shape))
    out["baseline_median_deg"] = float(np.median(berr))
    return out


def stage_evaluate(a):
    d = check_declaration(a)
    pf = load(a.preflight)
    E = Path(a.estimate_dir)
    er = load(E / "estimate_result.json")
    out = new_dir(a.out_dir)
    h42 = _h42()
    res = {"stage": "evaluate", "time": now(), "frozen_verified": {n: (E / n / "estimate.json").is_file() and sha(E / n / "estimate.json") == v["estimate_sha256"]
                                                                   for n, v in er["runs"].items()}, "runs": {}, "w142_comparison": {}, "edge_vs_w142": {}}
    if not all(res["frozen_verified"].values()):
        raise SystemExit(f"STOP estimates not frozen {res['frozen_verified']}")
    for name, dsn in RUNS143:
        est = load(E / name / "estimate.json")
        if dsn == "s1":
            res["runs"][name] = {"counts": est["counts"], "sfm_validity": est["sfm_validity"], "timing_s": est["timing_s"],
                                 "s1": s1_diag(est), "kp_on_excluded_total": est["kp_on_excluded_total"]}
            continue
        gt, kind = gt_for(dsn, pf)
        rep, Tg = evaluate_w143(est, gt, kind, h42)
        res["runs"][name] = rep
        # W142 comparison: historical (or frozen held-out run) on common frames
        w142p = W142_HIST.get(dsn) or (E / "w142_sitting_static_rgbmask" / "estimate.json")
        w = load(w142p)
        res["w142_comparison"][dsn] = {"w142_source": str(w142p),
                                       "w142_coverage": sum(f["orientation"]["status"] == "ESTIMATED" for f in w["frames"]) / len(w["frames"]),
                                       "w143_coverage": rep["coverage"],
                                       "common": common_compare(est, w, lambda f: f["orientation"]["status"] == "ORIENTATION_ESTIMATED",
                                                                lambda f: f["orientation"]["status"] == "ESTIMATED", gt, kind, h42)}
        w142_eval = h42.eval_w142(w, gt, kind, dsn)
        res["w142_comparison"][dsn]["w142_full"] = {k: w142_eval.get(k) for k in ("coverage",)}
        if "aligned" in w142_eval:
            res["w142_comparison"][dsn]["w142_full"].update({"median_deg": w142_eval["aligned"]["median_deg"], "baseline_median_deg": w142_eval["baseline_aligned"]["median_deg"],
                                                             "rel1s": w142_eval["relative_1s"]["median_deg"], "rel1s_base": w142_eval["relative_1s"]["baseline_median_deg"],
                                                             "sequence_regime": w142_eval["sequence_regime"]["regime"]})
        # W143 edge decision vs the W142 state of the same pair
        wst = {(e["i"], e["j"]): e["state"] for e in w["edges"]}
        tab = {}
        for e in est["edges"]:
            s = wst.get((e["i"], e["j"]), "NA")
            t = tab.setdefault(s, {"n": 0, "accepted": 0, "err": []})
            t["n"] += 1
            t["accepted"] += int(e["rotation_accepted"])
            Ti, Tj = Tg[e["i"]], Tg[e["j"]]
            if e["rotation_accepted"] and Ti is not None and Tj is not None:
                t["err"].append(float(CP.geodesic_deg(np.asarray(e["M"]).T, Ti[:3, :3].T @ Tj[:3, :3])))
        res["edge_vs_w142"][dsn] = {s: {"n": t["n"], "accepted": t["accepted"], "accepted_rot_err_median_deg": float(np.median(t["err"])) if t["err"] else None}
                                    for s, t in tab.items()}
    sfm_sit = load(E / "sfm_sitting_static_rgbmask" / "estimate.json")
    gt_sit, _ = gt_for("sitting_static", pf)
    res["sfm_sitting_static"] = {k: v for k, v in h42.eval_sfm(sfm_sit["frames"], gt_sit, "real").items() if k in ("coverage", "n_valid", "aligned")}
    if "aligned" in res["sfm_sitting_static"]:
        res["sfm_sitting_static"]["aligned"] = {k: v for k, v in res["sfm_sitting_static"]["aligned"].items() if k != "per_frame"}
    res["w142_on_held_out"] = {k: v for k, v in h42.eval_w142(load(E / "w142_sitting_static_rgbmask" / "estimate.json"), gt_sit, "real", "sitting_static").items()
                               if k in ("coverage", "sequence_regime", "edge_state_counts", "orientation_status_counts")}
    res["gates"] = gates(res, E)
    CP.dump(out / "evaluation.json", res)
    summ = {}
    for n, r in res["runs"].items():
        s = {"counts_camera": r.get("counts", {}).get("camera_state"), "translation": r.get("counts", {}).get("translation")}
        if "aligned" in r:
            s.update({"cov": round(r["coverage"], 3), "med": round(r["aligned"]["median_deg"], 3), "p90": round(r["aligned"]["p90_deg"], 3),
                      "max": round(r["aligned"]["max_deg"], 3), "base": round(r["baseline_aligned"]["median_deg"], 3),
                      "rel1s": r["relative_1s"]["median_deg"], "rel1s_base": r["relative_1s"]["baseline_median_deg"],
                      "edges": r["edges"], "tr_edges": r["translation_edges"], "tr_frames": r["translation_frames"], "conf": r["regime_confusion"]})
        summ[n] = s
    print(json.dumps({"runs": summ, "w142_comparison": res["w142_comparison"], "edge_vs_w142": res["edge_vs_w142"],
                      "sfm_sitting": res["sfm_sitting_static"], "w142_held_out": res["w142_on_held_out"], "gates": res["gates"]}, indent=1, default=str))
    return 0


def s1_diag(est):
    e1 = [e for e in est["edges"] if e["span"] == 1 and "flow_raw_median_px" in e]
    return {"calibration_status": est["calibration_status"], "flow_raw_median_px_span1": float(np.median([e["flow_raw_median_px"] for e in e1])) if e1 else None,
            "rotation_accepted_edges": est["counts"]["edges_rotation_accepted"], "edges_total": est["counts"]["edges_total"],
            "reject_reasons": est["counts"]["edge_reject_reasons"], "fit_models": est["counts"]["edges_fit_model"],
            "nominal_rotation_deg_accepted_median": float(np.median([e["rotation_deg"] for e in est["edges"] if e.get("rotation_accepted")]))
            if est["counts"]["edges_rotation_accepted"] else None,
            "orientation_counts": est["counts"]["orientation"], "translation_counts": est["counts"]["translation"],
            "note": "nominal focal; rotation magnitudes are not calibrated measurements"}


def gates(res, E):
    R = res["runs"]
    g = {}
    a1, a2, a3, a4 = (R[f"w143_syn_A{k}"] for k in (1, 2, 3, 4))
    seg = [r for fid, r in enumerate(load(E / "w143_syn_A4" / "estimate.json")["frames"]) if 20 <= fid <= 69]
    syn = {"A1": {"cov": a1["coverage"], "max": a1.get("aligned", {}).get("max_deg"), "false_dir": a1["translation_edges"]["false_dir_frac"]},
           "A2": {"cov": a2["coverage"], "med": a2.get("aligned", {}).get("median_deg"), "false_dir": a2["translation_edges"]["false_dir_frac"]},
           "A3": {"cov": a3["coverage"], "med": a3.get("aligned", {}).get("median_deg"), "dir_err": a3["translation_edges"]["dir_err_median_deg"]},
           "A4": {"cov": a4["coverage"], "med": a4.get("aligned", {}).get("median_deg"),
                  "seg_dir_frac": sum(f["translation_window"]["translation"] == "TRANSLATION_DIRECTION_ONLY" for f in seg) / len(seg)},
           "numeric_translation_without_valid_sfm": sum(R[n].get("numeric_translation_without_valid_sfm", 0) for n in R if n != "w143_s1"),
           "kp_on_excluded": sum(R[n]["kp_on_excluded_total"] for n in R)}
    syn["pass"] = bool(syn["A1"]["cov"] >= G["syn_cov"] and syn["A1"]["max"] is not None and syn["A1"]["max"] < G["syn_A1_max_deg"]
                       and (syn["A1"]["false_dir"] or 0) <= G["syn_false_dir_frac"]
                       and syn["A2"]["cov"] >= G["syn_cov"] and syn["A2"]["med"] is not None and syn["A2"]["med"] < G["syn_A2_med_deg"]
                       and (syn["A2"]["false_dir"] or 0) <= G["syn_false_dir_frac"]
                       and syn["A3"]["cov"] >= G["syn_cov"] and syn["A3"]["med"] is not None and syn["A3"]["med"] < G["syn_A3_med_deg"]
                       and (syn["A3"]["dir_err"] is None or syn["A3"]["dir_err"] < G["syn_A3_dir_err_deg"])
                       and syn["A4"]["cov"] >= G["syn_cov"] and syn["A4"]["med"] is not None and syn["A4"]["med"] < G["syn_A4_med_deg"]
                       and syn["A4"]["seg_dir_frac"] >= G["syn_A4_seg_dir_frac"]
                       and syn["numeric_translation_without_valid_sfm"] == 0 and syn["kp_on_excluded"] == 0)
    g["synthetic"] = syn

    def real(dsn, heldout):
        r = R[f"w143_{dsn}"]
        x = {"coverage": r["coverage"]}
        if "aligned" not in r:
            x["pass"] = False
            return x
        x.update({"median": r["aligned"]["median_deg"], "baseline": r["baseline_aligned"]["median_deg"], "rel1s": r["relative_1s"]["median_deg"],
                  "rel1s_base": r["relative_1s"]["baseline_median_deg"], "spike_frac": r["relative_consecutive"]["frac_gt_spike"],
                  "numeric_translation_without_valid_sfm": r["numeric_translation_without_valid_sfm"]})
        x["beats_baseline"] = bool(x["median"] < x["baseline"] and x["rel1s"] is not None and x["rel1s"] < x["rel1s_base"])
        cm = res["w142_comparison"][dsn]["common"]
        x["common_vs_w142"] = cm
        x["nondeteriorated"] = bool(cm.get("n", 0) < 3 or cm["a_median_deg"] <= cm["b_median_deg"] + G["nondeterioration_deg"])
        base_ok = x["beats_baseline"]
        if heldout:
            x["gt_rot_range_valid_deg"] = r.get("gt_rot_range_valid_deg")
            x["meaningful_rotation"] = bool((x["gt_rot_range_valid_deg"] or 0) >= G["sit_meaningful_rot_deg"])
            base_ok = x["beats_baseline"] if x["meaningful_rotation"] else True
            x["false_direction_frac"] = r["translation_frames"]["frac_frames_false_direction"]
            tr_ok = x["false_direction_frac"] <= G["sit_false_dir_frac"]
        else:
            tr_ok = True
        x["pass"] = bool(x["coverage"] >= G["real_cov"] and x["median"] < G["real_med_deg"] and base_ok and tr_ok
                         and x["numeric_translation_without_valid_sfm"] == 0 and (heldout or x["nondeteriorated"]))
        return x
    g["walking_static"] = real("walking_static", False)
    g["sitting_static"] = real("sitting_static", True)
    xe = load(E / "w143_walking_xyz" / "estimate.json")
    n = len(xe["frames"])
    full = sum(f["camera_state"] == "FULL_6DOF_AVAILABLE" for f in xe["frames"])
    downgraded = sum(1 for f in xe["frames"] if f["sfm"]["frame_valid"] and f["camera_state"] != "FULL_6DOF_AVAILABLE")
    ns = sum(f["translation_window"].get("regime") == "NEAR_STATIC_OR_BELOW_DETECTABILITY" for f in xe["frames"]) / n
    g["walking_xyz"] = {"full6dof_frac": full / n, "near_static_frac": ns, "sfm_valid_downgraded": downgraded,
                        "sfm_validity": xe["sfm_validity"], "orientation_coverage": R["w143_walking_xyz"]["coverage"],
                        "orientation_median_deg": R["w143_walking_xyz"].get("aligned", {}).get("median_deg")}
    g["walking_xyz"]["pass"] = bool(full / n >= G["xyz_full6dof_frac"] and ns <= G["xyz_near_static_frac"] and downgraded == 0)
    ws = g["walking_static"]
    gain = res["w142_comparison"]["walking_static"]["w143_coverage"] - res["w142_comparison"]["walking_static"]["w142_coverage"]
    g["walking_static_coverage_gain"] = gain
    if g["synthetic"]["pass"] and ws["pass"] and g["sitting_static"]["pass"] and g["walking_xyz"]["pass"]:
        j = "SUPPORTED"
    elif not ws["pass"] and not g["sitting_static"]["pass"] and (gain < G["notyet_cov_gain"] or not ws.get("nondeteriorated", False)):
        j = "NOT_YET_SUPPORTED"
    else:
        j = "PARTIALLY_SUPPORTED"
    g["judgment"] = j
    return g


# ------------------------------------------------------------------------------------------------ visualize ---
def stage_visualize(a):
    import w143_visualize as V
    check_declaration(a)
    return V.visualize(a, ROOT, W141, W142, W143, RUNS143, gt_for, _h42())


def main():
    global H42
    H42 = _h42()
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
