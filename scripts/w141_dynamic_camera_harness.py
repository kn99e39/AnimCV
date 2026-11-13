#!/usr/bin/env python3
"""W141 (EXPERIMENTAL, opt-in, scratch only): Dynamic-Scene Camera Pose Robustness harness.

Question: does the FROZEN W140 RGB-only estimator (COLMAP incremental SfM, scripts/w140_colmap_estimate.py) recover the
camera trajectory when independently moving humans are in view, and does excluding RGB-predicted human regions from
feature extraction improve it beyond an area-matched static-region control?

Datasets: TUM RGB-D fr3 walking_static and walking_xyz (300 uniformly selected frames each), and a controlled synthetic
dynamic scene (W140 static scene + camera trajectory, plus two walking human-shaped occluders; scripts/w141_synthetic_dynamic_scene.py).
Conditions (only the feature-exclusion factor differs; same frames, calibration, estimator configuration):
  A  unmasked            W140 estimator unchanged (scripts/w140_colmap_estimate.py)
  B  rgbmask             keypoints on disk-dilated RGB person masks dropped by COLMAP's extractor (scripts/w141_colmap_masked_estimate.py)
  C  areactrl            same pixel area excluded from static regions, humans left available (area_control_mask of B)
  D  oracle              synthetic only: exact foreground identity (ORACLE diagnostic, never a real-world claim)
  syn_oracle_areactrl    synthetic only: area control of D (oracle-derived diagnostic)
  syn_clean              W140 clean static frames (reference)

Stages (each refuses an existing output directory):
  preflight  repo/branch, W140 identities, dataset identities + rights + fr3 calibration text, frame selection, estimator
             inputs, synthetic dynamic render + background identity check, RGB person segmentation, B/C/D masks + contracts,
             extraction-only feature-exclusion check (no matching/mapping/poses), resources.
  declare    prospective declaration (frozen before any W141 estimator output or GT score).
  estimate   the 11 declared estimator runs; outputs frozen by SHA-256.
  evaluate   GT association (after freezing), Sim(3) metrics, common-intersection comparison, early-reference alignment +
             observability, feature/contamination accounting, synthetic causal analysis, gates, classification.
  visualize  mask/feature overlays, trajectories, error curves, synthetic comparisons, playback videos, 3D viewport videos.

    PY=C:/Projects/AnimCV/.venv_build_full/Scripts/python.exe; export PYTHONPATH=src
    $PY -X utf8 scripts/w141_dynamic_camera_harness.py <stage> --out-dir <dir> [--preflight P --declaration D
        --declaration-sha S --estimate-dir E --eval-dir V]
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
import app.experimental_dynamic_camera_pose as DP                     # noqa: E402

W141 = Path("C:/Users/dna10/animcv-w141")
DATA = W141 / "data"
W140 = Path("C:/Users/dna10/animcv-w140")
SYN140 = W140 / "data" / "synthetic_v3"
SYN_DYN = DATA / "synthetic_dynamic"
PC_PY = W140 / "env" / "pc311" / "Scripts" / "python.exe"
EST140 = ROOT / "scripts" / "w140_colmap_estimate.py"
EST141 = ROOT / "scripts" / "w141_colmap_masked_estimate.py"
SEG = ROOT / "scripts" / "w141_segment_people.py"
SYN_SCRIPT = ROOT / "scripts" / "w141_synthetic_dynamic_scene.py"
BLENDER = "C:/Program Files/Blender Foundation/Blender 5.1/blender.exe"
TORCH_HOME = W141 / "models"
BRANCH = "Dual_Stage_Generator"
SEQS = {"walking_static": {"archive": "rgbd_dataset_freiburg3_walking_static.tgz", "bytes": 474074105,
                           "url": "https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_walking_static.tgz",
                           "last_modified": "Tue, 07 Aug 2012 18:59:31 GMT",
                           "clauses": ["The Asus Xtion sensor has been kept in place manually.", "Ground-truth trajectory length: 0.282m"]},
        "walking_xyz": {"archive": "rgbd_dataset_freiburg3_walking_xyz.tgz", "bytes": 527550055,
                        "url": "https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_walking_xyz.tgz",
                        "last_modified": "Tue, 07 Aug 2012 18:50:52 GMT",
                        "clauses": ["The Asus Xtion sensor has manually been moved along three directions (xyz) while keeping the same orientation.",
                                    "Ground-truth trajectory length: 5.791m"]}}
FR3 = {"fx": 535.4, "fy": 539.2, "cx": 320.1, "cy": 247.6, "width": 640, "height": 480}
CALIB_ROW = "Freiburg 3 RGB 535.4 539.2 320.1 247.6 0 0 0 0 0"
UNDIST_CLAUSE = ("Note that both the color and IR images of the Freiburg 3 sequences have already been undistorted, therefore the "
                 "distortion parameters are all zero.")
LICENCE_CLAUSE = "licensed under a Creative Commons 4.0 Attribution License (CC BY 4.0)"
GT_CLAUSE = "give the position of the optical center of the color camera with respect to the world origin as defined by the motion capture system"
W140_IDS = {"evaluation.json": ("results/evaluate/evaluation.json", "3bb32caa90e76059e19a2cd8f95eb9ccb994e059754b07963a56f30d7e6fb953"),
            "evidence_manifest.json": ("results/evidence_manifest.json", "28888c2790842a45fde363028e2dd12e2275dc314317337b791c98726fcd7ab6"),
            "declaration.json": ("results/declare/declaration.json", "4a0a118c2012760c2fd8fb9a768681dbf9c17f3b53d379e64770409bbe0e9fbe"),
            "estimate_result.json": ("results/estimate/estimate_result.json", "305cb1d8d0959ae4d08dd16c799eaf1cd29866b8f937bc1fb93343b738d57175")}
INSTRUMENTS = ["src/app/experimental_dynamic_camera_pose.py", "scripts/w141_dynamic_camera_harness.py", "scripts/w141_colmap_masked_estimate.py",
               "scripts/w141_segment_people.py", "scripts/w141_synthetic_dynamic_scene.py", "tests/test_w141_dynamic_camera_pose.py",
               "src/app/experimental_camera_pose.py", "scripts/w140_colmap_estimate.py"]
# (run name, input key, mask condition or None, sequence/dataset key, role)
RUNS = [("syn_clean", "syn_clean", None, "synthetic", "reference: W140 clean static frames"),
        ("syn_unmasked", "syn_dynamic", None, "synthetic", "A: dynamic, unmasked"),
        ("syn_rgbmask", "syn_dynamic", "rgbmask", "synthetic", "B: dynamic, RGB person-mask exclusion"),
        ("syn_oracle", "syn_dynamic", "oracle", "synthetic", "D: dynamic, ORACLE foreground exclusion (diagnostic)"),
        ("syn_oracle_areactrl", "syn_dynamic", "oracle_areactrl", "synthetic", "oracle-derived area control (diagnostic)"),
        ("walking_static_A", "walking_static", None, "walking_static", "A: unmasked"),
        ("walking_static_B", "walking_static", "rgbmask", "walking_static", "B: RGB person-mask exclusion"),
        ("walking_static_C", "walking_static", "areactrl", "walking_static", "C: area-matched static-region control"),
        ("walking_xyz_A", "walking_xyz", None, "walking_xyz", "A: unmasked"),
        ("walking_xyz_B", "walking_xyz", "rgbmask", "walking_xyz", "B: RGB person-mask exclusion"),
        ("walking_xyz_C", "walking_xyz", "areactrl", "walking_xyz", "C: area-matched static-region control")]
LOW_BACKGROUND_KP = 200


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


def run(cmd, log):
    t0 = time.perf_counter()
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, encoding="utf-8", errors="replace")
    Path(log).write_text(r.stdout + "\n--- stderr ---\n" + r.stderr, encoding="utf-8")
    return r.returncode, time.perf_counter() - t0, r.stdout


def check_declaration(a):
    if sha(a.declaration) != a.declaration_sha:
        raise SystemExit("STOP INTEGRITY_FAILED: declaration changed")
    d = load(a.declaration)
    bad = [f for f, h in d["instrument_sha256_lf"].items() if lf_sha(f) != h]
    if bad:
        raise SystemExit(f"STOP DECLARATION_INVALID: instruments changed {bad}")
    return d


def read_mask(p):
    import cv2
    m = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise SystemExit(f"STOP MASK_CONTRACT: unreadable {p}")
    if m.ndim == 3:                                   # Blender writes the oracle masks as RGB (white = foreground)
        if not (m == m[..., :1]).all():
            raise SystemExit(f"STOP MASK_CONTRACT: non-grey mask {p}")
        m = m[..., 0]
    return m


def write_mask(p, m):
    import cv2
    cv2.imwrite(str(p), np.asarray(m, np.uint8))


# ------------------------------------------------------------------------------------------------ preflight ---
def build_tum_input(seq, out):
    import cv2
    src = DATA / "extracted" / f"rgbd_dataset_freiburg3_{seq}"
    ts, files = CP.read_rgb_list(src / "rgb.txt")
    sel = CP.select_uniform(len(ts))
    ind = new_dir(out / f"input_{seq}")
    (ind / "images").mkdir()
    frames, bad = [], []
    for k in sel:
        im = cv2.imread(str(src / files[k]), cv2.IMREAD_COLOR)
        if im is None or im.shape[:2] != (FR3["height"], FR3["width"]):
            bad.append({"frame_id": k, "file": files[k], "reason": "unreadable or wrong size"})
            continue
        nm = f"f{k:04d}_{ts[k]:.6f}.png"
        shutil.copyfile(src / files[k], ind / "images" / nm)          # byte copy: fr3 RGB is already undistorted
        frames.append({"frame_id": k, "timestamp": float(ts[k]), "image": nm, "source": files[k]})
    json.dump(frames, open(ind / "frames.json", "w"), indent=1)
    json.dump({"model": "PINHOLE", **FR3, "pixel_convention": "OpenCV (integer = pixel centre); COLMAP cx = cx + 0.5",
               "undistortion": "none applied: official page states fr3 colour images are already undistorted (d0..d4 = 0)",
               "source": "TUM file_formats page, 'Freiburg 3 RGB' row"}, open(ind / "camera.json", "w"), indent=1)
    man = {"selected_frame_ids": sel, "n_total_rgb": len(ts), "unreadable": bad,
           "rule": "uniform: round(linspace(0, N-1, 300)) over rgb.txt order (W140 rule, chosen without GT or outcomes); an unreadable "
                   "selected frame stays in the denominator as INVALID and is never replaced",
           "files_sha256": {f["image"]: sha(ind / "images" / f["image"]) for f in frames}}
    json.dump(man, open(ind / "input_manifest.json", "w"), indent=1)
    return ind, man


def copy_syn_input(name, img_dir, out):
    ind = new_dir(out / f"input_{name}")
    shutil.copytree(img_dir, ind / "images")
    shutil.copyfile(SYN140 / "input" / "camera.json", ind / "camera.json")
    shutil.copyfile(SYN140 / "input" / "frames.json", ind / "frames.json")
    fr = load(ind / "frames.json")
    json.dump({"files_sha256": {p.name: sha(p) for p in sorted((ind / "images").iterdir())}, "source": str(img_dir),
               "selected_frame_ids": [f["frame_id"] for f in fr]}, open(ind / "input_manifest.json", "w"), indent=1)
    return ind


def build_masks(key, ind, out, oracle_dir=None):
    """Segment people (RGB only), then build the declared exclusion masks and COLMAP keep masks for one input."""
    import cv2
    md = new_dir(out / "masks" / key)
    rc, dt, so = run([sys.executable, "-I", SEG, "--images", ind / "images", "--frames", ind / "frames.json", "--out", md / "raw",
                      "--torch-home", TORCH_HOME], md / "segment.log")
    if rc != 0:
        raise SystemExit(f"STOP SEGMENTATION_BLOCKED {key}: rc {rc}")
    det = load(md / "raw_detections.json")
    frames = load(ind / "frames.json")
    conds = ["rgbmask", "areactrl"] + (["oracle", "oracle_areactrl"] if oracle_dir else [])
    for c in conds:
        (md / f"keep_{c}").mkdir()
    per = []
    for f in frames:
        img = cv2.imread(str(ind / "images" / f["image"]))
        raw = read_mask(md / "raw" / f["image"]) > 127
        if raw.shape != img.shape[:2]:
            raise SystemExit("STOP MASK_CONTRACT: raw mask geometry")
        B = DP.disk_dilate(raw)
        C, cinfo = DP.area_control_mask(B)
        ex = {"rgbmask": B, "areactrl": C}
        rec = {"frame_id": f["frame_id"], "image": f["image"], "raw_frac": float(raw.mean()), "B_frac": float(B.mean()), "C_frac": float(C.mean()),
               "C_info": cinfo, "B_contains_raw": bool(not (raw & ~B).any()), "C_disjoint_B": bool(not (C & B).any())}
        if oracle_dir:
            orc = read_mask(oracle_dir / f["image"]) > 127
            D = DP.disk_dilate(orc)
            Dc, dinfo = DP.area_control_mask(D)
            ex.update({"oracle": D, "oracle_areactrl": Dc})
            rec.update({"oracle_frac": float(orc.mean()), "D_frac": float(D.mean()), "Dctrl_frac": float(Dc.mean()), "Dctrl_info": dinfo,
                        "rgb_vs_oracle_raw": DP.mask_quality(raw, orc), "rgb_vs_oracle_dilated": DP.mask_quality(B, D)})
        for c, E in ex.items():
            write_mask(md / f"keep_{c}" / (f["image"] + ".png"), DP.keep_from_exclusion(E))
        per.append(rec)
    summ = {"segment_wall_s": dt, "segment_model_s": det.get("total_s"), "n_frames": len(frames),
            "raw_frac": {q: float(np.percentile([r["raw_frac"] for r in per], q)) for q in (0, 10, 50, 90, 100)},
            "raw_frac_mean": float(np.mean([r["raw_frac"] for r in per])),
            "frames_with_person": int(sum(r["raw_frac"] > 0 for r in per)),
            "B_frac_mean": float(np.mean([r["B_frac"] for r in per])), "C_frac_mean": float(np.mean([r["C_frac"] for r in per])),
            "C_shortfall_frames": int(sum(r["C_info"]["shortfall_px"] > 0 for r in per)),
            "C_area_ratio_median": float(np.median([r["C_info"]["area_px"] / r["C_info"]["target_px"] for r in per if r["C_info"]["target_px"]] or [1.0])),
            "all_B_contains_raw": all(r["B_contains_raw"] for r in per), "all_C_disjoint_B": all(r["C_disjoint_B"] for r in per),
            "keep_dirs_sha256": {c: hashlib.sha256("".join(sha(p) for p in sorted((md / f"keep_{c}").iterdir())).encode()).hexdigest() for c in conds}}
    if oracle_dir:
        ious = [r["rgb_vs_oracle_raw"]["iou"] for r in per if r["rgb_vs_oracle_raw"]["iou"] is not None]
        rec_ = [r["rgb_vs_oracle_raw"]["recall"] for r in per if r["rgb_vs_oracle_raw"]["recall"] is not None]
        summ["rgb_vs_oracle"] = {"iou_median": float(np.median(ious)), "iou_p10": float(np.percentile(ious, 10)),
                                 "recall_median": float(np.median(rec_)), "frames_no_detection_with_fg": int(sum(1 for r in per if r["raw_frac"] == 0 and r["oracle_frac"] > 0)),
                                 "D_frac_mean": float(np.mean([r["D_frac"] for r in per]))}
    CP.dump(md / "mask_summary.json", {"summary": summ, "per_frame": per})
    return md, summ


def stage_preflight(a):
    out = new_dir(a.out_dir)
    rep = {"stage": "preflight", "time": now(), "branch": git("rev-parse", "--abbrev-ref", "HEAD"), "head": git("rev-parse", "HEAD"),
           "status_porcelain": git("status", "--porcelain").splitlines()}
    rep["branch_ok"] = rep["branch"] == BRANCH
    w140 = {k: {"path": str(W140 / p), "sha256": sha(W140 / p), "expected": h} for k, (p, h) in W140_IDS.items()}
    decl140 = load(ROOT / "examples" / "experimental_camera_pose" / "w140_declaration.json")
    w140["instruments_unchanged"] = {f: lf_sha(f) == h for f, h in decl140["instrument_sha256_lf"].items()}
    w140["ok"] = all(v["sha256"] == v["expected"] for k, v in w140.items() if k in W140_IDS) and all(w140["instruments_unchanged"].values())
    rep["w140_baseline"] = w140
    ds = {}
    ff, ov, dl = (page_text(DATA / "docs" / n) for n in ("rgbd-dataset_file_formats.html", "rgbd-dataset.html", "rgbd-dataset_download.html"))
    for seq, d in SEQS.items():
        arc = DATA / "archives" / d["archive"]
        src = DATA / "extracted" / f"rgbd_dataset_freiburg3_{seq}"
        ds[seq] = {"url": d["url"], "archive_sha256": sha(arc), "archive_bytes": arc.stat().st_size, "expected_bytes": d["bytes"],
                   "server_last_modified": d["last_modified"], "rgb_txt_sha256": sha(src / "rgb.txt"),
                   "groundtruth_txt_sha256": sha(src / "groundtruth.txt"), "n_rgb": len(CP.read_rgb_list(src / "rgb.txt")[0]),
                   "extracted_entries": sorted(p.name for p in src.iterdir()),
                   "description_clauses_present": {c: c in dl for c in d["clauses"]},
                   "authentication": "official TUM HTTPS URL; byte size equals the server Content-Length; TUM publishes no checksum, so the "
                                     "downloaded SHA-256 is recorded as the identity"}
        ds[seq]["ok"] = bool(ds[seq]["archive_bytes"] == d["bytes"] and all(ds[seq]["description_clauses_present"].values())
                             and "depth" not in ds[seq]["extracted_entries"])
    rep["datasets"] = ds
    rep["docs_sha256"] = {p.name: sha(p) for p in (DATA / "docs").iterdir()}
    rep["calibration"] = {"published_row_present": CALIB_ROW in ff, "undistorted_clause_present": UNDIST_CLAUSE in ff, "used": FR3,
                          "gt_semantics_clause_present": GT_CLAUSE in ff, "note": "Freiburg 3 (not Freiburg 1) calibration; no undistortion applied"}
    rep["rights"] = {"licence_clause_present": LICENCE_CLAUSE in ov, "licence": "CC BY 4.0 (TUM RGB-D benchmark data)",
                     "attribution": "J. Sturm, N. Engelhard, F. Endres, W. Burgard, D. Cremers, A Benchmark for the Evaluation of RGB-D SLAM Systems, IROS 2012",
                     "use": "local research evaluation only; data in scratch, never committed or redistributed",
                     "segmentation_model": {"code": "torchvision (BSD-3-Clause)", "weights": "maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth, published by "
                                            "PyTorch/torchvision, trained on COCO 2017 (annotations CC BY 4.0; images under Flickr terms); used for local "
                                            "inference only, not redistributed; no training"}}
    inputs = {}
    for seq in SEQS:
        ind, man = build_tum_input(seq, out)
        inputs[seq] = {"dir": str(ind), "n_selected": len(man["selected_frame_ids"]), "unreadable": man["unreadable"],
                       "manifest_sha256": sha(ind / "input_manifest.json"), "frames_sha256": sha(ind / "frames.json"),
                       "camera_sha256": sha(ind / "camera.json")}
    # ---- synthetic dynamic scene ----
    if not SYN_DYN.exists():
        rc, dt, _ = run([BLENDER, "-b", "--factory-startup", "--python-exit-code", "1", "-P", SYN_SCRIPT, "--", "--w140", SYN140, "--out", SYN_DYN],
                        out / "synthetic_render.log")
        if rc != 0:
            raise SystemExit("STOP synthetic render failed")
        rep["synthetic_render_s"] = dt
    import cv2
    bg = {}
    for p in sorted((SYN_DYN / "check").iterdir()):
        fi = int(p.stem.split("_f")[1])
        a_, b_ = cv2.imread(str(p)).astype(int), cv2.imread(str(SYN140 / "images" / f"f{fi:04d}.png")).astype(int)
        d_ = np.abs(a_ - b_)
        bg[fi] = {"max_abs": int(d_.max()), "mean_abs": float(d_.mean()), "frac_px_diff_gt_8": float((d_.max(2) > 8).mean())}
    same_cam = all(sha(SYN_DYN / r) == sha(SYN140 / r) for r in ("input/camera.json", "input/frames.json", "gt/camera_gt.txt", "gt/segments.json"))
    rep["synthetic"] = {"dir": str(SYN_DYN), "generator_lf_sha256": lf_sha("scripts/w141_synthetic_dynamic_scene.py"),
                        "w140_scene_blend_sha256": sha(SYN140 / "scene.blend"), "background_identity": bg,
                        "camera_timestamps_gt_identical_to_w140": same_cam, "gt_sha256": sha(SYN_DYN / "gt" / "camera_gt.txt"),
                        "background_ok": bool(same_cam and all(v["mean_abs"] < 0.05 and v["frac_px_diff_gt_8"] < 0.001 for v in bg.values()))}
    inputs["syn_clean"] = {"dir": str(copy_syn_input("syn_clean", SYN140 / "images", out))}
    inputs["syn_dynamic"] = {"dir": str(copy_syn_input("syn_dynamic", SYN_DYN / "images", out))}
    for k in ("syn_clean", "syn_dynamic"):
        ind = Path(inputs[k]["dir"])
        inputs[k].update({"n_selected": len(load(ind / "frames.json")), "manifest_sha256": sha(ind / "input_manifest.json")})
    rep["inputs"] = inputs
    # ---- masks ----
    masks = {}
    for k in ("walking_static", "walking_xyz", "syn_dynamic"):
        md, summ = build_masks(k, Path(inputs[k]["dir"]), out, oracle_dir=(SYN_DYN / "gt" / "fg_masks") if k == "syn_dynamic" else None)
        masks[k] = {"dir": str(md), "summary": summ, "mask_summary_sha256": sha(md / "mask_summary.json")}
    rep["masks"] = masks
    # ---- extraction-only feature-exclusion contract (no matching / mapping / poses) ----
    xc = {}
    for k in ("walking_static", "walking_xyz", "syn_dynamic"):
        rc, dt, so = run([PC_PY, "-I", EST141, "--input", inputs[k]["dir"], "--out", out / f"extract_check_{k}", "--keep-masks",
                          Path(masks[k]["dir"]) / "keep_rgbmask", "--extract-check", 8], out / f"extract_check_{k}.log")
        xc[k] = load(out / f"extract_check_{k}" / "extract_check.json") if (out / f"extract_check_{k}" / "extract_check.json").is_file() else {"ok": False}
        xc[k]["rc"] = rc
    rep["extract_check"] = xc
    envr = subprocess.run([str(PC_PY), "-I", "-c", "import pycolmap,numpy,cv2,sys;print(pycolmap.__version__,numpy.__version__,cv2.__version__,sys.version.split()[0])"],
                          capture_output=True, text=True).stdout.split()
    rep["estimator_env"] = {"python_exe": str(PC_PY), "pycolmap": envr[0] if envr else None, "numpy": envr[1] if len(envr) > 1 else None,
                            "opencv": envr[2] if len(envr) > 2 else None, "freeze_sha256": sha(W140 / "env" / "freeze.txt")}
    tot = sum(p.stat().st_size for p in W141.rglob("*") if p.is_file())
    rep["resources"] = {"disk_free_gb": round(shutil.disk_usage(str(W141)).free / 1e9, 1), "w141_scratch_gb": round(tot / 1e9, 2)}
    ok = (rep["branch_ok"] and w140["ok"] and all(v["ok"] for v in ds.values()) and rep["calibration"]["published_row_present"]
          and rep["calibration"]["undistorted_clause_present"] and rep["calibration"]["gt_semantics_clause_present"]
          and rep["rights"]["licence_clause_present"] and rep["synthetic"]["background_ok"] and all(v["ok"] for v in xc.values())
          and all(m["summary"]["all_B_contains_raw"] and m["summary"]["all_C_disjoint_B"] for m in masks.values())
          and rep["estimator_env"]["pycolmap"] == "4.2.1" and all(not v.get("unreadable") for k, v in inputs.items() if k in SEQS))
    rep["status"] = "OK" if ok else "BLOCKED"
    CP.dump(out / "preflight.json", rep)
    print(json.dumps({"status": rep["status"], "branch_ok": rep["branch_ok"], "w140_ok": w140["ok"], "datasets_ok": {k: v["ok"] for k, v in ds.items()},
                      "calibration": {k: v for k, v in rep["calibration"].items() if k != "used"}, "rights": rep["rights"]["licence_clause_present"],
                      "synthetic_background": rep["synthetic"]["background_ok"], "extract_check": {k: v["ok"] for k, v in xc.items()},
                      "masks": {k: {kk: m["summary"][kk] for kk in ("raw_frac_mean", "raw_frac", "frames_with_person", "B_frac_mean", "C_frac_mean",
                                                                     "C_shortfall_frames", "C_area_ratio_median")} | ({"rgb_vs_oracle": m["summary"]["rgb_vs_oracle"]}
                                                                                                                      if "rgb_vs_oracle" in m["summary"] else {})
                                for k, m in masks.items()}, "resources": rep["resources"]}, indent=1, default=str))
    return 0 if ok else 3


# ------------------------------------------------------------------------------------------------ declare ---
def outputs_exist():
    r = W141 / "results"
    return sorted(str(p) for p in r.rglob("estimate.json")) if r.is_dir() else []


def stage_declare(a):
    out = new_dir(a.out_dir)
    pf = load(a.preflight)
    if pf["status"] != "OK":
        raise SystemExit("preflight not OK")
    if outputs_exist():
        raise SystemExit(f"STOP: estimator output already exists {outputs_exist()}")
    d = {"schema": "animcv_exp_w141_declaration_v1", "time": now(), "estimates_exist": False, "gt_scores_seen": False,
         "branch": pf["branch"], "repo_head_at_declaration": git("rev-parse", "HEAD"), "preflight_sha256": sha(a.preflight),
         "w140_baseline": pf["w140_baseline"], "datasets": pf["datasets"], "inputs": pf["inputs"],
         "masks": {k: {"dir": v["dir"], "mask_summary_sha256": v["mask_summary_sha256"], "keep_dirs_sha256": v["summary"]["keep_dirs_sha256"]}
                   for k, v in pf["masks"].items()},
         "estimator": {"method": "FROZEN W140 COLMAP incremental SfM (pycolmap 4.2.1, CPU), scripts/w140_colmap_estimate.py unchanged",
                       "config": "SIFT max 8192; sequential pairing overlap 10 quadratic, no loop detection; single PINHOLE camera with the given "
                                 "intrinsics, NOT refined; mapper defaults; random_seed 0; 8 threads; multiple models allowed; primary = largest",
                       "feature_exclusion": "conditions B/C/D/oracle_areactrl run scripts/w141_colmap_masked_estimate.py, which executes the W140 "
                                            "estimator code unchanged with pycolmap.ImageReaderOptions().mask_path set to the condition's keep masks "
                                            "(COLMAP drops keypoints on zero pixels; images are never painted). Verified pre-declaration by the "
                                            "extraction-only check (0 keypoints on excluded pixels; masked set == unmasked set minus excluded).",
                       "inputs_only": ["RGB PNG frames (fr3: byte copies, already undistorted)", "original timestamps", "published fr3 intrinsics",
                                       "B/C: exclusion masks derived from the same RGB frames only"],
                       "never_inputs": ["groundtruth.txt / synthetic camera_gt.txt", "depth", "accelerometer/IMU", "GT-derived poses, scale or alignment",
                                        "oracle masks (except the labelled synthetic oracle diagnostics)"],
                       "determinism": "seeded, but multi-threaded mapping not bit-reproducible; exactly one official run per declared run"},
         "segmentation": {"model": "torchvision maskrcnn_resnet50_fpn_v2 COCO_V1", "weights_sha256": "73cbd0190fcbe3ba339921fbce2c3a0b6bb9126c9a133c85e43a2a8e060a109e",
                          "settings": "CPU float32, model's own resize, label person, score >= 0.5, soft mask >= 0.5, union over instances",
                          "dilation_px": DP.DILATE_PX, "selection_rationale": "one bounded, widely used, licensed instance-segmentation checkpoint "
                          "with a COCO person class, available through the existing torchvision install; no model search, no training"},
         "conditions": {r[0]: {"input": r[1], "mask": r[2], "dataset": r[3], "role": r[4]} for r in RUNS},
         "area_control": {"construction": DP.area_control_mask.__doc__, "offsets": DP.AREA_OFFSETS,
                          "limitation": "pixel-area matched, not feature-count matched; the removed-keypoint counts of B and C are reported"},
         "conventions": CP.__doc__ + "\n" + DP.__doc__,
         "selection": {"rule": "fr3: round(linspace(0, N-1, 300)) over rgb.txt order (no GT, no outcome); synthetic: all 140 frames",
                       "ids": {s: load(Path(pf["inputs"][s]["dir"]) / "input_manifest.json")["selected_frame_ids"] for s in SEQS},
                       "missing_rule": "kept in the denominator, never replaced"},
         "gt_association": f"W140 rule: linear position + SLERP between the bracketing GT samples, gap <= {CP.GT_MAX_BRACKET_GAP_S} s; evaluate stage only, "
                           "after all estimate.json are frozen by SHA-256",
         "alignment": {"benchmark": "full-trajectory Umeyama Sim(3) on camera centres (evaluation only)",
                       "common_intersection": "each condition re-aligned (own Sim(3)) on the frames ESTIMATED+associated in all of A, B, C",
                       "early_reference": f"first {int(DP.EARLY_FRACTION * 100)} % by time; OBSERVABLE iff n >= {DP.EARLY_MIN_FRAMES}, GT RMS spread sv1 >= "
                                          f"{DP.EARLY_MIN_SPREAD_M} m and scale > 0: rotation from orientations (chordal), scale/translation from "
                                          "positions; otherwise ROTATION_ONLY with scale/translation UNOBSERVABLE; never repaired with later GT",
                       "w140_position_only_early": "computed for continuity, never interpreted as drift",
                       "statement": "no alignment is estimation or localisation; ATE after Sim(3) is not metric scale; GT world origin not recovered"},
         "metrics": ["coverage over all selected frames", "Sim(3) ATE RMSE/median/p90", "rotation median/p90", "RPE consecutive and ~1 s",
                     f"GT-referenced spikes (consecutive relative error > max({DP.SPIKE_FACTOR} x median, {DP.SPIKE_FLOOR_M} m / {DP.SPIKE_FLOOR_DEG} deg))",
                     "path length estimated (scaled) vs GT", "drift slopes", "lost intervals", "Sim(3) scale/rotation", "keypoints / retained / "
                     "removed per frame from the COLMAP database", "model observations inside RGB person masks (contamination proxy)",
                     "person-anchored landmarks", f"frames with < {LOW_BACKGROUND_KP} retained keypoints (inadequate background evidence)",
                     "Spearman correlation of per-frame occupancy with A error and with A-B error difference", "runtime",
                     "synthetic: per-segment errors, stationary-interval spurious motion, rotation-interval spurious translation, landmark distance "
                     "to static boxes, oracle-mask contamination, RGB-vs-oracle mask IoU"],
         "gates": {"config": f"per real sequence x condition: coverage >= {DP.COVERAGE_FLOOR:.0%} (denominator 300) AND Sim(3) ATE RMSE <= "
                             f"{DP.ATE_TARGET_M} AND median rotation error <= {DP.ROT_MED_TARGET_DEG} deg AND both ATE and median rotation below the "
                             "stationary best-constant-pose baseline (added because walking_static's GT extent makes the ATE target non-discriminating)",
                   "masking_benefit": f"B improves on A: (B coverage >= floor and A < floor) OR on the common intersection (n >= {DP.MIN_COMMON}) ATE or median "
                                      f"rotation lower by >= {DP.BENEFIT_REL:.0%} and >= {DP.BENEFIT_ABS_ATE_M} m / {DP.BENEFIT_ABS_ROT_DEG} deg; AND B coverage "
                                      f">= A - {DP.COVERAGE_LOSS_MAX}; AND C distinct (C does not improve on A by the same rule, or B beats C by it). Verdicts "
                                      "DEMONSTRATED / NOT_ATTRIBUTABLE_TO_HUMAN_EXCLUSION / HARMFUL / UNNECESSARY (A passes) / NOT_DEMONSTRATED",
                   "synthetic": f"clean run must reproduce W140's synthetic recovery/stationary gates; a condition is DISTURBED if ATE_rel > "
                                f"{DP.SYN_DISTURB_ATE_RATIO}x clean and > clean + {DP.SYN_DISTURB_ATE_ABS_REL:.1%} of extent, or median rotation > clean + "
                                f"{DP.SYN_DISTURB_ROT_DEG} deg, or coverage < clean - {DP.SYN_DISTURB_COVERAGE}, or stationary-interval spurious motion "
                                "fails the W140 thresholds; exclusion PREVENTS disturbance if the masked run is not DISTURBED while unmasked is",
                   "classification": "SUPPORTED: A or B passes the config gate on both real sequences; PARTIALLY_SUPPORTED: on one, or some "
                                     "configuration reaches the coverage floor and beats the baseline; NOT_YET_SUPPORTED otherwise; BLOCKED: a "
                                     "prerequisite fails. The masking-benefit verdict is reported separately. Never changed after results."},
         "runs": {"n": len(RUNS), "names": [r[0] for r in RUNS], "order": "sequential, one official run each"},
         "budget": {"estimator_wall_h_max": 3.0, "segmentation_wall_h_max": 1.0, "w141_scratch_gb_max": 12.0, "real_frames_max_per_sequence": 300,
                    "training": "none", "on_exceed": "RESOURCE_BLOCKED"},
         "pre_declaration_disclosures": [
             "dev check of COLMAP mask_path on two W140 synthetic frames (dev/masktest): 0 keypoints on excluded pixels, masked = unmasked minus excluded",
             "dev prototype of the synthetic dynamic scene on 4 frames (dev/syn_proto) and of Mask R-CNN on them (person IoU 0.29-0.86 vs the "
             "oracle); afterwards the nearer occluder was moved closer to the camera (y -1.0 -> -1.6, start x -1.4 -> -0.9, speed 0.90 -> 0.85) to "
             "raise image occupancy; no estimator, matching or GT-scored output existed at that point; the occluders were not adjusted to "
             "improve detection",
             "the RGB masks and their occupancy statistics were computed in the preflight (no GT involved) before this declaration; DILATE_PX, "
             "thresholds and area-control offsets were fixed in the instrument code before the masks were generated"],
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
    for name, inp, cond, ds, role in RUNS:
        ind = Path(pf["inputs"][inp]["dir"])
        if sha(ind / "input_manifest.json") != d["inputs"][inp]["manifest_sha256"]:
            raise SystemExit(f"STOP input {inp} changed")
        if cond is None:
            cmd = [PC_PY, "-I", EST140, "--input", ind, "--out", out / name]
        else:
            mk = Path(pf["masks"][inp]["dir"]) / f"keep_{cond}"
            hh = hashlib.sha256("".join(sha(p) for p in sorted(mk.iterdir())).encode()).hexdigest()
            if hh != d["masks"][inp]["keep_dirs_sha256"][cond]:
                raise SystemExit(f"STOP masks {inp}/{cond} changed")
            cmd = [PC_PY, "-I", EST141, "--input", ind, "--out", out / name, "--keep-masks", mk]
        rc, dt, _ = run(cmd, out / f"{name}.log")
        ep = out / name / "estimate.json"
        res["runs"][name] = {"rc": rc, "wall_s": dt, "estimate_sha256": sha(ep) if ep.is_file() else None, "cmd": [str(c) for c in cmd]}
        print(name, rc, round(dt, 1), flush=True)
        CP.dump(out / "estimate_result.json", res)
    print(json.dumps({k: {kk: v[kk] for kk in ("rc", "wall_s")} for k, v in res["runs"].items()}, indent=1))
    return 0


# ------------------------------------------------------------------------------------------------ evaluate ---
def feature_accounting(name, inp, cond, est, pf, est_dir):
    """Keypoints from the COLMAP database vs the RGB person masks / condition masks, and model-observation contamination."""
    md = Path(pf["masks"][inp]["dir"])
    kps = DP.read_db_keypoints(est_dir / name / "database.db")
    obs_p = est_dir / name / "primary_model_txt" / "images.txt"
    obs = DP.read_images_txt(obs_p) if obs_p.is_file() else {}
    oracle_dir = SYN_DYN / "gt" / "fg_masks" if inp == "syn_dynamic" else None
    per = []
    pid_in, pid_all = {}, {}
    for f in est["frames"]:
        nm = f["image"]
        raw = read_mask(md / "raw" / nm) > 127 if (md / "raw" / nm).is_file() else None
        Bx = read_mask(md / "keep_rgbmask" / (nm + ".png")) == 0
        Cx = read_mask(md / "keep_areactrl" / (nm + ".png")) == 0
        own = (read_mask(md / f"keep_{cond}" / (nm + ".png")) == 0) if cond else None
        orc = (read_mask(oracle_dir / nm) > 127) if oracle_dir else None
        xy = kps.get(nm, np.zeros((0, 2)))
        r = {"frame_id": f["frame_id"], "status": f["status"], "n_kp": int(len(xy)), "kp_in_raw_person": int(DP.keypoints_in_mask(xy, raw).sum()),
             "kp_in_B": int(DP.keypoints_in_mask(xy, Bx).sum()), "kp_in_C": int(DP.keypoints_in_mask(xy, Cx).sum()),
             "occupancy_raw": float(raw.mean()), "B_frac": float(Bx.mean())}
        if own is not None:
            r["kp_on_own_excluded"] = int(DP.keypoints_in_mask(xy, own).sum())
        if orc is not None:
            r["kp_in_oracle_fg"] = int(DP.keypoints_in_mask(xy, orc).sum())
            r["oracle_frac"] = float(orc.mean())
        o = obs.get(nm)
        if o is not None and len(o["xy"]):
            inr = DP.keypoints_in_mask(o["xy"], raw)
            r.update({"n_obs": int(len(o["xy"])), "obs_in_raw_person": int(inr.sum()), "obs_in_B": int(DP.keypoints_in_mask(o["xy"], Bx).sum())})
            if orc is not None:
                r["obs_in_oracle_fg"] = int(DP.keypoints_in_mask(o["xy"], orc).sum())
            ref = orc if orc is not None else raw
            inref = DP.keypoints_in_mask(o["xy"], ref)
            for p, b in zip(o["pid"], inref):
                pid_all[p] = pid_all.get(p, 0) + 1
                pid_in[p] = pid_in.get(p, 0) + int(b)
        per.append(r)
    n_pts = len(pid_all)
    anchored = sum(1 for p in pid_all if pid_in.get(p, 0) * 2 > pid_all[p])
    reg = [r for r in per if "n_obs" in r]
    summ = {"kp_median": float(np.median([r["n_kp"] for r in per])), "kp_in_raw_person_total": int(sum(r["kp_in_raw_person"] for r in per)),
            "kp_in_B_total": int(sum(r["kp_in_B"] for r in per)), "kp_in_C_total": int(sum(r["kp_in_C"] for r in per)),
            "kp_on_own_excluded_total": int(sum(r.get("kp_on_own_excluded", 0) for r in per)) if cond else None,
            "obs_in_person_fraction": (float(sum(r["obs_in_raw_person"] for r in reg) / max(1, sum(r["n_obs"] for r in reg))) if reg else None),
            "obs_in_oracle_fraction": (float(sum(r.get("obs_in_oracle_fg", 0) for r in reg) / max(1, sum(r["n_obs"] for r in reg))) if reg and oracle_dir else None),
            "landmarks": n_pts, "person_anchored_landmarks": anchored, "person_anchored_fraction": anchored / n_pts if n_pts else None,
            "anchoring_reference": "oracle foreground" if oracle_dir else "RGB person mask (undilated)",
            "low_background_frames": [r["frame_id"] for r in per if r["n_kp"] < LOW_BACKGROUND_KP]}
    return {"summary": summ, "per_frame": per}


def box_distance(P, boxes):
    """Distance of points to the nearest static box surface (0 inside)."""
    best = np.full(len(P), np.inf)
    for b in boxes:
        M = np.asarray(b["matrix_world"], float)
        R, t = M[:3, :3], M[:3, 3]
        Rn = R / np.linalg.norm(R, axis=0)
        loc = (Rn.T @ (P - t).T).T - np.asarray(b["center_local"])
        q = np.abs(loc) - np.asarray(b["half_extents"])
        d = np.linalg.norm(np.maximum(q, 0), axis=1)
        best = np.minimum(best, d)
    return best


def synthetic_segments(ev, seg):
    out = {}
    if "full_sim3" not in ev:
        return out
    ids = ev["frame_ids"]
    Pa, Pg = np.asarray(ev["aligned_positions"]), np.asarray(ev["gt_positions"])
    Ra, Rg = np.asarray(ev["aligned_rotations"]), np.asarray(ev["gt_rotations"])
    for s, (lo, hi) in seg.items():
        ix = [i for i, f in enumerate(ids) if lo <= f <= hi]
        if len(ix) < 2:
            out[s] = {"n": len(ix)}
            continue
        out[s] = {"n": len(ix), "ate_rmse": float(np.sqrt(np.mean(np.sum((Pa[ix] - Pg[ix]) ** 2, 1)))),
                  "rot_err_median_deg": float(np.median(CP.geodesic_deg(Ra[ix], Rg[ix]))),
                  "est_path_length": float(np.sum(np.linalg.norm(np.diff(Pa[ix], axis=0), axis=1))),
                  "gt_path_length": float(np.sum(np.linalg.norm(np.diff(Pg[ix], axis=0), axis=1))),
                  "est_rotation_range_deg": float(CP.geodesic_deg(Ra[ix][:, None], Ra[ix][None]).max()),
                  "gt_rotation_range_deg": float(CP.geodesic_deg(Rg[ix][:, None], Rg[ix][None]).max())}
    return out


def spearman(x, y):
    from scipy.stats import spearmanr
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 10 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    r = spearmanr(x, y)
    return {"rho": float(r.statistic), "p": float(r.pvalue), "n": int(len(x))}


def stage_evaluate(a):
    d = check_declaration(a)
    pf = load(a.preflight)
    est_dir = Path(a.estimate_dir)
    er = load(est_dir / "estimate_result.json")
    out = new_dir(a.out_dir)
    gts = {"synthetic": SYN_DYN / "gt" / "camera_gt.txt", **{s: DATA / "extracted" / f"rgbd_dataset_freiburg3_{s}" / "groundtruth.txt" for s in SEQS}}
    want_gt = {"synthetic": pf["synthetic"]["gt_sha256"], **{s: d["datasets"][s]["groundtruth_txt_sha256"] for s in SEQS}}
    res = {"stage": "evaluate", "time": now(), "runs": {}, "frozen_estimates_verified": {}}
    # 1) verify every estimate is frozen BEFORE any GT is read
    ests = {}
    for name, inp, cond, ds, role in RUNS:
        ep = est_dir / name / "estimate.json"
        ok = ep.is_file() and sha(ep) == er["runs"][name]["estimate_sha256"]
        res["frozen_estimates_verified"][name] = bool(ok)
        ests[name] = load(ep) if ok else None
    for k, p in gts.items():
        if sha(p) != want_gt[k]:
            raise SystemExit(f"STOP GT identity {k}")
    gt = {k: CP.read_groundtruth(p) for k, p in gts.items()}
    seg = load(SYN_DYN / "gt" / "segments.json")
    boxes = load(SYN_DYN / "gt" / "static_boxes.json")
    w140_contract_ok = load(W140 / "results" / "preflight" / "preflight.json")["oracle_contract"]["ok"]
    for name, inp, cond, ds, role in RUNS:
        e = ests[name]
        if e is None:
            res["runs"][name] = {"uncomputable": "estimate missing or changed after freezing", "role": role}
            continue
        ev = DP.evaluate_dynamic(e["frames"], gt[ds])
        pts = np.load(est_dir / name / "points3D.npy")
        ev["role"], ev["dataset"], ev["condition_mask"] = role, ds, cond
        ev["estimator_diagnostics"] = {"models": e["models"], "primary_model_id": e["primary_model_id"], "timing_s": e["timing_s"],
                                       "wall_s": er["runs"][name]["wall_s"], "n_points3D": int(len(pts)),
                                       "mean_track_length": float(pts[:, 7].mean()) if len(pts) else None,
                                       "mean_point_error_px": float(pts[:, 6].mean()) if len(pts) else None,
                                       "keypoints_median": float(np.median([f["n_keypoints"] for f in e["frames"]])),
                                       "observed_points_median": float(np.median([f.get("n_points3D_observed") or 0 for f in e["frames"] if f["status"] == "ESTIMATED"]))
                                       if ev["status_counts"]["ESTIMATED"] else None,
                                       "consecutive_inlier_ratio_median": float(np.median([p["inlier_ratio"] for p in e["pair_stats_consecutive"] if p["inlier_ratio"] is not None]))
                                       if any(p["inlier_ratio"] is not None for p in e["pair_stats_consecutive"]) else None,
                                       "consecutive_inliers_min": min((p["inliers"] for p in e["pair_stats_consecutive"]), default=None),
                                       "consecutive_inliers_median": float(np.median([p["inliers"] for p in e["pair_stats_consecutive"]])),
                                       "lost_frame_ids": [f["frame_id"] for f in e["frames"] if f["status"] == "LOST"],
                                       "invalid_frame_ids": [f["frame_id"] for f in e["frames"] if f["status"] == "INVALID"],
                                       "mask_binding_sha256": sha(est_dir / name / "mask_binding.json") if cond else None}
        if inp != "syn_clean":
            fa = feature_accounting(name, inp, cond, e, pf, est_dir)
            ev["features"] = fa
            if "full_sim3" in ev:
                occ = {r["frame_id"]: r["occupancy_raw"] for r in fa["per_frame"]}
                ev["occupancy_vs_error"] = {"pos": spearman([occ[i] for i in ev["frame_ids"]], ev["full_sim3"]["ate"]["per_frame"]),
                                            "rot": spearman([occ[i] for i in ev["frame_ids"]], ev["full_sim3"]["rot"]["per_frame"])}
        if ds == "synthetic":
            ev["segments"] = synthetic_segments(ev, seg)
            st = list(range(seg["stationary"][0], seg["stationary"][1] + 1))
            ev["gates_w140_synthetic"] = CP.synthetic_gates(ev, st, contracts_ok=w140_contract_ok and pf["synthetic"]["background_ok"])
            if "full_sim3" in ev and len(pts):
                s, R, t = ev["full_sim3"]["sim3"]["scale"], np.asarray(ev["full_sim3"]["sim3"]["R"]), np.asarray(ev["full_sim3"]["sim3"]["t"])
                Pw = (s * (R @ pts[:, :3].T)).T + t
                dist = box_distance(Pw, boxes)
                ev["landmark_consistency"] = {"n": int(len(Pw)), "median_dist_to_static_box_m": float(np.median(dist)),
                                              "p90_dist_m": float(np.percentile(dist, 90)), "frac_within_2cm": float((dist < 0.02).mean()),
                                              "frac_beyond_10cm": float((dist > 0.10).mean())}
        else:
            ev["gates"] = DP.config_gate(ds, ev)
        res["runs"][name] = ev
    # 2) matched comparisons
    comp = {}
    for seq in SEQS:
        names = {c: f"{seq}_{c}" for c in "ABC"}
        evs = {c: res["runs"][n] for c, n in names.items()}
        reg = [set(evs[c].get("frame_ids", [])) for c in "ABC"]
        common = sorted(set.intersection(*reg)) if all(reg) else []
        cm = {c: (DP.evaluate_on_ids(ests[names[c]]["frames"], gt[seq], common) if common and ests[names[c]] else {"n": 0}) for c in "ABC"}
        cov = {c: evs[c].get("coverage_estimated", 0.0) for c in "ABC"}
        gates = {c: evs[c].get("gates", {"pass": False}) for c in "ABC"}
        mb = DP.masking_benefit(cov, cm, gates)
        occ = {r["frame_id"]: r["occupancy_raw"] for r in evs["A"]["features"]["per_frame"]} if "features" in evs["A"] else {}
        corr = {}
        if cm["A"].get("frame_ids") and cm["B"].get("frame_ids"):
            ids = cm["A"]["frame_ids"]
            o = [occ[i] for i in ids]
            corr = {"occ_vs_errA_pos": spearman(o, cm["A"]["ate_per_frame"]),
                    "occ_vs_errA_minus_errB_pos": spearman(o, np.asarray(cm["A"]["ate_per_frame"]) - np.asarray(cm["B"]["ate_per_frame"])),
                    "occ_vs_errA_rot": spearman(o, cm["A"]["rot_per_frame"]),
                    "occ_vs_errA_minus_errB_rot": spearman(o, np.asarray(cm["A"]["rot_per_frame"]) - np.asarray(cm["B"]["rot_per_frame"]))}
        comp[seq] = {"coverage": cov, "n_common": len(common), "common": {c: {k: v for k, v in cm[c].items() if k not in ("ate_per_frame", "rot_per_frame", "frame_ids")}
                                                                           for c in "ABC"},
                     "common_per_frame": {"frame_ids": cm["A"].get("frame_ids"), **{c: {"ate": cm[c].get("ate_per_frame"), "rot": cm[c].get("rot_per_frame")} for c in "ABC"}},
                     "gates": gates, "masking_benefit": mb, "occupancy_correlation": corr}
    # synthetic causal analysis
    syn = {}
    rc_ = res["runs"].get("syn_clean", {})
    if "full_sim3" in rc_:
        ext = rc_["gt_extent_m"]
        for n in ("syn_unmasked", "syn_rgbmask", "syn_oracle", "syn_oracle_areactrl"):
            ev = res["runs"].get(n, {})
            stok = bool(ev.get("gates_w140_synthetic", {}).get("stationary_ok"))
            syn[n] = DP.synthetic_disturbance(ev, rc_, ext, stok)
        syn["clean_reproduces_w140_gates"] = bool(rc_["gates_w140_synthetic"]["pass"])
        un = syn["syn_unmasked"]["disturbed"]
        syn["dynamic_foreground_disturbs"] = un
        syn["prevented_by"] = {n: bool(un and not syn[n]["disturbed"]) for n in ("syn_rgbmask", "syn_oracle", "syn_oracle_areactrl")}
        regs = [set(res["runs"][n].get("frame_ids", [])) for n in ("syn_unmasked", "syn_rgbmask", "syn_oracle", "syn_oracle_areactrl")]
        common = sorted(set.intersection(*regs)) if all(regs) else []
        syn["common_intersection"] = {"n": len(common), **{n: {k: v for k, v in DP.evaluate_on_ids(ests[n]["frames"], gt["synthetic"], common).items()
                                                               if k not in ("ate_per_frame", "rot_per_frame", "frame_ids")}
                                                           for n in ("syn_clean", "syn_unmasked", "syn_rgbmask", "syn_oracle", "syn_oracle_areactrl") if common}}
    comp["synthetic"] = syn
    res["comparisons"] = comp
    real_pass = {s: bool(comp[s]["gates"]["A"].get("pass") or comp[s]["gates"]["B"].get("pass")) for s in SEQS}
    real_partial = {s: bool(any(comp[s]["gates"][c].get("coverage_ok") and comp[s]["gates"][c].get("beats_baseline") for c in "AB")) for s in SEQS}
    res["judgment"] = DP.judgment(real_pass, real_partial)
    res["real_pass"], res["real_partial"] = real_pass, real_partial
    CP.dump(out / "evaluation.json", res)
    summ = {}
    for name, ev in res["runs"].items():
        if "full_sim3" not in ev:
            summ[name] = {"uncomputable": ev.get("uncomputable"), "status": ev.get("status_counts")}
            continue
        f = ev["full_sim3"]
        summ[name] = {"cov": round(ev["coverage_estimated"], 3), "status": ev["status_counts"], "ate": round(f["ate"]["rmse"], 4),
                      "ate_p90": round(float(np.percentile(f["ate"]["per_frame"], 90)), 4), "rot_med": round(f["rot"]["median_deg"], 3),
                      "rot_p90": round(f["rot"]["p90_deg"], 3), "base_ate": round(ev["baseline_stationary"]["ate_rmse"], 4),
                      "base_rot": round(ev["baseline_stationary"]["rot_median_deg"], 3), "path_est_gt": [round(ev["path_length"]["est_scaled_m"], 3), round(ev["path_length"]["gt_m"], 3)],
                      "spikes": ev["spikes_gt_referenced"]["n"], "early": ev["early_reference"]["status"],
                      "pass": (ev.get("gates") or ev.get("gates_w140_synthetic") or {}).get("pass"),
                      "contam": (ev.get("features") or {}).get("summary", {}).get("obs_in_person_fraction"),
                      "anchored": (ev.get("features") or {}).get("summary", {}).get("person_anchored_fraction")}
    print(json.dumps({"runs": summ, "comparisons": {s: {"coverage": comp[s]["coverage"], "n_common": comp[s]["n_common"], "common": {c: {k: comp[s]["common"][c].get(k) for k in ("ate_rmse", "rot_median_deg")} for c in "ABC"},
                                                         "masking_benefit": comp[s]["masking_benefit"]["verdict"]} for s in SEQS},
                      "synthetic": {k: v for k, v in syn.items() if k != "common_intersection"}, "judgment": res["judgment"]}, indent=1, default=str))
    return 0


# ------------------------------------------------------------------------------------------------ visualize ---
def stage_visualize(a):
    import w141_visualize as V                                       # noqa: E402  (scripts/ on sys.path via script dir)
    check_declaration(a)
    return V.visualize(a, ROOT, W141, SYN_DYN, BLENDER, SEQS, RUNS)


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
