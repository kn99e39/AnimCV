#!/usr/bin/env python3
"""W142 (EXPERIMENTAL, opt-in, scratch only): Observability-Aware Near-Static Camera Reconstruction harness.

Question: can static-background correspondences give accurate camera ROTATION and an honest motion-state / translation-
observability report in low-parallax, human-containing video where incremental SfM fails -- without fabricating
translation, and without damaging the moving-camera case?

Estimator under test: scripts/w142_background_rotation_estimate.py (app.experimental_background_rotation).
Frozen controls: W140 COLMAP SfM (scripts/w140_colmap_estimate.py) and the W141 masked wrapper; W141 historical estimates.

Datasets: synthetic A1 stationary / A2 fixed-centre rotation / A3 small translation (new renders with the UNCHANGED W141
dynamic-scene script and a new camera GT), A4 = W141 synthetic dynamic scene (W140 trajectory) reused; TUM fr3
walking_static (primary real) and walking_xyz (moving-camera control) with the exact W141 inputs and masks; authentic
AnimCV S1 (Wikimedia kettlebell clip, segment 79-318) as an uncalibrated observational diagnostic.

Stages: preflight, declare, estimate, evaluate, visualize (each refuses an existing output directory).

    PY=C:/Projects/AnimCV/.venv_build_full/Scripts/python.exe; export PYTHONPATH=src
    $PY -X utf8 scripts/w142_harness.py <stage> --out-dir <dir> [--preflight P --declaration D --declaration-sha S --estimate-dir E --eval-dir V]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
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
import app.experimental_dynamic_camera_pose as DP                     # noqa: E402

W142 = Path("C:/Users/dna10/animcv-w142")
W141 = Path("C:/Users/dna10/animcv-w141")
W140 = Path("C:/Users/dna10/animcv-w140")
SYN140 = W140 / "data" / "synthetic_v3"
PC_PY = W140 / "env" / "pc311" / "Scripts" / "python.exe"
BLENDER = "C:/Program Files/Blender Foundation/Blender 5.1/blender.exe"
EST142 = ROOT / "scripts" / "w142_background_rotation_estimate.py"
EST141 = ROOT / "scripts" / "w141_colmap_masked_estimate.py"
SEG141 = ROOT / "scripts" / "w141_segment_people.py"
SYN141 = ROOT / "scripts" / "w141_synthetic_dynamic_scene.py"
BRANCH = "Dual_Stage_Generator"
S1_VIDEO = Path("C:/Users/dna10/animcv-w121/data/originals/64498918_Double_Kettlebell_Strict_Press_side_view_.webm")
S1_SHA = "de8feb9d515daa2bfaeed7167fdcc62028dd3f6450872c4fd81f8fb03dd6359c"
S1_SEGMENT = (79, 318)                    # the AnimCV S1 clip segment (W125 action sidecar: segment_source_frames)
S1_NOMINAL_F = 640.0                      # NOMINAL focal (= image width, ~53 deg HFOV): NOT a calibration
W141_DECL = ROOT / "examples" / "experimental_camera_pose" / "w141_declaration.json"
W141_DECL_SHA = "6dd621bbb9c9fb4a442220271edd9824def4985abdd54d8ea15fb5402585ffbc"
W141_IDS = {"evaluation.json": ("results/evaluate/evaluation.json", "c370522e244bd0d0af7081becdeeb811c27b8c9620e40003ed2edb6767d294e2"),
            "estimate_result.json": ("results/estimate/estimate_result.json", "ebce5a7d9dff7916120a76f35930fc9bac5dce9cd71d89ff3c497dfcb45d8ea9"),
            "evidence_manifest.json": ("results/evidence_manifest.json", "df19ba5480786ad95553a1af8849e5e9791da1ff550efca33955da22a19178a5"),
            "amendment1.json": ("results/amend1/amendment1.json", "2d7f8c396376806441de5868605601581cd2ed163910fc644fc72ce52ec242f9")}
W140_IDS = {"evaluation.json": ("results/evaluate/evaluation.json", "3bb32caa90e76059e19a2cd8f95eb9ccb994e059754b07963a56f30d7e6fb953"),
            "evidence_manifest.json": ("results/evidence_manifest.json", "28888c2790842a45fde363028e2dd12e2275dc314317337b791c98726fcd7ab6")}
INSTRUMENTS = ["src/app/experimental_background_rotation.py", "scripts/w142_background_rotation_estimate.py", "scripts/w142_harness.py",
               "tests/test_w142_background_rotation.py", "src/app/experimental_camera_pose.py", "src/app/experimental_dynamic_camera_pose.py",
               "scripts/w140_colmap_estimate.py", "scripts/w141_colmap_masked_estimate.py", "scripts/w141_segment_people.py",
               "scripts/w141_synthetic_dynamic_scene.py"]
SYN_SCENES = ("A1", "A2", "A3")
# ---- frozen synthetic camera trajectories (declared before any W142 render or estimate) ----
TRAJ = {"A1": "exactly stationary camera at the W140 frame-0 pose",
        "A2": "fixed centre (W140 frame-0); R = R0 Rx(pitch) Ry(yaw) Rz(roll) in optical axes; yaw 12 sin(2pi t/4.0), pitch 5 sin(2pi t/2.3+0.5), "
              "roll 4 sin(2pi t/3.1+1.0) deg",
        "A3": "c = c0 + smoothstep(f/139) [0.04, 0, 0.015] m (4.3 cm total); R = R0 Ry(2 sin(2pi t/4.6) deg)"}
# ---- runs: (name, input key, mask key or None, dataset, role) ----
RUNS = []
for sc in ("A1", "A2", "A3", "A4"):
    RUNS += [(f"syn_{sc}_rgbmask", f"syn_{sc}", "rgbmask", f"syn_{sc}", "primary (RGB person mask)"),
             (f"syn_{sc}_oracle", f"syn_{sc}", "oracle", f"syn_{sc}", "ORACLE mask diagnostic"),
             (f"syn_{sc}_none", f"syn_{sc}", None, f"syn_{sc}", "no-mask ablation")]
RUNS += [("walking_static_rgbmask", "walking_static", "rgbmask", "walking_static", "primary (W141 RGB person mask)"),
         ("walking_static_none", "walking_static", None, "walking_static", "no-mask ablation"),
         ("walking_static_areactrl", "walking_static", "areactrl", "walking_static", "W141 area-control mask ablation"),
         ("walking_xyz_rgbmask", "walking_xyz", "rgbmask", "walking_xyz", "primary (moving-camera control)"),
         ("walking_xyz_none", "walking_xyz", None, "walking_xyz", "no-mask ablation"),
         ("s1_rgbmask", "s1", "rgbmask", "s1", "authentic S1 observational diagnostic (uncalibrated)"),
         ("s1_none", "s1", None, "s1", "S1 no-mask ablation")]
SFM_RUNS = [(f"sfm_syn_{sc}_rgbmask", f"syn_{sc}", "rgbmask") for sc in SYN_SCENES]
HIST_SFM = {"walking_static": ["walking_static_A", "walking_static_B", "walking_static_C"], "walking_xyz": ["walking_xyz_A", "walking_xyz_B", "walking_xyz_C"],
            "syn_A4": ["syn_unmasked", "syn_rgbmask", "syn_oracle"]}
ROUTE_SFM = {"walking_static": ("w141", "walking_static_B"), "walking_xyz": ("w141", "walking_xyz_B"), "syn_A4": ("w141", "syn_rgbmask"),
             **{f"syn_{sc}": ("w142", f"sfm_syn_{sc}_rgbmask") for sc in SYN_SCENES}}
# ---- frozen gates ----
G = {"syn_stationary_false_rot_max_deg": 0.5, "syn_rotation_median_deg": 1.0, "syn_false_translation_edge_frac": 0.05,
     "syn_A4_translation_segment_frac": 0.5, "static_coverage": 0.80, "static_median_deg": 3.0, "static_spike_deg": 2.0,
     "static_spike_frac_max": 0.02, "xyz_near_static_frac_max": 0.20, "partial_static_coverage": 0.50,
     "gt_label_disp_m": 0.05, "gt_label_rot_deg": 1.0, "spike_deg_report": 1.0}
PARALLEL = 3


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


# ------------------------------------------------------------------------------------------------ synthetic trajectories ---
def rot_axis(axis, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def trajectory(scene, times, c0, R0):
    out = []
    for f, t in enumerate(times):
        if scene == "A1":
            c, R = c0.copy(), R0.copy()
        elif scene == "A2":
            yaw = 12 * math.sin(2 * math.pi * t / 4.0)
            pitch = 5 * math.sin(2 * math.pi * t / 2.3 + 0.5)
            roll = 4 * math.sin(2 * math.pi * t / 3.1 + 1.0)
            c, R = c0.copy(), R0 @ rot_axis("x", pitch) @ rot_axis("y", yaw) @ rot_axis("z", roll)
        else:
            u = f / (len(times) - 1)
            s = 0.5 - 0.5 * math.cos(math.pi * u)
            c, R = c0 + s * np.array([0.04, 0.0, 0.015]), R0 @ rot_axis("y", 2 * math.sin(2 * math.pi * t / 4.6))
        out.append((c, R))
    return out


def build_pseudo_w140(scene, out):
    """A W140-format directory (scene.blend + intrinsics + timestamps from W140, NEW camera GT) for the unchanged W141 renderer."""
    d = new_dir(out / f"pseudo_w140_{scene}")
    for sub in ("input", "gt"):
        (d / sub).mkdir()
    shutil.copyfile(SYN140 / "scene.blend", d / "scene.blend")
    shutil.copyfile(SYN140 / "input" / "camera.json", d / "input" / "camera.json")
    shutil.copyfile(SYN140 / "input" / "frames.json", d / "input" / "frames.json")
    gt0 = CP.read_groundtruth(SYN140 / "gt" / "camera_gt.txt")
    c0, R0 = gt0[0, 1:4], CP.quat_xyzw_to_R(gt0[0, 4:8])
    times = [f["timestamp"] for f in load(SYN140 / "input" / "frames.json")]
    lines = [f"# W142 synthetic {scene} GT: timestamp tx ty tz qx qy qz qw (T_WC, Blender world metres, OpenCV optical axes). {TRAJ[scene]}"]
    for t, (c, R) in zip(times, trajectory(scene, times, c0, R0)):
        q = CP.R_to_quat_xyzw(R)
        lines.append(f"{t:.6f} {c[0]:.9f} {c[1]:.9f} {c[2]:.9f} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}")
    (d / "gt" / "camera_gt.txt").write_text("\n".join(lines) + "\n")
    json.dump({"A1": {"stationary": [0, 139]}, "A2": {"rotation": [0, 139]}, "A3": {"small_translation": [0, 139]}}[scene],
              open(d / "gt" / "segments.json", "w"))
    return d


def make_input(name, img_dir, frames, cam, out):
    ind = new_dir(out / f"input_{name}")
    (ind / "images").mkdir()
    for f in frames:
        shutil.copyfile(Path(img_dir) / f["image"], ind / "images" / f["image"])
    json.dump(frames, open(ind / "frames.json", "w"), indent=1)
    json.dump(cam, open(ind / "camera.json", "w"), indent=1)
    json.dump({"files_sha256": {f["image"]: sha(ind / "images" / f["image"]) for f in frames}, "source": str(img_dir),
               "selected_frame_ids": [f["frame_id"] for f in frames]}, open(ind / "input_manifest.json", "w"), indent=1)
    return ind


def segment_and_masks(key, ind, out, oracle_dir=None, extra_exclude=None):
    import cv2
    md = new_dir(out / "masks" / key)
    rc, dt = run([sys.executable, "-I", SEG141, "--images", ind / "images", "--frames", ind / "frames.json", "--out", md / "raw",
                  "--torch-home", W141 / "models"], md / "segment.log")
    if rc != 0:
        raise SystemExit(f"STOP SEGMENTATION {key}")
    frames = load(ind / "frames.json")
    conds = ["rgbmask"] + (["oracle"] if oracle_dir else [])
    for c in conds:
        (md / f"keep_{c}").mkdir()
    per = []
    for f in frames:
        raw = cv2.imread(str(md / "raw" / f["image"]), cv2.IMREAD_UNCHANGED) > 127
        B = DP.disk_dilate(raw)
        if extra_exclude is not None:
            B = B | extra_exclude
        cv2.imwrite(str(md / "keep_rgbmask" / (f["image"] + ".png")), DP.keep_from_exclusion(B))
        r = {"frame_id": f["frame_id"], "raw_frac": float(raw.mean()), "excl_frac": float(B.mean())}
        if oracle_dir:
            o = cv2.imread(str(oracle_dir / f["image"]), cv2.IMREAD_UNCHANGED)
            o = (o[..., 0] if o.ndim == 3 else o) > 127
            D = DP.disk_dilate(o)
            cv2.imwrite(str(md / "keep_oracle" / (f["image"] + ".png")), DP.keep_from_exclusion(D))
            q = DP.mask_quality(raw, o)
            r.update({"oracle_frac": float(o.mean()), "iou": q["iou"], "recall": q["recall"]})
        per.append(r)
    summ = {"segment_s": dt, "raw_frac_mean": float(np.mean([r["raw_frac"] for r in per])),
            "raw_frac_median": float(np.median([r["raw_frac"] for r in per])), "excl_frac_mean": float(np.mean([r["excl_frac"] for r in per])),
            "frames_with_person": int(sum(r["raw_frac"] > 0 for r in per)), "keep_dirs_sha256": {c: dir_sha(md / f"keep_{c}") for c in conds}}
    if oracle_dir:
        ious = [r["iou"] for r in per if r["iou"] is not None]
        summ.update({"rgb_vs_oracle_iou_median": float(np.median(ious)) if ious else None,
                     "frames_fg_no_detection": int(sum(1 for r in per if r["raw_frac"] == 0 and r["oracle_frac"] > 0))})
    CP.dump(md / "mask_summary.json", {"summary": summ, "per_frame": per})
    return md, summ


def build_s1(out):
    """Authentic S1: the AnimCV clip segment (source frames 79-318) of the Wikimedia kettlebell video; NOMINAL intrinsics."""
    import cv2
    tmp = new_dir(out / "s1_frames")
    cap = cv2.VideoCapture(str(S1_VIDEO))
    frames, k = [], 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if S1_SEGMENT[0] <= k <= S1_SEGMENT[1]:
            nm = f"f{k:04d}.png"
            cv2.imwrite(str(tmp / nm), img)
            frames.append({"frame_id": k, "timestamp": k / 30.0, "image": nm, "container_pos_ms": cap.get(cv2.CAP_PROP_POS_MSEC)})
        k += 1
    cap.release()
    h, w = cv2.imread(str(tmp / frames[0]["image"])).shape[:2]
    cam = {"model": "PINHOLE", "width": w, "height": h, "fx": S1_NOMINAL_F, "fy": S1_NOMINAL_F, "cx": (w - 1) / 2, "cy": (h - 1) / 2,
           "calibration_status": "NOMINAL_UNCALIBRATED", "pixel_convention": "OpenCV",
           "note": "no trusted S1 intrinsics exist; focal = image width is an arbitrary nominal value: calibrated rotation accuracy is "
                   "NOT claimed; K-free (homography vs fundamental) and image-space diagnostics are primary"}
    ind = make_input("s1", tmp, frames, cam, out)
    stack = np.stack([cv2.imread(str(ind / "images" / f["image"])) for f in frames[::8]])
    med = np.median(stack, axis=0).max(axis=2)
    rows = med.max(axis=1) < 16
    letter = np.zeros((h, w), bool)
    letter[rows] = True
    letter = DP.disk_dilate(letter, 4)
    return ind, letter, {"n_frames": len(frames), "decoded_total": k, "letterbox_rows": int(rows.sum()), "letterbox_excl_frac": float(letter.mean())}


# ------------------------------------------------------------------------------------------------ preflight ---
def stage_preflight(a):
    import cv2
    out = new_dir(a.out_dir)
    rep = {"stage": "preflight", "time": now(), "branch": git("rev-parse", "--abbrev-ref", "HEAD"), "head": git("rev-parse", "HEAD"),
           "status_porcelain": git("status", "--porcelain").splitlines()}
    rep["branch_ok"] = rep["branch"] == BRANCH
    hist = {"w140": {k: {"sha256": sha(W140 / p), "ok": sha(W140 / p) == h} for k, (p, h) in W140_IDS.items()},
            "w141": {k: {"sha256": sha(W141 / p), "ok": sha(W141 / p) == h} for k, (p, h) in W141_IDS.items()},
            "w141_declaration_ok": sha(W141_DECL) == W141_DECL_SHA}
    d141 = load(W141_DECL)
    hist["w141_instruments_unchanged"] = {f: lf_sha(f) == h for f, h in load(ROOT / "examples/experimental_camera_pose/w141_declaration_amendment1.json")["instrument_sha256_lf"].items()}
    er141 = load(W141 / "results/estimate/estimate_result.json")
    hist["w141_estimates_frozen"] = {n: sha(W141 / "results/estimate" / n / "estimate.json") == v["estimate_sha256"] for n, v in er141["runs"].items()}
    hist["ok"] = (all(v["ok"] for v in hist["w140"].values()) and all(v["ok"] for v in hist["w141"].values()) and hist["w141_declaration_ok"]
                  and all(hist["w141_instruments_unchanged"].values()) and all(hist["w141_estimates_frozen"].values()))
    rep["historical"] = hist
    inputs, masks = {}, {}
    # ---- real inputs + masks: exact W141 ones, verified against the W141 declaration ----
    for s in ("walking_static", "walking_xyz"):
        ind = W141 / "results" / "preflight" / f"input_{s}"
        inputs[s] = {"dir": str(ind), "manifest_sha256": sha(ind / "input_manifest.json"), "camera_sha256": sha(ind / "camera.json"),
                     "ok": sha(ind / "input_manifest.json") == d141["inputs"][s]["manifest_sha256"] and sha(ind / "camera.json") == d141["inputs"][s]["camera_sha256"],
                     "n": len(load(ind / "frames.json")), "source": "W141 preflight input (exact frame selection, fr3 calibration, timestamps)"}
        md = W141 / "results" / "preflight" / "masks" / s
        masks[s] = {"dirs": {"rgbmask": str(md / "keep_rgbmask"), "areactrl": str(md / "keep_areactrl")},
                    "ok": all(dir_sha(md / f"keep_{c}") == d141["masks"][s]["keep_dirs_sha256"][c] for c in ("rgbmask", "areactrl")),
                    "source": "W141 frozen RGB person masks (Mask R-CNN COCO_V1, score>=0.5, mask>=0.5, +15 px) and area control"}
    ind = W141 / "results" / "preflight" / "input_syn_dynamic"
    inputs["syn_A4"] = {"dir": str(ind), "manifest_sha256": sha(ind / "input_manifest.json"), "n": 140,
                        "ok": sha(ind / "input_manifest.json") == d141["inputs"]["syn_dynamic"]["manifest_sha256"],
                        "gt": str(W141 / "data" / "synthetic_dynamic" / "gt" / "camera_gt.txt"), "source": "W141 synthetic dynamic scene (W140 trajectory)"}
    md = W141 / "results" / "preflight" / "masks" / "syn_dynamic"
    masks["syn_A4"] = {"dirs": {"rgbmask": str(md / "keep_rgbmask"), "oracle": str(md / "keep_oracle")},
                       "ok": all(dir_sha(md / f"keep_{c}") == d141["masks"]["syn_dynamic"]["keep_dirs_sha256"][c] for c in ("rgbmask", "oracle"))}
    # ---- new synthetic scenes A1-A3: unchanged W141 renderer + new camera GT ----
    scen = {}
    for sc in SYN_SCENES:
        pdir = build_pseudo_w140(sc, out)
        sdir = W142 / "data" / f"scene_{sc}"
        if not sdir.exists():
            rc, dt = run([BLENDER, "-b", "--factory-startup", "--python-exit-code", "1", "-P", SYN141, "--", "--w140", pdir, "--out", sdir],
                         out / f"render_{sc}.log")
            if rc != 0:
                raise SystemExit(f"STOP render {sc}")
            scen[sc] = {"render_s": dt}
        if sha(sdir / "gt" / "camera_gt.txt") != sha(pdir / "gt" / "camera_gt.txt"):
            raise SystemExit(f"STOP scene {sc} GT differs from the declared trajectory")
        frames = load(sdir / "input" / "frames.json")
        ind = make_input(f"syn_{sc}", sdir / "images", frames, load(sdir / "input" / "camera.json"), out)
        inputs[f"syn_{sc}"] = {"dir": str(ind), "manifest_sha256": sha(ind / "input_manifest.json"), "n": len(frames), "ok": True,
                               "gt": str(sdir / "gt" / "camera_gt.txt"), "gt_sha256": sha(sdir / "gt" / "camera_gt.txt"), "trajectory": TRAJ[sc]}
        md_, summ = segment_and_masks(f"syn_{sc}", ind, out, oracle_dir=sdir / "gt" / "fg_masks")
        masks[f"syn_{sc}"] = {"dirs": {"rgbmask": str(md_ / "keep_rgbmask"), "oracle": str(md_ / "keep_oracle")}, "summary": summ, "ok": True}
        scen[sc] = {**scen.get(sc, {}), "dir": str(sdir), "w141_renderer_lf_sha256": lf_sha("scripts/w141_synthetic_dynamic_scene.py"),
                    "scene_blend_identical_to_w140": sha(pdir / "scene.blend") == sha(SYN140 / "scene.blend")}
        # background identity of the clean check renders is meaningful only where the camera pose equals a W140 pose (frame 0)
        chk = sdir / "check" / "clean_f0000.png"
        if chk.is_file():
            dd = np.abs(cv2.imread(str(chk)).astype(int) - cv2.imread(str(SYN140 / "images" / "f0000.png")).astype(int))
            scen[sc]["frame0_clean_vs_w140_mean_abs"] = float(dd.mean())
    rep["synthetic"] = scen
    # ---- S1 ----
    s1 = {"video": str(S1_VIDEO), "sha256": sha(S1_VIDEO), "sha_ok": sha(S1_VIDEO) == S1_SHA, "segment": S1_SEGMENT,
          "rights": "Wikimedia Commons 'Double Kettlebell Strict Press (side view)', author Taco fleur, CC BY-SA 4.0 (W121 source record); "
                    "local research analysis only, not redistributed", "intrinsics": "NONE TRUSTED -> nominal f = width, labelled NOMINAL_UNCALIBRATED"}
    if s1["sha_ok"]:
        ind, letter, info = build_s1(out)
        s1.update(info)
        inputs["s1"] = {"dir": str(ind), "manifest_sha256": sha(ind / "input_manifest.json"), "n": info["n_frames"], "ok": True}
        md_, summ = segment_and_masks("s1", ind, out, extra_exclude=letter)
        masks["s1"] = {"dirs": {"rgbmask": str(md_ / "keep_rgbmask")}, "summary": summ, "ok": True,
                       "note": "keep mask also excludes the black letterbox rows (+4 px)"}
    rep["s1"] = s1
    rep["inputs"], rep["masks"] = inputs, masks
    # ---- extraction-only mask contract (features only: no pairs, no poses) ----
    xc = {}
    for k in ("walking_static", "syn_A1", "s1"):
        if k not in inputs:
            continue
        fr = load(Path(inputs[k]["dir"]) / "frames.json")[:5]
        tot, bad = 0, 0
        for f in fr:
            img = cv2.imread(str(Path(inputs[k]["dir"]) / "images" / f["image"]))
            keep = cv2.imread(str(Path(masks[k]["dirs"]["rgbmask"]) / (f["image"] + ".png")), cv2.IMREAD_UNCHANGED) > 0
            xy, _, _ = BR.sift_features(img, keep)
            tot += len(xy)
            bad += int((~BR.on_keep(xy, keep)).sum())
        xc[k] = {"kp": tot, "kp_on_excluded": bad, "ok": bad == 0 and tot > 0}
    rep["extract_check"] = xc
    rep["resources"] = {"disk_free_gb": round(shutil.disk_usage(str(W142)).free / 1e9, 1),
                        "w142_scratch_gb": round(sum(p.stat().st_size for p in W142.rglob("*") if p.is_file()) / 1e9, 2)}
    ok = (rep["branch_ok"] and hist["ok"] and all(v["ok"] for v in inputs.values()) and all(v["ok"] for v in masks.values())
          and all(v["ok"] for v in xc.values()))
    rep["status"] = "OK" if ok else "BLOCKED"
    CP.dump(out / "preflight.json", rep)
    print(json.dumps({"status": rep["status"], "historical_ok": hist["ok"], "inputs_ok": {k: v["ok"] for k, v in inputs.items()},
                      "masks_ok": {k: v["ok"] for k, v in masks.items()}, "extract_check": xc, "synthetic": scen,
                      "mask_summaries": {k: v.get("summary") for k, v in masks.items() if v.get("summary")},
                      "s1": {k: v for k, v in s1.items() if k != "rights"}, "resources": rep["resources"]}, indent=1, default=str))
    return 0 if ok else 3


# ------------------------------------------------------------------------------------------------ declare ---
def stage_declare(a):
    out = new_dir(a.out_dir)
    pf = load(a.preflight)
    if pf["status"] != "OK":
        raise SystemExit("preflight not OK")
    if (W142 / "results").is_dir() and list((W142 / "results").rglob("estimate.json")):
        raise SystemExit("STOP: W142 estimator output already exists")
    d = {"schema": "animcv_exp_w142_declaration_v1", "time": now(), "estimates_exist": False, "gt_scores_seen": False,
         "branch": pf["branch"], "repo_head_at_declaration": git("rev-parse", "HEAD"), "preflight_sha256": sha(a.preflight),
         "historical": pf["historical"], "inputs": pf["inputs"], "masks": pf["masks"], "s1": pf["s1"],
         "estimator": {"name": "W142 background rotation + GRIC observability", "module_doc": BR.__doc__,
                       "params": {k: getattr(BR, k) for k in dir(BR) if k.isupper()},
                       "may_consume": ["RGB frames", "RGB timestamps", "approved intrinsics (fr3 published / synthetic exact / S1 NOMINAL)",
                                       "RGB-derived keep masks (W141 person masks; S1 also letterbox)"],
                       "never": ["GT pose / scale", "depth", "IMU", "oracle masks (except labelled synthetic oracle diagnostics)", "SfM outputs"],
                       "competing_hypotheses": "per edge: pure rotation (incl. no motion) vs non-rotational planar warp (homography) vs general "
                                               "motion with parallax (essential); K-free: homography vs fundamental",
                       "translation_policy": "value always null; status UNRESOLVED_BELOW_PARALLAX_DETECTABILITY (with a p90 residual / f bound), "
                                             "TRANSLATION_DIRECTION_OBSERVED_SCALE_UNOBSERVABLE or INSUFFICIENT_EVIDENCE; never a measured zero",
                       "routing": "after estimation and before GT, frames whose regime window is TRANSLATING are routed to the frozen 6DoF SfM "
                                  "estimate where it registered the frame; other frames get orientation-only state; nothing is discarded"},
         "runs": {r[0]: {"input": r[1], "mask": r[2], "dataset": r[3], "role": r[4]} for r in RUNS},
         "sfm_runs": {r[0]: {"input": r[1], "mask": r[2], "estimator": "frozen W140 COLMAP via the W141 masked wrapper"} for r in SFM_RUNS},
         "synthetic_trajectories": TRAJ,
         "gt": {"real": "W141 rule: linear position + SLERP between bracketing GT samples (gap <= 50 ms), W141 duplicate-timestamp removal",
                "synthetic": "GT sampled exactly at frame timestamps: exact timestamp match (|dt| < 1e-6 s)",
                "access": "evaluate stage only, after all W142 estimate.json, SfM estimates and routing files are frozen by SHA-256"},
         "evaluation": {"orientation_alignment": "ONE evaluation-only rotation R_a = chordal argmin sum ||R_a R_est_i - R_gt_i||_F over the "
                                                 "condition's valid associated frames; no position-based alignment is used for rotation",
                        "relative": "consecutive selected frames and ~1 s pairs: angle((R_est_i^T R_est_j)^T (R_gt_i^T R_gt_j)); no alignment",
                        "constant_camera_baseline": "R_est = I for every frame, evaluated with the same alignment (= GT deviation from its chordal "
                                                    "mean) and the same relative pairs (= GT relative rotation angle)",
                        "coverage": "orientation status ESTIMATED / all selected frames; UNSUPPORTED, OTHER_COMPONENT and INSUFFICIENT_EVIDENCE "
                                    "count as not covered; metrics also on the frames common to compared conditions",
                        "regime_labels_for_evaluation": f"GT TRANSLATING if GT centre displacement within +-{BR.WINDOW_S} s >= {G['gt_label_disp_m']} m; "
                                                        f"ROTATING if GT rotation range >= {G['gt_label_rot_deg']} deg; else NEAR_STATIC",
                        "false_translation": "edges classified TRANSLATION_OBSERVED or TRANSLATION_PRESENT_PLANAR_AMBIGUOUS when GT translation is "
                                             "exactly zero (A1, A2)",
                        "no_ate": "no pseudo-positions are created; ATE is not computed for this estimator",
                        "historical_sfm": "W140/W141 and the new SfM runs are re-evaluated with the same orientation alignment (frozen outputs, new metric only)"},
         "gates": {"synthetic (primary rgbmask)": f"A1 aligned orientation error MAX < {G['syn_stationary_false_rot_max_deg']} deg; A2 aligned median < "
                                                  f"{G['syn_rotation_median_deg']} deg; A1 and A2 false-translation edge fraction <= "
                                                  f"{G['syn_false_translation_edge_frac']}; no numeric translation anywhere; A1 sequence regime "
                                                  "NEAR_STATIC, A2 ROTATING; A4 frames 20-69 TRANSLATING fraction >= "
                                                  f"{G['syn_A4_translation_segment_frac']}; 0 keypoints on excluded pixels",
                   "walking_static (primary rgbmask)": f"orientation coverage >= {G['static_coverage']}; aligned median < {G['static_median_deg']} deg; "
                                                       "aligned median AND 1 s relative median both below the constant-camera baseline on the same "
                                                       f"frames; fraction of consecutive relative errors > {G['static_spike_deg']} deg <= "
                                                       f"{G['static_spike_frac_max']}",
                   "walking_xyz control (primary rgbmask)": f"sequence regime TRANSLATING; NEAR_STATIC frame fraction <= {G['xyz_near_static_frac_max']}; "
                                                            "no numeric translation; the historical W141 6DoF estimate verified and routed for TRANSLATING frames",
                   "classification": "SUPPORTED: synthetic + walking_static + walking_xyz gates pass. NOT_YET_SUPPORTED: walking_static coverage < "
                                     f"{G['partial_static_coverage']} or neither orientation metric beats the constant-camera baseline. "
                                     "PARTIALLY_SUPPORTED otherwise. BLOCKED: a prerequisite fails. A3 and S1 are reported, not gated."},
         "gate_values": G,
         "budget": {"primary_estimation_h_max": 2.0, "scratch_gb_max": 15.0, "parallel_processes": PARALLEL, "training": "none", "downloads": "none"},
         "pre_declaration_disclosures": [
             "dev smoke of the W142 estimator on the first 40 W140 clean synthetic frames (dev/smoke_out, no masks): code paths and runtime "
             "only (0.23 s per pair); no GT score was computed; no parameter was changed afterwards",
             "the S1 clip was located through the W125 action sidecar (source_record) and one frame was viewed to identify letterbox bars; "
             "the letterbox rule (rows whose median-frame max < 16, +4 px) was fixed before any estimate"],
         "instrument_sha256_lf": {f: lf_sha(f) for f in INSTRUMENTS}}
    CP.dump(out / "declaration.json", d)
    print(json.dumps({"declaration_sha256": sha(out / "declaration.json")}, indent=1))
    return 0


# ------------------------------------------------------------------------------------------------ estimate ---
def route(est, sfm):
    """Pre-GT combination: frames in a TRANSLATING window take the frozen 6DoF SfM pose where it registered them."""
    sf = {f["frame_id"]: f for f in sfm["frames"]} if sfm else {}
    out = []
    for f in est["frames"]:
        s = sf.get(f["frame_id"])
        if f["regime_window"]["regime"] == "TRANSLATING":
            src = "SFM_6DOF" if s and s["status"] == "ESTIMATED" else "ORIENTATION_ONLY_SFM_MISSING"
        else:
            src = "ORIENTATION_ONLY"
        out.append({"frame_id": f["frame_id"], "regime": f["regime_window"]["regime"], "source": src,
                    "sfm_status": s["status"] if s else None, "orientation_status": f["orientation"]["status"]})
    c = {}
    for r in out:
        c[r["source"]] = c.get(r["source"], 0) + 1
    return {"frames": out, "counts": c}


def stage_estimate(a):
    d = check_declaration(a)
    pf = load(a.preflight)
    if sha(a.preflight) != d["preflight_sha256"]:
        raise SystemExit("STOP preflight differs")
    out = new_dir(a.out_dir)
    res = {"stage": "estimate", "time": now(), "head": git("rev-parse", "HEAD"), "runs": {}, "sfm_runs": {}}

    def one(r):
        name, inp, mk, ds, role = r
        cmd = [sys.executable, "-I", EST142, "--input", pf["inputs"][inp]["dir"], "--out", out / name, "--mask-label", mk or "none"]
        if mk:
            cmd += ["--keep-masks", pf["masks"][inp]["dirs"][mk]]
        rc, dt = run(cmd, out / f"{name}.log")
        ep = out / name / "estimate.json"
        return name, {"rc": rc, "wall_s": dt, "estimate_sha256": sha(ep) if ep.is_file() else None}

    def one_sfm(r):
        name, inp, mk = r
        rc, dt = run([PC_PY, "-I", EST141, "--input", pf["inputs"][inp]["dir"], "--out", out / name, "--keep-masks", pf["masks"][inp]["dirs"][mk]],
                     out / f"{name}.log")
        ep = out / name / "estimate.json"
        return name, {"rc": rc, "wall_s": dt, "estimate_sha256": sha(ep) if ep.is_file() else None}
    t0 = time.perf_counter()
    with ThreadPoolExecutor(PARALLEL) as ex:
        for name, v in ex.map(one, RUNS):
            res["runs"][name] = v
            print(name, v["rc"], round(v["wall_s"], 1), flush=True)
            CP.dump(out / "estimate_result.json", res)
    res["w142_estimation_wall_s"] = time.perf_counter() - t0
    t1 = time.perf_counter()
    for r in SFM_RUNS:
        name, v = one_sfm(r)
        res["sfm_runs"][name] = v
        print(name, v["rc"], round(v["wall_s"], 1), flush=True)
    res["sfm_wall_s"] = time.perf_counter() - t1
    routing = {}
    for ds, (src, nm) in ROUTE_SFM.items():
        est = load(out / f"{ds}_rgbmask" / "estimate.json")
        sp = (W141 / "results/estimate" / nm / "estimate.json") if src == "w141" else (out / nm / "estimate.json")
        sfm = load(sp) if sp.is_file() else None
        routing[ds] = {"sfm_source": str(sp), "sfm_sha256": sha(sp) if sp.is_file() else None, **route(est, sfm)}
    CP.dump(out / "routing.json", routing)
    res["routing_sha256"] = sha(out / "routing.json")
    CP.dump(out / "estimate_result.json", res)
    print(json.dumps({"w142_wall_s": res["w142_estimation_wall_s"], "sfm_wall_s": res["sfm_wall_s"],
                      "routing": {k: v["counts"] for k, v in routing.items()}}, indent=1))
    return 0


# ------------------------------------------------------------------------------------------------ evaluate ---
def gt_for(ds, pf):
    if ds in ("walking_static", "walking_xyz"):
        return DP.read_groundtruth_dedup(W141 / "data" / "extracted" / f"rgbd_dataset_freiburg3_{ds}" / "groundtruth.txt")[0], "real"
    return CP.read_groundtruth(pf["inputs"][ds]["gt"]), "synthetic"


def associate(gt, kind, t):
    if kind == "synthetic":
        k = int(np.argmin(np.abs(gt[:, 0] - t)))
        return CP.T_WC_from_tum(gt[k, 1:4], gt[k, 4:8]) if abs(gt[k, 0] - t) < 1e-6 else None
    return CP.interpolate_gt(gt, t)[0]


def orient_eval(frames_status, R_est, Tg, times, ids):
    """frames_status: per frame valid flag; R_est per frame (or None); Tg per frame GT (or None)."""
    n = len(ids)
    ok = [k for k in range(n) if frames_status[k] and R_est[k] is not None and Tg[k] is not None]
    rep = {"n_selected": n, "n_valid": sum(frames_status), "n_valid_associated": len(ok), "coverage": sum(frames_status) / n if n else 0.0}
    if len(ok) < 3:
        rep["uncomputable"] = "fewer than 3 valid associated frames"
        return rep, ok
    Rg = np.asarray([Tg[k][:3, :3] for k in ok])
    Re = np.asarray([R_est[k] for k in ok])
    t = np.asarray([times[k] for k in ok])
    Ra, err = BR.orientation_errors(Rg, Re)
    _, berr = BR.orientation_errors(Rg, np.broadcast_to(np.eye(3), Rg.shape))
    rc, rcb, pc = BR.relative_errors(t, Rg, Re)
    r1, r1b, p1 = BR.relative_errors(t, Rg, Re, dt=1.0)
    rep.update({"aligned": {"median_deg": float(np.median(err)), "p90_deg": float(np.percentile(err, 90)), "max_deg": float(err.max()),
                            "per_frame": err.tolist(), "R_align_angle_deg": BR.angle_deg(Ra)},
                "baseline_aligned": {"median_deg": float(np.median(berr)), "p90_deg": float(np.percentile(berr, 90)), "max_deg": float(berr.max()),
                                     "per_frame": berr.tolist()},
                "relative_consecutive": {"median_deg": float(np.median(rc)) if len(rc) else None, "p90_deg": float(np.percentile(rc, 90)) if len(rc) else None,
                                         "baseline_median_deg": float(np.median(rcb)) if len(rcb) else None,
                                         "frac_gt_spike": float((rc > G["static_spike_deg"]).mean()) if len(rc) else None,
                                         "spikes": [{"frame_ids": [ids[ok[i]], ids[ok[j]]], "t": float(t[j]), "err_deg": float(e)}
                                                    for (i, j), e in zip(pc, rc) if e > max(G["spike_deg_report"], 5 * float(np.median(rc)))],
                                         "per_pair": rc.tolist(), "baseline_per_pair": rcb.tolist()},
                "relative_1s": {"median_deg": float(np.median(r1)) if len(r1) else None, "p90_deg": float(np.percentile(r1, 90)) if len(r1) else None,
                                "baseline_median_deg": float(np.median(r1b)) if len(r1b) else None, "n": int(len(r1))},
                "drift_slope_deg_per_s": float(np.polyfit(t - t[0], err, 1)[0]) if np.ptp(t) > 0 else None,
                "valid_ids": [ids[k] for k in ok], "times": t.tolist()})
    return rep, ok


def eval_w142(est, gt, kind, gate_ds):
    frames = est["frames"]
    ids = [f["frame_id"] for f in frames]
    times = [f["timestamp"] for f in frames]
    Tg = [associate(gt, kind, t) for t in times]
    st = [f["orientation"]["status"] == "ESTIMATED" for f in frames]
    Re = [np.asarray(f["orientation"]["R_WC"]) if f["orientation"]["R_WC"] is not None else None for f in frames]
    rep, ok = orient_eval(st, Re, Tg, times, ids)
    # edges vs GT
    ed = []
    for e in est["edges"]:
        Ti, Tj = Tg[e["i"]], Tg[e["j"]]
        if Ti is None or Tj is None:
            continue
        Grel = Ti[:3, :3].T @ Tj[:3, :3]
        tg_i = Ti[:3, :3].T @ (Tj[:3, 3] - Ti[:3, 3])
        r = {"i": e["i"], "j": e["j"], "span": e["span"], "state": e["state"], "gt_rot_deg": BR.angle_deg(Grel), "gt_trans_m": float(np.linalg.norm(tg_i))}
        if e.get("rotation_valid"):
            r["rel_err_deg"] = float(CP.geodesic_deg(np.asarray(e["M"]).T, Grel))
        if e.get("translation_direction_in_cam_i") and np.linalg.norm(tg_i) > 0.01:
            r["dir_err_deg"] = float(np.degrees(np.arccos(np.clip(np.dot(e["translation_direction_in_cam_i"], tg_i / np.linalg.norm(tg_i)), -1, 1))))
        ed.append(r)
    rep["edges_vs_gt"] = {}
    for s in BR.STATES:
        es = [r for r in ed if r["state"] == s]
        rep["edges_vs_gt"][s] = {"n": len(es), "gt_rot_median_deg": float(np.median([r["gt_rot_deg"] for r in es])) if es else None,
                                 "gt_trans_median_m": float(np.median([r["gt_trans_m"] for r in es])) if es else None,
                                 "rel_err_median_deg": float(np.median([r["rel_err_deg"] for r in es if "rel_err_deg" in r])) if any("rel_err_deg" in r for r in es) else None,
                                 "dir_err_median_deg": float(np.median([r["dir_err_deg"] for r in es if "dir_err_deg" in r])) if any("dir_err_deg" in r for r in es) else None}
    zero_t = [r for r in ed if r["gt_trans_m"] < 1e-9]
    rep["false_translation"] = {"n_edges_gt_zero_translation": len(zero_t), "n_claimed": sum(r["state"] in ("TRANSLATION_OBSERVED", "TRANSLATION_PRESENT_PLANAR_AMBIGUOUS") for r in zero_t)}
    rep["false_translation"]["frac"] = rep["false_translation"]["n_claimed"] / len(zero_t) if zero_t else None
    nonstat = [r for r in ed if r["state"] == "NO_DETECTABLE_MOTION"]
    rep["no_detectable_motion_edges"] = {"n": len(nonstat), "gt_rot_p90_deg": float(np.percentile([r["gt_rot_deg"] for r in nonstat], 90)) if nonstat else None,
                                         "n_gt_rot_over_0p5deg": sum(r["gt_rot_deg"] > 0.5 for r in nonstat)}
    # regimes vs GT labels
    gi = [k for k in range(len(frames)) if Tg[k] is not None]
    lab = BR.gt_regime_labels([times[k] for k in gi], [Tg[k][:3, 3] for k in gi], [Tg[k][:3, :3] for k in gi], G["gt_label_disp_m"], G["gt_label_rot_deg"])
    conf = {}
    for k, l in zip(gi, lab):
        key = f"{l}->{frames[k]['regime_window']['regime']}"
        conf[key] = conf.get(key, 0) + 1
    rep["regime_confusion"] = conf
    rep["regime_frames"] = [f["regime_window"]["regime"] for f in frames]
    rep["gt_labels"] = {ids[k]: l for k, l in zip(gi, lab)}
    rep["sequence_regime"] = est["sequence_regime"]
    rep["numeric_translation_frames"] = sum(f["translation"]["value"] is not None for f in frames)
    rep["kp_on_excluded_total"] = est["kp_on_excluded_total"]
    rep["edge_state_counts"] = est["edge_state_counts"]
    rep["orientation_status_counts"] = est["orientation_status_counts"]
    rep["timing_s"] = est["timing_s"]
    rep["rotation_averaging"] = est["rotation_averaging"]
    rep["gt_motion"] = None
    if gi:
        Pg = np.asarray([Tg[k][:3, 3] for k in gi])
        Rg = np.asarray([Tg[k][:3, :3] for k in gi])
        rep["gt_motion"] = {"max_dev_from_mean_m": float(np.linalg.norm(Pg - Pg.mean(0), axis=1).max()),
                            "rot_range_deg": float(CP.geodesic_deg(Rg[:, None], Rg[None]).max()),
                            "path_m": float(np.sum(np.linalg.norm(np.diff(Pg, axis=0), axis=1)))}
    return rep


def eval_sfm(est_frames, gt, kind):
    ids = [f["frame_id"] for f in est_frames]
    times = [f["timestamp"] for f in est_frames]
    Tg = [associate(gt, kind, t) for t in times]
    st = [f["status"] == "ESTIMATED" for f in est_frames]
    Re = [np.asarray(f["T_WC"])[:3, :3] if f["status"] == "ESTIMATED" else None for f in est_frames]
    rep, _ = orient_eval(st, Re, Tg, times, ids)
    return {k: v for k, v in rep.items() if k not in ("valid_ids",)}


def feature_accounting(est_dir, name, inp, pf):
    """Background-feature accounting: keypoints per frame, best-model inliers inside the RGB person mask (contamination proxy)."""
    import cv2
    est = load(est_dir / name / "estimate.json")
    F = np.load(est_dir / name / "features.npz")
    Eg = np.load(est_dir / name / "edges.npz")
    raw_dir = (Path(pf["masks"][inp]["dirs"]["rgbmask"]).parent / "raw") if "rgbmask" in pf["masks"].get(inp, {}).get("dirs", {}) else None
    raws = {}
    tot_inl, in_person = 0, 0
    for e in est["edges"]:
        if e["span"] != 1:
            continue
        key = f"e{e['i']}_{e['j']}"
        mm = Eg[key]
        if raw_dir is None or not len(mm):
            continue
        f = est["frames"][e["i"]]
        if e["i"] not in raws:
            raws[e["i"]] = cv2.imread(str(raw_dir / f["image"]), cv2.IMREAD_UNCHANGED) > 127
        xy = F[f"f{e['i']}"][mm[:, 0]]
        tot_inl += len(xy)
        in_person += int(BR.on_keep(xy, raws[e["i"]]).sum())
    kp = [len(F[f"f{k}"]) for k in range(len(est["frames"]))]
    return {"kp_median": float(np.median(kp)), "kp_min": int(min(kp)), "frames_kp_lt_100": int(sum(k < 100 for k in kp)),
            "span1_inliers_total": tot_inl, "span1_inliers_on_person_mask": in_person,
            "span1_inlier_person_fraction": in_person / tot_inl if tot_inl else None,
            "span1_inliers_median": float(np.median([e["n_inliers"][e["best_calibrated_model"]] for e in est["edges"] if e["span"] == 1 and "n_inliers" in e]))
            if any(e["span"] == 1 and "n_inliers" in e for e in est["edges"]) else None,
            "kp_on_excluded_total": est["kp_on_excluded_total"]}


def stage_evaluate(a):
    d = check_declaration(a)
    pf = load(a.preflight)
    E = Path(a.estimate_dir)
    er = load(E / "estimate_result.json")
    out = new_dir(a.out_dir)
    res = {"stage": "evaluate", "time": now(), "frozen_verified": {}, "runs": {}, "sfm": {}, "historical_sfm": {}}
    for name, v in {**er["runs"], **er["sfm_runs"]}.items():
        ep = E / name / "estimate.json"
        res["frozen_verified"][name] = bool(ep.is_file() and sha(ep) == v["estimate_sha256"])
    res["frozen_verified"]["routing"] = sha(E / "routing.json") == er["routing_sha256"]
    if not all(res["frozen_verified"].values()):
        raise SystemExit(f"STOP estimates not frozen {res['frozen_verified']}")
    gts = {}
    for name, inp, mk, ds, role in RUNS:
        est = load(E / name / "estimate.json")
        r = {"role": role, "dataset": ds, "mask": mk, "sequence_regime": est["sequence_regime"], "edge_state_counts": est["edge_state_counts"],
             "orientation_status_counts": est["orientation_status_counts"], "timing_s": est["timing_s"]}
        if ds != "s1":
            if ds not in gts:
                gts[ds] = gt_for(ds, pf)
            r.update(eval_w142(est, *gts[ds], ds))
        else:
            r["s1_diagnostics"] = s1_diag(est)
        r["features"] = feature_accounting(E, name, inp, pf)
        res["runs"][name] = r
    for name, inp, mk in SFM_RUNS:
        e = load(E / name / "estimate.json")
        res["sfm"][name] = eval_sfm(e["frames"], *gts[inp]) if inp in gts else None
    for ds, names in HIST_SFM.items():
        for n in names:
            e = load(W141 / "results/estimate" / n / "estimate.json")
            res["historical_sfm"][f"w141_{n}"] = {"label": "W141 frozen estimate re-evaluated with the W142 orientation alignment (new metric only)",
                                                   **eval_sfm(e["frames"], *gts[ds])}
    res["routing"] = {k: {"counts": v["counts"], "sfm_source": v["sfm_source"]} for k, v in load(E / "routing.json").items()}
    res["gates"] = gates(res)
    CP.dump(out / "evaluation.json", res)
    summ = {}
    for n, r in res["runs"].items():
        s = {"seq_regime": r["sequence_regime"]["regime"], "orient": r["orientation_status_counts"], "states": r["edge_state_counts"]}
        if "aligned" in r:
            s.update({"cov": round(r["coverage"], 3), "aligned_med": round(r["aligned"]["median_deg"], 3), "aligned_p90": round(r["aligned"]["p90_deg"], 3),
                      "aligned_max": round(r["aligned"]["max_deg"], 3), "base_med": round(r["baseline_aligned"]["median_deg"], 3),
                      "rel1s": r["relative_1s"]["median_deg"], "rel1s_base": r["relative_1s"]["baseline_median_deg"],
                      "relc": r["relative_consecutive"]["median_deg"], "relc_base": r["relative_consecutive"]["baseline_median_deg"],
                      "spikefrac": r["relative_consecutive"]["frac_gt_spike"], "false_T": r["false_translation"], "conf": r["regime_confusion"]})
        s["contam"] = r["features"]["span1_inlier_person_fraction"]
        summ[n] = s
    print(json.dumps({"runs": summ, "sfm": {k: ({kk: v.get(kk) for kk in ("coverage",)} | ({"med": v["aligned"]["median_deg"]} if v and "aligned" in v else {})) if v else None
                                            for k, v in res["sfm"].items()},
                      "historical": {k: {"cov": v["coverage"], **({"med": v["aligned"]["median_deg"], "base": v["baseline_aligned"]["median_deg"]} if "aligned" in v else {})}
                                     for k, v in res["historical_sfm"].items()},
                      "routing": res["routing"], "gates": res["gates"]}, indent=1, default=str))
    return 0


def s1_diag(est):
    e1 = [e for e in est["edges"] if e["span"] == 1 and "gric" in e]
    kf = {}
    for e in est["edges"]:
        if "best_kfree_model" in e:
            kf.setdefault(e["span"], {}).setdefault(e["best_kfree_model"], 0)
            kf[e["span"]][e["best_kfree_model"]] += 1
    return {"calibration_status": est["calibration_status"],
            "kfree_model_by_span": kf,
            "flow_raw_median_px_span1": float(np.median([e["flow_raw_median_px"] for e in e1])) if e1 else None,
            "rot_residual_median_px_span1": float(np.median([e["rot_residual_median_px"] for e in e1 if e.get("rot_residual_median_px") is not None])) if e1 else None,
            "nominal_rotation_deg_span1_median": float(np.median([e["rotation_deg"] for e in e1 if e.get("rotation_deg") is not None])) if e1 else None,
            "note": "rotation magnitudes use the NOMINAL focal and are not calibrated measurements"}


def gates(res):
    R = res["runs"]
    g = {}
    a1, a2, a4 = R["syn_A1_rgbmask"], R["syn_A2_rgbmask"], R["syn_A4_rgbmask"]
    a4_trans = [r for fid, r in enumerate(a4["regime_frames"]) if 20 <= fid <= 69]      # A4 frame_id == index (W140 segments)
    syn = {"A1_false_rot_max_deg": a1.get("aligned", {}).get("max_deg"), "A2_median_deg": a2.get("aligned", {}).get("median_deg"),
           "A1_false_T_frac": a1["false_translation"]["frac"], "A2_false_T_frac": a2["false_translation"]["frac"],
           "A1_regime": a1["sequence_regime"]["regime"], "A2_regime": a2["sequence_regime"]["regime"],
           "A4_translation_segment_frac": sum(r == "TRANSLATING" for r in a4_trans) / len(a4_trans),
           "numeric_translation_frames": sum(r.get("numeric_translation_frames", 0) for r in R.values()),
           "kp_on_excluded_total": sum(r["features"]["kp_on_excluded_total"] for r in R.values())}
    syn["pass"] = bool(syn["A1_false_rot_max_deg"] is not None and syn["A1_false_rot_max_deg"] < G["syn_stationary_false_rot_max_deg"]
                       and syn["A2_median_deg"] is not None and syn["A2_median_deg"] < G["syn_rotation_median_deg"]
                       and syn["A1_false_T_frac"] is not None and syn["A1_false_T_frac"] <= G["syn_false_translation_edge_frac"]
                       and syn["A2_false_T_frac"] is not None and syn["A2_false_T_frac"] <= G["syn_false_translation_edge_frac"]
                       and syn["A1_regime"] == "NEAR_STATIC_TRANSLATION_UNRESOLVED" and syn["A2_regime"] == "ROTATING_TRANSLATION_UNRESOLVED"
                       and syn["A4_translation_segment_frac"] >= G["syn_A4_translation_segment_frac"]
                       and syn["numeric_translation_frames"] == 0 and syn["kp_on_excluded_total"] == 0)
    g["synthetic"] = syn
    s = R["walking_static_rgbmask"]
    st = {"coverage": s["coverage"]}
    if "aligned" in s:
        st.update({"median_deg": s["aligned"]["median_deg"], "baseline_median_deg": s["baseline_aligned"]["median_deg"],
                   "rel1s_median_deg": s["relative_1s"]["median_deg"], "rel1s_baseline_deg": s["relative_1s"]["baseline_median_deg"],
                   "spike_frac": s["relative_consecutive"]["frac_gt_spike"]})
        st["beats_aligned"] = bool(st["median_deg"] < st["baseline_median_deg"])
        st["beats_rel1s"] = bool(st["rel1s_median_deg"] is not None and st["rel1s_median_deg"] < st["rel1s_baseline_deg"])
        st["pass"] = bool(st["coverage"] >= G["static_coverage"] and st["median_deg"] < G["static_median_deg"] and st["beats_aligned"]
                          and st["beats_rel1s"] and st["spike_frac"] is not None and st["spike_frac"] <= G["static_spike_frac_max"])
    else:
        st.update({"beats_aligned": False, "beats_rel1s": False, "pass": False})
    g["walking_static"] = st
    x = R["walking_xyz_rgbmask"]
    ns = sum(r == "NEAR_STATIC_TRANSLATION_UNRESOLVED" for r in x["regime_frames"]) / len(x["regime_frames"])
    rt = res["routing"]["walking_xyz"]["counts"]
    xc = {"sequence_regime": x["sequence_regime"]["regime"], "near_static_frac": ns, "numeric_translation_frames": x["numeric_translation_frames"],
          "routing": rt, "historical_6dof_verified": res["frozen_verified"].get("routing", False)}
    xc["pass"] = bool(xc["sequence_regime"] == "TRANSLATING" and ns <= G["xyz_near_static_frac_max"] and xc["numeric_translation_frames"] == 0
                      and rt.get("SFM_6DOF", 0) > 0)
    g["walking_xyz"] = xc
    if not (st.get("coverage", 0) >= G["partial_static_coverage"] and (st["beats_aligned"] or st["beats_rel1s"])):
        j = "NOT_YET_SUPPORTED"
    elif syn["pass"] and st["pass"] and xc["pass"]:
        j = "SUPPORTED"
    else:
        j = "PARTIALLY_SUPPORTED"
    g["judgment"] = j
    return g


# ------------------------------------------------------------------------------------------------ visualize ---
def stage_visualize(a):
    import w142_visualize as V
    check_declaration(a)
    return V.visualize(a, ROOT, W141, W142, RUNS)


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
