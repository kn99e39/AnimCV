#!/usr/bin/env python3
"""DIAGNOSTIC (Worklog 83): MPI-INF-3DHP official test set -> Worklog 73 ExternalBank (detector input, GT target).

Feasibility checks first (each STOPs the build):
  * annotation <-> image identity: the nominal mapping is row k -> ``img_{k+1:06d}.jpg``.  A constant
    per-sequence image offset is estimated from the re-run detector's agreement with ``annot2`` (identity
    evidence only, no candidate model involved) on probe rows (every 10th valid row), offsets -10..+10:
    the nominal offset 0 is kept unless the mean agreement there is > 5% worse than the curve minimum, in
    which case the argmin is used; the first- and second-half argmins must lie within +-1 frame of the
    chosen offset, and the agreement 3 frames away on both sides must be >= 25% worse (identity resolved to
    +-1 frame); otherwise STOP.  (Worklog 83 first assumed offset 0; TS3 contradicted it, then the 50 fps
    TS5 curve was flat over +-1 frame, so the gate was amended twice -- both before any candidate prediction.)
    Every used image has the calibrated size; valid rows whose offset image does not exist are dropped and
    counted;
  * camera convention: ``annot3`` (camera mm, OpenCV axes, +y down) is a pinhole camera-frame representation
    of ``annot2``: a per-sequence least-squares pinhole fit must leave <= 2 px median residual with focal
    lengths within 1% and principal point within 10 px of the official calibration (TS2's annotation uses
    slightly different intrinsics than the single listed studio camera, so the official-intrinsics
    reprojection is reported, not required).  ``annot2`` is undistorted; the listed TS5/TS6 lens distortion is
    reported, and on TS5/TS6 probe rows the detector is also compared with the DISTORTED projection, which
    tells whether the shipped images are undistorted.  Intrinsics never enter an input or a metric;
  * left/right: the detector's left wrist lies nearer GT left than GT right on associated rows.
Then for every official ``valid_frame`` row: the re-run OpenPose COCO-18 detector on the full image, the
unchanged Worklog 73 GT-2D identity association (GT used ONLY to pick the subject among detections), the
3DPW input conversion, and the AnimCV canonical target.  Writes ``<out>/mpi3dhp_test.{json,npz}`` and a
build report.  Requires h5py (MATLAB v7.3 annotation files).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from framepose.bedlam_supervision import ASSOCIATION, ExternalBank, ExternalSample, ImageRef, associate
from framepose.mpi3dhp_test_transfer import (
    SEQUENCES, association_uv, canonical_target, fit_pinhole, image_name, parse_calibration, project_opencv, select_frames,
)
from framepose.openpose_coco18 import OpenPoseCOCO18, to_canonical_input

CALIBRATION = {"TS1": "ts1-4cameras.calib", "TS2": "ts1-4cameras.calib", "TS3": "ts1-4cameras.calib",
               "TS4": "ts1-4cameras.calib", "TS5": "ts5-6cameras.calib", "TS6": "ts5-6cameras.calib"}
IMAGE_SIZE = {"TS1": (2048, 2048), "TS2": (2048, 2048), "TS3": (2048, 2048), "TS4": (2048, 2048),
              "TS5": (1920, 1080), "TS6": (1920, 1080)}
MAX_PINHOLE_RESIDUAL_MEDIAN_PX = 2.0
MAX_FOCAL_RELATIVE = 0.01
MAX_PRINCIPAL_PX = 10.0
PROBE_EVERY = 10          # every 10th valid row is an identity probe
PROBE_OFFSETS = (-10, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 10)
MAX_MISSING_IMAGE_SHARE = 0.01
NOMINAL_TOLERANCE = 1.05
RISE_AT_3 = 1.25
LIMB_COCO = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)
COCO_TO_TEST17 = {2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13}


def _read(path: Path):
    data = path.read_bytes()
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR), hashlib.sha256(data).hexdigest()


def _agreement(people, annot2_17, size):
    """Best (over detections) median limb-joint distance to GT 2D, as a fraction of the GT limb-box diagonal."""
    g = np.array([annot2_17[COCO_TO_TEST17[c]] for c in LIMB_COCO])
    diag = float(np.linalg.norm(g.max(0) - g.min(0)))
    best = None
    for person in people:
        kp = person["keypoints"]
        det = np.array([kp[c, 2] > 0 for c in LIMB_COCO])
        if det.sum() < 6:
            continue
        d = float(np.median(np.linalg.norm(kp[list(LIMB_COCO), :2][det] - g[det], axis=1))) / max(diag, 1e-6)
        best = d if best is None else min(best, d)
    return best


def estimate_offset(detector, image_dir, annot2, annot3, rows, size, cal, n_images):
    """Constant image offset from detector-vs-GT-2D agreement (identity evidence only)."""
    probes = []
    for k in rows[::PROBE_EVERY]:
        k = int(k)
        agree, people0 = {}, None
        for o in PROBE_OFFSETS:
            path = image_dir / image_name(k + o)
            if 0 <= k + o < n_images and path.exists():
                people = detector(_read(path)[0])
                agree[o] = _agreement(people, annot2[k], size)
                if o == 0:
                    people0 = people
        distorted = None
        if np.any(cal["dist"]) and people0 is not None:
            distorted = _agreement(people0, project_opencv(annot3[k], cal["K"], cal["dist"]), size)
        probes.append({"row": k, "agreement_by_offset": agree, "agreement_distorted_projection": distorted})

    def curve(part):
        return {o: float(np.mean(v)) for o in PROBE_OFFSETS
                if (v := [p["agreement_by_offset"][o] for p in part if p["agreement_by_offset"].get(o) is not None])}
    half = len(probes) // 2
    curves = {"all": curve(probes), "first_half": curve(probes[:half]), "second_half": curve(probes[half:])}
    best = {k: min(c, key=c.get) for k, c in curves.items()}
    c = curves["all"]
    o = 0 if c.get(0, math.inf) <= NOMINAL_TOLERANCE * c[best["all"]] else best["all"]
    ok = (abs(best["first_half"] - o) <= 1 and abs(best["second_half"] - o) <= 1
          and all((o + d) in c and c[o + d] >= RISE_AT_3 * c[o] for d in (-3, 3)))
    printable = {k: {str(kk): vv for kk, vv in v.items()} for k, v in curves.items()}
    return (o if ok else None), probes, printable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="extracted mpi_inf_3dhp_test_set directory")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import h5py
    import torch

    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit("STOP: output directory is not empty; never overwrite experimental artifacts")
    detector = OpenPoseCOCO18("cuda")
    samples, arrays, drops, report, started = [], {k: [] for k in ExternalBank.ARRAYS}, Counter(), {}, time.time()
    for seq in SEQUENCES:
        mat = args.root / seq / "annot_data.mat"
        with h5py.File(mat, "r") as f:
            annot2 = np.array(f["annot2"])[:, 0]          # (N, 17, 2) pixels
            annot3 = np.array(f["annot3"])[:, 0]          # (N, 17, 3) camera mm
            valid = np.array(f["valid_frame"]).reshape(-1)
        n, size = len(annot2), IMAGE_SIZE[seq]
        image_dir = args.root / seq / "imageSequence"
        n_images = len(list(image_dir.glob("img_*.jpg")))
        cal = parse_calibration((args.root / "test_util/camera_calibration" / CALIBRATION[seq]).read_text(), size)
        rows = select_frames(valid)
        reproj = np.linalg.norm(project_opencv(annot3[rows], cal["K"], np.zeros(5)) - annot2[rows], axis=2)
        reproj_distorted = np.linalg.norm(project_opencv(annot3[rows], cal["K"], cal["dist"]) - annot2[rows], axis=2)
        fit = fit_pinhole(annot3[rows], annot2[rows])
        k_off = cal["K"]
        if (fit["residual_median_px"] > MAX_PINHOLE_RESIDUAL_MEDIAN_PX
                or abs(fit["fx"] / k_off[0, 0] - 1) > MAX_FOCAL_RELATIVE or abs(fit["fy"] / k_off[1, 1] - 1) > MAX_FOCAL_RELATIVE
                or abs(fit["cx"] - k_off[0, 2]) > MAX_PRINCIPAL_PX or abs(fit["cy"] - k_off[1, 2]) > MAX_PRINCIPAL_PX):
            raise SystemExit(f"STOP: {seq}: annotation does not reproject as a pinhole camera frame: {fit}")
        offset, probes, curves = estimate_offset(detector, image_dir, annot2, annot3, rows, size, cal, n_images)
        if offset is None:
            raise SystemExit(f"STOP: {seq}: RGB/annotation identity is not one constant offset: {curves}")
        present = np.array([(image_dir / image_name(int(k) + offset)).exists() for k in rows])
        if (~present).mean() > MAX_MISSING_IMAGE_SHARE:
            raise SystemExit(f"STOP: {seq}: {int((~present).sum())} valid rows have no image at offset {offset}")
        drops[f"{seq}:no_image_at_offset"] += int((~present).sum())
        rows = rows[present]
        kept, lr = Counter(), []
        for k in rows:
            k = int(k)
            rel = f"{seq}/imageSequence/{image_name(k + offset)}"
            img, sha = _read(args.root / rel)
            if (img.shape[1], img.shape[0]) != size:
                raise SystemExit(f"STOP: image size {img.shape[1::-1]} != calibrated {size} for {rel}")
            people = detector(img)
            idx, info = associate(people, association_uv(annot2[k]), size)
            if idx is None:
                drops[f"{seq}:association:{info['reason']}"] += 1
                continue
            kp = people[idx]["keypoints"]
            if kp[7, 2] > 0 and kp[4, 2] > 0:      # COCO 7 = LWrist, 4 = RWrist
                lr.append(bool(np.linalg.norm(kp[7, :2] - annot2[k, 7]) < np.linalg.norm(kp[7, :2] - annot2[k, 4])))
            inp, visible = to_canonical_input(kp, size)
            target, gt_valid = canonical_target(annot3[k])
            target_valid = visible & gt_valid
            if not target_valid[0]:
                drops[f"{seq}:pelvis_invalid"] += 1
                continue
            samples.append(ExternalSample(
                sample_id=f"3dhp:{seq}#row{k:06d}", source="MPI-INF-3DHP", sequence_id=f"3dhp:{seq}", frame_index=k,
                split="test", image_size=size, image_reference=ImageRef("mpi3dhp_test", rel),
                detail={"annotation_row": k, "image_sha256": sha, "association": info, "people_detected": len(people)}))
            arrays["input_2d"].append(inp)
            arrays["input_valid"].append(inp[:, 2] > 0)
            arrays["target_3d"].append(target)
            arrays["target_valid"].append(target_valid)
            kept[seq] += 1
        report[seq] = {"annotation_rows": n, "images": n_images, "image_offset": offset,
                       "image_mapping": f"row k -> img_(k+1{offset:+d})", "valid_frames": int(valid.astype(bool).sum()),
                       "valid_rows_with_image": int(len(rows)),
                       "kept_rows": kept[seq], "reprojection_px": {"median": float(np.median(reproj)),
                                                                   "p95": float(np.percentile(reproj, 95)), "max": float(reproj.max())},
                       "pinhole_fit": fit, "official_intrinsics": {"fx": float(k_off[0, 0]), "fy": float(k_off[1, 1]),
                                                                   "cx": float(k_off[0, 2]), "cy": float(k_off[1, 2])},
                       "reprojection_with_listed_distortion_px": {"median": float(np.median(reproj_distorted)),
                                                                  "p95": float(np.percentile(reproj_distorted, 95))},
                       "lens_distortion_listed": cal["dist"].tolist(),
                       "identity_mean_agreement_by_offset": curves,
                       "distorted_projection_agreement_mean": (float(np.mean([p["agreement_distorted_projection"] for p in probes
                                                                             if p["agreement_distorted_projection"] is not None]))
                                                               if np.any(cal["dist"]) else None),
                       "identity_probes": probes, "left_wrist_nearer_gt_left": [int(sum(lr)), len(lr)],
                       "mat_sha256": hashlib.sha256(mat.read_bytes()).hexdigest()}
        print(json.dumps({"sequence": seq, **{k: v for k, v in report[seq].items() if k != "identity_probes"},
                          "seconds": round(time.time() - started)}), flush=True)
        if lr and sum(lr) / len(lr) < 0.9:
            raise SystemExit(f"STOP: {seq}: left/right convention does not match the detector: {sum(lr)}/{len(lr)}")

    bank = ExternalBank(samples, {k: np.stack(v) for k, v in arrays.items()},
                        {"source": "MPI-INF-3DHP official test set", "association": ASSOCIATION, "detector": detector.identity(),
                         "population": "every official valid_frame row"})
    args.out.mkdir(parents=True, exist_ok=True)
    bank.save(args.out / "mpi3dhp_test")
    full = {"schema": "animcv_w83_mpi3dhp_bank_build_v1", "rows": len(bank), "content_digest": bank.content_digest(),
            "drops": dict(drops), "sequences": report, "seconds": time.time() - started, "torch": torch.__version__}
    (args.out / "mpi3dhp_test_build_report.json").write_text(json.dumps(full, indent=1))
    print(json.dumps({k: v for k, v in full.items() if k != "sequences"}, indent=1))


if __name__ == "__main__":
    main()
