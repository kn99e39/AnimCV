#!/usr/bin/env python3
"""DIAGNOSTIC (Worklog 87): detector-vs-GT 2D discrepancy for every FrameBank joint (observation-quality audit).

GT 2D per source (never a model input; observation-quality measurement only):
  * 3DPW: world SMPL-24 ``jointPositions`` projected through the per-frame ``cam_poses`` (world -> camera) and
    ``cam_intrinsics`` of the official sequence file;
  * MPI-INF-3DHP TEST: ``annot2`` of the Worklog 83 annotation row (TS3 image offset handled by the bank);
  * BEDLAM VALIDATION: ``gtkps`` (SMPL-X body joints) of the Worklog 73 label row.
Discrepancy = |detector - GT| in pixels / diagonal of the GT 2D box over the 12 limb joints (the Worklog 73
association scale), for detector-visible joints (NaN otherwise).  Note: the detector's COCO keypoints and the
datasets' joint definitions differ systematically (e.g. SMPL shoulder centres), so this measures detector
localization AND annotation-definition discrepancy together.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from framepose.bank import load_bank
from framepose.bedlam_supervision import CANONICAL_SMPLX, ExternalBank
from framepose.contract import JOINT_INDEX
from framepose.mpi3dhp_test_transfer import CANONICAL_TEST17

LIMBS = [JOINT_INDEX[n] for n in ("left_shoulder", "left_elbow", "left_wrist", "right_shoulder", "right_elbow", "right_wrist",
                                  "left_hip", "left_knee", "left_ankle", "right_hip", "right_knee", "right_ankle")]


def discrepancy(gt_px, det_norm, det_valid, size):
    w, h = size
    det = np.asarray(det_norm, float)[:, :2] * [w, h]
    box = gt_px[LIMBS]
    diag = float(np.linalg.norm(box.max(0) - box.min(0)))
    d = np.linalg.norm(det - gt_px, axis=1) / max(diag, 1e-6)
    return np.where(det_valid, d, np.nan), diag


def threedpw_gt(bank, raw_root: Path):
    cache, out = {}, np.full((len(bank), 17, 2), np.nan)
    for p, s in enumerate(bank.samples):
        _, seq, actor = s.sequence_id.split(":")
        key = (s.split, seq)
        if key not in cache:
            cache[key] = pickle.load(open(raw_root / s.split / f"{seq}.pkl", "rb"), encoding="latin1")
        r = cache[key]
        a = int(actor.replace("actor", ""))
        joints = np.asarray(r["jointPositions"][a][s.frame_index]).reshape(24, 3)
        cam = np.asarray(r["cam_poses"][s.frame_index])
        pc = joints @ cam[:3, :3].T + cam[:3, 3]
        uv = pc @ np.asarray(r["cam_intrinsics"]).T
        out[p] = (uv[:, :2] / uv[:, 2:3])[CANONICAL_SMPLX]
    return out


def mpi3dhp_gt(bank, root: Path):
    import h5py

    out, cache = np.full((len(bank), 17, 2), np.nan), {}
    for p, s in enumerate(bank.samples):
        seq = s.sequence_id.split(":")[1]
        if seq not in cache:
            with h5py.File(root / seq / "annot_data.mat", "r") as f:
                cache[seq] = np.array(f["annot2"])[:, 0]
        out[p] = cache[seq][s.detail["annotation_row"]][CANONICAL_TEST17]
    return out


def bedlam_gt(bank, labels: Path):
    out, cache = np.full((len(bank), 17, 2), np.nan), {}
    for p, s in enumerate(bank.samples):
        f = s.detail["label_file"]
        if f not in cache:
            cache[f] = np.load(labels / f)["gtkps"]
        out[p] = cache[f][s.detail["label_row"]][:22, :2][CANONICAL_SMPLX]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("framebank", "threedpw_raw", "mpi3dhp_bank", "mpi3dhp_root", "bedlam_val", "bedlam_val_labels", "out"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit("STOP: output exists; never overwrite")
    report = {"schema": "animcv_w87_detector_discrepancy_v1", "populations": {}}
    arrays = {}
    for name, bank, gt in (("3dpw", (fb := load_bank(args.framebank)), None), ("mpi3dhp_test", (mx := ExternalBank.load(args.mpi3dhp_bank)), None),
                           ("bedlam_validation", (bv := ExternalBank.load(args.bedlam_val)), None)):
        gt = threedpw_gt(fb, args.threedpw_raw) if name == "3dpw" else (mpi3dhp_gt(mx, args.mpi3dhp_root) if name == "mpi3dhp_test" else bedlam_gt(bv, args.bedlam_val_labels))
        d = np.full((len(bank), 17), np.nan)
        diag = np.zeros(len(bank))
        for p, s in enumerate(bank.samples):
            d[p], diag[p] = discrepancy(gt[p], bank.arrays["input_2d"][p], bank.arrays["input_valid"][p], tuple(s.image_size))
        arrays[f"{name}_gt2d_px"], arrays[f"{name}_discrepancy"], arrays[f"{name}_limb_diag_px"] = gt, d, diag
        report["populations"][name] = {"rows": len(bank), "bank_content_digest": bank.content_digest(),
                                       "median_discrepancy_by_joint": {n: float(np.nanmedian(d[:, j])) for n, j in JOINT_INDEX.items()}}
        print(json.dumps({name: {k: round(v, 4) for k, v in report["populations"][name]["median_discrepancy_by_joint"].items()}}), flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out / "discrepancy.npz", **arrays)
    (args.out / "discrepancy_report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
