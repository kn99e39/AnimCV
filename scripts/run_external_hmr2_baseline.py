#!/usr/bin/env python3
"""EXPERIMENT (Worklog 97): official 4DHumans / HMR 2.0b as an external RGB baseline against frozen S0.

Subcommands (``PREDECLARED`` is fixed before any HMR 2.0 prediction exists):
  population  (AnimCV env)  the evaluation rows = the W95 rows (3DPW VAL/TEST, MPI-INF-3DHP), each with its RGB path and
                            a person box built ONLY from the dataset-provided 2D detections that S0 consumed (no GT).
  infer       (animcv-w97-hmr2 image, GPU)  upstream ``load_hmr2`` + ``ViTDetDataset`` with those boxes; one image at a
                            time, no temporal information; saves predicted SMPL rotations, betas, camera and the 24 SMPL
                            joints (``smplx.SMPLLayer`` forward of the predicted parameters).
  evaluate    (AnimCV env)  S (frozen S0) vs E (HMR 2.0b) on identical rows in AnimCV canonical camera axes; GT only for
                            scoring; conditional common-rig replay S+Z vs E+Z through the unchanged W95 leg IK.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

PREDECLARED = {
    "model": {"repo": "https://github.com/shubham-goel/4D-Humans", "commit": "efe18deff163b29dff87ddbd575fa29b716a356c",
              "checkpoint": "logs/train/multiruns/hmr2/0/checkpoints/epoch=35-step=1000000.ckpt (HMR2.0b, upstream default)",
              "checkpoint_sha256": "c9ea09c834161468638c9de7a782c614e300c622709bf52967274c543ec72fcf",
              "archive_sha256": "0fdf9e66ec97503fe1b995f4942e021a6df748f4ccbc5574718742d975a6e19b",
              "smpl": "basicModel_neutral_lbs_10_207_0_v1.0.0.pkl (sha256 98e65c74ad9b998783132f00880d1025a8d64b158e040e6ef13a557e5098bc42), "
                      "converted by upstream check_smpl_exists (array-identical)"},
    "populations": {
        "main_selection_free": "every W95 3DPW TEST row (identical sequence/actor/frame rows as W93-W96); no sampling",
        "secondary_selection_free": "every W95 3DPW VALIDATION row",
        "challenge_gt_labelled": "subsets of the TEST rows defined by evaluation-only GT labels (reported separately): gravity "
                                 "waist_bend, other (seated/crouched/kneeling), GT knee flexion >= 45 deg, GT knee flexion < 5 deg, "
                                 "S0 knee plane > 45 deg from GT (W96 wrong-plane rows), out-of-plane limbs (GT thigh or upper-arm "
                                 "depth component > 0.5 of its length)",
        "cross_domain": "every W95 MPI-INF-3DHP row (secondary; HMR 2.0 trained on 3DHP train S1-S8, same studio as TS1-4; the "
                        "3DHP joint definitions differ from SMPL joints, so only limb-joint metrics are primary there)",
    },
    "box": {"source": "bank input_2d (dataset-provided 2D detections; the same observations S0 consumed), normalized x/W, y/H",
            "rule": "tight box of joints with input_valid, enlarged by 1.2 about its centre, passed as (x1,y1,x2,y2) to the "
                    "official ViTDetDataset (which expands it to the 192:256 aspect)", "min_valid_joints": 6,
            "invalid": "fewer valid joints -> no box -> E unavailable (coverage)"},
    "canonical": "HMR 2.0 SMPL joints are in OpenCV camera axes (x right, y down, z forward) of the crop camera; AnimCV "
                 "canonical = (x, z, -y); 17 joints from the 24 SMPL joints with the same index map as the 3DPW GT adapter; "
                 "root-relative, metric (no scale, no Procrustes in primary metrics; PA-MPJPE secondary, labelled)",
    "metrics_primary": ["MPJPE root-relative (mm)", "upper-arm/forearm/thigh/shin direction error (deg)",
                        "wrist/ankle position error relative to the pelvis (mm)", "camera-forward depth error of limb joints (mm)",
                        "knee plane error (both GT and prediction flexed >= 20 deg) and wrong-branch rate (> 90 deg)",
                        "torso up/forward axis error (W93 thorax frame)", "pelvis orientation vs SMPL (3DPW): E from predicted "
                        "global orient, E from joints (W94 lower-trunk proxy), S from joints"],
    "material_rule": "E materially stronger on a population if paired E - S has a sequence-cluster 95% interval below 0 AND a "
                     ">= 10% relative reduction, for MPJPE, thigh/shin direction and knee-plane error on the main TEST population "
                     "and on the GT knee flexion >= 45 challenge subset",
    "rig_replay": "W95 two-bone legs, zero twist (Z), root = W95 B_S0 control (SMPL pelvis Object, S0 thorax), J1X arm; only the "
                  "leg joint source changes (S0 vs E); W96 candidate T not used",
    "temporal": "matched adjacent rows; no smoothing of either source; HMR 2.0 sees single frames only",
}
SMPL24_TO_17 = (0, 1, 4, 7, 2, 5, 8, 3, 9, 12, 15, 16, 18, 20, 17, 19, 21)  # same map as the 3DPW GT adapter (W93 NAMES order)
CV_TO_CANON = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])


def person_box(xy_norm, valid, image_size, factor=1.2, min_valid=6):
    """(x1, y1, x2, y2) pixels from the valid normalized 2D detections, or None."""
    v = np.asarray(valid, bool)
    if v.sum() < min_valid:
        return None
    W, H = image_size
    pts = np.asarray(xy_norm, float)[v][:, :2] * np.array([W, H])
    lo, hi = pts.min(0), pts.max(0)
    c, half = (lo + hi) / 2, (hi - lo) / 2 * factor
    return [float(c[0] - half[0]), float(c[1] - half[1]), float(c[0] + half[0]), float(c[1] + half[1])]


def smpl24_to_canonical17(j24):
    """(..., 24, 3) OpenCV-axis SMPL joints -> (..., 17, 3) AnimCV canonical, pelvis-relative."""
    j = np.asarray(j24, float)[..., SMPL24_TO_17, :] @ CV_TO_CANON.T
    return j - j[..., :1, :]


# ------------------------------------------------------------------ population (AnimCV env)

def cmd_population(args):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from framepose.bank import load_bank
    from framepose.bedlam_supervision import ExternalBank
    from pose.framepose_bridge import sequence_frame_positions, sequence_ids_in_split

    rows = []
    bank = load_bank(args.bank)
    for split in ("validation", "test"):
        for sid in sequence_ids_in_split(bank, split):
            for p in sequence_frame_positions(bank, sid):
                s = bank.samples[p]
                rows.append({"population": f"3dpw_{split}", "bank_position": int(p), "sequence_id": sid, "frame_index": int(s.frame_index),
                             "image": "3dpw/" + s.image_reference.relative_path, "image_size": list(s.image_size),
                             "box": person_box(bank.arrays["input_2d"][p], bank.arrays["input_valid"][p], s.image_size)})
    mx = ExternalBank.load(args.hp_bank)
    for p, s in enumerate(mx.samples):
        rows.append({"population": "3dhp", "bank_position": int(p), "sequence_id": s.sequence_id, "frame_index": int(s.frame_index),
                     "image": "3dhp/" + s.image_reference.relative_path, "image_size": list(s.image_size),
                     "box": person_box(mx.arrays["input_2d"][p], mx.arrays["input_valid"][p], s.image_size)})
    out = {"schema": "worklog97_population", "predeclared": PREDECLARED, "rows": rows}
    args.out.write_text(json.dumps(out))
    by = {}
    for r in rows:
        k = r["population"]
        by.setdefault(k, [0, 0])
        by[k][0] += 1
        by[k][1] += r["box"] is not None
    print({k: {"rows": a, "with_box": b} for k, (a, b) in by.items()})


# ------------------------------------------------------------------ inference (HMR 2.0 image)

def cmd_infer(args):
    import cv2
    import smplx
    import torch
    from hmr2.datasets.vitdet_dataset import ViTDetDataset
    from hmr2.models import DEFAULT_CHECKPOINT, load_hmr2

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    data = json.loads(args.population.read_text())
    rows = [r for r in data["rows"] if args.only is None or r["population"] in args.only.split(",")]
    model, cfg = load_hmr2(DEFAULT_CHECKPOINT)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = model.to(device).eval()
    n = len(rows)
    keys = ("global_orient", "body_pose", "betas", "pred_cam", "joints24", "keypoints3d")
    res = {k: None for k in keys}
    valid = np.zeros(n, bool)
    t0 = __import__("time").time()
    for i, r in enumerate(rows):
        if r["box"] is None:
            continue
        img = cv2.imread(str(args.image_root / r["image"]))
        if img is None:
            continue
        ds = ViTDetDataset(cfg, img, np.array([r["box"]], np.float32))
        batch = torch.utils.data.default_collate([ds[0]])
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            out = model(batch)
            sp = out["pred_smpl_params"]
            raw = smplx.SMPLLayer.forward(model.smpl, global_orient=sp["global_orient"], body_pose=sp["body_pose"], betas=sp["betas"])
        vals = {"global_orient": sp["global_orient"][0, 0], "body_pose": sp["body_pose"][0], "betas": sp["betas"][0],
                "pred_cam": out["pred_cam"][0], "joints24": raw.joints[0, :24], "keypoints3d": out["pred_keypoints_3d"][0]}
        for k, v in vals.items():
            a = v.detach().cpu().numpy().astype(np.float32)
            if res[k] is None:
                res[k] = np.full((n,) + a.shape, np.nan, np.float32)
            res[k][i] = a
        valid[i] = True
        if i % 500 == 0:
            print(f"{i}/{n} {__import__('time').time() - t0:.0f}s", flush=True)
    elapsed = __import__("time").time() - t0
    meta = {"rows": n, "valid": int(valid.sum()), "seconds": elapsed, "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "max_memory_allocated_mib": (torch.cuda.max_memory_allocated() / 2 ** 20) if device.type == "cuda" else None,
            "torch": torch.__version__}
    np.savez_compressed(args.out, valid=valid, index=np.array([[r["bank_position"] for r in rows]]), population=np.array([r["population"] for r in rows]),
                        **{k: v for k, v in res.items() if v is not None})
    Path(str(args.out) + ".json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("population")
    p.add_argument("--bank", type=Path, required=True)
    p.add_argument("--hp-bank", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("infer")
    p.add_argument("--population", type=Path, required=True)
    p.add_argument("--image-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--only", default=None)
    p.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    {"population": cmd_population, "infer": cmd_infer}[args.cmd](args)


if __name__ == "__main__":
    main()
