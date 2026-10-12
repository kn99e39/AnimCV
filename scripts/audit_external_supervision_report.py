#!/usr/bin/env python3
"""DIAGNOSTIC (Worklog 73): post-hoc audits requested by the report review (descriptive; nothing re-trained).

1. Qwen extraction parity: the wrapper's re-extraction of the first N FrameBank
   samples (image-path order) vs the Worklog 68 cache, field by field.
2. Domain nearest-neighbour baselines in the arm-endpoint readout space
   (L2-normalized shoulder/elbow/wrist readouts), with same-sequence
   neighbours excluded so the 3DPW baseline is not a temporal near-duplicate.
3. Detection sharing: rows of one image whose detector input is identical
   (two labelled people associated with the same OpenPose skeleton).
4. Cache provenance: per-scene bank parts rebuilt from scratch with the
   committed builder vs the cached parts used for the reported TRAIN bank.
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

from framepose.arm_depth_probe import forward_targets
from framepose.bank import load_bank
from framepose.bedlam_supervision import ExternalBank
from framepose.visual_coreset import arm_features


def _stats(v):
    v = np.asarray(v, float)
    return {"n": int(len(v)), "mean": float(v.mean()), "p05": float(np.percentile(v, 5)), "p50": float(np.median(v)),
            "p95": float(np.percentile(v, 95))} if len(v) else {"n": 0}


def nearest(query, base, query_seq=None, base_seq=None, device="cuda"):
    """Max cosine of each query row against base rows, excluding identical sequence ids when given."""
    import torch

    b = torch.as_tensor(base, device=device, dtype=torch.float32)
    out = []
    bseq = None if base_seq is None else np.asarray(base_seq)
    for s in range(0, len(query), 2048):
        q = torch.as_tensor(query[s:s + 2048], device=device, dtype=torch.float32)
        sim = q @ b.T
        if query_seq is not None:
            same = torch.as_tensor(np.asarray(query_seq[s:s + 2048])[:, None] == bseq[None, :], device=device)
            sim = sim.masked_fill(same, -2.0)
        out += sim.max(1).values.cpu().tolist()
    return _stats(out)


def shared_detections(bank: ExternalBank) -> dict:
    by_image = defaultdict(list)
    for i, s in enumerate(bank.samples):
        by_image[s.image_reference.relative_path].append(i)
    rows_sharing, images_sharing = 0, 0
    for idx in by_image.values():
        if len(idx) < 2:
            continue
        x = bank.arrays["input_2d"][idx]
        dup = np.zeros(len(idx), bool)
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                if np.array_equal(x[a], x[b]):
                    dup[a] = dup[b] = True
        rows_sharing += int(dup.sum())
        images_sharing += int(dup.any())
    return {"rows": len(bank), "images_with_multiple_rows": sum(len(v) > 1 for v in by_image.values()),
            "rows_sharing_a_detection": rows_sharing, "images_with_shared_detection": images_sharing,
            "share": rows_sharing / max(len(bank), 1)}


def compare_parts(cached: Path, rebuilt: Path) -> dict:
    a, b = pickle.loads(cached.read_bytes()), pickle.loads(rebuilt.read_bytes())
    sa, aa, ra, _ = a
    sb, ab, rb, _ = b
    strip = lambda s: (s.sample_id, s.sequence_id, s.frame_index, s.split, tuple(s.image_size),
                       s.image_reference.relative_path, json.dumps(s.detail, sort_keys=True))
    same_samples = [strip(x) for x in sa] == [strip(x) for x in sb]
    same_arrays = all(np.array_equal(np.asarray(aa[k]), np.asarray(ab[k])) for k in ExternalBank.ARRAYS)
    return {"rows_cached": len(sa), "rows_rebuilt": len(sb), "samples_identical": same_samples,
            "arrays_identical": same_arrays, "kept_rows": [ra["kept_rows"], rb["kept_rows"]]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("framebank", "w68_cache", "parity_dir", "external_train", "external_train_cache", "external_val",
                 "external_val_cache", "train_bank_presplit", "out"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--part", action="append", default=[], help="CACHED_PKL=REBUILT_PKL")
    args = parser.parse_args()
    report = {"schema": "animcv_w73_report_audit_v1"}

    # 1. Qwen parity
    w68 = np.load(args.w68_cache / "readout.npz")
    par = np.load(args.parity_dir / "readout.npz")
    pos = np.load(args.parity_dir / "parity_positions.npy")
    report["qwen_parity"] = {
        "samples": int(len(pos)), "selection": "first N FrameBank samples in image-path order",
        **{f"{k}_identical": bool(np.array_equal(par[k], w68[k][pos]))
           for k in ("readout", "joint_valid", "crop_sha256", "crop_box_xy_side", "image_sha256")},
        "readout_max_abs_diff": float(np.abs(par["readout"].astype(np.float32) - w68["readout"][pos].astype(np.float32)).max())}

    # 2. nearest-neighbour baselines (eligible rows, as in the main report)
    bank = load_bank(args.framebank)
    tr, va = ExternalBank.load(args.external_train), ExternalBank.load(args.external_val)
    f3 = arm_features(w68["readout"])
    ftr = arm_features(np.load(args.external_train_cache / "readout.npz")["readout"])
    fva = arm_features(np.load(args.external_val_cache / "readout.npz")["readout"])
    el3 = forward_targets(bank.arrays["target_3d"], bank.arrays["target_valid"])[1].any(axis=1)
    eltr = forward_targets(tr.arrays["target_3d"], tr.arrays["target_valid"])[1].any(axis=1)
    elva = forward_targets(va.arrays["target_3d"], va.arrays["target_valid"])[1].any(axis=1)
    seq3 = np.array([s.sequence_id for s in bank.samples])
    t3 = bank.indices("train")[el3[bank.indices("train")]]
    v3 = bank.indices("validation")[el3[bank.indices("validation")]]
    report["nearest_neighbour_cosine"] = {
        "3dpw_train_to_3dpw_train_other_sequence": nearest(f3[t3], f3[t3], seq3[t3], seq3[t3]),
        "3dpw_validation_to_3dpw_train": nearest(f3[v3], f3[t3]),
        "3dpw_validation_to_bedlam_train": nearest(f3[v3], ftr[eltr]),
        "bedlam_train_to_3dpw_train": nearest(ftr[eltr], f3[t3]),
        "bedlam_validation_to_bedlam_train": nearest(fva[elva], ftr[eltr]),
        "eligible_rows": {"3dpw_train": int(len(t3)), "3dpw_validation": int(len(v3)),
                          "bedlam_train": int(eltr.sum()), "bedlam_validation": int(elva.sum())},
        "ineligible_train_rows": {"3dpw": [int((~el3[bank.indices('train')]).sum()), int(len(bank.indices('train')))],
                                  "bedlam": [int((~eltr).sum()), int(len(tr))]}}

    # 3. detection sharing
    report["shared_detections"] = {"train_presplit": shared_detections(ExternalBank.load(args.train_bank_presplit)),
                                   "train_final": shared_detections(tr), "validation_final": shared_detections(va)}

    # 4. cache provenance
    report["cache_provenance"] = {}
    for spec in args.part:
        cached, rebuilt = (Path(p) for p in spec.split("="))
        report["cache_provenance"][cached.stem] = compare_parts(cached, rebuilt)

    args.out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
