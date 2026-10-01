#!/usr/bin/env python3
"""EVALUATION ONLY: attribute Worklog 62's H0 shoulder->wrist direction error upstream.

H0 / runtime observable: frozen H0 predictions from the 3DPW benchmark detector
observation in the bank.  GT / evaluation oracle: 3DPW jointPositions (3D) and
their projection with the official camera (oracle_geometry 2D).  The frozen
checkpoint is re-run with ONLY the 2D xy substituted; validity, confidence,
crop rule, sign fields and weights are unchanged.  Nothing is written back.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_root_motion_contact_h0_replay import _find_3dpw_pkl, _parse_3dpw_sequence_id  # noqa: E402

from framepose.bank import load_bank  # noqa: E402
from framepose.crops import crop_box, geometry_in_crop  # noqa: E402
from framepose.signs import mask_fields, oracle_sign_states  # noqa: E402
from pose.framepose_bridge import assemble_h0  # noqa: E402
from pose.framepose_chain_direction_attribution import (  # noqa: E402
    ARM, JOINT_INDEX, chain_direction_row, detector_vs_oracle_2d, joint_coordinate_errors,
    oracle_input_2d, project_animcv_to_pixels, xz_vs_image_direction,
)
from pose.three_dpw_adapter import _world_to_animcv_camera  # noqa: E402

SIGN_FIELDS = ("shoulder_forward_depth", "hip_forward_depth")
ARM_IDX = [JOINT_INDEX[n] for n in ARM]
SMPL_ARM = (16, 18, 20)


def _stats(values):
    values = sorted(v for v in values if v is not None and math.isfinite(v))
    if not values:
        return {"n": 0}

    def pct(p):
        k = (len(values) - 1) * p / 100
        lo, hi = math.floor(k), math.ceil(k)
        return values[lo] + (values[hi] - values[lo]) * (k - lo)
    return {"n": len(values), "mean": sum(values) / len(values), "min": values[0],
            "p05": pct(5), "p25": pct(25), "p50": pct(50), "p75": pct(75), "p95": pct(95),
            "max": values[-1]}


def _pearson(x, y):
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    xs, ys = np.array(pairs).T
    return float(np.corrcoef(xs, ys)[0, 1])


def _summary(rows, keys):
    return {k: _stats([r[k] for r in rows]) for k in keys}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--h0-train", type=Path, required=True)
    parser.add_argument("--h0-validation", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--semantics-report", type=Path, required=True)
    parser.add_argument("--w62-rows", type=Path, required=True)
    parser.add_argument("--w62-report", type=Path, required=True)
    parser.add_argument("--threedpw-raw", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    import torch
    from framepose.train import load_checkpoint, predict

    bank = load_bank(args.bank)
    h0, identity = assemble_h0(bank, {"train": args.h0_train, "validation": args.h0_validation})
    lineage = json.loads(args.semantics_report.read_text())["h0_identity"]
    if identity.bank_content_digest != lineage["bank_content_digest"] or identity.split_sha256 != lineage["split_sha256"]:
        raise SystemExit("frozen H0 / bank differ from the docs/57 AnimationSemantics lineage")
    w62_rows = json.loads(args.w62_rows.read_text())
    w62_cases = json.loads(args.w62_report.read_text())["owner_cases"]
    position_of = {(s.sequence_id, s.frame_index): i for i, s in enumerate(bank.samples)}
    positions = [position_of[(r["sequence_id"], r["frame_index"])] for r in w62_rows]

    # GT absolute camera joints + official intrinsics (evaluation oracle only).
    gt_abs, intrinsics = {}, {}
    for sid in sorted({r["sequence_id"] for r in w62_rows}):
        name, actor = _parse_3dpw_sequence_id(sid)
        with _find_3dpw_pkl(args.threedpw_raw, name).open("rb") as handle:
            raw = pickle.load(handle, encoding="latin1")
        world = np.asarray(raw["jointPositions"][actor], float).reshape(-1, 24, 3)
        camera = np.asarray(raw["cam_poses"], float)
        gt_abs[sid] = (world, camera, np.asarray(raw["campose_valid"][actor], bool))
        intrinsics[sid] = np.asarray(raw["cam_intrinsics"], float)

    detector_inputs, oracle_all, oracle_arm, valids, gt_root_rel, gt_pixels = [], [], [], [], [], []
    for row, position in zip(w62_rows, positions):
        sid, frame = row["sequence_id"], row["frame_index"]
        sample = bank.samples[position]
        world, camera, campose_valid = gt_abs[sid]
        if not campose_valid[frame]:
            raise SystemExit(f"{sid} #{frame}: GT camera pose invalid on a Worklog 62 row")
        absolute = _world_to_animcv_camera(world[frame], camera[frame])
        K = intrinsics[sid]
        if tuple(sample.image_size) != (int(round(K[0, 2] * 2)), int(round(K[1, 2] * 2))):
            raise SystemExit(f"{sid}: bank image size differs from 3DPW intrinsics")
        pixels = project_animcv_to_pixels(absolute, K)
        det = np.asarray(bank.arrays["input_2d"][position], float)
        detector_inputs.append(det)
        oracle_all.append(oracle_input_2d(pixels, sample.image_size, det))
        oracle_arm.append(oracle_input_2d(pixels, sample.image_size, det, joints=ARM))
        valids.append(np.asarray(bank.arrays["input_valid"][position], bool))
        gt_root_rel.append(absolute - absolute[0])
        gt_pixels.append(pixels)

    model, checkpoint = load_checkpoint(args.checkpoint, device=args.device)
    signs = mask_fields(oracle_sign_states(bank.arrays["target_3d"], bank.arrays["target_valid"]),
                        list(SIGN_FIELDS))[positions]
    device = torch.device(args.device)

    def run(inputs):
        geometry = np.stack([geometry_in_crop(x, v, bank.samples[p].image_size,
                                              crop_box(x, v, bank.samples[p].image_size))
                             for x, v, p in zip(inputs, valids, positions)]).astype(np.float32)
        return predict(model, torch, geometry, None, np.arange(len(inputs)), device, signs=signs)

    regimes = {"benchmark_detector_observation": run(detector_inputs),
               "oracle_geometry_all_joints": run(oracle_all),
               "oracle_geometry_left_arm_only": run(oracle_arm)}
    reproduction = float(np.abs(regimes["benchmark_detector_observation"] - h0[positions]).max())
    if reproduction > 1e-4:
        raise SystemExit(f"frozen H0 not reproduced from detector input: max |diff| {reproduction}")

    rows = []
    w62_consistency = 0.0
    for k, (row, position) in enumerate(zip(w62_rows, positions)):
        sample = bank.samples[position]
        w, h = sample.image_size
        det_px = detector_inputs[k][ARM_IDX, :2] * [w, h]
        ora_px = gt_pixels[k][list(SMPL_ARM)]
        gt3 = gt_root_rel[k][list(SMPL_ARM)]
        out = {"sequence_id": row["sequence_id"], "frame_index": row["frame_index"],
               "bank_position": position,
               "w62_endpoint_direction_error_degrees": row["endpoint_direction_error_degrees"],
               "w62_ratio_log_error": row["ratio_log_error"],
               "w62_reach_fraction_error": row["reach_fraction_error"],
               "w62_bend_plane_error_degrees": row["bend_plane_error_degrees"],
               "w62_fk_h0_endpoint_error": row["fk_h0_endpoint_error"],
               "w62_ik_h0_endpoint_error": row["ik_h0_endpoint_error"]}
        out.update(detector_vs_oracle_2d(det_px, ora_px))
        out["gt_xz_vs_oracle_2d_degrees"] = xz_vs_image_direction(gt3[2] - gt3[0], ora_px[2] - ora_px[0])
        for tag, prediction in regimes.items():
            short = {"benchmark_detector_observation": "h0",
                     "oracle_geometry_all_joints": "oracle_all",
                     "oracle_geometry_left_arm_only": "oracle_arm"}[tag]
            p3 = prediction[k][ARM_IDX]
            for key, value in chain_direction_row(p3, gt3).items():
                out[f"{short}_{key}"] = value
            out[f"{short}_xz_vs_detector_2d_degrees"] = xz_vs_image_direction(p3[2] - p3[0], det_px[2] - det_px[0])
            for key, value in joint_coordinate_errors(p3, gt3, prediction[k][0], gt_root_rel[k][0]).items():
                out[f"{short}_{key}"] = value
        w62_consistency = max(w62_consistency, abs(out["h0_full_3d_angle_degrees"]
                                                   - row["endpoint_direction_error_degrees"]))
        rows.append(out)
    if w62_consistency > 1e-3:
        raise SystemExit(f"camera-frame direction error differs from Worklog 62: {w62_consistency}")

    two_d = ["left_shoulder_2d_error_px", "left_elbow_2d_error_px", "left_wrist_2d_error_px",
             "shoulder_wrist_2d_direction_error_degrees", "shoulder_elbow_2d_direction_error_degrees",
             "elbow_wrist_2d_direction_error_degrees", "shoulder_wrist_2d_endpoint_error_over_chain",
             "oracle_chain_length_px", "gt_xz_vs_oracle_2d_degrees"]
    three_d_suffix = ["full_3d_angle_degrees", "xz_image_plane_angle_degrees", "forward_fraction_error",
                      "depth_elevation_error_degrees", "component_h0_full", "component_h0_image_plane_gt_depth",
                      "component_gt_image_plane_h0_depth", "component_gt_full",
                      "segment_h0_upper_h0_lower", "segment_gt_upper_h0_lower", "segment_h0_upper_gt_lower",
                      "segment_gt_upper_gt_lower", "segment_upper_direction_error_degrees",
                      "segment_lower_direction_error_degrees", "upper_xz_angle_degrees",
                      "lower_xz_angle_degrees", "xz_vs_detector_2d_degrees"]
    joint_suffix = [f"{j}_{a}" for j in ARM for a in ("x_error_m", "y_error_m", "z_error_m", "error_m")]

    def block(subset):
        out = {"n": len(subset), "two_d": _summary(subset, two_d)}
        for short in ("h0", "oracle_all", "oracle_arm"):
            out[short] = _summary(subset, [f"{short}_{s}" for s in three_d_suffix])
            out[short + "_abs_axis"] = {k: _stats([abs(r[f"{short}_{k}"]) for r in subset])
                                       for k in joint_suffix}
            out[short + "_depth_sign_disagree"] = sum(r[f"{short}_depth_sign_disagrees"] for r in subset)
            out[short + "_abs_depth_elevation_error"] = _stats(
                [abs(r[f"{short}_depth_elevation_error_degrees"]) for r in subset])
        return out

    sequences = sorted({r["sequence_id"] for r in rows})
    report = {
        "schema": "animcv_framepose_chain_direction_attribution_v1",
        "labels": {"H0": "frozen H0 from benchmark_detector_observation (3DPW poses2d)",
                   "oracle_geometry": "3DPW jointPositions projected with official cam_poses/cam_intrinsics; evaluation only",
                   "GT": "3DPW jointPositions root-relative canonical camera frame; evaluation only"},
        "bank_content_digest": identity.bank_content_digest, "h0_split_sha256": identity.split_sha256,
        "checkpoint": str(args.checkpoint), "checkpoint_candidate": checkpoint.get("candidate"),
        "sign_fields": list(SIGN_FIELDS),
        "substitution": "only input_2d x,y replaced; input_valid, confidence, crop rule, signs, weights unchanged",
        "oracle_2d_convention": "SMPL joints 16-21/1,2,4,5,7,8 direct; head=SMPL15 (detector uses nose); neck/pelvis/spine midpoints; thorax=neck",
        "reproduction_max_abs_m": reproduction, "w62_direction_consistency_max_degrees": w62_consistency,
        "rows": len(rows), "pooled": block(rows),
        "per_sequence": {sid: block([r for r in rows if r["sequence_id"] == sid]) for sid in sequences},
    }
    pooled = report["pooled"]
    factors = {k: [r[k] for r in rows] for k in ("shoulder_wrist_2d_direction_error_degrees",
                                                 "h0_xz_image_plane_angle_degrees",
                                                 "h0_depth_elevation_error_degrees",
                                                 "h0_segment_upper_direction_error_degrees",
                                                 "h0_segment_lower_direction_error_degrees")}
    target = [r["h0_full_3d_angle_degrees"] for r in rows]
    report["correlation_with_h0_full_3d_angle"] = {
        k: _pearson([abs(v) for v in vals], target) for k, vals in factors.items()}
    # Fraction of H0 full-angle error closed by each oracle substitution (paired medians).
    report["closure"] = {
        name: {"median_reduction_degrees": float(np.median([r["h0_full_3d_angle_degrees"] - r[key] for r in rows])),
               "mean_fraction_closed": float(np.mean([r[key] for r in rows]) / np.mean(target))}
        for name, key in (("oracle_2d_all_joints", "oracle_all_full_3d_angle_degrees"),
                          ("oracle_2d_left_arm_only", "oracle_arm_full_3d_angle_degrees"),
                          ("gt_depth_component", "h0_component_h0_image_plane_gt_depth"),
                          ("gt_image_plane_component", "h0_component_gt_image_plane_h0_depth"),
                          ("gt_upper_segment", "h0_segment_gt_upper_h0_lower"),
                          ("gt_lower_segment", "h0_segment_h0_upper_gt_lower"))}

    cases = {label: (c["sequence_id"], c["frame_index"]) for label, c in w62_cases.items()}
    for label, key in (("largest_2d_shoulder_wrist_direction_error", "shoulder_wrist_2d_direction_error_degrees"),
                       ("largest_depth_component_error", "h0_depth_elevation_error_degrees")):
        best = max(rows, key=lambda r: (abs(r[key]), r["sequence_id"], r["frame_index"]))
        cases[label] = (best["sequence_id"], best["frame_index"])
    by_key = {(r["sequence_id"], r["frame_index"]): r for r in rows}
    report["owner_cases"] = {
        label: {"sequence_id": key[0], "frame_index": key[1],
                "overlaps": sorted(o for o, k in cases.items() if k == key and o != label),
                "row": by_key.get(key),
                "excluded": None if key in by_key else "not a Worklog 62 matched row (H0 IK unavailable)"}
        for label, key in cases.items()}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "rows.json").write_text(json.dumps(rows, indent=1))
    path = args.out_dir / "report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "rows": len(rows),
                      "reproduction": reproduction, "w62_consistency": w62_consistency,
                      "closure": report["closure"],
                      "correlation": report["correlation_with_h0_full_3d_angle"]}, indent=2))


if __name__ == "__main__":
    main()
