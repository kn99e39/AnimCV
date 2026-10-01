#!/usr/bin/env python3
"""DIAGNOSTIC (Worklog 64): train the explicit 2.5D left-arm depth probe and compare with frozen H0.

Same bank, split identity, geometry tensor and O_BILATERAL sign array as H0;
only the target/output interpretation differs.  GT is used for the training
target and for evaluation only, never at inference beyond the two historical
oracle sign fields that H0 itself receives.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from framepose.arm_depth_probe import (
    OUTPUT_TOKEN, SCHEMA, SEGMENT_NAMES, ProbeConfig, angle_degrees, elevation_degrees,
    forward_fraction, forward_targets, observed_plane, predict_probe, reconstruct_direction,
    segment_vectors, train_probe,
)
from framepose.bank import load_bank
from framepose.contract import JOINT_INDEX
from framepose.model import ModelConfig, build_model, parameter_report
from framepose.signs import mask_fields, oracle_sign_states
from framepose.train import geometry_tensor, load_checkpoint, predict
from pose.framepose_bridge import assemble_h0

SIGN_FIELDS = ("shoulder_forward_depth", "hip_forward_depth")
CLEAN = ("courtyard_dancing_00", "courtyard_hug_00")
STABLE_SINE = math.sin(math.radians(10))


def _stats(values):
    v = np.sort(np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float))
    if not len(v):
        return {"n": 0}
    q = lambda p: float(np.percentile(v, p))  # noqa: E731
    return {"n": int(len(v)), "mean": float(v.mean()), "p25": q(25), "p50": q(50), "p75": q(75),
            "p95": q(95), "max": float(v[-1])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bank", "h0_train", "h0_validation", "h0_test", "h0_checkpoint", "semantics_report",
                 "w62_rows", "w63_report", "out_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    import torch

    bank = load_bank(args.bank)
    h0, identity = assemble_h0(bank, {"train": args.h0_train, "validation": args.h0_validation,
                                      "test": args.h0_test})
    lineage = json.loads(args.semantics_report.read_text())["h0_identity"]
    if identity.bank_content_digest != lineage["bank_content_digest"] or any(
            identity.split_sha256[k] != v for k, v in lineage["split_sha256"].items()):
        raise SystemExit("bank / frozen H0 differ from the docs/57 lineage")

    # Candidate input identity: exactly H0's geometry tensor and sign array.
    geometry = geometry_tensor(bank)
    signs = mask_fields(oracle_sign_states(bank.arrays["target_3d"], bank.arrays["target_valid"]),
                        list(SIGN_FIELDS))
    h0_model, h0_payload = load_checkpoint(args.h0_checkpoint, device=args.device)
    validation = bank.indices("validation")
    test = bank.indices("test")
    rerun = predict(h0_model, torch, geometry, None, validation, torch.device(args.device), signs=signs)
    input_identity = float(np.abs(rerun - h0[validation]).max())
    if input_identity > 1e-4:
        raise SystemExit(f"candidate inputs do not reproduce frozen H0: {input_identity}")

    model_config = ModelConfig(**{k: v for k, v in h0_payload["model_config"].items()
                                  if k in ("visual_dim", "visual_tokens", "sign_fields", "width", "heads",
                                           "fusion_depth", "feedforward_multiplier")})
    candidate = h0_payload["candidate"]
    config = ProbeConfig(epochs=candidate["epochs"], batch_size=candidate["batch_size"],
                         learning_rate=candidate["learning_rate"], weight_decay=candidate["weight_decay"],
                         minimum_learning_rate=candidate["minimum_learning_rate"], seed=candidate["seed"],
                         mixed_precision=candidate["mixed_precision"],
                         evaluate_every=candidate["evaluate_every"], device=args.device)
    targets, masks = forward_targets(bank.arrays["target_3d"], bank.arrays["target_valid"])
    model, interpret, training = train_probe(geometry, signs, targets, masks, bank.indices("train"),
                                             validation, model_config, config)
    params = parameter_report(model)
    h0_params = parameter_report(build_model(model_config))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"schema": SCHEMA, "model_config": model_config.to_dict(), "probe_config": config.to_dict(),
                "output_token": OUTPUT_TOKEN, "bank_content_digest": bank.content_digest(),
                "selection": training["selection"], "state_dict": model.state_dict()},
               args.out_dir / "checkpoint.pt")

    def evaluate(positions):
        positions = np.asarray(positions)
        cand = predict_probe(model, interpret, geometry, signs, positions, device=args.device)
        gt_vec = segment_vectors(bank.arrays["target_3d"][positions])
        h0_vec = segment_vectors(h0[positions])
        sizes = np.asarray([bank.samples[p].image_size for p in positions], dtype=float)
        planes, plane_ok = observed_plane(bank.arrays["input_2d"][positions], sizes)
        valid_in = bank.arrays["input_valid"][positions]
        arm_in = np.all([valid_in[:, JOINT_INDEX[j]] for j in ("left_shoulder", "left_elbow", "left_wrist")], axis=0)
        eligible = masks[positions].all(axis=1) & arm_in & np.all([plane_ok[n] for n in SEGMENT_NAMES], axis=0)
        rows = []
        per = {}
        # Ineligible rows (degenerate observed segment) are excluded below; give
        # them a placeholder plane so vectorized reconstruction stays defined.
        planes = {name: np.where(plane_ok[name][:, None], planes[name], [[1.0, 0.0]]) for name in planes}
        for k, name in enumerate(SEGMENT_NAMES):
            g = gt_vec[name]
            gu = g / np.linalg.norm(g, axis=1, keepdims=True)
            f_gt = targets[positions, k]
            f_h0, _ = forward_fraction(h0_vec[name])
            f_c = cand[:, k]
            gt_xz = np.stack([g[:, 0], g[:, 2]], axis=1)
            safe_gt_xz = np.where(np.linalg.norm(gt_xz, axis=1, keepdims=True) > 1e-9, gt_xz, [[1.0, 0.0]])
            per[name] = {
                "f_gt": f_gt, "f_h0": f_h0, "f_cand": f_c,
                "h0_full": angle_degrees(h0_vec[name], g),
                "cand_raw_plane": angle_degrees(reconstruct_direction(f_c, planes[name]), gu),
                "cand_oracle_xz": angle_degrees(reconstruct_direction(f_c, safe_gt_xz), gu),
                "h0_f_raw_plane": angle_degrees(reconstruct_direction(f_h0, planes[name]), gu),
                "h0_f_oracle_xz": angle_degrees(reconstruct_direction(f_h0, safe_gt_xz), gu),
                "perspective_floor_gt_f_raw_plane": angle_degrees(reconstruct_direction(f_gt, planes[name]), gu),
            }
        for i, p in enumerate(positions):
            if not eligible[i]:
                continue
            sample = bank.samples[p]
            row = {"sequence_id": sample.sequence_id, "frame_index": sample.frame_index, "position": int(p)}
            for name in SEGMENT_NAMES:
                d = per[name]
                row.update({f"{name}_{key}": float(value[i]) for key, value in d.items()})
            rows.append(row)
        return rows, int(len(positions)), int(eligible.sum())

    def summarize(rows):
        out = {"n": len(rows)}
        for name in SEGMENT_NAMES:
            f_gt = np.array([r[f"{name}_f_gt"] for r in rows])
            seg = {}
            for who in ("h0", "cand"):
                f = np.array([r[f"{name}_f_{who}"] for r in rows])
                seg[f"{who}_abs_forward_error"] = _stats(np.abs(f - f_gt))
                seg[f"{who}_abs_elevation_error_degrees"] = _stats(np.abs(elevation_degrees(f) - elevation_degrees(f_gt)))
                stable = np.abs(f_gt) > STABLE_SINE
                seg[f"{who}_forward_sign_agreement_stable"] = (
                    float(np.mean(np.sign(f[stable]) == np.sign(f_gt[stable]))) if stable.any() else None)
            for key in ("h0_full", "cand_raw_plane", "cand_oracle_xz", "h0_f_raw_plane", "h0_f_oracle_xz",
                        "perspective_floor_gt_f_raw_plane"):
                seg[f"{key}_angle_degrees"] = _stats([r[f"{name}_{key}"] for r in rows])
            diff = np.array([r[f"{name}_cand_raw_plane"] - r[f"{name}_h0_full"] for r in rows])
            seg["paired_cand_raw_minus_h0_full_degrees"] = _stats(diff)
            seg["cand_raw_better_than_h0_full_fraction"] = float(np.mean(diff < 0)) if len(diff) else None
            fdiff = np.array([abs(r[f"{name}_f_cand"] - r[f"{name}_f_gt"]) - abs(r[f"{name}_f_h0"] - r[f"{name}_f_gt"])
                              for r in rows])
            seg["paired_abs_forward_error_cand_minus_h0"] = _stats(fdiff)
            out[name] = seg
        return out

    val_rows, val_n, val_e = evaluate(validation)
    test_rows, test_n, test_e = evaluate(test)
    w62 = {(r["sequence_id"], r["frame_index"]) for r in json.loads(args.w62_rows.read_text())}
    scopes = {
        "validation_all": val_rows,
        "validation_clean_dancing_hug": [r for r in val_rows if any(c in r["sequence_id"] for c in CLEAN)],
        "validation_crosscountry_only": [r for r in val_rows if "crosscountry" in r["sequence_id"]],
        "test_all": test_rows,
        "worklog62_matched_rows": [r for r in val_rows if (r["sequence_id"], r["frame_index"]) in w62],
    }
    owner = {}
    by_key = {(r["sequence_id"], r["frame_index"]): r for r in val_rows}
    for label, case in json.loads(args.w63_report.read_text())["owner_cases"].items():
        key = (case["sequence_id"], case["frame_index"])
        row = by_key.get(key)
        owner[label] = {"sequence_id": key[0], "frame_index": key[1], "overlaps": case["overlaps"],
                        "row": row, "excluded": None if row else "not an eligible validation row"}

    report = {
        "schema": SCHEMA + "_report",
        "labels": {"H0": "frozen O_BILATERAL Cartesian XYZ", "candidate": "explicit 2.5D forward-fraction probe",
                   "oracle_signs": "shoulder_forward_depth, hip_forward_depth from GT (identical to H0; optimistic control)",
                   "GT": "bank target_3d; training target + evaluation only"},
        "bank_content_digest": bank.content_digest(), "h0_split_sha256": identity.split_sha256,
        "input_identity_max_abs_m": input_identity,
        "model": {"h0_parameters": h0_params, "candidate_parameters": params,
                  "model_config": model_config.to_dict(), "output_token": OUTPUT_TOKEN,
                  "unused_outputs": "all other joints/channels of the unchanged head (gradient-free)"},
        "training": training,
        "splits": {"train": int(len(bank.indices("train"))), "validation": val_n, "test": test_n,
                   "validation_eligible": val_e, "test_eligible": test_e},
        "eligibility": "GT valid for all three arm joints AND detector input valid for all three AND non-degenerate observed 2D segments",
        "scopes": {name: summarize(rows) for name, rows in scopes.items()},
        "owner_cases": owner,
    }
    (args.out_dir / "rows_validation.json").write_text(json.dumps(val_rows))
    (args.out_dir / "rows_test.json").write_text(json.dumps(test_rows))
    path = args.out_dir / "report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                      "input_identity": input_identity, "params": [h0_params, params],
                      "selection": training["selection"], "splits": report["splits"]}, indent=2))


if __name__ == "__main__":
    main()
