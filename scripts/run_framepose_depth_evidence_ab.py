#!/usr/bin/env python3
"""DIAGNOSTIC (Worklog 65): raw Depth Anything arm-depth signal, then D0 vs D1 on the Worklog 64 task.

References: A frozen H0 (Cartesian XYZ), B Worklog 64 geometry-only probe,
C D0_ZERO_DEPTH, D D1_DEPTH_ANYTHING.  C vs D is the causal comparison
(identical 5-channel graph and parameter count; only channel 5 differs).
GT is the training target and evaluation reference only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from framepose.arm_depth_probe import (
    SEGMENTS, SEGMENT_NAMES, ProbeConfig, angle_degrees, build_probe, elevation_degrees, forward_fraction,
    forward_targets, observed_plane, predict_probe, reconstruct_direction, segment_vectors,
)
from framepose.bank import load_bank
from framepose.contract import JOINT_INDEX
from framepose.depth_evidence_probe import (
    CANDIDATES, build_depth_conditioned, candidate_geometry, predict_depth_candidate, train_depth_candidate,
)
from framepose.model import ModelConfig, build_model, parameter_report
from framepose.signs import mask_fields, oracle_sign_states
from framepose.train import geometry_tensor, load_checkpoint, predict
from pose.framepose_bridge import assemble_h0

SIGN_FIELDS = ("shoulder_forward_depth", "hip_forward_depth")
CLEAN = ("courtyard_dancing_00", "courtyard_hug_00")
STABLE_SINE = math.sin(math.radians(10))
F_BINS = ((0.0, STABLE_SINE), (STABLE_SINE, 0.5), (0.5, 1.0001))
MODELS = ("h0", "w64", "d0", "d1")


def _stats(values):
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if not len(v):
        return {"n": 0}
    q = lambda p: float(np.percentile(v, p))  # noqa: E731
    return {"n": int(len(v)), "mean": float(v.mean()), "p05": q(5), "p50": q(50), "p95": q(95), "max": float(v.max())}


def _rank(x):
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(len(x))
    # average ties
    values = np.asarray(x)[order]
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[j + 1] == values[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2
        i = j + 1
    return ranks


def _corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return {"pearson": None, "spearman": None, "n": int(len(x))}
    return {"pearson": float(np.corrcoef(x, y)[0, 1]),
            "spearman": float(np.corrcoef(_rank(x), _rank(y))[0, 1]), "n": int(len(x))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bank", "h0_train", "h0_validation", "h0_test", "h0_checkpoint", "w64_checkpoint",
                 "semantics_report", "evidence_dir", "w63_report", "out_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    import torch

    bank = load_bank(args.bank)
    h0, identity = assemble_h0(bank, {"train": args.h0_train, "validation": args.h0_validation, "test": args.h0_test})
    lineage = json.loads(args.semantics_report.read_text())["h0_identity"]
    if identity.bank_content_digest != lineage["bank_content_digest"] or any(
            identity.split_sha256[k] != v for k, v in lineage["split_sha256"].items()):
        raise SystemExit("bank / frozen H0 differ from the docs/57 lineage")
    manifest = json.loads((args.evidence_dir / "manifest.json").read_text())
    npz_path = args.evidence_dir / "evidence.npz"
    if (manifest["identity"]["bank_content_digest"] != bank.content_digest()
            or manifest["sample_ids_sha256"] != hashlib.sha256("\n".join(s.sample_id for s in bank.samples).encode()).hexdigest()
            or manifest["evidence_npz_sha256"] != hashlib.sha256(npz_path.read_bytes()).hexdigest()):
        raise SystemExit("depth evidence cache identity does not match this bank")
    evidence = np.load(npz_path)
    forward_ev, available = evidence["forward_evidence"], evidence["available"]

    geometry = geometry_tensor(bank)
    signs = mask_fields(oracle_sign_states(bank.arrays["target_3d"], bank.arrays["target_valid"]), list(SIGN_FIELDS))
    h0_model, h0_payload = load_checkpoint(args.h0_checkpoint, device=args.device)
    validation, test, train = bank.indices("validation"), bank.indices("test"), bank.indices("train")
    rerun = predict(h0_model, torch, geometry, None, validation, torch.device(args.device), signs=signs)
    if float(np.abs(rerun - h0[validation]).max()) > 1e-4:
        raise SystemExit("inputs do not reproduce frozen H0")
    model_config = ModelConfig(**{k: v for k, v in h0_payload["model_config"].items()
                                  if k in ("visual_dim", "visual_tokens", "sign_fields", "width", "heads",
                                           "fusion_depth", "feedforward_multiplier")})
    c = h0_payload["candidate"]
    config = ProbeConfig(epochs=c["epochs"], batch_size=c["batch_size"], learning_rate=c["learning_rate"],
                         weight_decay=c["weight_decay"], minimum_learning_rate=c["minimum_learning_rate"],
                         seed=c["seed"], mixed_precision=c["mixed_precision"], evaluate_every=c["evaluate_every"],
                         device=args.device)
    targets, masks = forward_targets(bank.arrays["target_3d"], bank.arrays["target_valid"])

    # B: Worklog 64 probe, loaded unchanged.
    w64_payload = torch.load(args.w64_checkpoint, map_location=args.device, weights_only=False)
    w64_model, w64_interpret = build_probe(model_config)
    w64_model.load_state_dict(w64_payload["state_dict"])
    w64_model = w64_model.to(args.device).eval()

    # C / D: identical graph, only channel 5 differs.
    inputs = {name: candidate_geometry(geometry, forward_ev, name) for name in CANDIDATES}
    if not np.array_equal(inputs["D0_ZERO_DEPTH"][..., :4], inputs["D1_DEPTH_ANYTHING"][..., :4]) or \
            np.any(inputs["D0_ZERO_DEPTH"][..., 4] != 0):
        raise SystemExit("D0/D1 input contract violated")
    trained, training = {}, {}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name in CANDIDATES:
        model, interpret, report = train_depth_candidate(inputs[name], signs, targets, masks, train, validation,
                                                         model_config, config)
        trained[name] = (model, interpret)
        training[name] = {**report, "parameters": parameter_report(model)}
        torch.save({"candidate": name, "model_config": model_config.to_dict(), "probe_config": config.to_dict(),
                    "evidence_identity": manifest["identity"], "selection": report["selection"],
                    "state_dict": model.state_dict()}, args.out_dir / f"{name}.pt")

    def rows_for(positions):
        positions = np.asarray(positions)
        preds = {"w64": predict_probe(w64_model, w64_interpret, geometry, signs, positions, device=args.device),
                 "d0": predict_depth_candidate(*trained["D0_ZERO_DEPTH"], inputs["D0_ZERO_DEPTH"], signs, positions),
                 "d1": predict_depth_candidate(*trained["D1_DEPTH_ANYTHING"], inputs["D1_DEPTH_ANYTHING"], signs, positions)}
        gt_vec = segment_vectors(bank.arrays["target_3d"][positions])
        h0_vec = segment_vectors(h0[positions])
        sizes = np.asarray([bank.samples[p].image_size for p in positions], float)
        planes, plane_ok = observed_plane(bank.arrays["input_2d"][positions], sizes)
        planes = {n: np.where(plane_ok[n][:, None], planes[n], [[1.0, 0.0]]) for n in planes}
        valid_in = bank.arrays["input_valid"][positions]
        arm_in = np.all([valid_in[:, JOINT_INDEX[j]] for j in ("left_shoulder", "left_elbow", "left_wrist")], axis=0)
        eligible = masks[positions].all(axis=1) & arm_in & np.all([plane_ok[n] for n in SEGMENT_NAMES], axis=0)
        per = {}
        for k, name in enumerate(SEGMENT_NAMES):
            a, b = (JOINT_INDEX[j] for j in SEGMENTS[name])
            g = gt_vec[name]
            gu = g / np.linalg.norm(g, axis=1, keepdims=True)
            gxz = np.stack([g[:, 0], g[:, 2]], axis=1)
            gxz = np.where(np.linalg.norm(gxz, axis=1, keepdims=True) > 1e-9, gxz, [[1.0, 0.0]])
            f = {"gt": targets[positions, k], "h0": forward_fraction(h0_vec[name])[0],
                 "w64": preds["w64"][:, k], "d0": preds["d0"][:, k], "d1": preds["d1"][:, k]}
            seg = {f"f_{m}": v for m, v in f.items()}
            seg["h0_full_xyz"] = angle_degrees(h0_vec[name], g)
            for m in MODELS:
                seg[f"{m}_raw_plane"] = angle_degrees(reconstruct_direction(f[m], planes[name]), gu)
                seg[f"{m}_oracle_xz"] = angle_degrees(reconstruct_direction(f[m], gxz), gu)
            pos_ab = (available[positions, a] & available[positions, b])
            seg["da_delta"] = np.where(pos_ab, forward_ev[positions, b] - forward_ev[positions, a], np.nan)
            seg["gt_delta_y_m"] = g[:, 1]
            per[name] = seg
        rows = []
        for i, p in enumerate(positions):
            if not eligible[i]:
                continue
            sample = bank.samples[p]
            row = {"sequence_id": sample.sequence_id, "frame_index": sample.frame_index, "position": int(p),
                   "depth_frame_available": bool(evidence["frame_available"][p]),
                   "da_forward_shoulder_elbow_wrist": [float(forward_ev[p, JOINT_INDEX[j]]) if available[p, JOINT_INDEX[j]] else None
                                                       for j in ("left_shoulder", "left_elbow", "left_wrist")]}
            for name, seg in per.items():
                row.update({f"{name}_{key}": (None if not np.isfinite(v[i]) else float(v[i])) for key, v in seg.items()})
            rows.append(row)
        return rows

    def raw_signal(rows):
        out = {}
        for name in SEGMENT_NAMES:
            r = [x for x in rows if x[f"{name}_da_delta"] is not None]
            d = np.array([x[f"{name}_da_delta"] for x in r])
            fg = np.array([x[f"{name}_f_gt"] for x in r])
            fh = np.array([x[f"{name}_f_h0"] for x in r])
            dy = np.array([x[f"{name}_gt_delta_y_m"] for x in r])
            stable = np.abs(fg) > STABLE_SINE
            seg = {"rows_with_depth": len(r), "rows_without_depth": sum(x[f"{name}_da_delta"] is None for x in rows),
                   "sign_agreement": float(np.mean(np.sign(d) == np.sign(fg))) if len(r) else None,
                   "stable_sign_agreement": float(np.mean(np.sign(d[stable]) == np.sign(fg[stable]))) if stable.any() else None,
                   "h0_stable_sign_agreement_same_rows": float(np.mean(np.sign(fh[stable]) == np.sign(fg[stable]))) if stable.any() else None,
                   "corr_vs_gt_forward_fraction": _corr(d, fg), "corr_vs_gt_delta_y": _corr(d, dy),
                   "h0_corr_vs_gt_forward_fraction": _corr(fh, fg), "by_abs_gt_f": {}}
            for lo, hi in F_BINS:
                sel = (np.abs(fg) >= lo) & (np.abs(fg) < hi)
                seg["by_abs_gt_f"][f"{lo:.3f}-{min(hi, 1):.3f}"] = {
                    "n": int(sel.sum()),
                    "sign_agreement": float(np.mean(np.sign(d[sel]) == np.sign(fg[sel]))) if sel.any() else None,
                    "h0_sign_agreement": float(np.mean(np.sign(fh[sel]) == np.sign(fg[sel]))) if sel.any() else None,
                    "corr": _corr(d[sel], fg[sel])}
            out[name] = seg
        return out

    def summarize(rows):
        out = {"n": len(rows), "depth_frame_available": int(sum(r["depth_frame_available"] for r in rows))}
        for name in SEGMENT_NAMES:
            fg = np.array([r[f"{name}_f_gt"] for r in rows])
            seg = {}
            for m in MODELS:
                f = np.array([r[f"{name}_f_{m}"] for r in rows])
                seg[m] = {"abs_f_error": _stats(np.abs(f - fg)),
                          "abs_elevation_error_degrees": _stats(np.abs(elevation_degrees(f) - elevation_degrees(fg))),
                          "full_raw_plane_degrees": _stats([r[f"{name}_{m}_raw_plane"] for r in rows]),
                          "full_oracle_xz_degrees": _stats([r[f"{name}_{m}_oracle_xz"] for r in rows])}
            seg["h0"]["full_xyz_degrees"] = _stats([r[f"{name}_h0_full_xyz"] for r in rows])
            f0 = np.array([r[f"{name}_f_d0"] for r in rows])
            f1 = np.array([r[f"{name}_f_d1"] for r in rows])
            paired = {
                "abs_f_error": np.abs(f1 - fg) - np.abs(f0 - fg),
                "abs_elevation_error_degrees": np.abs(elevation_degrees(f1) - elevation_degrees(fg)) - np.abs(elevation_degrees(f0) - elevation_degrees(fg)),
                "full_raw_plane_degrees": np.array([r[f"{name}_d1_raw_plane"] - r[f"{name}_d0_raw_plane"] for r in rows]),
                "full_oracle_xz_degrees": np.array([r[f"{name}_d1_oracle_xz"] - r[f"{name}_d0_oracle_xz"] for r in rows]),
            }
            seg["d1_minus_d0"] = {k: {**_stats(v), "d1_better_fraction": float(np.mean(v < 0)) if len(v) else None}
                                  for k, v in paired.items()}
            seg["d1_minus_d0_by_abs_gt_f"] = {
                f"{lo:.3f}-{min(hi, 1):.3f}": _stats(paired["abs_f_error"][(np.abs(fg) >= lo) & (np.abs(fg) < hi)])
                for lo, hi in F_BINS}
            out[name] = seg
        return out

    val_rows, test_rows = rows_for(validation), rows_for(test)
    scopes = {"test_all": test_rows, "validation_all": val_rows,
              "validation_clean_dancing_hug": [r for r in val_rows if any(c in r["sequence_id"] for c in CLEAN)],
              "validation_crosscountry_only": [r for r in val_rows if "crosscountry" in r["sequence_id"]],
              "test_depth_available": [r for r in test_rows if r["depth_frame_available"]]}
    per_sequence = defaultdict(list)
    for r in val_rows + test_rows:
        per_sequence[r["sequence_id"]].append(r)
    seq_table = {}
    for sid, rows in sorted(per_sequence.items()):
        entry = {"n": len(rows)}
        for name in SEGMENT_NAMES:
            fg = np.array([r[f"{name}_f_gt"] for r in rows])
            for m in ("d0", "d1"):
                entry[f"{name}_{m}_abs_f_error_mean"] = float(np.mean(np.abs(np.array([r[f"{name}_f_{m}"] for r in rows]) - fg)))
        seq_table[sid] = entry
    sequences_d1_better = {name: sum(e[f"{name}_d1_abs_f_error_mean"] < e[f"{name}_d0_abs_f_error_mean"]
                                     for e in seq_table.values()) for name in SEGMENT_NAMES}
    owner = {}
    by_key = {(r["sequence_id"], r["frame_index"]): r for r in val_rows}
    for label, case in json.loads(args.w63_report.read_text())["owner_cases"].items():
        key = (case["sequence_id"], case["frame_index"])
        owner[label] = {"sequence_id": key[0], "frame_index": key[1], "overlaps": case["overlaps"],
                        "row": by_key.get(key), "excluded": None if key in by_key else "not an eligible validation row"}

    report = {
        "schema": "animcv_framepose_depth_evidence_ab_v1",
        "labels": {"A": "frozen H0 Cartesian XYZ", "B": "Worklog 64 geometry-only explicit-depth probe",
                   "C": "D0_ZERO_DEPTH (5-channel graph, channel 5 = 0)",
                   "D": "D1_DEPTH_ANYTHING (5-channel graph, channel 5 = Depth Anything forward evidence)",
                   "oracle_signs": "shoulder_forward_depth, hip_forward_depth (historical O_BILATERAL; optimistic control)"},
        "evidence_manifest": {k: v for k, v in manifest.items() if k != "image_sha256_by_relative_path"},
        "bank_content_digest": bank.content_digest(),
        "parameters": {"h0": parameter_report(build_model(model_config)), "w64": parameter_report(w64_model),
                       **{name: training[name]["parameters"] for name in CANDIDATES}},
        "training": training,
        "raw_depth_signal": {name: raw_signal(rows) for name, rows in scopes.items()},
        "scopes": {name: summarize(rows) for name, rows in scopes.items()},
        "per_sequence_abs_f_error": seq_table, "sequences_where_d1_better": sequences_d1_better,
        "sequence_count": len(seq_table), "owner_cases": owner,
    }
    (args.out_dir / "rows_validation.json").write_text(json.dumps(val_rows))
    (args.out_dir / "rows_test.json").write_text(json.dumps(test_rows))
    path = args.out_dir / "report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "parameters": report["parameters"],
                      "selection": {k: v["selection"] for k, v in training.items()},
                      "sequences_where_d1_better": sequences_d1_better, "sequence_count": len(seq_table)}, indent=2))


if __name__ == "__main__":
    main()
