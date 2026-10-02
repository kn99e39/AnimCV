#!/usr/bin/env python3
"""DIAGNOSTIC (Worklog 66): R0_ZERO_RELATION vs R1_DEPTH_RELATION late fusion over frozen H0.

References: A frozen H0, B Worklog 65 D0, C Worklog 65 D1 (read from the
Worklog 65 per-row outputs), D R0, E R1.  Worklog 65 evidence is consumed
byte-for-byte from its cache; H0 is consumed byte-for-byte from its .npy files.
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
    SEGMENT_NAMES, ProbeConfig, angle_degrees, elevation_degrees, forward_targets, observed_plane,
    reconstruct_direction, segment_vectors,
)
from framepose.bank import load_bank
from framepose.relational_depth_fusion import (
    CANDIDATES, depth_relations, fusion_inputs, h0_forward, predict_fusion, train_fusion,
)
from pose.framepose_bridge import assemble_h0

CLEAN = ("courtyard_dancing_00", "courtyard_hug_00")
STABLE_SINE = math.sin(math.radians(10))
F_BINS = ((0.0, STABLE_SINE), (STABLE_SINE, 0.5), (0.5, 1.0001))
MODELS = ("h0", "d0", "d1", "r0", "r1")


def _stats(values):
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if not len(v):
        return {"n": 0}
    q = lambda p: float(np.percentile(v, p))  # noqa: E731
    return {"n": int(len(v)), "mean": float(v.mean()), "p50": q(50), "p95": q(95)}


def _paired(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    return {**_stats(d), "improved_share": float(np.mean(d < 0)) if len(d) else None,
            "worsened_share": float(np.mean(d > 0)) if len(d) else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bank", "h0_train", "h0_validation", "h0_test", "h0_checkpoint", "semantics_report",
                 "evidence_dir", "w65_dir", "out_dir"):
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
    npz = args.evidence_dir / "evidence.npz"
    npz_sha = hashlib.sha256(npz.read_bytes()).hexdigest()
    if (npz_sha != manifest["evidence_npz_sha256"] or manifest["identity"]["bank_content_digest"] != bank.content_digest()
            or manifest["sample_ids_sha256"] != hashlib.sha256("\n".join(s.sample_id for s in bank.samples).encode()).hexdigest()):
        raise SystemExit("Worklog 65 evidence cache identity changed")
    evidence = np.load(npz)
    relations, relation_ok = depth_relations(evidence["forward_evidence"], evidence["available"])
    h0_f = h0_forward(h0)
    targets, masks = forward_targets(bank.arrays["target_3d"], bank.arrays["target_valid"])
    train, validation, test = bank.indices("train"), bank.indices("validation"), bank.indices("test")

    c = torch.load(args.h0_checkpoint, map_location="cpu", weights_only=False)["candidate"]
    config = ProbeConfig(epochs=c["epochs"], batch_size=c["batch_size"], learning_rate=c["learning_rate"],
                         weight_decay=c["weight_decay"], minimum_learning_rate=c["minimum_learning_rate"],
                         seed=c["seed"], mixed_precision=c["mixed_precision"], evaluate_every=c["evaluate_every"],
                         device=args.device)
    inputs = {name: fusion_inputs(h0_f, relations, name) for name in CANDIDATES}
    if not np.array_equal(inputs["R0_ZERO_RELATION"][:, :3], inputs["R1_DEPTH_RELATION"][:, :3]) or \
            np.any(inputs["R0_ZERO_RELATION"][:, 3:] != 0) or \
            not np.array_equal(inputs["R1_DEPTH_RELATION"][:, 3:], relations.astype(np.float32)):
        raise SystemExit("R0/R1 input contract violated")
    models, training = {}, {}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name in CANDIDATES:
        model, report = train_fusion(inputs[name], targets, masks, train, validation, config)
        models[name] = model
        training[name] = {**report, "parameters": int(sum(p.numel() for p in model.parameters()))}
        torch.save({"candidate": name, "probe_config": config.to_dict(), "evidence_npz_sha256": npz_sha,
                    "selection": report["selection"], "state_dict": model.state_dict()},
                   args.out_dir / f"{name}.pt")

    def split_error(positions):
        m = masks[positions]
        return float((np.abs(h0_f[positions] - targets[positions]) * m).sum() / m.sum())
    h0_in_sample = {"train": split_error(train), "validation": split_error(validation), "test": split_error(test)}

    def load_rows(path):
        rows = json.loads(path.read_text())
        positions = np.array([r["position"] for r in rows])
        sizes = np.asarray([bank.samples[p].image_size for p in positions], float)
        planes, _ = observed_plane(bank.arrays["input_2d"][positions], sizes)
        gt_vec = segment_vectors(bank.arrays["target_3d"][positions])
        preds = {"r0": predict_fusion(models["R0_ZERO_RELATION"], inputs["R0_ZERO_RELATION"], positions),
                 "r1": predict_fusion(models["R1_DEPTH_RELATION"], inputs["R1_DEPTH_RELATION"], positions)}
        consistency = 0.0
        for i, row in enumerate(rows):
            p = positions[i]
            for k, name in enumerate(SEGMENT_NAMES):
                consistency = max(consistency, abs(row[f"{name}_f_h0"] - h0_f[p, k]), abs(row[f"{name}_f_gt"] - targets[p, k]))
                g = gt_vec[name][i]
                gu = g / np.linalg.norm(g)
                for m in ("r0", "r1"):
                    f = preds[m][i, k]
                    row[f"{name}_f_{m}"] = float(f)
                    row[f"{name}_{m}_raw_plane"] = float(angle_degrees(reconstruct_direction(np.array([f]), planes[name][i:i + 1]), gu[None])[0])
                row[f"{name}_relation"] = float(relations[p, k]) if relation_ok[p, k] else None
        return rows, consistency

    val_rows, c1 = load_rows(args.w65_dir / "rows_validation.json")
    test_rows, c2 = load_rows(args.w65_dir / "rows_test.json")
    if max(c1, c2) > 1e-9:
        raise SystemExit(f"Worklog 65 rows disagree with frozen H0 / targets: {max(c1, c2)}")

    def metrics(rows):
        out = {"n": len(rows)}
        for name in SEGMENT_NAMES:
            fg = np.array([r[f"{name}_f_gt"] for r in rows])
            seg = {}
            err = {}
            for m in MODELS:
                f = np.array([r[f"{name}_f_{m}"] for r in rows])
                err[m] = {"abs_f": np.abs(f - fg),
                          "elevation": np.abs(elevation_degrees(f) - elevation_degrees(fg)),
                          "full": np.array([r[f"{name}_{m}_raw_plane"] for r in rows])}
                seg[m] = {k: _stats(v) for k, v in err[m].items()}
            seg["r1_minus_r0"] = {k: _paired(err["r1"][k], err["r0"][k]) for k in ("abs_f", "elevation", "full")}
            seg["r1_minus_h0"] = {k: _paired(err["r1"][k], err["h0"][k]) for k in ("abs_f", "elevation", "full")}
            seg["by_abs_gt_f"] = {}
            for lo, hi in F_BINS:
                sel = (np.abs(fg) >= lo) & (np.abs(fg) < hi)
                seg["by_abs_gt_f"][f"{lo:.3f}-{min(hi, 1):.3f}"] = {
                    "n": int(sel.sum()), **{f"{m}_abs_f_mean": float(err[m]["abs_f"][sel].mean()) if sel.any() else None
                                            for m in MODELS},
                    "r1_minus_r0_abs_f": _paired(err["r1"]["abs_f"][sel], err["r0"]["abs_f"][sel]),
                    "r1_minus_h0_abs_f": _paired(err["r1"]["abs_f"][sel], err["h0"]["abs_f"][sel])}
            out[name] = seg
        return out

    heldout = val_rows + test_rows
    other_val = [r for r in val_rows if not any(c in r["sequence_id"] for c in CLEAN) and "crosscountry" not in r["sequence_id"]]
    scopes = {"test_all": test_rows, "validation_all": val_rows,
              "validation_clean_dancing_hug": [r for r in val_rows if any(c in r["sequence_id"] for c in CLEAN)],
              "validation_crosscountry_only": [r for r in val_rows if "crosscountry" in r["sequence_id"]],
              "all_other_heldout": other_val + test_rows, "validation_other": other_val}

    quartiles = {}
    for name in SEGMENT_NAMES:
        rows = [r for r in heldout if r[f"{name}_relation"] is not None]
        mag = np.abs(np.array([r[f"{name}_relation"] for r in rows]))
        edges = np.quantile(mag, [0.25, 0.5, 0.75]).tolist()
        bounds = [-np.inf, *edges, np.inf]
        seg = {"edges": edges, "rows_without_relation": len(heldout) - len(rows)}
        fg = np.array([r[f"{name}_f_gt"] for r in rows])
        errs = {m: np.abs(np.array([r[f"{name}_f_{m}"] for r in rows]) - fg) for m in MODELS}
        for q in range(4):
            sel = (mag >= bounds[q]) & (mag < bounds[q + 1])
            seg[f"Q{q + 1}"] = {"n": int(sel.sum()), **{f"{m}_abs_f_mean": float(errs[m][sel].mean()) for m in MODELS},
                                "r1_minus_r0": _paired(errs["r1"][sel], errs["r0"][sel]),
                                "r1_minus_h0": _paired(errs["r1"][sel], errs["h0"][sel])}
        quartiles[name] = seg

    complementarity = {}
    for name in SEGMENT_NAMES:
        rows = [r for r in heldout if r[f"{name}_relation"] is not None and abs(r[f"{name}_f_gt"]) > STABLE_SINE]
        table = {}
        for label, h_ok, d_ok in (("h0_wrong_da_right", False, True), ("h0_right_da_wrong", True, False),
                                  ("both_right", True, True), ("both_wrong", False, False)):
            sel = [r for r in rows
                   if (np.sign(r[f"{name}_f_h0"]) == np.sign(r[f"{name}_f_gt"])) == h_ok
                   and (np.sign(r[f"{name}_relation"]) == np.sign(r[f"{name}_f_gt"])) == d_ok]
            fg = np.array([r[f"{name}_f_gt"] for r in sel])
            entry = {"n": len(sel)}
            for m in MODELS:
                if sel:
                    f = np.array([r[f"{name}_f_{m}"] for r in sel])
                    entry[f"{m}_abs_f_mean"] = float(np.abs(f - fg).mean())
                    entry[f"{m}_sign_correct"] = float(np.mean(np.sign(f) == np.sign(fg)))
            table[label] = entry
        complementarity[name] = {"stable_rows": len(rows), "definition": "|GT f| > sin(10 deg); sign of H0 f and of the DA relation vs GT", **table}

    per_seq = defaultdict(list)
    for r in heldout:
        per_seq[r["sequence_id"]].append(r)
    sequence = {}
    for name in SEGMENT_NAMES:
        contributions = {}
        better_r0 = better_h0 = 0
        for sid, rows in per_seq.items():
            fg = np.array([r[f"{name}_f_gt"] for r in rows])
            e = {m: np.abs(np.array([r[f"{name}_f_{m}"] for r in rows]) - fg) for m in ("h0", "r0", "r1")}
            better_r0 += e["r1"].mean() < e["r0"].mean()
            better_h0 += e["r1"].mean() < e["h0"].mean()
            contributions[sid] = float((e["r1"] - e["r0"]).sum())
        total = sum(contributions.values())
        top = min(contributions.items(), key=lambda kv: kv[1])
        sequence[name] = {"sequences": len(per_seq), "r1_better_than_r0": int(better_r0),
                          "r1_better_than_h0": int(better_h0),
                          "total_r1_minus_r0_abs_f_sum": total,
                          "largest_single_sequence_gain": {"sequence_id": top[0], "sum": top[1],
                                                           "share_of_total": top[1] / total if total else None}}

    owner_src = json.loads((args.w65_dir / "report.json").read_text())["owner_cases"]
    by_key = {(r["sequence_id"], r["frame_index"]): r for r in val_rows}
    owner = {label: {"sequence_id": c["sequence_id"], "frame_index": c["frame_index"], "overlaps": c["overlaps"],
                     "row": by_key.get((c["sequence_id"], c["frame_index"]))} for label, c in owner_src.items()}

    report = {
        "schema": "animcv_framepose_relational_depth_fusion_v1",
        "labels": {"A": "frozen H0", "B": "Worklog 65 D0", "C": "Worklog 65 D1", "D": "R0_ZERO_RELATION",
                   "E": "R1_DEPTH_RELATION"},
        "evidence_npz_sha256": npz_sha, "evidence_identity": manifest["identity"],
        "h0_split_sha256": identity.split_sha256, "h0_abs_f_error_by_split": h0_in_sample,
        "graph": "6 -> Linear(32) -> GELU -> Linear(32) -> GELU -> Linear(3) -> tanh",
        "training": training, "scopes": {k: metrics(v) for k, v in scopes.items()},
        "relation_quartiles_heldout": quartiles, "complementarity_heldout": complementarity,
        "per_sequence": sequence, "owner_cases": owner,
    }
    (args.out_dir / "rows_validation.json").write_text(json.dumps(val_rows))
    (args.out_dir / "rows_test.json").write_text(json.dumps(test_rows))
    path = args.out_dir / "report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                      "selection": {k: v["selection"] for k, v in training.items()},
                      "parameters": {k: v["parameters"] for k, v in training.items()},
                      "h0_abs_f_error_by_split": h0_in_sample}, indent=2))


if __name__ == "__main__":
    main()
