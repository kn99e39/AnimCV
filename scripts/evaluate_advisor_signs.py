#!/usr/bin/env python3
"""S0 / S1 / S2 on one common frame subset, with frame-level review exports.

S2 reuses the **S1 checkpoint unchanged** and only swaps the sign input, per the
batch contract: the geometry core is not retrained to compensate for advisor
mistakes. That separates two different questions —

    S1 - S0   what correct sign evidence is worth (the upper bound)
    S2 - S0   what this advisor's sign evidence is actually worth
    S1 - S2   how much of the bound the advisor's sign quality loses

All three are scored on exactly the frames the advisor covered, so the
comparison is not contaminated by a different frame population.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common.serialization import read_json, write_json
from framepose.bank import load_bank
from framepose.evaluate import compare, evaluate_predictions
from framepose.signs import (
    SIGN_FIELDS, SIGN_FIELD_NAMES, agreement, neutral_sign_states, oracle_sign_states,
)
from framepose.train import geometry_tensor, load_checkpoint, predict


_FIELD_JOINTS = {definition.name: definition.joints for definition in SIGN_FIELDS}

REVIEW_CATEGORIES = {
    "facing_camera_reconstructed_as_away": ("torso_facing", -1),
    "facing_away_reconstructed_as_camera": ("torso_facing", 1),
    "shoulder_orientation_failure": ("shoulder_forward_depth", None),
    "hip_orientation_failure": ("hip_forward_depth", None),
    "left_elbow_flip": ("left_elbow_forward_bend", None),
    "right_elbow_flip": ("right_elbow_forward_bend", None),
    "left_knee_flip": ("left_knee_forward_bend", None),
    "right_knee_flip": ("right_knee_forward_bend", None),
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate advisor-supplied signs against S0/S1")
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--experiment-root", required=True, type=Path, help="sign_v1 root (S0/S1)")
    parser.add_argument("--sign-bank", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--review-per-category", type=int, default=6)
    args = parser.parse_args()

    import torch

    bank = load_bank(args.bank)
    geometry = geometry_tensor(bank)
    with np.load(args.sign_bank / "signs.npz") as handle:
        advisor = handle["signs"].astype(np.int8)
        covered = handle["covered"].astype(bool)
    summary = read_json(args.sign_bank / "sign_bank_summary.json")

    oracle = oracle_sign_states(bank.arrays["target_3d"], bank.arrays["target_valid"])
    neutral = neutral_sign_states(len(bank))
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")

    s0_model, _ = load_checkpoint(args.experiment_root / "S0" / "checkpoint.pt", device=str(device))
    s1_model, s1_payload = load_checkpoint(args.experiment_root / "S1" / "checkpoint.pt",
                                           device=str(device))
    candidates = {
        "S0": (s0_model, neutral, "neutral"),
        "S1": (s1_model, oracle, "oracle"),
        # Same weights as S1; only the sign input changes.
        "S2": (s1_model, advisor, "advisor"),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    matrix = {
        "schema": "animcv_frame_pose_advisor_evaluation_v1",
        "bank_content_digest": bank.content_digest(),
        "s2_uses_checkpoint": str(args.experiment_root / "S1" / "checkpoint.pt"),
        "s2_retrained": False,
        "advisor": {key: summary[key] for key in
                    ("model", "prompt", "generation", "invalid_response_count",
                     "invalid_response_rate", "frames_per_second", "covered_positions")},
        "advisor_sign_accuracy": summary["accuracy_vs_oracle"],
        "advisor_sign_accuracy_if_inverted": summary.get("accuracy_vs_oracle_if_inverted"),
        "splits": {},
    }

    reports: dict[str, dict[str, dict]] = {}
    for split in ("validation", "test"):
        positions = np.asarray([index for index in bank.indices(split) if covered[index]],
                               dtype=np.int64)
        if not len(positions):
            continue
        matrix["splits"][split] = {"scored_frames": int(len(positions))}
        for key, (model, signs, source) in candidates.items():
            prediction = predict(model, torch, geometry, None, positions, device, signs=signs)
            np.save(args.out / f"prediction_{key}_{split}.npy", prediction.astype(np.float32))
            report = evaluate_predictions(bank, positions, prediction, candidate=f"{key}_{source}")
            write_json(args.out / f"evaluation_{key}_{split}.json", report)
            reports.setdefault(key, {})[split] = report
            matrix["splits"][split][key] = {
                "sign_source": source,
                "aggregate": report["aggregate"],
                "sign_agreement": report["sign_agreement"],
            }
        matrix["splits"][split]["advisor_sign_accuracy_on_scored_frames"] = agreement(
            advisor[positions], oracle[positions])

    matrix["comparisons"] = {}
    for split in matrix["splits"]:
        for baseline, candidate in (("S0", "S1"), ("S0", "S2"), ("S1", "S2")):
            delta = compare(reports[baseline][split], reports[candidate][split])
            matrix["comparisons"][f"{split}:{candidate}_vs_{baseline}"] = delta

    matrix["review"] = _review(bank, reports, advisor, oracle, covered,
                               args.review_per_category)
    write_json(args.out / "advisor_evaluation.json", matrix)
    write_json(args.out / "frame_review.json",
               {"schema": "animcv_frame_pose_sign_review_v1", "categories": matrix["review"]})

    print(json.dumps({split: {key: {
        "mpjpe_mm": value["aggregate"]["mpjpe_mm"]["mean"],
        "yaw_p95": value["aggregate"]["root_yaw_error_degrees"]["p95"],
        "hinge_flip_rate": value["aggregate"]["hinge_flip_rate"]["mean"],
        "sign_agreement": value["sign_agreement"]["overall"]["agreement"],
    } for key, value in payload.items() if key in ("S0", "S1", "S2")}
        for split, payload in matrix["splits"].items()}, indent=2, sort_keys=True))
    return 0


def _review(bank, reports, advisor, oracle, covered, per_category: int) -> dict:
    """Frames where a named flip category actually occurred under S0."""
    split = "test" if "test" in reports["S0"] else "validation"
    frames = {key: {record["sample_id"]: record for record in reports[key][split]["frames"]}
              for key in reports}
    positions = np.asarray([index for index in bank.indices(split) if covered[index]], dtype=np.int64)

    output: dict[str, list] = {}
    for category, (field, wanted_oracle) in REVIEW_CATEGORIES.items():
        index = SIGN_FIELD_NAMES.index(field)
        picked = []
        for position in positions:
            sample = bank.samples[int(position)]
            reference = int(oracle[position, index])
            if reference == 0:
                continue
            if wanted_oracle is not None and reference != wanted_oracle:
                continue
            s0_record = frames["S0"].get(sample.sample_id)
            if s0_record is None:
                continue
            # The S0 reconstruction must actually have the wrong branch here.
            s0_sign = _predicted_sign(reports, "S0", split, sample.sample_id, index)
            if s0_sign is None or s0_sign == reference:
                continue
            entry = {
                "sample_id": sample.sample_id,
                "sequence_id": sample.sequence_id,
                "frame_index": sample.frame_index,
                "image_reference": sample.image_reference.to_dict() if sample.image_reference else None,
                "sign_field": field,
                "joints": list(_FIELD_JOINTS[field]),
                "oracle_sign": reference,
                "advisor_sign": int(advisor[position, index]),
                "advisor_sign_correct": bool(advisor[position, index] == reference),
            }
            for key in ("S0", "S1", "S2"):
                record = frames[key].get(sample.sample_id, {})
                entry[key] = {
                    "reconstructed_sign": _predicted_sign(reports, key, split, sample.sample_id, index),
                    "mpjpe_mm": record.get("mpjpe_mm"),
                    "root_yaw_error_degrees": record.get("root_yaw_error_degrees"),
                    "hinge_flip_rate": record.get("hinge_flip_rate"),
                    "shoulder_forward_depth_sign_disagreement": record.get(
                        "shoulder_forward_depth_sign_disagreement"),
                    "hip_forward_depth_sign_disagreement": record.get(
                        "hip_forward_depth_sign_disagreement"),
                }
            entry["sign_changed_S0_to_S1"] = entry["S0"]["reconstructed_sign"] != entry["S1"]["reconstructed_sign"]
            entry["sign_changed_S0_to_S2"] = entry["S0"]["reconstructed_sign"] != entry["S2"]["reconstructed_sign"]
            entry["position_delta_S1_minus_S0_mm"] = (
                None if entry["S1"]["mpjpe_mm"] is None or entry["S0"]["mpjpe_mm"] is None
                else entry["S1"]["mpjpe_mm"] - entry["S0"]["mpjpe_mm"])
            entry["position_delta_S2_minus_S0_mm"] = (
                None if entry["S2"]["mpjpe_mm"] is None or entry["S0"]["mpjpe_mm"] is None
                else entry["S2"]["mpjpe_mm"] - entry["S0"]["mpjpe_mm"])
            picked.append(entry)
            if len(picked) >= per_category:
                break
        output[category] = picked
    return output


_SIGN_CACHE: dict[tuple[str, str], dict[str, dict]] = {}


def _predicted_sign(reports, key: str, split: str, sample_id: str, index: int) -> int | None:
    """The reconstructed sign the evaluator read back out of that frame."""
    cache = _SIGN_CACHE.setdefault((key, split), {})
    if not cache:
        for record in reports[key][split]["frames"]:
            cache[record["sample_id"]] = record
    record = cache.get(sample_id)
    if record is None:
        return None
    return int(record["sign_state"][SIGN_FIELD_NAMES[index]])


if __name__ == "__main__":
    raise SystemExit(main())
