#!/usr/bin/env python3
"""Sign Advisor experiment: does discrete sign evidence resolve the flips?

    S0  neutral   sign conditioning present, every field UNKNOWN
    S1  oracle    sign conditioning present, signs derived from ground-truth 3D
    S2  advisor   sign conditioning present, signs supplied by a VLM sign bank

S0 and S1 share one model graph, one parameter count, one seed, one frame set,
one optimizer, one loss and one evaluator. **The only thing that differs is the
sign information**, which is what makes this a capacity-matched control rather
than a repeat of the dense visual-fusion experiment.

This path is not introduced to lower MPJPE. The primary questions are whether
root/torso yaw, bilateral orientation and hinge flips resolve; continuous
position metrics are a guardrail reported separately, never a success score.

The oracle is an architecture control, not a production mechanism: it answers
"is sign evidence sufficient at all", with no RGB involved.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common.serialization import write_json
from framepose.bank import load_bank
from framepose.evaluate import compare, evaluate_predictions
from framepose.observations import assert_quality_interpretable
from framepose.signs import (
    SIGN_FIELD_NAMES, agreement, contract, mask_fields, oracle_sign_states, summarize,
)
from framepose.train import CandidateConfig, geometry_tensor, predict, sign_tensor, train_candidate


_HINGE_FIELDS = [name for name in SIGN_FIELD_NAMES if name.endswith("_forward_bend")]
_BILATERAL_FIELDS = ["shoulder_forward_depth", "hip_forward_depth"]

# Attribution candidates isolate WHICH sign information matters. They share the
# graph, all seven embedding tables, the parameter count, the seed, the frames,
# the optimizer, the loss, the evaluator and the schedule with S0/S1; only which
# fields carry oracle values changes, and every other field is UNKNOWN.
CANDIDATES = {
    "S0": {"name": "S0_neutral_sign", "sign_source": "neutral", "fields": []},
    "S1": {"name": "S1_oracle_sign", "sign_source": "oracle", "fields": list(SIGN_FIELD_NAMES)},
    "S2": {"name": "S2_advisor_sign", "sign_source": "advisor", "fields": list(SIGN_FIELD_NAMES)},
    "O_TORSO": {"name": "O_TORSO_oracle_facing_only", "sign_source": "oracle",
                "fields": ["torso_facing"]},
    "O_BILATERAL": {"name": "O_BILATERAL_oracle_forward_depth_only", "sign_source": "oracle",
                    "fields": list(_BILATERAL_FIELDS)},
    "O_HINGE": {"name": "O_HINGE_oracle_bend_only", "sign_source": "oracle",
                "fields": list(_HINGE_FIELDS)},
    "O_ORIENTATION": {"name": "O_ORIENTATION_oracle_facing_and_forward_depth",
                      "sign_source": "oracle",
                      "fields": ["torso_facing"] + list(_BILATERAL_FIELDS)},
    # Leave-one-field-out reads of the successful groups: a field is not shown
    # necessary by the group containing it having worked.
    "O_SHOULDER": {"name": "O_SHOULDER_oracle_shoulder_depth_only", "sign_source": "oracle",
                   "fields": ["shoulder_forward_depth"]},
    "O_HIP": {"name": "O_HIP_oracle_hip_depth_only", "sign_source": "oracle",
              "fields": ["hip_forward_depth"]},
    "O_ELBOWS": {"name": "O_ELBOWS_oracle_elbow_bend_only", "sign_source": "oracle",
                 "fields": ["left_elbow_forward_bend", "right_elbow_forward_bend"]},
    "O_KNEES": {"name": "O_KNEES_oracle_knee_bend_only", "sign_source": "oracle",
                "fields": ["left_knee_forward_bend", "right_knee_forward_bend"]},
    "O_LEFT_ELBOW": {"name": "O_LEFT_ELBOW_oracle", "sign_source": "oracle",
                     "fields": ["left_elbow_forward_bend"]},
    "O_RIGHT_ELBOW": {"name": "O_RIGHT_ELBOW_oracle", "sign_source": "oracle",
                      "fields": ["right_elbow_forward_bend"]},
    "O_LEFT_KNEE": {"name": "O_LEFT_KNEE_oracle", "sign_source": "oracle",
                    "fields": ["left_knee_forward_bend"]},
    "O_RIGHT_KNEE": {"name": "O_RIGHT_KNEE_oracle", "sign_source": "oracle",
                     "fields": ["right_knee_forward_bend"]},
    # Direct 3-of-4 leave-one-out from the successful O_HINGE set. Necessity of
    # a field can only be read from removing it from a set that works.
    "H_NO_LEFT_ELBOW": {"name": "H_NO_LEFT_ELBOW_oracle", "sign_source": "oracle",
                        "fields": [name for name in _HINGE_FIELDS
                                   if name != "left_elbow_forward_bend"]},
    "H_NO_RIGHT_ELBOW": {"name": "H_NO_RIGHT_ELBOW_oracle", "sign_source": "oracle",
                         "fields": [name for name in _HINGE_FIELDS
                                    if name != "right_elbow_forward_bend"]},
    "H_NO_LEFT_KNEE": {"name": "H_NO_LEFT_KNEE_oracle", "sign_source": "oracle",
                       "fields": [name for name in _HINGE_FIELDS
                                  if name != "left_knee_forward_bend"]},
    "H_NO_RIGHT_KNEE": {"name": "H_NO_RIGHT_KNEE_oracle", "sign_source": "oracle",
                        "fields": [name for name in _HINGE_FIELDS
                                   if name != "right_knee_forward_bend"]},
    # docs/32's conditional local-hinge topology candidates. Same fields as
    # their H_*/O_HINGE counterparts; run these with
    # --hinge-sign-injection post_attention. Never run with the default
    # pre_attention topology -- that would just reproduce O_HINGE/H_NO_*.
    "L_HINGE": {"name": "L_HINGE_oracle_bend_only_post_attention", "sign_source": "oracle",
                "fields": list(_HINGE_FIELDS)},
    "L_NO_LEFT_KNEE": {"name": "L_NO_LEFT_KNEE_oracle_post_attention", "sign_source": "oracle",
                       "fields": [name for name in _HINGE_FIELDS if name != "left_knee_forward_bend"]},
    "L_NO_RIGHT_KNEE": {"name": "L_NO_RIGHT_KNEE_oracle_post_attention", "sign_source": "oracle",
                        "fields": [name for name in _HINGE_FIELDS if name != "right_knee_forward_bend"]},
    "L_NO_RIGHT_ELBOW": {"name": "L_NO_RIGHT_ELBOW_oracle_post_attention", "sign_source": "oracle",
                         "fields": [name for name in _HINGE_FIELDS if name != "right_elbow_forward_bend"]},
    "L_NEUTRAL": {"name": "L_NEUTRAL_post_attention", "sign_source": "neutral", "fields": []},
}

COMPARISON_SEMANTICS = {
    "S1_vs_S0": ("capacity-matched: same graph, same parameter count, same seed; the only "
                 "variable is the sign information supplied"),
    "O_*_vs_S0": ("capacity-matched attribution: identical to S1 except that only the named "
                  "sign group carries oracle values, every other field being UNKNOWN"),
    "evidence_tiers": {
        "direct_contract_identical": ["shoulder_forward_depth_sign_disagreement_rate",
                                      "hip_forward_depth_sign_disagreement_rate",
                                      "sign_agreement on a fed field"],
        "structurally_coupled_downstream": ["root_yaw_error_degrees", "hinge_flip_rate",
                                            "hinge_direction_mae_degrees"],
        "independent_position_guardrails": ["mpjpe_mm", "pa_mpjpe_mm",
                                            "per_joint_mean_error_mm"],
    },
    "S2_vs_S1": "sign-recovery quality of the advisor against the sign-information upper bound",
    "S2_vs_S0": "end-to-end value of advisor-supplied sign evidence",
    "not_comparable_with": ("historical F1/F2, which tested a different hypothesis (dense visual "
                            "fusion of patch tokens into continuous XYZ)"),
    "primary_metrics": ["root_yaw_error_degrees", "shoulder_forward_depth_sign_disagreement_rate",
                        "hip_forward_depth_sign_disagreement_rate", "hinge_flip_rate",
                        "hinge_direction_mae_degrees", "sign_agreement"],
    "guardrail_metrics": ["mpjpe_mm", "pa_mpjpe_mm", "per_joint_mean_error_mm"],
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Sign-conditioned frame-pose experiment")
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--candidates", default="S0,S1")
    parser.add_argument("--advisor-signs", type=Path, default=None,
                        help="VLM sign bank .npz for the S2 candidate")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--evaluate-every", type=int, default=10)
    parser.add_argument("--no-mixed-precision", action="store_true")
    parser.add_argument("--compile-training-graph", action="store_true")
    parser.add_argument("--hinge-sign-injection", choices=("pre_attention", "post_attention"),
                        default="pre_attention",
                        help="docs/32: where the four hinge sign fields are injected relative to "
                             "global joint self-attention. Applies to every candidate in this "
                             "invocation; orientation fields are unaffected either way.")
    args = parser.parse_args()

    bank = load_bank(args.bank)
    bank.assert_split_isolation()
    regime = assert_quality_interpretable(bank.regime())
    geometry = geometry_tensor(bank)
    args.out.mkdir(parents=True, exist_ok=True)

    oracle = oracle_sign_states(bank.arrays["target_3d"], bank.arrays["target_valid"])
    advisor = None
    if args.advisor_signs is not None:
        with np.load(args.advisor_signs) as handle:
            advisor = handle["signs"].astype(np.int8)

    matrix = {
        "schema": "animcv_frame_pose_sign_experiment_v1",
        "bank_fingerprint": bank.fingerprint(args.bank),
        "observation_regime": regime,
        "sign_contract": contract(),
        "oracle_sign_distribution": summarize(oracle),
        "comparison_semantics": COMPARISON_SEMANTICS,
        "shared": {"epochs": args.epochs, "batch_size": args.batch_size,
                   "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
                   "seed": args.seed, "loss_contract": "baseline_geometry_v1",
                   "selection_split": "validation",
                   "execution_backend": "compiled" if args.compile_training_graph else "eager"},
        "candidates": {},
    }
    if advisor is not None:
        matrix["advisor_sign_accuracy_vs_oracle"] = agreement(advisor, oracle)

    reports: dict[str, dict] = {}
    for key in [item.strip() for item in args.candidates.split(",") if item.strip()]:
        definition = CANDIDATES[key]
        config = CandidateConfig(
            name=definition["name"], backbone="none", sign_source=definition["sign_source"],
            loss_contract="baseline_geometry_v1", epochs=args.epochs, batch_size=args.batch_size,
            learning_rate=args.learning_rate, weight_decay=args.weight_decay, seed=args.seed,
            device=args.device, mixed_precision=not args.no_mixed_precision,
            compile_training_graph=args.compile_training_graph, evaluate_every=args.evaluate_every,
            hinge_sign_injection=args.hinge_sign_injection)
        if definition["sign_source"] == "advisor":
            signs = advisor
        elif definition["sign_source"] == "neutral":
            signs = sign_tensor(bank, "neutral")
        else:
            # Oracle values for the declared group only; everything else UNKNOWN.
            signs = mask_fields(oracle, definition["fields"])
        directory = args.out / key
        directory.mkdir(parents=True, exist_ok=True)
        training = train_candidate(bank, config, geometry=geometry, signs=signs,
                                   checkpoint_path=directory / "checkpoint.pt")
        write_json(directory / "training_report.json", training)
        evaluation = _evaluate(bank, geometry, signs, directory, config, args)
        for split, report in evaluation.items():
            write_json(directory / f"evaluation_{split}.json", report)
        reports[key] = evaluation
        matrix["candidates"][key] = {
            "active_sign_fields": definition["fields"],
            "inactive_sign_fields": [name for name in SIGN_FIELD_NAMES if name not in definition["fields"]],
            "sign_source": definition["sign_source"],
            "config": config.to_dict(), "model": training["model"],
            "sign": training["sign"], "selection": training["selection"],
            "performance": training["performance"], "execution": training["execution"],
            "aggregate": {split: report["aggregate"] for split, report in evaluation.items()},
            "sign_agreement": {split: report["sign_agreement"] for split, report in evaluation.items()},
        }

    matrix["comparisons"] = {}
    pairs = [("S0", key) for key in reports if key != "S0"] + [("S1", "S2")]
    for split in ("validation", "test"):
        for baseline, candidate in pairs:
            if baseline not in reports or candidate not in reports:
                continue
            if split not in reports[baseline] or split not in reports[candidate]:
                continue
            delta = compare(reports[baseline][split], reports[candidate][split])
            matrix["comparisons"][f"{split}:{candidate}_vs_{baseline}"] = delta
            write_json(args.out / f"compare_{split}_{candidate}_vs_{baseline}.json", delta)

    write_json(args.out / "sign_experiment_matrix.json", matrix)
    print(json.dumps({key: {
        "test_mpjpe_mm": value["aggregate"].get("test", {}).get("mpjpe_mm", {}).get("mean"),
        "test_yaw_p95": value["aggregate"].get("test", {}).get("root_yaw_error_degrees", {}).get("p95"),
        "test_hinge_flip_rate": value["aggregate"].get("test", {}).get("hinge_flip_rate", {}).get("mean"),
        "test_sign_agreement": value["sign_agreement"].get("test", {}).get("overall", {}).get("agreement"),
    } for key, value in matrix["candidates"].items()}, indent=2, sort_keys=True))
    return 0


def _evaluate(bank, geometry, signs, directory: Path, config: CandidateConfig, args):
    import torch

    from framepose.train import load_checkpoint

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    model, _ = load_checkpoint(directory / "checkpoint.pt", device=str(device))
    evaluation = {}
    for split in ("validation", "test"):
        positions = bank.indices(split)
        if not len(positions):
            continue
        prediction = predict(model, torch, geometry, None, positions, device, signs=signs)
        np.save(directory / f"prediction_{split}.npy", prediction.astype(np.float32))
        evaluation[split] = evaluate_predictions(bank, positions, prediction, candidate=config.name)
    return evaluation


if __name__ == "__main__":
    raise SystemExit(main())
