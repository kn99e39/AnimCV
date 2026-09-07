#!/usr/bin/env python3
"""Frozen-weight sign-to-joint influence: does one hinge sign stay local?

Training-level attribution cannot separate two explanations of the observed
all-or-none hinge pattern:

    learned global adaptation      the model rearranged itself around the set
                                   of signs it was trained with
    runtime propagation            a single sign, injected into one joint query,
                                   spreads through global joint self-attention

This measures the second directly. Weights are frozen, the geometry input is
fixed, and exactly one hinge sign is toggled at a time. Whatever moves in the
output moved because of that one bit.

No retraining, no loss, no data change.

CONTRACT REPAIR (docs/32): the v1 probe fed every checkpoint the full 7-field
oracle as its baseline sign state, regardless of which fields that checkpoint
was actually trained with. For a single-group candidate like O_HINGE (trained
with the three orientation fields permanently UNKNOWN), that put the model in
an input state it never saw during training. v2 requires an explicit
per-checkpoint active-field contract and constructs the baseline as
``mask_fields(oracle, active_fields)`` -- every field the checkpoint was not
trained with stays UNKNOWN, exactly as it did during training -- and toggles
only fields that were active for that checkpoint. The contract is read from
``run_sign_experiments.CANDIDATES``, the single place that already recorded
what each checkpoint was trained with, never guessed from the checkpoint name.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_sign_experiments import CANDIDATES  # noqa: E402  -- the one training-time contract source

from common.serialization import write_json
from framepose.bank import load_bank
from framepose.signs import (
    HINGE_CHAINS_BY_JOINT, SIGN_FIELD_NAMES, UNKNOWN, mask_fields, oracle_sign_states, sign_state,
)
from framepose.train import geometry_tensor, load_checkpoint


HINGE_FIELDS = tuple(name for name in SIGN_FIELD_NAMES if name.endswith("_forward_bend"))
SCHEMA = "animcv_sign_influence_matrix_v2"


def active_fields_for(name: str) -> list[str]:
    """The one place this script is allowed to learn a checkpoint's training-
    time active sign fields: the CANDIDATES contract used to train it. Refuses
    rather than guessing from the checkpoint name for anything not in it."""
    if name not in CANDIDATES:
        raise ValueError(
            f"no active-sign-field contract for checkpoint name {name!r}; "
            f"add it to CANDIDATES in run_sign_experiments.py or pass --active-fields explicitly "
            f"(known names: {sorted(CANDIDATES)})"
        )
    return list(CANDIDATES[name]["fields"])


def verify_checkpoint_identity(name: str, payload: dict) -> dict:
    """Check that the checkpoint agrees with the contract it is filed under.

    docs/33 Section 2: the CLI label alone is not evidence. A checkpoint's own
    stored candidate name and conditioning topology must both match the
    CANDIDATES entry being used to build its baseline sign state, or the
    diagnostic is measuring one thing and reporting it as another.
    """
    contract = CANDIDATES[name]
    stored_candidate = dict(payload.get("candidate") or {})
    stored_model = dict(payload.get("model_config") or {})
    # Checkpoints trained before the topology switch existed carry no field;
    # they were all trained with the historical pre-attention injection.
    stored_topology = stored_model.get("hinge_sign_injection",
                                       stored_candidate.get("hinge_sign_injection", "pre_attention"))
    expected_topology = contract["expected_hinge_sign_injection"]
    stored_name = stored_candidate.get("name")
    if stored_name is not None and stored_name != contract["name"]:
        raise ValueError(
            f"checkpoint filed as {name!r} stores candidate name {stored_name!r} but the contract "
            f"declares {contract['name']!r}; refusing to attribute one candidate's weights to another")
    if stored_topology != expected_topology:
        raise ValueError(
            f"checkpoint filed as {name!r} was trained with hinge_sign_injection={stored_topology!r} "
            f"but the contract declares {expected_topology!r}; refusing to report a topology the "
            "weights do not have")
    return {"stored_candidate_name": stored_name, "contract_candidate_name": contract["name"],
            "hinge_sign_injection": stored_topology, "identity_verified": True}


def _chain_joints(field: str) -> tuple[str, ...]:
    joint = field[: -len("_forward_bend")]
    return HINGE_CHAINS_BY_JOINT[joint]


def _joint_groups(field: str, joint_names: tuple[str, ...]) -> dict[str, list[int]]:
    """Partition output joints relative to the toggled field."""
    routed = [joint_names.index(field[: -len("_forward_bend")])]
    chain = [joint_names.index(name) for name in _chain_joints(field)]
    other_chain = sorted({joint_names.index(name)
                          for other in HINGE_FIELDS if other != field
                          for name in _chain_joints(other)} - set(chain))
    rest = [index for index in range(len(joint_names))
            if index not in set(chain) | set(other_chain)]
    return {"routed": routed, "own_chain": chain, "other_hinge_chains": other_chain, "rest": rest}


def main() -> int:
    parser = argparse.ArgumentParser(description="Frozen-weight sign-to-joint influence matrix (v2, contract-repaired)")
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--checkpoint", action="append", required=True, help="NAME=PATH; NAME must be a run_sign_experiments.CANDIDATES key")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--frames", type=int, default=1500)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    import torch

    from framepose.contract import JOINT_NAMES

    bank = load_bank(args.bank)
    geometry = geometry_tensor(bank)
    oracle = oracle_sign_states(bank.arrays["target_3d"], bank.arrays["target_valid"])

    positions = bank.indices(args.split)
    if args.frames and args.frames < len(positions):
        step = len(positions) / args.frames
        positions = positions[np.unique(np.floor(np.arange(args.frames) * step).astype(np.int64))]

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    report = {
        "schema": SCHEMA,
        "bank_content_digest": bank.content_digest(),
        "split": args.split,
        "frames": int(len(positions)),
        "contract_source": "run_sign_experiments.CANDIDATES",
        "method": ("frozen weights, fixed geometry; baseline sign state is the oracle masked down to "
                   "exactly the checkpoint's own training-time active_sign_fields (every other field "
                   "UNKNOWN, matching training); exactly one ACTIVE hinge sign is toggled from its "
                   "oracle value to the opposite branch at a time; degenerate fields are skipped per frame"),
        "joint_names": list(JOINT_NAMES),
        "checkpoints": {},
    }

    for entry in args.checkpoint:
        name, path = entry.split("=", 1)
        active_fields = active_fields_for(name)
        active_hinge_fields = [field for field in HINGE_FIELDS if field in active_fields]

        model, payload = load_checkpoint(path, device=str(device))
        identity = verify_checkpoint_identity(name, payload)
        model.eval()
        geometry_batch = torch.as_tensor(geometry[positions], device=device)
        base_signs = mask_fields(oracle[positions], active_fields).astype(np.int64)
        with torch.no_grad():
            baseline = model(geometry_batch, None,
                             torch.as_tensor(base_signs, device=device)).float().cpu().numpy()

        if not active_hinge_fields:
            report["checkpoints"][name] = {
                "path": path, "active_sign_fields": active_fields, "identity": identity,
                "note": "no active hinge field for this checkpoint; primary in-distribution probe has nothing to toggle",
                "fields": {},
            }
            print(f"\n== {name}: no active hinge field, skipped")
            continue

        per_field = {}
        for field in active_hinge_fields:
            index = SIGN_FIELD_NAMES.index(field)
            toggled = base_signs.copy()
            active_frame_mask = base_signs[:, index] != UNKNOWN
            toggled[active_frame_mask, index] = -toggled[active_frame_mask, index]
            with torch.no_grad():
                moved = model(geometry_batch, None,
                              torch.as_tensor(toggled, device=device)).float().cpu().numpy()

            # Millimetre displacement per output joint, over frames where the
            # toggle actually changed the input.
            displacement = np.linalg.norm(moved - baseline, axis=-1) * 1000.0
            scored = displacement[active_frame_mask]
            groups = _joint_groups(field, tuple(JOINT_NAMES))
            group_means = {key: float(scored[:, members].mean()) if members else None
                           for key, members in groups.items()}

            # Did the reconstructed sign of each hinge chain actually change?
            # Read from the OUTPUT pose, so this is meaningful for every hinge
            # chain regardless of whether that chain's field was active for
            # this checkpoint's input.
            valid = bank.arrays["target_valid"][positions]
            base_state = np.stack([sign_state(baseline[order], valid[order])
                                   for order in range(len(positions))])
            moved_state = np.stack([sign_state(moved[order], valid[order])
                                    for order in range(len(positions))])
            sign_changes = {
                other: float((base_state[active_frame_mask, SIGN_FIELD_NAMES.index(other)]
                              != moved_state[active_frame_mask, SIGN_FIELD_NAMES.index(other)]).mean())
                for other in HINGE_FIELDS
            }

            per_field[field] = {
                "in_distribution": True,
                "toggled_frames": int(active_frame_mask.sum()),
                "per_joint_mean_displacement_mm": {
                    JOINT_NAMES[joint]: float(scored[:, joint].mean())
                    for joint in range(len(JOINT_NAMES))},
                "group_mean_displacement_mm": group_means,
                "target_over_non_target_ratio": (
                    None if not group_means["other_hinge_chains"]
                    else group_means["own_chain"] / group_means["other_hinge_chains"]),
                "own_chain_over_rest_ratio": (
                    None if not group_means["rest"]
                    else group_means["own_chain"] / group_means["rest"]),
                "reconstructed_sign_change_rate": sign_changes,
                "displacement_percentiles_mm": {
                    "own_chain": {q: float(np.percentile(scored[:, groups["own_chain"]], q))
                                  for q in (50, 90, 99)},
                    "other_hinge_chains": {
                        q: float(np.percentile(scored[:, groups["other_hinge_chains"]], q))
                        for q in (50, 90, 99)},
                },
            }
        report["checkpoints"][name] = {"path": path, "active_sign_fields": active_fields,
                                       "identity": identity, "fields": per_field}

    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "sign_influence.json", report)
    for name, payload in report["checkpoints"].items():
        print(f"\n== {name} (active: {payload.get('active_sign_fields')})")
        if not payload["fields"]:
            continue
        print("   %-26s %10s %10s %10s %8s %8s" % (
            "toggled sign", "own chain", "other hinge", "rest", "own/oth", "own-sign"))
        for field, value in payload["fields"].items():
            groups = value["group_mean_displacement_mm"]
            print("   %-26s %10.3f %10.3f %10.3f %8s %8.3f" % (
                field, groups["own_chain"], groups["other_hinge_chains"], groups["rest"],
                ("%.2f" % value["target_over_non_target_ratio"])
                if value["target_over_non_target_ratio"] else "-",
                value["reconstructed_sign_change_rate"][field]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
