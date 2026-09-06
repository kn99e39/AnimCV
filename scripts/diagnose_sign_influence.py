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
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common.serialization import write_json
from framepose.bank import load_bank
from framepose.signs import (
    HINGE_CHAINS_BY_JOINT, NEGATIVE, POSITIVE, SIGN_FIELD_NAMES, UNKNOWN, oracle_sign_states,
    sign_state,
)
from framepose.train import geometry_tensor, load_checkpoint


HINGE_FIELDS = tuple(name for name in SIGN_FIELD_NAMES if name.endswith("_forward_bend"))


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
    parser = argparse.ArgumentParser(description="Frozen-weight sign-to-joint influence matrix")
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--checkpoint", action="append", required=True, help="NAME=PATH")
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
        "schema": "animcv_sign_influence_matrix_v1",
        "bank_content_digest": bank.content_digest(),
        "split": args.split,
        "frames": int(len(positions)),
        "method": ("frozen weights, fixed geometry, exactly one hinge sign toggled from its oracle "
                   "value to the opposite branch; degenerate fields are skipped per frame"),
        "joint_names": list(JOINT_NAMES),
        "checkpoints": {},
    }

    for entry in args.checkpoint:
        name, path = entry.split("=", 1)
        model, _ = load_checkpoint(path, device=str(device))
        model.eval()
        geometry_batch = torch.as_tensor(geometry[positions], device=device)
        base_signs = oracle[positions].astype(np.int64)
        with torch.no_grad():
            baseline = model(geometry_batch, None,
                             torch.as_tensor(base_signs, device=device)).float().cpu().numpy()

        per_field = {}
        for field in HINGE_FIELDS:
            index = SIGN_FIELD_NAMES.index(field)
            toggled = base_signs.copy()
            active = base_signs[:, index] != UNKNOWN
            toggled[active, index] = -toggled[active, index]
            with torch.no_grad():
                moved = model(geometry_batch, None,
                              torch.as_tensor(toggled, device=device)).float().cpu().numpy()

            # Millimetre displacement per output joint, over frames where the
            # toggle actually changed the input.
            displacement = np.linalg.norm(moved - baseline, axis=-1) * 1000.0
            scored = displacement[active]
            groups = _joint_groups(field, tuple(JOINT_NAMES))
            group_means = {key: float(scored[:, members].mean()) if members else None
                           for key, members in groups.items()}

            # Did the reconstructed sign of each hinge chain actually change?
            valid = bank.arrays["target_valid"][positions]
            base_state = np.stack([sign_state(baseline[order], valid[order])
                                   for order in range(len(positions))])
            moved_state = np.stack([sign_state(moved[order], valid[order])
                                    for order in range(len(positions))])
            sign_changes = {
                other: float((base_state[active, SIGN_FIELD_NAMES.index(other)]
                              != moved_state[active, SIGN_FIELD_NAMES.index(other)]).mean())
                for other in HINGE_FIELDS}

            per_field[field] = {
                "toggled_frames": int(active.sum()),
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
        report["checkpoints"][name] = {"path": path, "fields": per_field}

    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "sign_influence.json", report)
    for name, payload in report["checkpoints"].items():
        print(f"\n== {name}")
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
