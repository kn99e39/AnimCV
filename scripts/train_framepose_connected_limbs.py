#!/usr/bin/env python3
"""DIAGNOSTIC (Worklog 86): G0 / G1 connected-limb models for the right arm and the right leg.

Exactly the Worklog 80 / 85 J1X training contract (``train_framepose_joint_arm_geometry.train_joint``,
called unchanged): 3DPW TRAIN + BEDLAM TRAIN (Worklog 73 historical + Worklog 85 expansion), 50/50
source-balanced draws over chain-valid rows, 11,334 draws/epoch, 200 epochs, AdamW 3e-4 -> 1e-5 cosine,
batch 256, seed 1337, selection = lowest 3DPW VALIDATION chain objective.  Per chain, G0 and G1 differ
only in the visual tensor (exact zeros vs the frozen-Qwen chain readout); same rows, init and draws.
Chain-specific inputs only: the chain's targets, the Worklog 69 pair features of the chain's detector
joints and the chain's readout.  No new mechanism, width, loss or sampler.
Order (each check stops the run):
  1. identities (banks, readout caches, J1X checkpoint);
  2. GENERIC-PIPELINE GATE: the left arm through this chain code path (stored Worklog 68/85 readouts,
     ``chain_pair_geometry``, ``chain_targets``) must retrain J1X bit-exactly;
  3. left-arm reference control G0X (zero visual, the J1X population) -- the matched 2D-only partner of J1X;
  4. right arm G0 / G1 and right leg G0 / G1, once each.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_external_visual_supervision_transfer import load_cache  # noqa: E402
from train_framepose_joint_arm_geometry import train_joint  # noqa: E402

from framepose.bank import load_bank  # noqa: E402
from framepose.bedlam_supervision import ExternalBank  # noqa: E402
from framepose.connected_limb_transfer import chain_pair_geometry, chain_targets  # noqa: E402
from framepose.train import geometry_tensor  # noqa: E402

J1X_SHA256 = "82fd6bcb2cf49aac8c5aebb124335cc3fd890ec701160bab714f54fa6c7839f5"
NEW_CHAINS = ("right_arm", "right_leg")   # right leg chosen by the data-only LEG_RULE (contract audit)
G = {"g0": "J0_GEOMETRY_ONLY", "g1": "J1_GEOMETRY_PLUS_VISION"}
BANK_ORDER = ("3dpw", "bedlam_train_historical", "bedlam_train_w85_new", "bedlam_validation")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assemble(banks, readouts, chain):
    """Row-stacked chain inputs in the Worklog 85 order [3DPW, BEDLAM hist, BEDLAM new, BEDLAM val]."""
    geometry = np.concatenate([chain_pair_geometry(geometry_tensor(banks[k]), chain) for k in BANK_ORDER]).reshape(-1, 39)
    t3 = np.concatenate([banks[k].arrays["target_3d"] for k in BANK_ORDER])
    tv = np.concatenate([banks[k].arrays["target_valid"] for k in BANK_ORDER])
    elbow_t, wrist_t, mask = chain_targets(t3, tv, chain)
    n = [len(banks[k]) for k in BANK_ORDER]
    off = np.cumsum([0] + n)
    fb = banks["3dpw"]
    train3, val3 = fb.indices("train"), fb.indices("validation")
    ext_train = np.arange(off[1], off[3])
    ext_val = np.arange(off[3], off[4])
    return {"readouts": np.concatenate([readouts[k] for k in BANK_ORDER]), "geometry": geometry, "elbow_t": elbow_t,
            "wrist_t": wrist_t, "mask": mask, "primary": train3[mask[train3]], "external": ext_train[mask[ext_train]],
            "val": val3[mask[val3]], "ext_val": ext_val[mask[ext_val]]}


def train(d, cand, device):
    return train_joint(G[cand], d["readouts"], d["geometry"], d["elbow_t"], d["wrist_t"], d["mask"], d["primary"],
                       d["external"], d["val"], d["ext_val"], device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bank", "cache_dir", "hist_train", "hist_train_cache", "new_train", "new_train_cache", "bedlam_val",
                 "bedlam_val_cache", "limb_readouts", "w85_training", "contract_audit", "out_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    import torch

    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise SystemExit("STOP: output directory is not empty; never overwrite experimental artifacts")
    device = torch.device(args.device)
    audit = json.loads(args.contract_audit.read_text())
    if audit["leg_selection"]["selected"] != "right_leg":
        raise SystemExit("STOP: the contract audit selected a different leg")
    j1x_path = args.w85_training / "J1_BEDLAM_EXPANDED.pt"
    if _sha(j1x_path) != J1X_SHA256:
        raise SystemExit("STOP: J1X checkpoint identity mismatch")
    banks = {"3dpw": load_bank(args.bank), "bedlam_train_historical": ExternalBank.load(args.hist_train),
             "bedlam_train_w85_new": ExternalBank.load(args.new_train), "bedlam_validation": ExternalBank.load(args.bedlam_val)}
    caches = {"3dpw": args.cache_dir, "bedlam_train_historical": args.hist_train_cache, "bedlam_train_w85_new": args.new_train_cache,
              "bedlam_validation": args.bedlam_val_cache}
    stored, ident3 = {}, None
    for k in BANK_ORDER:
        stored[k], ident = load_cache(caches[k], banks[k].content_digest(), [s.sample_id for s in banks[k].samples], ident3)
        ident3 = ident3 or ident
    rep = json.loads((args.limb_readouts / "readout_report.json").read_text())
    limb = {}
    for k in BANK_ORDER:
        info = rep["banks"][k]
        if info["bank_content_digest"] != banks[k].content_digest() or info["npz_sha256"] != _sha(args.limb_readouts / f"{k}.npz"):
            raise SystemExit(f"STOP: limb readout cache identity mismatch for {k}")
        limb[k] = np.load(args.limb_readouts / f"{k}.npz")
    identities = {k: banks[k].content_digest() for k in BANK_ORDER} | {"j1x": J1X_SHA256, "contract_audit": _sha(args.contract_audit)}

    # --- generic-pipeline gate: left arm through the chain code must retrain J1X bit-exactly ---
    left = assemble(banks, stored, "left_arm")
    model, info, vis = train(left, "g1", device)
    hist = torch.load(j1x_path, map_location="cpu", weights_only=False)
    state = model.state_dict()
    gate = {"selected_epoch": info["selection"]["epoch"], "j1x_selected_epoch": hist["selection"]["epoch"],
            "state_bit_exact": all(torch.equal(state[k].cpu(), hist["state_dict"][k].cpu()) for k in hist["state_dict"])}
    gate["passed"] = gate["state_bit_exact"] and gate["selected_epoch"] == gate["j1x_selected_epoch"]
    print(json.dumps({"generic_pipeline_gate": gate}), flush=True)
    if not gate["passed"]:
        raise SystemExit("STOP: the chain pipeline does not reproduce J1X")
    del model, vis
    torch.cuda.empty_cache()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    jobs = [("left_arm", "g0", stored)] + [(c, g, {k: limb[k][f"readout_{c}"] for k in BANK_ORDER}) for c in NEW_CHAINS for g in ("g0", "g1")]
    for chain, cand, readouts in jobs:
        d = assemble(banks, readouts, chain)
        model, info, vis = train(d, cand, device)
        name = f"{chain}_{cand}"
        torch.save({"candidate": name, "architecture": G[cand], "chain": chain, "selection": info["selection"],
                    "state_dict": model.state_dict()}, args.out_dir / f"{name}.pt")
        results[name] = {**info, "eligibility": {"3dpw_train": int(len(d["primary"])), "bedlam_train": int(len(d["external"])),
                                                 "3dpw_validation": int(len(d["val"])), "bedlam_validation": int(len(d["ext_val"]))}}
        print(json.dumps({name: {"selection": info["selection"]["epoch"], "validation": info["selection"]["validation"]["objective"],
                                 "external_validation": info["selection"]["external_validation"]["objective"]}}), flush=True)
        del model, vis
        torch.cuda.empty_cache()
    control = {c: {k: results[f"{c}_g0"][k] == results[f"{c}_g1"][k] for k in ("init_digest", "parameters", "steps", "draws", "source_draws")}
               for c in NEW_CHAINS}
    if not all(all(v.values()) for v in control.values()):
        raise SystemExit(f"STOP: G0/G1 control identity broken: {control}")
    report = {"schema": "animcv_w86_connected_limb_training_v1", "identities": identities, "generic_pipeline_gate": gate,
              "control_identity": control, "candidates": results,
              "checkpoints_sha256": {n: _sha(args.out_dir / f"{n}.pt") for n in results}}
    (args.out_dir / "training_report.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps({"checkpoints": report["checkpoints_sha256"], "control": control}, indent=1))


if __name__ == "__main__":
    main()
