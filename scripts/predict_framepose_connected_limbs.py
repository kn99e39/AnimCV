#!/usr/bin/env python3
"""EVALUATION ONLY (Worklog 86): frozen connected-limb candidates on 3DPW TEST and MPI-INF-3DHP TEST.

Per chain, identical observations for every candidate (detector 2D, chain pair geometry, chain readout):
  * left arm (reference): J0 (Worklog 80), J1X (Worklog 85), G0X (Worklog 86 matched 2D-only control);
  * right arm: G0, G1;  right leg: G0, G1.
Gate: J0 / J1X left-arm outputs reproduce the stored Worklog 85 predictions (<= 1e-6).
Writes ``predictions_{3dpw,3dhp}.npz`` with ``<chain>_<candidate>_{elbow,wrist}`` (middle / distal) and
``<chain>_target_{elbow,wrist,valid}``.
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

from framepose.bank import load_bank  # noqa: E402
from framepose.bedlam_supervision import ExternalBank  # noqa: E402
from framepose.connected_limb_transfer import chain_pair_geometry, chain_targets  # noqa: E402
from framepose.joint_arm_geometry import build_joint_arm, candidate_visual, decode_numpy, predict_joint  # noqa: E402
from framepose.train import geometry_tensor  # noqa: E402

CANDIDATES = {"left_arm": {"j0": "J0_GEOMETRY_ONLY", "j1x": "J1_GEOMETRY_PLUS_VISION", "g0x": "J0_GEOMETRY_ONLY"},
              "right_arm": {"g0": "J0_GEOMETRY_ONLY", "g1": "J1_GEOMETRY_PLUS_VISION"},
              "right_leg": {"g0": "J0_GEOMETRY_ONLY", "g1": "J1_GEOMETRY_PLUS_VISION"}}
W85_PREDICTIONS = {"3dpw": "81eae1cb0f18a584e576de0e096f99227632a480c02d4ec46448be461c11980c",
                   "3dhp": "c70f3eb229d11c08cc722b63a6b969d77b51bef707837501ba73febd30e65db5"}


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bank", "cache_dir", "w83_dir", "w80_joint_dir", "w85_dir", "limb_readouts", "limb_training", "out_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    import torch

    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise SystemExit("STOP: output directory is not empty; never overwrite experimental artifacts")
    device = torch.device(args.device)
    training = json.loads((args.limb_training / "training_report.json").read_text())
    ckpt = {"left_arm": {"j0": args.w80_joint_dir / "J0_GEOMETRY_ONLY.pt", "j1x": args.w85_dir / "training" / "J1_BEDLAM_EXPANDED.pt",
                         "g0x": args.limb_training / "left_arm_g0.pt"},
            "right_arm": {g: args.limb_training / f"right_arm_{g}.pt" for g in ("g0", "g1")},
            "right_leg": {g: args.limb_training / f"right_leg_{g}.pt" for g in ("g0", "g1")}}
    for chain in ("right_arm", "right_leg"):
        for g in ("g0", "g1"):
            if _sha(ckpt[chain][g]) != training["checkpoints_sha256"][f"{chain}_{g}"]:
                raise SystemExit(f"STOP: {chain}_{g} changed after training")
    if _sha(ckpt["left_arm"]["g0x"]) != training["checkpoints_sha256"]["left_arm_g0"]:
        raise SystemExit("STOP: left_arm_g0 changed after training")
    fb = load_bank(args.bank)
    mx = ExternalBank.load(args.w83_dir / "bank" / "mpi3dhp_test")
    r3, ident3 = load_cache(args.cache_dir, fb.content_digest(), [s.sample_id for s in fb.samples], None)
    rx, _ = load_cache(args.w83_dir / "qwen", mx.content_digest(), [s.sample_id for s in mx.samples], ident3)
    limb_rep = json.loads((args.limb_readouts / "readout_report.json").read_text())
    limb = {}
    for k, b in (("3dpw", fb), ("mpi3dhp_test", mx)):
        info = limb_rep["banks"][k]
        if info["bank_content_digest"] != b.content_digest() or info["npz_sha256"] != _sha(args.limb_readouts / f"{k}.npz"):
            raise SystemExit(f"STOP: limb readout identity mismatch for {k}")
        limb[k] = np.load(args.limb_readouts / f"{k}.npz")
    out = {"3dpw": {}, "3dhp": {}}
    for pop, bank, stored, lk in (("3dpw", fb, r3, "3dpw"), ("3dhp", mx, rx, "mpi3dhp_test")):
        gt = geometry_tensor(bank)
        for chain, cands in CANDIDATES.items():
            geometry = chain_pair_geometry(gt, chain).reshape(len(bank), -1)
            readouts = stored if chain == "left_arm" else limb[lk][f"readout_{chain}"]
            e_t, w_t, ok = chain_targets(bank.arrays["target_3d"], bank.arrays["target_valid"], chain)
            out[pop].update({f"{chain}_target_elbow": e_t, f"{chain}_target_wrist": w_t, f"{chain}_target_valid": ok})
            for g, arch in cands.items():
                model = build_joint_arm(readouts.shape[-1])
                model.load_state_dict(torch.load(ckpt[chain][g], map_location="cpu", weights_only=False)["state_dict"])
                model = model.to(device).eval()
                raw = predict_joint(model, candidate_visual(readouts, arch), geometry, np.arange(len(bank)), device)
                e, w, _ = decode_numpy(raw)
                out[pop][f"{chain}_{g}_elbow"], out[pop][f"{chain}_{g}_wrist"] = e, w
    gate = {}
    for pop in ("3dpw", "3dhp"):
        path = args.w85_dir / "predictions" / f"predictions_{pop}.npz"
        if _sha(path) != W85_PREDICTIONS[pop]:
            raise SystemExit(f"STOP: Worklog 85 {pop} predictions changed")
        w85 = np.load(path)
        for g in ("j0", "j1x"):
            gate[f"{pop}_{g}"] = float(max(np.abs(out[pop][f"left_arm_{g}_elbow"] - w85[f"{g}_elbow"]).max(),
                                           np.abs(out[pop][f"left_arm_{g}_wrist"] - w85[f"{g}_wrist"]).max()))
    passed = max(gate.values()) <= 1e-6
    print(json.dumps({"reproduction": gate, "passed": passed}), flush=True)
    if not passed:
        raise SystemExit("STOP: left-arm reference does not reproduce Worklog 85")
    validity = {}
    for pop in out:
        for chain, cands in CANDIDATES.items():
            for g in cands:
                e, w = out[pop][f"{chain}_{g}_elbow"], out[pop][f"{chain}_{g}_wrist"]
                validity[f"{pop}:{chain}:{g}"] = {"finite": bool(np.isfinite(e).all() and np.isfinite(w).all()),
                                                  "length_sum_error": float(np.abs(np.linalg.norm(e, axis=1) + np.linalg.norm(w - e, axis=1) - 1).max())}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for pop in out:
        np.savez(args.out_dir / f"predictions_{pop}.npz", **out[pop])
    report = {"schema": "animcv_w86_limb_predictions_v1", "reproduction": gate, "passed": passed, "validity": validity,
              "predictions_sha256": {p: _sha(args.out_dir / f"predictions_{p}.npz") for p in out}}
    (args.out_dir / "prediction_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in validity.items()}, indent=0)[:2000])


if __name__ == "__main__":
    main()
