#!/usr/bin/env python3
"""DIAGNOSTIC (Worklog 87): historical-parity gate and frozen predictions for the near-plane attribution.

Frozen candidates only (no training, no checkpoint reselection): left arm J1X (Worklog 85) and G0X, right arm
G0 / G1, right leg G0 / G1 (Worklog 86).  The Worklog 86 prediction path is re-run on 3DPW (all FrameBank rows)
and MPI-INF-3DHP TEST and must reproduce the stored Worklog 86 predictions exactly; the same frozen models are
additionally run on BEDLAM VALIDATION (a selection-free validation source for the diagnosis).
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

PINNED = {"J1_BEDLAM_EXPANDED.pt": "82fd6bcb2cf49aac8c5aebb124335cc3fd890ec701160bab714f54fa6c7839f5",
          "left_arm_g0.pt": "a7aecc6e906e14b438f238ba0a8056d27aad2e56b86e26aa26c1df2e0313462c",
          "right_arm_g0.pt": "ae1ccea9d5d8d049fe79adecec014c2a56b8c76722337d64c7059ef6807c9a8e",
          "right_arm_g1.pt": "e7f220d6cbc8e35d7bebd9195b65184067b3bfb4b0d7b700278eaf46cd87a986",
          "right_leg_g0.pt": "5186a471b66e1758e09aee5ee08db7e5350db1a2a2f976886c5ad90675152499",
          "right_leg_g1.pt": "25b660c4b055f5c7f5249d13c31ae315e0257976e3af911961d11c98871c1e26",
          "w86_predictions_3dpw.npz": "a13f06bc8a77a2ff73daebb6448baa4f012c3eb3251c9d775fcd0a4eb0fc1ffd",
          "w86_predictions_3dhp.npz": "ef8ca102301c4aa2cde4e697419964d3e391128878d42e70050c1038de31b2fc"}
# chain -> {name: (checkpoint key, architecture)}; g0 = 2D only, g1 = 2D + frozen Qwen (J1X for the left arm)
CANDIDATES = {"left_arm": {"g0": ("left_arm_g0.pt", "J0_GEOMETRY_ONLY"), "g1": ("J1_BEDLAM_EXPANDED.pt", "J1_GEOMETRY_PLUS_VISION")},
              "right_arm": {"g0": ("right_arm_g0.pt", "J0_GEOMETRY_ONLY"), "g1": ("right_arm_g1.pt", "J1_GEOMETRY_PLUS_VISION")},
              "right_leg": {"g0": ("right_leg_g0.pt", "J0_GEOMETRY_ONLY"), "g1": ("right_leg_g1.pt", "J1_GEOMETRY_PLUS_VISION")}}
W86_NAMES = {"left_arm": {"g0": "g0x", "g1": "j1x"}, "right_arm": {"g0": "g0", "g1": "g1"}, "right_leg": {"g0": "g0", "g1": "g1"}}


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bank", "cache_dir", "w83_dir", "bedlam_val", "bedlam_val_cache", "w85_training", "w86_dir", "out_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    import torch

    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise SystemExit("STOP: output directory is not empty; never overwrite experimental artifacts")
    device = torch.device(args.device)
    paths = {"J1_BEDLAM_EXPANDED.pt": args.w85_training / "J1_BEDLAM_EXPANDED.pt",
             **{k: args.w86_dir / "training" / k for k in PINNED if k.endswith(".pt") and k != "J1_BEDLAM_EXPANDED.pt"},
             "w86_predictions_3dpw.npz": args.w86_dir / "predictions" / "predictions_3dpw.npz",
             "w86_predictions_3dhp.npz": args.w86_dir / "predictions" / "predictions_3dhp.npz"}
    identities = {k: _sha(p) for k, p in paths.items()}
    bad = {k: v for k, v in identities.items() if v != PINNED[k]}
    if bad:
        raise SystemExit(f"STOP: frozen artifact identity mismatch: {bad}")
    fb, mx, bv = load_bank(args.bank), ExternalBank.load(args.w83_dir / "bank" / "mpi3dhp_test"), ExternalBank.load(args.bedlam_val)
    r3, ident3 = load_cache(args.cache_dir, fb.content_digest(), [s.sample_id for s in fb.samples], None)
    rx, _ = load_cache(args.w83_dir / "qwen", mx.content_digest(), [s.sample_id for s in mx.samples], ident3)
    rv, _ = load_cache(args.bedlam_val_cache, bv.content_digest(), [s.sample_id for s in bv.samples], ident3)
    limb_rep = json.loads((args.w86_dir / "readouts" / "readout_report.json").read_text())
    limb = {}
    for k, b in (("3dpw", fb), ("mpi3dhp_test", mx), ("bedlam_validation", bv)):
        info = limb_rep["banks"][k]
        if info["bank_content_digest"] != b.content_digest() or info["npz_sha256"] != _sha(args.w86_dir / "readouts" / f"{k}.npz"):
            raise SystemExit(f"STOP: limb readout identity mismatch for {k}")
        limb[k] = np.load(args.w86_dir / "readouts" / f"{k}.npz")
    models = {}
    for chain, cands in CANDIDATES.items():
        for g, (ck, _) in cands.items():
            m = build_joint_arm(r3.shape[-1])
            m.load_state_dict(torch.load(paths[ck], map_location="cpu", weights_only=False)["state_dict"])
            models[(chain, g)] = m.to(device).eval()
    out = {}
    for pop, bank, stored, lk in (("3dpw", fb, r3, "3dpw"), ("3dhp", mx, rx, "mpi3dhp_test"), ("bedlam_validation", bv, rv, "bedlam_validation")):
        gt = geometry_tensor(bank)
        out[pop] = {}
        for chain, cands in CANDIDATES.items():
            geometry = chain_pair_geometry(gt, chain).reshape(len(bank), -1)
            readouts = stored if chain == "left_arm" else limb[lk][f"readout_{chain}"]
            e_t, w_t, ok = chain_targets(bank.arrays["target_3d"], bank.arrays["target_valid"], chain)
            out[pop].update({f"{chain}_target_elbow": e_t, f"{chain}_target_wrist": w_t, f"{chain}_target_valid": ok})
            for g, (_, arch) in cands.items():
                raw = predict_joint(models[(chain, g)], candidate_visual(readouts, arch), geometry, np.arange(len(bank)), device)
                out[pop][f"{chain}_{g}_elbow"], out[pop][f"{chain}_{g}_wrist"], _ = decode_numpy(raw)
    parity = {}
    for pop in ("3dpw", "3dhp"):
        w86 = np.load(paths[f"w86_predictions_{pop}.npz"])
        for chain, names in W86_NAMES.items():
            for g, n in names.items():
                parity[f"{pop}:{chain}:{g}"] = float(max(np.abs(out[pop][f"{chain}_{g}_elbow"] - w86[f"{chain}_{n}_elbow"]).max(),
                                                         np.abs(out[pop][f"{chain}_{g}_wrist"] - w86[f"{chain}_{n}_wrist"]).max()))
            for k in ("target_elbow", "target_wrist", "target_valid"):
                parity[f"{pop}:{chain}:{k}"] = float(np.abs(out[pop][f"{chain}_{k}"].astype(float) - w86[f"{chain}_{k}"].astype(float)).max())
    passed = max(parity.values()) == 0.0
    print(json.dumps({"parity": parity, "passed": passed}), flush=True)
    if not passed:
        raise SystemExit("STOP: frozen predictions do not reproduce Worklog 86")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for pop, d in out.items():
        np.savez(args.out_dir / f"predictions_{pop}.npz", **d)
    report = {"schema": "animcv_w87_frozen_predictions_v1", "identities": identities, "parity": parity, "passed": passed,
              "bank_digests": {"3dpw": fb.content_digest(), "3dhp": mx.content_digest(), "bedlam_validation": bv.content_digest()},
              "predictions_sha256": {p: _sha(args.out_dir / f"predictions_{p}.npz") for p in out}}
    (args.out_dir / "prediction_report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
