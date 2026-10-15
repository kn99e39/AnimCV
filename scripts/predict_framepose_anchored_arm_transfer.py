#!/usr/bin/env python3
"""EVALUATION ONLY (Worklog 83): frozen J0/J1 (Worklog 80) and R0/R1 (Worklog 82) on MPI-INF-3DHP test.

Order (each check stops the run):
  1. checkpoint / stored-prediction / Qwen-cache identities (hashes pinned below);
  2. historical reproduction gate: every frozen model is re-run on its ORIGINAL 3DPW FrameBank inputs
     (all 21,817 rows) and must reproduce the stored Worklog 80 / Worklog 82 outputs; the stored corrected
     arms must satisfy the connected-arm contract;
  3. only then: the same frozen models on the Worklog 83 3DHP external bank (detector 2D, detector crop,
     Worklog 68 Qwen readout).  No weight is trained, re-selected or adapted; eval mode throughout.
Writes ``predictions_3dhp.npz``, ``predictions_3dpw_reproduced.npz`` and ``prediction_report.json``.
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

from framepose.anchored_arm_correction import (  # noqa: E402
    anchor_features, anchor_from_raw, build_correction, candidate_visual as correction_visual, correct_numpy,
    predict_correction,
)
from framepose.bank import load_bank  # noqa: E402
from framepose.bedlam_supervision import ExternalBank  # noqa: E402
from framepose.joint_arm_geometry import build_joint_arm, candidate_visual, decode_numpy, normalized_arm_targets, predict_joint  # noqa: E402
from framepose.learned_vision_sensor import pair_geometry  # noqa: E402
from framepose.train import geometry_tensor  # noqa: E402

PINNED = {"J0_GEOMETRY_ONLY.pt": "5332b691a58b4135ce9cb4188dba0b0f4050bf11d56b478a88de03386d5f3ba5",
          "J1_GEOMETRY_PLUS_VISION.pt": "25f0c60f48c51bd474369fab924b3ea70e2d040364091479cd1697a8f8e8ee4b",
          "w80_predictions_3dpw.npz": "eaf3f57c64acd9c3eb086cd84635884aef795e0fd635ea7d3a5776d08aead00f",
          "R0_GEOMETRY_CORRECTION.pt": "2f015d67d17775c917db17dd23da735b51c278dcfeb579f8f4a31bb4f8c9b79b",
          "R1_VISUAL_CORRECTION.pt": "873777734d9f44d18951830b111ff2ce6f9cf3d8912a52741e0125053cdd3fab",
          "w82_predictions_3dpw.npz": "a8bfbaea1cdc79173ea97083e1c1b8d6f6097676a147c2b5fd3c732404e15362",
          "w68_readout.npz": "f5d0eb9c34a7eb0dbae24ace7b0ee0c6a87951dcd8c847e291e44471414ec3b5"}
J = ("J0_GEOMETRY_ONLY", "J1_GEOMETRY_PLUS_VISION")
R = ("R0_GEOMETRY_CORRECTION", "R1_VISUAL_CORRECTION")
# Same build / same image as Worklogs 80 and 82 (animcv-framepose:cuda118): agreement is expected to round-off.
REPRODUCTION_TOLERANCE = 1e-6
CONNECTED_TOLERANCE = 1e-9


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _state_digest(model) -> str:
    d = hashlib.sha256()
    for k, v in model.state_dict().items():
        d.update(k.encode())
        d.update(v.detach().cpu().numpy().tobytes())
    return d.hexdigest()


def _load(builder, path, channels, device):
    import torch

    model = builder(channels)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["state_dict"])
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def run_all(models, readouts, geometry, device):
    """Frozen inference from runtime inputs only: J raw outputs, the J0 anchor and R corrections."""
    out, n = {}, len(geometry)
    pos = np.arange(n)
    for cand in J:
        raw = predict_joint(models[cand], candidate_visual(readouts, cand), geometry, pos, device)
        elbow, wrist, rho = decode_numpy(raw)
        out.update({f"{cand}_raw": raw, f"{cand}_elbow": elbow, f"{cand}_wrist": wrist, f"{cand}_rho": rho})
    u0, v0, rho0 = anchor_from_raw(out["J0_GEOMETRY_ONLY_raw"])
    anchor = anchor_features(u0, v0, rho0)
    out.update({"anchor_u": u0, "anchor_v": v0, "anchor_rho": rho0,
                "anchor_elbow": anchor[:, 7:10].astype(np.float64), "anchor_wrist": anchor[:, 10:13].astype(np.float64)})
    for cand in R:
        c = predict_correction(models[cand], correction_visual(readouts, cand), geometry, anchor, pos, device)
        elbow, wrist, rho, u, v = correct_numpy(c, u0, v0, rho0)
        out.update({f"{cand}_correction": c, f"{cand}_elbow": elbow, f"{cand}_wrist": wrist, f"{cand}_rho": rho,
                    f"{cand}_u": u, f"{cand}_v": v})
    return out


def connected(pred, names) -> dict:
    """Max violation of the unit connected-arm contract per candidate (|E| + |W-E| = 1, |W| <= 1)."""
    out = {}
    for c in names:
        e, w = pred[f"{c}_elbow"], pred[f"{c}_wrist"]
        out[c] = {"length_sum_error": float(np.abs(np.linalg.norm(e, axis=1) + np.linalg.norm(w - e, axis=1) - 1).max()),
                  "reach_over_one": float(max(0.0, (np.linalg.norm(w, axis=1) - 1).max())),
                  "finite": bool(np.isfinite(e).all() and np.isfinite(w).all())}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bank", "cache_dir", "w80_joint_dir", "w82_dir", "external_bank", "external_cache", "out_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    import torch

    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise SystemExit("STOP: output directory is not empty; never overwrite experimental artifacts")
    device = torch.device(args.device)
    paths = {"J0_GEOMETRY_ONLY.pt": args.w80_joint_dir / "J0_GEOMETRY_ONLY.pt",
             "J1_GEOMETRY_PLUS_VISION.pt": args.w80_joint_dir / "J1_GEOMETRY_PLUS_VISION.pt",
             "w80_predictions_3dpw.npz": args.w80_joint_dir / "predictions_3dpw.npz",
             "R0_GEOMETRY_CORRECTION.pt": args.w82_dir / "R0_GEOMETRY_CORRECTION.pt",
             "R1_VISUAL_CORRECTION.pt": args.w82_dir / "R1_VISUAL_CORRECTION.pt",
             "w82_predictions_3dpw.npz": args.w82_dir / "predictions_3dpw.npz",
             "w68_readout.npz": args.cache_dir / "readout.npz"}
    identities = {k: _sha(p) for k, p in paths.items()}
    bad = {k: v for k, v in identities.items() if v != PINNED[k]}
    if bad:
        raise SystemExit(f"STOP: frozen artifact identity mismatch: {bad}")

    # --- 1. historical inputs (3DPW FrameBank, Worklog 68 cache) ---
    bank = load_bank(args.bank)
    r3, ident3 = load_cache(args.cache_dir, bank.content_digest(), [s.sample_id for s in bank.samples], None)
    g3 = pair_geometry(geometry_tensor(bank)).reshape(len(bank), -1)
    models = {c: _load(build_joint_arm, paths[f"{c}.pt"], r3.shape[-1], device) for c in J}
    models.update({c: _load(build_correction, paths[f"{c}.pt"], r3.shape[-1], device) for c in R})
    digests_before = {c: _state_digest(m) for c, m in models.items()}

    # --- 2. historical reproduction gate ---
    rep = run_all(models, r3, g3, device)
    w80, w82 = np.load(paths["w80_predictions_3dpw.npz"]), np.load(paths["w82_predictions_3dpw.npz"])
    checks = {}
    for c in J:
        for k in ("raw", "elbow", "wrist", "rho"):
            checks[f"{c}_{k}_vs_w80"] = float(np.abs(rep[f"{c}_{k}"] - w80[f"{c}_{k}"]).max())
    for k in ("u", "v", "rho", "elbow", "wrist"):
        checks[f"anchor_{k}_vs_w82"] = float(np.abs(rep[f"anchor_{k}"] - w82[f"anchor_{k}"]).max())
    for c in R:
        for k in ("correction", "elbow", "wrist", "rho", "u", "v"):
            checks[f"{c}_{k}_vs_w82"] = float(np.abs(rep[f"{c}_{k}"] - w82[f"{c}_{k}"]).max())
    e_t, w_t, ok_t = normalized_arm_targets(bank.arrays["target_3d"], bank.arrays["target_valid"])
    checks["targets_vs_w82"] = float(max(np.abs(e_t - w82["target_elbow"]).max(), np.abs(w_t - w82["target_wrist"]).max()))
    targets_mask_equal = bool((ok_t == w82["target_valid"]).all())
    stored = {c: {"elbow": w80[f"{c}_elbow"], "wrist": w80[f"{c}_wrist"]} for c in J}
    stored.update({c: {"elbow": w82[f"{c}_elbow"], "wrist": w82[f"{c}_wrist"]} for c in R})
    historical_connected = connected({f"{c}_{k}": v for c, d in stored.items() for k, v in d.items()}, (*J, *R))
    worst = max(checks.values())
    gate = {"max_abs_differences": checks, "tolerance": REPRODUCTION_TOLERANCE, "worst": worst,
            "targets_mask_equal": targets_mask_equal, "stored_connected_arm": historical_connected,
            "passed": worst <= REPRODUCTION_TOLERANCE and targets_mask_equal
            and all(v["length_sum_error"] <= CONNECTED_TOLERANCE and v["reach_over_one"] <= CONNECTED_TOLERANCE and v["finite"]
                    for v in historical_connected.values())}
    print(json.dumps({"historical_reproduction": gate}, indent=1), flush=True)
    if not gate["passed"]:
        raise SystemExit("STOP: historical reproduction failed; no 3DHP prediction is run")

    # --- 3. independent source (3DHP external bank, its own Worklog 68 cache) ---
    ext = ExternalBank.load(args.external_bank)
    rx, identx = load_cache(args.external_cache, ext.content_digest(), [s.sample_id for s in ext.samples], ident3)
    gx = pair_geometry(geometry_tensor(ext)).reshape(len(ext), -1)
    pred = run_all(models, rx, gx, device)
    e_x, w_x, ok_x = normalized_arm_targets(ext.arrays["target_3d"], ext.arrays["target_valid"])
    pred.update({"target_elbow": e_x, "target_wrist": w_x, "target_valid": ok_x})
    digests_after = {c: _state_digest(m) for c, m in models.items()}
    if digests_after != digests_before or any(_sha(paths[f"{c}.pt"]) != PINNED[f"{c}.pt"] for c in (*J, *R)):
        raise SystemExit("STOP: a frozen model changed during evaluation")
    validity = connected(pred, (*J, *R, "anchor"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(args.out_dir / "predictions_3dhp.npz", **pred)
    np.savez(args.out_dir / "predictions_3dpw_reproduced.npz", **rep)
    report = {"schema": "animcv_framepose_anchored_arm_transfer_predictions_v1", "identities": identities,
              "framebank_digest": bank.content_digest(), "w68_cache_digest": ident3["digest"],
              "external_bank_digest": ext.content_digest(), "external_cache_digest": identx["digest"],
              "external_cache_identity": identx, "historical_reproduction": gate,
              "frozen_state_digests": digests_before, "frozen_unchanged": digests_after == digests_before,
              "external_rows": len(ext), "external_target_valid_rows": int(ok_x.sum()),
              "external_connected_arm": validity, "torch": torch.__version__,
              "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
              "predictions_3dhp_sha256": _sha(args.out_dir / "predictions_3dhp.npz")}
    (args.out_dir / "prediction_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ("external_rows", "external_target_valid_rows", "external_connected_arm")}, indent=1))


if __name__ == "__main__":
    main()
