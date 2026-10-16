#!/usr/bin/env python3
"""EVALUATION ONLY (Worklog 85): frozen J0 / J1 / R1 (historical) and J1/R1_BEDLAM_EXPANDED on 3DPW and 3DHP.

Identical observations for every candidate: the 3DPW FrameBank + Worklog 68 cache, and the fixed Worklog 83
MPI-INF-3DHP bank + its Qwen cache.  Gates (each STOPs): checkpoint / bank / cache identities; historical J1
and R1 3DPW outputs reproduce Worklog 80 / 82 (R1 fed the stored Worklog 82 anchor, as in Worklog 83);
historical 3DHP outputs reproduce the stored Worklog 83 predictions.  R1 candidates always use the frozen
Worklog 80 J0 anchor computed from runtime inputs.  Writes ``predictions_{3dpw,3dhp}.npz`` and a report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict_framepose_anchored_arm_transfer import PINNED as W83_PINNED  # noqa: E402
from run_external_visual_supervision_transfer import load_cache  # noqa: E402

from framepose.anchored_arm_correction import (  # noqa: E402
    anchor_features, anchor_from_raw, build_correction, candidate_visual as correction_visual, correct_numpy, predict_correction,
)
from framepose.bank import load_bank  # noqa: E402
from framepose.bedlam_supervision import ExternalBank  # noqa: E402
from framepose.joint_arm_geometry import build_joint_arm, candidate_visual, decode_numpy, normalized_arm_targets, predict_joint  # noqa: E402
from framepose.learned_vision_sensor import pair_geometry  # noqa: E402
from framepose.train import geometry_tensor  # noqa: E402

J1, R1 = "J1_GEOMETRY_PLUS_VISION", "R1_VISUAL_CORRECTION"
TOLERANCE = 1e-6
W83_PREDICTIONS_SHA256 = "48a0c845c604616a868abf4b9ee47bb669f13682b522fc097eb3f2857f08972f"
W83_BANK_DIGEST = "7260782e49a037fe9a583de8ef2917ce324686f3d19543d9a7bda95729d21aa9"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(builder, path, channels, device):
    import torch

    m = builder(channels)
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["state_dict"])
    m = m.to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def predict(models, readouts, geometry, device, anchor_raw=None):
    """All candidates on one population from runtime inputs (``anchor_raw`` only for historical reproduction)."""
    pos = np.arange(len(geometry))
    out = {}
    raw0 = predict_joint(models["j0"], candidate_visual(readouts, "J0_GEOMETRY_ONLY"), geometry, pos, device)
    out["j0_elbow"], out["j0_wrist"], _ = decode_numpy(raw0)
    out["anchor_raw"] = raw0 if anchor_raw is None else np.asarray(anchor_raw, np.float64)
    u0, v0, rho0 = anchor_from_raw(out["anchor_raw"])
    anchor = anchor_features(u0, v0, rho0)
    out["anchor_elbow"], out["anchor_wrist"] = anchor[:, 7:10].astype(np.float64), anchor[:, 10:13].astype(np.float64)
    out["anchor_rho"] = rho0
    for g in ("j1", "j1x"):
        raw = predict_joint(models[g], candidate_visual(readouts, J1), geometry, pos, device)
        out[f"{g}_elbow"], out[f"{g}_wrist"], _ = decode_numpy(raw)
    for g in ("r1", "r1x"):
        c = predict_correction(models[g], correction_visual(readouts, R1), geometry, anchor, pos, device)
        e, w, rho, _, _ = correct_numpy(c, u0, v0, rho0)
        out.update({f"{g}_elbow": e, f"{g}_wrist": w, f"{g}_rho": rho, f"{g}_correction": c})
    return out


def connected(pred) -> dict:
    out = {}
    for g in ("j0", "j1", "j1x", "r1", "r1x"):
        e, w = pred[f"{g}_elbow"], pred[f"{g}_wrist"]
        out[g] = {"length_sum_error": float(np.abs(np.linalg.norm(e, axis=1) + np.linalg.norm(w - e, axis=1) - 1).max()),
                  "reach_over_one": float(max(0.0, (np.linalg.norm(w, axis=1) - 1).max())),
                  "finite": bool(np.isfinite(e).all() and np.isfinite(w).all())}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bank", "cache_dir", "w80_joint_dir", "w82_dir", "w83_dir", "expanded_dir", "out_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    import torch

    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise SystemExit("STOP: output directory is not empty; never overwrite experimental artifacts")
    device = torch.device(args.device)
    paths = {"J0_GEOMETRY_ONLY.pt": args.w80_joint_dir / "J0_GEOMETRY_ONLY.pt", "J1_GEOMETRY_PLUS_VISION.pt": args.w80_joint_dir / f"{J1}.pt",
             "R1_VISUAL_CORRECTION.pt": args.w82_dir / f"{R1}.pt", "w80_predictions_3dpw.npz": args.w80_joint_dir / "predictions_3dpw.npz",
             "w82_predictions_3dpw.npz": args.w82_dir / "predictions_3dpw.npz", "w68_readout.npz": args.cache_dir / "readout.npz"}
    identities = {k: _sha(p) for k, p in paths.items()}
    bad = {k: v for k, v in identities.items() if v != W83_PINNED[k]}
    w83_pred = args.w83_dir / "predictions" / "predictions_3dhp.npz"
    identities["w83_predictions_3dhp"] = _sha(w83_pred)
    if bad or identities["w83_predictions_3dhp"] != W83_PREDICTIONS_SHA256:
        raise SystemExit(f"STOP: frozen identity mismatch: {bad} / {identities['w83_predictions_3dhp']}")
    training = json.loads((args.expanded_dir / "training_report.json").read_text())
    for n, h in training["checkpoints_sha256"].items():
        if _sha(args.expanded_dir / f"{n}.pt") != h:
            raise SystemExit(f"STOP: expanded checkpoint {n} changed after training")
    identities.update({f"{n}.pt": h for n, h in training["checkpoints_sha256"].items()})

    bank = load_bank(args.bank)
    r3, ident3 = load_cache(args.cache_dir, bank.content_digest(), [s.sample_id for s in bank.samples], None)
    g3 = pair_geometry(geometry_tensor(bank)).reshape(len(bank), -1)
    ext = ExternalBank.load(args.w83_dir / "bank" / "mpi3dhp_test")
    if ext.content_digest() != W83_BANK_DIGEST:
        raise SystemExit("STOP: the Worklog 83 3DHP bank changed")
    rx, identx = load_cache(args.w83_dir / "qwen", ext.content_digest(), [s.sample_id for s in ext.samples], ident3)
    gx = pair_geometry(geometry_tensor(ext)).reshape(len(ext), -1)
    ch = r3.shape[-1]
    models = {"j0": _load(build_joint_arm, paths["J0_GEOMETRY_ONLY.pt"], ch, device),
              "j1": _load(build_joint_arm, paths["J1_GEOMETRY_PLUS_VISION.pt"], ch, device),
              "j1x": _load(build_joint_arm, args.expanded_dir / "J1_BEDLAM_EXPANDED.pt", ch, device),
              "r1": _load(build_correction, paths["R1_VISUAL_CORRECTION.pt"], ch, device),
              "r1x": _load(build_correction, args.expanded_dir / "R1_BEDLAM_EXPANDED.pt", ch, device)}

    # --- historical reproduction gate ---
    w80, w82, w83 = np.load(paths["w80_predictions_3dpw.npz"]), np.load(paths["w82_predictions_3dpw.npz"]), np.load(w83_pred)
    hist3 = predict(models, r3, g3, device, anchor_raw=w82["anchor_raw"])
    p3 = predict(models, r3, g3, device)
    px = predict(models, rx, gx, device)
    checks = {"j0_3dpw_vs_w80": max(np.abs(p3["j0_elbow"] - w80["J0_GEOMETRY_ONLY_elbow"]).max(), np.abs(p3["j0_wrist"] - w80["J0_GEOMETRY_ONLY_wrist"]).max()),
              "j1_3dpw_vs_w80": max(np.abs(p3["j1_elbow"] - w80[f"{J1}_elbow"]).max(), np.abs(p3["j1_wrist"] - w80[f"{J1}_wrist"]).max()),
              "r1_3dpw_vs_w82_on_stored_anchor": max(np.abs(hist3["r1_elbow"] - w82[f"{R1}_elbow"]).max(), np.abs(hist3["r1_wrist"] - w82[f"{R1}_wrist"]).max()),
              **{f"{g}_3dhp_vs_w83": max(np.abs(px[f"{g}_elbow"] - w83[f"{c}_elbow"]).max(), np.abs(px[f"{g}_wrist"] - w83[f"{c}_wrist"]).max())
                 for g, c in (("j0", "J0_GEOMETRY_ONLY"), ("j1", J1), ("r1", R1))}}
    checks = {k: float(v) for k, v in checks.items()}
    gate = {"max_abs_differences": checks, "tolerance": TOLERANCE, "passed": max(checks.values()) <= TOLERANCE}
    print(json.dumps({"historical_reproduction": gate}, indent=1), flush=True)
    if not gate["passed"]:
        raise SystemExit("STOP: historical reproduction failed")
    for d, b in ((p3, bank), (px, ext)):
        d["target_elbow"], d["target_wrist"], d["target_valid"] = normalized_arm_targets(b.arrays["target_3d"], b.arrays["target_valid"])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(args.out_dir / "predictions_3dpw.npz", **p3)
    np.savez(args.out_dir / "predictions_3dhp.npz", **px)
    report = {"schema": "animcv_w85_expanded_predictions_v1", "identities": identities, "historical_reproduction": gate,
              "framebank_digest": bank.content_digest(), "w83_bank_digest": ext.content_digest(), "w83_cache_digest": identx["digest"],
              "connected_arm": {"3dpw": connected(p3), "3dhp": connected(px)},
              "predictions_sha256": {k: _sha(args.out_dir / f"predictions_{k}.npz") for k in ("3dpw", "3dhp")}}
    (args.out_dir / "prediction_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report["connected_arm"], indent=1))


if __name__ == "__main__":
    main()
