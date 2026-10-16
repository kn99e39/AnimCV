#!/usr/bin/env python3
"""DIAGNOSTIC (Worklog 85): J1_BEDLAM_EXPANDED and R1_BEDLAM_EXPANDED under the unchanged Worklog 80 / 82 contracts.

The only intervention is the BEDLAM TRAIN population: the 21,232 historical Worklog 73 TRAIN rows plus the
Worklog 85 expansion rows.  Everything else is the historical code path, called unchanged:
  * J1: ``train_framepose_joint_arm_geometry.train_joint`` (Worklog 80 representation, loss, optimizer,
    9,000 steps, 50/50 source-balanced draws, selection = lowest 3DPW validation objective);
  * R1: ``train_framepose_anchored_arm_correction.train_correction`` on the frozen Worklog 80 J0 anchor
    (zero visual input, never trained), zero-initialized Rodrigues correction, the Worklog 82 coupled loss.
Order (each check stops the run):
  1. identities: frozen checkpoints, Qwen caches bound to bank digests, split/subject isolation (no BEDLAM
     validation subject, sequence or image in TRAIN; 3DPW splits unchanged);
  2. TRAINING-PIPELINE REPRODUCTION GATE: this wrapper retrains J1 and R1 on the HISTORICAL population with
     the historical row order and must reproduce the Worklog 80 J1 / Worklog 82 R1 checkpoints (same selected
     epoch; 3DPW outputs within ``REPRODUCTION_TOLERANCE``; bit-exactness reported);
  3. the frozen-J0 anchor audit of Worklog 82 (same criterion) on the expanded population;
  4. J1_BEDLAM_EXPANDED and R1_BEDLAM_EXPANDED, trained once each.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_external_visual_supervision_transfer import load_cache  # noqa: E402
from train_framepose_anchored_arm_correction import (  # noqa: E402
    AUDIT_CRITERION, J0_SHA256, _objective, train_correction,
)
from train_framepose_joint_arm_geometry import train_joint  # noqa: E402

from framepose.anchored_arm_correction import anchor_features, anchor_from_raw, correct_numpy, predict_correction  # noqa: E402
from framepose.bank import load_bank  # noqa: E402
from framepose.bedlam_supervision import ExternalBank, is_validation_subject  # noqa: E402
from framepose.joint_arm_geometry import build_joint_arm, decode_numpy, normalized_arm_targets, predict_joint  # noqa: E402
from framepose.learned_vision_sensor import pair_geometry  # noqa: E402
from framepose.train import geometry_tensor  # noqa: E402

PINNED = {"J0_GEOMETRY_ONLY.pt": J0_SHA256,
          "J1_GEOMETRY_PLUS_VISION.pt": "25f0c60f48c51bd474369fab924b3ea70e2d040364091479cd1697a8f8e8ee4b",
          "R1_VISUAL_CORRECTION.pt": "873777734d9f44d18951830b111ff2ce6f9cf3d8912a52741e0125053cdd3fab",
          "w80_predictions_3dpw.npz": "eaf3f57c64acd9c3eb086cd84635884aef795e0fd635ea7d3a5776d08aead00f",
          "w82_predictions_3dpw.npz": "a8bfbaea1cdc79173ea97083e1c1b8d6f6097676a147c2b5fd3c732404e15362",
          "historical_bedlam_train_digest": "56f416639516632f444586c5e26e6610c08b5c1e1db3b7ca8afb6e0dd71fa419",
          "historical_bedlam_validation_digest": "9556799f9040f5fb799318cf7f8cba2dbb94bd3d19f02bbee676881fcaedb632"}
J1, R1 = "J1_GEOMETRY_PLUS_VISION", "R1_VISUAL_CORRECTION"
# Declared before training: same selected epoch and 3DPW elbow/wrist outputs within 1e-4 (0.01 %L).
REPRODUCTION_TOLERANCE = 1e-4


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(state) -> str:
    return hashlib.sha256(b"".join(v.detach().cpu().numpy().tobytes() for v in state.values())).hexdigest()


def assemble(bank, r3, externals, val, rval):
    """Row-stacked inputs in the historical order [3DPW, BEDLAM TRAIN parts..., BEDLAM validation]."""
    banks = [bank, *[b for b, _ in externals], val]
    readouts = np.concatenate([r3, *[r for _, r in externals], rval])
    geometry = np.concatenate([pair_geometry(geometry_tensor(b)) for b in banks]).reshape(len(readouts), -1)
    elbow_t, wrist_t, mask = normalized_arm_targets(np.concatenate([b.arrays["target_3d"] for b in banks]),
                                                    np.concatenate([b.arrays["target_valid"] for b in banks]))
    n3 = len(bank)
    ntr = sum(len(b) for b, _ in externals)
    train3, val3 = bank.indices("train"), bank.indices("validation")
    ext_train = n3 + np.arange(ntr)
    ext_val = n3 + ntr + np.arange(len(val))
    return {"readouts": readouts, "geometry": geometry, "elbow_t": elbow_t, "wrist_t": wrist_t, "mask": mask, "n3": n3,
            "primary": train3[mask[train3]], "external": ext_train[mask[ext_train]], "val": val3[mask[val3]],
            "ext_val": ext_val[mask[ext_val]], "ext_train_rows": int(ntr)}


def frozen_j0(path: Path, channels: int, device):
    import torch

    j0 = build_joint_arm(channels)
    j0.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["state_dict"])
    j0 = j0.to(device).eval()
    for p in j0.parameters():
        p.requires_grad_(False)
    return j0


def anchor_for(j0, d, device):
    raw = predict_joint(j0, np.zeros_like(d["readouts"], dtype=np.float16), d["geometry"], np.arange(len(d["readouts"])), device)
    u0, v0, rho0 = anchor_from_raw(raw)
    return raw, (u0, v0, rho0), anchor_features(u0, v0, rho0)


def train_pair(d, j0, device):
    """J1 and R1 on one population, exactly the historical calls; returns models, infos, 3DPW outputs."""
    import torch

    model_j, info_j, vis_j = train_joint(J1, d["readouts"], d["geometry"], d["elbow_t"], d["wrist_t"], d["mask"],
                                         d["primary"], d["external"], d["val"], d["ext_val"], device)
    raw = predict_joint(model_j, vis_j, d["geometry"], np.arange(d["n3"]), device)
    e_j, w_j, _ = decode_numpy(raw)
    del vis_j
    torch.cuda.empty_cache()
    raw0, (u0, v0, rho0), anchor = anchor_for(j0, d, device)
    model_r, info_r, vis_r = train_correction(R1, d["readouts"], d["geometry"], anchor, d["elbow_t"], d["wrist_t"], d["mask"],
                                              d["primary"], d["external"], d["val"], d["ext_val"], device)
    c = predict_correction(model_r, vis_r, d["geometry"], anchor, np.arange(d["n3"]), device)
    e_r, w_r, _, _, _ = correct_numpy(c, u0[:d["n3"]], v0[:d["n3"]], rho0[:d["n3"]])
    del vis_r
    torch.cuda.empty_cache()
    return {"j1": (model_j, info_j, e_j, w_j, raw), "r1": (model_r, info_r, e_r, w_r, c), "anchor_raw": raw0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bank", "cache_dir", "hist_train", "hist_train_cache", "external_val", "external_val_cache",
                 "w80_joint_dir", "w82_dir", "out_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    for name in ("new_train", "new_train_cache", "manifest"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path)
    parser.add_argument("--reproduce-only", action="store_true", help="run only the training-pipeline reproduction gate")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    import torch

    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise SystemExit("STOP: output directory is not empty; never overwrite experimental artifacts")
    device = torch.device(args.device)
    paths = {"J0_GEOMETRY_ONLY.pt": args.w80_joint_dir / "J0_GEOMETRY_ONLY.pt", "J1_GEOMETRY_PLUS_VISION.pt": args.w80_joint_dir / f"{J1}.pt",
             "R1_VISUAL_CORRECTION.pt": args.w82_dir / f"{R1}.pt", "w80_predictions_3dpw.npz": args.w80_joint_dir / "predictions_3dpw.npz",
             "w82_predictions_3dpw.npz": args.w82_dir / "predictions_3dpw.npz"}
    identities = {k: _sha(p) for k, p in paths.items()}
    bank = load_bank(args.bank)
    hist, val = ExternalBank.load(args.hist_train), ExternalBank.load(args.external_val)
    identities.update({"historical_bedlam_train_digest": hist.content_digest(), "historical_bedlam_validation_digest": val.content_digest(),
                       "framebank_digest": bank.content_digest()})
    bad = {k: identities[k] for k in PINNED if identities[k] != PINNED[k]}
    if bad:
        raise SystemExit(f"STOP: frozen identity mismatch: {bad}")
    r3, ident3 = load_cache(args.cache_dir, bank.content_digest(), [s.sample_id for s in bank.samples], None)
    rh, _ = load_cache(args.hist_train_cache, hist.content_digest(), [s.sample_id for s in hist.samples], ident3)
    rv, _ = load_cache(args.external_val_cache, val.content_digest(), [s.sample_id for s in val.samples], ident3)
    if args.reproduce_only:
        new, rn, identn, manifest = ExternalBank([], {k: np.zeros(sh, dt) for k, sh, dt in (
            ("input_2d", (0, 17, 3), np.float32), ("input_valid", (0, 17), bool), ("target_3d", (0, 17, 3), np.float32),
            ("target_valid", (0, 17), bool))}), None, None, None
    else:
        new = ExternalBank.load(args.new_train)
        identities["new_bedlam_train_digest"] = new.content_digest()
        manifest = json.loads(args.manifest.read_text())
        if new.metadata.get("expansion_manifest_sha256") != manifest["manifest_sha256"]:
            raise SystemExit("STOP: expansion bank was not built from this manifest")
        rn, identn = load_cache(args.new_train_cache, new.content_digest(), [s.sample_id for s in new.samples], ident3)

    # --- 1. split / identity isolation ---
    tr_ids = [s.sample_id for s in hist.samples] + [s.sample_id for s in new.samples]
    if len(set(tr_ids)) != len(tr_ids) or set(tr_ids) & {s.sample_id for s in val.samples}:
        raise SystemExit("STOP: duplicate or train/validation sample identities")
    tr_subj = {s.detail["subject"] for b in (hist, new) for s in b.samples}
    va_subj = {s.detail["subject"] for s in val.samples}
    tr_seq = {s.sequence_id for b in (hist, new) for s in b.samples}
    tr_img = {s.image_reference.relative_path for b in (hist, new) for s in b.samples}
    isolation = {"subject_overlap": len(tr_subj & va_subj), "sequence_overlap": len(tr_seq & {s.sequence_id for s in val.samples}),
                 "image_overlap": len(tr_img & {s.image_reference.relative_path for s in val.samples}),
                 "validation_subjects_in_train": sum(is_validation_subject(s.detail["person"]) for s in new.samples),
                 "new_rows_from_validation_scenes": sum(s.sequence_id.split(":")[1] in {v.sequence_id.split(":")[1] for v in val.samples}
                                                       for s in new.samples)}
    if any(isolation.values()):
        raise SystemExit(f"STOP: BEDLAM train/validation leakage: {isolation}")
    if {s.sequence_id for s in bank.samples if s.split == "train"} & {s.sequence_id for s in bank.samples if s.split != "train"}:
        raise SystemExit("STOP: 3DPW TRAIN/held-out sequence overlap")
    j0 = frozen_j0(paths["J0_GEOMETRY_ONLY.pt"], r3.shape[-1], device)
    j0_before = _digest(j0.state_dict())

    # --- 2. training-pipeline reproduction gate (historical population, historical row order) ---
    hd = assemble(bank, r3, [(hist, rh)], val, rv)
    w80, w82 = np.load(paths["w80_predictions_3dpw.npz"]), np.load(paths["w82_predictions_3dpw.npz"])
    rep = train_pair(hd, j0, device)
    hist_ckpt = {"j1": torch.load(paths["J1_GEOMETRY_PLUS_VISION.pt"], map_location="cpu", weights_only=False),
                 "r1": torch.load(paths["R1_VISUAL_CORRECTION.pt"], map_location="cpu", weights_only=False)}
    gate = {"anchor_raw_vs_w82_3dpw": float(np.abs(rep["anchor_raw"][:hd["n3"]] - w82["anchor_raw"]).max())}
    for g, (model, info, e, w, _) in ((k, rep[k]) for k in ("j1", "r1")):
        ref_e = w80[f"{J1}_elbow"] if g == "j1" else w82[f"{R1}_elbow"]
        ref_w = w80[f"{J1}_wrist"] if g == "j1" else w82[f"{R1}_wrist"]
        state = model.state_dict()
        hstate = hist_ckpt[g]["state_dict"]
        gate[g] = {"selected_epoch": info["selection"]["epoch"], "historical_selected_epoch": hist_ckpt[g]["selection"]["epoch"],
                   "state_bit_exact": all(torch.equal(state[k].cpu(), hstate[k].cpu()) for k in hstate),
                   "state_max_abs": float(max((state[k].cpu().float() - hstate[k].cpu().float()).abs().max() for k in hstate)),
                   "elbow_max_abs_3dpw": float(np.abs(e - ref_e).max()), "wrist_max_abs_3dpw": float(np.abs(w - ref_w).max())}
        gate[g]["passed"] = (gate[g]["selected_epoch"] == gate[g]["historical_selected_epoch"]
                             and max(gate[g]["elbow_max_abs_3dpw"], gate[g]["wrist_max_abs_3dpw"]) <= REPRODUCTION_TOLERANCE)
    gate["passed"] = gate["j1"]["passed"] and gate["r1"]["passed"]
    print(json.dumps({"training_reproduction_gate": gate}, indent=1), flush=True)
    if args.reproduce_only:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "reproduction_gate.json").write_text(json.dumps({"identities": identities, "isolation": isolation,
                                                                          "training_reproduction_gate": gate}, indent=1))
    if not gate["passed"]:
        raise SystemExit("STOP: the training pipeline does not reproduce the historical J1/R1 checkpoints")
    if args.reproduce_only:
        return
    del rep, hd
    torch.cuda.empty_cache()

    # --- 3. expanded population: anchor audit (Worklog 82 criterion) ---
    xd = assemble(bank, r3, [(hist, rh), (new, rn)], val, rv)
    raw0, _, anchor = anchor_for(j0, xd, device)
    e0, w0 = anchor[:, 7:10].astype(np.float64), anchor[:, 10:13].astype(np.float64)
    n3, nh = xd["n3"], len(hist)
    scopes = {"3dpw_train": hd_idx(bank, "train"), "3dpw_validation": hd_idx(bank, "validation"),
              "bedlam_train_historical": n3 + np.arange(nh), "bedlam_train_new": n3 + nh + np.arange(len(new)),
              "bedlam_validation": n3 + xd["ext_train_rows"] + np.arange(len(val))}
    audit = {"criterion": AUDIT_CRITERION, "scopes": {k: _objective(e0, w0, xd["elbow_t"], xd["wrist_t"], xd["mask"], v) for k, v in scopes.items()}}
    s = audit["scopes"]
    bedlam_train = _objective(e0, w0, xd["elbow_t"], xd["wrist_t"], xd["mask"], n3 + np.arange(xd["ext_train_rows"]))
    audit["ratios_train_over_validation"] = {"3dpw": s["3dpw_train"]["objective"] / s["3dpw_validation"]["objective"],
                                             "bedlam_expanded": bedlam_train["objective"] / s["bedlam_validation"]["objective"]}
    audit["severe_mismatch"] = any(v < 0.5 for v in audit["ratios_train_over_validation"].values())
    print(json.dumps({"anchor_audit": audit}, default=str), flush=True)
    if audit["severe_mismatch"]:
        raise SystemExit(f"STOP: severe frozen-anchor stacking mismatch: {audit['ratios_train_over_validation']}")

    # --- 4. the two expanded models ---
    res = train_pair(xd, j0, device)
    if _digest(j0.state_dict()) != j0_before or _sha(paths["J0_GEOMETRY_ONLY.pt"]) != J0_SHA256:
        raise SystemExit("STOP: frozen J0 changed")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    names = {"j1": "J1_BEDLAM_EXPANDED", "r1": "R1_BEDLAM_EXPANDED"}
    hd_external = assemble_counts(hist, rh)
    exposure = {"historical": {"bedlam_eligible_rows": hd_external},
                "expanded": {"bedlam_eligible_rows": int(len(xd["external"])), "bedlam_new_eligible_rows": int((xd["external"] >= n3 + nh).sum())}}
    for g in ("j1", "r1"):
        model, info, *_ = res[g]
        torch.save({"candidate": names[g], "architecture": J1 if g == "j1" else R1, "selection": info["selection"],
                    "anchor_checkpoint_sha256": J0_SHA256 if g == "r1" else None,
                    "training_population": {"historical_bedlam_train_digest": hist.content_digest(),
                                            "new_bedlam_train_digest": new.content_digest(), "manifest_sha256": manifest["manifest_sha256"]},
                    "state_dict": model.state_dict()}, args.out_dir / f"{names[g]}.pt")
        draws = info["source_draws"]
        exposure[names[g]] = {"source_draws": draws, "bedlam_draws_per_eligible_row": draws.get("bedlam", 0) / len(xd["external"]),
                              "historical_bedlam_draws_per_eligible_row": draws.get("bedlam", 0) / hd_external}
    report = {"schema": "animcv_w85_bedlam_expanded_training_v1", "identities": identities, "new_cache_identity": identn,
              "isolation": isolation, "training_reproduction_gate": gate, "anchor_audit": audit, "exposure": exposure,
              "candidates": {names[g]: {k: v for k, v in res[g][1].items()} for g in ("j1", "r1")},
              "frozen_j0_unchanged": True, "checkpoints_sha256": {n: _sha(args.out_dir / f"{n}.pt") for n in names.values()}}
    (args.out_dir / "training_report.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps({"checkpoints": report["checkpoints_sha256"], "exposure": exposure,
                      **{n: report["candidates"][n]["selection"] for n in names.values()}}, indent=1, default=str))


def hd_idx(bank, split):
    return bank.indices(split)


def assemble_counts(hist, rh):
    """Number of GT-valid historical BEDLAM TRAIN rows (the historical sampler pool)."""
    _, _, ok = normalized_arm_targets(hist.arrays["target_3d"], hist.arrays["target_valid"])
    return int(ok.sum())


if __name__ == "__main__":
    main()
