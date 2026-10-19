#!/usr/bin/env python3
"""EVALUATION (Worklog 91, cross-population): GT-free runtime heading on MPI-INF-3DHP TEST (TS1-TS6).

The Worklog 83 external bank: ``input_2d`` is the re-run OpenPose COCO-18 detector (GT 2D only associates the subject
among detections and never enters the input), ``target_3d`` is pelvis-relative AnimCV camera coordinates.  S0 and H0 were
trained on 3DPW only, so this is an out-of-domain check (studio TS1-TS4 at 0.4 s rows, outdoor TS5-TS6 at ~50 fps).

  A_h0_oracle   O_BILATERAL checkpoint fed 3DHP-GT-derived torso signs (``oracle_sign_states`` of target_3d):
                EVALUATION-ONLY reference, never a runtime candidate.
  B_s0_runtime  S0 checkpoint with neutral signs (all UNKNOWN) + unchanged ``current_root_orientation``.
  C_s0_reliable ``motion.runtime_heading`` (B's yaw, KNOWN rows only).
  F_fixed_180   constant 180 deg.
GT heading: ``observe_yaw`` of the 3DHP GT joints (``target_3d``/``target_valid``), offline evaluation only.
Frozen-model inference only; nothing is trained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_runtime_heading_eval import evaluate  # noqa: E402

from framepose.bedlam_supervision import ExternalBank  # noqa: E402
from framepose.mpi3dhp_test_transfer import FPS  # noqa: E402
from framepose.signs import mask_fields, neutral_sign_states, oracle_sign_states  # noqa: E402
from framepose.train import geometry_tensor, load_checkpoint, predict  # noqa: E402
from motion.animation_semantics_v2_bridge import current_root_orientation  # noqa: E402
from motion.runtime_heading import runtime_heading  # noqa: E402
from pose.framepose_bridge import build_h0_lifted_sequence, sequence_frame_positions  # noqa: E402
from pose.root_orientation_diagnostic import observe_yaw  # noqa: E402

CHECKPOINT_CANDIDATE = {"S0": "S0_neutral_sign", "H0": "O_BILATERAL_oracle_forward_depth_only"}


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def shim(bank, input_2d, input_valid):
    """FrameBank-shaped view for the unchanged ``build_h0_lifted_sequence`` (fps / timestamp per 3DHP sequence)."""
    samples = [SimpleNamespace(sequence_id=s.sequence_id, frame_index=s.frame_index, fps=float(FPS[s.sequence_id.split(":")[1]]),
                               timestamp=s.frame_index / float(FPS[s.sequence_id.split(":")[1]])) for s in bank.samples]
    return SimpleNamespace(samples=samples, arrays={"input_2d": input_2d, "input_valid": input_valid})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("w83_dir", "s0_checkpoint", "h0_checkpoint", "out_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    import torch

    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise SystemExit("STOP: output directory is not empty")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    mx = ExternalBank.load(args.w83_dir / "bank" / "mpi3dhp_test")
    geometry = geometry_tensor(mx)
    positions = np.arange(len(mx))
    preds, ident = {}, {}
    for name, path, signs in (("S0", args.s0_checkpoint, neutral_sign_states(len(mx))),
                              ("H0", args.h0_checkpoint, mask_fields(oracle_sign_states(mx.arrays["target_3d"], mx.arrays["target_valid"]),
                                                                     ["shoulder_forward_depth", "hip_forward_depth"]))):
        model, ck = load_checkpoint(path, device=args.device)
        if ck.get("candidate", {}).get("name") != CHECKPOINT_CANDIDATE[name]:
            raise SystemExit(f"STOP: {name} checkpoint is not {CHECKPOINT_CANDIDATE[name]}")
        preds[name] = predict(model, torch, geometry, None, positions, torch.device(args.device), signs=signs).astype(np.float32)
        np.save(args.out_dir / f"prediction_{name}.npy", preds[name])
        ident[name] = {"checkpoint_sha256": _sha(path), "prediction_sha256": _sha(args.out_dir / f"prediction_{name}.npy"),
                       "sign_source": "neutral" if name == "S0" else "oracle (3DHP GT, evaluation reference only)"}
    runtime_view = shim(mx, mx.arrays["input_2d"], mx.arrays["input_valid"])
    gt_2d = np.concatenate([np.zeros_like(mx.arrays["input_2d"][..., :2]), np.ones_like(mx.arrays["input_2d"][..., 2:])], axis=-1)
    gt_view = shim(mx, gt_2d, mx.arrays["target_valid"])
    rows = {k: [] for k in ("A_h0_oracle", "B_s0_runtime", "C_s0_reliable", "F_fixed_180")}
    seqs = sorted({s.sequence_id for s in mx.samples})
    for sid in seqs:
        pos = sequence_frame_positions(runtime_view, sid)
        gt = observe_yaw(build_h0_lifted_sequence(gt_view, mx.arrays["target_3d"], sid))
        a = current_root_orientation(build_h0_lifted_sequence(runtime_view, preds["H0"], sid))
        lifted_s0 = build_h0_lifted_sequence(runtime_view, preds["S0"], sid)
        b = current_root_orientation(lifted_s0)
        c = runtime_heading(lifted_s0)
        for i, p in enumerate(pos):
            g = gt[i].yaw_radians
            if g is None:
                continue
            s = runtime_view.samples[p]
            conf = float(min(mx.arrays["input_2d"][p][j, 2] for j in (1, 4, 11, 14)))
            base = {"sequence_id": sid, "frame": s.frame_index, "t": s.timestamp, "gt": g, "conf": conf}
            rows["A_h0_oracle"].append({**base, "known": a[i].known, "yaw": a[i].yaw_radians, "reliability": None})
            rows["B_s0_runtime"].append({**base, "known": b[i].known, "yaw": b[i].yaw_radians, "reliability": None})
            rows["C_s0_reliable"].append({**base, "known": c[i].status == "KNOWN", "yaw": c[i].yaw_radians if c[i].status == "KNOWN" else None,
                                          "reliability": c[i].min_torso_confidence})
            rows["F_fixed_180"].append({**base, "known": True, "yaw": math.pi, "reliability": None})
    report = {"schema": "worklog91_runtime_heading_3dhp", "bank_content_digest": mx.content_digest(), "identities": ident, "results": {}}
    groups = {"all": lambda r: True, "studio_TS1_TS4": lambda r: r["sequence_id"].split(":")[1] in ("TS1", "TS2", "TS3", "TS4"),
              "outdoor_TS5_TS6": lambda r: r["sequence_id"].split(":")[1] in ("TS5", "TS6")}
    for gname, keep in groups.items():
        report["results"][gname] = {k: evaluate([r for r in v if keep(r)]) for k, v in rows.items()}
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps({g: {k: {"coverage": round(v["coverage"], 4), **{m: round(x, 4) for m, x in v.get("error", {}).items()}}
                          for k, v in x.items()} for g, x in report["results"].items()}, indent=1))


if __name__ == "__main__":
    main()
