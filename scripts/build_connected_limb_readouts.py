#!/usr/bin/env python3
"""DIAGNOSTIC (Worklog 86): frozen-Qwen readouts for new chains, read from the cached Worklog 68 token grids.

No Qwen forward pass: every bank already stores its frozen merger token grid (fp16, (N, 196, 4096)) and the
448 px crop box of every row.  For each requested chain the readout is the Worklog 68 bilinear sample at the
DETECTOR's proximal / middle / distal positions (invalid detector joint -> zero vector) plus the bank's stored
global mean.  Provenance check: the left arm is recomputed the same way and compared with the stored
(float32-token) left-arm readouts; the deviation is reported (fp16-token rounding only).
Writes ``<out>/<bank>.npz`` (``readout_<chain>`` fp16 (N, 4, 4096), ``valid_<chain>`` (N, 3)) and a report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from framepose.bank import load_bank
from framepose.bedlam_supervision import ExternalBank
from framepose.connected_limb_transfer import chain_indices, chain_readout


def build(bank, cache_dir: Path, chains):
    grid = np.load(cache_dir / "token_grid_fp16.npy", mmap_mode="r")
    stored = np.load(cache_dir / "readout.npz")
    readout, boxes = stored["readout"], stored["crop_box_xy_side"]
    n = len(bank)
    if grid.shape[0] != n or readout.shape[0] != n:
        raise SystemExit(f"STOP: grid/readout rows differ from the bank ({grid.shape[0]}, {readout.shape[0]}, {n})")
    out = {f"readout_{c}": np.zeros((n, 4, readout.shape[-1]), np.float16) for c in chains}
    out.update({f"valid_{c}": np.zeros((n, 3), bool) for c in chains})
    dev = []
    for p in range(n):
        s = bank.samples[p]
        rows = lambda idx, p=p: np.asarray(grid[p, idx])  # noqa: E731
        inp, vis = bank.arrays["input_2d"][p], bank.arrays["input_valid"][p]
        for c in chains:
            r = chain_readout(rows, inp, vis, tuple(s.image_size), boxes[p], c, readout[p, 3].astype(np.float32))
            out[f"readout_{c}"][p] = r.astype(np.float16)
            out[f"valid_{c}"][p] = vis[chain_indices(c)]
        if "left_arm" in chains:
            d = np.abs(out["readout_left_arm"][p, :3].astype(np.float32) - readout[p, :3].astype(np.float32))
            dev.append((float(d.max()), float(d.mean()), float(np.abs(readout[p, :3].astype(np.float32)).mean())))
    dev = np.array(dev) if dev else np.zeros((0, 3))
    return out, {"rows": n, "left_arm_vs_stored_max_abs": float(dev[:, 0].max()) if len(dev) else None,
                 "left_arm_vs_stored_mean_abs": float(dev[:, 1].mean()) if len(dev) else None,
                 "stored_mean_abs_magnitude": float(dev[:, 2].mean()) if len(dev) else None,
                 "global_readout_equals_stored": bool(all(np.array_equal(out[f"readout_{c}"][:, 3], readout[:, 3]) for c in chains)),
                 "source_readout_npz_sha256": json.loads((cache_dir / "manifest.json").read_text())["readout_npz_sha256"],
                 "grid_shape": list(grid.shape)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", action="append", nargs=3, metavar=("NAME", "BANK", "CACHE"), required=True,
                        help="name, bank path (FrameBank .json or ExternalBank stem), Qwen cache dir")
    parser.add_argument("--chains", nargs="+", default=["left_arm", "right_arm", "right_leg"])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit("STOP: output directory is not empty; never overwrite experimental artifacts")
    args.out.mkdir(parents=True, exist_ok=True)
    report, started = {"schema": "animcv_w86_limb_readouts_v1", "chains": args.chains, "banks": {}}, time.time()
    for name, bank_path, cache in args.bank:
        bank = load_bank(Path(bank_path)) if bank_path.endswith(".json") else ExternalBank.load(Path(bank_path))
        manifest = json.loads((Path(cache) / "manifest.json").read_text())
        if manifest["identity"]["bank_content_digest"] != bank.content_digest():
            raise SystemExit(f"STOP: {name}: cache was not built for this bank")
        arrays, info = build(bank, Path(cache), args.chains)
        np.savez(args.out / f"{name}.npz", **arrays)
        info.update({"bank_content_digest": bank.content_digest(),
                     "sample_ids_sha256": hashlib.sha256("\n".join(s.sample_id for s in bank.samples).encode()).hexdigest(),
                     "npz_sha256": hashlib.sha256((args.out / f"{name}.npz").read_bytes()).hexdigest(),
                     "seconds": round(time.time() - started)})
        report["banks"][name] = info
        print(json.dumps({name: info}), flush=True)
    (args.out / "readout_report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
