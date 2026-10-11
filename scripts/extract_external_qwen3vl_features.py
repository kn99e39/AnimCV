#!/usr/bin/env python3
"""DIAGNOSTIC (Worklog 73): Worklog 68 Qwen3-VL cache for an external (Worklog 73) bank.

Runs the UNCHANGED Worklog 68 extractor (``extract_qwen3vl_visual_features.main``)
with its bank loader pointed at a Worklog 73 ``ExternalBank``: same pinned
snapshot/fingerprint check, 4-bit NF4 load, 448 px crop contract, processor,
merger feature point, readout and manifest.  ``--parity-3dpw N`` instead builds
an ExternalBank from the first N FrameBank rows (image order) and extracts them
so the readouts can be compared with the existing Worklog 68 cache.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import extract_qwen3vl_visual_features as w68
from framepose.bank import load_bank
from framepose.bedlam_supervision import ExternalBank, ExternalSample, ImageRef


def framebank_subset(path: Path, n: int) -> ExternalBank:
    bank = load_bank(path)
    order = sorted(range(len(bank.samples)), key=lambda p: bank.samples[p].image_reference.relative_path)[:n]
    samples = [ExternalSample(bank.samples[p].sample_id, bank.samples[p].source, bank.samples[p].sequence_id,
                              bank.samples[p].frame_index, bank.samples[p].split, tuple(bank.samples[p].image_size),
                              ImageRef("3dpw_images", bank.samples[p].image_reference.relative_path),
                              {"framebank_position": int(p)}) for p in order]
    return ExternalBank(samples, {k: bank.arrays[k][order] for k in ExternalBank.ARRAYS},
                        {"parity_subset_of": str(path), "positions": [int(p) for p in order]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-bank", type=Path)
    parser.add_argument("--parity-3dpw", type=int, default=0)
    parser.add_argument("--framebank", type=Path)
    args, rest = parser.parse_known_args()
    if args.parity_3dpw:
        external = framebank_subset(args.framebank, args.parity_3dpw)
    else:
        external = ExternalBank.load(args.external_bank)
    w68.load_bank = lambda _path: external
    sys.argv = [sys.argv[0], "--bank", "external", *rest]
    w68.main()
    if args.parity_3dpw:
        out = Path(rest[rest.index("--out-dir") + 1])
        np.save(out / "parity_positions.npy", np.array(external.metadata["positions"]))


if __name__ == "__main__":
    main()
