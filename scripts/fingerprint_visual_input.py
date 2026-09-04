#!/usr/bin/env python3
"""Compute a frame bank's visual-input identity (no training, no model).

A frozen visual feature is a pure function of the exact source image content,
the geometry used to build the person crop, the crop contract and the backbone's
preprocessing. This writes that identity so a feature cache can be verified
against it at load time instead of being trusted.

It reads every referenced image once (each distinct file once, even when several
actors share it), so it costs one pass over the imagery and no GPU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from framepose.backbones import resolve_backbone
from framepose.bank import load_bank
from framepose.visual_input import preprocessing_identity, save_identity, visual_input_identity


def main() -> int:
    parser = argparse.ArgumentParser(description="Fingerprint a frame bank's visual input")
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--image-root", action="append", required=True, help="KEY=PATH (repeatable)")
    parser.add_argument("--backbone", required=True,
                        help="Backbone whose preprocessing contract the identity binds")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cpu",
                        help="Only used to read the backbone's declared preprocessing")
    args = parser.parse_args()

    roots = dict(value.split("=", 1) for value in args.image_root)
    bank = load_bank(args.bank)
    spec = resolve_backbone(args.backbone)
    if spec.kind == "none":
        raise ValueError("the geometry-only candidate consumes no visual input")

    # Instantiating the tower is how its declared preprocessing is read; no
    # inference is run and no features are produced here.
    from framepose.backbones import FrozenVisualBackbone

    backbone = FrozenVisualBackbone(spec, device=args.device)
    identity = visual_input_identity(
        bank, image_roots=roots,
        preprocessing=preprocessing_identity(backbone.mean, backbone.std, backbone.input_size,
                                             backbone.prefix_tokens),
        crop_resolution=spec.input_resolution)
    save_identity(identity, args.out)
    print(json.dumps({key: identity[key] for key in
                      ("fingerprint", "image_content_summary", "bank_content_digest",
                       "crop_contract_digest", "crop_resolution", "sample_count")},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
