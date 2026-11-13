#!/usr/bin/env python3
"""W141 (EXPERIMENTAL) feature-exclusion wrapper around the FROZEN W140 estimator (scripts/w140_colmap_estimate.py).

Runs ONLY in the scratch environment C:/Users/dna10/animcv-w140/env/pc311 (pycolmap 4.2.1, CPU).

    pc311/Scripts/python.exe -I scripts/w141_colmap_masked_estimate.py --input <in_dir> --out <out_dir> --keep-masks <mask_dir>

The W140 file is NOT modified and its code path is executed unchanged (same SIFT / matching / mapper configuration,
same leak guard on <in_dir>, same output schema).  The single difference is that pycolmap.ImageReaderOptions().mask_path
is set to <mask_dir>, so COLMAP's own feature extractor drops every keypoint located on a zero-valued mask pixel
(COLMAP convention: mask file <mask_dir>/<image_name>.png, 0 = exclude, non-zero = keep).  The RGB images themselves are
never painted: no artificial mask boundaries can become SIFT features.

The unmasked Condition A runs scripts/w140_colmap_estimate.py directly (the original path stays reproducible).
Mask-directory contract (refused otherwise): exactly one <image>.png per frame in frames.json, single-channel uint8 with the
image's height x width, values in {0, 255}, nothing else in the directory.
Writes <out>/mask_binding.json (mask dir, per-file SHA-256, keep fraction per frame) next to the W140 outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
W140_EST = HERE / "w140_colmap_estimate.py"


def check_masks(in_dir, mask_dir):
    import cv2
    import numpy as np
    frames = json.load(open(Path(in_dir) / "frames.json"))
    cam = json.load(open(Path(in_dir) / "camera.json"))
    want = {f["image"] + ".png" for f in frames}
    have = {p.name for p in Path(mask_dir).iterdir()}
    if have != want:
        raise SystemExit(f"STOP MASK_CONTRACT: mask files != frames (missing {sorted(want - have)[:3]}, extra {sorted(have - want)[:3]})")
    rec = {}
    for f in frames:
        p = Path(mask_dir) / (f["image"] + ".png")
        m = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if m is None or m.ndim != 2 or m.dtype != np.uint8 or m.shape != (cam["height"], cam["width"]):
            raise SystemExit(f"STOP MASK_CONTRACT: {p.name} not a {cam['height']}x{cam['width']} uint8 single-channel image")
        u = set(np.unique(m).tolist())
        if not u <= {0, 255}:
            raise SystemExit(f"STOP MASK_CONTRACT: {p.name} values {sorted(u)[:6]}")
        rec[f["image"]] = {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "keep_fraction": float((m > 0).mean())}
    return rec


def extract_check(in_dir, mask_dir, out, n):
    """PRE-DECLARATION contract check (feature extraction only; no matching, mapping or pose): extract SIFT on the first n
    frames with and without the keep masks using the W140 extraction settings; report keypoints on excluded pixels and
    whether masked keypoints == unmasked keypoints minus those on excluded pixels."""
    import cv2
    import numpy as np
    import pycolmap
    out = Path(out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    cam = json.load(open(Path(in_dir) / "camera.json"))
    names = [f["image"] for f in json.load(open(Path(in_dir) / "frames.json"))[:n]]
    params = f"{cam['fx']},{cam['fy']},{cam['cx'] + 0.5},{cam['cy'] + 0.5}"
    kps = {}
    for tag, mp in (("unmasked", None), ("masked", mask_dir)):
        r = pycolmap.ImageReaderOptions()
        r.camera_model, r.camera_params = "PINHOLE", params
        if mp:
            r.mask_path = str(mp)
        e = pycolmap.FeatureExtractionOptions()
        e.num_threads, e.use_gpu = 8, False
        db = out / f"{tag}.db"
        pycolmap.extract_features(db, Path(in_dir) / "images", image_names=names, camera_mode=pycolmap.CameraMode.SINGLE,
                                  reader_options=r, extraction_options=e, device=pycolmap.Device.cpu)
        d = pycolmap.Database.open(str(db))
        kps[tag] = {im.name: np.asarray(d.read_keypoints(im.image_id))[:, :2] for im in d.read_all_images()}
        d.close()
    rep = []
    for nm in names:
        keep = cv2.imread(str(Path(mask_dir) / (nm + ".png")), cv2.IMREAD_UNCHANGED) > 0
        def excl(xy):
            ix = np.clip(np.floor(xy[:, 0]).astype(int), 0, keep.shape[1] - 1)
            iy = np.clip(np.floor(xy[:, 1]).astype(int), 0, keep.shape[0] - 1)
            return ~keep[iy, ix]
        u, m = kps["unmasked"][nm], kps["masked"][nm]
        rep.append({"image": nm, "n_unmasked": int(len(u)), "n_masked": int(len(m)), "unmasked_on_excluded": int(excl(u).sum()),
                    "masked_on_excluded": int(excl(m).sum()), "exact_subset": bool(len(m) == len(u) - int(excl(u).sum())),
                    "excluded_fraction_px": float((~keep).mean())})
    res = {"frames": rep, "ok": bool(rep and all(r["masked_on_excluded"] == 0 and r["exact_subset"] for r in rep))}
    (out / "extract_check.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res))
    return 0 if res["ok"] else 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep-masks", required=True)
    ap.add_argument("--extract-check", type=int, default=0, help="PRE-DECLARATION: extraction-only mask contract on the first N frames")
    a = ap.parse_args()
    masks = check_masks(a.input, a.keep_masks)
    if a.extract_check:
        return extract_check(a.input, a.keep_masks, a.out, a.extract_check)
    import pycolmap
    orig = pycolmap.ImageReaderOptions

    def reader_with_masks():
        o = orig()
        o.mask_path = str(Path(a.keep_masks))
        return o
    pycolmap.ImageReaderOptions = reader_with_masks
    spec = importlib.util.spec_from_file_location("w140_colmap_estimate", W140_EST)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.argv = [str(W140_EST), "--input", a.input, "--out", a.out]
    rc = mod.main()
    pycolmap.ImageReaderOptions = orig
    json.dump({"wrapper": "w141_colmap_masked_estimate", "w140_estimator": str(W140_EST),
               "w140_estimator_sha256_lf": hashlib.sha256(W140_EST.read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
               "mask_path": str(Path(a.keep_masks)), "mask_convention": "COLMAP: <image>.png, 0 = exclude, 255 = keep",
               "masks": masks}, open(Path(a.out) / "mask_binding.json", "w"), indent=1)
    return rc


if __name__ == "__main__":
    sys.exit(main())
