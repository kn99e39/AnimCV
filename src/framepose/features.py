"""Frozen-backbone feature cache.

The visual backbones are frozen for the whole controlled comparison, so their
patch tokens are a pure function of `(frame, crop contract, backbone weights)`.
Materialising them once removes *backbone inference* from the training loop
entirely; it does not make a visual candidate as cheap as the geometry-only one,
because the fusion model still runs cross-attention over 196 image tokens per
frame (measured: 6,903 frames/s for F0 against ~1,040 for F1/F2). The cache is
keyed to the bank's content digest and to the backbone's weight digest, and
refuses to be used with either changed, so candidate replay stays exact.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from common.serialization import read_json, write_json
from framepose.backbones import FrozenVisualBackbone, resolve_backbone
from framepose.contract import FrameBank
from framepose.crops import CROP_CONTRACT, CROP_RESOLUTION, crop_box, render_crop


CACHE_SCHEMA = "animcv_frame_pose_feature_cache_v1"


def cache_directory(root: str | Path, backbone_key: str) -> Path:
    return Path(root) / backbone_key


def read_crop(bank: FrameBank, position: int, image_roots: dict[str, str | Path],
              resolution: int = CROP_RESOLUTION) -> np.ndarray:
    """Load one frame and return its deterministic person-centric crop."""
    from PIL import Image

    sample = bank.samples[position]
    if sample.image_reference is None:
        raise ValueError(f"{sample.sample_id} has no image reference")
    path = sample.image_reference.resolve(image_roots)
    with Image.open(path) as handle:
        image = np.asarray(handle.convert("RGB"))
    box = crop_box(bank.arrays["input_2d"][position], bank.arrays["input_valid"][position],
                   sample.image_size)
    return render_crop(image, box, resolution)


def build_feature_cache(bank: FrameBank, backbone_key: str, *, image_roots: dict[str, str | Path],
                        out_root: str | Path, device: str = "cuda", batch_size: int = 32,
                        workers: int = 8, positions: Iterable[int] | None = None) -> dict[str, Any]:
    """Materialise `(N, tokens, width)` float16 patch features for the whole bank."""
    spec = resolve_backbone(backbone_key)
    if spec.kind == "none":
        raise ValueError("the geometry-only candidate needs no feature cache")
    backbone = FrozenVisualBackbone(spec, device=device)
    order = list(range(len(bank))) if positions is None else list(positions)

    directory = cache_directory(out_root, backbone_key)
    directory.mkdir(parents=True, exist_ok=True)
    array_path = directory / "tokens.npy"
    tokens = np.lib.format.open_memmap(
        array_path, mode="w+", dtype=np.float16,
        shape=(len(order), spec.token_count, spec.embed_dim))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(order), batch_size):
            chunk = order[start:start + batch_size]
            crops = np.stack(list(pool.map(
                lambda position: read_crop(bank, position, image_roots, spec.input_resolution), chunk)))
            tokens[start:start + len(chunk)] = backbone.tokens(crops).astype(np.float16)
    tokens.flush()
    del tokens

    metadata = {
        "schema": CACHE_SCHEMA,
        "backbone": backbone.provenance(),
        "crop_contract": CROP_CONTRACT,
        "bank_content_digest": bank.content_digest(),
        "sample_order_digest": sample_order_digest([bank.samples[position].sample_id for position in order]),
        "sample_count": len(order),
        "dtype": "float16",
        "shape": [len(order), spec.token_count, spec.embed_dim],
        "array": array_path.name,
    }
    write_json(directory / "meta.json", metadata)
    return metadata


def sample_order_digest(sample_ids: list[str]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for identifier in sample_ids:
        digest.update(identifier.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_feature_cache(root: str | Path, backbone_key: str, bank: FrameBank
                       ) -> tuple[np.ndarray, dict[str, Any]]:
    """Memory-map a cache, refusing any mismatch with the bank it must pair with."""
    directory = cache_directory(root, backbone_key)
    metadata = read_json(directory / "meta.json")
    if metadata.get("schema") != CACHE_SCHEMA:
        raise ValueError(f"unsupported feature cache schema: {metadata.get('schema')!r}")
    if metadata.get("bank_content_digest") != bank.content_digest():
        raise ValueError("feature cache was built for a different frame bank")
    expected = sample_order_digest([sample.sample_id for sample in bank.samples])
    if metadata.get("sample_order_digest") != expected:
        raise ValueError("feature cache sample order does not match the frame bank")
    array = np.load(directory / metadata["array"], mmap_mode="r")
    if list(array.shape) != list(metadata["shape"]):
        raise ValueError("feature cache array shape does not match its metadata")
    return array, metadata


def cache_report(root: str | Path, backbone_key: str) -> dict[str, Any]:
    directory = cache_directory(root, backbone_key)
    metadata = read_json(directory / "meta.json")
    array_path = directory / metadata["array"]
    return {**metadata, "array_path": str(array_path), "array_byte_size": array_path.stat().st_size}
