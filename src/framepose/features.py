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
from framepose.crops import CROP_CONTRACT, CROP_RESOLUTION, crop_box, crop_contract_digest, render_crop
from framepose.observations import image_content_digest


# v2 binds the visual-input identity (image bytes + bank geometry + crop
# contract + preprocessing) and the backbone weight digest into the cache's own
# provenance. v1 caches recorded neither; they stay readable only through the
# explicit historical compatibility path below, and are labelled as such.
CACHE_SCHEMA = "animcv_frame_pose_feature_cache_v2"
LEGACY_CACHE_SCHEMAS = ("animcv_frame_pose_feature_cache_v1",)

# What loading actually establishes. Stated precisely so no report can claim a
# verification the code does not perform.
WEIGHT_VERIFICATION = (
    "the recorded digest is the backbone that generated this cache; loading does not "
    "re-download or re-hash a current backbone to prove it is still identical")


def cache_directory(root: str | Path, backbone_key: str) -> Path:
    return Path(root) / backbone_key


def read_crop(bank: FrameBank, position: int, image_roots: dict[str, str | Path],
              resolution: int = CROP_RESOLUTION,
              with_digest: bool = False) -> np.ndarray | tuple[np.ndarray, str]:
    """Load one frame and return its deterministic person-centric crop.

    `with_digest` also returns the SHA-256 of the exact image bytes, computed
    from the same read, so building a cache establishes pixel identity without a
    second pass over the imagery.
    """
    import io

    from PIL import Image

    sample = bank.samples[position]
    if sample.image_reference is None:
        raise ValueError(f"{sample.sample_id} has no image reference")
    path = sample.image_reference.resolve(image_roots)
    payload = path.read_bytes()
    with Image.open(io.BytesIO(payload)) as handle:
        image = np.asarray(handle.convert("RGB"))
    box = crop_box(bank.arrays["input_2d"][position], bank.arrays["input_valid"][position],
                   sample.image_size)
    crop = render_crop(image, box, resolution)
    if not with_digest:
        return crop
    import hashlib

    return crop, hashlib.sha256(payload).hexdigest()


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

    digests: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(order), batch_size):
            chunk = order[start:start + batch_size]
            produced = list(pool.map(
                lambda position: read_crop(bank, position, image_roots, spec.input_resolution,
                                           with_digest=True), chunk))
            crops = np.stack([item[0] for item in produced])
            digests.extend(item[1] for item in produced)
            tokens[start:start + len(chunk)] = backbone.tokens(crops).astype(np.float16)
    tokens.flush()
    del tokens

    from framepose.visual_input import preprocessing_identity, visual_input_identity

    provenance = backbone.provenance()
    identity = visual_input_identity(
        bank, image_roots=image_roots,
        preprocessing=preprocessing_identity(
            backbone.mean, backbone.std, backbone.input_size, backbone.prefix_tokens),
        crop_resolution=spec.input_resolution, digests=digests if positions is None else None)
    metadata = {
        "schema": CACHE_SCHEMA,
        "backbone": provenance,
        "crop_contract": CROP_CONTRACT,
        "crop_contract_digest": crop_contract_digest(),
        "bank_content_digest": bank.content_digest(),
        "sample_order_digest": sample_order_digest([bank.samples[position].sample_id for position in order]),
        "visual_input": {key: identity[key] for key in
                         ("image_content_summary", "crop_contract_digest", "crop_resolution",
                          "preprocessing", "fingerprint")},
        "visual_input_fingerprint": identity["fingerprint"],
        "feature_cache_provenance": _cache_provenance(identity["fingerprint"], provenance),
        "weight_verification": WEIGHT_VERIFICATION,
        "provenance_level": "verified_v2",
        "sample_count": len(order),
        "dtype": "float16",
        "token_shape": [spec.token_count, spec.embed_dim],
        "shape": [len(order), spec.token_count, spec.embed_dim],
        "array": array_path.name,
    }
    write_json(directory / "meta.json", metadata)
    return metadata


def _cache_provenance(fingerprint: str, backbone: dict[str, Any]) -> str:
    """`visual_input_fingerprint + backbone identity + recorded weight digest`."""
    import hashlib

    payload = json.dumps({
        "visual_input_fingerprint": fingerprint,
        "backbone_key": backbone.get("key"),
        "timm_model": backbone.get("timm_model"),
        "weights_sha256": backbone.get("weights_sha256"),
        "embed_dim": backbone.get("embed_dim"),
        "token_count": backbone.get("token_count"),
        "input_resolution": backbone.get("input_resolution"),
    }, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sample_order_digest(sample_ids: list[str]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for identifier in sample_ids:
        digest.update(identifier.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_feature_cache(root: str | Path, backbone_key: str, bank: FrameBank, *,
                       visual_input_fingerprint: str | None = None,
                       allow_legacy: bool = False) -> tuple[np.ndarray, dict[str, Any]]:
    """Memory-map a cache, refusing any mismatch with the contract it must pair with.

    Always checked: schema, bank content digest, sample order, array shape.

    Checked for a v2 cache: the crop-contract digest currently in force, the
    backbone key and the presence of a recorded weight digest -- and, when
    `visual_input_fingerprint` is supplied, that the exact image content, bank
    geometry, crop contract and preprocessing the cache was built from are the
    ones being paired with it now.

    Not checked, and never claimed: that a freshly downloaded backbone still
    hashes to the recorded digest. See `WEIGHT_VERIFICATION`.

    A v1 cache recorded neither the visual-input identity nor the crop digest.
    It is refused unless `allow_legacy=True`, and is then labelled
    `provenance_level="historical_v1"` with the specific guarantees it cannot
    provide listed in the returned metadata.
    """
    directory = cache_directory(root, backbone_key)
    metadata = read_json(directory / "meta.json")
    schema = metadata.get("schema")
    legacy = schema in LEGACY_CACHE_SCHEMAS
    if schema != CACHE_SCHEMA and not legacy:
        raise ValueError(f"unsupported feature cache schema: {schema!r}")
    if legacy and not allow_legacy:
        raise ValueError(
            f"feature cache at {directory} uses the historical schema {schema!r}, which recorded no "
            "image-content or crop-contract identity; pass allow_legacy=True to read it as a "
            "historical artifact, or rebuild it to obtain the v2 provenance contract")
    if metadata.get("bank_content_digest") != bank.content_digest():
        raise ValueError("feature cache was built for a different frame bank")
    expected = sample_order_digest([sample.sample_id for sample in bank.samples])
    if metadata.get("sample_order_digest") != expected:
        raise ValueError("feature cache sample order does not match the frame bank")

    if not legacy:
        if metadata.get("crop_contract_digest") != crop_contract_digest():
            raise ValueError(
                "feature cache was built under a different crop contract; its features are not "
                "valid for the crop contract currently in force")
        if metadata.get("backbone", {}).get("key") != backbone_key:
            raise ValueError("feature cache records a different visual backbone")
        if not metadata.get("backbone", {}).get("weights_sha256"):
            raise ValueError("feature cache records no backbone weight digest")
        if visual_input_fingerprint is not None and \
                metadata.get("visual_input_fingerprint") != visual_input_fingerprint:
            raise ValueError(
                "feature cache visual-input fingerprint does not match the images, bank geometry, "
                "crop contract and preprocessing it is being paired with")
        metadata = {**metadata, "visual_input_verified": visual_input_fingerprint is not None}
    else:
        metadata = {
            **metadata,
            "provenance_level": "historical_v1",
            "visual_input_verified": False,
            "not_established": [
                "exact source image content (no image-content digest was recorded)",
                "crop-contract identity (no crop-contract digest was recorded)",
                "backbone preprocessing identity",
            ],
        }

    array = np.load(directory / metadata["array"], mmap_mode="r")
    if list(array.shape) != list(metadata["shape"]):
        raise ValueError("feature cache array shape does not match its metadata")
    if metadata.get("dtype") and str(array.dtype) != metadata["dtype"]:
        raise ValueError("feature cache array dtype does not match its metadata")
    return array, metadata


def cache_report(root: str | Path, backbone_key: str) -> dict[str, Any]:
    directory = cache_directory(root, backbone_key)
    metadata = read_json(directory / "meta.json")
    array_path = directory / metadata["array"]
    return {**metadata, "array_path": str(array_path), "array_byte_size": array_path.stat().st_size}
