"""Visual input identity.

A frozen visual feature is a pure function of four things:

```
exact source image content
the geometry used to construct the person crop
the crop contract
the visual backbone's preprocessing
```

`visual_input_fingerprint` binds all four. It exists because the bank's
`content_digest` deliberately covers only frame identity and numeric arrays
(so a metadata correction cannot invalidate a cache), and
`provenance_fingerprint` binds image *references* rather than image *bytes*.
Neither is allowed to grow a third responsibility, so pixel-level cache
identity lives here.

Paths are never part of the identity: the same bank consumed from a different
host, or from a different image root, must produce the same fingerprint.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from common.serialization import read_json, write_json
from framepose.contract import FrameBank
from framepose.crops import CROP_CONTRACT, CROP_CONTRACT_VERSION, CROP_RESOLUTION, crop_contract_digest
from framepose.observations import image_content_digest


VISUAL_INPUT_SCHEMA = "animcv_frame_pose_visual_input_v1"


def preprocessing_identity(mean: Sequence[float], std: Sequence[float], resolution: int,
                           prefix_tokens_dropped: int) -> dict[str, Any]:
    """The backbone-side preprocessing that turns a crop into model input."""
    return {
        "mean": [float(value) for value in mean],
        "std": [float(value) for value in std],
        "input_resolution": int(resolution),
        "prefix_tokens_dropped": int(prefix_tokens_dropped),
    }


def image_content_digests(bank: FrameBank, image_roots: dict[str, str | Path],
                          positions: Iterable[int] | None = None) -> list[str]:
    """SHA-256 of every referenced image's bytes, in bank order.

    Images are frequently shared between actors of the same sequence, so each
    distinct file is read once.
    """
    order = list(range(len(bank))) if positions is None else list(positions)
    cache: dict[str, str] = {}
    digests = []
    for position in order:
        sample = bank.samples[position]
        if sample.image_reference is None:
            digests.append("")
            continue
        key = f"{sample.image_reference.root_key}/{sample.image_reference.relative_path}"
        if key not in cache:
            cache[key] = image_content_digest(sample.image_reference.resolve(image_roots))
        digests.append(cache[key])
    return digests


def image_content_summary(digests: Sequence[str]) -> str:
    """One digest over the ordered per-image content digests."""
    digest = hashlib.sha256()
    digest.update(b"animcv_frame_pose_image_content_v1")
    for value in digests:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def visual_input_fingerprint(*, image_content: str, bank_content_digest: str,
                             sample_order_digest: str, crop_digest: str, crop_resolution: int,
                             preprocessing: dict[str, Any]) -> str:
    payload = {
        "schema": VISUAL_INPUT_SCHEMA,
        "image_content": image_content,
        "bank_content_digest": bank_content_digest,
        "sample_order_digest": sample_order_digest,
        "crop_contract_digest": crop_digest,
        "crop_contract_version": CROP_CONTRACT_VERSION,
        "crop_resolution": int(crop_resolution),
        "preprocessing": preprocessing,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def visual_input_identity(bank: FrameBank, *, image_roots: dict[str, str | Path],
                          preprocessing: dict[str, Any], crop_resolution: int = CROP_RESOLUTION,
                          digests: Sequence[str] | None = None) -> dict[str, Any]:
    """Full, path-independent identity of the visual input for one bank."""
    from framepose.features import sample_order_digest as order_digest

    values = list(digests) if digests is not None else image_content_digests(bank, image_roots)
    if len(values) != len(bank):
        raise ValueError("image content digests must align with the bank sample order")
    summary = image_content_summary(values)
    order = order_digest([sample.sample_id for sample in bank.samples])
    crop_digest = crop_contract_digest()
    return {
        "schema": VISUAL_INPUT_SCHEMA,
        "image_content_summary": summary,
        "bank_content_digest": bank.content_digest(),
        "sample_order_digest": order,
        "crop_contract": CROP_CONTRACT,
        "crop_contract_digest": crop_digest,
        "crop_resolution": int(crop_resolution),
        "preprocessing": preprocessing,
        "sample_count": len(bank),
        "fingerprint": visual_input_fingerprint(
            image_content=summary, bank_content_digest=bank.content_digest(),
            sample_order_digest=order, crop_digest=crop_digest,
            crop_resolution=crop_resolution, preprocessing=preprocessing),
    }


def save_identity(identity: dict[str, Any], path: str | Path) -> Path:
    write_json(path, identity)
    return Path(path)


def load_identity(path: str | Path) -> dict[str, Any]:
    payload = read_json(path)
    if payload.get("schema") != VISUAL_INPUT_SCHEMA:
        raise ValueError(f"unsupported visual input identity schema: {payload.get('schema')!r}")
    return payload
