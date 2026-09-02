"""Frozen visual backbones for the three controlled observation candidates.

F0 uses none. F1 and F2 use the *same* ViT-B/16 architecture at the *same*
224x224 resolution producing the *same* 14x14 patch-token grid, so the only
variable between them is what the weights were pretrained on:

    F1  vit_base_patch16_224.augreg_in21k_ft_in1k   vision-only supervision
    F2  vit_base_patch16_siglip_224.webli           image-text (vision-language)

Both are used as representation extractors. No text encoder, no language
decoder, no autoregressive generation is present in the pose inference path.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BackboneSpec:
    key: str
    kind: str                       # "none" | "vision" | "vision_language"
    timm_model: str | None
    hub_id: str | None
    pretraining: str
    license: str
    embed_dim: int | None
    token_grid: tuple[int, int] | None
    input_resolution: int | None

    @property
    def token_count(self) -> int:
        if self.token_grid is None:
            return 0
        return self.token_grid[0] * self.token_grid[1]

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key, "kind": self.kind, "timm_model": self.timm_model,
            "hub_id": self.hub_id, "pretraining": self.pretraining, "license": self.license,
            "embed_dim": self.embed_dim, "token_grid": list(self.token_grid) if self.token_grid else None,
            "token_count": self.token_count, "input_resolution": self.input_resolution,
        }


BACKBONES: dict[str, BackboneSpec] = {
    "none": BackboneSpec(
        key="none", kind="none", timm_model=None, hub_id=None,
        pretraining="none (F0 geometry-only control)", license="n/a",
        embed_dim=None, token_grid=None, input_resolution=None,
    ),
    "vit_in21k": BackboneSpec(
        key="vit_in21k", kind="vision",
        timm_model="vit_base_patch16_224.augreg_in21k_ft_in1k",
        hub_id="timm/vit_base_patch16_224.augreg_in21k_ft_in1k",
        pretraining="ImageNet-21k supervised classification, fine-tuned on ImageNet-1k (AugReg)",
        license="Apache-2.0",
        embed_dim=768, token_grid=(14, 14), input_resolution=224,
    ),
    "siglip": BackboneSpec(
        key="siglip", kind="vision_language",
        timm_model="vit_base_patch16_siglip_224.webli",
        hub_id="timm/ViT-B-16-SigLIP",
        pretraining="SigLIP image-text sigmoid contrastive pretraining on WebLI",
        license="Apache-2.0",
        embed_dim=768, token_grid=(14, 14), input_resolution=224,
    ),
}

# The vision tower shared by lightweight open VLM stacks (PaliGemma, SmolVLM,
# Idefics) is exactly this SigLIP ViT. Using the tower alone is the documented
# preference of the VLM usage contract: it is the multimodal visual
# representation, taken without the language decoder.
VLM_BACKBONE_KEYS = ("siglip",)


def resolve_backbone(key: str) -> BackboneSpec:
    if key not in BACKBONES:
        raise ValueError(f"unknown visual backbone {key!r}; known: {sorted(BACKBONES)}")
    return BACKBONES[key]


class FrozenVisualBackbone:
    """Frozen patch-token extractor.  Never trained in this batch."""

    def __init__(self, spec: BackboneSpec, device: str = "cpu") -> None:
        if spec.kind == "none":
            raise ValueError("the 'none' backbone has no module")
        import timm
        import torch

        self.spec = spec
        self.torch = torch
        self.model = timm.create_model(spec.timm_model, pretrained=True, num_classes=0)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model.to(device)
        self.device = device
        config = timm.data.resolve_model_data_config(self.model)
        self.mean = np.asarray(config["mean"], dtype=np.float32)
        self.std = np.asarray(config["std"], dtype=np.float32)
        self.input_size = int(config["input_size"][-1])
        if self.input_size != spec.input_resolution:
            raise ValueError(f"{spec.timm_model} resolves to {self.input_size}px, expected {spec.input_resolution}")
        self.prefix_tokens = int(getattr(self.model, "num_prefix_tokens", 0))

    def preprocess(self, crops: np.ndarray) -> np.ndarray:
        """`(B, H, W, 3)` uint8 -> `(B, 3, H, W)` float32, backbone normalization."""
        values = np.asarray(crops, dtype=np.float32) / 255.0
        values = (values - self.mean) / self.std
        return np.ascontiguousarray(values.transpose(0, 3, 1, 2))

    def tokens(self, crops: np.ndarray) -> np.ndarray:
        """`(B, H, W, 3)` uint8 -> `(B, token_count, embed_dim)` float32 patch tokens."""
        torch = self.torch
        batch = torch.as_tensor(self.preprocess(crops), device=self.device)
        with torch.no_grad():
            features = self.model.forward_features(batch)
        if features.ndim != 3:
            raise ValueError(f"{self.spec.timm_model} did not return sequence features")
        if self.prefix_tokens:
            features = features[:, self.prefix_tokens:]
        if features.shape[1] != self.spec.token_count:
            raise ValueError(
                f"{self.spec.timm_model} returned {features.shape[1]} patch tokens, "
                f"expected {self.spec.token_count}")
        if features.shape[2] != self.spec.embed_dim:
            raise ValueError(
                f"{self.spec.timm_model} returned width {features.shape[2]}, expected {self.spec.embed_dim}")
        return features.float().cpu().numpy()

    def provenance(self) -> dict[str, Any]:
        """Exact, offline-checkable identity of the weights this run consumed."""
        digest = hashlib.sha256()
        for name, tensor in sorted(self.model.state_dict().items()):
            digest.update(name.encode("utf-8"))
            digest.update(tensor.detach().float().cpu().numpy().tobytes())
        total = sum(parameter.numel() for parameter in self.model.parameters())
        trainable = sum(parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad)
        return {
            **self.spec.to_dict(),
            "weights_sha256": digest.hexdigest(),
            "parameter_count": int(total),
            "trainable_parameter_count": int(trainable),
            "frozen": trainable == 0,
            "preprocessing": {
                "mean": [float(value) for value in self.mean],
                "std": [float(value) for value in self.std],
                "input_resolution": self.input_size,
                "prefix_tokens_dropped": self.prefix_tokens,
            },
            "text_encoder_loaded": False,
            "language_decoder_loaded": False,
            "autoregressive_generation_used": False,
        }
