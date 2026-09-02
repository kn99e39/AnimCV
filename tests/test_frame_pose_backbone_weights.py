"""Real frozen-backbone behaviour.

Skipped where `timm` is not installed (the macOS authoring environment); these
run on the training host, where the controlled comparison is actually executed.
"""

import numpy as np
import pytest

timm = pytest.importorskip("timm")
torch = pytest.importorskip("torch")

from framepose.backbones import FrozenVisualBackbone, resolve_backbone


@pytest.mark.parametrize("key", ["vit_in21k", "siglip"])
def test_backbone_is_frozen_and_matches_its_declared_contract(key):
    spec = resolve_backbone(key)
    backbone = FrozenVisualBackbone(spec, device="cpu")
    provenance = backbone.provenance()

    assert provenance["frozen"] is True
    assert provenance["trainable_parameter_count"] == 0
    assert provenance["parameter_count"] > 50_000_000
    assert provenance["weights_sha256"]
    assert provenance["preprocessing"]["input_resolution"] == 224
    assert provenance["language_decoder_loaded"] is False
    assert provenance["autoregressive_generation_used"] is False
    # No text tower is instantiated by the pose path.
    for attribute in ("text_model", "text", "text_encoder", "lm_head"):
        assert not hasattr(backbone.model, attribute), attribute

    crops = np.full((2, 224, 224, 3), 128, dtype=np.uint8)
    tokens = backbone.tokens(crops)
    assert tokens.shape == (2, spec.token_count, spec.embed_dim)
    assert np.isfinite(tokens).all()
    assert np.allclose(tokens, backbone.tokens(crops)), "frozen features must be deterministic"

    # The backbone contributes no gradient to anything downstream.
    assert all(not parameter.requires_grad for parameter in backbone.model.parameters())
    assert not backbone.model.training


def test_the_two_visual_backbones_are_architecture_matched_but_differently_pretrained():
    vision = FrozenVisualBackbone(resolve_backbone("vit_in21k"), device="cpu").provenance()
    language = FrozenVisualBackbone(resolve_backbone("siglip"), device="cpu").provenance()
    assert vision["token_count"] == language["token_count"] == 196
    assert vision["embed_dim"] == language["embed_dim"] == 768
    assert vision["input_resolution"] == language["input_resolution"] == 224
    assert vision["kind"] == "vision" and language["kind"] == "vision_language"
    assert vision["weights_sha256"] != language["weights_sha256"]
