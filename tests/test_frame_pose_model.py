import numpy as np
import pytest

torch = pytest.importorskip("torch")

from framepose.backbones import BACKBONES, VLM_BACKBONE_KEYS, resolve_backbone
from framepose.model import GEOMETRY_FEATURES, ModelConfig, build_model, parameter_report


GEOMETRY_ONLY = ModelConfig()
VISION = ModelConfig(visual_dim=768, visual_tokens=196)


def _inputs(batch=4, tokens=196, width=768):
    geometry = torch.randn(batch, 17, GEOMETRY_FEATURES)
    return geometry, torch.randn(batch, tokens, width)


def test_geometry_only_path_produces_a_canonical_pose():
    torch.manual_seed(0)
    model = build_model(GEOMETRY_ONLY)
    geometry, _ = _inputs()
    assert model(geometry).shape == (4, 17, 3)


def test_visual_and_vlm_paths_share_one_fusion_and_head_contract():
    reports = {}
    for key in ("vit_in21k", "siglip"):
        spec = resolve_backbone(key)
        torch.manual_seed(0)
        model = build_model(ModelConfig(visual_dim=spec.embed_dim, visual_tokens=spec.token_count))
        geometry, tokens = _inputs(tokens=spec.token_count, width=spec.embed_dim)
        assert model(geometry, tokens).shape == (4, 17, 3)
        reports[key] = parameter_report(model)
    # F1 and F2 differ only in what the frozen backbone was pretrained on, so the
    # trainable model must be parameter-identical between them.
    assert reports["vit_in21k"] == reports["siglip"]


def test_candidate_paths_refuse_each_other_s_inputs():
    torch.manual_seed(0)
    geometry_only = build_model(GEOMETRY_ONLY)
    vision = build_model(VISION)
    geometry, tokens = _inputs()
    with pytest.raises(ValueError, match="must not receive image tokens"):
        geometry_only(geometry, tokens)
    with pytest.raises(ValueError, match="requires image tokens"):
        vision(geometry)
    with pytest.raises(ValueError, match="geometry must be"):
        vision(torch.randn(4, 17, 3), tokens)


def test_joint_queries_own_their_own_output_row():
    """Each canonical joint's prediction must be produced by its own query."""
    torch.manual_seed(0)
    model = build_model(GEOMETRY_ONLY).eval()
    geometry = torch.randn(1, 17, GEOMETRY_FEATURES, requires_grad=True)
    output = model(geometry)
    assert output.shape[1] == 17
    gradient, = torch.autograd.grad(output[0, 5].sum(), geometry, retain_graph=True)
    # Self-attention lets joints inform one another, but the queried joint must
    # dominate its own readout.
    magnitudes = gradient[0].abs().sum(dim=-1)
    assert int(magnitudes.argmax()) == 5


def test_image_tokens_actually_reach_the_prediction():
    torch.manual_seed(0)
    model = build_model(VISION).eval()
    geometry, tokens = _inputs(batch=2)
    first = model(geometry, tokens)
    second = model(geometry, tokens + 1.0)
    assert not torch.allclose(first, second), "the visual path must influence the pose"


def test_model_config_validation():
    with pytest.raises(ValueError):
        ModelConfig(visual_dim=768)
    with pytest.raises(ValueError):
        ModelConfig(width=250, heads=8)
    with pytest.raises(ValueError):
        ModelConfig(fusion_depth=0)


def test_backbone_registry_declares_provenance_and_excludes_language_generation():
    assert set(BACKBONES) == {"none", "vit_in21k", "siglip"}
    assert BACKBONES["vit_in21k"].kind == "vision"
    assert BACKBONES["siglip"].kind == "vision_language"
    assert VLM_BACKBONE_KEYS == ("siglip",)
    for key in ("vit_in21k", "siglip"):
        spec = BACKBONES[key]
        # The controlled pair must be architecture-, resolution- and token-matched.
        assert spec.embed_dim == 768
        assert spec.token_grid == (14, 14)
        assert spec.input_resolution == 224
        assert spec.license and spec.hub_id and spec.timm_model
    with pytest.raises(ValueError):
        resolve_backbone("clip-large")


def test_pose_inference_never_depends_on_text_generation():
    import inspect

    import framepose.backbones as backbones
    import framepose.model as model_module
    import framepose.train as train_module

    for module in (backbones, model_module, train_module):
        source = inspect.getsource(module)
        for forbidden in (".generate(", "AutoTokenizer", "AutoModelForCausalLM", "lm_head",
                          ".text_model", ".text_encoder", "get_text_features"):
            assert forbidden not in source, f"{module.__name__} must not reach for {forbidden}"
    # The backbone wrapper only ever calls the vision tower's feature extractor.
    assert "forward_features" in inspect.getsource(backbones)
    assert "num_classes=0" in inspect.getsource(backbones)
