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


# --------------------------- docs/32: local-hinge injection topology ----

def _sign_model(injection: str, seed: int = 0):
    from framepose.signs import SIGN_FIELD_COUNT

    torch.manual_seed(seed)
    return build_model(ModelConfig(sign_fields=SIGN_FIELD_COUNT, hinge_sign_injection=injection)).eval()


def test_hinge_sign_injection_rejects_unknown_values():
    with pytest.raises(ValueError, match="hinge_sign_injection"):
        ModelConfig(hinge_sign_injection="mid_attention")


def test_pre_attention_is_the_default_and_matches_historical_construction():
    """The default must be exactly the historical FramePoseEstimator: no new
    buffers, no behavior change, for every existing (S0/S1/O_*/H_*) config."""
    default_config = ModelConfig()
    assert default_config.hinge_sign_injection == "pre_attention"
    model = build_model(ModelConfig(sign_fields=7))
    assert not hasattr(model, "orientation_joint_mask")
    assert not hasattr(model, "hinge_only_joint_mask")
    assert hasattr(model, "sign_joint_mask")


def test_post_attention_and_pre_attention_share_one_parameter_count():
    """Pure topology attribution: same embeddings, same mask, same width --
    only WHEN the hinge contribution is added differs."""
    pre = build_model(ModelConfig(sign_fields=7, hinge_sign_injection="pre_attention"))
    post = build_model(ModelConfig(sign_fields=7, hinge_sign_injection="post_attention"))
    assert parameter_report(pre) == parameter_report(post)


def test_post_attention_hinge_toggle_changes_only_its_own_routed_joint():
    """The architectural claim docs/32 exists to test: after moving hinge
    injection past every attention block, toggling one hinge sign must
    change EXACTLY that joint's 3 output coordinates and nothing else."""
    from framepose.contract import JOINT_NAMES
    from framepose.signs import SIGN_FIELD_COUNT, SIGN_FIELD_NAMES

    model = _sign_model("post_attention")
    geometry = torch.randn(2, 17, GEOMETRY_FEATURES)
    base_state = torch.zeros(2, SIGN_FIELD_COUNT, dtype=torch.long)

    field = "left_knee_forward_bend"
    index = SIGN_FIELD_NAMES.index(field)
    toggled_state = base_state.clone()
    toggled_state[:, index] = 1

    with torch.no_grad():
        baseline = model(geometry, None, base_state)
        moved = model(geometry, None, toggled_state)

    displacement = (moved - baseline).abs().sum(dim=-1)  # (2, 17)
    routed_joint = JOINT_NAMES.index("left_knee")
    assert displacement[:, routed_joint].gt(0).all(), "the routed joint must move"
    other_joints = [i for i in range(len(JOINT_NAMES)) if i != routed_joint]
    assert torch.equal(displacement[:, other_joints], torch.zeros_like(displacement[:, other_joints])), (
        "no other joint may move at all under post_attention hinge injection")


def test_post_attention_orientation_toggle_still_propagates_globally():
    """Orientation fields keep the historical (pre-attention) topology even
    under hinge_sign_injection='post_attention' -- their behaviour is
    unchanged, so a torso_facing toggle can still move other joints."""
    from framepose.contract import JOINT_NAMES
    from framepose.signs import SIGN_FIELD_COUNT, SIGN_FIELD_NAMES

    model = _sign_model("post_attention")
    geometry = torch.randn(2, 17, GEOMETRY_FEATURES)
    base_state = torch.zeros(2, SIGN_FIELD_COUNT, dtype=torch.long)

    index = SIGN_FIELD_NAMES.index("torso_facing")
    toggled_state = base_state.clone()
    toggled_state[:, index] = 1

    with torch.no_grad():
        baseline = model(geometry, None, base_state)
        moved = model(geometry, None, toggled_state)

    displacement = (moved - baseline).abs().sum(dim=-1)
    # torso_facing is not one of the joints it directly governs (it governs
    # torso/hip/shoulder joints); a joint OUTSIDE that set moving confirms
    # attention-mediated propagation is intact for orientation fields.
    outside_joint = JOINT_NAMES.index("left_wrist")
    assert displacement[:, outside_joint].gt(0).any(), (
        "orientation-field propagation must remain global under post_attention")


def test_pre_attention_hinge_toggle_still_propagates_globally():
    """Sanity check that the historical topology's own cross-joint
    propagation (the mechanism docs/32 attributes the knee leakage to) is
    reproduced by this exact model/config path, for contrast with the
    post_attention result above."""
    from framepose.contract import JOINT_NAMES
    from framepose.signs import SIGN_FIELD_COUNT, SIGN_FIELD_NAMES

    model = _sign_model("pre_attention")
    geometry = torch.randn(2, 17, GEOMETRY_FEATURES)
    base_state = torch.zeros(2, SIGN_FIELD_COUNT, dtype=torch.long)

    index = SIGN_FIELD_NAMES.index("left_knee_forward_bend")
    toggled_state = base_state.clone()
    toggled_state[:, index] = 1

    with torch.no_grad():
        baseline = model(geometry, None, base_state)
        moved = model(geometry, None, toggled_state)

    displacement = (moved - baseline).abs().sum(dim=-1)
    other_joint = JOINT_NAMES.index("right_elbow")
    assert displacement[:, other_joint].gt(0).any(), (
        "pre_attention hinge injection is expected to leak to unrelated joints")
