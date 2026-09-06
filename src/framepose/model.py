"""FramePoseEstimator — Layer A's model interface.

    (explicit 2D joint geometry [, RGB-derived visual tokens]) -> (B, 17, 3)

Explicit geometry is a first-class input and is present in every candidate; the
visual path complements it and never replaces it. Fusion is geometry-aware:
canonical joint queries read the image's spatial tokens through cross-attention
rather than being handed one globally pooled image vector.

Width and depth are fixed once (width 256 matches the Legacy Temporal Pose
Baseline's `channels=256`); this batch runs no depth/width sweep.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

from framepose.contract import JOINT_COUNT
from framepose.signs import SIGN_FIELD_COUNT, joint_field_matrix


MODEL_SCHEMA = "animcv_frame_pose_estimator_v1"

# Fixed design constants. Deliberately not swept in this batch.
DEFAULT_WIDTH = 256
DEFAULT_HEADS = 8
DEFAULT_FUSION_DEPTH = 2
DEFAULT_FEEDFORWARD_MULTIPLIER = 4

# Geometry token features per joint: x, y (crop-normalized), confidence, validity.
GEOMETRY_FEATURES = 4

# A sign field takes three values (-1, 0, +1), so conditioning is one small
# learned categorical embedding per (field, value). Fixed, never swept.
SIGN_VALUE_COUNT = 3


@dataclass(frozen=True)
class ModelConfig:
    visual_dim: int | None = None
    visual_tokens: int = 0
    # Discrete sign conditioning (Sign Contract). 0 disables it entirely; the
    # S0 neutral control keeps it enabled and feeds every field UNKNOWN, so the
    # control is capacity-matched by construction.
    sign_fields: int = 0
    width: int = DEFAULT_WIDTH
    heads: int = DEFAULT_HEADS
    fusion_depth: int = DEFAULT_FUSION_DEPTH
    feedforward_multiplier: int = DEFAULT_FEEDFORWARD_MULTIPLIER

    def __post_init__(self) -> None:
        if self.width <= 0 or self.width % self.heads:
            raise ValueError("width must be positive and divisible by heads")
        if self.fusion_depth < 1:
            raise ValueError("fusion_depth must be at least 1")
        if (self.visual_dim is None) != (self.visual_tokens == 0):
            raise ValueError("visual_dim and visual_tokens must be set together")
        if self.visual_dim is not None and self.visual_dim <= 0:
            raise ValueError("visual_dim must be positive when set")
        if self.sign_fields not in (0, SIGN_FIELD_COUNT):
            raise ValueError(f"sign_fields must be 0 or the contract's {SIGN_FIELD_COUNT}")

    @property
    def uses_vision(self) -> bool:
        return self.visual_dim is not None

    @property
    def uses_signs(self) -> bool:
        return self.sign_fields > 0

    def to_dict(self) -> dict[str, Any]:
        return {"schema": MODEL_SCHEMA, **asdict(self), "uses_vision": self.uses_vision,
                "uses_signs": self.uses_signs, "sign_value_count": SIGN_VALUE_COUNT,
                "joint_count": JOINT_COUNT, "geometry_features": GEOMETRY_FEATURES}


def build_model(config: ModelConfig):
    """Construct the estimator.  `torch` is imported lazily, as elsewhere."""
    torch, nn = _torch()

    class FusionBlock(nn.Module):
        """Pre-norm self-attention over joint queries, optional cross-attention
        into image tokens, then a position-wise feed-forward."""

        def __init__(self) -> None:
            super().__init__()
            self.self_norm = nn.LayerNorm(config.width)
            self.self_attention = nn.MultiheadAttention(config.width, config.heads, batch_first=True)
            self.uses_vision = config.uses_vision
            if config.uses_vision:
                self.cross_norm = nn.LayerNorm(config.width)
                self.cross_attention = nn.MultiheadAttention(config.width, config.heads, batch_first=True)
            hidden = config.width * config.feedforward_multiplier
            self.feed_norm = nn.LayerNorm(config.width)
            self.feed_forward = nn.Sequential(
                nn.Linear(config.width, hidden), nn.GELU(), nn.Linear(hidden, config.width))

        def forward(self, queries, image_tokens):
            normalized = self.self_norm(queries)
            attended, _ = self.self_attention(normalized, normalized, normalized, need_weights=False)
            queries = queries + attended
            if self.uses_vision:
                normalized = self.cross_norm(queries)
                attended, _ = self.cross_attention(normalized, image_tokens, image_tokens, need_weights=False)
                queries = queries + attended
            return queries + self.feed_forward(self.feed_norm(queries))

    class FramePoseEstimator(nn.Module):
        schema = MODEL_SCHEMA

        def __init__(self) -> None:
            super().__init__()
            self.config = config
            self.geometry_projection = nn.Linear(GEOMETRY_FEATURES, config.width)
            self.joint_embedding = nn.Parameter(torch.zeros(JOINT_COUNT, config.width))
            nn.init.normal_(self.joint_embedding, std=0.02)
            if config.uses_signs:
                # One embedding per (sign field, value). Each field reaches only
                # the joints the Sign Contract says it governs, so a changed
                # sign is traceable to a changed pose component.
                self.sign_embedding = nn.Parameter(
                    torch.zeros(config.sign_fields, SIGN_VALUE_COUNT, config.width))
                nn.init.normal_(self.sign_embedding, std=0.02)
                self.register_buffer("sign_joint_mask",
                                     torch.as_tensor(joint_field_matrix()), persistent=False)
            if config.uses_vision:
                self.image_projection = nn.Linear(config.visual_dim, config.width)
                self.image_norm = nn.LayerNorm(config.width)
                self.patch_embedding = nn.Parameter(torch.zeros(config.visual_tokens, config.width))
                nn.init.normal_(self.patch_embedding, std=0.02)
            self.blocks = nn.ModuleList(FusionBlock() for _ in range(config.fusion_depth))
            self.output_norm = nn.LayerNorm(config.width)
            self.head = nn.Sequential(
                nn.Linear(config.width, config.width), nn.GELU(), nn.Linear(config.width, 3))

        def forward(self, geometry, image_tokens=None, sign_state=None):
            """geometry: (B, 17, 4).  image_tokens: (B, T, visual_dim) or None.
            sign_state: (B, sign_fields) with values in {-1, 0, +1}, or None."""
            if geometry.shape[-2:] != (JOINT_COUNT, GEOMETRY_FEATURES):
                raise ValueError(f"geometry must be (B, {JOINT_COUNT}, {GEOMETRY_FEATURES})")
            queries = self.geometry_projection(geometry) + self.joint_embedding
            if self.config.uses_signs:
                if sign_state is None:
                    raise ValueError("this candidate requires a sign state")
                if sign_state.shape[-1] != self.config.sign_fields:
                    raise ValueError(f"sign_state must be (B, {self.config.sign_fields})")
                # -1/0/+1 -> 0/1/2, then gather each field's value embedding and
                # route it only to the joints that field governs.
                indices = (sign_state.long() + 1).clamp(0, SIGN_VALUE_COUNT - 1)
                fields = torch.arange(self.config.sign_fields, device=indices.device)
                selected = self.sign_embedding[fields.unsqueeze(0), indices]
                queries = queries + torch.einsum("jf,bfw->bjw", self.sign_joint_mask, selected)
            elif sign_state is not None:
                raise ValueError("this candidate must not receive a sign state")
            tokens = None
            if self.config.uses_vision:
                if image_tokens is None:
                    raise ValueError("this candidate requires image tokens")
                tokens = self.image_norm(self.image_projection(image_tokens)) + self.patch_embedding
            elif image_tokens is not None:
                raise ValueError("the geometry-only candidate must not receive image tokens")
            for block in self.blocks:
                queries = block(queries, tokens)
            return self.head(self.output_norm(queries))

    return FramePoseEstimator()


def parameter_report(model) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"parameter_count": int(total), "trainable_parameter_count": int(trainable),
            "frozen_parameter_count": int(total - trainable)}


def _torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - environment guard
        raise ImportError("frame pose training requires torch; install the training extra") from exc
    return torch, nn
