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

import numpy as np

from framepose.contract import JOINT_COUNT
from framepose.signs import SIGN_FIELD_COUNT, SIGN_FIELD_NAMES, joint_field_matrix


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
    # docs/32's conditional locality candidate. "pre_attention" is the
    # historical FramePoseEstimator, unchanged: every sign field is injected
    # into the joint query before global joint self-attention, so a hinge
    # sign can reach any joint through fusion_depth rounds of attention.
    # "post_attention" changes ONLY where the four hinge fields are added:
    # after every self-/cross-attention block, routed by the same
    # sign_joint_mask to only the one joint that field governs, immediately
    # before the per-joint output head -- so a hinge sign can no longer
    # reach any other joint's output at all, by construction. Orientation
    # fields (torso_facing, shoulder_forward_depth, hip_forward_depth) are
    # injected pre-attention in both modes; their behaviour is unchanged.
    hinge_sign_injection: str = "pre_attention"

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
        if self.hinge_sign_injection not in ("pre_attention", "post_attention"):
            raise ValueError("hinge_sign_injection must be 'pre_attention' or 'post_attention'")

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
                mask = joint_field_matrix()
                self.register_buffer("sign_joint_mask", torch.as_tensor(mask), persistent=False)
                if config.hinge_sign_injection == "post_attention":
                    # Same mask, same embedding table -- split only by WHEN each
                    # field's contribution is added, not by any new routing or
                    # capacity. Computed once from the fixed field/joint
                    # contract, never learned or tuned.
                    hinge_columns = [index for index, name in enumerate(SIGN_FIELD_NAMES)
                                     if name.endswith("_forward_bend")]
                    hinge_mask = np.zeros_like(mask)
                    hinge_mask[:, hinge_columns] = mask[:, hinge_columns]
                    orientation_mask = mask - hinge_mask
                    self.register_buffer("orientation_joint_mask",
                                         torch.as_tensor(orientation_mask), persistent=False)
                    self.register_buffer("hinge_only_joint_mask",
                                         torch.as_tensor(hinge_mask), persistent=False)
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
            hinge_injection = None
            if self.config.uses_signs:
                if sign_state is None:
                    raise ValueError("this candidate requires a sign state")
                if sign_state.shape[-1] != self.config.sign_fields:
                    raise ValueError(f"sign_state must be (B, {self.config.sign_fields})")
                # -1/0/+1 -> 0/1/2, then gather each field's value embedding and
                # route it only to the joints that field governs. Membership is
                # exact, not a range: a range test would let 0.5 through and
                # `.long()` would then truncate it into a confident legal branch,
                # and NaN/inf would pass an unordered comparison entirely.
                finite = torch.isfinite(sign_state.float())
                exact = ((sign_state == -1) | (sign_state == 0) | (sign_state == 1))
                if not bool((finite & exact).all()):
                    raise ValueError(
                        "sign_state values must be exactly -1, 0 or +1; anything else -- including "
                        "0.5, NaN and inf -- is refused, never clamped, rounded or truncated into "
                        "a branch")
                indices = sign_state.long() + 1
                fields = torch.arange(self.config.sign_fields, device=indices.device)
                selected = self.sign_embedding[fields.unsqueeze(0), indices]
                if self.config.hinge_sign_injection == "post_attention":
                    # Historical topology for orientation fields, unchanged.
                    queries = queries + torch.einsum("jf,bfw->bjw", self.orientation_joint_mask, selected)
                    # Hinge fields: computed now, added only after every
                    # attention block, so they never reach self-attention.
                    hinge_injection = torch.einsum("jf,bfw->bjw", self.hinge_only_joint_mask, selected)
                else:
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
            if hinge_injection is not None:
                # output_norm/head are both per-joint (LayerNorm over width,
                # then a position-wise MLP) with no joint-mixing left in the
                # graph, so from this point on a hinge sign can only change
                # its own routed joint's 3D output.
                queries = queries + hinge_injection
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
