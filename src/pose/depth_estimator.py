"""Depth Anything V2 backend adapter (relative monocular depth).

Not part of Architecture_v2.md's original v2 scope — section 1.3/14.1
explicitly excludes Depth Anything V2 from v2 ("must not be implemented
in this version... may be added later through extension interfaces").
Added here per explicit follow-up request to improve retargeting
quality beyond the 2D-only MVP. Mirrors the "DepthProvider"
future-extension interface already sketched in section 13, and reuses
the Depth Anything V2 checkout at third_party/Depth-Anything-V2 as the
API reference (confirmed real function names/signatures from
third_party/Depth-Anything-V2/run.py rather than guessed).

``depth_anything_v2``/``torch`` are imported lazily inside
``_load_model`` so the rest of the project keeps working without them
installed — same pattern as every other heavy optional backend
(mmpose, pyassimp) here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# vits ("small") is the only Depth Anything V2 checkpoint size released
# under Apache-2.0; vitb/vitl/vitg are CC-BY-NC-4.0 (see the v1 doc's
# section 13.2 license note, still accurate for v2's usage here).
_MODEL_CONFIGS: dict[str, dict] = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}


@dataclass
class DepthEstimatorConfig:
    checkpoint_path: str
    encoder: str = "vits"
    # "auto" resolves to whatever third_party/Depth-Anything-V2's own
    # infer_image will actually use (see _load_model) -- its input
    # tensor's device isn't configurable, so "auto" is the only choice
    # that's guaranteed not to crash with a device mismatch. Only pass
    # an explicit device if you know it matches that.
    device: str = "auto"
    input_size: int = 518
    batch_size: int = 4  # frames per forward pass in infer_frames (1 = the per-frame infer_image path)


class DepthEstimator:
    """Wraps Depth Anything V2 behind a project-native numpy interface."""

    def __init__(self, config: DepthEstimatorConfig):
        if config.encoder not in _MODEL_CONFIGS:
            raise ValueError(
                f"unknown encoder {config.encoder!r}; expected one of {sorted(_MODEL_CONFIGS)}"
            )
        self._config = config
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                import torch
                from depth_anything_v2.dpt import DepthAnythingV2
            except ImportError as exc:
                raise ImportError(
                    "depth_anything_v2 is not installed. Install its requirements (see "
                    "third_party/Depth-Anything-V2/requirements.txt) and pass --depth-checkpoint."
                ) from exc

            # Confirmed by actually running the real (untrained-weights)
            # model: third_party/Depth-Anything-V2's own infer_image ->
            # image2tensor hardcodes its input tensor's device as
            # "cuda if available, else mps, else cpu" -- it has no
            # parameter to override this and ignores the model's own
            # device entirely. So requesting a device that upstream
            # won't actually place the input on always crashes with a
            # device-mismatch RuntimeError deep in the forward pass,
            # not a clear error. Resolve "auto" to match, and fail fast
            # with a clear message for any other mismatched value.
            auto_device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available() else "cpu"
            )
            if self._config.device == "auto":
                resolved_device = auto_device
            elif self._config.device != auto_device:
                raise ValueError(
                    f"depth_anything_v2's infer_image always places its input on "
                    f"{auto_device!r} on this machine (cuda > mps > cpu, whichever is "
                    f"available) and has no way to override that, so device={self._config.device!r} "
                    f"would crash with a device mismatch. Use device='auto' or {auto_device!r}."
                )
            else:
                resolved_device = self._config.device

            model = DepthAnythingV2(**_MODEL_CONFIGS[self._config.encoder])
            model.load_state_dict(torch.load(self._config.checkpoint_path, map_location="cpu"))
            self._model = model.to(resolved_device).eval()
        return self._model

    def infer_frame(self, image: np.ndarray) -> np.ndarray:
        """Relative depth map, same (H, W) as `image`. Larger values are
        Depth Anything V2's own convention for "closer to the camera".
        This project only ever uses relative ordering/deltas across
        frames, never treats these as metric distances (matches the v1
        doc's section 13.2: depth is a hint, not ground truth)."""
        model = self._load_model()
        return model.infer_image(image, self._config.input_size)

    def infer_frames(self, images: list[np.ndarray]) -> list[np.ndarray]:
        """``infer_frame`` for many frames, ``batch_size`` same-sized frames per forward pass.

        Uses upstream's own preprocessing (``image2tensor``) and the same bilinear resize back to each frame's size as
        ``infer_image``; falls back to the per-frame path when batching is off or the model has no ``image2tensor``.
        """
        bs = max(1, int(self._config.batch_size))
        if bs == 1 or len(images) <= 1:
            return [self.infer_frame(image) for image in images]
        model = self._load_model()
        if not hasattr(model, "image2tensor"):
            return [self.infer_frame(image) for image in images]
        import torch
        import torch.nn.functional as F

        out: list[np.ndarray | None] = [None] * len(images)
        groups: dict[tuple, list[int]] = {}
        for i, image in enumerate(images):
            groups.setdefault(tuple(image.shape), []).append(i)
        with torch.no_grad():
            for idx in groups.values():
                for k in range(0, len(idx), bs):
                    chunk = idx[k:k + bs]
                    tensors, sizes = zip(*(model.image2tensor(images[i], self._config.input_size) for i in chunk))
                    depth = model(torch.cat(tensors, dim=0))
                    h, w = sizes[0]
                    depth = F.interpolate(depth[:, None], (h, w), mode="bilinear", align_corners=True)[:, 0]
                    for i, d in zip(chunk, depth.cpu().numpy()):
                        out[i] = d
        return out
