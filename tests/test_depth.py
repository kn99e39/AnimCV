import numpy as np
import pytest

from pose.depth_estimator import DepthEstimator, DepthEstimatorConfig
from pose.depth_sampling import sample_depth_at_landmarks
from pose.pose_types import PoseFrame, PoseLandmark


def test_depth_estimator_config_rejects_unknown_encoder():
    with pytest.raises(ValueError):
        DepthEstimator(DepthEstimatorConfig(checkpoint_path="x.pth", encoder="not_a_real_size"))


def test_depth_estimator_raises_import_error_without_dependency(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "depth_anything_v2.dpt" or name.startswith("depth_anything_v2"):
            raise ModuleNotFoundError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    estimator = DepthEstimator(DepthEstimatorConfig(checkpoint_path="x.pth"))

    with pytest.raises(ImportError):
        estimator.infer_frame(np.zeros((4, 4, 3), dtype=np.uint8))


def test_sample_depth_at_landmarks_reads_nearest_pixel():
    depth_map = np.zeros((10, 10), dtype=np.float32)
    depth_map[3, 7] = 42.0  # row=y=3, col=x=7

    frame = PoseFrame(
        frame_index=0,
        timestamp=0.0,
        landmarks={"left_wrist": PoseLandmark(name="left_wrist", x=7.0, y=3.0, confidence=0.9, visible=True)},
    )

    sampled = sample_depth_at_landmarks(frame, depth_map)

    assert sampled.landmarks["left_wrist"].z == pytest.approx(42.0)
    # original frame is untouched (sampling returns a new object)
    assert frame.landmarks["left_wrist"].z is None


def test_sample_depth_at_landmarks_out_of_bounds_gets_none():
    depth_map = np.zeros((10, 10), dtype=np.float32)
    frame = PoseFrame(
        frame_index=0,
        timestamp=0.0,
        landmarks={
            "left_wrist": PoseLandmark(name="left_wrist", x=999.0, y=999.0, confidence=0.9, visible=True)
        },
    )

    sampled = sample_depth_at_landmarks(frame, depth_map)

    assert sampled.landmarks["left_wrist"].z is None


def test_pose_landmark_z_roundtrips_through_json():
    landmark = PoseLandmark(name="left_wrist", x=1.0, y=2.0, confidence=0.9, visible=True, z=5.5)

    restored = PoseLandmark.from_dict(landmark.to_dict())

    assert restored == landmark


def test_pose_landmark_without_z_defaults_to_none_for_old_json():
    # Simulates a pose.json written before the `z` field existed.
    data = {"name": "left_wrist", "x": 1.0, "y": 2.0, "confidence": 0.9, "visible": True}

    landmark = PoseLandmark.from_dict(data)

    assert landmark.z is None


def _install_fake_torch_and_depth_anything_v2(monkeypatch, cuda_available, dpt_class):
    import sys
    import types

    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: cuda_available),
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False)),
        load=lambda path, map_location=None: {},
    )
    fake_pkg = types.ModuleType("depth_anything_v2")
    fake_dpt = types.ModuleType("depth_anything_v2.dpt")
    fake_dpt.DepthAnythingV2 = dpt_class
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "depth_anything_v2", fake_pkg)
    monkeypatch.setitem(sys.modules, "depth_anything_v2.dpt", fake_dpt)


def test_depth_estimator_raises_on_device_mismatch_with_upstream(monkeypatch):
    # Confirmed against the real third_party/Depth-Anything-V2 code
    # (see pose/depth_estimator.py's comment): its infer_image always
    # places the input tensor on cuda when available, regardless of
    # what device the model itself was moved to.
    _install_fake_torch_and_depth_anything_v2(monkeypatch, cuda_available=True, dpt_class=object)

    estimator = DepthEstimator(DepthEstimatorConfig(checkpoint_path="x.pth", device="cpu"))

    with pytest.raises(ValueError, match="cuda"):
        estimator.infer_frame(np.zeros((4, 4, 3), dtype=np.uint8))


def test_depth_estimator_auto_device_resolves_without_raising(monkeypatch):
    class FakeModel:
        def load_state_dict(self, state_dict):
            pass

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            return self

        def infer_image(self, image, input_size):
            return np.zeros(image.shape[:2], dtype=np.float32)

    _install_fake_torch_and_depth_anything_v2(
        monkeypatch, cuda_available=False, dpt_class=lambda **kwargs: FakeModel()
    )

    estimator = DepthEstimator(DepthEstimatorConfig(checkpoint_path="x.pth", device="auto"))

    depth = estimator.infer_frame(np.zeros((6, 8, 3), dtype=np.uint8))

    assert depth.shape == (6, 8)
