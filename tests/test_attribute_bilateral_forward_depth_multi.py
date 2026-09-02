"""Focused tests for the multi-checkpoint forward-depth attribution script.

Diagnostic-only units: verifies checkpoint-argument parsing and that the
hard-set definition is fixed from exactly the nominated label.
"""
import importlib.util
from pathlib import Path
import sys

import pytest

pytest.importorskip("torch", reason="attribution checks require the optional training extra")

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _ROOT / "scripts" / "attribute_bilateral_forward_depth_multi.py"


def _load_module():
    sys.path.insert(0, str(_ROOT / "src"))
    try:
        spec = importlib.util.spec_from_file_location("attribute_bilateral_forward_depth_multi", _SCRIPT)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def test_parse_checkpoint_arg_splits_label_and_path():
    module = _load_module()
    label, path = module._parse_checkpoint_arg("historical_a9=/some/path/direct_mix.pth")
    assert label == "historical_a9"
    assert path == Path("/some/path/direct_mix.pth")


def test_parse_checkpoint_arg_rejects_missing_equals():
    import argparse

    module = _load_module()
    with pytest.raises(argparse.ArgumentTypeError):
        module._parse_checkpoint_arg("no_equals_sign_here")


def test_parse_checkpoint_arg_handles_paths_containing_equals():
    module = _load_module()
    label, path = module._parse_checkpoint_arg("label=/weird=path/model.pth")
    assert label == "label"
    assert path == Path("/weird=path/model.pth")


def test_per_frame_yaw_matches_root_yaw_error_degrees():
    import numpy as np
    from pose.pose_lifter import H36M_NAMES
    from training.temporal_lifter import _root_yaw_error_degrees

    module = _load_module()
    n = 3
    rng = np.random.default_rng(5)
    target = rng.standard_normal((n, len(H36M_NAMES), 3)).astype(np.float32)
    prediction = target + rng.standard_normal((n, len(H36M_NAMES), 3)).astype(np.float32) * 0.2
    valid = np.ones((n, len(H36M_NAMES)), dtype=bool)

    result = module._per_frame_yaw(prediction, target, valid)
    expected = np.array([_root_yaw_error_degrees(p, t, v) for p, t, v in zip(prediction, target, valid)])
    assert np.allclose(result, expected, equal_nan=True)
