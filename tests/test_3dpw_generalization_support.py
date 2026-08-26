"""Focused contracts for the diagnostic-only 3DPW support analysis."""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _ROOT / "scripts" / "diagnose_3dpw_generalization_support.py"


def _load_module():
    sys.path.insert(0, str(_ROOT / "src"))
    try:
        spec = importlib.util.spec_from_file_location("diagnose_3dpw_generalization_support", _SCRIPT)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def _valid(n):
    import training.temporal_lifter as temporal_lifter
    return np.ones((n, len(temporal_lifter.H36M_NAMES)), dtype=bool)


def test_gt_descriptor_preserves_requested_signed_z_and_forward_y_axes():
    module = _load_module()
    targets = np.zeros((2, 17, 3), dtype=float)
    valid = _valid(2)
    targets[:, module.LEFT_SHOULDER] = [-1.0, 0.2, 0.3]
    targets[:, module.RIGHT_SHOULDER] = [1.0, 0.7, -0.1]
    targets[:, module.LEFT_HIP] = [-0.8, 0.1, -0.2]
    targets[:, module.RIGHT_HIP] = [0.8, 0.4, 0.4]

    geometry = module._target_frame_geometry(targets, valid)
    offsets = module._window_offsets(len(targets), 5)
    temporal = module._target_temporal_geometry(geometry, 30.0, offsets)
    descriptor, descriptor_valid = module._target_descriptor(geometry, temporal)

    assert geometry["shoulder"]["signed_z"].tolist() == pytest.approx([-0.4, -0.4])
    assert geometry["shoulder"]["signed_forward_y"].tolist() == pytest.approx([0.5, 0.5])
    assert geometry["hip"]["signed_z"].tolist() == pytest.approx([0.6, 0.6])
    assert np.isfinite(geometry["root_orientation"]).all()
    assert descriptor.shape == (2, 27)
    assert descriptor_valid.all()


def test_input_descriptor_keeps_canonical_geometry_and_temporal_context():
    module = _load_module()
    inputs = np.zeros((5, 17, 3), dtype=float)
    inputs[..., 2] = 0.9
    inputs[:, module.THORAX, :2] = [0.0, 1.0]
    inputs[:, module.LEFT_SHOULDER, :2] = [-0.5, 0.5]
    inputs[:, module.RIGHT_SHOULDER, :2] = [0.5, 0.5]
    inputs[:, module.LEFT_HIP, :2] = [-0.4, 0.1]
    inputs[:, module.RIGHT_HIP, :2] = [0.4, 0.1]
    raw = inputs.copy()
    geometry = module._input_frame_geometry(inputs, raw)
    offsets = module._window_offsets(len(inputs), 5)
    temporal = module._input_temporal_geometry(geometry, inputs, 30.0, offsets)
    descriptor, descriptor_valid = module._input_descriptor(geometry, inputs, temporal)

    assert geometry["torso_height"].tolist() == pytest.approx([1.0] * 5)
    assert descriptor.shape == (5, 119)
    assert descriptor_valid.all()
    assert np.isfinite(temporal["window_net_joint_displacement_normalized"]).all()


def test_temporal_window_boundary_is_clipped_inside_one_sequence():
    module = _load_module()
    offsets = module._window_offsets(3, 81)

    assert offsets.shape == (3, 81)
    assert offsets[0, 0] == 0
    assert offsets[0, -1] == 2
    assert offsets[-1, 0] == 0
    assert offsets[-1, -1] == 2


def test_hard_set_selection_is_deterministic_and_uses_a9_only():
    module = _load_module()
    metadata = [
        {"sequence_id": "seq", "sequence_index": 0, "local_index": index, "frame_index": index}
        for index in range(4)
    ]
    split = {"name": "test", "metadata": metadata, "ranges": [(0, 4)]}
    a9 = np.asarray([1.0, 9.0, 5.0, 3.0])
    a12 = np.asarray([100.0, 1.0, 2.0, 3.0])

    first = module._select_tail(split, a9, a12, 0.5)
    second = module._select_tail(split, a9, a12, 0.5)

    assert first["indices"].tolist() == [1, 2]
    assert first["indices"].tolist() == second["indices"].tolist()
    assert [item["frame_id"] for item in first["records"]] == [1, 2]


def test_sequence_disjoint_nearest_support_excludes_same_sequence():
    module = _load_module()
    query = np.asarray([[0.0]], dtype=float)
    query_valid = np.asarray([True])
    query_sequences = np.asarray(["query"], dtype=object)
    support = np.asarray([[0.01], [0.02], [10.0]], dtype=float)
    support_valid = np.asarray([True, True, True])
    support_sequences = np.asarray(["query", "query", "other"], dtype=object)
    scaler = {"mean": np.asarray([0.0]), "scale": np.asarray([1.0])}

    distance, index = module._nearest_support(
        query, query_valid, query_sequences, support, support_valid, support_sequences, scaler,
        np.asarray([0]),
    )

    assert index.tolist() == [2]
    assert distance.tolist() == pytest.approx([10.0])
