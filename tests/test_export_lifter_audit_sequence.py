import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _ROOT / "scripts" / "export_lifter_audit_sequence.py"


def _load_module():
    sys.path.insert(0, str(_ROOT / "src"))
    try:
        spec = importlib.util.spec_from_file_location("export_lifter_audit_sequence", _SCRIPT)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def _metadata(actions):
    return [{"action": action} for action in actions]


def test_action_bounds_returns_the_contiguous_global_index_range():
    module = _load_module()
    metadata = _metadata(["seq_a", "seq_a", "seq_a", "seq_b", "seq_b"])

    assert module._action_bounds(metadata, "seq_a") == (0, 2)
    assert module._action_bounds(metadata, "seq_b") == (3, 4)


def test_action_bounds_rejects_an_interleaved_action():
    module = _load_module()
    metadata = _metadata(["seq_a", "seq_b", "seq_a"])

    with pytest.raises(ValueError, match="not contiguous"):
        module._action_bounds(metadata, "seq_a")


def test_action_bounds_rejects_an_unknown_action():
    module = _load_module()
    with pytest.raises(ValueError, match="no frames found"):
        module._action_bounds(_metadata(["seq_a"]), "seq_missing")


def test_resolve_window_maps_local_frames_onto_global_indices():
    module = _load_module()
    metadata = _metadata(["seq_x"] * 3 + ["seq_a"] * 10 + ["seq_y"] * 2)

    assert module._resolve_window(metadata, "seq_a", 0, None) == (3, 12)
    assert module._resolve_window(metadata, "seq_a", 2, 5) == (5, 8)


def test_resolve_window_rejects_a_window_outside_the_action():
    module = _load_module()
    metadata = _metadata(["seq_a"] * 4)

    with pytest.raises(ValueError, match="outside action"):
        module._resolve_window(metadata, "seq_a", 0, 10)
    with pytest.raises(ValueError, match="outside action"):
        module._resolve_window(metadata, "seq_a", -1, 2)


def test_build_sequence_export_orders_frames_from_zero_and_pairs_gt_with_estimate():
    module = _load_module()
    from pose.pose_lifter import H36M_NAMES

    n = len(H36M_NAMES)
    targets = np.arange(4 * n * 3, dtype=np.float32).reshape(4, n, 3)
    prediction = targets + 1000.0

    export = module._build_sequence_export(prediction, targets, global_start=1, global_end=2, action="seq_a", fps=30.0)

    assert export["schema"] == module.SCHEMA
    assert export["action"] == "seq_a"
    assert export["fps"] == 30.0
    assert [frame["frame_index"] for frame in export["frames"]] == [0, 1]
    first = export["frames"][0]
    assert first["reference"]["pelvis"] == [float(value) for value in targets[1, 0]]
    assert first["estimate"]["pelvis"] == [float(value) for value in prediction[1, 0]]


def test_cli_writes_a_contiguous_window_json(tmp_path, monkeypatch, capsys):
    pytest.importorskip("torch", reason="the export CLI requires the optional training extra")
    module = _load_module()
    from pose.pose_lifter import LiftedPoseFrame, LiftedPosePoint, LiftedPoseSequence, H36M_NAMES
    from pose.pose_types import PoseFrame, PoseLandmark, PoseSequence
    from training.temporal_lifter import TrainingConfig, build_dataset, save_dataset, train

    names = set(H36M_NAMES) - {"thorax"}
    pose = PoseSequence(
        [PoseFrame(i, i / 25, {name: PoseLandmark(name, 10 + i, 20, 1.0, True) for name in names}) for i in range(8)],
        25,
    )
    target = LiftedPoseSequence(
        [LiftedPoseFrame(i, i / 25, {name: LiftedPosePoint(name, (i / 10, 0, 0), 1.0, 0.0) for name in names})
         for i in range(8)],
        25,
    )
    dataset = build_dataset(pose, target, (100, 100), "sequence-smoke")
    dataset_path, checkpoint = tmp_path / "holdout.json", tmp_path / "model.pth"
    save_dataset(dataset, dataset_path)
    train(dataset, checkpoint, TrainingConfig(window=3, channels=8, epochs=1, batch_size=2))

    out = tmp_path / "window.json"
    monkeypatch.setattr(sys, "argv", [
        "export_lifter_audit_sequence.py",
        "--checkpoint", str(checkpoint), "--holdout", str(dataset_path), "--device", "cpu",
        "--action", "sequence-smoke", "--start-frame", "1", "--end-frame", "5",
        "--out", str(out),
    ])

    assert module.main() == 0

    export = json.loads(out.read_text())
    assert export["schema"] == module.SCHEMA
    assert export["fps"] == pytest.approx(25.0)
    assert [frame["frame_index"] for frame in export["frames"]] == list(range(5))

    summary = json.loads(capsys.readouterr().out)
    assert summary["frame_count"] == 5
    assert summary["global_index_range"] == [1, 5]
