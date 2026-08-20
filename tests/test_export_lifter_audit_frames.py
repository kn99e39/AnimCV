import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _ROOT / "scripts" / "export_lifter_audit_frames.py"


def _load_module():
    sys.path.insert(0, str(_ROOT / "src"))
    try:
        spec = importlib.util.spec_from_file_location("export_lifter_audit_frames", _SCRIPT)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def _diagnostic(index, action, worst_hinge_deg, flipped, yaw_error_deg):
    return {
        "global_index": index, "action": action, "worst_hinge_deg": worst_hinge_deg,
        "flipped": flipped, "yaw_error_deg": yaw_error_deg,
    }


def test_frame_diagnostics_matches_official_hinge_and_yaw_metrics():
    """The exported per-frame diagnostics must reuse the same functions the
    official evaluation report calls, so a frame flagged here is flagged the
    same way in reports/*.json -- not a reimplementation that can drift."""
    module = _load_module()
    from pose.pose_lifter import H36M_NAMES

    n = len(H36M_NAMES)
    reference = np.zeros((1, n, 3), dtype=np.float32)
    reference[0, H36M_NAMES.index("left_shoulder")] = [0, 0, 0]
    reference[0, H36M_NAMES.index("left_elbow")] = [0, -0.1, 0.3]
    reference[0, H36M_NAMES.index("left_wrist")] = [0, 0, 0.6]
    estimate = reference.copy()
    # Flip the elbow bend to the opposite side of the shoulder-wrist axis.
    estimate[0, H36M_NAMES.index("left_elbow")] = [0, 0.1, 0.3]
    valid = np.ones((1, n), dtype=bool)
    metadata = [{"action": "seq_a"}]

    diagnostics = module._frame_diagnostics(estimate, reference, valid, metadata)

    assert len(diagnostics) == 1
    row = diagnostics[0]
    assert row["global_index"] == 0
    assert row["action"] == "seq_a"
    assert row["flipped"] is True
    assert row["worst_hinge_deg"] == pytest.approx(180.0, abs=1e-3)


def test_select_picks_finds_worst_hinge_worst_yaw_and_mid_per_action():
    module = _load_module()
    rows = [
        _diagnostic(0, "seq_a", worst_hinge_deg=10.0, flipped=False, yaw_error_deg=5.0),
        _diagnostic(1, "seq_a", worst_hinge_deg=90.0, flipped=True, yaw_error_deg=40.0),
        _diagnostic(2, "seq_a", worst_hinge_deg=30.0, flipped=False, yaw_error_deg=60.0),
        _diagnostic(3, "seq_b", worst_hinge_deg=5.0, flipped=False, yaw_error_deg=1.0),
    ]

    picks = module._select_picks(rows, actions=None)

    assert set(picks) == {"seq_a", "seq_b"}
    assert picks["seq_a"]["worst_hinge"]["global_index"] == 1
    assert picks["seq_a"]["worst_yaw"]["global_index"] == 2
    assert picks["seq_a"]["mid"]["global_index"] == 2  # median of [10, 30, 90] by hinge error
    assert picks["seq_b"]["worst_hinge"]["global_index"] == 3


def test_select_picks_filters_to_requested_actions():
    module = _load_module()
    rows = [
        _diagnostic(0, "seq_a", worst_hinge_deg=10.0, flipped=False, yaw_error_deg=5.0),
        _diagnostic(1, "seq_b", worst_hinge_deg=5.0, flipped=False, yaw_error_deg=1.0),
    ]

    picks = module._select_picks(rows, actions=["seq_b"])

    assert set(picks) == {"seq_b"}


def test_select_picks_falls_back_to_worst_hinge_when_yaw_is_unavailable():
    module = _load_module()
    rows = [_diagnostic(0, "seq_a", worst_hinge_deg=12.0, flipped=False, yaw_error_deg=None)]

    picks = module._select_picks(rows, actions=None)

    assert picks["seq_a"]["worst_yaw"]["global_index"] == 0


def test_build_export_writes_character_points_and_deduplicates_shared_frames():
    module = _load_module()
    from pose.pose_lifter import H36M_NAMES

    n = len(H36M_NAMES)
    targets = np.arange(2 * n * 3, dtype=np.float32).reshape(2, n, 3)
    prediction = targets + 100.0
    shared = _diagnostic(0, "seq_a", 20.0, False, 3.0)
    other = _diagnostic(1, "seq_a", 5.0, False, 1.0)
    # worst_hinge and worst_yaw point at the same frame -- must not duplicate it.
    picks = {"seq_a": {"worst_hinge": shared, "worst_yaw": shared, "mid": other}}

    gt_export, pred_export, picks_export = module._build_export(picks, targets, prediction)

    assert gt_export["schema"] == module.SCHEMA
    assert [frame["frame_index"] for frame in gt_export["frames"]] == [0, 1]
    assert [frame["frame_index"] for frame in pred_export["frames"]] == [0, 1]
    gt_pelvis = gt_export["frames"][0]["character_points"]["pelvis"]
    pred_pelvis = pred_export["frames"][0]["character_points"]["pelvis"]
    assert pred_pelvis == [value + 100.0 for value in gt_pelvis]
    assert picks_export["picks"] == picks


def test_cli_writes_gt_and_pred_json_and_reports_exported_frames(tmp_path, monkeypatch, capsys):
    pytest.importorskip("torch", reason="the export CLI requires the optional training extra")
    module = _load_module()
    from pose.pose_lifter import LiftedPoseFrame, LiftedPosePoint, LiftedPoseSequence, H36M_NAMES
    from pose.pose_types import PoseFrame, PoseLandmark, PoseSequence
    from training.temporal_lifter import TrainingConfig, build_dataset, save_dataset, train

    names = set(H36M_NAMES) - {"thorax"}
    pose = PoseSequence(
        [PoseFrame(i, i / 25, {name: PoseLandmark(name, 10 + i, 20, 1.0, True) for name in names}) for i in range(6)],
        25,
    )
    target = LiftedPoseSequence(
        [LiftedPoseFrame(i, i / 25, {name: LiftedPosePoint(name, (i / 10, 0, 0), 1.0, 0.0) for name in names})
         for i in range(6)],
        25,
    )
    dataset = build_dataset(pose, target, (100, 100), "audit-smoke")
    dataset_path, checkpoint = tmp_path / "holdout.json", tmp_path / "model.pth"
    save_dataset(dataset, dataset_path)
    train(dataset, checkpoint, TrainingConfig(window=3, channels=8, epochs=1, batch_size=2))

    out_gt, out_pred, out_picks = tmp_path / "gt.json", tmp_path / "pred.json", tmp_path / "picks.json"
    monkeypatch.setattr(sys, "argv", [
        "export_lifter_audit_frames.py",
        "--checkpoint", str(checkpoint), "--holdout", str(dataset_path), "--device", "cpu",
        "--out-gt", str(out_gt), "--out-pred", str(out_pred), "--out-picks", str(out_picks),
    ])

    assert module.main() == 0

    gt_export = json.loads(out_gt.read_text())
    pred_export = json.loads(out_pred.read_text())
    picks_export = json.loads(out_picks.read_text())
    assert gt_export["schema"] == module.SCHEMA
    assert len(gt_export["frames"]) == len(pred_export["frames"])
    assert set(picks_export["picks"]) == {"audit-smoke"}

    summary = json.loads(capsys.readouterr().out)
    assert summary["out_gt"] == str(out_gt)
    assert summary["exported_frame_count"] == len(gt_export["frames"])
