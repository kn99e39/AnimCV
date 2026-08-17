import json

import numpy as np

from pose import amass_adapter
from scripts.prepare_amass import _camera_views, _repair_cached_sequence_id, _source_metadata_error, _stratified_sources


def test_project_centres_forward_point_and_rejects_behind_camera():
    joints = np.array([[[0.0, 2.0, 0.0], [0.0, -1.0, 0.0]]], dtype=np.float32)

    pixels, visible = amass_adapter._project(joints, 1920, 1080, 1500.0)

    assert pixels[0, 0].tolist() == [960.0, 540.0]
    assert visible[0].tolist() == [True, False]


def test_import_amass_downsamples_and_emits_supervised_contract(tmp_path, monkeypatch):
    source = tmp_path / "motion.npz"
    np.savez(
        source,
        poses=np.zeros((8, 156), dtype=np.float32),
        trans=np.zeros((8, 3), dtype=np.float32),
        betas=np.zeros(16, dtype=np.float32),
        gender=np.array("male"),
        mocap_framerate=np.array(60.0),
    )

    def fake_evaluate(poses, trans, betas, gender, root, device):
        assert len(poses) == 3  # 60 -> 30 FPS, then max_frames=3.
        assert gender == "male"
        return np.zeros((len(poses), 24, 3), dtype=np.float32)

    monkeypatch.setattr(amass_adapter, "_evaluate_smplh", fake_evaluate)
    output = tmp_path / "prepared.json"
    report = amass_adapter.import_amass_motion(
        source, output, body_model_root=tmp_path, split="train",
        max_frames=3, target_fps=30.0,
    )
    dataset = json.loads(output.read_text())

    assert report["frame_count"] == 3
    assert dataset["schema"] == "animcv_supervised_3d_lifter_dataset_v2"
    assert dataset["source_fps"] == 30.0
    assert dataset["source"]["input_kind"] == "synthetic_virtual_camera_gt_2d"
    assert len(dataset["sequences"]) == 1


def test_amass_selection_round_robins_across_subsets(tmp_path):
    raw = tmp_path / "raw"
    sources = [
        raw / "A" / "a1.npz", raw / "A" / "a2.npz", raw / "A" / "a3.npz",
        raw / "B" / "b1.npz", raw / "B" / "b2.npz",
    ]

    selected = _stratified_sources(sources, raw, 4)

    assert [path.name for path in selected] == ["a1.npz", "b1.npz", "a2.npz", "b2.npz"]


def test_amass_relative_source_identifier_prevents_same_stem_collision():
    first = amass_adapter.amass_sequence_id("CMU/subject/walk", 0.0)
    second = amass_adapter.amass_sequence_id("KIT/subject/walk.npz", 0.0)

    assert first == "amass:CMU/subject/walk:yaw0"
    assert second == "amass:KIT/subject/walk:yaw0"
    assert first != second


def test_amass_source_metadata_rejects_shape_auxiliary(tmp_path):
    shape = tmp_path / "shape.npz"
    np.savez(shape, betas=np.zeros(16, dtype=np.float32), gender=np.array("male"))

    error = _source_metadata_error(shape)

    assert error is not None
    assert "mocap_framerate" in error
    assert "poses" in error
    assert "trans" in error


def test_cached_amass_sequence_id_is_repaired():
    dataset = {"sequence_id": "amass:walk:yaw0", "sequences": [{"sequence_id": "amass:walk:yaw0"}]}

    changed = _repair_cached_sequence_id(dataset, "amass:CMU/subject/walk:yaw0")

    assert changed is True
    assert dataset["sequence_id"] == "amass:CMU/subject/walk:yaw0"
    assert dataset["sequences"][0]["sequence_id"] == "amass:CMU/subject/walk:yaw0"


def test_amass_camera_view_parser_is_explicit_and_non_cartesian():
    views = _camera_views("0,0,4.5,1500;-45,10,5,1300")

    assert views == [(0.0, 0.0, 4.5, 1500.0), (-45.0, 10.0, 5.0, 1300.0)]
