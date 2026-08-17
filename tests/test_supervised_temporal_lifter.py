import pytest

pytest.importorskip("torch", reason="supervised temporal lifter tests require the optional training extra")

from pose.pose_lifter import LiftedPoseFrame, LiftedPosePoint, LiftedPoseSequence, H36M_NAMES
from pose.pose_types import PoseFrame, PoseLandmark, PoseSequence
from training.temporal_lifter import TrainingConfig, _augment_inputs, _rank_shard, build_dataset, combine_datasets, evaluate, infer, load_dataset, preflight, save_dataset, train


def _pose(index):
    names = set(H36M_NAMES) - {"thorax"}
    return PoseFrame(index, index / 25, {name: PoseLandmark(name, 10 + index, 20, 1.0, True) for name in names})


def _target(index):
    names = set(H36M_NAMES) - {"thorax"}
    return LiftedPoseFrame(index, index / 25, {name: LiftedPosePoint(name, (index / 10, 0, 0), 1.0, 0.0) for name in names})


def test_supervised_dataset_and_training_smoke(tmp_path):
    pose = PoseSequence([_pose(i) for i in range(4)], 25)
    target = LiftedPoseSequence([_target(i) for i in range(4)], 25)
    dataset = build_dataset(pose, target, (100, 100), "licensed-smoke")
    assert len(dataset["frames"]) == 4
    path, checkpoint = tmp_path / "dataset.json", tmp_path / "model.pth"
    save_dataset(dataset, path)
    report = train(load_dataset(path), checkpoint, TrainingConfig(window=3, channels=8, epochs=1, batch_size=2))
    assert report["frame_count"] == 4
    assert evaluate(load_dataset(path), checkpoint)["frame_count"] == 4
    assert infer(pose, checkpoint, (100, 100)).backend == "animcv_supervised_temporal_lifter_v1"
    assert report["parallelism"]["mode"] == "single_gpu"


def test_rank_shards_pad_only_to_equalize_ddp_steps():
    import torch
    indices = torch.arange(5)
    shards = [_rank_shard(torch, indices, rank, 2).tolist() for rank in range(2)]
    assert shards == [[0, 1, 2], [3, 4, 0]]


def test_combined_dataset_keeps_windows_inside_each_sequence():
    first = build_dataset(PoseSequence([_pose(i) for i in range(3)], 25), LiftedPoseSequence([_target(i) for i in range(3)], 25), (100, 100), "first")
    second = build_dataset(PoseSequence([_pose(i + 10) for i in range(3)], 25), LiftedPoseSequence([_target(i + 10) for i in range(3)], 25), (100, 100), "second")
    combined = combine_datasets([first, second])
    assert [item["sequence_id"] for item in combined["sequences"]] == ["first", "second"]
    assert all(frame["target_valid"][0] for frame in combined["frames"])


def test_invalid_joint_is_masked_from_supervised_dataset():
    pose = PoseSequence([_pose(0)], 25)
    target = _target(0)
    points = dict(target.points)
    points["left_wrist"] = LiftedPosePoint("left_wrist", (99, 99, 99), 1.0, 0.0, observation_valid=False)
    dataset = build_dataset(pose, LiftedPoseSequence([LiftedPoseFrame(0, 0.0, points)], 25), (100, 100), "masked")
    wrist = H36M_NAMES.index("left_wrist")
    assert not dataset["frames"][0]["target_valid"][wrist]


def test_combining_rejects_wrong_declared_split():
    dataset = build_dataset(PoseSequence([_pose(i) for i in range(3)], 25), LiftedPoseSequence([_target(i) for i in range(3)], 25), (100, 100), "train")
    dataset["source"] = {"split": "holdout"}
    with pytest.raises(ValueError, match="expected 'train'"):
        combine_datasets([dataset], expected_split="train")


def test_training_preflight_reports_cpu_runtime():
    report = preflight("cpu")
    assert report["passed"]
    assert report["requested_device"] == "cpu"


def test_input_augmentation_drops_observations_but_keeps_tensor_shape():
    import torch

    values = torch.ones((2, 17, 3), dtype=torch.float32)
    config = TrainingConfig(window=3, channels=8, epochs=1, batch_size=2, input_dropout_probability=0.999)

    augmented = _augment_inputs(torch, values, config, torch.Generator().manual_seed(7))

    assert augmented.shape == values.shape
    assert (augmented[..., 2] == 0).any()
    assert (augmented[..., :2][augmented[..., 2] == 0] == 0).all()


def test_training_can_initialize_from_compatible_checkpoint(tmp_path):
    pose = PoseSequence([_pose(i) for i in range(4)], 25)
    target = LiftedPoseSequence([_target(i) for i in range(4)], 25)
    dataset = build_dataset(pose, target, (100, 100), "init-smoke")
    first, second = tmp_path / "first.pth", tmp_path / "second.pth"
    train(dataset, first, TrainingConfig(window=3, channels=8, epochs=1, batch_size=2))

    report = train(dataset, second, TrainingConfig(
        window=3, channels=8, epochs=1, batch_size=2, init_checkpoint=str(first),
    ))

    assert report["initialization"] == {"mode": "checkpoint", "checkpoint": str(first)}
