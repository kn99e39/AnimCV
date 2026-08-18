import numpy as np
import pytest

pytest.importorskip("torch", reason="supervised temporal lifter tests require the optional training extra")

from pose.pose_lifter import LiftedPoseFrame, LiftedPosePoint, LiftedPoseSequence, H36M_NAMES
from pose.pose_types import PoseFrame, PoseLandmark, PoseSequence
from training.temporal_lifter import BONES, H36M_NAMES, HINGE_CHAINS, TrainingConfig, _augment_inputs, _hinge_flip_loss, _hinge_loss, _model, _normalize_inputs, _rank_shard, _source_balanced_permutation, _supervision_loss, _vector_loss, _yaw_axis_loss, _yaw_tail_loss, build_dataset, combine_datasets, evaluate, infer, load_dataset, preflight, save_dataset, train


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


def test_training_seed_reproduces_model_initialization_and_report(tmp_path):
    import torch

    pose = PoseSequence([_pose(i) for i in range(4)], 25)
    target = LiftedPoseSequence([_target(i) for i in range(4)], 25)
    dataset = build_dataset(pose, target, (100, 100), "seed-smoke")
    config = TrainingConfig(window=3, channels=8, epochs=1, batch_size=2, seed=29)
    first, second = tmp_path / "first.pth", tmp_path / "second.pth"
    first_report, second_report = train(dataset, first, config), train(dataset, second, config)
    first_state = torch.load(first, weights_only=True)["state_dict"]
    second_state = torch.load(second, weights_only=True)["state_dict"]

    assert first_report["reproducibility"] == {"training_seed": 29, "model_initialization": "torch.manual_seed"}
    assert first_report["training_mpjpe_mm"] == pytest.approx(second_report["training_mpjpe_mm"])
    assert first_state.keys() == second_state.keys()
    assert all(torch.equal(first_state[key], second_state[key]) for key in first_state)


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


def test_pelvis_torso_coordinate_contract_removes_translation_and_scale():
    values = np.array([[[100.0, 200.0, 1.0]] * 17], dtype="float32")
    values[0, H36M_NAMES.index("thorax"), :2] = (100.0, 300.0)
    values[0, H36M_NAMES.index("left_wrist"), :2] = (150.0, 250.0)
    values[0, H36M_NAMES.index("right_wrist")] = 0.0

    normalized = _normalize_inputs(values, "pelvis_torso_v1")

    assert normalized[0, 0, :2].tolist() == [0.0, 0.0]
    assert normalized[0, H36M_NAMES.index("thorax"), :2].tolist() == [0.0, 1.0]
    assert normalized[0, H36M_NAMES.index("left_wrist"), :2].tolist() == [0.5, 0.5]
    assert normalized[0, H36M_NAMES.index("right_wrist")].tolist() == [0.0, 0.0, 0.0]


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


def test_dilated_model_has_full_window_receptive_field():
    import torch
    from torch import nn

    model = _model(nn, 8, "dilated_tcn_v1")

    assert model.receptive_field >= 81
    assert model(torch.zeros((2, 81, 17, 3))).shape == (2, 17, 3)


def test_source_balanced_sampling_upsamples_small_source():
    import torch

    indices = torch.arange(102)
    source_ids = torch.tensor([0] * 100 + [1] * 2)
    sampled = _source_balanced_permutation(torch, indices, source_ids, torch.Generator().manual_seed(4))

    assert len(sampled) == len(indices)
    assert (source_ids[sampled] == 0).sum() == 51
    assert (source_ids[sampled] == 1).sum() == 51


def test_camera_and_temporal_augmentations_preserve_missing_joint_contract():
    import torch

    values = torch.ones((15, 17, 3), dtype=torch.float32)
    values[:, 0] = 0
    config = TrainingConfig(
        window=3, channels=8, epochs=1, batch_size=2, input_global_scale_std=.1,
        input_translation_std=.1, input_rotation_degrees=10, temporal_occlusion_probability=.8,
        temporal_occlusion_frames=3,
    )
    augmented = _augment_inputs(torch, values, config, torch.Generator().manual_seed(9), [(0, 7), (7, 15)])

    assert augmented.shape == values.shape
    assert (augmented[:, 0] == 0).all()
    assert (augmented[..., 2] == 0).any()
    assert (augmented[..., :2][augmented[..., 2] == 0] == 0).all()


def test_structural_loss_is_zero_for_matching_pose_and_positive_for_wrong_hinge():
    import torch

    target = torch.zeros((1, 17, 3))
    for name, point in {"left_shoulder": (-1, 0, 0), "left_elbow": (0, 1, 0), "left_wrist": (1, 0, 0)}.items():
        target[0, H36M_NAMES.index(name)] = torch.tensor(point)
    valid = torch.zeros((1, 17, 1))
    valid[0, [H36M_NAMES.index(name) for name in ("left_shoulder", "left_elbow", "left_wrist")]] = 1
    config = TrainingConfig(window=3, channels=8, epochs=1, batch_size=1, hinge_loss_weight=1.0)

    matching = _supervision_loss(torch, target, target, valid, config)
    wrong = target.clone()
    wrong[0, H36M_NAMES.index("left_elbow"), 1] = -1

    assert matching == pytest.approx(0)
    assert _supervision_loss(torch, wrong, target, valid, config) > matching


def test_yaw_axis_loss_matches_bilateral_xy_orientation_not_torso_size():
    import torch

    target = torch.zeros((1, 17, 3))
    left, right = (H36M_NAMES.index(name) for name in ("left_shoulder", "right_shoulder"))
    target[0, left, :2] = torch.tensor((-1.0, 0.0))
    target[0, right, :2] = torch.tensor((1.0, 0.0))
    valid = torch.zeros((1, 17), dtype=torch.bool)
    valid[0, [left, right]] = True

    same_heading_different_width = target.clone()
    same_heading_different_width[0, left, 0] = -3.0
    same_heading_different_width[0, right, 0] = 3.0
    reversed_heading = target.clone()
    reversed_heading[0, left, 0], reversed_heading[0, right, 0] = 1.0, -1.0

    assert _yaw_axis_loss(torch, same_heading_different_width, target, valid) == pytest.approx(0)
    assert _yaw_axis_loss(torch, reversed_heading, target, valid) == pytest.approx(2.0)
    assert _yaw_tail_loss(torch, same_heading_different_width, target, valid) == pytest.approx(0)
    assert _yaw_tail_loss(torch, reversed_heading, target, valid) == pytest.approx(2.0)


def test_hinge_flip_loss_only_penalizes_reversed_bend_direction():
    import torch

    target = torch.zeros((1, 17, 3))
    names = ("left_shoulder", "left_elbow", "left_wrist")
    for name, point in zip(names, ((-1, 0, 0), (0, 1, 0), (1, 0, 0))):
        target[0, H36M_NAMES.index(name)] = torch.tensor(point)
    valid = torch.zeros((1, 17), dtype=torch.bool)
    valid[0, [H36M_NAMES.index(name) for name in names]] = True
    reversed_bend = target.clone()
    reversed_bend[0, H36M_NAMES.index("left_elbow"), 1] = -1

    assert _hinge_flip_loss(torch, target, target, valid) == pytest.approx(0)
    assert _hinge_flip_loss(torch, reversed_bend, target, valid) == pytest.approx(1.0)


def test_vectorized_structural_losses_match_per_chain_reference_reduction():
    import torch

    torch.manual_seed(7)
    prediction, target = torch.randn((9, 17, 3)), torch.randn((9, 17, 3))
    valid = torch.rand((9, 17)) > 0.3

    def reference_vector(pairs, transform):
        values = []
        for first, second in pairs:
            first_index, second_index = H36M_NAMES.index(first), H36M_NAMES.index(second)
            pair_valid = valid[:, first_index] & valid[:, second_index]
            if pair_valid.any():
                values.append(torch.nn.functional.smooth_l1_loss(
                    transform(prediction[pair_valid, first_index], prediction[pair_valid, second_index]),
                    transform(target[pair_valid, first_index], target[pair_valid, second_index]), reduction="mean",
                ))
        return torch.stack(values).mean() if values else prediction.new_zeros(())

    def reference_hinge():
        values = []
        for proximal, joint, distal in HINGE_CHAINS:
            indices = [H36M_NAMES.index(name) for name in (proximal, joint, distal)]
            chain_valid = valid[:, indices[0]] & valid[:, indices[1]] & valid[:, indices[2]]
            if chain_valid.any():
                values.append(torch.nn.functional.smooth_l1_loss(
                    _bend(prediction[chain_valid, indices[0]], prediction[chain_valid, indices[1]], prediction[chain_valid, indices[2]]),
                    _bend(target[chain_valid, indices[0]], target[chain_valid, indices[1]], target[chain_valid, indices[2]]),
                    reduction="mean",
                ))
        return torch.stack(values).mean() if values else prediction.new_zeros(())

    def _bend(proximal, joint, distal):
        axis = distal - proximal
        projection = (joint - proximal).mul(axis).sum(-1, keepdim=True) / axis.square().sum(-1, keepdim=True).clamp_min(1e-8)
        return joint - (proximal + projection * axis)

    assert _vector_loss(torch, prediction, target, valid, BONES, lambda first, second: first - second) == pytest.approx(
        reference_vector(BONES, lambda first, second: first - second)
    )
    assert _hinge_loss(torch, prediction, target, valid) == pytest.approx(reference_hinge())
