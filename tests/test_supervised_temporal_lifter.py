from pose.pose_lifter import LiftedPoseFrame, LiftedPosePoint, LiftedPoseSequence, H36M_NAMES
from pose.pose_types import PoseFrame, PoseLandmark, PoseSequence
from training.temporal_lifter import TrainingConfig, build_dataset, evaluate, infer, load_dataset, save_dataset, train


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
