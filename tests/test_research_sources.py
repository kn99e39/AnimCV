from pose.pose_lifter import LiftedPoseFrame, LiftedPosePoint, LiftedPoseSequence
from pose.pose_types import PoseFrame, PoseLandmark, PoseSequence
from training.research_sources import import_mpi3dhp_dataset


def test_mpi_source_adapter_preserves_provenance_and_supervision(monkeypatch, tmp_path):
    pose = PoseSequence([PoseFrame(0, 0.0, {"pelvis": PoseLandmark("pelvis", 10, 10, 1, True), "neck": PoseLandmark("neck", 11, 11, 1, True)})], 25)
    points = {"pelvis": LiftedPosePoint("pelvis", (0, 0, 0), 1, 0), "neck": LiftedPosePoint("neck", (0, 0, 1), 1, 0)}
    target = LiftedPoseSequence([LiftedPoseFrame(0, 0.0, points)], 25)
    monkeypatch.setattr("training.research_sources.load_mpi3dhp_ground_truth", lambda *args, **kwargs: (pose, target))
    # The generic dataset builder requires all canonical joints; use a targeted fake to isolate source provenance.
    monkeypatch.setattr("training.research_sources.build_dataset", lambda *args: {
        "schema": "animcv_supervised_3d_lifter_dataset_v2", "joint_names": ["pelvis"],
        "frames": [{"target_valid": [True]}], "sequences": [{"frames": [{"target_valid": [True]}]}],
    })
    out = tmp_path / "mpi.json"
    report = import_mpi3dhp_dataset("annot.mat", 0, (100, 100), "seq", out, split="holdout")
    assert report == {"sequence_id": "seq", "split": "holdout", "frame_count": 1, "valid_joint_count": 1}
