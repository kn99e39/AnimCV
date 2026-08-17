import pickle

import numpy as np

from pose.three_dpw_adapter import import_3dpw_dataset, load_3dpw_ground_truth
from training.temporal_lifter import load_dataset


def _sequence_payload():
    joints = np.zeros((2, 24, 3), dtype=float)
    joints[0, 1] = (1.0, 2.0, 3.0)  # SMPL left hip
    joints[0, 4] = (4.0, 5.0, 6.0)  # SMPL left knee
    poses2d = np.zeros((2, 3, 18), dtype=float)
    poses2d[:, 0, :] = np.arange(18)
    poses2d[:, 1, :] = np.arange(18) + 100
    poses2d[:, 2, :] = 1.0
    return {
        "sequence": "synthetic", "poses2d": [poses2d], "jointPositions": [joints.reshape(2, -1)],
        "cam_poses": np.repeat(np.eye(4)[None], 2, axis=0), "campose_valid": [np.array([True, False])],
        "cam_intrinsics": np.array([[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]]),
    }


def test_3dpw_camera_conversion_and_invalid_frame_filtering(tmp_path):
    annotation = tmp_path / "sequence.pkl"
    with annotation.open("wb") as handle:
        pickle.dump(_sequence_payload(), handle)

    sequence_id, pose, lifted, image_size = load_3dpw_ground_truth(annotation)[0]

    assert sequence_id == "3dpw:synthetic:actor0"
    assert image_size == (1920, 1080)
    assert pose.frames[0].landmarks["left_hip"].x == 11.0
    assert pose.frames[1].landmarks == {}
    # Identity OpenCV camera -> AnimCV (x, z, -y), then pelvis-relative.
    assert lifted.frames[0].points["left_hip"].position == (1.0, 3.0, -2.0)

    output = tmp_path / "prepared.json"
    report = import_3dpw_dataset(annotation, output, split="train")
    dataset = load_dataset(output)
    assert report["frame_count"] == 1
    assert len(dataset["sequences"]) == 1
    assert dataset["source"]["dataset"] == "3DPW"
