import numpy as np
import pytest

from pose.pose_lifter import H36M_NAMES
from training.temporal_lifter import _evaluation_report


def test_evaluation_reports_pa_yaw_hinge_and_provenance_slices():
    points = {
        "pelvis": (0, 0, 0), "left_hip": (-.2, 0, 0), "right_hip": (.2, 0, 0),
        "left_knee": (-.3, .1, -.5), "right_knee": (.3, .1, -.5),
        "left_ankle": (-.2, 0, -1), "right_ankle": (.2, 0, -1),
        "spine": (0, 0, .3), "thorax": (0, 0, .6), "neck": (0, 0, .8), "head": (0, 0, 1),
        "left_shoulder": (-.3, 0, .7), "left_elbow": (-.5, .15, .55), "left_wrist": (-.7, 0, .4),
        "right_shoulder": (.3, 0, .7), "right_elbow": (.5, .15, .55), "right_wrist": (.7, 0, .4),
    }
    target = np.asarray([[points[name] for name in H36M_NAMES]], dtype=np.float32)
    report = _evaluation_report(target.copy(), target, np.ones((1, 17), dtype=bool), [{
        "source": "AMASS", "view": "yaw=45", "action": "walk",
    }])

    assert report["pa_mpjpe_mm"] == pytest.approx(0, abs=.01)
    assert report["root_yaw_mae_degrees"] == pytest.approx(0)
    assert report["hinge_flip_rate"] == pytest.approx(0)
    assert report["slices"]["source"]["AMASS"]["evaluated_frame_count"] == 1
    assert report["passed"]


def test_evaluation_detects_a_reversed_elbow_bend():
    target = np.zeros((1, 17, 3), dtype=np.float32)
    for name, value in {"left_shoulder": (-1, 0, 0), "left_elbow": (0, 1, 0), "left_wrist": (1, 0, 0)}.items():
        target[0, H36M_NAMES.index(name)] = value
    prediction = target.copy()
    prediction[0, H36M_NAMES.index("left_elbow"), 1] = -1
    valid = np.zeros((1, 17), dtype=bool)
    valid[0, [H36M_NAMES.index(name) for name in ("left_shoulder", "left_elbow", "left_wrist")]] = True
    report = _evaluation_report(prediction, target, valid, [{"source": "test", "view": None, "action": "bend"}])

    assert report["hinge_sample_count"] == 1
    assert report["hinge_flip_rate"] == pytest.approx(1)
