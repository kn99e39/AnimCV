import pytest

from pose.dataset_3d_audit import audit_supervised_3d
from pose.pose_lifter import LiftedPoseFrame, LiftedPosePoint, LiftedPoseSequence


def _sequence(offset=0.0):
    frames = []
    for index in range(2):
        points = {name: LiftedPosePoint(name, (x + offset, y, z), 1.0, 0.0)
                  for name, (x, y, z) in {"pelvis": (0, 0, 0), "neck": (0, 0, 1), "left_wrist": (-1, 0, .5)}.items()}
        frames.append(LiftedPoseFrame(index, index / 25, points))
    return LiftedPoseSequence(frames, 25)


def test_dataset_neutral_audit_reports_mpjpe_and_pa_mpjpe():
    report = audit_supervised_3d(_sequence(1.0), _sequence())
    assert report["mpjpe_mm"] == pytest.approx(1000)
    assert report["pa_mpjpe_mm"] == pytest.approx(0)
    assert report["root_yaw_mae_degrees"] is None
