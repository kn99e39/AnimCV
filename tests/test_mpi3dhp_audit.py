from pose.mpi3dhp_audit import audit_mpi3dhp_2d
from pose.pose_types import PoseFrame, PoseLandmark, PoseSequence


def _sequence(offset: float) -> PoseSequence:
    landmarks = {
        "pelvis": PoseLandmark("pelvis", 100 + offset, 100, 1.0, True),
        "neck": PoseLandmark("neck", 100 + offset, 0, 1.0, True),
    }
    return PoseSequence([PoseFrame(0, 0.0, landmarks)], 25.0)


def test_mpi3dhp_audit_reports_pck_against_ground_truth():
    report = audit_mpi3dhp_2d(_sequence(2), _sequence(0))
    assert report["matched_joints"] == 2
    assert report["pck_at_0_2"] == 1.0
    assert report["passed"]
