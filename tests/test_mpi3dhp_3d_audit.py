from pose.mpi3dhp_3d_audit import audit_mpi3dhp_3d
from pose.pose_lifter import LiftedPoseFrame, LiftedPosePoint, LiftedPoseSequence
from pose.root_motion import RootMotionFrame, RootMotionSequence


def _lifted(scale=1.0):
    points = {name: LiftedPosePoint(name, tuple(scale * x for x in xyz), 1.0, 0.0)
              for name, xyz in {"pelvis": (0, 0, 0), "neck": (0, 0, 1), "left_hip": (-1, 0, 0)}.items()}
    return LiftedPoseSequence([LiftedPoseFrame(0, 0.0, points)], 25.0)


def _root():
    frame = RootMotionFrame(0, 0.0, 0.0, (0, 1, 0), (1, 0, 0), 1.0, None, {})
    return RootMotionSequence([frame], 25.0)


def test_mpi3dhp_3d_audit_reports_similarity_aligned_error():
    report = audit_mpi3dhp_3d(_lifted(2), _lifted(), _root(), _root())
    assert report["matched_joints"] == 3
    assert report["pa_mpjpe_mm"] < 1e-6
    assert report["root_yaw_mae_degrees"] == 0.0
