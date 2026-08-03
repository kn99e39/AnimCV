from pose.audit_visualization import render_audit_views
from pose.root_motion import RootMotionFrame, RootMotionSequence


def test_audit_view_renderer_writes_three_view_svg(tmp_path):
    root = RootMotionSequence(frames=[RootMotionFrame(0, 0.0, 0.0, (0, 1, 0), (1, 0, 0), 1.0, None, {
        "pelvis": (0, 0, 0), "left_hip": (-1, 0, 0), "right_hip": (1, 0, 0), "spine": (0, 0, 1),
    })])
    output = tmp_path / "audit.svg"
    render_audit_views(root, output)
    text = output.read_text()
    assert "정면 X/Z" in text and "측면 Y/Z" in text and "상단 X/Y" in text
