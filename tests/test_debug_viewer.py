import cv2
import numpy as np

from mediaio.frame_sequence import Frame, FrameSequence
from pose.pose_types import PoseFrame, PoseLandmark, PoseSequence
from ui.debug_viewer import render_debug_overlay


def _sample_frames() -> FrameSequence:
    frames = [
        Frame(index=i, timestamp=i / 24.0, image=np.zeros((48, 64, 3), dtype=np.uint8), width=64, height=48)
        for i in range(3)
    ]
    return FrameSequence(frames=frames, fps=24.0, width=64, height=48, source_path="clip.mp4")


def _sample_poses() -> PoseSequence:
    landmark = PoseLandmark(name="left_wrist", x=10.0, y=20.0, confidence=0.9, visible=True)
    frames = [
        PoseFrame(frame_index=i, timestamp=i / 24.0, landmarks={"left_wrist": landmark})
        for i in range(3)
    ]
    return PoseSequence(frames=frames, source_fps=24.0)


def test_render_debug_overlay_as_image_sequence(tmp_path):
    out_dir = tmp_path / "overlay"

    render_debug_overlay(_sample_frames(), _sample_poses(), str(out_dir))

    written = sorted(out_dir.glob("*.png"))
    assert len(written) == 3
    image = cv2.imread(str(written[0]))
    assert image is not None
    assert image.any()  # landmark drawing changed at least one pixel


def test_render_debug_overlay_as_video(tmp_path):
    out_path = tmp_path / "overlay.mp4"

    render_debug_overlay(_sample_frames(), _sample_poses(), str(out_path))

    assert out_path.exists()
    assert out_path.stat().st_size > 0
