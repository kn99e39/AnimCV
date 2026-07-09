from mediaio.frame_sequence import Frame, FrameSequence, FrameSequenceMetadata
from pose.pose_types import PoseFrame, PoseLandmark, PoseSequence


def test_frame_sequence_metadata_roundtrip():
    frames = [
        Frame(index=0, timestamp=0.0, image=None, width=1920, height=1080),
        Frame(index=1, timestamp=1 / 24, image=None, width=1920, height=1080),
    ]
    sequence = FrameSequence(
        frames=frames, fps=24.0, width=1920, height=1080, source_path="reference.mp4"
    )

    metadata = FrameSequenceMetadata.from_sequence(sequence)
    restored = FrameSequenceMetadata.from_dict(metadata.to_dict())

    assert restored == metadata
    assert restored.frame_count == 2


def test_pose_sequence_roundtrip():
    landmark = PoseLandmark(name="left_wrist", x=0.4, y=0.6, confidence=0.95, visible=True)
    frame = PoseFrame(frame_index=0, timestamp=0.0, landmarks={"left_wrist": landmark})
    sequence = PoseSequence(frames=[frame], source_fps=24.0, landmark_schema="canonical_v1")

    restored = PoseSequence.from_dict(sequence.to_dict())

    assert restored == sequence
