import cv2
import numpy as np

from mediaio.video_loader import VideoLoader


def _write_synthetic_video(path, fps=12.0, num_frames=24, size=(64, 48)):
    width, height = size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    for i in range(num_frames):
        frame = np.full((height, width, 3), fill_value=i % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_load_video_reads_all_frames_at_source_fps(tmp_path):
    video_path = tmp_path / "clip.mp4"
    _write_synthetic_video(video_path, fps=12.0, num_frames=24)

    sequence = VideoLoader().load_video(str(video_path))

    assert sequence.fps == 12.0
    assert sequence.width == 64
    assert sequence.height == 48
    assert len(sequence.frames) == 24
    assert sequence.frames[0].index == 0
    assert sequence.frames[-1].index == len(sequence.frames) - 1


def test_load_video_downsamples_to_target_fps(tmp_path):
    video_path = tmp_path / "clip.mp4"
    _write_synthetic_video(video_path, fps=24.0, num_frames=48)

    sequence = VideoLoader().load_video(str(video_path), target_fps=12.0)

    assert sequence.fps == 12.0
    assert len(sequence.frames) == 24


def test_load_video_start_frame_skips_leading_frames(tmp_path):
    video_path = tmp_path / "clip.mp4"
    _write_synthetic_video(video_path, fps=12.0, num_frames=24)

    sequence = VideoLoader().load_video(str(video_path), start_frame=12)

    assert len(sequence.frames) == 12


def test_load_video_end_frame_truncates_trailing_frames(tmp_path):
    video_path = tmp_path / "clip.mp4"
    _write_synthetic_video(video_path, fps=12.0, num_frames=24)

    sequence = VideoLoader().load_video(str(video_path), end_frame=11)

    assert len(sequence.frames) == 12


def test_load_video_start_and_end_frame_together(tmp_path):
    video_path = tmp_path / "clip.mp4"
    _write_synthetic_video(video_path, fps=12.0, num_frames=24)

    sequence = VideoLoader().load_video(str(video_path), start_frame=5, end_frame=14)

    assert len(sequence.frames) == 10


def test_load_video_start_frame_after_end_frame_raises(tmp_path):
    import pytest

    video_path = tmp_path / "clip.mp4"
    _write_synthetic_video(video_path, fps=12.0, num_frames=24)

    with pytest.raises(ValueError):
        VideoLoader().load_video(str(video_path), start_frame=10, end_frame=5)


def test_load_video_start_frame_combined_with_target_fps(tmp_path):
    video_path = tmp_path / "clip.mp4"
    _write_synthetic_video(video_path, fps=24.0, num_frames=24)

    sequence = VideoLoader().load_video(str(video_path), start_frame=12, target_fps=12.0)

    # 12 remaining source frames (indices 12..23) at 24fps, downsampled to
    # 12fps (stride 2) -> every other one of those 12 frames.
    assert sequence.fps == 12.0
    assert len(sequence.frames) == 6


def test_load_video_missing_file_raises(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError):
        VideoLoader().load_video(str(tmp_path / "missing.mp4"))


def test_load_image_sequence_reads_sorted_images(tmp_path):
    for i in range(5):
        image = np.full((10, 20, 3), fill_value=i, dtype=np.uint8)
        cv2.imwrite(str(tmp_path / f"{i:04d}.png"), image)

    sequence = VideoLoader().load_image_sequence(str(tmp_path))

    assert len(sequence.frames) == 5
    assert [frame.index for frame in sequence.frames] == [0, 1, 2, 3, 4]
    assert sequence.width == 20
    assert sequence.height == 10


def test_load_image_sequence_empty_directory_raises(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError):
        VideoLoader().load_image_sequence(str(tmp_path))


def test_scrubber_reports_video_metadata(tmp_path):
    video_path = tmp_path / "clip.mp4"
    _write_synthetic_video(video_path, fps=12.0, num_frames=24, size=(64, 48))

    with VideoLoader().open_scrubber(str(video_path)) as scrubber:
        assert scrubber.frame_count == 24
        assert scrubber.fps == 12.0
        assert scrubber.width == 64
        assert scrubber.height == 48


def test_scrubber_reads_frame_at_index(tmp_path):
    video_path = tmp_path / "clip.mp4"
    _write_synthetic_video(video_path, num_frames=24)

    with VideoLoader().open_scrubber(str(video_path)) as scrubber:
        frame = scrubber.read_frame(10)
        assert frame is not None
        assert frame.shape == (48, 64, 3)


def test_scrubber_sequential_reads_advance(tmp_path):
    video_path = tmp_path / "clip.mp4"
    _write_synthetic_video(video_path, num_frames=24)

    with VideoLoader().open_scrubber(str(video_path)) as scrubber:
        first = scrubber.read_frame(5)
        second = scrubber.read_frame(6)
        assert first is not None and second is not None
        # Consecutive synthetic frames have different fill values, so the
        # scrubber genuinely advanced rather than re-returning frame 5.
        assert not np.array_equal(first, second)


def test_scrubber_out_of_range_returns_none(tmp_path):
    video_path = tmp_path / "clip.mp4"
    _write_synthetic_video(video_path, num_frames=24)

    with VideoLoader().open_scrubber(str(video_path)) as scrubber:
        assert scrubber.read_frame(9999) is None
        assert scrubber.read_frame(-1) is None


def test_scrubber_missing_file_raises(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError):
        VideoLoader().open_scrubber(str(tmp_path / "missing.mp4"))
