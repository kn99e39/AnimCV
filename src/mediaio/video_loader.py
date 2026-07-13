"""OpenCV-backed video/image-sequence loading (Architecture_v2.md section 3.1).

No other module should call ``cv2.VideoCapture`` or raw OpenCV loading
code directly; everything must go through ``VideoLoader``.
"""

from __future__ import annotations

from pathlib import Path

import cv2

from mediaio.frame_sequence import Frame, FrameSequence

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class VideoLoader:
    default_image_sequence_fps: float = 24.0

    def load_video(
        self,
        path: str,
        target_fps: float | None = None,
        start_frame: int | None = None,
        end_frame: int | None = None,
    ) -> FrameSequence:
        """start_frame/end_frame (Architecture_v2.md section 1.1's "Start
        frame / end frame" optional input) are inclusive source-video
        frame indices, i.e. positions in the original video before any
        target_fps resampling -- not indices into the resampled output.
        Both are optional and independent (only a start, only an end, or
        neither)."""
        if start_frame is not None and end_frame is not None and start_frame > end_frame:
            raise ValueError(f"start_frame ({start_frame}) must not be after end_frame ({end_frame})")

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise FileNotFoundError(f"Could not open video: {path}")

        try:
            source_fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
            if source_fps <= 0:
                raise ValueError(f"Video reports an invalid fps: {path}")

            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

            range_start = start_frame or 0
            if range_start > 0:
                # CAP_PROP_POS_FRAMES seeking is only approximate on some
                # codecs (it typically snaps to the nearest keyframe
                # internally before decoding forward) -- exact for the
                # synthetic mp4v test fixtures here, close enough in
                # general for a "reference clip" trim, not frame-accurate
                # for every container/codec.
                capture.set(cv2.CAP_PROP_POS_FRAMES, range_start)

            output_fps = min(target_fps, source_fps) if target_fps else source_fps
            stride = source_fps / output_fps

            frames: list[Frame] = []
            source_index = 0
            next_sample_at = 0.0
            output_index = 0
            while True:
                if end_frame is not None and range_start + source_index > end_frame:
                    break
                ok, image = capture.read()
                if not ok:
                    break
                if source_index >= next_sample_at:
                    frames.append(
                        Frame(
                            index=output_index,
                            timestamp=output_index / output_fps,
                            image=image,
                            width=width,
                            height=height,
                        )
                    )
                    output_index += 1
                    next_sample_at += stride
                source_index += 1
        finally:
            capture.release()

        return FrameSequence(
            frames=frames, fps=output_fps, width=width, height=height, source_path=str(path)
        )

    def load_image_sequence(self, directory: str) -> FrameSequence:
        directory_path = Path(directory)
        image_paths = sorted(
            p for p in directory_path.iterdir() if p.suffix.lower() in _IMAGE_EXTENSIONS
        )
        if not image_paths:
            raise FileNotFoundError(f"No images found in: {directory}")

        fps = self.default_image_sequence_fps
        frames: list[Frame] = []
        width = height = 0
        for index, image_path in enumerate(image_paths):
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Failed to read image: {image_path}")
            height, width = image.shape[:2]
            frames.append(
                Frame(index=index, timestamp=index / fps, image=image, width=width, height=height)
            )

        return FrameSequence(
            frames=frames, fps=fps, width=width, height=height, source_path=str(directory_path)
        )
