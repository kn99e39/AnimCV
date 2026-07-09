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

    def load_video(self, path: str, target_fps: float | None = None) -> FrameSequence:
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise FileNotFoundError(f"Could not open video: {path}")

        try:
            source_fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
            if source_fps <= 0:
                raise ValueError(f"Video reports an invalid fps: {path}")

            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

            output_fps = min(target_fps, source_fps) if target_fps else source_fps
            stride = source_fps / output_fps

            frames: list[Frame] = []
            source_index = 0
            next_sample_at = 0.0
            output_index = 0
            while True:
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
