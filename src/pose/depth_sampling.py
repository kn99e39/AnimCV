"""Sample a depth map at 2D landmark pixel coordinates.

Matches the "sample depth at landmark/custom-point coordinates" step
sketched (as future work) in the v1 doc's section 13.2 for
Depth Anything V2 integration. Kept separate from pose/depth_estimator.py
so the pure sampling logic — which the ``estimate-pose`` CLI command
calls once per frame — is testable without needing a real depth model.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from pose.pose_types import PoseFrame


def sample_depth_at_landmarks(pose_frame: PoseFrame, depth_map: np.ndarray) -> PoseFrame:
    """Returns a new PoseFrame with every landmark's `z` set from the
    nearest pixel in depth_map. Landmarks that fall outside the depth
    map's bounds are left with z=None rather than raising, consistent
    with this project's "degrade gracefully" handling of missing/low-
    confidence observations elsewhere (e.g. fk_solver holding the last
    valid value)."""
    height, width = depth_map.shape[:2]
    updated_landmarks = {}
    for name, landmark in pose_frame.landmarks.items():
        px, py = int(round(landmark.x)), int(round(landmark.y))
        if 0 <= px < width and 0 <= py < height:
            z = float(depth_map[py, px])
        else:
            z = None
        updated_landmarks[name] = replace(landmark, z=z)
    return replace(pose_frame, landmarks=updated_landmarks)
