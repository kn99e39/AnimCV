"""Canonical semantic landmark names produced by any pose backend adapter.

Adapters (e.g. ``pose/mmpose_adapter.py``) must translate backend-specific
joint names into this project-native schema before constructing a
``PoseFrame`` (Architecture_v2.md section 3.2).
"""

from __future__ import annotations

CANONICAL_LANDMARKS: list[str] = [
    "pelvis",
    "spine",
    "neck",
    "head",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "left_hip",
    "left_knee",
    "left_ankle",
    "right_hip",
    "right_knee",
    "right_ankle",
]
