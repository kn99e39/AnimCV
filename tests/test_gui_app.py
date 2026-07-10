"""Tests for ui/gui_app.py's pure logic -- the click-mapping math and
small formatters -- kept separate from the Tkinter widget class itself
so they're testable without a display, same split as
ui/mapping_ui.py's parse_mapping_line vs run_interactive_mapping."""

from __future__ import annotations

from pose.pose_types import PoseLandmark
from rig.bone_mapping import BoneMappingEntry
from ui.gui_app import describe_mapping_entry, frame_index_from_path, nearest_landmark


def _landmark(name, x, y, visible=True):
    return PoseLandmark(name=name, x=x, y=y, confidence=1.0, visible=visible)


def test_nearest_landmark_picks_closest_within_radius():
    landmarks = {
        "left_elbow": _landmark("left_elbow", 100, 100),
        "left_wrist": _landmark("left_wrist", 200, 200),
    }
    assert nearest_landmark(landmarks, 102, 101) == "left_elbow"
    assert nearest_landmark(landmarks, 198, 202) == "left_wrist"


def test_nearest_landmark_returns_none_outside_radius():
    landmarks = {"left_elbow": _landmark("left_elbow", 100, 100)}
    assert nearest_landmark(landmarks, 100 + 100, 100, max_distance=25.0) is None


def test_nearest_landmark_ignores_invisible_landmarks():
    landmarks = {"left_elbow": _landmark("left_elbow", 100, 100, visible=False)}
    assert nearest_landmark(landmarks, 100, 100) is None


def test_nearest_landmark_empty_dict():
    assert nearest_landmark({}, 0, 0) is None


def test_frame_index_from_path_parses_zero_padded_stem():
    assert frame_index_from_path("/cache/frames/00042.png") == 42


def test_frame_index_from_path_defaults_to_zero_for_non_numeric_stem():
    assert frame_index_from_path("/cache/frames/reference.png") == 0


def test_describe_mapping_entry_direction():
    entry = BoneMappingEntry(
        target_bone="upper_arm.L",
        source_type="landmark",
        source_names=["left_shoulder", "left_elbow"],
        mapping_mode="direction",
    )
    assert describe_mapping_entry(entry) == "direction left_shoulder -> left_elbow"


def test_describe_mapping_entry_landmark():
    entry = BoneMappingEntry(
        target_bone="hips", source_type="landmark", source_names=["pelvis"], mapping_mode="landmark"
    )
    assert describe_mapping_entry(entry) == "landmark pelvis"


def test_describe_mapping_entry_custom_point():
    entry = BoneMappingEntry(
        target_bone="wing.L", source_type="custom_point", source_names=["wing_tip_1"], mapping_mode="point"
    )
    assert describe_mapping_entry(entry) == "custom_point wing_tip_1"
