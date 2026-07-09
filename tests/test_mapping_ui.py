import io

import pytest

from ui.mapping_ui import MappingCommandError, parse_mapping_line, run_interactive_mapping


def test_parse_direction_command():
    entry = parse_mapping_line("upper_arm.L", "direction left_shoulder left_elbow +Y")

    assert entry.target_bone == "upper_arm.L"
    assert entry.source_type == "landmark"
    assert entry.mapping_mode == "direction"
    assert entry.source_names == ["left_shoulder", "left_elbow"]
    assert entry.axis_hint == "+Y"


def test_parse_direction_command_without_axis_hint():
    entry = parse_mapping_line("forearm.L", "direction left_elbow left_wrist")

    assert entry.axis_hint is None


def test_parse_landmark_command():
    entry = parse_mapping_line("head", "landmark head")

    assert entry.mapping_mode == "landmark"
    assert entry.source_names == ["head"]


def test_parse_custom_point_command():
    entry = parse_mapping_line("wing_tip.L", "custom_point custom_point_03")

    assert entry.source_type == "custom_point"
    assert entry.mapping_mode == "point"
    assert entry.source_names == ["custom_point_03"]


@pytest.mark.parametrize("line", ["", "skip", "SKIP", "   "])
def test_parse_skip_returns_none(line):
    assert parse_mapping_line("head", line) is None


def test_parse_unknown_command_raises():
    with pytest.raises(MappingCommandError):
        parse_mapping_line("head", "teleport somewhere")


def test_parse_direction_with_wrong_arg_count_raises():
    with pytest.raises(MappingCommandError):
        parse_mapping_line("upper_arm.L", "direction only_one_source")


def test_run_interactive_mapping_covers_acceptance_bones():
    bone_names = ["upper_arm.L", "forearm.L", "thigh.L", "shin.L", "head", "hips"]
    script = "\n".join(
        [
            "direction left_shoulder left_elbow",  # upper_arm.L
            "direction left_elbow left_wrist",  # forearm.L
            "direction left_hip left_knee",  # thigh.L
            "direction left_knee left_ankle",  # shin.L
            "landmark head",  # head
            "landmark pelvis",  # hips
        ]
    )
    input_stream = io.StringIO(script + "\n")
    output_stream = io.StringIO()

    profile = run_interactive_mapping(
        bone_names=bone_names,
        rig_id="character_01",
        created_from_frame=3,
        input_stream=input_stream,
        output_stream=output_stream,
    )

    mapped_bones = {entry.target_bone for entry in profile.entries}
    assert mapped_bones == set(bone_names)
    assert profile.rig_id == "character_01"
    assert profile.created_from_frame == 3


def test_run_interactive_mapping_allows_skipping_bones():
    input_stream = io.StringIO("skip\nlandmark head\n")

    profile = run_interactive_mapping(
        bone_names=["unmappable_prop", "head"],
        rig_id="character_01",
        output_stream=io.StringIO(),
        input_stream=input_stream,
    )

    assert [entry.target_bone for entry in profile.entries] == ["head"]


def test_run_interactive_mapping_stops_on_done():
    input_stream = io.StringIO("landmark head\ndone\nlandmark pelvis\n")

    profile = run_interactive_mapping(
        bone_names=["head", "hips", "spine"],
        rig_id="character_01",
        output_stream=io.StringIO(),
        input_stream=input_stream,
    )

    assert [entry.target_bone for entry in profile.entries] == ["head"]


def test_run_interactive_mapping_stops_on_eof():
    input_stream = io.StringIO("landmark head\n")

    profile = run_interactive_mapping(
        bone_names=["head", "hips"],
        rig_id="character_01",
        output_stream=io.StringIO(),
        input_stream=input_stream,
    )

    assert [entry.target_bone for entry in profile.entries] == ["head"]


def test_run_interactive_mapping_reports_bad_command_and_continues():
    input_stream = io.StringIO("teleport somewhere\nlandmark head\n")
    output_stream = io.StringIO()

    profile = run_interactive_mapping(
        bone_names=["hips", "head"],
        rig_id="character_01",
        output_stream=output_stream,
        input_stream=input_stream,
    )

    assert [entry.target_bone for entry in profile.entries] == ["head"]
    assert "unrecognized mapping command" in output_stream.getvalue()
