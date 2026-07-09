import io

import pytest

from ui.mapping_ui import (
    MappingCommandError,
    parse_ik_chain_line,
    parse_mapping_line,
    run_interactive_mapping,
)


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


def test_parse_ik_chain_line_full():
    chain = parse_ik_chain_line(
        "upper_arm.L forearm.L hand.L left_shoulder left_elbow left_wrist +Y +X", 1
    )

    assert chain.name == "ik_chain_1"
    assert chain.root_bone == "upper_arm.L"
    assert chain.mid_bone == "forearm.L"
    assert chain.end_bone == "hand.L"
    assert chain.root_source == "left_shoulder"
    assert chain.mid_source == "left_elbow"
    assert chain.end_source == "left_wrist"
    assert chain.root_axis_hint == "+Y"
    assert chain.mid_axis_hint == "+X"


def test_parse_ik_chain_line_without_axis_hints():
    chain = parse_ik_chain_line(
        "upper_arm.L forearm.L hand.L left_shoulder left_elbow left_wrist", 2
    )

    assert chain.root_axis_hint is None
    assert chain.mid_axis_hint is None
    assert chain.name == "ik_chain_2"


@pytest.mark.parametrize("line", ["", "done", "DONE", "   "])
def test_parse_ik_chain_line_stop_returns_none(line):
    assert parse_ik_chain_line(line, 1) is None


def test_parse_ik_chain_line_wrong_arg_count_raises():
    with pytest.raises(MappingCommandError):
        parse_ik_chain_line("upper_arm.L forearm.L hand.L", 1)


def test_run_interactive_mapping_collects_ik_chains():
    input_stream = io.StringIO(
        "\n".join(
            [
                "done",  # skip the per-bone loop immediately
                "upper_arm.L forearm.L hand.L left_shoulder left_elbow left_wrist",
                "thigh.L shin.L foot.L left_hip left_knee left_ankle",
                "done",
            ]
        )
        + "\n"
    )

    profile = run_interactive_mapping(
        bone_names=["upper_arm.L", "forearm.L"],
        rig_id="character_01",
        output_stream=io.StringIO(),
        input_stream=input_stream,
    )

    assert profile.entries == []
    assert [chain.root_bone for chain in profile.ik_chains] == ["upper_arm.L", "thigh.L"]
    assert profile.ik_chains[0].mid_bone == "forearm.L"
    assert profile.ik_chains[1].end_source == "left_ankle"


def test_run_interactive_mapping_ik_chain_bad_line_reports_and_continues():
    input_stream = io.StringIO("done\nnot enough tokens\ndone\n")
    output_stream = io.StringIO()

    profile = run_interactive_mapping(
        bone_names=["upper_arm.L"],
        rig_id="character_01",
        output_stream=output_stream,
        input_stream=input_stream,
    )

    assert profile.ik_chains == []
    assert "ik chain needs" in output_stream.getvalue()
