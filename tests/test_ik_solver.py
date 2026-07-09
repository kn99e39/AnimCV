import math

import pytest

from motion.motion_graph import MotionFrame, MotionGraph, MotionPoint
from retarget.axis_utils import quaternion_from_axis_angle
from retarget.ik_solver import solve_ik_chain
from rig.bone_mapping import IKChainEntry

_IDENTITY = (0.0, 0.0, 0.0, 1.0)


def _point(name, x, y, frame_index=0, confidence=0.9, visible=True):
    return MotionPoint(
        semantic_name=name,
        frame_index=frame_index,
        position_2d=(x, y),
        position_3d=None,
        confidence=confidence,
        visible=visible,
    )


def _chain(**overrides):
    defaults = dict(
        name="left_arm",
        root_bone="upper_arm.L",
        mid_bone="forearm.L",
        end_bone="hand.L",
        root_source="left_shoulder",
        mid_source="left_elbow",
        end_source="left_wrist",
    )
    defaults.update(overrides)
    return IKChainEntry(**defaults)


def _frame(frame_index, root, mid, end):
    return MotionFrame(
        frame_index=frame_index,
        timestamp=frame_index / 24.0,
        points={
            "left_shoulder": _point("left_shoulder", *root, frame_index),
            "left_elbow": _point("left_elbow", *mid, frame_index),
            "left_wrist": _point("left_wrist", *end, frame_index),
        },
    )


def test_solve_ik_chain_reference_frame_is_identity():
    # Fully extended arm at the reference frame: root=(0,0), mid=(10,0),
    # end=(20,0) -- root_length=10, mid_length=10, calibrated here.
    graph = MotionGraph(
        frames=[_frame(0, (0.0, 0.0), (10.0, 0.0), (20.0, 0.0))], tracks={}, fps=24.0
    )

    result = solve_ik_chain(graph, _chain())

    assert result["upper_arm.L"][0].rotation == pytest.approx(_IDENTITY)
    assert result["forearm.L"][0].rotation == pytest.approx(_IDENTITY)


def test_solve_ik_chain_bends_elbow_toward_closer_target():
    # Frame 0 (reference): fully extended, same as above.
    # Frame 1: wrist pulled in to (10, 10) while the observed elbow
    # stays at (10, 0) -- a right-angle bend, hand-verified via the law
    # of cosines: mid_bend goes from 0 to +90deg, root_angle stays 0
    # (the target angle 45deg exactly cancels the root offset angle
    # 45deg for this particular geometry).
    graph = MotionGraph(
        frames=[
            _frame(0, (0.0, 0.0), (10.0, 0.0), (20.0, 0.0)),
            _frame(1, (0.0, 0.0), (10.0, 0.0), (10.0, 10.0)),
        ],
        tracks={},
        fps=24.0,
    )

    result = solve_ik_chain(graph, _chain())

    assert result["upper_arm.L"][1].rotation == pytest.approx(_IDENTITY)
    assert result["forearm.L"][1].rotation == pytest.approx(
        quaternion_from_axis_angle((0.0, 0.0, 1.0), math.pi / 2)
    )


def test_solve_ik_chain_bend_side_follows_observed_elbow_position():
    # Mirror image of the bend test above: the elbow is observed on the
    # opposite side of the root->target line, so the bend should come
    # out with the opposite sign.
    graph = MotionGraph(
        frames=[
            _frame(0, (0.0, 0.0), (10.0, 0.0), (20.0, 0.0)),
            _frame(1, (0.0, 0.0), (0.0, 10.0), (10.0, 10.0)),
        ],
        tracks={},
        fps=24.0,
    )

    result = solve_ik_chain(graph, _chain())

    assert result["forearm.L"][1].rotation == pytest.approx(
        quaternion_from_axis_angle((0.0, 0.0, 1.0), -math.pi / 2)
    )


def test_solve_ik_chain_clamps_unreachable_target():
    # end pulled far beyond max_reach (root_length + mid_length = 20);
    # the solver should clamp rather than raise or produce NaN.
    graph = MotionGraph(
        frames=[
            _frame(0, (0.0, 0.0), (10.0, 0.0), (20.0, 0.0)),
            _frame(1, (0.0, 0.0), (10.0, 0.0), (1000.0, 0.0)),
        ],
        tracks={},
        fps=24.0,
    )

    result = solve_ik_chain(graph, _chain())

    root_rotation = result["upper_arm.L"][1].rotation
    mid_rotation = result["forearm.L"][1].rotation
    assert all(math.isfinite(c) for c in root_rotation)
    assert all(math.isfinite(c) for c in mid_rotation)
    # clamped to max reach == fully extended == same as the reference frame
    assert mid_rotation == pytest.approx(_IDENTITY)


def test_solve_ik_chain_holds_last_rotation_when_a_point_is_lost():
    frames = [
        _frame(0, (0.0, 0.0), (10.0, 0.0), (20.0, 0.0)),
        _frame(1, (0.0, 0.0), (10.0, 0.0), (10.0, 10.0)),
    ]
    # frame 2: wrist goes missing entirely
    frames.append(
        MotionFrame(
            frame_index=2,
            timestamp=2 / 24.0,
            points={
                "left_shoulder": _point("left_shoulder", 0.0, 0.0, 2),
                "left_elbow": _point("left_elbow", 10.0, 0.0, 2),
            },
        )
    )
    graph = MotionGraph(frames=frames, tracks={}, fps=24.0)

    result = solve_ik_chain(graph, _chain())

    assert result["upper_arm.L"][2].rotation == result["upper_arm.L"][1].rotation
    assert result["forearm.L"][2].rotation == result["forearm.L"][1].rotation
    assert result["upper_arm.L"][2].confidence == 0.0
    assert result["forearm.L"][2].confidence == 0.0


def test_solve_ik_chain_applies_rest_pose_correction():
    graph = MotionGraph(
        frames=[
            _frame(0, (0.0, 0.0), (10.0, 0.0), (20.0, 0.0)),
            _frame(1, (0.0, 0.0), (10.0, 0.0), (10.0, 10.0)),
        ],
        tracks={},
        fps=24.0,
    )
    rest_x_90 = (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, -1.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )

    uncorrected = solve_ik_chain(graph, _chain())
    corrected = solve_ik_chain(
        graph, _chain(), root_rest_local_matrix=rest_x_90, mid_rest_local_matrix=rest_x_90
    )

    assert corrected["forearm.L"][1].rotation != pytest.approx(
        uncorrected["forearm.L"][1].rotation
    )


def test_solve_ik_chain_respects_axis_hints():
    graph = MotionGraph(
        frames=[
            _frame(0, (0.0, 0.0), (10.0, 0.0), (20.0, 0.0)),
            _frame(1, (0.0, 0.0), (10.0, 0.0), (10.0, 10.0)),
        ],
        tracks={},
        fps=24.0,
    )

    result = solve_ik_chain(graph, _chain(root_axis_hint="+Y", mid_axis_hint="+X"))

    assert result["forearm.L"][1].rotation == pytest.approx(
        quaternion_from_axis_angle((1.0, 0.0, 0.0), math.pi / 2)
    )
