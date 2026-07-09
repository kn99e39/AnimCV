import math

import pytest

from motion.motion_graph import MotionFrame, MotionGraph, MotionPoint
from retarget.axis_utils import quaternion_from_axis_angle
from retarget.fk_solver import frame_direction, signed_angle_delta, solve_anchor_bone, solve_direction_bone
from rig.bone_mapping import BoneMappingEntry


def _point(name, x, y, frame_index=0, confidence=0.9, visible=True):
    return MotionPoint(
        semantic_name=name,
        frame_index=frame_index,
        position_2d=(x, y),
        position_3d=None,
        confidence=confidence,
        visible=visible,
    )


def test_frame_direction_returns_unit_vector():
    frame = MotionFrame(
        frame_index=0,
        timestamp=0.0,
        points={"a": _point("a", 0.0, 0.0), "b": _point("b", 3.0, 4.0)},
    )

    direction = frame_direction(frame, "a", "b")

    assert direction == pytest.approx((0.6, 0.8))


@pytest.mark.parametrize(
    "points",
    [
        {},
        {"a": _point("a", 0.0, 0.0)},
        {"a": _point("a", 0.0, 0.0), "b": _point("b", 1.0, 0.0, visible=False)},
        {"a": _point("a", 1.0, 1.0), "b": _point("b", 1.0, 1.0)},
    ],
)
def test_frame_direction_returns_none_when_unavailable(points):
    frame = MotionFrame(frame_index=0, timestamp=0.0, points=points)

    assert frame_direction(frame, "a", "b") is None


def test_signed_angle_delta_quarter_turn():
    assert signed_angle_delta((1.0, 0.0), (0.0, 1.0)) == pytest.approx(math.pi / 2)


def test_signed_angle_delta_zero_for_identical_directions():
    assert signed_angle_delta((1.0, 0.0), (1.0, 0.0)) == pytest.approx(0.0)


def _direction_motion_graph():
    # source_a fixed at origin; source_b rotates 90 degrees per frame,
    # tracing frame0 -> frame1 -> frame2 as reference, +90 deg, +180 deg.
    positions = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)]
    frames = []
    for i, (bx, by) in enumerate(positions):
        frames.append(
            MotionFrame(
                frame_index=i,
                timestamp=i / 24.0,
                points={"shoulder": _point("shoulder", 0.0, 0.0, i), "elbow": _point("elbow", bx, by, i)},
            )
        )
    return MotionGraph(frames=frames, tracks={}, fps=24.0)


def test_solve_direction_bone_tracks_chain_movement():
    graph = _direction_motion_graph()
    entry = BoneMappingEntry(
        target_bone="upper_arm.L",
        source_type="landmark",
        source_names=["shoulder", "elbow"],
        mapping_mode="direction",
    )

    samples = solve_direction_bone(graph, "upper_arm.L", entry)

    assert len(samples) == 3
    assert samples[0].rotation == pytest.approx((0.0, 0.0, 0.0, 1.0))
    assert samples[1].rotation == pytest.approx(
        quaternion_from_axis_angle((0.0, 0.0, 1.0), math.pi / 2)
    )
    assert samples[2].rotation == pytest.approx(
        quaternion_from_axis_angle((0.0, 0.0, 1.0), math.pi)
    )
    assert all(sample.bone_name == "upper_arm.L" for sample in samples)
    assert all(sample.confidence > 0 for sample in samples)


def test_solve_direction_bone_holds_last_rotation_when_point_lost():
    frames = [
        MotionFrame(
            frame_index=0,
            timestamp=0.0,
            points={"a": _point("a", 0.0, 0.0), "b": _point("b", 0.0, 1.0)},
        ),
        MotionFrame(
            frame_index=1,
            timestamp=1 / 24.0,
            points={"a": _point("a", 0.0, 0.0), "b": _point("b", 0.0, 1.0, visible=False)},
        ),
    ]
    graph = MotionGraph(frames=frames, tracks={}, fps=24.0)
    entry = BoneMappingEntry(
        target_bone="head", source_type="landmark", source_names=["a", "b"], mapping_mode="direction"
    )

    samples = solve_direction_bone(graph, "head", entry)

    assert samples[1].rotation == samples[0].rotation
    assert samples[1].confidence == 0.0


def test_solve_direction_bone_respects_axis_hint():
    graph = _direction_motion_graph()
    entry = BoneMappingEntry(
        target_bone="upper_arm.L",
        source_type="landmark",
        source_names=["shoulder", "elbow"],
        mapping_mode="direction",
        axis_hint="+Y",
    )

    samples = solve_direction_bone(graph, "upper_arm.L", entry)

    assert samples[1].rotation == pytest.approx(
        quaternion_from_axis_angle((0.0, 1.0, 0.0), math.pi / 2)
    )


def test_solve_direction_bone_wrong_source_count_raises():
    entry = BoneMappingEntry(
        target_bone="head", source_type="landmark", source_names=["a"], mapping_mode="direction"
    )

    with pytest.raises(ValueError):
        solve_direction_bone(MotionGraph(), "head", entry)


def test_solve_anchor_bone_tracks_offset_from_reference():
    frames = [
        MotionFrame(frame_index=0, timestamp=0.0, points={"wrist": _point("wrist", 10.0, 10.0)}),
        MotionFrame(frame_index=1, timestamp=1 / 24.0, points={"wrist": _point("wrist", 15.0, 13.0)}),
    ]
    graph = MotionGraph(frames=frames, tracks={}, fps=24.0)
    entry = BoneMappingEntry(
        target_bone="hand.L", source_type="landmark", source_names=["wrist"], mapping_mode="landmark"
    )

    samples = solve_anchor_bone(graph, "hand.L", entry)

    assert samples[0].location == pytest.approx((0.0, 0.0, 0.0))
    assert samples[1].location == pytest.approx((5.0, 3.0, 0.0))


def test_solve_anchor_bone_holds_last_location_when_point_lost():
    frames = [
        MotionFrame(frame_index=0, timestamp=0.0, points={"wrist": _point("wrist", 10.0, 10.0)}),
        MotionFrame(frame_index=1, timestamp=1 / 24.0, points={}),
    ]
    graph = MotionGraph(frames=frames, tracks={}, fps=24.0)
    entry = BoneMappingEntry(
        target_bone="hand.L", source_type="landmark", source_names=["wrist"], mapping_mode="landmark"
    )

    samples = solve_anchor_bone(graph, "hand.L", entry)

    assert samples[1].location == samples[0].location
    assert samples[1].confidence == 0.0
