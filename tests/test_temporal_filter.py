from motion.motion_graph import MotionFrame, MotionGraph, MotionPoint
from retarget.temporal_filter import median_filter_motion_graph


def _point(frame_index: int, x: float, *, visible: bool = True) -> MotionPoint:
    return MotionPoint(
        semantic_name="left_wrist",
        frame_index=frame_index,
        position_2d=(x, 0.0),
        position_3d=None,
        confidence=0.9,
        visible=visible,
    )


def _graph(values: list[float]) -> MotionGraph:
    return MotionGraph(
        frames=[
            MotionFrame(frame_index=i, timestamp=i / 24, points={"left_wrist": _point(i, x)})
            for i, x in enumerate(values)
        ],
        fps=24,
    )


def test_median_filter_removes_single_frame_position_spike():
    graph = _graph([0.0, 1.0, 100.0, 3.0, 4.0])

    filtered = median_filter_motion_graph(graph, window_size=3)

    assert [frame.points["left_wrist"].position_2d[0] for frame in filtered.frames] == [
        0.0,
        1.0,
        3.0,
        4.0,
        4.0,
    ]
    assert graph.frames[2].points["left_wrist"].position_2d == (100.0, 0.0)


def test_median_filter_does_not_invent_data_across_occlusion():
    graph = _graph([0.0, 1.0, 100.0, 3.0, 4.0])
    graph.frames[2].points["left_wrist"].visible = False

    filtered = median_filter_motion_graph(graph, window_size=3)

    assert [frame.points["left_wrist"].position_2d[0] for frame in filtered.frames] == [
        0.0,
        1.0,
        100.0,
        3.0,
        4.0,
    ]


def test_median_filter_requires_odd_positive_window():
    graph = _graph([0.0])

    for size in (0, 2):
        try:
            median_filter_motion_graph(graph, window_size=size)
        except ValueError as exc:
            assert "odd positive" in str(exc)
        else:
            raise AssertionError("expected invalid smoothing window to fail")
