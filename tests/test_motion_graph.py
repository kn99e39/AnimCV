from motion.motion_graph import MotionFrame, MotionGraph, MotionPoint, MotionTrack


def _sample_graph() -> MotionGraph:
    point = MotionPoint(
        semantic_name="left_wrist",
        frame_index=0,
        position_2d=(0.5, 0.5),
        position_3d=None,
        confidence=0.9,
        visible=True,
    )
    frame = MotionFrame(
        frame_index=0,
        timestamp=0.0,
        points={"left_wrist": point},
        importance=0.8,
        locked=False,
    )
    track = MotionTrack(semantic_name="left_wrist", points=[point])
    return MotionGraph(
        frames=[frame],
        tracks={"left_wrist": track},
        fps=24.0,
        source_metadata={"video": "reference.mp4"},
    )


def test_motion_graph_roundtrip():
    graph = _sample_graph()

    restored = MotionGraph.from_dict(graph.to_dict())

    assert restored == graph


def test_motion_graph_json_roundtrip(tmp_path):
    from motion.motion_io import load_motion_graph, save_motion_graph

    graph = _sample_graph()
    path = tmp_path / "sample.motion.json"

    save_motion_graph(graph, path)
    restored = load_motion_graph(path)

    assert restored == graph
