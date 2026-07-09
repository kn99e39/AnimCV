from motion.motion_builder import MotionGraphBuilder
from pose.pose_types import PoseFrame, PoseLandmark, PoseSequence


def _sample_pose_sequence() -> PoseSequence:
    frames = []
    for i in range(3):
        landmark = PoseLandmark(
            name="left_wrist", x=10.0 + i, y=20.0 + i, confidence=0.9, visible=True
        )
        frames.append(
            PoseFrame(frame_index=i, timestamp=i / 24.0, landmarks={"left_wrist": landmark})
        )
    return PoseSequence(frames=frames, source_fps=24.0, landmark_schema="canonical_v1")


def test_build_creates_one_motion_frame_per_pose_frame():
    poses = _sample_pose_sequence()

    graph = MotionGraphBuilder().build(poses, source_metadata={"video": "clip.mp4"})

    assert len(graph.frames) == 3
    assert graph.fps == 24.0
    assert graph.source_metadata == {"video": "clip.mp4"}


def test_build_groups_points_into_tracks_by_semantic_name():
    poses = _sample_pose_sequence()

    graph = MotionGraphBuilder().build(poses)

    assert set(graph.tracks) == {"left_wrist"}
    track = graph.tracks["left_wrist"]
    assert [p.frame_index for p in track.points] == [0, 1, 2]
    assert track.points[1].position_2d == (11.0, 21.0)


def test_build_defaults_importance_and_lock_to_pre_optimization_values():
    poses = _sample_pose_sequence()

    graph = MotionGraphBuilder().build(poses)

    assert all(frame.importance == 0.0 and frame.locked is False for frame in graph.frames)


def test_build_populates_position_3d_when_depth_was_sampled():
    landmark = PoseLandmark(
        name="left_wrist", x=10.0, y=20.0, confidence=0.9, visible=True, z=3.5
    )
    poses = PoseSequence(
        frames=[PoseFrame(frame_index=0, timestamp=0.0, landmarks={"left_wrist": landmark})],
        source_fps=24.0,
    )

    graph = MotionGraphBuilder().build(poses)

    point = graph.frames[0].points["left_wrist"]
    assert point.position_3d == (10.0, 20.0, 3.5)


def test_build_leaves_position_3d_none_without_depth():
    poses = _sample_pose_sequence()  # landmarks have no z set

    graph = MotionGraphBuilder().build(poses)

    assert all(
        point.position_3d is None
        for frame in graph.frames
        for point in frame.points.values()
    )
