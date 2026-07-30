from pose.subject_tracker import Detection, SubjectTracker


def test_initial_selection_prefers_overlap_with_user_box():
    tracker = SubjectTracker((0, 0, 100, 100))
    selected = tracker.select([
        Detection((200, 0, 300, 100), 0.99),
        Detection((5, 5, 105, 105), 0.7),
    ])
    assert selected.bbox == (5, 5, 105, 105)


def test_selection_keeps_previous_subject_when_confidence_changes():
    tracker = SubjectTracker((0, 0, 100, 100))
    tracker.select([Detection((0, 0, 100, 100), 0.8)])
    selected = tracker.select([
        Detection((4, 0, 104, 100), 0.55),
        Detection((200, 0, 300, 100), 0.99),
    ])
    assert selected.bbox == (4, 0, 104, 100)


def test_empty_detections_do_not_change_tracker_state():
    tracker = SubjectTracker((0, 0, 100, 100))
    assert tracker.select([]) is None
    assert tracker.select([Detection((2, 0, 102, 100), 0.5)]).bbox == (2, 0, 102, 100)
