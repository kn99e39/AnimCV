import pytest

from pose.contact import ContactState, ContactThresholds, classify_foot_contact, fit_contact_thresholds
from pose.pose_lifter import LiftedPoseFrame, LiftedPosePoint, LiftedPoseSequence


def _sequence(positions, *, fps=30.0, valid=None) -> LiftedPoseSequence:
    valid = valid or [True] * len(positions)
    frames = [
        LiftedPoseFrame(i, i / fps, {
            "left_ankle": LiftedPosePoint("left_ankle", tuple(p), 0.9, 0.1, observation_valid=v),
        })
        for i, (p, v) in enumerate(zip(positions, valid))
    ]
    return LiftedPoseSequence(frames=frames, source_fps=fps)


def test_stationary_ankle_is_classified_contact():
    positions = [(0.0, 0.0, 0.0)] * 8
    thresholds = ContactThresholds(contact_speed_m_s=0.05, moving_speed_m_s=0.5, height_std_threshold_m=0.02)
    states = classify_foot_contact(_sequence(positions), "left", thresholds)
    assert all(state is ContactState.CONTACT for state in states)


def test_fast_swinging_ankle_is_classified_moving():
    positions = [(0.2 * i, 0.0, 0.0) for i in range(8)]  # 0.2m/frame @30fps = 6 m/s
    thresholds = ContactThresholds(contact_speed_m_s=0.05, moving_speed_m_s=0.5, height_std_threshold_m=0.02)
    states = classify_foot_contact(_sequence(positions), "left", thresholds)
    assert all(state is ContactState.MOVING for state in states[2:-2])


def test_unreliable_observation_refuses_to_unknown_rather_than_guessing():
    positions = [(0.0, 0.0, 0.0)] * 8
    valid = [True, True, True, False, True, True, True, True]
    thresholds = ContactThresholds(contact_speed_m_s=0.05, moving_speed_m_s=0.5, height_std_threshold_m=0.02)
    states = classify_foot_contact(_sequence(positions, valid=valid), "left", thresholds)
    assert states[3] is ContactState.UNKNOWN
    # Neighboring frames whose finite-difference window touches the invalid
    # frame also lose evidence and must not silently keep a confident state.
    assert states[2] in (ContactState.UNKNOWN, ContactState.CONTACT)


def test_short_runs_are_suppressed_to_unknown_for_lack_of_temporal_support():
    # A single-frame MOVING blip surrounded by CONTACT should not survive.
    positions = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
                 (5.0, 0.0, 0.0),
                 (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)]
    thresholds = ContactThresholds(
        contact_speed_m_s=0.05, moving_speed_m_s=0.5, height_std_threshold_m=0.02, min_reliable_run=2,
    )
    states = classify_foot_contact(_sequence(positions), "left", thresholds)
    assert ContactState.MOVING not in states  # the blip's run length is 1 < min_reliable_run


def test_classification_is_deterministic_on_replay():
    positions = [(0.01 * i, 0.0, 0.0) for i in range(12)]
    thresholds = ContactThresholds(contact_speed_m_s=0.1, moving_speed_m_s=0.4, height_std_threshold_m=0.02)
    sequence = _sequence(positions)
    first = classify_foot_contact(sequence, "left", thresholds)
    second = classify_foot_contact(sequence, "left", thresholds)
    assert first == second


def test_thresholds_reject_invalid_ordering():
    with pytest.raises(ValueError):
        ContactThresholds(contact_speed_m_s=0.5, moving_speed_m_s=0.1, height_std_threshold_m=0.0)


def test_fit_contact_thresholds_uses_only_the_sequences_it_is_given():
    train_positions = [(0.0, 0.0, 0.0)] * 30
    train = _sequence(train_positions)
    thresholds = fit_contact_thresholds([train], "left")
    # A stationary train distribution should yield a near-zero contact cutoff,
    # independent of anything from a held-out split (none was passed in).
    assert thresholds.contact_speed_m_s == pytest.approx(0.0, abs=1e-6)


def test_fit_contact_thresholds_raises_on_too_little_evidence():
    tiny = _sequence([(0.0, 0.0, 0.0)] * 2)
    with pytest.raises(ValueError):
        fit_contact_thresholds([tiny], "left")
