import pytest

from pose.contact import ContactState
from pose.contact_reference import kinematic_proxy_contact_reference


def test_stationary_positions_are_contact():
    positions = [(0.0, 0.0, 0.0)] * 5
    states = kinematic_proxy_contact_reference(positions, fps=30.0, speed_threshold_m_s=0.05)
    assert all(state is ContactState.CONTACT for state in states)


def test_fast_moving_positions_are_moving():
    positions = [(0.5 * i, 0.0, 0.0) for i in range(5)]
    states = kinematic_proxy_contact_reference(positions, fps=30.0, speed_threshold_m_s=0.05)
    assert all(state is ContactState.MOVING for state in states[1:-1])


def test_missing_position_is_unknown_not_guessed():
    positions = [(0.0, 0.0, 0.0), None, (0.0, 0.0, 0.0)]
    states = kinematic_proxy_contact_reference(positions, fps=30.0, speed_threshold_m_s=0.05)
    assert states[1] is ContactState.UNKNOWN


def test_rejects_non_positive_fps():
    with pytest.raises(ValueError):
        kinematic_proxy_contact_reference([(0.0, 0.0, 0.0)], fps=0.0, speed_threshold_m_s=0.05)
