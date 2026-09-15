import numpy as np

from framepose.signs import UNIT_FORWARD_EPSILON, UNKNOWN
from scripts.diagnose_pose_reconciliation_readback import boundary_probe


def test_solver_and_canonical_are_unknown_just_below_existing_boundary():
    probe = boundary_probe("below")

    assert probe["solver_unit_forward"] < UNIT_FORWARD_EPSILON
    assert probe["canonical_bend_direction_y"] < UNIT_FORWARD_EPSILON
    assert probe["canonical_sign_state"] == UNKNOWN


def test_mathematically_equal_boundary_is_recorded_at_machine_precision():
    probe = boundary_probe("equal")

    assert abs(probe["solver_margin"]) <= 16 * np.finfo(np.float64).eps
    assert abs(probe["canonical_margin"]) <= 16 * np.finfo(np.float64).eps
    assert probe["canonical_sign_state"] in (UNKNOWN, 1)


def test_solver_and_canonical_are_readable_just_above_existing_boundary():
    probe = boundary_probe("above")

    assert probe["solver_unit_forward"] > UNIT_FORWARD_EPSILON
    assert probe["canonical_bend_direction_y"] > UNIT_FORWARD_EPSILON
    assert probe["canonical_sign_state"] == 1
