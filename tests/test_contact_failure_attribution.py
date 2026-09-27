import pytest

from pose.contact import ContactState
from pose.contact_failure_attribution import (
    FailureCategory, attribute_disagreements, fit_position_error_threshold, summarize_attribution,
)


def _errors(ankle, other=0.01):
    return {"left_ankle": ankle, "right_ankle": other, "pelvis": other, "left_knee": other}


def test_no_disagreement_produces_no_attribution():
    oracle = [ContactState.CONTACT, ContactState.MOVING]
    h0 = [ContactState.CONTACT, ContactState.MOVING]
    reference = [ContactState.CONTACT, ContactState.MOVING]
    joint_errors = [_errors(0.01), _errors(0.01)]
    attributions = attribute_disagreements(joint_errors, "left", oracle, h0, reference, position_error_threshold_m=0.1)
    assert attributions == []


def test_preexisting_oracle_disagreement_is_ambiguity_not_h0_fault():
    oracle = [ContactState.MOVING]  # oracle itself already disagreed with reference
    h0 = [ContactState.CONTACT]
    reference = [ContactState.CONTACT]
    joint_errors = [_errors(0.01)]  # H0 position is actually fine
    attributions = attribute_disagreements(joint_errors, "left", oracle, h0, reference, position_error_threshold_m=0.1)
    assert attributions[0].category is FailureCategory.BODY_ROOT_RELATIVE_MOTION_AMBIGUITY


def test_invalid_ankle_observation_is_observation_invalidity():
    oracle = [ContactState.CONTACT]
    h0 = [ContactState.MOVING]
    reference = [ContactState.CONTACT]  # oracle agreed with reference, so not ambiguity
    joint_errors = [{"left_ankle": None, "right_ankle": 0.01, "pelvis": 0.01, "left_knee": 0.01}]
    attributions = attribute_disagreements(joint_errors, "left", oracle, h0, reference, position_error_threshold_m=0.1)
    assert attributions[0].category is FailureCategory.OBSERVATION_INVALIDITY


def test_large_ankle_only_error_is_ankle_specific_failure():
    oracle = [ContactState.CONTACT]
    h0 = [ContactState.MOVING]
    reference = [ContactState.CONTACT]
    joint_errors = [_errors(ankle=0.5, other=0.01)]  # ankle far worse than the rest of the pose
    attributions = attribute_disagreements(joint_errors, "left", oracle, h0, reference, position_error_threshold_m=0.1)
    assert attributions[0].category is FailureCategory.ANKLE_SPECIFIC_FAILURE


def test_large_whole_pose_error_is_pose_position_error():
    oracle = [ContactState.CONTACT]
    h0 = [ContactState.MOVING]
    reference = [ContactState.CONTACT]
    joint_errors = [_errors(ankle=0.5, other=0.4)]  # ankle error is large but so is the rest of the pose
    attributions = attribute_disagreements(joint_errors, "left", oracle, h0, reference, position_error_threshold_m=0.1)
    assert attributions[0].category is FailureCategory.POSE_POSITION_ERROR


def test_small_position_error_is_temporal_jitter():
    oracle = [ContactState.CONTACT]
    h0 = [ContactState.MOVING]
    reference = [ContactState.CONTACT]
    joint_errors = [_errors(ankle=0.01, other=0.01)]  # position is fine; disagreement must be velocity-side
    attributions = attribute_disagreements(joint_errors, "left", oracle, h0, reference, position_error_threshold_m=0.1)
    assert attributions[0].category is FailureCategory.TEMPORAL_JITTER


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError):
        attribute_disagreements([_errors(0.01)], "left", [ContactState.CONTACT] * 2,
                                 [ContactState.MOVING] * 2, [ContactState.CONTACT] * 2, 0.1)


def test_summarize_attribution_counts_every_category_key():
    oracle = [ContactState.CONTACT, ContactState.CONTACT]
    h0 = [ContactState.MOVING, ContactState.MOVING]
    reference = [ContactState.CONTACT, ContactState.CONTACT]
    joint_errors = [_errors(ankle=0.01), {"left_ankle": None, "right_ankle": 0.01, "pelvis": 0.01, "left_knee": 0.01}]
    attributions = attribute_disagreements(joint_errors, "left", oracle, h0, reference, position_error_threshold_m=0.1)
    summary = summarize_attribution(attributions)
    assert summary["total_disagreements"] == 2
    assert set(summary) == {category.value for category in FailureCategory} | {"total_disagreements"}


def test_fit_position_error_threshold_uses_train_only_percentile():
    errors = [0.01 * i for i in range(1, 101)]
    threshold = fit_position_error_threshold(errors, percentile=90.0)
    assert threshold == pytest.approx(0.901, abs=1e-6)


def test_fit_position_error_threshold_raises_on_too_little_evidence():
    with pytest.raises(ValueError):
        fit_position_error_threshold([0.01, 0.02])
