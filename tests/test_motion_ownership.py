from pose.motion_ownership import OWNERSHIP_MODEL, Observability, Owner, audit_for


def test_ownership_model_covers_all_six_quantities_with_distinct_owners():
    names = {entry.name for entry in OWNERSHIP_MODEL}
    assert names == {
        "local_articulation", "body_heading_yaw", "global_horizontal_translation",
        "ground_height_placement", "foot_planted_or_moving", "observation_reliability",
    }
    owners = [entry.owner for entry in OWNERSHIP_MODEL]
    assert len(owners) == len(set(owners)), "each quantity must have a distinct owner (no shared 'root confidence')"
    assert set(owners) == set(Owner)


def test_global_translation_is_not_observable_from_frame_pose_alone():
    entry = audit_for("global_horizontal_translation")
    assert entry.owner is Owner.ROOT_MOTION
    assert entry.observability is Observability.NOT_OBSERVABLE


def test_contact_is_only_temporally_inferable():
    entry = audit_for("foot_planted_or_moving")
    assert entry.owner is Owner.CONTACT
    assert entry.observability is Observability.ONLY_TEMPORALLY_INFERABLE


def test_audit_for_unknown_quantity_raises():
    try:
        audit_for("not_a_real_quantity")
    except KeyError:
        return
    raise AssertionError("expected KeyError for an undefined quantity")
