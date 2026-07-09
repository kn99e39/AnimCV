from rig.bone_mapping import BoneMappingEntry, BoneMappingProfile, IKChainEntry
from rig.rig_profile import BoneInfo, RigProfile

_IDENTITY = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)


def _sample_rig_profile() -> RigProfile:
    upper_arm = BoneInfo(
        name="upper_arm.L",
        parent="clavicle.L",
        children=["forearm.L"],
        rest_local_matrix=_IDENTITY,
        rest_world_matrix=_IDENTITY,
        local_axis_hint={"primary": (0.0, 1.0, 0.0)},
    )
    return RigProfile(
        rig_id="character_01",
        source_path="character.fbx",
        bones={"upper_arm.L": upper_arm},
        root_bone="root",
        scale=1.0,
        metadata={"exporter": "blender"},
    )


def _sample_mapping_profile() -> BoneMappingProfile:
    entry = BoneMappingEntry(
        target_bone="upper_arm.L",
        source_type="landmark",
        source_names=["left_shoulder", "left_elbow"],
        mapping_mode="direction",
        weight=1.0,
        axis_hint="+Y",
        locked=False,
    )
    return BoneMappingProfile(
        rig_id="character_01",
        entries=[entry],
        created_from_frame=0,
        user_notes="initial mapping",
    )


def test_rig_profile_roundtrip():
    profile = _sample_rig_profile()

    restored = RigProfile.from_dict(profile.to_dict())

    assert restored == profile


def test_bone_mapping_profile_roundtrip():
    profile = _sample_mapping_profile()

    restored = BoneMappingProfile.from_dict(profile.to_dict())

    assert restored == profile


def test_bone_mapping_profile_partial_mapping_is_allowed():
    profile = BoneMappingProfile(rig_id="character_01", entries=[])

    assert profile.entries == []


def test_rig_profile_json_roundtrip(tmp_path):
    from rig.rig_profile import load_rig_profile, save_rig_profile

    profile = _sample_rig_profile()
    path = tmp_path / "rig_profile.json"

    save_rig_profile(profile, path)
    restored = load_rig_profile(path)

    assert restored == profile


def test_bone_mapping_profile_json_roundtrip(tmp_path):
    from rig.bone_mapping import load_bone_mapping_profile, save_bone_mapping_profile

    profile = _sample_mapping_profile()
    path = tmp_path / "mapping.json"

    save_bone_mapping_profile(profile, path)
    restored = load_bone_mapping_profile(path)

    assert restored == profile


def _sample_ik_chain() -> IKChainEntry:
    return IKChainEntry(
        name="left_arm",
        root_bone="upper_arm.L",
        mid_bone="forearm.L",
        end_bone="hand.L",
        root_source="left_shoulder",
        mid_source="left_elbow",
        end_source="left_wrist",
        root_axis_hint="+Y",
        mid_axis_hint=None,
        enabled=True,
    )


def test_ik_chain_entry_roundtrip():
    chain = _sample_ik_chain()

    restored = IKChainEntry.from_dict(chain.to_dict())

    assert restored == chain


def test_bone_mapping_profile_with_ik_chains_roundtrip():
    profile = BoneMappingProfile(
        rig_id="character_01",
        entries=[],
        ik_chains=[_sample_ik_chain()],
        created_from_frame=0,
    )

    restored = BoneMappingProfile.from_dict(profile.to_dict())

    assert restored == profile
    assert restored.ik_chains[0].root_bone == "upper_arm.L"


def test_bone_mapping_profile_from_dict_defaults_ik_chains_for_old_json():
    # Simulates a mapping.json written before ik_chains existed.
    data = {
        "rig_id": "character_01",
        "entries": [],
        "created_from_frame": 0,
        "user_notes": None,
    }

    profile = BoneMappingProfile.from_dict(data)

    assert profile.ik_chains == []
