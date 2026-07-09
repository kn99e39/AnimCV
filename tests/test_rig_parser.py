from dataclasses import dataclass, field

import pytest

from rig.rig_parser import build_bone_tree, detect_root_bone

_IDENTITY = ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))


@dataclass
class FakeNode:
    name: str
    transformation: tuple = _IDENTITY
    children: list["FakeNode"] = field(default_factory=list)


def _humanoid_scene():
    hand = FakeNode("hand.L")
    forearm = FakeNode("forearm.L", children=[hand])
    upper_arm = FakeNode("upper_arm.L", children=[forearm])
    hips = FakeNode("hips", children=[upper_arm])
    scene_root = FakeNode("RootNode", children=[hips])
    return scene_root


def test_build_bone_tree_walks_full_hierarchy():
    scene_root = _humanoid_scene()

    bones = build_bone_tree(scene_root)

    assert set(bones) == {"RootNode", "hips", "upper_arm.L", "forearm.L", "hand.L"}
    assert bones["RootNode"].parent is None
    assert bones["hips"].parent == "RootNode"
    assert bones["upper_arm.L"].parent == "hips"
    assert bones["hips"].children == ["upper_arm.L"]


def test_build_bone_tree_converts_transformation_to_matrix4():
    scene_root = FakeNode("root", transformation=((2, 0, 0, 0), (0, 2, 0, 0), (0, 0, 2, 0), (0, 0, 0, 1)))

    bones = build_bone_tree(scene_root)

    assert bones["root"].rest_local_matrix == (
        (2.0, 0.0, 0.0, 0.0),
        (0.0, 2.0, 0.0, 0.0),
        (0.0, 0.0, 2.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def test_detect_root_bone_prefers_name_hint():
    bones = build_bone_tree(_humanoid_scene())

    assert detect_root_bone(bones) == "hips"


def test_detect_root_bone_falls_back_to_only_child_of_scene_root():
    scene_root = FakeNode("RootNode", children=[FakeNode("skeleton_top", children=[FakeNode("child_a")])])

    bones = build_bone_tree(scene_root)

    assert detect_root_bone(bones) == "skeleton_top"


def test_detect_root_bone_falls_back_to_scene_root_when_ambiguous():
    scene_root = FakeNode("RootNode", children=[FakeNode("branch_a"), FakeNode("branch_b")])

    bones = build_bone_tree(scene_root)

    assert detect_root_bone(bones) == "RootNode"


def test_rig_parser_load_without_assimp_raises_import_error(monkeypatch):
    import builtins

    from rig.rig_parser import RigParser

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pyassimp":
            raise ModuleNotFoundError("No module named 'pyassimp'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError):
        RigParser().load("character.fbx")
