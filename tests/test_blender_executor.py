import sys

import pytest

from fake_bpy import FakeArmatureObject, FakeBpyModule
from retarget.solver import AnimationClip, AnimationTrack, BoneTransformSample


@pytest.fixture
def fake_bpy(monkeypatch):
    fake = FakeBpyModule()
    monkeypatch.setitem(sys.modules, "bpy", fake)
    return fake


def _rotation_track(bone_name, count=3):
    samples = [
        BoneTransformSample(
            frame_index=i, bone_name=bone_name, location=None,
            rotation=(0.0, 0.0, 0.0, 1.0), scale=None, confidence=0.9,
        )
        for i in range(count)
    ]
    return AnimationTrack(bone_name=bone_name, samples=samples)


def _location_track(bone_name, count=3):
    samples = [
        BoneTransformSample(
            frame_index=i, bone_name=bone_name, location=(float(i), 0.0, 0.0),
            rotation=(0.0, 0.0, 0.0, 1.0), scale=None, confidence=0.9,
        )
        for i in range(count)
    ]
    return AnimationTrack(bone_name=bone_name, samples=samples)


def test_write_keyframes_inserts_rotation_and_location(fake_bpy):
    from blender.keyframe_writer import write_keyframes

    armature = FakeArmatureObject("Armature", ["upper_arm.L", "hand.L"])
    clip = AnimationClip(
        name="Generated_Motion",
        fps=24.0,
        tracks={
            "upper_arm.L": _rotation_track("upper_arm.L"),
            "hand.L": _location_track("hand.L"),
        },
        frame_start=0,
        frame_end=2,
    )

    inserted = write_keyframes(armature, clip)

    assert inserted == 6
    upper_arm_bone = armature.pose.bones["upper_arm.L"]
    assert upper_arm_bone.rotation_mode == "QUATERNION"
    assert [call[0] for call in upper_arm_bone.keyframe_calls] == ["rotation_quaternion"] * 3

    hand_bone = armature.pose.bones["hand.L"]
    assert [call[0] for call in hand_bone.keyframe_calls] == ["location"] * 3

    assert armature.animation_data.action.name == "Generated_Motion"


def test_write_keyframes_skips_bone_absent_from_armature(fake_bpy):
    from blender.keyframe_writer import write_keyframes

    armature = FakeArmatureObject("Armature", ["upper_arm.L"])
    clip = AnimationClip(
        name="Generated_Motion",
        fps=24.0,
        tracks={"forearm.L": _rotation_track("forearm.L")},
        frame_start=0,
        frame_end=2,
    )

    inserted = write_keyframes(armature, clip)

    assert inserted == 0


def test_write_keyframes_creates_animation_data_if_missing(fake_bpy):
    from blender.keyframe_writer import write_keyframes

    armature = FakeArmatureObject("Armature", ["head"])
    assert armature.animation_data is None

    write_keyframes(
        armature,
        AnimationClip(
            name="Generated_Motion", fps=24.0, tracks={"head": _rotation_track("head", count=1)}
        ),
    )

    assert armature.animation_data is not None


def test_import_rig_dispatches_by_extension_and_finds_armature(fake_bpy):
    from blender.executor import BlenderExecutor

    armature = FakeArmatureObject("Armature", ["root"])
    fake_bpy.data.objects.append(armature)

    result = BlenderExecutor().import_rig("character.blend")

    assert fake_bpy.ops.open_mainfile_calls == ["character.blend"]
    assert result is armature


def test_import_rig_fbx_extension_calls_fbx_importer(fake_bpy):
    from blender.executor import BlenderExecutor

    fake_bpy.data.objects.append(FakeArmatureObject("Armature", ["root"]))

    BlenderExecutor().import_rig("character.fbx")

    assert fake_bpy.ops.import_fbx_calls == ["character.fbx"]


def test_import_rig_unsupported_extension_raises(fake_bpy):
    from blender.executor import BlenderExecutor

    with pytest.raises(ValueError):
        BlenderExecutor().import_rig("character.obj")


def test_import_rig_no_armature_found_raises(fake_bpy):
    from blender.executor import BlenderExecutor

    with pytest.raises(ValueError):
        BlenderExecutor().import_rig("character.blend")


def test_apply_animation_writes_keyframes_onto_scene_armature(fake_bpy):
    from blender.executor import BlenderExecutor

    armature = FakeArmatureObject("Armature", ["upper_arm.L"])
    fake_bpy.data.objects.append(armature)
    clip = AnimationClip(
        name="Generated_Motion", fps=24.0, tracks={"upper_arm.L": _rotation_track("upper_arm.L")}
    )

    BlenderExecutor().apply_animation(clip)

    assert len(armature.pose.bones["upper_arm.L"].keyframe_calls) == 3


def test_apply_animation_no_armature_raises(fake_bpy):
    from blender.executor import BlenderExecutor

    with pytest.raises(ValueError):
        BlenderExecutor().apply_animation(AnimationClip(name="x", fps=24.0))


def test_save_blend_and_export_fbx_call_bpy_ops(fake_bpy):
    from blender.executor import BlenderExecutor

    executor = BlenderExecutor()
    executor.save_blend("out.blend")
    executor.export_fbx("out.fbx")

    assert fake_bpy.ops.save_as_mainfile_calls == ["out.blend"]
    assert fake_bpy.ops.export_fbx_calls == ["out.fbx"]
