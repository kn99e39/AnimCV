"""A minimal fake ``bpy`` module for testing blender/*.py without Blender.

Real end-to-end verification against actual Blender is done separately
(see result/result_mil7.txt) — this only exercises the Python-side
logic in keyframe_writer.py / executor.py: which bones get keyframed,
on which data_path, with what values, in what order.
"""

from __future__ import annotations

from types import SimpleNamespace


class FakeAction:
    def __init__(self, name):
        self.name = name
        self.fcurves = []


class FakeActionsCollection:
    def __init__(self):
        self.created = []

    def new(self, name):
        action = FakeAction(name)
        self.created.append(action)
        return action


class FakeAnimData:
    def __init__(self):
        self.action = None


class FakePoseBone:
    def __init__(self, name):
        self.name = name
        self.rotation_mode = "XYZ"
        self.rotation_quaternion = (0.0, 0.0, 0.0, 1.0)
        self.location = (0.0, 0.0, 0.0)
        self.keyframe_calls: list[tuple[str, int]] = []

    def keyframe_insert(self, data_path, frame):
        self.keyframe_calls.append((data_path, frame))


class FakePose:
    def __init__(self, bone_names):
        self.bones = {name: FakePoseBone(name) for name in bone_names}


class FakeArmatureObject:
    def __init__(self, name, bone_names):
        self.name = name
        self.type = "ARMATURE"
        self.animation_data = None
        self.pose = FakePose(bone_names)

    def animation_data_create(self):
        self.animation_data = FakeAnimData()


class FakeScene:
    def __init__(self):
        self.frame_current = 0

    def frame_set(self, frame):
        self.frame_current = frame


class FakeData:
    def __init__(self):
        self.actions = FakeActionsCollection()
        self.objects: list = []


class FakeOps:
    def __init__(self):
        self.open_mainfile_calls: list[str] = []
        self.import_fbx_calls: list[str] = []
        self.save_as_mainfile_calls: list[str] = []
        self.export_fbx_calls: list[str] = []

        self.wm = SimpleNamespace(
            open_mainfile=lambda filepath: self.open_mainfile_calls.append(filepath),
            save_as_mainfile=lambda filepath: self.save_as_mainfile_calls.append(filepath),
        )
        self.import_scene = SimpleNamespace(
            fbx=lambda filepath: self.import_fbx_calls.append(filepath)
        )
        self.export_scene = SimpleNamespace(
            fbx=lambda filepath, **kwargs: self.export_fbx_calls.append(filepath)
        )


class FakeBpyModule:
    def __init__(self):
        self.data = FakeData()
        self.context = SimpleNamespace(scene=FakeScene())
        self.ops = FakeOps()
