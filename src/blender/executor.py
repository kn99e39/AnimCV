"""Blender execution layer (Architecture_v2.md section 3.4).

Only this module (and ``keyframe_writer.py`` / ``export.py``) may import
``bpy``, and only inside function bodies — this module is only ever
actually run from inside Blender's own bundled Python interpreter (see
``scripts/apply_motion.py``), never from the project's regular venv, so
importing it there is harmless as long as nothing calls a method that
needs bpy.
"""

from __future__ import annotations

from typing import Any

from retarget.solver import AnimationClip
from rig.rig_profile import RigProfile

BlenderRigHandle = Any


def _find_armature_object():
    import bpy

    for obj in bpy.data.objects:
        if obj.type == "ARMATURE":
            return obj
    return None


class BlenderExecutor:
    def import_rig(self, path: str) -> BlenderRigHandle:
        import bpy

        lowered = path.lower()
        if lowered.endswith(".blend"):
            bpy.ops.wm.open_mainfile(filepath=path)
        elif lowered.endswith(".fbx"):
            bpy.ops.import_scene.fbx(filepath=path)
        else:
            raise ValueError(f"unsupported rig file type (expected .fbx or .blend): {path}")

        armature_object = _find_armature_object()
        if armature_object is None:
            raise ValueError(f"no armature object found after importing rig: {path}")
        return armature_object

    def apply_animation(self, animation: AnimationClip, rig_profile: RigProfile | None = None) -> None:
        from blender.keyframe_writer import write_keyframes

        armature_object = _find_armature_object()
        if armature_object is None:
            raise ValueError("no armature object in the current scene")

        # rig_profile is accepted to match the documented interface
        # (section 3.4) and for future per-bone axis correction; the
        # Milestone 5 fk_solver already bakes axis_hint into its
        # rotations, so this MVP writer doesn't need it yet.
        write_keyframes(armature_object, animation)

    def save_blend(self, path: str) -> None:
        import bpy

        bpy.ops.wm.save_as_mainfile(filepath=path)

    def export_fbx(self, path: str) -> None:
        from blender.export import export_fbx

        export_fbx(path)
