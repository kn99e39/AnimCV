"""FBX export from Blender (Architecture_v2.md section 3.4)."""

from __future__ import annotations


def export_fbx(path: str) -> None:
    import bpy

    bpy.ops.export_scene.fbx(filepath=path, use_selection=False, bake_anim=True)
