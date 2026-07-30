"""FBX export from Blender (Architecture_v2.md section 3.4)."""

from __future__ import annotations


def export_fbx(path: str) -> None:
    import bpy

    # This tool exports animation for an existing rig, not a whole Blender
    # scene. Limiting the FBX to armatures prevents cameras/lights from the
    # source .blend leaking into a game asset (and avoids importer failures
    # caused by unsupported light properties in newer Blender releases).
    bpy.ops.export_scene.fbx(
        filepath=path,
        use_selection=False,
        object_types={"ARMATURE"},
        bake_anim=True,
    )
