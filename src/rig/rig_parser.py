"""FBX / rig hierarchy parser via Assimp (Architecture_v2.md section 3.3).

Uses Assimp (``pyassimp``) rather than the proprietary Autodesk FBX SDK:
the FBX SDK has no official PyPI wheel and requires a manual
download/license from Autodesk, while Assimp reads FBX (and many other
formats) through a pip-installable binding. ``pyassimp`` — and the
native ``assimp`` shared library it wraps — is imported lazily inside
``RigParser.load`` so the rest of the project runs without it installed
(Milestone 1 constraint). Note that ``import pyassimp`` itself raises if
the native library is missing (not just a plain ``ImportError``), so
that import is wrapped broadly.

Assimp exposes only a generic scene node tree, not a first-class bone
concept. Every scene node is treated as a candidate bone; this matches
the project's "assisted, not fully automatic" design principle (section
6) — the user prunes/confirms real bones later through the Bone Mapping
Profile, and only mapped bones are animated (section 6.5).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from common.types import Matrix4
from rig.rig_profile import BoneInfo, RigProfile

_ROOT_NAME_HINTS = ("root", "hips", "pelvis", "armature")


class SceneNode(Protocol):
    """Structural shape of an Assimp scene node (and of the fake nodes
    used in tests) — duck-typed so bone-tree extraction is testable
    without the native assimp library."""

    name: str
    transformation: Any
    children: list["SceneNode"]


class RigParser:
    def load(self, path: str) -> RigProfile:
        try:
            import pyassimp
        except Exception as exc:  # pyassimp raises its own AssimpError at
            # import time when the native library is missing, not a plain
            # ImportError, so this must catch broadly.
            raise ImportError(
                "pyassimp / the native assimp shared library is not available. "
                "Install libassimp for your OS and `pip install pyassimp` to use RigParser."
            ) from exc

        with pyassimp.load(path) as scene:
            bones = build_bone_tree(scene.rootnode)

        return RigProfile(
            rig_id=Path(path).stem,
            source_path=str(path),
            bones=bones,
            root_bone=detect_root_bone(bones),
            scale=1.0,
            metadata={"parser": "assimp"},
        )


def build_bone_tree(root_node: SceneNode) -> dict[str, BoneInfo]:
    bones: dict[str, BoneInfo] = {}
    _walk(root_node, parent_name=None, bones=bones)
    return bones


def _walk(node: SceneNode, parent_name: str | None, bones: dict[str, BoneInfo]) -> None:
    bones[node.name] = BoneInfo(
        name=node.name,
        parent=parent_name,
        children=[child.name for child in node.children],
        rest_local_matrix=_to_matrix4(node.transformation),
        # World matrices require accumulating parent transforms down the
        # chain; left None here and computed where needed (Milestone 5).
        rest_world_matrix=None,
        local_axis_hint=None,
    )
    for child in node.children:
        _walk(child, node.name, bones)


def _to_matrix4(transformation: Any) -> Matrix4:
    return tuple(tuple(float(v) for v in row) for row in transformation)


def detect_root_bone(bones: dict[str, BoneInfo]) -> str | None:
    """Heuristic root-bone candidate detection (section 3.3).

    Assimp scenes have a single absolute scene-root node (often named
    e.g. "RootNode") that is not itself a skeletal joint. This first
    looks for a node name hinting at a skeleton root (hips/pelvis/root/
    armature); failing that, it falls back to the absolute root node's
    only child if there's exactly one (the common case for
    skeleton-only exports), else the absolute root itself. This is a
    best-effort suggestion — RigProfile.root_bone can always be
    corrected by hand.
    """
    scene_roots = [bone for bone in bones.values() if bone.parent is None]
    scene_root_name = scene_roots[0].name if len(scene_roots) == 1 else None

    for name in bones:
        # The absolute scene-root wrapper node (e.g. "RootNode") is
        # excluded from hint matching even if its own name contains a
        # hint like "root" — it is conventionally not a skeletal joint.
        if name == scene_root_name:
            continue
        if any(hint in name.lower() for hint in _ROOT_NAME_HINTS):
            return name

    if scene_root_name is not None:
        root = bones[scene_root_name]
        if len(root.children) == 1:
            return root.children[0]
        return root.name

    return None
