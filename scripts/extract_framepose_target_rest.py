#!/usr/bin/env python3
"""Run under Blender: import an FBX and save its authoritative rest snapshot."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--rig", type=Path, required=True)
    parser.add_argument("--rig-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    options = parser.parse_args(args)
    try:
        import bpy
        from blender.executor import BlenderExecutor
        from blender.framepose_rest_adapter import extract_target_rest_pose
        from retarget.framepose_target_rest import save_target_rest_pose

        bpy.ops.wm.read_factory_settings(use_empty=True)
        armature = BlenderExecutor().import_rig(str(options.rig))
        snapshot = extract_target_rest_pose(armature, rig_id=options.rig_id,
                                            source_path=str(options.rig))
        options.out.parent.mkdir(parents=True, exist_ok=True)
        save_target_rest_pose(snapshot, options.out)
        print(f"rest pose: {len(snapshot.bones)} bones; sha256={snapshot.digest()}; {options.out}")
        return 0
    except Exception as exc:
        print(f"rest extraction failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
