"""Headless Blender entry point (Architecture_v2.md section 4.10 / 12.3).

Run via:

    blender --background --python scripts/apply_motion.py -- \\
        --rig path/to/rig.fbx_or_blend \\
        --animation path/to/animation.json \\
        --out path/to/result.blend \\
        [--fbx-out path/to/result.fbx]

This (plus src/blender/*.py, which it imports) is the only place in the
project that touches ``bpy``. It only runs correctly inside Blender's
own bundled Python interpreter; running it with a plain ``python`` will
fail at the ``import bpy`` inside BlenderExecutor's methods, not here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    # Blender's bundled Python has no knowledge of this project's src/
    # layout, so it has to be added by hand before importing our code.
    sys.path.insert(0, str(_SRC_DIR))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply an AnimationClip to a rig inside Blender")
    parser.add_argument("--rig", required=True, help="Path to .fbx or .blend rig file")
    parser.add_argument("--animation", required=True, help="Path to animation(_optimized).json")
    parser.add_argument("--out", required=True, help="Output .blend path")
    parser.add_argument("--fbx-out", default=None, help="Optional output .fbx path")
    return parser


def args_after_double_dash(argv: list[str]) -> list[str]:
    """Blender puts its own args before "--" and the script's args after
    it, but leaves the *entire* original argv in sys.argv — the split is
    the script's own responsibility (this is Blender's documented
    pattern, not a workaround). Falls back to argv[1:] so this also
    works when run directly with a plain ``python`` (e.g. in tests)."""
    if "--" in argv:
        return argv[argv.index("--") + 1 :]
    return argv[1:]


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(args_after_double_dash(sys.argv if argv is None else argv))

    # Confirmed by testing against a real Blender build: an unhandled
    # exception in a --python script does NOT make blender.exe exit
    # non-zero (only SystemExit does) — so without this try/except, a
    # crash here would look like success to whoever invoked blender
    # (e.g. app.cli's subprocess.run check), silently producing no
    # output file. Catch broadly and turn it into an explicit exit code.
    try:
        from blender.executor import BlenderExecutor
        from retarget.solver import load_animation_clip

        executor = BlenderExecutor()
        executor.import_rig(args.rig)

        clip = load_animation_clip(args.animation)
        executor.apply_animation(clip)

        executor.save_blend(args.out)
        print(f"[apply_motion] wrote {args.out}")

        if args.fbx_out:
            executor.export_fbx(args.fbx_out)
            print(f"[apply_motion] wrote {args.fbx_out}")
    except Exception as exc:
        print(f"[apply_motion] failed: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
