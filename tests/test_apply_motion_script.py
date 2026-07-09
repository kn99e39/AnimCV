import importlib.util
import sys
from pathlib import Path

import pytest

from fake_bpy import FakeBpyModule

_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "apply_motion.py"


def _load_apply_motion_module():
    spec = importlib.util.spec_from_file_location("apply_motion", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_args_after_double_dash_splits_on_blender_style_separator():
    apply_motion = _load_apply_motion_module()

    argv = ["blender", "--background", "--python", "apply_motion.py", "--", "--rig", "r.fbx", "--out", "o.blend"]

    assert apply_motion.args_after_double_dash(argv) == [
        "--rig", "r.fbx", "--out", "o.blend",
    ]


def test_args_after_double_dash_falls_back_when_run_directly():
    apply_motion = _load_apply_motion_module()

    argv = ["apply_motion.py", "--rig", "r.fbx", "--animation", "a.json", "--out", "o.blend"]

    assert apply_motion.args_after_double_dash(argv) == [
        "--rig", "r.fbx", "--animation", "a.json", "--out", "o.blend",
    ]


def test_build_parser_requires_rig_animation_and_out():
    apply_motion = _load_apply_motion_module()
    parser = apply_motion.build_parser()

    args = parser.parse_args(["--rig", "r.fbx", "--animation", "a.json", "--out", "o.blend"])

    assert args.rig == "r.fbx"
    assert args.animation == "a.json"
    assert args.out == "o.blend"
    assert args.fbx_out is None


def test_build_parser_accepts_optional_fbx_out():
    apply_motion = _load_apply_motion_module()
    parser = apply_motion.build_parser()

    args = parser.parse_args(
        ["--rig", "r.fbx", "--animation", "a.json", "--out", "o.blend", "--fbx-out", "o.fbx"]
    )

    assert args.fbx_out == "o.fbx"


def test_main_catches_exceptions_and_returns_1_instead_of_crashing(tmp_path, monkeypatch):
    # Confirmed against a real Blender build: blender.exe's own process
    # exit code stays 0 when a --python script raises an unhandled,
    # non-SystemExit exception. So main() must convert failures into an
    # explicit return code itself, or callers can never detect them.
    fake_bpy = FakeBpyModule()
    # No armature in fake_bpy.data.objects -> BlenderExecutor.import_rig
    # raises ValueError("no armature object found...").
    monkeypatch.setitem(sys.modules, "bpy", fake_bpy)

    rig_path = tmp_path / "rig.blend"
    rig_path.write_text("not a real blend file, just needs to exist for the path check")

    apply_motion = _load_apply_motion_module()
    exit_code = apply_motion.main(
        ["--", "--rig", str(rig_path), "--animation", "missing.json", "--out", str(tmp_path / "out.blend")]
    )

    assert exit_code == 1
