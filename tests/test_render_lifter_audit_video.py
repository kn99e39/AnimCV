import importlib.util
from pathlib import Path
import sys


_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "render_lifter_audit_video.py"


def _load_module():
    # The real script imports bpy; argument parsing and pure geometry helpers
    # are kept independent of it so they can be tested in the normal
    # project environment, matching test_render_blender_animation_video.py.
    sys.modules.setdefault("bpy", type("Bpy", (), {})())
    mathutils = type("Mathutils", (), {"Vector": tuple})()
    sys.modules.setdefault("mathutils", mathutils)
    spec = importlib.util.spec_from_file_location("render_lifter_audit_video", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_args_accepts_the_sequence_review_contract(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(sys, "argv", ["blender", "--", "--sequence", "window.json", "--out", "review.mp4",
                                      "--camera", "front", "--fps", "24"])

    args = module._args()

    assert args.sequence == "window.json"
    assert args.out == "review.mp4"
    assert args.camera == "front"
    assert args.fps == 24.0


def test_args_defaults_camera_to_three_quarter(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(sys, "argv", ["blender", "--", "--sequence", "window.json", "--out", "review.mp4"])

    args = module._args()

    assert args.camera == "three_quarter"
    assert args.fps is None
