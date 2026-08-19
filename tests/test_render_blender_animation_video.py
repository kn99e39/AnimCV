import importlib.util
from pathlib import Path
import sys


_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "render_blender_animation_video.py"


def _load_module():
    # The real script imports bpy, but argument parsing is deliberately kept
    # independent so it can be tested in the normal project environment.
    sys.modules.setdefault("bpy", type("Bpy", (), {})())
    mathutils = type("Mathutils", (), {"Vector": tuple})()
    sys.modules.setdefault("mathutils", mathutils)
    spec = importlib.util.spec_from_file_location("render_animation_video", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_args_accepts_review_video_contract(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(sys, "argv", ["blender", "--", "--blend", "result.blend", "--out", "review.mp4",
                                       "--camera", "side", "--frame-step", "2", "--hide-original-mesh"])

    args = module._args()

    assert args.blend == "result.blend"
    assert args.out == "review.mp4"
    assert args.camera == "side"
    assert args.frame_step == 2
    assert args.hide_original_mesh is True
