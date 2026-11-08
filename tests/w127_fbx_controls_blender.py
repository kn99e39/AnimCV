#!/usr/bin/env python3
"""W127 focused-test helper (Blender side): builds the prospectively declared CONTROL inputs on the REAL BaseRig from a
synthetic-value fixture .blend.  Never used by the product commands.  Runs as
``blender -b --python-exit-code 1 -P tests/w127_fbx_controls_blender.py -- <mode> <args...>``.

  variant <src.blend> <out.blend> <name>       linear | mixed | perkey | axis | bezier | extra_action | wrong_rig
  multitake_fbx <src.blend> <out.fbx> <take_action>   the selected Action + an extra Action -> FBX with TWO takes
  wrongrig_fbx <src.blend> <out.fbx> <take_action>    hand_l renamed -> FBX of a NON-BaseRig with the given take name
  blanket_constant <src.blend> <out.blend>     every key of every F-curve CONSTANT (the W108 negative intervention)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _arm():
    import bpy
    return [o for o in bpy.data.objects if o.type == "ARMATURE"][0]


def _fcurves(act):
    from bpy_extras import anim_utils
    arm = _arm()
    cb = anim_utils.action_get_channelbag_for_slot(act, arm.animation_data.action_slot)
    return list(cb.fcurves)


def _set(fc, interp):
    for k in fc.keyframe_points:
        k.interpolation = interp


def variant(src, out, name):
    import bpy
    bpy.ops.wm.open_mainfile(filepath=src)
    arm = _arm()
    act = arm.animation_data.action
    ua = 'pose.bones["upperarm_l"].rotation_quaternion'
    for fc in _fcurves(act):
        if name == "linear":
            _set(fc, "LINEAR")
        elif name == "mixed" and fc.data_path == ua:
            _set(fc, "LINEAR")
        elif name == "perkey" and fc.data_path == ua:
            for i, k in enumerate(fc.keyframe_points):
                k.interpolation = "LINEAR" if i % 2 == 0 else "CONSTANT"
        elif name == "axis" and fc.data_path == ua and fc.array_index > 0:
            _set(fc, "LINEAR")
        elif name == "bezier" and fc.data_path == ua:
            _set(fc, "BEZIER")
    if name == "extra_action":
        ex = act.copy()
        ex.name = "AnimCV_EXTRA_unassigned_take"
        ex.use_fake_user = True
    if name == "wrong_rig":
        arm.data.bones["hand_l"].name = "hand_left"
    if Path(out).exists():
        raise RuntimeError("refusing to overwrite")
    bpy.ops.wm.save_as_mainfile(filepath=out)


def _export(out, take_action, keep_extra):
    import bpy
    import blender.experimental_fbx_interchange as X
    arm = _arm()
    act = arm.animation_data.action
    for a in [a for a in bpy.data.actions if a != act]:
        bpy.data.actions.remove(a)
    act.name = take_action
    if keep_extra:
        ex = act.copy()
        ex.name = take_action + "_EXTRA"
    res = bpy.ops.export_scene.fbx(filepath=out, **X.EXPORT_KWARGS)
    if "FINISHED" not in res:
        raise RuntimeError(str(res))


def multitake_fbx(src, out, take_action):
    import bpy
    bpy.ops.wm.open_mainfile(filepath=src)
    _export(out, take_action, keep_extra=True)


def wrongrig_fbx(src, out, take_action):
    import bpy
    bpy.ops.wm.open_mainfile(filepath=src)
    _arm().data.bones["hand_l"].name = "hand_left"
    _export(out, take_action, keep_extra=False)


def blanket_constant(src, out):
    import bpy
    bpy.ops.wm.open_mainfile(filepath=src)
    for fc in _fcurves(_arm().animation_data.action):
        _set(fc, "CONSTANT")
    bpy.ops.wm.save_as_mainfile(filepath=out)


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:]
    try:
        {"variant": variant, "multitake_fbx": multitake_fbx, "wrongrig_fbx": wrongrig_fbx,
         "blanket_constant": blanket_constant}[argv[0]](*argv[1:])
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
