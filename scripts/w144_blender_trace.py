"""W144 read-only fresh-process Blender evaluation and endpoint interventions.

No file is saved. Diagnostic replacements are not animation candidates.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]


def main():
    import bpy
    from mathutils import Matrix
    from blender.experimental_action_stage import _fcurves, _audit
    ap = argparse.ArgumentParser()
    ap.add_argument('--blend', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args(sys.argv[sys.argv.index('--')+1:])
    if Path(a.out).exists():
        raise RuntimeError('refusing to overwrite evidence')
    bpy.ops.wm.open_mainfile(filepath=str(Path(a.blend).resolve()))
    arms = [o for o in bpy.data.objects if o.type == 'ARMATURE']
    assert len(arms) == 1
    arm = arms[0]
    scene = bpy.context.scene
    names = sorted(b.name for b in arm.pose.bones if not any(
        x in b.name for x in ('index','middle','pinky','ring','thumb')))
    act = arm.animation_data.action
    digest, stored, census = _audit(act)
    def mat(m):
        return [list(r) for r in m]
    rest = {n: {'parent': arm.data.bones[n].parent.name if arm.data.bones[n].parent else None,
                'matrix_local': mat(arm.data.bones[n].matrix_local),
                'head_local': list(arm.data.bones[n].head_local),
                'tail_local': list(arm.data.bones[n].tail_local),
                'inherit_scale': arm.data.bones[n].inherit_scale,
                'use_inherit_rotation': arm.data.bones[n].use_inherit_rotation,
                'constraints': [c.type for c in arm.pose.bones[n].constraints]}
            for n in names}
    def capture(f, sub=0):
        mw = arm.matrix_world
        return {'action': f, 'subframe': sub, 'object': mat(mw),
                'basis': {n: mat(arm.pose.bones[n].matrix_basis) for n in names},
                'pose': {n: mat(arm.pose.bones[n].matrix) for n in names},
                'world': {n: {'head': list(mw @ arm.pose.bones[n].head),
                              'tail': list(mw @ arm.pose.bones[n].tail)} for n in names}}
    baseline, subframes = [], []
    for f in range(1, 241):
        scene.frame_set(f)
        bpy.context.view_layer.update()
        baseline.append(capture(f))
        if f < 240:
            for sub in (0.25,0.5,0.75,0.999):
                scene.frame_set(f, subframe=sub)
                subframes.append(capture(f,sub))
    # Detach only in memory. Exact baseline matrix_basis for every nonfinger bone.
    arm.animation_data.action = None
    def apply(state):
        arm.matrix_world = Matrix(state['object'])
        for n in names:
            arm.pose.bones[n].matrix_basis = Matrix(state['basis'][n])
        bpy.context.view_layer.update()
    def compact():
        mw = arm.matrix_world
        return {n: list(mw @ arm.pose.bones[n].head) for n in names}
    def intervene(b, c, factor):
        apply(b)
        if factor in ('object_rotation','object_translation','object_rotation+spine_01'):
            m = arm.matrix_world.copy()
            if factor == 'object_translation':
                m.translation = Matrix(c['object']).translation
            else:
                for i in range(3):
                    for j in range(3):
                        m[i][j] = c['object'][i][j]
            arm.matrix_world = m
        if factor.startswith('bone:') or factor == 'object_rotation+spine_01':
            n = factor[5:] if factor.startswith('bone:') else 'spine_01'
            arm.pose.bones[n].matrix_basis = Matrix(c['basis'][n])
        bpy.context.view_layer.update()
        return compact()
    factors = ['object_translation','object_rotation','bone:spine_01',
               'bone:clavicle_l','bone:clavicle_r','bone:upperarm_l',
               'bone:lowerarm_l','bone:thigh_l','bone:thigh_r','object_rotation+spine_01']
    counter = []
    for i in range(1,240):
        prev,cur = baseline[i-1],baseline[i]
        # Hold one endpoint factor at previous value (necessity) and inject
        # current factor into previous state (sufficiency). Full FK each time.
        counter.append({'to_action': i+1,
                        'hold': {k: intervene(cur,prev,k) for k in factors},
                        'inject': {k: intervene(prev,cur,k) for k in factors}})
    # Validate instrumentation on the actual hierarchy, known injected factors.
    base = baseline[0]
    apply(base)
    synthetic = {'base': capture(0)}
    import copy
    for tag in ('stable','root_translation','root_rotation','torso_rotation','upstream_pose_jump','parent_defect'):
        state = copy.deepcopy(base)
        if tag == 'root_translation':
            state['object'][0][3] += .1
        if tag in ('root_rotation','upstream_pose_jump','parent_defect'):
            r = Matrix.Rotation(0.3490658503988659,4,'Z')
            state['object'] = mat(r @ Matrix(base['object']))
        if tag in ('torso_rotation','parent_defect'):
            state['basis']['spine_01'] = mat(Matrix.Rotation(.3490658503988659,4,'Y') @ Matrix(base['basis']['spine_01']))
        apply(state)
        synthetic[tag] = capture(0)
    result = {'schema':'animcv_w144_blender_trace_v1','blender':bpy.app.version_string,
              'action':act.name,'value_digest':digest,'census':census,'rest':rest,
              'baseline':baseline,'subframes':subframes,'counterfactuals':counter,
              'synthetic':synthetic,'object_parent':arm.parent.name if arm.parent else None,
              'object_constraints':[c.type for c in arm.constraints],
              'nla_tracks':len(arm.animation_data.nla_tracks),
              'drivers':len(arm.animation_data.drivers)}
    Path(a.out).write_text(json.dumps(result),encoding='utf8')
    print('W144_TRACE_COMPLETE',a.out)


if __name__ == '__main__':
    main()
