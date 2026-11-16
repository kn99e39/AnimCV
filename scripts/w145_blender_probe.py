"""Fresh Blender W145 baseline and matched orientation diagnostic probes; never saves a blend."""
from __future__ import annotations
import sys,json,argparse,hashlib
from pathlib import Path


def main():
    import bpy
    from mathutils import Matrix
    ap=argparse.ArgumentParser();ap.add_argument('--blend',required=True);ap.add_argument('--spec',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args(sys.argv[sys.argv.index('--')+1:]);spec=json.loads(Path(a.spec).read_text('utf8'))
    assert hashlib.sha256(Path(a.blend).read_bytes()).hexdigest()==spec['original_A_sha256']
    assert not Path(a.out).exists()
    bpy.ops.wm.open_mainfile(filepath=str(Path(a.blend).resolve()))
    arms=[o for o in bpy.data.objects if o.type=='ARMATURE'];assert len(arms)==1;arm=arms[0]
    names=list(spec['probes'][0]['basis']);baseline=[]
    def capture():
        return {n:{'head':list(arm.matrix_world@arm.pose.bones[n].head),'tail':list(arm.matrix_world@arm.pose.bones[n].tail)} for n in names}
    for frame in range(1,241):
        bpy.context.scene.frame_set(frame);bpy.context.view_layer.update();baseline.append({'action':frame,'world':capture()})
    action=arm.animation_data.action.name;arm.animation_data.action=None
    probes=[]
    for probe in spec['probes']:
        arm.matrix_world=Matrix(probe['object'])
        for n,m in probe['basis'].items():arm.pose.bones[n].matrix_basis=Matrix(m)
        bpy.context.view_layer.update();probes.append({'tag':probe['tag'],'action':probe['action'],'world':capture()})
    Path(a.out).write_text(json.dumps({'schema':'animcv_w145_blender_probe_v1','action':action,'blender':bpy.app.version_string,
        'baseline':baseline,'probes':probes,'saved_blend':False}),encoding='utf8')
    print('W145_BLENDER_PROBES',len(probes))


if __name__=='__main__':main()
