"""Blender-only FBX round-trip and deformation checks for the mannequin."""
from pathlib import Path
import hashlib
import json
import math
import bpy
from mathutils import Vector
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'examples/skinned_mannequin'
def load(path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=str(path))
    return next(o for o in bpy.context.scene.objects if o.type=='ARMATURE')
def rest(rig):
    return {b.name:{'parent':b.parent.name if b.parent else None,'matrix':[list(row) for row in rig.matrix_world@b.matrix_local]} for b in rig.data.bones}
original=rest(load(ROOT/'examples/BaseRig.fbx'))
rig=load(OUT/'BaseRig_Skinned.fbx'); exported=rest(rig)
assert original.keys()==exported.keys()
assert all(original[n]['parent']==exported[n]['parent'] for n in original)
error=max(abs(original[n]['matrix'][i][j]-exported[n]['matrix'][i][j]) for n in original for i in range(4) for j in range(4))
assert error<1e-5,error
skin=next(o for o in bpy.context.scene.objects if o.type=='MESH')
assert any(m.type=='ARMATURE' and m.object==rig for m in skin.modifiers)
assert all(1<=len(v.groups)<=4 and abs(sum(g.weight for g in v.groups)-1)<1e-5 for v in skin.data.vertices)
def coords():
    bpy.context.view_layer.update()
    obj=skin.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh=obj.to_mesh(); result=[skin.matrix_world@v.co for v in mesh.vertices]; obj.to_mesh_clear(); return result
baseline=coords(); motions={}
for name in ['lowerarm_l','calf_r','spine_03','neck_02','index_01_l']:
    pb=rig.pose.bones[name]; pb.rotation_mode='XYZ'; pb.rotation_euler.x=math.radians(45)
    moved=coords(); delta=[(a-b).length for a,b in zip(baseline,moved)]
    assert all(math.isfinite(c) for v in moved for c in v)
    assert max(delta)>.001,(name,max(delta))
    motions[name]={'rotation_local_x_degrees':45,'max_vertex_displacement_m':max(delta),'vertices_moved_over_1mm':sum(d>.001 for d in delta)}
    pb.rotation_euler.x=0
report={'scope':'asset_authoring_validation; no perception evaluation regime or model quality gate measured','source_sha256':hashlib.sha256((ROOT/'examples/BaseRig.fbx').read_bytes()).hexdigest(),'fbx_sha256':hashlib.sha256((OUT/'BaseRig_Skinned.fbx').read_bytes()).hexdigest(),'blender_version':bpy.app.version_string,'bones':len(exported),'vertices':len(skin.data.vertices),'polygons':len(skin.data.polygons),'bone_names_and_parents_preserved':True,'max_world_rest_matrix_absolute_error':error,'all_vertices_normalized_1_to_4_weights':True,'deformation_checks':motions,'limitations':['Low-poly overlapping body sections; not a welded production character surface.','No separate head or pelvis bones in source; existing hierarchy preserved.','Pose checks verify finite movement, not collision-free deformation for all motions.']}
(OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
