"""Build a simple low-poly mannequin using BaseRig's unchanged skeleton.

Run with Blender: blender -b --python scripts/build_skinned_mannequin.py
All coordinates below are centimetres in the imported armature's local space.
No external models, textures, or Python packages are required.
"""
from pathlib import Path
import hashlib
import json
import math
import bpy
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'examples/skinned_mannequin'
OUT.mkdir(parents=True, exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=str(ROOT / 'examples/BaseRig.fbx'))
rig = next(o for o in bpy.context.scene.objects if o.type == 'ARMATURE')
bones = rig.data.bones
parts = []

def ellipsoid(name, center, scale, direction=None):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=8, ring_count=4, location=center)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    if direction is not None:
        obj.rotation_euler = Vector(direction).to_track_quat('Z', 'Y').to_euler()
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    parts.append(obj)
    return obj

def segment(name, a, b, radius, depth=None):
    a, b = Vector(a), Vector(b)
    return ellipsoid(name, (a+b)/2, (radius, depth or radius, (b-a).length/2+radius*.55), b-a)

# Sparse flat-shaded rings create an intentionally geometric mannequin.
def torso():
    rings=[(84,11,7),(94,15,9),(106,12,7),(119,13,8),(139,19,9),(146,15,7)]
    verts=[]
    for z,rx,ry in rings:
        for i in range(8):
            angle=2*math.pi*i/8+math.pi/8
            verts.append((rx*math.cos(angle),ry*math.sin(angle),z))
    faces=[tuple(reversed(range(8)))]
    for j in range(len(rings)-1):
        for i in range(8):
            k=j*8+i; n=j*8+(i+1)%8
            faces.append((k,n,n+8,k+8))
    faces.append(tuple(range(40,48)))
    mesh=bpy.data.meshes.new('TorsoGeometry'); mesh.from_pydata(verts,[],faces); mesh.update()
    obj=bpy.data.objects.new('Torso',mesh); bpy.context.collection.objects.link(obj); parts.append(obj)

torso()
segment('Neck',(0,1,145),(0,0,160),4)
ellipsoid('Head',(0,0,169),(8,7.5,11))
# One small wedge-like nose and forward feet are the only orientation details.
ellipsoid('Nose',(0,-7,169),(1.5,2.5,2))
for suffix in ['l','r']:
    for name,r in [('upperarm',5.2),('lowerarm',4.1),('thigh',7.3),('calf',5.2)]:
        b=bones[f'{name}_{suffix}']
        segment(name,b.head_local,b.tail_local,r)
    for name,r in [('upperarm',4.8),('lowerarm',3.8),('calf',4.7)]:
        ellipsoid('Joint',bones[f'{name}_{suffix}'].head_local,(r,r,r))
    hand=bones[f'hand_{suffix}'].head_local
    tip=bones[f'middle_02_{suffix}'].tail_local
    segment('Mitten',hand,tip,3.2,2.1)
    thumb=bones[f'thumb_01_{suffix}']
    segment('Thumb',thumb.head_local,bones[f'thumb_02_{suffix}'].tail_local,1.5)
    ankle=bones[f'foot_{suffix}'].head_local
    ellipsoid('Foot',(ankle.x,ankle.y-5,4.5),(4.8,10.5,4.5))
    segment('Ankle',ankle,(ankle.x,ankle.y,4),3)

bpy.ops.object.select_all(action='DESELECT')
for o in parts: o.select_set(True)
bpy.context.view_layer.objects.active=parts[0]
bpy.ops.object.join()
skin=bpy.context.object
skin.name='AnimCV_Mannequin_Skin'
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
# Keep the sparse authored topology and visible planar faces.
for p in skin.data.polygons: p.use_smooth=False

# Region-restricted distance weights avoid opposite-limb/control-bone influence.
active=[b for b in bones if b.name not in {'interaction','center_of_mass'}]
groups={b.name:skin.vertex_groups.new(name=b.name) for b in active}
def distance(p,b):
    a=b.head_local; d=b.tail_local-a
    t=max(0,min(1,(p-a).dot(d)/d.length_squared))
    return (p-(a+t*d)).length
for v in skin.data.vertices:
    p=v.co; side='l' if p.x>=0 else 'r'
    if p.z>160:
        weights=[('neck_02',1.0)]
    else:
        candidates=active
        if p.z<82:
            candidates=[b for b in active if b.name in {f'thigh_{side}',f'calf_{side}',f'foot_{side}'}]
        elif abs(p.x)>30:
            candidates=[b for b in active if b.name.endswith('_'+side) and not b.name.startswith(('thigh','calf','foot'))]
        else:
            candidates=[b for b in active if not any(f in b.name for f in ['metacarpal','index','middle','ring','pinky','thumb','hand','foot','calf'])]
        nearest=sorted(((distance(p,b),b.name) for b in candidates))[:4]
        # Compact support limits blending to nearby joints; max four influences.
        raw=[(n,math.exp(-((d-nearest[0][0])/3.5)**2)/(d+1)**2) for d,n in nearest]
        total=sum(w for _,w in raw)
        weights=[(n,w/total) for n,w in raw if w/total>1e-5]
        total=sum(w for _,w in weights); weights=[(n,w/total) for n,w in weights]
    for n,w in weights: groups[n].add([v.index],w,'REPLACE')
skin.parent=rig
skin.matrix_world=rig.matrix_world.copy()
mod=skin.modifiers.new('BaseRig skin deformation','ARMATURE'); mod.object=rig
mat=bpy.data.materials.new('Slate blue matte'); mat.diffuse_color=(.16,.32,.44,1)
mat.use_nodes=True
bsdf=mat.node_tree.nodes.get('Principled BSDF'); bsdf.inputs['Base Color'].default_value=mat.diffuse_color; bsdf.inputs['Roughness'].default_value=.65
skin.data.materials.append(mat)
rig.show_in_front=True
bpy.context.view_layer.update()

bpy.ops.object.select_all(action='DESELECT'); rig.select_set(True); skin.select_set(True)
bpy.context.view_layer.objects.active=rig
bpy.ops.export_scene.fbx(filepath=str(OUT/'BaseRig_Skinned.fbx'),use_selection=True,object_types={'ARMATURE','MESH'},add_leaf_bones=False,bake_anim=False,use_armature_deform_only=False)

# CPU-only preview rendering is asset authoring, not a GPU experiment.
scene=bpy.context.scene
scene.render.engine='CYCLES'; scene.cycles.device='CPU'; scene.cycles.samples=24
scene.render.resolution_x=700; scene.render.resolution_y=900; scene.render.resolution_percentage=100
scene.world=bpy.data.worlds.new('Studio'); scene.world.use_nodes=True
scene.world.node_tree.nodes['Background'].inputs[0].default_value=(.15,.18,.22,1)
scene.world.node_tree.nodes['Background'].inputs[1].default_value=.5
for loc,power,size in [((2,-3,4),450,3),((-3,-1,2),250,3),((0,3,3),500,2)]:
    bpy.ops.object.light_add(type='AREA',location=loc)
    light=bpy.context.object; light.data.energy=power; light.data.shape='DISK'; light.data.size=size
    light.rotation_euler=(Vector((0,0,1))-light.location).to_track_quat('-Z','Y').to_euler()
bpy.ops.object.camera_add(location=(2.5,-5,2.1))
camera=bpy.context.object; camera.data.type='ORTHO'; camera.data.ortho_scale=2.1; scene.camera=camera
camera.rotation_euler=(Vector((0,0,.93))-camera.location).to_track_quat('-Z','Y').to_euler()
bpy.context.preferences.filepaths.save_version=0
bpy.ops.wm.save_as_mainfile(filepath=str(OUT/'BaseRig_Skinned.blend'))
for label,loc in [('front',(0,-5,1.05)),('back',(0,5,1.05)),('three_quarter',(2.5,-5,2.1))]:
    camera.location=loc; camera.rotation_euler=(Vector((0,0,.93))-camera.location).to_track_quat('-Z','Y').to_euler()
    scene.render.filepath=str(OUT/f'preview_{label}.png'); bpy.ops.render.render(write_still=True)
print('MANNEQUIN_BUILD_COMPLETE',len(skin.data.vertices),len(skin.data.polygons))
