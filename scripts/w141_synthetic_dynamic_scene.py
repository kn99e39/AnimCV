"""W141 (EXPERIMENTAL) Dataset C: controlled synthetic DYNAMIC scene (run with Blender 5.1):

    blender -b --factory-startup --python-exit-code 1 -P scripts/w141_synthetic_dynamic_scene.py -- \
        --w140 C:/Users/dna10/animcv-w140/data/synthetic_v3 --out <dir> [--frames 0,70]

Reuses, unchanged, the frozen W140 static scene (``<w140>/scene.blend``: static textured geometry, materials, Workbench FLAT
render settings, camera intrinsics) and the W140 camera trajectory (``<w140>/gt/camera_gt.txt``, read by the RENDERER only).
Adds two articulated, human-shaped, textured foreground occluders that walk independently of the camera.

Writes  <out>/images/f%04d.png       dynamic RGB frames (estimator input)
        <out>/input/camera.json, frames.json   byte copies of the W140 synthetic intrinsics / timestamps (no poses)
        <out>/gt/camera_gt.txt       byte copy of the W140 GT (T_WC, TUM format)
        <out>/gt/fg_masks/f%04d.png  exact foreground identity, 255 = human occluder (ORACLE: Condition D / diagnostics only)
        <out>/gt/static_boxes.json   world pose + half extents of every static box (landmark-consistency diagnostic)
        <out>/gt/humans.json         per-frame root position / heading of each occluder
        <out>/gt/segments.json       byte copy of the W140 segments
        <out>/check/clean_f%04d.png  frames re-rendered with the occluders hidden (background identity check vs W140)
        <out>/scene_dynamic.blend
The static environment, intrinsics, timestamps and camera trajectory are identical to W140; only the occluders differ.
"""
import json
import math
import os
import shutil
import sys

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

argv = sys.argv[sys.argv.index("--") + 1:]
W140 = argv[argv.index("--w140") + 1]
OUT = argv[argv.index("--out") + 1]
ONLY = [int(v) for v in argv[argv.index("--frames") + 1].split(",")] if "--frames" in argv else None
FPS = 30.0
CHECK_FRAMES = (0, 45, 95, 130)

for d in ("images", "input", "gt", "gt/fg_masks", "check"):
    os.makedirs(os.path.join(OUT, d), exist_ok=False)
for src, dst in (("input/camera.json", "input/camera.json"), ("input/frames.json", "input/frames.json"),
                 ("gt/camera_gt.txt", "gt/camera_gt.txt"), ("gt/segments.json", "gt/segments.json")):
    shutil.copyfile(os.path.join(W140, src), os.path.join(OUT, dst))

bpy.ops.wm.open_mainfile(filepath=os.path.join(W140, "scene.blend"))
scene = bpy.context.scene
cam = scene.camera
static_objs = [o for o in scene.objects if o.type == "MESH"]
boxes = []
for o in static_objs:
    if o.name.startswith("marker_"):
        continue
    bb = np.array([list(v) for v in o.bound_box])
    boxes.append({"name": o.name, "matrix_world": [list(r) for r in o.matrix_world],
                  "center_local": ((bb.max(0) + bb.min(0)) / 2).tolist(), "half_extents": ((bb.max(0) - bb.min(0)) / 2).tolist()})
json.dump(boxes, open(os.path.join(OUT, "gt", "static_boxes.json"), "w"), indent=1)


def texture_mat(seed, tint, n=512):
    """Seeded multi-scale rectangle texture (same family as W140, colours in [0.15, 0.85]) with a clothing tint."""
    rng = np.random.default_rng(seed)
    img = np.ones((n, n, 4), np.float32)
    base = np.asarray(tint, np.float32)
    img[..., :3] = base
    for size in (64, 24, 9):
        for _ in range(int(2.2 * (n / size) ** 2 * 0.35)):
            x, y = rng.integers(0, n - size, 2)
            w_, h_ = rng.integers(max(2, size // 3), size + 1, 2)
            img[y:y + h_, x:x + w_, :3] = np.clip(0.55 * base + 0.45 * rng.uniform(0.15, 0.85, 3), 0.15, 0.85)
    im = bpy.data.images.new(f"htex{seed}", n, n)
    im.pixels.foreach_set(img.ravel())
    im.pack()
    mat = bpy.data.materials.new(f"hm{seed}")
    mat.use_nodes = True
    nt = mat.node_tree
    tn = nt.nodes.new("ShaderNodeTexImage")
    tn.image = im
    nt.links.new(tn.outputs["Color"], nt.nodes["Principled BSDF"].inputs["Base Color"])
    return mat


def part(name, kind, dims, mat, parent, loc):
    if kind == "cyl":
        bpy.ops.mesh.primitive_cylinder_add(radius=dims[0], depth=dims[1], vertices=24, location=(0, 0, 0))
    elif kind == "sph":
        bpy.ops.mesh.primitive_uv_sphere_add(radius=dims[0], segments=24, ring_count=12, location=(0, 0, 0))
    else:
        bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0))
        bpy.context.active_object.scale = dims
        bpy.ops.object.transform_apply(scale=True)
    ob = bpy.context.active_object
    ob.name = name
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.uv.cube_project(cube_size=0.5)
    bpy.ops.object.mode_set(mode="OBJECT")
    ob.data.materials.append(mat)
    ob.parent = parent
    ob.location = loc
    return ob


def empty(name, parent, loc):
    e = bpy.data.objects.new(name, None)
    scene.collection.objects.link(e)
    e.parent = parent
    e.location = loc
    return e


def humanoid(tag, seed, shirt, trousers, skin):
    """Articulated human-shaped occluder (~1.75 m): local +Y = walking direction, +Z up, root on the ground."""
    m_shirt, m_trou, m_skin = texture_mat(seed, shirt), texture_mat(seed + 1, trousers), texture_mat(seed + 2, skin)
    root = empty(f"{tag}_root", None, (0, 0, 0))
    parts = [part(f"{tag}_pelvis", "box", (0.34, 0.20, 0.16), m_trou, root, (0, 0, 0.95)),
             part(f"{tag}_torso", "box", (0.40, 0.23, 0.52), m_shirt, root, (0, 0, 1.28)),
             part(f"{tag}_neck", "cyl", (0.05, 0.10), m_skin, root, (0, 0, 1.58)),
             part(f"{tag}_head", "sph", (0.11,), m_skin, root, (0, 0, 1.68))]
    joints = {}
    for side, sx in (("L", 1), ("R", -1)):
        hip = empty(f"{tag}_hip{side}", root, (0.10 * sx, 0, 0.90))
        knee = empty(f"{tag}_knee{side}", hip, (0, 0, -0.44))
        parts += [part(f"{tag}_thigh{side}", "cyl", (0.075, 0.44), m_trou, hip, (0, 0, -0.22)),
                  part(f"{tag}_shin{side}", "cyl", (0.06, 0.42), m_trou, knee, (0, 0, -0.21)),
                  part(f"{tag}_foot{side}", "box", (0.10, 0.24, 0.07), m_skin, knee, (0, 0.05, -0.44))]
        sh = empty(f"{tag}_shoulder{side}", root, (0.25 * sx, 0, 1.50))
        el = empty(f"{tag}_elbow{side}", sh, (0, 0, -0.30))
        parts += [part(f"{tag}_uparm{side}", "cyl", (0.05, 0.30), m_shirt, sh, (0, 0, -0.15)),
                  part(f"{tag}_forearm{side}", "cyl", (0.045, 0.28), m_skin, el, (0, 0, -0.14))]
        joints[side] = (hip, knee, sh, el)
    return root, joints, parts


HUMANS = [  # path: start xy, unit direction, speed m/s, gait phase
    {"tag": "h1", "seed": 501, "shirt": (0.70, 0.30, 0.25), "trousers": (0.25, 0.30, 0.55), "skin": (0.80, 0.62, 0.50),
     "start": (-0.9, -1.6), "dir": (1.0, 0.0), "speed": 0.85, "phase": 0.0},
    {"tag": "h2", "seed": 601, "shirt": (0.30, 0.60, 0.35), "trousers": (0.35, 0.30, 0.25), "skin": (0.55, 0.40, 0.30),
     "start": (2.2, 0.3), "dir": (-1.0, 0.0), "speed": 0.80, "phase": 0.37}]
built = []
for h in HUMANS:
    root, joints, parts = humanoid(h["tag"], h["seed"], h["shirt"], h["trousers"], h["skin"])
    built.append((h, root, joints, parts))
human_parts = [p for b in built for p in b[3]]


def pose_humans(f):
    t = f / FPS
    rec = []
    for h, root, joints, _ in built:
        d = np.asarray(h["dir"], float)
        xy = np.asarray(h["start"], float) + h["speed"] * t * d
        ph = 2 * math.pi * (1.8 * t / 2 + h["phase"])            # stride cycle 0.9 Hz (one step per leg per cycle)
        root.location = (float(xy[0]), float(xy[1]), 0.02 * abs(math.sin(ph)))
        root.rotation_euler = (0, 0, math.atan2(d[1], d[0]) - math.pi / 2)   # local +Y -> walking direction
        for side, sgn in (("L", 1), ("R", -1)):
            hip, knee, sh, el = joints[side]
            s = math.sin(ph) * sgn
            hip.rotation_euler = (math.radians(28) * s, 0, 0)
            knee.rotation_euler = (math.radians(-35) * max(0.0, -s), 0, 0)
            sh.rotation_euler = (math.radians(-22) * s, 0, 0)
            el.rotation_euler = (math.radians(25), 0, 0)
        rec.append({"tag": h["tag"], "root_xy": xy.tolist(), "heading_deg": math.degrees(math.atan2(d[1], d[0]))})
    return rec


gt = [l.split() for l in open(os.path.join(W140, "gt", "camera_gt.txt")) if l.strip() and not l.startswith("#")]
frames = json.load(open(os.path.join(W140, "input", "frames.json")))
assert len(gt) == len(frames)


def set_camera(i):
    t, x, y, z, qx, qy, qz, qw = (float(v) for v in gt[i])
    R_opt = Quaternion((qw, qx, qy, qz)).to_matrix()
    R_bcam = R_opt @ Matrix(((1, 0, 0), (0, -1, 0), (0, 0, -1)))     # optical -> Blender camera axes (self-inverse)
    cam.matrix_world = Matrix.Translation(Vector((x, y, z))) @ R_bcam.to_4x4()


disp = scene.display
rgb_state = (disp.shading.color_type, disp.render_aa, tuple(scene.world.color))


def render_rgb(path, hide_humans=False):
    disp.shading.color_type, disp.render_aa = rgb_state[0], rgb_state[1]
    scene.world.color = rgb_state[2]
    for p in human_parts:
        p.hide_render = hide_humans
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)


def render_mask(path):
    disp.shading.color_type, disp.render_aa = "OBJECT", "OFF"
    scene.world.color = (0, 0, 0)
    for o in scene.objects:
        if o.type == "MESH":
            o.color = (1, 1, 1, 1) if o in human_parts else (0, 0, 0, 1)
            o.hide_render = False
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)


hum = []
for i in range(len(frames)):
    if ONLY is not None and i not in ONLY:
        continue
    set_camera(i)
    hum.append({"frame_id": i, "humans": pose_humans(i)})
    scene.frame_set(1)
    bpy.context.view_layer.update()
    render_rgb(os.path.join(OUT, "images", f"f{i:04d}.png"))
    render_mask(os.path.join(OUT, "gt", "fg_masks", f"f{i:04d}.png"))
    if i in CHECK_FRAMES:
        render_rgb(os.path.join(OUT, "check", f"clean_f{i:04d}.png"), hide_humans=True)
json.dump(hum, open(os.path.join(OUT, "gt", "humans.json"), "w"), indent=1)
render_rgb(os.path.join(OUT, "check", "_last_state.png"))   # restores RGB state before saving
os.remove(os.path.join(OUT, "check", "_last_state.png"))
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, "scene_dynamic.blend"))
print("W141_SYNTHETIC_DYNAMIC_DONE", len(hum))
