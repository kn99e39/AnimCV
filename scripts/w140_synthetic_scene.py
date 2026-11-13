"""W140 (EXPERIMENTAL) Dataset A: deterministic static synthetic scene rendered in Blender (run with Blender 5.1):

    blender -b --factory-startup --python-exit-code 1 -P scripts/w140_synthetic_scene.py -- --out <dir>

Writes  <out>/images/f%04d.png  (estimator input: RGB only)
        <out>/input/camera.json, <out>/input/frames.json   (intrinsics + timestamps; NO poses)
        <out>/gt/camera_gt.txt     (TUM format: t tx ty tz qx qy qz qw = T_WC in the Blender world, OpenCV optical axes)
        <out>/gt/oracle_landmarks.json (marker sphere centres, for oracle projection tests ONLY)
        <out>/gt/segments.json     (stationary / translation / rotation / mixed frame ranges)
        <out>/scene.blend
Static textured geometry (procedural multi-scale coloured-rectangle images, seeded), Workbench FLAT shading (no lighting
change between frames), 640x480, lens 24 mm / sensor 36 mm (fx = fy = 426.667 px), 30 fps nominal timestamps.
"""
import json
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Euler, Matrix, Vector

argv = sys.argv[sys.argv.index("--") + 1:]
OUT = argv[argv.index("--out") + 1]
W, H, LENS, SENSOR, FPS = 640, 480, 24.0, 36.0, 30.0
SEG = {"stationary": [0, 19], "translation": [20, 69], "rotation": [70, 119], "mixed": [120, 139]}
N = 140

for d in ("images", "input", "gt"):
    os.makedirs(os.path.join(OUT, d), exist_ok=False)

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = "BLENDER_WORKBENCH"
scene.display.shading.light = "FLAT"
scene.display.shading.color_type = "TEXTURE"
scene.display.render_aa = "8"
scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage = W, H, 100
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGB"
scene.view_settings.view_transform = "Standard"
scene.world = bpy.data.worlds.new("w") if scene.world is None else scene.world
scene.world.color = (0.05, 0.05, 0.06)


def texture(seed, n=1024):
    rng = np.random.default_rng(seed)
    img = np.ones((n, n, 4), np.float32)
    img[..., :3] = rng.uniform(0.2, 0.8, 3)
    for size in (160, 64, 24, 9):
        for _ in range(int(2.2 * (n / size) ** 2 * 0.35)):
            x, y = rng.integers(0, n - size, 2)
            w_, h_ = rng.integers(max(2, size // 3), size + 1, 2)
            img[y:y + h_, x:x + w_, :3] = rng.uniform(0.15, 0.85, 3)   # never fully saturated: only oracle markers are
    im = bpy.data.images.new(f"tex{seed}", n, n)
    im.pixels.foreach_set(img.ravel())
    im.pack()
    mat = bpy.data.materials.new(f"m{seed}")
    mat.use_nodes = True
    nt = mat.node_tree
    tn = nt.nodes.new("ShaderNodeTexImage")
    tn.image = im
    tn.interpolation = "Linear"
    nt.links.new(tn.outputs["Color"], nt.nodes["Principled BSDF"].inputs["Base Color"])
    return mat


def add_box(name, loc, size, seed, rot=(0, 0, 0)):
    bpy.ops.mesh.primitive_cube_add(size=1, location=loc, rotation=rot)
    ob = bpy.context.active_object
    ob.name = name
    ob.scale = size
    bpy.ops.object.transform_apply(scale=True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.uv.cube_project(cube_size=float(max(size)))           # one texture copy per face: no repetitive pattern
    bpy.ops.object.mode_set(mode="OBJECT")
    ob.data.materials.append(texture(seed))
    return ob


add_box("floor", (0, 0, -0.05), (12, 12, 0.1), 1)
add_box("back_wall", (0, 5, 2), (12, 0.1, 4), 2)
add_box("left_wall", (-5, 0, 2), (0.1, 12, 4), 3)
add_box("right_wall", (5, 0, 2), (0.1, 12, 4), 4)
for i, (loc, size, rot) in enumerate([((-1.5, 2.0, 0.5), (1.0, 1.0, 1.0), (0, 0, 0.3)), ((1.2, 2.6, 0.75), (0.8, 1.2, 1.5), (0, 0, -0.5)),
                                      ((0.0, 3.5, 1.2), (2.0, 0.6, 2.4), (0, 0, 0.1)), ((-2.8, 3.2, 0.9), (0.7, 0.7, 1.8), (0, 0, 0.8)),
                                      ((2.6, 1.4, 0.4), (0.9, 0.9, 0.8), (0, 0, 1.1)), ((0.4, 1.0, 0.25), (0.5, 0.5, 0.5), (0.3, 0.2, 0.6))]):
    add_box(f"box{i}", loc, size, 10 + i, rot)
markers = {"red": ((-0.6, 1.6, 0.9), (1, 0, 0)), "green": ((1.0, 2.0, 1.9), (0, 1, 0)), "blue": ((-2.0, 2.4, 2.3), (0, 0, 1)),
           "yellow": ((2.2, 1.1, 1.3), (1, 1, 0))}
for nm, (loc, col) in markers.items():
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.035, location=loc, segments=32, ring_count=16)
    ob = bpy.context.active_object
    ob.name = f"marker_{nm}"
    m = bpy.data.materials.new(f"mk_{nm}")
    m.diffuse_color = (*col, 1)
    m.use_nodes = True
    m.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (*col, 1)
    ob.data.materials.append(m)

cam_data = bpy.data.cameras.new("cam")
cam_data.lens, cam_data.sensor_width, cam_data.sensor_fit = LENS, SENSOR, "HORIZONTAL"
cam_data.clip_start, cam_data.clip_end = 0.05, 100
cam = bpy.data.objects.new("cam", cam_data)
scene.collection.objects.link(cam)
scene.camera = cam

P0 = np.array([0.0, -3.0, 1.4])
TARGET = np.array([0.0, 2.5, 1.0])


def look_rot(pos, target):
    fwd = Vector((target - pos).tolist()).normalized()
    return fwd.to_track_quat("-Z", "Y").to_matrix()


def smooth(u):
    return 0.5 - 0.5 * math.cos(math.pi * min(max(u, 0.0), 1.0))


R0 = look_rot(P0, TARGET)
poses = []
for f in range(N):
    if f <= 19:
        p, R = P0.copy(), R0.copy()
    elif f <= 69:
        u = smooth((f - 20) / 49.0)
        p, R = P0 + u * np.array([1.6, 0.0, 0.35]), R0.copy()
    elif f <= 119:
        u = (f - 70) / 49.0
        p = P0 + np.array([1.6, 0.0, 0.35])
        yaw, pitch = math.radians(22.0) * math.sin(2 * math.pi * u), math.radians(8.0) * math.sin(4 * math.pi * u)
        R = Euler((0, 0, yaw)).to_matrix() @ R0 @ Euler((pitch, 0, 0)).to_matrix()
    else:
        u = smooth((f - 120) / 19.0)
        p = P0 + np.array([1.6, 0.0, 0.35]) + u * np.array([-0.8, 0.5, -0.2])
        R = Euler((0, 0, math.radians(-12.0) * u)).to_matrix() @ R0
    poses.append((p, Matrix(R)))

frames, gt_lines = [], ["# W140 synthetic GT: timestamp tx ty tz qx qy qz qw (T_WC, Blender world metres, OpenCV optical axes)"]
for f, (p, R) in enumerate(poses):
    cam.matrix_world = Matrix.Translation(Vector(p.tolist())) @ R.to_4x4()
    scene.frame_set(1)
    bpy.context.view_layer.update()
    scene.render.filepath = os.path.join(OUT, "images", f"f{f:04d}.png")
    bpy.ops.render.render(write_still=True)
    t = f / FPS
    frames.append({"frame_id": f, "timestamp": t, "image": f"f{f:04d}.png"})
    Rw = np.array(cam.matrix_world.to_3x3())
    Ro = Rw @ np.diag([1.0, -1.0, -1.0])
    q = Matrix(Ro.tolist()).to_quaternion()
    gt_lines.append(f"{t:.6f} {p[0]:.9f} {p[1]:.9f} {p[2]:.9f} {q.x:.9f} {q.y:.9f} {q.z:.9f} {q.w:.9f}")

fx = LENS / SENSOR * W
json.dump({"model": "PINHOLE", "width": W, "height": H, "fx": fx, "fy": fx, "cx": W / 2 - 0.5, "cy": H / 2 - 0.5,
           "pixel_convention": "OpenCV (integer = pixel centre); COLMAP cx = cx + 0.5", "distortion": [0, 0, 0, 0, 0],
           "source": "Blender camera lens 24 mm / sensor 36 mm horizontal fit, square pixels, centred principal point"},
          open(os.path.join(OUT, "input", "camera.json"), "w"), indent=1)
json.dump(frames, open(os.path.join(OUT, "input", "frames.json"), "w"), indent=1)
open(os.path.join(OUT, "gt", "camera_gt.txt"), "w").write("\n".join(gt_lines) + "\n")
json.dump({k: list(v[0]) for k, v in markers.items()}, open(os.path.join(OUT, "gt", "oracle_landmarks.json"), "w"), indent=1)
json.dump(SEG, open(os.path.join(OUT, "gt", "segments.json"), "w"), indent=1)
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, "scene.blend"))
print("W140_SYNTHETIC_DONE", N)
