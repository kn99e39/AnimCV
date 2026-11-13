"""W140 POST-HOC visualisation (Blender side): third-person 3D perspective viewport of camera motion.

    blender -b --factory-startup --python-exit-code 1 -P scripts/w140_viewport_blender.py -- --spec <spec.json> --out <frames_dir>

The spec (written by scripts/w140_viewport.py from FROZEN W140 outputs) holds, per selected frame, the GT T_WC and the
Sim(3)-ALIGNED estimated T_WC (evaluation-only alignment; OpenCV optical axes), the camera intrinsics, and Sim(3)-aligned
landmarks.  Renders a Blender-viewport-like scene: XY floor grid at z = 0 (0.1 m minor / 1 m major, red X and green Y axis
lines), GT path (white) and ALIGNED ESTIMATE path (orange), FOV-true camera frustums with an 'up' triangle for the GT
(white) and the estimate (orange), and landmarks (grey points), from a FIXED external perspective camera (not the
reconstructed camera's view).  Workbench, 1280x720.  Writes <out>/v%04d.png (one per selected frame) + view.json.
"""
import json
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector

argv = sys.argv[sys.argv.index("--") + 1:]
SPEC = json.load(open(argv[argv.index("--spec") + 1]))
OUT = argv[argv.index("--out") + 1]
os.makedirs(OUT, exist_ok=False)
W, H = 1280, 720

bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.render.engine = "BLENDER_WORKBENCH"
sc.display.shading.light = "STUDIO"
sc.display.shading.color_type = "MATERIAL"
sc.display.shading.show_cavity = False
sc.display.render_aa = "8"
sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = W, H, 100
sc.render.image_settings.file_format = "PNG"
sc.view_settings.view_transform = "Standard"
sc.world = bpy.data.worlds.new("w")
sc.world.color = (0.046, 0.046, 0.046)                     # Blender viewport grey (#3d3d3d after the sRGB transform)


def mat(name, rgb):
    m = bpy.data.materials.new(name)
    m.diffuse_color = (*rgb, 1.0)
    return m


M = {"grid": mat("grid", (0.2, 0.2, 0.2)), "major": mat("major", (0.42, 0.42, 0.42)), "x": mat("x", (0.75, 0.08, 0.1)),
     "y": mat("y", (0.25, 0.6, 0.03)), "gt": mat("gt", (0.92, 0.92, 0.92)), "est": mat("est", (1.0, 0.3, 0.02)),
     "pts": mat("pts", (0.22, 0.3, 0.46))}


def edges_object(name, verts, edges, material, thickness):
    """Line segments as a bevelled POLY curve (renders as thin tubes in Workbench; a wireframe modifier needs faces)."""
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth = thickness / 2
    cu.bevel_resolution = 1
    cu.use_fill_caps = True
    verts = [tuple(map(float, v)) for v in verts]
    for a_, b_ in edges:
        sp = cu.splines.new("POLY")
        sp.points.add(1)
        sp.points[0].co = (*verts[a_], 1.0)
        sp.points[1].co = (*verts[b_], 1.0)
    ob = bpy.data.objects.new(name, cu)
    sc.collection.objects.link(ob)
    ob.data.materials.append(material)
    return ob


traj = np.asarray([f[k] for f in SPEC["frames"] for k in ("gt_T", "est_T") if f.get(k)])[:, :3, 3]
lo, hi = traj.min(0), traj.max(0)
centre = (lo + hi) / 2
ext = float(max(np.linalg.norm(hi - lo), 0.3))
scale_line = ext / 300.0
GZ = 0.0 if SPEC.get("grid_at_floor", False) else math.floor((lo[2] - 0.25) * 10) / 10   # horizontal reference plane height

# ---- floor grid (z = 0) ----
half = max(1.0, math.ceil((max(hi[0] - lo[0], hi[1] - lo[1]) / 2 + 0.5 * ext) * 2) / 2)
cx, cy = round(centre[0], 1), round(centre[1], 1)
for kind, step, mtl, th in (("grid", 0.1, M["grid"], scale_line * 0.9), ("major", 1.0, M["major"], scale_line * 2.0)):
    vs, es = [], []
    n = int(round(half / step))
    for i in range(-n, n + 1):
        x = (round(cx / step) + i) * step
        y = (round(cy / step) + i) * step
        for a, b in (((x, cy - half, GZ), (x, cy + half, GZ)), ((cx - half, y, GZ), (cx + half, y, GZ))):
            vs += [a, b]
            es.append((len(vs) - 2, len(vs) - 1))
    edges_object(f"floor_{kind}", vs, es, mtl, th)
if cy - half <= 0 <= cy + half:                            # world X axis (y = 0), red, if inside the grid
    edges_object("axis_x", [(cx - half, 0, GZ + 0.0005), (cx + half, 0, GZ + 0.0005)], [(0, 1)], M["x"], scale_line * 2.0)
if cx - half <= 0 <= cx + half:                            # world Y axis (x = 0), green, if inside the grid
    edges_object("axis_y", [(0, cy - half, GZ + 0.0005), (0, cy + half, GZ + 0.0005)], [(0, 1)], M["y"], scale_line * 2.0)

# ---- trajectories ----
for key, mtl in (("gt_T", M["gt"]), ("est_T", M["est"])):
    P = [np.asarray(f[key])[:3, 3] for f in SPEC["frames"] if f.get(key)]
    edges_object(f"path_{key}", P, [(i, i + 1) for i in range(len(P) - 1)], mtl, scale_line * 1.1)

# ---- landmarks ----
pts = np.asarray(SPEC.get("points", []), float)
if len(pts):
    me = bpy.data.meshes.new("pts")
    me.from_pydata([tuple(p) for p in pts], [], [])
    ob = bpy.data.objects.new("landmarks", me)
    sc.collection.objects.link(ob)
    ng = bpy.data.node_groups.new("pts", "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    gi, go = ng.nodes.new("NodeGroupInput"), ng.nodes.new("NodeGroupOutput")
    mp = ng.nodes.new("GeometryNodeMeshToPoints")
    ico = ng.nodes.new("GeometryNodeMeshIcoSphere")
    ICO = ico
    ico.inputs["Radius"].default_value = scale_line * 1.1
    ico.inputs["Subdivisions"].default_value = 1
    inst = ng.nodes.new("GeometryNodeInstanceOnPoints")
    real = ng.nodes.new("GeometryNodeRealizeInstances")
    sm = ng.nodes.new("GeometryNodeSetMaterial")
    sm.inputs["Material"].default_value = M["pts"]
    L = ng.links
    L.new(gi.outputs[0], mp.inputs["Mesh"])
    L.new(mp.outputs["Points"], inst.inputs["Points"])
    L.new(ico.outputs["Mesh"], inst.inputs["Instance"])
    L.new(inst.outputs["Instances"], real.inputs["Geometry"])
    L.new(real.outputs["Geometry"], sm.inputs["Geometry"])
    L.new(sm.outputs["Geometry"], go.inputs[0])
    ob.modifiers.new("pts", "NODES").node_group = ng

# ---- frustums (optical axes: +x right, +y down, +z forward), FOV-true ----
K = SPEC["K"]
d = 0.12 * ext + 0.03
wx, hy = d * K["width"] / (2 * K["fx"]), d * K["height"] / (2 * K["fy"])
fv = [(0, 0, 0), (-wx, -hy, d), (wx, -hy, d), (wx, hy, d), (-wx, hy, d), (-0.6 * wx, -1.15 * hy, d), (0.6 * wx, -1.15 * hy, d), (0, -1.7 * hy, d)]
fe = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1), (5, 6), (6, 7), (7, 5)]
fr = {"gt_T": edges_object("frustum_gt", fv, fe, M["gt"], scale_line * 2.2), "est_T": edges_object("frustum_est", fv, fe, M["est"], scale_line * 2.2)}

# ---- fixed external third-person viewer ----
gtR = [np.asarray(f["gt_T"])[:3, :3] for f in SPEC["frames"] if f.get("gt_T")]
fwd = np.mean([R[:, 2] for R in gtR], axis=0)
fwd_xy = np.array([fwd[0], fwd[1], 0.0])
fwd_xy = fwd_xy / (np.linalg.norm(fwd_xy) + 1e-9)
az = math.radians(SPEC.get("view_azimuth_deg", 55.0))
rot = np.array([[math.cos(az), -math.sin(az), 0], [math.sin(az), math.cos(az), 0], [0, 0, 1]])
back = rot @ (-fwd_xy)
elev = math.radians(SPEC.get("view_elevation_deg", 22.0))
target = np.array([centre[0], centre[1], centre[2] - 0.15 * (centre[2] - GZ)])
LENS, SENSOR = 32.0, 36.0
need = np.vstack([traj + o for o in (np.zeros(3), [d, 0, 0], [-d, 0, 0], [0, d, 0], [0, -d, 0], [0, 0, d], [0, 0, -d])])  # centres +- frustum size


def fits(dist):
    eye_ = target + back * dist * math.cos(elev) + np.array([0, 0, dist * math.sin(elev)])
    f_ = (target - eye_) / np.linalg.norm(target - eye_)
    r_ = np.cross(f_, [0, 0, 1.0])
    r_ /= np.linalg.norm(r_)
    u_ = np.cross(r_, f_)
    rel = need - eye_
    z_ = rel @ f_
    if np.any(z_ <= 0.05):
        return False, eye_
    fpx = LENS / SENSOR * W
    x_, y_ = fpx * (rel @ r_) / z_, fpx * (rel @ u_) / z_
    return bool(np.all(np.abs(x_) <= 0.42 * W) and np.all(np.abs(y_) <= 0.40 * H)), eye_


dist = 0.5
ok_, eye = fits(dist)
while not ok_ and dist < 50:
    dist *= 1.05
    ok_, eye = fits(dist)
if len(pts):
    ICO.inputs["Radius"].default_value = 0.0016 * dist                    # ~2 px landmarks at the viewing distance
cam_data = bpy.data.cameras.new("viewer")
cam_data.lens = LENS
cam_data.sensor_width = SENSOR
cam_data.clip_start, cam_data.clip_end = 0.01, 1000
viewer = bpy.data.objects.new("viewer", cam_data)
sc.collection.objects.link(viewer)
viewer.location = Vector(eye.tolist())
viewer.rotation_euler = (Vector(target.tolist()) - viewer.location).to_track_quat("-Z", "Y").to_euler()   # camera up = local +Y
sc.camera = viewer
json.dump({"grid_z_m": GZ, "eye": eye.tolist(), "target": target.tolist(), "lens_mm": LENS, "elevation_deg": math.degrees(elev), "distance_m": dist, "frustum_depth": d, "grid_half_m": half,
           "note": "fixed external viewer; world = GT world (z up); estimate shown after evaluation-only Sim(3)"},
          open(os.path.join(OUT, "view.json"), "w"), indent=1)

for i, f in enumerate(SPEC["frames"]):
    for key, ob in fr.items():
        T = f.get(key)
        ob.hide_render = T is None
        if T is not None:
            ob.matrix_world = Matrix(T)
    bpy.context.view_layer.update()
    sc.render.filepath = os.path.join(OUT, f"v{i:04d}.png")
    bpy.ops.render.render(write_still=True)
print("W140_VIEWPORT_DONE", len(SPEC["frames"]))
