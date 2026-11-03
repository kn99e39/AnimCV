#!/usr/bin/env python3
"""W107 evaluation contract: FBX Action round-trip failure ownership (pure Python; evidence only).

Shared by the host stages, the Blender stages and the tests.  Nothing here touches Blender, a rig file or a
production module.  It defines

  * the frozen V0 exporter arguments (parsed from the unchanged ``src/blender/export.py``) and the one-factor
    diagnostic variants V0d / V1 / V2 (each differs from V0 in at most one declared operator property),
  * the V3 synthetic world-equivalent 2x2 design (object static/animated x object scale 1/0.01) and its motion,
  * the parity metrics (W106 gate metric reproduced exactly, plus Euclidean / object / armature-space / local),
  * the predeclared tolerances, the causal-support rule and the CASE mapping.

No production exporter or importer function is modified or monkey-patched; diagnostic variants are separate
``bpy.ops.export_scene.fbx`` calls on scratch files only.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path

import numpy as np

SCHEMA = "w107_fbx_ownership_v1"

REAL_CASES = ("complete_control", "arm_natural_most", "real_clip")   # W106 cases, fixed; never substituted
W106_SYNTHETIC = "w106_synthetic"                                    # W106 minimal armature; separately labelled
REAL_ACTIONS = {c: f"W91 {c} FIXED" for c in REAL_CASES}             # M_Main slot; never R11 RUNTIME
W106_SYNTHETIC_ACTION = "SYN baseline"

# --- exporter variants ------------------------------------------------------------------------------------------
# V0 calls the real production function ``blender.export.export_fbx``.  V0d repeats its arguments verbatim through a
# separate diagnostic operator call (negative control: the diagnostic path must equal the production path).  V1 and
# V2 each change exactly one property; values are the documented control values, chosen before any outcome.
VARIANTS = {
    "V0": {"path": "production export_fbx", "override": {}},
    "V0d": {"path": "diagnostic operator call", "override": {}},
    "V1": {"path": "diagnostic operator call", "override": {"bake_anim_simplify_factor": 0.0},
           "factor": "default bake key simplification",
           "rationale": "0.0 is the operator's documented no-simplification value (keeps every baked key); the "
                        "Blender default is 1.0"},
    "V2": {"path": "diagnostic operator call", "override": {"bake_anim_use_all_actions": False},
           "factor": "Action/take enumeration (export all Actions as takes)",
           "rationale": "False exports only the object's assigned Action/scene take instead of every compatible "
                        "stored Action; the Blender default is True"},
}
VARIANT_ORDER = ("V0", "V0d", "V1", "V2")
CAUSAL_VARIANTS = ("V1", "V2")

IMPORTER = {"call": "bpy.ops.import_scene.fbx(filepath=...) with every other property at its default",
            "scene": "bpy.ops.wm.read_factory_settings(use_empty=True)"}
IMPORTER_PROBE = {"status": "NOT_RUN",
                  "reason": "no single importer property could be predeclared with a mechanism specific to the W106 "
                            "drift independently of the exporter variants; importer ownership is recorded as "
                            "unresolved except for what the exporter-only variants imply"}

ROTATION_CONVERSION = {"status": "NOT_INDEPENDENTLY_MANIPULABLE",
                       "reason": "FBX stores Euler curves; the Blender exporter has no operator property that keeps "
                                 "quaternions, so rotation conversion cannot be isolated by a one-factor argument "
                                 "change; it is characterised only by residual error"}


def export_fbx_kwargs(source_text):
    """Literal keyword arguments of the single ``bpy.ops.export_scene.fbx`` call in ``src/blender/export.py``."""
    tree = ast.parse(source_text)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "fbx" and isinstance(n.func.value, ast.Attribute) and n.func.value.attr == "export_scene"]
    if len(calls) != 1:
        raise ValueError(f"expected one export_scene.fbx call, found {len(calls)}")
    out = {}
    for kw in calls[0].keywords:
        if kw.arg == "filepath":
            continue
        out[kw.arg] = ast.literal_eval(kw.value)
    return out


def variant_kwargs(v0_kwargs, variant):
    """Full diagnostic operator kwargs (without filepath) for a variant; V0 itself is the production function."""
    kw = {k: (sorted(v) if isinstance(v, set) else v) for k, v in v0_kwargs.items()}
    ov = VARIANTS[variant]["override"]
    clash = set(ov) & set(kw)
    if clash:
        raise ValueError(f"{variant} overrides a property export_fbx already sets: {sorted(clash)}")
    kw.update(ov)
    return kw


def one_factor_ok(v0_kwargs, defaults):
    """Every causal variant differs from V0's effective arguments in exactly one property, and that property is set
    to a value different from its runtime default (otherwise the variant would be a null intervention)."""
    res = {}
    for v in CAUSAL_VARIANTS:
        ov = VARIANTS[v]["override"]
        (k, val), = ov.items()
        res[v] = {"property": k, "value": val, "default": defaults.get(k),
                  "supported": k in defaults, "not_in_v0_kwargs": k not in v0_kwargs,
                  "differs_from_default": k in defaults and defaults[k] != val}
        res[v]["ok"] = res[v]["supported"] and res[v]["not_in_v0_kwargs"] and res[v]["differs_from_default"]
    return res


# --- V3 synthetic world-equivalent 2x2 ---------------------------------------------------------------------------
V3_CELLS = ("static_s1", "static_s001", "animated_s1", "animated_s001")   # object motion x object scale
V3_FRAMES = [float(f) for f in range(1, 50)]                              # keys on every frame (dense, like real_clip)
V3_SCALE = {"static_s1": 1.0, "static_s001": 0.01, "animated_s1": 1.0, "animated_s001": 0.01}
V3_ANIMATED = {"static_s1": False, "static_s001": False, "animated_s1": True, "animated_s001": True}
V3_REFERENCE = "static_s1"
# bones in reference (scale-1) armature units; geometry is multiplied by 1/scale in the scale-0.01 cells
V3_BONES = [("base", (0, 0, 0), (0, 0, 0.2), None),
            ("spine", (0, 0, 0.2), (0, 0, 0.6), "base"),
            ("upperarm_l", (0, 0, 0.6), (0.3, 0, 0.6), "spine"),
            ("lowerarm_l", (0.3, 0, 0.6), (0.6, 0, 0.6), "upperarm_l"),
            ("hand_l", (0.6, 0, 0.6), (0.7, 0, 0.6), "lowerarm_l"),
            ("upperarm_r", (0, 0, 0.6), (-0.3, 0, 0.6), "spine"),
            ("thigh_l", (0.1, 0, 0), (0.1, 0, -0.45), "base"),
            ("calf_l", (0.1, 0, -0.45), (0.1, 0, -0.9), "thigh_l")]
V3_CHILD_BONES = [b[0] for b in V3_BONES if b[0] != "base"]
V3_BASE_PREROT = ((1.0, 0.0, 0.0), 90.0)     # constant 90 deg about X, as in an FBX-imported rig object


def qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw*bw - ax*bx - ay*by - az*bz, aw*bx + ax*bw + ay*bz - az*by,
            aw*by - ax*bz + ay*bw + az*bx, aw*bz + ax*by - ay*bx + az*bw)


def qaxis_angle(axis, deg):
    n = math.sqrt(sum(c*c for c in axis))
    h = math.radians(deg)/2
    s = math.sin(h)/n
    return (math.cos(h), axis[0]*s, axis[1]*s, axis[2]*s)


def v3_global_rotation(frame):
    """Rigid global rotation G(t) = Rx(90) * R(t): a yaw sweep plus small (sub-degree) tilts, deterministic."""
    t = frame - 1.0
    yaw = 35.0*math.sin(0.11*t)
    tilt_x = 0.6*math.sin(0.73*t + 0.4)
    tilt_y = 0.4*math.sin(1.31*t + 1.1)
    r = qmul(qaxis_angle((0, 0, 1), yaw), qmul(qaxis_angle((1, 0, 0), tilt_x), qaxis_angle((0, 1, 0), tilt_y)))
    return qmul(qaxis_angle(*V3_BASE_PREROT), r)


def v3_child_rotation(bone_index, frame):
    """Local rotation of a non-base bone; identical in every cell (local rotations are scale-invariant)."""
    t = frame - 1.0
    j = bone_index + 1
    ang = 25.0*math.sin(0.09*j*t + 0.5*j) + 0.3*math.sin(1.7*t + j)
    axis = (math.sin(j), math.cos(2*j + 1), 0.3*j)
    return qaxis_angle(axis, ang)


# --- tolerances (fixed a priori; units metres unless noted) ------------------------------------------------------
TOL = {
    "fbx_rt_parity_m": 1e-3,          # W106 gate, unchanged: max abs world head/tail coordinate, integer frames
    "deterministic_m": 1e-9,          # V0 vs W106 fbx_n0, repeats, V0d vs V0 (bitwise expected)
    "source_fidelity_m": 0.0,         # W107 source sample vs W106 n0_reopen (bitwise expected)
    "substantial_reduction_ratio": 0.5,   # a factor is substantial on a case if V_F <= 0.5 * V0
    "collateral_bone_increase_m": 1e-4,   # no bone's per-bone max error may grow by more than 0.1 mm vs V0
    "v3_equivalence_m": 1e-6,         # V3 cells' Blender world heads/tails vs the reference cell
    "synthetic_control_m": 1e-5,      # W106 synthetic V0 parity (W106 measured ~1e-6)
}


# --- metrics ------------------------------------------------------------------------------------------------------
def _idx(names_src, names_imp):
    pos = {n: i for i, n in enumerate(names_imp)}
    common = [n for n in names_src if n in pos]
    return common, [names_src.index(n) for n in common], [pos[n] for n in common]


def rot_angle_deg(ma, mb):
    ra = ma[:3, :3]/np.linalg.norm(ma[:3, :3], axis=0)
    rb = mb[:3, :3]/np.linalg.norm(mb[:3, :3], axis=0)
    return float(np.degrees(np.arccos(np.clip((np.trace(ra.T @ rb) - 1)/2, -1, 1))))


def quat_angle_deg(a, b):
    d = np.abs(np.sum(a*b, axis=-1))/(np.linalg.norm(a, axis=-1)*np.linalg.norm(b, axis=-1))
    return np.degrees(2*np.arccos(np.clip(d, 0, 1)))


def _stats(x):
    x = np.asarray(x, dtype=float).ravel()
    if x.size == 0:
        return {"max": None, "p95": None, "median": None}
    return {"max": float(x.max()), "p95": float(np.percentile(x, 95)), "median": float(np.median(x))}


def parity(src, imp, rest_comparable=None):
    """Compare an imported sample set with its Blender source at identical sample times.

    ``src``/``imp``: dicts of arrays as written by the W106 ``sample`` helper (times, bones, world_head, world_tail,
    world_rot, object_matrix, local_quat, local_loc).  Returns the W106 gate metric (``gate_max_abs_coord_m``: max
    abs coordinate difference of heads and tails over all source bones at integer frames) plus diagnostics.
    """
    if not np.array_equal(src["times"], imp["times"]):
        raise ValueError("sample times differ")
    names_s = [str(b) for b in src["bones"]]
    names_i = [str(b) for b in imp["bones"]]
    common, ia, ib = _idx(names_s, names_i)
    t = src["times"]
    integer = t == np.floor(t)
    half = ~integer
    out = {"bones_compared": len(common), "missing_in_import": sorted(set(names_s) - set(names_i)),
           "integer_samples": int(integer.sum()), "half_samples": int(half.sum())}

    def coord(mask):
        dh = src["world_head"][mask][:, ia] - imp["world_head"][mask][:, ib]
        dt = src["world_tail"][mask][:, ia] - imp["world_tail"][mask][:, ib]
        return dh, dt
    dh, dt = coord(integer)
    out["gate_max_abs_coord_m"] = float(max(np.abs(dh).max(), np.abs(dt).max())) if dh.size else 0.0
    eu = np.maximum(np.linalg.norm(dh, axis=-1), np.linalg.norm(dt, axis=-1))          # (T_int, B)
    out["euclid_integer_m"] = _stats(eu)
    per_frame = eu.max(1)
    out["euclid_per_frame_m"] = _stats(per_frame)
    out["frames_gt_1mm"] = int((per_frame > TOL["fbx_rt_parity_m"]).sum())
    per_bone = np.maximum(np.abs(dh).max(axis=(0, 2)), np.abs(dt).max(axis=(0, 2)))     # gate metric per bone
    out["per_bone_max_abs_coord_m"] = {common[k]: float(per_bone[k]) for k in range(len(common))}
    order = np.argsort(-per_bone)[:5]
    out["worst_bones"] = [[common[k], float(per_bone[k])] for k in order]
    fr = t[integer]
    out["worst_frames"] = [[float(fr[k]), float(per_frame[k])] for k in np.argsort(-per_frame)[:5]]
    if half.any():
        h1, h2 = coord(half)
        out["half_frame_max_abs_coord_m"] = float(max(np.abs(h1).max(), np.abs(h2).max()))
    else:
        out["half_frame_max_abs_coord_m"] = None
    # armature object
    om_s, om_i = src["object_matrix"][integer], imp["object_matrix"][integer]
    out["object_rotation_deg"] = _stats([rot_angle_deg(a, b) for a, b in zip(om_s, om_i)])
    out["object_translation_m"] = _stats(np.linalg.norm(om_s[:, :3, 3] - om_i[:, :3, 3], axis=-1))
    sc_s = np.linalg.norm(om_s[:, :3, :3], axis=1)
    sc_i = np.linalg.norm(om_i[:, :3, :3], axis=1)
    out["object_scale_max_abs_diff"] = float(np.abs(sc_s - sc_i).max())
    # armature space (each side expressed in its own object frame), rescaled to world metres by the source scale
    arm = []
    for k in range(om_s.shape[0]):
        inv_s, inv_i = np.linalg.inv(om_s[k]), np.linalg.inv(om_i[k])
        ps = (inv_s[:3, :3] @ src["world_tail"][integer][k][ia].T).T + inv_s[:3, 3]
        pi = (inv_i[:3, :3] @ imp["world_tail"][integer][k][ib].T).T + inv_i[:3, 3]
        arm.append(np.linalg.norm(ps - pi, axis=-1)*float(np.linalg.norm(om_s[k][:3, 0])))
    arm = np.array(arm)
    out["armature_space_tail_m"] = _stats(arm.max(1) if arm.size else [])
    # local bone channels (comparable only where the imported rest matrix equals the source rest matrix)
    lq = quat_angle_deg(src["local_quat"][integer][:, ia], imp["local_quat"][integer][:, ib])
    keep = [k for k, n in enumerate(common) if rest_comparable is None or rest_comparable.get(n, False)]
    out["local_quat_angle_deg"] = _stats(lq[:, keep] if keep else [])
    out["local_quat_bones_comparable"] = len(keep)
    return out


def arrays_max_diff(a, b, keys=("world_head", "world_tail", "world_rot", "object_matrix", "local_quat", "local_loc",
                                "local_scale")):
    """Max abs difference over the sampled arrays of two runs of the same condition (same bones/times required)."""
    if [str(x) for x in a["bones"]] != [str(x) for x in b["bones"]] or not np.array_equal(a["times"], b["times"]):
        return math.inf
    return float(max(np.abs(np.asarray(a[k]) - np.asarray(b[k])).max() for k in keys))


def collateral(v0_per_bone, vf_per_bone):
    """Bones whose gate metric grows by more than the declared tolerance under a variant."""
    tol = TOL["collateral_bone_increase_m"]
    return sorted(b for b in v0_per_bone if b in vf_per_bone and vf_per_bone[b] - v0_per_bone[b] > tol)


# --- causal-support rule and CASE mapping ---------------------------------------------------------------------
def factor_support(per_case):
    """``per_case``: case -> {v0, vf, collateral (list), repeat_ok, action_identity_ok}.

    Per case: substantial iff vf <= ratio*v0; clean iff no collateral, repeat agrees and the selected Action is the
    declared one.  The factor is *supported* iff every real case is substantial and clean and at least one real case
    reaches vf <= 1 mm.  *Mixed* iff some but not all cases are substantial (or substantial but not clean).
    """
    r = TOL["substantial_reduction_ratio"]
    res = {"cases": {}}
    for c, v in per_case.items():
        sub = v["vf"] is not None and v["v0"] > 0 and v["vf"] <= r*v["v0"]
        clean = not v["collateral"] and v["repeat_ok"] and v["action_identity_ok"]
        res["cases"][c] = {"reduction_ratio": (None if v["vf"] is None else 1 - v["vf"]/v["v0"]),
                           "substantial": bool(sub), "clean": bool(clean),
                           "passes_1mm": bool(v["vf"] is not None and v["vf"] <= TOL["fbx_rt_parity_m"])}
    cs = res["cases"].values()
    res["all_substantial_clean"] = all(x["substantial"] and x["clean"] for x in cs)
    res["any_1mm"] = any(x["passes_1mm"] for x in cs)
    res["all_1mm"] = all(x["passes_1mm"] for x in cs)
    res["supported"] = bool(res["all_substantial_clean"] and res["any_1mm"])
    res["mixed"] = bool(not res["supported"] and any(x["substantial"] for x in cs))
    return res


def decide_case(pre):
    """``pre`` keys: preflight_ok, source_fidelity_ok, v0_reproduces_w106, v0_deterministic, v0d_equals_v0,
    actions_unmodified, originals_unchanged, synthetic_control_ok, v3_status ('TESTED'|'NOT_TESTABLE'),
    variants_testable (dict variant -> bool), factors (dict variant -> factor_support result).

    D: preflight / identity / source fidelity / V0 reproduction / determinism / diagnostic-path equality / mutation.
    A: >= 1 causal factor supported, synthetic control passes, V3 tested.
    C: a factor is mixed, a causal variant is untestable, V3 not testable, the synthetic control fails, or a factor
       meets the A rule but a non-D control is missing.
    B: every causal variant tested, none substantial on any real case, controls intact.
    """
    for k, why in (("preflight_ok", "preflight / artifact identity failed"),
                   ("source_fidelity_ok", "W107 Blender source sampling differs from W106 N0"),
                   ("v0_reproduces_w106", "V0 does not reproduce the W106 FBX baseline"),
                   ("v0_deterministic", "V0 repeat on fresh scratch outputs is not deterministic"),
                   ("v0d_equals_v0", "diagnostic operator path does not equal the production export_fbx path"),
                   ("actions_unmodified", "an export modified the source Action"),
                   ("originals_unchanged", "an original artifact changed")):
        if not pre[k]:
            return "D", why
    f = pre["factors"]
    supported = [v for v in CAUSAL_VARIANTS if pre["variants_testable"].get(v) and f.get(v, {}).get("supported")]
    controls = pre["synthetic_control_ok"] and pre["v3_status"] == "TESTED"
    if supported and controls and all(pre["variants_testable"].values()):
        return "A", f"causal factor supported on every real case: {supported}"
    if supported:
        return "C", f"factor {supported} meets the support rule but a control/variant is missing or failed"
    if any(f.get(v, {}).get("mixed") for v in CAUSAL_VARIANTS):
        return "C", "mixed: a factor is substantial on some but not all real cases, or not clean"
    if not all(pre["variants_testable"].values()) or pre["v3_status"] != "TESTED" or not pre["synthetic_control_ok"]:
        return "C", "a variant is untestable or a synthetic control failed"
    return "B", "no tested independent factor yields a substantial real-Action geometric reduction"


# --- hashing ----------------------------------------------------------------------------------------------------
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def json_sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, indent=1).encode()).hexdigest()


def load_npz(path):
    z = np.load(Path(path))
    return {k: z[k] for k in z.files}
