#!/usr/bin/env python3
"""W108 evaluation contract: FBX subframe motion semantics and interpolation ownership (pure Python; evidence only).

Shared by the host stages, the Blender stages and the tests.  Nothing here touches Blender, a rig file or a
production module.  It defines

  * the fixed cases (three W106/W107 real M_Main Actions) and the two synthetic controls (step / smooth) built on the
    W106 synthetic rig, never pooled with the real Actions,
  * the exact sample grid (every integer frame plus +0.25 / +0.50 / +0.75 inside the scene range) and the
    transition / quiet interval classes derived from the source key times only,
  * the world / armature-space / rotation error metrics per sample phase and interval class,
  * an independent binary-FBX node reader (no add-on code) and the AnimationCurve key-attribute decoder,
  * the single declared scratch intervention (P2: interpolation property -> CONSTANT on a copy of the imported
    Action), the predeclared tolerances, expected signatures and the CASE mapping.

No production exporter or importer function is modified or monkey-patched.
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
import zlib
from collections import namedtuple
from pathlib import Path

import numpy as np

SCHEMA = "w108_fbx_subframe_v1"

REAL_CASES = ("complete_control", "arm_natural_most", "real_clip")   # W106/W107 cases, fixed; never substituted
REAL_ACTIONS = {c: f"W91 {c} FIXED" for c in REAL_CASES}             # M_Main slot; never R11 RUNTIME
SYN_INPUT = "w106_synthetic"                                         # W107 scratch copy of the W106 synthetic N0
SYN_BASE_ACTION = "SYN baseline"
SYN_CONTROLS = {
    "syn_step": {"action": "W108 SYN step CONSTANT", "source_interpolation": "CONSTANT",
                 "reference": "held pose: value of the latest key at or before t (Blender CONSTANT evaluation)"},
    "syn_smooth": {"action": "W108 SYN smooth LINEAR", "source_interpolation": "LINEAR",
                   "reference": "piecewise-linear F-curves between the same keys (Blender LINEAR evaluation)"},
}
SYN_CONSTRUCTION = ("scratch copy of the W106 synthetic N0 file; the 'SYN baseline' Action is duplicated (key times, "
                    "values, slot identical), the duplicate's key interpolation is set by foreach_set to the control's "
                    "source interpolation, the duplicate is assigned to the armature with the same slot identifier and "
                    "the file saved as a scratch copy; rig, scene fps and frame range are untouched")

EXPORT_PATHS = ("V0", "V1")
# V0: real unchanged blender.export.export_fbx.  V1: W107's isolated diagnostic call (V0 arguments +
# bake_anim_simplify_factor=0.0).  Neither is a production fix.
V1_OVERRIDE = {"bake_anim_simplify_factor": 0.0}

PHASES = (0.0, 0.25, 0.5, 0.75)
SUB_PHASES = (0.25, 0.5, 0.75)

P2 = {"name": "P2_post_import_constant",
      "operation": "on a fresh default import of the V1 FBX: Action.copy() of the selected imported Action only; "
                   "keyframe_points.foreach_set('interpolation', CONSTANT) on every key of every F-curve of the copy "
                   "(no fcurve.update(), no key insert/remove/move, no value/handle change); assign the copy to the "
                   "imported armature with the slot whose identifier equals the original's assigned slot",
      "unchanged": ["key times", "key values", "key counts", "handles and handle types", "easing/back/amplitude/period",
                    "extrapolation", "modifiers", "rig", "scene fps / frame range", "source Action", "original "
                    "imported Action"],
      "not_testable_if": "copy changes any key field other than interpolation, the slot cannot be matched, or the "
                         "original imported Action changes"}

# FBX SDK FbxAnimCurveDef::EInterpolationType bits of KeyAttrFlags
FBX_INTERP_BITS = {0x00000002: "CONSTANT", 0x00000004: "LINEAR", 0x00000008: "CUBIC"}
FBX_INTERP_MASK = 0x0000000E
FBX_CONSTANT_NEXT = 0x00000100     # eConstantNext (only meaningful with CONSTANT)
FBX_KTIME_PER_SECOND = 46186158000

# --- tolerances (fixed a priori; metres unless noted) -----------------------------------------------------------
TOL = {
    "deterministic_m": 1e-9,           # P0 replay vs W107, repeat probe, P2 vs V1 at integer frames
    "source_fidelity_m": 0.0,          # W108 source samples vs W107 source samples (bitwise)
    "w107_metric_reproduce_m": 1e-9,   # recomputed integer gate / half-frame max vs the W107 analysis values
    "material_reduction_ratio": 0.5,   # P2 subframe max <= 0.5 * V1 subframe max (W107 'substantial' convention)
    "smooth_harm_ratio": 2.0,          # smooth control: P2 subframe max >= 2 * V1 subframe max
    "sufficiency_m": 1e-3,             # W106 1 mm gate applied to subframe samples (representational sufficiency)
    "synthetic_rest_confound_m": 1e-5, # synthetic V1 integer gate above this -> original gate confounded
    "key_time_integer_eps": 0.0,       # imported key time deviation from an integer frame (exact expected)
}


# --- sample grid and interval classes ----------------------------------------------------------------------------
def sample_times(frame_start, frame_end):
    """Every integer frame of [start, end] plus +0.25/+0.5/+0.75 of every frame except the last."""
    out = []
    for f in range(frame_start, frame_end + 1):
        for p in PHASES:
            if f == frame_end and p > 0:
                break
            out.append(float(f) + p)
    return out


def phase_of(times):
    t = np.asarray(times, dtype=float)
    return np.round(t - np.floor(t), 6)


def interval_classes(frame_start, frame_end, key_frames):
    """Interval [f, f+1) is *transition* iff some source key lies at f+1 (the held pose may change there), else
    *quiet*.  Derived from source key times only (identity metadata, never from outcomes)."""
    keys = {float(k) for k in key_frames}
    out = {}
    for f in range(frame_start, frame_end):
        out[f] = "transition" if float(f + 1) in keys else "quiet"
    return out


def time_classes(times, classes):
    """Per sample: 'integer' or the class of its interval."""
    out = []
    for t in times:
        f = int(math.floor(t))
        out.append("integer" if t == f else classes[f])
    return out


# --- metrics ------------------------------------------------------------------------------------------------------
def _stats(x):
    x = np.asarray(x, dtype=float).ravel()
    if x.size == 0:
        return {"max": None, "p95": None, "median": None, "n": 0}
    return {"max": float(x.max()), "p95": float(np.percentile(x, 95)), "median": float(np.median(x)), "n": int(x.size)}


def common_index(src, imp):
    ns = [str(b) for b in src["bones"]]
    ni = [str(b) for b in imp["bones"]]
    pos = {n: i for i, n in enumerate(ni)}
    common = [n for n in ns if n in pos]
    return common, [ns.index(n) for n in common], [pos[n] for n in common]


def rot_angle_deg(ra, rb):
    """Angle between rotation matrices (..., 3, 3) after column normalisation."""
    ra = ra/np.linalg.norm(ra, axis=-2, keepdims=True)
    rb = rb/np.linalg.norm(rb, axis=-2, keepdims=True)
    tr = np.einsum("...ji,...ji->...", ra, rb)
    return np.degrees(np.arccos(np.clip((tr - 1)/2, -1, 1)))


def error_fields(src, imp):
    """Per (time, bone) error fields at identical sample times."""
    if not np.array_equal(src["times"], imp["times"]):
        raise ValueError("sample times differ")
    common, ia, ib = common_index(src, imp)
    dh = src["world_head"][:, ia] - imp["world_head"][:, ib]
    dt = src["world_tail"][:, ia] - imp["world_tail"][:, ib]
    coord = np.maximum(np.abs(dh).max(-1), np.abs(dt).max(-1))                     # (T, B) gate-style
    euclid = np.maximum(np.linalg.norm(dh, axis=-1), np.linalg.norm(dt, axis=-1))    # (T, B)
    head = np.abs(dh).max(-1)                                                       # (T, B) heads only
    rot = rot_angle_deg(src["world_rot"][:, ia], imp["world_rot"][:, ib])           # (T, B) degrees
    om_s, om_i = src["object_matrix"], imp["object_matrix"]
    obj_rot = rot_angle_deg(om_s[:, :3, :3], om_i[:, :3, :3])
    obj_tr = np.linalg.norm(om_s[:, :3, 3] - om_i[:, :3, 3], axis=-1)
    arm = np.zeros(coord.shape)
    for k in range(om_s.shape[0]):
        inv_s, inv_i = np.linalg.inv(om_s[k]), np.linalg.inv(om_i[k])
        ps = (inv_s[:3, :3] @ src["world_tail"][k][ia].T).T + inv_s[:3, 3]
        pi = (inv_i[:3, :3] @ imp["world_tail"][k][ib].T).T + inv_i[:3, 3]
        arm[k] = np.linalg.norm(ps - pi, axis=-1)*float(np.linalg.norm(om_s[k][:3, 0]))
    return {"bones": common, "coord": coord, "euclid": euclid, "head": head, "rot_deg": rot,
            "object_rot_deg": obj_rot, "object_trans_m": obj_tr, "armature_tail_m": arm}


def summarize(fields, times, tclass, mask_name, mask):
    t = np.asarray(times, dtype=float)
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return {"samples": 0}
    c = fields["coord"][m]
    pf = c.max(1)
    pb = c.max(0)
    order = np.argsort(-pb)[:5]
    fo = np.argsort(-pf)[:5]
    return {"samples": int(m.sum()),
            "max_abs_coord_m": float(c.max()),
            "euclid_m": _stats(fields["euclid"][m]),
            "per_sample_max_m": _stats(pf),
            "samples_gt_1mm": int((pf > TOL["sufficiency_m"]).sum()),
            "head_max_abs_coord_m": float(fields["head"][m].max()),
            "world_rot_deg": _stats(fields["rot_deg"][m]),
            "object_rot_deg_max": float(fields["object_rot_deg"][m].max()),
            "object_trans_m_max": float(fields["object_trans_m"][m].max()),
            "armature_tail_m": _stats(fields["armature_tail_m"][m]),
            "worst_bones": [[fields["bones"][k], float(pb[k])] for k in order],
            "worst_times": [[float(t[m][k]), float(pf[k])] for k in fo]}


def subframe_report(src, imp, times, classes):
    """Errors by phase (0, .25, .5, .75), all subframes, and subframes by interval class."""
    f = error_fields(src, imp)
    ph = phase_of(times)
    tc = np.array(time_classes(times, classes))
    out = {"bones_compared": len(f["bones"])}
    out["integer"] = summarize(f, times, tc, "integer", ph == 0)
    for p in SUB_PHASES:
        out[f"phase_{p:.2f}"] = summarize(f, times, tc, str(p), np.isclose(ph, p))
    out["subframe"] = summarize(f, times, tc, "subframe", ph > 0)
    out["subframe_transition"] = summarize(f, times, tc, "transition", (ph > 0) & (tc == "transition"))
    out["subframe_quiet"] = summarize(f, times, tc, "quiet", (ph > 0) & (tc == "quiet"))
    per_bone = f["coord"][ph > 0].max(0) if (ph > 0).any() else np.zeros(len(f["bones"]))
    out["subframe_per_bone_max_abs_coord_m"] = {b: float(v) for b, v in zip(f["bones"], per_bone)}
    return out


def arrays_max_diff(a, b, keys=("world_head", "world_tail", "world_rot", "object_matrix", "local_quat", "local_loc",
                                "local_scale"), mask_a=None, mask_b=None):
    """Max abs difference of two sample sets (same bone order); optional time masks select common times."""
    if [str(x) for x in a["bones"]] != [str(x) for x in b["bones"]]:
        return math.inf
    ta = a["times"] if mask_a is None else a["times"][mask_a]
    tb = b["times"] if mask_b is None else b["times"][mask_b]
    if not np.array_equal(ta, tb):
        return math.inf
    out = 0.0
    for k in keys:
        xa = np.asarray(a[k]) if mask_a is None else np.asarray(a[k])[mask_a]
        xb = np.asarray(b[k]) if mask_b is None else np.asarray(b[k])[mask_b]
        out = max(out, float(np.abs(xa - xb).max()) if xa.size else 0.0)
    return out


def common_time_masks(t_new, t_old):
    """Masks selecting the times present in both grids (W107 integer+half grid is a subset of the W108 grid)."""
    so = set(np.asarray(t_old).tolist())
    sn = set(np.asarray(t_new).tolist())
    return np.array([t in so for t in t_new]), np.array([t in sn for t in t_old])


def w107_gate(src, imp):
    """W107/W106 gate metric (integer frames) and W107 half-frame max, recomputed on a W108 sample set."""
    f = error_fields(src, imp)
    ph = phase_of(src["times"])
    return {"gate_max_abs_coord_m": float(f["coord"][ph == 0].max()),
            "half_frame_max_abs_coord_m": float(f["coord"][np.isclose(ph, 0.5)].max())}


# --- independent binary FBX reader -------------------------------------------------------------------------------
Elem = namedtuple("Elem", ("id", "props", "props_type", "elems"))
_SCALAR = {b"Y": ("<h", 2), b"C": ("<?", 1), b"I": ("<i", 4), b"F": ("<f", 4), b"D": ("<d", 8), b"L": ("<q", 8)}
_ARRAY = {b"f": "<f4", b"d": "<f8", b"l": "<i8", b"i": "<i4", b"b": "<i1"}
FBX_MAGIC = b"Kaydara FBX Binary  \x00\x1a\x00"


def fbx_read(path):
    """Minimal binary FBX node reader written from the public format description (header, node records, typed
    properties, zlib arrays).  Independent of ``io_scene_fbx``.  Returns (root Elem, version)."""
    data = Path(path).read_bytes()
    if not data.startswith(FBX_MAGIC):
        raise ValueError("not a binary FBX")
    version = struct.unpack_from("<I", data, 23)[0]
    wide = version >= 7500
    hdr = struct.Struct("<QQQ" if wide else "<III")
    null = hdr.size + 1

    def read_prop(off):
        t = data[off:off+1]
        off += 1
        if t in _SCALAR:
            fmt, n = _SCALAR[t]
            return struct.unpack_from(fmt, data, off)[0], t, off + n
        if t in _ARRAY:
            n, enc, clen = struct.unpack_from("<III", data, off)
            off += 12
            raw = data[off:off+clen]
            if enc == 1:
                raw = zlib.decompress(raw)
            return np.frombuffer(raw, dtype=_ARRAY[t], count=n).copy(), t, off + clen
        if t in (b"S", b"R"):
            n = struct.unpack_from("<I", data, off)[0]
            off += 4
            return data[off:off+n], t, off + n
        raise ValueError(f"unknown property type {t!r} at {off-1}")

    def read_node(off):
        end, nprops, _plen = hdr.unpack_from(data, off)
        off += hdr.size
        nlen = data[off]
        off += 1
        if end == 0:
            return None, off + nlen
        name = data[off:off+nlen]
        off += nlen
        props, types = [], []
        for _ in range(nprops):
            v, t, off = read_prop(off)
            props.append(v)
            types.append(t)
        elems = []
        while off < end:
            if end - off == null and data[off:end] == b"\x00"*null:
                off = end
                break
            child, off = read_node(off)
            if child is None:
                break
            elems.append(child)
        return Elem(name, props, b"".join(types), elems), end

    off = 27
    top = []
    while off < len(data):
        node, off2 = read_node(off)
        if node is None:
            break
        top.append(node)
        off = off2
    return Elem(b"", [], b"", top), version


def _child(e, name):
    for c in e.elems:
        if c.id == name:
            return c
    return None


def decode_interp(flag):
    bits = int(flag) & FBX_INTERP_MASK
    return FBX_INTERP_BITS.get(bits, f"UNKNOWN(0x{bits:x})")


def fbx_anim_summary(root, fps):
    """Per AnimationStack: curves, key counts, key-time grid and decoded KeyAttrFlags interpolation.

    Works on any tree of (id, props, props_type, elems) nodes -- this module's reader or ``io_scene_fbx.parse_fbx``."""
    objs = _child(root, b"Objects")
    conns = _child(root, b"Connections")
    if objs is None or conns is None:
        return {"status": "NO_OBJECTS_OR_CONNECTIONS"}
    kind, name, curves = {}, {}, {}
    for c in objs.elems:
        if c.id in (b"AnimationStack", b"AnimationLayer", b"AnimationCurveNode", b"AnimationCurve"):
            uid = int(c.props[0])
            kind[uid] = c.id.decode()
            nm = bytes(c.props[1]) if len(c.props) > 1 else b""
            name[uid] = nm.split(b"\x00")[0].decode(errors="replace")
            if c.id == b"AnimationCurve":
                kt = _child(c, b"KeyTime")
                kv = _child(c, b"KeyValueFloat")
                kf = _child(c, b"KeyAttrFlags")
                kr = _child(c, b"KeyAttrRefCount")
                curves[uid] = {"times": np.asarray(kt.props[0], dtype=np.int64) if kt else np.zeros(0, np.int64),
                               "n_values": len(kv.props[0]) if kv else 0,
                               "flags": [int(x) for x in kf.props[0]] if kf else [],
                               "refcount": [int(x) for x in kr.props[0]] if kr else []}
    parent = {}
    for c in conns.elems:
        if c.id == b"C" and len(c.props) >= 3:
            ch, pa = int(c.props[1]), int(c.props[2])
            parent.setdefault(ch, []).append(pa)

    def up(uid, want):
        for p in parent.get(uid, []):
            if kind.get(p) == want:
                return p
        return None
    tick = FBX_KTIME_PER_SECOND/fps
    stacks = {}
    for uid, cv in curves.items():
        cn = up(uid, "AnimationCurveNode")
        ly = up(cn, "AnimationLayer") if cn is not None else None
        st = up(ly, "AnimationStack") if ly is not None else None
        sname = name.get(st, "<unconnected>")
        s = stacks.setdefault(sname, {"curves": 0, "keys": 0, "flags": set(), "interp": set(),
                                      "refcount_covers_keys": True, "values_match_times": True,
                                      "nonint_key_times": 0, "frames": set()})
        s["curves"] += 1
        s["keys"] += int(cv["times"].size)
        s["flags"].update(cv["flags"])
        s["interp"].update(decode_interp(f) for f in cv["flags"])
        s["refcount_covers_keys"] &= sum(cv["refcount"]) == cv["times"].size
        s["values_match_times"] &= cv["n_values"] == cv["times"].size
        fr = cv["times"].astype(np.float64)/tick
        s["nonint_key_times"] += int((cv["times"] % round(tick) != 0).sum()) if float(tick).is_integer() else \
            int((np.abs(fr - np.round(fr)) > 1e-9).sum())
        s["frames"].update(np.round(fr, 6).tolist())
    out = {}
    for sname, s in stacks.items():
        fr = sorted(s["frames"])
        out[sname] = {"curves": s["curves"], "keys": s["keys"], "flags_hex": sorted(f"0x{f:x}" for f in s["flags"]),
                      "interpolation": sorted(s["interp"]), "refcount_covers_keys": bool(s["refcount_covers_keys"]),
                      "values_match_times": bool(s["values_match_times"]), "nonint_key_times": s["nonint_key_times"],
                      "frame_min": fr[0] if fr else None, "frame_max": fr[-1] if fr else None,
                      "distinct_frames": len(fr)}
    return {"status": "PARSED", "ticks_per_frame": tick, "stacks": out}


def select_stack(stacks, action):
    """Stack of the selected Action: exact name or the importer's '<obj>|<action>' convention."""
    m = [s for s in stacks if s == action or s.endswith("|" + action)]
    return m[0] if len(m) == 1 else None


# --- expected signatures / CASE mapping -------------------------------------------------------------------------
def boundary(source_interp, fbx_independent, fbx_addon, imported_interp):
    """First evidenced loss of hold semantics, from direct encoding evidence only.

    ``fbx_*``: selected-stack summaries from the two parsers.  Reliable iff both parsers agree on the stack's curve
    and key counts and decoded interpolation set, and every curve's key attributes cover all keys."""
    rel = (fbx_independent is not None and fbx_addon is not None
           and all(fbx_independent.get(k) == fbx_addon.get(k) for k in ("curves", "keys", "interpolation",
                                                                        "flags_hex", "nonint_key_times"))
           and fbx_independent.get("refcount_covers_keys") and fbx_independent.get("values_match_times"))
    res = {"fbx_flags_reliable": bool(rel), "source": source_interp, "imported": imported_interp,
           "fbx": fbx_independent.get("interpolation") if fbx_independent else None}
    if not rel:
        res["boundary"] = "UNRESOLVED_FBX_FLAGS_UNRELIABLE"
    elif source_interp != ["CONSTANT"]:
        res["boundary"] = "NOT_APPLICABLE_SOURCE_NOT_HELD"
    elif "CONSTANT" not in res["fbx"]:
        res["boundary"] = "EXPORT_SERIALIZATION"   # FBX already encodes non-hold interpolation
    elif imported_interp != ["CONSTANT"]:
        res["boundary"] = "IMPORT"                 # FBX encodes CONSTANT, importer discards it
    else:
        res["boundary"] = "NONE_OBSERVED"
    return res


def decide_case(pre):
    """``pre`` keys: preflight_ok, p0 (dict case -> bool), deterministic, originals_unchanged, sources_ok,
    synthetic_valid, boundary (dict case -> boundary()), p2 (dict case -> {testable, integer_unchanged, keydata_ok,
    material, sufficient}), synthetic (dict control -> {testable, contrast_ok, confounded}).

    D: preflight / P0 replay / identity / determinism / originals / invalid synthetic construction.
    A: every real case has a reliable boundary in {EXPORT_SERIALIZATION, IMPORT}; P2 testable, key data unchanged,
       integer-frame motion unchanged and material on all three; step and smooth contrasts both hold, unconfounded.
    B: P2 testable with integer parity on all three real cases, synthetic controls valid, and P2 not material on any
       real case (interpolation hypothesis fails to explain the subframe error).
    C: otherwise (mixed real results, unreliable FBX flags, untestable intervention, synthetic contrast failed or
       confounded, integer parity broken by P2).
    """
    for k, why in (("preflight_ok", "preflight / artifact identity failed"),
                   ("sources_ok", "source Action / slot / timing identity mismatch"),
                   ("deterministic", "fresh repeat disagrees"),
                   ("originals_unchanged", "an original artifact changed"),
                   ("synthetic_valid", "synthetic controls invalid (construction or source semantics)")):
        if not pre[k]:
            return "D", why
    if not all(pre["p0"].values()):
        return "D", f"P0 replay of W107 failed: {sorted(c for c, v in pre['p0'].items() if not v)}"
    p2 = pre["p2"]
    real_ok = {c: p2[c]["testable"] and p2[c]["keydata_ok"] and p2[c]["integer_unchanged"] for c in REAL_CASES}
    bnd = {c: pre["boundary"][c]["boundary"] in ("EXPORT_SERIALIZATION", "IMPORT") for c in REAL_CASES}
    syn = pre["synthetic"]
    syn_ok = all(syn[k]["testable"] and syn[k]["contrast_ok"] and not syn[k]["confounded"] for k in SYN_CONTROLS)
    material = {c: real_ok[c] and p2[c]["material"] for c in REAL_CASES}
    if all(bnd.values()) and all(material.values()) and syn_ok:
        return "A", "boundary evidenced, P2 material with unchanged integer motion on all three real Actions, " \
                    "step/smooth contrast holds"
    if all(real_ok.values()) and not any(p2[c]["material"] for c in REAL_CASES) and \
            all(syn[k]["testable"] for k in SYN_CONTROLS):
        return "B", "P2 does not materially reduce subframe error on any real Action"
    why = []
    if not all(bnd.values()):
        why.append(f"boundary not evidenced: {sorted(c for c, v in bnd.items() if not v)}")
    if not all(real_ok.values()):
        why.append(f"P2 untestable or integer parity broken: {sorted(c for c, v in real_ok.items() if not v)}")
    if not all(material.values()):
        why.append(f"P2 not material: {sorted(c for c, v in material.items() if not v)}")
    if not syn_ok:
        why.append("synthetic contrast failed, untestable or confounded")
    return "C", "; ".join(why)


# --- hashing ----------------------------------------------------------------------------------------------------
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_npz(path):
    z = np.load(Path(path))
    return {k: z[k] for k in z.files}
