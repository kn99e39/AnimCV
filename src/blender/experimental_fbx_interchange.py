"""EXPERIMENTAL (opt-in, Batch 127) FBX continuous-time interchange helper for the video -> editable Action slice.

NOT part of the default pipeline.  ``src/blender/export.py`` (the default exporter) and the stock Blender importer are
unchanged and never patched.  This module is used only by the explicit ``experimental-video-action fbx-interchange`` /
``fbx-consume`` commands and by their Blender stage modes.

It is a MAINTAINED adaptation of the W107-W111 scratch mechanisms (evidence: docs/107-111 Worklogs):

  * W107: the stock exporter's key simplification causes integer-frame drift -> the producer exports with
    ``bake_anim_simplify_factor = 0`` (a key at every frame of the scene range).
  * W108/W109: the exporter writes every curve LINEAR (``KeyAttrFlags 0x6104``) and the stock importer ignores the flags
    -> the producer rewrites, byte-exactly and in place, ONLY the ``KeyAttrFlags`` value of FBX curves whose driving
    source (item, transform group) is uniformly CONSTANT to the FBX SDK CONSTANT encoding ``0x6002``; a consumer
    adapter reads the flags back and sets ONLY keyframe ``interpolation`` on the uniquely mapped imported F-curves.
  * W110/W111: the curve correspondence is resolved from the FBX connection graph only (never exporter UIDs), so it is
    invariant to the exporter's per-process UID / element-order / source-path variation.

The FBX file stays a standard binary FBX.  Scene frame range and the declared take/rig/timebase identity travel in a
slice-private, sealed package manifest (``animcv_exp_fbx_interchange_package_v1``) that is a REQUIRED consumer input:
raw FBX alone in a stock importer is NOT lossless (it keeps LINEAR interpolation and drops the scene range).

Pure-Python parts (no bpy) are importable by the host CLI and the tests.  ``bpy`` parts are imported lazily.
"""
from __future__ import annotations

import hashlib
import json
import struct
import zlib
from collections import namedtuple

PACKAGE_SCHEMA = "animcv_exp_fbx_interchange_package_v1"
CONSUMER_SCHEMA = "animcv_exp_fbx_interchange_consume_v1"
ADAPTER_SCHEMA = "animcv_exp_fbx_interchange_adapter_v1"
PACKAGE_FBX = "animation.fbx"
PACKAGE_MANIFEST = "package_manifest.json"
# The consumer's naming/mapping rule depends on the stock importer's behaviour; only the tested pair is accepted.
SUPPORTED_BLENDER = ("5.1",)

# --- FBX SDK FbxAnimCurveDef bits (public SDK enumerations; W108/W109) --------------------------------------------
FBX_CONSTANT = 0x2
FBX_LINEAR = 0x4
FBX_CUBIC = 0x8
FBX_INTERP_MASK = 0xE
FBX_CONSTANT_NEXT = 0x100
EXPORTER_FLAG = 0x6104            # what Blender's exporter writes for every baked key (LINEAR + inert tangent bits)
CONSTANT_FLAG = 0x6002            # W109/W110 CONSTANT-standard: interpolation field -> CONSTANT, other bits kept
FBX_TICKS_PER_SECOND = 46186158000

SUPPORTED = ("CONSTANT", "LINEAR")
TRANSFORM_PROPS = {"Lcl Translation": "location", "Lcl Rotation": "rotation", "Lcl Scaling": "scale"}
CHANNELS = {"d|X": 0, "d|Y": 1, "d|Z": 2}
SEPARABILITY_DEFAULTS = {"RotationOffset": (0.0, 0.0, 0.0), "RotationPivot": (0.0, 0.0, 0.0),
                         "ScalingOffset": (0.0, 0.0, 0.0), "ScalingPivot": (0.0, 0.0, 0.0),
                         "GeometricTranslation": (0.0, 0.0, 0.0), "GeometricRotation": (0.0, 0.0, 0.0),
                         "GeometricScaling": (1.0, 1.0, 1.0)}

# Exporter arguments of the opt-in producer.  Differences from ``src/blender/export.py`` (which stays the default):
#   bake_anim_simplify_factor 0.0       W107: no destructive key simplification (a key at every scene frame)
#   bake_anim_use_all_actions True   \  ONE per-Action take: the producer removes every other Action from the opened file
#   bake_anim_use_nla_strips  False  /  IN MEMORY ONLY (the .blend is never saved) so no extra take can leak in.  A per-Action
#                                       take starts at FBX time 0 (= Action frame 1), which the stock importer's default
#                                       anim_offset 1 maps back to frame 1.  (W127 development probe: a GLOBAL scene take
#                                       writes absolute time frame/fps and re-imports one frame late, 2..N+1.)
#   add_leaf_bones            False     no synthetic "<bone>_end" bones: the imported topology equals the source rig
#   object_types {ARMATURE, MESH}       the rig's skinned mesh is exported so the FBX carries a BindPose.  Without one
#                                       (armature-only, as the default exporter) the stock importer rebuilds every bone rest
#                                       from the exported current-frame pose (W127 development probe on the synthetic BaseRig
#                                       fixture: world rest change up to 0.92 m; with the mesh <= 1 um).  Only meshes bound
#                                       to the armature are allowed (producer check); no camera / light is exported.
#   bake_anim_force_startend_keying True (Blender default, stated explicitly)
EXPORT_KWARGS = {"use_selection": False, "object_types": {"ARMATURE", "MESH"}, "bake_anim": True,
                 "bake_anim_use_all_actions": True, "bake_anim_use_nla_strips": False,
                 "bake_anim_simplify_factor": 0.0, "bake_anim_force_startend_keying": True, "add_leaf_bones": False}


class InterchangeStop(RuntimeError):
    """A precise refusal (machine-readable ``code``).  The package / recovered scene is never produced on a stop."""

    def __init__(self, code, detail=""):
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code, self.detail = code, detail


# ======================================================================================= binary FBX reading ===
Elem = namedtuple("Elem", ("id", "props", "props_type", "elems"))
_SCALAR = {b"Y": ("<h", 2), b"C": ("<b", 1), b"I": ("<i", 4), b"F": ("<f", 4), b"D": ("<d", 8), b"L": ("<q", 8)}
_ARRAY = {b"f": ("<f", 4), b"d": ("<d", 8), b"l": ("<q", 8), b"i": ("<i", 4), b"b": ("<b", 1), b"c": ("<b", 1)}
FBX_MAGIC = b"Kaydara FBX Binary  \x00\x1a\x00"


def read_fbx(data):
    """Fail-closed wrapper of ``_read_fbx``: a truncated / corrupt file is a declared ``FBX_PARSE_ERROR``."""
    try:
        return _read_fbx(data)
    except InterchangeStop:
        raise
    except (struct.error, zlib.error, IndexError, ValueError, UnicodeDecodeError, KeyError, TypeError) as e:
        raise InterchangeStop("FBX_PARSE_ERROR", f"{type(e).__name__}: {e}") from None


def _read_fbx(data):
    """Binary FBX -> (root Elem, version, flag_locations).  Written from the public format description (adapted from the
    independent W108/W109 readers).  ``flag_locations`` maps every Objects/AnimationCurve uid to its KeyAttrFlags array
    record (value offset, length, encoding, byte size) for an in-place same-length rewrite."""
    if not data.startswith(FBX_MAGIC):
        raise InterchangeStop("FBX_NOT_BINARY", "not a binary FBX")
    version = struct.unpack_from("<I", data, 23)[0]
    hdr = struct.Struct("<QQQ" if version >= 7500 else "<III")
    null = hdr.size + 1
    arrays = {}

    def read_prop(off):
        t = data[off:off + 1]
        off += 1
        if t in _SCALAR:
            fmt, n = _SCALAR[t]
            return struct.unpack_from(fmt, data, off)[0], t, off + n, None
        if t in _ARRAY:
            n, enc, clen = struct.unpack_from("<III", data, off)
            raw = data[off + 12:off + 12 + clen]
            rec = {"value_offset": off + 12, "length": n, "encoding": enc, "size": clen, "type": t}
            if enc == 1:
                raw = zlib.decompress(raw)
            fmt, sz = _ARRAY[t]
            vals = list(struct.unpack_from(f"<{n}{fmt[1]}", raw, 0)) if n else []
            return vals, t, off + 12 + clen, rec
        if t in (b"S", b"R"):
            n = struct.unpack_from("<I", data, off)[0]
            return data[off + 4:off + 4 + n], t, off + 4 + n, None
        raise InterchangeStop("FBX_PARSE_ERROR", f"unknown property type {t!r} at {off - 1}")

    def read_node(off, path):
        end, nprops, _ = hdr.unpack_from(data, off)
        off += hdr.size
        nlen = data[off]
        off += 1
        if end == 0:
            return None, off + nlen
        name = data[off:off + nlen]
        off += nlen
        props, types, recs = [], [], []
        for _ in range(nprops):
            v, t, off, rec = read_prop(off)
            props.append(v)
            types.append(t)
            recs.append(rec)
        elems = []
        while off < end:
            if end - off == null and data[off:end] == b"\x00" * null:
                off = end
                break
            child, off = read_node(off, path + (name,))
            if child is None:
                break
            elems.append(child)
        e = Elem(name, props, b"".join(types), elems)
        if name == b"KeyAttrFlags" and path[-2:] == (b"Objects", b"AnimationCurve"):
            arrays.setdefault("pending", []).append((e, recs[0] if recs else None))
        return e, end

    off, top, sentinel = 27, [], False
    while off < len(data):
        node, off2 = read_node(off, ())
        if node is None:
            sentinel = True
            break
        top.append(node)
        off = off2
    ids = {t.id for t in top}
    if not sentinel or not {b"GlobalSettings", b"Objects", b"Connections"} <= ids:
        raise InterchangeStop("FBX_PARSE_ERROR", f"truncated or incomplete FBX (sentinel {sentinel}, sections {sorted(ids)})")
    root = Elem(b"", [], b"", top)
    pend = {id(e): r for e, r in arrays.get("pending", [])}
    flags = {}
    for t in top:
        if t.id != b"Objects":
            continue
        for c in t.elems:
            if c.id != b"AnimationCurve":
                continue
            uid = int(c.props[0])
            fl = [e for e in c.elems if e.id == b"KeyAttrFlags"]
            if uid in flags or len(fl) != 1 or id(fl[0]) not in pend:
                raise InterchangeStop("FBX_CURVE_FLAGS_AMBIGUOUS", f"AnimationCurve {uid}")
            flags[uid] = pend[id(fl[0])]
    return root, version, flags


def _child(e, name):
    for c in e.elems:
        if c.id == name:
            return c
    return None


def _name(raw):
    return bytes(raw).split(b"\x00\x01")[0].split(b"\x00")[0].decode("utf-8", errors="replace")


def _str(raw):
    return bytes(raw).decode("utf-8", errors="replace")


def flag_class(flag):
    f = int(flag)
    bits = f & FBX_INTERP_MASK
    if bits == FBX_CONSTANT:
        return "CONSTANT_NEXT" if f & FBX_CONSTANT_NEXT else "CONSTANT"
    return {FBX_LINEAR: "LINEAR", FBX_CUBIC: "CUBIC"}.get(bits, f"UNKNOWN(0x{bits:x})")


def _p70(elem):
    p = _child(elem, b"Properties70")
    out = {}
    if p is None:
        return out
    for e in p.elems:
        if e.id == b"P" and len(e.props) >= 4:
            out[_str(e.props[0])] = [x for x in e.props[4:]]
    return out


def separable(model_elem):
    p = _p70(model_elem)
    bad = {k: p[k] for k, d in SEPARABILITY_DEFAULTS.items()
           if k in p and tuple(float(x) for x in p[k][:3]) != d}
    return not bad, bad


# ===================================================================== take / timebase inventory (FBX side) ===
def take_inventory(root):
    """Every AnimationStack: name, LocalStart/LocalStop (ticks), its layers; and the GlobalSettings timebase."""
    objs = _child(root, b"Objects")
    conns = _child(root, b"Connections")
    nodes = {int(c.props[0]): c for c in (objs.elems if objs else []) if len(c.props) >= 2}
    parents = {}
    for c in (conns.elems if conns else []):
        if c.id == b"C" and len(c.props) >= 3:
            parents.setdefault(int(c.props[1]), []).append(int(c.props[2]))
    stacks = []
    for u, n in sorted(nodes.items(), key=lambda kv: _name(kv[1].props[1])):
        if n.id != b"AnimationStack":
            continue
        p = _p70(n)
        layers = sorted(_name(nodes[l].props[1]) for l, ps in parents.items()
                        if l in nodes and nodes[l].id == b"AnimationLayer" and u in ps)
        # an absent LocalStart / LocalStop property has the FBX default value 0 (the exporter omits default values)
        stacks.append({"name": _name(n.props[1]), "local_start_ticks": int(p["LocalStart"][0]) if "LocalStart" in p else 0,
                       "local_stop_ticks": int(p["LocalStop"][0]) if "LocalStop" in p else 0, "layers": layers,
                       "local_start_present": "LocalStart" in p, "local_stop_present": "LocalStop" in p})
    gs = _child(root, b"GlobalSettings")
    g = _p70(gs) if gs is not None else {}
    timebase = {"TimeMode": int(g["TimeMode"][0]) if "TimeMode" in g else None,
                "CustomFrameRate": float(g["CustomFrameRate"][0]) if "CustomFrameRate" in g else None,
                "TimeSpanStart": int(g["TimeSpanStart"][0]) if "TimeSpanStart" in g else None,
                "TimeSpanStop": int(g["TimeSpanStop"][0]) if "TimeSpanStop" in g else None}
    return {"stacks": stacks, "global_settings": timebase}


def ticks_to_frame(ticks, fps):
    return ticks / FBX_TICKS_PER_SECOND * fps


# ============================================================ curve correspondence (W110 rule, maintained copy) ===
def fbx_curve_map(root, stack_name):
    """Per-curve FBX correspondence records of the selected AnimationStack (connection graph only).  Same rule as the
    W110 scratch adapter (``scripts/w110_fbx_consumer_adapter.py::fbx_curve_map``); parity is checked by the tests."""
    objs, conns = _child(root, b"Objects"), _child(root, b"Connections")
    nodes = {}
    for c in objs.elems:
        if len(c.props) >= 2:
            nodes[int(c.props[0])] = c
    kind = {u: c.id.decode() for u, c in nodes.items()}
    name = {u: _name(c.props[1]) for u, c in nodes.items()}
    parents = {}
    for c in conns.elems:
        if c.id == b"C" and len(c.props) >= 3:
            parents.setdefault(int(c.props[1]), []).append(
                (int(c.props[2]), _str(c.props[0]), _str(c.props[3]) if len(c.props) > 3 else None))
    out = {"stack": stack_name, "status": "OK", "curves": [], "models": {}}
    stacks = [u for u in nodes if kind[u] == "AnimationStack" and name[u] == stack_name]
    if len(stacks) != 1:
        out["status"] = f"NO_UNIQUE_STACK({len(stacks)})"
        return out
    layers = [u for u in nodes if kind[u] == "AnimationLayer" and any(p == stacks[0] for p, _, _ in parents.get(u, []))]
    if len(layers) != 1:
        out["status"] = f"UNSUPPORTED_LAYER_COUNT({len(layers)})"
        return out
    out["layer"] = name[layers[0]]
    if out["layer"] != stack_name:
        out["status"] = "UNSUPPORTED_LAYER_NAME"
        return out
    cnodes = {u for u in nodes if kind[u] == "AnimationCurveNode" and any(p == layers[0] for p, _, _ in parents.get(u, []))}
    model_names = {}
    for u in nodes:
        if kind[u] == "Model":
            model_names.setdefault(name[u], []).append(u)
    seen = {}
    for u in sorted(nodes):
        if kind[u] != "AnimationCurve":
            continue
        cps = [(p, t, pn) for p, t, pn in parents.get(u, []) if kind.get(p) == "AnimationCurveNode"]
        if not any(p in cnodes for p, _, _ in cps):
            continue
        rec = {"uid": u, "status": "OK"}
        c = nodes[u]
        kf, kr, kt = _child(c, b"KeyAttrFlags"), _child(c, b"KeyAttrRefCount"), _child(c, b"KeyTime")
        flags = [int(x) for x in kf.props[0]] if kf is not None else []
        refs = [int(x) for x in kr.props[0]] if kr is not None else []
        nkeys = len(kt.props[0]) if kt is not None else 0
        classes = sorted({flag_class(f) for f in flags})
        rec.update({"keys": nkeys, "flags_hex": sorted({f"0x{f:x}" for f in flags}), "flag_classes": classes})
        rec["flag_class"] = classes[0] if len(classes) == 1 else "MIXED_OR_MISSING"
        if len(cps) != 1:
            rec["status"] = "AMBIGUOUS_CURVE_NODE"
            out["curves"].append(rec)
            continue
        cn, ctype, chan = cps[0]
        rec.update({"curve_node": cn, "channel_name": chan, "channel": CHANNELS.get(chan)})
        mps = [(p, t, pn) for p, t, pn in parents.get(cn, []) if kind.get(p) == "Model"]
        if ctype != "OP" or rec["channel"] is None:
            rec["status"] = "BAD_CHANNEL"
        elif len(mps) != 1:
            rec["status"] = f"MODEL_COUNT({len(mps)})"
        else:
            m, mtype, mprop = mps[0]
            rec.update({"model": m, "model_name": name[m], "model_type": _str(nodes[m].props[2]).split("\x00")[0]
                        if len(nodes[m].props) > 2 else "", "fbx_property": mprop, "group": TRANSFORM_PROPS.get(mprop)})
            if mtype != "OP" or rec["group"] is None:
                rec["status"] = "UNSUPPORTED_PROPERTY"
            elif rec["model_type"] not in ("Null", "LimbNode"):
                rec["status"] = "UNSUPPORTED_MODEL_TYPE"
            elif len(model_names[name[m]]) != 1:
                rec["status"] = "DUPLICATE_MODEL_NAME"
            elif sum(refs) != nkeys or not flags:
                rec["status"] = "BAD_REFCOUNT"
            if rec["status"] == "OK":
                key = (m, mprop, rec["channel"])
                if key in seen:
                    rec["status"] = "DUPLICATE_CHANNEL"
                    seen[key]["status"] = "DUPLICATE_CHANNEL"
                else:
                    seen[key] = rec
            if m not in out["models"]:
                mp = [p for p, t, _ in parents.get(m, []) if kind.get(p) == "Model" and t == "OO"]
                sep, bad = separable(nodes[m])
                out["models"][m] = {"name": name[m], "type": rec["model_type"],
                                    "parent": name[mp[0]] if len(mp) == 1 else None,
                                    "parent_type": (_str(nodes[mp[0]].props[2]).split("\x00")[0]
                                                    if len(mp) == 1 and len(nodes[mp[0]].props) > 2 else None),
                                    "parent_count": len(mp), "separable": sep, "non_default": {k: list(v) for k, v in bad.items()}}
        out["curves"].append(rec)
    out["models"] = {str(k): v for k, v in sorted(out["models"].items())}
    return out


def group_table(fmap):
    """UID-free per-(item, group) view of a curve map: {"<model_type>:<name>|<group>": {curves, flag classes, keys}}.
    Used to compare the FBX against the package manifest independently of exporter UIDs (W111)."""
    out = {}
    for r in fmap["curves"]:
        if "model" not in r:
            k = f"UNMAPPED|{r['status']}"
        else:
            k = f"{r['model_type']}:{r['model_name']}|{r['group']}"
        g = out.setdefault(k, {"curves": 0, "flag_classes": set(), "keys": set(), "status": set(), "channels": []})
        g["curves"] += 1
        g["flag_classes"].update(r["flag_classes"])
        g["keys"].add(r["keys"])
        g["status"].add(r["status"])
        if r.get("channel") is not None:
            g["channels"].append(r["channel"])
    return {k: {"curves": v["curves"], "flag_classes": sorted(v["flag_classes"]), "keys": sorted(v["keys"]),
                "status": sorted(v["status"]), "channels": sorted(v["channels"])} for k, v in sorted(out.items())}


# ================================================================================= producer (source side) ===
def source_group_key(data_path, armature_object):
    """Source F-curve data path -> (item key, group) or None if unsupported.  Items: ``object:<armature name>`` or
    ``bone:<bone name>``; groups: location / rotation / scale."""
    prop = data_path
    item = f"object:{armature_object}"
    if data_path.startswith('pose.bones["'):
        end = data_path.find('"]', 12)
        if end < 0:
            return None
        bone = data_path[12:end].replace('\\"', '"').replace("\\\\", "\\")
        prop = data_path[end + 3:] if data_path[end + 2:end + 3] == "." else None
        item = f"bone:{bone}"
        if prop is None:
            return None
    grp = {"location": "location", "rotation_quaternion": "rotation", "rotation_euler": "rotation",
           "rotation_axis_angle": "rotation", "scale": "scale"}.get(prop)
    return (item, grp) if grp else None


def producer_directives(fmap, source_groups, armature_object):
    """Per FBX curve of the single take: the directive from the uniformly-held/-smooth DRIVING source group.

    ``source_groups``: {"object:<n>|<g>" / "bone:<n>|<g>": sorted set of source key interpolations}.
    CONSTANT if the source group is uniformly CONSTANT; LINEAR if uniformly LINEAR; UNDRIVEN if the source has no
    F-curve for that (item, group); UNSUPPORTED otherwise (any mixture / BEZIER / other) -> the producer refuses."""
    out = {}
    for r in fmap["curves"]:
        if r["status"] != "OK":
            out[r["uid"]] = ("UNMATCHED", [], None)
            continue
        if r["model_type"] == "Null":
            key = f"object:{r['model_name']}|{r['group']}" if r["model_name"] == armature_object else None
        else:
            key = f"bone:{r['model_name']}|{r['group']}"
        if key is None:
            out[r["uid"]] = ("UNMATCHED", [], None)
            continue
        interp = sorted(set(source_groups.get(key, [])))
        if not interp:
            d = "UNDRIVEN"
        elif interp == ["CONSTANT"]:
            d = "CONSTANT"
        elif interp == ["LINEAR"]:
            d = "LINEAR"
        else:
            d = "UNSUPPORTED"
        out[r["uid"]] = (d, interp, key)
    return out


def patch_flags(data, flag_locations, target_uids, new_value, expected_old=EXPORTER_FLAG):
    """Byte-exact in-place rewrite of the KeyAttrFlags values of ``target_uids`` only (same length; W109 rule).
    Refuses if a target array is compressed / not int32 / holds a value other than ``expected_old``."""
    out = bytearray(data)
    recs = []
    for uid in sorted(target_uids):
        p = flag_locations.get(uid)
        if p is None:
            raise InterchangeStop("PRODUCER_FLAG_TARGET_MISSING", str(uid))
        if p["type"] != b"i" or p["encoding"] != 0 or p["size"] != 4 * p["length"] or p["length"] < 1:
            raise InterchangeStop("PRODUCER_FLAG_ARRAY_UNSUPPORTED", f"curve {uid}: {p}")
        old = list(struct.unpack_from(f"<{p['length']}i", data, p["value_offset"]))
        if any(v != expected_old for v in old):
            raise InterchangeStop("PRODUCER_UNEXPECTED_EXPORTER_FLAG", f"curve {uid}: {[hex(v) for v in old]}")
        struct.pack_into(f"<{p['length']}i", out, p["value_offset"], *([int(new_value)] * p["length"]))
        recs.append({"offset": p["value_offset"], "bytes": p["size"]})
    if len(out) != len(data):
        raise AssertionError("length changed")
    return bytes(out), recs


def byte_diff_confined(a, b, spans):
    if len(a) != len(b):
        return False, None
    allowed = bytearray(len(a))
    for s, n in spans:
        allowed[s:s + n] = b"\x01" * n
    nd = 0
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            nd += 1
            if not allowed[i]:
                return False, nd
    return True, nd


# ===================================================================================== consumer planning ===
def escape_identifier(s):
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _rot_path(mode):
    if mode == "QUATERNION":
        return "rotation_quaternion", 4
    if mode == "AXIS_ANGLE":
        return "rotation_axis_angle", 4
    return "rotation_euler", 3


def plan(fmap, scene):
    """Resolve the FBX correspondence against an imported-scene description (W110 rule, maintained copy).  Returns
    per-group directives (CONSTANT / LINEAR / UNRESOLVED / NO_FBX_CURVES) and the complete accounting."""
    res = {"stack": fmap["stack"], "status": fmap["status"], "items": {}, "groups": [], "unresolved": []}
    if fmap["status"] != "OK":
        res["accounting"] = {"fbx_curves": len(fmap["curves"]), "fbx_curves_mapped": 0}
        return res
    curves_by_model = {}
    for r in fmap["curves"]:
        if "model" in r:
            curves_by_model.setdefault(r["model"], []).append(r)
    unmodelled = [r for r in fmap["curves"] if "model" not in r]
    objects = scene["objects"]
    claimed = {}
    for m, recs in sorted(curves_by_model.items()):
        info = fmap["models"][str(m)]
        it = {"name": info["name"], "type": info["type"], "status": "OK"}
        if info["type"] == "Null":
            if info["name"] in objects:
                it.update({"id_name": info["name"], "prefix": "", "rotation_mode": objects[info["name"]]["rotation_mode"]})
            else:
                it["status"] = "NO_OBJECT"
        else:
            arms = [o for o, d in objects.items() if d["type"] == "ARMATURE" and d.get("bones") and info["name"] in d["bones"]]
            if len(arms) != 1:
                it["status"] = f"BONE_OWNER_COUNT({len(arms)})"
            else:
                b = objects[arms[0]]["bones"][info["name"]]
                want = None if info["parent_type"] == "Null" and info["parent"] == arms[0] else info["parent"]
                if info["parent_count"] != 1 or b["parent"] != want:
                    it["status"] = "PARENT_MISMATCH"
                it.update({"id_name": arms[0], "prefix": f'pose.bones["{escape_identifier(info["name"])}"].',
                           "rotation_mode": b["rotation_mode"]})
        if it["status"] == "OK":
            act = f'{it["id_name"]}|{fmap["stack"]}'
            a = scene["actions"].get(act)
            if a is None:
                it["status"] = "NO_ACTION"
            elif len(a["slots"]) != 1:
                it["status"] = f"SLOT_COUNT({len(a['slots'])})"
            else:
                it.update({"action": act, "slot": a["slots"][0]})
        res["items"][info["name"]] = it
        ok_recs = [r for r in recs if r["status"] == "OK"]
        coupled = not info["separable"]
        all_classes = sorted({r["flag_class"] for r in recs})
        for g in ("location", "rotation", "scale"):
            grecs = [r for r in recs if r.get("group") == g]
            grp = {"item": info["name"], "item_type": info["type"], "group": g, "fbx_curves": len(grecs),
                   "coupled": coupled, "fcurves": []}
            if it["status"] == "OK":
                prop, n = _rot_path(it["rotation_mode"]) if g == "rotation" else (g, 3)
                fl = scene["actions"][it["action"]]["fcurves"][it["slot"]]
                for i in range(n):
                    dp = it["prefix"] + prop
                    grp["fcurves"].append([dp, i, sum(1 for p, j in fl if p == dp and j == i)])
            reasons = []
            if it["status"] != "OK":
                reasons.append(it["status"])
            if any(h != 1 for _, _, h in grp["fcurves"]):
                reasons.append("FCURVE_MISSING_OR_DUPLICATE")
            bad = [r for r in grecs if r["status"] != "OK"]
            if bad:
                reasons.append("FBX_CURVE_" + "/".join(sorted({r["status"] for r in bad})))
            if coupled:
                cls = all_classes if len(ok_recs) == len(recs) else ["<bad curve>"]
            else:
                cls = sorted({r["flag_class"] for r in grecs})
            if not grecs and not coupled:
                directive = "NO_FBX_CURVES"
            elif reasons:
                directive = "UNRESOLVED"
            elif len(cls) != 1:
                directive = "UNRESOLVED"
                reasons.append("FLAG_DISAGREEMENT" + ("_COUPLED_ITEM" if coupled else ""))
            elif cls[0] not in SUPPORTED:
                directive = "UNRESOLVED"
                reasons.append(f"UNSUPPORTED_FLAG_{cls[0]}")
            else:
                directive = cls[0]
            grp["directive"], grp["reasons"] = directive, reasons
            if it["status"] == "OK":
                grp["action"], grp["slot"] = it["action"], it["slot"]
                for dp, i, _ in grp["fcurves"]:
                    claimed.setdefault((it["action"], it["slot"], dp, i), []).append(len(res["groups"]))
            res["groups"].append(grp)
            if directive == "UNRESOLVED":
                res["unresolved"].append({"item": info["name"], "group": g, "reasons": reasons})
    for key, gi in claimed.items():
        if len(gi) > 1:
            for k in gi:
                if res["groups"][k]["directive"] in SUPPORTED:
                    res["groups"][k]["directive"] = "UNRESOLVED"
                    res["groups"][k]["reasons"].append("FCURVE_CLAIMED_TWICE")
                    res["unresolved"].append({"item": res["groups"][k]["item"], "group": res["groups"][k]["group"],
                                              "reasons": ["FCURVE_CLAIMED_TWICE"]})
    for r in unmodelled:
        res["unresolved"].append({"item": None, "group": None, "reasons": [f"FBX_CURVE_{r['status']}"]})
    acts = sorted({g["action"] for g in res["groups"] if "action" in g})
    imported = {(a, s, p, i) for a in acts for s, fl in scene["actions"][a]["fcurves"].items() for p, i in fl}
    covered = set(claimed)
    n_dir = {}
    for g in res["groups"]:
        n_dir[g["directive"]] = n_dir.get(g["directive"], 0) + 1
    res["accounting"] = {"fbx_curves": len(fmap["curves"]), "groups": len(res["groups"]), "group_directives": n_dir,
                         "target_actions": acts, "imported_fcurves_in_target_actions": len(imported),
                         "imported_fcurves_mapped": len(imported & covered),
                         "imported_fcurves_unmapped": sorted(f"{a}:{p}[{i}]" for a, s, p, i in imported - covered)}
    return res


def group_directives_view(pl):
    """UID-free {"<item_type>:<item>|<group>": directive} of a plan (compared with the manifest's declaration)."""
    return {f"{g['item_type']}:{g['item']}|{g['group']}": g["directive"] for g in pl["groups"]}


# ============================================================================================ manifest ===
def canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def seal(obj, key="manifest_digest"):
    body = {k: v for k, v in obj.items() if k != key}
    body = json.loads(json.dumps(body))
    body[key] = hashlib.sha256(canon({k: v for k, v in body.items() if k != key})).hexdigest()
    return body


def seal_ok(obj, key="manifest_digest"):
    return isinstance(obj, dict) and obj.get(key) == hashlib.sha256(canon({k: v for k, v in obj.items() if k != key})).hexdigest()


def topology_digest(bones):
    """{bone: parent or None} -> SHA-256 (rest-pose independent rig identity: names + hierarchy)."""
    return hashlib.sha256(canon({str(k): v for k, v in sorted(bones.items())})).hexdigest()


# Known W105 BaseRig topology (all bones incl. fingers, names + parents) as built by the slice from
# examples/skinned_mannequin/BaseRig_Skinned.fbx (SHA-256 c2947ba7...).  Frozen before any W127 outcome.
KNOWN_BASERIG_TOPOLOGY_SHA256 = "ea931abd049055f04ffcef68da4624d129aa8245df4da2dc1aa2cb975ac38077"
KNOWN_RIG_NOTE = "W105 BaseRig (examples/skinned_mannequin/BaseRig_Skinned.fbx) as keyed by the W124-W126 slice"


# ============================================================================== Blender runtime (bpy only) ===
def describe_scene():
    import bpy
    from bpy_extras import anim_utils
    objs = {}
    for o in bpy.data.objects:
        d = {"type": o.type, "rotation_mode": o.rotation_mode, "bones": None}
        if o.type == "ARMATURE" and o.pose is not None:
            d["bones"] = {pb.name: {"parent": pb.parent.name if pb.parent else None, "rotation_mode": pb.rotation_mode}
                          for pb in o.pose.bones}
        objs[o.name] = d
    acts = {}
    for a in bpy.data.actions:
        fc = {}
        for s in a.slots:
            cb = anim_utils.action_get_channelbag_for_slot(a, s)
            fc[s.identifier] = [[f.data_path, f.array_index] for f in cb.fcurves] if cb is not None else []
        acts[a.name] = {"slots": [s.identifier for s in a.slots], "fcurves": fc}
    return {"objects": objs, "actions": acts}


def apply_plan(pl):
    """Set ONLY keyframe ``interpolation`` on the F-curves of CONSTANT / LINEAR groups.  Times, values, handles, curves,
    Actions, slots, rig, objects and timebase are never written."""
    import bpy
    import numpy as np
    from bpy_extras import anim_utils
    vals = {i.identifier: i.value for i in bpy.types.Keyframe.bl_rna.properties["interpolation"].enum_items}
    out = {"fcurves_changed": 0, "fcurves_already_correct": 0, "keys_changed": 0}
    for g in pl["groups"]:
        if g["directive"] not in SUPPORTED:
            continue
        a = bpy.data.actions[g["action"]]
        s = [x for x in a.slots if x.identifier == g["slot"]][0]
        cb = anim_utils.action_get_channelbag_for_slot(a, s)
        want = vals[g["directive"]]
        for dp, i, _ in g["fcurves"]:
            f = cb.fcurves.find(dp, index=i)
            n = len(f.keyframe_points)
            cur = np.empty(n, dtype=np.int32)
            f.keyframe_points.foreach_get("interpolation", cur)
            if bool((cur == want).all()):
                out["fcurves_already_correct"] += 1
                continue
            f.keyframe_points.foreach_set("interpolation", np.full(n, want, dtype=np.int32))
            out["fcurves_changed"] += 1
            out["keys_changed"] += int((cur != want).sum())
    return out


# ================================================================================== Blender-side producer ===
FAULT_ENV = "ANIMCV_EXP_FBXI_FAULT"   # tests / evidence only: after_raw_export | after_patch (producer), after_import (consumer)


def _fault(point):
    import os
    if os.environ.get(FAULT_ENV) == point:
        raise RuntimeError(f"INJECTED_FAULT {point}")


def _sha(path):
    from pathlib import Path
    h = hashlib.sha256()
    with open(Path(path), "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fbx_addon_version():
    try:
        import io_scene_fbx
        v = getattr(io_scene_fbx, "bl_info", {}).get("version")
        return ".".join(str(x) for x in v) if v else None
    except Exception:  # pragma: no cover
        return None


def source_census(arm):
    """The selected armature's assigned Action: every F-curve's group, interpolation set, key frames.  Records what the
    producer cannot represent truthfully (unknown paths, modifiers, non-integer keys, mixtures) as problems."""
    from bpy_extras import anim_utils
    ad = arm.animation_data
    act = ad.action if ad else None
    if act is None:
        raise InterchangeStop("UNSUPPORTED_NO_ASSIGNED_ACTION", arm.name)
    slot = getattr(ad, "action_slot", None)
    cb = anim_utils.action_get_channelbag_for_slot(act, slot) if slot is not None else None
    fcurves = list(cb.fcurves) if cb is not None else []
    if not fcurves:
        raise InterchangeStop("UNSUPPORTED_EMPTY_ACTION", act.name)
    groups, curves, problems = {}, [], []
    for fc in sorted(fcurves, key=lambda f: (f.data_path, f.array_index)):
        kp = list(fc.keyframe_points)
        interp = sorted({k.interpolation for k in kp})
        gk = source_group_key(fc.data_path, arm.name)
        rec = {"data_path": fc.data_path, "index": fc.array_index, "keys": len(kp), "interpolation": interp,
               "first": float(kp[0].co[0]) if kp else None, "last": float(kp[-1].co[0]) if kp else None,
               "non_integer_key_times": sum(1 for k in kp if abs(k.co[0] - round(k.co[0])) > 1e-6),
               "modifiers": len(fc.modifiers), "group": None if gk is None else f"{gk[0]}|{gk[1]}"}
        curves.append(rec)
        if gk is None or (gk[0].startswith("bone:") and gk[0][5:] not in arm.pose.bones):
            problems.append(("UNSUPPORTED_SOURCE_CHANNEL", fc.data_path))
            continue
        if rec["modifiers"]:
            problems.append(("UNSUPPORTED_FCURVE_MODIFIER", fc.data_path))
        if rec["non_integer_key_times"]:
            problems.append(("UNSUPPORTED_NON_INTEGER_KEY_TIMES", fc.data_path))
        if not kp:
            problems.append(("UNSUPPORTED_EMPTY_FCURVE", fc.data_path))
        g = groups.setdefault(rec["group"], {"interpolation": set(), "fcurves": 0, "keys": 0})
        g["interpolation"].update(interp)
        g["fcurves"] += 1
        g["keys"] += len(kp)
    groups = {k: {"interpolation": sorted(v["interpolation"]), "fcurves": v["fcurves"], "keys": v["keys"]}
              for k, v in sorted(groups.items())}
    for k, v in groups.items():
        if v["interpolation"] not in (["CONSTANT"], ["LINEAR"]):
            problems.append(("UNSUPPORTED_INTERPOLATION_MIXTURE" if len(v["interpolation"]) > 1 else
                             "UNSUPPORTED_INTERPOLATION_" + "_".join(v["interpolation"]), k))
    firsts = [c["first"] for c in curves if c["first"] is not None]
    lasts = [c["last"] for c in curves if c["last"] is not None]
    return {"action": act.name, "slot": getattr(slot, "identifier", None), "fcurves": curves, "groups": groups,
            "key_frame_min": min(firsts) if firsts else None, "key_frame_max": max(lasts) if lasts else None,
            "problems": problems}


def _scene_checks(arm):
    import bpy
    probs = []
    for o in bpy.data.objects:
        if o == arm:
            continue
        if o.type == "ARMATURE":
            probs.append(("UNSUPPORTED_MULTIPLE_ARMATURES", o.name))
        elif o.type == "MESH":
            mods = [m for m in o.modifiers if m.type == "ARMATURE"]
            if o.parent != arm or len(mods) != 1 or mods[0].object != arm:
                probs.append(("UNSUPPORTED_MESH_NOT_BOUND_TO_ARMATURE", o.name))
            if o.animation_data and (o.animation_data.action or len(o.animation_data.drivers) or len(o.animation_data.nla_tracks)):
                probs.append(("UNSUPPORTED_ANIMATED_MESH", o.name))
            sk = o.data.shape_keys
            if sk is not None and sk.animation_data is not None:
                probs.append(("UNSUPPORTED_SHAPE_KEY_ANIMATION", o.name))
    ad = arm.animation_data
    if ad is not None and len(ad.nla_tracks):
        probs.append(("UNSUPPORTED_NLA_TRACKS", arm.name))
    if ad is not None and len(ad.drivers):
        probs.append(("UNSUPPORTED_DRIVERS", arm.name))
    if len(arm.constraints) or any(len(pb.constraints) for pb in arm.pose.bones):
        probs.append(("UNSUPPORTED_CONSTRAINTS", arm.name))
    if arm.data.animation_data is not None and (arm.data.animation_data.action or len(arm.data.animation_data.drivers)):
        probs.append(("UNSUPPORTED_ARMATURE_DATA_ANIMATION", arm.name))
    return probs


def rig_record(arm):
    import bpy
    bones = {b.name: (b.parent.name if b.parent else None) for b in arm.data.bones}
    meshes = [{"name": o.name, "vertices": len(o.data.vertices), "vertex_groups": len(o.vertex_groups)}
              for o in sorted(bpy.data.objects, key=lambda x: x.name) if o.type == "MESH"]
    return {"armature_object": arm.name, "bones": bones, "bone_count": len(bones), "topology_sha256": topology_digest(bones),
            "mesh_objects": meshes}


def produce(blend, raw_fbx, fbx, take_action, report):
    """Open ``blend`` (never saved), check the declared regime, export ONE per-Action take without key simplification,
    rewrite only the KeyAttrFlags of CONSTANT-driven curves, verify the edit, and report.  Writes ``raw_fbx`` (exporter
    output, evidence) and ``fbx`` (the package FBX).  Raises InterchangeStop on anything unsupported."""
    import bpy
    from pathlib import Path
    for p in (raw_fbx, fbx):
        if Path(p).exists():
            raise InterchangeStop("REFUSED_EXISTING_OUTPUT", str(p))
    pre = _sha(blend)
    bpy.ops.wm.open_mainfile(filepath=str(blend))
    arms = [o for o in bpy.data.objects if o.type == "ARMATURE"]
    if len(arms) != 1:
        raise InterchangeStop("UNSUPPORTED_ARMATURE_COUNT", str(len(arms)))
    arm = arms[0]
    rig = rig_record(arm)
    if rig["topology_sha256"] != KNOWN_BASERIG_TOPOLOGY_SHA256:
        raise InterchangeStop("UNSUPPORTED_RIG", f"armature topology {rig['topology_sha256'][:12]} is not the known BaseRig")
    probs = _scene_checks(arm)
    cen = source_census(arm)
    probs += cen["problems"]
    s = bpy.context.scene
    tb = {"fps": int(s.render.fps), "fps_base": float(s.render.fps_base), "frame_start": int(s.frame_start),
          "frame_end": int(s.frame_end), "key_frame_min": cen["key_frame_min"], "key_frame_max": cen["key_frame_max"]}
    if tb["fps_base"] != 1.0:
        probs.append(("UNSUPPORTED_TIMEBASE_FPS_BASE", str(tb["fps_base"])))
    act = arm.animation_data.action
    tb["action_frame_range"] = [float(act.frame_range[0]), float(act.frame_range[1])]
    if tb["frame_start"] != 1 or tb["action_frame_range"] != [1.0, float(tb["frame_end"])] or             cen["key_frame_min"] != 1.0 or cen["key_frame_max"] != float(tb["frame_end"]):
        # declared time convention: FBX take time 0 <-> Action frame 1 = scene frame_start; keys span the scene range
        probs.append(("UNSUPPORTED_TIME_ORIGIN_OR_RANGE", json.dumps({"scene": [tb["frame_start"], tb["frame_end"]],
                      "action_frame_range": tb["action_frame_range"], "keys": [cen["key_frame_min"], cen["key_frame_max"]]})))
    if probs:
        raise InterchangeStop(probs[0][0], json.dumps(probs[:20]))
    other_actions = sorted(a.name for a in bpy.data.actions if a.name != cen["action"])
    for a in [a for a in bpy.data.actions if a != act]:     # IN MEMORY ONLY: exactly one take (the selected Action)
        bpy.data.actions.remove(a)
    act.name = take_action                                   # in memory only: take name "<armature>|<take_action>"
    take = f"{arm.name}|{take_action}"
    s.frame_set(tb["frame_start"])
    Path(raw_fbx).parent.mkdir(parents=True, exist_ok=True)
    res = bpy.ops.export_scene.fbx(filepath=str(raw_fbx), **EXPORT_KWARGS)
    if "FINISHED" not in res or not Path(raw_fbx).exists():
        raise InterchangeStop("PRODUCER_EXPORT_FAILED", str(res))
    _fault("after_raw_export")
    data = Path(raw_fbx).read_bytes()
    root, version, flocs = read_fbx(data)
    inv = take_inventory(root)
    if len(inv["stacks"]) != 1 or inv["stacks"][0]["name"] != take or inv["stacks"][0]["layers"] != [take]:
        raise InterchangeStop("PRODUCER_TAKE_NOT_UNIQUE", json.dumps(inv["stacks"]))
    fmap = fbx_curve_map(root, take)
    if fmap["status"] != "OK":
        raise InterchangeStop("PRODUCER_CURVE_MAP_" + fmap["status"])
    dirs = producer_directives(fmap, {k: v["interpolation"] for k, v in cen["groups"].items()}, arm.name)
    bad = sorted({(d, str(k)) for d, _, k in dirs.values() if d in ("UNMATCHED", "UNSUPPORTED")})
    if bad:
        raise InterchangeStop("PRODUCER_UNMAPPED_OR_UNSUPPORTED_CURVES", json.dumps(bad[:20]))
    missing = sorted(set(cen["groups"]) - {k for _, _, k in dirs.values()})
    if missing:
        raise InterchangeStop("PRODUCER_SOURCE_GROUP_NOT_EXPORTED", json.dumps(missing))
    by_uid = {c["uid"]: c for c in fmap["curves"]}
    objs = _child(root, b"Objects")
    nodes = {int(c.props[0]): c for c in objs.elems if len(c.props) >= 2}
    varying = []
    for u, (d, _, k) in dirs.items():      # an UNDRIVEN group must be constant-valued (else its interpolation would matter)
        if d == "UNDRIVEN":
            kv = _child(nodes[u], b"KeyValueFloat")
            vals = list(kv.props[0]) if kv is not None else []
            if vals and (max(vals) - min(vals)) > 1e-4 * max(1.0, max(abs(x) for x in vals)):
                varying.append(k)
    if varying:
        raise InterchangeStop("UNSUPPORTED_UNDRIVEN_GROUP_VARIES", json.dumps(sorted(set(varying))[:20]))
    const = sorted(u for u, (d, _, _) in dirs.items() if d == "CONSTANT")
    new, spans = patch_flags(data, flocs, const, CONSTANT_FLAG) if const else (data, [])
    ok_span, nd = byte_diff_confined(data, new, [(r["offset"], r["bytes"]) for r in spans])
    _fault("after_patch")
    Path(fbx).write_bytes(new)
    root2, _, _ = read_fbx(Path(fbx).read_bytes())
    gt1, gt2 = group_table(fmap), group_table(fbx_curve_map(root2, take))
    pd = {}
    for u, (d, _, _) in dirs.items():
        r = by_uid[u]
        k = f"{r['model_type']}:{r['model_name']}|{r['group']}"
        if pd.setdefault(k, d) != d:
            raise InterchangeStop("PRODUCER_GROUP_DIRECTIVE_INCONSISTENT", k)
    cons = {k: ("CONSTANT" if v == "CONSTANT" else "LINEAR") for k, v in pd.items()}

    def strip(t):
        return {k: {kk: vv for kk, vv in v.items() if kk != "flag_classes"} for k, v in t.items()}
    checks = {"byte_diff_confined_to_flag_values": ok_span, "same_length": len(new) == len(data),
              "every_target_changed": (nd or 0) > 0 if const else nd == 0,
              "curve_map_unchanged_except_flags": strip(gt1) == strip(gt2),
              "decoded_flag_classes_as_directed": {k: v["flag_classes"] for k, v in gt2.items()} == {k: [v] for k, v in cons.items()},
              "take_inventory_unchanged": take_inventory(root2) == inv}
    if not all(checks.values()):
        raise InterchangeStop("PRODUCER_PATCH_VERIFICATION_FAILED", json.dumps(checks))
    post = _sha(blend)
    if post != pre:
        raise InterchangeStop("PRODUCER_SOURCE_BLEND_CHANGED", f"{pre} -> {post}")
    st = inv["stacks"][0]
    rep = {"schema": "animcv_exp_fbx_interchange_produce_v1", "blender_version": bpy.app.version_string,
           "io_scene_fbx_version": _fbx_addon_version(), "source_blend_sha256_before": pre, "source_blend_sha256_after": post,
           "source_action": cen["action"], "source_slot": cen["slot"], "other_actions_in_file_not_exported": other_actions,
           "source_census": {"fcurves": cen["fcurves"], "groups": cen["groups"]}, "timebase": tb, "rig": rig,
           "exporter_kwargs": {k: sorted(v) if isinstance(v, set) else v for k, v in EXPORT_KWARGS.items()},
           "raw_fbx_sha256": hashlib.sha256(data).hexdigest(), "raw_fbx_bytes": len(data),
           "fbx_sha256": _sha(fbx), "fbx_bytes": Path(fbx).stat().st_size, "fbx_version": version,
           "take": {"name": take, "layers": st["layers"], "stack_count": len(inv["stacks"]),
                    "local_start_ticks": st["local_start_ticks"], "local_stop_ticks": st["local_stop_ticks"],
                    "time_convention": "FBX take time 0 = Action frame 1 = scene frame_start (per-Action take; stock "
                                       "importer default anim_offset 1.0)",
                    "local_start_frame": ticks_to_frame(st["local_start_ticks"], tb["fps"]) + tb["frame_start"],
                    "local_stop_frame": ticks_to_frame(st["local_stop_ticks"], tb["fps"]) + tb["frame_start"]},
           "global_settings": inv["global_settings"], "fbx_curves": len(fmap["curves"]),
           "fbx_keys_total": sum(c["keys"] for c in fmap["curves"]), "patched_curves": len(const), "patch_checks": checks,
           "fbx_group_table": gt2, "producer_directives": dict(sorted(pd.items())), "consumer_directives": dict(sorted(cons.items()))}
    Path(report).write_text(json.dumps(rep, sort_keys=True))
    return rep


# ================================================================================== Blender-side consumer ===
class Audit:
    """``sys.addaudithook`` recorder of every Python-level ``open`` (including module loads via io.open_code).  C-level
    reads by Blender itself are not visible to it (disclosed)."""

    def __init__(self):
        self.events = []

    def __call__(self, event, args):
        if event == "open" and args and isinstance(args[0], (str, bytes)):
            p = args[0].decode(errors="replace") if isinstance(args[0], bytes) else args[0]
            self.events.append([p, str(args[1]) if len(args) > 1 else None])


def classify_access(events, package_dir, out_paths, code_roots, install_roots, user_roots=()):
    """Each audited open -> DECLARED_INPUT (the package manifest / FBX), OUTPUT (the consumer's own outputs), CODE
    (``.py``/``.pyc`` under the repository code roots, or anything under the Blender / Python installation), UNDECLARED
    (anything else, including any other file inside the package directory).  BLENDER_USER_RESOURCES: Blender's own per-user
    configuration / extension cache (not an animation input; disclosed separately)."""
    import os
    from pathlib import Path

    def n(p):
        try:
            return os.path.normcase(str(Path(p).resolve()))
        except (OSError, ValueError):
            return os.path.normcase(str(p))
    pk = n(package_dir)
    declared = {n(Path(package_dir) / x) for x in (PACKAGE_MANIFEST, PACKAGE_FBX)}
    outs = [n(p) for p in out_paths]
    code = [n(r) for r in code_roots]
    inst = [n(r) for r in install_roots]
    usr = [n(r) for r in user_roots]
    cats = {"DECLARED_INPUT": set(), "OUTPUT": set(), "CODE": set(), "BLENDER_USER_RESOURCES": set(), "UNDECLARED": set()}
    for p, _ in events:
        q = n(p)
        if q in declared:
            cats["DECLARED_INPUT"].add(q)
        elif any(q == o or q.startswith(o + os.sep) for o in outs):
            cats["OUTPUT"].add(q)
        elif q == pk or q.startswith(pk + os.sep):
            cats["UNDECLARED"].add(q)
        elif any(q.startswith(r + os.sep) for r in inst):
            cats["CODE"].add(q)
        elif any(q.startswith(r + os.sep) for r in code) and q.endswith((".py", ".pyc")):
            cats["CODE"].add(q)
        elif any(q.startswith(r + os.sep) for r in usr) and not q.endswith((".fbx", ".blend", ".json", ".npz", ".npy")):
            cats["BLENDER_USER_RESOURCES"].add(q)
        else:
            cats["UNDECLARED"].add(q)
    return {k: sorted(v) for k, v in cats.items()}


def verify_manifest_bytes(raw):
    try:
        man = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        raise InterchangeStop("MANIFEST_UNPARSABLE") from None
    if not isinstance(man, dict) or man.get("schema") != PACKAGE_SCHEMA:
        raise InterchangeStop("MANIFEST_SCHEMA_UNSUPPORTED", str(man.get("schema") if isinstance(man, dict) else None))
    if not seal_ok(man):
        raise InterchangeStop("MANIFEST_SEAL_INVALID", "the manifest does not bind its content (edited or torn)")
    return man


def consume(package_dir, out_blend, mode="adapter"):
    """Fresh factory-empty scene; ONLY the package manifest + FBX are read.  ``mode='adapter'``: package-aware recovery
    (interpolation from the FBX flags, cross-checked with the manifest; scene range from the manifest).  ``mode='raw'``:
    the stock importer alone (labelled RAW_FBX_STOCK_IMPORT; nothing recovered) for the A/B."""
    import bpy
    from pathlib import Path
    pdir = Path(package_dir)
    if Path(out_blend).exists():
        raise InterchangeStop("REFUSED_EXISTING_OUTPUT", str(out_blend))
    mp = pdir / PACKAGE_MANIFEST
    if not mp.exists():
        raise InterchangeStop("PACKAGE_INCOMPLETE", "no sealed package manifest (an export that did not complete is never a package)")
    mraw = mp.read_bytes()
    man = verify_manifest_bytes(mraw)
    ver = ".".join(str(x) for x in bpy.app.version[:2])
    if ver not in SUPPORTED_BLENDER:
        raise InterchangeStop("UNSUPPORTED_CONSUMER_BLENDER_VERSION", f"{bpy.app.version_string} not in {SUPPORTED_BLENDER}")
    pv = str(man["producer"]["blender_version"]).split(" ")[0]
    if ".".join(pv.split(".")[:2]) not in SUPPORTED_BLENDER:
        raise InterchangeStop("UNSUPPORTED_PRODUCER_BLENDER_VERSION", pv)
    if man["producer"].get("io_scene_fbx_version") != _fbx_addon_version():
        raise InterchangeStop("UNSUPPORTED_IMPORTER_VERSION", f"{man['producer'].get('io_scene_fbx_version')} vs {_fbx_addon_version()}")
    if man["fbx"]["file"] != PACKAGE_FBX or not (pdir / PACKAGE_FBX).exists():
        raise InterchangeStop("PACKAGE_INCOMPLETE", "package FBX missing")
    fp = pdir / PACKAGE_FBX
    data = fp.read_bytes()
    if hashlib.sha256(data).hexdigest() != man["fbx"]["sha256"]:
        raise InterchangeStop("FBX_HASH_MISMATCH", "the FBX is not the one the manifest declares")
    root, _, _ = read_fbx(data)
    inv = take_inventory(root)
    tk, tb, rig = man["take"], man["timebase"], man["rig"]
    if len(inv["stacks"]) != 1:
        raise InterchangeStop("TAKE_NOT_UNIQUE", json.dumps([s["name"] for s in inv["stacks"]]))
    st = inv["stacks"][0]
    if st["name"] != tk["name"] or st["layers"] != [tk["name"]]:
        raise InterchangeStop("TAKE_MISMATCH", f"FBX take {st['name']!r} vs manifest {tk['name']!r}")
    gs = inv["global_settings"]
    if gs["CustomFrameRate"] is None or abs(gs["CustomFrameRate"] - tb["fps"] / tb["fps_base"]) > 1e-9:
        raise InterchangeStop("TIMEBASE_MISMATCH", f"FBX frame rate {gs['CustomFrameRate']} vs manifest {tb['fps']}/{tb['fps_base']}")
    f0 = ticks_to_frame(st["local_start_ticks"], tb["fps"]) + tb["frame_start"]
    f1 = ticks_to_frame(st["local_stop_ticks"], tb["fps"]) + tb["frame_start"]
    if abs(f0 - tb["frame_start"]) > 1e-6 or abs(f1 - tb["frame_end"]) > 1e-6:
        raise InterchangeStop("TAKE_RANGE_MISMATCH", f"FBX take {f0}..{f1} vs manifest {tb['frame_start']}..{tb['frame_end']}")
    if rig["topology_sha256"] != KNOWN_BASERIG_TOPOLOGY_SHA256 or topology_digest(rig["bones"]) != rig["topology_sha256"]:
        raise InterchangeStop("UNSUPPORTED_RIG", "the package does not declare the known BaseRig")
    fmap = fbx_curve_map(root, tk["name"])
    if fmap["status"] != "OK":
        raise InterchangeStop("FBX_CURVE_MAP_" + fmap["status"])
    gt = group_table(fmap)
    if gt != man["fbx"]["group_table"]:
        diff = sorted(k for k in set(gt) | set(man["fbx"]["group_table"]) if gt.get(k) != man["fbx"]["group_table"].get(k))
        raise InterchangeStop("METADATA_FBX_DISAGREE", json.dumps(diff[:20]))
    # ---- the stock importer with its default flags (unchanged, not patched) ----
    bpy.ops.wm.read_factory_settings(use_empty=True)
    try:
        res = bpy.ops.import_scene.fbx(filepath=str(fp))
    except Exception as e:  # noqa: BLE001
        raise InterchangeStop("CONSUMER_IMPORT_FAILED", f"{type(e).__name__}: {e}") from None
    if "FINISHED" not in res:
        raise InterchangeStop("CONSUMER_IMPORT_FAILED", str(res))
    _fault("after_import")
    arms = [o for o in bpy.data.objects if o.type == "ARMATURE"]
    if len(arms) != 1:
        raise InterchangeStop("RIG_MISMATCH", f"{len(arms)} armatures")
    arm = arms[0]
    got = {b.name: (b.parent.name if b.parent else None) for b in arm.data.bones}
    if arm.name != rig["armature_object"] or got != rig["bones"]:
        raise InterchangeStop("RIG_MISMATCH", f"imported armature {arm.name!r}/{topology_digest(got)[:12]} vs manifest "
                              f"{rig['armature_object']!r}/{rig['topology_sha256'][:12]}")
    others = sorted([o.name, o.type] for o in bpy.data.objects if o != arm)
    if others != sorted([m["name"], "MESH"] for m in rig["mesh_objects"]):
        raise InterchangeStop("SCENE_OBJECTS_MISMATCH", json.dumps(others))
    want_act = f"{arm.name}|{tk['name']}"
    acts = sorted(a.name for a in bpy.data.actions)
    if acts != [want_act]:
        raise InterchangeStop("ACTION_NOT_UNIQUE_OR_WRONG_TAKE", json.dumps(acts))
    act = bpy.data.actions[want_act]
    if not arm.animation_data or arm.animation_data.action != act or len(act.slots) != 1:
        raise InterchangeStop("ACTION_NOT_ASSIGNED", want_act)
    sc = bpy.context.scene
    if int(sc.render.fps) != int(tb["fps"]) or abs(sc.render.fps_base - tb["fps_base"]) > 1e-9:
        raise InterchangeStop("TIMEBASE_MISMATCH", f"imported scene {sc.render.fps}/{sc.render.fps_base} vs manifest")
    out = {"schema": CONSUMER_SCHEMA, "mode": mode, "blender_version": bpy.app.version_string,
           "io_scene_fbx_version": _fbx_addon_version(), "package_manifest_sha256": hashlib.sha256(mraw).hexdigest(),
           "manifest_digest": man["manifest_digest"], "fbx_sha256": man["fbx"]["sha256"], "source_identity": man["source"],
           "take": tk["name"], "action": want_act, "imported_scene_range_before": [sc.frame_start, sc.frame_end]}
    if mode == "adapter":
        from bpy_extras import anim_utils
        pl = plan(fmap, describe_scene())
        if pl["status"] != "OK" or pl["unresolved"] or pl["accounting"]["imported_fcurves_unmapped"]:
            raise InterchangeStop("UNRESOLVED_CURVES", json.dumps({"unresolved": pl["unresolved"][:10],
                                                                    "unmapped": pl["accounting"].get("imported_fcurves_unmapped", [])[:10]}))
        view = group_directives_view(pl)
        want = man["interpolation"]["consumer_directives"]
        if view != want:
            diff = sorted(k for k in set(view) | set(want) if view.get(k) != want.get(k))
            raise InterchangeStop("METADATA_FBX_DISAGREE", "directives: " + json.dumps(diff[:20]))
        applied = apply_plan(pl)
        sc.frame_start, sc.frame_end = int(tb["frame_start"]), int(tb["frame_end"])
        cb = anim_utils.action_get_channelbag_for_slot(act, act.slots[0])
        kmin = min(f.keyframe_points[0].co[0] for f in cb.fcurves)
        kmax = max(f.keyframe_points[-1].co[0] for f in cb.fcurves)
        if abs(kmin - tb["frame_start"]) > 1e-6 or abs(kmax - tb["frame_end"]) > 1e-6:
            raise InterchangeStop("KEY_RANGE_MISMATCH", f"imported keys {kmin}..{kmax}")
        census = {}
        for g in pl["groups"]:
            for dp, i, _ in g["fcurves"]:
                ip = sorted({k.interpolation for k in cb.fcurves.find(dp, index=i).keyframe_points})
                if ip != [g["directive"]]:
                    raise InterchangeStop("POST_APPLY_CENSUS_MISMATCH", f"{dp}[{i}] {ip} vs {g['directive']}")
                census[g["directive"]] = census.get(g["directive"], 0) + 1
        out.update({"plan_accounting": pl["accounting"], "applied": applied, "recovered_interpolation_census": census,
                    "scene_range_restored_from_manifest": [sc.frame_start, sc.frame_end], "imported_key_range": [kmin, kmax],
                    "label": "PACKAGE_AWARE_ADAPTER_RECOVERY"})
    else:
        out["label"] = "RAW_FBX_STOCK_IMPORT (stock importer only; nothing recovered)"
    out["libraries_linked"] = len(bpy.data.libraries)
    out["blend_filepath_before_save"] = bpy.data.filepath
    bpy.ops.wm.save_as_mainfile(filepath=str(out_blend))
    out["recovered_blend_sha256"] = _sha(out_blend)
    return out


def build_manifest(prod, source, created_at, helper_sha, stage_sha):
    """Sealed package manifest from a producer report.  ``source`` is identity/provenance only: the consumer never opens
    anything it names."""
    return seal({
        "schema": PACKAGE_SCHEMA, "created_at": created_at,
        "fbx": {"file": PACKAGE_FBX, "sha256": prod["fbx_sha256"], "bytes": prod["fbx_bytes"], "version": prod["fbx_version"],
                "raw_exporter_output_sha256": prod["raw_fbx_sha256"], "patched_curves": prod["patched_curves"],
                "curves": prod["fbx_curves"], "keys_total": prod["fbx_keys_total"], "group_table": prod["fbx_group_table"],
                "exporter_kwargs": prod["exporter_kwargs"]},
        "take": prod["take"], "timebase": prod["timebase"], "global_settings": prod["global_settings"],
        "rig": dict(prod["rig"], known_rig=KNOWN_RIG_NOTE),
        "interpolation": {"regime": "PER_ITEM_TRANSFORM_GROUP_UNIFORM (CONSTANT or LINEAR); anything else is refused",
                          "carrier": "FBX AnimationCurve KeyAttrFlags (CONSTANT 0x6002 / exporter LINEAR 0x6104); the manifest "
                                     "repeats the directives only as a cross-check",
                          "producer_directives": prod["producer_directives"], "consumer_directives": prod["consumer_directives"],
                          "source_groups": prod["source_census"]["groups"]},
        "source": source,
        "producer": {"blender_version": prod["blender_version"], "io_scene_fbx_version": prod["io_scene_fbx_version"],
                     "helper_sha256_lf": helper_sha, "stage_sha256_lf": stage_sha,
                     "other_actions_in_source_file_not_exported": prod["other_actions_in_file_not_exported"]},
        "consumer_requirements": {
            "required_inputs": [PACKAGE_MANIFEST, PACKAGE_FBX],
            "consumer": "AnimCV opt-in Blender consumer adapter " + ADAPTER_SCHEMA + " (experimental-video-action fbx-consume)",
            "supported_blender": list(SUPPORTED_BLENDER),
            "raw_fbx_alone": "NOT lossless: a stock importer reads every curve as LINEAR (held keys interpolate between "
                             "frames) and does not set the scene frame range; Unreal / Unity / other FBX consumers NOT TESTED"},
    })
