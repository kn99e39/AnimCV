#!/usr/bin/env python3
"""W109 evaluation contract: FBX importer CONSTANT-flag behavioural ownership (pure Python; evidence only).

Shared by the host stages, the Blender stages and the tests.  Nothing here touches Blender, a rig file or a
production module.  It defines

  * the fixed cases: the three W106-W108 real M_Main Actions (frozen W107 V1 FBX as the only key-data basis) and the
    W108 synthetic held-step control (both mutated), plus the W108 smooth LINEAR control (left LINEAR, never mutated),
  * the two predeclared CONSTANT encodings of FBX ``KeyAttrFlags`` (not a sweep) and the exact derivation rule from
    the W107/W108 source flag ``0x6104``,
  * an offset-aware binary FBX node index used *only* to locate and byte-patch the targeted ``KeyAttrFlags`` values
    in place (same length, so no node offset changes), and a canonical tree digest that works on both this repo's
    independent reader (W108 contract) and ``io_scene_fbx.parse_fbx`` trees, with the targeted flags masked,
  * the declared expected temporal functions (CONSTANT-standard holds the key at or before t, CONSTANT-next takes the
    next key), their error metrics, the per-import classification and the frozen CASE mapping.

No production exporter or importer function is modified or monkey-patched.
"""
from __future__ import annotations

import hashlib
import math
import struct
from collections import namedtuple

import numpy as np

import w108_fbx_subframe_contract as C8   # W108 independent reader, flag decoder, metrics (read-only reuse)

SCHEMA = "w109_fbx_constant_import_v1"

REAL_CASES = C8.REAL_CASES                       # complete_control, arm_natural_most, real_clip -- fixed
STEP_CONTROL = "syn_step"                        # W108 synthetic held-step control (mutated like the real cases)
SMOOTH_CONTROL = "syn_smooth"                    # W108 smooth LINEAR negative control: flags left LINEAR
TARGET_CASES = REAL_CASES + (STEP_CONTROL,)      # cases receiving the two CONSTANT variants
ALL_CASES = TARGET_CASES + (SMOOTH_CONTROL,)

CONTROL = "control"                              # unchanged byte copy of the V1 FBX
VARIANTS = ("CONSTANT_standard", "CONSTANT_next")

# --- FBX SDK FbxAnimCurveDef bit semantics (public SDK enumerations) ---------------------------------------------
FBX_CONSTANT = 0x00000002            # eInterpolationConstant
FBX_LINEAR = 0x00000004              # eInterpolationLinear
FBX_INTERP_MASK = C8.FBX_INTERP_MASK  # 0xE
FBX_CONSTANT_NEXT = C8.FBX_CONSTANT_NEXT  # 0x100: eConstantNext under CONSTANT; the same bit is eTangentAuto (only
#                                        meaningful for CUBIC) -- the exporter literal sets it as "tangent auto"
SOURCE_FLAG = 0x6104                 # W107/W108: LINEAR | tangent auto | time-independent | clamp-progressive
ALTERABLE_BITS = FBX_INTERP_MASK | FBX_CONSTANT_NEXT   # the only bits a variant may change


def variant_flag(old, variant):
    """Declared derivation: clear the interpolation field and the constant-mode bit, set CONSTANT, set eConstantNext
    only for CONSTANT_next.  All other bits (0x6000: generic time-independent / clamp-progressive tangent bits, inert
    for non-cubic keys) are retained unchanged."""
    if variant not in VARIANTS:
        raise ValueError(variant)
    new = (int(old) & ~ALTERABLE_BITS) | FBX_CONSTANT
    if variant == "CONSTANT_next":
        new |= FBX_CONSTANT_NEXT
    return new


EXPECTED_FLAGS = {v: variant_flag(SOURCE_FLAG, v) for v in VARIANTS}   # {standard: 0x6002, next: 0x6102}


def decode_flag(flag):
    f = int(flag)
    interp = C8.decode_interp(f)
    return {"hex": f"0x{f:x}", "interpolation": interp,
            "constant_mode": (("NEXT" if f & FBX_CONSTANT_NEXT else "STANDARD") if interp == "CONSTANT" else None),
            "bit_0x100_meaning": ("eConstantNext" if interp == "CONSTANT" else "eTangentAuto (cubic-only)"),
            "other_bits_hex": f"0x{f & ~ALTERABLE_BITS:x}"}


EXPECTED_FUNCTION = {
    "CONSTANT_standard": "floor",   # value at t = key at or before t (held previous key)
    "CONSTANT_next": "ceil",        # value at t in (k, k+1) = key at k+1 (next key)
}

# --- tolerances (fixed a priori; metres unless noted) -------------------------------------------------------------
TOL = {
    "deterministic_m": 1e-9,          # repeat, P0 replay vs W108/W107 samples, integer frames vs control
    "w108_metric_reproduce_m": 1e-9,  # control subframe max vs the W108 analysis value
    "source_fidelity_m": 0.0,         # fresh source samples vs W108 source.npz (bitwise)
    "follows_declared_m": 1e-3,       # subframe max error vs the declared CONSTANT function (W106 1 mm gate)
    "discriminable_m": 1e-2,          # control must differ from both declared functions by > 1 cm somewhere
    "smooth_sufficiency_m": 1e-3,     # smooth control (LINEAR) subframe max vs source
    "smooth_min_interval_change_m": 1e-6,
}


# --- offset-aware binary FBX index (locating the bytes to patch only) --------------------------------------------
INode = namedtuple("INode", ("id", "props", "elems"))     # props: list of dicts (type, value_offset, ...)
_SCALAR_FMT = {b"Y": "<h", b"C": "<b", b"I": "<i", b"F": "<f", b"D": "<d", b"L": "<q", b"Z": "<b"}
_ARRAY_SIZE = {b"f": 4, b"d": 8, b"l": 8, b"i": 4, b"b": 1, b"c": 1}


def fbx_index(data):
    """Top-level node list with, per property, its type, the byte offset of its payload and (arrays) the length,
    encoding and payload size.  Scalars carry their decoded value; ``L`` scalars decode as int64."""
    if not data.startswith(C8.FBX_MAGIC):
        raise ValueError("not a binary FBX")
    version = struct.unpack_from("<I", data, 23)[0]
    hdr = struct.Struct("<QQQ" if version >= 7500 else "<III")
    null = hdr.size + 1

    def read_node(off):
        end, nprops, _plen = hdr.unpack_from(data, off)
        off += hdr.size
        nlen = data[off]
        off += 1
        if end == 0:
            return None, off + nlen
        name = data[off:off+nlen]
        off += nlen
        props = []
        for _ in range(nprops):
            t = data[off:off+1]
            off += 1
            if t in _SCALAR_FMT:
                n = struct.calcsize(_SCALAR_FMT[t])
                val = struct.unpack_from(_SCALAR_FMT[t], data, off)[0]
                props.append({"type": t, "value_offset": off, "size": n, "value": val})
                off += n
            elif t in _ARRAY_SIZE:
                n, enc, clen = struct.unpack_from("<III", data, off)
                props.append({"type": t, "header_offset": off, "value_offset": off + 12, "length": n,
                              "encoding": enc, "size": clen, "stride": _ARRAY_SIZE[t]})
                off += 12 + clen
            elif t in (b"S", b"R"):
                n = struct.unpack_from("<I", data, off)[0]
                props.append({"type": t, "value_offset": off + 4, "size": n, "value": data[off+4:off+4+n]})
                off += 4 + n
            else:
                raise ValueError(f"unknown property type {t!r} at {off-1}")
        elems = []
        while off < end:
            if end - off == null and data[off:end] == b"\x00"*null:
                off = end
                break
            child, off = read_node(off)
            if child is None:
                break
            elems.append(child)
        return INode(name, props, elems), end

    off, top = 27, []
    while off < len(data):
        node, off2 = read_node(off)
        if node is None:
            break
        top.append(node)
        off = off2
    return INode(b"", [], top), version


def curve_flag_locations(data):
    """{AnimationCurve uid: KeyAttrFlags property record} for every Objects/AnimationCurve."""
    root, _ = fbx_index(data)
    out = {}
    for top in root.elems:
        if top.id != b"Objects":
            continue
        for c in top.elems:
            if c.id != b"AnimationCurve":
                continue
            uid = int(c.props[0]["value"])
            fl = [e for e in c.elems if e.id == b"KeyAttrFlags"]
            if uid in out or len(fl) != 1 or len(fl[0].props) != 1:
                raise ValueError(f"AnimationCurve {uid}: duplicate uid or not exactly one KeyAttrFlags array")
            out[uid] = fl[0].props[0]
    return out


def patch_flags(data, target_uids, new_value, expected_old=SOURCE_FLAG):
    """Byte-exact in-place rewrite of the KeyAttrFlags values of ``target_uids`` only.  Refuses (ValueError) if a
    target array is compressed, has an unexpected type/size, or holds any value other than ``expected_old``.
    Returns (new bytes, per-target records with the byte span written)."""
    locs = curve_flag_locations(data)
    out = bytearray(data)
    recs = []
    for uid in sorted(target_uids):
        if uid not in locs:
            raise ValueError(f"target curve {uid} not found")
        p = locs[uid]
        if p["type"] != b"i" or p["encoding"] != 0 or p["size"] != 4*p["length"] or p["length"] < 1:
            raise ValueError(f"curve {uid}: KeyAttrFlags not an uncompressed int32 array ({p})")
        old = list(struct.unpack_from(f"<{p['length']}i", data, p["value_offset"]))
        if any(v != expected_old for v in old):
            raise ValueError(f"curve {uid}: KeyAttrFlags {[hex(v) for v in old]} != declared source "
                             f"0x{expected_old:x}")
        struct.pack_into(f"<{p['length']}i", out, p["value_offset"], *([int(new_value)]*p["length"]))
        recs.append({"uid": uid, "offset": p["value_offset"], "bytes": p["size"], "old": [f"0x{v:x}" for v in old],
                     "new": f"0x{int(new_value):x}"})
    if len(out) != len(data):
        raise AssertionError("length changed")
    return bytes(out), recs


def byte_diff_spans(a, b, spans):
    """Positions where ``a`` and ``b`` differ; whether all lie inside ``spans`` [(start, nbytes)]."""
    if len(a) != len(b):
        return {"same_length": False, "diff_bytes": None, "all_in_targets": False}
    x = np.frombuffer(a, dtype=np.uint8)
    y = np.frombuffer(b, dtype=np.uint8)
    pos = np.nonzero(x != y)[0]
    allowed = np.zeros(len(a), dtype=bool)
    for s, n in spans:
        allowed[s:s+n] = True
    return {"same_length": True, "diff_bytes": int(pos.size), "all_in_targets": bool(allowed[pos].all()),
            "first_last": [int(pos[0]), int(pos[-1])] if pos.size else None}


# --- tree-level helpers for both parsers -------------------------------------------------------------------------
def _child(e, name):
    for c in e.elems:
        if c.id == name:
            return c
    return None


def curve_stack_map(root):
    """{AnimationCurve uid: stack name} via Connections (curve -> curve node -> layer -> stack), on any tree of
    (id, props, props_type, elems) nodes.  Curves whose chain is missing or ambiguous map to None."""
    objs, conns = _child(root, b"Objects"), _child(root, b"Connections")
    kind, name = {}, {}
    for c in objs.elems:
        if c.id in (b"AnimationStack", b"AnimationLayer", b"AnimationCurveNode", b"AnimationCurve"):
            uid = int(c.props[0])
            kind[uid] = c.id
            name[uid] = bytes(c.props[1]).split(b"\x00")[0].decode(errors="replace") if len(c.props) > 1 else ""
    parents = {}
    for c in conns.elems:
        if c.id == b"C" and len(c.props) >= 3:
            parents.setdefault(int(c.props[1]), []).append(int(c.props[2]))

    def up(uid, want):
        p = [q for q in parents.get(uid, []) if kind.get(q) == want]
        return p[0] if len(p) == 1 else None
    out = {}
    for uid, k in kind.items():
        if k != b"AnimationCurve":
            continue
        cn = up(uid, b"AnimationCurveNode")
        ly = up(cn, b"AnimationLayer") if cn is not None else None
        st = up(ly, b"AnimationStack") if ly is not None else None
        out[uid] = name[st] if st is not None else None
    return out


def target_uids(root, action):
    """Curves of the uniquely name-matched stack (exact name or '<obj>|<action>')."""
    m = curve_stack_map(root)
    stacks = sorted({s for s in m.values() if s is not None})
    sel = C8.select_stack(stacks, action)
    return sel, sorted(u for u, s in m.items() if s == sel and sel is not None), m


def curve_flags(root):
    """{uid: [KeyAttrFlags values]} for every AnimationCurve."""
    out = {}
    for c in _child(root, b"Objects").elems:
        if c.id == b"AnimationCurve":
            kf = _child(c, b"KeyAttrFlags")
            out[int(c.props[0])] = [int(x) for x in kf.props[0]] if kf is not None else []
    return out


_ARR_DT = {"f": "<f4", "d": "<f8", "l": "<i8", "i": "<i4", "b": "<i1", "c": "<u1"}
_SC_FMT = {"Y": "<h", "I": "<i", "F": "<f", "D": "<d", "L": "<q", "Z": "<b"}


def _canon(t, v):
    if t in _ARR_DT:
        return np.asarray(v).astype(_ARR_DT[t], copy=False).tobytes()
    if t in ("S", "R"):
        return bytes(v)
    if t == "C":   # add-on: 1-byte bytes; W108 reader: bool -- canonical boolean byte
        return b"\x01" if (v[0] if isinstance(v, (bytes, bytearray)) else int(v)) else b"\x00"
    return struct.pack(_SC_FMT[t], v)


def tree_digest(root, masked_uids=()):
    """SHA-256 of a canonical serialisation of the whole node tree (ids, property types and values, child order).
    The KeyAttrFlags values of AnimationCurves in ``masked_uids`` are replaced by a fixed token."""
    masked = {int(u) for u in masked_uids}
    h = hashlib.sha256()

    def walk(e, mask_flags):
        types = bytes(e.props_type).decode()
        h.update(struct.pack("<I", len(e.id)) + bytes(e.id) + struct.pack("<I", len(types)) + types.encode())
        for t, v in zip(types, e.props):
            b = b"<MASKED>" if (mask_flags and e.id == b"KeyAttrFlags") else _canon(t, v)
            h.update(struct.pack("<Q", len(b)) + b)
        h.update(struct.pack("<I", len(e.elems)))
        child_mask = e.id == b"AnimationCurve" and len(e.props) > 0 and int(e.props[0]) in masked
        for c in e.elems:
            walk(c, child_mask)
    for top in root.elems:
        walk(top, False)
    return h.hexdigest()


def file_tree_record(root, action, variant):
    """Everything both parsers must agree on for one file: digests (full / targets masked), selected stack, target
    uids, decoded target flags and all-curve flag census, plus the W108 per-stack animation summary."""
    sel, tg, smap = target_uids(root, action)
    fl = curve_flags(root)
    tflags = sorted({v for u in tg for v in fl[u]})
    others = sorted({v for u, vs in fl.items() if u not in set(tg) for v in vs})
    return {"variant": variant, "selected_stack": sel, "target_curves": len(tg),
            "target_uids_sha256": hashlib.sha256(",".join(map(str, tg)).encode()).hexdigest(),
            "curves_total": len(fl), "unmapped_curves": sum(s is None for s in smap.values()),
            "stacks": sorted({s for s in smap.values() if s is not None}),
            "target_flags_hex": [f"0x{v:x}" for v in tflags],
            "target_flags_decoded": [decode_flag(v) for v in tflags],
            "nontarget_flags_hex": [f"0x{v:x}" for v in others],
            "digest_full": tree_digest(root), "digest_targets_masked": tree_digest(root, tg)}


# --- evaluated temporal functions --------------------------------------------------------------------------------
def declared_function_error(imp, ctrl, times, mode, classes):
    """Max |coordinate| difference of world heads/tails between ``imp`` at every subframe time t and the control
    import's integer sample at floor(t) ('floor', CONSTANT-standard) or ceil(t) ('ceil', CONSTANT-next).  The control
    import's integer samples are the FBX key values as evaluated by Blender, so the reference isolates interpolation
    semantics from key data.  Reported overall, per phase and per interval class."""
    t = np.asarray(times, dtype=float)
    if not (np.array_equal(imp["times"], t) and np.array_equal(ctrl["times"], t)):
        raise ValueError("sample times differ")
    if [str(b) for b in imp["bones"]] != [str(b) for b in ctrl["bones"]]:
        raise ValueError("bone order differs")
    idx = {float(x): i for i, x in enumerate(t) if x == math.floor(x)}
    ph = C8.phase_of(t)
    sub = np.nonzero(ph > 0)[0]
    ref = np.array([idx[float(math.floor(t[i]) if mode == "floor" else math.floor(t[i]) + 1)] for i in sub])
    dh = np.abs(imp["world_head"][sub] - ctrl["world_head"][ref]).max(axis=(1, 2))
    dt = np.abs(imp["world_tail"][sub] - ctrl["world_tail"][ref]).max(axis=(1, 2))
    per = np.maximum(dh, dt)
    cls = np.array([classes[int(math.floor(t[i]))] for i in sub])
    out = {"mode": mode, "subframes": int(sub.size), "max_m": float(per.max()) if per.size else 0.0}
    for p in C8.SUB_PHASES:
        m = np.isclose(ph[sub], p)
        out[f"phase_{p:.2f}_m"] = float(per[m].max()) if m.any() else None
    for c in ("transition", "quiet"):
        m = cls == c
        out[f"{c}_m"] = float(per[m].max()) if m.any() else None
        out[f"{c}_subframes"] = int(m.sum())
    return out


def classify(rec):
    """Per (variant, case) importer-behaviour outcome.

    ``rec`` keys: imported_interpolation (sorted list), keydata_equal_all, keydata_equal_excl_interpolation,
    delta_vs_control_m, integer_vs_control_m, declared_error_m, discriminable.

    LINEAR_UNCHANGED  import is indistinguishable from the unchanged-control import (LINEAR, all key fields equal,
                      every sample within deterministic_m): the flag had no read or evaluated effect
    FOLLOWS_DECLARED  key data (excluding interpolation) and integer motion unchanged, and every subframe sample
                      within follows_declared_m of the variant's declared CONSTANT function
    NOT_DISCRIMINABLE control and declared function do not differ by > discriminable_m (no inference)
    OTHER             anything else (e.g. flag read but evaluated differently, or key data altered)"""
    T = TOL
    if not rec["discriminable"]:
        return "NOT_DISCRIMINABLE"
    if (rec["imported_interpolation"] == ["LINEAR"] and rec["keydata_equal_all"]
            and rec["delta_vs_control_m"] <= T["deterministic_m"]):
        return "LINEAR_UNCHANGED"
    if (rec["keydata_equal_excl_interpolation"] and rec["integer_vs_control_m"] <= T["deterministic_m"]
            and rec["declared_error_m"] <= T["follows_declared_m"]):
        return "FOLLOWS_DECLARED"
    return "OTHER"


def decide_case(pre):
    """``pre`` keys: preflight_ok, encoding_valid, sources_ok, matching_ok, p0 (dict case -> bool), deterministic,
    originals_unchanged, smooth_valid, results (dict variant -> dict case -> {outcome, keydata_ok, integer_ok}).

    D: preflight / provenance, flag encoding or unrelated-semantics preservation, source identity, Action/slot/rest
       matching, P0 baseline, determinism or originals failure -- no importer conclusion.
    A: both CONSTANT variants imported on all three real Actions and the step control, none follows its declared
       CONSTANT function, key values/times (excluding interpolation) and integer motion unchanged, smooth LINEAR
       negative control valid.
    B: CONSTANT_standard follows the declared held function on all three real Actions and the step control with key
       data and integer motion preserved; smooth control valid (no synthetic contradiction).
    C: otherwise (disagreement by Action / encoding / control, NOT_DISCRIMINABLE, altered key data, invalid smooth
       control)."""
    for k, why in (("preflight_ok", "preflight / provenance failed"),
                   ("encoding_valid", "flag encoding or unrelated-semantics preservation failed"),
                   ("sources_ok", "source Action identity mismatch"),
                   ("matching_ok", "imported Action / slot / rest / take matching failed"),
                   ("deterministic", "repeat disagrees"),
                   ("originals_unchanged", "an original artifact changed")):
        if not pre[k]:
            return "D", why
    if not all(pre["p0"].values()):
        return "D", f"P0 baseline failed: {sorted(c for c, v in pre['p0'].items() if not v)}"
    res = pre["results"]
    cells = [(v, c) for v in VARIANTS for c in TARGET_CASES]
    if any(res[v][c]["outcome"] == "NOT_DISCRIMINABLE" for v, c in cells):
        return "C", "a declared function is not discriminable from the control on some case"
    intact = all(res[v][c]["keydata_ok"] and res[v][c]["integer_ok"] for v, c in cells)
    if pre["smooth_valid"] and intact and all(res[v][c]["outcome"] != "FOLLOWS_DECLARED" for v, c in cells):
        return "A", "neither CONSTANT variant is imported as its declared function on any real Action or the step " \
                    "control; key data and integer motion unchanged; smooth control valid"
    std = [res["CONSTANT_standard"][c] for c in TARGET_CASES]
    if pre["smooth_valid"] and all(r["outcome"] == "FOLLOWS_DECLARED" and r["keydata_ok"] and r["integer_ok"]
                                   for r in std):
        return "B", "CONSTANT_standard imported as the declared held function on all real Actions and the step control"
    why = []
    if not pre["smooth_valid"]:
        why.append("smooth negative control invalid")
    if not intact:
        why.append("key data or integer motion altered: " + str(sorted(f"{v}:{c}" for v, c in cells
                                                                       if not (res[v][c]["keydata_ok"]
                                                                               and res[v][c]["integer_ok"]))))
    why.append("outcomes: " + str({f"{v}:{c}": res[v][c]["outcome"] for v, c in cells}))
    return "C", "; ".join(why)


sha256_file = C8.sha256_file
load_npz = C8.load_npz
