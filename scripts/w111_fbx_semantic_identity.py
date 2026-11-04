#!/usr/bin/env python3
"""W111 prospective FBX semantic identity (pure Python; evidence only, not a product feature).

A UID-independent canonical representation of a binary FBX file as written by Blender 5.1 / io_scene_fbx 5.15 with
the W107 V1 armature-only options, plus a strict comparison, a raw-variation accounting, a minimal binary writer used
ONLY to build scratch negative-control copies, the negative-control mutations and the UID normalisation of W110
adapter plans.  It works on ``w108_fbx_subframe_contract.fbx_read`` trees ((id, props, props_type, elems) nodes) and
imports nothing from Blender.

Canonicalisation rule (frozen by the W111 declaration; anything not listed below is RETAINED verbatim):

  Excluded exporter-volatile fields (and only these):
    * UID references: ``Objects/<element>`` property 0, ``Connections/C`` properties 1 and 2 (child, parent; parent 0
      is the scene root and is retained as ``ROOT``), ``Documents/Document`` property 0.  UID values are replaced by
      the referenced element's typed semantic identity (Objects, Connections) or a fixed token (Document).
    * ``FBXHeaderExtension/CreationTimeStamp`` (the whole element; must occur exactly once).
    * The value of ``FBXHeaderExtension/SceneInfo/Properties70/P "Original|ApplicationNativeFile"`` (the source
      .blend path; must occur exactly once and be an absolute ``*.blend`` path).
  Typed identity:
    * named classes (Model, NodeAttribute, AnimationStack, AnimationLayer): (class, subtype, raw name bytes); a
      repeated named identity is REJECTED (no disambiguation by connections, no name-based fallback);
    * anonymous classes (AnimationCurveNode, AnimationCurve): (class, subtype, raw name, sorted outgoing connections
      (type, property, destination identity)), resolved to a fixpoint; unresolved, orphaned or colliding identities
      are REJECTED;
    * any other Objects class is REJECTED (unexpected semantic field for the tested workflow).
  Relations (rejected when violated): every Model exactly one OO parent (a Model or the root); every NodeAttribute
  exactly one OO destination, a Model; every AnimationLayer exactly one OO destination, a Stack; every curve node
  exactly one OO destination (a Layer) and exactly one OP destination (a Model); every curve exactly one OP
  destination (a curve node); connections must reference existing elements (or parent 0); connection types OO/OP only.
  UID leak guard: any int64 scalar, or int64 array other than ``KeyTime``, outside the declared UID positions whose
  value equals an element UID or the Document UID is REJECTED.
  Order: Objects are keyed by identity (file order ignored); Connections are a sorted multiset; every other node keeps
  its file order and every property value (full bytes) is compared.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
import struct
from collections import Counter

import numpy as np

SCHEMA = "w111_fbx_semantic_identity_v1"
NAMED = ("Model", "NodeAttribute", "AnimationStack", "AnimationLayer")
ANONYMOUS = ("AnimationCurveNode", "AnimationCurve")
DECLARED_CLASSES = NAMED + ANONYMOUS
PATH_PROP = b"Original|ApplicationNativeFile"
PATH_RE = re.compile(r"^[A-Za-z]:[\\/].*\.blend$", re.IGNORECASE)
EXCLUDED_FIELDS = {
    "uid_references": "Objects/<element> prop 0; Connections/C props 1-2 (non-zero); Documents/Document prop 0",
    "creation_timestamp": "FBXHeaderExtension/CreationTimeStamp (whole element)",
    "source_path": "FBXHeaderExtension/SceneInfo/Properties70/P 'Original|ApplicationNativeFile' value (prop 4)"}

_ARR_DT = {"f": "<f4", "d": "<f8", "l": "<i8", "i": "<i4", "b": "<i1", "c": "<u1"}
_SC_FMT = {"Y": "<h", "I": "<i", "F": "<f", "D": "<d", "L": "<q", "Z": "<b"}


class Reject(Exception):
    """The canonicaliser refuses the file: ``code`` is one of the declared rejection classes."""

    def __init__(self, code, detail=""):
        super().__init__(f"{code}: {detail}")
        self.code, self.detail = code, detail


def _t(e):
    return bytes(e.props_type).decode()


def _val(t, v):
    """Full canonical bytes of one property value (W109 _canon semantics)."""
    if t in _ARR_DT:
        return np.asarray(v).astype(_ARR_DT[t], copy=False).tobytes()
    if t in ("S", "R"):
        return bytes(v)
    if t == "C":
        return b"\x01" if (v[0] if isinstance(v, (bytes, bytearray)) else int(v)) else b"\x00"
    return struct.pack(_SC_FMT[t], v)


def _child(e, name):
    for c in e.elems:
        if c.id == name:
            return c
    return None


def _children(e, name):
    return [c for c in e.elems if c.id == name]


def _subtree(e, replace=None, path=()):
    """Nested tuple (id, ((type, bytes), ...), (children...)).  ``replace(path, index, e)`` may return a token
    (bytes) that replaces a property value, or the sentinel ``_SKIP`` for the whole element."""
    if replace is not None:
        r = replace(path + (bytes(e.id),), None, e)
        if r is _SKIP:
            return (bytes(e.id), (("<EXCLUDED>", b"element"),), ())
    props = []
    for i, (t, v) in enumerate(zip(_t(e), e.props)):
        tok = replace(path + (bytes(e.id),), i, e) if replace is not None else None
        props.append((t, tok) if isinstance(tok, bytes) else (t, _val(t, v)))
    return (bytes(e.id), tuple(props), tuple(_subtree(c, replace, path + (bytes(e.id),)) for c in e.elems))


_SKIP = object()


def _digest(obj):
    h = hashlib.sha256()

    def walk(x):
        if isinstance(x, bytes):
            h.update(b"B" + struct.pack("<Q", len(x)) + x)
        elif isinstance(x, str):
            b = x.encode()
            h.update(b"S" + struct.pack("<Q", len(b)) + b)
        elif isinstance(x, tuple):
            h.update(b"T" + struct.pack("<Q", len(x)))
            for y in x:
                walk(y)
        elif x is None:
            h.update(b"N")
        elif isinstance(x, int):
            h.update(b"I" + str(x).encode() + b";")
        else:
            raise TypeError(type(x))
    walk(obj)
    return h.hexdigest()


def _subtype(e):
    return bytes(e.props[2]).decode("utf-8", "backslashreplace") if len(e.props) > 2 else ""


def canonicalize(root):
    """-> dict with ``objects`` {identity: subtree}, ``connections`` (sorted tuple), ``header`` {section index: subtree},
    ``uid_of`` {identity: uid}, ``order`` (file-order identities), ``accounting``.  Raises Reject."""
    tops = [bytes(e.id) for e in root.elems]
    for need in (b"Objects", b"Connections", b"FBXHeaderExtension", b"Documents"):
        if tops.count(need) != 1:
            raise Reject("MISSING_OR_REPEATED_SECTION", need.decode())
    objs, conns = _child(root, b"Objects"), _child(root, b"Connections")
    # --- elements and UIDs
    node, cls = {}, {}
    for c in objs.elems:
        cid = bytes(c.id).decode()
        if cid not in DECLARED_CLASSES:
            raise Reject("UNEXPECTED_OBJECT_CLASS", cid)
        if _t(c)[:3] != "LSS":
            raise Reject("UNEXPECTED_OBJECT_SIGNATURE", f"{cid} {_t(c)}")
        u = int(c.props[0])
        if u in node or u == 0:
            raise Reject("DUPLICATE_OR_ZERO_UID", str(u))
        node[u], cls[u] = c, cid
    doc = [d for d in _children(_child(root, b"Documents"), b"Document")]
    if len(doc) != 1 or _t(doc[0])[:1] != "L":
        raise Reject("DOCUMENT_COUNT", str(len(doc)))
    doc_uid = int(doc[0].props[0])
    # --- connections
    out = {u: [] for u in node}
    raw_conns = []
    for c in conns.elems:
        if c.id != b"C":
            raise Reject("UNEXPECTED_CONNECTION_ELEMENT", bytes(c.id).decode())
        t = _t(c)
        ctype = bytes(c.props[0]).decode() if t[:1] == "S" else None
        if ctype not in ("OO", "OP") or t not in ("SLL", "SLLS") or (ctype == "OP") != (t == "SLLS"):
            raise Reject("UNEXPECTED_CONNECTION_SIGNATURE", f"{ctype} {t}")
        ch, pa = int(c.props[1]), int(c.props[2])
        prop = bytes(c.props[3]) if t == "SLLS" else None
        if ch not in node or (pa != 0 and pa not in node):
            raise Reject("UNRESOLVED_CONNECTION", f"{ctype} {ch} -> {pa}")
        out[ch].append((ctype, prop, pa))
        raw_conns.append((ctype, ch, pa, prop))
    # --- relations
    for u, k in cls.items():
        oo = [p for t, _, p in out[u] if t == "OO"]
        op = [p for t, _, p in out[u] if t == "OP"]
        if k == "Model":
            ok = len(oo) == 1 and (oo[0] == 0 or cls.get(oo[0]) == "Model") and not op
        elif k == "NodeAttribute":
            ok = len(oo) == 1 and cls.get(oo[0]) == "Model" and not op
        elif k == "AnimationLayer":
            ok = len(oo) == 1 and cls.get(oo[0]) == "AnimationStack" and not op
        elif k == "AnimationStack":
            ok = not oo and not op
        elif k == "AnimationCurveNode":
            ok = (len(oo) == 1 and cls.get(oo[0]) == "AnimationLayer" and len(op) == 1 and cls.get(op[0]) == "Model")
        else:   # AnimationCurve
            ok = not oo and len(op) == 1 and cls.get(op[0]) == "AnimationCurveNode"
        if not ok:
            raise Reject("AMBIGUOUS_OR_MISSING_RELATION", f"{k} {u}: {out[u]}")
    # --- identities
    ident = {}
    for u, k in cls.items():
        if k in NAMED:
            ident[u] = f"{k}[{_subtype(node[u])}]{bytes(node[u].props[1])!r}"
    seen = Counter(ident.values())
    dup = sorted(i for i, n in seen.items() if n > 1)
    if dup:
        raise Reject("DUPLICATE_NAMED_IDENTITY", dup[0])
    pending = [u for u, k in cls.items() if k in ANONYMOUS]
    while pending:
        progressed, rest = False, []
        for u in pending:
            if all(p == 0 or p in ident for _, _, p in out[u]):
                sig = ";".join(sorted(f"{t}:{pr!r}->{'ROOT' if p == 0 else ident[p]}" for t, pr, p in out[u]))
                ident[u] = f"{cls[u]}[{_subtype(node[u])}]{bytes(node[u].props[1])!r}<{sig}>"
                progressed = True
            else:
                rest.append(u)
        if not progressed:
            raise Reject("UNRESOLVED_IDENTITY", f"{len(rest)} anonymous elements")
        pending = rest
    seen = Counter(ident.values())
    dup = sorted(i for i, n in seen.items() if n > 1)
    if dup:
        raise Reject("NON_INJECTIVE_IDENTITY", dup[0][:300])
    # --- UID leak guard
    uids = set(node) | {doc_uid}
    declared_pos = 0

    def scan(e, path):
        nonlocal declared_pos
        for i, (t, v) in enumerate(zip(_t(e), e.props)):
            is_decl = ((len(path) == 2 and path[0] == b"Objects" and i == 0)
                       or (path[-2:] == (b"Connections", b"C") and i in (1, 2))
                       or (path[-2:] == (b"Documents", b"Document") and i == 0))
            if is_decl:
                declared_pos += 1 if (t == "L" and int(v) != 0) else 0
                continue
            if t == "L" and int(v) in uids:
                raise Reject("UNEXPECTED_UID_REFERENCE", f"{b'/'.join(path).decode()}[{i}]")
            if t == "l" and e.id != b"KeyTime" and np.isin(np.asarray(v), list(uids)).any():
                raise Reject("UNEXPECTED_UID_REFERENCE", f"{b'/'.join(path).decode()}[{i}] (array)")
        for c in e.elems:
            scan(c, path + (bytes(c.id),))
    for top in root.elems:
        scan(top, (bytes(top.id),))
    # --- header (every top-level section except Objects / Connections), with the declared exclusions
    hdr = _child(root, b"FBXHeaderExtension")
    ts = _children(hdr, b"CreationTimeStamp")
    if len(ts) != 1:
        raise Reject("TIMESTAMP_COUNT", str(len(ts)))
    si = _children(hdr, b"SceneInfo")
    p70 = _child(si[0], b"Properties70") if len(si) == 1 else None
    pp = [p for p in (p70.elems if p70 is not None else []) if p.id == b"P" and p.props and bytes(p.props[0]) == PATH_PROP]
    if len(pp) != 1 or len(pp[0].props) != 5:
        raise Reject("SOURCE_PATH_FIELD_COUNT", str(len(pp)))
    path_value = bytes(pp[0].props[4]).decode("utf-8", "backslashreplace")
    if not PATH_RE.match(path_value):
        raise Reject("SOURCE_PATH_UNEXPECTED_VALUE", path_value)
    ts_value = tuple((bytes(c.id).decode(), int(c.props[0])) for c in ts[0].elems)

    def rep(path, i, e):
        if i is None:
            return _SKIP if path[:2] == (b"FBXHeaderExtension", b"CreationTimeStamp") and len(path) == 2 else None
        if path[:3] == (b"FBXHeaderExtension", b"SceneInfo", b"Properties70") and e.id == b"P" and i == 4 \
                and bytes(e.props[0]) == PATH_PROP:
            return b"<EXCLUDED:SOURCE_PATH>"
        if path == (b"Documents", b"Document") and i == 0:
            return b"<EXCLUDED:DOCUMENT_UID>"
        return None
    header = {}
    for k, top in enumerate(root.elems):
        if top.id in (b"Objects", b"Connections"):
            header[k] = (bytes(top.id), "<keyed>")
            continue
        header[k] = _subtree(top, rep)

    def orep(path, i, e):
        return b"<UID>" if (i == 0 and len(path) == 1) else None
    objects = {ident[u]: _subtree(node[u], orep) for u in node}
    connections = tuple(sorted((t, ident[ch], "ROOT" if pa == 0 else ident[pa], pr if pr is not None else b"")
                               for t, ch, pa, pr in raw_conns))
    # --- accounting
    by_cls = Counter(cls.values())
    stacks = {u: bytes(node[u].props[1]).split(b"\x00")[0].decode("utf-8", "backslashreplace")
              for u, k in cls.items() if k == "AnimationStack"}
    layer_stack = {u: [p for t, _, p in out[u] if t == "OO"][0] for u, k in cls.items() if k == "AnimationLayer"}
    node_layer = {u: [p for t, _, p in out[u] if t == "OO"][0] for u, k in cls.items() if k == "AnimationCurveNode"}
    curves_by_stack = Counter()
    for u, k in cls.items():
        if k == "AnimationCurve":
            cn = [p for t, _, p in out[u] if t == "OP"][0]
            curves_by_stack[stacks[layer_stack[node_layer[cn]]]] += 1
    n_props = sum(len(o[1]) for o in objects.values())
    acc = {"objects": len(node), "objects_by_class": dict(sorted(by_cls.items())), "connections": len(raw_conns),
           "connections_to_root": sum(1 for c in raw_conns if c[2] == 0),
           "excluded": {"uid_reference_values": declared_pos, "object_uid_values": len(node),
                        "connection_uid_values": sum(1 + (c[2] != 0) for c in raw_conns), "document_uid_values": 1,
                        "creation_timestamp_elements": 1, "creation_timestamp_values": len(ts_value),
                        "source_path_values": 1},
           "curves": by_cls.get("AnimationCurve", 0), "curves_with_identity": by_cls.get("AnimationCurve", 0),
           "curves_by_stack": dict(sorted(curves_by_stack.items())), "identity_collisions": 0,
           "object_property_values_retained": n_props, "header_sections": len(header) - 2}
    return {"schema": SCHEMA, "objects": objects, "connections": connections, "header": header,
            "uid_of": {ident[u]: u for u in node}, "order": [ident[u] for u in (int(c.props[0]) for c in objs.elems)],
            "conn_order": [(t, ident[ch], "ROOT" if pa == 0 else ident[pa], pr or b"") for t, ch, pa, pr in raw_conns],
            "document_uid": doc_uid, "timestamp": ts_value, "source_path": path_value,
            "class_of": {ident[u]: cls[u] for u in node}, "accounting": acc}


def try_canonicalize(root):
    try:
        return canonicalize(root), None
    except Reject as r:
        return None, {"code": r.code, "detail": r.detail[:400]}


def canonical_digest(can):
    """Digest of the full canonical content (header, objects by identity, connection multiset)."""
    return _digest((tuple(sorted((k, v) for k, v in can["header"].items())),
                    tuple(sorted(can["objects"].items())), can["connections"]))


def compare(a, b, limit=20):
    """Full-record comparison of two canonical forms (every byte of every retained value)."""
    ha, hb = a["header"], b["header"]
    hdr = sorted({k for k in set(ha) | set(hb) if ha.get(k) != hb.get(k)})
    ka, kb = set(a["objects"]), set(b["objects"])
    only_a, only_b = sorted(ka - kb), sorted(kb - ka)
    differ = sorted(k for k in ka & kb if a["objects"][k] != b["objects"][k])
    ca, cb = Counter(a["connections"]), Counter(b["connections"])
    conn_a = sorted((ca - cb).elements())
    conn_b = sorted((cb - ca).elements())
    n = len(hdr) + len(only_a) + len(only_b) + len(differ) + len(conn_a) + len(conn_b)
    return {"equal": n == 0, "differences": n, "header_sections_differ": hdr,
            "objects_only_a": only_a[:limit], "objects_only_b": only_b[:limit], "objects_differ": differ[:limit],
            "n_objects_only_a": len(only_a), "n_objects_only_b": len(only_b), "n_objects_differ": len(differ),
            "connections_only_a": [list(map(_s, c)) for c in conn_a[:limit]],
            "connections_only_b": [list(map(_s, c)) for c in conn_b[:limit]],
            "n_connections_only_a": len(conn_a), "n_connections_only_b": len(conn_b),
            "_sets": {"only_a": set(only_a), "only_b": set(only_b), "differ": set(differ),
                      "conn_a": conn_a, "conn_b": conn_b}}


def _s(x):
    return x.decode("utf-8", "backslashreplace") if isinstance(x, bytes) else x


def public(cmp):
    return {k: v for k, v in cmp.items() if k != "_sets"}


def raw_variation(a, b, sha_a=None, sha_b=None):
    """Exporter-volatile differences between two canonicalised files of the same case (raw, before exclusion)."""
    common = set(a["uid_of"]) & set(b["uid_of"])
    changed = sum(1 for k in common if a["uid_of"][k] != b["uid_of"][k])
    oa, ob = a["order"], b["order"]
    pos = sum(1 for x, y in zip(oa, ob) if x != y) + abs(len(oa) - len(ob))
    ca, cb = a["conn_order"], b["conn_order"]
    cpos = sum(1 for x, y in zip(ca, cb) if x != y) + abs(len(ca) - len(cb))
    return {"identities_compared": len(common), "uid_changed": changed,
            "document_uid_changed": a["document_uid"] != b["document_uid"],
            "object_order_positions_differ": pos, "object_order_equal": oa == ob,
            "connection_order_positions_differ": cpos, "connection_order_equal": ca == cb,
            "timestamp_differs": a["timestamp"] != b["timestamp"], "source_path_differs": a["source_path"] != b["source_path"],
            "source_paths": [a["source_path"], b["source_path"]],
            "bytes_equal": (sha_a == sha_b) if sha_a and sha_b else None}


def record_summary(can):
    """JSON-able per-file record: per-identity subtree digests, connection multiset digest, header digests."""
    return {"schema": SCHEMA + ":record", "canonical_digest": canonical_digest(can),
            "objects": {k: _digest(v) for k, v in sorted(can["objects"].items())},
            "connections_digest": _digest(can["connections"]), "connections": len(can["connections"]),
            "header": {str(k): _digest(v) for k, v in sorted(can["header"].items())},
            "accounting": can["accounting"]}


# --- UID normalisation of W110 adapter records ---------------------------------------------------------------------
def normalize_plan(plan, uid_of):
    """Replace every UID in a W110 adapter plan by the semantic identity of the same FBX element; unknown UIDs become
    '<UNMAPPED:uid>' (counted).  Lists of groups are ordered by (item identity, group)."""
    inv = {u: k for k, u in uid_of.items()}
    unmapped = []

    def m(u):
        u = int(u)
        if u not in inv:
            unmapped.append(u)
            return f"<UNMAPPED:{u}>"
        return inv[u]
    p = copy.deepcopy(plan)
    p["items"] = {m(k): {**v, "model": m(v["model"])} for k, v in plan.get("items", {}).items()}
    groups = []
    for g in plan.get("groups", []):
        g2 = dict(g)
        g2["model"] = m(g["model"])
        g2["fbx_curves"] = sorted(m(u) for u in g["fbx_curves"])
        groups.append(g2)
    p["groups"] = sorted(groups, key=lambda g: (g["model"], g["group"]))
    p["curve_status"] = {m(k): v for k, v in sorted(plan.get("curve_status", {}).items())}
    p["_unmapped_uids"] = len(unmapped)
    return p


def normalize_directives(directives, uid_of):
    inv = {u: k for k, u in uid_of.items()}
    return {inv.get(int(u), f"<UNMAPPED:{u}>"): list(v) for u, v in directives.items()}


def plan_digest(nplan):
    return hashlib.sha256(json.dumps(nplan, sort_keys=True, default=str).encode()).hexdigest()


# --- minimal binary FBX writer (scratch negative-control copies only) -----------------------------------------------
FBX_MAGIC = b"Kaydara FBX Binary  \x00\x1a\x00"
_ALWAYS_SENTINEL = {b"AnimationStack", b"AnimationLayer"}
_FOOTER = b"\xfa\xbc\xab\x09\xd0\xc8\xd4\x66\xb1\x76\xfb\x83\x1c\xf7\x26\x7e" + b"\x00"*4


def _enc(t, v):
    if t in _ARR_DT:
        a = np.asarray(v).astype(_ARR_DT[t], copy=False)
        b = a.tobytes()
        return struct.pack("<3I", a.size, 0, len(b)) + b
    if t in ("S", "R"):
        b = bytes(v)
        return struct.pack("<I", len(b)) + b
    return _val(t, v)


def write_fbx(root, version=7400):
    """Serialise a node tree (uncompressed arrays) so that ``fbx_read`` returns the same tree."""
    wide = version >= 7500
    hdr = struct.Struct("<3Q" if wide else "<3I")
    sentinel = b"\x00"*(hdr.size + 1)

    def node(e, start):
        nid = bytes(e.id)
        props = b"".join(t.encode() + _enc(t, v) for t, v in zip(_t(e), e.props))
        off = start + hdr.size + 1 + len(nid) + len(props)
        body = bytearray()
        for c in e.elems:
            cb = node(c, off)
            body += cb
            off += len(cb)
        if e.elems or not e.props or nid in _ALWAYS_SENTINEL:
            body += sentinel
            off += len(sentinel)
        return hdr.pack(off, len(e.props), len(props)) + bytes([len(nid)]) + nid + props + bytes(body)
    out = bytearray(FBX_MAGIC + struct.pack("<I", version))
    for top in root.elems:
        out += node(top, len(out))
    out += sentinel + _FOOTER
    return bytes(out)


# --- negative / invariance controls on scratch copies ---------------------------------------------------------------
class M:
    """Mutable node with the (id, props, props_type, elems) interface."""
    __slots__ = ("id", "props", "props_type", "elems")

    def __init__(self, nid, props, props_type, elems):
        self.id, self.props, self.props_type, self.elems = nid, props, props_type, elems


def mcopy(e):
    return M(bytes(e.id), [np.array(p, copy=True) if isinstance(p, np.ndarray) else p for p in e.props],
             bytes(e.props_type), [mcopy(c) for c in e.elems])


CONTROLS = {
    "NC0_rewrite_identity": ("EQUAL", "re-serialise the unmodified tree"),
    "INV1_uid_relabel": ("EQUAL", "bijective relabel of every element UID and the Document UID at every declared UID "
                                  "position"),
    "INV2_order_permutation": ("EQUAL", "reverse the order of Objects elements and of Connections"),
    "INV3_volatile_fields": ("EQUAL", "change every CreationTimeStamp value and the source .blend path string"),
    "NC1_curve_flag": ("DETECTED_LOCAL", "one selected-take CONSTANT curve's KeyAttrFlags 0x6002 -> 0x6104 (LINEAR "
                                         "curve 0x6104 -> 0x6002 if the take has no CONSTANT curve)"),
    "NC2_key_value": ("DETECTED_LOCAL", "one selected-take curve: middle KeyValueFloat += 1e-3"),
    "NC3_key_time": ("DETECTED_LOCAL", "one selected-take curve: middle KeyTime += 1 FBX tick"),
    "NC4_take_connection": ("DETECTED_LOCAL_OR_REJECTED:NON_INJECTIVE_IDENTITY",
                            "the OO connection of one selected-take curve node re-pointed from the selected layer to "
                            "another take's layer"),
    "NC5_parent": ("DETECTED_LOCAL", "one LimbNode's OO parent re-pointed to its grandparent"),
    "NC6_duplicate_name": ("REJECTED:DUPLICATE_NAMED_IDENTITY", "one LimbNode Model renamed to another LimbNode's "
                                                                "name"),
    "NC7_unresolved_connection": ("REJECTED:UNRESOLVED_CONNECTION", "one LimbNode parent connection re-pointed to a "
                                                                    "non-existent UID"),
    "NC8_unexpected_uid_reference": ("REJECTED:UNEXPECTED_UID_REFERENCE", "Documents/Document/RootNode 0 -> the UID of "
                                                                         "the armature Null Model"),
    "NC9_non_target_curve": ("DETECTED_LOCAL", "one curve of a NON-selected take: middle KeyValueFloat += 1e-3"),
    "NC10_take_swap": ("DETECTED_LOCAL", "swap the take (layer) membership of the selected-take curve node holding the "
                                         "NC1 target curve and the same item/property curve node of another take"),
}


def _objs(root):
    return _child(root, b"Objects")


def _conns(root):
    return _child(root, b"Connections")


def _targets(can, stack):
    """Deterministic mutation targets from the canonical form of the unmodified file."""
    cls = can["class_of"]
    lay = [k for k, c in cls.items() if c == "AnimationLayer" and k.startswith("AnimationLayer[]b'" + stack + "\\x00")]
    if len(lay) != 1:
        raise ValueError(f"selected layer not unique: {lay}")
    sel_layer = lay[0]
    sel_curves = sorted(k for k, c in cls.items() if c == "AnimationCurve" and sel_layer in k)
    other_curves = sorted(k for k, c in cls.items() if c == "AnimationCurve" and sel_layer not in k)
    other_layers = sorted(k for k, c in cls.items() if c == "AnimationLayer" and k != sel_layer)
    parent = {}
    for t, ch, pa, _ in can["connections"]:
        if t == "OO" and cls.get(ch) == "Model":
            parent[ch] = pa
    limbs = sorted(k for k, c in cls.items() if c == "Model" and k.startswith("Model[LimbNode]"))
    nulls = sorted(k for k, c in cls.items() if c == "Model" and k.startswith("Model[Null]"))
    gp = [k for k in limbs if parent.get(k) in limbs and parent.get(parent[k]) in limbs]
    return {"selected_layer": sel_layer, "selected_curves": sel_curves, "other_curves": other_curves,
            "other_layers": other_layers, "limbs": limbs, "nulls": nulls, "grandchild": gp[0] if gp else None,
            "parent": parent}


def _by_uid(root):
    return {int(e.props[0]): e for e in _objs(root).elems}


def _flags_of(index, uid):
    c = index[uid]
    return c, _child(c, b"KeyAttrFlags")


def mutate(root0, can, stack, control):
    """Return (mutated mutable tree, record of what was changed and which canonical records may change)."""
    r = mcopy(root0)
    uid = can["uid_of"]
    tg = _targets(can, stack)
    rec = {"control": control, "expect": CONTROLS[control][0], "description": CONTROLS[control][1]}
    allowed_ids, allowed_conn = set(), []
    if control == "NC0_rewrite_identity":
        pass
    elif control == "INV1_uid_relabel":
        uids = sorted(uid.values())
        new = {u: 900_000_000_000_001 + 2*i for i, u in enumerate(reversed(uids))}   # odd: never an FBX tick
        for e in _objs(r).elems:
            e.props[0] = new[int(e.props[0])]
        for c in _conns(r).elems:
            c.props[1] = new[int(c.props[1])]
            c.props[2] = new[int(c.props[2])] if int(c.props[2]) != 0 else 0
        d = _child(_child(r, b"Documents"), b"Document")
        d.props[0] = int(d.props[0]) + 1
        rec["relabelled"] = len(new)
    elif control == "INV2_order_permutation":
        _objs(r).elems.reverse()
        _conns(r).elems.reverse()
    elif control == "INV3_volatile_fields":
        ts = _child(_child(r, b"FBXHeaderExtension"), b"CreationTimeStamp")
        for c in ts.elems:
            c.props[0] = int(c.props[0]) + 1
        si = _child(_child(r, b"FBXHeaderExtension"), b"SceneInfo")
        for p in _child(si, b"Properties70").elems:
            if p.props and bytes(p.props[0]) == PATH_PROP:
                p.props[4] = b"D:\\w111_invariance\\other_location\\renamed_source.blend"
    elif control in ("NC1_curve_flag", "NC2_key_value", "NC3_key_time", "NC9_non_target_curve"):
        pool = tg["other_curves"] if control == "NC9_non_target_curve" else tg["selected_curves"]
        if not pool:
            raise ValueError(f"{control}: no candidate curve")
        target = pool[0]
        if control == "NC1_curve_flag":
            i0 = _by_uid(root0)
            const = [k for k in pool if int(np.asarray(_flags_of(i0, uid[k])[1].props[0])[0]) == 0x6002]
            target = const[0] if const else pool[0]
        c, kf = _flags_of(_by_uid(r), uid[target])
        if control == "NC1_curve_flag":
            old = int(np.asarray(kf.props[0])[0])
            kf.props[0] = np.full_like(np.asarray(kf.props[0]), 0x6104 if old == 0x6002 else 0x6002)
            rec["change"] = [hex(old), hex(int(kf.props[0][0]))]
        elif control in ("NC2_key_value", "NC9_non_target_curve"):
            kv = _child(c, b"KeyValueFloat")
            a = np.asarray(kv.props[0]).astype(np.float32).copy()
            i = len(a)//2
            old = float(a[i])
            a[i] = np.float32(old + 1e-3)
            kv.props[0] = a
            rec["change"] = [i, old, float(a[i])]
        else:
            kt = _child(c, b"KeyTime")
            a = np.asarray(kt.props[0]).astype(np.int64).copy()
            i = len(a)//2
            if not (len(a) >= 3 and a[i] + 1 < a[i+1]):
                raise ValueError("NC3: cannot shift the middle key by one tick")
            a[i] += 1
            kt.props[0] = a
            rec["change"] = [i, int(a[i]-1), int(a[i])]
        rec["target"] = target
        allowed_ids.add(target)
    elif control == "NC4_take_connection":
        curve = tg["selected_curves"][0]
        cn = [pa for t, ch, pa, _ in can["connections"] if ch == curve and t == "OP"][0]
        if not tg["other_layers"]:
            raise ValueError("NC4: no other take layer")
        new_layer = tg["other_layers"][0]
        for c in _conns(r).elems:
            if int(c.props[1]) == uid[cn] and int(c.props[2]) == uid[tg["selected_layer"]]:
                c.props[2] = uid[new_layer]
        rec["target"], rec["new_layer"] = cn, new_layer
        allowed_ids |= {cn} | {ch for t, ch, pa, _ in can["connections"] if pa == cn}
    elif control == "NC10_take_swap":
        i0 = _by_uid(root0)
        const = [k for k in tg["selected_curves"] if int(np.asarray(_flags_of(i0, uid[k])[1].props[0])[0]) == 0x6002]
        curve = (const or tg["selected_curves"])[0]
        a = [pa for t, ch, pa, _ in can["connections"] if ch == curve and t == "OP"][0]
        if not tg["other_layers"]:
            raise ValueError("NC10: no other take layer")
        b = a.replace(tg["selected_layer"], tg["other_layers"][0])
        if b not in uid:
            raise ValueError("NC10: no counterpart curve node in the other take")
        la, lb = uid[tg["selected_layer"]], uid[tg["other_layers"][0]]
        for c in _conns(r).elems:
            if bytes(c.props[0]) == b"OO" and int(c.props[1]) == uid[a] and int(c.props[2]) == la:
                c.props[2] = lb
            elif bytes(c.props[0]) == b"OO" and int(c.props[1]) == uid[b] and int(c.props[2]) == lb:
                c.props[2] = la
        rec["target"], rec["counterpart"] = a, b
        allowed_ids |= {a, b} | {ch for t, ch, pa, _ in can["connections"] if pa in (a, b)}
    elif control in ("NC5_parent", "NC7_unresolved_connection"):
        ch = tg["grandchild"]
        if ch is None:
            raise ValueError("no LimbNode with a LimbNode grandparent")
        pa = tg["parent"][ch]
        new = uid[tg["parent"][pa]] if control == "NC5_parent" else max(uid.values()) + 123457
        for c in _conns(r).elems:
            if bytes(c.props[0]) == b"OO" and int(c.props[1]) == uid[ch] and int(c.props[2]) == uid[pa]:
                c.props[2] = new
        rec["target"], rec["old_parent"], rec["new_parent"] = ch, pa, (tg["parent"][pa] if control == "NC5_parent"
                                                                       else f"<nonexistent {new}>")
        allowed_conn = [("OO", ch, pa), ("OO", ch, tg["parent"][pa])]
    elif control == "NC6_duplicate_name":
        a, b = tg["limbs"][0], tg["limbs"][1]
        _by_uid(r)[uid[b]].props[1] = bytes(_by_uid(root0)[uid[a]].props[1])
        rec["target"], rec["renamed_to"] = b, a
    elif control == "NC8_unexpected_uid_reference":
        rn = _child(_child(_child(r, b"Documents"), b"Document"), b"RootNode")
        rn.props[0] = uid[tg["nulls"][0]]
        rec["target"] = "Documents/Document/RootNode"
    else:
        raise ValueError(control)
    rec["allowed_ids"], rec["allowed_connections"] = sorted(allowed_ids), allowed_conn
    return r, rec


def judge_control(rec, base_can, mut_can, rejection):
    """Is the control discriminating as declared?  Localisation: every differing record must be one the mutation
    could legitimately change; nothing else (header, other curves / takes, connections) may differ."""
    exp = rec["expect"]
    out = {"expect": exp, "rejection": rejection}
    if rejection is not None:
        out["outcome"] = f"REJECTED:{rejection['code']}"
        out["ok"] = exp.startswith("REJECTED") and exp.endswith(rejection["code"]) or \
            (exp.startswith("DETECTED_LOCAL_OR_REJECTED") and exp.endswith(rejection["code"]))
        return out
    cmp = compare(base_can, mut_can)
    out["comparison"] = public(cmp)
    if exp == "EQUAL":
        out["outcome"] = "EQUAL" if cmp["equal"] else "DIFFERENT"
        out["ok"] = cmp["equal"]
        return out
    if cmp["equal"]:
        out["outcome"], out["ok"] = "MASKED", False
        return out
    s = cmp["_sets"]
    allowed = set(rec["allowed_ids"])
    touched = s["only_a"] | s["only_b"] | s["differ"]
    if rec["control"] == "NC4_take_connection":
        old = allowed
        ok_ids = all(k in old or any(a in k for a in old) or "AnimationCurve" in k for k in touched) and \
            s["only_a"] == old and not s["differ"]
        ok_conn = all(c[1] in touched or c[2] in touched for c in s["conn_a"] + s["conn_b"])
        loc = ok_ids and ok_conn and not cmp["header_sections_differ"] and len(s["only_b"]) == len(old)
    elif rec["control"] == "NC5_parent":
        want_a = {(t, ch, pa) for t, ch, pa in [rec["allowed_connections"][0]]}
        want_b = {(t, ch, pa) for t, ch, pa in [rec["allowed_connections"][1]]}
        loc = (not touched and not cmp["header_sections_differ"]
               and {tuple(c[:3]) for c in s["conn_a"]} == want_a and {tuple(c[:3]) for c in s["conn_b"]} == want_b
               and len(s["conn_a"]) == 1 and len(s["conn_b"]) == 1)
    elif rec["control"] == "NC10_take_swap":
        loc = (bool(touched) and touched <= allowed and not s["only_a"] and not s["only_b"] and not s["conn_a"]
               and not s["conn_b"] and not cmp["header_sections_differ"])
    else:
        loc = (touched == allowed and s["differ"] == allowed and not s["conn_a"] and not s["conn_b"]
               and not cmp["header_sections_differ"])
    out["outcome"] = "DETECTED_LOCAL" if loc else "DETECTED_NOT_LOCAL"
    out["ok"] = bool(loc and exp.startswith("DETECTED"))
    return out
