#!/usr/bin/env python3
"""W110 consumer-only FBX interpolation recovery adapter (scratch diagnostic; evidence only, not a product feature).

Inputs are restricted by contract to

  * the bytes of one FBX file (read from its path),
  * the selected take = AnimationStack name, and
  * the freshly imported Blender scene (``bpy.data`` after an unchanged default ``import_scene.fbx``).

It never reads a .blend, a source Action, ground truth, a key census, a declaration or any sidecar listing desired
curve semantics; it imports nothing from this repository.  After the stock importer has run, it may change only the
``interpolation`` field of keyframes of imported F-curves that it uniquely matched to FBX AnimationCurves of the
selected take, as directed by those curves' ``KeyAttrFlags``.  Key times, values, handles, handle types, curves,
Actions, slots, rig, objects and timebase are never written.

Mapping rule (frozen by the W110 declaration):

  FBX side (connection graph only):   AnimationCurve --OP 'd|X|Y|Z'--> AnimationCurveNode --OP 'Lcl Translation |
  Lcl Rotation | Lcl Scaling'--> Model;  AnimationCurveNode --OO--> the single AnimationLayer of the selected
  AnimationStack.
  Blender side (stock importer naming, Blender 5.1 / io_scene_fbx 5.15):  a ``Null`` Model is the object of the same
  name; a ``LimbNode`` Model is the pose bone of the same name in exactly one armature, whose bone parent must equal
  the FBX parent Model (or none when the FBX parent is the armature's Null).  The Action is ``"<id name>|<stack>"``
  (single-layer stack whose layer name equals the stack name), with exactly one slot.
  Correspondence unit: (item, transform property).  The importer bakes all of an item's Lcl curves through one matrix
  per key time and decomposes it into location / rotation / scale F-curves (code inspection), so no per-axis FBX-curve
  -> F-curve bijection exists (Euler X/Y/Z -> quaternion W/X/Y/Z, rest-frame axis mixing).  With zero rotation /
  scaling offsets and pivots and identity geometric transforms on the Model (checked from the FBX), location depends
  only on Lcl Translation, rotation only on Lcl Rotation and scale only on Lcl Scaling; the directive of each group is
  then the consensus of its FBX curves.  Otherwise the item is ``COUPLED`` and every group needs the consensus of all of
  the item's curves.  Disagreement, unsupported flags (CONSTANT-next, CUBIC, unknown, per-key mixtures), duplicate
  channels, missing / ambiguous items, Actions, slots or F-curves are counted as unresolved and never changed.
"""
from __future__ import annotations

import hashlib
import json

SCHEMA = "w110_fbx_consumer_adapter_v1"

FBX_CONSTANT = 0x2
FBX_LINEAR = 0x4
FBX_CUBIC = 0x8
FBX_INTERP_MASK = 0xE
FBX_CONSTANT_NEXT = 0x100          # eConstantNext under CONSTANT (eTangentAuto otherwise)
SUPPORTED = ("CONSTANT", "LINEAR")  # FBX classes this adapter can express as Blender keyframe interpolation
TRANSFORM_PROPS = {"Lcl Translation": "location", "Lcl Rotation": "rotation", "Lcl Scaling": "scale"}
CHANNELS = {"d|X": 0, "d|Y": 1, "d|Z": 2}
# Model properties that make Lcl Translation / Rotation / Scaling interact in the importer's per-key matrix
SEPARABILITY_DEFAULTS = {"RotationOffset": (0.0, 0.0, 0.0), "RotationPivot": (0.0, 0.0, 0.0),
                         "ScalingOffset": (0.0, 0.0, 0.0), "ScalingPivot": (0.0, 0.0, 0.0),
                         "GeometricTranslation": (0.0, 0.0, 0.0), "GeometricRotation": (0.0, 0.0, 0.0),
                         "GeometricScaling": (1.0, 1.0, 1.0)}


# --- FBX side (pure; works on io_scene_fbx.parse_fbx trees and on any (id, props, props_type, elems) tree) --------
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
            out[_str(e.props[0])] = [float(x) for x in e.props[4:] if isinstance(x, (int, float))]
    return out


def separable(model_elem):
    """True iff every offset / pivot / geometric property of the Model is absent or at its default."""
    p = _p70(model_elem)
    bad = {k: p[k] for k, d in SEPARABILITY_DEFAULTS.items() if k in p and tuple(p[k][:3]) != d}
    return not bad, bad


def fbx_curve_map(root, stack_name):
    """Per-curve FBX correspondence records of the selected AnimationStack (connection graph only)."""
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
    out = {"schema": SCHEMA + ":fbx_map", "stack": stack_name, "status": "OK", "curves": [], "models": {}}
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
            continue                                   # curve of another take
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
                        if len(nodes[m].props) > 2 else "", "fbx_property": mprop,
                        "group": TRANSFORM_PROPS.get(mprop)})
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
                if key in seen:                        # the importer would keep only the first: many-to-one
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
                                    "parent_count": len(mp), "separable": sep, "non_default": bad}
        out["curves"].append(rec)
    out["models"] = {str(k): v for k, v in sorted(out["models"].items())}
    return out


def fbx_map_digest(fmap, with_flags=True):
    """Digest of the correspondence records (optionally without the flag fields)."""
    drop = () if with_flags else ("flags_hex", "flag_classes", "flag_class")
    recs = [{k: v for k, v in r.items() if k not in drop} for r in fmap["curves"]]
    blob = json.dumps({"stack": fmap["stack"], "status": fmap["status"], "curves": recs, "models": fmap["models"]},
                      sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


# --- Blender side (pure planning over a scene description) ------------------------------------------------------
def escape_identifier(s):
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _rot_path(mode):
    if mode == "QUATERNION":
        return "rotation_quaternion", 4
    if mode == "AXIS_ANGLE":
        return "rotation_axis_angle", 4
    return "rotation_euler", 3


def plan(fmap, scene):
    """Resolve the FBX correspondence against an imported-scene description.

    ``scene``: {"objects": {name: {"type", "rotation_mode", "bones": {bone: {"parent", "rotation_mode"}} | None}},
                "actions": {name: {"slots": [identifier], "fcurves": {slot: [[data_path, index], ...]}}}}

    Returns per-group directives (CONSTANT / LINEAR / UNRESOLVED / NO_FBX_CURVES), the F-curves they cover and the
    complete accounting of FBX curves and imported F-curves."""
    res = {"schema": SCHEMA + ":plan", "stack": fmap["stack"], "status": fmap["status"], "items": {}, "groups": [],
           "curve_status": {}, "unresolved": []}
    if fmap["status"] != "OK":
        res["accounting"] = {"fbx_curves": len(fmap["curves"]), "fbx_curves_mapped": 0}
        return res
    curves_by_model = {}
    for r in fmap["curves"]:
        if "model" in r:
            curves_by_model.setdefault(r["model"], []).append(r)
    objects = scene["objects"]
    claimed = {}
    for m, recs in sorted(curves_by_model.items()):
        info = fmap["models"][str(m)]
        it = {"model": m, "name": info["name"], "type": info["type"], "status": "OK"}
        if info["type"] == "Null":
            if info["name"] in objects:
                it.update({"id_name": info["name"], "prefix": "", "rotation_mode": objects[info["name"]]["rotation_mode"]})
            else:
                it["status"] = "NO_OBJECT"
        else:   # LimbNode
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
        res["items"][str(m)] = it
        ok_recs = [r for r in recs if r["status"] == "OK"]
        groups = {}
        for g in ("location", "rotation", "scale"):
            groups[g] = [r for r in recs if r.get("group") == g]
        coupled = not info["separable"]
        all_classes = sorted({r["flag_class"] for r in recs})
        for g, grecs in groups.items():
            grp = {"model": m, "item": info["name"], "group": g, "fbx_curves": [r["uid"] for r in grecs],
                   "coupled": coupled, "fcurves": []}
            if it["status"] == "OK":
                if g == "rotation":
                    prop, n = _rot_path(it["rotation_mode"])
                else:
                    prop, n = g, 3
                fl = scene["actions"][it["action"]]["fcurves"][it["slot"]]
                for i in range(n):
                    dp = it["prefix"] + prop
                    hits = sum(1 for p, j in fl if p == dp and j == i)
                    grp["fcurves"].append([dp, i, hits])
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
            for r in grecs:
                res["curve_status"][str(r["uid"])] = directive if directive in SUPPORTED else "UNRESOLVED"
            if directive == "UNRESOLVED":
                res["unresolved"].append({"item": info["name"], "group": g, "reasons": reasons})
    # an imported F-curve claimed by two groups is ambiguous: unresolve both
    for key, gi in claimed.items():
        if len(gi) > 1:
            for k in gi:
                if res["groups"][k]["directive"] in SUPPORTED:
                    res["groups"][k]["directive"] = "UNRESOLVED"
                    res["groups"][k]["reasons"].append("FCURVE_CLAIMED_TWICE")
    for r in fmap["curves"]:
        res["curve_status"].setdefault(str(r["uid"]), "UNRESOLVED" if r["status"] != "OK" else "NO_ITEM")
    acts = sorted({g["action"] for g in res["groups"] if "action" in g})
    imported = {(a, s, p, i) for a in acts for s, fl in scene["actions"][a]["fcurves"].items() for p, i in fl}
    covered = set(claimed)
    n_dir = {}
    for g in res["groups"]:
        n_dir[g["directive"]] = n_dir.get(g["directive"], 0) + 1
    cs = list(res["curve_status"].values())
    res["accounting"] = {
        "fbx_curves": len(fmap["curves"]),
        "fbx_curves_mapped": sum(1 for v in cs if v in SUPPORTED),
        "fbx_curves_constant": cs.count("CONSTANT"), "fbx_curves_linear": cs.count("LINEAR"),
        "fbx_curves_unresolved": sum(1 for v in cs if v not in SUPPORTED),
        "fbx_curve_status_counts": {k: cs.count(k) for k in sorted(set(cs))},
        "groups": len(res["groups"]), "group_directives": n_dir,
        "items": len(res["items"]), "items_unresolved": sum(1 for i in res["items"].values() if i["status"] != "OK"),
        "coupled_items": sum(1 for m in fmap["models"].values() if not m["separable"]),
        "target_actions": acts,
        "imported_fcurves_in_target_actions": len(imported), "imported_fcurves_mapped": len(imported & covered),
        "imported_fcurves_unmapped": sorted(f"{a}:{p}[{i}]" for a, s, p, i in imported - covered),
        "fcurves_claimed_twice": sum(1 for v in claimed.values() if len(v) > 1),
        "axis_level_one_to_one_curves": 0,
        "axis_level_note": "the importer bakes each item's Lcl curves through one matrix per key time; no FBX curve "
                           "maps to exactly one imported F-curve (Euler XYZ -> quaternion WXYZ; rest-frame axis mixing)"}
    return res


# --- Blender runtime (bpy) -----------------------------------------------------------------------------------------
def describe_scene():
    import bpy
    from bpy_extras import anim_utils
    objs = {}
    esc_mismatch = []
    for o in bpy.data.objects:
        d = {"type": o.type, "rotation_mode": o.rotation_mode, "bones": None}
        if o.type == "ARMATURE" and o.pose is not None:
            d["bones"] = {pb.name: {"parent": pb.parent.name if pb.parent else None, "rotation_mode": pb.rotation_mode}
                          for pb in o.pose.bones}
            esc_mismatch += [n for n in d["bones"] if bpy.utils.escape_identifier(n) != escape_identifier(n)]
        objs[o.name] = d
    acts = {}
    for a in bpy.data.actions:
        fc = {}
        for s in a.slots:
            cb = anim_utils.action_get_channelbag_for_slot(a, s)
            fc[s.identifier] = [[f.data_path, f.array_index] for f in cb.fcurves] if cb is not None else []
        acts[a.name] = {"slots": [s.identifier for s in a.slots], "fcurves": fc}
    return {"objects": objs, "actions": acts, "escape_mismatch": esc_mismatch}


def apply(pl):
    """Set only keyframe ``interpolation`` on the F-curves of CONSTANT / LINEAR groups (the F-curve is not re-updated)."""
    import bpy
    import numpy as np
    from bpy_extras import anim_utils
    vals = {i.identifier: i.value for i in bpy.types.Keyframe.bl_rna.properties["interpolation"].enum_items}
    out = {"fcurves_changed": 0, "fcurves_already_correct": 0, "keys_changed": 0, "per_group": []}
    for g in pl["groups"]:
        if g["directive"] not in SUPPORTED:
            continue
        a = bpy.data.actions[g["action"]]
        s = [x for x in a.slots if x.identifier == g["slot"]][0]
        cb = anim_utils.action_get_channelbag_for_slot(a, s)
        want = vals[g["directive"]]
        rec = {"item": g["item"], "group": g["group"], "directive": g["directive"], "changed": 0, "already": 0}
        for dp, i, _ in g["fcurves"]:
            f = cb.fcurves.find(dp, index=i)
            n = len(f.keyframe_points)
            cur = np.empty(n, dtype=np.int32)
            f.keyframe_points.foreach_get("interpolation", cur)
            if bool((cur == want).all()):
                rec["already"] += 1
                out["fcurves_already_correct"] += 1
                continue
            f.keyframe_points.foreach_set("interpolation", np.full(n, want, dtype=np.int32))
            rec["changed"] += 1
            out["fcurves_changed"] += 1
            out["keys_changed"] += int((cur != want).sum())
        out["per_group"].append(rec)
    return out


def recover(fbx_path, stack_name):
    """The adapter: FBX bytes + selected take name + the current (freshly imported) Blender scene -> report."""
    from pathlib import Path
    from io_scene_fbx import parse_fbx
    sha_before = hashlib.sha256(Path(fbx_path).read_bytes()).hexdigest()
    root, version = parse_fbx.parse(str(fbx_path))
    sha_after = hashlib.sha256(Path(fbx_path).read_bytes()).hexdigest()
    fmap = fbx_curve_map(root, stack_name)
    scene = describe_scene()
    pl = plan(fmap, scene)
    rep = {"schema": SCHEMA + ":report", "fbx_sha256": sha_before, "fbx_unchanged_during_read": sha_before == sha_after,
           "fbx_version": version, "stack": stack_name, "fbx_map_digest": fbx_map_digest(fmap),
           "fbx_map_digest_without_flags": fbx_map_digest(fmap, with_flags=False),
           "escape_mismatch": scene["escape_mismatch"], "plan": pl}
    rep["applied"] = apply(pl) if pl["status"] == "OK" and not scene["escape_mismatch"] else None
    return rep
