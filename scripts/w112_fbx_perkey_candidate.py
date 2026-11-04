#!/usr/bin/env python3
"""W112 source-free per-key FBX interpolation candidate (scratch diagnostic; evidence only, not a product feature).

Tested CONDITIONALLY: only on cases whose privileged import-level feasibility bound passed, only on their metadata FBX.

Inputs are restricted exactly as for the W110 adapter: the bytes of one FBX file, the selected take (AnimationStack)
name and the freshly imported Blender scene.  It never reads a .blend, an Action of origin, ground truth, a key census,
a declaration, a sidecar or any desired-motion description.  The only repository module it imports is the unchanged
W110 consumer adapter, whose FBX-side map, scene description and (item, transform property) correspondence it reuses
verbatim.  It writes nothing but keyframe ``interpolation``.

Rule (frozen by the W112 declaration):
  * A group the W110 plan resolves to CONSTANT or LINEAR is applied exactly as W110 applies it (every key).
  * A non-coupled group the W110 plan leaves unresolved ONLY because a curve's flags vary across keys
    (``UNSUPPORTED_FLAG_MIXED_OR_MISSING``) is resolved PER KEY when all of the following hold; otherwise it stays
    unresolved and is never changed:
      - every FBX curve of the group has the identical KeyTime array and RefCount runs summing to its key count;
      - at every key the decoded flag class is identical across the group's curves (no per-axis semantics: the
        importer bakes the property through one matrix per key, so an FBX axis has no imported F-curve of its own)
        and is CONSTANT or LINEAR;
      - every imported F-curve of the group has exactly as many keys as the FBX curves, and its key frames are spaced
        exactly as the FBX ticks converted with the imported scene frame rate (|difference| <= 1e-4 frame).
    Key j of every imported F-curve of the group then receives the class of FBX key j.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

import w110_fbx_consumer_adapter as W

SCHEMA = "w112_fbx_perkey_candidate_v1"
KTIME_PER_SECOND = 46186158000
TIME_TOL_FRAMES = 1e-4
FLAG_REASON = "UNSUPPORTED_FLAG_MIXED_OR_MISSING"


def curve_keys(root, uids):
    """{uid: (KeyTime int64 array, per-key flag class list | None, refcount_ok)} for the given curve UIDs."""
    objs = W._child(root, b"Objects")
    out = {}
    for c in objs.elems:
        if c.id != b"AnimationCurve" or int(c.props[0]) not in uids:
            continue
        kt, kf, kr = W._child(c, b"KeyTime"), W._child(c, b"KeyAttrFlags"), W._child(c, b"KeyAttrRefCount")
        t = np.asarray(kt.props[0], dtype=np.int64) if kt is not None else np.zeros(0, dtype=np.int64)
        flags = [int(x) for x in kf.props[0]] if kf is not None else []
        refs = [int(x) for x in kr.props[0]] if kr is not None else []
        ok = len(flags) == len(refs) and sum(refs) == len(t) and all(r > 0 for r in refs)
        per = []
        if ok:
            for f, r in zip(flags, refs):
                per += [W.flag_class(f)]*r
        out[int(c.props[0])] = (t, per if ok else None, ok)
    return out


def perkey_plan(fmap, keys, pl, scene_frames, fps):
    """Per-key directives for groups left unresolved only by per-key flag variation.  ``scene_frames``:
    {(action, slot, data_path, index): [key frames]} of the imported F-curves."""
    out = []
    for gi, g in enumerate(pl["groups"]):
        if g["directive"] != "UNRESOLVED" or g["reasons"] != [FLAG_REASON] or g["coupled"] or "action" not in g:
            continue
        rec = {"group_index": gi, "item": g["item"], "group": g["group"], "status": "OK"}
        ks = [keys.get(int(u)) for u in g["fbx_curves"]]
        if not ks or any(k is None or not k[2] for k in ks):
            rec["status"] = "BAD_REFCOUNT"
        elif any(not np.array_equal(ks[0][0], k[0]) for k in ks):
            rec["status"] = "KEYTIME_MISMATCH_ACROSS_AXES"
        else:
            n = len(ks[0][0])
            per = []
            for j in range(n):
                cl = sorted({k[1][j] for k in ks})
                per.append(cl[0] if len(cl) == 1 else "AXIS_DISAGREEMENT")
            bad = sorted({c for c in per if c not in W.SUPPORTED})
            if bad:
                rec["status"] = "UNSUPPORTED_PER_KEY_" + "/".join(bad)
            else:
                rel = (ks[0][0] - ks[0][0][0])*float(fps)/KTIME_PER_SECOND
                for dp, i, _ in g["fcurves"]:
                    fr = scene_frames.get((g["action"], g["slot"], dp, i))
                    if fr is None or len(fr) != n:
                        rec["status"] = "IMPORTED_KEY_COUNT_MISMATCH"
                        break
                    if float(np.abs((np.asarray(fr) - fr[0]) - rel).max()) > TIME_TOL_FRAMES:
                        rec["status"] = "IMPORTED_KEY_TIME_MISMATCH"
                        break
            rec["per_key"] = per
            rec["runs"] = [[a, b] for a, b in _rle(per)]
        out.append(rec)
    return out


def _rle(seq):
    runs = []
    for x in seq:
        if runs and runs[-1][0] == x:
            runs[-1][1] += 1
        else:
            runs.append([x, 1])
    return runs


def _imported_frames(pl):
    import bpy
    from bpy_extras import anim_utils
    out = {}
    for g in pl["groups"]:
        if "action" not in g:
            continue
        a = bpy.data.actions[g["action"]]
        s = [x for x in a.slots if x.identifier == g["slot"]][0]
        cb = anim_utils.action_get_channelbag_for_slot(a, s)
        for dp, i, _ in g["fcurves"]:
            f = cb.fcurves.find(dp, index=i)
            if f is not None:
                n = len(f.keyframe_points)
                co = np.empty(2*n, dtype=np.float32)
                f.keyframe_points.foreach_get("co", co)
                out[(g["action"], g["slot"], dp, i)] = co[0::2].astype(np.float64).tolist()
    return out


def apply(pl, pk):
    """W110 apply for uniform groups, then the per-key groups (interpolation only)."""
    import bpy
    from bpy_extras import anim_utils
    out = W.apply(pl)
    vals = {i.identifier: i.value for i in bpy.types.Keyframe.bl_rna.properties["interpolation"].enum_items}
    out["perkey_groups_applied"] = 0
    for rec in pk:
        if rec["status"] != "OK":
            continue
        g = pl["groups"][rec["group_index"]]
        a = bpy.data.actions[g["action"]]
        s = [x for x in a.slots if x.identifier == g["slot"]][0]
        cb = anim_utils.action_get_channelbag_for_slot(a, s)
        want = np.asarray([vals[c] for c in rec["per_key"]], dtype=np.int32)
        for dp, i, _ in g["fcurves"]:
            f = cb.fcurves.find(dp, index=i)
            n = len(f.keyframe_points)
            cur = np.empty(n, dtype=np.int32)
            f.keyframe_points.foreach_get("interpolation", cur)
            if bool((cur == want).all()):
                out["fcurves_already_correct"] += 1
                continue
            f.keyframe_points.foreach_set("interpolation", want)
            out["fcurves_changed"] += 1
            out["keys_changed"] += int((cur != want).sum())
        out["perkey_groups_applied"] += 1
    return out


def recover(fbx_path, stack_name):
    """The candidate: FBX bytes + selected take name + the current (freshly imported) Blender scene -> report."""
    from pathlib import Path
    import bpy
    from io_scene_fbx import parse_fbx
    sha_before = hashlib.sha256(Path(fbx_path).read_bytes()).hexdigest()
    root, version = parse_fbx.parse(str(fbx_path))
    sha_after = hashlib.sha256(Path(fbx_path).read_bytes()).hexdigest()
    fmap = W.fbx_curve_map(root, stack_name)
    scene = W.describe_scene()
    pl = W.plan(fmap, scene)
    sc = bpy.context.scene
    fps = sc.render.fps/sc.render.fps_base
    uids = {int(u) for g in pl["groups"] for u in g["fbx_curves"]}
    pk = perkey_plan(fmap, curve_keys(root, uids), pl, _imported_frames(pl), fps) if pl["status"] == "OK" else []
    n_ok = sum(1 for r in pk if r["status"] == "OK")
    acc = dict(pl["accounting"])
    acc["perkey_groups_considered"] = len(pk)
    acc["perkey_groups_resolved"] = n_ok
    acc["groups_unresolved_after_perkey"] = sum(1 for g in pl["groups"] if g["directive"] == "UNRESOLVED") - n_ok
    rep = {"schema": SCHEMA + ":report", "fbx_sha256": sha_before, "fbx_unchanged_during_read": sha_before == sha_after,
           "fbx_version": version, "stack": stack_name, "fbx_map_digest": W.fbx_map_digest(fmap),
           "escape_mismatch": scene["escape_mismatch"], "plan": pl, "perkey": pk, "accounting": acc,
           "scene_fps": fps}
    rep["applied"] = apply(pl, pk) if pl["status"] == "OK" and not scene["escape_mismatch"] else None
    rep["digest"] = hashlib.sha256(json.dumps({"pk": pk, "acc": acc}, sort_keys=True, default=str).encode()).hexdigest()
    return rep
