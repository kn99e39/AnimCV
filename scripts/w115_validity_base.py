"""W115 scratch-only measurement library (evidence only; NON-PRODUCTION).

Re-tests the unchanged W114 source-preserving Animation Base candidate with two corrected measurement contracts:

  storage model       how a case-level float64 channel quaternion becomes a stored Blender Action F-curve value, reproduced
                      from the unchanged historical renderer (`render_runtime_heading_rig_blender.render`):
                        bone key    pose_bone.rotation_quaternion = (w, x, y, z)  -> float32 cast, keyed as is
                        Object key  armature.rotation_quaternion = rest_q @ to_q(objq) -> float32 mathutils product
                      and the renderer's key order at one frame (a later keyframe_insert on the same channel/frame wins)
  layered change      L0 source cells (float32 state bytes) / L1 case-level float64 channel bits and continuous deltas /
                      L2 predicted stored float32 values / L3 audited Action values. No tolerance anywhere: every layer
                      compares exact values in its own representation
  semantic compare    field-scoped path allowlist (declared before outcomes) + normalize-then-rehash of the evidence /
                      constraints / edit-record / state-id chain; a path is normalized only in an allowlisted field, only if
                      it lies under the build's own recorded root, and only if its referent bytes are equal across builds
  lock semantics      local channel-value preservation (tail-joint projection) and posed world-space joint-position
                      preservation, evaluated separately; neither is adopted as a policy
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import w114_rig_influence as R  # noqa: E402
import w114_source_preserving_base as B  # noqa: E402

F32 = np.float32
CH = R.CHANNELS
# F-curve data paths of the eight keyed W105 channels (index = w, x, y, z)
DATA_PATH = {c: ("rotation_quaternion" if c == "__object__" else f'pose.bones["{c}"].rotation_quaternion') for c in CH}
PLACEHOLDER = "<BUILD_ROOT>"


# ------------------------------------------------------------------------------------------- storage model ---

def blender_mul_f32(a, b):
    """mathutils Quaternion @ Quaternion (BLI mul_qt_qtqt) in float32, wxyz, evaluated in the C source's term order."""
    a = np.asarray(a, F32)
    b = np.asarray(b, F32)
    t0 = a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1] - a[..., 2] * b[..., 2] - a[..., 3] * b[..., 3]
    t1 = a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0] + a[..., 2] * b[..., 3] - a[..., 3] * b[..., 2]
    t2 = a[..., 0] * b[..., 2] + a[..., 2] * b[..., 0] + a[..., 3] * b[..., 1] - a[..., 1] * b[..., 3]
    t3 = a[..., 0] * b[..., 3] + a[..., 3] * b[..., 0] + a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1]
    return np.stack([t0, t1, t2, t3], -1).astype(F32)


def _wxyz(xyzw):
    x, y, z, w = xyzw
    return np.array([w, x, y, z], np.float64)


def stored_case_values(frames, slot, rest_q_wxyz, key_offset_from_first=True):
    """Predicted stored F-curve values {channel: {key_frame: float32[4] (w, x, y, z)}} for one slot of a case.

    Mirrors the renderer's per-frame key writes in order: Object (when an Object root or yaw exists), every entry of
    extra_bone_locals_xyzw, then (when the slot has a mode) proximal/distal arm locals and the mode's extra bone locals.
    Only the object_rotation_xyzw root path is implemented (the W114/W115 case files always provide it when keyed); a
    yaw-only frame raises so a missing branch cannot pass silently.
    """
    rq = np.asarray(rest_q_wxyz, F32)
    f0 = frames[0]["frame_index"]
    out = {c: {} for c in CH}
    for fr in frames:
        f = fr["frame_index"] - f0 + 1 if key_offset_from_first else fr["frame_index"] + 1
        m = fr["modes"].get(slot)
        root = fr.get("object_rotation_xyzw", {}).get(slot)
        yaw = fr["yaw_deg"].get(slot)
        if root is not None:
            out["__object__"][f] = blender_mul_f32(rq, _wxyz(root).astype(F32))
        elif yaw is not None or m is not None:
            raise ValueError("storage model: Object key without object_rotation_xyzw is outside the modelled cases")
        for bone, q in fr.get("extra_bone_locals_xyzw", {}).get(slot, {}).items():
            out[bone][f] = _wxyz(q).astype(F32)
        if m is not None:
            out["upperarm_l"][f] = _wxyz(m["proximal_local_xyzw"]).astype(F32)
            out["lowerarm_l"][f] = _wxyz(m["distal_local_xyzw"]).astype(F32)
            for bone, q in m.get("extra_bone_locals_xyzw", {}).items():
                out[bone][f] = _wxyz(q).astype(F32)
    return out


def stored_arrays(stored, keys):
    """{channel: (len(keys), 4) float64 of float32 stored values; NaN where the channel has no key}."""
    res = {}
    for c in CH:
        a = np.full((len(keys), 4), np.nan)
        for i, k in enumerate(keys):
            if k in stored[c]:
                a[i] = stored[c][k].astype(np.float64)
        res[c] = a
    return res


def audit_arrays(action, keys):
    """Audited Action F-curve values -> {channel: (len(keys), 4) float64; NaN where unkeyed}."""
    by = {(c["data_path"], c["index"]): dict(zip([int(round(f)) for f in c["frames"]], c["values"])) for c in action["channels"]}
    res = {}
    for c in CH:
        a = np.full((len(keys), 4), np.nan)
        for k in range(4):
            col = by.get((DATA_PATH[c], k))
            if col is None:
                continue
            for i, f in enumerate(keys):
                if f in col:
                    a[i, k] = col[f]
        res[c] = a
    return res


def case_channels(frames, slot):
    """Case-level float64 channel quaternions (w, x, y, z) as the renderer would assign them (pre-cast)."""
    T = len(frames)
    res = {c: np.full((T, 4), np.nan) for c in CH}
    for i, fr in enumerate(frames):
        m = fr["modes"].get(slot)
        root = fr.get("object_rotation_xyzw", {}).get(slot)
        if root is not None:
            res["__object__"][i] = _wxyz(root)
        for bone, q in fr.get("extra_bone_locals_xyzw", {}).get(slot, {}).items():
            res[bone][i] = _wxyz(q)
        if m is not None:
            res["upperarm_l"][i] = _wxyz(m["proximal_local_xyzw"])
            res["lowerarm_l"][i] = _wxyz(m["distal_local_xyzw"])
            for bone, q in m.get("extra_bone_locals_xyzw", {}).items():
                res[bone][i] = _wxyz(q)
    return res


# ------------------------------------------------------------------------------------- layered classification ---

def _neq(a, b):
    """Per-row: any component differs (exact), NaN==NaN; NaN vs value counts as different."""
    na, nb = np.isnan(a), np.isnan(b)
    return (~((a == b) | (na & nb))).any(-1)


def _hemi_delta(a, b):
    sign = np.where((np.nan_to_num(a) * np.nan_to_num(b)).sum(-1, keepdims=True) < 0, -1.0, 1.0)
    return np.abs(a - sign * b).max(-1)


def _geodesic_deg(a, b):
    na = np.linalg.norm(a, axis=-1)
    nb = np.linalg.norm(b, axis=-1)
    dot = np.abs((a * b).sum(-1)) / (na * nb)
    return np.degrees(2 * np.arccos(np.clip(dot, 0.0, 1.0)))


def classify(l1_before, l1_after, l2_before, l2_after, rows=None):
    """Per channel: counts of invariant / below_output_resolution / material_stored / missing channel-frames, plus
    continuous pre-cast deltas and stored (post-cast) deltas. No tolerance: every predicate is exact."""
    out = {}
    for c in CH:
        a1, b1, a2, b2 = (np.asarray(x[c], np.float64) for x in (l1_before, l1_after, l2_before, l2_after))
        sel = np.arange(len(a1)) if rows is None else np.asarray(rows)
        a1, b1, a2, b2 = a1[sel], b1[sel], a2[sel], b2[sel]
        nan_mismatch = (np.isnan(a1).any(-1) != np.isnan(b1).any(-1)) | (np.isnan(a2).any(-1) != np.isnan(b2).any(-1))
        l1 = _neq(a1, b1)
        l2 = _neq(a2, b2)
        both = ~np.isnan(a2).any(-1) & ~np.isnan(b2).any(-1)
        l2_rot = np.zeros(len(sel), bool)
        l2_rot[both] = _hemi_delta(a2[both], b2[both]) > 0
        d1 = np.where(~np.isnan(a1).any(-1) & ~np.isnan(b1).any(-1), _hemi_delta(np.nan_to_num(a1), np.nan_to_num(b1)), np.nan)
        g1 = np.where(~np.isnan(d1), _geodesic_deg(np.nan_to_num(a1, nan=1.0), np.nan_to_num(b1, nan=1.0)), np.nan)
        d2 = np.where(both, np.abs(np.nan_to_num(a2) - np.nan_to_num(b2)).max(-1), np.nan)
        rec = {
            "rows": int(len(sel)),
            "invariant": int((~l1 & ~l2 & ~nan_mismatch).sum()),
            "below_output_resolution": int((l1 & ~l2 & ~nan_mismatch).sum()),
            "material_stored": int((l2 & ~nan_mismatch).sum()),
            "material_stored_rotation": int((l2_rot & ~nan_mismatch).sum()),
            "stored_sign_flip_only": int((l2 & ~l2_rot & both).sum()),
            "missing_or_contradictory": int((nan_mismatch | (~l1 & l2)).sum()),
            "l1_bit_changed": int(l1.sum()),
            "l1_max_abs_component_delta": float(np.nanmax(d1)) if np.isfinite(d1).any() else 0.0,
            "l1_max_geodesic_deg": float(np.nanmax(g1)) if np.isfinite(g1).any() else 0.0,
            "l1_min_nonzero_abs_component_delta": float(d1[l1 & np.isfinite(d1)].min()) if (l1 & np.isfinite(d1)).any() else None,
            "l2_max_abs_component_delta": float(np.nanmax(d2)) if np.isfinite(d2).any() else 0.0,
            "changed_rows_l1": [int(r) for r in sel[l1]],
            "changed_rows_l2": [int(r) for r in sel[l2]],
        }
        out[c] = rec
    return out


def storage_rig(name, frames_fn, rest_q, depicts, owners):
    """A W114 RigModel whose channels are the PREDICTED STORED float32 values (as float64), with tolerance 0.

    `frames_fn(values)` -> single-slot ("SLOT") case frames. A channel-frame is changed iff its stored rotation changes at
    float32 resolution (hemisphere-aligned exact comparison, the unchanged W114 `quat_delta` with `> 0`)."""
    def fn(values):
        frames = frames_fn(values)
        keys = list(range(1, len(frames) + 1))
        return stored_arrays(stored_case_values(frames, "SLOT", rest_q), keys)
    return B.RigModel(name, fn, depicts, owners, tol=0.0)


# ------------------------------------------------------------------------------------- mixed-window case frames ---

def slot_frames(slots, times, label):
    """Single-slot ("SLOT") case frames in the W105 layout from per-frame (pose, arm), same fields as
    `w114_rig_influence.real_case_frames` (used for the mixed-window Blender replay)."""
    import math
    frames = []
    for i, ((p, a), t) in enumerate(zip(slots, times)):
        fr = {"bank_position": -1, "frame_index": i, "time_s": float(t), "image": f"window_{i:04d}", "identity": "not_read",
              "runtime_status": "DIAGNOSTIC", "runtime_reasons": [], "yaw_deg": {}, "modes": {}, "forward_cam": {},
              "mode_lines": {}, "object_rotation_xyzw": {}, "extra_bone_locals_xyzw": {}, "fk_points_armature": {},
              "gt_valid": False}
        if p is None:
            fr["yaw_deg"]["SLOT"] = None
        else:
            fr["object_rotation_xyzw"]["SLOT"] = p["export"]["object"]
            fr["extra_bone_locals_xyzw"]["SLOT"] = dict(p["export"]["bones"])
            fr["fk_points_armature"]["SLOT"] = p["export"]["points"]
            fr["forward_cam"]["SLOT"] = p["export"]["forward"]
            fr["yaw_deg"]["SLOT"] = math.degrees(math.atan2(-p["export"]["forward"][0], p["export"]["forward"][1]))
            if a is not None:
                fr["modes"]["SLOT"] = a
            else:
                for bone in ("upperarm_l", "lowerarm_l"):
                    fr["extra_bone_locals_xyzw"]["SLOT"][bone] = list(R.REST_Q)
        frames.append(fr)
    return frames


# ----------------------------------------------------------------------------------------- lock semantics ---

# Declared correspondence of the 17 H36M joints to posed rig points (head of the bone that starts at the joint).
# A declared, approximate geometric correspondence; the torso/neck/head entries are not anatomical equivalences.
WORLD_POINT = {"pelvis": "spine_01", "left_hip": "thigh_l", "left_knee": "calf_l", "left_ankle": "foot_l",
               "right_hip": "thigh_r", "right_knee": "calf_r", "right_ankle": "foot_r", "spine": "spine_03",
               "thorax": "neck_01", "neck": "neck_02", "head": "neck_02.tail", "left_shoulder": "upperarm_l",
               "left_elbow": "lowerarm_l", "left_wrist": "hand_l", "right_shoulder": "upperarm_r",
               "right_elbow": "lowerarm_r", "right_wrist": "hand_r"}
# joints whose local channel value placement the tail-joint projection assigns to a keyed channel
LOCAL_DEPICTED_BY = {j: [c for c, js in R.DEPICTS.items() if j in js] for j in B.JOINTS}


def local_lock(stored_before, stored_after, lock_mask, rows=None):
    """Local channel-value preservation: a locked (frame, joint) cell holds iff every keyed channel depicting the joint has
    bit-identical stored values at that frame. Joints depicted by no keyed channel are reported as `no_keyed_channel`."""
    T = lock_mask.shape[0]
    sel = np.arange(T) if rows is None else np.asarray(rows)
    changed = {c: _neq(np.asarray(stored_before[c])[sel], np.asarray(stored_after[c])[sel]) for c in CH}
    held = violated = undepicted = 0
    viol_joints = {}
    for k, r in enumerate(sel):
        for j, n in enumerate(B.JOINTS):
            if not lock_mask[r, j]:
                continue
            chans = LOCAL_DEPICTED_BY[n]
            if not chans:
                undepicted += 1
                held += 1
                continue
            if any(changed[c][k] for c in chans):
                violated += 1
                viol_joints[n] = viol_joints.get(n, 0) + 1
            else:
                held += 1
    locked = held + violated
    return {"locked_cells": int(locked), "held": int(held), "violated": int(violated), "no_keyed_channel_cells": int(undepicted),
            "violated_per_joint": viol_joints,
            "verdict": "VACUOUS" if locked == 0 else ("VIOLATED" if violated else "SATISFIED")}


def world_lock(pos_before, pos_after, lock_mask, rows=None):
    """Posed world-space joint-position preservation: a locked cell holds iff the Blender-evaluated world position of the
    joint's declared rig point is bit-identical. Magnitudes are reported, never thresholded."""
    T = lock_mask.shape[0]
    sel = np.arange(T) if rows is None else np.asarray(rows)
    held = violated = 0
    viol_joints, max_disp = {}, {}
    for r in sel:
        for j, n in enumerate(B.JOINTS):
            if not lock_mask[r, j]:
                continue
            a = np.asarray(pos_before[r][WORLD_POINT[n]], np.float64)
            b = np.asarray(pos_after[r][WORLD_POINT[n]], np.float64)
            if np.array_equal(a, b):
                held += 1
            else:
                violated += 1
                viol_joints[n] = viol_joints.get(n, 0) + 1
                max_disp[n] = max(max_disp.get(n, 0.0), float(np.linalg.norm(a - b)))
    locked = held + violated
    return {"locked_cells": int(locked), "held": int(held), "violated": int(violated), "violated_per_joint": viol_joints,
            "max_world_displacement_m_per_joint": max_disp,
            "verdict": "VACUOUS" if locked == 0 else ("VIOLATED" if violated else "SATISFIED")}


def rest_ancestors(rest_json):
    bones = json.loads(Path(rest_json).read_text())["bones"]
    anc = {}
    for b in bones:
        chain, cur = [], bones[b]["parent"]
        while cur is not None:
            chain.append(cur)
            cur = bones[cur]["parent"]
        anc[b] = chain
    return anc


def world_dependencies(rest_json):
    """Keyed channels whose rotation can move each declared world point (strict bone ancestors + the Object)."""
    anc = rest_ancestors(rest_json)
    out = {}
    for n, p in WORLD_POINT.items():
        bone = p.split(".")[0]
        chain = list(anc[bone]) + ([bone] if p.endswith(".tail") else [])
        out[n] = ["__object__"] + [c for c in CH if c in chain]
    return out


# ------------------------------------------------------------------------------------- semantic comparison ---

def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def get_field(obj, path):
    for k in path:
        obj = obj[k]
    return obj


def set_field(obj, path, value):
    for k in path[:-1]:
        obj = obj[k]
    obj[path[-1]] = value


def own_root_relative(value: str, root: str):
    """Relative remainder if `value` lies under `root` (separator-aware, slash-insensitive), else None."""
    v = value.replace("\\", "/")
    r = root.replace("\\", "/").rstrip("/") + "/"
    return v[len(r):] if v.startswith(r) else None


def save_bundle(path: Path, ev: dict, constraints: dict, lineage, extra_records=None):
    """Serialize one population's evidence + constraints + lineage (records, per-state values/origin) for comparison."""
    path.mkdir(parents=True, exist_ok=False)
    (path / "evidence_meta.json").write_text(B.canonical(ev["meta"]))
    np.savez(path / "evidence_arrays.npz", **{k: np.asarray(v) for k, v in ev["arrays"].items()})
    (path / "constraints.json").write_text(B.canonical({k: constraints[k] for k in sorted(constraints)}))
    states = []
    vals = {}
    for i, sid in enumerate(lineage.order):
        s = lineage.states[sid]
        states.append({"state_id": sid, "parent": s["parent"], "record": s["record"]})
        vals[f"values_{i:03d}"] = np.asarray(s["values"])
        vals[f"origin_{i:03d}"] = np.asarray(s["origin"])
    (path / "lineage.json").write_text(json.dumps(states, sort_keys=True, separators=(",", ":"), allow_nan=False))
    np.savez(path / "lineage_arrays.npz", **vals)


def load_bundle(path: Path) -> dict:
    meta = json.loads((path / "evidence_meta.json").read_text())
    with np.load(path / "evidence_arrays.npz", allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    with np.load(path / "lineage_arrays.npz", allow_pickle=False) as z:
        la = {k: z[k] for k in z.files}
    return {"meta": meta, "arrays": arrays, "constraints": json.loads((path / "constraints.json").read_text()),
            "states": json.loads((path / "lineage.json").read_text()), "lineage_arrays": la}


def bundle_integrity(bd: dict) -> dict:
    """Recompute every digest from stored content: evidence digest, constraints digest bound in records, state ids."""
    ev = {"meta": bd["meta"], "arrays": bd["arrays"]}
    problems = []
    try:
        ok_ev = B.evidence_digest(ev) == bd["meta"].get("evidence_digest")
    except Exception as e:  # malformed content
        ok_ev, problems = False, problems + [f"evidence digest not computable: {e}"]
    if not ok_ev:
        problems.append("evidence_digest does not equal the digest of the stored content")
    ed = bd["meta"].get("evidence_digest")
    cdig = {cid: B.constraints_digest(c) for cid, c in bd["constraints"].items()}
    for cid, c in bd["constraints"].items():
        if c.get("evidence_digest") != ed:
            problems.append(f"constraints {cid} bound to another evidence digest")
    ids = [s["state_id"] for s in bd["states"]]
    for i, s in enumerate(bd["states"]):
        v, o = bd["lineage_arrays"][f"values_{i:03d}"], bd["lineage_arrays"][f"origin_{i:03d}"]
        if B.state_id(ed, s["parent"], B.record_digest(s["record"]), v, o) != s["state_id"]:
            problems.append(f"state {i} id does not recompute")
        if s["parent"] is not None and s["parent"] not in ids[:i]:
            problems.append(f"state {i} parent is not an earlier state")
        rec = s["record"]
        if rec is not None:
            if rec.get("evidence_digest") != ed:
                problems.append(f"state {i} record bound to another evidence digest")
            if rec.get("base_state_id") != s["parent"]:
                problems.append(f"state {i} record base differs from its parent link")
            if "constraints_digest" in rec and rec["constraints_digest"] != cdig.get(rec.get("constraints_id")):
                problems.append(f"state {i} record constraints digest does not match its constraints")
    for k, src in bd["meta"].get("sources", {}).items():
        if not isinstance(src, dict) or not isinstance(src.get("sha256"), str) or len(src["sha256"]) != 64:
            problems.append(f"source {k} lacks a sha256")
    return {"ok": not problems, "problems": problems}


def normalized_chain(bd: dict, allow_fields, build_root: str, actual_dir: Path):
    """Normalize ONLY the allowlisted evidence-meta path fields (guarded), then re-hash the whole chain.

    Returns (chain, referents, problems). chain = normalized evidence digest, constraints digests, state ids in order."""
    meta = copy.deepcopy(bd["meta"])
    meta.pop("evidence_digest", None)
    problems, referents = [], {}
    for path in allow_fields:
        try:
            v = get_field(meta, path)
        except (KeyError, TypeError):
            continue
        rel = own_root_relative(v, build_root) if isinstance(v, str) else None
        if rel is None:
            problems.append(f"allowlisted field {'.'.join(path)} does not lie under the build's own root")
            continue
        f = Path(actual_dir) / rel
        referents[".".join(path)] = {"relative": rel, "sha256": B.sha_file(f) if f.exists() else None}
        set_field(meta, path, f"{PLACEHOLDER}/{rel}")
    h = hashlib.sha256(B.DIGEST_DOMAIN)
    h.update(B.canonical(meta).encode())
    B._hash_arrays(h, bd["arrays"])
    ed_n = h.hexdigest()
    ed = bd["meta"].get("evidence_digest")
    cmap, cn = {}, {}
    for cid, c in bd["constraints"].items():
        raw = B.constraints_digest(c)
        c2 = dict(c, evidence_digest=ed_n if c.get("evidence_digest") == ed else c.get("evidence_digest"))
        cmap[raw] = cn[cid] = B.constraints_digest(c2)
    smap, chain = {None: None}, []
    for i, s in enumerate(bd["states"]):
        rec = s["record"]
        rec_n = None
        if rec is not None:
            rec_n = dict(rec)
            if rec_n.get("evidence_digest") == ed:
                rec_n["evidence_digest"] = ed_n
            if "constraints_digest" in rec_n:
                rec_n["constraints_digest"] = cmap.get(rec_n["constraints_digest"], rec_n["constraints_digest"])
            rec_n["base_state_id"] = smap.get(rec_n.get("base_state_id"), "UNMAPPED:" + str(rec_n.get("base_state_id")))
        v, o = bd["lineage_arrays"][f"values_{i:03d}"], bd["lineage_arrays"][f"origin_{i:03d}"]
        sid_n = B.state_id(ed_n, smap.get(s["parent"], "UNMAPPED"), B.record_digest(rec_n), v, o)
        smap[s["state_id"]] = sid_n
        chain.append(sid_n)
    return {"evidence_digest": ed_n, "constraints": cn, "states": chain}, referents, problems


def json_diff(p, q, path=()):
    out = []
    if isinstance(p, dict) and isinstance(q, dict):
        for k in sorted(set(p) | set(q)):
            if k not in p or k not in q:
                out.append(path + (k,))
            else:
                out.extend(json_diff(p[k], q[k], path + (k,)))
    elif isinstance(p, list) and isinstance(q, list) and len(p) == len(q):
        for i, (u, v) in enumerate(zip(p, q)):
            out.extend(json_diff(u, v, path + (i,)))
    elif p != q or type(p) is not type(q):
        out.append(path)
    return out


def matches(path, pattern):
    """Field-path match; '*' in the pattern matches exactly one list index (never a dict key, never a prefix)."""
    if len(path) != len(pattern):
        return False
    return all(pp == "*" and isinstance(x, int) or pp == x for x, pp in zip(path, pattern))
