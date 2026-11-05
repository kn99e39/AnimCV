#!/usr/bin/env python3
"""W116 scratch Semantic Constraint prototype: WORLD_POSITION_PIN only (NON-PRODUCTION, evidence only).

The seven responsibilities of the Batch hypothesis are kept in separate sections of this file:

  1. Immutable Source Evidence   -> `SourceGuard` (hash-bound files, checked before/after every stage)
  2. Editable Animation State    -> `Store` (content-addressed, read-only, parent lineage, refs)
  3. Semantic Constraint         -> `make_pin` / `validate_request` (declarative; no solver field is read by the solver
                                    except the authorised DOF list and the target)
  4. Rig Binding                 -> `verify_binding` (declared candidate rig points; coincidence + existence verified
                                    against Blender-evaluated geometry; never inferred from names)
  5. Constraint Resolver         -> `resolve_frame` (bounded analytic two-bone / aim solvers, bounded CCD; operates on a
                                    copy of the control state's values through the Python FK model `Rig`)
  6. Constraint Evaluator        -> `evaluate_candidate` (reads ONLY Blender-evaluated output + Blender rest lengths; never
                                    the resolver's internal positions)
  7. Edit Commit / Rejection     -> `decide` + `Store.commit` (atomic accept, or reject leaving the parent/ref unchanged)

WORLD = the Blender scene world (Armature Object transform x posed skeleton), metres.  NOT reconstructed physical space.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------------------------------- declared constants ---
TOL_M = 1e-3                    # provisional diagnostic acceptance target (NOT a production tolerance)
BOUNDARY_BAND_M = 1e-5          # GT-feasibility margins inside this band are reported as BOUNDARY, not counted
NUMERIC_FLOOR_M = 1e-5          # reporting floor for "materially changed" world points (bit-exact counts reported too)
LENGTH_TOL_M = 1e-6             # evaluated bone length vs Blender rest length * scale
ELBOW_FLEX_LIMIT_DEG = (0.0, 150.0)   # declared joint limit of the two-bone chain (flex = angle(upper dir, lower dir))
CCD_TOL_M = 1e-4                # CCD internal convergence criterion
COINCIDENT_REST = 1e-4          # binding candidate coincidence at rest (armature units)
COINCIDENT_EVAL_M = 1e-6        # binding candidate coincidence on every Blender-evaluated parent frame
SUPPORTED_TYPES = ("WORLD_POSITION_PIN",)
NOT_IMPLEMENTED_TYPES = ("ORIENTATION_PIN", "MOTION_PRESERVATION", "CONTACT_PRESERVATION", "EDIT_REGION", "SOFT_STYLE")
SUPPORTED_FRAMES = ("SCENE_WORLD",)
SUPPORTED_UNITS = ("m",)
OBJECT = "__object__"

FAILURE_CLASSES = ("NONE", "INFEASIBLE_REQUEST", "UNSUPPORTED_BINDING", "SOLVER_FAILURE", "CONTRACT_VIOLATION",
                   "INVALID_EXPERIMENT")


def canon(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(obj) -> str:
    return hashlib.sha256(canon(obj)).hexdigest()


def sha_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ quaternions ---
def q_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])


def q_conj(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])


def q_axis_angle(axis, angle):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    s = math.sin(angle / 2)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, math.cos(angle / 2)])


def q_to_m(q):
    q = np.asarray(q, float)
    q = q / np.linalg.norm(q)
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def m_to_q(m):
    m = np.asarray(m, float)
    t = np.trace(m)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def q_between(u, v):
    """Minimal (swing) rotation taking direction u to direction v."""
    u = np.asarray(u, float) / np.linalg.norm(u)
    v = np.asarray(v, float) / np.linalg.norm(v)
    c = np.cross(u, v)
    n = np.linalg.norm(c)
    if n < 1e-15:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return q_axis_angle(c / n, math.atan2(n, float(np.dot(u, v))))


def wxyz_to_xyzw(q):
    return [q[1], q[2], q[3], q[0]]


# ---------------------------------------------------------------------------------------------- rig FK model ---
class Rig:
    """Python FK model of a Blender armature (rotation-only pose bones), used by the RESOLVER only.

    pose(b) = pose(parent) @ inv(rest(parent)) @ rest(b) @ basis(b); world = T(loc) R(rest_q * objq) S(scale)."""

    def __init__(self, spec: dict):
        self.spec = spec
        self.rig_id = spec["rig_id"]
        self.bones = {}
        for n, b in spec["bones"].items():
            M = np.array(b["matrix_local"], float)
            head = np.array(b["head"], float)
            tail = np.array(b["tail"], float)
            self.bones[n] = {"parent": b["parent"], "M": M, "Minv": np.linalg.inv(M), "len": float(np.linalg.norm(tail - head))}
        self.order = []
        seen = set()
        while len(self.order) < len(self.bones):
            for n, b in sorted(self.bones.items()):
                if n not in seen and (b["parent"] is None or b["parent"] in seen):
                    self.order.append(n)
                    seen.add(n)
        o = spec["object"]
        self.loc = np.array(o["location"], float)
        self.rest_q = np.array(wxyz_to_xyzw(o["rest_rotation_wxyz"]), float)
        self.scale = float(o["scale"][0])
        if any(abs(s - self.scale) > 0 for s in o["scale"]):
            raise ValueError("non-uniform armature scale is outside the W116 FK model")
        self.children = {n: [c for c, b in self.bones.items() if b["parent"] == n] for n in self.bones}

    def ancestors(self, bone):
        out = []
        while bone is not None:
            out.append(bone)
            bone = self.bones[bone]["parent"]
        return out[::-1]  # root -> bone

    def descendants(self, bone):
        out, stack = [], [bone]
        while stack:
            b = stack.pop()
            out.append(b)
            stack.extend(self.children[b])
        return sorted(out)

    def is_root(self, bone):
        return self.bones[bone]["parent"] is None

    def fk(self, vals: dict):
        """vals: channel -> xyzw (missing bone = identity basis).  Returns object rotation and per-bone pose matrices."""
        objq = np.asarray(vals.get(OBJECT, [0, 0, 0, 1]), float)
        Robj = q_to_m(q_mul(self.rest_q, objq))
        P = {}
        for n in self.order:
            b = self.bones[n]
            B = np.eye(4)
            B[:3, :3] = q_to_m(vals[n]) if n in vals else np.eye(3)
            P[n] = (b["M"] @ B) if b["parent"] is None else (P[b["parent"]] @ self.bones[b["parent"]]["Minv"] @ b["M"] @ B)
        return {"Robj": Robj, "P": P}

    def world(self, fk, bone, point):
        P = fk["P"][bone]
        p = P[:3, 3] if point == "head" else (P @ np.array([0.0, self.bones[bone]["len"], 0.0, 1.0]))[:3]
        return self.loc + fk["Robj"] @ (self.scale * p)

    def all_points(self, fk):
        return {b: {"head": self.world(fk, b, "head"), "tail": self.world(fk, b, "tail")} for b in self.bones}

    def frame_of_rest(self, fk, bone):
        """World rotation A_b with world_rot(b) = A_b @ basis(b)."""
        b = self.bones[bone]
        if b["parent"] is None:
            R = b["M"][:3, :3]
        else:
            R = (fk["P"][b["parent"]] @ self.bones[b["parent"]]["Minv"] @ b["M"])[:3, :3]
        return fk["Robj"] @ R

    def world_rot(self, fk, bone):
        return fk["Robj"] @ fk["P"][bone][:3, :3]

    def apply_world_swing(self, vals, fk, bone, q_world):
        """basis' = (A^-1 S A) basis, i.e. pre-multiply the bone's world rotation by S (pivot = the bone head)."""
        qA = m_to_q(self.frame_of_rest(fk, bone))
        local = q_mul(q_mul(q_conj(qA), q_world), qA)
        new = dict(vals)
        new[bone] = q_mul(local, np.asarray(vals[bone], float))
        return new

    def world_len(self, bone):
        return self.scale * self.bones[bone]["len"]


# ------------------------------------------------------------------------------------- 1. immutable source ---
class SourceGuard:
    def __init__(self, files: dict):
        self.files = {k: str(v) for k, v in files.items()}
        self.before = {k: sha_file(v) for k, v in self.files.items()}

    def check(self):
        now = {k: sha_file(v) for k, v in self.files.items()}
        return {"unchanged": now == self.before, "before": self.before, "after": now}


# ---------------------------------------------------------------------------------- 2. editable animation state ---
def make_state(parent_id, rig_id, frames, channels, values, op, source=None):
    st = {"schema": "w116_animation_state_v1", "parent_id": parent_id, "rig_id": rig_id, "frames": list(map(int, frames)),
          "channels": list(channels), "values": {c: [[float(x) for x in q] for q in values[c]] for c in channels},
          "op": op, "source": source}
    st["state_id"] = digest({k: v for k, v in st.items() if k != "state_id"})
    return st


def verify_state(st):
    return st["state_id"] == digest({k: v for k, v in st.items() if k != "state_id"})


def frame_vals(st, i):
    return {c: np.array(st["values"][c][i], float) for c in st["channels"]}


class Store:
    """Content-addressed scratch store.  States are written once (read-only); refs move only on an atomic commit."""

    def __init__(self, root: Path):
        self.root = Path(root)
        (self.root / "states").mkdir(parents=True, exist_ok=True)
        self.refs = {}
        self.log = []

    def path(self, sid):
        return self.root / "states" / f"{sid}.json"

    def put(self, st):
        if not verify_state(st):
            raise ValueError("state id does not bind its content")
        p = self.path(st["state_id"])
        if p.exists():
            if p.read_bytes() != canon(st):
                raise ValueError("refusing to overwrite a different state under the same id")
            return st["state_id"]
        p.write_bytes(canon(st))
        p.chmod(0o444)
        return st["state_id"]

    def get(self, sid):
        st = json.loads(self.path(sid).read_text())
        if not verify_state(st):
            raise ValueError(f"stored state {sid} fails its content binding")
        return st

    def init_ref(self, ref, sid):
        if ref in self.refs:
            raise ValueError("ref exists")
        self.refs[ref] = sid
        self.log.append({"op": "init", "ref": ref, "head": sid})

    def commit(self, ref, expected_parent, candidate, decision):
        """Atomic: the ref moves to the candidate only if the decision is ACCEPT and the parent is still the head."""
        head = self.refs[ref]
        rec = {"op": "commit", "ref": ref, "expected_parent": expected_parent, "head_before": head,
               "candidate": candidate["state_id"] if candidate else None, "decision": decision["decision"],
               "reasons": decision["reasons"]}
        if head != expected_parent:
            rec.update(decision="REJECT", reasons=["STALE_PARENT"], failure_class="CONTRACT_VIOLATION")
        elif decision["decision"] == "ACCEPT":
            if candidate["parent_id"] != head:
                rec.update(decision="REJECT", reasons=["CANDIDATE_PARENT_MISMATCH"], failure_class="CONTRACT_VIOLATION")
            else:
                self.put(candidate)
                self.refs[ref] = candidate["state_id"]
        rec["head_after"] = self.refs[ref]
        self.log.append(rec)
        return rec

    def revert(self, ref):
        """New state whose values are the parent's values, parented on the current head (history is never rewritten)."""
        head = self.get(self.refs[ref])
        parent = self.get(head["parent_id"])
        st = make_state(head["state_id"], parent["rig_id"], parent["frames"], parent["channels"], parent["values"],
                        {"kind": "REVERT", "reverts": head["state_id"], "restores": parent["state_id"]}, parent.get("source"))
        self.put(st)
        self.refs[ref] = st["state_id"]
        rec = {"op": "revert", "ref": ref, "from": head["state_id"], "to": st["state_id"], "restores": parent["state_id"],
               "values_equal_restored": canon(st["values"]) == canon(parent["values"])}
        self.log.append(rec)
        return st, rec

    def lineage(self, sid):
        out = []
        while sid is not None:
            st = self.get(sid)
            out.append({"state_id": sid, "op": st["op"].get("kind"), "parent_id": st["parent_id"], "verifies": verify_state(st)})
            sid = st["parent_id"]
        return out


# ------------------------------------------------------------------------------------------- 4. rig binding ---
def verify_binding(joint, candidates, rig_id, rest_digest, blender_rest, evaluated_frames, state_channels, rig: Rig):
    """candidates: declared list of (bone, point).  BOUND only if every candidate exists and all candidates coincide at
    rest and on every Blender-evaluated parent frame; otherwise fails safe as MISSING or AMBIGUOUS."""
    rec = {"semantic_joint": joint, "rig_id": rig_id, "candidates": [list(c) for c in candidates]}
    if not candidates:
        return dict(rec, status="BINDING_MISSING", reason="no declared rig point for this semantic joint")
    missing = [b for b, _ in candidates if b not in blender_rest]
    if missing:
        return dict(rec, status="BINDING_MISSING", reason=f"bone(s) absent from the evaluated rig: {missing}")
    pts = lambda fr, b, p: np.array(fr["world"][b][p])  # noqa: E731
    rest_pt = lambda b, p: np.array(blender_rest[b]["head_local" if p == "head" else "tail_local"])  # noqa: E731
    rest_gap = max(float(np.linalg.norm(rest_pt(*c) - rest_pt(*candidates[0]))) for c in candidates)
    eval_gap = max((float(np.linalg.norm(pts(fr, *c) - pts(fr, *candidates[0]))) for fr in evaluated_frames for c in candidates),
                   default=0.0)
    rec.update(rest_gap_armature_units=rest_gap, evaluated_gap_m=eval_gap, frames_checked=len(evaluated_frames))
    if rest_gap > COINCIDENT_REST or eval_gap > COINCIDENT_EVAL_M:
        return dict(rec, status="BINDING_AMBIGUOUS", reason="declared candidate points are different rig points")
    bone, point = candidates[0]
    chain = rig.ancestors(bone) if point == "tail" else rig.ancestors(bone)[:-1]
    rec.update(status="BOUND", bone=bone, point=point, rest_digest=rest_digest,
               dependency_channels=[OBJECT] + chain,
               state_controllable_channels=[c for c in [OBJECT] + chain if c in state_channels])
    rec["binding_id"] = digest({k: rec[k] for k in ("semantic_joint", "rig_id", "rest_digest", "bone", "point", "candidates")})
    return rec


# -------------------------------------------------------------------------------------- 3. semantic constraint ---
def make_pin(constraint_id, joint, binding_id, target, target_source, frames, authorized, parent_state_id,
             frame="SCENE_WORLD", unit="m", hard=True, tolerance_m=TOL_M, root_permitted=False, ctype="WORLD_POSITION_PIN"):
    c = {"schema": "w116_semantic_constraint_v1", "constraint_id": constraint_id, "type": ctype, "semantic_joint": joint,
         "binding_id": binding_id, "coordinate_frame": frame, "unit": unit, "target": [float(x) for x in target],
         "target_source": target_source, "frames": list(map(int, frames)), "hard": hard, "tolerance_m": tolerance_m,
         "authorized_channels": sorted(authorized), "root_permitted": root_permitted, "parent_state_id": parent_state_id}
    c["target_digest"] = digest({"target": c["target"], "source": target_source, "binding_id": binding_id, "frames": c["frames"]})
    c["constraint_digest"] = digest({k: v for k, v in c.items() if k != "constraint_digest"})
    return c


def captured_target(capture, src):
    """Recompute a target from its declared source.  capture: (state_id, frame_index, binding_id) -> evaluated position."""
    key = (src.get("state_id"), src.get("frame_index"), src.get("binding_id"))
    if key not in capture:
        return None
    p = np.array(capture[key], float)
    if src["kind"] == "CAPTURED_FROM_EVALUATED_STATE":
        return p
    if src["kind"] == "CAPTURED_PLUS_DECLARED_OFFSET":
        return p + np.array(src["offset_m"], float)
    return None


def validate_request(req, ctx):
    """Contract validation only (no solving).  Returns a list of (failure_class, reason)."""
    errs = []
    bindings, failed, capture = ctx["bindings"], ctx["failed_bindings"], ctx["capture"]
    parent = ctx["parent"]
    edit_ch = set(req["edit"]["channels"]) if req.get("edit") else set()
    for c in req["constraints"]:
        if c.get("constraint_digest") != digest({k: v for k, v in c.items() if k != "constraint_digest"}):
            errs.append(("CONTRACT_VIOLATION", "CONSTRAINT_DIGEST_MISMATCH"))
        if c.get("target_digest") != digest({"target": c["target"], "source": c["target_source"], "binding_id": c["binding_id"],
                                             "frames": c["frames"]}):
            errs.append(("CONTRACT_VIOLATION", "TARGET_ALTERED_AFTER_BINDING"))
        if c["type"] not in SUPPORTED_TYPES:
            errs.append(("CONTRACT_VIOLATION", "TYPE_NOT_IMPLEMENTED" if c["type"] in NOT_IMPLEMENTED_TYPES else "TYPE_UNKNOWN"))
            continue
        if c["coordinate_frame"] not in SUPPORTED_FRAMES:
            errs.append(("CONTRACT_VIOLATION", "COORDINATE_FRAME_UNSUPPORTED"))
        if c["unit"] not in SUPPORTED_UNITS:
            errs.append(("CONTRACT_VIOLATION", "UNIT_UNSUPPORTED"))
        if c["hard"] is not True:
            errs.append(("CONTRACT_VIOLATION", "SOFT_CONSTRAINT_NOT_IMPLEMENTED"))
        if c["tolerance_m"] != TOL_M:
            errs.append(("CONTRACT_VIOLATION", "TOLERANCE_NOT_DECLARED"))
        b = bindings.get(c["binding_id"])
        if b is None:
            fb = failed.get(c["semantic_joint"])
            errs.append(("UNSUPPORTED_BINDING", fb["status"] if fb else "BINDING_UNVERIFIED"))
            continue
        if b["semantic_joint"] != c["semantic_joint"]:
            errs.append(("UNSUPPORTED_BINDING", "BINDING_JOINT_MISMATCH"))
        if b["rig_id"] != parent["rig_id"]:
            errs.append(("UNSUPPORTED_BINDING", "BINDING_RIG_MISMATCH"))
        if c["parent_state_id"] != req["parent_state_id"]:
            errs.append(("CONTRACT_VIOLATION", "CONSTRAINT_PARENT_MISMATCH"))
        if not set(c["frames"]) <= set(parent["frames"]):
            errs.append(("CONTRACT_VIOLATION", "FRAME_NOT_IN_STATE"))
        if c["coordinate_frame"] in SUPPORTED_FRAMES and c["unit"] in SUPPORTED_UNITS:
            t = captured_target(capture, c["target_source"])
            if t is None or canon([float(x) for x in t]) != canon(c["target"]):
                errs.append(("CONTRACT_VIOLATION", "TARGET_PROVENANCE_MISMATCH"))
        auth = set(c["authorized_channels"])
        if not auth <= set(parent["channels"]):
            errs.append(("CONTRACT_VIOLATION", "DOF_NOT_IN_STATE"))
        if not c["root_permitted"] and (OBJECT in auth or any(a in ctx["root_bones"] for a in auth)):
            errs.append(("CONTRACT_VIOLATION", "ROOT_NOT_PERMITTED"))
        if c["root_permitted"]:
            errs.append(("CONTRACT_VIOLATION", "ROOT_AUTHORIZATION_NOT_APPROVED_IN_W116"))
        if auth & edit_ch:
            errs.append(("CONTRACT_VIOLATION", "AUTHORIZED_OVERLAPS_USER_EDIT"))
    pins = [c for c in req["constraints"] if c["type"] in SUPPORTED_TYPES]
    if len({tuple(c["authorized_channels"]) for c in pins}) > 1:
        errs.append(("CONTRACT_VIOLATION", "AUTHORIZATION_INCONSISTENT"))
    for i, a in enumerate(pins):
        for b in pins[i + 1:]:
            if a["binding_id"] == b["binding_id"] and set(a["frames"]) & set(b["frames"]):
                if float(np.linalg.norm(np.array(a["target"]) - np.array(b["target"]))) > 2 * TOL_M:
                    errs.append(("INFEASIBLE_REQUEST", "CONFLICTING_HARD_CONSTRAINTS"))
    return errs


# --------------------------------------------------------------------------------------------- user edits ---
def apply_edit(rig: Rig, vals: dict, edit: dict, frame_param):
    """Declared user edit on ONE frame (copy).  kinds:
       local_axis_rotation: basis' = basis * R(axis_local, angle)
       world_plane_rotation: rotate the bone about its head around the current bend-plane normal of (bone, child)"""
    if edit is None:
        return dict(vals)
    new = dict(vals)
    ch = edit["channels"][0]
    ang = frame_param["angle_rad"]
    if edit["kind"] == "local_axis_rotation":
        new[ch] = q_mul(np.asarray(vals[ch], float), q_axis_angle(edit["axis_local"], ang))
        return new
    if edit["kind"] == "world_plane_rotation":
        fk = rig.fk(vals)
        child = edit["child"]
        S, E, W = rig.world(fk, ch, "head"), rig.world(fk, ch, "tail"), rig.world(fk, child, "tail")
        n = np.cross(E - S, W - S)
        return rig.apply_world_swing(vals, fk, ch, q_axis_angle(n / np.linalg.norm(n), ang))
    raise ValueError(f"unknown edit kind {edit['kind']}")


def edit_param_for_chord(rig: Rig, vals, edit, pin_bone, pin_point, chord_m):
    """Angle such that the pinned point would move by exactly `chord_m` under the declared edit (exact arithmetic)."""
    fk = rig.fk(vals)
    ch = edit["channels"][0]
    pivot = rig.world(fk, ch, "head")
    p = rig.world(fk, pin_bone, pin_point)
    if edit["kind"] == "local_axis_rotation":
        axis = rig.world_rot(fk, ch) @ np.asarray(edit["axis_local"], float)
    else:
        S, E, W = pivot, rig.world(fk, ch, "tail"), rig.world(fk, edit["child"], "tail")
        axis = np.cross(E - S, W - S)
    axis = axis / np.linalg.norm(axis)
    lever = p - pivot
    perp = float(np.linalg.norm(lever - np.dot(lever, axis) * axis))
    return {"angle_rad": edit["sign"] * 2 * math.asin(chord_m / (2 * perp)), "lever_perp_m": perp}


# --------------------------------------------------------------------------------------- 5. constraint resolver ---
def _structure(rig: Rig, pins, auth):
    """Map (pins, authorised channels) to a declared analytic mode."""
    pb = [(p["bone"], p["point"]) for p in pins]
    if any(pt != "tail" for _, pt in pb):
        return None, "PIN_POINT_NOT_CANONICAL_TAIL"
    if len(pins) == 1:
        d = pb[0][0]
        chain = [b for b in rig.ancestors(d) if b in auth]
        dep = set(rig.ancestors(d)) | {OBJECT}
        if not (set(auth) & dep):
            return "none", "NO_AUTHORIZED_DOF_IN_DEPENDENCY"
        p = rig.bones[d]["parent"]
        if set(auth) & dep == {d, p}:
            return "two_bone", None
        if set(auth) & dep == {d}:
            return "aim_distal", None
        if p is not None and set(auth) & dep == {p}:
            return "aim_rigid", None
        return None, "RESOLVER_UNSUPPORTED_DOF_SET"
    if len(pins) == 2:
        (b1, _), (b2, _) = pb
        if rig.bones[b2]["parent"] == b1 and set(auth) == {b1, b2}:
            return "two_pin_chain", None
        return None, "RESOLVER_UNSUPPORTED_PIN_SET"
    return None, "RESOLVER_UNSUPPORTED_PIN_SET"


def _limits_reach(L1, L2):
    lo, hi = ELBOW_FLEX_LIMIT_DEG
    rmin = math.sqrt(L1 * L1 + L2 * L2 + 2 * L1 * L2 * math.cos(math.radians(hi)))
    rmax = math.sqrt(L1 * L1 + L2 * L2 + 2 * L1 * L2 * math.cos(math.radians(lo)))
    return rmin, rmax


def resolve_frame(rig: Rig, ctrl_vals: dict, pins, auth, method="analytic", max_iter=64):
    """pins: list of {"bone","point","target"(np)}.  Returns (status, proposal values, info).  Operates on a copy.
    status in SOLVED | NO_CHANGE_NEEDED | UNREACHABLE | INSUFFICIENT_DOF | NON_CONVERGED | RESOLVER_UNSUPPORTED."""
    vals = {k: np.array(v, float) for k, v in ctrl_vals.items()}
    fk = rig.fk(vals)
    res0 = [float(np.linalg.norm(rig.world(fk, p["bone"], p["point"]) - p["target"])) for p in pins]
    info = {"method": method, "control_internal_residual_m": res0}
    if all(r <= TOL_M for r in res0):
        info["mode"] = "already_satisfied"
        return "NO_CHANGE_NEEDED", vals, info
    mode, why = _structure(rig, pins, auth)
    info["mode"] = mode
    if mode == "none":
        info["reason"] = why
        return "INSUFFICIENT_DOF", vals, info
    if mode is None:
        info["reason"] = why
        return "RESOLVER_UNSUPPORTED", vals, info
    if method == "ccd":
        return _ccd(rig, vals, pins, auth, max_iter, info)
    status = "SOLVED"
    if mode == "two_bone":
        d = pins[0]["bone"]
        p = rig.bones[d]["parent"]
        L1, L2 = rig.world_len(p), rig.world_len(d)
        S, E, T = rig.world(fk, p, "head"), rig.world(fk, p, "tail"), pins[0]["target"]
        rmin, rmax = _limits_reach(L1, L2)
        dist = float(np.linalg.norm(T - S))
        info.update(reach_min_m=rmin, reach_max_m=rmax, shoulder_target_m=dist)
        if dist > rmax + TOL_M or dist < rmin - TOL_M:
            status = "UNREACHABLE"
        dc = min(max(dist, rmin), rmax)
        t = (T - S) / dist
        e = E - S
        bperp = e - np.dot(e, t) * t
        if np.linalg.norm(bperp) < 1e-12:
            ax = rig.world_rot(fk, p) @ np.array([1.0, 0.0, 0.0])
            bperp = ax - np.dot(ax, t) * t
            info["pole_fallback"] = True
        bhat = bperp / np.linalg.norm(bperp)
        ca = max(-1.0, min(1.0, (L1 * L1 + dc * dc - L2 * L2) / (2 * L1 * dc)))
        Enew = S + L1 * (ca * t + math.sqrt(max(0.0, 1 - ca * ca)) * bhat)
        vals = rig.apply_world_swing(vals, fk, p, q_between(E - S, Enew - S))
        fk = rig.fk(vals)
        E2, W2 = rig.world(fk, d, "head"), rig.world(fk, d, "tail")
        vals = rig.apply_world_swing(vals, fk, d, q_between(W2 - E2, T - E2))
    elif mode == "aim_distal":
        d = pins[0]["bone"]
        E, W, T = rig.world(fk, d, "head"), rig.world(fk, d, "tail"), pins[0]["target"]
        gap = abs(float(np.linalg.norm(T - E)) - rig.world_len(d))
        info["sphere_gap_m"] = gap
        if gap > TOL_M:
            status = "UNREACHABLE"
        vals = rig.apply_world_swing(vals, fk, d, q_between(W - E, T - E))
    elif mode == "aim_rigid":
        d = pins[0]["bone"]
        p = rig.bones[d]["parent"]
        S, W, T = rig.world(fk, p, "head"), rig.world(fk, d, "tail"), pins[0]["target"]
        gap = abs(float(np.linalg.norm(T - S)) - float(np.linalg.norm(W - S)))
        info["sphere_gap_m"] = gap
        if gap > TOL_M:
            status = "UNREACHABLE"
        vals = rig.apply_world_swing(vals, fk, p, q_between(W - S, T - S))
    elif mode == "two_pin_chain":
        (pe, pw) = pins
        p, d = pe["bone"], pw["bone"]
        S, E = rig.world(fk, p, "head"), rig.world(fk, p, "tail")
        gap1 = abs(float(np.linalg.norm(pe["target"] - S)) - rig.world_len(p))
        vals = rig.apply_world_swing(vals, fk, p, q_between(E - S, pe["target"] - S))
        fk = rig.fk(vals)
        E2, W2 = rig.world(fk, d, "head"), rig.world(fk, d, "tail")
        gap2 = abs(float(np.linalg.norm(pw["target"] - E2)) - rig.world_len(d))
        info.update(sphere_gap_elbow_m=gap1, sphere_gap_wrist_m=gap2)
        if gap1 > TOL_M or gap2 > TOL_M:
            status = "UNREACHABLE"
        vals = rig.apply_world_swing(vals, fk, d, q_between(W2 - E2, pw["target"] - E2))
    fk = rig.fk(vals)
    info["internal_residual_m"] = [float(np.linalg.norm(rig.world(fk, p["bone"], p["point"]) - p["target"])) for p in pins]
    return status, vals, info


def _ccd(rig, vals, pins, auth, max_iter, info):
    if len(pins) != 1:
        info["reason"] = "CCD_SINGLE_PIN_ONLY"
        return "RESOLVER_UNSUPPORTED", vals, info
    pin = pins[0]
    chain = [b for b in rig.ancestors(pin["bone"]) if b in auth]
    it = 0
    for it in range(1, max_iter + 1):
        for b in reversed(chain):
            fk = rig.fk(vals)
            j = rig.world(fk, b, "head")
            e = rig.world(fk, pin["bone"], pin["point"])
            vals = rig.apply_world_swing(vals, fk, b, q_between(e - j, pin["target"] - j))
        fk = rig.fk(vals)
        r = float(np.linalg.norm(rig.world(fk, pin["bone"], pin["point"]) - pin["target"]))
        if r <= CCD_TOL_M:
            break
    info.update(iterations=it, max_iter=max_iter, internal_residual_m=[r])
    return ("SOLVED" if r <= CCD_TOL_M else "NON_CONVERGED"), vals, info


# ----------------------------------------------------------------------------------- 6. constraint evaluator ---
def _p(ev_frame, bone, point):
    return np.array(ev_frame["world"][bone][point], float)


def gt_feasibility(structure, ctrl_frame, pins, blender_rest, scale):
    """Ground-truth reachability from BLENDER-evaluated control positions and Blender rest lengths (independent of the
    resolver's FK model).  Returns (class, margin_m) with class FEASIBLE | INFEASIBLE | BOUNDARY."""
    L = lambda b: blender_rest[b]["length"] * scale  # noqa: E731
    if structure == "none" or structure is None:
        m = TOL_M - max(float(np.linalg.norm(_p(ctrl_frame, p["bone"], p["point"]) - p["target"])) for p in pins)
    elif structure == "two_bone":
        d = pins[0]["bone"]
        par = blender_rest[d]["parent"]
        S = _p(ctrl_frame, par, "head")
        dist = float(np.linalg.norm(pins[0]["target"] - S))
        L1, L2 = L(par), L(d)
        lo, hi = ELBOW_FLEX_LIMIT_DEG
        rmax = math.sqrt(L1 ** 2 + L2 ** 2 + 2 * L1 * L2 * math.cos(math.radians(lo)))
        rmin = math.sqrt(L1 ** 2 + L2 ** 2 + 2 * L1 * L2 * math.cos(math.radians(hi)))
        m = min(rmax + TOL_M - dist, dist - (rmin - TOL_M))
    elif structure == "aim_distal":
        d = pins[0]["bone"]
        m = TOL_M - abs(float(np.linalg.norm(pins[0]["target"] - _p(ctrl_frame, d, "head"))) - L(d))
    elif structure == "aim_rigid":
        d = pins[0]["bone"]
        S = _p(ctrl_frame, blender_rest[d]["parent"], "head")
        m = TOL_M - abs(float(np.linalg.norm(pins[0]["target"] - S)) - float(np.linalg.norm(_p(ctrl_frame, d, "tail") - S)))
    elif structure == "two_pin_chain":
        pe, pw = pins
        S = _p(ctrl_frame, pe["bone"], "head")
        m = min(TOL_M - abs(float(np.linalg.norm(pe["target"] - S)) - L(pe["bone"])),
                TOL_M - abs(float(np.linalg.norm(pw["target"] - pe["target"])) - L(pw["bone"])))
    else:
        raise ValueError(structure)
    cls = "BOUNDARY" if abs(m) < BOUNDARY_BAND_M else ("FEASIBLE" if m > 0 else "INFEASIBLE")
    return cls, m


def evaluate_candidate(cand_frame, ctrl_frame, parent_frame, pins, changed_allowed, domain_points, blender_rest, scale,
                       chain=None):
    """Independent output evaluation of ONE frame from Blender output only."""
    out = {"pins": []}
    for p in pins:
        after = _p(cand_frame, p["bone"], p["point"])
        out["pins"].append({"semantic_joint": p["semantic_joint"], "target": [float(x) for x in p["target"]],
                            "before_parent": _p(parent_frame, p["bone"], p["point"]).tolist(),
                            "unpinned_control": _p(ctrl_frame, p["bone"], p["point"]).tolist(), "pinned_candidate": after.tolist(),
                            "control_residual_m": float(np.linalg.norm(_p(ctrl_frame, p["bone"], p["point"]) - p["target"])),
                            "candidate_residual_m": float(np.linalg.norm(after - p["target"]))})
    out["pins_satisfied"] = all(x["candidate_residual_m"] <= TOL_M for x in out["pins"])
    # stored channel changes (exact float32 F-curve values)
    ch_vs_parent = sorted(c for c in cand_frame["stored_wxyz"] if cand_frame["stored_wxyz"][c] != parent_frame["stored_wxyz"][c])
    ch_vs_ctrl = sorted(c for c in cand_frame["stored_wxyz"] if cand_frame["stored_wxyz"][c] != ctrl_frame["stored_wxyz"][c])
    out["stored_changed_vs_parent"] = ch_vs_parent
    out["stored_changed_vs_control"] = ch_vs_ctrl
    out["scope_ok"] = set(ch_vs_parent) <= set(changed_allowed["vs_parent"]) and set(ch_vs_ctrl) <= set(changed_allowed["vs_control"])
    # world-point changes
    moved_parent, moved_ctrl, outside_exact = {}, {}, []
    for b in cand_frame["world"]:
        for pt in ("head", "tail"):
            dpar = float(np.linalg.norm(_p(cand_frame, b, pt) - _p(parent_frame, b, pt)))
            dctl = float(np.linalg.norm(_p(cand_frame, b, pt) - _p(ctrl_frame, b, pt)))
            if dpar > NUMERIC_FLOOR_M:
                moved_parent[f"{b}.{pt}"] = dpar
            if dctl > NUMERIC_FLOOR_M:
                moved_ctrl[f"{b}.{pt}"] = dctl
            if f"{b}.{pt}" not in domain_points and cand_frame["world"][b][pt] != parent_frame["world"][b][pt]:
                outside_exact.append(f"{b}.{pt}")
    out["moved_vs_parent_m"] = moved_parent
    out["moved_vs_control_m"] = moved_ctrl
    out["outside_domain_changed_exact"] = outside_exact
    out["locality_ok"] = not outside_exact
    # bone lengths
    worst = 0.0
    for b in cand_frame["world"]:
        ln = float(np.linalg.norm(_p(cand_frame, b, "tail") - _p(cand_frame, b, "head")))
        worst = max(worst, abs(ln - blender_rest[b]["length"] * scale))
    out["max_bone_length_error_m"] = worst
    out["lengths_ok"] = worst <= LENGTH_TOL_M
    # root: Armature Object matrix and root-bone heads identical to the control
    out["root_unchanged"] = (cand_frame["matrix_world"] == ctrl_frame["matrix_world"]
                             and cand_frame["stored_wxyz"].get(OBJECT) == ctrl_frame["stored_wxyz"].get(OBJECT))
    out["root_displacement_vs_parent_m"] = float(np.linalg.norm(np.array(cand_frame["matrix_world"])[:3, 3]
                                                                - np.array(parent_frame["matrix_world"])[:3, 3]))
    # articulation of the pinned chain
    if chain:
        p, d = chain
        S, E, W = _p(cand_frame, p, "head"), _p(cand_frame, d, "head"), _p(cand_frame, d, "tail")
        u, v = E - S, W - E
        flex = math.degrees(math.atan2(np.linalg.norm(np.cross(u, v)), float(np.dot(u, v))))
        Sc, Ec, Wc = _p(ctrl_frame, p, "head"), _p(ctrl_frame, d, "head"), _p(ctrl_frame, d, "tail")
        nc = np.cross(Ec - Sc, Wc - Ec)
        n = np.cross(u, v)
        ctrl_flex = math.degrees(math.atan2(np.linalg.norm(nc), float(np.dot(Ec - Sc, Wc - Ec))))
        out.update(elbow_flex_deg=flex, control_elbow_flex_deg=ctrl_flex,
                   bend_flip=bool(np.dot(n, nc) < 0) if np.linalg.norm(n) > 1e-9 and np.linalg.norm(nc) > 1e-9 else None,
                   joint_limit_ok=ELBOW_FLEX_LIMIT_DEG[0] - 1e-9 <= flex <= ELBOW_FLEX_LIMIT_DEG[1] + 1e-9)
    else:
        out["joint_limit_ok"] = True
    return out


# ------------------------------------------------------------------------------------------ 7. commit / reject ---
def decide(validation_errors, resolver_status, evaluations):
    """ACCEPT only if the contract is valid, the resolver claims a solution, and EVERY evaluated frame satisfies every
    hard pin and the scope/locality/length/root/limit checks.  The resolver's internal residual is never consulted."""
    if validation_errors:
        cls = validation_errors[0][0]
        return {"decision": "REJECT", "failure_class": cls, "reasons": sorted({r for _, r in validation_errors})}
    statuses = set(resolver_status)
    if statuses - {"SOLVED", "NO_CHANGE_NEEDED"}:
        order = ["RESOLVER_UNSUPPORTED", "NON_CONVERGED", "INSUFFICIENT_DOF", "UNREACHABLE"]
        bad = [s for s in order if s in statuses]
        cls = "SOLVER_FAILURE" if bad[0] in ("RESOLVER_UNSUPPORTED", "NON_CONVERGED") else "INFEASIBLE_REQUEST"
        return {"decision": "REJECT", "failure_class": cls, "reasons": sorted(statuses - {"SOLVED", "NO_CHANGE_NEEDED"})}
    reasons = set()
    for ev in evaluations:
        if not ev["pins_satisfied"]:
            reasons.add("EVALUATED_PIN_UNSATISFIED")
        if not ev["scope_ok"]:
            reasons.add("UNAUTHORIZED_CHANNEL_CHANGE")
        if not ev["locality_ok"]:
            reasons.add("OUTPUT_OUTSIDE_DECLARED_DOMAIN")
        if not ev["lengths_ok"]:
            reasons.add("BONE_LENGTH_CHANGED")
        if not ev["root_unchanged"]:
            reasons.add("ROOT_MOVED")
        if not ev["joint_limit_ok"]:
            reasons.add("JOINT_LIMIT")
    if reasons:
        cls = "CONTRACT_VIOLATION" if reasons & {"UNAUTHORIZED_CHANNEL_CHANGE", "OUTPUT_OUTSIDE_DECLARED_DOMAIN", "BONE_LENGTH_CHANGED",
                                                "ROOT_MOVED"} else "SOLVER_FAILURE"
        return {"decision": "REJECT", "failure_class": cls, "reasons": sorted(reasons)}
    if statuses == {"NO_CHANGE_NEEDED"}:
        return {"decision": "ACCEPT", "failure_class": "NONE", "reasons": ["TRIVIALLY_SATISFIED_NOT_EVIDENCE"]}
    return {"decision": "ACCEPT", "failure_class": "NONE", "reasons": []}


def domain_points(rig: Rig, channels):
    """Declared output domain: every evaluated point whose dependency chain contains a changed channel."""
    pts = set()
    for c in channels:
        if c == OBJECT:
            return {f"{b}.{p}" for b in rig.bones for p in ("head", "tail")}
        for b in rig.descendants(c):
            pts.add(f"{b}.tail")
            if b != c:
                pts.add(f"{b}.head")
    return pts
