"""EXPERIMENTAL (opt-in, Batch 131) adapter: frozen BaseRig Action <-> CondMDI HumanML3D-263 partial conditioning.

Never imported by the default CLI/GUI path.  numpy (+ scipy's gaussian filter, as in the official HumanML3D encoder) only.

* ``Rig``: numpy forward kinematics of the BaseRig from Blender rest matrices (Blender's pose-matrix rule), validated
  against Blender-evaluated pose matrices by the harness.
* Declared, duration-preserving time map: 240 Action frames at 30 fps (half-open [0, 8.0) s) <-> 160 model samples at
  20 fps.  Even samples are exact Action frames; odd samples lie exactly half-way between two frames and are SO(3)-slerped
  (INTERPOLATED hypothesis).  The 20 -> 30 reconstruction evaluates the SAME 240 original timestamps.
* ``encode``: the official HumanML3D-263 recipe (uniform skeleton, canonical frame, smoothed root heading, root-relative
  positions) with ONE declared deviation: the floor uses only supported joints (the official rule takes the minimum over
  all joints, including the UNKNOWN feet).  Only declared feature groups are observed; everything UNKNOWN is masked.
* ``project_arm``: generated left shoulder->elbow->wrist directions -> ``upperarm_l`` / ``lowerarm_l`` basis by a
  minimal swing from the frozen W129 B (endpoint slerp) orientation, which supplies the (unobservable) twist.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np

from app.experimental_motionbricks import (AdapterStop, canon_sign, mat_to_wxyz, rot_part, sha256_file,  # noqa: F401
                                           slerp_wxyz, wxyz_to_mat)

SCHEMA = "animcv_exp_w131_condmdi_v1"
A_BLENDER_TO_HML = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])    # Z-up/-Y-fwd -> Y-up/+Z-fwd
FPS_ACTION, FPS_MODEL = 30, 20
N_ACTION = 240
EDIT_CHANNELS = ("upperarm_l", "lowerarm_l")
INTERIOR_ACTION = (172, 199)
KEYED = ("__object__", "spine_01", "thigh_l", "thigh_r", "calf_l", "calf_r", "upperarm_l", "lowerarm_l")

HML_JOINTS = ("pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2", "left_ankle",
              "right_ankle", "spine3", "left_foot", "right_foot", "neck", "left_collar", "right_collar", "head",
              "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist")
CHAINS = ((0, 2, 5, 8, 11), (0, 1, 4, 7, 10), (0, 3, 6, 9, 12, 15), (9, 14, 17, 19, 21), (9, 13, 16, 18, 20))
PARENTS = [-1] + [0] * 21
for _c in CHAINS:
    for _p, _ch in zip(_c[:-1], _c[1:]):
        PARENTS[_ch] = _p
RAW_OFFSETS = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, -1, 0], [0, 1, 0], [0, -1, 0],
                        [0, -1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1], [0, 1, 0], [1, 0, 0], [-1, 0, 0], [0, 0, 1],
                        [0, -1, 0], [0, -1, 0], [0, -1, 0], [0, -1, 0], [0, -1, 0], [0, -1, 0]], float)
# bone lengths of the official HumanML3D target sequence 000021 (= motius HML263_TARGET_OFFSETS, parity-checked)
TARGET_LEN = np.array([0.0, 0.1030739695, 0.1098833680, 0.1315682381, 0.3936232626, 0.3901882172, 0.1431902200,
                       0.4324331284, 0.4256435037, 0.0573647358, 0.1433817595, 0.1494190246, 0.2193600535, 0.1374867707,
                       0.1433828324, 0.1030392274, 0.1316139847, 0.1229843721, 0.2568399906, 0.2630918622, 0.2660117149,
                       0.2698763907])

# BaseRig point for every HML joint ("H:" head, "T:" tail of a bone; HIPMID = thigh-head midpoint)
RIG_POINT = {0: "HIPMID", 1: "H:thigh_l", 2: "H:thigh_r", 3: "H:spine_02", 4: "H:calf_l", 5: "H:calf_r", 6: "H:spine_04",
             7: "H:foot_l", 8: "H:foot_r", 9: "H:spine_05", 10: "T:foot_l", 11: "T:foot_r", 12: "H:neck_01",
             13: "H:clavicle_l", 14: "H:clavicle_r", 15: "T:neck_02", 16: "H:upperarm_l", 17: "H:upperarm_r",
             18: "H:lowerarm_l", 19: "H:lowerarm_r", 20: "H:hand_l", 21: "H:hand_r"}
OBSERVED_HYP, PROXY_PELVIS, PROXY_TORSO, UNKNOWN = "OBSERVED_HYPOTHESIS", "PROXY_PELVIS", "PROXY_RIGID_TORSO", "UNKNOWN"
JOINT_CLASS = {0: PROXY_PELVIS, 1: PROXY_PELVIS, 2: PROXY_PELVIS, 3: PROXY_TORSO, 4: OBSERVED_HYP, 5: OBSERVED_HYP,
               6: PROXY_TORSO, 7: OBSERVED_HYP, 8: OBSERVED_HYP, 9: PROXY_TORSO, 10: UNKNOWN, 11: UNKNOWN, 12: PROXY_TORSO,
               13: PROXY_TORSO, 14: PROXY_TORSO, 15: UNKNOWN, 16: PROXY_TORSO, 17: PROXY_TORSO, 18: OBSERVED_HYP,
               19: UNKNOWN, 20: OBSERVED_HYP, 21: UNKNOWN}
CLASS_NOTE = {
    OBSERVED_HYP: "position follows a DRIVEN Action channel (frozen W102 M1+S0 3D hypothesis via W105; GT-free, not truth)",
    PROXY_PELVIS: "rigid on the Armature Object rotation (W93 GT-free torso-frame hypothesis); no pelvis bone",
    PROXY_TORSO: ("rigid on the driven spine_01 through the UNDRIVEN rest spine_02..05 / neck_01 / clavicles: the upper-spine and "
                  "clavicle bending is UNKNOWN and is the rig's rest; passed as a labelled PROXY because every HML263 arm "
                  "position is pelvis-relative through this chain (declared coupling)"),
    UNKNOWN: "not determined by the source evidence (feet orientation, head/neck orientation, right arm); MASKED",
}
ARM_JOINTS = (18, 20)                                          # left elbow / wrist: observed outside the edit interior only
OBSERVED_JOINTS = tuple(j for j in range(1, 22) if JOINT_CLASS[j] != UNKNOWN)
FLOOR_JOINTS = (7, 8)                                          # supported joints for the declared floor rule
ROOT_FEATURES_OBSERVED = (0, 3)                                # absolute yaw (PROXY), root height (PROXY); xz = UNKNOWN travel


# ------------------------------------------------------------------------------------------------ quaternions ---
def qmul(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    w1, x1, y1, z1 = np.moveaxis(a, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(b, -1, 0)
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)


def qinv(q):
    return np.asarray(q, float) * np.array([1.0, -1.0, -1.0, -1.0])


def qrot(q, v):
    q, v = np.asarray(q, float), np.asarray(v, float)
    qv = q[..., 1:]
    uv = np.cross(qv, v)
    uuv = np.cross(qv, uv)
    return v + 2 * (q[..., :1] * uv + uuv)


def qbetween(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    xyz = np.cross(a, b)
    w = np.sqrt((a * a).sum(-1, keepdims=True) * (b * b).sum(-1, keepdims=True)) + (a * b).sum(-1, keepdims=True)
    q = np.concatenate([w, xyz], -1)
    return q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-8)


def normalize(v, eps=1e-8):
    v = np.asarray(v, float)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), eps)


def swing(a, b):
    """Minimal rotation matrix taking unit direction a onto b."""
    a, b = normalize(a), normalize(b)
    c = float(np.dot(a, b))
    if c < -1 + 1e-9:
        raise AdapterStop("DEGENERATE_SWING", "antiparallel swing")
    v = np.cross(a, b)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1.0 + c)


# ------------------------------------------------------------------------------------------------ BaseRig FK ---
class Rig:
    """BaseRig forward kinematics: pose = parent_pose @ (parent_rest^-1 @ rest) @ basis (root: rest @ basis)."""

    def __init__(self, rest, object_location=(0.0, 0.0, 0.0), object_scale=0.009999999776482582):
        need = {s.split(":", 1)[1] for s in RIG_POINT.values() if ":" in s} | {"thigh_l", "thigh_r"} | set(EDIT_CHANNELS)
        missing = sorted(need - set(rest))
        if missing:
            raise AdapterStop("BASERIG_MISSING_BONES", ",".join(missing))
        self.rest = rest
        self.L = {b: np.asarray(v["matrix_local"], float) for b, v in rest.items()}
        self.length = {b: float(np.linalg.norm(np.asarray(v["tail_local"]) - np.asarray(v["head_local"]))) for b, v in rest.items()}
        order, seen = [], set()

        def visit(b):
            if b in seen:
                return
            p = rest[b]["parent"]
            if p:
                visit(p)
            seen.add(b)
            order.append(b)
        for b in rest:
            visit(b)
        self.order = order
        self.rel = {b: np.linalg.inv(self.L[rest[b]["parent"]]) @ self.L[b] for b in rest if rest[b]["parent"]}
        self.loc = np.asarray(object_location, float)
        self.scale = float(object_scale)

    def pose(self, basis, object_wxyz):
        P = {}
        for b in self.order:
            B = np.eye(4)
            if b in basis:
                B[:3, :3] = wxyz_to_mat(basis[b])
            p = self.rest[b]["parent"]
            P[b] = (P[p] @ self.rel[b] if p else self.L[b]) @ B
        M = np.eye(4)
        M[:3, :3] = wxyz_to_mat(object_wxyz) * self.scale
        M[:3, 3] = self.loc
        return M, P

    def point(self, M, P, spec):
        if spec == "HIPMID":
            return 0.5 * (self.point(M, P, "H:thigh_l") + self.point(M, P, "H:thigh_r"))
        kind, b = spec.split(":", 1)
        loc = P[b][:3, 3] if kind == "H" else P[b][:3, 3] + P[b][:3, 1] * self.length[b]
        return (M @ np.r_[loc, 1.0])[:3]

    def hml_joints(self, basis, object_wxyz, A=A_BLENDER_TO_HML):
        M, P = self.pose(basis, object_wxyz)
        return np.stack([np.asarray(A) @ self.point(M, P, RIG_POINT[j]) for j in range(22)])


def stored_key(stored, ch, f):
    return np.array([stored[ch][str(c)][str(int(f))] for c in range(4)], float)


def pose_at(stored, u, override=None):
    """Keyed channels at fractional Action frame u: exact key at integers; SO(3) slerp between the two keys otherwise.
    ``override`` = {channel: callable(u) -> wxyz} replaces a channel (used for the B arm and for controls)."""
    f0 = int(np.floor(u + 1e-12))
    w = float(u - f0)
    basis, obj = {}, None
    for ch in KEYED:
        if override and ch in override:
            q = np.asarray(override[ch](u), float)
        else:
            q0 = stored_key(stored, ch, f0)
            q = q0 if w < 1e-12 else slerp_wxyz(q0, canon_sign(stored_key(stored, ch, f0 + 1), q0), w)
        if ch == "__object__":
            obj = q
        else:
            basis[ch] = q
    return basis, obj


def b_slerp_fn(stored, ch, f_ctx=171, f_tgt=200):
    """Frozen W129 B: per-channel endpoint-matched slerp of the local basis, t = (f - f_ctx) / (f_tgt - f_ctx)."""
    q0 = stored_key(stored, ch, f_ctx)
    q1 = canon_sign(stored_key(stored, ch, f_tgt), q0)
    return lambda u: slerp_wxyz(q0, q1, (u - f_ctx) / (f_tgt - f_ctx))


# ------------------------------------------------------------------------------------------------ time contract ---
def time_map(n_action=N_ACTION, fps_action=FPS_ACTION, fps_model=FPS_MODEL):
    """Rational, duration-preserving map.  Action frame f (1..n) sits at t = (f-1)/fps_action on the half-open axis
    [0, n/fps_action); model sample k at t = k/fps_model.  Returns exact Fractions."""
    duration = Fraction(n_action, fps_action)
    n_model = duration * fps_model
    if n_model.denominator != 1:
        raise AdapterStop("DURATION_NOT_REPRESENTABLE", f"{n_action} frames at {fps_action} fps = {n_model} model samples")
    n_model = int(n_model)
    u = [1 + Fraction(k * fps_action, fps_model) for k in range(n_model)]
    s = [Fraction((f - 1) * fps_model, fps_action) for f in range(1, n_action + 1)]
    return {"n_action": n_action, "n_model": n_model, "duration_s": duration, "u": u, "s": s,
            "kind": ["EXACT" if x.denominator == 1 else "INTERPOLATED_HALF_FRAME" for x in u]}


def validate_time_map(tm):
    u, s = tm["u"], tm["s"]
    if any(b <= a for a, b in zip(u, u[1:])) or any(b <= a for a, b in zip(s, s[1:])):
        raise AdapterStop("TIME_MAP_NON_MONOTONE")
    if Fraction(len(u), FPS_MODEL) != tm["duration_s"] or Fraction(len(s), FPS_ACTION) != tm["duration_s"]:
        raise AdapterStop("TIME_MAP_DURATION_CHANGED", f"{len(u)} samples / {len(s)} frames vs {tm['duration_s']} s")
    # half-open axis: the last frame(s) may fall inside the last sample's interval [n_model-1, n_model); such frames
    # (Action 240 only) are not reconstructable without extrapolation and are never written (outside the interior).
    if u[0] != 1 or u[-1] > tm["n_action"] or s[0] != 0 or s[-1] >= tm["n_model"]:
        raise AdapterStop("TIME_MAP_OUT_OF_RANGE")
    for x in u:
        if x.denominator not in (1, 2):
            raise AdapterStop("TIME_MAP_NOT_HALF_FRAME", str(x))
    return True


def interior_samples(tm, interior=INTERIOR_ACTION):
    """Model samples strictly between the last context frame (interior[0]-1) and the first target frame (interior[1]+1)."""
    lo, hi = Fraction(interior[0] - 2, FPS_ACTION), Fraction(interior[1], FPS_ACTION)
    return [k for k in range(tm["n_model"]) if lo < Fraction(k, FPS_MODEL) < hi]


def protected_check(interior, protected_action):
    ov = sorted(set(range(interior[0], interior[1] + 1)) & set(protected_action))
    if ov:
        raise AdapterStop("PROTECTED_OVERLAP", f"interior overlaps protected Action frames {ov[:5]}")


# ------------------------------------------------------------------------------------------------ axes / handedness ---
def check_axes(A, rig):
    """A must be a proper rotation, map Blender up to HML up, and keep the rig's left on the HML body-left (+X when the
    structural rest forward - the rest foot direction - maps to +Z)."""
    A = np.asarray(A, float)
    if abs(np.linalg.det(A) - 1.0) > 1e-9:
        raise AdapterStop("AXIS_MAPPING_IMPROPER", f"det = {np.linalg.det(A):.3f}")
    if float(np.dot(A @ np.array([0.0, 0.0, 1.0]), [0.0, 1.0, 0.0])) < 0.99:
        raise AdapterStop("AXIS_UP_MISMATCH")
    M, P = rig.pose({}, (1.0, 0.0, 0.0, 0.0))
    left = A @ (rig.point(M, P, "H:thigh_l") - rig.point(M, P, "H:thigh_r"))
    fwd = A @ (rig.point(M, P, "T:foot_l") - rig.point(M, P, "H:foot_l"))
    fwd[1] = 0.0
    if float(np.dot(np.cross([0.0, 1.0, 0.0], normalize(fwd)), normalize(left))) < 0.5:
        raise AdapterStop("HANDEDNESS_MISMATCH", "rig left is not HML body-left")
    return True


def check_left_right(joints):
    """Per sample: left shoulder/collar on the left-hip side; each elbow nearer its own shoulder; each wrist nearer its own
    elbow.  Catches swapped L/R joint mappings."""
    j = np.asarray(joints, float)
    lat = normalize(j[:, 1] - j[:, 2])
    bad = []
    for k in range(len(j)):
        if (np.dot(j[k, 16] - j[k, 17], lat[k]) <= 0 or np.dot(j[k, 13] - j[k, 14], lat[k]) <= 0
                or np.linalg.norm(j[k, 18] - j[k, 16]) >= np.linalg.norm(j[k, 18] - j[k, 17])
                or np.linalg.norm(j[k, 20] - j[k, 18]) >= np.linalg.norm(j[k, 20] - j[k, 19])):
            bad.append(k)
    if bad:
        raise AdapterStop("LEFT_RIGHT_MISMATCH", f"{len(bad)} samples, first {bad[:5]}")
    return True


# ------------------------------------------------------------------------------------------------ encoder ---
def uniform_skeleton(joints):
    """Official HumanML3D uniform-skeleton retarget: root scaled by the target/source leg ratio (frame 0 offsets of joints
    5 and 8), every bone re-laid with the target length along its source direction (IK->FK equivalent)."""
    j = np.asarray(joints, float)
    src_leg = np.linalg.norm(j[0, 5] - j[0, 2]) + np.linalg.norm(j[0, 8] - j[0, 5])
    tgt_leg = TARGET_LEN[5] + TARGET_LEN[8]
    out = np.zeros_like(j)
    out[:, 0] = j[:, 0] * (tgt_leg / src_leg)
    for c in CHAINS:
        for p, ch in zip(c[:-1], c[1:]):
            out[:, ch] = out[:, p] + TARGET_LEN[ch] * normalize(j[:, ch] - j[:, p])
    return out


def encode(joints, floor_rule="supported", c_ankle=0.0, n_out=None):
    """Standard (relative) HML263 from 22-joint HML-axis positions [T,22,3].  Official recipe; ``floor_rule`` 'official'
    (min over all joints) or 'supported' (min over the ankles minus ``c_ankle``).  Rotation, velocity and contact
    features are NOT produced (zero) - they are never observed in W131.  ``n_out`` rows (default T); the root yaw
    velocity of the last row is 0 (it is never used by the absolute conversion)."""
    from scipy.ndimage import gaussian_filter1d
    pos = uniform_skeleton(joints)
    T = len(pos) if n_out is None else n_out
    if floor_rule == "official":
        floor = pos[:, :, 1].min()
    elif floor_rule == "supported":
        floor = pos[:, FLOOR_JOINTS, 1].min() - c_ankle
    else:
        raise AdapterStop("FLOOR_RULE", floor_rule)
    pos[:, :, 1] -= floor
    init = pos[0].copy()
    pos -= init[0] * np.array([1.0, 0.0, 1.0])
    across0 = normalize(init[2] - init[1] + init[17] - init[16])
    fwd0 = normalize(np.cross([0.0, 1.0, 0.0], across0))
    r0 = qbetween(fwd0, np.array([0.0, 0.0, 1.0]))
    pos = qrot(np.broadcast_to(r0, pos.shape[:-1] + (4,)), pos)
    across = normalize(pos[:, 1] - pos[:, 2] + pos[:, 17] - pos[:, 16])
    fwd = np.cross(np.array([[0.0, 1.0, 0.0]]), across)
    fwd = normalize(gaussian_filter1d(fwd, 20, axis=0, mode="nearest"))
    rq = qbetween(fwd, np.repeat([[0.0, 0.0, 1.0]], len(pos), 0))
    rq[0] = [1.0, 0.0, 0.0, 0.0]
    loc = pos.copy()
    loc[..., 0] -= loc[:, 0:1, 0]
    loc[..., 2] -= loc[:, 0:1, 2]
    loc = qrot(np.repeat(rq[:, None], 22, 1), loc)
    rel = np.zeros((T, 263))
    rv = qmul(rq[1:], qinv(rq[:-1]))
    nr = min(T, len(pos) - 1)
    rel[:nr, 0] = np.arcsin(np.clip(rv[:nr, 2], -1, 1))
    vel = qrot(rq[1:], pos[1:, 0] - pos[:-1, 0])
    rel[:nr, 1:3] = vel[:nr][:, [0, 2]]
    rel[:T, 3] = loc[:T, 0, 1]
    rel[:T, 4:67] = loc[:T, 1:].reshape(T, -1)
    return {"relative": rel, "canonical": pos[:T], "root_quat": rq[:T], "r0": r0, "floor": float(floor),
            "frame0_root_xz": init[0][[0, 2]]}


def rel_to_abs(rel):
    """numpy replica of motius ``relative_to_absolute`` (parity-checked in the scratch env)."""
    rel = np.asarray(rel, float)
    ang = np.zeros(len(rel))
    ang[1:] = rel[:-1, 0]
    ang = np.cumsum(ang)
    rq = np.stack([np.cos(ang), np.zeros_like(ang), np.sin(ang), np.zeros_like(ang)], -1)
    rp = np.zeros((len(rel), 3))
    rp[1:, [0, 2]] = rel[:-1, 1:3]
    rp = np.cumsum(qrot(qinv(rq), rp), 0)
    out = rel.copy()
    out[:, 0] = ang
    out[:, 1:3] = rp[:, [0, 2]]
    out[:, 3] = rel[:, 3]
    return out


def abs_to_joints(absm):
    """numpy replica of motius ``recover_from_absolute``: canonical-world 22 joints [T,22,3]."""
    absm = np.asarray(absm, float)
    a = absm[:, 0]
    q = np.stack([np.cos(a), np.zeros_like(a), -np.sin(a), np.zeros_like(a)], -1)
    root = np.stack([absm[:, 1], absm[:, 3], absm[:, 2]], -1)
    p = qrot(np.repeat(q[:, None], 21, 1), absm[:, 4:67].reshape(-1, 21, 3))
    p[..., 0] += root[:, None, 0]
    p[..., 2] += root[:, None, 2]
    return np.concatenate([root[:, None], p], 1)


def canonical_dirs_to_blender(vec, r0, A=A_BLENDER_TO_HML):
    """Directions in the canonical HML frame -> Blender world (undo the frame-0 facing rotation, then A^T)."""
    v = qrot(np.broadcast_to(qinv(r0), np.shape(vec)[:-1] + (4,)), np.asarray(vec, float))
    return v @ np.asarray(A)                                      # (A^T v) row-wise


def observation_mask(n_model, interior_k, observe_root=ROOT_FEATURES_OBSERVED, observed_joints=OBSERVED_JOINTS):
    """Absolute-space [263, T] mask: root yaw + height; ric positions of every non-UNKNOWN joint; the left elbow/wrist
    only outside the edit interior.  Rotations, velocities, contacts and root xz are never observed."""
    m = np.zeros((263, n_model), bool)
    for f in observe_root:
        m[f, :] = True
    outside = np.ones(n_model, bool)
    outside[list(interior_k)] = False
    for j in observed_joints:
        sl = slice(4 + 3 * (j - 1), 4 + 3 * j)
        m[sl, :] = True if j not in ARM_JOINTS else outside[None, :]
    return m


def feature_owner(f):
    """(group, joint) of an absolute HML263 feature index."""
    if f == 0:
        return "root_yaw", 0
    if f in (1, 2):
        return "root_xz", 0
    if f == 3:
        return "root_height", 0
    if f < 67:
        return "pos", (f - 4) // 3 + 1
    if f < 193:
        return "rot", (f - 67) // 6 + 1
    if f < 259:
        return "vel", (f - 193) // 3
    return "contact", None


def validate_mask(mask, interior_k):
    m = np.asarray(mask, bool)
    if m.ndim != 2 or m.shape[0] != 263:
        raise AdapterStop("MASK_SHAPE", str(m.shape))
    for f in np.nonzero(m.any(1))[0]:
        g, j = feature_owner(int(f))
        if g in ("rot", "vel", "contact", "root_xz"):
            raise AdapterStop("MASK_OBSERVES_UNKNOWN", f"feature {f} ({g}) is never source-supported")
        if g == "pos" and JOINT_CLASS[j] == UNKNOWN:
            raise AdapterStop("MASK_OBSERVES_UNKNOWN", f"feature {f}: joint {HML_JOINTS[j]} is UNKNOWN")
        if g == "pos" and j in ARM_JOINTS and m[f, list(interior_k)].any():
            raise AdapterStop("MASK_OBSERVES_EDIT_DOMAIN", f"{HML_JOINTS[j]} observed inside the edit interior")
    outside = [k for k in range(m.shape[1]) if k not in set(interior_k)]
    before = [k for k in outside if k < min(interior_k)]
    after = [k for k in outside if k > max(interior_k)]
    for j in ARM_JOINTS:
        f = 4 + 3 * (j - 1)
        if not (m[f, before].any() and m[f, after].any()):
            raise AdapterStop("MASK_NO_SOURCE_ARM_EVIDENCE", HML_JOINTS[j])
    return True


def coverage_matrix(tm, interior_k, mask):
    """Per joint x sample: class, passed (observed in the mask), interpolated."""
    rows = {}
    for j in range(22):
        f = 1 if j == 0 else 4 + 3 * (j - 1)
        passed = mask[f] if j else (mask[0] | mask[3])
        rows[HML_JOINTS[j]] = {"class": JOINT_CLASS[j], "rig_point": RIG_POINT[j], "note": CLASS_NOTE[JOINT_CLASS[j]],
                               "passed_samples": int(passed.sum()),
                               "passed_interpolated_samples": int(sum(passed[k] for k in range(tm["n_model"]) if tm["kind"][k] != "EXACT")),
                               "masked_samples": int((~passed).sum())}
    rows["pelvis"]["root_features"] = {"yaw": "PROXY (official heading of hips + rigid-torso shoulders), passed",
                                       "height": "PROXY (pelvis above the declared supported-joint floor), passed",
                                       "xz": "UNKNOWN travel (no root translation in the Action), MASKED"}
    return {"joints": rows, "feature_groups": {"rot_6d": "MASKED (IK chain through UNKNOWN twist / rest joints)",
                                               "velocity": "MASKED (contains UNKNOWN root travel)",
                                               "foot_contact": "MASKED (UNKNOWN)"},
            "interior_samples": list(interior_k), "n_observed_features": int(mask.sum()),
            "n_samples_exact": int(sum(1 for x in tm["kind"] if x == "EXACT")),
            "n_samples_interpolated": int(sum(1 for x in tm["kind"] if x != "EXACT"))}


# ------------------------------------------------------------------------------------------------ projection ---
def arm_dirs_from_abs(absm, r0, samples):
    """Generated (or encoded) upper-arm / forearm directions in Blender world for the given samples."""
    J = abs_to_joints(absm)
    u = canonical_dirs_to_blender(J[samples, 18] - J[samples, 16], r0)
    v = canonical_dirs_to_blender(J[samples, 20] - J[samples, 18], r0)
    return normalize(u), normalize(v)


def project_arm(rig, basis, obj, ref_upper, ref_lower, u_dir, v_dir):
    """upperarm_l / lowerarm_l basis whose bone Y axes point along u_dir / v_dir (Blender world), by the minimal swing
    from the reference orientation (ref_* basis wxyz, which supplies twist); torso/parents from ``basis``/``obj``."""
    b = dict(basis)
    b["upperarm_l"], b["lowerarm_l"] = ref_upper, ref_lower
    M, P = rig.pose(b, obj)
    Ro = rot_part(M)
    Rua = Ro @ rot_part(P["upperarm_l"])
    Rua_new = swing(Rua[:, 1], u_dir) @ Rua
    Bc, Bu, Bl = (rot_part(rig.L[x]) for x in ("clavicle_l", "upperarm_l", "lowerarm_l"))
    Pc = rot_part(P["clavicle_l"])
    Pu_new = Ro.T @ Rua_new
    bu = mat_to_wxyz((Bc.T @ Bu).T @ Pc.T @ Pu_new)
    b["upperarm_l"] = bu
    M, P = rig.pose(b, obj)
    Rla = Ro @ rot_part(P["lowerarm_l"])
    Rla_new = swing(Rla[:, 1], v_dir) @ Rla
    bl = mat_to_wxyz((Bu.T @ Bl).T @ Pu_new.T @ (Ro.T @ Rla_new))
    return bu, bl


def reconstruct_30(per_sample, frames):
    """20 -> 30 fps on the original timestamps: frame f at model position s = 2(f-1)/3; exact sample when integral,
    otherwise SO(3) slerp between the two neighbouring samples (weights 1/3, 2/3)."""
    out = {}
    for f in frames:
        s = Fraction((f - 1) * FPS_MODEL, FPS_ACTION)
        k0 = int(s)
        w = float(s - k0)
        if k0 not in per_sample or (w > 0 and k0 + 1 not in per_sample):
            raise AdapterStop("RECONSTRUCTION_SAMPLE_MISSING", f"frame {f} needs samples {k0}, {k0 + 1}")
        q0 = np.asarray(per_sample[k0], float)
        out[f] = q0 if w == 0 else slerp_wxyz(q0, canon_sign(per_sample[k0 + 1], q0), w)
    return out
