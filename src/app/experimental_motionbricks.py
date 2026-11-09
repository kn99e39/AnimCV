"""EXPERIMENTAL (opt-in, Batch 129): BaseRig Action <-> MotionBricks SOMA adapter for ONE left-arm inbetweening proposal.

Scope (frozen by the W129 declaration; nothing here is a Production path):
  * read a frozen real-video-derived W124/W126 BaseRig Action (via the Blender ``dump`` of
    ``blender/experimental_motionbricks_stage.py``),
  * map its evaluated pose at the declared context/target Action frames into MotionBricks SOMA local joint transforms
    (77-joint SOMA skeleton; the 30 model joints are named in the model manifest),
  * map the generated SOMA motion back into BaseRig ``upperarm_l`` / ``lowerarm_l`` pose-bone rotations ONLY,
  * compute the A/B/C diagnostics from fresh-process Blender readbacks.

Representation (closed form, no IK, no optimisation):
  Blender world (Z up, character faces -Y) -> SOMA model space (Y up, faces +Z) by the proper rotation
  ``A_BLENDER_TO_SOMA`` (x, y, z) -> (x, z, -y).  Each mapped SOMA joint j driven by BaseRig source b gets the world
  rotation  R_soma(j) = A . R_base(b) . D_j  with the constant  D_j = R_base_rest(b)^T . A^T . Q_j . R_soma_rest(j),
  where Q_j aligns the SOMA rest bone frame to the BaseRig rest bone frame (primary bone direction + a declared
  secondary direction).  Parent-relative (channel) rotations therefore map exactly both ways:
      L_soma(j) = D_p^T . L_base(b) . D_j       L_base(b) = D_p . L_soma(j) . D_j^T.
  Joints with no BaseRig counterpart keep their SOMA reference local transform and are labelled SYNTHETIC.

Only numpy (+ stdlib) is imported, so the module loads in the AnimCV venv and in tests without Blender/TensorRT.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA = "animcv_exp_w129_motionbricks_v1"
A_BLENDER_TO_SOMA = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
EDIT_CHANNELS = ("upperarm_l", "lowerarm_l")
FPS = 30.0
POSES_PER_TOKEN = 4
BOUNDARY_FRAMES = 3

SOMA_MODEL_JOINTS = (
    "Hips", "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head", "Jaw", "LeftEye", "RightEye",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand", "LeftHandThumbEnd", "LeftHandMiddleEnd",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand", "RightHandThumbEnd", "RightHandMiddleEnd",
    "LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase", "RightLeg", "RightShin", "RightFoot", "RightToeBase")

# SOMA joint -> (kind, BaseRig source).  OBJECT = Armature Object frame (BaseRig has no pelvis bone; spine_01 and the
# thighs are root bones).  PROXY = rigid stand-in because the BaseRig has no such bone.
JOINT_MAP = {
    "Hips": ("OBJECT", None),
    "Spine1": ("BONE", "spine_01"), "Spine2": ("BONE", "spine_03"), "Chest": ("BONE", "spine_05"),
    "Neck1": ("BONE", "neck_01"), "Neck2": ("BONE", "neck_02"), "Head": ("PROXY", "neck_02"),
    "LeftShoulder": ("BONE", "clavicle_l"), "LeftArm": ("BONE", "upperarm_l"), "LeftForeArm": ("BONE", "lowerarm_l"),
    "LeftHand": ("BONE", "hand_l"),
    "RightShoulder": ("BONE", "clavicle_r"), "RightArm": ("BONE", "upperarm_r"), "RightForeArm": ("BONE", "lowerarm_r"),
    "RightHand": ("BONE", "hand_r"),
    "LeftLeg": ("BONE", "thigh_l"), "LeftShin": ("BONE", "calf_l"), "LeftFoot": ("BONE", "foot_l"), "LeftToeBase": ("PROXY", "foot_l"),
    "RightLeg": ("BONE", "thigh_r"), "RightShin": ("BONE", "calf_r"), "RightFoot": ("BONE", "foot_r"), "RightToeBase": ("PROXY", "foot_r"),
}
SYNTHETIC_MODEL_JOINTS = tuple(j for j in SOMA_MODEL_JOINTS if j not in JOINT_MAP)

# Rest-frame alignment: (SOMA primary (from, to), BaseRig primary (from-point, to-point), secondary).
# Points: "H:<bone>" head, "T:<bone>" tail, "HIPMID" mid of thigh heads.  Secondary: "FWD" (+Z SOMA forward),
# "UP" (+Y), or ("VEC", (soma from, to), (base from, to)).
ALIGN = {
    "Hips": (("Hips", "Chest"), ("HIPMID", "H:spine_05"), ("VEC", ("RightLeg", "LeftLeg"), ("H:thigh_r", "H:thigh_l"))),
    "Spine1": (("Spine1", "Spine2"), ("H:spine_01", "T:spine_01"), "FWD"),
    "Spine2": (("Spine2", "Chest"), ("H:spine_03", "T:spine_03"), "FWD"),
    "Chest": (("Chest", "Neck1"), ("H:spine_05", "H:neck_01"), "FWD"),
    "Neck1": (("Neck1", "Neck2"), ("H:neck_01", "T:neck_01"), "FWD"),
    "Neck2": (("Neck2", "Head"), ("H:neck_02", "T:neck_02"), "FWD"),
    "Head": (("Head", "HeadEnd"), ("H:neck_02", "T:neck_02"), "FWD"),
}
for _s, _side in (("Left", "l"), ("Right", "r")):
    ALIGN.update({
        f"{_s}Shoulder": ((f"{_s}Shoulder", f"{_s}Arm"), (f"H:clavicle_{_side}", f"T:clavicle_{_side}"), "FWD"),
        f"{_s}Arm": ((f"{_s}Arm", f"{_s}ForeArm"), (f"H:upperarm_{_side}", f"T:upperarm_{_side}"), "FWD"),
        f"{_s}ForeArm": ((f"{_s}ForeArm", f"{_s}Hand"), (f"H:lowerarm_{_side}", f"T:lowerarm_{_side}"), "FWD"),
        f"{_s}Hand": ((f"{_s}Hand", f"{_s}HandMiddle1"), (f"H:hand_{_side}", f"H:middle_01_{_side}"),
                      ("VEC", (f"{_s}Hand", f"{_s}HandThumb1"), (f"H:hand_{_side}", f"H:thumb_01_{_side}"))),
        f"{_s}Leg": ((f"{_s}Leg", f"{_s}Shin"), (f"H:thigh_{_side}", f"T:thigh_{_side}"), "FWD"),
        f"{_s}Shin": ((f"{_s}Shin", f"{_s}Foot"), (f"H:calf_{_side}", f"T:calf_{_side}"), "FWD"),
        f"{_s}Foot": ((f"{_s}Foot", f"{_s}ToeBase"), (f"H:foot_{_side}", f"T:foot_{_side}"), "UP"),
        f"{_s}ToeBase": ((f"{_s}ToeBase", f"{_s}ToeEnd"), (f"H:foot_{_side}", f"T:foot_{_side}"), "UP"),
    })

# W105 driven channels of the frozen W124/W126 Action (sidecar ``driven_channels``); everything else is rig rest.
DRIVEN_BASE = {"__object__", "spine_01", "thigh_l", "thigh_r", "calf_l", "calf_r", "upperarm_l", "lowerarm_l"}


class AdapterStop(RuntimeError):
    def __init__(self, code, detail=""):
        super().__init__(f"{code}: {detail}")
        self.code, self.detail = code, detail


# ------------------------------------------------------------------------------------------------ math helpers ---
def quat_xyzw_to_mat(q):
    x, y, z, w = (float(v) for v in q)
    n = x * x + y * y + z * z + w * w
    s = 2.0 / n
    return np.array([[1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
                     [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
                     [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)]])


def mat_to_quat_xyzw(m):
    m = np.asarray(m, float)
    t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def wxyz_to_mat(q):
    w, x, y, z = q
    return quat_xyzw_to_mat((x, y, z, w))


def mat_to_wxyz(m):
    x, y, z, w = mat_to_quat_xyzw(m)
    return np.array([w, x, y, z])


def rot_part(m4):
    """Rotation part of a 4x4 (column-normalised; the Armature Object carries a uniform 0.01 scale)."""
    r = np.asarray(m4, float)[:3, :3]
    r = r / np.linalg.norm(r, axis=0, keepdims=True)
    u, _, vt = np.linalg.svd(r)
    out = u @ vt
    if np.linalg.det(out) < 0:
        raise AdapterStop("IMPROPER_ROTATION", "negative determinant in a rig matrix")
    return out


def angle_between(r1, r2):
    c = (np.trace(np.asarray(r1).T @ np.asarray(r2)) - 1.0) / 2.0
    return float(np.arccos(np.clip(c, -1.0, 1.0)))


def frame_from(primary, secondary):
    e1 = np.asarray(primary, float) / np.linalg.norm(primary)
    s = np.asarray(secondary, float) - np.dot(secondary, e1) * e1
    if np.linalg.norm(s) < 1e-6:
        raise AdapterStop("DEGENERATE_ALIGNMENT", "secondary direction parallel to primary")
    e2 = s / np.linalg.norm(s)
    return np.column_stack([e1, e2, np.cross(e1, e2)])


def slerp_wxyz(q0, q1, t):
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9999995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    th = np.arccos(d)
    return (np.sin((1 - t) * th) * q0 + np.sin(t * th) * q1) / np.sin(th)


def canon_sign(q, ref):
    q = np.asarray(q, float)
    return -q if float(np.dot(q, ref)) < 0 else q


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ time contract ---
def episode(segment_first_source, context_last_source, target_first_source, n_tokens=9, protected_source=()):
    """Declared native-time episode.  Valid output sample k (0..V-1) is SOURCE frame context_last + k; sample 0 is the last
    context pose and sample V-1 the first target pose (MotionBricks trims 3 boundary frames per end and endpoint-corrects)."""
    valid = n_tokens * POSES_PER_TOKEN - 2 * BOUNDARY_FRAMES
    if target_first_source - context_last_source != valid - 1:
        raise AdapterStop("UNSUPPORTED_TIME_ALIGNMENT",
                          f"{n_tokens} tokens give {valid} valid samples; context {context_last_source} -> target "
                          f"{target_first_source} needs {target_first_source - context_last_source + 1}")
    to_action = lambda s: s - segment_first_source + 1                                       # noqa: E731
    interior = list(range(context_last_source + 1, target_first_source))
    overlap = sorted(set(interior) & set(protected_source))
    if overlap:
        raise AdapterStop("PROTECTED_OVERLAP", f"interior overlaps protected source frames {overlap[:5]}")
    ctx = list(range(context_last_source - 3, context_last_source + 1))
    tgt = list(range(target_first_source, target_first_source + 4))
    return {"n_tokens": n_tokens, "raw_frames": n_tokens * POSES_PER_TOKEN, "valid_samples": valid,
            "context_source": ctx, "target_source": tgt, "context_action": [to_action(s) for s in ctx],
            "target_action": [to_action(s) for s in tgt],
            "valid_source": list(range(context_last_source, target_first_source + 1)),
            "valid_action": [to_action(s) for s in range(context_last_source, target_first_source + 1)],
            "interior_source": [interior[0], interior[-1]], "interior_action": [to_action(interior[0]), to_action(interior[-1])],
            "allowed_token_counts": [n_tokens], "segment_first_source": segment_first_source}


# ------------------------------------------------------------------------------------------------ skeletons ---
class SomaSkeleton:
    def __init__(self, names, parents, ref_local):
        self.names = list(names)
        self.parents = [int(p) for p in parents]
        self.ref = np.asarray(ref_local, float)                       # [J, 7] xyz + qxyzw
        missing = [j for j in SOMA_MODEL_JOINTS if j not in self.names]
        if missing:
            raise AdapterStop("SOMA_SKELETON_MISSING_JOINTS", ",".join(missing))
        if self.parents[0] != -1 or self.names[0] != "Hips":
            raise AdapterStop("SOMA_SKELETON_ROOT", "root must be Hips at index 0")
        for i, p in enumerate(self.parents[1:], 1):
            if not 0 <= p < i:
                raise AdapterStop("SOMA_SKELETON_ORDER", f"joint {i} parent {p}")
        self.idx = {n: i for i, n in enumerate(self.names)}
        self.rest_rot, self.rest_pos = self.fk(self.ref)

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        return cls(d["names"], d["parents"], d["reference_local_xyz_qxyzw"])

    def fk(self, local):
        local = np.asarray(local, float)
        n = len(self.names)
        R, P = np.zeros((n, 3, 3)), np.zeros((n, 3))
        for i in range(n):
            r = quat_xyzw_to_mat(local[i, 3:])
            if self.parents[i] < 0:
                R[i], P[i] = r, local[i, :3]
            else:
                p = self.parents[i]
                R[i], P[i] = R[p] @ r, P[p] + R[p] @ local[i, :3]
        return R, P


class BaseFrame:
    """One evaluated BaseRig frame from the Blender dump (armature space -> Blender world)."""

    def __init__(self, rest, frame):
        self.M = np.asarray(frame["matrix_world"], float)
        self.Robj = rot_part(self.M)
        self.P = {b: np.asarray(m, float) for b, m in frame["pose_matrix"].items()}
        self.rest = rest
        self.frame = frame

    def world_rot(self, b):
        return self.Robj @ rot_part(self.P[b])

    def head(self, b):
        return (self.M @ np.r_[self.P[b][:3, 3], 1.0])[:3]

    def tail(self, b):
        length = np.linalg.norm(np.asarray(self.rest[b]["tail_local"]) - np.asarray(self.rest[b]["head_local"]))
        return (self.M @ np.r_[self.P[b][:3, 3] + self.P[b][:3, 1] * length, 1.0])[:3]

    def point(self, spec):
        if spec == "HIPMID":
            return 0.5 * (self.head("thigh_l") + self.head("thigh_r"))
        kind, b = spec.split(":", 1)
        return self.head(b) if kind == "H" else self.tail(b)


def rest_frame(rest, object_rest_wxyz, object_scale):
    """A BaseFrame at the rig rest pose (identity basis everywhere, the declared object rest rotation)."""
    R = wxyz_to_mat(object_rest_wxyz) * float(object_scale)
    M = np.eye(4)
    M[:3, :3] = R
    pm = {b: np.asarray(v["matrix_local"], float) for b, v in rest.items()}
    return BaseFrame(rest, {"matrix_world": M, "pose_matrix": pm})


# ------------------------------------------------------------------------------------------------ mapping ---
class Mapping:
    def __init__(self, soma: SomaSkeleton, base_rest: BaseFrame, A=A_BLENDER_TO_SOMA):
        self.soma, self.base_rest, self.A = soma, base_rest, np.asarray(A, float)
        if abs(np.linalg.det(self.A) - 1.0) > 1e-9:
            raise AdapterStop("AXIS_MAPPING_IMPROPER", f"det(A) = {np.linalg.det(self.A):.3f}: mirroring is refused")
        need = {b for k, b in JOINT_MAP.values() if b}
        for spec in ALIGN.values():
            for s in spec[1]:
                if s != "HIPMID":
                    need.add(s.split(":", 1)[1])
            if isinstance(spec[2], tuple):
                need |= {s.split(":", 1)[1] for s in spec[2][2]}
        missing = sorted(b for b in need | {"thigh_l", "thigh_r"} if b not in base_rest.P)
        if missing:
            raise AdapterStop("BASERIG_MISSING_BONES", ",".join(missing))
        self.Q, self.D, self.align_report = {}, {}, {}
        sp = soma.rest_pos
        for j in JOINT_MAP:
            (sa, sb), (ba, bb), sec = ALIGN[j]
            ps = sp[soma.idx[sb]] - sp[soma.idx[sa]]
            pb = self.A @ (base_rest.point(bb) - base_rest.point(ba))
            if sec == "FWD":
                ss, sbv = np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0])
            elif sec == "UP":
                ss, sbv = np.array([0.0, 1.0, 0.0]), np.array([0.0, 1.0, 0.0])
            else:
                (s0, s1), (b0, b1) = sec[1], sec[2]
                ss = sp[soma.idx[s1]] - sp[soma.idx[s0]]
                sbv = self.A @ (base_rest.point(b1) - base_rest.point(b0))
            Q = frame_from(pb, sbv) @ frame_from(ps, ss).T
            kind, b = JOINT_MAP[j]
            Rb0 = base_rest.Robj if kind == "OBJECT" else base_rest.world_rot(b)
            self.Q[j] = Q
            self.D[j] = Rb0.T @ self.A.T @ Q @ soma.rest_rot[soma.idx[j]]
            self.align_report[j] = {"rest_alignment_deg": float(np.degrees(angle_between(np.eye(3), Q))),
                                    "primary_residual_deg": float(np.degrees(np.arccos(np.clip(
                                        np.dot(Q @ ps / np.linalg.norm(ps), pb / np.linalg.norm(pb)), -1, 1))))}
        self.hip_scale = float(sp[0][1] / (self.A @ base_rest.point("HIPMID"))[1])

    # ---- BaseRig -> SOMA ----
    def soma_world_rot(self, bf: BaseFrame, j):
        kind, b = JOINT_MAP[j]
        Rb = bf.Robj if kind == "OBJECT" else bf.world_rot(b)
        return self.A @ Rb @ self.D[j]

    def to_soma_local(self, bf: BaseFrame):
        """77-joint SOMA local transforms [J, 7] (xyz, qxyzw) for one BaseRig frame."""
        s = self.soma
        n = len(s.names)
        Rw = np.zeros((n, 3, 3))
        out = np.array(s.ref, float, copy=True)
        for i, name in enumerate(s.names):
            p = s.parents[i]
            if name in JOINT_MAP:
                Rw[i] = self.soma_world_rot(bf, name)
            else:
                Rw[i] = (Rw[p] if p >= 0 else np.eye(3)) @ quat_xyzw_to_mat(s.ref[i, 3:])
            local = Rw[i] if p < 0 else Rw[p].T @ Rw[i]
            out[i, 3:] = mat_to_quat_xyzw(local)
        out[0, :3] = (self.A @ bf.point("HIPMID")) * self.hip_scale
        return out

    # ---- SOMA -> BaseRig (left-arm channels only) ----
    def base_basis_from_soma(self, soma_local, bf_parent_rest_rel=None):
        """BaseRig pose-bone basis (wxyz) for EDIT_CHANNELS from one SOMA local pose [J, 7]."""
        s = self.soma
        Rw, _ = s.fk(soma_local)
        out = {}
        for b, j in (("upperarm_l", "LeftArm"), ("lowerarm_l", "LeftForeArm")):
            pj = s.names[s.parents[s.idx[j]]]
            Ls = Rw[s.idx[pj]].T @ Rw[s.idx[j]]
            rel_world = self.D[pj] @ Ls @ self.D[j].T                 # = R_base(parent)^T R_base(b)
            bp = self.base_rest.rest[b]["parent"]
            if JOINT_MAP[pj][1] != bp:
                raise AdapterStop("PARENT_CORRESPONDENCE", f"{j} parent {pj} vs BaseRig {b} parent {bp}")
            Bp = rot_part(self.base_rest.rest[bp]["matrix_local"])
            Bb = rot_part(self.base_rest.rest[b]["matrix_local"])
            basis = (Bp.T @ Bb).T @ rel_world
            out[b] = mat_to_wxyz(basis)
        return out

    def basis_from_base_frame(self, bf: BaseFrame):
        """Independent BaseRig-only basis of the edit channels from evaluated pose matrices (round-trip reference)."""
        out = {}
        for b in EDIT_CHANNELS:
            bp = self.base_rest.rest[b]["parent"]
            Bp = rot_part(self.base_rest.rest[bp]["matrix_local"])
            Bb = rot_part(self.base_rest.rest[b]["matrix_local"])
            out[b] = mat_to_wxyz((Bp.T @ Bb).T @ rot_part(bf.P[bp]).T @ rot_part(bf.P[b]))
        return out

    def report(self):
        return {"A_blender_to_soma": self.A.tolist(), "det_A": float(np.linalg.det(self.A)), "hip_scale": self.hip_scale,
                "joint_map": {j: list(v) for j, v in JOINT_MAP.items()}, "synthetic_model_joints": list(SYNTHETIC_MODEL_JOINTS),
                "alignment": self.align_report, "coverage": coverage()}


def coverage():
    rows = {}
    for j in SOMA_MODEL_JOINTS:
        if j not in JOINT_MAP:
            rows[j] = "SYNTHETIC_SOMA_REFERENCE (no BaseRig counterpart; SOMA rest relative to its parent)"
            continue
        kind, b = JOINT_MAP[j]
        if kind == "OBJECT":
            rows[j] = ("OBJECT_PROXY: Armature Object rotation = S0 W93 torso-frame hypothesis (GT-free); hips position "
                       "= thigh-head midpoint, NO root travel observed (SYNTHETIC zero travel)")
        elif kind == "PROXY":
            rows[j] = f"PROXY of {b} ({'driven S0 hypothesis' if b in DRIVEN_BASE else 'rig REST, undriven'})"
        elif b in DRIVEN_BASE:
            rows[j] = f"DRIVEN from {b}: frozen W102 M1+S0 3D hypothesis via W105 (GT-free; not observed truth)"
        else:
            rows[j] = f"RIG_REST {b}: undriven in the frozen Action (UNKNOWN in the video; rest is SYNTHETIC)"
    return rows


def no_mirroring_check(mapping: Mapping, soma_local):
    """Engine heading convention (forward = up x (right_hip - left_hip)) must agree with the toe direction, and the SOMA
    left hand must stay on the body's left.  Returns a dict; ``ok`` False means the axis mapping is refused."""
    s = mapping.soma
    _, P = s.fk(soma_local)
    i = s.idx
    across = P[i["RightLeg"]] - P[i["LeftLeg"]]
    across[1] = 0
    fwd = np.cross([0.0, 1.0, 0.0], across)
    fwd /= np.linalg.norm(fwd)
    toe = 0.5 * ((P[i["LeftToeBase"]] - P[i["LeftFoot"]]) + (P[i["RightToeBase"]] - P[i["RightFoot"]]))
    toe[1] = 0
    toe /= np.linalg.norm(toe)
    left_axis = -across / np.linalg.norm(across)
    lh = float(np.dot(P[i["LeftHand"]] - P[i["Hips"]], left_axis))
    rh = float(np.dot(P[i["RightHand"]] - P[i["Hips"]], left_axis))
    ok = bool(np.dot(fwd, toe) > 0.5 and lh > rh)
    return {"ok": ok, "heading_dot_toe": float(np.dot(fwd, toe)), "left_hand_lateral_m": lh, "right_hand_lateral_m": rh}


# ------------------------------------------------------------------------------------------------ diagnostics ---
def wrist_series(readback, lo, hi):
    fr = {f["blender_frame"]: f for f in readback["frames"]}
    return np.array([fr[f]["world"]["hand_l"]["head"] for f in range(lo, hi + 1)])


def arm_points(readback, lo, hi):
    fr = {f["blender_frame"]: f for f in readback["frames"]}
    sh = np.array([fr[f]["world"]["upperarm_l"]["head"] for f in range(lo, hi + 1)])
    el = np.array([fr[f]["world"]["lowerarm_l"]["head"] for f in range(lo, hi + 1)])
    wr = np.array([fr[f]["world"]["hand_l"]["head"] for f in range(lo, hi + 1)])
    return sh, el, wr


def flex_deg(sh, el, wr):
    u, v = el - sh, wr - el
    c = np.sum(u * v, 1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1))
    return 180.0 - np.degrees(np.arccos(np.clip(c, -1, 1)))         # 180 = straight arm, small = folded


def quality(readback, readback_a, interior, window, hinge_ref=None):
    """Diagnostics of one branch over ``window`` (Action frames, inclusive) around ``interior``."""
    lo, hi = window
    i0, i1 = interior
    sh, el, wr = arm_points(readback, lo, hi)
    _, _, wa = arm_points(readback_a, lo, hi)
    fr = np.arange(lo, hi + 1)
    step = np.linalg.norm(np.diff(wr, axis=0), axis=1)              # step[k] = |w[k+1]-w[k]| (frame fr[k] -> fr[k+1])
    vel = np.diff(wr, axis=0) * FPS
    acc = np.diff(vel, axis=0) * FPS
    jerk = np.diff(acc, axis=0) * FPS
    seg = (fr[:-1] >= i0 - 1) & (fr[:-1] <= i1)                        # steps 171->172 .. 199->200
    d = np.diff(wr, axis=0)
    flicks = []
    for k in range(len(d) - 1):
        if seg[k] and seg[k + 1] and min(step[k], step[k + 1]) > 0.020:
            cosv = float(np.dot(d[k], d[k + 1]) / (step[k] * step[k + 1]))
            if cosv < 0.0:
                flicks.append({"frames": [int(fr[k]), int(fr[k + 1]), int(fr[k + 2])], "cos": cosv,
                               "steps_mm": [float(step[k] * 1e3), float(step[k + 1] * 1e3)]})
    flex = flex_deg(sh, el, wr)
    n = np.cross(el - sh, wr - el)
    nn = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    inside = (fr >= i0) & (fr <= i1)
    bend_flips = [int(fr[k + 1]) for k in range(len(fr) - 1)
                  if (inside[k] or inside[k + 1]) and flex[k] < 170 and flex[k + 1] < 170 and np.dot(nn[k], nn[k + 1]) < 0]
    hinge_wrong = []
    if hinge_ref is not None:
        hinge_wrong = [int(fr[k]) for k in range(len(fr)) if inside[k] and flex[k] < 170 and np.dot(nn[k], hinge_ref) < 0]
    k0 = int(np.where(fr == i0 - 1)[0][0])                          # index of frame 171
    k1 = int(np.where(fr == i1 + 1)[0][0])                          # index of frame 200
    c0 = {"entry_step_mm": float(step[k0] * 1e3), "exit_step_mm": float(step[k1 - 1] * 1e3)}
    c1 = {"entry_dv_mps": float(np.linalg.norm((wr[k0 + 1] - wr[k0]) - (wr[k0] - wr[k0 - 1])) * FPS),
          "exit_dv_mps": float(np.linalg.norm((wr[k1 + 1] - wr[k1]) - (wr[k1] - wr[k1 - 1])) * FPS)}
    dev = np.linalg.norm(wr - wa, axis=1)[inside]
    chord = wr[k1] - wr[k0]
    cl = float(np.linalg.norm(chord))

    def progress(w):
        return np.dot(w - w[k0], chord) / max(cl * cl, 1e-12)
    prog = progress(wr)[inside]
    prog_a = (np.dot(wa - wa[k0], wa[k1] - wa[k0]) / max(np.linalg.norm(wa[k1] - wa[k0]) ** 2, 1e-12))[inside]
    path = float(np.sum(step[k0:k1]))
    off_chord = np.linalg.norm(np.cross(wr[k0:k1 + 1] - wr[k0], chord / max(cl, 1e-12)), axis=1)
    jseg = (fr[3:] >= i0) & (fr[3:] <= i1 + 1)
    aseg = (fr[2:] >= i0) & (fr[2:] <= i1 + 1)
    return {"window_action": [lo, hi], "interior_action": [i0, i1],
            "peak_step_mm": float(step[seg].max() * 1e3), "peak_step_frame": int(fr[:-1][seg][int(np.argmax(step[seg]))] + 1),
            "peak_speed_mps": float(np.linalg.norm(vel[seg], axis=1).max()),
            "peak_acc_mps2": float(np.linalg.norm(acc[aseg[:len(acc)]], axis=1).max()),
            "peak_jerk_mps3": float(np.linalg.norm(jerk[jseg[:len(jerk)]], axis=1).max()),
            "rms_jerk_mps3": float(np.sqrt(np.mean(np.sum(jerk[jseg[:len(jerk)]] ** 2, 1)))),
            "reversal_flicks": flicks, "n_reversal_flicks": len(flicks),
            "boundary_c0": c0, "boundary_c1": c1,
            "flex_min_deg": float(flex[inside].min()), "flex_max_deg": float(flex[inside].max()),
            "near_straight_frames": [int(f) for f in fr[inside][flex[inside] > 175.0]],
            "bend_direction_flips": bend_flips, "hinge_opposite_frames": hinge_wrong,
            "upperarm_len_range_mm": float(np.ptp(np.linalg.norm(el - sh, axis=1)) * 1e3),
            "lowerarm_len_range_mm": float(np.ptp(np.linalg.norm(wr - el, axis=1)) * 1e3),
            "wrist_dev_from_A_mean_mm": float(dev.mean() * 1e3), "wrist_dev_from_A_max_mm": float(dev.max() * 1e3),
            "progress_vs_A_mean_abs": float(np.mean(np.abs(prog - prog_a))),
            "path_length_mm": path * 1e3, "chord_mm": cl * 1e3, "path_over_chord": path / max(cl, 1e-12),
            "max_off_chord_mm": float(off_chord.max() * 1e3),
            "per_frame": {"frames": fr.tolist(), "wrist": wr.tolist(), "step_mm": (step * 1e3).tolist(), "flex_deg": flex.tolist()}}


def armlocal_wrist(dump_report, lo, hi, scale=0.01):
    """Left wrist (hand_l head) in the clavicle_l bone frame, metres, from Blender-evaluated pose matrices (dump).
    Only the edited channels (upperarm_l, lowerarm_l) move it; torso / object motion is factored out."""
    fr = {f["blender_frame"]: f for f in dump_report["frames"]}
    out = []
    for f in range(lo, hi + 1):
        pc = np.asarray(fr[f]["pose_matrix"]["clavicle_l"], float)
        ph = np.asarray(fr[f]["pose_matrix"]["hand_l"], float)
        out.append((np.linalg.inv(pc) @ np.r_[ph[:3, 3], 1.0])[:3] * scale)
    return np.array(out)


def armlocal_quality(wl, interior, window):
    """Arm-local (edit-attributable) wrist kinematics: steps, reversal flicks and boundary C0/C1 (same definitions as
    ``quality`` but in the clavicle_l frame)."""
    lo, hi = window
    i0, i1 = interior
    fr = np.arange(lo, hi + 1)
    d = np.diff(wl, axis=0)
    step = np.linalg.norm(d, axis=1)
    seg = (fr[:-1] >= i0 - 1) & (fr[:-1] <= i1)
    flicks = []
    for k in range(len(d) - 1):
        if seg[k] and seg[k + 1] and min(step[k], step[k + 1]) > 0.020 and np.dot(d[k], d[k + 1]) < 0:
            flicks.append({"frames": [int(fr[k]), int(fr[k + 1]), int(fr[k + 2])],
                           "cos": float(np.dot(d[k], d[k + 1]) / (step[k] * step[k + 1])),
                           "steps_mm": [float(step[k] * 1e3), float(step[k + 1] * 1e3)]})
    k0 = int(np.where(fr == i0 - 1)[0][0])
    k1 = int(np.where(fr == i1 + 1)[0][0])
    acc = np.diff(d, axis=0) * FPS * FPS
    jerk = np.diff(acc, axis=0) * FPS
    aseg = (fr[2:] >= i0) & (fr[2:] <= i1 + 1)
    jseg = (fr[3:] >= i0) & (fr[3:] <= i1 + 1)
    return {"space": "clavicle_l frame (edit-attributable)", "peak_step_mm": float(step[seg].max() * 1e3),
            "peak_step_frame": int(fr[:-1][seg][int(np.argmax(step[seg]))] + 1),
            "peak_acc_mps2": float(np.linalg.norm(acc[aseg], axis=1).max()),
            "peak_jerk_mps3": float(np.linalg.norm(jerk[jseg], axis=1).max()),
            "rms_jerk_mps3": float(np.sqrt(np.mean(np.sum(jerk[jseg] ** 2, 1)))),
            "n_reversal_flicks": len(flicks), "reversal_flicks": flicks,
            "boundary_c0": {"entry_step_mm": float(step[k0] * 1e3), "exit_step_mm": float(step[k1 - 1] * 1e3)},
            "boundary_c1": {"entry_dv_mps": float(np.linalg.norm(d[k0] - d[k0 - 1]) * FPS),
                            "exit_dv_mps": float(np.linalg.norm(d[k1] - d[k1 - 1]) * FPS)},
            "path_length_mm": float(np.sum(step[k0:k1]) * 1e3),
            "per_frame": {"frames": fr.tolist(), "wrist_armlocal": wl.tolist(), "step_mm": (step * 1e3).tolist()}}


def hinge_reference(readback_a):
    """Dominant elbow bend-plane normal of the untouched source-derived Action (all frames, flexed only)."""
    n = len(readback_a["frames"])
    sh, el, wr = arm_points(readback_a, 1, n)
    flex = flex_deg(sh, el, wr)
    nv = np.cross(el - sh, wr - el)
    nv = nv[flex < 170]
    nv = nv / np.linalg.norm(nv, axis=1, keepdims=True)
    m = nv.mean(0)
    return m / np.linalg.norm(m)


def preservation(rb_variant, rb_base, interior, edit_bones=("upperarm_l", "lowerarm_l", "hand_l")):
    """Fresh-readback footprint: which (frame, bone) world points differ from the untouched baseline."""
    i0, i1 = interior
    out = {"outside_changed": [], "inside_changed_frames": set(), "nonarm_changed": [], "sub_outside_changed": []}
    for fv, fb in zip(rb_variant["frames"], rb_base["frames"]):
        f = fv["blender_frame"]
        assert f == fb["blender_frame"]
        if fv["matrix_world"] != fb["matrix_world"]:
            out["nonarm_changed"].append([f, "__object__"])
        for b in fb["world"]:
            same = fv["world"][b] == fb["world"][b]
            if same:
                continue
            if b not in edit_bones:
                out["nonarm_changed"].append([f, b])
            elif not i0 <= f <= i1:
                out["outside_changed"].append([f, b])
            else:
                out["inside_changed_frames"].add(f)
    for sv, sb in zip(rb_variant["subframes"], rb_base["subframes"]):
        f = sv["blender_frame"]
        if sv["world"] != sb["world"] or sv["matrix_world"] != sb["matrix_world"]:
            if not i0 <= f <= i1:
                out["sub_outside_changed"].append([f, sv["subframe"]])
    out["inside_changed_frames"] = sorted(out["inside_changed_frames"])
    return out


def changed_keys(stored_v, stored_b):
    diff = []
    for ch in sorted(set(stored_v) | set(stored_b)):
        for comp in sorted(set(stored_v.get(ch, {})) | set(stored_b.get(ch, {}))):
            a, b = stored_v.get(ch, {}).get(comp, {}), stored_b.get(ch, {}).get(comp, {})
            for f in sorted(set(a) | set(b), key=int):
                if a.get(f) != b.get(f):
                    diff.append([ch, int(comp), int(f)])
    return diff
