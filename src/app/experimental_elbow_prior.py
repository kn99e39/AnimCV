"""EXPERIMENTAL (opt-in, Batch 135): source-constrained elbow-swivel prior on the frozen W129 left-arm candidates.

Scope (frozen by the W135 declaration; nothing here is a Production path):
  * read Blender ``dump`` reports (``blender/experimental_motionbricks_stage.py dump``) of the frozen W129 A (baseline),
    B (endpoint slerp) and C (MotionBricks projection) ``.blend`` files,
  * extract from C ONLY its per-frame elbow SWIVEL angle (the one redundant DOF of a two-bone arm once the shoulder and
    wrist positions are fixed), measured in the ``clavicle_l`` frame against a declared reference axis,
  * re-pose B's left arm by rotating it rigidly about B's own shoulder->wrist axis so that its swivel equals the target
    swivel (D: C's swivel; E: B's own swivel -> neutral control).  The wrist position, both bone lengths, the elbow
    flexion angle and the ``lowerarm_l`` key (elbow-local relation) are B's by construction; only ``upperarm_l`` changes.

Representation (closed form, no optimisation, no smoothing, no regularisation):
  Blender pose: P_b = P_parent . (Rest_parent^-1 . Rest_b) . Basis_b   (armature space, rotation-only channels).
  Arm-local frame = the evaluated ``clavicle_l`` pose matrix K (parent of ``upperarm_l``; unkeyed, follows spine/object).
  For shoulder S (upperarm_l head), elbow E (lowerarm_l head), wrist W (hand_l head) in K-local coordinates:
      a = (W - S)/|W - S|,  u = (E - S) - ((E - S).a) a,  r_p = r - (r.a) a,
      swivel psi = atan2(a . (r_p x u), r_p . u)        (right-handed about a; r = declared lateral reference)
  New upperarm basis:  Basis' = M^T . G . M . Basis_B  with  M = Rot(K) Rot(Rest_clav)^T Rot(Rest_ua)  and G the
  rotation by wrap(psi_target - psi_B) about B's armature-space axis through S and W.  G fixes S and W exactly.

Only numpy (+ stdlib) is imported.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

SCHEMA = "animcv_exp_w135_elbow_prior_v1"
FPS = 30.0
SIDE = "l"
UPPER, LOWER, HAND, PARENT = "upperarm_l", "lowerarm_l", "hand_l", "clavicle_l"
EDIT_CHANNELS = (UPPER, LOWER)
INTERIOR = (172, 199)                       # Action frames (SOURCE 250-277)
CONTEXT_LAST, TARGET_FIRST = 171, 200
PIN_ACTION = tuple(range(132, 141))         # W126V WORLD_POSITION_PIN, Action 132-140 (SOURCE 210-218)
REFERENCE_ARMATURE = (1.0, 0.0, 0.0)        # rig lateral (+X = left side) at REST, expressed once in the clavicle_l frame
REFERENCE_CANDIDATES = {"anterior": (0.0, -1.0, 0.0), "lateral": (1.0, 0.0, 0.0), "superior": (0.0, 0.0, 1.0)}
MIN_REF_MARGIN_SIN = float(np.sin(np.radians(15.0)))     # axis within 15 deg of +/- reference -> refuse
MIN_ELBOW_OFFSET_SIN = float(np.sin(np.radians(10.0)))   # near-straight arm (swivel undefined) -> refuse
MAX_SWIVEL_STEP_DEG = 90.0                  # consecutive-frame swivel change -> FLIP refusal
REACH_TOL = 1e-6                            # relative tolerance on |S-W| <= L1+L2, >= |L1-L2|


class PriorStop(RuntimeError):
    def __init__(self, code, detail=""):
        super().__init__(f"{code}: {detail}")
        self.code, self.detail = code, detail


# ------------------------------------------------------------------------------------------------ math ---
def rot(m4):
    """Rotation part of a rig matrix; refuses an improper (mirrored) matrix."""
    r = np.asarray(m4, float)[:3, :3]
    r = r / np.linalg.norm(r, axis=0, keepdims=True)
    u, _, vt = np.linalg.svd(r)
    out = u @ vt
    if np.linalg.det(np.asarray(m4, float)[:3, :3]) <= 0 or np.linalg.det(out) < 0:
        raise PriorStop("IMPROPER_ROTATION", "mirrored / non-proper rig matrix")
    return out


def wxyz_to_mat(q):
    w, x, y, z = (float(v) for v in q)
    n = w * w + x * x + y * y + z * z
    s = 2.0 / n
    return np.array([[1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
                     [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
                     [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)]])


def mat_to_wxyz(m):
    m = np.asarray(m, float)
    t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def axis_angle(axis, ang):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def wrap(x):
    return float((x + np.pi) % (2 * np.pi) - np.pi)


def canon_sign(q, ref):
    q = np.asarray(q, float)
    return -q if float(np.dot(q, ref)) < 0 else q


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# ------------------------------------------------------------------------------------------------ rig geometry ---
class ArmFrame:
    """One evaluated frame of a dump: arm points in the clavicle_l pose frame (armature units)."""

    def __init__(self, dump_report, frame):
        fr = {f["blender_frame"]: f for f in dump_report["frames"]}
        if frame not in fr:
            raise PriorStop("FRAME_MISSING", f"frame {frame} not in dump")
        self.f = fr[frame]
        self.rest = dump_report["rest"]
        self.P = {b: np.asarray(self.f["pose_matrix"][b], float) for b in (PARENT, UPPER, LOWER, HAND)}
        self.K = self.P[PARENT]
        self.Kinv = np.linalg.inv(self.K)
        rot(self.K)                                                      # proper-rotation check (mirror refusal)

    def local(self, b):
        return (self.Kinv @ np.r_[self.P[b][:3, 3], 1.0])[:3]

    def SEW(self):
        return self.local(UPPER), self.local(LOWER), self.local(HAND)


def bone_lengths(rest):
    l1 = float(np.linalg.norm(np.subtract(rest[LOWER]["head_local"], rest[UPPER]["head_local"])))
    l2 = float(np.linalg.norm(np.subtract(rest[HAND]["head_local"], rest[LOWER]["head_local"])))
    return l1, l2


def reference_local(rest, ref=REFERENCE_ARMATURE):
    """The declared swivel reference (armature-space direction at REST) in clavicle_l bone coordinates (constant)."""
    rc = rot(np.asarray(rest[PARENT]["matrix_local"], float))
    return rc.T @ np.asarray(ref, float)


def select_reference(dumps, frames):
    """Outcome-blind declared rule: the anatomical candidate axis with the largest minimum axis-reference margin over
    the given branches/frames (uses only shoulder-wrist axis directions, never swivel values)."""
    rows = {}
    for name, v in REFERENCE_CANDIDATES.items():
        r = reference_local(dumps[0]["rest"], v)
        rows[name] = min(swivel(*ArmFrame(d, f).SEW(), r)[1] for d in dumps for f in frames)
    best = max(rows, key=rows.get)
    return best, rows


def swivel(S, E, W, r):
    """Swivel angle (rad) of the elbow about the shoulder->wrist axis against reference r, plus validity margins."""
    a = (W - S) / np.linalg.norm(W - S)
    e = E - S
    u = e - np.dot(e, a) * a
    rp = r - np.dot(r, a) * a
    margin_ref = float(np.linalg.norm(rp) / np.linalg.norm(r))
    margin_elbow = float(np.linalg.norm(u) / np.linalg.norm(e))
    psi = float(np.arctan2(np.dot(a, np.cross(rp, u)), np.dot(rp, u)))
    return psi, margin_ref, margin_elbow


def check_reachable(S, W, l1, l2):
    d = float(np.linalg.norm(W - S))
    lo, hi = abs(l1 - l2), l1 + l2
    if not (lo * (1 - REACH_TOL) <= d <= hi * (1 + REACH_TOL)):
        raise PriorStop("WRIST_UNREACHABLE", f"|S-W|={d:.4f} outside [{lo:.4f}, {hi:.4f}]")
    return d


def check_dump(dump_report, n_frames=240):
    sc = dump_report["scene"]
    eff = sc["fps"] / sc["fps_base"]
    if eff != FPS:
        raise PriorStop("NOT_30_FPS", f"scene effective fps {eff}")
    if (sc["frame_start"], sc["frame_end"]) != (1, n_frames):
        raise PriorStop("FRAME_RANGE", f"{sc['frame_start']}..{sc['frame_end']}")
    if len(dump_report["actions_in_file"]) != 1:
        raise PriorStop("ACTION_COUNT", str(dump_report["actions_in_file"]))


# ------------------------------------------------------------------------------------------------ prior ---
def extract_swivel_series(dump_report, frames, expected_sha=None):
    """Per-frame swivel of a branch (only S/E/W of the left arm in its own clavicle_l frame are read)."""
    check_dump(dump_report)
    if expected_sha is not None and dump_report["blend_sha256"] != expected_sha:
        raise PriorStop("SOURCE_IDENTITY", f"dump of {dump_report['blend_sha256'][:12]} != {expected_sha[:12]}")
    r = reference_local(dump_report["rest"])
    l1, l2 = bone_lengths(dump_report["rest"])
    out = {}
    for f in frames:
        S, E, W = ArmFrame(dump_report, f).SEW()
        psi, mr, me = swivel(S, E, W, r)
        out[int(f)] = {"psi_rad": psi, "margin_ref_sin": mr, "margin_elbow_sin": me,
                       "upper_len": float(np.linalg.norm(E - S)), "lower_len": float(np.linalg.norm(W - E)),
                       "rest_upper_len": l1, "rest_lower_len": l2}
    return out


def build_prior(dump_c, c_sha):
    """The ONLY learned information handed to the solver: C's swivel (rad) on the interior Action frames."""
    s = extract_swivel_series(dump_c, range(INTERIOR[0], INTERIOR[1] + 1), expected_sha=c_sha)
    for f, v in s.items():
        if v["margin_ref_sin"] < MIN_REF_MARGIN_SIN or v["margin_elbow_sin"] < MIN_ELBOW_OFFSET_SIN:
            raise PriorStop("REPRESENTATION_SINGULAR", f"frame {f}: C swivel undefined (margins {v['margin_ref_sin']:.3f}, {v['margin_elbow_sin']:.3f})")
    vals = {str(f): s[f]["psi_rad"] for f in sorted(s)}
    prior = {"schema": SCHEMA + "_prior", "kind": "LEARNED_ELBOW_SWIVEL", "side": SIDE,
             "source_branch": "W129 C (MotionBricks-SOMA-RP-v1 primary sample R1, projected)",
             "source_blend_sha256": c_sha, "source_bones": [UPPER, LOWER, HAND, PARENT],
             "frames": [INTERIOR[0], INTERIOR[1]], "reference": "lateral (armature +X at rest) in clavicle_l frame",
             "psi_rad": vals,
             "excluded": "C wrist position/path, root, object, spine, timing (frame ids are identity-mapped), right arm"}
    prior["content_sha256"] = sha256_json(vals)
    return prior


def neutral_prior(dump_b, b_sha):
    """E: B's own swivel (no learned information)."""
    s = extract_swivel_series(dump_b, range(INTERIOR[0], INTERIOR[1] + 1), expected_sha=b_sha)
    vals = {str(f): s[f]["psi_rad"] for f in sorted(s)}
    p = {"schema": SCHEMA + "_prior", "kind": "NEUTRAL_B_SWIVEL", "side": SIDE, "source_blend_sha256": b_sha,
         "source_branch": "W129 B (endpoint slerp, no model)", "frames": [INTERIOR[0], INTERIOR[1]], "psi_rad": vals}
    p["content_sha256"] = sha256_json(vals)
    return p


def verify_prior(prior, source_dump, expected_kind):
    """Fail closed unless the prior is exactly what the bound source file yields (catches corrupted, mirrored,
    time-shifted or substituted priors)."""
    if prior.get("kind") != expected_kind:
        raise PriorStop("PRIOR_KIND", f"{prior.get('kind')} != {expected_kind}")
    if prior.get("side") != SIDE:
        raise PriorStop("PRIOR_SIDE_MISMATCH", str(prior.get("side")))
    want = [str(f) for f in range(INTERIOR[0], INTERIOR[1] + 1)]
    if sorted(prior["psi_rad"], key=int) != want:
        raise PriorStop("PRIOR_FRAME_IDENTITY", f"frames {sorted(prior['psi_rad'], key=int)[:3]}... != interior")
    if sha256_json(prior["psi_rad"]) != prior.get("content_sha256"):
        raise PriorStop("PRIOR_CONTENT_HASH", "prior values do not match their own content hash")
    ref = (build_prior if expected_kind == "LEARNED_ELBOW_SWIVEL" else neutral_prior)(source_dump, prior["source_blend_sha256"])
    if ref["content_sha256"] != prior["content_sha256"]:
        raise PriorStop("PRIOR_PROVENANCE_MISMATCH", "prior is not the swivel of the bound source file")
    return True


# ------------------------------------------------------------------------------------------------ solver ---
def check_domain(frames, channels=EDIT_CHANNELS):
    frames = [int(f) for f in frames]
    if set(frames) & set(PIN_ACTION):
        raise PriorStop("PIN_OVERLAP", "request overlaps the W126V pin")
    if sorted(frames) != list(range(INTERIOR[0], INTERIOR[1] + 1)):
        raise PriorStop("DOMAIN_FRAMES", f"requested frames {min(frames)}..{max(frames)} != {INTERIOR}")
    bad = sorted(set(channels) - set(EDIT_CHANNELS))
    if bad:
        raise PriorStop("DOMAIN_CHANNEL", str(bad))


def solve(dump_b, dump_a, prior_psi, wrist_targets=None, frames=None):
    """Re-pose B's left arm on the interior so that its swivel equals ``prior_psi`` (dict frame->rad).

    Returns per-frame upperarm_l / lowerarm_l basis quaternions (wxyz, sign-canonicalised to the A key) and the
    numpy-FK diagnostics.  ``wrist_targets`` (control use only) replaces B's wrist as the target to exercise the
    reachability refusal; the declared D/E runs never pass it.
    """
    frames = list(range(INTERIOR[0], INTERIOR[1] + 1)) if frames is None else [int(f) for f in frames]
    check_domain(frames)
    check_dump(dump_b)
    check_dump(dump_a)
    rest = dump_b["rest"]
    l1, l2 = bone_lengths(rest)
    r = reference_local(rest)
    Rc_rest, Ru_rest, Rl_rest, Rh_rest = (rot(np.asarray(rest[b]["matrix_local"], float)) for b in (PARENT, UPPER, LOWER, HAND))
    st_b, st_a = dump_b["stored"], dump_a["stored"]

    def key(st, ch, f):
        return np.array([st[ch][str(c)][str(f)] for c in range(4)])
    out, diag = {UPPER: {}, LOWER: {}}, {}
    prev = None
    a171 = ArmFrame(dump_a, CONTEXT_LAST)
    psi_prev = swivel(*a171.SEW(), r)[0]
    for f in frames:
        if str(f) not in prior_psi:
            raise PriorStop("PRIOR_FRAME_IDENTITY", f"no prior at frame {f}")
        tgt = float(prior_psi[str(f)])
        af = ArmFrame(dump_b, f)
        S, E, W = af.SEW()
        if wrist_targets is not None:
            W = np.asarray(wrist_targets[f], float)
        check_reachable(S, W, l1, l2)
        psi_b, mr, me = swivel(S, E, W, r)
        if mr < MIN_REF_MARGIN_SIN:
            raise PriorStop("REFERENCE_SINGULAR", f"frame {f}: arm axis within 15 deg of the swivel reference")
        if me < MIN_ELBOW_OFFSET_SIN:
            raise PriorStop("ELBOW_STRAIGHT_SINGULAR", f"frame {f}: swivel undefined (arm near straight)")
        if abs(np.degrees(wrap(tgt - psi_prev))) > MAX_SWIVEL_STEP_DEG:
            raise PriorStop("ELBOW_FLIP", f"frame {f}: swivel step {np.degrees(wrap(tgt - psi_prev)):.1f} deg")
        psi_prev = tgt
        delta = wrap(tgt - psi_b)
        Kr = rot(af.K)
        axis_arm = Kr @ ((W - S) / np.linalg.norm(W - S))
        G = axis_angle(axis_arm, delta)
        M = Kr @ Rc_rest.T @ Ru_rest
        basis_b = wxyz_to_mat(key(st_b, UPPER, f))
        basis_new = M.T @ G @ M @ basis_b
        qu = canon_sign(mat_to_wxyz(basis_new), key(st_a, UPPER, f))
        ql = canon_sign(key(st_b, LOWER, f), key(st_a, LOWER, f))         # B's elbow-local relation, unchanged
        # numpy FK verification (armature space)
        Pu = af.K @ _m4(Rc_rest.T @ Ru_rest, _rel(rest, PARENT, UPPER)) @ _m4(wxyz_to_mat(qu), np.zeros(3))
        Pl = Pu @ _m4(Ru_rest.T @ Rl_rest, _rel(rest, UPPER, LOWER)) @ _m4(wxyz_to_mat(ql), np.zeros(3))
        Ph = Pl @ _m4(Rl_rest.T @ Rh_rest, _rel(rest, LOWER, HAND)) @ _m4(wxyz_to_mat(key(st_b, HAND, f)) if HAND in st_b else np.eye(3), np.zeros(3))
        S2, E2, W2 = ((af.Kinv @ np.r_[P[:3, 3], 1.0])[:3] for P in (Pu, Pl, Ph))
        psi_new = swivel(S2, E2, W2, r)[0]
        d = {"psi_b_rad": psi_b, "psi_target_rad": tgt, "psi_solved_rad": psi_new, "delta_rad": delta,
             "margin_ref_sin": mr, "margin_elbow_sin": me, "wrist_err": float(np.linalg.norm(W2 - W)),
             "shoulder_err": float(np.linalg.norm(S2 - S)), "upper_len_err": float(abs(np.linalg.norm(E2 - S2) - l1)),
             "lower_len_err": float(abs(np.linalg.norm(W2 - E2) - l2)), "swivel_err_rad": abs(wrap(psi_new - tgt)),
             "elbow_shift": float(np.linalg.norm(E2 - E))}
        for k_, tol in (("wrist_err", 1e-3), ("shoulder_err", 1e-3), ("upper_len_err", 1e-3), ("lower_len_err", 1e-3), ("swivel_err_rad", 1e-6)):
            if d[k_] > tol:
                raise PriorStop("SOLVER_CONSTRAINT_VIOLATION", f"frame {f}: {k_}={d[k_]:.3g}")
        out[UPPER][str(f)] = qu.tolist()
        out[LOWER][str(f)] = ql.tolist()
        diag[str(f)] = d
        prev = f
    # exit continuity check against the frozen frame 200 (A == B == C there)
    psi200 = swivel(*ArmFrame(dump_a, TARGET_FIRST).SEW(), r)[0]
    if abs(np.degrees(wrap(psi200 - psi_prev))) > MAX_SWIVEL_STEP_DEG:
        raise PriorStop("ELBOW_FLIP", f"exit step {np.degrees(wrap(psi200 - psi_prev)):.1f} deg")
    return out, {"per_frame": diag, "last_frame": prev, "lengths": [l1, l2], "reference_local": r.tolist()}


def _rel(rest, parent, child):
    """Child head in the parent bone's rest frame (translation of Rest_parent^-1 . Rest_child)."""
    Mp = np.asarray(rest[parent]["matrix_local"], float)
    Mc = np.asarray(rest[child]["matrix_local"], float)
    return (np.linalg.inv(Mp) @ Mc)[:3, 3]


def _m4(R, t):
    m = np.eye(4)
    m[:3, :3] = R
    m[:3, 3] = t
    return m


# ------------------------------------------------------------------------------------------------ metrics ---
def arm_local_series(dump_report, lo, hi, scale=0.01):
    """S/E/W (metres) in the clavicle_l frame for Action lo..hi."""
    S, E, W = [], [], []
    for f in range(lo, hi + 1):
        s, e, w = ArmFrame(dump_report, f).SEW()
        S.append(s * scale)
        E.append(e * scale)
        W.append(w * scale)
    return np.array(S), np.array(E), np.array(W)


def elbow_metrics(dump_report, interior=INTERIOR, window=(164, 207)):
    lo, hi = window
    i0, i1 = interior
    fr = np.arange(lo, hi + 1)
    S, E, W = arm_local_series(dump_report, lo, hi)
    r = reference_local(dump_report["rest"])
    psi = np.unwrap(np.array([swivel(S[k], E[k], W[k], r)[0] for k in range(len(fr))]))
    d = np.diff(E, axis=0)
    step = np.linalg.norm(d, axis=1)
    seg = (fr[:-1] >= i0 - 1) & (fr[:-1] <= i1)                    # steps 171->172 .. 199->200
    flicks = [int(fr[k + 1]) for k in range(len(d) - 1)
              if seg[k] and seg[k + 1] and min(step[k], step[k + 1]) > 0.020 and np.dot(d[k], d[k + 1]) < 0]
    n = np.cross(E - S, W - E)
    n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    plane_step = np.degrees(np.arccos(np.clip(np.sum(n[1:] * n[:-1], 1), -1, 1)))
    u, v = E - S, W - E
    flex = 180.0 - np.degrees(np.arccos(np.clip(np.sum(u * v, 1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1)), -1, 1)))
    k0 = int(np.where(fr == i0 - 1)[0][0])
    k1 = int(np.where(fr == i1 + 1)[0][0])
    psid = np.degrees(psi)
    vel = np.diff(psid) * FPS
    acc = np.diff(vel) * FPS
    acc_e = np.diff(d, axis=0) * FPS * FPS
    jerk_e = np.diff(acc_e, axis=0) * FPS
    jseg = (fr[3:] >= i0) & (fr[3:] <= i1 + 1)
    vseg = seg
    aseg = (fr[2:] >= i0) & (fr[2:] <= i1 + 1)
    # elbow pointing direction (unit offset from the S-W axis) in clavicle-frame anatomical terms
    rest = dump_report["rest"]
    rc = rot(np.asarray(rest[PARENT]["matrix_local"], float))
    ant = rc.T @ np.array([0.0, -1.0, 0.0])
    lat = rc.T @ np.array([1.0, 0.0, 0.0])
    sup = rc.T @ np.array([0.0, 0.0, 1.0])
    a = (W - S) / np.linalg.norm(W - S, axis=1, keepdims=True)
    off = (E - S) - np.sum((E - S) * a, 1, keepdims=True) * a
    off = off / np.linalg.norm(off, axis=1, keepdims=True)
    inside = (fr >= i0) & (fr <= i1)
    return {"space": "clavicle_l frame (arm-local, metres / degrees)", "window": [lo, hi],
            "elbow_peak_step_mm": float(step[seg].max() * 1e3), "elbow_peak_step_frame": int(fr[:-1][seg][int(np.argmax(step[seg]))] + 1),
            "elbow_entry_step_mm": float(step[k0] * 1e3), "elbow_exit_step_mm": float(step[k1 - 1] * 1e3),
            "elbow_path_mm": float(step[k0:k1].sum() * 1e3), "elbow_reversal_flicks": flicks,
            "elbow_rms_jerk_mps3": float(np.sqrt(np.mean(np.sum(jerk_e[jseg] ** 2, 1)))),
            "bend_plane_step_max_deg": float(plane_step[seg].max()), "bend_plane_entry_deg": float(plane_step[k0]),
            "bend_plane_exit_deg": float(plane_step[k1 - 1]),
            "swivel_step_max_deg": float(np.abs(np.diff(psid))[seg].max()),
            "swivel_entry_step_deg": float(abs(psid[k0 + 1] - psid[k0])), "swivel_exit_step_deg": float(abs(psid[k1] - psid[k1 - 1])),
            "swivel_peak_vel_dps": float(np.abs(vel[vseg]).max()), "swivel_peak_acc_dps2": float(np.abs(acc[aseg[:len(acc)]]).max()),
            "swivel_rms_acc_dps2": float(np.sqrt(np.mean(acc[aseg[:len(acc)]] ** 2))),
            "swivel_range_interior_deg": [float(psid[inside].min()), float(psid[inside].max())],
            "flex_range_interior_deg": [float(flex[inside].min()), float(flex[inside].max())],
            "near_straight_frames": [int(x) for x in fr[inside][flex[inside] > 175.0]],
            "elbow_dir_interior_mean": {"anterior": float(np.mean(off[inside] @ ant)), "lateral": float(np.mean(off[inside] @ lat)),
                                        "superior": float(np.mean(off[inside] @ sup))},
            "per_frame": {"frames": fr.tolist(), "swivel_deg": psid.tolist(), "flex_deg": flex.tolist(),
                          "elbow_local_m": E.tolist(), "wrist_local_m": W.tolist(), "elbow_step_mm": (step * 1e3).tolist(),
                          "bend_plane_step_deg": plane_step.tolist(),
                          "elbow_dir": {"anterior": (off @ ant).tolist(), "lateral": (off @ lat).tolist(), "superior": (off @ sup).tolist()}}}
