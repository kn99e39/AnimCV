"""EXPERIMENTAL (opt-in, Batch 136): UniMate v3 source-bound left-arm generation on the authentic S1 BaseRig Action.

Scope (frozen by the W136 declaration; nothing here is a Production path):
  * the frozen W129 B Action (endpoint slerp) is exported over ONE 61-frame window (Action 156..216) to a GLB by a
    disclosed, world-pose-preserving export adapter (one static ``pelvis`` root above ``spine_01``/``thigh_l``/``thigh_r``;
    the armature-object rotation baked onto that root) and turned into a UniMate own-rig asset + saved clip by the
    OFFICIAL ``data_process.rig_preprocess`` (offline ``rule`` labels, ``--save_clips``).  That clip is a MODEL INPUT
    PROXY ("PROXY_B"): a complete GT-free animation hypothesis, never measured 3D ground truth.
  * the official replacement sampler keeps every PROXY_B feature except the left-arm descendants of ``upperarm_l``
    (``lowerarm_l``, ``hand_l``, the five left finger chains) at Action 172..199 (+ their 171->172 velocity channels),
  * the generated ``upperarm_l``/``lowerarm_l`` rotations at Action 172..199 are decoded back to BaseRig pose-basis
    quaternions and imported (only those 2 channels x 28 frames) into a separate scratch Action.

Representation (UniMate 12-D, per joint, HML order): RIFKE position(3) | 6D rotation(6) | velocity t->t+1 (3).
Joint j's 6D slot holds the T-pose-rebased local rotation of its PARENT (``hml[j] = bvh[parent(j)]``).  So
  bvh(upperarm_l) = slot(lowerarm_l),  bvh(lowerarm_l) = slot(hand_l)   (both have exactly one child on the BaseRig).
Rebase (``compute_rots_from_tpos``):  new_j = cum_p * L_j * tq_j^-1 * cum_p^-1,  cum_j = cum_p * tq_j  (tq = T-pose local).
Inverse:  L_j = cum_p^-1 * new_j * cum_p * tq_j.   The glTF/bpy-4.0 export keeps BaseRig bone frames, so
  L_j = Rest_rel_j * Basis_j  and  tq_j = Rest_rel_j   =>   Basis_j = tq_j^-1 * L_j.
(Verified on the real S1 data before declaration; the instrument re-verifies it every run and refuses otherwise.)

Only numpy (+ stdlib) is imported; Python >= 3.10 (also imported by the UniMate-side runner).
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

SCHEMA = "animcv_exp_w136_unimate_v1"
FPS = 30
SOURCE_MINUS_ACTION = 78                      # Action = SOURCE - 78
N_ACTION = 240
INTERIOR = (172, 199)                         # authorised footprint frames (SOURCE 250-277)
CONTEXT_LAST, TARGET_FIRST = 171, 200
PIN_ACTION = tuple(range(132, 141))           # W126V WORLD_POSITION_PIN (SOURCE 210-218)
EDIT_CHANNELS = ("upperarm_l", "lowerarm_l")
WINDOW = (156, 215)                           # 60 model frames (feature frame i <-> Action 156 + i)
EXPORT_FRAMES = (156, 216)                    # 61 exported poses (velocity of frame 215 needs pose 216)
T_MODEL = 60
FEAT = 12
VEL = (9, 10, 11)
GEN_ROOT = "upperarm_l"                       # generated joints = strict descendants of upperarm_l
EXPECTED_LABELS = {"clavicle_l": "Left Shoulder", "upperarm_l": "Left Upper Arm", "lowerarm_l": "Left Forearm",
                   "hand_l": "Left Hand", "clavicle_r": "Right Shoulder", "upperarm_r": "Right Upper Arm",
                   "lowerarm_r": "Right Forearm", "hand_r": "Right Hand", "thigh_l": "Left Thigh", "thigh_r": "Right Thigh",
                   "pelvis": "Hips"}
FACE_PAIR = ("thigh_r", "thigh_l")            # (right, left)
N_JOINTS = 50                                 # 49 anatomical BaseRig bones + the adapter's pelvis root
ADAPTER_ROOT = "pelvis"
EXCLUDED_HELPERS = ("interaction", "center_of_mass")   # zero-weight helper roots, pruned by the official exporter
PROMPT = "An object presses its left arm overhead."
QUAT_TOL = 1e-5                               # roundtrip tolerance on |q| components (sign-canonical)
SCALE_SPREAD_MAX = 1e-3                       # float32 offsets of 3-4 cm finger bones give ~2e-4 (input-only calibration)


class UniStop(RuntimeError):
    def __init__(self, code, detail=""):
        super().__init__(f"{code}: {detail}")
        self.code, self.detail = code, detail


# ------------------------------------------------------------------------------------------------ hashing ---
def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# ------------------------------------------------------------------------------------------------ time ---
def action_to_index(f):
    i = int(f) - WINDOW[0]
    if not 0 <= i < T_MODEL:
        raise UniStop("WINDOW_FRAME", f"Action {f} outside window {WINDOW}")
    return i


def index_to_action(i):
    return WINDOW[0] + int(i)


def source_of(f):
    return int(f) + SOURCE_MINUS_ACTION


def check_window():
    """The declared window: 60 frames, native 30 fps, contains the footprint with context on both sides, no pin."""
    lo, hi = WINDOW
    if hi - lo + 1 != T_MODEL or EXPORT_FRAMES != (lo, hi + 1):
        raise UniStop("WINDOW_SHAPE", f"{WINDOW} / {EXPORT_FRAMES}")
    if not (lo < CONTEXT_LAST and TARGET_FIRST <= hi and lo <= INTERIOR[0] and INTERIOR[1] <= hi):
        raise UniStop("WINDOW_CONTEXT", "footprint without context on both sides")
    if any(lo <= f <= hi for f in PIN_ACTION):
        raise UniStop("PIN_OVERLAP", "window overlaps the hard pin")
    return True


def check_fps(fps):
    if abs(float(fps) - FPS) > 1e-9:
        raise UniStop("NOT_30_FPS", f"{fps}")
    return True


# ------------------------------------------------------------------------------------------------ quaternions ---
def qmul(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    w1, x1, y1, z1 = np.moveaxis(a, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(b, -1, 0)
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)


def qinv(a):
    a = np.asarray(a, float)
    return a * np.array([1.0, -1.0, -1.0, -1.0]) / np.sum(a * a, -1, keepdims=True)


def qnorm(a):
    a = np.asarray(a, float)
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


def qdist(a, b):
    """Sign-invariant max-abs component distance between unit quaternions."""
    a, b = qnorm(a), qnorm(b)
    return np.minimum(np.abs(a - b).max(-1), np.abs(a + b).max(-1))


def canon_sign(q, ref):
    q = np.asarray(q, float)
    return -q if float(np.dot(q, np.asarray(ref, float))) < 0 else q


def mat_to_quat(m):
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
    return qnorm(np.array(q))


def quat_to_mat(q):
    w, x, y, z = (float(v) for v in qnorm(q))
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def quat_to_6d(q):
    m = quat_to_mat(q)
    return np.concatenate([m[:, 0], m[:, 1]])


def sixd_to_mat(d, strict=True):
    """UniMate's 6D convention (first two matrix columns, Gram-Schmidt).  Unlike the official decoder, a degenerate or
    non-finite vector is REFUSED (never silently replaced by the identity)."""
    d = np.asarray(d, float)
    x_raw, y_raw = d[0:3], d[3:6]
    nx, ny = np.linalg.norm(x_raw), np.linalg.norm(y_raw)
    if strict and (not np.isfinite(d).all() or nx < 1e-6 or ny < 1e-6):
        raise UniStop("RIG_PROJECTION_FAILED", "non-finite or degenerate 6D rotation")
    x = x_raw / nx
    z = np.cross(x, y_raw)
    nz = np.linalg.norm(z)
    if strict and nz < 1e-6 * ny:
        raise UniStop("RIG_PROJECTION_FAILED", "parallel 6D axes")
    z = z / nz
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


# ------------------------------------------------------------------------------------------------ skeleton ---
def children(parents):
    ch = {j: [] for j in range(len(parents))}
    for j, p in enumerate(parents):
        if p >= 0:
            ch[int(p)].append(j)
    return ch


def descendants(parents, j):
    ch, out, stack = children(parents), [], [j]
    while stack:
        k = stack.pop()
        for c in ch[k]:
            out.append(c)
            stack.append(c)
    return sorted(out)


def generated_joints(names, parents):
    if GEN_ROOT not in names:
        raise UniStop("MISSING_JOINT", GEN_ROOT)
    return descendants(parents, names.index(GEN_ROOT))


def validate_asset(names, parents, clean_names, face_raw, rig_rest_parent, raw_annotation_labels=None):
    """Own-rig conditioning gate (labels, side, hierarchy, facing, joint budget).  ``rig_rest_parent`` maps every
    BaseRig bone to its BaseRig parent (``None`` for roots)."""
    names, clean_names = [str(n) for n in names], [str(n) for n in clean_names]
    if len(set(names)) != len(names):
        raise UniStop("JOINT_NAMES_DUPLICATE", "")
    if len(names) != N_JOINTS:
        raise UniStop("JOINT_COUNT", f"{len(names)} != {N_JOINTS}")
    expect = sorted([b for b in rig_rest_parent if b not in EXCLUDED_HELPERS] + [ADAPTER_ROOT])
    if sorted(names) != expect:
        missing = sorted(set(expect) - set(names))
        raise UniStop("MISSING_JOINT" if missing else "EXTRA_JOINT", str(missing or sorted(set(names) - set(expect))))
    if int(parents[0]) != -1 or names[0] != ADAPTER_ROOT or sum(int(p) == -1 for p in parents) != 1:
        raise UniStop("ROOT", f"{names[0]} / {sum(int(p) == -1 for p in parents)} roots")
    for j, n in enumerate(names[1:], 1):
        want = rig_rest_parent[n] if rig_rest_parent[n] is not None else ADAPTER_ROOT
        if names[int(parents[j])] != want:
            raise UniStop("HIERARCHY", f"{n}: parent {names[int(parents[j])]} != {want}")
    for b, lab in EXPECTED_LABELS.items():
        got = clean_names[names.index(b)]
        if got != lab:
            raise UniStop("LABEL_MISMATCH", f"{b}: '{got}' != '{lab}'")
    for j, n in enumerate(names):
        lab = clean_names[j]
        if n.endswith("_l") and lab.startswith("Right") or n.endswith("_r") and lab.startswith("Left"):
            raise UniStop("SIDE_MISMATCH", f"{n}: {lab}")
    if tuple(face_raw) != FACE_PAIR:
        raise UniStop("FACING_PAIR", f"{face_raw}")
    if raw_annotation_labels is not None and [str(x) for x in raw_annotation_labels] != clean_names:
        raise UniStop("LABEL_MISMATCH", "cond labels differ from the reviewed annotation")
    return True


def check_facing(tpos, names, rig_anterior_yup=(0.0, 0.0, 1.0), min_cos=0.95):
    """Canonical T-pose forward (up x (right - left) thigh, Y-up, right-handed) must be the rig's anterior (+Z after
    canonicalisation) and the right thigh must lie on -X (mirror / handedness check)."""
    tp = np.asarray(tpos, float)
    r, l = tp[names.index(FACE_PAIR[0])], tp[names.index(FACE_PAIR[1])]
    across = r - l
    across[1] = 0.0
    fwd = np.cross(np.array([0.0, 1.0, 0.0]), across)
    fwd = fwd / np.linalg.norm(fwd)
    c = float(np.dot(fwd, rig_anterior_yup))
    head = tp[names.index("neck_02")]
    up_ok = head[1] > tp[names.index(ADAPTER_ROOT)][1]
    if c < min_cos or not up_ok:
        raise UniStop("FACING", f"forward cos {c:.3f}, head above pelvis {up_ok}")
    if not (r[0] < 0 < l[0]):
        raise UniStop("HANDEDNESS", f"right thigh x {r[0]:.3f}, left thigh x {l[0]:.3f}")
    return c


def check_scale(offsets, names, parents, rig_rest_head):
    """Bone offsets must be the BaseRig rest offsets up to ONE positive global scale (no non-uniform scaling)."""
    ratios = []
    for j, n in enumerate(names[1:], 1):
        p = names[int(parents[j])]
        if p == ADAPTER_ROOT or n == ADAPTER_ROOT:
            continue
        lr = float(np.linalg.norm(np.subtract(rig_rest_head[n], rig_rest_head[p])))
        lo = float(np.linalg.norm(offsets[j]))
        if lr > 1e-6:
            ratios.append(lo / lr)
    ratios = np.asarray(ratios)
    spread = float(ratios.max() / ratios.min() - 1.0)
    if spread > SCALE_SPREAD_MAX:
        raise UniStop("SCALING", f"bone-length ratio spread {spread:.2e}")
    return float(np.median(ratios)), spread


# ------------------------------------------------------------------------------------------------ mask ---
def build_keep_mask(names, parents, max_joints=None):
    """(J, 12, T) bool keep mask: True = held at the PROXY_B feature, False = generated by the model.
    Generated: strict descendants of upperarm_l, all channels, Action 172..199, + their velocity channels at 171."""
    J = len(names)
    mj = max_joints or J
    keep = np.ones((mj, FEAT, T_MODEL), bool)
    keep[J:] = False                                   # padding joints: as the official joint mask (ignored)
    g = generated_joints(list(names), parents)
    i0, i1 = action_to_index(INTERIOR[0]), action_to_index(INTERIOR[1])
    for j in g:
        keep[j, :, i0:i1 + 1] = False
        keep[j, list(VEL), action_to_index(CONTEXT_LAST)] = False
    return keep


def validate_mask(keep, names, parents):
    """Fail closed unless the free (generated) set is exactly the declared left-arm footprint support."""
    keep = np.asarray(keep, bool)
    J = len(names)
    if keep.ndim != 3 or keep.shape[1:] != (FEAT, T_MODEL) or keep.shape[0] < J:
        raise UniStop("MASK_SHAPE", f"{keep.shape}")
    g = set(generated_joints(list(names), parents))
    lo = names.index("lowerarm_l") if "lowerarm_l" in names else None
    hd = names.index("hand_l") if "hand_l" in names else None
    if lo not in g or hd not in g:
        raise UniStop("MASK_DOMAIN", "lowerarm_l / hand_l not generated")
    allowed = np.zeros((J, FEAT, T_MODEL), bool)
    i0, i1 = action_to_index(INTERIOR[0]), action_to_index(INTERIOR[1])
    for j in g:
        allowed[j, :, i0:i1 + 1] = True
        allowed[j, list(VEL), action_to_index(CONTEXT_LAST)] = True
    free = ~keep[:J]
    bad = free & ~allowed
    if bad.any():
        j, c, t = (int(v[0]) for v in np.nonzero(bad))
        raise UniStop("MASK_DOMAIN", f"generated outside footprint: joint {names[j]} channel {c} Action {index_to_action(t)}")
    if not free[:, :, i0:i1 + 1].any():
        raise UniStop("MASK_EMPTY", "nothing generated")
    missing = allowed & ~free
    if missing.any():
        j, c, t = (int(v[0]) for v in np.nonzero(missing))
        raise UniStop("MASK_INCOMPLETE", f"{names[j]} channel {c} Action {index_to_action(t)} kept")
    for f in PIN_ACTION:
        if WINDOW[0] <= f <= WINDOW[1] and free[:, :, action_to_index(f)].any():
            raise UniStop("PIN_OVERLAP", f"Action {f}")
    return {"generated_joints": [names[j] for j in sorted(g)], "n_free": int(free.sum()),
            "frames_action": [INTERIOR[0], INTERIOR[1]], "velocity_only_frame": CONTEXT_LAST}


# ------------------------------------------------------------------------------------------------ decode ---
def cumulative_tpose(tq, parents):
    tq = np.asarray(tq, float)
    cum = tq.copy()
    for j in range(1, len(parents)):
        cum[j] = qmul(cum[int(parents[j])], tq[j])
    return cum


def bvh_from_slot(feats, names, parents, joint):
    """Rebased (T-pose-relative) rotation of ``joint`` = the 6D slot of its single child (HML order)."""
    j = names.index(joint)
    ch = children(parents)[j]
    if len(ch) != 1:
        raise UniStop("REPRESENTATION_BLOCKED", f"{joint} has {len(ch)} children; single-child slot required")
    c = ch[0]
    return np.stack([mat_to_quat(sixd_to_mat(feats[t, c, 3:9])) for t in range(feats.shape[0])])


def decode_basis(feats, names, parents, tq, joints=EDIT_CHANNELS):
    """Denormalised UniMate features (T, J, 12) -> BaseRig pose-basis quaternions (wxyz) per window frame."""
    feats = np.asarray(feats, float)
    if feats.ndim != 3 or feats.shape[1] != len(names) or feats.shape[2] != FEAT:
        raise UniStop("FEATURE_SHAPE", f"{feats.shape}")
    if not np.isfinite(feats).all():
        raise UniStop("RIG_PROJECTION_FAILED", "non-finite features")
    names = [str(n) for n in names]
    tq = qnorm(np.asarray(tq, float))
    cum = cumulative_tpose(tq, parents)
    out = {}
    for b in joints:
        j = names.index(b)
        p = int(parents[j])
        new = bvh_from_slot(feats, names, parents, b)
        L = qmul(qmul(qmul(qinv(cum[p])[None], new), cum[p][None]), tq[j][None])
        out[b] = qnorm(qmul(qinv(tq[j])[None], L))
    return out


def encode_slot(basis_by_frame, names, parents, tq, joint):
    """Forward map used by the synthetic tests: BaseRig basis -> the child's 6D slot (inverse of ``decode_basis``)."""
    j = names.index(joint)
    p = int(parents[j])
    tq = qnorm(np.asarray(tq, float))
    cum = cumulative_tpose(tq, parents)
    L = qmul(tq[j][None], np.asarray(basis_by_frame, float))
    new = qmul(qmul(qmul(cum[p][None], L), qinv(tq[j])[None]), qinv(cum[p])[None])
    return np.stack([quat_to_6d(q) for q in new])


def roundtrip_error(decoded, stored, joint, shift=0):
    """Max sign-invariant quaternion distance between decoded window frames and Blender stored keys of ``joint``
    (``stored`` = dump ``stored`` dict), with the window mapped to Action frames shifted by ``shift``."""
    q = decoded[joint]
    err = []
    for i in range(q.shape[0]):
        f = index_to_action(i) + shift
        ref = np.array([stored[joint][str(c)][str(f)] for c in range(4)])
        err.append(float(qdist(q[i], ref)))
    return max(err)


def time_identity(decoded, stored, joints=EDIT_CHANNELS):
    """Frame-identity proof: shift 0 must be within QUAT_TOL AND >= 100x better than shifts -1 and +1."""
    rep = {}
    for b in joints:
        e0 = roundtrip_error(decoded, stored, b, 0)
        em = roundtrip_error(decoded, stored, b, -1)
        ep = roundtrip_error(decoded, stored, b, +1)
        rep[b] = {"shift0": e0, "shift-1": em, "shift+1": ep}
        if e0 > QUAT_TOL:
            raise UniStop("REPRESENTATION_BLOCKED", f"{b} roundtrip error {e0:.2e} > {QUAT_TOL}")
        if min(em, ep) < 100 * max(e0, 1e-9):
            raise UniStop("TIME_SHIFT", f"{b}: shift errors {em:.2e}/{ep:.2e} not separable from {e0:.2e}")
    return rep


# ------------------------------------------------------------------------------------------------ import ---
def check_domain(frames, channels=EDIT_CHANNELS):
    bad_ch = sorted(set(channels) - set(EDIT_CHANNELS))
    if bad_ch:
        raise UniStop("DOMAIN_CHANNEL", str(bad_ch))
    fr = sorted(int(f) for f in frames)
    if any(f in PIN_ACTION for f in fr):
        raise UniStop("PIN_OVERLAP", f"{fr[0]}..{fr[-1]}")
    if fr and (fr[0] < INTERIOR[0] or fr[-1] > INTERIOR[1]):
        raise UniStop("DOMAIN_FRAMES", f"{fr[0]}..{fr[-1]}")
    return True


def import_values(decoded, stored_ref, frames=range(INTERIOR[0], INTERIOR[1] + 1)):
    """Authorised import: ``{channel: {frame: [w,x,y,z]}}`` for upperarm_l/lowerarm_l at 172..199 only, unit-norm,
    sign-canonicalised to the reference Action's key at the same frame (rotation unchanged)."""
    frames = list(frames)
    check_domain(frames, list(decoded))
    out = {}
    for ch in EDIT_CHANNELS:
        out[ch] = {}
        for f in frames:
            q = qnorm(decoded[ch][action_to_index(f)])
            if not np.isfinite(q).all():
                raise UniStop("RIG_PROJECTION_FAILED", f"{ch} {f}")
            ref = [stored_ref[ch][str(c)][str(f)] for c in range(4)]
            out[ch][str(f)] = [float(v) for v in canon_sign(q, ref)]
    return out


# ------------------------------------------------------------------------------------------------ source 2D ---
COCO = {"l_sho": 5, "r_sho": 6, "l_elb": 7, "r_elb": 8, "l_wri": 9, "r_wri": 10, "l_hip": 11, "r_hip": 12}
RIG_OF = {"l_sho": "upperarm_l", "r_sho": "upperarm_r", "l_hip": "thigh_l", "r_hip": "thigh_r",
          "l_elb": "lowerarm_l", "l_wri": "hand_l"}
FIT_POINTS = ("l_sho", "r_sho", "l_hip", "r_hip")     # unedited in every branch (right arm is unkeyed at rig rest)
FIT_RMS_MAX_PX = 20.0


def fit_affine_camera(world_pts, img_pts):
    """Least-squares affine camera  u = P [X;1]  (2x4) from correspondences (N,3)->(N,2)."""
    X = np.c_[np.asarray(world_pts, float), np.ones(len(world_pts))]
    P, *_ = np.linalg.lstsq(X, np.asarray(img_pts, float), rcond=None)
    res = X @ P - img_pts
    return P.T, float(np.sqrt(np.mean(np.sum(res ** 2, 1))))


def project(P, pts):
    pts = np.asarray(pts, float)
    return (np.c_[pts, np.ones(len(pts))] @ P.T)
