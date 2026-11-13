"""EXPERIMENTAL (opt-in, Batch 138): BaseRig Action <-> Kimodo SOMA adapter for ONE left-arm interior proposal.

Scope (frozen by the W138 declaration; nothing here is a Production path):
  * the frozen W129 PROXY_B Action (== A outside Action 172..199) is mapped into Kimodo's own 77-joint SOMA skeleton by the
    unchanged W129 closed-form ``Mapping`` (built here on Kimodo's neutral joints, identity rest rotations),
  * Kimodo receives vendor ``fullbody`` keyframes ONLY at the declared context frames (window indices 0..8 and 37..45 =
    Action 163..171 and 200..208, 18 < 20 keyframes) and NOTHING inside the interior (indices 9..36 = Action 172..199),
  * the generated SOMA motion is mapped back into BaseRig ``upperarm_l`` / ``lowerarm_l`` pose-bone rotations at
    Action 172..199 only; every other generated joint and frame is analysed, never imported.

Time map: native 30 fps; Action = SOURCE - 78; window index i <-> Action 163 + i (i = 0..45).

Only numpy (+ stdlib and the numpy-only W129/W136 modules) is imported, so the module loads in the AnimCV venv and in tests.
"""
from __future__ import annotations

import numpy as np

import app.experimental_motionbricks as M
import app.experimental_unimate as U

SCHEMA = "animcv_exp_w138_kimodo_v1"
FPS = 30
WINDOW = (163, 208)                                   # Action frames of the generated clip (inclusive)
N_WIN = WINDOW[1] - WINDOW[0] + 1                     # 46 frames
INTERIOR = U.INTERIOR                                 # (172, 199): authorised import footprint
EDIT_CHANNELS = U.EDIT_CHANNELS
PIN_ACTION = U.PIN_ACTION
KEY_ACTION = tuple(range(163, 172)) + tuple(range(200, 209))      # 18 vendor full-body keyframes (context only)
MAX_KEYFRAMES = 19                                    # vendor guidance: fewer than 20 constrained frames per type
STEP_FRAMES = tuple(range(INTERIOR[0] - 1, INTERIOR[1] + 2))     # Action 171..200 -> 29 steps
UNUSABLE_STEP_MM = 100.0                              # W136's explicit unusable-continuity diagnostic
FLICK_MIN_MM = 20.0
ADHERENCE_MEAN_MAX_M = 0.05                           # declared: mean keyframe joint-position error of the raw output
COPY_MAX_FRACTION = 0.5
TEXT_DIM = 4096
SOMA30 = M.SOMA_MODEL_JOINTS                          # identical 30 names and order as kimodo SOMASkeleton30
GROUPS30 = {
    "root": ("Hips",), "torso": ("Spine1", "Spine2", "Chest"), "head": ("Neck1", "Neck2", "Head", "Jaw", "LeftEye", "RightEye"),
    "left_clavicle": ("LeftShoulder",), "left_arm": ("LeftArm", "LeftForeArm", "LeftHand", "LeftHandThumbEnd", "LeftHandMiddleEnd"),
    "right_clavicle": ("RightShoulder",), "right_arm": ("RightArm", "RightForeArm", "RightHand", "RightHandThumbEnd", "RightHandMiddleEnd"),
    "legs": ("LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase", "RightLeg", "RightShin", "RightFoot", "RightToeBase"),
}
ARM = {"shoulder": "LeftArm", "elbow": "LeftForeArm", "wrist": "LeftHand"}   # SOMA joint positions = BaseRig bone heads


class KimodoStop(RuntimeError):
    def __init__(self, code, detail=""):
        super().__init__(f"{code}: {detail}")
        self.code, self.detail = code, detail


# ------------------------------------------------------------------------------------------------ time contract ---
def action_to_index(f):
    f = int(f)
    if not WINDOW[0] <= f <= WINDOW[1]:
        raise KimodoStop("WINDOW_DOMAIN", f"Action {f} outside {WINDOW}")
    return f - WINDOW[0]


def index_to_action(i):
    i = int(i)
    if not 0 <= i < N_WIN:
        raise KimodoStop("WINDOW_DOMAIN", f"index {i}")
    return WINDOW[0] + i


def key_indices():
    return [action_to_index(f) for f in KEY_ACTION]


def interior_indices():
    return [action_to_index(f) for f in range(INTERIOR[0], INTERIOR[1] + 1)]


def check_fps_frames(fps, n_frames):
    if int(fps) != FPS or float(fps) != float(FPS):
        raise KimodoStop("NOT_30_FPS", str(fps))
    if int(n_frames) != N_WIN:
        raise KimodoStop("WINDOW_LENGTH", f"{n_frames} != {N_WIN}")
    return True


def check_keyframes(idx):
    """The declared keyframe set exactly: ints, strictly increasing, inside the window, fewer than 20, none inside the
    interior, both boundary context frames (Action 171, 200) present."""
    try:
        idx = [int(i) for i in idx]
    except (TypeError, ValueError):
        raise KimodoStop("KEYFRAMES_FORMAT", str(idx)[:80]) from None
    if not idx or any(b <= a for a, b in zip(idx, idx[1:])):
        raise KimodoStop("KEYFRAMES_FORMAT", "empty, unsorted or duplicated")
    if idx[0] < 0 or idx[-1] >= N_WIN:
        raise KimodoStop("KEYFRAMES_DOMAIN", "outside window")
    if len(idx) > MAX_KEYFRAMES:
        raise KimodoStop("KEYFRAMES_DOMAIN", f"{len(idx)} keyframes >= 20 (vendor sparse-keyframe guidance)")
    inside = sorted(set(idx) & set(interior_indices()))
    if inside:
        raise KimodoStop("KEYFRAMES_DOMAIN", f"keyframe inside the interior {inside[:4]} (would assert unobserved 3D)")
    if idx != key_indices():
        raise KimodoStop("KEYFRAMES_DOMAIN", "not the declared keyframe set")
    return True


def validate_motion_mask(mask):
    """Vendor ``motion_mask`` [T, D] (or [1, T, D]) as built by ``create_conditions_from_constraints``: constrained rows must
    be exactly the declared keyframes; every interior row must be entirely unconstrained."""
    m = np.asarray(mask)
    if m.ndim == 3:
        if m.shape[0] != 1:
            raise KimodoStop("MASK_SHAPE", str(m.shape))
        m = m[0]
    if m.ndim != 2 or m.shape[0] != N_WIN:
        raise KimodoStop("MASK_SHAPE", str(m.shape))
    if not (m.dtype == bool or np.isin(m, (0, 1)).all()):
        raise KimodoStop("MASK_DTYPE", str(m.dtype))
    rows = [int(i) for i in np.where(m.astype(bool).any(1))[0]]
    if not rows:
        raise KimodoStop("MASK_EMPTY", "no constrained row")
    if rows != key_indices():
        raise KimodoStop("MASK_DOMAIN", f"constrained rows {rows[:6]}... != declared keyframes")
    return {"constrained_rows": rows, "n_constrained_rows": len(rows), "channels_per_row": sorted({int(c) for c in m.astype(bool).sum(1) if c})}


def check_null_text(texts):
    """No-text mode: every prompt must be empty after the vendor's own strip; any text would need the gated encoder."""
    if isinstance(texts, str):
        texts = [texts]
    bad = [t for t in texts if len(str(t).strip()) != 0]
    if bad:
        raise KimodoStop("TEXT_MODE_BLOCKED", f"non-empty prompt {bad[0][:40]!r}: the text-conditioned path needs the gated Llama 3 base")
    return True


# ------------------------------------------------------------------------------------------------ skeleton ---
def hip_height(neutral, names):
    """Kimodo neutral joints have Hips at the origin; the declared rest hip height is its height above the lowest joint."""
    neutral = np.asarray(neutral, float)
    if str(names[0]) != "Hips" or np.abs(neutral[0]).max() > 1e-9:
        raise KimodoStop("SOMA_SKELETON_ROOT", "Hips must be joint 0 at the origin")
    h = float(-neutral[:, 1].min())
    if not 0.5 < h < 1.5:
        raise KimodoStop("SOMA_SKELETON_SCALE", f"hip height {h:.3f} m")
    return h


def kimodo_soma(names, parents, neutral):
    """W129 ``SomaSkeleton`` on Kimodo's 77-joint SOMA: identity rest rotations, local offsets = neutral[j] - neutral[parent];
    the Hips reference position is (0, hip height, 0) (only its height is used, by the W129 hip scale)."""
    names = [str(n) for n in names]
    parents = [int(p) for p in parents]
    neutral = np.asarray(neutral, float)
    if len(names) != 77 or neutral.shape != (77, 3) or len(parents) != 77:
        raise KimodoStop("SOMA_SKELETON_SHAPE", f"{len(names)} names, {neutral.shape}")
    ref = np.zeros((77, 7))
    ref[:, 6] = 1.0
    for i, p in enumerate(parents):
        ref[i, :3] = neutral[i] - (neutral[p] if p >= 0 else 0.0)
    ref[0, :3] = [0.0, hip_height(neutral, names), 0.0]
    return M.SomaSkeleton(names, parents, ref)


def mats_from_soma_local(soma_local):
    sl = np.asarray(soma_local, float)
    return np.stack([M.quat_xyzw_to_mat(q) for q in sl[:, 3:]])


def soma_local_from_mats(soma, mats, root_pos):
    """[J, 7] SOMA local (xyz = reference offsets, root xyz = ``root_pos``; qxyzw from the rotation matrices)."""
    mats = np.asarray(mats, float)
    if mats.shape != (len(soma.names), 3, 3) or not np.isfinite(mats).all():
        raise KimodoStop("RIG_PROJECTION_FAILED", f"local rotation shape {mats.shape} or non-finite")
    det = np.linalg.det(mats)
    if np.abs(det - 1.0).max() > 1e-3:
        raise KimodoStop("RIG_PROJECTION_FAILED", f"improper/degenerate rotation det {det.min():.4f}..{det.max():.4f}")
    out = np.array(soma.ref, float, copy=True)
    for i, m in enumerate(mats):
        u, _, vt = np.linalg.svd(m)
        out[i, 3:] = M.mat_to_quat_xyzw(u @ vt)
    out[0, :3] = np.asarray(root_pos, float)
    return out


def positions30(soma, soma_local):
    """Joint positions of the 30 model joints by the numpy W129 FK on Kimodo's skeleton."""
    _, P = soma.fk(soma_local)
    return P[[soma.idx[j] for j in SOMA30]]


def canonical_shift(root_seq):
    """Vendor canonical origin: root XZ = (0, 0) at frame 0 (Y unchanged).  Returns the additive shift."""
    r = np.asarray(root_seq, float)
    return np.array([-r[0, 0], 0.0, -r[0, 2]])


def check_canonical(root_seq, tol=1e-9):
    r = np.asarray(root_seq, float)
    if abs(r[0, 0]) > tol or abs(r[0, 2]) > tol:
        raise KimodoStop("CANONICAL_ORIGIN", f"root XZ at frame 0 = ({r[0, 0]:.4g}, {r[0, 2]:.4g})")
    return True


def decode_edit_basis(mapping, soma, local_mats_seq, root_seq):
    """BaseRig basis (wxyz) of upperarm_l / lowerarm_l for every window frame from Kimodo 77-joint local rotations."""
    out = {ch: [] for ch in EDIT_CHANNELS}
    for mats, r in zip(local_mats_seq, root_seq):
        b = mapping.base_basis_from_soma(soma_local_from_mats(soma, mats, r))
        for ch in EDIT_CHANNELS:
            out[ch].append(b[ch])
    return {ch: np.array(v) for ch, v in out.items()}


def time_identity(decoded, stored, joints=EDIT_CHANNELS, tol=U.QUAT_TOL):
    """Shift 0 within ``tol`` AND >= 100x better than shifts -1/+1 (window index i <-> Action 163 + i)."""
    rep = {}
    for b in joints:
        errs = {}
        for s in (-1, 0, 1):
            e = 0.0
            for i in range(N_WIN):
                f = index_to_action(i) + s
                if not 1 <= f <= U.N_ACTION:
                    continue
                ref = np.array([stored[b][str(c)][str(f)] for c in range(4)])
                q = M.canon_sign(np.asarray(decoded[b][i], float), ref)
                e = max(e, float(np.abs(q - ref).max()))
            errs[s] = e
        rep[b] = {"shift0": errs[0], "shift-1": errs[-1], "shift+1": errs[1]}
        if errs[0] > tol:
            raise KimodoStop("REPRESENTATION_BLOCKED", f"{b} roundtrip {errs[0]:.2e} > {tol}")
        if min(errs[-1], errs[1]) < 100 * max(errs[0], 1e-9):
            raise KimodoStop("TIME_SHIFT", f"{b} shift errors {errs[-1]:.2e}/{errs[1]:.2e} vs {errs[0]:.2e}")
    return rep


def import_values(decoded, stored_ref):
    """Authorised import {channel: {Action: wxyz}} at 172..199 only, unit-norm, sign-canonical to the reference key."""
    frames = list(range(INTERIOR[0], INTERIOR[1] + 1))
    U.check_domain(frames, list(decoded))
    out = {}
    for ch in EDIT_CHANNELS:
        out[ch] = {}
        for f in frames:
            q = np.asarray(decoded[ch][action_to_index(f)], float)
            if not np.isfinite(q).all() or np.linalg.norm(q) < 1e-6:
                raise KimodoStop("RIG_PROJECTION_FAILED", f"{ch} {f}")
            q = q / np.linalg.norm(q)
            ref = [stored_ref[ch][str(c)][str(f)] for c in range(4)]
            out[ch][str(f)] = [float(v) for v in M.canon_sign(q, ref)]
    return out


# ------------------------------------------------------------------------------------------------ conditioning ---
def group_index(names=SOMA30):
    names = [str(n) for n in names]
    return {g: [names.index(j) for j in js if j in names] for g, js in GROUPS30.items()}


def conditioning_report(raw30, proxy30, keys=None):
    """Raw output joint positions vs the PROXY_B keyframe targets [N, 30, 3] (same canonical model space, metres)."""
    raw30, proxy30 = np.asarray(raw30, float), np.asarray(proxy30, float)
    if raw30.shape != (N_WIN, 30, 3) or proxy30.shape != (N_WIN, 30, 3):
        raise KimodoStop("POSITION_SHAPE", f"{raw30.shape} {proxy30.shape}")
    keys = key_indices() if keys is None else list(keys)
    inter = interior_indices()
    err = np.linalg.norm(raw30[keys] - proxy30[keys], axis=-1)                 # (K, 30)
    groups = {g: {"key_err_max_m": float(err[:, ix].max()), "key_err_mean_m": float(err[:, ix].mean())}
              for g, ix in group_index().items()}
    exact = float(np.mean(np.all(np.isclose(raw30[inter], proxy30[inter], rtol=0, atol=1e-7), axis=-1)))
    return {"key_err_mean_m": float(err.mean()), "key_err_max_m": float(err.max()), "groups": groups,
            "boundary_key_err_mean_m": {"action_171": float(err[keys.index(action_to_index(171))].mean()),
                                        "action_200": float(err[keys.index(action_to_index(200))].mean())},
            "interior_exact_equal_fraction": exact,
            "interior_dev_from_proxy_max_m": float(np.linalg.norm(raw30[inter] - proxy30[inter], axis=-1).max())}


def check_conditioning(rep, mean_max=ADHERENCE_MEAN_MAX_M, copy_max=COPY_MAX_FRACTION):
    if rep["interior_exact_equal_fraction"] > copy_max:
        raise KimodoStop("INTERIOR_COPIED", f"{rep['interior_exact_equal_fraction']:.3f} of interior joints equal PROXY_B")
    if not np.isfinite(rep["key_err_mean_m"]) or rep["key_err_mean_m"] > mean_max:
        raise KimodoStop("CONDITIONING_FIDELITY", f"mean keyframe joint error {rep['key_err_mean_m']:.4f} m > {mean_max}")
    return True


# ------------------------------------------------------------------------------------------------ raw analysis ---
def _steps(series):
    return (np.linalg.norm(np.diff(np.asarray(series, float), axis=0), axis=1) * 1e3).tolist()


def flicks(series, min_mm=FLICK_MIN_MM):
    d = np.diff(np.asarray(series, float), axis=0)
    s = np.linalg.norm(d, axis=1) * 1e3
    return [k for k in range(len(d) - 1) if s[k] > min_mm and s[k + 1] > min_mm and float(np.dot(d[k], d[k + 1])) < 0]


def continuity(series):
    """``series``: positions at Action 171..200 (30 rows) -> entry/exit/peak steps (mm) and flicks."""
    st = _steps(series)
    return {"entry_step_mm": st[0], "exit_step_mm": st[-1], "peak_step_mm": float(max(st)),
            "peak_step_to_action": int(STEP_FRAMES[int(np.argmax(st)) + 1]), "n_flicks": len(flicks(series)), "steps_mm": st}


def _arm_series(pos30, joint):
    j = SOMA30.index(joint)
    return np.asarray(pos30)[[action_to_index(f) for f in STEP_FRAMES], j]


def attribution(raw30, proxy30, hyb30):
    """Raw (model full body) vs projected (PROXY_B parents + generated LeftArm/LeftForeArm local rotations) vs PROXY_B, all
    Kimodo FK in the same canonical model space."""
    views = {"raw": raw30, "projected": hyb30, "proxyB": proxy30}
    cont = {v: {k: continuity(_arm_series(p, j)) for k, j in ARM.items()} for v, p in views.items()}
    gap = {}
    for k in ("elbow", "wrist"):
        jk, js = SOMA30.index(ARM[k]), SOMA30.index(ARM["shoulder"])
        tot, sh, rot = [], [], []
        for f in range(INTERIOR[0], INTERIOR[1] + 1):
            i = action_to_index(f)
            r, h, rs, hs = raw30[i, jk], hyb30[i, jk], raw30[i, js], hyb30[i, js]
            tot.append(np.linalg.norm(r - h) * 1e3)
            sh.append(np.linalg.norm(rs - hs) * 1e3)
            rot.append(np.linalg.norm((r - rs) - (h - hs)) * 1e3)
        gap[k] = {"raw_minus_projected_max_mm": float(max(tot)), "raw_minus_projected_mean_mm": float(np.mean(tot)),
                  "shoulder_translation_part_max_mm": float(max(sh)), "parent_orientation_part_max_mm": float(max(rot))}
    excess = {}
    for k in ("elbow", "wrist"):
        e = np.array(cont["raw"][k]["steps_mm"]) - np.array(cont["proxyB"][k]["steps_mm"])
        excess[k] = {"max_excess_mm": float(e.max()), "at_step_to_action": int(STEP_FRAMES[int(np.argmax(e)) + 1])}
    return {"continuity": cont, "raw_vs_projected": gap, "raw_excess_step_over_proxyB": excess}


def body_report(raw30, proxy30):
    out = {}
    fi = interior_indices()
    si = [action_to_index(f) for f in STEP_FRAMES]
    for g, ix in group_index().items():
        dev = np.linalg.norm(raw30[fi][:, ix] - proxy30[fi][:, ix], axis=-1) * 1e3
        sr = np.linalg.norm(np.diff(raw30[si][:, ix], axis=0), axis=-1) * 1e3
        sg = np.linalg.norm(np.diff(proxy30[si][:, ix], axis=0), axis=-1) * 1e3
        out[g] = {"interior_dev_max_mm": float(dev.max()), "interior_dev_mean_mm": float(dev.mean()),
                  "raw_entry_step_max_mm": float(sr[0].max()), "proxyB_entry_step_max_mm": float(sg[0].max()),
                  "raw_exit_step_max_mm": float(sr[-1].max()), "proxyB_exit_step_max_mm": float(sg[-1].max()),
                  "raw_peak_step_mm": float(sr.max()), "proxyB_peak_step_mm": float(sg.max()),
                  "boundary_excess_max_mm": float(max((sr[0] - sg[0]).max(), (sr[-1] - sg[-1]).max())),
                  "excess_max_mm": float((sr - sg).max())}
    return out


def raw_unusable(att, body, limit=UNUSABLE_STEP_MM):
    reasons = [f"raw {k} step exceeds PROXY_B by {v['max_excess_mm']:.0f} mm (-> Action {v['at_step_to_action']})"
               for k, v in att["raw_excess_step_over_proxyB"].items() if v["max_excess_mm"] > limit]
    reasons += [f"group {g} boundary step exceeds PROXY_B by {v['boundary_excess_max_mm']:.0f} mm"
                for g, v in body.items() if v["boundary_excess_max_mm"] > limit]
    return {"unusable": bool(reasons), "reasons": reasons, "limit_mm": limit}


def hybrid_local(proxy_local, raw_local, soma):
    """PROXY_B 77-joint local poses with ONLY the LeftArm / LeftForeArm local rotations taken from the raw output."""
    h = np.array(proxy_local, float, copy=True)
    for j in ("LeftArm", "LeftForeArm"):
        h[:, soma.idx[j], 3:] = np.asarray(raw_local)[:, soma.idx[j], 3:]
    return h


def neck_height_mm(pos30):
    """Neck2 height above Hips per frame (body-collapse diagnostic)."""
    p = np.asarray(pos30)
    return ((p[:, SOMA30.index("Neck2"), 1] - p[:, SOMA30.index("Hips"), 1]) * 1e3).tolist()
