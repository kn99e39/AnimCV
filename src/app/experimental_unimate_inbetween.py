"""EXPERIMENTAL (opt-in, Batch 137): official UniMate ``--inbetween`` temporal conditioning on the authentic S1 rig.

Scope (frozen by the W137 declaration; nothing here is a Production path):
  * the frozen W136 own-rig PROXY_B asset (official ``rig_preprocess`` output of the W129 B Action, Action 156..216) is
    re-used byte-for-byte; window Action 156..215 = feature frames 0..59 (``app.experimental_unimate`` time map),
  * the OFFICIAL frame-only mode keeps every whole-body feature at window frames 0..15 and 44..59 (32 frames, Action
    156..171 and 200..215) and generates ALL joints and channels at window frames 16..43 (Action 172..199),
  * only ``upperarm_l``/``lowerarm_l`` rotations at Action 172..199 are decoded (W136 decoder, unchanged) and imported;
    the raw full-body model output is analysed separately and never imported.

This module adds only what the temporal mode needs: the declared keep-frame set and its validator, the
model-space -> rig-world similarity map (fitted on the NO-MODEL PROXY_B positions), and the raw-versus-projected
continuity / parent-context attribution metrics.  Only numpy (+ stdlib); Python >= 3.10.
"""
from __future__ import annotations

import numpy as np

import app.experimental_unimate as U

SCHEMA = "animcv_exp_w137_unimate_inbetween_v1"
KEEP_FRAMES = tuple(range(0, 16)) + tuple(range(44, 60))      # 32 window indices held at PROXY_B (all joints, all channels)
FREE_FRAMES = tuple(range(16, 44))                             # 28 window indices generated (Action 172..199)
KEEP_FRAMES_ARG = ",".join(str(i) for i in KEEP_FRAMES)       # the literal official --keep_frames argument
FLICK_MIN_MM = 20.0                                            # W129 reversal-flick definition (both steps > 20 mm, negative dot)
UNUSABLE_STEP_MM = 100.0                                       # W136's explicit unusable-continuity diagnostic
SIM_RMS_MAX_M = 1e-4                                           # world-mapping gate (NO-MODEL PROXY_B FK vs B readback; input-only 2.6e-6)
STEP_FRAMES = tuple(range(U.CONTEXT_LAST, U.TARGET_FIRST + 1))  # Action 171..200 -> 29 steps 171->172 .. 199->200
GROUPS = {
    "root": ("pelvis",),
    "torso": ("spine_01", "spine_02", "spine_03", "spine_04", "spine_05"),
    "head": ("neck_01", "neck_02"),
    "left_clavicle": ("clavicle_l",),
    "left_arm": ("upperarm_l", "lowerarm_l", "hand_l"),
    "right_arm": ("clavicle_r", "upperarm_r", "lowerarm_r", "hand_r"),
    "legs": ("thigh_l", "calf_l", "foot_l", "thigh_r", "calf_r", "foot_r"),
}


# ------------------------------------------------------------------------------------------------ keep frames ---
def check_keep_frames(keep):
    """The declared official keep-frame list: exactly window 0..15 + 44..59, non-negative, sorted, unique; its complement
    is exactly the authorised footprint Action 172..199 with frozen 171/200 kept; the window holds no pin frame."""
    U.check_window()
    k = [int(x) for x in keep]
    if k != sorted(set(k)) or any(x < 0 or x >= U.T_MODEL for x in k):
        raise U.UniStop("KEEP_FRAMES_FORMAT", f"{k}")
    if tuple(k) != KEEP_FRAMES:
        raise U.UniStop("KEEP_FRAMES_DOMAIN", f"{k} != declared")
    free = [i for i in range(U.T_MODEL) if i not in k]
    if [U.index_to_action(i) for i in free] != list(range(U.INTERIOR[0], U.INTERIOR[1] + 1)):
        raise U.UniStop("KEEP_FRAMES_DOMAIN", "complement != Action 172..199")
    if U.action_to_index(U.CONTEXT_LAST) not in k or U.action_to_index(U.TARGET_FIRST) not in k:
        raise U.UniStop("KEEP_FRAMES_DOMAIN", "frame 171 or 200 not kept")
    return {"n_keep": len(k), "n_free": len(free), "free_action": [U.index_to_action(free[0]), U.index_to_action(free[-1])]}


def parse_keep_arg(spec):
    """Mirror of the official comma parser for non-negative indices (the official one resolves negatives against max_T)."""
    return [int(t) for t in str(spec).split(",") if t.strip()]


def temporal_mask():
    m = np.zeros((1, 1, 1, U.T_MODEL), bool)
    m[0, 0, 0, list(KEEP_FRAMES)] = True
    return m


def validate_temporal_mask(mask):
    """Fail closed unless ``mask`` is the official frame-only shape (1,1,1,60) with True exactly at the declared frames."""
    m = np.asarray(mask)
    if m.dtype != bool:
        raise U.UniStop("MASK_DTYPE", str(m.dtype))
    if m.shape != (1, 1, 1, U.T_MODEL):
        raise U.UniStop("MASK_NOT_TEMPORAL", f"shape {m.shape} != (1, 1, 1, {U.T_MODEL})")
    got = [int(i) for i in np.nonzero(m[0, 0, 0])[0]]
    if not got:
        raise U.UniStop("MASK_EMPTY", "nothing kept")
    if len(got) == U.T_MODEL:
        raise U.UniStop("MASK_EMPTY", "nothing generated")
    extra, missing = sorted(set(got) - set(KEEP_FRAMES)), sorted(set(KEEP_FRAMES) - set(got))
    if extra:
        raise U.UniStop("MASK_DOMAIN", f"kept frames outside the declared set {extra}")
    if missing:
        raise U.UniStop("MASK_DOMAIN", f"declared kept frames generated {missing}")
    return {"shape": list(m.shape), "kept": got, "n_kept": len(got)}


def check_mode(inbetween, motion_edit, motion_expand=False):
    if not inbetween or motion_edit or motion_expand:
        raise U.UniStop("CONDITIONING_BLOCKED", f"inbetween={inbetween} motion_edit={motion_edit} motion_expand={motion_expand}")
    return True


# ------------------------------------------------------------------------------------------------ conditioning ---
def group_index(names):
    names = [str(n) for n in names]
    out = {g: [names.index(b) for b in bs if b in names] for g, bs in GROUPS.items()}
    covered = {j for v in out.values() for j in v}
    out["fingers_left"] = [j for j, n in enumerate(names) if n.endswith("_l") and j not in covered]
    out["fingers_right"] = [j for j, n in enumerate(names) if n.endswith("_r") and j not in covered]
    return out


def conditioning_report(sample, gt, names, keep=KEEP_FRAMES):
    """Denormalised (T, J, 12) sample vs PROXY_B: kept frames must be exact; interior must not be a copy of PROXY_B."""
    s, g = np.asarray(sample, float), np.asarray(gt, float)
    if s.shape != g.shape or s.ndim != 3 or s.shape[0] != U.T_MODEL or s.shape[2] != U.FEAT:
        raise U.UniStop("FEATURE_SHAPE", f"{s.shape} / {g.shape}")
    if not (np.isfinite(s).all() and np.isfinite(g).all()):
        raise U.UniStop("RIG_PROJECTION_FAILED", "non-finite features")
    k = list(keep)
    f = [i for i in range(U.T_MODEL) if i not in k]
    d = np.abs(s - g)
    rep = {"kept_max_abs": float(d[k].max()), "groups": {}}
    for nm, idx in group_index(names).items():
        if not idx:
            continue
        rep["groups"][nm] = {"kept_max_abs": float(d[k][:, idx].max()), "interior_max_abs": float(d[f][:, idx].max()),
                             "interior_mean_abs": float(d[f][:, idx].mean()),
                             "interior_exact_equal_fraction": float((d[f][:, idx] == 0).mean())}
    rep["interior_exact_equal_fraction"] = float((d[f] == 0).mean())
    rep["interior_max_abs"] = float(d[f].max())
    return rep


KEPT_TOL = 1e-4                                                # denormalised feature units (W136 M4 tolerance)
COPY_MAX_FRACTION = 0.5                                        # interior entries exactly equal to PROXY_B (copy detector)


def check_conditioning(rep, tol=KEPT_TOL, copy_max=COPY_MAX_FRACTION):
    """Kept frames exact (every joint group) and interior not a PROXY_B copy (overall and for the left arm)."""
    if rep["kept_max_abs"] > tol:
        bad = [g for g, v in rep["groups"].items() if v["kept_max_abs"] > tol]
        raise U.UniStop("CONDITIONING_FIDELITY", f"kept frames differ by {rep['kept_max_abs']:.3e} in {bad}")
    if rep["interior_exact_equal_fraction"] > copy_max or rep["groups"]["left_arm"]["interior_exact_equal_fraction"] > copy_max:
        raise U.UniStop("INTERIOR_COPIED", f"interior equals PROXY_B on {rep['interior_exact_equal_fraction']:.2f} of entries")
    return True


# ------------------------------------------------------------------------------------------------ world mapping ---
def similarity_fit(X, Y):
    """Umeyama: Y ~ s R X + t (proper rotation, s > 0).  Returns (s, R, t, rms)."""
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    C = Yc.T @ Xc / len(X)
    Uu, S, Vt = np.linalg.svd(C)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(Uu @ Vt))
    R = Uu @ D @ Vt
    s = float(np.trace(np.diag(S) @ D) / (Xc ** 2).sum(1).mean())
    t = my - s * R @ mx
    rms = float(np.sqrt(np.mean(np.sum((X @ R.T * s + t - Y) ** 2, 1))))
    return s, R, t, rms


def apply_sim(sim, X):
    s, R, t = sim[0], np.asarray(sim[1]), np.asarray(sim[2])
    return np.asarray(X, float) @ R.T * s + t


def world_points(readback, bones, frames):
    fr = {f["blender_frame"]: f for f in readback["frames"]}
    return np.array([[fr[f]["world"][b]["head"] for b in bones] for f in frames])


def fit_world_map(model_pos, names, readback, bones, window=U.WINDOW):
    """Fit model-space (T=60, J, 3) positions to Blender world bone heads over the window; refuse a bad fit."""
    frames = list(range(window[0], window[1] + 1))
    X = np.concatenate([np.asarray(model_pos)[:, [list(names).index(b) for b in bones]].reshape(-1, 3)])
    Y = world_points(readback, bones, frames).reshape(-1, 3)
    s, R, t, rms = similarity_fit(X, Y)
    if not np.isfinite(rms) or rms > SIM_RMS_MAX_M:
        raise U.UniStop("REPRESENTATION_BLOCKED", f"world map rms {rms:.3e} m > {SIM_RMS_MAX_M}")
    return {"s": s, "R": R.tolist(), "t": t.tolist(), "rms_m": rms, "n_points": int(len(X)), "bones": list(bones)}


# ------------------------------------------------------------------------------------------------ continuity ---
def steps_mm(series):
    """(N, 3) metres -> (N-1,) step norms in mm."""
    return np.linalg.norm(np.diff(np.asarray(series, float), axis=0), axis=1) * 1e3


def flicks(series, min_mm=FLICK_MIN_MM):
    d = np.diff(np.asarray(series, float), axis=0) * 1e3
    out = []
    for i in range(len(d) - 1):
        if np.linalg.norm(d[i]) > min_mm and np.linalg.norm(d[i + 1]) > min_mm and float(np.dot(d[i], d[i + 1])) < 0:
            out.append(i + 1)
    return out


def continuity(series_by_action):
    """Series over Action 171..200 (30 points) -> entry/exit/peak steps (mm) and reversal flicks (W129 definition)."""
    pts = np.array([series_by_action[f] for f in STEP_FRAMES])
    st = steps_mm(pts)
    return {"entry_mm": float(st[0]), "exit_mm": float(st[-1]), "peak_mm": float(st.max()),
            "peak_step_to_action": int(STEP_FRAMES[int(np.argmax(st)) + 1]), "steps_mm": [float(x) for x in st],
            "n_flicks": len(flicks(pts)), "flick_at_action": [int(STEP_FRAMES[i]) for i in flicks(pts)]}


def _series(pos, j):
    """Window positions (60, J, 3) -> {Action: (3,)} for joint j."""
    return {U.index_to_action(i): np.asarray(pos)[i, j] for i in range(np.asarray(pos).shape[0])}


def attribution(raw_w, gt_w, hyb_w, names):
    """Raw (model full body, FK) vs projected (frozen PROXY_B parent + generated upperarm/lowerarm rotations, FK) vs
    PROXY_B, all already mapped to rig world metres, window frames (60, J, 3).

    Returns continuity of each view (elbow = lowerarm_l head, wrist = hand_l head, shoulder = upperarm_l head), the
    raw-to-projected elbow/wrist gap per frame, its split into shoulder translation and parent-orientation parts, and the
    raw excess step over PROXY_B's step at the same transition."""
    names = [str(n) for n in names]
    ix = {k: names.index(b) for k, b in (("shoulder", "upperarm_l"), ("elbow", "lowerarm_l"), ("wrist", "hand_l"))}
    views = {"raw": raw_w, "projected": hyb_w, "proxyB": gt_w}
    cont = {v: {k: continuity(_series(p, ix[k])) for k in ix} for v, p in views.items()}
    fr = list(range(U.INTERIOR[0], U.INTERIOR[1] + 1))
    gap = {}
    for k in ("elbow", "wrist"):
        d_tot, d_sh, d_rot = [], [], []
        for f in fr:
            i = U.action_to_index(f)
            r, h = raw_w[i, ix[k]], hyb_w[i, ix[k]]
            rs, hs = raw_w[i, ix["shoulder"]], hyb_w[i, ix["shoulder"]]
            d_tot.append(np.linalg.norm(r - h) * 1e3)
            d_sh.append(np.linalg.norm(rs - hs) * 1e3)
            d_rot.append(np.linalg.norm((r - rs) - (h - hs)) * 1e3)
        gap[k] = {"raw_minus_projected_max_mm": float(max(d_tot)), "raw_minus_projected_mean_mm": float(np.mean(d_tot)),
                  "shoulder_translation_part_max_mm": float(max(d_sh)), "parent_orientation_part_max_mm": float(max(d_rot)),
                  "per_frame_mm": {str(f): float(v) for f, v in zip(fr, d_tot)}}
    excess = {}
    for k in ("elbow", "wrist"):
        e = np.array(cont["raw"][k]["steps_mm"]) - np.array(cont["proxyB"][k]["steps_mm"])
        excess[k] = {"max_excess_mm": float(e.max()), "at_step_to_action": int(STEP_FRAMES[int(np.argmax(e)) + 1])}
    # shoulder-relative vectors: the projected one equals the raw one in the raw clavicle frame (same local arm rotations),
    # so its world difference is purely the parent orientation; recorded per view for the full picture.
    return {"continuity": cont, "raw_vs_projected": gap, "raw_excess_step_over_proxyB": excess}


def body_report(raw_w, gt_w, names):
    """Raw full-body motion vs PROXY_B per joint group: interior deviation, entry/exit/peak world steps and excess."""
    out = {}
    fi = [U.action_to_index(f) for f in range(U.INTERIOR[0], U.INTERIOR[1] + 1)]
    si = [U.action_to_index(f) for f in STEP_FRAMES]
    for nm, idx in group_index(names).items():
        if not idx:
            continue
        dev = np.linalg.norm(raw_w[fi][:, idx] - gt_w[fi][:, idx], axis=-1) * 1e3
        sr = np.linalg.norm(np.diff(raw_w[si][:, idx], axis=0), axis=-1) * 1e3          # (29, n)
        sg = np.linalg.norm(np.diff(gt_w[si][:, idx], axis=0), axis=-1) * 1e3
        out[nm] = {"interior_dev_max_mm": float(dev.max()), "interior_dev_mean_mm": float(dev.mean()),
                   "raw_entry_step_max_mm": float(sr[0].max()), "proxyB_entry_step_max_mm": float(sg[0].max()),
                   "raw_exit_step_max_mm": float(sr[-1].max()), "proxyB_exit_step_max_mm": float(sg[-1].max()),
                   "raw_peak_step_mm": float(sr.max()), "proxyB_peak_step_mm": float(sg.max()),
                   "boundary_excess_max_mm": float(max((sr[0] - sg[0]).max(), (sr[-1] - sg[-1]).max())),
                   "excess_max_mm": float((sr - sg).max())}
    return out


def raw_unusable(att, body):
    """Declared raw-continuity rule: any raw world elbow/wrist step exceeding PROXY_B's step at the same transition by
    > 100 mm, or any joint group's raw entry/exit step exceeding PROXY_B's by > 100 mm."""
    reasons = []
    for k in ("elbow", "wrist"):
        if att["raw_excess_step_over_proxyB"][k]["max_excess_mm"] > UNUSABLE_STEP_MM:
            reasons.append(f"raw {k} excess step {att['raw_excess_step_over_proxyB'][k]['max_excess_mm']:.1f} mm")
    for g, v in body.items():
        if v["boundary_excess_max_mm"] > UNUSABLE_STEP_MM:
            reasons.append(f"raw {g} boundary excess {v['boundary_excess_max_mm']:.1f} mm")
    return {"unusable": bool(reasons), "reasons": reasons}


def hybrid_features(gt, sample, names, parents, joints=U.EDIT_CHANNELS):
    """PROXY_B features with ONLY the 6D slots that carry the bvh rotation of ``joints`` (their single child's slot)
    taken from the sample: the frozen-parent projection expressed in feature space (for FK only, never imported)."""
    h = np.array(gt, float, copy=True)
    ch = U.children(parents)
    for b in joints:
        c = ch[[str(n) for n in names].index(b)]
        if len(c) != 1:
            raise U.UniStop("REPRESENTATION_BLOCKED", f"{b} has {len(c)} children")
        h[:, c[0], 3:9] = np.asarray(sample, float)[:, c[0], 3:9]
    return h
