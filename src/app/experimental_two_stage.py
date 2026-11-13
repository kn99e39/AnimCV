"""W139 (EXPERIMENTAL, opt-in, scratch only): Two-Stage Generative Animation feasibility -- pure numpy instruments.

Stage 1 = frozen W129 MotionBricks left-arm completion (no re-run).  Bridge = W136 world-pose-preserving export +
official UniMate rig_preprocess + official features.  Stage 2 = frozen UniMate v3 through the scheduled-replacement
sampler (``app.experimental_two_stage_sampler``; flow-SDEdit at t_s = ts[K_S] on the W136 left-arm free set).

Nothing here touches Production code, the canonical Action lineage, the exporter, CLI or GUI.  Every function is
deterministic numpy; the torch sampler lives in ``experimental_two_stage_sampler``.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

import app.experimental_unimate as U

SCHEMA = "animcv_exp_w139_two_stage_v1"
N_STEPS = 50                      # official replacement sampler default (inbetween_sample_ode num_steps)
K_S = 25                          # scheduled pin through produced state 25 -> t_s = ts[25] = 0.5 (logit-normal(0,1) mode)
SEED_COMPLETION = 10              # the published v3 config seed (W136/W137); every Stage-2 completion run
SEED_NATIVE_REF = 11              # model-native reference ONLY (independent noise; never used for a completion)
N_VALID = U.N_JOINTS              # 50 own-rig joints (49 BaseRig bones + adapter pelvis); padding rows 50..70
MAX_J = 71
N_FREE_W136 = 5424                # 16 joints x (12 x 28 + 3)
SLOT_UPPER, SLOT_LOWER = 19, 21   # HML slots holding bvh(upperarm_l), bvh(lowerarm_l)

# ---- prospective thresholds (declared before any W139 model output) ----
GAMMA_MIN = 0.75                  # I2: network clean-data estimate at the first free step keeps >= 75 % of the Stage-1 difference
NET_RATIO_MIN = 0.25              # I3: ||network term|| >= 0.25 ||carry-through|| on G
NET_MM_MIN = 10.0                 # I3: decoded |out_C - null_C| world wrist/elbow max >= 10 mm
NOISE_FACTOR = 10.0               # every effect threshold must also exceed 10 x the measured replicate noise floor
CROSS_TOL = 1e-6                  # cross-run state difference vs t * dx1 (float32 rounding)
K0_PAIR_TOL = 1e-5                # in-W139 K0: k_s=0 on PROXY_B vs PROXY_C, normalised valid entries (primary)
K0_W136_TOL = 1e-4                # secondary: K0 on PROXY_B vs frozen W136 U raw output (normalised)
K0_W136_MM = 1.0                  # secondary: decoded wrist/elbow
UNUSABLE_STEP_MM = 100.0          # W136/W137 unusable-continuity diagnostic
P2_DROP_MM = 100.0                # intent P2: min interior wrist z >= z(171) - 100 mm
FIDELITY_MARGIN_MM = 20.0         # superiority on GT-free source-derived agreement (mean wrist deviation from A)
FLICK_MIN_MM = 20.0
X1_OFFLINE_TOL = 1e-5            # offline float64 bridge normalisation vs the official float32 tensor
EPS_CORR_MAX = 0.03               # native reference noise independence: |corr| over 36,000 valid eps entries (~5.7 sigma)
JUDGMENTS = ("BLOCKED", "NOT_YET_SUPPORTED", "PARTIALLY_SUPPORTED", "SUPPORTED")


class TwoStageStop(U.UniStop):
    """W139 refusal; a UniStop subclass so the shared W136 control helper records its code."""


# ------------------------------------------------------------------------------------------------ masks ---
def validate_k_s(k_s, n_steps=N_STEPS):
    if not isinstance(k_s, (int, np.integer)) or isinstance(k_s, (bool, np.bool_)) or not 0 <= int(k_s) <= n_steps:
        raise TwoStageStop("SCHEDULE", f"k_s must be an int in [0, {n_steps}], got {k_s!r}")
    return int(k_s)


def sched_set(keep, n_valid=N_VALID):
    """G = complement of the keep mask restricted to VALID joints (padding rows stay exactly as in the official mask)."""
    keep = np.asarray(keep, bool)
    if keep.ndim != 3 or keep.shape[1] != U.FEAT:
        raise TwoStageStop("MASK_SHAPE", f"{keep.shape}")
    g = ~keep
    g[n_valid:] = False
    return g


def state_mask(keep, g, j, k_s):
    """Mask applied to produced state j (integer state index; never a float t comparison)."""
    return (np.asarray(keep, bool) | np.asarray(g, bool)) if j <= k_s else np.asarray(keep, bool)


def check_w136_free_set(keep, names, parents):
    """The scheduled set must be exactly W136's validated free set (5424 valid entries)."""
    info = U.validate_mask(keep, names, parents)
    g = sched_set(keep)
    if int(g.sum()) != N_FREE_W136 or info["n_free"] != N_FREE_W136:
        raise TwoStageStop("MASK_DOMAIN", f"free set {int(g.sum())} != {N_FREE_W136}")
    return {"n_sched": int(g.sum()), **info}


def frame_mask(keep_frames, n=U.T_MODEL):
    m = np.zeros((1, 1, 1, n), bool)
    m[..., list(keep_frames)] = True
    return m


# ------------------------------------------------------------------------------------------------ trajectory checks ---
def interpolant(eps, x1, t):
    return ((1.0 - np.float32(t)) * eps + np.float32(t) * x1).astype(np.float32)


def check_recorded_states(states, eps, x1, keep_valid, g_valid, k_s, ts):
    """Each run: every recorded state j<=k_s equals the interpolant on keep|G bit-exactly; j>k_s on keep only.
    ``states`` (S,1,J,12,T), ``eps``/``x1`` (1,J,12,T) float32 valid-joint slices; masks (J,12,T)."""
    S = len(states)
    bad = []
    for j in range(S):
        it = interpolant(eps, x1, ts[j])
        m = (keep_valid | g_valid) if j <= k_s else keep_valid
        if not np.array_equal(states[j][0][m], it[0][m]):
            bad.append(j)
    return {"n_states": S, "states_not_bit_equal_interpolant": bad, "ok": not bad}


def _dot(a, b):
    return float(np.sum(np.asarray(a, np.float64) * np.asarray(b, np.float64)))


def transfer_decomposition(recB, recC, x1B, x1C, g_valid, k_s=K_S):
    """Decompose the Stage-2 output difference C - B on G into carry-through and network term (feature space, normalised).

    recX: dict with 'states' (51,1,J,12,T), 'velocities' (50,1,J,12,T), 't' (51,), 'eps' (1,J,12,T) -- valid joints.
    Same eps is required (gate checked by the caller via eps sha).  On G:
        dstate_ks = t_s * dx1                         (carry-through P, by construction)
        dout      = dstate_ks + sum_{i>=k_s} dt_i dv_i (network term N)
    beta  = <dout, dx1>/|dx1|^2           (passive state-blind network -> t_s; regenerate-from-context -> ~0; copy -> 1)
    gamma = <dx1hat_ks, dx1>/|dx1|^2  with x1hat_i = x_i + (1 - t_i) v_i (network's own clean-data estimate)
            (state-blind -> t_s; perfect denoiser of the Stage-1 signal -> 1; context-only regeneration -> ~0)."""
    g = np.asarray(g_valid, bool)
    t = np.asarray(recB["t"], np.float64)
    sB, sC = recB["states"][:, 0], recC["states"][:, 0]
    vB, vC = recB["velocities"][:, 0], recC["velocities"][:, 0]
    dx1 = (np.asarray(x1C, np.float64) - np.asarray(x1B, np.float64))[0][g]
    n2 = _dot(dx1, dx1)
    if n2 <= 0:
        raise TwoStageStop("NO_STAGE1_DIFFERENCE", "x1_C == x1_B on G")
    t_s = float(t[k_s])
    d_state_ks = (sC[k_s].astype(np.float64) - sB[k_s].astype(np.float64))[g]
    P = t_s * dx1
    dout = (sC[-1].astype(np.float64) - sB[-1].astype(np.float64))[g]
    N = dout - d_state_ks
    dt = np.diff(t)
    N_sum = sum(dt[i] * (vC[i].astype(np.float64) - vB[i].astype(np.float64))[g] for i in range(k_s, len(dt)))

    def x1hat(s, v, i):
        return (s[i].astype(np.float64) + (1.0 - t[i]) * v[i].astype(np.float64))[g]
    gam = {str(i): _dot(x1hat(sC, vC, i) - x1hat(sB, vB, i), dx1) / n2 for i in sorted({k_s, min(k_s + 5, 49), 40, 49})}
    disc = [float(np.abs((vC[i] - vB[i])[g]).max()) for i in range(0, k_s)]   # discarded evaluations (still recorded)
    nP, nN = float(np.linalg.norm(P)), float(np.linalg.norm(N))
    return {"t_s": t_s, "n_G": int(g.sum()), "dx1_norm": float(np.sqrt(n2)),
            "carry_through_check_max_abs": float(np.abs(d_state_ks - P).max()),
            "network_term_sum_check_max_abs": float(np.abs(N - N_sum).max()),
            "carry_norm": nP, "network_norm": nN, "network_over_carry": nN / nP if nP > 0 else float("inf"),
            "beta": _dot(dout, dx1) / n2, "gamma": gam, "gamma_first_free": gam[str(k_s)],
            "cos_network_dx1": _dot(N, dx1) / (nN * np.sqrt(n2)) if nN > 0 else 0.0,
            "discarded_steps": list(range(0, k_s)), "discarded_step_velocity_diff_max": max(disc) if disc else 0.0,
            "passive_null": {"beta": t_s, "gamma": t_s}}


def null_output(outB, x1B, x1C, g_valid, t_s):
    """Inert-network null for C: B's output plus the passive carry-through t_s*(x1C - x1B) on G."""
    out = np.array(outB, np.float32, copy=True)
    g = np.asarray(g_valid, bool)
    out[0][g] = (np.asarray(outB, np.float64)[0][g] + t_s * (np.asarray(x1C, np.float64) - np.asarray(x1B, np.float64))[0][g]).astype(np.float32)
    return out


def injected_signal_ratio(x1B, x1C, g_valid, t_s=0.5, slots=(SLOT_UPPER, SLOT_LOWER)):
    """PRE-OUTPUT power check: ||t_s dx1|| vs expected ||(1-t_s) eps|| (eps ~ N(0,1)) on G and on the imported 6D slots."""
    g = np.asarray(g_valid, bool)
    d = (np.asarray(x1C, np.float64) - np.asarray(x1B, np.float64))[0]
    out = {"G": float(t_s * np.linalg.norm(d[g]) / ((1 - t_s) * np.sqrt(g.sum())))}
    for s in slots:
        m = np.zeros_like(g)
        m[s, 3:9] = g[s, 3:9]
        out[f"slot{s}_rot6d"] = float(t_s * np.linalg.norm(d[m]) / ((1 - t_s) * np.sqrt(max(m.sum(), 1))))
    return out


# ------------------------------------------------------------------------------------------------ decision rules ---
def information_path(i1, decomposition, decoded_net_mm, noise_floor_mm, noise_floor_feat):
    """I1 implementation visibility (non-discarded evaluation), I2 learned reading, I3 network response on the output."""
    gam = decomposition["gamma_first_free"]
    i2 = bool(gam >= GAMMA_MIN)
    floor_mm = NOISE_FACTOR * float(noise_floor_mm)
    floor_feat = NOISE_FACTOR * float(noise_floor_feat)
    i3 = bool(decomposition["network_over_carry"] >= NET_RATIO_MIN and decoded_net_mm >= NET_MM_MIN and decoded_net_mm >= floor_mm
              and decomposition["network_norm"] >= floor_feat * np.sqrt(decomposition["n_G"]))
    return {"I1": bool(i1), "I2": i2, "I3": i3, "learned_path": bool(i1 and i2 and i3), "gamma_first_free": gam,
            "network_over_carry": decomposition["network_over_carry"], "decoded_network_term_mm": decoded_net_mm,
            "noise_floor_mm_x10": floor_mm, "noise_floor_feat_x10_per_entry": floor_feat,
            "direction": ("AMPLIFIES_STAGE1" if decomposition["cos_network_dx1"] > 0.1 else
                          "WASHES_OUT_STAGE1" if decomposition["cos_network_dx1"] < -0.1 else "ORTHOGONAL")}


def intent_metrics(world_by_frame, interior=U.INTERIOR):
    """Camera-free press-intent terms from a fresh world readback {frame: {'world': {bone: {'head': xyz}}}} (metres).
    P1: hand_l head above upperarm_l head on every interior frame.  P2: min interior wrist z >= z(171) - 100 mm."""
    i0, i1 = interior
    dz = [(world_by_frame[f]["world"]["hand_l"]["head"][2] - world_by_frame[f]["world"]["upperarm_l"]["head"][2]) * 1e3 for f in range(i0, i1 + 1)]
    wz = [world_by_frame[f]["world"]["hand_l"]["head"][2] * 1e3 for f in range(i0, i1 + 1)]
    z171 = world_by_frame[i0 - 1]["world"]["hand_l"]["head"][2] * 1e3
    p1_frac = float(np.mean(np.asarray(dz) > 0))
    return {"P1_frac_wrist_above_shoulder": p1_frac, "P1_min_dz_mm": float(min(dz)), "P1": bool(p1_frac == 1.0),
            "P2_min_wrist_z_minus_z171_mm": float(min(wz) - z171), "P2": bool(min(wz) - z171 >= -P2_DROP_MM)}


def max_steps(q):
    al, el = q["arm_local"], q["elbow"]
    return {"wrist_entry": al["boundary_c0"]["entry_step_mm"], "wrist_exit": al["boundary_c0"]["exit_step_mm"], "wrist_peak": al["peak_step_mm"],
            "elbow_entry": el["elbow_entry_step_mm"], "elbow_exit": el["elbow_exit_step_mm"], "elbow_peak": el["elbow_peak_step_mm"],
            "wrist_flicks": int(al["n_reversal_flicks"]), "elbow_flicks": len(el["elbow_reversal_flicks"])}


def usability_floor(q):
    s = max_steps(q)
    return bool(max(s["wrist_peak"], s["elbow_peak"]) <= UNUSABLE_STEP_MM and s["wrist_flicks"] == 0 and s["elbow_flicks"] == 0
                and not q["near_straight_frames"] and not q["hinge_opposite_frames"])


def non_inferior(q, ref):
    """Q1/Q3 form: every entry/exit/peak step (arm-local wrist, elbow) <= 2 x ref + 10 mm."""
    s, r = max_steps(q), max_steps(ref)
    return bool(all(s[k] <= 2 * r[k] + 10.0 for k in ("wrist_entry", "wrist_exit", "wrist_peak", "elbow_entry", "elbow_exit", "elbow_peak")))


def acceptable(q, intent, ref_slerp):
    return bool(usability_floor(q) and intent["P1"] and intent["P2"] and non_inferior(q, ref_slerp))


def superiority(cand, others):
    """cand/others: {'q','intent','fidelity_dev_A_mean_mm','acceptable'}.  S2C is metric-superior to X iff S2C is acceptable
    and (X is not acceptable, or S2C's mean wrist deviation from A is >= 20 mm lower)."""
    out = {}
    for k, o in others.items():
        out[k] = bool(cand["acceptable"] and (not o["acceptable"] or cand["fidelity_dev_A_mean_mm"] <= o["fidelity_dev_A_mean_mm"] - FIDELITY_MARGIN_MM))
    return {"per_control": out, "all": bool(out and all(out.values()))}


def judgment(prereq_ok, controls_ok, info, sup_all, human_native_rate_review):
    """Prospective hypothesis judgment (distinct from test PASS)."""
    if not (prereq_ok and controls_ok):
        return {"judgment": "BLOCKED", "sub": "prerequisite / mechanism control failed"}
    if not info["learned_path"]:
        return {"judgment": "NOT_YET_SUPPORTED", "sub": "no learned Stage-1 information path beyond passive carry-through at t_s=0.5 (this sampler only)"}
    if sup_all and human_native_rate_review:
        return {"judgment": "SUPPORTED", "sub": "learned path + metric superiority + human native-rate review"}
    return {"judgment": "PARTIALLY_SUPPORTED",
            "sub": ("METRIC_SUPERIORITY_INDICATED; perceptual acceptance PENDING (no human native-rate review)" if sup_all
                    else "learned information path shown; final quality superiority unproven/absent")}


# ------------------------------------------------------------------------------------------------ quaternions ---
def qslerp(q0, q1, w):
    q0, q1 = U.qnorm(np.asarray(q0, float)), U.qnorm(np.asarray(q1, float))
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        return U.qnorm(q0 + w * (q1 - q0))
    th = np.arccos(np.clip(d, -1, 1))
    return (np.sin((1 - w) * th) * q0 + np.sin(w * th) * q1) / np.sin(th)


def xyzw_to_wxyz(q):
    q = np.asarray(q, float)
    return np.concatenate([q[..., 3:4], q[..., :3]], axis=-1)


def undo_endpoint_correction(corrected_xyzw, start_delta_xyzw, end_delta_xyzw):
    """Invert MotionBricks' endpoint correction for one joint's local rotations (kernels.py apply_endpoint_deltas_kernel):
       q_corr[f] = q_raw[f] * slerp(1, dS, 1 - w_f) * slerp(1, dE, w_f),  w_f = f / (n - 1).
    Returns raw (n,4) wxyz and the per-frame correction quaternion (wxyz)."""
    qc = xyzw_to_wxyz(corrected_xyzw)
    dS, dE = xyzw_to_wxyz(start_delta_xyzw), xyzw_to_wxyz(end_delta_xyzw)
    n = len(qc)
    one = np.array([1.0, 0, 0, 0])
    raw, corr = [], []
    for f in range(n):
        w = f / (n - 1)
        c = U.qmul(qslerp(one, dS, 1 - w), qslerp(one, dE, w))
        corr.append(U.qnorm(c))
        raw.append(U.qnorm(U.qmul(qc[f], U.qinv(c))))
    return np.asarray(raw), np.asarray(corr)


def qangle_deg(a, b):
    d = np.abs(np.sum(U.qnorm(np.asarray(a, float)) * U.qnorm(np.asarray(b, float)), -1))
    return np.degrees(2 * np.arccos(np.clip(d, 0, 1)))


# ------------------------------------------------------------------------------------------------ native control ---
def poison_hidden(feats, g_valid):
    """Copy of (T,J,12) features with every hidden (G) entry set to NaN -- proves an init builder never reads G."""
    f = np.array(feats, np.float64, copy=True)
    gT = np.transpose(np.asarray(g_valid, bool)[:f.shape[1]], (2, 0, 1))
    f[gT] = np.nan
    return f


def slerp_hidden_rotations(bvh_wxyz, hidden_joints, f0, f1):
    """Per-joint slerp of local (T-pose-rebased bvh) rotations between kept window frames f0 and f1 for hidden joints.
    Reads only frames f0 and f1 of those joints (refuses NaN there)."""
    q = np.array(bvh_wxyz, float, copy=True)
    for j in hidden_joints:
        a, b = q[f0, j], q[f1, j]
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            raise TwoStageStop("HIDDEN_READ", f"boundary rotation of joint {j} not available")
        for f in range(f0 + 1, f1):
            q[f, j] = qslerp(a, b, (f - f0) / (f1 - f0))
    return q


def arm_errors(pred_w, ref_w, names, frames):
    """World/model-space elbow (lowerarm_l head) and wrist (hand_l head) errors (same units as input x 1e3)."""
    e = {}
    for nm, b in (("elbow", "lowerarm_l"), ("wrist", "hand_l")):
        j = names.index(b)
        d = np.linalg.norm(np.asarray(pred_w)[frames, j] - np.asarray(ref_w)[frames, j], axis=-1) * 1e3
        e[nm] = {"mean": float(d.mean()), "max": float(d.max())}
    return e


# ------------------------------------------------------------------------------------------------ misc ---
def blind_key(seed_text, labels):
    """Deterministic label permutation keyed by a hash only known after the run (e.g. run_result sha)."""
    h = hashlib.sha256(seed_text.encode()).digest()
    order = sorted(labels, key=lambda s: hashlib.sha256(h + s.encode()).hexdigest())
    letters = [chr(ord("P") + i) for i in range(len(order))]
    return dict(zip(letters, order))


def sha_json(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()
