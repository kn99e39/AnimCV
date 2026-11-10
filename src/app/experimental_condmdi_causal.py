"""EXPERIMENTAL (opt-in, Batch 132) CondMDI conditioning-mechanism causal gate: pure-numpy masks, metrics and the
prospectively frozen decision rules.  Never imported by the default CLI/GUI path; W131 code is reused unchanged.

* ``official_feature_mask`` / ``official_middle_mask``: numpy replicas of the official Motius/CondMDI
  ``joint_mask_to_feature_mask`` / ``build_observation_mask(mode="middle")`` (parity-checked against the official
  function in the scratch env by the harness).  NOTE the official semantics: the pelvis ``pos`` group is root xz + height
  (features 1..3), the pelvis ``rot`` group is root yaw (feature 0), and the four foot-contact features are ALWAYS added
  for the ankle/foot joints, whatever the feature mode.
* model-space metrics on absolute HML263 (root-relative heading-frame ``ric`` positions and canonical world joints);
* frozen rules: S1 flag isolation, observed-context gate, mechanism classification, candidate QA and CASE precedence.
"""
from __future__ import annotations

import numpy as np

from app.experimental_condmdi import abs_to_joints, feature_owner

SCHEMA = "animcv_exp_w132_condmdi_causal_v1"
N_FEAT = 263
IMPUTE_KEYS = frozenset({"imputate", "stop_imputation_at", "replacement_distribution", "inpainted_motion", "inpainting_mask"})
L_ELBOW, L_WRIST = 18, 20

# ------------------------------------------------------------------------------------------- frozen thresholds ---
CTX_ARM_WINDOW_MAX_M = 0.050          # left elbow / wrist ric error at observed samples 103..113 and 133..143
CTX_OBS_MEAN_M = 0.050                # mean ric error over every observed joint-sample
CTX_ROOT_HEIGHT_MAX_M = 0.050         # max |root height error| over observed samples
CTX_ROOT_YAW_MAX_RAD = float(np.radians(5.0))
MECH_RATIO = 0.5                      # "removed" = R1 boundary jump <= 0.5 x R0's
REF_DISCONTINUITY_MM = 50.0           # reference cell boundary excess above the reference's own step
S1_BOUNDARY = {"last_obs_before": 113, "first_gen": 114, "last_gen": 132, "first_obs_after": 133}
S1_CTX_WINDOWS = (tuple(range(103, 114)), tuple(range(133, 144)))


# ------------------------------------------------------------------------------------------- official masks ---
def _feature_matrices():
    pos = np.zeros((22, N_FEAT), bool)
    pos[0, 1:4] = True
    for j in range(1, 22):
        pos[j, 4 + 3 * (j - 1): 4 + 3 * j] = True
    rot = np.zeros((22, N_FEAT), bool)
    rot[0, 0] = True
    for j in range(1, 22):
        rot[j, 67 + 6 * (j - 1): 67 + 6 * j] = True
    vel = np.zeros((22, N_FEAT), bool)
    for j in range(22):
        vel[j, 193 + 3 * j: 193 + 3 * (j + 1)] = True
    con = np.zeros((22, N_FEAT), bool)
    con[7, -4] = con[10, -3] = con[8, -2] = con[11, -1] = True
    return pos, rot, vel, con


MAT_POS, MAT_ROT, MAT_VEL, MAT_CONTACT = _feature_matrices()


def official_feature_mask(joint_mask, feature_mode="pos_rot_vel"):
    """[22, T] joint mask -> [263, T] feature mask with the official grouping (contact always included)."""
    if feature_mode not in ("pos", "pos_rot", "pos_rot_vel"):
        raise ValueError(feature_mode)
    mats = [MAT_POS, MAT_CONTACT] + ([MAT_ROT] if feature_mode != "pos" else []) + ([MAT_VEL] if feature_mode == "pos_rot_vel" else [])
    mapping = np.logical_or.reduce(mats)
    return (np.asarray(joint_mask, float).T @ mapping.astype(float)).T > 0


def official_middle_frames(length, width):
    """Official ``mode="middle"``: observed frames [0, start) U [start+width, length), start = (length-width)//2."""
    width = min(length, int(width))
    start = (length - width) // 2
    hidden = list(range(start, start + width))
    observed = list(range(0, start)) + list(range(start + width, length))
    return observed, hidden


def official_middle_mask(length, width, feature_mode, joints=range(22)):
    observed, _ = official_middle_frames(length, width)
    jm = np.zeros((22, length), bool)
    for j in joints:
        jm[j, observed] = True
    return official_feature_mask(jm, feature_mode)


def mask_group_census(mask):
    """Observed feature count per group (root_yaw / root_xz / root_height / pos / rot / vel / contact)."""
    out = {}
    for f in range(mask.shape[0]):
        g, _ = feature_owner(f)
        out[g] = out.get(g, 0) + int(np.asarray(mask[f]).sum())
    return out


# ------------------------------------------------------------------------------------------- model-space geometry ---
def ric(absm):
    """Root-relative heading-frame joint positions [T,22,3] (root = (0, height, 0)); what the position mask observes."""
    absm = np.asarray(absm, float)
    T = len(absm)
    out = np.zeros((T, 22, 3))
    out[:, 0, 1] = absm[:, 3]
    out[:, 1:] = absm[:, 4:67].reshape(T, 21, 3)
    return out


def steps(P, k):
    """|P[k+1] - P[k]| per joint, metres."""
    return np.linalg.norm(P[k + 1] - P[k], axis=-1)


def s1_boundary(out_abs, in_abs, mask):
    """Left elbow / wrist model-space boundary numbers for one S1 output (metres; ric and canonical world)."""
    b = S1_BOUNDARY
    R, Ri = ric(out_abs), ric(in_abs)
    W, Wi = abs_to_joints(out_abs), abs_to_joints(in_abs)
    res = {}
    for name, j in (("left_wrist", L_WRIST), ("left_elbow", L_ELBOW)):
        r = {}
        for space, P, Pi in (("ric", R, Ri), ("world", W, Wi)):
            r[space] = {
                "internal_entry_m": float(np.linalg.norm(P[b["first_gen"], j] - P[b["last_obs_before"], j])),
                "internal_exit_m": float(np.linalg.norm(P[b["first_obs_after"], j] - P[b["last_gen"], j])),
                "effective_entry_m": float(np.linalg.norm(P[b["first_gen"], j] - Pi[b["last_obs_before"], j])),
                "effective_exit_m": float(np.linalg.norm(Pi[b["first_obs_after"], j] - P[b["last_gen"], j])),
                "context_err_113_m": float(np.linalg.norm(P[b["last_obs_before"], j] - Pi[b["last_obs_before"], j])),
                "context_err_133_m": float(np.linalg.norm(P[b["first_obs_after"], j] - Pi[b["first_obs_after"], j])),
                "source_step_112_113_m": float(np.linalg.norm(Pi[113, j] - Pi[112, j])),
                "source_step_133_134_m": float(np.linalg.norm(Pi[134, j] - Pi[133, j])),
            }
        res[name] = r
    return res


def observed_residuals(out_abs, in_abs, mask):
    """Observed-context residuals of an output against its own conditioning input (absolute-space features)."""
    out_abs, in_abs, mask = np.asarray(out_abs, float), np.asarray(in_abs, float), np.asarray(mask, bool)
    T = len(out_abs)
    d = np.abs(out_abs - in_abs)
    res = {"max_abs_feature_err": float(d[mask.T].max()) if mask.any() else 0.0}
    obs_samples = np.nonzero(mask[0])[0] if mask[0].any() else np.arange(0)
    res["root_yaw_max_rad"] = float(d[obs_samples, 0].max()) if len(obs_samples) else None
    hs = np.nonzero(mask[3])[0]
    res["root_height_max_m"] = float(d[hs, 3].max()) if len(hs) else None
    R, Ri = ric(out_abs), ric(in_abs)
    e = np.linalg.norm(R - Ri, axis=-1)                       # [T,22]
    obs_js = np.zeros((T, 22), bool)
    for j in range(1, 22):
        obs_js[:, j] = mask[4 + 3 * (j - 1)]
    res["obs_joint_samples"] = int(obs_js.sum())
    res["obs_joint_ric_err_mean_m"] = float(e[obs_js].mean()) if obs_js.any() else None
    res["obs_joint_ric_err_max_m"] = float(e[obs_js].max()) if obs_js.any() else None
    per_joint = {}
    for j in range(1, 22):
        if obs_js[:, j].any():
            per_joint[str(j)] = {"mean_m": float(e[obs_js[:, j], j].mean()), "max_m": float(e[obs_js[:, j], j].max())}
    res["per_joint_ric_err"] = per_joint
    win = {}
    for j in (L_ELBOW, L_WRIST):
        for w in S1_CTX_WINDOWS:
            ks = [k for k in w if obs_js[k, j]]
            if ks:
                win[f"{j}_{w[0]}_{w[-1]}"] = float(e[ks, j].max())
    res["arm_window_ric_err_max_m"] = max(win.values()) if win else None
    res["arm_window_ric_err"] = win
    return res


def trajectory_summary(out_abs, interior=range(114, 133), window=range(104, 143)):
    """Raw left-wrist trajectory, timing and root travel of an S1 output (model space)."""
    R, W = ric(out_abs), abs_to_joints(out_abs)
    interior = list(interior)
    h = R[:, L_WRIST, 1]
    root = W[:, 0, [0, 2]]
    return {"left_wrist_height_ric_m": {str(k): float(h[k]) for k in window},
            "left_wrist_minus_shoulder_height_m": {str(k): float(R[k, L_WRIST, 1] - R[k, 16, 1]) for k in window},
            "interior_peak_height_sample": int(interior[int(np.argmax(h[interior]))]),
            "interior_height_at": {str(k): float(h[k]) for k in (interior[0], interior[len(interior) // 2], interior[-1])},
            "interior_max_step_m": float(max(np.linalg.norm(R[k + 1, L_WRIST] - R[k, L_WRIST]) for k in interior[:-1])),
            "root_travel_net_m": float(np.linalg.norm(root[-1] - root[0])),
            "root_path_m": float(np.sum(np.linalg.norm(np.diff(root, axis=0), axis=1))),
            "root_yaw_range_rad": float(np.ptp(np.asarray(out_abs)[:, 0]))}


# ------------------------------------------------------------------------------------------- reference control ---
def refctl_metrics(out_abs, ref_abs, mask, hidden):
    """Score one reference-control cell against the known complete reference (absolute HML263)."""
    out_abs, ref_abs, mask = np.asarray(out_abs, float), np.asarray(ref_abs, float), np.asarray(mask, bool)
    hidden = list(hidden)
    k0, k1 = hidden[0] - 1, hidden[-1]                         # boundary steps k0->k0+1 (entry), k1->k1+1 (exit)
    res = {}
    for space, P, Q in (("ric", ric(out_abs), ric(ref_abs)), ("world", abs_to_joints(out_abs), abs_to_joints(ref_abs))):
        e = np.linalg.norm(P - Q, axis=-1)
        bd = {}
        for nm, k in (("entry", k0), ("exit", k1)):
            so, sr = steps(P, k), steps(Q, k)
            bd[nm] = {"max_excess_mm": float((so - sr).max() * 1e3), "argmax_joint": int(np.argmax(so - sr)),
                      "max_out_step_mm": float(so.max() * 1e3), "max_ref_step_mm": float(sr.max() * 1e3)}
        res[space] = {"hidden_mpjpe_mm": float(e[hidden].mean() * 1e3), "hidden_max_mm": float(e[hidden].max() * 1e3),
                      "endpoint_mpjpe_mm": {"first_hidden": float(e[hidden[0]].mean() * 1e3), "last_hidden": float(e[hidden[-1]].mean() * 1e3)},
                      "observed_frames_mpjpe_mm": float(np.delete(e, hidden, axis=0).mean() * 1e3),
                      "boundary": bd, "boundary_max_excess_mm": max(bd["entry"]["max_excess_mm"], bd["exit"]["max_excess_mm"])}
    d = np.abs(out_abs - ref_abs)
    grp = {}
    for f in range(N_FEAT):
        g, _ = feature_owner(f)
        m = mask[f]
        if m.any():
            grp[g] = max(grp.get(g, 0.0), float(d[m, f].max()))
    res["observed_feature_residual_max_by_group"] = grp
    res["observed_feature_residual_max"] = float(d[mask.T].max()) if mask.any() else 0.0
    return res


def refctl_readings(cells):
    """Descriptive, prospectively fixed readings over the four cells (keys i, ii, iii, iv; ric boundary excess)."""
    ex = {k: cells[k]["ric"]["boundary_max_excess_mm"] for k in ("i", "ii", "iii", "iv")}
    disc = {k: v > REF_DISCONTINUITY_MM for k, v in ex.items()}
    return {"boundary_max_excess_mm_ric": ex, "discontinuity_present": disc,
            "replacement_effect_pos_rot_vel": "ON>OFF" if disc["i"] and not disc["iii"] else ("OFF>ON" if disc["iii"] and not disc["i"] else "no class change"),
            "replacement_effect_pos": "ON>OFF" if disc["ii"] and not disc["iv"] else ("OFF>ON" if disc["iv"] and not disc["ii"] else "no class change"),
            "feature_group_effect_imputeON": "pos>pos_rot_vel" if disc["ii"] and not disc["i"] else ("pos_rot_vel>pos" if disc["i"] and not disc["ii"] else "no class change"),
            "feature_group_effect_imputeOFF": "pos>pos_rot_vel" if disc["iv"] and not disc["iii"] else ("pos_rot_vel>pos" if disc["iii"] and not disc["iv"] else "no class change"),
            "hidden_mpjpe_world_mm": {k: cells[k]["world"]["hidden_mpjpe_mm"] for k in ("i", "ii", "iii", "iv")},
            "observed_feature_residual_max": {k: cells[k]["observed_feature_residual_max"] for k in ("i", "ii", "iii", "iv")}}


# ------------------------------------------------------------------------------------------- frozen rules ---
def flag_isolation(on, off, io_on, io_off, n_steps):
    """ON/OFF must differ ONLY by the diffusion-step x0 replacement.  ``on``/``off``: runner instrumentation dicts;
    ``io_*``: model_io arrays.  Returns (pass, checks)."""
    c = {}
    for k in ("input_relative", "input_absolute", "observation_mask_abs", "effective_conditioning_normalized"):
        c[f"{k}_bitwise_equal"] = bool(np.array_equal(io_on[k], io_off[k]))
    for k in ("seed", "text", "length", "guidance", "respacing"):
        c[f"{k}_equal"] = on["request"][k] == off["request"][k]
    c["impute_flags"] = [on["request"]["impute"], off["request"]["impute"]]
    c["impute_flags_ok"] = c["impute_flags"] == [True, False]
    c["net_calls"] = [on["n_net_calls"], off["n_net_calls"]]
    c["net_calls_equal_and_cfg_doubled"] = on["n_net_calls"] == off["n_net_calls"] == 2 * n_steps
    c["obs_x0_single_and_equal"] = len(on["obs_x0_sha"]) == 1 and on["obs_x0_sha"] == off["obs_x0_sha"]
    c["obs_mask_single_and_equal"] = len(on["obs_mask_sha"]) == 1 and on["obs_mask_sha"] == off["obs_mask_sha"]
    c["text_tokens_equal"] = on["text_tokens_sha"] == off["text_tokens_sha"] and len(on["text_tokens_sha"]) == 1
    kon, koff = set(on["y_keys"]), set(off["y_keys"])
    c["y_keys_only_on"] = sorted(kon - koff)
    c["y_keys_only_off"] = sorted(koff - kon)
    c["y_keys_differ_only_by_imputation"] = (kon - koff) == set(IMPUTE_KEYS) and not (koff - kon)
    c["requires_imputation_on_all_true"] = len(on["requires_imputation"]) == n_steps and all(on["requires_imputation"])
    c["requires_imputation_off_all_false"] = len(off["requires_imputation"]) == n_steps and not any(off["requires_imputation"])
    c["reconstruction_guidance_never"] = not any(on["requires_reconstruction_guidance"]) and not any(off["requires_reconstruction_guidance"])
    c["first_step_net_inputs_equal"] = on["x_sha"][:2] == off["x_sha"][:2]
    c["first_step_net_outputs_equal"] = on["out_sha"][:2] == off["out_sha"][:2]
    div = next((i for i, (a, b) in enumerate(zip(on["x_sha"], off["x_sha"])) if a != b), None)
    c["first_divergent_net_call"] = div
    c["divergence_after_first_step"] = div is not None and div >= 2
    ok = all(v for k, v in c.items() if isinstance(v, bool))
    return bool(ok), c


def context_gate(res):
    """Observed-context preservation of the OFF output (frozen thresholds)."""
    c = {"arm_window_ric_err_max_m": res["arm_window_ric_err_max_m"], "obs_joint_ric_err_mean_m": res["obs_joint_ric_err_mean_m"],
         "root_height_max_m": res["root_height_max_m"], "root_yaw_max_rad": res["root_yaw_max_rad"]}
    c["pass"] = bool(res["arm_window_ric_err_max_m"] <= CTX_ARM_WINDOW_MAX_M and res["obs_joint_ric_err_mean_m"] <= CTX_OBS_MEAN_M
                     and res["root_height_max_m"] <= CTX_ROOT_HEIGHT_MAX_M and res["root_yaw_max_rad"] <= CTX_ROOT_YAW_MAX_RAD)
    return c


def mechanism(b0, b1, ctx_pass):
    """Classify the S1 boundary mechanism from R0 (ON) and R1 (OFF) left-wrist ric boundary numbers."""
    w0, w1 = b0["left_wrist"]["ric"], b1["left_wrist"]["ric"]
    e0 = max(w0["effective_entry_m"], w0["effective_exit_m"])
    e1 = max(w1["effective_entry_m"], w1["effective_exit_m"])
    i1 = max(w1["internal_entry_m"], w1["internal_exit_m"])
    flags = {"R0_effective_max_m": e0, "R1_effective_max_m": e1, "R1_internal_max_m": i1,
             "R1_internal_jump_persists": i1 > MECH_RATIO * e0, "R1_effective_jump_removed": e1 <= MECH_RATIO * e0,
             "R1_context_pass": bool(ctx_pass)}
    if flags["R1_internal_jump_persists"]:
        label = "PERSISTS_UNDER_LEARNED_CONDITIONING"
    elif not ctx_pass or not flags["R1_effective_jump_removed"]:
        label = "OFF_CONTINUOUS_ONLY_BY_DEPARTING_FROM_CONTEXT"
    else:
        label = "REPLACEMENT_IMPUTATION_PRIMARY_SUPPORTED"
    return label, flags


def candidate_qa(q, review):
    """QA1-QA6 for C_OFF against B and the frozen W131 C_ON (``q``: per-branch quality dicts; ``review``: unblinded)."""
    off, on, b = q["C_OFF"], q["C_ON"], q["B"]
    aoff, aon, ab = off["arm_local"], on["arm_local"], b["arm_local"]
    bmax = lambda a: max(a["boundary_c0"]["entry_step_mm"], a["boundary_c0"]["exit_step_mm"])     # noqa: E731
    qa = {"QA1": aoff["n_reversal_flicks"] == 0,
          "QA2": bool(all(aoff["boundary_c0"][k] <= 2 * ab["boundary_c0"][k] + 10.0 for k in ("entry_step_mm", "exit_step_mm"))
                      and bmax(aoff) < bmax(aon) and aoff["rms_jerk_mps3"] < aon["rms_jerk_mps3"]),
          "QA3": not off["near_straight_frames"] and not off["bend_direction_flips"] and not off["hinge_opposite_frames"],
          "QA4": off["wrist_dev_from_A_mean_mm"] < b["wrist_dev_from_A_mean_mm"] and off["wrist_dev_from_A_mean_mm"] < on["wrist_dev_from_A_mean_mm"],
          "QA5": bool(off["press_intent"]["pass"])}
    if review is None:
        qa["QA6"] = False
    else:
        rank = review["V4_rank_most_to_least_faithful"]
        qa["QA6"] = bool(review["V2_most_natural"] == "C_OFF" and "C_OFF" not in review["V1_flick"]
                         and "C_OFF" not in review["V3_wrong_elbow"] and rank[-1] != "C_OFF"
                         and rank.index("C_OFF") < rank.index("B") and rank.index("C_OFF") < rank.index("C_ON")
                         and review["V6_upward_press"].get("C_OFF") == "yes")
    return qa


def case_rule(data_blocked, integrity_ok, unsafe, qa):
    """DATA_BLOCKED > D > C > B > A."""
    if data_blocked:
        return "DATA_BLOCKED"
    if not integrity_ok:
        return "D"
    if unsafe:
        return "C"
    if qa is not None and all(qa.values()):
        return "A"
    return "B"
