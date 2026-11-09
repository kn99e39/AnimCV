#!/usr/bin/env python3
"""W131 (EXPERIMENTAL, opt-in) harness: licensed CondMDI partial-joint completion of the S1 left arm on the frozen real
BaseRig Action.

Stages (each refuses an existing output directory; nothing in an AnimCV run directory or lineage is ever written):

  controls  representation controls BEFORE any S1 model output: Blender dump of the frozen S1 baseline, numpy-FK parity,
            official-encoder / absolute-conversion / decoder parity (Motius reference in the scratch env), round trip,
            static / rotated / parent-child / asymmetric / time controls, the masked-UNKNOWN rig perturbation, and the
            fail-closed negative controls.
  declare   prospective declaration (source/run/model/checkpoint hashes, coverage matrix, mapping, time map, mask, text,
            sampler, seed, baselines, visual questions, metric thresholds, frozen CASE precedence, instrument hashes)
            written BEFORE any primary S1 learned output exists.  ``outcomes_inspected: false``.
  run       S1: conditioning -> R1 primary CondMDI sample -> R2 exact determinism rerun -> R3 leak control (rig-level
            UNKNOWN perturbation + tensor-level masked-feature noise; must be bitwise equal) -> left-arm projection ->
            C, B_rs (B through the same 30->20->30 path), Cpin (.blend) -> fresh readbacks -> footprint / pin / quality ->
            renders -> native-rate video, contact sheets, blinded copies.
  assess    apply the frozen CASE rule to the run evidence + the agent's blinded visual review file.

    PY=C:/Projects/AnimCV/.venv_build_full/Scripts/python.exe; export PYTHONPATH=src
    $PY -X utf8 scripts/w131_condmdi_harness.py <stage> --out-dir <dir> [...]
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]
import numpy as np                                                    # noqa: E402

import app.experimental_condmdi as C                                  # noqa: E402
import app.experimental_motionbricks as M                             # noqa: E402

W = Path("C:/Users/dna10/animcv-w131")
W129 = Path("C:/Users/dna10/animcv-w129")
W126 = Path("C:/Users/dna10/animcv-w126/runs")
BLENDER = "C:/Program Files/Blender Foundation/Blender 5.1/blender.exe"
CM_PY = W / "env" / "cm312" / "Scripts" / "python.exe"
INFER = ROOT / "scripts" / "w131_condmdi_infer.py"
STAGE = ROOT / "src" / "blender" / "experimental_motionbricks_stage.py"      # W129 stage, reused UNCHANGED
ACTION_STAGE = ROOT / "src" / "blender" / "experimental_action_stage.py"     # W124 stage, reused UNCHANGED
HF_REV = "1e84d9ebf9a2562a2e3aa01926b9fb3606bd23de"
HF_DIR = W / "hf" / HF_REV
LFS = {"model.safetensors": "3d9e299f6096405c028b3b8b9f8a663794a1e938bfae8ff27102caaa0e1c0e29",
       "clip.safetensors": "431adb635d7a35aaaf5b70acf8738712c6f9ae30d08e17698c6f505c9a686fb8",
       "Mean_abs_3d.npy": "1bb2633e77a99f1120fc1eb424b7489360bac0a8dfb579e18bd5065431b4b07e",
       "Std_abs_3d.npy": "9f5f12cae2003ef19fea119347346d0e46ad85c86dcd1096f1e319ee2a69f940"}
INSTRUMENTS = ["src/app/experimental_condmdi.py", "scripts/w131_condmdi_infer.py", "scripts/w131_condmdi_harness.py",
               "scripts/w131_condmdi_preflight.py", "tests/test_experimental_condmdi.py",
               "src/app/experimental_motionbricks.py", "src/blender/experimental_motionbricks_stage.py",
               "src/blender/experimental_action_stage.py"]
S1 = {"run": W126 / "S1_replay", "segment_first": 79, "n_action": 240,
      "baseline_sha": "0803ab2bf31210fbb729bc81b76931729fdc3e12cce27b9507a5b8caf1d12402",
      "value_digest": "25a09802ffa210df56f0e930295468ef4cdb675e2945cb95491c080b5101c8a9",
      "head_state": "73dcd3dde3a65ffbd23f36361173ac7a92ee7c4db5e8436e4e85947649a524ab",
      "pin_accepted": "accepted/accepted_03_674ba278e9a0.blend",
      "pin_accepted_sha": "7e2ab16511f4cbe111171df9af0bc2e1f7d8c09a57bd5a4feff284d34e908f40",
      "w129_B": W129 / "results/run/S1/B_S1.blend", "w129_B_sha": "8899f43341ac1a7a565229a9551b1f8a79943cc038ad519e5544af89e772a33f",
      "pin_request": "edits/W126V", "context_last": 171, "target_first": 200}
TEXT = "a person stands in place and presses both arms straight up overhead"
SEED = 42
GUIDANCE = 2.5
RESPACING = "ddim100"
LEAK_SEED = 131
UNKNOWN_RIG_CHANNELS = ("upperarm_r", "lowerarm_r", "hand_l", "hand_r", "foot_l", "foot_r", "neck_01", "neck_02")


# ------------------------------------------------------------------------------------------------ utilities ---
def now():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def dump(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(obj, indent=1, sort_keys=True, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    return M.sha256_file(p)


def load(p):
    return json.loads(Path(p).read_text())


def new_dir(p):
    p = Path(p)
    if p.exists():
        raise SystemExit(f"refusing to overwrite {p}")
    p.mkdir(parents=True)
    return p


def lf_sha(rel):
    return hashlib.sha256((ROOT / rel).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def blender(script, mode, log, **kw):
    args = [BLENDER, "-b", "-t", "2", "--python-exit-code", "1", "-P", str(script), "--", mode]
    for k, v in kw.items():
        args += ["--" + k.replace("_", "-"), str(v)]
    t = time.time()
    r = subprocess.run(args, capture_output=True, text=True)
    Path(log).write_text(r.stdout + "\n--- stderr ---\n" + r.stderr)
    if r.returncode != 0:
        raise RuntimeError(f"Blender {mode} failed ({r.returncode}); see {log}")
    return time.time() - t


def cm(req, req_path):
    dump(req_path, req)
    t = time.time()
    r = subprocess.run([str(CM_PY), "-I", str(INFER), str(req_path)], capture_output=True, text=True, cwd=str(W))
    Path(str(req_path) + ".log").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr)
    return r.returncode, time.time() - t


def check_declaration(a):
    if not a.declaration:
        raise SystemExit("--declaration required")
    if M.sha256_file(a.declaration) != a.declaration_sha:
        raise SystemExit("BLOCKED_DECLARATION_CHANGED")
    d = load(a.declaration)
    changed = [f for f, h in d["instrument_sha256_lf"].items() if lf_sha(f) != h]
    if changed:
        raise SystemExit(f"BLOCKED_INSTRUMENT_CHANGED {changed}")
    return d


def c_ankle():
    """Declared floor constant from the checkpoint's own statistics: mean ankle height minus mean foot-joint height."""
    m = np.load(HF_DIR / "Mean_abs_3d.npy").astype(np.float64)
    y = lambda j: m[4 + 3 * (j - 1) + 1]                                           # noqa: E731
    return float((y(7) + y(8)) / 2 - (y(10) + y(11)) / 2)


def inputs():
    inp = W / "inputs"
    if not inp.exists():
        inp.mkdir(parents=True)
        shutil.copy2(S1["run"] / "baseline" / "baseline.blend", inp / "S1_baseline.blend")
        shutil.copy2(S1["run"] / S1["pin_accepted"], inp / "S1_W126V_accepted.blend")
        shutil.copy2(S1["w129_B"], inp / "S1_W129_B.blend")
    for p, h in ((inp / "S1_baseline.blend", S1["baseline_sha"]), (inp / "S1_W126V_accepted.blend", S1["pin_accepted_sha"]),
                 (inp / "S1_W129_B.blend", S1["w129_B_sha"]), (S1["run"] / "baseline" / "baseline.blend", S1["baseline_sha"])):
        if M.sha256_file(p) != h:
            raise SystemExit(f"BLOCKED_SOURCE_IDENTITY {p}")
    return inp


def conditioning(rig, stored, tm, kint, override=None, cank=None):
    J = np.stack([rig.hml_joints(*C.pose_at(stored, float(u), override)) for u in tm["u"]])
    enc = C.encode(J, "supported", c_ankle() if cank is None else cank)
    mask = C.observation_mask(tm["n_model"], kint)
    return J, enc, mask


def arm_override(rig_basis_fn_u, rig_basis_fn_l):
    return {"upperarm_l": rig_basis_fn_u, "lowerarm_l": rig_basis_fn_l}


def project_samples(rig, stored, tm, kint, absm, r0):
    """Model (or encoded) absolute HML263 -> per interior sample upperarm_l / lowerarm_l basis (B = twist reference)."""
    bu_fn, bl_fn = C.b_slerp_fn(stored, "upperarm_l"), C.b_slerp_fn(stored, "lowerarm_l")
    U, V = C.arm_dirs_from_abs(absm, r0, kint)
    up, lo = {}, {}
    for i, k in enumerate(kint):
        u = float(tm["u"][k])
        basis, obj = C.pose_at(stored, u)
        up[k], lo[k] = C.project_arm(rig, basis, obj, bu_fn(u), bl_fn(u), U[i], V[i])
    return up, lo


# ------------------------------------------------------------------------------------------------ controls ---
def stage_controls(a):
    out = new_dir(a.out_dir)
    pf = load(a.preflight)
    idn = load(a.identity)
    if pf["status"] != "OK_PENDING_IDENTITY" or not idn["identity_match"] or not idn["hf_files_sha256_ok"]:
        raise SystemExit("DATA_BLOCKED: preflight/identity not OK")
    inp = inputs()
    res = {"stage": "controls", "time": now(), "head": git("rev-parse", "HEAD")}
    blender(STAGE, "dump", out / "dump.log", blend=inp / "S1_baseline.blend", report=out / "S1_dump.json")
    bd = load(out / "S1_dump.json")
    st = bd["stored"]
    rig = C.Rig(bd["rest"])
    # (1) numpy FK vs Blender-evaluated pose matrices, all 240 integer frames
    e_w = 0.0
    for fr in bd["frames"]:
        basis, obj = C.pose_at(st, fr["blender_frame"])
        Mw, P = rig.pose(basis, obj)
        for b, pm in fr["pose_matrix"].items():
            e_w = max(e_w, float(np.abs((Mw @ P[b])[:3, 3] - (np.asarray(fr["matrix_world"]) @ np.asarray(pm))[:3, 3]).max()))
    res["fk_parity_max_world_m"] = e_w
    # (2) time map
    tm = C.time_map()
    C.validate_time_map(tm)
    kint = C.interior_samples(tm)
    res["time"] = {"n_action": tm["n_action"], "n_model": tm["n_model"], "duration_s": str(tm["duration_s"]),
                   "interior_samples": [kint[0], kint[-1], len(kint)],
                   "interior_sample_action_positions": [str(tm["u"][kint[0]]), str(tm["u"][kint[-1]])],
                   "observed_arm_neighbours": {"before": str(tm["u"][kint[0] - 1]), "after": str(tm["u"][kint[-1] + 1])},
                   "exact": sum(1 for x in tm["kind"] if x == "EXACT"), "interpolated": sum(1 for x in tm["kind"] if x != "EXACT"),
                   "reconstruct_172_199_positions": [str(tm["s"][f - 1]) for f in (172, 173, 174, 199)],
                   "frame_240_position": str(tm["s"][239])}
    # (3) real S1 conditioning + round trip
    J, enc, mask = conditioning(rig, st, tm, kint)
    C.check_axes(C.A_BLENDER_TO_HML, rig)
    C.check_left_right(J)
    C.validate_mask(mask, kint)
    ab = C.rel_to_abs(enc["relative"])
    U, V = C.arm_dirs_from_abs(ab, enc["r0"], list(range(tm["n_model"])))
    rt = 0.0
    for k, u in enumerate(tm["u"]):
        Mw, P = rig.pose(*C.pose_at(st, float(u)))
        w1 = M.frame_from(rig.point(Mw, P, "H:lowerarm_l") - rig.point(Mw, P, "H:upperarm_l"), [0, 0, 1])[:, 0]
        w2 = M.frame_from(rig.point(Mw, P, "H:hand_l") - rig.point(Mw, P, "H:lowerarm_l"), [0, 0, 1])[:, 0]
        rt = max(rt, float(np.degrees(np.arccos(np.clip(np.dot(w1, U[k]), -1, 1)))), float(np.degrees(np.arccos(np.clip(np.dot(w2, V[k]), -1, 1)))))
    res["arm_direction_round_trip_max_deg"] = rt
    # projection identity: projecting A's own directions with A as twist reference returns A
    pid = 0.0
    for f in range(172, 200):
        b, o = C.pose_at(st, f)
        Mw, P = rig.pose(b, o)
        bu, bl = C.project_arm(rig, b, o, b["upperarm_l"], b["lowerarm_l"], rig.point(Mw, P, "H:lowerarm_l") - rig.point(Mw, P, "H:upperarm_l"),
                               rig.point(Mw, P, "H:hand_l") - rig.point(Mw, P, "H:lowerarm_l"))
        pid = max(pid, float(np.abs(M.canon_sign(bu, b["upperarm_l"]) - b["upperarm_l"]).max()), float(np.abs(M.canon_sign(bl, b["lowerarm_l"]) - b["lowerarm_l"]).max()))
    res["projection_identity_max_quat_err"] = pid
    # B_rs (no model) through the same path must reproduce B: lossless-conversion diagnostic
    bu_fn, bl_fn = C.b_slerp_fn(st, "upperarm_l"), C.b_slerp_fn(st, "lowerarm_l")
    Jb, encb, _ = conditioning(rig, st, tm, kint, arm_override(bu_fn, bl_fn))
    upb, lob = project_samples(rig, st, tm, kint, C.rel_to_abs(encb["relative"]), encb["r0"])
    rb_u, rb_l = C.reconstruct_30(upb, range(172, 200)), C.reconstruct_30(lob, range(172, 200))
    res["B_rs_vs_B_max_quat_err"] = max(max(float(np.abs(M.canon_sign(rb_u[f], bu_fn(f)) - bu_fn(f)).max()) for f in rb_u),
                                        max(float(np.abs(M.canon_sign(rb_l[f], bl_fn(f)) - bl_fn(f)).max()) for f in rb_l))
    # (4) official-implementation parity (scratch env, Motius reference; no model)
    # official encoder needs T+1 positions for T rows: the 160 S1 samples + Action 240 (formula parity only)
    Jp = np.concatenate([J, rig.hml_joints(*C.pose_at(st, 240))[None]])
    enc_off = C.encode(Jp, "official", n_out=len(Jp) - 1)
    np.savez(out / "parity_input.npz", joints=Jp.astype(np.float32), relative=enc["relative"].astype(np.float32))
    rc, _ = cm({"mode": "parity", "input_npz": str(out / "parity_input.npz"), "out_dir": str(out / "parity")}, out / "req_parity.json")
    par = {"rc": rc}
    if rc == 0:
        z = np.load(out / "parity" / "official.npz")
        off = z["hml263"].astype(np.float64)
        obs_f = [0, 3] + [f for j in range(1, 22) for f in range(4 + 3 * (j - 1), 4 + 3 * j)]
        par["encoder_ric_root_max_abs"] = float(np.abs(off[:, obs_f] - enc_off["relative"][:, obs_f]).max())
        par["encoder_root_xz_vel_max_abs"] = float(np.abs(off[:, 1:3] - enc_off["relative"][:, 1:3]).max())
        par["rel_to_abs_max_abs"] = float(np.abs(z["absolute"] - C.rel_to_abs(enc["relative"].astype(np.float32))).max())
        par["abs_to_joints_max_abs"] = float(np.abs(z["joints"] - C.abs_to_joints(z["absolute"])).max())
        par["target_offsets_max_abs"] = float(np.abs(np.linalg.norm(z["target_offsets"], axis=1) - C.TARGET_LEN).max())
    res["official_parity"] = par
    # (5) synthetic posed controls on the real rig
    rest_obj = (1.0, 0.0, 0.0, 0.0)

    def joints_of(bases, obj=rest_obj):
        return rig.hml_joints(bases, obj)
    Jrest = joints_of({})
    Jyaw = joints_of({}, _qaa([0, 0, 1], 30))
    Jarm = joints_of({"upperarm_l": _qaa([1, 0, 0], -60)})            # swing (local Y = bone axis = twist)
    Jsp = joints_of({"spine_01": _qaa([1, 0, 0], 20)})
    Jboth = joints_of({"spine_01": _qaa([1, 0, 0], 20), "upperarm_l": _qaa([1, 0, 0], -60)})
    seq = lambda *js: np.stack(js)                                                 # noqa: E731

    def ric(js):
        return C.encode(seq(*js), "supported", 0.0)["relative"][:, 4:67].reshape(-1, 21, 3)
    # NOTE: compare the SAME frame (index 1) of two sequences: the official recipe forces frame 0's root rotation to identity
    r_static = ric([Jrest, Jrest])
    r_yaw = ric([Jyaw, Jyaw])
    sc = {"static_left_wrist_x_positive": bool(r_static[0, 19, 0] > 0.1), "static_left_shoulder_x": float(r_static[0, 15, 0]),
          "rotated_yaw_invariance_max": float(np.abs(r_static - r_yaw).max())}
    r_arm = ric([Jrest, Jarm])
    moved = np.linalg.norm(r_arm[1] - r_static[1], axis=1)
    sc["asymmetric_left_arm_only_moved_joints"] = [C.HML_JOINTS[j + 1] for j in np.nonzero(moved > 1e-6)[0]]
    r_sp = ric([Jrest, Jsp])
    r_bo = ric([Jrest, Jboth])
    msp = np.linalg.norm(Jsp - Jrest, axis=1)                                     # rig-world positions
    sc["parent_child_spine_moves"] = [C.HML_JOINTS[j] for j in np.nonzero(msp > 1e-6)[0]]
    # informational: the official heading uses the (rigid-torso) shoulders, so a spine bend also changes every ric
    sc["parent_child_ric_changed_joints_info"] = [C.HML_JOINTS[j + 1] for j in np.nonzero(np.linalg.norm(r_sp[1] - r_static[1], axis=1) > 1e-6)[0]]
    rq_a, rq_b = (C.encode(seq(Jrest, x), "supported", 0.0)["root_quat"][1] for x in (Jrest, Jsp))
    sc["parent_child_heading_change_deg_info"] = float(np.degrees(2 * np.arccos(np.clip(abs(np.dot(rq_a, rq_b)), -1, 1))))
    # arm direction relative to the chest frame must be the same with/without the spine bend (child follows parent)
    Mb, Pb = rig.pose({"spine_01": _qaa([1, 0, 0], 20), "upperarm_l": _qaa([1, 0, 0], -60)}, rest_obj)
    Ma, Pa = rig.pose({"upperarm_l": _qaa([1, 0, 0], -60)}, rest_obj)
    rel_b = M.rot_part(Pb["spine_05"]).T @ M.rot_part(Pb["upperarm_l"])
    rel_a = M.rot_part(Pa["spine_05"]).T @ M.rot_part(Pa["upperarm_l"])
    sc["parent_child_arm_relative_to_chest_unchanged"] = float(np.abs(rel_a - rel_b).max())
    sc["parent_child_legs_unchanged"] = float(np.abs(Jboth[[1, 2, 4, 5, 7, 8]] - Jarm[[1, 2, 4, 5, 7, 8]]).max())   # rig-world
    sc["pass"] = bool(sc["static_left_wrist_x_positive"] and sc["rotated_yaw_invariance_max"] < 1e-6
                      and set(sc["asymmetric_left_arm_only_moved_joints"]) == {"left_elbow", "left_wrist"}
                      and "left_wrist" in sc["parent_child_spine_moves"] and "left_knee" not in sc["parent_child_spine_moves"]
                      and sc["parent_child_arm_relative_to_chest_unchanged"] < 1e-9 and sc["parent_child_legs_unchanged"] < 1e-6)
    res["synthetic_controls"] = sc
    # (6) masked-UNKNOWN rig perturbation: observed features must be bitwise identical
    rng = np.random.default_rng(LEAK_SEED)
    pert = {ch: (lambda q_: (lambda u: q_))(_qaa(rng.normal(size=3), float(rng.uniform(20, 60)))) for ch in UNKNOWN_RIG_CHANNELS}
    Jx = np.stack([rig.hml_joints(*_pose_with(st, float(u), pert)) for u in tm["u"]])
    encx = C.encode(Jx, "supported", c_ankle())
    obs = mask.T
    absA, absX = C.rel_to_abs(enc["relative"]), C.rel_to_abs(encx["relative"])
    res["unknown_rig_perturbation"] = {"channels": list(UNKNOWN_RIG_CHANNELS),
                                       "masked_joints_changed_m": float(np.abs(Jx - J)[:, [10, 11, 15, 19, 21]].max()),
                                       "observed_joints_changed_m": float(np.abs(Jx - J)[:, list(C.OBSERVED_JOINTS)].max()),
                                       "observed_abs_features_bitwise_equal": bool(np.array_equal(absA[obs], absX[obs])),
                                       "masked_abs_features_changed": bool(not np.array_equal(absA[~obs], absX[~obs]))}
    # (7) fail-closed negative controls
    neg = {}

    def expect(name, code, fn):
        try:
            fn()
            neg[name] = {"pass": False, "note": "did not refuse"}
        except M.AdapterStop as e:
            neg[name] = {"pass": e.code == code, "code": e.code}
    expect("mirrored_axis", "AXIS_MAPPING_IMPROPER", lambda: C.check_axes(np.diag([-1.0, 1, 1]) @ C.A_BLENDER_TO_HML, rig))
    expect("wrong_axis_identity", "AXIS_UP_MISMATCH", lambda: C.check_axes(np.eye(3), rig))
    swapJ = J.copy()
    swapJ[:, [18, 19]] = swapJ[:, [19, 18]]
    swapJ[:, [20, 21]] = swapJ[:, [21, 20]]
    expect("left_right_swapped_arm", "LEFT_RIGHT_MISMATCH", lambda: C.check_left_right(swapJ))
    expect("baserig_missing_bone", "BASERIG_MISSING_BONES", lambda: C.Rig({k: v for k, v in bd["rest"].items() if k != "lowerarm_l"}))
    bad = mask.copy()
    bad[4 + 3 * 18: 4 + 3 * 19, :] = True                                          # right elbow (UNKNOWN)
    expect("mask_observes_unknown_joint", "MASK_OBSERVES_UNKNOWN", lambda: C.validate_mask(bad, kint))
    bad2 = mask.copy()
    bad2[67:73, :] = True                                                          # rot features
    expect("mask_observes_rotation", "MASK_OBSERVES_UNKNOWN", lambda: C.validate_mask(bad2, kint))
    bad3 = mask.copy()
    bad3[1:3, :] = True                                                            # root travel
    expect("mask_observes_root_travel", "MASK_OBSERVES_UNKNOWN", lambda: C.validate_mask(bad3, kint))
    bad4 = mask.copy()
    bad4[4 + 3 * 19: 4 + 3 * 20, kint[3]] = True                                   # left wrist inside the interior
    expect("mask_observes_edit_domain", "MASK_OBSERVES_EDIT_DOMAIN", lambda: C.validate_mask(bad4, kint))
    expect("mask_bad_shape", "MASK_SHAPE", lambda: C.validate_mask(mask[:262], kint))
    tmx = dict(tm)
    tmx["u"] = list(tm["u"])
    tmx["u"][10], tmx["u"][11] = tmx["u"][11], tmx["u"][10]
    expect("time_non_monotone", "TIME_MAP_NON_MONOTONE", lambda: C.validate_time_map(tmx))
    tmy = dict(tm)
    tmy["u"] = list(tm["u"])[:-1]
    expect("time_duration_changed", "TIME_MAP_DURATION_CHANGED", lambda: C.validate_time_map(tmy))
    expect("time_window_not_representable", "DURATION_NOT_REPRESENTABLE", lambda: C.time_map(n_action=239))
    prop = load(S1["run"] / S1["pin_request"] / "proposal.json")
    pin_act = [f - S1["segment_first"] + 1 for f in prop["pin_frames"]]
    expect("protected_overlap", "PROTECTED_OVERLAP", lambda: C.protected_check((128, 155), pin_act))
    C.protected_check(C.INTERIOR_ACTION, pin_act)
    for name, vals in (("channel_outside_domain", {"spine_01": {"180": [1, 0, 0, 0]}}),
                       ("frame_outside_interior", {"upperarm_l": {"210": [1, 0, 0, 0]}})):
        spec = {"out_blend": str(out / f"neg_{name}.blend"), "interior_action_frames": [172, 199], "values_wxyz": vals,
                "branch": "NEG", "action_name": "NEG", "expected_blend_sha256": S1["baseline_sha"]}
        dump(out / f"neg_{name}_spec.json", spec)
        try:
            blender(STAGE, "build_variant", out / f"neg_{name}.log", blend=inp / "S1_baseline.blend", spec=out / f"neg_{name}_spec.json",
                    report=out / f"neg_{name}_report.json")
            neg[name] = {"pass": False}
        except RuntimeError:
            txt = (out / f"neg_{name}.log").read_text()
            neg[name] = {"pass": ("REFUSED_" in txt) and not (out / f"neg_{name}.blend").exists(),
                         "refusal": [ln for ln in txt.splitlines() if "REFUSED_" in ln][:1]}
    res["negative_controls"] = neg
    res["coverage"] = C.coverage_matrix(tm, kint, mask)
    res["c_ankle_m"] = c_ankle()
    res["floor_supported_minus_official_m"] = float(C.encode(J, "supported", c_ankle())["floor"] - C.encode(J, "official")["floor"])
    g1 = (e_w <= 1e-5 and rt <= 1e-3 and pid <= 1e-6 and res["B_rs_vs_B_max_quat_err"] <= 1e-6 and par.get("rc") == 0
          and par["encoder_ric_root_max_abs"] <= 1e-4 and par["rel_to_abs_max_abs"] <= 1e-4 and par["abs_to_joints_max_abs"] <= 1e-4
          and par["target_offsets_max_abs"] <= 1e-6 and sc["pass"] and res["unknown_rig_perturbation"]["observed_abs_features_bitwise_equal"]
          and res["unknown_rig_perturbation"]["masked_abs_features_changed"])
    res["G1_pass"] = bool(g1)
    res["G6_adapter_pass"] = bool(all(v["pass"] for v in neg.values()))
    sha = dump(out / "controls.json", res)
    print(json.dumps({"G1_pass": res["G1_pass"], "G6": res["G6_adapter_pass"], "fk": e_w, "rt_deg": rt, "pid": pid,
                      "B_rs": res["B_rs_vs_B_max_quat_err"], "parity": par, "synthetic": sc, "leak": res["unknown_rig_perturbation"],
                      "neg": {k: v["pass"] for k, v in neg.items()}, "sha": sha}, indent=1, default=str))
    return 0 if (g1 and res["G6_adapter_pass"]) else 3


def _qaa(axis, deg):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    h = np.radians(deg) / 2
    return np.array([np.cos(h), *(np.sin(h) * axis)])


def _pose_with(stored, u, extra):
    basis, obj = C.pose_at(stored, u)
    for ch, fn in extra.items():
        basis[ch] = fn(u)
    return basis, obj


# ------------------------------------------------------------------------------------------------ declaration ---
CASE_RULE = {
    "DATA_BLOCKED": "a prerequisite (rights, checkpoint identity, authentic S1/W126/W129 baselines, runtime, Blender, mask semantics, honest mapping, bounded 20<->30 conversion) is missing before primary inference",
    "D": "declared safety/integrity falsification: the masked-input leak control R3 is not bitwise equal to R1 (input-leak/contract failure), a negative control does not fail closed, the sample is not the pinned real model, or an instrument changed after this declaration",
    "C": "the real model ran but a representation/clock/Action/constraint gate (G1, G2, G4, G5, G7) fails or the projection is unsafe",
    "B": "real model, honest partial conditioning and a safe separate Blender Action are demonstrated (all gates pass) but any quality criterion QA1-QA6 fails (no demonstrated superiority)",
    "A": "all gates pass AND QA1-QA6 pass: credible visual/semantic improvement vs BOTH A and B without source-intent loss (visual evidence = agent-only blinded stills; human native-rate review NOT RUN)",
    "precedence": "DATA_BLOCKED > D > C > B > A",
}
GATES = {
    "G0": "preflight OK_PENDING_IDENTITY + identity: HF model.safetensors bitwise = official random_joints model_avg; LFS hashes",
    "G1": "representation controls (controls.json G1_pass): FK parity <= 1e-5 m; arm-direction round trip <= 1e-3 deg; projection identity <= 1e-6; B_rs vs B <= 1e-6; official encoder/abs/decoder parity <= 1e-4; synthetic static/rotated/asymmetric/parent-child controls; UNKNOWN rig perturbation leaves every observed feature bitwise equal",
    "G2": "clock: 160 samples x 1/20 s = 240 x 1/30 s; interior samples 114..132 only; frames 172..199 reconstructed from samples 114..132 on the original timestamps; scene 30 fps, keys 1..240 unchanged",
    "G3": "authentic inference: pinned weights (hash-checked in the runner), official CondMDIPipeline._sample, mask validated (no UNKNOWN observed, left arm observed on both sides, not in the interior); R2 bitwise = R1; R3 (leak control) bitwise = R1",
    "G4": "C (and B_rs) separate .blend + separately named Action; fresh-process readback: 1 Action, 32 CONSTANT F-curves keyed 1..240, 30 fps, scene 1..240; changed keys only upperarm_l/lowerarm_l in 172..199; every non-left-arm bone world point and the object matrix bit-equal on every integer frame and subframe; left arm bit-equal outside 172..199; baseline/lineage files unchanged",
    "G5": "hard pin: Action 132..140 bit-equal C vs A; C composed onto the W126V-accepted .blend keeps every pin-frame residual equal to W126V's and <= 1e-6 m (authoritative tolerance 1 mm)",
    "G6": "negative controls fail closed (controls.json G6_adapter_pass)",
    "G7": "native-rate synchronized video: 240 frames written and decoded at 30.000 fps",
}
QA = {
    "QA1": "C has 0 ARM-LOCAL wrist reversal flicks (clavicle_l frame) in steps 171->172 .. 199->200 (consecutive steps both > 20 mm with negative dot)",
    "QA2": "C ARM-LOCAL boundary C0 entry/exit wrist steps each <= 2 x B's + 10 mm",
    "QA3": "C has no near-straight (> 175 deg) interior frame, no bend-direction flip, no frame bending against the A hinge reference",
    "QA4": "source fidelity: C mean interior WORLD wrist deviation from A <= 1.25 x B's + 25 mm",
    "QA5": "overhead-press intent (no swing): over Action 172..199, C's wrist height above the left shoulder (world z) never falls more than 50 mm below the lower of its values at Action 171 and 200",
    "QA6": "blinded agent visual review: C judged most natural (V2), not named for a flick (V1) or wrong elbow (V3), not least faithful (V4), and read as an upward press (V6 = yes)",
}
VISUAL_QUESTIONS = {
    "V1": "Which panel(s) show a visible unnatural one-frame flick/teleport/reversal of the left arm (around Action 171-175)?",
    "V2": "Which panel's left-arm motion through the interior looks most natural and temporally coherent?",
    "V3": "Which panel(s) show an anatomically wrong elbow (hyperextension, flip) or limb-length artefact?",
    "V4": "Rank the panels by agreement with the source video's visible left-arm pose and timing (rack -> press); which is least faithful?",
    "V5": "Is any boundary discontinuity visible at 171/172 or 199/200 in any panel?",
    "V6": "For each panel: does the interior read as an upward overhead press (yes) or as a swing / detour / other motion (no)?",
}


def stage_declare(a):
    out = new_dir(a.out_dir)
    pf, idn, ctl = load(a.preflight), load(a.identity), load(a.controls)
    if not (pf["status"] == "OK_PENDING_IDENTITY" and idn["identity_match"] and ctl["G1_pass"] and ctl["G6_adapter_pass"]):
        raise SystemExit("prerequisites/controls not OK: no declaration")
    tm = C.time_map()
    kint = C.interior_samples(tm)
    prop = load(S1["run"] / S1["pin_request"] / "proposal.json")
    d = {"schema": "animcv_exp_w131_declaration_v1", "time": now(), "outcomes_inspected": False,
         "primary_model_output_exists": (W / "results" / "run").exists(),
         "batch": 131, "starting_commit": "d2e68f1a3e7c2148ad81d3d8256b6b48c67ed697", "instruction_blob": "a06baf169f296195561dedcf123de506ffe5853b",
         "repo_head_at_declaration": git("rev-parse", "HEAD"),
         "preflight_sha256": M.sha256_file(a.preflight), "identity_sha256": M.sha256_file(a.identity), "controls_sha256": M.sha256_file(a.controls),
         "source": {"episode": "S1 (CC BY-SA 4.0 double kettlebell strict press, side view)", "clip_sha256": "de8feb9d515daa2bfaeed7167fdcc62028dd3f6450872c4fd81f8fb03dd6359c",
                    "run": str(S1["run"]), "baseline_sha256": S1["baseline_sha"], "value_digest": S1["value_digest"],
                    "lineage_head": S1["head_state"], "segment": "SOURCE 79..318 -> Action 1..240 (Action = SOURCE - 78)",
                    "edit_domain": "upperarm_l / lowerarm_l local quaternion keys at Action 172..199 = SOURCE 250..277",
                    "protected_pin": {"request": S1["pin_request"], "source_frames": prop["pin_frames"],
                                      "action_frames": [f - 78 for f in prop["pin_frames"]], "target_m": prop["constraint"]["target"],
                                      "tolerance_m": prop["constraint"]["tolerance_m"], "state": "accepted in W126 then REVERTED (no active pin at the head)"},
                    "w129_B_sha256": S1["w129_B_sha"]},
         "model": {"name": "CondMDI (Cohan et al., SIGGRAPH 2024), official 'randomly sampled frames and joints' release, EMA weights",
                   "hf_id": "ZeyuLing/motius-condmdi-humanml3d", "hf_revision": HF_REV, "lfs_sha256": LFS,
                   "official_release": "condmdi_random_joints.zip (Google Drive 1aP-z1JxSCTcUHhMqqdL2wbwQJUZWHT2j), model000750000.pt model_avg",
                   "identity": "196/196 tensors bitwise equal (identity.json)", "code": "setarehc/diffusion-motion-inbetweening 0c58473 (MIT)",
                   "runtime": "ZeyuLing/Motius d04a38c (CondMDI MIT + CLIP MIT retained), scratch Python 3.12 / torch 2.9.1+cu128",
                   "licence": "checkpoint card license: mit; code MIT; CLIP MIT; training data HumanML3D/AMASS (non-commercial research) - declared use is local non-commercial research evaluation"},
         "config": {"text": TEXT, "seed": SEED, "guidance": GUIDANCE, "respacing": RESPACING, "length": tm["n_model"],
                    "impute": True, "stop_imputation_at": 0, "deterministic_cuda": "cudnn.deterministic, no TF32, use_deterministic_algorithms(warn_only)",
                    "rerun": "R2 = exact rerun in a new process (determinism only)",
                    "leak_control": f"R3 = UNKNOWN rig channels {list(UNKNOWN_RIG_CHANNELS)} randomly rotated (seed {LEAK_SEED}) AND every masked input feature replaced by seeded noise (seed {LEAK_SEED}); output must be bitwise equal to R1",
                    "no_tuning": "one model, one episode, one text, one mask, one sampler, one seed; no prompt/seed/window/mask/guidance sweep, no post-hoc smoothing"},
         "time_map": {"rule": "Action frame f at t=(f-1)/30 s on [0, 8.0); model sample k at t=k/20 s; 240 frames <-> 160 samples; even k = exact frame 1+1.5k; odd k = half-frame, every keyed channel SO(3)-slerped at 0.5 (INTERPOLATED hypothesis)",
                      "reconstruction": "frame f <- model position s=2(f-1)/3: exact sample when integral, else slerp of the per-sample left-arm basis between samples floor(s), floor(s)+1 (weights 1/3, 2/3); only Action 172..199 (s=114..132) are written; Action 240 (s=159.33) is beyond the last sample and never reconstructed",
                      "interior_samples": kint, "no_retiming": True},
         "representation": {"axes": C.A_BLENDER_TO_HML.tolist(), "rig_points": {C.HML_JOINTS[j]: C.RIG_POINT[j] for j in range(22)},
                            "encoder": "official HumanML3D recipe (uniform skeleton with 000021 lengths, canonical frame-0, smoothed official root heading, root-relative positions); the floor deviates: min over ankles minus c_ankle",
                            "c_ankle_m": ctl["c_ankle_m"], "coverage": ctl["coverage"],
                            "mask": "absolute-space: root yaw + height; ric positions of every non-UNKNOWN joint at all 160 samples; left elbow/wrist only outside samples 114..132; no rotation, velocity, contact or root-xz feature",
                            "n_observed_features": ctl["coverage"]["n_observed_features"],
                            "declared_coupling": "HML263 positions are pelvis-relative: the left elbow/wrist conditioning is anchored on the rigid rest upper torso (PROXY) carried by the driven spine_01; the upper-spine/clavicle bending is UNKNOWN and is the rig's rest. Disclosed, not hidden.",
                            "projection": "generated left_shoulder->elbow and elbow->wrist directions (canonical -> undo frame-0 facing -> A^T) -> minimal swing from the frozen W129 B orientation (B supplies twist) -> upperarm_l, then lowerarm_l; per sample, then 20->30 slerp"},
         "branches": {"A": "untouched frozen baseline (no-op)", "B": "frozen W129 B .blend (endpoint-matched per-channel slerp)",
                      "C": "CondMDI R1 left arm, projected on Action 172..199 only",
                      "B_rs": "diagnostic: B through the SAME 30->20->30 + HML263 encode/decode + projection path (no model); does not replace B",
                      "W129_C": "historical MotionBricks C, reference only (frozen numbers; not rerun)"},
         "gates": GATES, "quality_criteria": QA, "visual_questions": VISUAL_QUESTIONS, "case_rule": CASE_RULE,
         "visual_protocol": "panels X/Y/Z = random permutation of A/B/C sealed in blind_key_SEALED.json; the agent answers from blinded stills BEFORE reading the key; AGENT-ONLY, not human; native-rate perceptual review NOT RUN (cannot be done by the agent)",
         "stop_conditions": ["leak control R3 not bitwise equal -> STOP (D, input-leak/contract failure)",
                             "projection would touch any non-left-arm channel or frame outside 172..199 -> refuse (C)",
                             "no retiming / window shift / tuning after output"],
         "instrument_sha256_lf": {f: lf_sha(f) for f in INSTRUMENTS}}
    sha = dump(out / "declaration.json", d)
    print(json.dumps({"declaration_sha256": sha, "observed": d["representation"]["n_observed_features"], "interior": [kint[0], kint[-1]]}, indent=1))
    return 0


# ------------------------------------------------------------------------------------------------ run ---
def readback(blend, rep, log):
    return blender(ACTION_STAGE, "readback", log, blend=blend, report=rep, subframes="0.25,0.5,0.75")


def full_dump(name, blend, out):
    p = out / f"dump_{name}.json"
    if not p.exists():
        blender(STAGE, "dump", out / f"dump_{name}.log", blend=blend, report=p)
    return load(p)


def press_intent(rb, lo=171, hi=200):
    fr = {f["blender_frame"]: f for f in rb["frames"]}
    h = {f: float(fr[f]["world"]["hand_l"]["head"][2] - fr[f]["world"]["upperarm_l"]["head"][2]) for f in range(lo, hi + 1)}
    floor_ = min(h[lo], h[hi])
    dip = floor_ - min(h[f] for f in range(lo + 1, hi))
    return {"wrist_above_shoulder_mm": {str(f): v * 1e3 for f, v in h.items()}, "endpoint_lower_mm": floor_ * 1e3,
            "max_dip_below_endpoints_mm": dip * 1e3, "pass": dip * 1e3 <= 50.0}


def stage_run(a):
    d = check_declaration(a)
    ctl = load(a.controls)
    if not ctl["G1_pass"]:
        raise SystemExit("STOP: representation controls failed")
    out = new_dir(Path(a.out_dir) / "S1")
    log = {"stage": "run", "start": now(), "steps": []}
    inp = inputs()
    refs_before = M.sha256_file(S1["run"] / "lineage" / "refs.json")
    base_blend, pin_blend, b_blend = inp / "S1_baseline.blend", inp / "S1_W126V_accepted.blend", inp / "S1_W129_B.blend"
    log["steps"].append(["dump_A", blender(STAGE, "dump", out / "dump_A.log", blend=base_blend, report=out / "dump_A.json")])
    bd = load(out / "dump_A.json")
    st = bd["stored"]
    rig = C.Rig(bd["rest"])
    tm = C.time_map()
    C.validate_time_map(tm)
    kint = C.interior_samples(tm)
    if kint != d["time_map"]["interior_samples"]:
        raise SystemExit("BLOCKED_TIME_MAP_CHANGED")
    J, enc, mask = conditioning(rig, st, tm, kint)
    C.check_left_right(J)
    C.validate_mask(mask, kint)
    np.savez(out / "conditioning.npz", relative_hml263=enc["relative"].astype(np.float32), observation_mask_abs=mask,
             joints_hml=J, canonical=enc["canonical"], root_quat=enc["root_quat"], r0=enc["r0"])
    cond = {"sha256": M.sha256_file(out / "conditioning.npz"), "coverage": C.coverage_matrix(tm, kint, mask),
            "floor_m": enc["floor"], "observed_features": int(mask.sum()),
            "left_arm_observed_samples": {"before_interior": int(mask[4 + 3 * 19, :kint[0]].sum()), "after_interior": int(mask[4 + 3 * 19, kint[-1] + 1:].sum()),
                                          "inside_interior": int(mask[4 + 3 * 19, kint].sum())}}
    dump(out / "conditioning.json", cond)
    # leak-control input: UNKNOWN rig channels perturbed, re-encoded (masked features change; observed must not)
    rng = np.random.default_rng(LEAK_SEED)
    pert = {ch: (lambda q_: (lambda u: q_))(_qaa(rng.normal(size=3), float(rng.uniform(20, 60)))) for ch in UNKNOWN_RIG_CHANNELS}
    Jx = np.stack([rig.hml_joints(*_pose_with(st, float(u), pert)) for u in tm["u"]])
    encx = C.encode(Jx, "supported", c_ankle())
    np.savez(out / "conditioning_leak.npz", relative_hml263=encx["relative"].astype(np.float32), observation_mask_abs=mask)
    base = {"mode": "infer", "hf_dir": str(HF_DIR), "lfs_sha256": LFS, "respacing": RESPACING, "guidance": GUIDANCE,
            "text": TEXT, "seed": SEED, "length": tm["n_model"]}
    runs = {}
    for name, extra in (("R1", {"input_npz": str(out / "conditioning.npz")}),
                        ("R2_rerun", {"input_npz": str(out / "conditioning.npz")}),
                        ("R3_leak", {"input_npz": str(out / "conditioning_leak.npz"), "perturb_masked_seed": LEAK_SEED})):
        rc, t = cm({**base, **extra, "out_dir": str(out / f"model_{name}")}, out / f"req_{name}.json")
        runs[name] = {"rc": rc, "s": t}
        log["steps"].append([f"infer_{name}", t, rc])
        if rc != 0:
            dump(out / "run_log.json", log)
            raise SystemExit(f"model run {name} failed rc={rc}")
    io1, io2, io3 = (np.load(out / f"model_{n}" / "model_io.npz") for n in ("R1", "R2_rerun", "R3_leak"))
    det = {"R2_output_bitwise_equal": bool(np.array_equal(io1["output_absolute"], io2["output_absolute"])),
           "R3_input_differs_from_R1": bool(not np.array_equal(io1["input_relative"], io3["input_relative"])),
           "R3_effective_conditioning_bitwise_equal": bool(np.array_equal(io1["effective_conditioning_normalized"], io3["effective_conditioning_normalized"])),
           "R3_output_bitwise_equal": bool(np.array_equal(io1["output_absolute"], io3["output_absolute"])),
           "R2_max_abs": float(np.abs(io1["output_absolute"] - io2["output_absolute"]).max()),
           "R3_max_abs": float(np.abs(io1["output_absolute"] - io3["output_absolute"]).max())}
    r1 = load(out / "model_R1" / "result.json")
    g3 = bool(det["R2_output_bitwise_equal"] and det["R3_output_bitwise_equal"] and det["R3_effective_conditioning_bitwise_equal"]
              and det["R3_input_differs_from_R1"] and r1["output_finite"] and cond["left_arm_observed_samples"]["inside_interior"] == 0
              and cond["left_arm_observed_samples"]["before_interior"] > 0 and cond["left_arm_observed_samples"]["after_interior"] > 0)
    res = {"source": "S1", "determinism_and_leak": det, "G3_pass": g3, "model_result": r1, "conditioning": cond, "runs": runs}
    if not (det["R3_output_bitwise_equal"] and det["R3_effective_conditioning_bitwise_equal"]):
        res["STOP"] = "INPUT_LEAK_CONTRACT_FAILURE (D): masked UNKNOWN inputs changed the output"
        dump(out / "run_result.json", res)
        print(json.dumps(det, indent=1))
        return 3
    # ---- projection (interior only) + B_rs ----
    out_abs = io1["output_absolute"].astype(np.float64)
    upC, loC = project_samples(rig, st, tm, kint, out_abs, enc["r0"])
    bu_fn, bl_fn = C.b_slerp_fn(st, "upperarm_l"), C.b_slerp_fn(st, "lowerarm_l")
    Jb, encb, _ = conditioning(rig, st, tm, kint, arm_override(bu_fn, bl_fn))
    upB, loB = project_samples(rig, st, tm, kint, C.rel_to_abs(encb["relative"]), encb["r0"])
    frames = range(C.INTERIOR_ACTION[0], C.INTERIOR_ACTION[1] + 1)
    vals = {}
    for br, (pu, pl) in (("C", (upC, loC)), ("Brs", (upB, loB))):
        ru, rl = C.reconstruct_30(pu, frames), C.reconstruct_30(pl, frames)
        vals[br] = {"upperarm_l": {str(f): M.canon_sign(ru[f], C.stored_key(st, "upperarm_l", f)).tolist() for f in frames},
                    "lowerarm_l": {str(f): M.canon_sign(rl[f], C.stored_key(st, "lowerarm_l", f)).tolist() for f in frames}}
    Jg = C.abs_to_joints(out_abs)
    gen_info = {"generated_vs_observed_left_arm_at_boundary_samples_m": {
        str(k): float(np.linalg.norm(Jg[k, 20] - C.abs_to_joints(io1["input_absolute"].astype(np.float64))[k, 20])) for k in (kint[0] - 1, kint[0], kint[-1], kint[-1] + 1)},
        "generated_root_travel_m": float(np.linalg.norm(Jg[-1, 0, [0, 2]] - Jg[0, 0, [0, 2]])),
        "generated_left_wrist_height_canonical_m": {str(k): float(Jg[k, 20, 1]) for k in kint}}
    dump(out / "projection.json", {"interior_samples": kint, "values": vals, "generated": gen_info,
                                   "transform": "canonical -> undo frame-0 facing -> A^T; swing from W129 B; per sample; 20->30 slerp; NO filtering"})
    # ---- Blender candidates ----
    blends = {"A": base_blend, "B": b_blend}
    for br, note in (("C", "CondMDI random_joints EMA, R1, seed 42"), ("Brs", "diagnostic: frozen B through the 30->20->30 + HML263 path, no model")):
        spec = {"out_blend": str(out / f"{br}_S1.blend"), "interior_action_frames": list(C.INTERIOR_ACTION), "values_wxyz": vals[br],
                "branch": br, "action_name": f"AnimCV_EXP W131 {br} S1", "expected_blend_sha256": S1["baseline_sha"], "note": note}
        dump(out / f"build_{br}_spec.json", spec)
        log["steps"].append([f"build_{br}", blender(STAGE, "build_variant", out / f"build_{br}.log", blend=base_blend,
                                                     spec=out / f"build_{br}_spec.json", report=out / f"build_{br}_report.json")])
        blends[br] = out / f"{br}_S1.blend"
    spec = {"out_blend": str(out / "Cpin_S1.blend"), "interior_action_frames": list(C.INTERIOR_ACTION), "values_wxyz": vals["C"],
            "branch": "C_on_W126V", "action_name": "AnimCV_EXP W131 C_on_W126V S1", "expected_blend_sha256": S1["pin_accepted_sha"],
            "note": "pin-preservation control"}
    dump(out / "build_Cpin_spec.json", spec)
    log["steps"].append(["build_Cpin", blender(STAGE, "build_variant", out / "build_Cpin.log", blend=pin_blend, spec=out / "build_Cpin_spec.json",
                                                report=out / "build_Cpin_report.json")])
    rb = {}
    for br, bl in list(blends.items()) + [("Cpin", out / "Cpin_S1.blend"), ("Pin", pin_blend)]:
        log["steps"].append([f"readback_{br}", readback(bl, out / f"readback_{br}.json", out / f"readback_{br}.log")])
        rb[br] = load(out / f"readback_{br}.json")
    i0, i1 = C.INTERIOR_ACTION
    g4 = {}
    for br in ("C", "Brs"):
        r = rb[br]
        pres = M.preservation(r, rb["A"], (i0, i1))
        ck = M.changed_keys(full_dump(br, blends[br], out)["stored"], st)
        census = r["fcurve_census"]
        g = {"blend_sha256": r["blend_sha256"], "action_name": r["action_name"], "actions_in_file": r["actions_in_file"],
             "single_action": len(r["actions_in_file"]) == 1, "action_name_distinct": r["action_name"] != rb["A"]["action_name"],
             "n_fcurves": census["n_fcurves"], "interp": census["interpolation_census_curves"],
             "key_range": [census["key_frame_min"], census["key_frame_max"]], "scene": r["scene"],
             "changed_keys_channels": sorted({c[0] for c in ck}), "changed_keys_frames": [min((c[2] for c in ck), default=None), max((c[2] for c in ck), default=None)],
             "n_changed_keys": len(ck), "nonarm_changed": pres["nonarm_changed"][:10], "outside_changed": pres["outside_changed"][:10],
             "sub_outside_changed": pres["sub_outside_changed"][:10], "inside_changed_frames": pres["inside_changed_frames"]}
        g["pass"] = bool(g["single_action"] and g["action_name_distinct"] and g["n_fcurves"] == 32 and set(g["interp"]) == {"CONSTANT"}
                         and g["key_range"] == [1.0, 240.0] and r["scene"]["effective_fps"] == 30.0 and r["scene"]["frame_start"] == 1
                         and r["scene"]["frame_end"] == 240 and set(g["changed_keys_channels"]) <= set(C.EDIT_CHANNELS)
                         and all(i0 <= c[2] <= i1 for c in ck) and not pres["nonarm_changed"] and not pres["outside_changed"]
                         and not pres["sub_outside_changed"])
        g4[br] = g
    a_same = M.sha256_file(blends["A"]) == S1["baseline_sha"] and rb["A"]["value_digest"] == S1["value_digest"]
    lineage_same = (M.sha256_file(S1["run"] / "lineage" / "refs.json") == refs_before
                    and M.sha256_file(S1["run"] / "baseline" / "baseline.blend") == S1["baseline_sha"]
                    and M.sha256_file(S1["run"] / S1["pin_accepted"]) == S1["pin_accepted_sha"] and M.sha256_file(S1["w129_B"]) == S1["w129_B_sha"])
    g4["A_untouched_and_digest"] = bool(a_same)
    g4["lineage_and_originals_unchanged"] = bool(lineage_same)
    g4["pass"] = bool(g4["C"]["pass"] and g4["Brs"]["pass"] and a_same and lineage_same)
    prop = load(S1["run"] / S1["pin_request"] / "proposal.json")
    tgt_m = np.asarray(prop["constraint"]["target"])
    pin_act = [f - S1["segment_first"] + 1 for f in prop["pin_frames"]]
    frp, frc, frC, frA = ({f["blender_frame"]: f for f in rb[k]["frames"]} for k in ("Pin", "Cpin", "C", "A"))
    res_pin = {f: float(np.linalg.norm(np.asarray(frp[f]["world"]["hand_l"]["head"]) - tgt_m)) for f in pin_act}
    res_cp = {f: float(np.linalg.norm(np.asarray(frc[f]["world"]["hand_l"]["head"]) - tgt_m)) for f in pin_act}
    g5 = {"pin_action_frames": pin_act, "pin_residual_W126V_m": res_pin, "pin_residual_C_composed_m": res_cp,
          "tolerance_m": prop["constraint"]["tolerance_m"],
          "pin_frames_bit_equal_C_vs_A": all(frC[f]["world"] == frA[f]["world"] for f in pin_act),
          "pin_frames_bit_equal_Cpin_vs_W126V": all(frc[f]["world"] == frp[f]["world"] for f in pin_act),
          "composed_preservation": M.preservation(rb["Cpin"], rb["Pin"], (i0, i1))}
    g5["pass"] = bool(g5["pin_frames_bit_equal_C_vs_A"] and g5["pin_frames_bit_equal_Cpin_vs_W126V"]
                      and all(res_cp[f] == res_pin[f] for f in pin_act) and max(res_cp.values()) <= 1e-6
                      and not g5["composed_preservation"]["nonarm_changed"] and not g5["composed_preservation"]["outside_changed"])
    # ---- quality ----
    hinge = M.hinge_reference(rb["A"])
    win = (i0 - 8, i1 + 8)
    q = {br: M.quality(rb[br], rb["A"], (i0, i1), win, hinge) for br in ("A", "B", "C", "Brs")}
    for br in ("A", "B", "C", "Brs"):
        dmp = bd if br == "A" else full_dump(br, blends[br], out)
        q[br]["arm_local"] = M.armlocal_quality(M.armlocal_wrist(dmp, *win), (i0, i1), win)
        q[br]["press_intent"] = press_intent(rb[br])
    q["Brs_vs_B_max_wrist_mm"] = float(max(np.linalg.norm(np.asarray(fb["world"]["hand_l"]["head"]) - np.asarray(fr_["world"]["hand_l"]["head"])) for fb, fr_ in zip(rb["B"]["frames"], rb["Brs"]["frames"])) * 1e3)
    q["note"] = "world-space wrist metrics include the source-derived torso/object motion (out of the edit domain, identical in all branches); arm_local isolates the edited channels"
    clock = {br: {"fps": rb[br]["scene"]["effective_fps"], "frames": [rb[br]["scene"]["frame_start"], rb[br]["scene"]["frame_end"]],
                  "key_range": [rb[br]["fcurve_census"]["key_frame_min"], rb[br]["fcurve_census"]["key_frame_max"]]} for br in ("A", "B", "C", "Brs")}
    g2 = bool(all(v["fps"] == 30.0 and v["frames"] == [1, 240] and v["key_range"] == [1.0, 240.0] for v in clock.values())
              and kint == list(range(114, 133)))
    res.update({"G2_pass": g2, "clock": clock, "G4": g4, "G5": g5, "quality": q, "hinge_reference": hinge.tolist(),
                "blends": {k: str(v) for k, v in blends.items()}, "readbacks": {k: M.sha256_file(out / f"readback_{k}.json") for k in rb}})
    dump(out / "run_result.json", res)
    vid = render_and_compose(out, blends, rb, log)
    res["video"] = vid
    log["end"] = now()
    dump(out / "run_log.json", log)
    dump(out / "run_result.json", res)
    print(json.dumps({"G2": g2, "G3": g3, "G4": g4["pass"], "G5": g5["pass"], "det": det, "video": vid.get("native_rate")}, indent=1))
    return 0


def render_and_compose(out, blends, rb, log):
    import cv2
    i0, i1 = C.INTERIOR_ACTION
    n, seg0 = 240, S1["segment_first"]
    fa = {f["blender_frame"]: f for f in rb["A"]["frames"]}
    allp = np.array([fa[f]["world"][b][p] for f in fa for b in fa[f]["world"] for p in ("head", "tail")])
    centre = ((allp.min(0) + allp.max(0)) / 2).tolist()
    ext = float(np.max(allp.max(0) - allp.min(0)))
    el = np.array([fa[f]["world"]["lowerarm_l"]["head"] for f in range(i0 - 8, i1 + 9)]).mean(0).tolist()
    views = [{"name": "body", "centre": centre, "offset": [0.0, -6.0, 0.0], "ortho_scale": ext * 1.15},
             {"name": "arm", "centre": el, "offset": [4.5, -4.5, 1.5], "ortho_scale": 1.3}]
    size = 320
    dump(out / "render_jobs.json", {"size": size, "views": views, "frames": [{"blender_frame": f, "markers": []} for f in range(1, n + 1)]})
    for br in ("A", "B", "C", "Brs"):
        log["steps"].append([f"render_{br}", blender(ACTION_STAGE, "render", out / f"render_{br}.log", blend=blends[br], jobs=out / "render_jobs.json",
                                                      out_dir=out / f"frames_{br}")])
    man = load(S1["run"] / "manifest" / "input_manifest.json")
    want = set(range(seg0, seg0 + n))
    cap = cv2.VideoCapture(man["intake"]["clip_path"])
    src, k = {}, 0
    while True:
        ok, im = cap.read()
        if not ok:
            break
        if k in want:
            src[k] = im
        k += 1
    cap.release()
    z = np.load(S1["run"] / "perception" / "m1s0" / "observations_predictions.npz")
    Wd, Hd = man["intake"]["frame_size"]
    pts = np.concatenate([z["input_2d"][kk][z["input_valid"][kk]][:, :2] for kk in sorted(want)]) * [Wd, Hd]
    c0, c1 = pts.min(0), pts.max(0)
    half = max(c1 - c0) * 0.6 + 20
    mid = (c0 + c1) / 2
    crop = [int(max(0, mid[0] - half)), int(max(0, mid[1] - half)), int(min(Wd, mid[0] + half)), int(min(Hd, mid[1] + half))]

    def lab(img, txt, y, col=(255, 255, 255), sc=0.42):
        cv2.putText(img, txt, (6, y), cv2.FONT_HERSHEY_SIMPLEX, sc, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, txt, (6, y), cv2.FONT_HERSHEY_SIMPLEX, sc, col, 1, cv2.LINE_AA)
        return img
    names = {"A": "A original (no-op)", "B": "B slerp (W129, no model)", "C": "C CondMDI (W131)", "Brs": "B_rs diag: B via 30-20-30"}

    def tile(br, f, label, changed):
        body = cv2.imread(str(out / f"frames_{br}" / f"body_f{f:04d}.png"))
        arm = cv2.imread(str(out / f"frames_{br}" / f"arm_f{f:04d}.png"))
        lab(body, label, 16)
        lab(body, f"Action {f}", 34)
        inside = i0 <= f <= i1
        lab(arm, "left-arm close-up", 16)
        lab(arm, "CHANGED interior" if (inside and changed) else ("interior" if inside else "unchanged"), 34,
            (80, 80, 255) if inside else (200, 200, 200))
        return np.vstack([body, arm])

    def source_tile(f):
        kk = f + seg0 - 1
        x0, y0, x1, y1 = crop
        im = src[kk][y0:y1, x0:x1].copy()
        im = cv2.resize(im, (int(im.shape[1] * 2 * size / im.shape[0]), 2 * size))
        lab(im, "SOURCE (fixed crop)", 16)
        lab(im, f"SOURCE frame {kk}  t={kk / 30.0:.3f}s", 34)
        return im

    def bar(width, f):
        b = np.full((34, width, 3), 245, np.uint8)
        Wb = width - 20
        for kk in range(1, n + 1):
            x0 = 10 + int((kk - 1) / n * Wb)
            x1 = 10 + int(kk / n * Wb)
            cv2.rectangle(b, (x0, 4), (max(x0, x1 - 1), 18), (91, 73, 219) if i0 <= kk <= i1 else (220, 220, 220), -1)
        xc = 10 + int((f - 0.5) / n * Wb)
        cv2.line(b, (xc, 0), (xc, 33), (0, 0, 0), 2)
        cv2.putText(b, f"edited interior Action {i0}-{i1} = SOURCE {i0 + seg0 - 1}-{i1 + seg0 - 1} (red); pin Action 132-140 protected; everything else unchanged",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 0, 0), 1)
        return b

    def write_video(path, order, labels, changed):
        wr, written = None, 0
        for f in range(1, n + 1):
            top = np.hstack([source_tile(f)] + [tile(br, f, lb, ch) for br, lb, ch in zip(order, labels, changed)])
            fr_ = np.vstack([top, bar(top.shape[1], f)])
            if wr is None:
                wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (fr_.shape[1], fr_.shape[0]))
            wr.write(fr_)
            written += 1
        wr.release()
        cp = cv2.VideoCapture(str(path))
        nd = 0
        while cp.read()[0]:
            nd += 1
        fps = cp.get(cv2.CAP_PROP_FPS)
        cp.release()
        return {"video": str(path), "sha256": M.sha256_file(path), "frames_written": written, "frames_decoded": nd, "fps": fps,
                "native_rate": nd == n and abs(fps - 30.0) < 1e-6}
    order = ["A", "B", "C", "Brs"]
    vid = write_video(out / "abc_S1_30fps.mp4", order, [names[b] for b in order], [b != "A" for b in order])
    sheets = {"entry": list(range(170, 176)), "midpoint": list(range(183, 189)), "exit": list(range(196, 202)), "overview": list(range(171, 202, 5))}

    def sheet(path, frames_, order_, labels, changed):
        rows = [np.hstack([np.vstack([cv2.resize(source_tile(f), (size, size)),
                                      lab(np.zeros((size, size, 3), np.uint8), f"SOURCE {f + seg0 - 1} / Action {f}", 20)]) for f in frames_])]
        for br, lb, ch in zip(order_, labels, changed):
            rows.append(np.hstack([tile(br, f, lb, ch) for f in frames_]))
        cv2.imwrite(str(path), np.vstack(rows))
        return str(path)
    sh = {k: sheet(out / f"sheet_{k}_S1.png", v, order, [names[b] for b in order], [b != "A" for b in order]) for k, v in sheets.items()}
    perm = list("ABC")
    random.SystemRandom().shuffle(perm)
    bdir = new_dir(out / "blind")
    dump(out / "blind_key_SEALED.json", {"X": perm[0], "Y": perm[1], "Z": perm[2], "note": "do not read before the blinded review is written"})
    bl_labels = [f"panel {x}" for x in "XYZ"]
    bsh = {k: sheet(bdir / f"blind_sheet_{k}_S1.png", v, perm, bl_labels, [True, True, True]) for k, v in sheets.items()}
    bvid = write_video(bdir / "blind_S1_30fps.mp4", perm, bl_labels, [True, True, True])
    vid.update({"sheets": sh, "blind_sheets": bsh, "blind_video": bvid, "blind_key_sha256": M.sha256_file(out / "blind_key_SEALED.json")})
    return vid


# ------------------------------------------------------------------------------------------------ assess ---
def stage_assess(a):
    d = check_declaration(a)
    out = new_dir(a.out_dir)
    pf, idn, ctl = load(a.preflight), load(a.identity), load(a.controls)
    rr = load(Path(a.run_dir) / "S1" / "run_result.json")
    vr = load(a.visual_review)
    key = load(Path(a.run_dir) / "S1" / "blind_key_SEALED.json")
    unblind = lambda x: key.get(x, x)                                              # noqa: E731
    g = {"G0": pf["status"] == "OK_PENDING_IDENTITY" and idn["identity_match"] and idn["hf_files_sha256_ok"],
         "G1": ctl["G1_pass"], "G2": rr["G2_pass"], "G3": rr["G3_pass"], "G4": rr["G4"]["pass"], "G5": rr["G5"]["pass"],
         "G6": ctl["G6_adapter_pass"], "G7": rr["video"]["native_rate"]}
    q = rr["quality"]
    ans = {k: [unblind(x) for x in v] if isinstance(v, list) else (unblind(v) if isinstance(v, str) else
                                                                    {unblind(kk): vv for kk, vv in v.items()} if isinstance(v, dict) else v)
           for k, v in vr["answers_blinded"].items()}
    qa = {"QA1": q["C"]["arm_local"]["n_reversal_flicks"] == 0,
          "QA2": all(q["C"]["arm_local"]["boundary_c0"][k] <= 2 * q["B"]["arm_local"]["boundary_c0"][k] + 10.0 for k in ("entry_step_mm", "exit_step_mm")),
          "QA3": not q["C"]["near_straight_frames"] and not q["C"]["bend_direction_flips"] and not q["C"]["hinge_opposite_frames"],
          "QA4": q["C"]["wrist_dev_from_A_mean_mm"] <= 1.25 * q["B"]["wrist_dev_from_A_mean_mm"] + 25.0,
          "QA5": bool(q["C"]["press_intent"]["pass"]),
          "QA6": bool(ans["V2_most_natural"] == "C" and "C" not in ans["V1_flick"] and "C" not in ans["V3_wrong_elbow"]
                      and ans["V4_least_faithful"] != "C" and ans["V6_upward_press"].get("C") == "yes")}
    if not g["G0"]:
        case = "DATA_BLOCKED"
    elif not (g["G3"] and g["G6"]):
        case = "D"
    elif not (g["G1"] and g["G2"] and g["G4"] and g["G5"] and g["G7"]):
        case = "C"
    elif all(qa.values()):
        case = "A"
    else:
        case = "B"
    rep = {"stage": "assess", "time": now(), "case": case, "gates": g, "quality_criteria": qa, "visual_answers_unblinded": ans,
           "visual_review_sha256": M.sha256_file(a.visual_review), "blind_key": key, "rule": CASE_RULE,
           "note": "visual review is the agent's own blinded review of stills, not human/animator validation; native-rate perceptual review NOT RUN"}
    sha = dump(out / "assessment.json", rep)
    print(json.dumps({"case": case, "gates": g, "qa": qa, "answers": ans, "sha": sha}, indent=1))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=("controls", "declare", "run", "assess"))
    ap.add_argument("--out-dir", required=True)
    for k in ("preflight", "identity", "controls", "declaration", "declaration-sha", "run-dir", "visual-review"):
        ap.add_argument("--" + k)
    a = ap.parse_args()
    return {"controls": stage_controls, "declare": stage_declare, "run": stage_run, "assess": stage_assess}[a.stage](a)


if __name__ == "__main__":
    sys.exit(main())
