#!/usr/bin/env python3
"""W135 (EXPERIMENTAL, opt-in): source-constrained MotionBricks elbow-swivel prior on the real S1 Blender Action.

Reuses the FROZEN W129 outputs (A baseline, B endpoint slerp, C MotionBricks projection); no model is run.
Stages (each refuses an existing output directory; no AnimCV run directory or lineage is written):

  preflight  identity / rights gate: repo state, W134 COMPLETE, full SHA-256 of S1 clip, A, B, C, W126V-accepted, W126
             lineage, W129 declaration / assessment / run record; byte copies of the frozen inputs; fresh Blender dumps
             and readbacks of A, B, C (independent recompute, no W129 JSON reused for geometry).
  declare    prospective declaration (bindings, convention, solver, reference, thresholds, controls, CASE rule,
             instrument hashes) BEFORE any D/E exists.  ``outcomes_inspected: false``.
  controls   real-data fail-closed controls (missing / corrupt C, mirrored / time-shifted / side-swapped prior, mirrored
             geometry, out-of-domain channel / frame build, pin overlap, unreachable wrist, elbow flip, non-30-fps) + the
             numpy-FK reproduction check against Blender.  Never builds D or E.
  run        prior extraction + verification, D and E solve, D / E / Dpin / Epin / Drev builds, fresh-process readbacks
             and dumps, mechanical gates, learned-effect and quality metrics, renders, native-rate A/B/C/D/E video and
             contact sheets.
  assess     apply the frozen CASE rule to the run evidence + the agent's visual review file.

    PY=C:/Projects/AnimCV/.venv_build_full/Scripts/python.exe; export PYTHONPATH=src
    $PY -X utf8 scripts/w135_elbow_prior_harness.py <stage> --out-dir <dir> [...]
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import numpy as np                                                    # noqa: E402

import app.experimental_elbow_prior as E                              # noqa: E402
import app.experimental_motionbricks as M                             # noqa: E402
import w129_motionbricks_harness as H                                 # noqa: E402

W135 = Path("C:/Users/dna10/animcv-w135")
W129R = Path("C:/Users/dna10/animcv-w129/results")
S1 = H.SOURCES["S1"]
STAGE, ACTION_STAGE = H.STAGE, H.ACTION_STAGE
INSTRUMENTS = ["src/app/experimental_elbow_prior.py", "scripts/w135_elbow_prior_harness.py",
               "tests/test_experimental_elbow_prior.py", "src/app/experimental_motionbricks.py",
               "src/blender/experimental_motionbricks_stage.py", "src/blender/experimental_action_stage.py",
               "scripts/w129_motionbricks_harness.py"]
FROZEN = {   # full SHA-256 from the official W129 records (run_result.json G4, preflight, W126 sources)
    "A": ("C:/Users/dna10/animcv-w129/inputs/S1_baseline.blend", S1["baseline_sha"]),
    "A_w126_original": (str(S1["run"] / "baseline" / "baseline.blend"), S1["baseline_sha"]),
    "B": (str(W129R / "run/S1/B_S1.blend"), "8899f43341ac1a7a565229a9551b1f8a79943cc038ad519e5544af89e772a33f"),
    "C": (str(W129R / "run/S1/C_S1.blend"), "e04efb3552f9808f8ae89520284b0d94e860a986c1e6ec475ed6b07683f9ef04"),
    "Cpin": (str(W129R / "run/S1/Cpin_S1.blend"), "3116d2ea3340d8356beb6adbf5f60f33c026ca92f47f7216137b4a00f6e31b02"),
    "W126V_accepted": ("C:/Users/dna10/animcv-w129/inputs/S1_W126V_accepted.blend", S1["pin_accepted_sha"]),
    "W126V_accepted_original": (str(S1["run"] / S1["pin_accepted"]), S1["pin_accepted_sha"]),
    "W126_refs": (str(S1["run"] / "lineage" / "refs.json"), "c0f1157dc7f067736059f754e23070de8ab5e425ec858e7415e2d7df17389b1b"),
    "W126V_proposal": (str(S1["run"] / "edits" / "W126V" / "proposal.json"), "582174bfcd3a284f9037409c945e2440b182a0c2593e618511b9effb280a9c84"),
    "S1_clip": (S1["clip"], S1["clip_sha"]),
    "W129_declaration_v2": (str(W129R / "declare_v2/declaration.json"), "0c34aab0755c24a7e30a43b11a356c9f5dc0c54e5adda39a5b54b8d6e9f0b6f8"),
    "W129_assessment": (str(W129R / "assess/assessment.json"), "df6693d22d58cb95f4e130e16e757ab9519c84fe359323560cf530da8b42791c"),
    "W129_run_result": (str(W129R / "run/S1/run_result.json"), "9c88961523fa41881fd87e58d366de23138b24baca316dc33403c40f76da9a83"),
    "W129_preflight": (str(W129R / "preflight/preflight.json"), "46119b9c5273e0fe038809f959ef12664abc89794085b94ec8aa351b44813868"),
    "W129_engine_output": (str(W129R / "run/S1/model_R1/engine_output.npz"), "8a528ef1bf7976f22bd6212eddf7c5b525ee5f30e27ee01f501d565baac6a100"),
    "W129_raw_model_io": (str(W129R / "run/S1/model_R1/raw_model_io.npz"), "1ad837e417b69531b35aa3108fccd6c99e05b4f924741b56d58ceec44accfb6f"),
    "W129_projection": (str(W129R / "run/S1/projection.json"), "e331344c8023da4b398ae3ac55c3be21d3955c486d6334c708a938fe41cf20aa"),
    "W129_evidence_manifest": (str(W129R / "evidence_manifest.json"), "b07154ed1b20da67a247fa4d9d5a28222ecb1087cc39d172bbe0af8eca113c02"),
}
RIGHTS_FILE = W135 / "rights" / "nvidia_open_model_agreement.html"


def _html_text(p):
    import html
    import re
    t = Path(p).read_text(encoding="utf-8", errors="replace")
    t = re.sub(r"<script.*?</script>|<style.*?</style>", "", t, flags=re.S)
    t = html.unescape(re.sub(r"<[^>]+>", "\n", t))
    return re.sub(r"\n\s*\n+", "\n", t)


RIGHTS_REQUIRED = ["NVIDIA Open Model Agreement", "Works are commercially usable.",
                   "NVIDIA does not claim ownership to any outputs generated using the Works or Derivative Works.",
                   "perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable license"]
INTERIOR = E.INTERIOR
WINDOW = (164, 207)
NAMES = {"A": "A original (no-op)", "B": "B slerp (no model)", "C": "C MotionBricks direct (W129)",
         "D": "D B-wrist + C elbow swivel", "E": "E B-wrist + neutral (B) swivel"}

load, dump, new_dir, now, git, lf_sha = H.load, H.dump, H.new_dir, H.now, H.git, H.lf_sha


def blender(script, mode, log, **kw):
    return H.blender(script, mode, log, **kw)


def readback(blend, rep, log):
    return blender(ACTION_STAGE, "readback", log, blend=blend, report=rep, subframes="0.25,0.5,0.75")


def bdump(blend, rep, log):
    return blender(STAGE, "dump", log, blend=blend, report=rep)


def check_declaration(a):
    if not a.declaration:
        raise SystemExit("--declaration required")
    if M.sha256_file(a.declaration) != a.declaration_sha:
        raise SystemExit("STOP INTEGRITY_FAILED: declaration changed")
    d = load(a.declaration)
    changed = [f for f, h in d["instrument_sha256_lf"].items() if lf_sha(f) != h]
    if changed:
        raise SystemExit(f"STOP DECLARATION_INVALID: instrument changed {changed}")
    return d


def frozen_hashes():
    out = {}
    for k, (p, h) in FROZEN.items():
        ok = Path(p).is_file()
        got = M.sha256_file(p) if ok else None
        out[k] = {"path": p, "expected": h, "sha256": got, "ok": bool(ok and got == h)}
    return out


# ------------------------------------------------------------------------------------------------ preflight ---
def stage_preflight(a):
    out = new_dir(a.out_dir)
    rep = {"stage": "preflight", "time": now(), "head": git("rev-parse", "HEAD"), "instruction_blob": git("rev-parse", "HEAD:docs/next_batch.md"),
           "origin_head": git("rev-parse", "origin/arch/single_frame_first"),
           "w134_complete": "AUTOMATION_BATCH_STATUS: COMPLETE" in (ROOT / "docs/134_WORKLOG_ARDY_SOMA_SPARSE_CONDITIONING_LLAMA_ACCESS_BLOCKED.md").read_text(encoding="utf-8")}
    rep["frozen"] = frozen_hashes()
    txt = _html_text(RIGHTS_FILE) if RIGHTS_FILE.is_file() else ""
    mb_lic = (H.MB_ROOT / "motionbricks/assets/onnx/LICENSE.md").read_text(encoding="utf-8")
    rep["rights"] = {
        "model_licence_file": mb_lic, "model_licence_file_sha256": M.sha256_file(H.MB_ROOT / "motionbricks/assets/onnx/LICENSE.md"),
        "agreement_url": "https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/",
        "agreement_html_sha256": M.sha256_file(RIGHTS_FILE) if RIGHTS_FILE.is_file() else None,
        "agreement_version_lines": [l.strip() for l in txt.splitlines() if "Release Date" in l or l.strip().startswith("(v.")],
        "required_clauses_present": {c: c in txt for c in RIGHTS_REQUIRED},
        "use": "local evaluation of the already generated W129 output (a model OUTPUT, not the Work); no redistribution of weights or outputs",
    }
    rep["rights"]["permitted"] = bool(txt) and all(rep["rights"]["required_clauses_present"].values()) and "NVIDIA Open Model Agreement" in mb_lic
    rep["blender"] = H.subprocess.run([H.BLENDER, "-b", "--version"], capture_output=True, text=True).stdout.splitlines()[:1]
    ok = rep["instruction_blob"] == "38de1e2e155e39c7f168373ca9b053418497ab0c" and rep["w134_complete"] and rep["rights"]["permitted"] \
        and all(v["ok"] for v in rep["frozen"].values()) and bool(rep["blender"])
    if ok:
        inp = new_dir(out / "inputs")
        for k in ("A", "B", "C", "W126V_accepted"):
            shutil.copyfile(FROZEN[k][0], inp / f"{k}.blend")
            if M.sha256_file(inp / f"{k}.blend") != FROZEN[k][1]:
                ok = False
        for k in ("A", "B", "C"):
            bdump(inp / f"{k}.blend", out / f"dump_{k}.json", out / f"dump_{k}.log")
            readback(inp / f"{k}.blend", out / f"readback_{k}.json", out / f"readback_{k}.log")
        readback(inp / "W126V_accepted.blend", out / "readback_Pin.json", out / "readback_Pin.log")
        # cross-check against the W129 records (diagnostic; the fresh dumps are authoritative here)
        x = {}
        for k in ("A", "B", "C"):
            w = load(W129R / "run/S1" / f"readback_{k}.json")
            f = load(out / f"readback_{k}.json")
            x[k] = {"world_bit_equal_to_W129_readback": all(p["world"] == q["world"] for p, q in zip(w["frames"], f["frames"])),
                    "fresh_fps": f["scene"]["effective_fps"], "fresh_blend_sha256": f["blend_sha256"]}
        rep["fresh_vs_w129"] = x
    rep["status"] = "OK" if ok else "DATA_BLOCKED"
    dump(out / "preflight.json", rep)
    print(json.dumps({"status": rep["status"], "rights": rep["rights"]["permitted"], "frozen_ok": {k: v["ok"] for k, v in rep["frozen"].items()},
                      "fresh_vs_w129": rep.get("fresh_vs_w129")}, indent=1))
    return 0 if ok else 3


# ------------------------------------------------------------------------------------------------ declaration ---
MECH_GATES = {
    "M0": "identity/rights: preflight OK (instruction blob, W134 COMPLETE, rights permitted, every frozen full SHA-256 equal, byte copies equal)",
    "M1": "prior provenance: the C prior is re-derivable bit-for-bit from the bound C file (verify_prior) and the solver raised no refusal for D or E",
    "M2": "Action: D and E each a separate .blend with exactly 1 separately named Action; 32 CONSTANT F-curves keyed 1..240; scene 30 fps, 1..240",
    "M3": "footprint: changed stored keys vs A only upperarm_l/lowerarm_l rotation at Action 172..199; every non-left-arm world point and the object matrix bit-equal to A on all 240 integer frames and all 717 subframes; left arm bit-equal outside 172..199; no subframe change outside [172,200)",
    "M4": "wrist tracking: |wrist_X - wrist_B| <= 1 mm (world, fresh readback) on every interior frame for X in {D,E}; entry 171->172 and exit 199->200 wrist steps differ from B's by <= 1 mm",
    "M5": "rig validity: upperarm/lowerarm world length range <= 0.1 mm over 171..200; lowerarm_l stored key equals B's bit-for-bit on the interior (B's elbow-local flexion/hinge relation kept); no near-straight (> 175 deg) interior frame",
    "M6": "hard pin: D and E composed onto the W126V-accepted .blend keep every Action 132..140 hand_l residual equal to W126V's and <= 1e-6 m; pin frames bit-equal; composed footprint as M3 relative to W126V",
    "M7": "reversibility: D with A's interior keys written back (Drev) is world-bit-equal to A on every integer frame and subframe",
    "M8": "solver neutrality: E (B's own swivel through the identical solver) equals B in world points within 1e-6 m on every frame",
    "M9": "integrity: A, B, C, Cpin, W126V-accepted, W126 refs/proposal, S1 clip and W129 records have identical SHA-256 after the run",
    "M10": "negative controls (controls stage) all fail closed",
    "M11": "native-rate video: 240 frames written and decoded at 30.000 fps",
}
LEARNED_RULE = ("LEARNED_EFFECT iff max over interior frames of |elbow_D - elbow_E| (world, fresh readback) >= 10 mm AND max |swivel_D - swivel_E| >= 3 deg; "
                "otherwise NO_LEARNED_EFFECT")
QUALITY = {
    "Q1": "D arm-local elbow entry (171->172) and exit (199->200) steps each <= 2 x B's corresponding step + 10 mm",
    "Q2": "D has 0 arm-local elbow reversal flicks (consecutive steps both > 20 mm with negative dot) in steps 171->172 .. 199->200",
    "Q3": "D consecutive-frame swivel change <= 30 deg on every step 171->172 .. 199->200",
    "Q4": "agent still-frame review: D judged more natural AND more faithful to the source overhead-press intent than both B and E, and D not named for a flick, wrong elbow or detour",
    "Q5": "native-rate perceptual review of the 30 fps video performed (this agent cannot perceive video playback: declared NOT REVIEWED in advance, so Q5 cannot pass in this run)",
}
CASE_RULE = {
    "DATA_BLOCKED": "rights, S1/A/B/C/pin or Blender unavailable, or any frozen hash mismatch before testing (preflight not OK)",
    "D": "declaration/instrument changed, a negative control does not fail closed (M10), integrity (M9) fails, or the C prior fails provenance verification",
    "C": "the solver refuses the real data (REPRESENTATION/GEOMETRY singularity, unreachable wrist, ELBOW_FLIP) or any of M2-M8, M11 fails",
    "B": "all mechanical gates pass but NO_LEARNED_EFFECT, or any of Q1-Q5 fails",
    "A": "all mechanical gates pass AND LEARNED_EFFECT AND Q1-Q5 pass",
    "precedence": "DATA_BLOCKED > D > C > B > A; a mechanical PASS alone never yields A",
}


def stage_declare(a):
    out = new_dir(a.out_dir)
    pf = load(a.preflight)
    if pf["status"] != "OK":
        raise SystemExit("preflight not OK: DATA_BLOCKED")
    pdir = Path(a.preflight).parent
    dA, dB, dC = (load(pdir / f"dump_{k}.json") for k in "ABC")
    best, rows = E.select_reference([dB, dC], range(E.CONTEXT_LAST, E.TARGET_FIRST + 1))
    if best != "lateral":
        raise SystemExit(f"STOP DECLARATION_INVALID: reference rule selects {best}, instrument hard-codes lateral")
    d = {"schema": "animcv_exp_w135_declaration_v1", "time": now(), "outcomes_inspected": False,
         "D_or_E_exists": any((W135 / "results").glob("run*/D_S1.blend")),
         "batch": 135, "starting_commit": "5b113ed930ab31a2cee713cd9e8a0ae1da3994bb",
         "instruction_blob": "38de1e2e155e39c7f168373ca9b053418497ab0c", "repo_head_at_declaration": git("rev-parse", "HEAD"),
         "preflight_sha256": M.sha256_file(a.preflight),
         "bindings": {k: {"path": v["path"], "sha256": v["sha256"]} for k, v in pf["frozen"].items()},
         "fresh_dump_sha256": {k: M.sha256_file(pdir / f"dump_{k}.json") for k in "ABC"},
         "rights": {"agreement_html_sha256": pf["rights"]["agreement_html_sha256"], "version": pf["rights"]["agreement_version_lines"],
                    "determination": "PERMITTED: NVIDIA Open Model Agreement grants a perpetual, worldwide, royalty-free licence to use/reproduce/prepare derivative works and states NVIDIA does not claim ownership of outputs; this run evaluates an already generated output locally and redistributes nothing"},
         "time_rule": "native 30 fps; Action = SOURCE - 78; interior Action 172..199 = SOURCE 250..277; frames 171 and 200 frozen; every prior value is read at the SAME Action frame id (no shift, no resampling)",
         "convention": {"space": "Blender armature space from fresh Blender dumps; arm-local = evaluated clavicle_l pose matrix (parent of upperarm_l); world = fresh readback (metres)",
                        "points": "S = upperarm_l head, E = lowerarm_l head, W = hand_l head (= lowerarm_l tail)",
                        "fk": "P_b = P_parent (Rest_parent^-1 Rest_b) Basis_b, rotation channels only; numpy reproduction of Blender pose matrices verified on A/B/C before declaration (max 3.8e-5 armature units)",
                        "swivel": "psi = atan2(a.(r_p x u), r_p.u); a = unit(W-S), u = (E-S) minus its a-component, r_p = r minus its a-component",
                        "reference": "r = rig lateral (armature +X at REST) expressed once in the rest clavicle_l frame, then carried by the posed clavicle_l frame",
                        "side": "left only (_l bones); proper rotations only (improper/mirrored matrices refused)"},
         "solver": {"name": "rigid swivel about B's own shoulder-wrist axis",
                    "rule": "for each interior frame f: G = rotation by wrap(psi_target(f) - psi_B(f)) about the armature-space axis through S_B(f) and W_B(f); upperarm_l basis' = M^T G M basis_B with M = Rot(K) Rot(Rest_clav)^T Rot(Rest_upperarm); lowerarm_l basis = B's stored key (unchanged)",
                    "D": "psi_target = psi_C(f) (the learned prior, verified against the bound C file)",
                    "E": "psi_target = psi_B(f) (neutral; C withheld); expected numerically identical to B (solver neutrality M8)",
                    "fixed": "S, W (B's wrist), both bone lengths, elbow flexion angle, lowerarm_l key, timestamps, all non-left-arm channels",
                    "regularisation": "NONE: no smoothing, no blending, no endpoint correction, no weights, no iteration",
                    "branch_choice": "unique: the swivel fully determines the elbow on the circle; wrap to (-pi, pi]",
                    "singularities": {"reference": f"refuse if sin(axis, reference) < sin 15 deg = {E.MIN_REF_MARGIN_SIN:.4f} for B or C",
                                      "straight_arm": f"refuse if elbow offset / upper length < sin 10 deg = {E.MIN_ELBOW_OFFSET_SIN:.4f}",
                                      "flip": f"refuse if the target swivel changes by > {E.MAX_SWIVEL_STEP_DEG} deg between consecutive frames 171..200",
                                      "reach": "refuse if |S-W| outside [|L1-L2|, L1+L2]",
                                      "post-solve": "refuse if numpy-FK wrist/shoulder error > 1e-3 armature units (0.01 mm), length error > 1e-3, swivel error > 1e-6 rad"}},
         "prior": {"extraction": "C's swivel psi_C(f), f = 172..199, from a fresh Blender dump of the bound C file, using only C's own S/E/W in C's own clavicle_l frame",
                   "provenance": "W129 C: MotionBricks-SOMA-RP-v1 primary sample R1 (ONNX 856419dc..., NVlabs/motionbricks 54fc563), MODEL-DERIVED, not observed truth",
                   "excluded": "C's wrist position/path (never a target), root/object/spine (factored out by the clavicle frame; C equals A there), right arm, timing (identity frame map)",
                   "note": "the swivel is by definition measured about C's own shoulder-wrist axis; this direction is used only to define C's elbow swivel, never as a D target"},
         "reference_selection": {"a_priori_choice": "anterior (armature -Y)",
                                 "pre_declaration_observation": "input-only margin inspection of B and C on 171..200 (no swivel values, no D/E) found the anterior reference near-singular for C (min sin 0.165 < 0.259)",
                                 "declared_rule": "choose among {anterior, lateral, superior} rig axes the one with the largest minimum axis-reference margin over B and C, frames 171..200 (uses only axis directions)",
                                 "margins": rows, "selected": best},
         "fixed_wrist_target": "W129 B evaluated hand_l head at Action 171..200 (GT-free, reference-derived rig trajectory; not ground truth)",
         "domain": {"channels": list(E.EDIT_CHANNELS), "frames": list(INTERIOR), "pin_action": [E.PIN_ACTION[0], E.PIN_ACTION[-1]],
                    "pin_target_m": [-0.0927, 0.2199, 1.4506], "pin_tolerance_m": 0.001},
         "mechanical_gates": MECH_GATES, "learned_effect_rule": LEARNED_RULE, "quality_criteria": QUALITY, "case_rule": CASE_RULE,
         "metrics": ["arm-local elbow trajectory/steps/flicks/jerk", "bend-plane normal step", "swivel angle, angular velocity, acceleration",
                     "wrist world + arm-local (W129 quality/armlocal_quality)", "entry/exit steps", "world shoulder motion (parent-induced, identical in all branches)",
                     "elbow pointing direction (anterior/lateral/superior components)"],
         "controls": ["missing C", "corrupt C (byte-modified copy)", "mirrored prior (negated swivel)", "mirrored geometry (reflected dump)",
                      "time-shifted prior (+1 frame)", "side-swapped prior", "C prior verified against B", "out-of-domain channel build (spine_01)",
                      "out-of-domain frame build (171 and 200)", "pin overlap request", "unreachable wrist target (2x reach)",
                      "elbow flip (pi jump in a target swivel)", "non-30-fps scene", "numpy FK reproduction vs Blender on A/B/C"],
         "review_protocol": "UNBLINDED agent review of fixed-time contact sheets (entry 170-175, middle 183-188, exit 196-201) and an overview strip; D vs B vs E vs source; still frames only; written to a review JSON before assess. Native-rate playback is NOT perceived by this agent (Q5 NOT REVIEWED). No human/animator review.",
         "stop_conditions": ["RIGHTS_UNRESOLVED", "DATA_BLOCKED", "REPRESENTATION_BLOCKED (C swivel singular)", "GEOMETRY_BLOCKED (solver refusal)",
                             "INTEGRITY_FAILED", "DECLARATION_INVALID", "no tuning, no other reference/threshold/prior after any D/E output"],
         "instrument_sha256_lf": {f: lf_sha(f) for f in INSTRUMENTS}}
    sha = dump(out / "declaration.json", d)
    print(json.dumps({"declaration_sha256": sha, "reference": best, "margins": rows}, indent=1))
    return 0


# ------------------------------------------------------------------------------------------------ controls ---
def _expect(name, fn, code, res):
    try:
        fn()
        res[name] = {"pass": False, "note": "NOT refused"}
    except E.PriorStop as e:
        res[name] = {"pass": e.code == code, "code": e.code, "expected": code}


def stage_controls(a):
    check_declaration(a)
    out = new_dir(a.out_dir)
    pdir = Path(a.preflight).parent
    dA, dB, dC = (load(pdir / f"dump_{k}.json") for k in "ABC")
    shaB, shaC = FROZEN["B"][1], FROZEN["C"][1]
    res = {}
    # numpy FK reproduction of Blender pose matrices (A/B/C, all frames)
    mx = 0.0
    for D in (dA, dB, dC):
        rest, st = D["rest"], D["stored"]
        Rc, Ru, Rl, Rh = (E.rot(np.asarray(rest[b]["matrix_local"], float)) for b in (E.PARENT, E.UPPER, E.LOWER, E.HAND))
        for fr in D["frames"]:
            f = fr["blender_frame"]
            K = np.asarray(fr["pose_matrix"][E.PARENT], float)
            q = lambda ch: np.array([st[ch][str(c)][str(f)] for c in range(4)])    # noqa: E731
            Pu = K @ E._m4(Rc.T @ Ru, E._rel(rest, E.PARENT, E.UPPER)) @ E._m4(E.wxyz_to_mat(q(E.UPPER)), np.zeros(3))
            Pl = Pu @ E._m4(Ru.T @ Rl, E._rel(rest, E.UPPER, E.LOWER)) @ E._m4(E.wxyz_to_mat(q(E.LOWER)), np.zeros(3))
            Ph = Pl @ E._m4(Rl.T @ Rh, E._rel(rest, E.LOWER, E.HAND))
            for b, Pm in ((E.UPPER, Pu), (E.LOWER, Pl), (E.HAND, Ph)):
                mx = max(mx, float(np.abs(Pm - np.asarray(fr["pose_matrix"][b], float)).max()))
    res["fk_reproduction_max_abs_armature_units"] = mx
    res["fk_pass"] = mx < 1e-3
    neg = {}
    # missing / corrupt C (file identity)
    miss = W135 / "absent" / "C.blend"
    neg["missing_C"] = {"pass": not miss.exists(), "code": "DATA_BLOCKED_MISSING_C" if not miss.exists() else None}
    bad = out / "C_corrupt.blend"
    raw = bytearray(Path(FROZEN["C"][0]).read_bytes())
    raw[len(raw) // 2] ^= 0xFF
    bad.write_bytes(bytes(raw))
    neg["corrupt_C"] = {"pass": M.sha256_file(bad) != shaC, "code": "INTEGRITY_FAILED_HASH", "sha256": M.sha256_file(bad)}
    bad.unlink()
    _expect("corrupt_C_dump_identity", lambda: E.build_prior({**dC, "blend_sha256": "0" * 64}, shaC), "SOURCE_IDENTITY", neg)
    good = E.build_prior(dC, shaC)
    E.verify_prior(good, dC, "LEARNED_ELBOW_SWIVEL")
    m = copy.deepcopy(good)
    m["psi_rad"] = {k: -v for k, v in good["psi_rad"].items()}
    m["content_sha256"] = E.sha256_json(m["psi_rad"])
    _expect("mirrored_prior", lambda: E.verify_prior(m, dC, "LEARNED_ELBOW_SWIVEL"), "PRIOR_PROVENANCE_MISMATCH", neg)
    ts = copy.deepcopy(good)
    ts["psi_rad"] = {str(int(k) + 1): v for k, v in good["psi_rad"].items()}
    ts["content_sha256"] = E.sha256_json(ts["psi_rad"])
    _expect("time_shifted_prior", lambda: E.verify_prior(ts, dC, "LEARNED_ELBOW_SWIVEL"), "PRIOR_FRAME_IDENTITY", neg)
    ts2 = copy.deepcopy(good)
    ks = sorted(good["psi_rad"], key=int)
    ts2["psi_rad"] = {k: good["psi_rad"][ks[min(i + 1, len(ks) - 1)]] for i, k in enumerate(ks)}   # same ids, values shifted
    ts2["content_sha256"] = E.sha256_json(ts2["psi_rad"])
    _expect("time_shifted_values_same_ids", lambda: E.verify_prior(ts2, dC, "LEARNED_ELBOW_SWIVEL"), "PRIOR_PROVENANCE_MISMATCH", neg)
    sd = copy.deepcopy(good)
    sd["side"] = "r"
    _expect("side_swapped_prior", lambda: E.verify_prior(sd, dC, "LEARNED_ELBOW_SWIVEL"), "PRIOR_SIDE_MISMATCH", neg)
    _expect("C_prior_against_B", lambda: E.verify_prior(good, dB, "LEARNED_ELBOW_SWIVEL"), "SOURCE_IDENTITY", neg)
    mir = copy.deepcopy(dC)
    Rf = np.diag([-1.0, 1.0, 1.0, 1.0])
    for fr in mir["frames"]:
        for b in fr["pose_matrix"]:
            fr["pose_matrix"][b] = (Rf @ np.asarray(fr["pose_matrix"][b])).tolist()
    _expect("mirrored_geometry_dump", lambda: E.build_prior(mir, shaC), "IMPROPER_ROTATION", neg)
    nf = copy.deepcopy(dC)
    nf["scene"]["fps"] = 24
    _expect("non_30fps_scene", lambda: E.build_prior(nf, shaC), "NOT_30_FPS", neg)
    neutral = E.neutral_prior(dB, shaB)["psi_rad"]
    _expect("pin_overlap_request", lambda: E.solve(dB, dA, neutral, frames=range(130, 158)), "PIN_OVERLAP", neg)
    _expect("frame_outside_interior_request", lambda: E.solve(dB, dA, neutral, frames=range(171, 201)), "DOMAIN_FRAMES", neg)
    _expect("channel_outside_domain_request", lambda: E.check_domain(range(172, 200), channels=("spine_01",)), "DOMAIN_CHANNEL", neg)
    tg = {}
    for f in range(INTERIOR[0], INTERIOR[1] + 1):
        S, _, Wp = E.ArmFrame(dB, f).SEW()
        tg[f] = S + (Wp - S) * 2.0 * (sum(E.bone_lengths(dB["rest"])) / np.linalg.norm(Wp - S))
    _expect("unreachable_wrist_target", lambda: E.solve(dB, dA, neutral, wrist_targets=tg), "WRIST_UNREACHABLE", neg)
    fl = dict(neutral)
    fl["185"] = E.wrap(fl["185"] + np.pi)
    _expect("elbow_flip_target", lambda: E.solve(dB, dA, fl), "ELBOW_FLIP", neg)
    # Blender-side refusals of out-of-domain builds (reusing the unchanged W129 stage)
    for name, vals in (("blender_channel_outside_domain", {"spine_01": {"180": [1, 0, 0, 0]}}),
                       ("blender_frame_171", {"upperarm_l": {"171": [1, 0, 0, 0]}}),
                       ("blender_frame_200", {"lowerarm_l": {"200": [1, 0, 0, 0]}})):
        spec = {"out_blend": str(out / f"neg_{name}.blend"), "interior_action_frames": list(INTERIOR), "values_wxyz": vals,
                "branch": "NEG", "action_name": "NEG", "expected_blend_sha256": FROZEN["A"][1]}
        dump(out / f"neg_{name}_spec.json", spec)
        try:
            blender(STAGE, "build_variant", out / f"neg_{name}.log", blend=pdir / "inputs" / "A.blend", spec=out / f"neg_{name}_spec.json",
                    report=out / f"neg_{name}_report.json")
            neg[name] = {"pass": False}
        except RuntimeError:
            txt = (out / f"neg_{name}.log").read_text()
            neg[name] = {"pass": ("REFUSED_" in txt) and not (out / f"neg_{name}.blend").exists(),
                         "refusal": [l for l in txt.splitlines() if "REFUSED_" in l][:1]}
    res["negative_controls"] = neg
    res["all_fail_closed"] = bool(all(v["pass"] for v in neg.values()))
    res["note"] = "no D or E prior was solved or written in this stage; the neutral prior is used only inside refused requests"
    dump(out / "controls.json", res)
    print(json.dumps({"fk": mx, "all_fail_closed": res["all_fail_closed"], "neg": {k: v["pass"] for k, v in neg.items()}}, indent=1))
    return 0 if (res["all_fail_closed"] and res["fk_pass"]) else 3


# ------------------------------------------------------------------------------------------------ run ---
def _frames(rb):
    return {f["blender_frame"]: f for f in rb["frames"]}


def stage_run(a):
    d = check_declaration(a)
    ctl = load(a.controls)
    if not (ctl["all_fail_closed"] and ctl["fk_pass"]):
        raise SystemExit("STOP INTEGRITY_FAILED: controls did not fail closed")
    pdir = Path(a.preflight).parent
    out = new_dir(Path(a.out_dir))
    log = {"stage": "run", "start": now(), "steps": []}
    inp = pdir / "inputs"
    for k in ("A", "B", "C", "W126V_accepted"):
        if M.sha256_file(inp / f"{k}.blend") != FROZEN[k][1]:
            raise SystemExit(f"STOP INTEGRITY_FAILED {k}")
    dA, dB, dC = (load(pdir / f"dump_{k}.json") for k in "ABC")
    for k, dd in (("A", dA), ("B", dB), ("C", dC)):
        if M.sha256_file(pdir / f"dump_{k}.json") != d["fresh_dump_sha256"][k] or dd["blend_sha256"] != FROZEN[k][1]:
            raise SystemExit(f"STOP INTEGRITY_FAILED dump {k}")
    res = {"declaration_sha256": a.declaration_sha}
    # ---- prior + solve ----
    try:
        prior = E.build_prior(dC, FROZEN["C"][1])
        E.verify_prior(prior, dC, "LEARNED_ELBOW_SWIVEL")
        neutral = E.neutral_prior(dB, FROZEN["B"][1])
        E.verify_prior(neutral, dB, "NEUTRAL_B_SWIVEL")
        dump(out / "prior_C.json", prior)
        dump(out / "prior_E_neutral.json", neutral)
        vD, diagD = E.solve(dB, dA, prior["psi_rad"])
        vE, diagE = E.solve(dB, dA, neutral["psi_rad"])
    except E.PriorStop as e:
        res["STOP"] = {"code": e.code, "detail": e.detail}
        dump(out / "run_result.json", res)
        print(json.dumps(res, indent=1))
        return 3
    dump(out / "solve_D.json", {"values_wxyz": vD, "diag": diagD})
    dump(out / "solve_E.json", {"values_wxyz": vE, "diag": diagE})
    res["solver"] = {br: {"max_wrist_err_units": max(x["wrist_err"] for x in dg["per_frame"].values()),
                          "max_len_err_units": max(max(x["upper_len_err"], x["lower_len_err"]) for x in dg["per_frame"].values()),
                          "max_abs_delta_deg": float(np.degrees(max(abs(x["delta_rad"]) for x in dg["per_frame"].values()))),
                          "max_elbow_shift_units": max(x["elbow_shift"] for x in dg["per_frame"].values()),
                          "min_margin_ref": min(x["margin_ref_sin"] for x in dg["per_frame"].values()),
                          "min_margin_elbow": min(x["margin_elbow_sin"] for x in dg["per_frame"].values())}
                     for br, dg in (("D", diagD), ("E", diagE))}
    # ---- builds ----
    blends = {"A": inp / "A.blend", "B": inp / "B.blend", "C": inp / "C.blend"}
    for br, vals, base, base_sha, nm in (("D", vD, inp / "A.blend", FROZEN["A"][1], "AnimCV_EXP W135 D S1"),
                                         ("E", vE, inp / "A.blend", FROZEN["A"][1], "AnimCV_EXP W135 E S1"),
                                         ("Dpin", vD, inp / "W126V_accepted.blend", FROZEN["W126V_accepted"][1], "AnimCV_EXP W135 D_on_W126V S1"),
                                         ("Epin", vE, inp / "W126V_accepted.blend", FROZEN["W126V_accepted"][1], "AnimCV_EXP W135 E_on_W126V S1")):
        spec = {"out_blend": str(out / f"{br}_S1.blend"), "interior_action_frames": list(INTERIOR), "values_wxyz": vals, "branch": f"W135 {br}",
                "action_name": nm, "expected_blend_sha256": base_sha,
                "note": "W135 scratch: B wrist path + " + ("C (MotionBricks) elbow swivel prior" if br.startswith("D") else "neutral B swivel")}
        dump(out / f"build_{br}_spec.json", spec)
        log["steps"].append([f"build_{br}", blender(STAGE, "build_variant", out / f"build_{br}.log", blend=base, spec=out / f"build_{br}_spec.json",
                                                     report=out / f"build_{br}_report.json")])
        blends[br] = out / f"{br}_S1.blend"
    # reversibility: write A's interior keys back into D
    stA = dA["stored"]
    vA = {ch: {str(f): [stA[ch][str(c)][str(f)] for c in range(4)] for f in range(INTERIOR[0], INTERIOR[1] + 1)} for ch in E.EDIT_CHANNELS}
    spec = {"out_blend": str(out / "Drev_S1.blend"), "interior_action_frames": list(INTERIOR), "values_wxyz": vA, "branch": "W135 Drev",
            "action_name": "AnimCV_EXP W135 D_reverted S1", "expected_blend_sha256": M.sha256_file(blends["D"]), "note": "reversibility control"}
    dump(out / "build_Drev_spec.json", spec)
    log["steps"].append(["build_Drev", blender(STAGE, "build_variant", out / "build_Drev.log", blend=blends["D"], spec=out / "build_Drev_spec.json",
                                                report=out / "build_Drev_report.json")])
    blends["Drev"] = out / "Drev_S1.blend"
    rb = {k: load(pdir / f"readback_{k}.json") for k in "ABC"}
    rb["Pin"] = load(pdir / "readback_Pin.json")
    dumps = {"A": dA, "B": dB, "C": dC}
    for br in ("D", "E", "Dpin", "Epin", "Drev"):
        log["steps"].append([f"readback_{br}", readback(blends[br], out / f"readback_{br}.json", out / f"readback_{br}.log")])
        rb[br] = load(out / f"readback_{br}.json")
    for br in ("D", "E"):
        log["steps"].append([f"dump_{br}", bdump(blends[br], out / f"dump_{br}.json", out / f"dump_{br}.log")])
        dumps[br] = load(out / f"dump_{br}.json")
    # ---- mechanical gates ----
    g = {}
    i0, i1 = INTERIOR
    for br in ("D", "E"):
        r = rb[br]
        pres = M.preservation(r, rb["A"], INTERIOR)
        ck = M.changed_keys(dumps[br]["stored"], dA["stored"])
        census = r["fcurve_census"]
        m2 = (len(r["actions_in_file"]) == 1 and r["action_name"] not in (rb["A"]["action_name"], rb["B"]["action_name"], rb["C"]["action_name"])
              and census["n_fcurves"] == 32 and set(census["interpolation_census_curves"]) == {"CONSTANT"}
              and [census["key_frame_min"], census["key_frame_max"]] == [1.0, 240.0] and r["scene"]["effective_fps"] == 30.0
              and r["scene"]["frame_start"] == 1 and r["scene"]["frame_end"] == 240)
        m3 = (set(c[0] for c in ck) <= set(E.EDIT_CHANNELS) and all(i0 <= c[2] <= i1 for c in ck)
              and not pres["nonarm_changed"] and not pres["outside_changed"] and not pres["sub_outside_changed"])
        fX, fB = _frames(r), _frames(rb["B"])
        werr = {f: float(np.linalg.norm(np.subtract(fX[f]["world"]["hand_l"]["head"], fB[f]["world"]["hand_l"]["head"]))) for f in range(i0, i1 + 1)}

        def wstep(fr_, f0, f1):
            return float(np.linalg.norm(np.subtract(fr_[f1]["world"]["hand_l"]["head"], fr_[f0]["world"]["hand_l"]["head"])))
        ent = abs(wstep(fX, i0 - 1, i0) - wstep(fB, i0 - 1, i0))
        ext = abs(wstep(fX, i1, i1 + 1) - wstep(fB, i1, i1 + 1))
        m4 = max(werr.values()) <= 1e-3 and ent <= 1e-3 and ext <= 1e-3
        L1 = [float(np.linalg.norm(np.subtract(fX[f]["world"]["lowerarm_l"]["head"], fX[f]["world"]["upperarm_l"]["head"]))) for f in range(i0 - 1, i1 + 2)]
        L2 = [float(np.linalg.norm(np.subtract(fX[f]["world"]["hand_l"]["head"], fX[f]["world"]["lowerarm_l"]["head"]))) for f in range(i0 - 1, i1 + 2)]
        low_same = all(dumps[br]["stored"][E.LOWER][str(c)][str(f)] == dB["stored"][E.LOWER][str(c)][str(f)] for c in range(4) for f in range(i0, i1 + 1))
        em = E.elbow_metrics(dumps[br])
        m5 = (np.ptp(L1) <= 1e-4 and np.ptp(L2) <= 1e-4 and low_same and not em["near_straight_frames"])
        g[br] = {"M2": bool(m2), "M3": bool(m3), "M4": bool(m4), "M5": bool(m5),
                 "changed_key_channels": sorted({c[0] for c in ck}), "changed_key_frames": [min((c[2] for c in ck), default=None), max((c[2] for c in ck), default=None)],
                 "n_changed_keys": len(ck), "nonarm_changed": pres["nonarm_changed"][:5], "outside_changed": pres["outside_changed"][:5],
                 "sub_outside_changed": pres["sub_outside_changed"][:5], "wrist_err_vs_B_max_m": max(werr.values()),
                 "entry_step_diff_vs_B_m": ent, "exit_step_diff_vs_B_m": ext, "upper_len_range_m": float(np.ptp(L1)),
                 "lower_len_range_m": float(np.ptp(L2)), "lowerarm_key_equals_B": bool(low_same), "near_straight": em["near_straight_frames"],
                 "blend_sha256": r["blend_sha256"], "action_name": r["action_name"], "n_subframes": len(r["subframes"])}
    # M6 pin composition
    pin_src = load(FROZEN["W126V_proposal"][0])
    tgt = np.asarray(pin_src["constraint"]["target"])
    pin_act = list(E.PIN_ACTION)
    fp = _frames(rb["Pin"])
    m6 = {}
    for br in ("Dpin", "Epin"):
        fc = _frames(rb[br])
        rp = {f: float(np.linalg.norm(np.asarray(fp[f]["world"]["hand_l"]["head"]) - tgt)) for f in pin_act}
        rc = {f: float(np.linalg.norm(np.asarray(fc[f]["world"]["hand_l"]["head"]) - tgt)) for f in pin_act}
        cp = M.preservation(rb[br], rb["Pin"], INTERIOR)
        m6[br] = {"residual_W126V_m": rp, "residual_composed_m": rc, "pin_frames_bit_equal": all(fc[f]["world"] == fp[f]["world"] for f in pin_act),
                  "composed_nonarm_changed": cp["nonarm_changed"][:5], "composed_outside_changed": cp["outside_changed"][:5],
                  "composed_sub_outside_changed": cp["sub_outside_changed"][:5]}
        m6[br]["pass"] = bool(m6[br]["pin_frames_bit_equal"] and all(rc[f] == rp[f] for f in pin_act) and max(rc.values()) <= 1e-6
                              and not cp["nonarm_changed"] and not cp["outside_changed"] and not cp["sub_outside_changed"])
    m6["pass"] = m6["Dpin"]["pass"] and m6["Epin"]["pass"]
    # M7 reversibility
    m7 = all(p["world"] == q["world"] and p["matrix_world"] == q["matrix_world"] for p, q in zip(rb["Drev"]["frames"], rb["A"]["frames"])) \
        and all(p["world"] == q["world"] for p, q in zip(rb["Drev"]["subframes"], rb["A"]["subframes"]))
    # M8 solver neutrality E == B
    eb = 0.0
    for p, q in zip(rb["E"]["frames"], rb["B"]["frames"]):
        for b in q["world"]:
            for k in ("head", "tail"):
                eb = max(eb, float(np.abs(np.subtract(p["world"][b][k], q["world"][b][k])).max()))
    m8 = eb <= 1e-6
    # M9 integrity
    after = frozen_hashes()
    m9 = all(v["ok"] for v in after.values())
    g.update({"M6": m6, "M7": bool(m7), "M8": bool(m8), "M8_max_world_diff_E_vs_B_m": eb, "M9": bool(m9), "M10": bool(ctl["all_fail_closed"]),
              "frozen_after": {k: v["ok"] for k, v in after.items()}})
    # ---- learned effect ----
    fD, fE = _frames(rb["D"]), _frames(rb["E"])
    el = {f: float(np.linalg.norm(np.subtract(fD[f]["world"]["lowerarm_l"]["head"], fE[f]["world"]["lowerarm_l"]["head"]))) for f in range(i0, i1 + 1)}
    emD, emE = E.elbow_metrics(dumps["D"]), E.elbow_metrics(dumps["E"])
    k_in = [k for k, f in enumerate(emD["per_frame"]["frames"]) if i0 <= f <= i1]
    dpsi = [abs(E.wrap(np.radians(emD["per_frame"]["swivel_deg"][k] - emE["per_frame"]["swivel_deg"][k]))) for k in k_in]
    learned = {"elbow_D_vs_E_max_mm": max(el.values()) * 1e3, "elbow_D_vs_E_mean_mm": float(np.mean(list(el.values()))) * 1e3,
               "swivel_D_vs_E_max_deg": float(np.degrees(max(dpsi))), "swivel_D_vs_E_mean_deg": float(np.degrees(np.mean(dpsi))),
               "per_frame_elbow_D_vs_E_mm": {str(f): v * 1e3 for f, v in el.items()}}
    learned["class"] = "LEARNED_EFFECT" if (learned["elbow_D_vs_E_max_mm"] >= 10.0 and learned["swivel_D_vs_E_max_deg"] >= 3.0) else "NO_LEARNED_EFFECT"
    # ---- quality metrics ----
    hinge = M.hinge_reference(rb["A"])
    q = {}
    for br in ("A", "B", "C", "D", "E"):
        qq = M.quality(rb[br], rb["A"], INTERIOR, WINDOW, hinge)
        qq.pop("per_frame", None)
        qq["arm_local"] = M.armlocal_quality(M.armlocal_wrist(dumps[br], *WINDOW), INTERIOR, WINDOW)
        qq["arm_local"].pop("per_frame", None)
        qq["elbow"] = E.elbow_metrics(dumps[br])
        fr_ = _frames(rb[br])
        sh = np.array([fr_[f]["world"]["upperarm_l"]["head"] for f in range(WINDOW[0], WINDOW[1] + 1)])
        qq["world_shoulder_step_max_mm"] = float(np.linalg.norm(np.diff(sh, axis=0), axis=1).max() * 1e3)
        wB = _frames(rb["B"])
        qq["wrist_dev_from_B_max_mm"] = max(float(np.linalg.norm(np.subtract(fr_[f]["world"]["hand_l"]["head"], wB[f]["world"]["hand_l"]["head"]))) for f in range(i0, i1 + 1)) * 1e3
        q[br] = qq
    qa = {"Q1": all(q["D"]["elbow"][k] <= 2 * q["B"]["elbow"][k] + 10.0 for k in ("elbow_entry_step_mm", "elbow_exit_step_mm")),
          "Q2": len(q["D"]["elbow"]["elbow_reversal_flicks"]) == 0,
          "Q3": q["D"]["elbow"]["swivel_step_max_deg"] <= 30.0}
    res.update({"gates": g, "learned": learned, "quality": q, "quality_criteria_metric": qa, "prior_sha256": M.sha256_file(out / "prior_C.json"),
                "blends": {k: str(v) for k, v in blends.items()}, "blend_sha256": {k: M.sha256_file(v) for k, v in blends.items()}})
    dump(out / "run_result.json", res)
    vid = render_and_compose(out, pdir, blends, log)
    res["video"] = vid
    log["end"] = now()
    dump(out / "run_log.json", log)
    dump(out / "run_result.json", res)
    print(json.dumps({"gates": {k: (v if isinstance(v, bool) else (v.get("pass") if isinstance(v, dict) and "pass" in v else
                                                                    {kk: vv for kk, vv in v.items() if kk.startswith("M")} if isinstance(v, dict) else v))
                                for k, v in g.items() if k != "frozen_after"},
                      "learned": {k: v for k, v in learned.items() if not k.startswith("per_frame")}, "qa": qa,
                      "video": {k: vid[k] for k in ("frames_written", "frames_decoded", "fps", "native_rate")}}, indent=1, default=str))
    return 0


def render_and_compose(out, pdir, blends, log):
    import cv2
    s = S1
    n = s["n_action"]
    i0, i1 = INTERIOR
    rbA = load(pdir / "readback_A.json")
    fa = _frames(rbA)
    allp = np.array([fa[f]["world"][b][p] for f in fa for b in fa[f]["world"] for p in ("head", "tail")])
    centre = ((allp.min(0) + allp.max(0)) / 2).tolist()
    ext = float(np.max(allp.max(0) - allp.min(0)))
    el = np.array([fa[f]["world"]["lowerarm_l"]["head"] for f in range(WINDOW[0], WINDOW[1] + 1)]).mean(0).tolist()
    views = [{"name": "body", "centre": centre, "offset": [0.0, -6.0, 0.0], "ortho_scale": ext * 1.15},
             {"name": "arm", "centre": el, "offset": [4.5, -4.5, 1.5], "ortho_scale": 1.3}]
    size = 320
    jobs = {"size": size, "views": views, "frames": [{"blender_frame": f, "markers": []} for f in range(1, n + 1)]}
    dump(out / "render_jobs.json", jobs)
    order = ["A", "B", "C", "D", "E"]
    for br in order:
        log["steps"].append([f"render_{br}", blender(ACTION_STAGE, "render", out / f"render_{br}.log", blend=blends[br], jobs=out / "render_jobs.json",
                                                      out_dir=out / f"frames_{br}")])
    man = load(s["run"] / "manifest" / "input_manifest.json")
    clip = man["intake"]["clip_path"]
    seg0 = s["segment"][0]
    want = set(range(seg0, seg0 + n))
    cap = cv2.VideoCapture(clip)
    src, k = {}, 0
    while True:
        ok, im = cap.read()
        if not ok:
            break
        if k in want:
            src[k] = im
        k += 1
    cap.release()
    z = np.load(s["run"] / "perception" / "m1s0" / "observations_predictions.npz")
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

    def tile(br, f, arm_only=False):
        arm = cv2.imread(str(out / f"frames_{br}" / f"arm_f{f:04d}.png"))
        lab(arm, NAMES[br] if arm_only else "left-arm close-up", 16)
        inside = i0 <= f <= i1
        lab(arm, f"Action {f} / SOURCE {f + seg0 - 1}" if arm_only else ("CHANGED interior" if inside and br != "A" else ("interior" if inside else "unchanged")),
            34, (80, 80, 255) if inside else (200, 200, 200))
        if arm_only:
            return arm
        body = cv2.imread(str(out / f"frames_{br}" / f"body_f{f:04d}.png"))
        lab(body, NAMES[br], 16)
        lab(body, f"Action {f}", 34)
        return np.vstack([body, arm])

    def source_tile(f, h=2 * size):
        kk = f + seg0 - 1
        x0, y0, x1, y1 = crop
        im = src[kk][y0:y1, x0:x1].copy()
        im = cv2.resize(im, (int(im.shape[1] * h / im.shape[0]), h))
        lab(im, "SOURCE (fixed crop)", 16)
        lab(im, f"SOURCE {kk}  t={kk / 30.0:.3f}s", 34)
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
        cv2.putText(b, f"edited interior Action {i0}-{i1} = SOURCE {i0 + seg0 - 1}-{i1 + seg0 - 1} (red); 30 fps native; Action = SOURCE - 78",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 0, 0), 1)
        return b
    path = out / "abcde_S1_30fps.mp4"
    wr, written = None, 0
    for f in range(1, n + 1):
        top = np.hstack([source_tile(f)] + [tile(br, f) for br in order])
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
    vid = {"video": str(path), "sha256": M.sha256_file(path), "frames_written": written, "frames_decoded": nd, "fps": fps,
           "native_rate": nd == n and abs(fps - 30.0) < 1e-6}
    sheets = {"entry": list(range(170, 176)), "middle": list(range(183, 189)), "exit": list(range(196, 202)),
              "overview": list(range(171, 201, 3))}
    sh = {}
    for name, frs in sheets.items():
        rows = [np.hstack([cv2.resize(source_tile(f, size), (size, size)) for f in frs])]
        rows += [np.hstack([tile(br, f, arm_only=True) for f in frs]) for br in order]
        p = out / f"sheet_arm_{name}_S1.png"
        cv2.imwrite(str(p), np.vstack(rows))
        sh[f"arm_{name}"] = {"path": str(p), "sha256": M.sha256_file(p)}
        rows = [np.hstack([cv2.resize(source_tile(f, size), (size, size)) for f in frs])]
        rows += [np.hstack([cv2.imread(str(out / f"frames_{br}" / f"body_f{f:04d}.png")) for f in frs]) for br in order]
        p = out / f"sheet_body_{name}_S1.png"
        cv2.imwrite(str(p), np.vstack(rows))
        sh[f"body_{name}"] = {"path": str(p), "sha256": M.sha256_file(p)}
    vid["sheets"] = sh
    return vid


# ------------------------------------------------------------------------------------------------ assess ---
def stage_assess(a):
    check_declaration(a)
    out = new_dir(a.out_dir)
    pf, ctl = load(a.preflight), load(a.controls)
    rr = load(Path(a.run_dir) / "run_result.json")
    vr = load(a.visual_review)
    if "STOP" in rr:
        stop = rr["STOP"]["code"]
        prov = stop.startswith("PRIOR_") or stop == "SOURCE_IDENTITY"
        case = "D" if prov else "C"
        rep = {"stage": "assess", "case": case, "stop": rr["STOP"]}
    else:
        g = rr["gates"]
        mech = {"M0": pf["status"] == "OK", "M1": True, "M2": g["D"]["M2"] and g["E"]["M2"], "M3": g["D"]["M3"] and g["E"]["M3"],
                "M4": g["D"]["M4"] and g["E"]["M4"], "M5": g["D"]["M5"] and g["E"]["M5"], "M6": g["M6"]["pass"], "M7": g["M7"],
                "M8": g["M8"], "M9": g["M9"], "M10": g["M10"] and ctl["fk_pass"], "M11": rr["video"]["native_rate"]}
        qa = dict(rr["quality_criteria_metric"])
        qa["Q4"] = bool(vr["answers"]["D_more_natural_than_B"] and vr["answers"]["D_more_natural_than_E"]
                        and vr["answers"]["D_more_press_faithful_than_B"] and vr["answers"]["D_more_press_faithful_than_E"]
                        and not vr["answers"]["D_flick_wrong_elbow_or_detour"])
        qa["Q5"] = bool(vr.get("native_rate_reviewed", False))
        if not mech["M0"]:
            case = "DATA_BLOCKED"
        elif not (mech["M9"] and mech["M10"]):
            case = "D"
        elif not all(mech[k] for k in ("M2", "M3", "M4", "M5", "M6", "M7", "M8", "M11")):
            case = "C"
        elif rr["learned"]["class"] == "LEARNED_EFFECT" and all(qa.values()):
            case = "A"
        else:
            case = "B"
        rep = {"stage": "assess", "case": case, "mechanical": mech, "learned": rr["learned"]["class"], "quality_criteria": qa,
               "visual_review": vr["answers"], "native_rate_reviewed": vr.get("native_rate_reviewed", False)}
    rep.update({"time": now(), "rule": CASE_RULE, "visual_review_sha256": M.sha256_file(a.visual_review),
                "run_result_sha256": M.sha256_file(Path(a.run_dir) / "run_result.json"),
                "note": "visual review is the agent's own unblinded still-frame review, not human/animator validation"})
    sha = dump(out / "assessment.json", rep)
    print(json.dumps({**{k: v for k, v in rep.items() if k not in ("rule",)}, "sha": sha}, indent=1, default=str))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=("preflight", "declare", "controls", "run", "assess"))
    ap.add_argument("--out-dir", required=True)
    for k in ("preflight", "declaration", "declaration-sha", "controls", "run-dir", "visual-review"):
        ap.add_argument("--" + k)
    a = ap.parse_args()
    return {"preflight": stage_preflight, "declare": stage_declare, "controls": stage_controls, "run": stage_run,
            "assess": stage_assess}[a.stage](a)


if __name__ == "__main__":
    sys.exit(main())
