#!/usr/bin/env python3
"""W129 (EXPERIMENTAL, opt-in) harness: pretrained MotionBricks left-arm inbetweening on a frozen real-video BaseRig Action.

Stages (each refuses an existing output directory; nothing in an AnimCV run directory or lineage is ever written):

  preflight  authenticate repo state, frozen S1/S3 baselines + W126 lineage heads + pins, source clips, the pinned
             MotionBricks checkout / ONNX bytes / manifest / licences, the separate runtime env, Blender; dump the SOMA
             skeleton; run the bundled independent smoke example and the runtime negative controls.
  declare    write the prospective declaration (episode/time rule, mapping, config, gates, CASE rule, instrument hashes)
             BEFORE any primary model output exists.  ``outcomes_inspected: false``.
  controls   mapping controls on the real BaseRig in Blender (static / rotated / left-right / parent-child / no-mirror) and
             adapter negative controls (mirrored axis, missing joint, protected overlap, bad timing, out-of-domain build).
  run        per source: dump -> SOMA conditioning -> ONE real TensorRT inference (+ labelled exact rerun, ORT parity) ->
             left-arm projection -> B (slerp) and C (model) candidate .blend files -> fresh-process readbacks ->
             preservation / pin / quality diagnostics -> renders -> native-rate A/B/C video, contact sheets, blinded sheets.
  assess     apply the frozen CASE rule to the run evidence + the (blinded) agent visual review file.

    PY=C:/Projects/AnimCV/.venv_build_full/Scripts/python.exe; export PYTHONPATH=src
    $PY -X utf8 scripts/w129_motionbricks_harness.py <stage> --out-dir <dir> [...]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]
import numpy as np                                                    # noqa: E402

import app.experimental_motionbricks as M                             # noqa: E402

W = Path("C:/Users/dna10/animcv-w129")
W126 = Path("C:/Users/dna10/animcv-w126/runs")
BLENDER = "C:/Program Files/Blender Foundation/Blender 5.1/blender.exe"
MB_ROOT = W / "ext" / "motionbricks"
MB_PY = W / "env" / "mb312" / "Scripts" / "python.exe"
MB_COMMIT = "54fc563aeaaf819a753d72cc9fdc6713989cdd43"
ONNX = MB_ROOT / "motionbricks" / "assets" / "onnx" / "soma" / "mmm_v6_soma_fp32.onnx"
ONNX_FP16 = MB_ROOT / "motionbricks" / "assets" / "onnx" / "soma" / "mmm_v6_soma_fp16.onnx"
ONNX_SHA = "856419dc72aeb14e591eebf4dd0ea05734a2506ddb00bcd39dd6dac2ad60f4fb"          # = Git LFS OID at MB_COMMIT
MANIFEST = MB_ROOT / "motionbricks" / "assets" / "onnx" / "soma" / "soma_n1.json"
TRT_CACHE = W / "trt-cache"
STAGE = ROOT / "src" / "blender" / "experimental_motionbricks_stage.py"
ACTION_STAGE = ROOT / "src" / "blender" / "experimental_action_stage.py"
INFER = ROOT / "scripts" / "w129_motionbricks_infer.py"
INSTRUMENTS = ["src/app/experimental_motionbricks.py", "src/blender/experimental_motionbricks_stage.py",
               "scripts/w129_motionbricks_infer.py", "scripts/w129_motionbricks_harness.py",
               "tests/test_experimental_motionbricks.py", "src/blender/experimental_action_stage.py"]
OBJECT_REST_WXYZ = [1.0, -1.410649872468639e-07, 0.0, 0.0]          # W124/W126 sidecar rig_info.rest_rotation_wxyz
OBJECT_SCALE = 0.009999999776482582

SOURCES = {
    "S1": {"role": "PRIMARY", "run": W126 / "S1_replay", "segment": [79, 318], "n_action": 240,
           "baseline_sha": "0803ab2bf31210fbb729bc81b76931729fdc3e12cce27b9507a5b8caf1d12402",
           "value_digest": "25a09802ffa210df56f0e930295468ef4cdb675e2945cb95491c080b5101c8a9",
           "head_state": "73dcd3dde3a65ffbd23f36361173ac7a92ee7c4db5e8436e4e85947649a524ab",
           "pin_accepted": "accepted/accepted_03_674ba278e9a0.blend",
           "pin_accepted_sha": "7e2ab16511f4cbe111171df9af0bc2e1f7d8c09a57bd5a4feff284d34e908f40",
           "pin_request": "edits/W126V", "context_last": 249, "target_first": 278,
           "clip": "C:/Users/dna10/animcv-w121/data/originals/64498918_Double_Kettlebell_Strict_Press_side_view_.webm",
           "clip_sha": "de8feb9d515daa2bfaeed7167fdcc62028dd3f6450872c4fd81f8fb03dd6359c"},
    "S3": {"role": "SECONDARY", "run": W126 / "S3_replay", "segment": [0, 153], "n_action": 154,
           "baseline_sha": "e1cfbae49e3af0e7a29758d93e5e3aec9781c3a16d4d5751d8406a1d3bba34e1",
           "value_digest": "b90587455a6aa2e831927b229e80462aee8634cd884bb16cc301289573b72937",
           "head_state": "be9a1f09107c2237b37db0d21fe73b950958f747f1bbd65054b2b10cf8af9ae6",
           "pin_accepted": "accepted/accepted_03_fd249fbb203f.blend",
           "pin_accepted_sha": "702ead2646319f657a515bb19d0fe288184ae0695f149e97f5e22b8016a34810",
           "pin_request": "edits/W126V", "context_last": 109, "target_first": 138,
           "clip": None, "clip_sha": "f6b90f2f72e9020d"},
}


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
    data = (ROOT / rel).read_bytes().replace(b"\r\n", b"\n")
    import hashlib
    return hashlib.sha256(data).hexdigest()


def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def blender(mode_script, mode, log, **kw):
    args = [BLENDER, "-b", "-t", "2", "--python-exit-code", "1", "-P", str(mode_script), "--", mode]
    for k, v in kw.items():
        args += ["--" + k.replace("_", "-"), str(v)]
    t = time.time()
    r = subprocess.run(args, capture_output=True, text=True)
    Path(log).write_text(r.stdout + "\n--- stderr ---\n" + r.stderr)
    if r.returncode != 0:
        raise RuntimeError(f"Blender {mode} failed ({r.returncode}); see {log}")
    return time.time() - t


def mb(req, out_json):
    dump(out_json, req)
    t = time.time()
    r = subprocess.run([str(MB_PY), "-I", str(INFER), str(out_json)], capture_output=True, text=True, cwd=str(W))
    Path(str(out_json) + ".log").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr)
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


# ------------------------------------------------------------------------------------------------ preflight ---
def stage_preflight(a):
    out = new_dir(a.out_dir)
    rep = {"stage": "preflight", "time": now(), "head": git("rev-parse", "HEAD"),
           "instruction_blob": git("rev-parse", "HEAD:docs/next_batch.md"),
           "w128_complete": "AUTOMATION_BATCH_STATUS: COMPLETE" in (ROOT / "docs/128_WORKLOG_OPT_IN_UNITY_GENERIC_FBX_PLAYBACK.md").read_text(encoding="utf-8")}
    ok = True
    srcs = {}
    for sid, s in SOURCES.items():
        run = s["run"]
        refs = load(run / "lineage" / "refs.json")
        r = {"baseline_sha_ok": M.sha256_file(run / "baseline" / "baseline.blend") == s["baseline_sha"],
             "refs_head": refs["head"], "refs_head_ok": refs["head"] == s["head_state"],
             "refs_head_value_digest_ok": refs["head_blend"]["value_digest"] == s["value_digest"],
             "refs_op": refs["op"], "refs_sha256": M.sha256_file(run / "lineage" / "refs.json"),
             "pin_accepted_sha_ok": M.sha256_file(run / s["pin_accepted"]) == s["pin_accepted_sha"]}
        prop = load(run / s["pin_request"] / "proposal.json")
        r["pin"] = {"request": s["pin_request"], "pin_source_frames": prop["pin_frames"],
                    "pin_action_frames": [f - s["segment"][0] + 1 for f in prop["pin_frames"]],
                    "target_m": prop["constraint"]["target"], "tolerance_m": prop["constraint"]["tolerance_m"],
                    "status": "ACCEPTED in W126 then REVERTED: the authoritative head has NO active pin; the historical pin domain is protected anyway"}
        man = load(run / "manifest" / "input_manifest.json")
        clip = man["intake"]["clip_path"]
        r["clip"] = clip
        r["clip_sha_ok"] = Path(clip).is_file() and M.sha256_file(clip).startswith(s["clip_sha"])
        side = load(run / "baseline" / "action_sidecar.json")
        r["sidecar_value_digest_ok"] = side["action_value_digest"] == s["value_digest"]
        r["fps"] = side["scene"]["effective_fps"]
        ok &= all(v for k, v in r.items() if k.endswith("_ok"))
        srcs[sid] = r
    rep["sources"] = srcs
    # MotionBricks checkout / model
    mbrep = {"commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=MB_ROOT, capture_output=True, text=True).stdout.strip(),
             "tag_v0_2_0": subprocess.run(["git", "rev-parse", "v0.2.0^{commit}"], cwd=MB_ROOT, capture_output=True, text=True).stdout.strip(),
             "worktree_clean": subprocess.run(["git", "status", "--porcelain"], cwd=MB_ROOT, capture_output=True, text=True).stdout.strip() == "",
             "lfs_oid_fp32": [l for l in subprocess.run(["git", "lfs", "ls-files", "-l"], cwd=MB_ROOT, capture_output=True, text=True).stdout.splitlines()
                              if l.endswith("mmm_v6_soma_fp32.onnx")],
             "onnx_bytes": ONNX.stat().st_size, "onnx_sha256": M.sha256_file(ONNX),
             "manifest_sha256": M.sha256_file(MANIFEST), "manifest": load(MANIFEST),
             "license_code_sha256": M.sha256_file(MB_ROOT / "LICENSE"),
             "license_code_first_line": (MB_ROOT / "LICENSE").read_text(encoding="utf-8").splitlines()[0:3],
             "license_model": (MB_ROOT / "motionbricks/assets/onnx/LICENSE.md").read_text(encoding="utf-8"),
             "sample_data_license_head": (MB_ROOT / "assets/motions/LICENSE.txt").read_text(encoding="utf-8", errors="replace")[:600]}
    mbrep["commit_ok"] = mbrep["commit"] == MB_COMMIT == mbrep["tag_v0_2_0"]
    mbrep["onnx_ok"] = mbrep["onnx_sha256"] == ONNX_SHA and bool(mbrep["lfs_oid_fp32"]) and mbrep["lfs_oid_fp32"][0].startswith(ONNX_SHA)
    ok &= mbrep["commit_ok"] and mbrep["onnx_ok"]
    rep["motionbricks"] = mbrep
    rep["blender"] = subprocess.run([BLENDER, "-b", "--version"], capture_output=True, text=True).stdout.splitlines()[:1]
    rep["runtime_env"] = subprocess.run([str(MB_PY), "-I", "-c", "import sys,tensorrt,warp,numpy,onnxruntime;print(sys.version.split()[0],tensorrt.__version__,warp.__version__,numpy.__version__,onnxruntime.__version__)"],
                                        capture_output=True, text=True).stdout.strip()
    base = {"motionbricks_root": str(MB_ROOT), "onnx": str(ONNX), "onnx_sha256": ONNX_SHA, "manifest": str(MANIFEST),
            "manifest_sha256": mbrep["manifest_sha256"], "cache_dir": str(TRT_CACHE)}
    runs = {}
    rc, t = mb({**base, "mode": "skeleton", "out_dir": str(out / "skeleton")}, out / "req_skeleton.json")
    runs["skeleton"] = {"rc": rc, "s": t}
    if rc == 0:
        shutil.copy(out / "skeleton" / "soma_skeleton.json", out / "soma_skeleton.json")
        M.SomaSkeleton.load(out / "soma_skeleton.json")
    rc, t = mb({**base, "mode": "smoke", "out_dir": str(out / "smoke")}, out / "req_smoke.json")
    runs["smoke"] = {"rc": rc, "s": t, "result": load(out / "smoke" / "result.json") if rc == 0 else None}
    ok &= rc == 0 and runs["smoke"]["result"]["valid_frames"] > 0
    # runtime negative controls
    neg = {}
    rc, _ = mb({**base, "mode": "infer", "onnx": str(W / "absent" / "mmm_v6_soma_fp32.onnx"), "out_dir": str(out / "neg_missing")}, out / "req_neg_missing.json")
    neg["missing_onnx"] = {"rc": rc, "stop": load(out / "neg_missing" / "stop.json") if rc == 3 else None, "pass": rc == 3}
    rc, _ = mb({**base, "mode": "infer", "onnx": str(ONNX_FP16), "out_dir": str(out / "neg_wrongsha")}, out / "req_neg_wrongsha.json")
    neg["wrong_model_bytes"] = {"rc": rc, "stop": load(out / "neg_wrongsha" / "stop.json") if rc == 3 else None, "pass": rc == 3}
    ptr = new_dir(out / "lfs_pointer_fixture") / "mmm_v6_soma_fp32.onnx"
    ptr.write_text(f"version https://git-lfs.github.com/spec/v1\noid sha256:{ONNX_SHA}\nsize 757864273\n")
    rc, _ = mb({**base, "mode": "infer", "onnx": str(ptr), "out_dir": str(out / "neg_pointer")}, out / "req_neg_pointer.json")
    neg["lfs_pointer"] = {"rc": rc, "stop": load(out / "neg_pointer" / "stop.json") if rc == 3 else None, "pass": rc == 3}
    rc, _ = mb({**base, "mode": "skeleton_missing_joint", "out_dir": str(out / "neg_skeleton")}, out / "req_neg_skeleton.json")
    neg["skeleton_missing_joint"] = {"rc": rc, "result": load(out / "neg_skeleton" / "result.json") if rc in (0, 4) else None, "pass": rc == 0}
    rep["runtime_negative_controls"] = neg
    ok &= all(v["pass"] for v in neg.values())
    rep["runs"] = runs
    rep["status"] = "OK" if ok else "DATA_BLOCKED"
    dump(out / "preflight.json", rep)
    print(json.dumps({"status": rep["status"], "smoke": runs["smoke"].get("result", {}) and runs["smoke"]["result"].get("valid_frames"),
                      "neg": {k: v["pass"] for k, v in neg.items()}}, indent=1))
    return 0 if ok else 3


# ------------------------------------------------------------------------------------------------ declaration ---
CASE_RULE = {
    "DATA_BLOCKED": "a prerequisite (model bytes/licence/runtime, authentic S1 baseline/lineage, Blender, native-time alignment) is missing before any primary evidence",
    "D": "invalid evidence or predeclared control-integrity failure: a mapping/runtime/adapter negative control does not fail closed, the primary sample is not the real pinned model (mock/substitution/provenance mismatch), or an instrument changed after this declaration",
    "C": "the real model ran but a timing (G2), mapping (G1), Action (G4) or hard-constraint (G5) gate fails, or the representation cannot be projected safely",
    "B": "authentic model and a safe separate Blender candidate are demonstrated (G0-G7 pass) but any A quality criterion (QA1-QA5) fails",
    "A": "G0-G7 pass AND QA1-QA5 pass (visible quality improvement vs BOTH the no-op A and the slerp B without materially worse source fidelity)",
    "precedence": "DATA_BLOCKED > D > C > B > A; S3 (secondary) never changes the S1 CASE",
}
GATES = {
    "G0": "prerequisites authenticated (preflight status OK)",
    "G1": "mapping controls pass: real-frame left-arm round trip <= 1e-5 (quaternion component), posed-control primary direction residual <= 0.05 deg, no-mirroring ok, left-right and parent-child behave as declared, mirrored axis refused",
    "G2": "timing: allowed token mask exactly {9}; pred_num_tokens == 9; valid samples == 30; endpoint-corrected LeftArm/LeftForeArm locals at sample 0 / 29 equal the last context / first target pose within 2e-3 rad",
    "G3": "real inference provenance: ONNX bytes = LFS OID 856419dc..., official MotionBricksSOMAEngine TensorRT plan, raw tensors saved",
    "G4": "candidate is a separate .blend + separately named Action; fresh-process readback: 1 Action, 32 CONSTANT F-curves keyed 1..N, 30 fps, scene 1..N; changed keys only upperarm_l/lowerarm_l on interior frames; every non-left-arm bone world point and the object matrix bit-equal on every integer frame and subframe; left arm bit-equal outside the interior; baseline/lineage files unchanged",
    "G5": "hard pin: the W126V pin domain (Action frames of source 210-218 for S1) is bit-equal; C composed onto the W126V-accepted .blend keeps every pin-frame residual equal to the W126V residual and <= 1e-6 m",
    "G6": "negative controls fail closed (runtime: missing ONNX, wrong bytes, LFS pointer, skeleton missing joint; adapter: mirrored axis, missing BaseRig bone, protected overlap, non-30-sample window, out-of-domain channel/frame build)",
    "G7": "native-rate synchronized A/B/C video: N_action frames written and decoded at 30.000 fps",
}
QA = {
    "QA1": "C has 0 ARM-LOCAL wrist reversal flicks (wrist in the clavicle_l frame, i.e. attributable to the edited channels) in steps context_last->+1 .. target_first-1->target_first (consecutive steps both > 20 mm with negative dot)",
    "QA2": "C ARM-LOCAL boundary C0 entry/exit wrist steps each <= 2 x B's corresponding step + 10 mm",
    "QA3": "C introduces no near-straight (> 175 deg) interior frame, no bend-direction flip and no frame bending against the A hinge reference",
    "QA4": "source fidelity: C mean interior wrist deviation from A <= 1.25 x B's + 25 mm",
    "QA5": "blinded agent visual review: C judged most natural (V2), C not named for a visible flick (V1) or wrong elbow (V3), and C not ranked least source-faithful (V4)",
}
VISUAL_QUESTIONS = {
    "V1": "Which panel(s) show a visible unnatural one-frame flick/teleport/reversal of the left arm (around Action 171-175)?",
    "V2": "Which panel's left-arm motion through the interior looks most natural and temporally coherent at native rate?",
    "V3": "Which panel(s) show an anatomically wrong elbow (hyperextension, flip) or limb-length artefact?",
    "V4": "Rank the panels by agreement with the source video's visible left-arm pose and timing (rack -> press); which is least faithful?",
    "V5": "Is any boundary discontinuity visible at 171/172 or 199/200 in any panel?",
}


def stage_declare(a):
    out = new_dir(a.out_dir)
    pf = load(a.preflight)
    if pf["status"] != "OK":
        raise SystemExit("preflight not OK: DATA_BLOCKED")
    eps = {}
    for sid, s in SOURCES.items():
        pin = pf["sources"][sid]["pin"]["pin_source_frames"]
        eps[sid] = M.episode(s["segment"][0], s["context_last"], s["target_first"], 9, protected_source=pin)
        eps[sid]["protected_source_frames"] = pin
        eps[sid]["protected_action_frames"] = pf["sources"][sid]["pin"]["pin_action_frames"]
    soma = M.SomaSkeleton.load(Path(a.preflight).parent / "soma_skeleton.json")
    d = {"schema": "animcv_exp_w129_declaration_v1", "time": now(), "outcomes_inspected": False,
         "primary_model_output_exists": (W / "results" / "run").exists(),
         "batch": 129, "starting_commit": "cd6db54fee0a354875c4e30df6eabd958be1ae1a",
         "instruction_blob": "390668973f13c2e952fe9c8db8a686609662ff07", "repo_head_at_declaration": git("rev-parse", "HEAD"),
         "preflight_sha256": M.sha256_file(a.preflight),
         "sources": {sid: {k: str(v) for k, v in s.items()} for sid, s in SOURCES.items()},
         "episodes": eps,
         "time_rule": "9 allowed tokens -> 36 raw frames; the official engine drops 3 boundary frames per end and endpoint-corrects, so valid sample k = SOURCE context_last + k (k = 0..29); sample 0 = last context pose, sample 29 = first target pose; only the 28 interior samples are written; no retiming, no resampling",
         "model": {"repo": "https://github.com/NVlabs/motionbricks", "commit": MB_COMMIT, "tag": "v0.2.0", "model_version": "MotionBricks-SOMA-RP-v1",
                   "onnx": "motionbricks/assets/onnx/soma/mmm_v6_soma_fp32.onnx", "onnx_sha256": ONNX_SHA, "precision": "FP32 (TF32 permitted, model default)",
                   "manifest_sha256": pf["motionbricks"]["manifest_sha256"], "licence_model": "NVIDIA Open Model Agreement",
                   "licence_code": "Apache-2.0", "runtime": "TensorRT 10.8.0.43 via the official MotionBricksSOMAEngine (CUDA graph)",
                   "config": {"allowed_token_counts": [9], "correct_generated_diff": True, "heading_alignment": [1.0, 0.0],
                              "target_num_poses": 4, "context_num_poses": 4, "model_to_runtime": "identity (poses supplied in SOMA model space)"},
                   "seed": "NOT APPLICABLE: the ONNX graph has no random operators (ArgMax/TopK decoding); one primary sample",
                   "rerun": "exactly one exact rerun R2 in a new process, labelled determinism only",
                   "ort_parity": "ONNX Runtime CPU on the identical encoded inputs, diagnostic only"},
         "mapping": {"A_blender_to_soma": M.A_BLENDER_TO_SOMA.tolist(), "joint_map": M.JOINT_MAP, "align": {k: str(v) for k, v in M.ALIGN.items()},
                     "synthetic_model_joints": list(M.SYNTHETIC_MODEL_JOINTS), "coverage": M.coverage(),
                     "root": "Hips position = A x (thigh-head midpoint) x (SOMA rest hip height / BaseRig rest hip height); no root travel exists in the Action (SYNTHETIC zero travel)",
                     "inverse": "upperarm_l / lowerarm_l pose-bone basis from the SOMA parent-relative rotation of LeftArm (parent LeftShoulder) / LeftForeArm (parent LeftArm) via L_base = D_p L_soma D_j^T; nothing else is read back from the model",
                     "framepose_limits": "17-joint FramePose / W105 partial BaseRig cannot determine: right arm (W105 does not drive it), clavicles, spine above spine_01, neck/head orientation, hands/fingers, feet/toes orientation, root translation / global travel, ground contact; these are rig REST or SYNTHETIC in the conditioning",
                     "soma_skeleton_joints": len(soma.names)},
         "branches": {"A": "untouched frozen baseline .blend (byte copy, no-op)",
                      "B": "per-channel endpoint-matched slerp of the upperarm_l / lowerarm_l basis between Action context_last and target_first, t = (f - f0) / 29, interior only",
                      "C": "MotionBricks primary sample, left-arm channels projected on the interior only"},
         "gates": GATES, "quality_criteria": QA, "visual_questions": VISUAL_QUESTIONS, "case_rule": CASE_RULE,
         "visual_protocol": "panels X/Y/Z = random permutation of A/B/C drawn at run time and sealed in blind_key.json; the agent writes its answers to the blinded sheets/video BEFORE reading the key; the review is AGENT-ONLY (no human/animator)",
         "stop_conditions": ["any G0 prerequisite missing -> DATA_BLOCKED", "G1 mapping control failure -> STOP before S1 inference (representation incompatible)",
                             "token count != 9 or valid != 30 -> no retiming, report C", "projection would touch any non-left-arm channel or frame outside the interior -> refuse (C)",
                             "no tuning: no other window, seed, token count, precision, model or post-processing after seeing output"],
         "secondary": "S3 runs with the identical frozen configuration only after the S1 primary completes; transfer check only",
         "pre_declaration_observations": ("A/B-only plumbing dry run on a scratch copy (no model output observed): on S1 the world-space wrist "
                                          "steps around Action 171-177 (300-400 mm) are partly carried by the SHOULDER itself (154-244 mm/frame "
                                          "from the source-derived object/spine_01 tracks, out of the edit domain); in the clavicle_l frame the "
                                          "largest A arm-channel jump is 170->171 (220 mm, inside the frozen context, not editable) followed by "
                                          "93-203 mm steps at 171-178. A world-space flick criterion would therefore not isolate the edit; QA1/QA2 "
                                          "use arm-local wrist kinematics. The frozen window is NOT changed."),
         "instrument_sha256_lf": {f: lf_sha(f) for f in INSTRUMENTS}}
    sha = dump(out / "declaration.json", d)
    print(json.dumps({"declaration_sha256": sha, "episodes": {k: {kk: v[kk] for kk in ("context_action", "target_action", "interior_action", "valid_source")} for k, v in eps.items()}}, indent=1))
    return 0


# ------------------------------------------------------------------------------------------------ controls ---
def quat_axis_angle(axis, deg):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    h = np.radians(deg) / 2
    return [float(np.cos(h)), *(np.sin(h) * axis).tolist()]


def stage_controls(a):
    d = check_declaration(a)
    out = new_dir(a.out_dir)
    soma = M.SomaSkeleton.load(Path(a.preflight).parent / "soma_skeleton.json")
    ow = OBJECT_REST_WXYZ
    yaw30 = quat_axis_angle([0, 0, 1], 30)
    poses = {
        "1": {"object": ow},                                                                    # static rest
        "2": {"object": yaw30, "bones": {"upperarm_l": quat_axis_angle([1, 0.2, 0], 40), "lowerarm_l": quat_axis_angle([0, 0, 1], 60)}},
        "3": {"object": ow, "bones": {"upperarm_l": quat_axis_angle([0, 0, 1], -70)}},       # left-right: left arm only
        "4": {"object": ow, "bones": {"spine_01": quat_axis_angle([1, 0, 0], 30)}},           # parent-child: spine only
        "5": {"object": ow, "bones": {"spine_01": quat_axis_angle([1, 0, 0], 30), "upperarm_l": quat_axis_angle([0, 0, 1], -70)}},
    }
    dump(out / "overrides.json", poses)
    blender(STAGE, "dump", out / "dump.log", blend=W / "inputs" / "S1_baseline.blend", report=out / "control_dump.json", overrides=out / "overrides.json")
    cd = load(out / "control_dump.json")
    fr = {f["blender_frame"]: M.BaseFrame(cd["rest"], f) for f in cd["frames"]}
    mp = M.Mapping(soma, M.rest_frame(cd["rest"], OBJECT_REST_WXYZ, OBJECT_SCALE))
    res = {"alignment": mp.report()}
    L = {k: mp.to_soma_local(v) for k, v in fr.items()}

    def dir_resid(k):
        Rw, P = soma.fk(L[k])
        bf = fr[k]
        worst = 0.0
        for j in ("LeftArm", "LeftForeArm", "RightArm", "Spine1", "LeftLeg", "LeftShin", "LeftShoulder"):
            (sa, sb), (ba, bb), _ = M.ALIGN[j]
            ds = P[soma.idx[sb]] - P[soma.idx[sa]]
            db = mp.A @ (bf.point(bb) - bf.point(ba))
            c = np.dot(ds, db) / np.linalg.norm(ds) / np.linalg.norm(db)
            worst = max(worst, float(np.degrees(np.arccos(np.clip(c, -1, 1)))))
        return worst

    def roundtrip(k):
        back, ind = mp.base_basis_from_soma(L[k]), mp.basis_from_base_frame(fr[k])
        e = 0.0
        for b in M.EDIT_CHANNELS:
            st = np.asarray(fr[k].frame["basis_wxyz"][b])
            for q in (back[b], ind[b]):
                e = max(e, float(min(np.abs(q - st).max(), np.abs(q + st).max())))
        return e
    rows = {k: {"direction_residual_deg": dir_resid(k), "roundtrip_err": roundtrip(k), "no_mirror": M.no_mirroring_check(mp, L[k])} for k in L}
    _, P1 = soma.fk(L["1"])
    _, P3 = soma.fk(L["3"])
    _, P4 = soma.fk(L["4"])
    _, P5 = soma.fk(L["5"])
    i = soma.idx
    lr = {"left_hand_moved_m": float(np.linalg.norm(P3[i["LeftHand"]] - P1[i["LeftHand"]])),
          "right_hand_moved_m": float(np.linalg.norm(P3[i["RightHand"]] - P1[i["RightHand"]])),
          "base_left_hand_moved_m": float(np.linalg.norm(fr["3"].head("hand_l") - fr["1"].head("hand_l")))}
    lr["pass"] = lr["left_hand_moved_m"] > 0.05 and lr["right_hand_moved_m"] < 1e-6
    loc = lambda Lx, j: Lx[i[j], 3:]                                                       # noqa: E731
    pc = {"chest_moved_m": float(np.linalg.norm(P4[i["Chest"]] - P1[i["Chest"]])),
          "left_hand_follows_m": float(np.linalg.norm(P4[i["LeftHand"]] - P1[i["LeftHand"]])),
          "leftarm_local_unchanged": float(min(np.abs(loc(L["4"], "LeftArm") - loc(L["1"], "LeftArm")).max(), np.abs(loc(L["4"], "LeftArm") + loc(L["1"], "LeftArm")).max())),
          "leftarm_local_same_in_5_as_3": float(min(np.abs(loc(L["5"], "LeftArm") - loc(L["3"], "LeftArm")).max(), np.abs(loc(L["5"], "LeftArm") + loc(L["3"], "LeftArm")).max())),
          "legs_unchanged_m": float(np.linalg.norm(P4[i["LeftFoot"]] - P1[i["LeftFoot"]]))}
    pc["pass"] = pc["chest_moved_m"] > 0.01 and pc["left_hand_follows_m"] > 0.01 and pc["leftarm_local_unchanged"] < 1e-5 \
        and pc["leftarm_local_same_in_5_as_3"] < 1e-5 and pc["legs_unchanged_m"] < 1e-6
    res["poses"] = rows
    res["left_right"] = lr
    res["parent_child"] = pc
    # real S1 frames round trip
    blender(STAGE, "dump", out / "dump_s1.log", blend=W / "inputs" / "S1_baseline.blend", report=out / "S1_dump.json")
    sd = load(out / "S1_dump.json")
    e_real = 0.0
    nm_ok = True
    for f in sd["frames"]:
        bf = M.BaseFrame(sd["rest"], f)
        Lf = mp.to_soma_local(bf)
        back = mp.base_basis_from_soma(Lf)
        for b in M.EDIT_CHANNELS:
            st = np.asarray(f["basis_wxyz"][b])
            e_real = max(e_real, float(min(np.abs(back[b] - st).max(), np.abs(back[b] + st).max())))
        nm_ok &= M.no_mirroring_check(mp, Lf)["ok"]
    res["real_S1_roundtrip_err_max"] = e_real
    res["real_S1_no_mirror_all_frames"] = bool(nm_ok)
    # adapter negative controls
    neg = {}
    try:
        M.Mapping(soma, M.rest_frame(cd["rest"], OBJECT_REST_WXYZ, OBJECT_SCALE), A=np.diag([-1.0, 1.0, 1.0]) @ M.A_BLENDER_TO_SOMA)
        neg["mirrored_axis"] = {"pass": False}
    except M.AdapterStop as e:
        neg["mirrored_axis"] = {"pass": e.code == "AXIS_MAPPING_IMPROPER", "code": e.code}
    # a proper but wrong axis (forward/up swapped) must be caught by the no-mirroring / heading check
    wrong = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    try:
        mw = M.Mapping(soma, M.rest_frame(cd["rest"], OBJECT_REST_WXYZ, OBJECT_SCALE), A=wrong)
        chk = M.no_mirroring_check(mw, mw.to_soma_local(fr["1"]))
        neg["wrong_axis_identity"] = {"pass": not chk["ok"], "check": chk}
    except M.AdapterStop as e:
        neg["wrong_axis_identity"] = {"pass": True, "code": e.code}
    rest_missing = {k: v for k, v in cd["rest"].items() if k != "upperarm_l"}
    try:
        M.Mapping(soma, M.rest_frame(rest_missing, OBJECT_REST_WXYZ, OBJECT_SCALE))
        neg["baserig_missing_bone"] = {"pass": False}
    except M.AdapterStop as e:
        neg["baserig_missing_bone"] = {"pass": e.code == "BASERIG_MISSING_BONES", "code": e.code}
    try:
        names = list(soma.names)
        names[soma.idx["LeftForeArm"]] = "X"
        M.SomaSkeleton(names, soma.parents, soma.ref)
        neg["soma_missing_joint"] = {"pass": False}
    except M.AdapterStop as e:
        neg["soma_missing_joint"] = {"pass": e.code == "SOMA_SKELETON_MISSING_JOINTS", "code": e.code}
    try:
        M.episode(79, 249, 278, 9, protected_source=list(range(241, 252)))
        neg["protected_overlap"] = {"pass": False}
    except M.AdapterStop as e:
        neg["protected_overlap"] = {"pass": e.code == "PROTECTED_OVERLAP", "code": e.code}
    try:
        M.episode(79, 249, 280, 9)
        neg["non_30_sample_window"] = {"pass": False}
    except M.AdapterStop as e:
        neg["non_30_sample_window"] = {"pass": e.code == "UNSUPPORTED_TIME_ALIGNMENT", "code": e.code}
    # out-of-domain build requests must be refused by the Blender stage before any change
    for name, vals in (("channel_outside_domain", {"spine_01": {"180": [1, 0, 0, 0]}}),
                       ("frame_outside_interior", {"upperarm_l": {"210": [1, 0, 0, 0]}})):
        spec = {"out_blend": str(out / f"neg_{name}.blend"), "interior_action_frames": [172, 199], "values_wxyz": vals,
                "branch": "NEG", "action_name": "NEG", "expected_blend_sha256": SOURCES["S1"]["baseline_sha"]}
        dump(out / f"neg_{name}_spec.json", spec)
        try:
            blender(STAGE, "build_variant", out / f"neg_{name}.log", blend=W / "inputs" / "S1_baseline.blend", spec=out / f"neg_{name}_spec.json",
                    report=out / f"neg_{name}_report.json")
            neg[name] = {"pass": False}
        except RuntimeError:
            txt = (out / f"neg_{name}.log").read_text()
            neg[name] = {"pass": ("REFUSED_" in txt) and not (out / f"neg_{name}.blend").exists(),
                         "refusal": [l for l in txt.splitlines() if "REFUSED_" in l][:1]}
    res["adapter_negative_controls"] = neg
    g1 = (res["real_S1_roundtrip_err_max"] <= 1e-5 and res["real_S1_no_mirror_all_frames"]
          and all(r["direction_residual_deg"] <= 0.05 and r["roundtrip_err"] <= 1e-5 and r["no_mirror"]["ok"] for r in rows.values())
          and lr["pass"] and pc["pass"] and neg["mirrored_axis"]["pass"] and neg["wrong_axis_identity"]["pass"])
    res["G1_pass"] = bool(g1)
    res["adapter_G6_pass"] = bool(all(v["pass"] for v in neg.values()))
    dump(out / "controls.json", res)
    print(json.dumps({"G1_pass": res["G1_pass"], "adapter_G6_pass": res["adapter_G6_pass"], "real_rt": e_real,
                      "poses": {k: {kk: (vv if not isinstance(vv, dict) else vv["ok"]) for kk, vv in v.items()} for k, v in rows.items()},
                      "lr": lr, "pc": pc, "neg": {k: v["pass"] for k, v in neg.items()}}, indent=1))
    return 0 if (g1 and res["adapter_G6_pass"]) else 3


# ------------------------------------------------------------------------------------------------ run ---
def readback(blend, rep, log):
    return blender(ACTION_STAGE, "readback", log, blend=blend, report=rep, subframes="0.25,0.5,0.75")


def stage_run(a):
    d = check_declaration(a)
    ctl = load(a.controls)
    if not ctl["G1_pass"]:
        raise SystemExit("STOP: mapping controls failed (G1) - representation incompatible; no S1 inference")
    sid = a.source
    s = SOURCES[sid]
    ep = d["episodes"][sid]
    out = new_dir(Path(a.out_dir) / sid)
    log = {"stage": "run", "source": sid, "role": s["role"], "start": now(), "steps": []}
    soma = M.SomaSkeleton.load(Path(a.preflight).parent / "soma_skeleton.json")
    inp = W / "inputs"
    base_blend = inp / f"{sid}_baseline.blend"
    pin_blend = inp / f"{sid}_W126V_accepted.blend"
    for p, h in ((base_blend, s["baseline_sha"]), (pin_blend, s["pin_accepted_sha"]), (s["run"] / "baseline" / "baseline.blend", s["baseline_sha"])):
        if M.sha256_file(p) != h:
            raise SystemExit(f"BLOCKED_SOURCE_IDENTITY {p}")
    refs_before = M.sha256_file(s["run"] / "lineage" / "refs.json")
    t = blender(STAGE, "dump", out / "dump.log", blend=base_blend, report=out / "baseline_dump.json")
    log["steps"].append(["dump", t])
    bd = load(out / "baseline_dump.json")
    frames = {f["blender_frame"]: f for f in bd["frames"]}
    mp = M.Mapping(soma, M.rest_frame(bd["rest"], OBJECT_REST_WXYZ, OBJECT_SCALE))
    ctx = np.stack([mp.to_soma_local(M.BaseFrame(bd["rest"], frames[f])) for f in ep["context_action"]])
    tgt = np.stack([mp.to_soma_local(M.BaseFrame(bd["rest"], frames[f])) for f in ep["target_action"]])
    np.savez(out / "conditioning.npz", context_local=ctx, target_local=tgt)
    cond = {"context_action": ep["context_action"], "target_action": ep["target_action"], "context_source": ep["context_source"],
            "target_source": ep["target_source"], "no_mirror": [M.no_mirroring_check(mp, x) for x in list(ctx) + list(tgt)],
            "coverage": M.coverage(), "conditioning_sha256": M.sha256_file(out / "conditioning.npz")}
    dump(out / "conditioning.json", cond)
    pf = load(a.preflight)
    base = {"motionbricks_root": str(MB_ROOT), "onnx": str(ONNX), "onnx_sha256": ONNX_SHA, "manifest": str(MANIFEST),
            "manifest_sha256": pf["motionbricks"]["manifest_sha256"], "cache_dir": str(TRT_CACHE),
            "context_local": ctx.tolist(), "target_local": tgt.tolist(), **{k: d["model"]["config"][k] for k in ("allowed_token_counts", "correct_generated_diff", "heading_alignment")}}
    rc, t = mb({**base, "mode": "infer", "ort_crosscheck": True, "out_dir": str(out / "model_R1")}, out / "req_R1.json")
    log["steps"].append(["infer_R1", t, rc])
    if rc != 0:
        dump(out / "run_log.json", log)
        raise SystemExit(f"model run R1 failed rc={rc}")
    rc2, t = mb({**base, "mode": "infer", "ort_crosscheck": False, "out_dir": str(out / "model_R2_rerun")}, out / "req_R2.json")
    log["steps"].append(["infer_R2_determinism_rerun", t, rc2])
    r1 = load(out / "model_R1" / "result.json")
    raw1, raw2 = np.load(out / "model_R1" / "raw_model_io.npz"), np.load(out / "model_R2_rerun" / "raw_model_io.npz")
    eo1, eo2 = np.load(out / "model_R1" / "engine_output.npz"), np.load(out / "model_R2_rerun" / "engine_output.npz")
    det = {"raw_bitwise_equal": bool(np.array_equal(raw1["pred_global_motions"], raw2["pred_global_motions"])),
           "tokens_equal": bool(np.array_equal(raw1["pred_num_tokens"], raw2["pred_num_tokens"])),
           "engine_output_bitwise_equal": bool(np.array_equal(eo1["local_poses"], eo2["local_poses"])),
           "raw_max_abs_diff": float(np.abs(raw1["pred_global_motions"] - raw2["pred_global_motions"]).max())}
    gen = eo1["local_poses"].astype(np.float64)                                   # [V, 77, 7] xyz + qxyzw
    nv = gen.shape[0]
    timing = {"pred_num_tokens": r1["pred_num_tokens"], "valid_frames": nv, "allowed_mask": raw1["in_allowed_pred_num_tokens"].tolist(),
              "has_local_poses": raw1["in_has_local_poses"].tolist(), "target_num_poses": r1["target_num_poses"]}

    def qerr(q1, q2):
        q1, q2 = np.asarray(q1, float), np.asarray(q2, float)
        return float(2 * np.arccos(np.clip(abs(np.dot(q1 / np.linalg.norm(q1), q2 / np.linalg.norm(q2))), -1, 1)))
    end = {}
    for j in ("LeftArm", "LeftForeArm", "LeftShoulder", "Hips"):
        k = soma.idx[j]
        end[j] = {"sample0_vs_context_last_rad": qerr(gen[0, k, 3:], ctx[3, k, 3:]),
                  "last_vs_target_first_rad": qerr(gen[-1, k, 3:], tgt[0, k, 3:])}
    sd, ed = eo1["start_deltas"], eo1["end_deltas"]
    corr = {j: {"start_delta_deg": float(np.degrees(2 * np.arccos(np.clip(abs(sd[soma.idx[j], 6]), 0, 1)))),
                "end_delta_deg": float(np.degrees(2 * np.arccos(np.clip(abs(ed[soma.idx[j], 6]), 0, 1))))}
            for j in ("LeftArm", "LeftForeArm", "LeftShoulder", "Chest", "Spine1", "RightArm", "LeftLeg")} if sd.size else {}
    timing["endpoint"] = end
    timing["endpoint_correction_magnitude"] = corr
    g2 = (timing["pred_num_tokens"] == 9 and nv == 30 and timing["allowed_mask"][0] == [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
          and all(v["sample0_vs_context_last_rad"] <= 2e-3 and v["last_vs_target_first_rad"] <= 2e-3 for k2, v in end.items() if k2 in ("LeftArm", "LeftForeArm")))
    timing["G2_pass"] = bool(g2)
    dump(out / "timing.json", timing)
    res = {"source": sid, "determinism": det, "timing": timing, "model_result": r1}
    if not g2:
        res["STOP"] = "TIMING_GATE_FAILED: no retiming / no window change; candidate not projected"
        dump(out / "run_result.json", res)
        print(json.dumps({"G2": False, "timing": timing}, indent=1, default=str))
        return 3
    # ---- projection to BaseRig (interior only) + B slerp ----
    i0, i1 = ep["interior_action"]
    f0, f1 = ep["valid_action"][0], ep["valid_action"][-1]
    st = bd["stored"]

    def stored_wxyz(ch, f):
        return np.array([st[ch][str(c)][str(f)] for c in range(4)])
    C, B, proj = {}, {}, {}
    for kf, f in enumerate(ep["valid_action"]):
        basis = mp.base_basis_from_soma(gen[kf])
        for ch in M.EDIT_CHANNELS:
            ref = stored_wxyz(ch, f)
            q = M.canon_sign(basis[ch], ref)
            proj.setdefault(ch, {})[str(f)] = {"basis_wxyz": q.tolist(), "baseline_wxyz": ref.tolist(), "angle_from_A_rad": qerr(q, ref)}
            if i0 <= f <= i1:
                C.setdefault(ch, {})[str(f)] = q.tolist()
                qb = M.slerp_wxyz(stored_wxyz(ch, f0), M.canon_sign(stored_wxyz(ch, f1), stored_wxyz(ch, f0)), (f - f0) / (f1 - f0))
                B.setdefault(ch, {})[str(f)] = M.canon_sign(qb, ref).tolist()
    endpoint_basis = {ch: {"context_last_rad": proj[ch][str(f0)]["angle_from_A_rad"], "target_first_rad": proj[ch][str(f1)]["angle_from_A_rad"]}
                      for ch in M.EDIT_CHANNELS}
    dump(out / "projection.json", {"interior_action": [i0, i1], "endpoint_basis_vs_A": endpoint_basis, "per_frame": proj,
                                   "transform": "L_base = D_p L_soma D_j^T, basis = (B_p^T B_b)^T L_base; sign canonicalised to the A key; NO filtering, NO retiming",
                                   "raw_model_io": str(out / "model_R1" / "raw_model_io.npz")})
    blends = {"A": base_blend}
    for br, vals, note in (("B", B, "endpoint-matched per-channel slerp, no learned model"), ("C", C, "MotionBricks-SOMA-RP-v1 primary sample R1")):
        spec = {"out_blend": str(out / f"{br}_{sid}.blend"), "interior_action_frames": [i0, i1], "values_wxyz": vals, "branch": br,
                "action_name": f"AnimCV_EXP W129 {br} {sid}", "expected_blend_sha256": s["baseline_sha"], "note": note}
        dump(out / f"build_{br}_spec.json", spec)
        log["steps"].append([f"build_{br}", blender(STAGE, "build_variant", out / f"build_{br}.log", blend=base_blend,
                                                     spec=out / f"build_{br}_spec.json", report=out / f"build_{br}_report.json")])
        blends[br] = out / f"{br}_{sid}.blend"
    # composed onto the W126V-accepted (pinned) state
    spec = {"out_blend": str(out / f"Cpin_{sid}.blend"), "interior_action_frames": [i0, i1], "values_wxyz": C, "branch": "C_on_W126V",
            "action_name": f"AnimCV_EXP W129 C_on_W126V {sid}", "expected_blend_sha256": s["pin_accepted_sha"], "note": "pin-preservation control"}
    dump(out / "build_Cpin_spec.json", spec)
    log["steps"].append(["build_Cpin", blender(STAGE, "build_variant", out / "build_Cpin.log", blend=pin_blend, spec=out / "build_Cpin_spec.json",
                                                report=out / "build_Cpin_report.json")])
    rb = {}
    for br, bl in list(blends.items()) + [("Cpin", out / f"Cpin_{sid}.blend"), ("Pin", pin_blend)]:
        log["steps"].append([f"readback_{br}", readback(bl, out / f"readback_{br}.json", out / f"readback_{br}.log")])
        rb[br] = load(out / f"readback_{br}.json")
    # ---- G4 / G5 ----
    g4 = {}
    for br in ("B", "C"):
        r = rb[br]
        pres = M.preservation(r, rb["A"], (i0, i1))
        ck = M.changed_keys(_full_dump(br, blends[br], out)["stored"], bd["stored"])
        census = r["fcurve_census"]
        g = {"blend_sha256": r["blend_sha256"], "action_name": r["action_name"], "actions_in_file": r["actions_in_file"],
             "single_action": len(r["actions_in_file"]) == 1, "action_name_distinct": r["action_name"] != rb["A"]["action_name"],
             "n_fcurves": census["n_fcurves"], "interp": census["interpolation_census_curves"],
             "key_range": [census["key_frame_min"], census["key_frame_max"]], "scene": r["scene"],
             "changed_keys_channels": sorted({c[0] for c in ck}), "changed_keys_frames": [min((c[2] for c in ck), default=None), max((c[2] for c in ck), default=None)],
             "n_changed_keys": len(ck), "nonarm_changed": pres["nonarm_changed"][:10], "outside_changed": pres["outside_changed"][:10],
             "sub_outside_changed": pres["sub_outside_changed"][:10], "inside_changed_frames": pres["inside_changed_frames"],
             "provenance": r.get("provenance_action")}
        g["pass"] = bool(g["single_action"] and g["action_name_distinct"] and g["n_fcurves"] == 32 and set(g["interp"]) == {"CONSTANT"}
                         and g["key_range"] == [1.0, float(s["n_action"])] and r["scene"]["effective_fps"] == 30.0
                         and r["scene"]["frame_start"] == 1 and r["scene"]["frame_end"] == s["n_action"]
                         and set(g["changed_keys_channels"]) <= set(M.EDIT_CHANNELS)
                         and all(i0 <= c[2] <= i1 for c in ck) and not pres["nonarm_changed"] and not pres["outside_changed"]
                         and not pres["sub_outside_changed"])
        g4[br] = g
    a_same = M.sha256_file(blends["A"]) == s["baseline_sha"] and rb["A"]["value_digest"] == s["value_digest"]
    lineage_same = M.sha256_file(s["run"] / "lineage" / "refs.json") == refs_before and M.sha256_file(s["run"] / "baseline" / "baseline.blend") == s["baseline_sha"]
    g4["A_untouched_and_digest"] = bool(a_same)
    g4["lineage_and_original_baseline_unchanged"] = bool(lineage_same)
    g4["pass"] = bool(g4["B"]["pass"] and g4["C"]["pass"] and a_same and lineage_same)
    pin_src = load(s["run"] / s["pin_request"] / "proposal.json")
    tgt_m = np.asarray(pin_src["constraint"]["target"])
    pin_act = [f - s["segment"][0] + 1 for f in pin_src["pin_frames"]]
    frp = {f["blender_frame"]: f for f in rb["Pin"]["frames"]}
    frc = {f["blender_frame"]: f for f in rb["Cpin"]["frames"]}
    frC = {f["blender_frame"]: f for f in rb["C"]["frames"]}
    frA = {f["blender_frame"]: f for f in rb["A"]["frames"]}
    res_pin = {f: float(np.linalg.norm(np.asarray(frp[f]["world"]["hand_l"]["head"]) - tgt_m)) for f in pin_act}
    res_cp = {f: float(np.linalg.norm(np.asarray(frc[f]["world"]["hand_l"]["head"]) - tgt_m)) for f in pin_act}
    g5 = {"pin_action_frames": pin_act, "pin_residual_W126V_m": res_pin, "pin_residual_C_composed_m": res_cp,
          "pin_frames_bit_equal_C_vs_A": all(frC[f]["world"] == frA[f]["world"] for f in pin_act),
          "pin_frames_bit_equal_Cpin_vs_W126V": all(frc[f]["world"] == frp[f]["world"] for f in pin_act),
          "composed_preservation": M.preservation(rb["Cpin"], rb["Pin"], (i0, i1))}
    g5["pass"] = bool(g5["pin_frames_bit_equal_C_vs_A"] and g5["pin_frames_bit_equal_Cpin_vs_W126V"]
                      and all(res_cp[f] == res_pin[f] for f in pin_act) and max(res_cp.values()) <= 1e-6
                      and not g5["composed_preservation"]["nonarm_changed"] and not g5["composed_preservation"]["outside_changed"])
    # ---- quality ----
    hinge = M.hinge_reference(rb["A"])
    win = (max(1, i0 - 8), min(s["n_action"], i1 + 8))
    q = {br: M.quality(rb[br], rb["A"], (i0, i1), win, hinge) for br in ("A", "B", "C")}
    for br in ("A", "B", "C"):
        dmp = bd if br == "A" else _full_dump(br, blends[br], out)
        q[br]["arm_local"] = M.armlocal_quality(M.armlocal_wrist(dmp, *win), (i0, i1), win)
    q["note"] = ("world-space wrist metrics include the source-derived torso/object rotation, which is OUT of the edit domain "
                 "and identical in A/B/C; arm_local metrics isolate the edited channels")
    res.update({"G4": g4, "G5": g5, "quality": q, "hinge_reference": hinge.tolist(), "blends": {k: str(v) for k, v in blends.items()},
                "readbacks": {k: M.sha256_file(out / f"readback_{k}.json") for k in rb}})
    dump(out / "run_result.json", res)
    # ---- renders + videos + sheets ----
    vid = render_and_compose(out, sid, s, ep, blends, rb, q, log)
    res["video"] = vid
    log["end"] = now()
    dump(out / "run_log.json", log)
    dump(out / "run_result.json", res)
    print(json.dumps({"G2": timing["G2_pass"], "G4": g4["pass"], "G5": g5["pass"], "det": det, "video": vid.get("native_rate")}, indent=1))
    return 0


def _full_dump(br, blend, out):
    p = out / f"dump_{br}.json"
    if not p.exists():
        blender(STAGE, "dump", out / f"dump_{br}.log", blend=blend, report=p)
    return load(p)


def render_and_compose(out, sid, s, ep, blends, rb, q, log):
    import cv2
    i0, i1 = ep["interior_action"]
    n = s["n_action"]
    fa = {f["blender_frame"]: f for f in rb["A"]["frames"]}
    allp = np.array([fa[f]["world"][b][p] for f in fa for b in fa[f]["world"] for p in ("head", "tail")])
    centre = ((allp.min(0) + allp.max(0)) / 2).tolist()
    ext = float(np.max(allp.max(0) - allp.min(0)))
    el = np.array([fa[f]["world"]["lowerarm_l"]["head"] for f in range(max(1, i0 - 8), min(n, i1 + 8) + 1)]).mean(0).tolist()
    views = [{"name": "body", "centre": centre, "offset": [0.0, -6.0, 0.0], "ortho_scale": ext * 1.15},
             {"name": "arm", "centre": el, "offset": [4.5, -4.5, 1.5], "ortho_scale": 1.3}]
    size = 320
    jobs = {"size": size, "views": views, "frames": [{"blender_frame": f, "markers": []} for f in range(1, n + 1)]}
    dump(out / "render_jobs.json", jobs)
    for br in ("A", "B", "C"):
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
    names = {"A": "A original (no-op)", "B": "B slerp (no model)", "C": "C MotionBricks"}

    def tile(br, f, label):
        body = cv2.imread(str(out / f"frames_{br}" / f"body_f{f:04d}.png"))
        arm = cv2.imread(str(out / f"frames_{br}" / f"arm_f{f:04d}.png"))
        lab(body, label, 16)
        lab(body, f"Action {f}", 34)
        inside = i0 <= f <= i1
        lab(arm, "left-arm close-up", 16)
        lab(arm, "CHANGED interior" if (inside and label[0] in "BC" and label != "?") else ("interior" if inside else "unchanged"),
            34, (80, 80, 255) if inside else (200, 200, 200))
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
            c = (91, 73, 219) if i0 <= kk <= i1 else (220, 220, 220)
            cv2.rectangle(b, (x0, 4), (max(x0, x1 - 1), 18), c, -1)
        xc = 10 + int((f - 0.5) / n * Wb)
        cv2.line(b, (xc, 0), (xc, 33), (0, 0, 0), 2)
        cv2.putText(b, f"edited interior Action {i0}-{i1} = SOURCE {i0 + seg0 - 1}-{i1 + seg0 - 1} (red); everything else unchanged",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 0, 0), 1)
        return b

    def write_video(path, order, labels):
        wr, written = None, 0
        for f in range(1, n + 1):
            top = np.hstack([source_tile(f)] + [tile(br, f, lab_) for br, lab_ in zip(order, labels)])
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

    vid = write_video(out / f"abc_{sid}_30fps.mp4", ["A", "B", "C"], [names["A"], names["B"], names["C"]])
    # contact sheets: around the event, midpoint and target (unblinded) + blinded copies
    c_last = ep["valid_action"][0]
    sheets = {"event": list(range(c_last - 1, c_last + 5)), "midpoint": list(range((i0 + i1) // 2 - 2, (i0 + i1) // 2 + 4)),
              "target": list(range(i1 - 3, i1 + 3)), "overview": list(range(c_last, i1 + 2, 5))}

    def sheet(path, frames_, order, labels):
        rows = []
        srow = np.hstack([np.vstack([cv2.resize(source_tile(f), (size, size)),
                                     lab(np.zeros((size, size, 3), np.uint8), f"SOURCE {f + seg0 - 1} / Action {f}", 20)]) for f in frames_])
        rows.append(srow)
        for br, lb in zip(order, labels):
            rows.append(np.hstack([tile(br, f, lb) for f in frames_]))
        cv2.imwrite(str(path), np.vstack(rows))
        return str(path)
    sh = {k: sheet(out / f"sheet_{k}_{sid}.png", v, ["A", "B", "C"], [names[b] for b in "ABC"]) for k, v in sheets.items()}
    perm = list("ABC")
    random.SystemRandom().shuffle(perm)
    bdir = new_dir(out / "blind")
    dump(out / "blind_key_SEALED.json", {"X": perm[0], "Y": perm[1], "Z": perm[2], "note": "do not read before the blinded review is written"})
    blabels = ["X", "Y", "Z"]
    bsh = {k: sheet(bdir / f"blind_sheet_{k}_{sid}.png", v, perm, [f"panel {x}" for x in blabels]) for k, v in sheets.items()}
    bvid = write_video(bdir / f"blind_{sid}_30fps.mp4", perm, [f"panel {x}" for x in blabels])
    vid.update({"sheets": sh, "blind_sheets": bsh, "blind_video": bvid, "blind_key": str(out / "blind_key_SEALED.json"),
                "blind_key_sha256": M.sha256_file(out / "blind_key_SEALED.json")})
    return vid


# ------------------------------------------------------------------------------------------------ assess ---
def stage_assess(a):
    d = check_declaration(a)
    out = new_dir(a.out_dir)
    pf, ctl = load(a.preflight), load(a.controls)
    rr = load(Path(a.run_dir) / "S1" / "run_result.json")
    vr = load(a.visual_review)
    key = load(Path(a.run_dir) / "S1" / "blind_key_SEALED.json")
    unblind = lambda x: key[x] if x in key else x                                   # noqa: E731
    g = {"G0": pf["status"] == "OK", "G1": ctl["G1_pass"], "G2": rr["timing"]["G2_pass"],
         "G3": rr["model_result"]["onnx_sha256"] == ONNX_SHA and rr["model_result"]["engine_class"].endswith("MotionBricksSOMAEngine"),
         "G4": rr["G4"]["pass"], "G5": rr["G5"]["pass"],
         "G6": all(v["pass"] for v in pf["runtime_negative_controls"].values()) and ctl["adapter_G6_pass"],
         "G7": rr["video"]["native_rate"]}
    qa_ = rr["quality"]
    qa = {"QA1": qa_["C"]["arm_local"]["n_reversal_flicks"] == 0,
          "QA2": all(qa_["C"]["arm_local"]["boundary_c0"][k] <= 2 * qa_["B"]["arm_local"]["boundary_c0"][k] + 10.0 for k in ("entry_step_mm", "exit_step_mm")),
          "QA3": not qa_["C"]["near_straight_frames"] and not qa_["C"]["bend_direction_flips"] and not qa_["C"]["hinge_opposite_frames"],
          "QA4": qa_["C"]["wrist_dev_from_A_mean_mm"] <= 1.25 * qa_["B"]["wrist_dev_from_A_mean_mm"] + 25.0}
    ans = {k: [unblind(x) for x in v] if isinstance(v, list) else unblind(v) for k, v in vr["answers_blinded"].items()}
    qa["QA5"] = bool(ans["V2_most_natural"] == "C" and "C" not in ans["V1_flick"] and "C" not in ans["V3_wrong_elbow"]
                     and ans["V4_least_faithful"] != "C")
    if not g["G0"]:
        case = "DATA_BLOCKED"
    elif not (g["G6"] and g["G3"]):
        case = "D"
    elif not (g["G1"] and g["G2"] and g["G4"] and g["G5"]):
        case = "C"
    elif not g["G7"]:
        case = "C"
    elif all(qa.values()):
        case = "A"
    else:
        case = "B"
    rep = {"stage": "assess", "time": now(), "case": case, "gates": g, "quality_criteria": qa, "visual_answers_unblinded": ans,
           "visual_review_sha256": M.sha256_file(a.visual_review), "blind_key": key, "rule": CASE_RULE,
           "note": "visual review is the agent's own blinded review, not human/animator validation"}
    sha = dump(out / "assessment.json", rep)
    print(json.dumps({"case": case, "gates": g, "qa": qa, "answers": ans, "sha": sha}, indent=1))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=("preflight", "declare", "controls", "run", "assess"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--preflight")
    ap.add_argument("--declaration")
    ap.add_argument("--declaration-sha")
    ap.add_argument("--controls")
    ap.add_argument("--source", default="S1", choices=tuple(SOURCES))
    ap.add_argument("--run-dir")
    ap.add_argument("--visual-review")
    a = ap.parse_args()
    return {"preflight": stage_preflight, "declare": stage_declare, "controls": stage_controls, "run": stage_run,
            "assess": stage_assess}[a.stage](a)


if __name__ == "__main__":
    sys.exit(main())
