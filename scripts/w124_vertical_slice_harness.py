#!/usr/bin/env python3
"""W124 evidence harness for the EXPERIMENTAL opt-in video -> editable Action -> WORLD_POSITION_PIN slice.

This harness is NOT part of the product path.  It drives the user-facing command
``motion-tool experimental-video-action ...`` (PYTHONPATH=src) exactly as a user would, on the declared engineering replays
(W121/W122 sources S1, S3 - not new independent sources), runs the declared negative / source / lineage controls, and
applies the frozen gate + CASE rule.  Every stage refuses to overwrite its output.

  declare   write the prospective declaration (requests, controls, gates, CASE rule, code blobs) BEFORE any S1/S3 outcome
  run       execute the declared plan in the declared order (subprocess calls to the user command; logs kept)
  assess    gates G1-G8 and the frozen CASE rule from the run outputs only (visual review is disclosure, not a gate)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = Path(os.environ.get("W124_BASE", "C:/Users/dna10/animcv-w124"))
PY = os.environ.get("W124_PYTHON", "C:/Projects/AnimCV/.venv_build_full/Scripts/python.exe")
BLENDER = os.environ.get("ANIMCV_BLENDER", "C:/Program Files/Blender Foundation/Blender 5.1/blender.exe")
W121 = Path("C:/Users/dna10/animcv-w121")
SETUP = BASE / "setup"

SOURCES = {  # engineering replays (W121/W122 sources); W121 frozen references for parity
    "S1": {"clip": W121 / "data/originals/64498918_Double_Kettlebell_Strict_Press_side_view_.webm",
           "sha256": "de8feb9d515daa2bfaeed7167fdcc62028dd3f6450872c4fd81f8fb03dd6359c", "segment": [79, 318],
           "w121_npz": W121 / "results/perceive/candidates/Double_Kettlebell_Strict_Press_side_view_.webm/m1s0/observations_predictions.npz",
           "w121_npz_sha256": "7988bf4da320d4def763c81b9d9210905f70b7854e54d71974482392c2510f51",
           "w121_parent": W121 / "results/prepare/S1/parent_state.json",
           "w121_value_digest": "25a09802ffa210df56f0e930295468ef4cdb675e2945cb95491c080b5101c8a9"},
    "S3": {"clip": W121 / "data/originals/162440161_Bent-over_row_-_exercise_demonstration_video.webm",
           "sha256": "f6b90f2f72e9020dcc261fadb09725913fa31719a8614de26fab372826987f62", "segment": [0, 153],
           "w121_npz": W121 / "results/perceive/candidates/Bent-over_row_-_exercise_demonstration_video.webm/m1s0/observations_predictions.npz",
           "w121_npz_sha256": "2a5f3ae555b099138f0b0ae1d49b5fc3b2863a23eb8f7f69f286e8b89f462c54",
           "w121_parent": W121 / "results/prepare/S3/parent_state.json",
           "w121_value_digest": "b90587455a6aa2e831927b229e80462aee8634cd884bb16cc301289573b72937"},
}
ARM = ["lowerarm_l", "upperarm_l"]


def req(rid, frames, capture, offset=None, **kw):
    t = {"kind": "CAPTURED_FROM_EVALUATED_STATE", "capture_frame": capture} if offset is None else \
        {"kind": "CAPTURED_PLUS_DECLARED_OFFSET", "capture_frame": capture, "offset_m": offset}
    r = {"schema": "animcv_exp_world_pin_request_v1", "request_id": rid, "parent_state_id": "@HEAD", "type": "WORLD_POSITION_PIN",
         "semantic_joint": "left_wrist", "coordinate_frame": "SCENE_WORLD", "unit": "m", "frames": frames, "target": t,
         "authorized_channels": list(ARM)}
    r.update(kw)
    return r


# Requests per source.  "@HEAD" is replaced by the baseline head id at proposal time (the user copies it from `build`).
PLAN = {
    "S1": {
        "positive": [req("S1_E1_hold_210_218", [210, 218], 210),
                     req("S1_E2_offset_up_3cm", [210, 218], 210, [0.0, 0.0, 0.03]),
                     req("S1_E3_hold_userreject", [210, 218], 210),
                     req("S1_E4_hold_stalecommit", [210, 218], 210)],
        "negative": {  # request -> (expected failure class, expected reason)
            "S1_N_REACH": (req("S1_N_REACH", [210, 218], 210, [0.0, 0.0, 1.0]), "INFEASIBLE_REQUEST", "UNREACHABLE"),
            "S1_N_WRONGBIND": (req("S1_N_WRONGBIND", [210, 218], 210, binding_id="@right_wrist"), "UNSUPPORTED_BINDING", "BINDING_JOINT_MISMATCH"),
            "S1_N_UNKNOWN_JOINT": (req("S1_N_UNKNOWN_JOINT", [210, 218], 210, semantic_joint="left_ankle"), "UNSUPPORTED_BINDING",
                                   "BINDING_UNDECLARED_JOINT"),
            "S1_N_AMBIGUOUS": (req("S1_N_AMBIGUOUS", [210, 218], 210, semantic_joint="left_hand"), "UNSUPPORTED_BINDING", "BINDING_AMBIGUOUS"),
            "S1_N_RIGHT_WRIST": (req("S1_N_RIGHT_WRIST", [210, 218], 210, semantic_joint="right_wrist"), "UNSUPPORTED_BINDING",
                                 "JOINT_NOT_SUPPORTED_BY_SLICE"),
            "S1_N_NAN": (req("S1_N_NAN", [210, 218], 210, ["NaN", 0.0, 0.0]), "CONTRACT_VIOLATION", "INVALID_NUMERIC_TARGET"),
            "S1_N_FRAME": (req("S1_N_FRAME", [210, 218], 210, coordinate_frame="ARMATURE_LOCAL"), "CONTRACT_VIOLATION",
                           "COORDINATE_FRAME_UNSUPPORTED"),
            "S1_N_UNIT": (req("S1_N_UNIT", [210, 218], 210, unit="cm"), "CONTRACT_VIOLATION", "UNIT_UNSUPPORTED"),
            "S1_N_UNAUTH": (req("S1_N_UNAUTH", [210, 218], 210, authorized_channels=["lowerarm_l", "spine_01", "upperarm_l"]),
                            "CONTRACT_VIOLATION", "UNAUTHORIZED_DOF_FOR_SLICE"),
            "S1_N_STALE": (req("S1_N_STALE", [210, 218], 210, parent_state_id="cc4bf0065938c89e516f478dbcceae18b72ab5498fa78d55d0bbff8c0e0a7dfb"),
                           "CONTRACT_VIOLATION", "STALE_PARENT"),
            "S1_N_OUTSIDE": (req("S1_N_OUTSIDE", [300, 330], 300), "CONTRACT_VIOLATION", "FRAME_NOT_IN_STATE"),
        },
        "primary": "S1_E1_hold_210_218", "user_reject": "S1_E3_hold_userreject", "gate_reject_accept_attempt": "S1_N_REACH",
        "stale_commit": "S1_E4_hold_stalecommit", "exploratory_after_primary": "S1_E2_offset_up_3cm", "post_revert_stale": req("S1_N_STALE_AFTER_REVERT", [210, 218], 210, parent_state_id="@PRIMARY"),
    },
    "S3": {"positive": [req("S3_E1_hold_41_49", [41, 49], 41)], "negative": {}, "primary": "S3_E1_hold_41_49"},
}
SOURCE_CONTROLS = {  # build-time controls (real files) -> expected STOP code
    "SC_BOX_CONTAINER_30000FPS": {"clip": W121 / "data/originals/77353232_Boxing_training.webm", "record": "source_BOX.json", "declared_fps": 30,
                                  "segment": [0, 100], "expect": "UNSUPPORTED_FPS_MISMATCH"},
    "SC_HISTORICAL_50FPS": {"clip": Path("C:/Users/dna10/animcv-w102/data/historical_source.mp4"), "record": "source_HIST.json",
                            "declared_fps": 50, "segment": [0, 100], "expect": "UNSUPPORTED_CADENCE"},
    "SC_CHANGED_BYTE": {"clip": "@S3_COPY_FLIPPED", "sha_of": "S3", "record": "source_S3.json", "declared_fps": 30, "segment": [0, 153],
                        "expect": "BLOCKED_SOURCE_IDENTITY"},
    "SC_WRONG_DECLARED_FPS": {"clip": "@S3", "record": "source_S3.json", "declared_fps": 25, "segment": [0, 153], "expect": "UNSUPPORTED_FPS_MISMATCH"},
    "SC_MISSING_CHECKPOINT": {"clip": "@S3", "record": "source_S3.json", "declared_fps": 30, "segment": [0, 153], "assets": "missing_checkpoint",
                              "expect": "DATA_BLOCKED_MISSING_CHECKPOINT"},
    "SC_ABSENT_BLENDER": {"clip": "@S3", "record": "source_S3.json", "declared_fps": 30, "segment": [0, 153], "blender": "C:/no/blender.exe",
                          "expect": "BLOCKED_BLENDER_UNAVAILABLE"},
    "SC_MISSING_OBSERVATION": {"clip": "@S3", "record": "source_S3.json", "declared_fps": 30, "segment": [0, 153],
                               "replay": "C:/no/observations.npz", "expect": "DATA_BLOCKED_MISSING_OBSERVATION"},
}
GATES = {
    "G1_live_e2e_and_perception_parity": "live frozen M1+S0 run completed on the source; observation arrays byte-equal to the W121 stored arrays",
    "G2_baseline_blend": "baseline .blend: reopened readback == in-process; FK parity <= 1e-5 m; Action value digest == W121 digest; values "
                         "canon-equal to the W121 parent state; 32 CONSTANT F-curves; Blender frames 1..N; scene 30 fps; sidecar mapping complete",
    "G3_nonzero_pin_edit": "primary request: gate ACCEPT, user ACCEPT moves head; >=1 changed frame; residual <= 1 mm on every pin frame "
                           "(integer frames); changed bones == {upperarm_l, lowerarm_l}; changed frames subset of pin frames; display truth ok "
                           "(proposed and final); subframe changes only inside pin intervals",
    "G4_negative_controls": "every declared negative request is REJECTed with the declared class and reason; no head move; immutable files unchanged",
    "G5_lineage": "user REJECT leaves head; accepting a gate-rejected candidate refused; stale commit refused (STALE_PARENT); revert exact "
                  "(values, value digest, integer frames and subframes bit-equal); post-revert stale proposal rejected",
    "G6_native_rate_review_video": "A/B video decodes at the source fps with one frame per Action frame (duration == source seconds)",
    "G7_fbx_roundtrip": "existing exporter + default importer: integer and subframe world error <= 1 mm, interpolation, key range, scene range, "
                        "fps and single Action all preserved (characterisation; PASS only if no mismatch)",
    "G8_regression_and_isolation": "focused tests + existing regression tests pass; default CLI parser digest unchanged; historical Worklogs unchanged",
    "G9_source_controls": "every declared source/runtime control STOPs with the declared code",
}
CASE_RULE = ("DATA_BLOCKED if no declared source could run live for asset/runtime reasons.  D if any integrity gate fails "
             "(G4, G5, G8, G9, or a parity part of G1/G2 on a source that ran).  A if G1-G6 pass on >=1 live source AND G7 PASS on "
             "the edited head.  B if G1-G6 pass on >=1 live source and G7 FAILs.  C otherwise (controls valid but core capability "
             "not demonstrated).  The agent visual review is disclosure only and never changes the CASE.")
FROZEN_FILES = ["src/app/experimental_video_action.py", "src/app/experimental_frozen_launch.py", "src/blender/experimental_action_stage.py",
                "src/app/cli.py", "tests/test_experimental_video_action.py", "scripts/w124_vertical_slice_harness.py",
                "examples/experimental_video_action/frozen_code_sha256.json"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def dump(p, o):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(o, indent=1, sort_keys=True, default=str), encoding="utf-8")
    return sha(p)


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def new_dir(p):
    p = Path(p)
    if p.exists():
        raise SystemExit(f"refusing to overwrite {p}")
    p.mkdir(parents=True)
    return p


def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()


# ------------------------------------------------------------------------------------------------- declare ---
def cmd_declare(a):
    out = new_dir(a.out_dir)
    decl = {"schema": "w124_declaration_v1", "batch": 124, "outcomes_inspected": False,
            "note": "S2 was used only as a pre-declaration DEBUG fixture (offline replay) and is excluded from all evidence",
            "head": git("rev-parse", "HEAD"), "code_sha256": {f: sha(ROOT / f) for f in FROZEN_FILES},
            "sources": {s: {k: str(v) for k, v in d.items()} for s, d in SOURCES.items()}, "plan": PLAN,
            "source_controls": {k: {kk: str(vv) for kk, vv in v.items()} for k, v in SOURCE_CONTROLS.items()},
            "order": ["S1: build(live) -> propose E1,E2,E3,E4 + negatives -> commit E3 reject -> commit N_REACH accept (refused) -> "
                      "commit E1 accept -> commit E4 accept (stale) -> commit E2 accept (exploratory; stale or gate-rejected) -> review E1 -> fbx head -> fbx baseline -> revert -> "
                      "post-revert stale proposal", "S3: build(live) -> propose E1 -> commit accept -> review -> fbx head -> revert",
                      "source controls", "stale-manifest/tamper control on a copy of the S3 run"],
            "gates": GATES, "case_rule": CASE_RULE, "experimental_diagnostic_tolerances": {"hard_pin_m": 1e-3, "fbx_m": 1e-3, "fk_parity_m": 1e-5,
                                                                                          "cadence_ms": 1.0},
            "subframes": [0.25, 0.5, 0.75], "visual_review": "agent inspects review stills + decoded video frames after render; disclosure only"}
    s = dump(out / "declaration.json", decl)
    print(s)


def check_decl(a):
    if sha(a.declaration) != a.declaration_sha:
        raise SystemExit("declaration hash mismatch")
    d = load(a.declaration)
    bad = [f for f, h in d["code_sha256"].items() if sha(ROOT / f) != h]
    if bad:
        raise SystemExit(f"code changed after declaration: {bad}")
    return d


# ----------------------------------------------------------------------------------------------------- run ---
class Runner:
    def __init__(self, out):
        self.out = out
        self.log = []
        self.env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), ANIMCV_BLENDER=BLENDER)

    def cli(self, tag, *args):
        cmd = [PY, "-X", "utf8", "-m", "app.cli", "experimental-video-action", *map(str, args)]
        t0 = time.time()
        r = subprocess.run(cmd, cwd=ROOT, env=self.env, capture_output=True, text=True, encoding="utf-8", errors="replace")
        rec = {"tag": tag, "cmd": cmd, "returncode": r.returncode, "seconds": time.time() - t0, "stdout": r.stdout[-6000:],
               "stderr_tail": r.stderr[-3000:]}
        stop = [ln for ln in r.stderr.splitlines() if "STOP " in ln]
        rec["stop"] = stop[-1].split("STOP ", 1)[1].split(":", 1)[0] if stop else None
        self.log.append(rec)
        dump(self.out / "command_log.json", self.log)
        print(f"[{tag}] rc={r.returncode} {rec['seconds']:.1f}s stop={rec['stop']}", flush=True)
        return rec


def materialise(rq, head, bindings, primary_id=None):
    r = json.loads(json.dumps(rq))
    if r["parent_state_id"] == "@HEAD":
        r["parent_state_id"] = head
    if r["parent_state_id"] == "@PRIMARY":
        r["parent_state_id"] = primary_id
    if r.get("binding_id") == "@right_wrist":
        r["binding_id"] = [b for b, x in bindings["bound"].items() if x["semantic_joint"] == "right_wrist"][0]
    t = r["target"]
    if "offset_m" in t:
        t["offset_m"] = [float("nan") if x == "NaN" else x for x in t["offset_m"]]
    return r


def cmd_run(a):
    d = check_decl(a)
    out = new_dir(a.out_dir)
    R = Runner(out)
    reqdir = out / "requests"
    reqdir.mkdir()
    runs = {}
    for sid in a.sources:
        src = SOURCES[sid]
        run = BASE / "runs" / f"{sid}_live"
        rec = R.cli(f"{sid}_build", "build", "--clip", src["clip"], "--clip-sha256", src["sha256"], "--source-record", SETUP / f"source_{sid}.json",
                    "--declared-fps", 30, "--segment", *src["segment"], "--perception-assets", SETUP / "perception_assets_main.json",
                    "--python", PY, "--blender", BLENDER, "--out", run)
        runs[sid] = str(run)
        if rec["returncode"] != 0:
            continue
        head = json.loads(rec["stdout"][rec["stdout"].index("{"):])["baseline_state_id"]
        bnd = load(run / "baseline" / "bindings.json")
        P = PLAN[sid]
        allreq = [(r["request_id"], r) for r in P["positive"]] + [(k, v[0]) for k, v in P["negative"].items()]
        for rid, rq in allreq:
            p = reqdir / f"{rid}.json"
            p.write_text(json.dumps(materialise(rq, head, bnd), allow_nan=True), encoding="utf-8")
            R.cli(f"{rid}_propose", "propose", "--run", run, "--request", p)
        if sid == "S1":
            R.cli("S1_commit_E3_userreject", "commit", "--run", run, "--request-id", P["user_reject"], "--decision", "reject")
            R.cli("S1_commit_NREACH_accept", "commit", "--run", run, "--request-id", P["gate_reject_accept_attempt"], "--decision", "accept")
        R.cli(f"{sid}_commit_primary", "commit", "--run", run, "--request-id", P["primary"], "--decision", "accept")
        if sid == "S1":
            R.cli("S1_commit_E4_stale", "commit", "--run", run, "--request-id", P["stale_commit"], "--decision", "accept")
            R.cli("S1_commit_E2_exploratory", "commit", "--run", run, "--request-id", P["exploratory_after_primary"], "--decision", "accept")
        R.cli(f"{sid}_review", "review", "--run", run, "--request-id", P["primary"])
        R.cli(f"{sid}_fbx_head", "fbx", "--run", run, "--state", "head")
        if sid == "S1":
            R.cli("S1_fbx_baseline", "fbx", "--run", run, "--state", "baseline")
        R.cli(f"{sid}_revert", "revert", "--run", run)
        if sid == "S1":
            prim = load(run / "edits" / P["primary"] / "proposal.json")["candidate"]
            p = reqdir / "S1_N_STALE_AFTER_REVERT.json"
            p.write_text(json.dumps(materialise(P["post_revert_stale"], head, bnd, prim)), encoding="utf-8")
            R.cli("S1_N_STALE_AFTER_REVERT_propose", "propose", "--run", run, "--request", p)
        R.cli(f"{sid}_status", "status", "--run", run)
    # source / runtime controls
    sc_dir = out / "source_controls"
    sc_dir.mkdir()
    bad_assets = load(SETUP / "perception_assets_main.json")
    bad_assets["checkpoint"] = "C:/no/s0_checkpoint.pt"
    dump(sc_dir / "assets_missing_checkpoint.json", bad_assets)
    flipped = sc_dir / "S3_copy_flipped.webm"
    b = bytearray(Path(SOURCES["S3"]["clip"]).read_bytes())
    b[len(b) // 2] ^= 0xFF
    flipped.write_bytes(bytes(b))
    for name, c in SOURCE_CONTROLS.items():
        clip = {"@S3": SOURCES["S3"]["clip"], "@S3_COPY_FLIPPED": flipped}.get(str(c["clip"]), c["clip"])
        csha = SOURCES[c["sha_of"]]["sha256"] if c.get("sha_of") else (sha(clip) if Path(clip).exists() else "0" * 64)
        args = ["build", "--clip", clip, "--clip-sha256", csha, "--source-record", SETUP / c["record"], "--declared-fps", c["declared_fps"],
                "--segment", *c["segment"], "--python", PY, "--blender", c.get("blender", BLENDER), "--out", sc_dir / f"{name}_run"]
        if c.get("replay"):
            args += ["--replay-observations", c["replay"], "--replay-sha256", "0" * 64]
        else:
            args += ["--perception-assets", sc_dir / "assets_missing_checkpoint.json" if c.get("assets") else SETUP / "perception_assets_main.json"]
        R.cli(name, *args)
    # stale-manifest / tamper controls on a COPY of the S3 run (the original run is never modified)
    if "S3" in runs and (Path(runs["S3"]) / "run.json").exists():
        for name, target in (("TC_STALE_MANIFEST", "manifest/input_manifest.json"), ("TC_BASELINE_BLEND_TAMPER", "baseline/baseline.blend")):
            cp = sc_dir / f"{name}_copy"
            shutil.copytree(runs["S3"], cp)
            f = cp / target
            os.chmod(f, stat.S_IWRITE | stat.S_IREAD)
            with open(f, "ab") as fh:
                fh.write(b" ")
            R.cli(name, "status", "--run", cp)
    dump(out / "run_index.json", {"runs": runs, "declaration_sha256": a.declaration_sha})


# -------------------------------------------------------------------------------------------------- assess ---
def cmd_assess(a):
    import numpy as np
    d = check_decl(a)
    out = new_dir(a.out_dir)
    idx = load(a.run_dir / "run_index.json")
    log = {r["tag"]: r for r in load(a.run_dir / "command_log.json")}
    rep = {"declaration_sha256": a.declaration_sha, "sources": {}, "gates": {}}
    sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
    import w116_semantic_pin_base as S
    for sid, run in idx["runs"].items():
        run = Path(run)
        g = {}
        src = SOURCES[sid]
        P = PLAN[sid]
        if log[f"{sid}_build"]["returncode"] != 0:
            rep["sources"][sid] = {"build_stop": log[f"{sid}_build"]["stop"]}
            continue
        man = load(run / "manifest/input_manifest.json")
        sc = load(run / "baseline/action_sidecar.json")
        reg = load(run / "lineage/blends.json")
        v0 = sc["baseline_state_id"]
        # G1
        live = man["perception"]["mode"] == "LIVE_FROZEN_M1_S0"
        z1, z0 = np.load(run / "perception/m1s0/observations_predictions.npz"), np.load(src["w121_npz"])
        arrays_eq = {k: bool(np.array_equal(z1[k], z0[k], equal_nan=True)) for k in z0.files}
        g["G1_live_e2e_and_perception_parity"] = {"live": live, "npz_sha256": man["perception"]["observation_prediction_sha256"],
                                                  "w121_npz_sha256": src["w121_npz_sha256"], "w121_npz_verified": sha(src["w121_npz"]) == src["w121_npz_sha256"],
                                                  "arrays_byte_equal": arrays_eq, "file_sha_equal": man["perception"]["observation_prediction_sha256"] == src["w121_npz_sha256"]}
        g["G1_live_e2e_and_perception_parity"]["pass"] = live and all(arrays_eq.values())
        # G2
        brb = load(run / reg[v0]["readback"])
        st0 = load(run / "lineage/states" / f"{v0}.json")
        w121p = load(src["w121_parent"])
        mp = sc["frame_mapping"]
        g2 = {"readback_equals_in_process": reg[v0]["readback_equals_in_process"], "fk_model_max_error_m": sc["fk_model_max_error_m"],
              "value_digest": brb["value_digest"], "w121_value_digest": src["w121_value_digest"],
              "values_equal_w121_parent": S.canon(st0["values"]) == S.canon(w121p["values"]) and st0["frames"] == w121p["frames"],
              "fcurves": brb["fcurve_census"]["n_fcurves"], "interpolation": brb["fcurve_census"]["interpolation_census_curves"],
              "key_range": [brb["fcurve_census"]["key_frame_min"], brb["fcurve_census"]["key_frame_max"]], "scene": brb["scene"],
              "mapping_complete": [m["source_frame"] for m in mp] == list(range(src["segment"][0], src["segment"][1] + 1))
              and [m["action_frame"] for m in mp] == list(range(1, len(mp) + 1)), "ownership": brb.get("ownership"),
              "driven": sorted(sc["driven_channels"]), "n_not_driven_bones": len(sc["not_driven_bones_at_rest"])}
        n = len(mp)
        g2["pass"] = (g2["readback_equals_in_process"] and g2["fk_model_max_error_m"] <= 1e-5 and g2["value_digest"] == src["w121_value_digest"]
                      and g2["values_equal_w121_parent"] and g2["fcurves"] == 32 and g2["interpolation"] == {"CONSTANT": 32}
                      and g2["key_range"] == [1.0, float(n)] and g2["scene"]["effective_fps"] == 30.0
                      and (g2["scene"]["frame_start"], g2["scene"]["frame_end"]) == (1, n) and g2["mapping_complete"])
        g["G2_baseline_blend"] = g2
        # G3
        pid = P["primary"]
        pr = load(run / "edits" / pid / "proposal.json")
        cm = load(run / "edits" / pid / "commit.json") if (run / "edits" / pid / "commit.json").exists() else {}
        fp = pr.get("footprint") or {}
        res = [e["pins"][0]["candidate_residual_m"] for e in (pr.get("evaluations") or {}).values()]
        g3 = {"gate_decision": pr["decision"], "final_status": cm.get("final_status"), "changed_frames": fp.get("frames_changed"),
              "changed_bones": sorted(fp.get("channel_frames", {})), "pin_frames": pr.get("pin_frames"), "residual_max_m": max(res) if res else None,
              "control_residual_max_m": pr.get("control_residual_max_m"), "display_check": pr.get("display_check"),
              "display_final_check": cm.get("display_final_check"), "subframes": pr.get("subframes"),
              "max_wrist_disp_mm": max((v["wrist_disp_m"] * 1e3 for v in fp.get("per_frame", {}).values()), default=None),
              "quality_disclosure": pr.get("quality_disclosure")}
        g3["pass"] = bool(pr["decision"]["decision"] == "ACCEPT" and cm.get("final_status") == "ACCEPTED" and fp.get("frames_changed")
                          and res and max(res) <= 1e-3 and set(g3["changed_bones"]) == set(ARM)
                          and set(fp["frames_changed"]) <= set(pr["pin_frames"]) and pr["display_check"]["ok"]
                          and cm["display_final_check"]["ok"] and pr["subframes"]["changed_intervals_within_pin_frames"]
                          and pr["subframes"]["pin_residual_at_subframes_max_m"] <= 1e-3)
        g["G3_nonzero_pin_edit"] = g3
        # G4
        neg = {}
        for rid, (_, cls, reason) in P["negative"].items():
            q = load(run / "edits" / rid / "proposal.json")
            reasons = set(q["decision"].get("reasons", [])) | {r for _, r in q.get("validation_errors", [])}
            neg[rid] = {"decision": q["decision"], "expected": [cls, reason],
                        "ok": q["decision"]["decision"] == "REJECT" and q["decision"]["failure_class"] == cls and reason in reasons}
        g["G4_negative_controls"] = {"controls": neg, "pass": all(v["ok"] for v in neg.values())}
        # G5
        g5 = {}
        if sid == "S1":
            ur = load(run / "edits" / P["user_reject"] / "commit.json")
            ga = load(run / "edits" / P["gate_reject_accept_attempt"] / "commit.json")
            sc2 = load(run / "edits" / P["stale_commit"] / "commit.json")
            prs = load(run / "edits" / "S1_N_STALE_AFTER_REVERT" / "proposal.json")
            g5.update(user_reject={"final": ur["final_status"], "head_unchanged": ur["head_before"] == ur["head_after"] == v0,
                                   "gate_decision_was": load(run / "edits" / P["user_reject"] / "proposal.json")["decision"]["decision"]},
                      gate_reject_accept_attempt={"final": ga["final_status"], "head_unchanged": ga["head_before"] == ga["head_after"],
                                                  "reasons": ga["store_record"].get("reasons")},
                      stale_commit={"final": sc2["final_status"], "reasons": sc2["store_record"].get("reasons"),
                                    "head_unchanged": sc2["head_before"] == sc2["head_after"]},
                      post_revert_stale={"decision": prs["decision"]})
            g5["immutable_unchanged_all_commits"] = all(load(p)["immutable_files_unchanged"] for p in run.glob("edits/*/commit.json"))
        rv = sorted(run.glob("reverted/*_report.json"))
        g5["revert"] = {k: load(rv[0])[k] for k in ("exact_undo", "values_equal_restored", "value_digest_equal", "blender_eval_integer_frames_bit_equal",
                                                     "blender_eval_subframes_bit_equal", "restored_state_id")} if rv else None
        ok5 = bool(rv) and g5["revert"]["exact_undo"] and g5["revert"]["restored_state_id"] == v0
        if sid == "S1":
            ok5 = ok5 and (g5["user_reject"]["final"] == "REJECTED" and g5["user_reject"]["head_unchanged"] and g5["user_reject"]["gate_decision_was"] == "ACCEPT"
                           and g5["gate_reject_accept_attempt"]["final"] == "REJECTED" and g5["gate_reject_accept_attempt"]["head_unchanged"]
                           and "GATE_REJECTED_CANNOT_ACCEPT" in (g5["gate_reject_accept_attempt"]["reasons"] or [])
                           and g5["stale_commit"]["final"] == "REJECTED" and g5["stale_commit"]["head_unchanged"]
                           and g5["stale_commit"]["reasons"] == ["STALE_PARENT"]
                           and g5["post_revert_stale"]["decision"]["decision"] == "REJECT"
                           and "STALE_PARENT" in g5["post_revert_stale"]["decision"]["reasons"] and g5["immutable_unchanged_all_commits"])
        g5["pass"] = ok5
        g["G5_lineage"] = g5
        # G6
        rr = run / "review" / pid / "review_report.json"
        g6 = load(rr) if rr.exists() else None
        g["G6_native_rate_review_video"] = {"report": g6, "pass": bool(g6 and g6["native_rate"] and g6["frames_decoded"] == n
                                                                         and abs(g6["seconds"] - g6["source_seconds"]) < 1e-9)}
        # G7
        fx = {}
        for p in sorted(run.glob("fbx/*/fbx_roundtrip_report.json")):
            r = load(p)
            fx[p.parent.name] = {k: r.get(k) for k in ("state_id", "fbx_file_exists", "reimport", "integer_world_max_err_m", "integer_world_err_median_m",
                                                       "integer_frames_over_tol", "integer_frames_compared", "subframe_world_max_err_m",
                                                       "subframe_samples_compared", "blend_interpolation", "fbx_interpolation",
                                                       "blend_key_range", "fbx_key_range", "fbx_scene", "fbx_actions", "fbx_rotation_modes",
                                                       "mismatches", "roundtrip_verdict", "blend_unchanged_by_export", "export")}
        head_fbx = [v for k, v in fx.items() if k.startswith("head_")]
        g["G7_fbx_roundtrip"] = {"reports": fx, "pass": bool(head_fbx) and head_fbx[0]["roundtrip_verdict"] == "PASS"}
        rep["sources"][sid] = g
    # G9 source controls + tamper controls
    scr = {}
    for name, c in SOURCE_CONTROLS.items():
        scr[name] = {"expected": c["expect"], "got": log[name]["stop"], "ok": log[name]["stop"] == c["expect"] and log[name]["returncode"] != 0}
    for name, exp in (("TC_STALE_MANIFEST", "BLOCKED_STALE_MANIFEST"), ("TC_BASELINE_BLEND_TAMPER", "BLOCKED_IMMUTABLE_OUTPUT_CHANGED")):
        if name in log:
            scr[name] = {"expected": exp, "got": log[name]["stop"], "ok": log[name]["stop"] == exp}
    rep["gates"]["G9_source_controls"] = {"controls": scr, "pass": all(v["ok"] for v in scr.values())}
    # G8 regression (results recorded by the agent from pytest junit files)
    g8 = {}
    for name in ("focused", "regression"):
        p = a.run_dir / f"pytest_{name}.json"
        g8[name] = load(p) if p.exists() else None
    g8["pass"] = all(v and v.get("failed") == 0 and v.get("errors") == 0 and v.get("passed", 0) > 0 for v in (g8["focused"], g8["regression"]))
    g8["historical_worklogs_changed"] = [f for f in git("diff", "--name-only", d["head"]).split() if f.startswith("docs/") and "_WORKLOG_" in f]
    g8["default_parser_digest_test"] = "tests/test_experimental_video_action.py::test_default_cli_commands_unchanged_and_experimental_is_additive"
    g8["pass"] = g8["pass"] and not g8["historical_worklogs_changed"]
    rep["gates"]["G8_regression_and_isolation"] = g8
    # CASE
    live_ok = [s for s, g in rep["sources"].items() if "G1_live_e2e_and_perception_parity" in g]
    core = [s for s in live_ok if all(rep["sources"][s][k]["pass"] for k in ("G1_live_e2e_and_perception_parity", "G2_baseline_blend",
                                                                            "G3_nonzero_pin_edit", "G6_native_rate_review_video"))]
    integrity = (all(rep["sources"][s]["G4_negative_controls"]["pass"] and rep["sources"][s]["G5_lineage"]["pass"] for s in live_ok)
                 and rep["gates"]["G9_source_controls"]["pass"] and g8["pass"]
                 and all(rep["sources"][s]["G1_live_e2e_and_perception_parity"]["pass"] and rep["sources"][s]["G2_baseline_blend"]["pass"]
                         for s in live_ok))
    if not live_ok:
        case = "DATA_BLOCKED"
    elif not integrity:
        case = "D"
    elif core and any(rep["sources"][s]["G7_fbx_roundtrip"]["pass"] for s in core):
        case = "A"
    elif core:
        case = "B"
    else:
        case = "C"
    rep.update(case=case, case_rule=CASE_RULE, live_sources=live_ok, core_sources=core, integrity=integrity)
    s = dump(out / "assessment.json", rep)
    print(json.dumps({"case": case, "live": live_ok, "core": core, "integrity": integrity, "sha256": s,
                      "gates": {sid: {k: v.get("pass") for k, v in g.items() if isinstance(v, dict)} for sid, g in rep["sources"].items()},
                      "G8": g8["pass"], "G9": rep["gates"]["G9_source_controls"]["pass"]}, indent=1))


REGRESSION = ["tests/test_cli.py", "tests/test_gui_app.py", "tests/test_blender_executor.py", "tests/test_motion_graph.py",
              "tests/test_keyframe_collapse.py", "tests/test_w116_semantic_pin.py", "tests/test_w118_transition_edit.py",
              "tests/test_w105_rig_utility.py", "tests/test_fk_solver.py", "tests/test_canonical_pose_parity.py", "tests/test_retarget_solver.py",
              "tests/test_w121_independent_transfer.py", "tests/test_render_blender_animation_video.py", "tests/test_w117_temporal_pin.py",
              "tests/test_w120_factor_attribution.py", "tests/test_w122_realtime_utility.py"]


def cmd_pytest(a):
    """Run focused + regression tests; record junit counts into the run directory (G8 input)."""
    import xml.etree.ElementTree as ET
    for name, files in (("focused", ["tests/test_experimental_video_action.py"]), ("regression", REGRESSION)):
        junit = a.run_dir / f"pytest_{name}.xml"
        if junit.exists():
            raise SystemExit(f"refusing to overwrite {junit}")
        r = subprocess.run([PY, "-X", "utf8", "-m", "pytest", *files, "-q", "-p", "no:cacheprovider", f"--junitxml={junit}"], cwd=ROOT,
                           capture_output=True, text=True, encoding="utf-8", errors="replace", env=dict(os.environ, ANIMCV_BLENDER=BLENDER))
        t = ET.parse(junit).getroot()
        ts = t if t.tag == "testsuite" else t.find("testsuite")
        tot, fail, err, skip = (int(ts.get(k, 0)) for k in ("tests", "failures", "errors", "skipped"))
        dump(a.run_dir / f"pytest_{name}.json", {"files": files, "returncode": r.returncode, "tests": tot, "failed": fail, "errors": err,
                                                  "skipped": skip, "passed": tot - fail - err - skip, "tail": r.stdout[-1500:],
                                                  "junit_sha256": sha(junit)})
        print(name, tot, fail, err, skip, r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=("declare", "run", "pytest", "assess"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--declaration", type=Path)
    ap.add_argument("--declaration-sha")
    ap.add_argument("--run-dir", type=Path)
    ap.add_argument("--sources", nargs="+", default=["S1", "S3"])
    a = ap.parse_args()
    for k in ("out_dir", "declaration", "run_dir"):
        if getattr(a, k) is not None:
            setattr(a, k, getattr(a, k).resolve())
    {"declare": cmd_declare, "run": cmd_run, "pytest": cmd_pytest, "assess": cmd_assess}[a.stage](a)


if __name__ == "__main__":
    main()
