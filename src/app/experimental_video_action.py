"""EXPERIMENTAL opt-in slice: authentic 30 fps video -> known BaseRig editable Blender Action -> one WORLD_POSITION_PIN.

Status: EXPERIMENTAL (Batch 124).  NOT part of the default pipeline.  The default CLI/GUI path (MotionGraph ->
RetargetSolver -> AnimationClip -> ``export-blender``) does not import this module; it is reached only through the
explicit ``motion-tool experimental-video-action ...`` subcommand.  See ``docs/EXPERIMENTAL_VIDEO_ACTION_WORLD_PIN.md``.

What it does (and nothing more):

  build    authenticate one source clip (SHA-256, declared 30 fps, container presentation-time cadence), run the FROZEN
           W102 M1 + S0 perception live (or an explicitly labelled offline replay of stored arrays), map the S0 hypothesis
           to the known W105 BaseRig with the frozen W105 GT-free FIXED-slot adapter, and write a NEW baseline .blend whose
           single Action is keyed with the frozen W116 rules.  A fresh Blender process re-opens the .blend and reads it back
           (F-curves, interpolation, timebase, world pose at integer frames and subframes).
  propose  one user-authored WORLD_POSITION_PIN request (left_wrist only, Blender scene-world metres, authorised DOF
           exactly {upperarm_l, lowerarm_l}).  The frozen W116 contract validates it and the frozen W116 analytic resolver
           solves the pin frames only (no automatic out-of-pin edit: budget 0).  The candidate is keyed into its own .blend,
           read back independently, and gated by the frozen W116/W118 evaluator (hard pin 1 mm EXPERIMENTAL diagnostic,
           scope, locality, root, lengths, joint limit, truthful footprint).  Nothing is committed.
  commit   explicit ACCEPT or REJECT of one proposal against the current head (atomic; stale parent refused; a gate-rejected
           candidate can never be accepted).
  revert   new state restoring the head's parent values; its .blend readback must equal the restored state's bitwise.
  review   30 fps side-by-side video: original source | baseline Action | accepted edited Action, footprint overlaid.
  fbx      export a state's .blend through the existing ``blender.export.export_fbx`` and re-import it with the existing
           default importer; integer-frame / subframe / interpolation / timebase readback is REPORTED, never repaired.

Unknown / invalid is never zero, rest pose, interpolation or user intent: an unsupported frame, cadence, binding or asset
stops the command with a precise BLOCKED / UNSUPPORTED code instead of fabricating data.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

SLICE_ID = "animcv_exp_video_action_world_pin_v1"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = PROJECT_ROOT / "scripts"
STAGE = PROJECT_ROOT / "src" / "blender" / "experimental_action_stage.py"
LAUNCHER = PROJECT_ROOT / "src" / "app" / "experimental_frozen_launch.py"

SUPPORTED_FPS = 30.0          # the only cadence validated for this slice (W121/W122 sources)
CADENCE_TOL_MS = 1.0          # W121 declared rule: container presentation step must equal 1000/fps within 1 ms
SLICE_JOINTS = ("left_wrist",)
SLICE_AUTHORIZED = ("lowerarm_l", "upperarm_l")
ARM_CHAIN = ("upperarm_l", "lowerarm_l")
SUBFRAMES = (0.25, 0.5, 0.75)
FBX_DIAGNOSTIC_TOL_M = 1e-3   # W106/W107 diagnostic 1 mm (EXPERIMENTAL; not a Production tolerance)
FK_PARITY_TOL_M = 1e-5        # W121 gate: Python FK model vs Blender evaluation
TARGET_KINDS = ("CAPTURED_FROM_EVALUATED_STATE", "CAPTURED_PLUS_DECLARED_OFFSET")
REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# Known rig (repository files, content-bound).
RIG_FILES = {
    "fbx": ("examples/skinned_mannequin/BaseRig_Skinned.fbx", "c2947ba7f0d088aa47dbf0e04729a991699f24c649acce252b7b902740bb61d7"),
    "rest": ("examples/e2e_demo/baserig_blender_rest.json", "1dcdc8268777bd433d1b73d56acef9a3f67b30f16456a80618d76fb6e70f9358"),
    "calibration": ("examples/e2e_demo/framepose_fk_calibration.json", "a2b8594d9301a2258446859bb4cb3f702de8ae940c509c20dd24ba6b18b0f69b"),
    "chains": ("examples/e2e_demo/framepose_ik_chains.json", "a3a8e52f797c67450e61d71c0ab9f1dabc89413d7b647d7663797b2a2febf916"),
}
# Frozen algorithms reused unchanged (SHA-256 of LF-normalised source).  A change stops the slice (no silent drift).
FROZEN_CODE = {
    "scripts/w116_semantic_pin_base.py": None, "scripts/w116_semantic_pin_contract.py": None, "scripts/w116_blender_stages.py": None,
    "scripts/w117_temporal_pin_base.py": None, "scripts/w118_transition_edit_base.py": None, "scripts/w105_rig_utility.py": None,
    "scripts/w102_rig_review.py": None, "scripts/w121_independent_transfer.py": None, "scripts/w121_independent_transfer_base.py": None,
    "scripts/render_runtime_heading_rig_blender.py": None, "scripts/review_framepose_fk_blender.py": None,
    "scripts/review_framepose_ik_gt_attribution_blender.py": None, "src/motion/torso_frame.py": None,
    "src/retarget/framepose_two_bone_ik.py": None, "src/retarget/framepose_fk.py": None, "src/retarget/framepose_target_rest.py": None,
    "src/blender/executor.py": None, "src/blender/export.py": None, "src/blender/framepose_rest_adapter.py": None,
    "src/blender/framepose_object_owner_adapter.py": None,
}
FROZEN_CODE_FILE = PROJECT_ROOT / "examples" / "experimental_video_action" / "frozen_code_sha256.json"
# Frozen W102 M1 + S0 perception identities (paths are supplied by the user in a perception-assets JSON).
PERCEPTION_EXPECTED = {
    "checkpoint": "6c12fea9f84ea0493fada8b9acc96cd5c70fdd5d9114603e954f9412f4247b44",
    "rtmdet": "78e30dcce0c6f594eaff0d6977b84b4103688b4aff0ad1aa16008a8cc854a7fb",
    "rtmpose": "77ffc7e802acf10951c353e8bc68b4f05218121177ceaea163aa124436ba6fb7",
    "gate_report": "d0be19367ca1a64acf8cc0355ba0f6129787bbf8a4c5f050635e24c5cbba8966",
    "lock": "5bf8e2caf0d40ef6057fc677af4d2482825fdc8ea3f46f321ac128c38b3b7277",
    "diagnostic_script": "6a2eff921d7db48ef91ea9d39da17c84338ac630c55790246e5f12f5fd53683a",
}
RTMDET_NAME = "rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth"
RTMPOSE_NAME = "rtmpose-tiny_simcc-coco_pt-aic-coco_420e-256x192-e613ba3f_20230127.pth"


class SliceStop(RuntimeError):
    """A precise BLOCKED / UNSUPPORTED / INVALID outcome.  ``code`` is machine-readable."""

    def __init__(self, code, detail=""):
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code, self.detail = code, detail


# ------------------------------------------------------------------------------------------------------ helpers ---
def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_code(path) -> str:
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def canon(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def dump(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, sort_keys=True, default=_jdefault), encoding="utf-8")
    return sha256_file(path)


def _jdefault(o):
    try:
        import numpy as np
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer, np.bool_)):
            return o.item()
    except ImportError:
        pass
    if isinstance(o, (set, tuple)):
        return sorted(o) if isinstance(o, set) else list(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_dir(path) -> Path:
    p = Path(path)
    if p.exists():
        raise SliceStop("REFUSED_EXISTING_OUTPUT", f"{p} already exists; outputs are never overwritten")
    p.mkdir(parents=True)
    return p


def _scripts_on_path():
    for p in (str(PROJECT_ROOT / "src"), str(SCRIPTS)):
        if p not in sys.path:
            sys.path.insert(0, p)


def frozen_code_report(expected=None):
    exp = expected if expected is not None else (load(FROZEN_CODE_FILE) if FROZEN_CODE_FILE.exists() else {})
    rep = {}
    for rel in FROZEN_CODE:
        p = PROJECT_ROOT / rel
        rep[rel] = {"sha256_lf": sha256_code(p) if p.exists() else None, "expected": exp.get(rel)}
        rep[rel]["ok"] = rep[rel]["sha256_lf"] is not None and rep[rel]["sha256_lf"] == rep[rel]["expected"]
    return rep


def require_frozen_code():
    rep = frozen_code_report()
    bad = sorted(k for k, v in rep.items() if not v["ok"])
    if bad:
        raise SliceStop("BLOCKED_FROZEN_ALGORITHM_CHANGED", f"frozen reused files changed or missing: {bad}")
    return rep


def rig_identity():
    rep = {}
    for k, (rel, want) in RIG_FILES.items():
        p = PROJECT_ROOT / rel
        got = sha256_file(p) if p.exists() else None
        rep[k] = {"path": rel, "sha256": got, "expected": want, "ok": got == want}
    if not all(v["ok"] for v in rep.values()):
        raise SliceStop("BLOCKED_RIG_IDENTITY", json.dumps({k: v["ok"] for k, v in rep.items()}))
    return rep


def find_blender(override=None):
    exe = override or os.environ.get("ANIMCV_BLENDER")
    if not exe:
        exe = shutil.which("blender")
    if not exe or not Path(exe).exists():
        raise SliceStop("BLOCKED_BLENDER_UNAVAILABLE", f"Blender executable not found ({exe!r}); pass --blender or set ANIMCV_BLENDER")
    return str(exe)


def run_blender(blender, mode, log, **kw):
    args = [blender, "-b", "-t", "2", "--python-exit-code", "1", "-P", str(STAGE), "--", mode]
    for k, v in kw.items():
        args += ["--" + k.replace("_", "-"), str(v)]
    r = subprocess.run(args, cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    Path(log).write_text(" ".join(args) + "\n" + r.stdout + "\n" + r.stderr, encoding="utf-8")
    if r.returncode != 0:
        raise SliceStop("BLOCKED_BLENDER_STAGE_FAILED", f"{mode} exited {r.returncode}; see {log}")
    return r


# --------------------------------------------------------------------------------------------- A. source intake ---
def decode_inspect(clip):
    import cv2
    import numpy as np
    cap = cv2.VideoCapture(str(clip))
    if not cap.isOpened():
        raise SliceStop("UNSUPPORTED_SOURCE_UNDECODABLE", str(clip))
    fps = cap.get(cv2.CAP_PROP_FPS)
    digests, pos, sizes = [], [], set()
    while True:
        ok, im = cap.read()
        if not ok:
            break
        pos.append(cap.get(cv2.CAP_PROP_POS_MSEC))
        sizes.add((im.shape[1], im.shape[0]))
        digests.append(hashlib.sha256(im.tobytes()).hexdigest())
    cap.release()
    pm = np.asarray(pos, float)
    dev = float(np.max(np.abs(np.diff(pm) - 1000.0 / fps))) if len(pm) > 2 and fps and np.isfinite(fps) and fps > 0 else None
    return {"fps": fps, "frames": len(digests), "sizes": sorted(sizes), "container_pos_msec": pos, "cadence_max_dev_ms": dev,
            "decoded_BGR_uint8_sha256": digests}


def intake(clip, expected_sha256, declared_fps, segment, source_record):
    """Authenticate the source and its timebase.  Returns the immutable input manifest body (no outcome data)."""
    clip = Path(clip)
    if not clip.exists():
        raise SliceStop("DATA_BLOCKED_SOURCE_MISSING", str(clip))
    got = sha256_file(clip)
    if got != expected_sha256:
        raise SliceStop("BLOCKED_SOURCE_IDENTITY", f"clip SHA-256 {got} != declared {expected_sha256}")
    for k in ("title", "license", "author", "source_uri"):
        if not source_record.get(k):
            raise SliceStop("BLOCKED_SOURCE_PROVENANCE_INCOMPLETE", f"source record lacks {k!r}")
    meta = decode_inspect(clip)
    fps = meta["fps"]
    if not meta["frames"] or not fps or not math.isfinite(fps) or fps <= 0:
        raise SliceStop("UNSUPPORTED_NO_CADENCE", "container reports no usable fps")
    if abs(float(declared_fps) - fps) > 1e-9:
        raise SliceStop("UNSUPPORTED_FPS_MISMATCH", f"declared {declared_fps} fps != container {fps} fps")
    if abs(fps - SUPPORTED_FPS) > 1e-9:
        raise SliceStop("UNSUPPORTED_CADENCE", f"{fps} fps; this slice supports only {SUPPORTED_FPS} fps (no resampling)")
    if meta["cadence_max_dev_ms"] is None or meta["cadence_max_dev_ms"] > CADENCE_TOL_MS:
        raise SliceStop("UNSUPPORTED_TIMESTAMPS_INCONSISTENT_OR_VFR",
                        f"container presentation step deviates {meta['cadence_max_dev_ms']} ms from 1000/fps (limit {CADENCE_TOL_MS} ms)")
    if len(meta["sizes"]) != 1:
        raise SliceStop("UNSUPPORTED_FRAME_SIZE_CHANGES", str(meta["sizes"]))
    lo, hi = int(segment[0]), int(segment[1])
    if not (0 <= lo <= hi < meta["frames"]):
        raise SliceStop("UNSUPPORTED_SEGMENT", f"[{lo}, {hi}] outside decoded frames 0..{meta['frames'] - 1}")
    return {"clip_path": str(clip), "clip_sha256": got, "source_record": source_record, "declared_fps": float(declared_fps),
            "container_fps": fps, "decoded_frames": meta["frames"], "frame_size": list(meta["sizes"][0]),
            "cadence_max_dev_ms": meta["cadence_max_dev_ms"], "cadence_rule_ms": CADENCE_TOL_MS,
            "decoded_BGR_uint8_sha256": meta["decoded_BGR_uint8_sha256"], "segment_source_frames": [lo, hi],
            "time_rule": "source time = source frame index / container fps (validated by container presentation timestamps)",
            "source_timestamps_s": [i / fps for i in range(lo, hi + 1)],
            "container_pos_msec_segment": meta["container_pos_msec"][lo:hi + 1],
            "video_subject_identity": "UNKNOWN (GT-free; tracked-subject continuity not verified by this slice)"}


# ----------------------------------------------------------------------------------------------- perception ---
def check_perception_assets(assets):
    """assets: {"frozen_code_dir", "weights_dir", "checkpoint", "gate_report", "lock"} -> identity report or SliceStop."""
    paths = {"checkpoint": assets.get("checkpoint"), "rtmdet": Path(assets.get("weights_dir") or "") / RTMDET_NAME,
             "rtmpose": Path(assets.get("weights_dir") or "") / RTMPOSE_NAME, "gate_report": assets.get("gate_report"),
             "lock": assets.get("lock"),
             "diagnostic_script": Path(assets.get("frozen_code_dir") or "") / "scripts" / "w102_video_diagnostic.py"}
    rep = {}
    for k, p in paths.items():
        if not p or not Path(p).exists():
            code = "DATA_BLOCKED_MISSING_CHECKPOINT" if k == "checkpoint" else "DATA_BLOCKED_MISSING_PERCEPTION_ASSET"
            raise SliceStop(code, f"{k}: {p}")
        got = sha256_file(p)
        rep[k] = {"path": str(p), "sha256": got, "expected": PERCEPTION_EXPECTED[k], "ok": got == PERCEPTION_EXPECTED[k]}
        if not rep[k]["ok"]:
            raise SliceStop("BLOCKED_FROZEN_PERCEPTION_IDENTITY", f"{k} SHA-256 {got} != frozen {PERCEPTION_EXPECTED[k]}")
    return rep


def run_live_perception(run, clip, inspection, assets, python_exe):
    out = run / "perception" / "m1s0"
    out.parent.mkdir(parents=True, exist_ok=True)
    args = [python_exe, "-X", "utf8", str(LAUNCHER), "--code-dir", assets["frozen_code_dir"], "--scratch-dir", str(run / "perception" / "scratch"),
            "w102_video_diagnostic.py", "run", "--clip", str(clip), "--inspection", str(inspection), "--weights", assets["weights_dir"],
            "--checkpoint", assets["checkpoint"], "--gate-report", assets["gate_report"], "--lock", assets["lock"], "--out-dir", str(out)]
    r = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace")
    (run / "perception" / "m1s0.log").write_text(" ".join(args) + "\n" + r.stdout + "\n" + r.stderr, encoding="utf-8")
    if r.returncode != 0 or not (out / "observations_predictions.npz").exists():
        raise SliceStop("BLOCKED_PERCEPTION_FAILED", f"frozen M1+S0 exited {r.returncode}; see perception/m1s0.log")
    return out / "observations_predictions.npz", out / "report.json"


def perception_frames(npz_path, segment):
    import numpy as np
    z = np.load(npz_path)
    lo, hi = segment
    need = ("landmarks", "input_2d", "input_valid", "predictions", "frame_index", "timestamp")
    if any(k not in z.files for k in need):
        raise SliceStop("DATA_BLOCKED_MISSING_OBSERVATION", f"observation file lacks {[k for k in need if k not in z.files]}")
    if hi >= len(z["predictions"]):
        raise SliceStop("DATA_BLOCKED_MISSING_OBSERVATION", "segment exceeds observed frames")
    present = np.isfinite(z["landmarks"][:, :, 0]).any(1)
    return z, present


# --------------------------------------------------------------------------------------------- rig adapter ---
def frozen_modules():
    _scripts_on_path()
    import w116_semantic_pin_base as S
    import w116_semantic_pin_contract as C
    import w118_transition_edit_base as XB
    import w121_independent_transfer as W121
    return S, C, XB, W121


def source_to_rig(z, present, segment, fps):
    """Frozen W105 GT-free S0-only FIXED-slot adapter (W121 ``fixed_slot`` + W116 value extraction), unchanged.
    Returns (cases, values, unsupported) where an unsupported frame is never filled."""
    S, C, XB, W121 = frozen_modules()
    rc = W121.rig_ctx()
    cases, unsupported = [], []
    for i in range(segment[0], segment[1] + 1):
        if not present[i]:
            unsupported.append({"source_frame": i, "reason": "NO_GT_FREE_SUBJECT_OBSERVATION"})
            continue
        fr = W121.fixed_slot(z["predictions"][i], rc)
        if fr is None or "FIXED" not in fr["modes"]:
            unsupported.append({"source_frame": i, "reason": "RIG_CONTRACT_FAILED (torso frame or left-arm chain unavailable)"})
            continue
        fr.update(frame_index=i, time_s=i / fps, identity="UNKNOWN (GT-free; tracked-subject continuity unverified)")
        cases.append(fr)
    values = W121.state_values_from_cases(cases) if not unsupported else None
    return cases, values, unsupported, rc


def driven_channel_report(rest_json):
    S, C, XB, W121 = frozen_modules()
    bones = sorted(load(rest_json)["bones"])
    driven = [c for c in C.REAL_CH if c != S.OBJECT]
    return {"driven_channels": {"__object__": "Armature Object rotation (rest rotation x S0 W93 torso-frame rotation); location is NOT keyed",
                                **{b: "pose-bone rotation_quaternion (W105 FIXED slot)" for b in driven}},
            "not_driven_bones_at_rest": [b for b in bones if b not in driven],
            "limitations": ["PARTIAL rig: right arm (clavicle_r, upperarm_r, lowerarm_r, hand_r) stays at REST",
                            "spine_02..05, neck, head, clavicles, hands, fingers, feet stay at rest",
                            "no root translation and no foot contact: the character rotates in place",
                            "left-arm and body quality are GT-free S0 hypotheses with no 3D accuracy evidence"]}


# ------------------------------------------------------------------------------------------------- run state ---
def run_paths(run):
    run = Path(run)
    return {"run": run / "run.json", "manifest": run / "manifest" / "input_manifest.json", "sidecar": run / "baseline" / "action_sidecar.json",
            "refs": run / "lineage" / "refs.json", "log": run / "lineage" / "log.jsonl", "blends": run / "lineage" / "blends.json",
            "bindings": run / "baseline" / "bindings.json"}


def load_run(run):
    run = Path(run)
    P = run_paths(run)
    if not P["run"].exists():
        raise SliceStop("BLOCKED_NOT_A_RUN", str(run))
    R = load(P["run"])
    if sha256_file(P["manifest"]) != R["input_manifest_sha256"]:
        raise SliceStop("BLOCKED_STALE_MANIFEST", "input manifest bytes differ from the run record")
    man = load(P["manifest"])
    clip = Path(man["intake"]["clip_path"])
    if clip.exists() and sha256_file(clip) != man["intake"]["clip_sha256"]:
        raise SliceStop("BLOCKED_SOURCE_IDENTITY", "source clip bytes changed since build")
    for name, rec in R["immutable_files"].items():
        if sha256_file(run / name) != rec:
            raise SliceStop("BLOCKED_IMMUTABLE_OUTPUT_CHANGED", name)
    return R, man


class PersistentStore:
    """Persists the frozen W116 ``Store`` (content-addressed read-only states; refs move only on atomic commit)."""

    def __init__(self, run):
        S, *_ = frozen_modules()
        self.P = run_paths(run)
        self.store = S.Store(Path(run) / "lineage")
        if self.P["refs"].exists():
            self.store.refs = load(self.P["refs"])

    def save(self, new_log):
        dump(self.P["refs"], self.store.refs)
        with open(self.P["log"], "a", encoding="utf-8") as f:
            for rec in new_log:
                f.write(json.dumps(dict(rec, at=now()), sort_keys=True) + "\n")


def blend_registry(run):
    p = run_paths(run)["blends"]
    return load(p) if p.exists() else {}


def register_blend(run, state_id, rec):
    """state id -> its .blend record.  An existing entry is never overwritten (an accepted copy is recorded alongside)."""
    reg = blend_registry(run)
    if state_id in reg:
        reg[state_id].setdefault("additional_records", []).append(rec)
    else:
        reg[state_id] = rec
    dump(run_paths(run)["blends"], reg)


def readback_by_source(rb, sidecar):
    """Blender readback frames keyed by SOURCE frame index (Blender frame -> source frame from the sidecar mapping)."""
    m = {r["action_frame"]: r["source_frame"] for r in sidecar["frame_mapping"]}
    out = {}
    for fr in rb["frames"]:
        f = fr["blender_frame"]
        if f not in m:
            raise SliceStop("INVALID_TIME_MAPPING", f"Blender frame {f} has no source frame")
        out[m[f]] = dict(fr, frame_index=m[f])
    return out


def build_state_blend(run, blender, state, name, folder, prov, rest_json, rig_fbx, sidecar=None):
    """Key one state into a NEW .blend and read it back in a fresh Blender process."""
    S, C, *_ = frozen_modules()
    d = Path(run) / folder
    d.mkdir(parents=True, exist_ok=True)
    states = d / f"{name}_state_for_blender.json"
    C.states_file(states, {name: state}, "w105")
    pv = d / f"{name}_provenance.json"
    dump(pv, prov)
    blend = d / f"{name}.blend"
    run_blender(blender, "build_blend", d / f"{name}_build.log", rig=rig_fbx, rest=rest_json, states=states, state_name=name,
                action_name=f"AnimCV_EXP {name} {state['state_id'][:12]}", out_blend=blend, provenance=pv, report=d / f"{name}_build.json",
                fps=SUPPORTED_FPS)
    run_blender(blender, "readback", d / f"{name}_readback.log", blend=blend, report=d / f"{name}_readback.json",
                subframes=",".join(str(s) for s in SUBFRAMES))
    b, rb = load(d / f"{name}_build.json"), load(d / f"{name}_readback.json")
    if b["blend_sha256"] != sha256_file(blend) or rb["blend_sha256"] != b["blend_sha256"]:
        raise SliceStop("INVALID_BLEND_IDENTITY", name)
    rec = {"state_id": state["state_id"], "blend": str(blend.relative_to(run)), "blend_sha256": b["blend_sha256"],
           "build_report": str((d / f"{name}_build.json").relative_to(run)), "readback": str((d / f"{name}_readback.json").relative_to(run)),
           "value_digest": rb["value_digest"], "readback_equals_in_process": [
               {k: f[k] for k in ("world", "matrix_world", "stored_wxyz")} for f in rb["frames"]]
           == [{k: f[k] for k in ("world", "matrix_world", "stored_wxyz")} for f in b["in_process_frames"]]}
    if not rec["readback_equals_in_process"] or rb["value_digest"] != b["value_digest"]:
        raise SliceStop("INVALID_BLEND_READBACK", f"{name}: reopened .blend differs from the in-process evaluation")
    return rec, b, rb


# -------------------------------------------------------------------------------------------------- build ---
def cmd_build(a):
    blender = find_blender(a.blender)
    rig_id_rep = rig_identity()
    code_rep = require_frozen_code()
    source_record = load(a.source_record)
    if a.replay_observations is None:
        assets = load(a.perception_assets) if a.perception_assets else None
        if assets is None:
            raise SliceStop("DATA_BLOCKED_MISSING_PERCEPTION_ASSET", "--perception-assets is required for live perception")
        perc_rep = check_perception_assets(assets)
    run = new_dir(Path(a.out).resolve())
    a.clip = str(Path(a.clip).resolve())
    try:
        return _build(a, run, blender, rig_id_rep, code_rep, source_record, perc_rep if a.replay_observations is None else None,
                      assets if a.replay_observations is None else None)
    except SliceStop as e:
        dump(run / "STOPPED.json", {"code": e.code, "detail": e.detail, "at": now(), "stage": "build"})
        raise


def _build(a, run, blender, rig_id_rep, code_rep, source_record, perc_rep, assets):
    import numpy as np
    S, C, XB, W121 = frozen_modules()
    man_dir = run / "manifest"
    man_dir.mkdir()
    ik = intake(a.clip, a.clip_sha256, a.declared_fps, a.segment, source_record)
    insp = man_dir / "inspection.json"   # W102 diagnostic input format (identity of clip + decoded frames)
    dump(insp, {"clip_sha256": ik["clip_sha256"], "frames": ik["decoded_frames"], "fps": ik["container_fps"], "size": ik["frame_size"],
                "timestamp": [i / ik["container_fps"] for i in range(ik["decoded_frames"])],
                "decoded_BGR_uint8_sha256": ik["decoded_BGR_uint8_sha256"], "source": ik["source_record"]["title"],
                "GT_available": False, "unseen": False})
    if a.replay_observations is None:
        npz, prep = run_live_perception(run, a.clip, insp, assets, a.python or sys.executable)
        perception = {"mode": "LIVE_FROZEN_M1_S0", "label": "live video end-to-end (frozen W102 M1 batch-1 + frozen S0)",
                      "assets": perc_rep, "observation_prediction_sha256": sha256_file(npz), "report_sha256": sha256_file(prep),
                      "report_summary": {k: load(prep).get(k) for k in ("frames", "fps", "environment_fingerprint", "missing_subject_frames",
                                                                        "observation_prediction_sha256", "clip_sha256", "no_GT")}}
        if load(prep)["clip_sha256"] != ik["clip_sha256"]:
            raise SliceStop("INVALID_PERCEPTION_IDENTITY", "perception report bound to another clip")
    else:
        npz = Path(a.replay_observations)
        if not npz.exists():
            raise SliceStop("DATA_BLOCKED_MISSING_OBSERVATION", str(npz))
        if sha256_file(npz) != a.replay_sha256:
            raise SliceStop("BLOCKED_OBSERVATION_IDENTITY", "replay observation SHA-256 mismatch")
        perception = {"mode": "OFFLINE_REPLAY", "label": "OFFLINE REPLAY of stored arrays - NOT live-video end-to-end",
                      "observation_prediction_sha256": sha256_file(npz), "replay_path": str(npz)}
    z, present = perception_frames(npz, ik["segment_source_frames"])
    seg = ik["segment_source_frames"]
    if len(z["predictions"]) != ik["decoded_frames"]:
        raise SliceStop("INVALID_PERCEPTION_IDENTITY", "observation frame count differs from the decoded clip")
    obs_t = np.asarray(z["timestamp"], float)
    if not np.array_equal(obs_t, np.arange(len(obs_t)) / ik["container_fps"]):
        raise SliceStop("INVALID_TIME_MAPPING", "observation timestamps are not index / fps")
    cases, values, unsupported, rc = source_to_rig(z, present, seg, ik["container_fps"])
    perception["segment_subject_present_frames"] = int(present[seg[0]:seg[1] + 1].sum())
    perception["segment_frames"] = seg[1] - seg[0] + 1
    perception["unsupported_frames"] = unsupported
    if unsupported:
        dump(run / "baseline_unsupported.json", {"unsupported_frames": unsupported})
        raise SliceStop("UNSUPPORTED_FRAMES", f"{len(unsupported)} segment frame(s) have no supported rig pose; choose a supported "
                        "segment (frames are never fabricated or interpolated)")
    frames = list(range(seg[0], seg[1] + 1))
    rest_json = PROJECT_ROOT / RIG_FILES["rest"][0]
    rig_fbx = PROJECT_ROOT / RIG_FILES["fbx"][0]
    manifest = {"schema": SLICE_ID + ":input_manifest", "created": now(), "intake": ik, "perception": perception,
                "rig": {"id": "BaseRig (W105 BaseRig_Skinned, known output rig)", "files": rig_id_rep,
                        "adapter": "W105 GT-free S0-only FIXED slot (scripts/w121_independent_transfer.fixed_slot, unchanged)",
                        "rest_snapshot_digest": rc["rest"].digest(), "canonical_to_armature_xyzw": list(rc["alignment"])},
                "frozen_code": code_rep, "supported": {"fps": SUPPORTED_FPS, "joints": list(SLICE_JOINTS),
                                                       "authorized_channels": list(SLICE_AUTHORIZED), "target_kinds": list(TARGET_KINDS)},
                "repo_head": _git_head()}
    man_sha = dump(run_paths(run)["manifest"], manifest)
    os.chmod(run_paths(run)["manifest"], 0o444)
    dump(run / "rig" / "video_cases.json", {"rest_snapshot_digest": rc["rest"].digest(), "frames": cases})
    src = {"kind": "LIVE_VIDEO_FROZEN_M1_S0_W105_FIXED" if perception["mode"] == "LIVE_FROZEN_M1_S0" else "OFFLINE_REPLAY_W105_FIXED",
           "input_manifest_sha256": man_sha, "clip_sha256": ik["clip_sha256"], "observation_prediction_sha256": perception["observation_prediction_sha256"],
           "fps_source": ik["container_fps"], "segment_source_frames": seg, "rig_identity": "W105 BaseRig_Skinned (known output rig)",
           "video_subject_identity": ik["video_subject_identity"]}
    v0 = S.make_state(None, "BaseRig", frames, C.REAL_CH, values, {"kind": "ROOT_COPY_OF_SOURCE_HYPOTHESIS"}, src)
    mapping = [{"source_frame": k, "source_time_s": k / ik["container_fps"], "action_frame": k - seg[0] + 1,
                "action_time_s_at_scene_fps": (k - seg[0]) / SUPPORTED_FPS} for k in frames]
    prov = {"slice": SLICE_ID, "status": "EXPERIMENTAL", "input_manifest_sha256": man_sha, "state_id": v0["state_id"],
            "role": "baseline (unedited source hypothesis)", "segment_source_frames": seg, "frame_rule": "Blender frame = source frame - first + 1",
            "scene_fps": SUPPORTED_FPS}
    rec, b, rb = build_state_blend(run, a.blender_exe, v0, "baseline", "baseline", prov, rest_json, rig_fbx)
    # Python FK model parity + bindings from the independent readback (W116 rules, unchanged)
    rig = C.rig_from(rest_json, b["rig"])
    sidecar_tmp = {"frame_mapping": mapping}
    ev = readback_by_source(rb, sidecar_tmp)
    worst = 0.0
    for i, k in enumerate(frames):
        fk = rig.fk(S.frame_vals(v0, i))
        for bn in ev[k]["world"]:
            for p in ("head", "tail"):
                worst = max(worst, float(np.linalg.norm(rig.world(fk, bn, p) - np.array(ev[k]["world"][bn][p]))))
    if worst > FK_PARITY_TOL_M:
        raise SliceStop("BLOCKED_PARITY_LOST", f"Python FK model vs Blender {worst} m > {FK_PARITY_TOL_M}")
    evf = [ev[k] for k in frames]
    ok, failed = {}, {}
    for j, cands in C.REAL_BINDING_CANDIDATES.items():
        r = S.verify_binding(j, cands, "BaseRig", b["rig"]["rest_snapshot_digest"], rb["rest"], evf, v0["channels"], rig)
        (ok.__setitem__(r["binding_id"], r) if r["status"] == "BOUND" else failed.__setitem__(j, r))
    dump(run_paths(run)["bindings"], {"bound": ok, "failed": failed, "declared_candidates": C.REAL_BINDING_CANDIDATES})
    fc = rb["fcurve_census"]
    sidecar = {"schema": SLICE_ID + ":action_sidecar", "status": "EXPERIMENTAL", "input_manifest_sha256": man_sha,
               "baseline_state_id": v0["state_id"], "baseline_blend": rec["blend"], "baseline_blend_sha256": rec["blend_sha256"],
               "action_name": rb["action_name"], "action_value_digest": rb["value_digest"], "perception_mode": perception["mode"],
               "perception_label": perception["label"], "source": {k: ik[k] for k in ("clip_sha256", "source_record", "container_fps",
                                                                                     "segment_source_frames", "video_subject_identity")},
               "frame_mapping": mapping, "scene": rb["scene"], "action_duration": {"action_frames": len(frames),
                                                                                  "seconds_at_scene_fps": len(frames) / SUPPORTED_FPS,
                                                                                  "source_seconds": len(frames) / ik["container_fps"],
                                                                                  "playback_factor": SUPPORTED_FPS / ik["container_fps"]},
               "fcurves": {"count": fc["n_fcurves"], "interpolation_census_curves": fc["interpolation_census_curves"],
                           "key_frame_range": [fc["key_frame_min"], fc["key_frame_max"]],
                           "subframe_semantics": "CONSTANT (held) keys - frozen W116/historical renderer keying rule"},
               **driven_channel_report(rest_json),
               "unavailable_frames": [], "bindings": {"bound": {r["semantic_joint"]: bid for bid, r in ok.items()},
                                                      "failed": {j: r["status"] for j, r in failed.items()}},
               "fk_model_max_error_m": worst, "rig_info": b["rig"]}
    sc_sha = dump(run_paths(run)["sidecar"], sidecar)
    ps = PersistentStore(run)
    ps.store.put(v0)
    ps.store.init_ref("head", v0["state_id"])
    ps.save(ps.store.log)
    register_blend(run, v0["state_id"], dict(rec, role="baseline"))
    immutable = {str(Path(rec["blend"])): rec["blend_sha256"], "baseline/action_sidecar.json": sc_sha,
                 "manifest/input_manifest.json": man_sha, "baseline/baseline_readback.json": sha256_file(run / rec["readback"])}
    dump(run_paths(run)["run"], {"schema": SLICE_ID + ":run", "created": now(), "input_manifest_sha256": man_sha,
                                 "immutable_files": immutable, "baseline_state_id": v0["state_id"], "blender": a.blender_exe,
                                 "rig_fbx": str(rig_fbx), "rest_json": str(rest_json)})
    print(json.dumps({"run": str(run), "baseline_state_id": v0["state_id"], "baseline_blend": rec["blend"],
                      "action_value_digest": rb["value_digest"], "perception": perception["label"],
                      "bindings_bound": sorted(sidecar["bindings"]["bound"])}, indent=1))
    return 0


def _git_head():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True).stdout.strip() or None
    except OSError:
        return None


# ---------------------------------------------------------------------------------------------- C. propose ---
def _finite3(v):
    return isinstance(v, (list, tuple)) and len(v) == 3 and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                                                                  and math.isfinite(x) for x in v)


def precheck_request(req, head_id, head_frames, bindings_ok, bindings_failed):
    """Slice-level request checks BEFORE the frozen W116 contract.  Returns (errs, binding_id or None, frames, capture)."""
    errs = []
    if req.get("schema") != "animcv_exp_world_pin_request_v1":
        errs.append(("CONTRACT_VIOLATION", "REQUEST_SCHEMA_UNKNOWN"))
    if req.get("type") != "WORLD_POSITION_PIN":
        errs.append(("CONTRACT_VIOLATION", "TYPE_NOT_SUPPORTED_IN_SLICE"))
    if req.get("parent_state_id") != head_id:
        errs.append(("CONTRACT_VIOLATION", "STALE_PARENT"))
    fr = req.get("frames")
    frames = None
    if not (isinstance(fr, list) and len(fr) == 2 and all(isinstance(x, int) and not isinstance(x, bool) for x in fr) and fr[0] <= fr[1]):
        errs.append(("CONTRACT_VIOLATION", "INVALID_FRAME_INTERVAL"))
    else:
        frames = list(range(fr[0], fr[1] + 1))
        if not set(frames) <= set(head_frames):
            errs.append(("CONTRACT_VIOLATION", "FRAME_NOT_IN_STATE"))
    t = req.get("target") or {}
    capture = t.get("capture_frame")
    if t.get("kind") not in TARGET_KINDS:
        errs.append(("CONTRACT_VIOLATION", "TARGET_KIND_UNSUPPORTED"))
    if not (isinstance(capture, int) and not isinstance(capture, bool)):
        errs.append(("CONTRACT_VIOLATION", "INVALID_NUMERIC_TARGET"))
    elif capture not in set(head_frames):
        errs.append(("CONTRACT_VIOLATION", "TARGET_CAPTURE_FRAME_NOT_IN_STATE"))
    if t.get("kind") == "CAPTURED_PLUS_DECLARED_OFFSET" and not _finite3(t.get("offset_m")):
        errs.append(("CONTRACT_VIOLATION", "INVALID_NUMERIC_TARGET"))
    if t.get("kind") == "CAPTURED_FROM_EVALUATED_STATE" and "offset_m" in t:
        errs.append(("CONTRACT_VIOLATION", "INVALID_NUMERIC_TARGET"))
    joint = req.get("semantic_joint")
    by_joint = {r["semantic_joint"]: bid for bid, r in bindings_ok.items()}
    bid = req.get("binding_id") or by_joint.get(joint)
    if req.get("binding_id") is None and joint not in by_joint:
        if joint in bindings_failed:
            errs.append(("UNSUPPORTED_BINDING", bindings_failed[joint]["status"]))
        else:
            errs.append(("UNSUPPORTED_BINDING", "BINDING_UNDECLARED_JOINT"))
        bid = None
    elif joint not in SLICE_JOINTS:
        errs.append(("UNSUPPORTED_BINDING", "JOINT_NOT_SUPPORTED_BY_SLICE"))
    auth = req.get("authorized_channels")
    if not isinstance(auth, list) or sorted(auth) != list(SLICE_AUTHORIZED):
        errs.append(("CONTRACT_VIOLATION", "UNAUTHORIZED_DOF_FOR_SLICE"))
    if bid is not None and bid not in bindings_ok:
        errs.append(("UNSUPPORTED_BINDING", "BINDING_UNVERIFIED"))
        bid = None
    return errs, bid, frames, capture


def propose_core(req, head, head_eval, rig, rest, scale, bindings_ok, bindings_failed, root_bones, keyer):
    """Rig-agnostic proposal.  head_eval: {source frame -> Blender-evaluated frame}.  keyer(candidate_state) -> eval dict.
    Returns the full proposal record (decision is the GATE decision; nothing is committed here)."""
    import numpy as np
    S, C, XB, W121 = frozen_modules()
    rec = {"request_id": req.get("request_id"), "parent_state_id": head["state_id"], "slice": SLICE_ID,
           "automatic_out_of_pin_budget_frames": 0, "mode": "W116 analytic framewise resolve on pin frames only (F0, zero budget)"}
    errs, bid, pin_frames, capture = precheck_request(req, head["state_id"], head["frames"], bindings_ok, bindings_failed)
    pin = None
    auth_ok = isinstance(req.get("authorized_channels"), list) and all(isinstance(x, str) for x in req["authorized_channels"])
    if auth_ok and bid is not None and pin_frames is not None and not any(r in ("INVALID_NUMERIC_TARGET", "TARGET_KIND_UNSUPPORTED",
                                                                    "TARGET_CAPTURE_FRAME_NOT_IN_STATE") for _, r in errs):
        b = bindings_ok[bid]
        cap = {(head["state_id"], k, bid): head_eval[k]["world"][b["bone"]][b["point"]] for k in head["frames"]}
        t = req["target"]
        src = {"kind": t["kind"], "state_id": req.get("parent_state_id"), "frame_index": capture, "binding_id": bid,
               "evaluated_by": "experimental_action_stage readback of the head .blend"}
        if t["kind"] == "CAPTURED_PLUS_DECLARED_OFFSET":
            src["offset_m"] = [float(x) for x in t["offset_m"]]
        tgt = S.captured_target(cap, src)
        if tgt is None:   # stale parent: the target cannot be re-derived from the head; use the head capture for the record only
            tgt = np.array(cap[(head["state_id"], capture, bid)], float) + np.array(src.get("offset_m", [0, 0, 0]), float)
        pin = S.make_pin("p0", req.get("semantic_joint"), bid, tgt.tolist(), src, pin_frames, req["authorized_channels"],
                         req.get("parent_state_id"), frame=req.get("coordinate_frame"), unit=req.get("unit"))
        w = S.validate_request({"constraints": [pin], "parent_state_id": req.get("parent_state_id"), "edit": None},
                               {"bindings": bindings_ok, "failed_bindings": bindings_failed, "capture": cap, "parent": head,
                                "root_bones": root_bones})
        errs += [e for e in w if e not in errs]
        rec["constraint"] = pin
    rec["validation_errors"] = [list(e) for e in errs]
    if errs or pin is None:
        rec["decision"] = S.decide(errs or [("CONTRACT_VIOLATION", "REQUEST_INCOMPLETE")], [], [])
        rec["candidate"] = None
        rec["status"] = "GATE_REJECTED_NO_CANDIDATE"
        return rec, None, None
    b = bindings_ok[bid]
    auth = sorted(pin["authorized_channels"])
    statuses, info, patches = {}, {}, {}
    for k in pin_frames:
        vals = S.frame_vals(head, head["frames"].index(k))
        rp = [{"bone": b["bone"], "point": b["point"], "target": np.array(pin["target"]), "semantic_joint": pin["semantic_joint"]}]
        st, pv, inf = S.resolve_frame(rig, vals, rp, auth)
        statuses[k], info[k], patches[k] = st, inf, pv
    cand = C.patched_state(head, patches, {"kind": "WORLD_POSITION_PIN_EDIT", "request_id": req.get("request_id"),
                                           "constraint_digest": pin["constraint_digest"], "automatic_budget_frames": 0})
    X = keyer(cand)
    U = head_eval
    frames = head["frames"]
    dom = S.domain_points(rig, auth)
    allowed = {"vs_parent": auth, "vs_control": auth}
    evals = {}
    for k in pin_frames:
        pins_k = [{"bone": b["bone"], "point": b["point"], "target": np.array(pin["target"]), "semantic_joint": pin["semantic_joint"]}]
        evals[k] = S.evaluate_candidate(X[k], U[k], U[k], pins_k, allowed, dom, rest, scale, chain=ARM_CHAIN)
    fp = XB.derive_footprint(X, U, frames, pin_frames, dom)
    spec = XB.display_spec(fp, frames, pin_frames, pin_frames, pin_frames, cand["state_id"], head["state_id"], "PROPOSED",
                           req.get("request_id"), "F0_FRAMEWISE_ZERO_BUDGET", frame_axis="source frame index (Action frame = index - first + 1)")
    dc = XB.check_display(spec, X, U, frames, pin_frames, cand["state_id"])
    decision = XB.decide_x([], [statuses[k] for k in pin_frames], list(evals.values()), fp, pin_frames, pin_frames, auth, dc)
    rec.update(candidate=cand["state_id"], resolver={str(k): {"status": statuses[k], "info": info[k]} for k in pin_frames},
               evaluations={str(k): evals[k] for k in pin_frames}, footprint=fp, display=spec, display_check=dc, decision=decision,
               pin_frames=pin_frames, hard_pin_residual_max_m=max(e["pins"][0]["candidate_residual_m"] for e in evals.values()),
               control_residual_max_m=max(e["pins"][0]["control_residual_m"] for e in evals.values()),
               status="PROPOSED_GATE_PASS" if decision["decision"] == "ACCEPT" else "GATE_REJECTED")
    try:
        rec["quality_disclosure"] = {k: v for k, v in XB.quality(X, U, frames, pin_frames, ARM_CHAIN[0], ARM_CHAIN[1], auth).items()
                                     if k not in ("per_transition", "wrist_path", "elbow_path")}
    except Exception as e:  # disclosure only; never gates
        rec["quality_disclosure"] = {"error": repr(e)}
    return rec, cand, X


def subframe_report(cand_rb, head_rb, sidecar, pin, binding):
    """Subframe footprint and pin residual (CONSTANT keys: frame k + s holds frame k)."""
    import numpy as np
    m = {r["action_frame"]: r["source_frame"] for r in sidecar["frame_mapping"]}
    changed, resid = [], []
    for c, h in zip(cand_rb["subframes"], head_rb["subframes"]):
        if (c["blender_frame"], c["subframe"]) != (h["blender_frame"], h["subframe"]):
            raise SliceStop("INVALID_SUBFRAME_ALIGNMENT")
        k = m[c["blender_frame"]]
        if c["world"] != h["world"] or c["matrix_world"] != h["matrix_world"]:
            changed.append([k, c["subframe"]])
        if pin is not None and k in set(pin["frames"]):
            p = np.array(c["world"][binding["bone"]][binding["point"]])
            resid.append(float(np.linalg.norm(p - np.array(pin["target"]))))
    intervals = sorted({k for k, _ in changed})
    return {"changed_subframe_samples": len(changed), "changed_intervals_source_frames": intervals,
            "changed_intervals_within_pin_frames": set(intervals) <= set(pin["frames"]) if pin else not intervals,
            "pin_residual_at_subframes_max_m": max(resid) if resid else None, "subframe_offsets": list(SUBFRAMES)}


def cmd_propose(a):
    import numpy as np
    S, C, XB, W121 = frozen_modules()
    run = Path(a.run).resolve()
    R, man = load_run(run)
    require_frozen_code()
    req_bytes = Path(a.request).read_bytes()
    req = json.loads(req_bytes)
    rid = req.get("request_id")
    if not isinstance(rid, str) or not REQUEST_ID_RE.match(rid):
        raise SliceStop("INVALID_REQUEST_ID", repr(rid))
    ed = new_dir(run / "edits" / rid)
    (ed / "request.json").write_bytes(req_bytes)
    ps = PersistentStore(run)
    head_id = ps.store.refs["head"]
    head = ps.store.get(head_id)
    reg = blend_registry(run)
    sidecar = load(run_paths(run)["sidecar"])
    head_rb = load(run / reg[head_id]["readback"])
    head_eval = readback_by_source(head_rb, sidecar)
    bnd = load(run_paths(run)["bindings"])
    rig = C.rig_from(R["rest_json"], sidecar["rig_info"])
    scale = sidecar["rig_info"]["scale"][0]
    root_bones = [n for n in rig.bones if rig.is_root(n)]
    blender = R["blender"]
    store_rb = {}

    def keyer(cand):
        prov = {"slice": SLICE_ID, "status": "EXPERIMENTAL", "input_manifest_sha256": R["input_manifest_sha256"], "state_id": cand["state_id"],
                "parent_state_id": cand["parent_id"], "role": f"candidate for request {rid} (PROPOSED; not committed)",
                "request_sha256": hashlib.sha256(req_bytes).hexdigest(), "frame_rule": "Blender frame = source frame - first + 1"}
        rec, b, rb = build_state_blend(run, blender, cand, "candidate", f"edits/{rid}", prov, R["rest_json"], R["rig_fbx"])
        store_rb.update(rec=rec, rb=rb)
        return readback_by_source(rb, sidecar)

    rec, cand, X = propose_core(req, head, head_eval, rig, head_rb["rest"], scale, bnd["bound"], bnd["failed"], root_bones, keyer)
    rec["request_sha256"] = hashlib.sha256(req_bytes).hexdigest()
    rec["input_manifest_sha256"] = R["input_manifest_sha256"]
    if cand is not None:
        dump(ed / "candidate_state.json", cand)
        rec["candidate_blend"] = store_rb["rec"]
        b = bnd["bound"][rec["constraint"]["binding_id"]]
        sub = subframe_report(store_rb["rb"], head_rb, sidecar, rec["constraint"], b)
        rec["subframes"] = sub
        if rec["decision"]["decision"] == "ACCEPT" and not sub["changed_intervals_within_pin_frames"]:
            rec["decision"] = {"decision": "REJECT", "failure_class": "CONTRACT_VIOLATION", "reasons": ["SUBFRAME_CHANGE_OUTSIDE_PIN_FRAMES"]}
            rec["status"] = "GATE_REJECTED"
    dump(ed / "proposal.json", rec)
    draw_timeline(ed / "timeline.png", rec, sidecar, head["frames"])
    print(json.dumps({"request_id": rid, "status": rec["status"], "decision": rec["decision"], "candidate": rec.get("candidate"),
                      "hard_pin_residual_max_m": rec.get("hard_pin_residual_max_m"),
                      "changed_frames": (rec.get("footprint") or {}).get("frames_changed"),
                      "changed_bones": sorted((rec.get("footprint") or {}).get("channel_frames", {}))}, indent=1))
    return 0


def draw_timeline(path, rec, sidecar, frames):
    """User-visible footprint: requested hard-pin frames, permitted temporal domain, ACTUAL changed frames per bone (Blender
    readback), unchanged frames and the gate status.  Drawn only from the evaluator footprint record."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fps = sidecar["source"]["container_fps"]
    pin = set(rec.get("pin_frames") or [])
    if not pin and rec.get("constraint"):
        pin = set(rec["constraint"]["frames"])
    fp = rec.get("footprint") or {"frames_changed": [], "channel_frames": {}, "per_frame": {}}
    rejected = rec["decision"]["decision"] != "ACCEPT"
    status = rec.get("final_status") or rec["status"]
    rows = ["requested hard pin", "permitted domain (budget 0)", "ACTUAL changed (any)"] + [f"  bone {c}" for c in sorted(fp["channel_frames"])] \
        + ["unchanged"]
    fig, axes = plt.subplots(2, 1, figsize=(14, 2.4 + 0.84 * len(rows)))
    changed = set(fp["frames_changed"])
    span = sorted(pin | changed) or [frames[0]]
    zoom = (max(frames[0], span[0] - 15), min(frames[-1], span[-1] + 15))
    for ax, (lo, hi), tag in ((axes[0], (frames[0], frames[-1]), "whole Action"), (axes[1], zoom, "zoom")):
        _timeline_axis(ax, rows, lo, hi, frames, pin, changed, fp, rejected, fps, rec, status, tag)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _timeline_axis(ax, rows, lo, hi, frames, pin, changed, fp, rejected, fps, rec, status, tag):
    from matplotlib.patches import Rectangle

    def cells(row, ks, color, hatch=None):
        for k in ks:
            ax.add_patch(Rectangle((k - 0.5, row - 0.35), 1, 0.7, facecolor=color, edgecolor="none" if not hatch else "k", hatch=hatch, lw=0.2))
    hatch = "///" if rejected else None
    cells(0, pin, "#2a7fd4")
    cells(1, pin, "#9cc3ea")
    cells(2, changed, "#d1495b" if not rejected else "#bbbbbb", hatch)
    for i, c in enumerate(sorted(fp["channel_frames"])):
        cells(3 + i, fp["channel_frames"][c], "#edae49" if not rejected else "#cccccc", hatch)
    cells(len(rows) - 1, [k for k in frames if k not in changed], "#e6e6e6")
    for k in sorted(changed):
        if lo <= k <= hi:
            mm = fp["per_frame"][str(k)]["wrist_disp_m"] * 1e3
            ax.text(k, 2, f"{mm:.0f}", ha="center", va="center", fontsize=5 if tag != "zoom" else 7, rotation=90)
    if tag == "zoom":
        ax.set_xticks(range(lo, hi + 1, 2))
        ax.tick_params(axis="x", labelsize=6)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=8)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlim(lo - 0.5, hi + 0.5)
    ax.set_xlabel(f"source frame index (Action frame = index - {lo} + 1); source time = index / {fps:g} s  |  numbers = wrist mm vs parent")
    sec = ax.secondary_xaxis("top", functions=(lambda x: x / fps, lambda t: t * fps))
    sec.set_xlabel("source time (s)", fontsize=8)
    title = (f"[{tag}] {rec.get('request_id')}: {status} | decision {rec['decision']['decision']} {rec['decision'].get('reasons', [])} | "
             f"changed frames {len(changed)} | bones {sorted(fp['channel_frames'])}")
    if rejected:
        title += " | REJECTED: not applied to the Action"
    ax.set_title(title, fontsize=9)


# ----------------------------------------------------------------------------------------- commit / revert ---
def cmd_commit(a):
    S, C, XB, W121 = frozen_modules()
    run = Path(a.run).resolve()
    R, man = load_run(run)
    ed = run / "edits" / a.request_id
    prop = load(ed / "proposal.json")
    if (ed / "commit.json").exists():
        raise SliceStop("REFUSED_ALREADY_DECIDED", a.request_id)
    ps = PersistentStore(run)
    n0 = len(ps.store.log)
    head_before = ps.store.refs["head"]
    cand = load(ed / "candidate_state.json") if (ed / "candidate_state.json").exists() else None
    if cand is not None and not S.verify_state(cand):
        raise SliceStop("INVALID_CANDIDATE_TAMPERED", "candidate state id does not bind its content")
    guard = {k: sha256_file(run / k) for k in R["immutable_files"]}
    if a.decision == "accept":
        if prop["decision"]["decision"] != "ACCEPT" or cand is None:
            out = {"op": "commit", "decision": "REJECT", "reasons": ["GATE_REJECTED_CANNOT_ACCEPT"] + prop["decision"].get("reasons", []),
                   "head_before": head_before, "head_after": head_before}
            ps.store.log.append(out)
        else:
            out = ps.store.commit("head", prop["parent_state_id"], cand, prop["decision"])
    else:
        out = ps.store.commit("head", prop["parent_state_id"], cand, {"decision": "REJECT", "reasons": ["USER_REJECTED"]})
    ps.save(ps.store.log[n0:])
    head_after = ps.store.refs["head"]
    final = "ACCEPTED" if head_after != head_before and head_after == (cand or {}).get("state_id") else "REJECTED"
    res = {"request_id": a.request_id, "user_decision": a.decision.upper(), "store_record": out, "final_status": final,
           "head_before": head_before, "head_after": head_after,
           "immutable_files_unchanged": guard == {k: sha256_file(run / k) for k in R["immutable_files"]}}
    if final == "ACCEPTED":
        src_blend = run / prop["candidate_blend"]["blend"]
        n_prev = len(list((run / "accepted").glob("*.blend"))) if (run / "accepted").exists() else 0
        dst = run / "accepted" / f"accepted_{n_prev + 1:02d}_{head_after[:12]}.blend"
        if dst.exists():
            raise SliceStop("REFUSED_EXISTING_OUTPUT", str(dst))
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src_blend, dst)
        if sha256_file(dst) != prop["candidate_blend"]["blend_sha256"]:
            raise SliceStop("INVALID_BLEND_IDENTITY", "accepted copy differs from the gated candidate")
        register_blend(run, head_after, dict(prop["candidate_blend"], role="accepted", accepted_copy=str(dst.relative_to(run))))
        res["accepted_blend"] = str(dst.relative_to(run))
    res["display_final"] = dict(prop.get("display") or {}, status=final) if prop.get("display") else None
    if prop.get("display"):
        sidecar = load(run_paths(run)["sidecar"])
        reg = blend_registry(run)
        U = readback_by_source(load(run / reg[prop["parent_state_id"]]["readback"]), sidecar)
        X = readback_by_source(load(run / prop["candidate_blend"]["readback"]), sidecar)
        head = ps.store.get(prop["parent_state_id"])
        res["display_final_check"] = XB.check_display(res["display_final"], X, U, head["frames"], prop["pin_frames"],
                                                      prop["candidate"], decision="ACCEPT" if final == "ACCEPTED" else "REJECT")
    dump(ed / "commit.json", res)
    prop2 = dict(prop, final_status=final)
    draw_timeline(ed / "timeline_final.png", prop2, load(run_paths(run)["sidecar"]), ps.store.get(prop["parent_state_id"])["frames"])
    print(json.dumps({k: res[k] for k in ("request_id", "user_decision", "final_status", "head_before", "head_after",
                                         "immutable_files_unchanged")}, indent=1))
    return 0


def cmd_revert(a):
    S, C, XB, W121 = frozen_modules()
    run = Path(a.run).resolve()
    R, man = load_run(run)
    ps = PersistentStore(run)
    n0 = len(ps.store.log)
    head = ps.store.get(ps.store.refs["head"])
    if head["parent_id"] is None:
        raise SliceStop("NOTHING_TO_REVERT", "head is the baseline")
    st, rec = ps.store.revert("head")
    ps.save(ps.store.log[n0:])
    restored = ps.store.get(rec["restores"])
    reg = blend_registry(run)
    sidecar = load(run_paths(run)["sidecar"])
    tag = f"reverted_{st['state_id'][:12]}"
    prov = {"slice": SLICE_ID, "status": "EXPERIMENTAL", "input_manifest_sha256": R["input_manifest_sha256"], "state_id": st["state_id"],
            "parent_state_id": st["parent_id"], "role": f"REVERT of {head['state_id']} restoring {restored['state_id']}"}
    brec, b, rb = build_state_blend(run, R["blender"], st, tag, "reverted", prov, R["rest_json"], R["rig_fbx"])
    register_blend(run, st["state_id"], dict(brec, role="reverted"))
    rb0 = load(run / reg[restored["state_id"]]["readback"])
    keys = ("world", "matrix_world", "stored_wxyz")
    same_int = [{k: f[k] for k in keys} for f in rb["frames"]] == [{k: f[k] for k in keys} for f in rb0["frames"]]
    same_sub = [{k: f[k] for k in ("world", "matrix_world")} for f in rb["subframes"]] == \
               [{k: f[k] for k in ("world", "matrix_world")} for f in rb0["subframes"]]
    out = {"revert_record": rec, "reverted_state_id": st["state_id"], "restored_state_id": restored["state_id"],
           "values_equal_restored": canon(st["values"]) == canon(restored["values"]), "value_digest_equal": rb["value_digest"] == rb0["value_digest"],
           "blender_eval_integer_frames_bit_equal": same_int, "blender_eval_subframes_bit_equal": same_sub,
           "reverted_blend": brec, "lineage": ps.store.lineage(st["state_id"])}
    out["exact_undo"] = all(out[k] for k in ("values_equal_restored", "value_digest_equal", "blender_eval_integer_frames_bit_equal",
                                            "blender_eval_subframes_bit_equal"))
    dump(run / "reverted" / f"{tag}_report.json", out)
    print(json.dumps({k: out[k] for k in ("reverted_state_id", "restored_state_id", "exact_undo")}, indent=1))
    return 0


def cmd_status(a):
    run = Path(a.run).resolve()
    R, man = load_run(run)
    ps = PersistentStore(run)
    print(json.dumps({"head": ps.store.refs.get("head"), "lineage": ps.store.lineage(ps.store.refs["head"]),
                      "blends": blend_registry(run)}, indent=1, default=str))
    return 0


# --------------------------------------------------------------------------------------- D. review (A/B video) ---
def cmd_review(a):
    import cv2
    import numpy as np
    run = Path(a.run).resolve()
    R, man = load_run(run)
    ed = run / "edits" / a.request_id
    prop, com = load(ed / "proposal.json"), load(ed / "commit.json")
    if com["final_status"] != "ACCEPTED":
        raise SliceStop("REVIEW_REQUIRES_ACCEPTED_EDIT", a.request_id)
    out = new_dir(run / "review" / a.request_id)
    sidecar = load(run_paths(run)["sidecar"])
    reg = blend_registry(run)
    base_rb = load(run / reg[prop["parent_state_id"]]["readback"])
    mapping = sidecar["frame_mapping"]
    pin = set(prop["pin_frames"])
    tgt = prop["constraint"]["target"]
    allp = np.array([fr["world"][b][p] for fr in base_rb["frames"] for b in fr["world"] for p in ("head", "tail")])
    centre = ((allp.min(0) + allp.max(0)) / 2).tolist()
    ext = float(np.max(allp.max(0) - allp.min(0)))
    views = [{"name": "body", "centre": centre, "offset": [0.0, -6.0, 0.0], "ortho_scale": ext * 1.15},
             {"name": "arm", "centre": tgt, "offset": [0.0, -6.0, 0.0], "ortho_scale": 0.9}]
    size = 360
    jobs = []
    for r in mapping:
        mk = [{"name": "pin_target", "position": tgt, "color": (0.1, 0.85, 0.2, 1), "radius": 0.02}] if r["source_frame"] in pin else []
        jobs.append({"blender_frame": r["action_frame"], "markers": mk})
    dump(out / "jobs.json", {"size": size, "views": views, "frames": jobs})
    blends = {"baseline": run / reg[prop["parent_state_id"]]["blend"], "edited": run / com["accepted_blend"]}
    for name, bl in blends.items():
        run_blender(R["blender"], "render", out / f"{name}_render.log", blend=bl, jobs=out / "jobs.json", out_dir=out / f"frames_{name}")
    clip = man["intake"]["clip_path"]
    cap = cv2.VideoCapture(clip)
    src = {}
    i = 0
    want = {r["source_frame"] for r in mapping}
    while True:
        ok, im = cap.read()
        if not ok:
            break
        if i in want:
            src[i] = im
        i += 1
    cap.release()
    npz = (run / "perception" / "m1s0" / "observations_predictions.npz") if man["perception"]["mode"] == "LIVE_FROZEN_M1_S0"         else Path(man["perception"]["replay_path"])
    z = np.load(npz)
    Wd, Hd = man["intake"]["frame_size"]
    ks = [r["source_frame"] for r in mapping]
    pts = np.concatenate([z["input_2d"][k][z["input_valid"][k]][:, :2] for k in ks]) * [Wd, Hd]
    c0, c1 = pts.min(0), pts.max(0)
    half = max(c1 - c0) * 0.6 + 20
    mid = (c0 + c1) / 2
    crop = [int(max(0, mid[0] - half)), int(max(0, mid[1] - half)), int(min(Wd, mid[0] + half)), int(min(Hd, mid[1] + half))]
    fp = prop["footprint"]
    disp = com["display_final"]
    H = size

    def lab(img, txt, y, col=(255, 255, 255), s=0.42):
        cv2.putText(img, txt, (6, y), cv2.FONT_HERSHEY_SIMPLEX, s, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, txt, (6, y), cv2.FONT_HERSHEY_SIMPLEX, s, col, 1, cv2.LINE_AA)
        return img

    def fit(im):
        s = H / im.shape[0]
        im = cv2.resize(im, (int(im.shape[1] * s), H))
        return im

    lo, hi = mapping[0]["source_frame"], mapping[-1]["source_frame"]
    vid = out / f"ab_{a.request_id}_30fps.mp4"
    frames_out = []
    for r in mapping:
        k, f = r["source_frame"], r["action_frame"]
        x0, y0, x1, y1 = crop
        s_im = lab(fit(src[k][y0:y1, x0:x1].copy()), "ORIGINAL SOURCE (fixed crop)", 16)
        lab(s_im, f"src frame {k}  t={r['source_time_s']:.3f}s", 34)
        full = fit(src[k].copy())
        cv2.rectangle(full, (int(x0 * H / Hd), int(y0 * H / Hd)), (int(x1 * H / Hd), int(y1 * H / Hd)), (0, 255, 255), 1)
        lab(full, "full source frame (crop box yellow)", 16)
        tiles = [s_im]
        for name in ("baseline", "edited"):
            body = cv2.imread(str(out / f"frames_{name}" / f"body_f{f:04d}.png"))
            arm = cv2.imread(str(out / f"frames_{name}" / f"arm_f{f:04d}.png"))
            lab(body, "UNEDITED Action (baseline)" if name == "baseline" else "EDITED Action (ACCEPTED)", 16)
            lab(body, f"Action frame {f}", 34)
            lab(arm, "left-arm close-up", 16)
            if name == "edited":
                pf = fp["per_frame"][str(k)]
                cell = disp["cells"][str(k)]
                col = (80, 80, 255) if pf["changed"] else (200, 200, 200)
                lab(arm, f"{cell}", 34, col)
                lab(arm, f"wrist {pf['wrist_disp_m'] * 1e3:.1f} mm, bones {','.join(pf['channels']) or '-'}", 52, col)
                if k in pin:
                    lab(arm, "HARD PIN frame (green = target)", 70, (120, 255, 120))
            tiles.append(np.vstack([body, arm]))
        left = np.full((2 * H, max(tiles[0].shape[1], full.shape[1]), 3), 30, np.uint8)
        left[:H, :tiles[0].shape[1]] = tiles[0]
        left[H:, :full.shape[1]] = full
        top = np.hstack([left, tiles[1], tiles[2]])
        bar = np.full((40, top.shape[1], 3), 245, np.uint8)
        W = top.shape[1] - 20
        for kk in range(lo, hi + 1):
            x0 = 10 + int((kk - lo) / (hi - lo + 1) * W)
            x1 = 10 + int((kk - lo + 1) / (hi - lo + 1) * W)
            c = (219, 73, 91)[::-1] if kk in set(fp["frames_changed"]) else ((212, 127, 42)[::-1] if kk in pin else (220, 220, 220))
            cv2.rectangle(bar, (x0, 6), (max(x0, x1 - 1), 26), c, -1)
        xc = 10 + int((k - lo + 0.5) / (hi - lo + 1) * W)
        cv2.line(bar, (xc, 0), (xc, 39), (0, 0, 0), 2)
        cv2.putText(bar, "changed frames (red) / pin (blue) / unchanged (grey)", (10, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 0, 0), 1)
        frames_out.append(np.vstack([top, bar]))
    h, w = frames_out[0].shape[:2]
    wr = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), SUPPORTED_FPS, (w, h))
    for fr in frames_out:
        wr.write(fr)
    wr.release()
    cap = cv2.VideoCapture(str(vid))
    n = 0
    while cap.read()[0]:
        n += 1
    vfps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    for k in sorted(pin | {min(pin) - 1, max(pin) + 1} | {lo, hi}):
        if lo <= k <= hi:
            cv2.imwrite(str(out / f"still_src{k:04d}.jpg"), frames_out[k - lo])
    rep = {"video": str(vid.relative_to(run)), "sha256": sha256_file(vid), "frames_written": len(frames_out), "frames_decoded": n,
           "fps": vfps, "seconds": n / vfps if vfps else None, "source_seconds": len(mapping) / sidecar["source"]["container_fps"],
           "native_rate": abs(vfps - sidecar["source"]["container_fps"]) < 1e-6, "panels": ["original source", "baseline .blend render",
                                                                                        "accepted edited .blend render"],
           "render_note": "Workbench renders of the saved .blend files (final medium); footprint text comes from the evaluator display spec"}
    dump(out / "review_report.json", rep)
    print(json.dumps(rep, indent=1))
    return 0


# ------------------------------------------------------------------------------------------------ D. FBX ---
def cmd_fbx(a):
    import numpy as np
    run = Path(a.run).resolve()
    R, man = load_run(run)
    ps = PersistentStore(run)
    sid = ps.store.refs["head"] if a.state == "head" else R["baseline_state_id"] if a.state == "baseline" else a.state
    reg = blend_registry(run)
    if sid not in reg:
        raise SliceStop("UNKNOWN_STATE", sid)
    out = new_dir(run / "fbx" / f"{a.state}_{sid[:12]}")
    blend = run / reg[sid]["blend"]
    pre = sha256_file(blend)
    fbx = out / f"{sid[:12]}.fbx"
    run_blender(R["blender"], "export_fbx", out / "export.log", blend=blend, fbx=fbx, report=out / "export.json")
    exp = load(out / "export.json")
    if not fbx.exists():
        raise SliceStop("FBX_EXPORT_FAILED", "no file written")
    try:
        run_blender(R["blender"], "fbx_readback", out / "readback.log", fbx=fbx, report=out / "fbx_readback.json",
                    subframes=",".join(str(s) for s in SUBFRAMES))
        frb = load(out / "fbx_readback.json")
    except SliceStop as e:
        frb = None
        imp_err = str(e)
    brb = load(run / reg[sid]["readback"])
    rep = {"state_id": sid, "blend": reg[sid]["blend"], "blend_unchanged_by_export": pre == sha256_file(blend), "export": exp,
           "fbx_file_exists": fbx.exists(), "diagnostic_tolerance_m": FBX_DIAGNOSTIC_TOL_M,
           "note": "an exported file on disk is not a roundtrip PASS; no exporter/importer flag was changed"}
    if frb is None:
        rep.update(reimport="FAIL", reimport_error=imp_err, roundtrip_verdict="FAIL")
    else:
        def worst(A, B):
            m, at = 0.0, None
            for x, y in zip(A, B):
                for bn in x["world"]:
                    if bn not in y["world"]:
                        return float("inf"), ("MISSING_BONE", bn)
                    for p in ("head", "tail"):
                        d = float(np.linalg.norm(np.array(x["world"][bn][p]) - np.array(y["world"][bn][p])))
                        if d > m:
                            m, at = d, (x["blender_frame"], x.get("subframe", 0.0), bn, p)
            return m, at
        n_b, n_f = len(brb["frames"]), len(frb["frames"])
        common = min(n_b, n_f)
        wi, wi_at = worst(brb["frames"][:common], frb["frames"][:common])
        sb = {(s["blender_frame"], s["subframe"]): s for s in brb["subframes"]}
        pairs = [(sb[(s["blender_frame"], s["subframe"])], s) for s in frb["subframes"] if (s["blender_frame"], s["subframe"]) in sb]
        ws, ws_at = worst([p[0] for p in pairs], [p[1] for p in pairs])
        per_frame = []
        for x, y in zip(brb["frames"][:common], frb["frames"][:common]):
            per_frame.append(max(float(np.linalg.norm(np.array(x["world"][bn][p]) - np.array(y["world"].get(bn, x["world"][bn])[p])))
                                 for bn in x["world"] for p in ("head", "tail")))
        mism = []
        if wi > FBX_DIAGNOSTIC_TOL_M:
            mism.append("INTEGER_FRAME_WORLD_DRIFT_OVER_1MM")
        if ws > FBX_DIAGNOSTIC_TOL_M:
            mism.append("SUBFRAME_WORLD_DEVIATION_OVER_1MM")
        if frb["fcurve_census"]["interpolation_census_curves"] != brb["fcurve_census"]["interpolation_census_curves"]:
            mism.append("INTERPOLATION_CHANGED")
        if [frb["fcurve_census"]["key_frame_min"], frb["fcurve_census"]["key_frame_max"]] != \
                [brb["fcurve_census"]["key_frame_min"], brb["fcurve_census"]["key_frame_max"]]:
            mism.append("KEY_RANGE_CHANGED")
        if (frb["scene"]["frame_start"], frb["scene"]["frame_end"]) != (brb["scene"]["frame_start"], brb["scene"]["frame_end"]):
            mism.append("SCENE_FRAME_RANGE_NOT_CARRIED")
        if abs(frb["scene"]["effective_fps"] - brb["scene"]["effective_fps"]) > 1e-9:
            mism.append("SCENE_FPS_DIFFERS")
        if len(frb["actions_in_file"]) != 1:
            mism.append("MULTIPLE_ACTIONS_AFTER_IMPORT")
        if n_f != n_b:
            mism.append("FRAME_COUNT_DIFFERS")
        rep.update(reimport="OK", integer_frames_compared=common, integer_world_max_err_m=wi, integer_world_max_at=wi_at,
                   integer_frames_over_tol=int(sum(1 for d in per_frame if d > FBX_DIAGNOSTIC_TOL_M)),
                   integer_world_err_median_m=float(np.median(per_frame)) if per_frame else None,
                   subframe_samples_compared=len(pairs), subframe_world_max_err_m=ws, subframe_world_max_at=ws_at,
                   blend_interpolation=brb["fcurve_census"]["interpolation_census_curves"],
                   fbx_interpolation=frb["fcurve_census"]["interpolation_census_curves"],
                   blend_fcurves=brb["fcurve_census"]["n_fcurves"], fbx_fcurves=frb["fcurve_census"]["n_fcurves"],
                   blend_key_range=[brb["fcurve_census"]["key_frame_min"], brb["fcurve_census"]["key_frame_max"]],
                   fbx_key_range=[frb["fcurve_census"]["key_frame_min"], frb["fcurve_census"]["key_frame_max"]],
                   blend_scene=brb["scene"], fbx_scene=frb["scene"], fbx_actions=frb["actions_in_file"],
                   fbx_rotation_modes=frb["pose_bone_rotation_modes"], fbx_armature_scale=frb["armature_scale"],
                   mismatches=mism, roundtrip_verdict="PASS" if not mism else "FAIL")
    dump(out / "fbx_roundtrip_report.json", rep)
    print(json.dumps({k: rep.get(k) for k in ("state_id", "fbx_file_exists", "reimport", "integer_world_max_err_m", "subframe_world_max_err_m",
                                              "mismatches", "roundtrip_verdict")}, indent=1, default=str))
    return 0


# --------------------------------------------------------------------------------------------------- CLI glue ---
def add_subparser(sub):
    """Register ``experimental-video-action`` on the motion-tool parser (called by app.cli; argparse only, no imports)."""
    p = sub.add_parser("experimental-video-action",
                       help="EXPERIMENTAL opt-in: authentic 30 fps video -> known BaseRig editable .blend Action + one left-wrist WORLD_POSITION_PIN")
    s = p.add_subparsers(dest="exp_action", required=True)
    b = s.add_parser("build", help="authenticate a clip, run frozen perception, write the baseline .blend")
    b.add_argument("--clip", required=True)
    b.add_argument("--clip-sha256", required=True)
    b.add_argument("--source-record", required=True, help="JSON with title, license, author, source_uri")
    b.add_argument("--declared-fps", type=float, required=True)
    b.add_argument("--segment", type=int, nargs=2, required=True, metavar=("FIRST", "LAST"), help="inclusive source frame interval")
    b.add_argument("--perception-assets", help="JSON: frozen_code_dir, weights_dir, checkpoint, gate_report, lock")
    b.add_argument("--replay-observations", help="OFFLINE REPLAY of stored observations_predictions.npz (labelled NOT live E2E)")
    b.add_argument("--replay-sha256")
    b.add_argument("--python", help="Python for the frozen perception subprocess (default: this interpreter)")
    b.add_argument("--blender")
    b.add_argument("--out", required=True, help="new run directory (never overwritten)")
    q = s.add_parser("propose", help="propose one WORLD_POSITION_PIN edit (nothing committed)")
    q.add_argument("--run", required=True)
    q.add_argument("--request", required=True)
    c = s.add_parser("commit", help="explicit ACCEPT / REJECT of a proposal")
    c.add_argument("--run", required=True)
    c.add_argument("--request-id", required=True)
    c.add_argument("--decision", choices=("accept", "reject"), required=True)
    r = s.add_parser("revert", help="revert the head to its parent's values (new state; history kept)")
    r.add_argument("--run", required=True)
    v = s.add_parser("review", help="30 fps side-by-side source | baseline | accepted edit video")
    v.add_argument("--run", required=True)
    v.add_argument("--request-id", required=True)
    f = s.add_parser("fbx", help="export a state's .blend with the existing exporter and re-import it (characterisation only)")
    f.add_argument("--run", required=True)
    f.add_argument("--state", default="head", help="head | baseline | <state id>")
    t = s.add_parser("status", help="print head, lineage and .blend registry")
    t.add_argument("--run", required=True)
    return p


def dispatch(args):
    try:
        if args.exp_action == "build":
            args.blender_exe = find_blender(args.blender)
            return cmd_build(args)
        return {"propose": cmd_propose, "commit": cmd_commit, "revert": cmd_revert, "review": cmd_review, "fbx": cmd_fbx,
                "status": cmd_status}[args.exp_action](args)
    except SliceStop as e:
        print(f"[experimental-video-action] STOP {e.code}: {e.detail}", file=sys.stderr)
        return 2


def main(argv=None):
    ap = argparse.ArgumentParser(prog="experimental-video-action")
    sub = ap.add_subparsers(dest="command", required=True)
    add_subparser(sub)
    return dispatch(ap.parse_args(["experimental-video-action"] + list(argv if argv is not None else sys.argv[1:])))


if __name__ == "__main__":
    raise SystemExit(main())
