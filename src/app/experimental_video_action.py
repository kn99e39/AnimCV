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
  recover  (W125) complete or roll back an interrupted commit/revert from its journal; never invents an accepted state.
  adopt    (W125) explicit, non-destructive upgrade of a W124 run (unversioned refs.json) to sealed transactional refs.

W125 transaction model (see docs/EXPERIMENTAL_VIDEO_ACTION_WORLD_PIN.md): every head-moving operation runs under one
OS-level non-blocking run lock, writes a sealed journal BEFORE staging, stages the content-addressed state, the .blend and
its fresh-process Blender readback, verifies them, and only then publishes by ONE atomic replace of lineage/refs.json
(compare-and-swap on the bytes read under the lock).  refs.json binds the head state id to its .blend SHA-256 and readback.
Log, registry, commit.json and the final timeline are written after publication and are re-derivable by ``recover``.
W126 commit-time semantic authority (see docs/EXPERIMENTAL_VIDEO_ACTION_WORLD_PIN.md): ``proposal.json`` is NEVER the gate
authority.  ``propose`` writes a versioned, sealed proposal (``animcv_exp_proposal_v2``) whose provenance block binds the request
bytes, the parent head evidence, the binding record, the candidate state/.blend/readback and the gate record - a CONSISTENCY
record only (an unkeyed SHA, not authentication).  ``commit --decision accept`` independently RE-DERIVES the frozen W116/W118
decision from authoritative inputs (request bytes, bindings recomputed from the immutable baseline, the verified head and its
readback, the frozen resolver/evaluator/footprint/display code) twice - on the hash-bound candidate readback before staging, and
on a FRESH Blender readback of the staged accepted copy before publication - and publishes only if both are ACCEPT and identical
to the sealed proposal.  Any divergence STOPs with ``REFUSED_SEMANTIC_AUTHORIZATION`` without moving the head; unsealed
(W124/W125) proposals must be re-proposed before they can be accepted.
  review   30 fps side-by-side video: original source | baseline Action | accepted edited Action, footprint overlaid.
  fbx      export a state's .blend through the existing ``blender.export.export_fbx`` and re-import it with the existing
           default importer; integer-frame / subframe / interpolation / timebase readback is REPORTED, never repaired.
  fbx-interchange / fbx-consume / fbx-verify   (Batch 127, opt-in) explicit FBX continuous-time interchange of ONE verified
           published state: a NEW package (FBX without key simplification, one global take, CONSTANT-driven curves flagged
           CONSTANT in KeyAttrFlags, sealed manifest with take/timebase/rig identity), a source-free Blender consumer that
           reads ONLY the package, and a separate evaluator that re-opens original and recovered .blend in fresh processes.
           ``fbx`` (stock) is unchanged; raw FBX alone is NOT lossless (see docs/EXPERIMENTAL_VIDEO_ACTION_WORLD_PIN.md).

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

from app import experimental_action_txn as TX

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


def _atomic_store_cls(S):
    """The frozen W116 ``Store`` with ONE change: ``put`` writes the content-addressed state file atomically (temp + fsync +
    rename) instead of a direct write, so a killed process can never leave a torn state file under a valid id.  The
    commit/revert/lineage logic is inherited unchanged."""

    class AtomicStore(S.Store):
        def put(self, st):
            if not S.verify_state(st):
                raise ValueError("state id does not bind its content")
            p = self.path(st["state_id"])
            if p.exists():
                if p.read_bytes() != S.canon(st):
                    raise SliceStop("BLOCKED_STATE_FILE_CONFLICT", f"{p.name}: existing bytes differ from the content-addressed "
                                    "state (torn or tampered); run `recover` / inspect manually")
                return st["state_id"]
            TX.atomic_write_bytes(p, S.canon(st))
            p.chmod(0o444)
            return st["state_id"]
    return AtomicStore


class PersistentStore:
    """Persists the frozen W116 ``Store`` (content-addressed read-only states).  Since W125 the head is published ONLY by an
    atomic replace of ``lineage/refs.json`` inside a journalled transaction (see ``_publish``); this class only loads it."""

    def __init__(self, run):
        S, *_ = frozen_modules()
        self.P = run_paths(run)
        self.store = _atomic_store_cls(S)(Path(run) / "lineage")
        if self.P["refs"].exists():
            self.store.refs = {"head": load(self.P["refs"])["head"]}

    def save(self, new_log):
        """W124 API kept for compatibility (unversioned refs, used by the W124 store test only).  The W125 commands never
        call it: they publish through ``_publish`` inside a journalled transaction."""
        TX.atomic_dump(self.P["refs"], self.store.refs)
        for rec in new_log:
            TX.append_line_durable(self.P["log"], json.dumps(dict(rec, at=now()), sort_keys=True))


def blend_registry(run):
    p = run_paths(run)["blends"]
    return load(p) if p.exists() else {}


def register_blend(run, state_id, rec):
    """state id -> its .blend record.  An existing entry is never overwritten (an accepted copy is recorded alongside).
    Written atomically; idempotent per transaction id (a recovery roll-forward never duplicates a record)."""
    reg = blend_registry(run)
    tid = rec.get("txn_id")
    if state_id in reg:
        if tid is not None and any(r.get("txn_id") == tid for r in [reg[state_id]] + reg[state_id].get("additional_records", [])):
            return
        reg[state_id].setdefault("additional_records", []).append(rec)
    else:
        reg[state_id] = rec
    TX.atomic_dump(run_paths(run)["blends"], reg, default=_jdefault)


# ------------------------------------------------------------------- W125 transaction authority and recovery ---
REFS_SCHEMA = "animcv_exp_refs_v2"
JOURNAL_SCHEMA = "animcv_exp_txn_journal_v1"
TERMINAL = ("FINALIZED", "ABORTED")


def _norm(obj):
    """JSON-normalise (numpy scalars, tuples, Paths) so a record's digest is stable across write/read."""
    return json.loads(json.dumps(obj, default=_jdefault))


def _digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _sealed(obj, key):
    obj = _norm({k: v for k, v in obj.items() if k != key})
    obj[key] = _digest(obj)
    return obj


def _seal_ok(obj, key):
    return isinstance(obj, dict) and obj.get(key) == _digest({k: v for k, v in obj.items() if k != key})


def read_refs(run):
    """-> (refs, raw bytes, kind) with kind 'V2' or 'LEGACY_W124'.  A V2 refs whose seal fails is a STOP (never repaired)."""
    p = run_paths(run)["refs"]
    if not p.exists():
        raise SliceStop("BLOCKED_REFS_MISSING", str(p))
    raw = p.read_bytes()
    try:
        refs = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        raise SliceStop("BLOCKED_REFS_CORRUPT", "lineage/refs.json does not parse (torn or tampered); it is never rewritten "
                        "automatically") from None
    if not isinstance(refs, dict) or not isinstance(refs.get("head"), str):
        raise SliceStop("BLOCKED_REFS_CORRUPT", "lineage/refs.json has no head")
    if refs.get("schema") is None and set(refs) == {"head"}:
        return refs, raw, "LEGACY_W124"
    if refs.get("schema") != REFS_SCHEMA or not _seal_ok(refs, "record_digest"):
        raise SliceStop("BLOCKED_REFS_TAMPERED", "lineage/refs.json seal does not bind its content")
    return refs, raw, "V2"


def make_refs(head, generation, head_blend, txn_id, prev_head, op):
    return _sealed({"schema": REFS_SCHEMA, "head": head, "generation": int(generation), "head_blend": head_blend, "txn_id": txn_id,
                    "prev_head": prev_head, "op": op, "published_at": now(),
                    "authority": "this file is the ONLY authority for the current head; it is replaced atomically at the "
                                 "publication point of a journalled transaction"}, "record_digest")


def verify_head(run, refs):
    """head -> content-addressed state -> .blend (SHA-256) -> independent readback report (SHA-256 + bound blend hash)."""
    S, *_ = frozen_modules()
    run, probs = Path(run), []
    hb = refs.get("head_blend") or {}
    sp = run / "lineage" / "states" / f"{refs['head']}.json"
    if not sp.exists():
        probs.append("HEAD_STATE_MISSING")
    else:
        try:
            if not S.verify_state(json.loads(sp.read_text())):
                probs.append("HEAD_STATE_FAILS_CONTENT_BINDING")
        except ValueError:
            probs.append("HEAD_STATE_UNPARSABLE")
    if hb.get("state_id") != refs["head"]:
        probs.append("HEAD_BLEND_RECORD_FOR_OTHER_STATE")
    for k in ("blend", "readback"):
        f = run / hb.get(k, "\0missing")
        if not hb.get(k) or not f.exists():
            probs.append(f"HEAD_{k.upper()}_MISSING")
        elif sha256_file(f) != hb.get(f"{k}_sha256"):
            probs.append(f"HEAD_{k.upper()}_SHA256_MISMATCH")
    if not probs:
        rb = load(run / hb["readback"])
        if rb.get("blend_sha256") != hb["blend_sha256"]:
            probs.append("HEAD_READBACK_NOT_OF_THIS_BLEND")
        if rb.get("value_digest") != hb.get("value_digest"):
            probs.append("HEAD_READBACK_VALUE_DIGEST_MISMATCH")
    return {"ok": not probs, "problems": probs}


def require_v2_head(run):
    refs, raw, kind = read_refs(run)
    if kind != "V2":
        raise SliceStop("BLOCKED_LEGACY_RUN_REQUIRES_ADOPT", "this run was written by W124 (unversioned refs.json); it stays "
                        "readable (status/review/fbx) but is not mutated until you run `experimental-video-action adopt --run "
                        f"{run}` explicitly (non-destructive: the legacy refs are preserved)")
    v = verify_head(run, refs)
    if not v["ok"]:
        raise SliceStop("BLOCKED_HEAD_UNVERIFIED", f"authoritative head does not resolve to a verified .blend: {v['problems']}")
    return refs, raw


def _txn_dir(run):
    return Path(run) / "lineage" / "txn"


def read_journals(run):
    """-> list of (path, journal or None if unreadable/tampered)."""
    out = []
    d = _txn_dir(run)
    if d.exists():
        for p in sorted(d.glob("*.json")):
            try:
                j = load(p)
                out.append((p, j if _seal_ok(j, "journal_digest") else None))
            except ValueError:
                out.append((p, None))
    return out


def pending_journals(run):
    return [(p, j) for p, j in read_journals(run) if j is None or j.get("state") not in TERMINAL]


def require_no_pending(run):
    pend = pending_journals(run)
    if pend:
        raise SliceStop("RECOVERY_REQUIRED", f"{len(pend)} interrupted transaction(s) "
                        f"{[p.stem for p, _ in pend]}; nothing was assumed - run `experimental-video-action recover --run {run}`")


def write_journal(run, j, state):
    j = _sealed(dict(j, state=state, updated_at=now(), history=list(j.get("history", [])) + [[state, now()]]), "journal_digest")
    TX.atomic_dump(_txn_dir(run) / f"{j['txn_id']}.json", j)
    return j


def new_txn(run, op, refs, **kw):
    import uuid
    tid = f"{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S')}_{op}_{uuid.uuid4().hex[:8]}"
    return {"schema": JOURNAL_SCHEMA, "txn_id": tid, "op": op, "pid": os.getpid(), "created_at": now(),
            "expected_head": refs["head"], "expected_generation": refs["generation"],
            "expected_refs_sha256": hashlib.sha256(json.dumps(refs, sort_keys=True).encode()).hexdigest(), **_norm(kw)}


def _publish(run, raw_before, new_refs):
    """THE publication point: compare-and-swap on the exact refs bytes read under the lock, then one atomic replace."""
    p = run_paths(run)["refs"]
    if p.read_bytes() != raw_before:
        raise SliceStop("BLOCKED_CONCURRENT_HEAD_CHANGE", "refs.json changed while this transaction held the run lock")
    TX.atomic_dump(p, new_refs)


def _log_append(run, rec, txn_id, torn_point=None):
    """Append one lineage log record for a transaction exactly once (idempotent on txn_id)."""
    recs, torn, bad = TX.read_jsonl(run_paths(run)["log"])
    if torn is not None or bad:
        raise SliceStop("RECOVERY_REQUIRED", "lineage/log.jsonl has a torn or unparsable line; run `recover`")
    if any(r.get("txn_id") == txn_id for r in recs):
        return False
    TX.append_line_durable(run_paths(run)["log"], json.dumps(_norm(dict(rec, at=now(), txn_id=txn_id)), sort_keys=True),
                           torn_point=torn_point)
    return True


def _quarantine(run, txn_id, rels):
    """Move (never delete) unpublished staged outputs to quarantine/<txn_id>/, recording their SHA-256."""
    run, moved = Path(run), []
    for rel in rels:
        src = run / rel
        if not src.exists():
            continue
        dst = run / "quarantine" / txn_id / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            raise SliceStop("BLOCKED_QUARANTINE_COLLISION", str(dst))
        moved.append({"from": str(rel), "to": str(dst.relative_to(run)), "sha256": sha256_file(src)})
        if not os.access(src, os.W_OK):
            os.chmod(src, 0o644)
        os.replace(src, dst)
    return moved


def _stray_tmp(run):
    run = Path(run)
    return sorted(str(p.relative_to(run)) for d in ("lineage", "lineage/txn", "accepted", "reverted") if (run / d).exists()
                  for p in (run / d).glob(".*.tmp"))


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


def _head_blend(run, rec, sid, role, blend=None, readback=None):
    """The head -> .blend binding stored INSIDE refs.json (so head and .blend identity are published by one atomic replace)."""
    run = Path(run)
    blend, readback = blend or rec["blend"], readback or rec["readback"]
    return {"state_id": sid, "role": role, "blend": str(blend), "blend_sha256": sha256_file(run / blend), "readback": str(readback),
            "readback_sha256": sha256_file(run / readback), "value_digest": rec["value_digest"]}


def build_state_blend(run, blender, state, name, folder, prov, rest_json, rig_fbx, sidecar=None, between=None):
    """Key one state into a NEW .blend and read it back in a fresh Blender process.  ``between`` (W125) is called after the
    build process and before the independent readback process (fault-injection boundary)."""
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
    if between is not None:
        between()
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
    init_lineage(run, v0, rec)
    immutable = {str(Path(rec["blend"])): rec["blend_sha256"], "baseline/action_sidecar.json": sc_sha,
                 "manifest/input_manifest.json": man_sha, "baseline/baseline_readback.json": sha256_file(run / rec["readback"])}
    dump(run_paths(run)["run"], {"schema": SLICE_ID + ":run", "created": now(), "input_manifest_sha256": man_sha,
                                 "immutable_files": immutable, "baseline_state_id": v0["state_id"], "blender": a.blender_exe,
                                 "rig_fbx": str(rig_fbx), "rest_json": str(rest_json)})
    print(json.dumps({"run": str(run), "baseline_state_id": v0["state_id"], "baseline_blend": rec["blend"],
                      "action_value_digest": rb["value_digest"], "perception": perception["label"],
                      "bindings_bound": sorted(sidecar["bindings"]["bound"])}, indent=1))
    return 0


def init_lineage(run, v0, rec):
    """New run: content-addressed baseline state, registry, log, then the sealed v2 refs (generation 0) last."""
    ps = PersistentStore(run)
    ps.store.put(v0)
    ps.store.init_ref("head", v0["state_id"])
    register_blend(run, v0["state_id"], dict(rec, role="baseline", txn_id="build"))
    _log_append(run, ps.store.log[-1], "build")
    TX.atomic_dump(run_paths(run)["refs"], make_refs(v0["state_id"], 0, _head_blend(run, rec, v0["state_id"], "baseline"), "build",
                                                     None, "INIT"))


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
    run = Path(a.run).resolve()
    with TX.RunLock(run / "lineage", "propose", SliceStop):
        return _propose(a, run)


# ------------------------------------------------------------------- W126 semantic gate (shared propose/commit) ---
PROPOSAL_SCHEMA = "animcv_exp_proposal_v2"
AUTH_REFUSED = "REFUSED_SEMANTIC_AUTHORIZATION"
# Every field of the proposal that carries (or is derived from) the frozen W116/W118 gate.  ``quality_disclosure`` is a
# never-gating disclosure and ``note`` is free text; neither is part of the gate record.
GATE_KEYS = ("request_id", "parent_state_id", "slice", "automatic_out_of_pin_budget_frames", "mode", "constraint",
             "validation_errors", "decision", "candidate", "status", "resolver", "evaluations", "footprint", "display",
             "display_check", "pin_frames", "hard_pin_residual_max_m", "control_residual_max_m", "subframes")
# Declared (pre-outcome) precedence of the W126 refusal reasons; the first present reason is the primary class.
AUTH_REASON_ORDER = (
    # stage PROVENANCE (hash-bound consistency; before any recomputation)
    "PROPOSAL_SEAL_INVALID", "PROPOSAL_DECISION_STATUS_CONFLICT", "REQUEST_MISSING", "REQUEST_BYTES_CHANGED",
    "BINDING_RECORD_CHANGED", "BINDING_PROVENANCE_MISMATCH", "PARENT_PROVENANCE_MISMATCH", "PARENT_EVIDENCE_CHANGED",
    "CANDIDATE_PROVENANCE_MISMATCH", "CANDIDATE_STATE_CHANGED", "CANDIDATE_BLEND_CHANGED", "CANDIDATE_READBACK_MISSING",
    "CANDIDATE_READBACK_CHANGED", "CANDIDATE_READBACK_NOT_OF_BLEND",
    # stage PRE_STAGING (frozen gate re-derived from authoritative inputs on the hash-bound candidate readback)
    "RECOMPUTED_GATE_REJECT", "CANDIDATE_NOT_REPRODUCED", "PROPOSAL_GATE_RECORD_DIVERGES",
    # stage FRESH_READBACK (frozen gate re-derived on a fresh-process Blender readback of the staged accepted copy)
    "FRESH_READBACK_GATE_REJECT", "FRESH_CANDIDATE_NOT_REPRODUCED", "FRESH_GATE_RECORD_DIVERGES",
    # stage PRE_PUBLISH (inputs re-hashed immediately before the W125 publication point)
    "INPUTS_CHANGED_DURING_TRANSACTION",
)
DECISION_STATUS = {"ACCEPT": ("PROPOSED_GATE_PASS",), "REJECT": ("GATE_REJECTED", "GATE_REJECTED_NO_CANDIDATE")}


def _cj(obj) -> str:
    """Canonical JSON text of a JSON-normalised object (NaN-safe string equality)."""
    return json.dumps(_norm(obj), sort_keys=True, separators=(",", ":"))


def gate_view(rec):
    return {k: _norm(rec.get(k)) for k in GATE_KEYS}


def gate_divergence(a, b):
    """Gate-record keys whose canonical value differs between two proposal-like records."""
    return [k for k in GATE_KEYS if _cj(a.get(k)) != _cj(b.get(k))]


def _gate_record(run, R, refs, req, bnd, readback_for):
    """THE frozen W116/W118 gate for one request against the CURRENT verified head (used by propose AND commit).

    ``readback_for(candidate_state) -> raw Blender readback dict`` supplies the evaluated candidate (propose: build + fresh
    readback; commit: the hash-bound stored readback, then a fresh readback of the staged accepted copy).  Returns
    (record, candidate state or None, raw candidate readback or None, head state, head readback, sidecar)."""
    S, C, XB, W121 = frozen_modules()
    ps = PersistentStore(run)
    head = ps.store.get(refs["head"])
    sidecar = load(run_paths(run)["sidecar"])
    head_rb = load(Path(run) / refs["head_blend"]["readback"])
    head_eval = readback_by_source(head_rb, sidecar)
    rig = C.rig_from(R["rest_json"], sidecar["rig_info"])
    scale = sidecar["rig_info"]["scale"][0]
    root_bones = [n for n in rig.bones if rig.is_root(n)]
    got = {}

    def keyer(cand):
        rb = readback_for(cand)
        got.update(cand=cand, rb=rb)
        return readback_by_source(rb, sidecar)

    rec, cand, X = propose_core(req, head, head_eval, rig, head_rb["rest"], scale, bnd["bound"], bnd["failed"], root_bones, keyer)
    if cand is not None:
        b = bnd["bound"][rec["constraint"]["binding_id"]]
        sub = subframe_report(got["rb"], head_rb, sidecar, rec["constraint"], b)
        rec["subframes"] = sub
        if rec["decision"]["decision"] == "ACCEPT" and not sub["changed_intervals_within_pin_frames"]:
            rec["decision"] = {"decision": "REJECT", "failure_class": "CONTRACT_VIOLATION", "reasons": ["SUBFRAME_CHANGE_OUTSIDE_PIN_FRAMES"]}
            rec["status"] = "GATE_REJECTED"
    return rec, cand, got.get("rb"), head, head_rb, sidecar


def recompute_bindings(run, R):
    """Re-derive the rig binding record from the IMMUTABLE baseline (state, .blend readback, sidecar, rig files) with the
    frozen W116 ``verify_binding`` - the binding authority at commit time (``baseline/bindings.json`` is only compared)."""
    S, C, XB, W121 = frozen_modules()
    run = Path(run)
    if sha256_file(R["rest_json"]) != RIG_FILES["rest"][1]:
        raise SliceStop("BLOCKED_RIG_IDENTITY", f"rest snapshot {R['rest_json']} differs from the known BaseRig")
    imm = {Path(k).as_posix(): v for k, v in R["immutable_files"].items()}
    if imm.get("baseline/baseline.blend") is None:
        raise SliceStop("BLOCKED_IMMUTABLE_OUTPUT_CHANGED", "run record does not guard baseline/baseline.blend")
    rb = load(run / "baseline" / "baseline_readback.json")
    if rb.get("blend_sha256") != imm["baseline/baseline.blend"]:
        raise SliceStop("BLOCKED_IMMUTABLE_OUTPUT_CHANGED", "baseline readback is not the readback of the immutable baseline .blend")
    sidecar = load(run_paths(run)["sidecar"])
    v0 = PersistentStore(run).store.get(R["baseline_state_id"])
    rig = C.rig_from(R["rest_json"], sidecar["rig_info"])
    ev = readback_by_source(rb, sidecar)
    evf = [ev[k] for k in v0["frames"]]
    ok, failed = {}, {}
    for j, cands in C.REAL_BINDING_CANDIDATES.items():
        r = S.verify_binding(j, cands, "BaseRig", sidecar["rig_info"]["rest_snapshot_digest"], rb["rest"], evf, v0["channels"], rig)
        (ok.__setitem__(r["binding_id"], r) if r["status"] == "BOUND" else failed.__setitem__(j, r))
    return _norm({"bound": ok, "failed": failed})


def _bindings_semantic(b):
    return {"bound": b.get("bound"), "failed": b.get("failed")}


def proposal_kind(prop):
    return "V2" if isinstance(prop, dict) and prop.get("schema") == PROPOSAL_SCHEMA and "proposal_seal" in prop else "LEGACY_UNSEALED"


def proposal_integrity(run, rid, prop, bindings_now=None, refs=None):
    """Hash-bound CONSISTENCY checks of a v2 proposal against the files it names (NOT gate authority: a hostile actor able
    to rewrite every file can also rewrite these hashes; the commit re-derives the gate independently).  -> ordered reasons."""
    S, *_ = frozen_modules()
    run, ed = Path(run), Path(run) / "edits" / rid
    reasons = []
    pv = prop.get("provenance") if isinstance(prop.get("provenance"), dict) else {}
    if not _seal_ok(prop, "proposal_seal") or pv.get("gate_view_digest") != _digest(gate_view(prop)):
        reasons.append("PROPOSAL_SEAL_INVALID")
    dec = (prop.get("decision") or {}).get("decision")
    if prop.get("status") not in DECISION_STATUS.get(dec, ()) or (dec == "ACCEPT") != (prop.get("candidate") is not None and
                                                                                      prop.get("status") == "PROPOSED_GATE_PASS"):
        reasons.append("PROPOSAL_DECISION_STATUS_CONFLICT")
    rq = ed / "request.json"
    if not rq.exists():
        reasons.append("REQUEST_MISSING")
    elif not (sha256_file(rq) == (pv.get("request") or {}).get("sha256") == prop.get("request_sha256")):
        reasons.append("REQUEST_BYTES_CHANGED")
    if bindings_now is not None:
        bf = run_paths(run)["bindings"]
        cur = load(bf) if bf.exists() else {}
        if _cj(_bindings_semantic(cur)) != _cj(bindings_now):
            reasons.append("BINDING_RECORD_CHANGED")
        if (pv.get("bindings") or {}).get("semantic_digest") != _digest(bindings_now):
            reasons.append("BINDING_PROVENANCE_MISMATCH")
    par = pv.get("parent") or {}
    if par.get("state_id") != prop.get("parent_state_id") or _cj(par.get("head_blend")) != _cj(prop.get("head_blend_at_proposal")):
        reasons.append("PARENT_PROVENANCE_MISMATCH")
    if refs is not None and refs.get("head") == par.get("state_id") and _cj(refs.get("head_blend")) != _cj(par.get("head_blend")):
        reasons.append("PARENT_EVIDENCE_CHANGED")
    c = pv.get("candidate")
    if c is not None or prop.get("candidate") is not None:
        c = c or {}
        cb = prop.get("candidate_blend") or {}
        if c.get("state_id") != prop.get("candidate") or c.get("blend_sha256") != cb.get("blend_sha256") \
                or c.get("readback") != cb.get("readback") or c.get("blend") != cb.get("blend"):
            reasons.append("CANDIDATE_PROVENANCE_MISMATCH")
        sf = ed / "candidate_state.json"
        try:
            st = load(sf)
            st_ok = S.verify_state(st) and st["state_id"] == c.get("state_id") and sha256_file(sf) == c.get("state_file_sha256")
        except (OSError, ValueError, KeyError, TypeError):
            st_ok = False
        if not st_ok:
            reasons.append("CANDIDATE_STATE_CHANGED")
        bp = run / str(c.get("blend") or "\0missing")
        if not c.get("blend") or not bp.exists() or sha256_file(bp) != c.get("blend_sha256"):
            reasons.append("CANDIDATE_BLEND_CHANGED")
        rp = run / str(c.get("readback") or "\0missing")
        if not c.get("readback") or not rp.exists():
            reasons.append("CANDIDATE_READBACK_MISSING")
        else:
            if sha256_file(rp) != c.get("readback_sha256"):
                reasons.append("CANDIDATE_READBACK_CHANGED")
            try:
                if load(rp).get("blend_sha256") != c.get("blend_sha256"):
                    reasons.append("CANDIDATE_READBACK_NOT_OF_BLEND")
            except ValueError:
                reasons.append("CANDIDATE_READBACK_CHANGED")
    return [r for r in AUTH_REASON_ORDER if r in reasons]


def _input_snapshot(run, rid, prop):
    """SHA-256 of every input file the decision depends on (re-checked immediately before publication)."""
    run, ed = Path(run), Path(run) / "edits" / rid
    files = {"proposal": ed / "proposal.json", "request": ed / "request.json", "bindings": run_paths(run)["bindings"],
             "candidate_state": ed / "candidate_state.json"}
    cb = prop.get("candidate_blend") or {}
    if cb.get("blend"):
        files["candidate_blend"] = run / cb["blend"]
    if cb.get("readback"):
        files["candidate_readback"] = run / cb["readback"]
    return {k: (sha256_file(p) if p.exists() else None) for k, p in files.items()}


def _refuse(run, rid, stage, reasons, detail=None, txn_id=None):
    """Record a refused semantic authorization (new file per attempt; never overwrites) and return the STOP to raise.
    The head, the accepted Action and every input file are left exactly as they are (nothing is repaired)."""
    import uuid
    reasons = [r for r in AUTH_REASON_ORDER if r in reasons] + [r for r in reasons if r not in AUTH_REASON_ORDER]
    rec = {"schema": "animcv_exp_authorization_refusal_v1", "request_id": rid, "at": now(), "stage": stage,
           "primary": reasons[0] if reasons else None, "reasons": reasons, "detail": detail, "txn_id": txn_id,
           "meaning": "commit-time semantic authorization FAILED: the proposal is NOT trusted and was NOT applied; the head "
                      "is unchanged.  Re-propose the request (new request id) to obtain a trusted proposal."}
    d = Path(run) / "edits" / rid / "authorization_refusals"
    d.mkdir(parents=True, exist_ok=True)
    TX.atomic_dump(d / f"{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}.json",
                   rec, default=_jdefault)
    return SliceStop(AUTH_REFUSED, f"{stage}: {reasons[0] if reasons else '?'} {reasons}; head unchanged; nothing published")


def _summ(rec):
    return {"decision": rec.get("decision"), "status": rec.get("status"), "candidate": rec.get("candidate"),
            "hard_pin_residual_max_m": rec.get("hard_pin_residual_max_m"),
            "changed_frames": (rec.get("footprint") or {}).get("frames_changed"),
            "changed_bones": sorted((rec.get("footprint") or {}).get("channel_frames", {})),
            "resolver_statuses": sorted({v["status"] for v in (rec.get("resolver") or {}).values()})}


def _propose(a, run):
    S, C, XB, W121 = frozen_modules()
    R, man = load_run(run)
    require_frozen_code()
    require_no_pending(run)
    refs, _ = require_v2_head(run)
    req_bytes = Path(a.request).read_bytes()
    req = json.loads(req_bytes)
    rid = req.get("request_id")
    if not isinstance(rid, str) or not REQUEST_ID_RE.match(rid):
        raise SliceStop("INVALID_REQUEST_ID", repr(rid))
    ed = new_dir(run / "edits" / rid)
    (ed / "request.json").write_bytes(req_bytes)
    bnd = load(run_paths(run)["bindings"])
    blender = R["blender"]
    store_rb = {}

    def build_and_read_back(cand):
        prov = {"slice": SLICE_ID, "status": "EXPERIMENTAL", "input_manifest_sha256": R["input_manifest_sha256"], "state_id": cand["state_id"],
                "parent_state_id": cand["parent_id"], "role": f"candidate for request {rid} (PROPOSED; not committed)",
                "request_sha256": hashlib.sha256(req_bytes).hexdigest(), "frame_rule": "Blender frame = source frame - first + 1"}
        rec, b, rb = build_state_blend(run, blender, cand, "candidate", f"edits/{rid}", prov, R["rest_json"], R["rig_fbx"])
        store_rb.update(rec=rec)
        return rb

    rec, cand, _rb, head, head_rb, sidecar = _gate_record(run, R, refs, req, bnd, build_and_read_back)
    rec["request_sha256"] = hashlib.sha256(req_bytes).hexdigest()
    rec["input_manifest_sha256"] = R["input_manifest_sha256"]
    cprov = None
    if cand is not None:
        cs_sha = dump(ed / "candidate_state.json", cand)
        rec["candidate_blend"] = store_rb["rec"]
        cb = store_rb["rec"]
        cprov = {"state_id": cand["state_id"], "state_file_sha256": cs_sha, "blend": cb["blend"], "blend_sha256": cb["blend_sha256"],
                 "readback": cb["readback"], "readback_sha256": sha256_file(run / cb["readback"]), "value_digest": cb["value_digest"]}
    rec["head_blend_at_proposal"] = refs["head_blend"]
    rec["note"] = "PROPOSED only: a proposal (and its candidate.blend) is NEVER an accepted edit; only refs.json is authority"
    rec["schema"] = PROPOSAL_SCHEMA
    rec["provenance"] = {
        "request": {"path": f"edits/{rid}/request.json", "sha256": rec["request_sha256"], "identity": "exact bytes"},
        "parent": {"state_id": head["state_id"], "generation": refs["generation"], "refs_txn_id": refs.get("txn_id"),
                   "head_blend": refs["head_blend"]},
        "bindings": {"file_sha256": sha256_file(run_paths(run)["bindings"]), "semantic_digest": _digest(_norm(_bindings_semantic(bnd))),
                     "identity": "semantic (bound/failed records); re-derived from the immutable baseline at commit"},
        "candidate": cprov, "input_manifest_sha256": R["input_manifest_sha256"],
        "frozen_code_sha256_lf": {k: v["sha256_lf"] for k, v in frozen_code_report().items()},
        "slice_code_sha256_lf": sha256_code(__file__), "gate_view_digest": _digest(gate_view(rec)),
        "authority": "CONSISTENCY RECORD ONLY (unkeyed SHA-256 seal; not authentication).  `commit --decision accept` re-derives "
                     "the frozen W116/W118 gate from authoritative inputs and never trusts this file's decision."}
    rec = _sealed(rec, "proposal_seal")
    TX.atomic_dump(ed / "proposal.json", rec, default=_jdefault)   # written last: an interrupted propose has no proposal.json
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
    untrusted = rec.get("untrusted") or []
    rejected = (rec.get("decision") or {}).get("decision") != "ACCEPT" or bool(untrusted)
    status = rec.get("final_status") or rec.get("status")
    if untrusted:          # W126: an unverified proposal's gate/footprint fields are never shown as authoritative
        status = f"{status} | UNTRUSTED PROPOSAL ({untrusted[0]}): gate/footprint below NOT authoritative"
        rec = dict(rec, decision={"decision": "UNVERIFIED", "reasons": list(untrusted)})
        fp = {"frames_changed": [], "channel_frames": {}, "per_frame": {}}
        pin = {k for k in pin if isinstance(k, int) and k in set(frames)}
    rows = ["requested hard pin", "permitted domain (budget 0)", "ACTUAL changed (any)"] + [f"  bone {c}" for c in sorted(fp["channel_frames"])] \
        + ["unchanged"]
    fig, axes = plt.subplots(2, 1, figsize=(14, 2.4 + 0.84 * len(rows)))
    changed = set(fp["frames_changed"])
    span = sorted(pin | changed) or [frames[0]]
    zoom = (max(frames[0], span[0] - 15), min(frames[-1], span[-1] + 15))
    for ax, (lo, hi), tag in ((axes[0], (frames[0], frames[-1]), "whole Action"), (axes[1], zoom, "zoom")):
        _timeline_axis(ax, rows, lo, hi, frames, pin, changed, fp, rejected, fps, rec, status, tag)
    fig.tight_layout()
    tmp = Path(path).with_name(f".{Path(path).name}.{os.getpid()}.tmp.png")
    fig.savefig(tmp, dpi=110, format="png")
    plt.close(fig)
    os.replace(tmp, path)          # W125: a killed process never leaves a torn timeline under the final name


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
RB_KEYS = ("world", "matrix_world", "stored_wxyz")


def _frames_equal(rb1, rb2):
    ints = [{k: f[k] for k in RB_KEYS} for f in rb1["frames"]] == [{k: f[k] for k in RB_KEYS} for f in rb2["frames"]]
    subs = [{k: f[k] for k in ("world", "matrix_world")} for f in rb1["subframes"]] == \
        [{k: f[k] for k in ("world", "matrix_world")} for f in rb2["subframes"]]
    return ints, subs


def _parent_readback(run, prop):
    hb = prop.get("head_blend_at_proposal")
    return load(Path(run) / (hb["readback"] if hb else blend_registry(run)[prop["parent_state_id"]]["readback"]))


def _display_final(run, prop, final):
    """W124 display-final record + frozen W118 display truth check (unchanged logic)."""
    S, C, XB, W121 = frozen_modules()
    out = {"display_final": dict(prop.get("display") or {}, status=final) if prop.get("display") else None}
    if prop.get("display"):
        sidecar = load(run_paths(run)["sidecar"])
        U = readback_by_source(_parent_readback(run, prop), sidecar)
        X = readback_by_source(load(Path(run) / prop["candidate_blend"]["readback"]), sidecar)
        frames = [r["source_frame"] for r in sidecar["frame_mapping"]]
        out["display_final_check"] = XB.check_display(out["display_final"], X, U, frames, prop["pin_frames"], prop["candidate"],
                                                      decision="ACCEPT" if final == "ACCEPTED" else "REJECT")
    return out


def _publication_checks(run, R, prop, rb_acc, rb_cand, rb_par, dst_sha):
    """Before publication: the accepted copy, read back by a FRESH Blender process, must be bit-identical to the gated
    candidate and its actual footprint (frozen W118 ``derive_footprint``, exact equality) must equal the proposal's."""
    import numpy as np
    S, C, XB, W121 = frozen_modules()
    sidecar = load(run_paths(run)["sidecar"])
    ints, subs = _frames_equal(rb_acc, rb_cand)
    rig = C.rig_from(R["rest_json"], sidecar["rig_info"])
    dom = S.domain_points(rig, sorted(prop["constraint"]["authorized_channels"]))
    X, U = readback_by_source(rb_acc, sidecar), readback_by_source(rb_par, sidecar)
    frames = [r["source_frame"] for r in sidecar["frame_mapping"]]
    fp = _norm(XB.derive_footprint(X, U, frames, prop["pin_frames"], dom))
    b = load(run_paths(run)["bindings"])["bound"][prop["constraint"]["binding_id"]]
    tgt = np.array(prop["constraint"]["target"], float)
    resid = max(float(np.linalg.norm(np.array(X[k]["world"][b["bone"]][b["point"]]) - tgt)) for k in prop["pin_frames"])
    c = {"accepted_blend_sha256_equals_gated_candidate": dst_sha == prop["candidate_blend"]["blend_sha256"],
         "readback_bound_to_accepted_blend": rb_acc.get("blend_sha256") == dst_sha,
         "value_digest_equals_gated_candidate": rb_acc["value_digest"] == rb_cand["value_digest"] == prop["candidate_blend"]["value_digest"],
         "integer_frames_bit_equal_gated_candidate": ints, "subframes_bit_equal_gated_candidate": subs,
         "actual_footprint_frames_equal_proposal": fp["frames_changed"] == prop["footprint"]["frames_changed"],
         "actual_footprint_bones_equal_proposal": fp["channel_frames"] == prop["footprint"]["channel_frames"]}
    return {"ok": all(c.values()), "checks": c, "actual_changed_frames": fp["frames_changed"],
            "actual_changed_bones": sorted(fp["channel_frames"]), "hard_pin_residual_max_m_integer_frames": resid,
            "proposal_hard_pin_residual_max_m": prop.get("hard_pin_residual_max_m")}


def _is_published(run, j):
    try:
        refs, _, kind = read_refs(run)
    except SliceStop:
        return None            # cannot tell -> caller must not guess
    return kind == "V2" and refs.get("txn_id") == j["txn_id"]


def _abort(run, j, reason):
    """Roll back an UNPUBLISHED transaction: staged outputs are moved to quarantine/<txn>/ (never deleted); a valid
    content-addressed state file it wrote is retained but unreferenced (it is never an accepted state by existence)."""
    S, *_ = frozen_modules()
    run = Path(run)
    rels = list(j.get("planned_outputs", []))
    if j["op"] == "REVERT" and (run / "reverted").exists():
        rels += [str(p.relative_to(run)) for p in sorted((run / "reverted").glob(f"reverted_{j['revert_state'][:12]}*"))
                 if str(p.relative_to(run)) not in rels]
    retained = None
    sf = j.get("state_file")
    if sf and not j.get("state_preexisting") and (run / sf).exists():
        try:
            st = json.loads((run / sf).read_text())
            valid = S.verify_state(st) and (run / sf).read_bytes() == S.canon(st)
        except ValueError:
            valid = False
        if valid:
            retained = sf
        else:
            rels.append(sf)
    moved = _quarantine(run, j["txn_id"], rels)
    write_journal(run, dict(j, abort_reason=reason, quarantined=moved, retained_unreferenced_state=retained), "ABORTED")
    return moved, retained


def _finish(run, j, recovered_by=None):
    """Everything AFTER the publication point.  Idempotent: a recovery roll-forward completes exactly what is missing."""
    run = Path(run)
    op = j["op"]
    _log_append(run, dict(j["store_record"], **({"completed_by_recovery": recovered_by} if recovered_by else {})), j["txn_id"],
                torn_point="torn_log_append")
    TX.fault("after_log_append")
    if op in ("ACCEPT", "REVERT"):
        register_blend(run, j["registry_state"], j["registry_record"])
    if op in ("ACCEPT", "DECISION"):
        ed = run / "edits" / j["request_id"]
        cr = j["commit_result"]
        if op == "ACCEPT" and "semantic_authorization" not in cr:      # a pre-W126 transaction: never reclassified, never invented
            cr = dict(cr, semantic_authorization={"authorized": None, "note": "NOT ESTABLISHED: published by a pre-W126 "
                                                                              "transaction (no commit-time semantic revalidation)"})
        if not (ed / "commit.json").exists():
            TX.atomic_dump(ed / "commit.json", dict(cr, **({"completed_by_recovery": recovered_by} if recovered_by else {})))
        TX.fault("before_timeline_final")
        if not (ed / "timeline_final.png").exists():
            prop = load(ed / "proposal.json")
            sidecar = load(run_paths(run)["sidecar"])
            untrusted = (cr.get("proposal_integrity") or {}).get("reasons") or []
            draw_timeline(ed / "timeline_final.png", dict(prop, final_status=cr["final_status"], untrusted=untrusted), sidecar,
                          [r["source_frame"] for r in sidecar["frame_mapping"]])
    TX.fault("before_finalize")
    sd = run / j["stage_dir"] if j.get("stage_dir") else None
    if sd is not None and sd.is_dir() and not any(sd.iterdir()):
        sd.rmdir()                     # an EMPTY staging dir only (all staged files were promoted); never deletes a file
    return write_journal(run, dict(j, **({"completed_by_recovery": recovered_by} if recovered_by else {})), "FINALIZED")


def _after_publish(run, j):
    try:
        TX.fault("after_publish")
        j = write_journal(run, j, "PUBLISHED")
        return _finish(run, j)
    except Exception as e:  # the head IS published (verified); only ancillary output is incomplete
        raise SliceStop("RECOVERY_REQUIRED_AFTER_PUBLICATION", f"head already published as {j['new_refs']['head']} (txn {j['txn_id']}); "
                        f"ancillary output incomplete ({type(e).__name__}: {e}); run `recover` to complete it") from e


def _staging_failed(run, j, e):
    pub = _is_published(run, j)
    if pub is False:
        moved, _ = _abort(run, j, f"{type(e).__name__}: {e}")
        if isinstance(e, SliceStop) and e.code == AUTH_REFUSED:     # W126: keep the semantic refusal class visible
            return SliceStop(AUTH_REFUSED, f"{e.detail}; txn {j['txn_id']} rolled back, head unchanged = {j['expected_head']}; "
                             f"{len(moved)} staged file(s) moved to quarantine/{j['txn_id']}/")
        return SliceStop("TRANSACTION_ABORTED_ROLLED_BACK", f"txn {j['txn_id']} failed before publication ({type(e).__name__}: {e}); "
                         f"head unchanged = {j['expected_head']}; {len(moved)} staged file(s) moved to quarantine/{j['txn_id']}/")
    return SliceStop("RECOVERY_REQUIRED", f"txn {j['txn_id']} failed ({type(e).__name__}: {e}); publication state "
                     f"{'PUBLISHED' if pub else 'UNKNOWN'}; run `recover`")


def cmd_commit(a):
    run = Path(a.run).resolve()
    if not REQUEST_ID_RE.match(a.request_id):
        raise SliceStop("INVALID_REQUEST_ID", repr(a.request_id))
    with TX.RunLock(run / "lineage", "commit", SliceStop):
        return _commit(a, run)


def _commit(a, run):
    S, C, XB, W121 = frozen_modules()
    R, man = load_run(run)
    require_frozen_code()            # W126: the commit re-derives the gate, so the frozen algorithms must be unchanged
    require_no_pending(run)
    refs, raw = require_v2_head(run)
    ed = run / "edits" / a.request_id
    if not (ed / "proposal.json").exists():
        raise SliceStop("PROPOSAL_MISSING_OR_INCOMPLETE", f"{a.request_id}: no proposal.json (absent or an interrupted propose); "
                        "nothing to decide")
    try:
        prop = load(ed / "proposal.json")
    except ValueError:
        raise SliceStop(AUTH_REFUSED if a.decision == "accept" else "PROPOSAL_UNREADABLE", "proposal.json does not parse") from None
    if (ed / "commit.json").exists():
        raise SliceStop("REFUSED_ALREADY_DECIDED", a.request_id)
    kind = proposal_kind(prop)
    if a.decision == "accept" and kind != "V2":
        raise SliceStop("BLOCKED_LEGACY_PROPOSAL_REQUIRES_REPROPOSE",
                        f"{a.request_id}: this proposal was written before W126 (unsealed, no provenance); its gate decision cannot be "
                        "re-established and is never grandfathered.  Nothing was changed.  Re-propose the same request under a new "
                        "request id (`experimental-video-action propose`), or `commit --decision reject` it.")
    cand = load(ed / "candidate_state.json") if (ed / "candidate_state.json").exists() else None
    if cand is not None and (not S.verify_state(cand) or cand["state_id"] != prop.get("candidate")):
        raise SliceStop("INVALID_CANDIDATE_TAMPERED", "candidate state id does not bind its content or is not the proposal's candidate")
    ps = PersistentStore(run)
    head_before = refs["head"]
    guard = {k: sha256_file(run / k) for k in R["immutable_files"]}
    if a.decision == "reject":
        integ = {"kind": kind, "reasons": proposal_integrity(run, a.request_id, prop, refs=refs) if kind == "V2" else ["LEGACY_UNSEALED_PROPOSAL"]}
        out = ps.store.commit("head", prop["parent_state_id"], cand, {"decision": "REJECT", "reasons": ["USER_REJECTED"]})
        if out["head_after"] != head_before:
            raise SliceStop("INTERNAL_INVARIANT", "a non-accept decision moved the in-memory head")
        return _record_decision(a, run, R, prop, refs, out, guard, integ)
    # ---- accept of a v2 proposal: declared precedence (W126 declaration) --------------------------------------------
    pd = (prop.get("decision") or {}).get("decision")
    if pd == "ACCEPT" and cand is not None and (prop.get("candidate_blend") or {}).get("blend") and \
            (not (run / prop["candidate_blend"]["blend"]).exists()
             or sha256_file(run / prop["candidate_blend"]["blend"]) != prop["candidate_blend"]["blend_sha256"]):
        raise SliceStop("INVALID_CANDIDATE_BLEND_TAMPERED", "staged candidate.blend differs from the gated candidate; "
                        "nothing decided, head unchanged")
    bind_auth = recompute_bindings(run, R)
    reasons = proposal_integrity(run, a.request_id, prop, bindings_now=bind_auth, refs=refs)
    if reasons:
        raise _refuse(run, a.request_id, "PROVENANCE", reasons)
    integ = {"kind": kind, "reasons": []}
    if pd != "ACCEPT" or cand is None:
        out = {"op": "commit", "decision": "REJECT", "reasons": ["GATE_REJECTED_CANNOT_ACCEPT"] + prop["decision"].get("reasons", []),
               "head_before": head_before, "head_after": head_before}
    elif prop["parent_state_id"] == head_before and cand["parent_id"] == head_before:
        return _accept(a, run, R, prop, cand, refs, raw, ps, guard, bind_auth)
    else:
        out = ps.store.commit("head", prop["parent_state_id"], cand, prop["decision"])   # STALE_PARENT / PARENT_MISMATCH record
    if out["head_after"] != head_before:
        raise SliceStop("INTERNAL_INVARIANT", "a non-accept decision moved the in-memory head")
    return _record_decision(a, run, R, prop, refs, out, guard, integ)


def _authorize(run, R, refs, rid, prop, cand, bind_auth, readback, stage):
    """Re-derive the frozen gate for the ORIGINAL request bytes against the verified head, with bindings re-derived from the
    immutable baseline, evaluating ``readback`` (stored hash-bound candidate readback, or a fresh readback of the staged copy).
    Returns (ok, reasons, recomputed record).  Never consults the proposal's decision."""
    req = json.loads((Path(run) / "edits" / rid / "request.json").read_bytes())
    rec, rcand, _rb, *_ = _gate_record(run, R, refs, req, bind_auth, lambda c: readback)
    pre = "" if stage == "PRE_STAGING" else "FRESH_"
    reasons = []
    if rec["decision"]["decision"] != "ACCEPT":
        reasons.append("RECOMPUTED_GATE_REJECT" if stage == "PRE_STAGING" else "FRESH_READBACK_GATE_REJECT")
    if rcand is None or rcand["state_id"] != cand["state_id"] or S_canon(rcand) != S_canon(cand):
        reasons.append(pre + "CANDIDATE_NOT_REPRODUCED")
    div = gate_divergence(rec, prop)
    if div:
        reasons.append(pre + "PROPOSAL_GATE_RECORD_DIVERGES" if stage == "PRE_STAGING" else "FRESH_GATE_RECORD_DIVERGES")
    return not reasons, reasons, rec, div


def S_canon(st):
    S, *_ = frozen_modules()
    return S.canon(st)


def _record_decision(a, run, R, prop, refs, out, guard, integ=None):
    """REJECT / gate-rejected / stale: the head NEVER moves.  Publication point of the decision = atomic commit.json."""
    ed = run / "edits" / a.request_id
    res = {"request_id": a.request_id, "user_decision": a.decision.upper(), "store_record": out, "final_status": "REJECTED",
           "head_before": refs["head"], "head_after": refs["head"], "generation": refs["generation"],
           "immutable_files_unchanged": guard == {k: sha256_file(run / k) for k in R["immutable_files"]},
           "proposal_integrity": integ}
    if integ and integ.get("reasons"):
        res.update(display_final=None, display_final_check={"ok": None, "skipped": "UNTRUSTED proposal: its display/gate fields are "
                                                                                 f"not authoritative ({integ['reasons']})"})
    else:
        res.update(_display_final(run, prop, "REJECTED"))
    j = new_txn(run, "DECISION", refs, request_id=a.request_id, user_decision=a.decision.upper(), store_record=out)
    res["txn_id"] = j["txn_id"]
    j["commit_result"] = _norm(res)
    j = write_journal(run, j, "PREPARED")
    try:
        TX.fault("reject_before_record")
    except BaseException as e:
        raise _staging_failed(run, j, e) from e
    TX.atomic_dump(ed / "commit.json", j["commit_result"])
    try:
        _finish(run, j)
    except Exception as e:
        raise SliceStop("RECOVERY_REQUIRED", f"decision recorded (REJECTED, head unchanged) but ancillary output incomplete "
                        f"({type(e).__name__}: {e}); run `recover`") from e
    print(json.dumps({k: res[k] for k in ("request_id", "user_decision", "final_status", "head_before", "head_after",
                                         "immutable_files_unchanged", "txn_id")}, indent=1))
    return 0


RECOMPUTED_AT_COMMIT = [
    "request validation (frozen slice precheck + W116 validate_request) from the ORIGINAL request bytes",
    "pin target re-captured from the verified head readback (W116 captured_target)",
    "rig bindings re-derived from the immutable baseline (W116 verify_binding)",
    "W116 analytic resolver statuses and patched candidate state (content id must equal the proposal's candidate)",
    "W116 evaluate_candidate (hard pin <= TOL_M, scope, locality, lengths, root, joint limit), W118 derive_footprint, display_spec, "
    "check_display, decide_x and the slice subframe rule - once on the hash-bound candidate readback, once on a FRESH Blender "
    "readback of the staged accepted copy"]
HASH_BOUND_AT_COMMIT = [
    "proposal seal + gate-view digest (consistency)", "request.json exact bytes", "bindings.json semantic content (vs re-derived)",
    "parent head state/.blend/readback (sealed refs.json, W125 verify_head)", "candidate state file, candidate .blend and its readback",
    "immutable baseline/manifest/sidecar/source clip (load_run)", "frozen algorithm sources (frozen_code_sha256.json)",
    "all decision inputs re-hashed immediately before the publication point"]


def _accept(a, run, R, prop, cand, refs, raw, ps, guard, bind_auth):
    rid = a.request_id
    ok, reasons, rec_pre, div = _authorize(run, R, refs, rid, prop, cand, bind_auth, load(run / prop["candidate_blend"]["readback"]),
                                           "PRE_STAGING")
    if not ok:
        raise _refuse(run, rid, "PRE_STAGING", reasons, detail={"recomputed": _summ(rec_pre), "proposal_claims": _summ(prop),
                                                                 "diverging_gate_keys": div})
    snap = _input_snapshot(run, rid, prop)
    acc = run / "accepted"
    n_prev = len(list(acc.glob("*.blend"))) if acc.exists() else 0
    name = f"accepted_{n_prev + 1:02d}_{cand['state_id'][:12]}"
    dst, rbp, rbl = acc / f"{name}.blend", acc / f"{name}_readback.json", acc / f"{name}_readback.log"
    for f in (dst, rbp, rbl):
        if f.exists():
            raise SliceStop("REFUSED_EXISTING_OUTPUT", f"{f} (an unrecovered orphan? run `status`)")
    rel = lambda f: str(Path(f).relative_to(run))  # noqa: E731
    sp = ps.store.path(cand["state_id"])
    TX.fault("before_stage")
    j = new_txn(run, "ACCEPT", refs, request_id=a.request_id, candidate=cand["state_id"], planned_outputs=[rel(dst), rel(rbp), rel(rbl)],
                state_file=rel(sp), state_preexisting=sp.exists())
    stage = _txn_dir(run) / f"{j['txn_id']}.stage"     # every staged byte (and any orphan Blender write) stays here
    j["stage_dir"] = rel(stage)
    j = write_journal(run, j, "PREPARED")
    try:
        TX.fault("after_journal_prepared")
        out = ps.store.commit("head", prop["parent_state_id"], cand, prop["decision"])   # atomic state put; in-memory ref only
        if out["head_after"] != cand["state_id"]:
            raise SliceStop("INTERNAL_INVARIANT", f"store refused the accept: {out.get('reasons')}")
        TX.fault("after_state_write")
        stage.mkdir(parents=True)
        sb = stage / "accepted.blend"
        shutil.copyfile(run / prop["candidate_blend"]["blend"], sb)
        TX.fsync_file(sb)
        dst_sha = sha256_file(sb)
        if dst_sha != prop["candidate_blend"]["blend_sha256"]:
            raise SliceStop("INVALID_BLEND_IDENTITY", "accepted copy differs from the gated candidate")
        TX.fault("after_blend_copy")
        TX.fault("during_blender")
        run_blender(R["blender"], "readback", stage / "accepted_readback.log", blend=sb, report=stage / "accepted_readback.json",
                    subframes=",".join(str(s) for s in SUBFRAMES))
        rb_acc = load(stage / "accepted_readback.json")
        if rb_acc.get("blend_sha256") != dst_sha:
            raise _refuse(run, rid, "FRESH_READBACK", ["FRESH_READBACK_GATE_REJECT"], detail="fresh readback is not of the staged copy",
                          txn_id=j["txn_id"])
        ok, reasons, rec_fresh, div = _authorize(run, R, refs, rid, prop, cand, bind_auth, rb_acc, "FRESH_READBACK")
        if not ok:
            raise _refuse(run, rid, "FRESH_READBACK", reasons, detail={"recomputed": _summ(rec_fresh), "proposal_claims": _summ(prop),
                                                                       "diverging_gate_keys": div}, txn_id=j["txn_id"])
        checks = _publication_checks(run, R, prop, rb_acc, load(run / prop["candidate_blend"]["readback"]), _parent_readback(run, prop), dst_sha)
        if not checks["ok"]:
            raise SliceStop("BLOCKED_PUBLICATION_CHECK_FAILED", json.dumps(checks["checks"]))
        acc.mkdir(parents=True, exist_ok=True)            # verified bytes only now get their final (still unpublished) names
        for f_src, f_dst in ((sb, dst), (stage / "accepted_readback.json", rbp), (stage / "accepted_readback.log", rbl)):
            TX._replace(f_src, f_dst)
        if sha256_file(dst) != dst_sha:
            raise SliceStop("INVALID_BLEND_IDENTITY", "accepted .blend changed while being promoted")
        res = {"request_id": a.request_id, "user_decision": "ACCEPT", "store_record": out, "final_status": "ACCEPTED",
               "head_before": refs["head"], "head_after": cand["state_id"], "generation": refs["generation"] + 1,
               "immutable_files_unchanged": guard == {k: sha256_file(run / k) for k in R["immutable_files"]},
               "accepted_blend": rel(dst), "accepted_readback": rel(rbp), "publication_checks": checks, "txn_id": j["txn_id"],
               "semantic_authorization": {
                   "schema": "animcv_exp_semantic_authorization_v1", "authorized": True, "proposal_kind": "V2",
                   "provenance_reasons": [], "pre_staging_gate": _summ(rec_pre), "fresh_readback_gate": _summ(rec_fresh),
                   "fresh_readback": rel(rbp), "fresh_readback_value_digest": rb_acc["value_digest"],
                   "hard_pin_residual_max_m_fresh_readback": rec_fresh["hard_pin_residual_max_m"],
                   "hard_pin_tolerance_m": rec_fresh["constraint"]["tolerance_m"],
                   "recomputed": RECOMPUTED_AT_COMMIT, "hash_bound": HASH_BOUND_AT_COMMIT,
                   "bindings_file_bytes_equal_proposal": snap["bindings"] == prop["provenance"]["bindings"]["file_sha256"],
                   "input_snapshot": snap,
                   "note": "the proposal's own decision field was never used as authority; seals are consistency checks, "
                           "not authentication"}}
        res.update(_display_final(run, prop, "ACCEPTED"))
        hb = dict(_head_blend(run, rb_acc, cand["state_id"], "accepted", blend=rel(dst), readback=rel(rbp)), request_id=a.request_id)
        new_refs = make_refs(cand["state_id"], refs["generation"] + 1, hb, j["txn_id"], refs["head"], "ACCEPT")
        j.update(_norm({"store_record": out, "commit_result": res, "new_refs": new_refs, "registry_state": cand["state_id"],
                        "registry_record": dict(prop["candidate_blend"], role="accepted", accepted_copy=rel(dst), accepted_readback=rel(rbp),
                                                txn_id=j["txn_id"])}))
        j = write_journal(run, j, "STAGED")
        TX.fault("before_publish")
        now_snap = _input_snapshot(run, rid, prop)       # W126: no edit between authorization and publication (TOCTOU)
        if now_snap != snap:
            raise _refuse(run, rid, "PRE_PUBLISH", ["INPUTS_CHANGED_DURING_TRANSACTION"],
                          detail={"changed": sorted(k for k in snap if snap[k] != now_snap.get(k))}, txn_id=j["txn_id"])
        _publish(run, raw, new_refs)
    except BaseException as e:
        raise _staging_failed(run, j, e) from e
    _after_publish(run, j)
    print(json.dumps({k: res[k] for k in ("request_id", "user_decision", "final_status", "head_before", "head_after",
                                         "immutable_files_unchanged", "accepted_blend", "txn_id")}, indent=1))
    return 0


def cmd_revert(a):
    run = Path(a.run).resolve()
    with TX.RunLock(run / "lineage", "revert", SliceStop):
        return _revert(a, run)


def _revert(a, run):
    S, C, XB, W121 = frozen_modules()
    R, man = load_run(run)
    require_no_pending(run)
    refs, raw = require_v2_head(run)
    ps = PersistentStore(run)
    head = ps.store.get(refs["head"])
    if head["parent_id"] is None:
        raise SliceStop("NOTHING_TO_REVERT", "head is the baseline")
    restored = ps.store.get(head["parent_id"])
    st_id = S.make_state(head["state_id"], restored["rig_id"], restored["frames"], restored["channels"], restored["values"],
                         {"kind": "REVERT", "reverts": head["state_id"], "restores": restored["state_id"]}, restored.get("source"))["state_id"]
    tag = f"reverted_{st_id[:12]}"
    rd = run / "reverted"
    planned = [f"reverted/{tag}{s}" for s in (".blend", "_state_for_blender.json", "_provenance.json", "_build.log", "_build.json",
                                               "_readback.log", "_readback.json", "_report.json")]
    if any((run / p).exists() for p in planned):
        raise SliceStop("REFUSED_EXISTING_OUTPUT", f"{tag}* already exists (an unrecovered orphan? run `status`)")
    sp = ps.store.path(st_id)
    TX.fault("before_stage")
    j = new_txn(run, "REVERT", refs, reverts=head["state_id"], restores=restored["state_id"], revert_state=st_id,
                planned_outputs=planned, state_file=str(sp.relative_to(run)), state_preexisting=sp.exists())
    stage_rel = f"lineage/txn/{j['txn_id']}.stage"     # Blender builds here; an orphan Blender can only write here
    j["stage_dir"] = stage_rel
    j = write_journal(run, j, "PREPARED")
    try:
        TX.fault("after_journal_prepared")
        st, rec = ps.store.revert("head")          # atomic put; in-memory ref only (published below)
        if st["state_id"] != st_id:
            raise SliceStop("INTERNAL_INVARIANT", "revert state id differs from the journalled id")
        TX.fault("after_state_write")
        reg = blend_registry(run)
        sidecar = load(run_paths(run)["sidecar"])
        prov = {"slice": SLICE_ID, "status": "EXPERIMENTAL", "input_manifest_sha256": R["input_manifest_sha256"], "state_id": st["state_id"],
                "parent_state_id": st["parent_id"], "role": f"REVERT of {head['state_id']} restoring {restored['state_id']}"}
        brec, b, rb = build_state_blend(run, R["blender"], st, tag, stage_rel, prov, R["rest_json"], R["rig_fbx"],
                                        between=lambda: TX.fault("during_blender"))
        rb0 = load(run / reg[restored["state_id"]]["readback"])
        same_int, same_sub = _frames_equal(rb, rb0)
        out = {"revert_record": rec, "reverted_state_id": st["state_id"], "restored_state_id": restored["state_id"],
               "values_equal_restored": canon(st["values"]) == canon(restored["values"]), "value_digest_equal": rb["value_digest"] == rb0["value_digest"],
               "blender_eval_integer_frames_bit_equal": same_int, "blender_eval_subframes_bit_equal": same_sub,
               "reverted_blend": brec, "lineage": ps.store.lineage(st["state_id"]), "txn_id": j["txn_id"]}
        out["exact_undo"] = all(out[k] for k in ("values_equal_restored", "value_digest_equal", "blender_eval_integer_frames_bit_equal",
                                                "blender_eval_subframes_bit_equal"))
        if not out["exact_undo"]:
            raise SliceStop("REVERT_NOT_EXACT", json.dumps({k: out[k] for k in ("values_equal_restored", "value_digest_equal",
                                                                               "blender_eval_integer_frames_bit_equal",
                                                                               "blender_eval_subframes_bit_equal")}))
        rd.mkdir(parents=True, exist_ok=True)              # verified -> final (still unpublished) names
        for f in sorted((run / stage_rel).glob(f"{tag}*")):
            TX._replace(f, rd / f.name)
        for k in ("blend", "build_report", "readback"):
            brec[k] = str(Path("reverted") / Path(brec[k]).name)
        if sha256_file(run / brec["blend"]) != brec["blend_sha256"]:
            raise SliceStop("INVALID_BLEND_IDENTITY", "reverted .blend changed while being promoted")
        out["reverted_blend"] = brec
        dump(rd / f"{tag}_report.json", out)
        new_refs = make_refs(st_id, refs["generation"] + 1, _head_blend(run, brec, st_id, "reverted"), j["txn_id"], head["state_id"], "REVERT")
        j.update(_norm({"store_record": rec, "new_refs": new_refs, "registry_state": st_id,
                        "registry_record": dict(brec, role="reverted", txn_id=j["txn_id"]), "report": f"reverted/{tag}_report.json"}))
        j = write_journal(run, j, "STAGED")
        TX.fault("before_publish")
        _publish(run, raw, new_refs)
    except BaseException as e:
        raise _staging_failed(run, j, e) from e
    _after_publish(run, j)
    print(json.dumps({k: out[k] for k in ("reverted_state_id", "restored_state_id", "exact_undo", "txn_id")}, indent=1))
    return 0


# ------------------------------------------------------------------------------------------- recover / adopt ---
def _recover_one(run, j, rid):
    refs, raw, kind = read_refs(run)
    if j["op"] == "DECISION":
        cj = Path(run) / "edits" / j["request_id"] / "commit.json"
        if cj.exists():
            if load(cj).get("txn_id") != j["txn_id"]:
                raise SliceStop("BLOCKED_DIVERGENT_JOURNAL", f"{j['txn_id']}: commit.json belongs to another transaction")
            _finish(run, j, recovered_by=rid)
            return {"txn_id": j["txn_id"], "op": j["op"], "outcome": "ROLLED_FORWARD (decision REJECTED was recorded; head never moved)"}
        _abort(run, j, f"recovered by {rid}: interrupted before the decision record; the proposal remains UNDECIDED")
        return {"txn_id": j["txn_id"], "op": j["op"], "outcome": "ROLLED_BACK (no decision recorded; proposal UNDECIDED; head never moved)"}
    if kind == "V2" and refs.get("txn_id") == j["txn_id"]:
        if j.get("state") not in ("STAGED", "PUBLISHED") or refs != j.get("new_refs"):
            raise SliceStop("BLOCKED_DIVERGENT_JOURNAL", f"{j['txn_id']}: refs name this transaction but differ from its staged refs")
        sa = (j.get("commit_result") or {}).get("semantic_authorization")
        if j["op"] == "ACCEPT" and sa is not None and sa.get("authorized") is not True:
            raise SliceStop("BLOCKED_DIVERGENT_JOURNAL", f"{j['txn_id']}: published ACCEPT whose journal records no semantic "
                            "authorization; recovery never invents one (inspect manually)")
        v = verify_head(run, refs)
        if not v["ok"]:
            raise SliceStop("BLOCKED_HEAD_UNVERIFIED", f"published head of {j['txn_id']} does not verify: {v['problems']}")
        _finish(run, j, recovered_by=rid)
        return {"txn_id": j["txn_id"], "op": j["op"], "outcome": f"ROLLED_FORWARD (head was already published = {refs['head']})"}
    if refs["head"] != j["expected_head"] or refs.get("generation") != j["expected_generation"]:
        raise SliceStop("BLOCKED_DIVERGENT_JOURNAL", f"{j['txn_id']}: unpublished, but the head moved since it started "
                        f"({j['expected_head'][:12]}@{j['expected_generation']} -> {refs['head'][:12]}@{refs.get('generation')})")
    moved, retained = _abort(run, j, f"recovered by {rid}: interrupted before publication")
    return {"txn_id": j["txn_id"], "op": j["op"], "outcome": f"ROLLED_BACK (head unchanged = {refs['head']})", "quarantined": moved,
            "retained_unreferenced_state": retained}


def cmd_recover(a):
    run = Path(a.run).resolve()
    with TX.RunLock(run / "lineage", "recover", SliceStop):
        return _recover(a, run)


def _recover(a, run):
    import uuid
    R, man = load_run(run)          # tampered immutable outputs / manifest STOP here: never repaired
    rid = f"recover_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}"
    refs, raw, kind = read_refs(run)   # torn / tampered refs STOP here: the authority is never invented
    actions = []
    recs, torn, bad = TX.read_jsonl(run_paths(run)["log"])
    if bad:
        raise SliceStop("BLOCKED_LOG_CORRUPT", f"{bad} unparsable non-final log line(s): not a crash signature; inspect manually")
    if torn is not None:     # crash signature: an incomplete final append.  Keep the bytes, restore the complete prefix.
        qd = run / "quarantine" / rid
        qd.mkdir(parents=True, exist_ok=True)
        (qd / "log_torn_tail.bin").write_bytes(torn)
        full = run_paths(run)["log"].read_bytes()
        TX.atomic_write_bytes(run_paths(run)["log"], full[: len(full) - len(torn)])
        actions.append({"action": "LOG_TORN_TAIL_QUARANTINED", "bytes": len(torn), "to": str((qd / "log_torn_tail.bin").relative_to(run))})
    for p, j in pending_journals(run):
        if j is None:
            raise SliceStop("BLOCKED_JOURNAL_CORRUPT_OR_TAMPERED", f"{p.name}: unreadable or seal mismatch; recovery will not guess")
        actions.append(_recover_one(run, j, rid))
    stray = _stray_tmp(run)
    if stray:
        actions.append({"action": "STRAY_TEMP_FILES_QUARANTINED", "files": _quarantine(run, rid, stray)})
    refs, raw, kind = read_refs(run)
    head_check = verify_head(run, refs) if kind == "V2" else {"ok": None, "problems": ["LEGACY_W124_REFS (not adopted)"]}
    if kind == "V2" and not head_check["ok"]:
        raise SliceStop("BLOCKED_HEAD_UNVERIFIED", f"after recovery the head does not verify: {head_check['problems']}")
    rep = {"recovery_id": rid, "at": now(), "refs_kind": kind, "head": refs["head"], "generation": refs.get("generation"),
           "head_verified": head_check, "actions": actions}
    dump(run / "recovery" / f"{rid}.json", rep)
    print(json.dumps({k: rep[k] for k in ("recovery_id", "head", "generation", "head_verified", "actions")}, indent=1, default=str))
    return 0


def cmd_adopt(a):
    run = Path(a.run).resolve()
    with TX.RunLock(run / "lineage", "adopt", SliceStop):
        return _adopt(a, run)


def _adopt(a, run):
    """Explicit, non-destructive upgrade of a W124 run (unversioned refs) to sealed v2 refs.  STOPs on any divergence."""
    S, *_ = frozen_modules()
    R, man = load_run(run)
    refs, raw, kind = read_refs(run)
    if kind != "LEGACY_W124":
        raise SliceStop("REFUSED_NOT_LEGACY", "refs.json is already v2")
    if read_journals(run):
        raise SliceStop("BLOCKED_DIVERGENT_JOURNAL", "a legacy run must not contain transaction journals")
    ps = PersistentStore(run)
    head = refs["head"]
    lin = ps.store.lineage(head)
    if not all(x["verifies"] for x in lin):
        raise SliceStop("BLOCKED_LEGACY_LINEAGE_INVALID", str(lin))
    recs, torn, bad = TX.read_jsonl(run_paths(run)["log"])
    if torn is not None or bad:
        raise SliceStop("BLOCKED_LEGACY_LOG_CORRUPT", "legacy log has torn/unparsable lines")
    moves = [r for r in recs if (r.get("op") == "commit" and r.get("head_after") != r.get("head_before")) or r.get("op") == "revert"]
    last = recs[0]["head"] if recs and recs[0].get("op") == "init" else None
    for r in moves:
        last = r.get("head_after") if r["op"] == "commit" else r.get("to")
    if last != head or len(lin) - 1 > len(moves):
        raise SliceStop("BLOCKED_LEGACY_DIVERGENT", f"legacy log final head {last} != refs head {head}")
    reg = blend_registry(run)
    if head not in reg:
        raise SliceStop("BLOCKED_LEGACY_HEAD_BLEND_UNREGISTERED", head)
    e = reg[head]
    blend = e.get("accepted_copy") or e["blend"]
    if sha256_file(run / blend) != e["blend_sha256"] or load(run / e["readback"]).get("blend_sha256") != e["blend_sha256"]:
        raise SliceStop("BLOCKED_LEGACY_HEAD_BLEND_UNVERIFIED", blend)
    legacy = run / "lineage" / "refs.legacy_w124.json"
    if legacy.exists():
        raise SliceStop("REFUSED_EXISTING_OUTPUT", str(legacy))
    TX.atomic_write_bytes(legacy, raw)
    hb = _head_blend(run, e, head, e.get("role", "legacy"), blend=blend, readback=e["readback"])
    tid = f"adopt_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    new_refs = make_refs(head, len(moves), hb, tid, None, "ADOPT_LEGACY_W124")
    j = write_journal(run, {"schema": JOURNAL_SCHEMA, "txn_id": tid, "op": "ADOPT", "legacy_refs_sha256": hashlib.sha256(raw).hexdigest(),
                            "legacy_refs_preserved_as": "lineage/refs.legacy_w124.json", "new_refs": new_refs,
                            "verified": {"lineage": lin, "log_head_moves": len(moves)}}, "STAGED")
    _publish(run, raw, new_refs)
    _log_append(run, {"op": "adopt", "ref": "head", "head": head, "generation": len(moves)}, tid)
    write_journal(run, j, "FINALIZED")
    print(json.dumps({"adopted": True, "head": head, "generation": len(moves), "head_blend": hb["blend"], "txn_id": tid}, indent=1))
    return 0


def cmd_status(a):
    """Read-only (takes no lock, writes nothing).  Everything shown derives from the authoritative refs.json."""
    S, *_ = frozen_modules()
    run = Path(a.run).resolve()
    R, man = load_run(run)
    refs, raw, kind = read_refs(run)
    ps = PersistentStore(run)
    lin = ps.store.lineage(refs["head"])
    chain = {x["state_id"] for x in lin}
    pend = []
    for p, j in pending_journals(run):
        if j is None:
            pend.append({"journal": p.name, "state": "UNREADABLE_OR_TAMPERED"})
            continue
        pub = kind == "V2" and refs.get("txn_id") == j["txn_id"]
        pend.append({"txn_id": j["txn_id"], "op": j["op"], "state": j["state"], "request_id": j.get("request_id"),
                     "published": pub, "meaning": ("head ALREADY moved to " + refs["head"] + "; ancillary records incomplete") if pub else
                     "NOT published: head unchanged; staged files are NOT accepted",
                     "staged_files_present": [f for f in j.get("planned_outputs", []) if (run / f).exists()]})
    edits = {}
    bind_now = None
    if (run / "edits").exists():
        for ed in sorted(p for p in (run / "edits").iterdir() if p.is_dir()):
            if not (ed / "proposal.json").exists():
                edits[ed.name] = {"state": "INCOMPLETE_PROPOSAL (interrupted propose; not decidable)"}
                continue
            try:
                pr = load(ed / "proposal.json")
            except ValueError:
                edits[ed.name] = {"state": "UNREADABLE_PROPOSAL (not decidable; never accepted)", "gate": "UNTRUSTED"}
                continue
            cm = load(ed / "commit.json") if (ed / "commit.json").exists() else None
            # W126: the displayed gate is the proposal's only if its hash-bound provenance verifies (consistency check; the
            # commit-time authority is the independent recomputation recorded in commit.json / authorization_refusals/).
            pkind = proposal_kind(pr)
            if pkind == "V2":
                if bind_now is None:
                    try:
                        bind_now = recompute_bindings(run, R)
                    except SliceStop as e:
                        bind_now = {"error": e.code}
                integ = proposal_integrity(run, ed.name, pr, bindings_now=bind_now if "error" not in bind_now else None,
                                           refs=refs if kind == "V2" else None)
            else:
                integ = ["LEGACY_UNSEALED_PROPOSAL"]
            gd = (pr.get("decision") or {}).get("decision")
            refusals = sorted((ed / "authorization_refusals").glob("*.json")) if (ed / "authorization_refusals").exists() else []
            if pkind != "V2":
                gate = f"LEGACY_UNVERIFIED:{gd} (pre-W126 proposal; cannot be accepted - re-propose)"
            elif integ:
                gate = f"UNTRUSTED ({integ[0]}): proposal differs from its sealed provenance; cannot be accepted"
            elif refusals:
                gate = f"REFUSED_AT_COMMIT ({load(refusals[-1]).get('primary')}): authoritative revalidation failed"
            else:
                gate = gd
            sa = (cm or {}).get("semantic_authorization")
            edits[ed.name] = {"gate": gate, "proposal_kind": pkind, "proposal_integrity": integ or "OK",
                              "candidate": pr.get("candidate"),
                              "decision": cm["final_status"] if cm else "UNDECIDED (proposal only; NOT applied)",
                              "semantic_authorization": (("AUTHORIZED_AT_COMMIT (W126 independent revalidation)" if sa.get("authorized")
                                                          else "NOT_ESTABLISHED (" + str(sa.get("note")) + ")") if sa else
                                                         ("NOT_ESTABLISHED (accepted before W126; historical, not reclassified)"
                                                          if cm and cm.get("final_status") == "ACCEPTED" else None)),
                              "authorization_refusals": [str(p.relative_to(run)) for p in refusals],
                              "in_current_head_lineage": pr.get("candidate") in chain if pr.get("candidate") else False,
                              "changed_frames": (pr.get("footprint") or {}).get("frames_changed") if not integ or pkind != "V2" else "NOT SHOWN (unverified)",
                              "changed_bones": sorted((pr.get("footprint") or {}).get("channel_frames", {})) if not integ or pkind != "V2"
                              else "NOT SHOWN (unverified)"}
    reg = blend_registry(run)
    all_states = {p.stem for p in (run / "lineage" / "states").glob("*.json")}
    reachable = set(chain)
    for x in list(reg):
        if x in all_states:
            reachable |= {y["state_id"] for y in ps.store.lineage(x)}
    acc_reg = {str(Path(r.get("accepted_copy", ""))) for r in reg.values() if r.get("accepted_copy")}
    if (refs.get("head_blend") or {}).get("blend"):
        acc_reg.add(str(Path(refs["head_blend"]["blend"])))
    orphan_acc = [str(p.relative_to(run)) for p in sorted((run / "accepted").glob("*.blend"))
                  if str(p.relative_to(run)) not in acc_reg] if (run / "accepted").exists() else []
    recs, torn, bad = TX.read_jsonl(run_paths(run)["log"])
    hv = verify_head(run, refs) if kind == "V2" else {"ok": None, "problems": ["LEGACY_W124_REFS: run `adopt` before editing"]}
    need = bool(pend) or torn is not None or bool(_stray_tmp(run)) or bool(orphan_acc)
    out = {"refs_kind": kind, "head": refs["head"], "generation": refs.get("generation"), "head_blend": refs.get("head_blend"),
           "head_verified": hv, "last_publication": {k: refs.get(k) for k in ("txn_id", "op", "prev_head", "published_at")},
           "lineage": lin, "pending_transactions": pend, "recovery_required": need,
           "recovery_action": f"experimental-video-action recover --run {run}" if need else None,
           "edits": edits, "unreferenced_states_not_accepted": sorted(all_states - reachable),
           "orphan_accepted_folder_blends_not_published": orphan_acc, "stray_temp_files": _stray_tmp(run),
           "staging_dirs_never_authority": sorted(p.name for p in _txn_dir(run).glob("*.stage")) if _txn_dir(run).exists() else [],
           "log": {"records": len(recs), "torn_tail_bytes": len(torn) if torn else 0, "bad_lines": bad},
           "quarantine": sorted(str(p.relative_to(run)) for p in (run / "quarantine").rglob("*") if p.is_file())
           if (run / "quarantine").exists() else [], "blends": reg}
    print(json.dumps(out, indent=1, default=str))
    return 0


# --------------------------------------------------------------------------------------- D. review (A/B video) ---
def cmd_review(a):
    import cv2
    import numpy as np
    run = Path(a.run).resolve()
    R, man = load_run(run)
    require_no_pending(run)
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
    require_no_pending(run)
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


# ----------------------------------------------------------------- E. opt-in FBX continuous-time interchange (W127) ---
# EXPERIMENTAL, opt-in (Batch 127).  ``fbx`` above is unchanged and keeps reporting the stock exporter/importer honestly.
# Three separate roles with separate inputs:
#   fbx-interchange  PRODUCER  (run + verified state -> NEW package directory; reads the run, never writes it)
#   fbx-consume      CONSUMER  (package directory ONLY -> NEW recovered .blend; never reads a run, .blend or report)
#   fbx-verify       EVALUATOR (run + verified state + consumer output -> fresh-process readbacks of BOTH .blends + report)
FBXI_SUBFRAMES = (0.25, 0.5, 0.75, 0.99)   # evaluator grid: quarter frames + near-boundary +0.99
FBXI_OBJECT_TOL = 1e-5                     # object matrix element tolerance (EXPERIMENTAL comparison diagnostic)
FBXI_REST_TOL_M = 1e-5                     # bone rest head/tail tolerance in metres (EXPERIMENTAL comparison diagnostic)
FBXI_FAULT_ENV = "ANIMCV_EXP_FBXI_FAULT"   # tests / evidence only (see blender/experimental_fbx_interchange.py)
FBXI_HELPER = PROJECT_ROOT / "src" / "blender" / "experimental_fbx_interchange.py"


def _fbxi():
    if str(PROJECT_ROOT / "src") not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
    import blender.experimental_fbx_interchange as X
    return X


def _fresh_dir(path):
    """Create a NEW directory atomically (a concurrent invocation on the same path gets REFUSED_EXISTING_OUTPUT)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        p.mkdir()
    except FileExistsError:
        raise SliceStop("REFUSED_EXISTING_OUTPUT", f"{p} already exists; outputs are never overwritten") from None
    return p


def _stage(blender, stage_mode, log, **kw):
    """Run one Blender stage; returns the exit code (3 = a declared STOP written to ``stop_report``)."""
    args = [blender, "-b", "-t", "2", "--python-exit-code", "1", "-P", str(STAGE), "--", stage_mode]
    for k, v in kw.items():
        args += ["--" + k.replace("_", "-"), str(v)]
    r = subprocess.run(args, cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    Path(log).write_text(" ".join(args) + "\n" + r.stdout + "\n" + r.stderr, encoding="utf-8")
    return r.returncode


def state_semantics(run, sid, reg):
    """Role and semantic-authorization label of a published state (read-only; never reclassifies anything)."""
    rec = reg[sid]
    role = rec.get("role")
    label = {"baseline": "NOT_APPLICABLE (unedited baseline Action)",
             "reverted": "NOT_APPLICABLE (exact revert: restores the parent's values; not a semantic edit)"}.get(role)
    commit = None
    if (run / "edits").exists():
        for cj in sorted((run / "edits").glob("*/commit.json")):
            c = load(cj)
            if c.get("final_status") == "ACCEPTED" and c.get("head_after") == sid:
                commit = (cj.parent.name, c)
    if role == "accepted":
        sa = (commit[1].get("semantic_authorization") if commit else None) or {}
        label = ("AUTHORIZED_AT_COMMIT (W126 independent revalidation)" if sa.get("authorized") else
                 "NOT_ESTABLISHED (accepted before W126; historical, not reclassified)")
    return {"role": role, "semantic_authorization": label, "accepted_request_id": commit[0] if commit else None}


def verify_state_blend(run, sid, refs, kind):
    """A published state of this run -> its verified .blend: content-addressed state, registry record, .blend SHA-256 and an
    independent readback bound to that .blend.  The head additionally verifies through sealed refs (W125)."""
    S, *_ = frozen_modules()
    ps = PersistentStore(run)
    chain = [x["state_id"] for x in ps.store.lineage(refs["head"])]
    if sid not in chain:
        raise SliceStop("NOT_A_PUBLISHED_STATE", f"{sid} is not in the published lineage of the current head")
    sp = run / "lineage" / "states" / f"{sid}.json"
    if not sp.exists() or not S.verify_state(json.loads(sp.read_text())):
        raise SliceStop("STATE_UNVERIFIED", sid)
    reg = blend_registry(run)
    if sid not in reg:
        raise SliceStop("UNKNOWN_STATE", sid)
    rec = reg[sid]
    blend, rbp = run / rec["blend"], run / rec["readback"]
    if not blend.exists() or sha256_file(blend) != rec["blend_sha256"]:
        raise SliceStop("STATE_BLEND_UNVERIFIED", rec["blend"])
    if not rbp.exists():
        raise SliceStop("STATE_READBACK_MISSING", rec["readback"])
    rb = load(rbp)
    if rb.get("blend_sha256") != rec["blend_sha256"] or rb.get("value_digest") != rec["value_digest"]:
        raise SliceStop("STATE_READBACK_NOT_OF_BLEND", rec["readback"])
    if sid == refs["head"] and kind == "V2":
        v = verify_head(run, refs)
        if not v["ok"]:
            raise SliceStop("BLOCKED_HEAD_UNVERIFIED", str(v["problems"]))
    return dict(rec, **state_semantics(run, sid, reg)), rb


def authority_snapshot(run):
    """Hashes of every authoritative / lineage file of the run (refs, log, registry, journals, states, baseline, accepted,
    reverted, edits).  An export must leave it identical."""
    out = {}
    for d in ("lineage", "baseline", "accepted", "reverted", "edits", "manifest"):
        base = run / d
        if base.exists():
            for p in sorted(base.rglob("*")):
                if p.is_file() and p.name not in ("txn.lock",):
                    out[str(p.relative_to(run))] = sha256_file(p)
    out["run.json"] = sha256_file(run / "run.json")
    return out


def _fbxi_fault(point):
    if os.environ.get(FBXI_FAULT_ENV) == point:
        os._exit(86)            # simulated kill: no cleanup, no status record


def produce_package(blender, blend, blend_sha256, out, take_action, source, guard=None):
    """The packaging core shared by ``fbx-interchange`` (a verified run state) and the focused tests (scratch control
    .blend files, labelled as such in ``source``).  ``out`` must not exist.  The sealed manifest is written LAST (atomic):
    a directory without it is never a package.  ``guard()`` (run commands) re-checks the run after the export."""
    X = _fbxi()
    out = _fresh_dir(out)
    (out / "producer").mkdir()
    (out / "package").mkdir()
    dump(out / "STATUS.json", {"status": "IN_PROGRESS (never a package: no sealed manifest yet)", "started": now(),
                               "state_id": source.get("state_id")})
    pr, stop = out / "producer" / "producer_report.json", out / "producer" / "producer_stop.json"
    t0 = datetime.datetime.now()

    def fail(code, detail):
        dump(out / "STATUS.json", {"status": "FAILED", "code": code, "detail": detail, "at": now(), "state_id": source.get("state_id"),
                                   "note": "no sealed manifest was written: this directory is NOT a package"})
        raise SliceStop(code, detail)
    rc = _stage(blender, "interchange_export", out / "producer" / "export.log", blend=blend, raw_fbx=out / "producer" / "raw_export.fbx",
                fbx=out / "package" / X.PACKAGE_FBX, take=take_action, report=pr, stop_report=stop)
    if rc == 3:
        s_ = load(stop)
        fail(s_["code"], s_["detail"])
    if rc != 0 or not pr.exists():
        fail("PRODUCER_STAGE_FAILED", f"Blender exited {rc}; see {out / 'producer' / 'export.log'}")
    prod = load(pr)
    if guard is not None:
        g = guard()
        if g:
            fail(*g)
    if sha256_file(blend) != blend_sha256 or prod["source_blend_sha256_before"] != blend_sha256 or             prod["fbx_sha256"] != sha256_file(out / "package" / X.PACKAGE_FBX):
        fail("PRODUCER_REPORT_INCONSISTENT", "source/package hashes")
    src = dict(source, action_name=prod["source_action"])
    mf = X.build_manifest(prod, src, now(), sha256_code(FBXI_HELPER), sha256_code(STAGE))
    _fbxi_fault("before_manifest")
    TX.atomic_write_bytes(out / "package" / X.PACKAGE_MANIFEST, json.dumps(mf, indent=1, sort_keys=True).encode("utf-8"))
    summ = {"status": "COMPLETE", "package": str(out / "package"), "state_id": source.get("state_id"),
            "state_role": source.get("state_role"), "semantic_authorization": source.get("semantic_authorization"),
            "take": prod["take"]["name"], "fbx_sha256": prod["fbx_sha256"], "fbx_bytes": prod["fbx_bytes"],
            "raw_exporter_output_bytes": prod["raw_fbx_bytes"], "fbx_curves": prod["fbx_curves"], "fbx_keys_total": prod["fbx_keys_total"],
            "patched_constant_curves": prod["patched_curves"], "manifest_digest": mf["manifest_digest"], "source_blend_unchanged": True,
            "other_actions_in_source_not_exported": prod["other_actions_in_file_not_exported"],
            "seconds": (datetime.datetime.now() - t0).total_seconds(),
            "next": f"experimental-video-action fbx-consume --package {out / 'package'} --out <new dir>"}
    dump(out / "STATUS.json", summ)
    return summ


def cmd_fbx_interchange(a):
    run = Path(a.run).resolve()
    R, man = load_run(run)
    require_no_pending(run)
    refs, raw, kind = read_refs(run)
    sid = refs["head"] if a.state == "head" else R["baseline_state_id"] if a.state == "baseline" else a.state
    rec, rb = verify_state_blend(run, sid, refs, kind)
    before = authority_snapshot(run)

    def guard():
        after = authority_snapshot(run)
        if after != before:
            return "RUN_CHANGED_DURING_EXPORT", json.dumps(sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))[:20])
        if a.state == "head" and read_refs(run)[0]["head"] != sid:
            return "HEAD_MOVED_DURING_EXPORT", sid
        return None
    source = {"kind": "RUN_STATE", "slice": SLICE_ID, "run_id": run.name, "state_id": sid, "state_role": rec["role"],
              "semantic_authorization": rec["semantic_authorization"], "accepted_request_id": rec["accepted_request_id"],
              "is_current_head_at_export": sid == refs["head"], "generation_at_export": refs.get("generation"),
              "blend_rel": rec["blend"], "blend_sha256": rec["blend_sha256"], "value_digest": rec["value_digest"],
              "readback_sha256": sha256_file(run / rec["readback"]), "source_clip_sha256": man["intake"]["clip_sha256"],
              "action_timebase_note": "30 fps source-to-Action mapping of the slice (frozen W124): Action frame f = first "
                                      "segment source frame + f - 1",
              "note": "identity/provenance only: the consumer never opens anything named here"}
    summ = produce_package(R["blender"], run / rec["blend"], rec["blend_sha256"], a.out, f"AnimCV_EXP_{sid[:12]}", source, guard)
    summ["run_authority_files_unchanged"] = True
    print(json.dumps(summ, indent=1))
    return 0


def cmd_fbx_consume(a):
    """CONSUMER: only the package directory is an input.  Nothing of a run is read (the run is not even an argument)."""
    X = _fbxi()
    blender = find_blender(a.blender)
    out = _fresh_dir(a.out)
    blend = out / ("recovered.blend" if a.mode == "adapter" else "raw_import.blend")
    rep, stop = out / "consume_report.json", out / "consume_stop.json"
    t0 = datetime.datetime.now()
    rc = _stage(blender, "interchange_consume", out / "consume.log", package=Path(a.package).resolve(), out_blend=blend, report=rep,
                stop_report=stop, mode=a.mode)
    secs = (datetime.datetime.now() - t0).total_seconds()
    if rc == 3:
        s = load(stop)
        raise SliceStop(s["code"], s.get("detail", ""))
    if rc != 0 or not rep.exists():
        raise SliceStop("CONSUMER_STAGE_FAILED", f"Blender exited {rc}; see {out / 'consume.log'}")
    r = load(rep)
    und = r.get("file_access_audit", {}).get("classified", {}).get("UNDECLARED", [])
    r["seconds"] = secs
    r["verdict"] = "RECOVERED" if a.mode == "adapter" else "RAW_IMPORT_ONLY"
    if und:
        r["verdict"] = "INVALID_UNDECLARED_INPUT_READ"
    dump(rep, r)
    print(json.dumps({k: r.get(k) for k in ("verdict", "label", "action", "take", "recovered_interpolation_census",
                                            "scene_range_restored_from_manifest", "recovered_blend_sha256", "seconds")}, indent=1))
    if und:
        raise SliceStop("INVALID_UNDECLARED_INPUT_READ", json.dumps(und[:10]))
    return 0


def pin_context(run, sid):
    """For an accepted WORLD_POSITION_PIN state: pin source frames -> Action frames, the target and the bound point."""
    sem = state_semantics(run, sid, blend_registry(run))
    if not sem["accepted_request_id"]:
        return None
    prop = load(run / "edits" / sem["accepted_request_id"] / "proposal.json")
    bnd = load(run_paths(run)["bindings"])["bound"][prop["constraint"]["binding_id"]]
    m = {r["source_frame"]: r["action_frame"] for r in load(run_paths(run)["sidecar"])["frame_mapping"]}
    return {"request_id": sem["accepted_request_id"], "source_frames": prop["pin_frames"],
            "action_frames": [m[k] for k in prop["pin_frames"]], "target": prop["constraint"]["target"],
            "bone": bnd["bone"], "point": bnd["point"], "semantic_joint": bnd["semantic_joint"]}


def _pt(fr, bone, point):
    return fr["world"][bone][point]


def compare_interchange(orig, rec, pin=None, tol=FBX_DIAGNOSTIC_TOL_M):
    """EVALUATOR comparison of two fresh-process readbacks (original .blend vs imported .blend) on one frozen grid."""
    import numpy as np
    X = _fbxi()
    mism, rep = [], {"diagnostic_tolerance_m": tol, "object_tolerance": FBXI_OBJECT_TOL, "rest_tolerance_m": FBXI_REST_TOL_M}
    bones = orig["bones"]
    if set(rec["bones"]) != set(bones):
        mism.append("BONE_SET_DIFFERS")
        rep["bone_set"] = {"missing": sorted(set(bones) - set(rec["bones"])), "extra": sorted(set(rec["bones"]) - set(bones))}
    common = [b for b in bones if b in rec["bones"]]

    def err(x, y):
        a = np.array([[x["world"][b][p] for p in ("head", "tail")] for b in common])
        c = np.array([[y["world"][b][p] for p in ("head", "tail")] for b in common])
        d = np.linalg.norm(a - c, axis=-1)
        return d.max(axis=1), float(np.abs(np.array(x["matrix_world"]) - np.array(y["matrix_world"])).max())
    fo = {f["blender_frame"]: f for f in orig["frames"]}
    fr = {f["blender_frame"]: f for f in rec["frames"]}
    if set(fo) != set(fr):
        mism.append("FRAME_SET_DIFFERS")
    ints = sorted(set(fo) & set(fr))
    per_frame, per_bone_i, mw = [], np.zeros(len(common)), 0.0
    worst_i = (0.0, None)
    for f in ints:
        d, m = err(fo[f], fr[f])
        per_bone_i = np.maximum(per_bone_i, d)
        per_frame.append(float(d.max()))
        mw = max(mw, m)
        if d.max() > worst_i[0]:
            worst_i = (float(d.max()), [f, 0.0, common[int(d.argmax())]])
    so = {(s["blender_frame"], s["subframe"]): s for s in orig["subframes"]}
    sr = {(s["blender_frame"], s["subframe"]): s for s in rec["subframes"]}
    subs = sorted(set(so) & set(sr))
    if set(so) != set(sr):
        mism.append("SUBFRAME_SET_DIFFERS")
    by_off, per_bone_s, worst_s, over_s = {}, np.zeros(len(common)), (0.0, None), []
    for k in subs:
        d, m = err(so[k], sr[k])
        per_bone_s = np.maximum(per_bone_s, d)
        mw = max(mw, m)
        by_off.setdefault(k[1], []).append(float(d.max()))
        if d.max() > worst_s[0]:
            worst_s = (float(d.max()), [k[0], k[1], common[int(d.argmax())]])
        if d.max() > tol:
            over_s.append([k[0], k[1]])
    pf = np.array(per_frame) if per_frame else np.zeros(1)
    rep["integer"] = {"frames_compared": len(ints), "max_m": float(pf.max()), "median_m": float(np.median(pf)), "worst_at": worst_i[1],
                      "frames_over_tol": [f for f, v in zip(ints, per_frame) if v > tol]}
    rep["subframe"] = {"samples_compared": len(subs), "max_m": worst_s[0], "worst_at": worst_s[1],
                       "median_m": float(np.median([v for vs in by_off.values() for v in vs])) if by_off else None,
                       "by_offset_max_m": {str(o): max(v) for o, v in sorted(by_off.items())},
                       "samples_over_tol": len(over_s), "frames_with_subframe_over_tol": sorted({f for f, _ in over_s})}
    rep["per_bone_max_m"] = {"integer": {b: float(v) for b, v in zip(common, per_bone_i)},
                             "subframe": {b: float(v) for b, v in zip(common, per_bone_s)}}
    rep["bones_over_tol"] = {"integer": [b for b, v in zip(common, per_bone_i) if v > tol],
                             "subframe": [b for b, v in zip(common, per_bone_s) if v > tol]}
    rep["object_matrix_world_max_abs"] = mw
    if pf.max() > tol:
        mism.append("INTEGER_FRAME_WORLD_DRIFT_OVER_1MM")
    if worst_s[0] > tol:
        mism.append("SUBFRAME_WORLD_DEVIATION_OVER_1MM")
    if mw > FBXI_OBJECT_TOL:
        mism.append("OBJECT_TRANSFORM_DIFFERS")
    sc_o = float(orig["armature_scale"][0])
    rest = 0.0
    for b in common:
        ro, rr = orig["rest"].get(b), rec["rest"].get(b)
        if ro is None or rr is None or ro["parent"] != rr["parent"]:
            rest = float("inf")
            continue
        for k in ("head_local", "tail_local"):
            rest = max(rest, float(np.abs(np.array(ro[k]) - np.array(rr[k])).max()) * sc_o)
    rep["rest_max_m"] = rest
    rep["armature_scale"] = {"original": orig["armature_scale"], "imported": rec["armature_scale"]}
    if rest > FBXI_REST_TOL_M:
        mism.append("REST_GEOMETRY_DIFFERS")
    if max(abs(x - y) for x, y in zip(orig["armature_scale"], rec["armature_scale"])) > 1e-6:
        mism.append("ARMATURE_SCALE_DIFFERS")
    # interpolation: every source-driven (item, group) must be recovered with the source interpolation
    def groups_of(rb):
        g = {}
        for c in rb["fcurve_census"]["fcurves"]:
            k = X.source_group_key(c["data_path"], rb["armature_object"])
            key = f"{k[0].split(':')[0]}|{k[1]}" if k and k[0].startswith("object:") else (f"{k[0]}|{k[1]}" if k else c["data_path"])
            g.setdefault(key, set()).update(c["interpolation"])
        return {k: sorted(v) for k, v in g.items()}
    go, gr = groups_of(orig), groups_of(rec)
    bad = {k: {"source": v, "imported": gr.get(k)} for k, v in go.items() if gr.get(k) != v}
    rep["interpolation"] = {"source_groups": go, "imported_groups_for_source_groups": {k: gr.get(k) for k in go},
                            "imported_undriven_groups": {k: v for k, v in gr.items() if k not in go},
                            "source_census_curves": orig["fcurve_census"]["interpolation_census_curves"],
                            "imported_census_curves": rec["fcurve_census"]["interpolation_census_curves"],
                            "source_fcurves": orig["fcurve_census"]["n_fcurves"], "imported_fcurves": rec["fcurve_census"]["n_fcurves"],
                            "not_recovered": bad}
    if bad:
        mism.append("INTERPOLATION_NOT_RECOVERED")
    rep["actions"] = {"original": orig["actions_in_file"], "imported": rec["actions_in_file"]}
    if len(rec["actions_in_file"]) != 1:
        mism.append("MULTIPLE_ACTIONS_AFTER_IMPORT")
    ko = [orig["fcurve_census"]["key_frame_min"], orig["fcurve_census"]["key_frame_max"]]
    kr = [rec["fcurve_census"]["key_frame_min"], rec["fcurve_census"]["key_frame_max"]]
    rep["key_range"] = {"original": ko, "imported": kr}
    if ko != kr:
        mism.append("KEY_RANGE_CHANGED")
    rep["scene"] = {"original": orig["scene"], "imported": rec["scene"]}
    if abs(orig["scene"]["effective_fps"] - rec["scene"]["effective_fps"]) > 1e-9:
        mism.append("SCENE_FPS_DIFFERS")
    if (orig["scene"]["frame_start"], orig["scene"]["frame_end"]) != (rec["scene"]["frame_start"], rec["scene"]["frame_end"]):
        mism.append("SCENE_FRAME_RANGE_NOT_CARRIED")
    if pin:
        res = {}
        for name, rb in (("original", orig), ("imported", rec)):
            fm = {f["blender_frame"]: f for f in rb["frames"]}
            sm = {(s["blender_frame"], s["subframe"]): s for s in rb["subframes"]}
            t = np.array(pin["target"])
            ri = [float(np.linalg.norm(np.array(_pt(fm[f], pin["bone"], pin["point"])) - t)) for f in pin["action_frames"] if f in fm]
            inner = [k for k in sm if k[0] in pin["action_frames"] and k[0] + 1 in pin["action_frames"]]
            rs = [float(np.linalg.norm(np.array(_pt(sm[k], pin["bone"], pin["point"])) - t)) for k in inner]
            res[name] = {"integer_pin_frames_max_m": max(ri) if ri else None, "subframes_between_pin_frames_max_m": max(rs) if rs else None,
                         "n_integer": len(ri), "n_subframes": len(rs)}
        rep["pin"] = dict(pin, residual=res)
        if (res["imported"]["integer_pin_frames_max_m"] or 0) > tol or (res["imported"]["subframes_between_pin_frames_max_m"] or 0) > tol:
            mism.append("PIN_RESIDUAL_OVER_1MM")
    rep["mismatches"] = mism
    rep["verdict"] = "PASS" if not mism else "FAIL"
    return rep


def evaluate_pair(blender, orig_blend, imported_blend, out, pin=None, label=""):
    """Two FRESH Blender processes (one per .blend) on the frozen grid, then ``compare_interchange``.  Used by
    ``fbx-verify`` and by the W127 harness for the stock / raw A/B on the same grid."""
    subs = ",".join(str(s) for s in FBXI_SUBFRAMES)
    for name, bl, fz in (("original", orig_blend, "1"), ("imported", imported_blend, "0")):
        rc = _stage(blender, "readback", out / f"{name}_readback.log", blend=bl, report=out / f"{name}_readback.json", subframes=subs,
                    frozen_audit=fz)
        if rc != 0:
            raise SliceStop("EVALUATOR_READBACK_FAILED", f"{name}: see {out / (name + '_readback.log')}")
    o, r = load(out / "original_readback.json"), load(out / "imported_readback.json")
    if o["blend_sha256"] != sha256_file(orig_blend) or r["blend_sha256"] != sha256_file(imported_blend):
        raise SliceStop("EVALUATOR_READBACK_NOT_OF_BLEND", label)
    rep = compare_interchange(o, r, pin)
    rep.update({"label": label, "original_blend_sha256": o["blend_sha256"], "imported_blend_sha256": r["blend_sha256"],
                "original_readback_sha256": sha256_file(out / "original_readback.json"),
                "imported_readback_sha256": sha256_file(out / "imported_readback.json"), "grid_subframes": list(FBXI_SUBFRAMES)})
    return rep


def cmd_fbx_verify(a):
    """EVALUATOR: separate from producer and consumer.  The original .blend is read ONLY here, after the import."""
    run = Path(a.run).resolve()
    R, man = load_run(run)
    require_no_pending(run)
    refs, raw, kind = read_refs(run)
    cdir = Path(a.consumed).resolve()
    crep = load(cdir / "consume_report.json")
    blend = cdir / ("recovered.blend" if crep.get("mode") == "adapter" else "raw_import.blend")
    if not blend.exists() or sha256_file(blend) != crep.get("recovered_blend_sha256"):
        raise SliceStop("CONSUMED_OUTPUT_UNVERIFIED", str(blend))
    sid = crep["source_identity"]["state_id"]
    if a.state not in (None, "package"):
        want = refs["head"] if a.state == "head" else R["baseline_state_id"] if a.state == "baseline" else a.state
        if want != sid:
            raise SliceStop("PACKAGE_NOT_OF_THIS_STATE", f"package state {sid} vs requested {want}")
    rec, _ = verify_state_blend(run, sid, refs, kind)
    if rec["blend_sha256"] != crep["source_identity"]["blend_sha256"]:
        raise SliceStop("PACKAGE_NOT_OF_THIS_STATE", "the package was produced from another .blend")
    out = _fresh_dir(a.out)
    pin = pin_context(run, sid)
    rep = evaluate_pair(R["blender"], run / rec["blend"], blend, out, pin, label=crep.get("label", ""))
    rep.update({"state_id": sid, "state_role": rec["role"], "semantic_authorization": rec["semantic_authorization"],
                "is_current_head": sid == refs["head"], "consume_mode": crep.get("mode"), "consume_report_sha256": sha256_file(cdir / "consume_report.json"),
                "consumer_file_access": crep.get("file_access_audit", {}).get("classified"),
                "interchange_verdict": ("INTERCHANGE_PASS" if rep["verdict"] == "PASS" and crep.get("mode") == "adapter" and
                                        not (crep.get("file_access_audit", {}).get("classified", {}).get("UNDECLARED")) else "FAIL")})
    dump(out / "fbx_verify_report.json", rep)
    print(json.dumps({k: rep.get(k) for k in ("state_id", "state_role", "semantic_authorization", "consume_mode", "verdict", "interchange_verdict",
                                              "mismatches")} | {"integer_max_m": rep["integer"]["max_m"], "subframe_max_m": rep["subframe"]["max_m"],
                                                                "pin": (rep.get("pin") or {}).get("residual")}, indent=1, default=str))
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
    xi = s.add_parser("fbx-interchange", help="EXPERIMENTAL opt-in (W127): export ONE verified published state as a NEW FBX "
                                             "interchange package (read-only on the run; the stock `fbx` command is unchanged)")
    xi.add_argument("--run", required=True)
    xi.add_argument("--state", default="head", help="head | baseline | <published state id>")
    xi.add_argument("--out", required=True, help="NEW output directory (never overwritten)")
    xc = s.add_parser("fbx-consume", help="EXPERIMENTAL opt-in (W127): source-free Blender consumer; reads ONLY the package directory")
    xc.add_argument("--package", required=True, help="the package/ directory written by fbx-interchange")
    xc.add_argument("--out", required=True, help="NEW output directory (never overwritten)")
    xc.add_argument("--mode", choices=("adapter", "raw"), default="adapter",
                    help="adapter: package-aware recovery (default); raw: stock importer only (A/B, nothing recovered)")
    xc.add_argument("--blender")
    xv = s.add_parser("fbx-verify", help="EXPERIMENTAL opt-in (W127): separate evaluator: original vs consumed .blend in fresh processes")
    xv.add_argument("--run", required=True)
    xv.add_argument("--consumed", required=True, help="the fbx-consume output directory")
    xv.add_argument("--state", default="package", help="package (default: the state the package declares) | head | baseline | <id>")
    xv.add_argument("--out", required=True, help="NEW output directory (never overwritten)")
    t = s.add_parser("status", help="print the authoritative head, lineage, .blend registry, pending transactions and recovery need")
    t.add_argument("--run", required=True)
    rc = s.add_parser("recover", help="complete (roll forward) or roll back interrupted transactions; never invents accepted state")
    rc.add_argument("--run", required=True)
    ad = s.add_parser("adopt", help="explicitly upgrade a W124 run to sealed transactional refs (non-destructive; legacy refs kept)")
    ad.add_argument("--run", required=True)
    return p


def dispatch(args):
    try:
        if args.exp_action == "build":
            args.blender_exe = find_blender(args.blender)
            return cmd_build(args)
        return {"propose": cmd_propose, "commit": cmd_commit, "revert": cmd_revert, "review": cmd_review, "fbx": cmd_fbx,
                "status": cmd_status, "recover": cmd_recover, "adopt": cmd_adopt, "fbx-interchange": cmd_fbx_interchange,
                "fbx-consume": cmd_fbx_consume, "fbx-verify": cmd_fbx_verify}[args.exp_action](args)
    except SliceStop as e:
        print(f"[experimental-video-action] STOP {e.code}: {e.detail}", file=sys.stderr)
        return 2
    except TX.InjectedFault as e:     # fault injection (tests / evidence harness only)
        print(f"[experimental-video-action] STOP INJECTED_FAULT: {e}", file=sys.stderr)
        return 3


def main(argv=None):
    ap = argparse.ArgumentParser(prog="experimental-video-action")
    sub = ap.add_subparsers(dest="command", required=True)
    add_subparser(sub)
    return dispatch(ap.parse_args(["experimental-video-action"] + list(argv if argv is not None else sys.argv[1:])))


if __name__ == "__main__":
    raise SystemExit(main())
