"""Batch 124 focused tests for the EXPERIMENTAL opt-in video -> editable Action -> WORLD_POSITION_PIN slice.

Test kinds (reported separately in the W124 Worklog):
  * unit / mock  - pure Python; the "Blender evaluation" is a Python FK mock (`fk_eval`), labelled as such.
  * blender      - real Blender 5.1 subprocess on the real BaseRig (skipped if Blender is absent).
No GT, no real video and no network are used here; real-video replays are run by the W124 harness, not by pytest.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from app import experimental_video_action as E  # noqa: E402
import w116_semantic_pin_base as S  # noqa: E402
import w118_transition_edit_base as XB  # noqa: E402

BLENDER = os.environ.get("ANIMCV_BLENDER", "C:/Program Files/Blender Foundation/Blender 5.1/blender.exe")
HAS_BLENDER = Path(BLENDER).exists()
# Structure digest of every pre-existing motion-tool subcommand (computed from the parser at the starting commit 0d6435a).
DEFAULT_PARSER_DIGEST = "ca4f210541a85710e22b69679f912c9adb4e221a3a0e4c3d7e214405b994758e"


# ------------------------------------------------------------------------------------------ default isolation ---
def _parser_struct(p):
    out = {}
    for name, sp in sorted(p._subparsers._group_actions[0].choices.items()):
        out[name] = sorted([a.dest, repr(a.default), a.required, getattr(a.type, "__name__", repr(a.type)), repr(a.choices),
                            repr(a.nargs)] for a in sp._actions)
    return out


def test_default_cli_commands_unchanged_and_experimental_is_additive():
    from app.cli import build_parser
    st = _parser_struct(build_parser())
    assert "experimental-video-action" in st
    st.pop("experimental-video-action")
    assert len(st) == 35
    assert hashlib.sha256(json.dumps(st, sort_keys=True).encode()).hexdigest() == DEFAULT_PARSER_DIGEST


def test_cli_import_and_default_help_do_not_load_experimental_dependencies():
    code = ("import sys; sys.path.insert(0, 'src'); from app.cli import build_parser; build_parser(); "
            "bad=[m for m in ('cv2','torch','bpy','numpy','w116_semantic_pin_base','w121_independent_transfer','matplotlib') if m in sys.modules];"
            "print(','.join(bad))")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == ""


def test_gui_module_does_not_reference_experimental_slice():
    gui = (ROOT / "src" / "app" / "gui.py").read_text(encoding="utf-8")
    assert "experimental" not in gui.lower()


def test_frozen_code_identity_and_tamper_detection():
    rep = E.frozen_code_report()
    assert all(v["ok"] for v in rep.values()), {k: v for k, v in rep.items() if not v["ok"]}
    exp = {k: v["expected"] for k, v in rep.items()}
    exp["scripts/w116_semantic_pin_base.py"] = "0" * 64
    assert not E.frozen_code_report(exp)["scripts/w116_semantic_pin_base.py"]["ok"]


def test_historical_worklogs_untouched_by_slice():
    """No historical Worklog is referenced for writing; the module only writes inside a new run directory."""
    src = (ROOT / "src" / "app" / "experimental_video_action.py").read_text(encoding="utf-8")
    assert "docs/" not in src.replace("docs/EXPERIMENTAL_VIDEO_ACTION_WORLD_PIN.md", "")


# ----------------------------------------------------------------------------------------------- A. intake ---
def _write_clip(path, fps, n=40, size=(64, 48), seed=0):
    import cv2
    rng = np.random.default_rng(seed)
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    for _ in range(n):
        wr.write(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8))
    wr.release()
    return path


REC = {"title": "synthetic", "license": "CC0", "author": "test", "source_uri": "file://synthetic"}


def test_intake_accepts_authentic_30fps_and_maps_time(tmp_path):
    p = _write_clip(tmp_path / "a.avi", 30.0)
    man = E.intake(p, E.sha256_file(p), 30, [5, 20], REC)
    assert man["container_fps"] == 30.0 and man["segment_source_frames"] == [5, 20]
    assert man["source_timestamps_s"][0] == 5 / 30.0 and len(man["source_timestamps_s"]) == 16
    assert man["video_subject_identity"].startswith("UNKNOWN")


@pytest.mark.parametrize("case,code", [
    ("changed_byte", "BLOCKED_SOURCE_IDENTITY"), ("wrong_declared_fps", "UNSUPPORTED_FPS_MISMATCH"),
    ("unsupported_cadence", "UNSUPPORTED_CADENCE"), ("missing", "DATA_BLOCKED_SOURCE_MISSING"),
    ("segment", "UNSUPPORTED_SEGMENT"), ("provenance", "BLOCKED_SOURCE_PROVENANCE_INCOMPLETE")])
def test_intake_stops(tmp_path, case, code):
    p = _write_clip(tmp_path / "a.avi", 25.0 if case == "unsupported_cadence" else 30.0)
    sha = E.sha256_file(p)
    rec, fps, seg = dict(REC), 25.0 if case == "unsupported_cadence" else 30.0, [0, 10]
    if case == "changed_byte":
        b = bytearray(p.read_bytes())
        b[len(b) // 2] ^= 0xFF
        p.write_bytes(bytes(b))
    if case == "wrong_declared_fps":
        fps = 29.97
    if case == "missing":
        p = tmp_path / "nope.avi"
    if case == "segment":
        seg = [10, 400]
    if case == "provenance":
        rec.pop("license")
    with pytest.raises(E.SliceStop) as ei:
        E.intake(p, sha, fps, seg, rec)
    assert ei.value.code == code


def test_perception_asset_and_runtime_stops(tmp_path):
    with pytest.raises(E.SliceStop) as ei:
        E.check_perception_assets({"checkpoint": str(tmp_path / "missing.pt"), "weights_dir": str(tmp_path), "gate_report": "x", "lock": "y",
                                   "frozen_code_dir": str(tmp_path)})
    assert ei.value.code == "DATA_BLOCKED_MISSING_CHECKPOINT"
    ck = tmp_path / "s0.pt"
    ck.write_bytes(b"not the frozen checkpoint")
    with pytest.raises(E.SliceStop) as ei:
        E.check_perception_assets({"checkpoint": str(ck), "weights_dir": str(tmp_path), "gate_report": str(ck), "lock": str(ck),
                                   "frozen_code_dir": str(tmp_path)})
    assert ei.value.code in ("BLOCKED_FROZEN_PERCEPTION_IDENTITY", "DATA_BLOCKED_MISSING_PERCEPTION_ASSET")
    with pytest.raises(E.SliceStop) as ei:
        E.find_blender(str(tmp_path / "no_blender.exe"))
    assert ei.value.code == "BLOCKED_BLENDER_UNAVAILABLE"


def test_missing_gt_free_observation_is_blocked(tmp_path):
    np.savez(tmp_path / "o.npz", landmarks=np.zeros((3, 17, 3)), predictions=np.zeros((3, 17, 3)))
    with pytest.raises(E.SliceStop) as ei:
        E.perception_frames(tmp_path / "o.npz", [0, 2])
    assert ei.value.code == "DATA_BLOCKED_MISSING_OBSERVATION"


def test_unsupported_frames_are_reported_not_fabricated():
    z = {"predictions": np.zeros((4, 17, 3))}
    present = np.array([True, False, True, True])
    cases, values, unsupported, _ = E.source_to_rig(z, present, [1, 1], 30.0)
    assert values is None and cases == [] and unsupported == [{"source_frame": 1, "reason": "NO_GT_FREE_SUBJECT_OBSERVATION"}]


def test_time_mapping_and_unknown_frame():
    sidecar = {"frame_mapping": [{"source_frame": 79 + i, "action_frame": 1 + i} for i in range(3)]}
    rb = {"frames": [{"blender_frame": f, "world": {}, "matrix_world": [], "stored_wxyz": {}} for f in (1, 2, 3)]}
    out = E.readback_by_source(rb, sidecar)
    assert sorted(out) == [79, 80, 81] and out[80]["blender_frame"] == 2
    rb["frames"].append({"blender_frame": 4})
    with pytest.raises(E.SliceStop) as ei:
        E.readback_by_source(rb, sidecar)
    assert ei.value.code == "INVALID_TIME_MAPPING"


# ------------------------------------------------------------------- C. synthetic rig + mock evaluator (unit) ---
def _mat(R, t):
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = t
    return M.tolist()


def syn_rig():
    """spine_01 (root, +Z) -> upperarm_l -> lowerarm_l (both +X); bone Y along the bone (Blender convention)."""
    Rz = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)
    Rx = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], float)
    bones = {"spine_01": {"parent": None, "matrix_local": _mat(Rz, [0, 0, 0]), "head": [0, 0, 0], "tail": [0, 0, 0.5]},
             "upperarm_l": {"parent": "spine_01", "matrix_local": _mat(Rx, [0, 0, 0.5]), "head": [0, 0, 0.5], "tail": [0.3, 0, 0.5]},
             "lowerarm_l": {"parent": "upperarm_l", "matrix_local": _mat(Rx, [0.3, 0, 0.5]), "head": [0.3, 0, 0.5], "tail": [0.55, 0, 0.5]}}
    return S.Rig({"rig_id": "SYN", "bones": bones, "object": {"location": [0, 0, 0], "scale": [1, 1, 1], "rest_rotation_wxyz": [1, 0, 0, 0]}})


CH = ["__object__", "spine_01", "upperarm_l", "lowerarm_l"]
FRAMES = list(range(10, 30))


def syn_state():
    vals = {c: [] for c in CH}
    for f in FRAMES:
        vals["__object__"].append([0, 0, 0, 1.0])
        vals["spine_01"].append(S.q_axis_angle([1, 0, 0], math.radians(15) * math.sin(math.pi * f / 20)).tolist())
        vals["upperarm_l"].append([0, 0, 0, 1.0])
        vals["lowerarm_l"].append(S.q_axis_angle([0, 0, 1], math.radians(60)).tolist())
    return S.make_state(None, "SYN", FRAMES, CH, vals, {"kind": "ROOT_SYNTHETIC"}, {"kind": "SYNTHETIC_FIXTURE"})


def fk_eval(rig, st, tamper=None):
    """MOCK evaluator (Python FK, not Blender): same record layout as the Blender readback."""
    out = {}
    for i, k in enumerate(st["frames"]):
        v = S.frame_vals(st, i)
        if tamper and k in tamper:
            v = dict(v, **tamper[k](v))
        fk = rig.fk(v)
        M = np.eye(4)
        M[:3, :3] = fk["Robj"]
        out[k] = {"frame_index": k, "world": {b: {"head": rig.world(fk, b, "head").tolist(), "tail": rig.world(fk, b, "tail").tolist()}
                                              for b in rig.bones},
                  "stored_wxyz": {c: [float(q[3]), float(q[0]), float(q[1]), float(q[2])] for c, q in v.items()}, "matrix_world": M.tolist()}
    return out


REST = {"spine_01": {"length": 0.5, "parent": None}, "upperarm_l": {"length": 0.3, "parent": "spine_01"},
        "lowerarm_l": {"length": 0.25, "parent": "upperarm_l"}}
CANDS = {"left_wrist": [("lowerarm_l", "tail")], "left_hand": [("lowerarm_l", "tail"), ("lowerarm_l", "head")],
         "right_wrist": [("upperarm_l", "tail")]}


def bindings(rig, st, ev):
    ok, failed = {}, {}
    rest = {b: {"head_local": rig.bones[b]["M"][:3, 3].tolist(), "tail_local": (rig.bones[b]["M"] @ np.array([0, rig.bones[b]["len"], 0, 1]))[:3].tolist(),
                **REST[b]} for b in rig.bones}
    for j, c in CANDS.items():
        r = S.verify_binding(j, c, "SYN", "restdigest", rest, [ev[k] for k in st["frames"]], st["channels"], rig)
        (ok.__setitem__(r["binding_id"], r) if r["status"] == "BOUND" else failed.__setitem__(j, r))
    return ok, failed


def request(st, **kw):
    r = {"schema": "animcv_exp_world_pin_request_v1", "request_id": "t", "parent_state_id": st["state_id"], "type": "WORLD_POSITION_PIN",
         "semantic_joint": "left_wrist", "coordinate_frame": "SCENE_WORLD", "unit": "m", "frames": [14, 20],
         "target": {"kind": "CAPTURED_FROM_EVALUATED_STATE", "capture_frame": 14}, "authorized_channels": ["lowerarm_l", "upperarm_l"]}
    r.update(kw)
    return r


def propose(req, tamper=None):
    rig, st = syn_rig(), syn_state()
    ev = fk_eval(rig, st)
    ok, failed = bindings(rig, st, ev)
    rec, cand, X = E.propose_core(req if not callable(req) else req(st, ok), st, ev, rig, REST, 1.0, ok, failed, ["spine_01"],
                                  lambda c: fk_eval(rig, c, tamper))
    return rec, cand, X, st, ev, ok


def test_feasible_nonzero_pin_accepted_with_truthful_footprint():
    rec, cand, X, st, ev, ok = propose(lambda st, ok: request(st))
    assert rec["decision"]["decision"] == "ACCEPT", rec["decision"]
    fp = rec["footprint"]
    assert fp["frames_changed"] and set(fp["frames_changed"]) <= set(range(14, 21)) and 14 not in fp["frames_changed"]
    assert set(fp["channel_frames"]) == {"upperarm_l", "lowerarm_l"}
    assert rec["hard_pin_residual_max_m"] <= 1e-3 and rec["control_residual_max_m"] > 1e-3
    assert rec["display_check"]["ok"]
    # unchanged outside the pin frames, spine/root untouched
    for k in FRAMES:
        if k not in range(14, 21):
            assert X[k] == ev[k]
        assert X[k]["stored_wxyz"]["spine_01"] == ev[k]["stored_wxyz"]["spine_01"]


def test_noop_pin_is_trivially_satisfied_not_evidence():
    rec, cand, X, *_ = propose(lambda st, ok: request(st, frames=[14, 14]))
    assert rec["decision"] == {"decision": "ACCEPT", "failure_class": "NONE", "reasons": ["TRIVIALLY_SATISFIED_NOT_EVIDENCE"]}
    assert rec["footprint"]["frames_changed"] == []


@pytest.mark.parametrize("mut,cls,reason", [
    (lambda st, ok: request(st, target={"kind": "CAPTURED_PLUS_DECLARED_OFFSET", "capture_frame": 14, "offset_m": [0, 0, 1.0]}),
     "INFEASIBLE_REQUEST", "UNREACHABLE"),
    (lambda st, ok: request(st, parent_state_id="0" * 64), "CONTRACT_VIOLATION", "STALE_PARENT"),
    (lambda st, ok: request(st, binding_id=[b for b, r in ok.items() if r["semantic_joint"] == "right_wrist"][0]),
     "UNSUPPORTED_BINDING", "BINDING_JOINT_MISMATCH"),
    (lambda st, ok: request(st, semantic_joint="left_ankle"), "UNSUPPORTED_BINDING", "BINDING_UNDECLARED_JOINT"),
    (lambda st, ok: request(st, semantic_joint="left_hand"), "UNSUPPORTED_BINDING", "BINDING_AMBIGUOUS"),
    (lambda st, ok: request(st, semantic_joint="right_wrist"), "UNSUPPORTED_BINDING", "JOINT_NOT_SUPPORTED_BY_SLICE"),
    (lambda st, ok: request(st, target={"kind": "CAPTURED_PLUS_DECLARED_OFFSET", "capture_frame": 14, "offset_m": [float("nan"), 0, 0]}),
     "CONTRACT_VIOLATION", "INVALID_NUMERIC_TARGET"),
    (lambda st, ok: request(st, target={"kind": "ABSOLUTE", "capture_frame": 14}), "CONTRACT_VIOLATION", "TARGET_KIND_UNSUPPORTED"),
    (lambda st, ok: request(st, coordinate_frame="ARMATURE_LOCAL"), "CONTRACT_VIOLATION", "COORDINATE_FRAME_UNSUPPORTED"),
    (lambda st, ok: request(st, unit="cm"), "CONTRACT_VIOLATION", "UNIT_UNSUPPORTED"),
    (lambda st, ok: request(st, authorized_channels=["lowerarm_l", "spine_01", "upperarm_l"]), "CONTRACT_VIOLATION",
     "UNAUTHORIZED_DOF_FOR_SLICE"),
    (lambda st, ok: request(st, frames=[25, 40]), "CONTRACT_VIOLATION", "FRAME_NOT_IN_STATE"),
])
def test_invalid_requests_rejected_without_mutation(mut, cls, reason):
    rec, cand, X, st, ev, ok = propose(mut)
    assert rec["decision"]["decision"] == "REJECT"
    assert reason in rec["decision"]["reasons"] or reason in [r for _, r in rec["validation_errors"]], rec
    if reason != "UNREACHABLE":
        assert cand is None and rec["candidate"] is None
    else:
        assert rec["decision"]["failure_class"] == cls


def test_hidden_out_of_pin_change_is_rejected():
    tamper = {25: lambda v: {"upperarm_l": S.q_mul(v["upperarm_l"], S.q_axis_angle([1, 0, 0], math.radians(1)))}}
    rec, *_ = propose(lambda st, ok: request(st), tamper=tamper)
    assert rec["decision"]["decision"] == "REJECT"
    assert "CHANGE_OUTSIDE_AUTOMATIC_BUDGET" in rec["decision"]["reasons"]


def test_unauthorised_bone_change_by_evaluator_is_rejected():
    tamper = {16: lambda v: {"spine_01": S.q_mul(v["spine_01"], S.q_axis_angle([1, 0, 0], math.radians(1)))}}
    rec, *_ = propose(lambda st, ok: request(st), tamper=tamper)
    assert rec["decision"]["decision"] == "REJECT"
    assert {"UNAUTHORIZED_CHANNEL_CHANGE", "POINT_OUTSIDE_ARM_DOMAIN_CHANGED_VS_CONTROL"} & set(rec["decision"]["reasons"])


def test_footprint_display_tamper_detected():
    rec, cand, X, st, ev, ok = propose(lambda st, ok: request(st))
    spec = copy.deepcopy(rec["display"])
    k = str(rec["footprint"]["frames_changed"][0])
    spec["cells"][k] = "PIN_UNCHANGED"
    assert not XB.check_display(spec, X, ev, st["frames"], rec["pin_frames"], cand["state_id"])["ok"]
    spec2 = dict(rec["display"], status="ACCEPTED")
    assert not XB.check_display(spec2, X, ev, st["frames"], rec["pin_frames"], cand["state_id"], decision="REJECT")["ok"]


def test_store_accept_reject_stale_and_exact_undo(tmp_path):
    rec, cand, X, st, ev, ok = propose(lambda st, ok: request(st))
    run = tmp_path / "run"
    (run / "lineage").mkdir(parents=True)
    ps = E.PersistentStore(run)
    ps.store.put(st)
    ps.store.init_ref("head", st["state_id"])
    ps.save(ps.store.log)
    ps = E.PersistentStore(run)                                    # reload from disk
    r = ps.store.commit("head", st["state_id"], cand, {"decision": "REJECT", "reasons": ["USER_REJECTED"]})
    assert r["head_after"] == st["state_id"]                       # rejected candidate leaves the head unchanged
    r = ps.store.commit("head", st["state_id"], cand, rec["decision"])
    assert r["head_after"] == cand["state_id"]
    r = ps.store.commit("head", st["state_id"], cand, rec["decision"])
    assert r["reasons"] == ["STALE_PARENT"] and r["head_after"] == cand["state_id"]
    rv, rr = ps.store.revert("head")
    assert rr["values_equal_restored"] and E.canon(rv["values"]) == E.canon(st["values"])
    assert fk_eval(syn_rig(), rv) == ev                            # mock evaluation bit-equal after undo
    ps.save(ps.store.log)
    assert E.PersistentStore(run).store.refs["head"] == rv["state_id"]
    tampered = dict(cand, values=dict(cand["values"], lowerarm_l=cand["values"]["upperarm_l"]))
    assert not S.verify_state(tampered)


# --------------------------------------------------------------------------- real Blender + real BaseRig ---
@pytest.mark.skipif(not HAS_BLENDER, reason="Blender 5.1 not available")
def test_blender_quaternion_and_time_mapping_on_real_baserig(tmp_path):
    """REAL Blender: a 3-frame state with known quaternions is keyed into a .blend, re-opened and read back."""
    import w116_semantic_pin_contract as C
    q = [S.q_axis_angle([0, 0, 1], math.radians(a)).tolist() for a in (0.0, 10.0, 25.0)]
    vals = {c: [[0, 0, 0, 1.0]] * 3 for c in C.REAL_CH}
    vals["lowerarm_l"] = q
    st = S.make_state(None, "BaseRig", [100, 101, 102], C.REAL_CH, vals, {"kind": "TEST"}, {"kind": "TEST"})
    rec, b, rb = E.build_state_blend(tmp_path, BLENDER, st, "t", "t", {"test": True}, E.PROJECT_ROOT / E.RIG_FILES["rest"][0],
                                     E.PROJECT_ROOT / E.RIG_FILES["fbx"][0])
    assert rb["scene"]["effective_fps"] == 30.0 and (rb["scene"]["frame_start"], rb["scene"]["frame_end"]) == (1, 3)
    assert [f["blender_frame"] for f in rb["frames"]] == [1, 2, 3]
    for i, f in enumerate(rb["frames"]):
        x, y, z, w = q[i]
        assert np.allclose(f["stored_wxyz"]["lowerarm_l"], [w, x, y, z], atol=1e-7)
    assert rb["fcurve_census"]["interpolation_census_curves"] == {"CONSTANT": 32}
    assert rec["readback_equals_in_process"]
    sub = [s for s in rb["subframes"] if s["blender_frame"] == 2]
    assert all(s["world"] == rb["frames"][1]["world"] for s in sub)   # CONSTANT keys hold between frames
    assert rb["provenance_object"] == {"test": True}
    with pytest.raises(E.SliceStop):
        E.build_state_blend(tmp_path, BLENDER, st, "t", "t", {"test": True}, E.PROJECT_ROOT / E.RIG_FILES["rest"][0],
                            E.PROJECT_ROOT / E.RIG_FILES["fbx"][0])   # never overwrites an existing .blend
