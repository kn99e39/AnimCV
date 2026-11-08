"""Batch 127 focused tests: OPT-IN FBX continuous-time interchange of the verified editable Action (EXPERIMENTAL).

Test kinds (reported separately in the W127 Worklog):
  * unit     - pure Python (manifest seal, access classification, source-path grouping, producer directive rule, W110 parity).
  * blender  - REAL Blender 5.1 on the REAL W105 BaseRig with SYNTHETIC VALUES (the W125/W126 8-frame fixture; not video, not
               a quality claim).  Producer / consumer / evaluator run as separate processes through the user CLI
               (``python -m app.cli experimental-video-action fbx-interchange | fbx-consume | fbx-verify``) or through the shared
               packaging core (scratch control .blend files, labelled as such).
The control matrix and every expected outcome below were frozen in the W127 declaration BEFORE these tests were first run
(only the C1 user path and its raw A/B were seen during development; disclosed in the Worklog).  They are not relaxed after a
failure.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT / "scripts", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from app import experimental_video_action as E  # noqa: E402
from app import experimental_action_txn as TX  # noqa: E402
import blender.experimental_fbx_interchange as X  # noqa: E402

BLENDER = os.environ.get("ANIMCV_BLENDER", "C:/Program Files/Blender Foundation/Blender 5.1/blender.exe")
HAS_BLENDER = Path(BLENDER).exists()
needs_blender = pytest.mark.skipif(not HAS_BLENDER, reason="Blender 5.1 not available")
ENV = dict(os.environ, PYTHONPATH=str(ROOT / "src"), ANIMCV_BLENDER=BLENDER)
for _k in (TX.FAULT_ENV, X.FAULT_ENV):
    ENV.pop(_k, None)
CONTROLS = ROOT / "tests" / "w127_fbx_controls_blender.py"
STARTING_COMMIT = "4229fe7eca6ce79bbb39a34678815d04cb45e177"

# Frozen expected classes (W127 declaration).  Raw = the package FBX through the stock importer alone.
RAW_HELD = ["SUBFRAME_WORLD_DEVIATION_OVER_1MM", "INTERPOLATION_NOT_RECOVERED", "SCENE_FRAME_RANGE_NOT_CARRIED"]
PACKAGE_NEGATIVES = {
    "missing_manifest": "PACKAGE_INCOMPLETE", "manifest_edit_no_reseal": "MANIFEST_SEAL_INVALID",
    "fbx_byte_flip": "FBX_HASH_MISMATCH", "stale_fbx_of_other_package": "FBX_HASH_MISMATCH",
    "multitake_fbx_resealed": "TAKE_NOT_UNIQUE", "take_renamed_resealed": "TAKE_MISMATCH",
    "fps25_resealed": "TIMEBASE_MISMATCH", "unknown_rig_declared_resealed": "UNSUPPORTED_RIG",
    "wrong_rig_fbx_consistent_resealed": "RIG_MISMATCH", "producer_version_resealed": "UNSUPPORTED_PRODUCER_BLENDER_VERSION",
    "directive_edit_resealed": "METADATA_FBX_DISAGREE", "fbx_flag_reverted_resealed": "METADATA_FBX_DISAGREE",
    "truncated_fbx_resealed": "FBX_PARSE_ERROR", "consumer_crash_after_import": "CONSUMER_STAGE_FAILED"}
UNSUPPORTED_SOURCES = {"perkey": "UNSUPPORTED_INTERPOLATION_MIXTURE", "axis": "UNSUPPORTED_INTERPOLATION_MIXTURE",
                       "bezier": "UNSUPPORTED_INTERPOLATION_BEZIER", "wrong_rig": "UNSUPPORTED_RIG"}


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def cli(*args, env=None, wait=True):
    e = dict(ENV, **(env or {}))
    cmd = [sys.executable, "-X", "utf8", "-m", "app.cli", "experimental-video-action", *[str(a) for a in args]]
    if not wait:
        return subprocess.Popen(cmd, cwd=ROOT, env=e, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return subprocess.run(cmd, cwd=ROOT, env=e, capture_output=True, text=True, timeout=900)


def blender_helper(*args):
    r = subprocess.run([BLENDER, "-b", "--python-exit-code", "1", "-P", str(CONTROLS), "--", *[str(a) for a in args]],
                       cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]


def scratch_source(tag):
    return {"kind": "SCRATCH_CONTROL_BLEND (W127 focused test; not a run state)", "state_id": None, "control": tag,
            "state_role": "control", "semantic_authorization": "NOT_APPLICABLE (synthetic control)"}


def package_of(blend, out, tag):
    return E.produce_package(BLENDER, Path(blend), E.sha256_file(blend), Path(out), f"AnimCV_EXP_{tag}", scratch_source(tag))


def consume(pkg, out, mode="adapter", env=None):
    return cli("fbx-consume", "--package", pkg, "--out", out, "--mode", mode, env=env)


def evaluate(orig_blend, consumed_dir, out):
    cd = Path(consumed_dir)
    crep = load(cd / "consume_report.json")
    blend = cd / ("recovered.blend" if crep["mode"] == "adapter" else "raw_import.blend")
    out = Path(out)
    out.mkdir(parents=True)
    return E.evaluate_pair(BLENDER, Path(orig_blend), blend, out, None, crep["label"])


# ------------------------------------------------------------------------------------------------------ unit ---
def test_manifest_seal_and_edit_detection():
    m = X.seal({"schema": X.PACKAGE_SCHEMA, "a": 1, "b": {"c": [1, 2]}})
    assert X.seal_ok(m) and X.verify_manifest_bytes(json.dumps(m).encode())["a"] == 1
    m2 = dict(m, a=2)
    assert not X.seal_ok(m2)
    with pytest.raises(X.InterchangeStop) as ei:
        X.verify_manifest_bytes(json.dumps(m2).encode())
    assert ei.value.code == "MANIFEST_SEAL_INVALID"
    with pytest.raises(X.InterchangeStop) as ei:
        X.verify_manifest_bytes(b'{"schema": "other"}')
    assert ei.value.code == "MANIFEST_SCHEMA_UNSUPPORTED"


def test_source_group_key_and_producer_directive_rule():
    assert X.source_group_key('pose.bones["upperarm_l"].rotation_quaternion', "root") == ("bone:upperarm_l", "rotation")
    assert X.source_group_key("rotation_quaternion", "root") == ("object:root", "rotation")
    assert X.source_group_key('pose.bones["a"]["custom"]', "root") is None
    assert X.source_group_key("data.bbone_x", "root") is None
    fm = {"curves": [{"uid": 1, "status": "OK", "model_type": "LimbNode", "model_name": "a", "group": "rotation"},
                     {"uid": 2, "status": "OK", "model_type": "LimbNode", "model_name": "a", "group": "location"},
                     {"uid": 3, "status": "OK", "model_type": "LimbNode", "model_name": "b", "group": "rotation"},
                     {"uid": 4, "status": "OK", "model_type": "Null", "model_name": "root", "group": "rotation"},
                     {"uid": 5, "status": "OK", "model_type": "LimbNode", "model_name": "c", "group": "rotation"}]}
    d = X.producer_directives(fm, {"bone:a|rotation": ["CONSTANT"], "bone:b|rotation": ["LINEAR"],
                                   "object:root|rotation": ["CONSTANT"], "bone:c|rotation": ["CONSTANT", "LINEAR"]}, "root")
    assert [d[u][0] for u in (1, 2, 3, 4, 5)] == ["CONSTANT", "UNDRIVEN", "LINEAR", "CONSTANT", "UNSUPPORTED"]


def test_access_classification(tmp_path):
    pk = tmp_path / "package"
    pk.mkdir()
    ev = [[str(pk / X.PACKAGE_MANIFEST), "r"], [str(pk / X.PACKAGE_FBX), "rb"], [str(pk / "other.json"), "r"],
          [str(tmp_path / "out" / "recovered.blend"), "w"], [str(ROOT / "src" / "blender" / "x.py"), "r"],
          [str(ROOT / "src" / "blender" / "x.json"), "r"], [str(tmp_path / "run" / "baseline.blend"), "rb"]]
    c = X.classify_access(ev, pk, [tmp_path / "out"], [ROOT / "src"], [tmp_path / "blender_install"])
    assert len(c["DECLARED_INPUT"]) == 2 and len(c["OUTPUT"]) == 1 and len(c["CODE"]) == 1
    assert len(c["UNDECLARED"]) == 3          # another package file, a json under src, a .blend elsewhere


# ---------------------------------------------------------------------------------------- real Blender fixture ---
@pytest.fixture(scope="session")
def fx(tmp_path_factory):
    """W125/W126 8-frame SYNTHETIC-VALUE fixture on the REAL BaseRig; P1 (nonzero left-wrist pin) accepted -> head."""
    if not HAS_BLENDER:
        pytest.skip("Blender 5.1 not available")
    import test_experimental_action_txn as T
    base = tmp_path_factory.mktemp("w127") / "base"
    v0 = T.build_fixture_run(base)
    pinned = base.parent / "pinned"
    shutil.copytree(base, pinned)
    r = cli("commit", "--run", pinned, "--request-id", "P1", "--decision", "accept")
    assert r.returncode == 0, r.stderr
    # POST-OUTCOME FIX (W127, disclosed): C1 reverts the shared pinned run, so C8 copies this untouched snapshot instead
    pristine = base.parent / "pinned_pristine_for_repeat"
    shutil.copytree(pinned, pristine)
    return {"base": base, "pinned": pinned, "pinned_pristine": pristine, "v0": v0, "head": E.read_refs(pinned)[0]["head"]}


@pytest.fixture(scope="session")
def head_pkg(fx, tmp_path_factory):
    out = tmp_path_factory.mktemp("w127pkg") / "head_export"
    r = cli("fbx-interchange", "--run", fx["pinned"], "--state", "head", "--out", out)
    assert r.returncode == 0, r.stderr
    return out


@pytest.fixture(scope="session")
def controls(fx, tmp_path_factory):
    d = tmp_path_factory.mktemp("w127ctl")
    src = fx["base"] / "baseline" / "baseline.blend"
    out = {}
    for v in ("linear", "mixed", "perkey", "axis", "bezier", "extra_action", "wrong_rig"):
        out[v] = d / f"{v}.blend"
        blender_helper("variant", src, out[v], v)
    return out


def _snapshot_all(run):
    return {str(p.relative_to(run)): E.sha256_file(p) for p in sorted(Path(run).rglob("*")) if p.is_file()
            and p.name not in ("txn.lock",)}


@needs_blender
def test_C1_user_path_nonzero_pin_head_and_authoring_safety(fx, head_pkg, tmp_path):
    run = fx["pinned"]
    st0 = cli("status", "--run", run)
    snap0 = _snapshot_all(run)
    # export of the baseline (explicitly chosen verified state) too
    rb = cli("fbx-interchange", "--run", run, "--state", "baseline", "--out", tmp_path / "base_export")
    assert rb.returncode == 0, rb.stderr
    s = load(head_pkg / "STATUS.json")
    assert s["status"] == "COMPLETE" and s["state_id"] == fx["head"] and s["semantic_authorization"].startswith("AUTHORIZED_AT_COMMIT")
    man = load(head_pkg / "package" / X.PACKAGE_MANIFEST)
    assert X.seal_ok(man) and man["take"]["stack_count"] == 1 and man["fbx"]["patched_curves"] == 24
    assert man["timebase"]["fps"] == 30 and [man["timebase"]["frame_start"], man["timebase"]["frame_end"]] == [1, 8]
    rc = consume(head_pkg / "package", tmp_path / "c")
    assert rc.returncode == 0, rc.stderr
    crep = load(tmp_path / "c" / "consume_report.json")
    acc = crep["file_access_audit"]["classified"]
    assert crep["verdict"] == "RECOVERED" and not acc["UNDECLARED"] and len(acc["DECLARED_INPUT"]) == 2
    assert not any(str(run).lower() in p for p in sum(acc.values(), []))          # nothing of the run was opened
    rv = cli("fbx-verify", "--run", run, "--consumed", tmp_path / "c", "--out", tmp_path / "v")
    assert rv.returncode == 0, rv.stderr
    v = load(tmp_path / "v" / "fbx_verify_report.json")
    assert v["interchange_verdict"] == "INTERCHANGE_PASS" and v["mismatches"] == [], v["mismatches"]
    assert v["pin"]["residual"]["imported"]["integer_pin_frames_max_m"] <= 1e-3
    assert v["pin"]["residual"]["imported"]["subframes_between_pin_frames_max_m"] <= 1e-3
    assert v["grid_subframes"] == [0.25, 0.5, 0.75, 0.99]
    # raw FBX alone through the stock importer
    rr = consume(head_pkg / "package", tmp_path / "craw", mode="raw")
    assert rr.returncode == 0, rr.stderr
    rv2 = cli("fbx-verify", "--run", run, "--consumed", tmp_path / "craw", "--out", tmp_path / "vraw")
    assert rv2.returncode == 0, rv2.stderr
    vr = load(tmp_path / "vraw" / "fbx_verify_report.json")
    assert vr["interchange_verdict"] == "FAIL" and vr["mismatches"] == RAW_HELD, vr["mismatches"]
    # baseline package also recovers
    rcb = consume(tmp_path / "base_export" / "package", tmp_path / "cb")
    assert rcb.returncode == 0, rcb.stderr
    rvb = cli("fbx-verify", "--run", run, "--consumed", tmp_path / "cb", "--out", tmp_path / "vb")
    assert rvb.returncode == 0 and load(tmp_path / "vb" / "fbx_verify_report.json")["interchange_verdict"] == "INTERCHANGE_PASS"
    # authoring safety: every run file byte-identical, status identical
    assert _snapshot_all(run) == snap0
    assert cli("status", "--run", run).stdout == st0.stdout
    # exact undo still works after export; the package stays bound to its state and is reported as not-the-head
    r = cli("revert", "--run", run)
    assert r.returncode == 0, r.stderr
    h2 = E.read_refs(run)[0]
    rep = load(run / f"reverted/reverted_{h2['head'][:12]}_report.json")
    assert rep["exact_undo"] and rep["blender_eval_integer_frames_bit_equal"] and rep["blender_eval_subframes_bit_equal"]
    rv3 = cli("fbx-verify", "--run", run, "--consumed", tmp_path / "c", "--out", tmp_path / "v3")
    assert rv3.returncode == 0 and load(tmp_path / "v3" / "fbx_verify_report.json")["is_current_head"] is False
    rv4 = cli("fbx-verify", "--run", run, "--consumed", tmp_path / "c", "--state", "head", "--out", tmp_path / "v4")
    assert rv4.returncode == 2 and "PACKAGE_NOT_OF_THIS_STATE" in rv4.stderr


@needs_blender
def test_C2_smooth_linear_control_and_blanket_constant_harm(controls, tmp_path):
    s = package_of(controls["linear"], tmp_path / "x", "linear")
    assert s["patched_constant_curves"] == 0
    assert consume(tmp_path / "x" / "package", tmp_path / "c").returncode == 0
    v = evaluate(controls["linear"], tmp_path / "c", tmp_path / "v")
    assert v["verdict"] == "PASS", v["mismatches"]
    blender_helper("blanket_constant", tmp_path / "c" / "recovered.blend", tmp_path / "blanket.blend")
    (tmp_path / "vk").mkdir()
    vk = E.evaluate_pair(BLENDER, controls["linear"], tmp_path / "blanket.blend", tmp_path / "vk", None, "blanket CONSTANT")
    assert "SUBFRAME_WORLD_DEVIATION_OVER_1MM" in vk["mismatches"] and vk["integer"]["max_m"] <= 1e-3
    assert consume(tmp_path / "x" / "package", tmp_path / "craw", mode="raw").returncode == 0
    vr = evaluate(controls["linear"], tmp_path / "craw", tmp_path / "vr")
    assert vr["mismatches"] == ["SCENE_FRAME_RANGE_NOT_CARRIED"], vr["mismatches"]


@needs_blender
def test_C3_mixed_per_group_held_and_smooth(controls, tmp_path):
    s = package_of(controls["mixed"], tmp_path / "x", "mixed")
    assert s["patched_constant_curves"] == 21
    man = load(tmp_path / "x" / "package" / X.PACKAGE_MANIFEST)
    assert man["interpolation"]["producer_directives"]["LimbNode:upperarm_l|rotation"] == "LINEAR"
    assert man["interpolation"]["producer_directives"]["LimbNode:lowerarm_l|rotation"] == "CONSTANT"
    assert consume(tmp_path / "x" / "package", tmp_path / "c").returncode == 0
    v = evaluate(controls["mixed"], tmp_path / "c", tmp_path / "v")
    assert v["verdict"] == "PASS", v["mismatches"]
    blender_helper("blanket_constant", tmp_path / "c" / "recovered.blend", tmp_path / "blanket.blend")
    (tmp_path / "vk").mkdir()
    vk = E.evaluate_pair(BLENDER, controls["mixed"], tmp_path / "blanket.blend", tmp_path / "vk", None, "blanket CONSTANT")
    assert "SUBFRAME_WORLD_DEVIATION_OVER_1MM" in vk["mismatches"]
    assert consume(tmp_path / "x" / "package", tmp_path / "craw", mode="raw").returncode == 0
    vr = evaluate(controls["mixed"], tmp_path / "craw", tmp_path / "vr")
    assert vr["mismatches"] == RAW_HELD, vr["mismatches"]


@needs_blender
@pytest.mark.parametrize("name", sorted(UNSUPPORTED_SOURCES))
def test_C4_unsupported_sources_stop_without_package(controls, tmp_path, name):
    sha = E.sha256_file(controls[name])
    with pytest.raises(E.SliceStop) as ei:
        package_of(controls[name], tmp_path / "x", name)
    assert load(tmp_path / "x" / "STATUS.json")["status"] == "FAILED"
    assert not (tmp_path / "x" / "package" / X.PACKAGE_MANIFEST).exists()
    assert E.sha256_file(controls[name]) == sha
    assert ei.value.code == UNSUPPORTED_SOURCES[name], f"CLASSIFICATION: {ei.value}"
    r = consume(tmp_path / "x" / "package", tmp_path / "c")
    assert r.returncode == 2 and not (tmp_path / "c" / "recovered.blend").exists()
    assert "PACKAGE_INCOMPLETE" in r.stderr, "CLASSIFICATION: " + r.stderr[-1500:]


@needs_blender
def test_C5a_extra_unassigned_action_is_excluded_not_exported(controls, tmp_path):
    s = package_of(controls["extra_action"], tmp_path / "x", "extra")
    assert s["other_actions_in_source_not_exported"] == ["AnimCV_EXTRA_unassigned_take"]
    man = load(tmp_path / "x" / "package" / X.PACKAGE_MANIFEST)
    assert man["take"]["stack_count"] == 1
    assert consume(tmp_path / "x" / "package", tmp_path / "c").returncode == 0
    v = evaluate(controls["extra_action"], tmp_path / "c", tmp_path / "v")
    assert v["verdict"] == "PASS" and v["actions"]["imported"] == ["root|root|AnimCV_EXP_extra"], v


def _reseal(pkg, fn):
    mp = Path(pkg) / X.PACKAGE_MANIFEST
    m = load(mp)
    fn(m)
    mp.write_text(json.dumps(X.seal(m), indent=1, sort_keys=True), encoding="utf-8")


def _set_fbx(pkg, data):
    (Path(pkg) / X.PACKAGE_FBX).write_bytes(data)
    import hashlib

    def f(m):
        m["fbx"]["sha256"] = hashlib.sha256(data).hexdigest()
        m["fbx"]["bytes"] = len(data)
    _reseal(pkg, f)


@needs_blender
@pytest.mark.parametrize("kind", sorted(PACKAGE_NEGATIVES))
def test_C5_C6_package_negative_controls_refuse_without_output(fx, head_pkg, tmp_path, kind):
    pkg = tmp_path / "package"
    shutil.copytree(head_pkg / "package", pkg)
    take_action = load(pkg / X.PACKAGE_MANIFEST)["take"]["name"].split("|", 1)[1]
    env = None
    if kind == "missing_manifest":
        (pkg / X.PACKAGE_MANIFEST).unlink()
    elif kind == "manifest_edit_no_reseal":
        m = load(pkg / X.PACKAGE_MANIFEST)
        m["timebase"]["fps"] = 25
        (pkg / X.PACKAGE_MANIFEST).write_text(json.dumps(m), encoding="utf-8")
    elif kind == "fbx_byte_flip":
        b = bytearray((pkg / X.PACKAGE_FBX).read_bytes())
        b[len(b) // 2] ^= 0xFF
        (pkg / X.PACKAGE_FBX).write_bytes(bytes(b))
    elif kind == "stale_fbx_of_other_package":
        other = tmp_path / "other"
        r = cli("fbx-interchange", "--run", fx["pinned"], "--state", "baseline", "--out", other)
        assert r.returncode == 0, r.stderr
        shutil.copyfile(other / "package" / X.PACKAGE_FBX, pkg / X.PACKAGE_FBX)
    elif kind == "multitake_fbx_resealed":
        blender_helper("multitake_fbx", fx["pinned"] / load(fx["pinned"] / "lineage/refs.json")["head_blend"]["blend"],
                       tmp_path / "mt.fbx", take_action)
        _set_fbx(pkg, (tmp_path / "mt.fbx").read_bytes())
    elif kind == "take_renamed_resealed":
        _reseal(pkg, lambda m: m["take"].__setitem__("name", "root|SOME_OTHER_TAKE"))
    elif kind == "fps25_resealed":
        _reseal(pkg, lambda m: m["timebase"].__setitem__("fps", 25))
    elif kind == "unknown_rig_declared_resealed":
        def f(m):
            b = m["rig"]["bones"]
            b["hand_left"] = b.pop("hand_l")
            for k, v in b.items():
                if v == "hand_l":
                    b[k] = "hand_left"
            m["rig"]["topology_sha256"] = X.topology_digest(b)
        _reseal(pkg, f)
    elif kind == "wrong_rig_fbx_consistent_resealed":
        blender_helper("wrongrig_fbx", fx["pinned"] / load(fx["pinned"] / "lineage/refs.json")["head_blend"]["blend"],
                       tmp_path / "wr.fbx", take_action)
        data = (tmp_path / "wr.fbx").read_bytes()
        root, _, _ = X.read_fbx(data)
        gt = X.group_table(X.fbx_curve_map(root, load(pkg / X.PACKAGE_MANIFEST)["take"]["name"]))
        (pkg / X.PACKAGE_FBX).write_bytes(data)
        import hashlib

        def f(m):
            m["fbx"].update(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data), group_table=gt)
        _reseal(pkg, f)
    elif kind == "producer_version_resealed":
        _reseal(pkg, lambda m: m["producer"].__setitem__("blender_version", "4.2.0"))
    elif kind == "directive_edit_resealed":
        def f(m):
            k = sorted(k for k, v in m["interpolation"]["consumer_directives"].items() if v == "CONSTANT")[0]
            m["interpolation"]["consumer_directives"][k] = "LINEAR"
        _reseal(pkg, f)
    elif kind == "fbx_flag_reverted_resealed":
        data = (pkg / X.PACKAGE_FBX).read_bytes()
        root, _, flocs = X.read_fbx(data)
        fm = X.fbx_curve_map(root, load(pkg / X.PACKAGE_MANIFEST)["take"]["name"])
        u = sorted(c["uid"] for c in fm["curves"] if c["flag_class"] == "CONSTANT")[0]
        new, _ = X.patch_flags(data, flocs, [u], X.EXPORTER_FLAG, expected_old=X.CONSTANT_FLAG)
        _set_fbx(pkg, new)
    elif kind == "truncated_fbx_resealed":
        data = (pkg / X.PACKAGE_FBX).read_bytes()
        _set_fbx(pkg, data[:len(data) // 2])
    elif kind == "consumer_crash_after_import":
        env = {X.FAULT_ENV: "after_import"}
    before = {p.name: E.sha256_file(p) for p in pkg.iterdir()}
    r = consume(pkg, tmp_path / "c", env=env)
    assert r.returncode == 2, r.stdout[-1000:] + r.stderr[-1000:]                   # refused (safety)
    assert not (tmp_path / "c" / "recovered.blend").exists()
    assert not (tmp_path / "c" / "consume_report.json").exists()
    assert {p.name: E.sha256_file(p) for p in pkg.iterdir()} == before          # nothing repaired
    assert f"STOP {PACKAGE_NEGATIVES[kind]}:" in r.stderr, "CLASSIFICATION: " + r.stderr[-1500:]


@needs_blender
def test_C6_producer_failures_leave_no_package(fx, controls, tmp_path):
    run = fx["pinned"]
    snap = _snapshot_all(run)
    r = cli("fbx-interchange", "--run", run, "--out", tmp_path / "f1", env={X.FAULT_ENV: "after_raw_export"})
    assert r.returncode == 2
    assert load(tmp_path / "f1" / "STATUS.json")["status"] == "FAILED"
    assert not (tmp_path / "f1" / "package" / X.PACKAGE_MANIFEST).exists()
    assert "PRODUCER_STAGE_FAILED" in r.stderr, "CLASSIFICATION: " + r.stderr[-1500:]
    r = cli("fbx-interchange", "--run", run, "--out", tmp_path / "f2", env={X.FAULT_ENV: "before_manifest"})
    assert r.returncode == 86
    assert load(tmp_path / "f2" / "STATUS.json")["status"].startswith("IN_PROGRESS")
    assert (tmp_path / "f2" / "package" / X.PACKAGE_FBX).exists() and not (tmp_path / "f2" / "package" / X.PACKAGE_MANIFEST).exists()
    for d in ("f1", "f2"):
        c = consume(tmp_path / d / "package", tmp_path / f"c_{d}")
        assert c.returncode == 2 and not (tmp_path / f"c_{d}" / "recovered.blend").exists()
        assert "PACKAGE_INCOMPLETE" in c.stderr, "CLASSIFICATION: " + c.stderr[-1500:]
    r = cli("fbx-interchange", "--run", run, "--out", tmp_path / "f2")                 # never overwritten
    assert r.returncode == 2 and "REFUSED_EXISTING_OUTPUT" in r.stderr
    assert _snapshot_all(run) == snap


@needs_blender
def test_C7_stock_fbx_command_unchanged_and_still_fails(fx, tmp_path):
    def fn_src(text, name):
        # POST-OUTCOME FIX (W127, disclosed): the frozen version cut the text at the next "def", which also captured the
        # W127 section comment inserted AFTER cmd_fbx; compare the function's own source segment only.
        import ast
        node = [n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == name][0]
        return ast.get_source_segment(text, node)
    old = subprocess.run(["git", "show", f"{STARTING_COMMIT}:src/app/experimental_video_action.py"], cwd=ROOT, capture_output=True,
                         text=True, encoding="utf-8").stdout
    assert old and fn_src(old, "cmd_fbx") == fn_src((ROOT / "src/app/experimental_video_action.py").read_text(encoding="utf-8"), "cmd_fbx")
    olds = subprocess.run(["git", "show", f"{STARTING_COMMIT}:src/blender/experimental_action_stage.py"], cwd=ROOT,
                          capture_output=True, text=True, encoding="utf-8").stdout
    news = (ROOT / "src/blender/experimental_action_stage.py").read_text(encoding="utf-8")
    for fn in ("export_fbx", "fbx_readback"):
        assert fn_src(olds, fn) == fn_src(news, fn)
    assert subprocess.run(["git", "diff", "--quiet", STARTING_COMMIT, "--", "src/blender/export.py", "src/blender/executor.py"],
                          cwd=ROOT).returncode == 0
    run = tmp_path / "run"
    shutil.copytree(fx["pinned"], run)
    r = cli("fbx", "--run", run, "--state", "head")
    assert r.returncode == 0, r.stderr
    rep = load(next((run / "fbx").glob("head_*")) / "fbx_roundtrip_report.json")
    assert rep["roundtrip_verdict"] == "FAIL"
    assert {"SUBFRAME_WORLD_DEVIATION_OVER_1MM", "INTERPOLATION_CHANGED", "SCENE_FRAME_RANGE_NOT_CARRIED"} <= set(rep["mismatches"])


@needs_blender
def test_C8_fresh_repeat_with_exporter_uid_and_path_variation(fx, head_pkg, tmp_path):
    run2 = tmp_path / "elsewhere" / "copied_run"
    shutil.copytree(fx["pinned_pristine"], run2)
    assert E.read_refs(run2)[0]["head"] == fx["head"]
    r = cli("fbx-interchange", "--run", run2, "--state", "head", "--out", tmp_path / "x2")
    assert r.returncode == 0, r.stderr
    a, b = load(head_pkg / "package" / X.PACKAGE_MANIFEST), load(tmp_path / "x2" / "package" / X.PACKAGE_MANIFEST)
    assert a["fbx"]["raw_exporter_output_sha256"] != b["fbx"]["raw_exporter_output_sha256"]      # the variation occurred
    assert a["fbx"]["group_table"] == b["fbx"]["group_table"] and a["interpolation"] == b["interpolation"]
    assert a["take"] == b["take"] and a["timebase"] == b["timebase"] and a["rig"] == b["rig"]
    for n, p in (("1", head_pkg / "package"), ("2", tmp_path / "x2" / "package")):
        assert consume(p, tmp_path / f"c{n}").returncode == 0
        bl = tmp_path / f"c{n}" / "recovered.blend"
        rc = E._stage(BLENDER, "readback", tmp_path / f"rb{n}.log", blend=bl, report=tmp_path / f"rb{n}.json",
                      subframes="0.25,0.5,0.75,0.99", frozen_audit="0")
        assert rc == 0
    r1, r2 = load(tmp_path / "rb1.json"), load(tmp_path / "rb2.json")
    assert [f["world"] for f in r1["frames"]] == [f["world"] for f in r2["frames"]]
    assert [s["world"] for s in r1["subframes"]] == [s["world"] for s in r2["subframes"]]
    assert r1["fcurve_census"]["interpolation_census_curves"] == r2["fcurve_census"]["interpolation_census_curves"]


@needs_blender
def test_C9_concurrent_invocations(fx, tmp_path):
    run = fx["pinned"]
    out = tmp_path / "same"
    ps = [cli("fbx-interchange", "--run", run, "--out", out, wait=False) for _ in range(2)]
    res = [(p.wait(600), p.communicate()) for p in ps]
    codes = sorted(c for c, _ in res)
    assert codes == [0, 2], res
    assert any("REFUSED_EXISTING_OUTPUT" in e for c, (o, e) in res if c == 2)
    assert X.seal_ok(load(out / "package" / X.PACKAGE_MANIFEST))
    # an export while a commit transaction is in flight (W125 hang before publication) refuses; the commit then completes
    run2 = tmp_path / "run2"
    shutil.copytree(fx["base"], run2)
    hang = tmp_path / "hang"
    pc = cli("commit", "--run", run2, "--request-id", "P1", "--decision", "accept", wait=False,
             env={TX.FAULT_ENV: "before_publish", TX.FAULT_MODE_ENV: "hang", TX.HANG_FILE_ENV: str(hang)})
    t0 = time.time()
    while not Path(str(hang) + ".reached").exists():
        assert time.time() - t0 < 300 and pc.poll() is None, pc.communicate()
        time.sleep(0.1)
    r = cli("fbx-interchange", "--run", run2, "--out", tmp_path / "during")
    assert r.returncode == 2 and "RECOVERY_REQUIRED" in r.stderr and not (tmp_path / "during").exists()
    Path(str(hang) + ".release").write_text("go")
    assert pc.wait(600) == 0, pc.communicate()
    assert E.read_refs(run2)[0]["head"] == load(run2 / "edits/P1/proposal.json")["candidate"]


def test_default_cli_isolation_and_gui():
    code = ("import sys; sys.path.insert(0, 'src'); from app.cli import build_parser; build_parser(); "
            "print(','.join(m for m in ('blender.experimental_fbx_interchange', 'bpy') if m in sys.modules))")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == "", r.stdout + r.stderr
    gui = ROOT / "src" / "app" / "gui.py"
    if gui.exists():
        t = gui.read_text(encoding="utf-8")
        assert "fbx-interchange" not in t and "experimental_fbx_interchange" not in t


@needs_blender
def test_parity_with_w110_scratch_adapter_curve_map(head_pkg):
    import w110_fbx_consumer_adapter as AD
    data = (head_pkg / "package" / X.PACKAGE_FBX).read_bytes()
    root, _, _ = X.read_fbx(data)
    take = load(head_pkg / "package" / X.PACKAGE_MANIFEST)["take"]["name"]
    mine, ref = X.fbx_curve_map(root, take), AD.fbx_curve_map(root, take)
    assert mine["status"] == ref["status"] == "OK"
    assert mine["curves"] == ref["curves"]
    assert {k: {kk: vv for kk, vv in v.items() if kk != "non_default"} for k, v in mine["models"].items()} == \
        {k: {kk: vv for kk, vv in v.items() if kk != "non_default"} for k, v in ref["models"].items()}
