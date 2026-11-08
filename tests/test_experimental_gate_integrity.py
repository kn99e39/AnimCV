"""Batch 126 focused tests: commit-time semantic gate integrity of the EXPERIMENTAL opt-in WORLD_POSITION_PIN editor.

Test kinds (reported separately in the W126 Worklog):
  * unit     - seal / gate-view / precedence helpers; no Blender.
  * blender  - REAL Blender 5.1 on the REAL W105 BaseRig with the W125 SYNTHETIC-VALUE 8-frame fixture (not video, not a quality
               claim), driven through the user CLI in separate processes (``python -m app.cli experimental-video-action ...``).
  * MOCK     - N23 only: the frozen resolver's STATUS is replaced in-process by a stub returning NON_CONVERGED (the analytic
               slice resolver never returns NON_CONVERGED); everything else is real Blender.

The negative-control matrix ``NEG`` and the valid cases ``VALID`` (expected exit code, STOP code, primary refusal class) were
frozen in the W126 declaration BEFORE any W126 outcome was examined; they are not adjusted after a failure.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from argparse import Namespace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(Path(__file__).resolve().parent)]
from app import experimental_video_action as E  # noqa: E402
from app import experimental_action_txn as TX  # noqa: E402
import w116_semantic_pin_base as S  # noqa: E402
from test_experimental_action_txn import (BLENDER, HAS_BLENDER, _req, _wait, assert_consistent, build_fixture_run, cli,  # noqa: E402
                                          head, load, status)

needs_blender = pytest.mark.skipif(not HAS_BLENDER, reason="Blender 5.1 not available")
AUTH = E.AUTH_REFUSED

# ---- frozen negative-control matrix: id -> (proposal, description, expected exit, expected STOP code, expected primary) ------
NEG = {
    "N01": ("REACH", "W125 bypass: UNREACHABLE candidate, proposal decision hand-edited to ACCEPT (no reseal)", 2, AUTH, "PROPOSAL_SEAL_INVALID"),
    "N02": ("REACH", "forged decision+status with a recomputed seal", 2, AUTH, "RECOMPUTED_GATE_REJECT"),
    "N03": ("REACH", "forged low residual + pins_satisfied + decision/status, resealed", 2, AUTH, "RECOMPUTED_GATE_REJECT"),
    "N04": ("P1", "forged residual (0.0) on a valid proposal, resealed", 2, AUTH, "PROPOSAL_GATE_RECORD_DIVERGES"),
    "N05": ("P1", "pin target moved +5 cm in the proposal constraint, resealed", 2, AUTH, "PROPOSAL_GATE_RECORD_DIVERGES"),
    "N06": ("P1", "pin frames shortened in the proposal, resealed", 2, AUTH, "PROPOSAL_GATE_RECORD_DIVERGES"),
    "N07": ("P1", "constraint digest altered, resealed", 2, AUTH, "PROPOSAL_GATE_RECORD_DIVERGES"),
    "N08": ("P1", "authorized channels widened (+spine_01) in the proposal, resealed", 2, AUTH, "PROPOSAL_GATE_RECORD_DIVERGES"),
    "N09": ("P1", "binding id swapped to right_wrist in the proposal, resealed", 2, AUTH, "PROPOSAL_GATE_RECORD_DIVERGES"),
    "N10": ("P1", "changed-frame/Bone footprint shrunk in the proposal, resealed", 2, AUTH, "PROPOSAL_GATE_RECORD_DIVERGES"),
    "N11": ("P1", "request.json bytes changed (capture frame 12->13), provenance not updated", 2, AUTH, "REQUEST_BYTES_CHANGED"),
    "N12": ("P1", "request.json changed AND its hash updated in the proposal, resealed", 2, AUTH, "CANDIDATE_NOT_REPRODUCED"),
    "N13": ("P1", "substituted binding record (left_wrist -> hand_l.head) in bindings.json", 2, AUTH, "BINDING_RECORD_CHANGED"),
    "N15": ("P1", "request.json reformatted (same JSON, different bytes): byte-identity policy", 2, AUTH, "REQUEST_BYTES_CHANGED"),
    "N16": ("P1", "candidate_state.json values edited (content address broken)", 2, "INVALID_CANDIDATE_TAMPERED", None),
    "N17": ("P1", "REACH candidate state/.blend/readback substituted into P1 with all hashes updated, resealed", 2, AUTH,
            "RECOMPUTED_GATE_REJECT"),
    "N18": ("P1", "candidate .blend byte flipped", 2, "INVALID_CANDIDATE_BLEND_TAMPERED", None),
    "N19": ("P1", "candidate .blend replaced by REACH's, readback+hashes made consistent: only the FRESH readback can tell", 2, AUTH,
            "FRESH_READBACK_GATE_REJECT"),
    "N21": ("REACH", "conflicting status PROPOSED_GATE_PASS with decision REJECT, resealed", 2, AUTH, "PROPOSAL_DECISION_STATUS_CONFLICT"),
    "N23": ("NC", "MOCK solver NON_CONVERGED with a real candidate file, forged ACCEPT, resealed (in-process)", None, AUTH,
            "RECOMPUTED_GATE_REJECT"),
    "N24": ("P1", "candidate readback deleted", 2, AUTH, "CANDIDATE_READBACK_MISSING"),
    "N25": ("P1", "candidate readback values altered, hash not updated", 2, AUTH, "CANDIDATE_READBACK_CHANGED"),
    "N25b": ("REACH", "readback forged to show the pin satisfied, hashes updated, proposal forged ACCEPT, resealed", 2, AUTH,
             "RECOMPUTED_GATE_REJECT"),
    "N26": ("P2", "STALE parent AND unsealed edit simultaneously (precedence: provenance before stale)", 2, AUTH, "PROPOSAL_SEAL_INVALID"),
    "N27": ("P1", "request bytes changed AND proposal edited (precedence: seal first)", 2, AUTH, "PROPOSAL_SEAL_INVALID"),
    "N28": ("P1", ".blend flipped AND proposal edited (precedence: W125 candidate-blend check first)", 2,
            "INVALID_CANDIDATE_BLEND_TAMPERED", None),
    "N29": ("P1", "proposal stripped to the unsealed pre-W126 format (decision ACCEPT)", 2, "BLOCKED_LEGACY_PROPOSAL_REQUIRES_REPROPOSE", None),
    "N30": ("P1", "request.json edited WHILE the accept transaction holds the lock (after fresh readback, before publish)", 2, AUTH,
            "INPUTS_CHANGED_DURING_TRANSACTION"),
    "N32": ("WB", "wrong-binding proposal (no candidate) forged ACCEPT + PASS status, resealed", 2, AUTH, "PROPOSAL_DECISION_STATUS_CONFLICT"),
    "N33": ("UA", "unauthorized-DOF proposal (no candidate) forged ACCEPT, resealed (status untouched)", 2, AUTH,
            "PROPOSAL_DECISION_STATUS_CONFLICT"),
    "N34": ("P1", "parent (head) readback altered", 2, "BLOCKED_HEAD_UNVERIFIED", None),
}
# Preserved (pre-W126) classified outcomes that are RECORDED decisions, not refusals.
RECORDED = {
    "N20": ("P2", "stale parent (P1 accepted first)", 0, "STALE_PARENT"),
    "N22": ("REACH", "honest UNREACHABLE proposal + accept flag", 0, "GATE_REJECTED_CANNOT_ACCEPT"),
    "N31": ("P1", "tampered proposal, explicit user REJECT: recorded REJECTED, display marked UNTRUSTED", 0, "USER_REJECTED"),
}
VALID = {
    "V01": "valid nonzero pin P1: ACCEPTED, fresh-readback residual <= 1 mm, actual Blender footprint, AUTHORIZED_AT_COMMIT",
    "V02": "no-op pin (NOOP): ACCEPT TRIVIALLY_SATISFIED_NOT_EVIDENCE, empty footprint (not evidence of improvement)",
    "V03": "explicit REJECT (P3), then accept flag refused REFUSED_ALREADY_DECIDED",
    "V04": "exact REVERT after V01 (bit-equal integer frames and subframes)",
    "V05": "benign reserialization of bindings.json and proposal.json (same content): ACCEPTED",
}
RECOVERY = {
    "R01": "published W126 ACCEPT whose (resealed) journal claims authorized=false -> recover BLOCKED_DIVERGENT_JOURNAL (never invents)",
    "R02": "published pre-W126-style ACCEPT journal (no semantic_authorization) -> recover rolls forward, marked NOT ESTABLISHED",
}


# ------------------------------------------------------------------------------------------------ helpers ---
def reseal(p):
    p = json.loads(json.dumps(p))
    p["provenance"]["gate_view_digest"] = E._digest(E.gate_view(p))
    return E._sealed(p, "proposal_seal")


def wprop(run, rid, p, indent=1):
    (run / "edits" / rid / "proposal.json").write_text(json.dumps(p, indent=indent, sort_keys=True), encoding="utf-8")


def rprop(run, rid):
    return load(run / "edits" / rid / "proposal.json")


def snapshot(run):
    files = {}
    for p in sorted(run.rglob("*")):
        if p.is_file() and "authorization_refusals" not in p.parts and "quarantine" not in p.parts and "txn" not in p.parts \
                and p.suffix != ".lock" and not p.name.endswith(".holder.json"):
            files[p.relative_to(run).as_posix()] = E.sha256_file(p)
    return files


def assert_refused_untouched(run, v0, rid, before_refs, before_files, code, primary, r):
    """Every rejected control: no head/generation change, no accepted .blend, no false ACCEPTED display, no commit.json, no
    silent repair of any input file (the tampered bytes are still there), immutable baseline intact."""
    assert r.returncode == 2, (r.returncode, r.stderr)
    assert f"STOP {code}" in r.stderr, r.stderr
    assert head(run) == before_refs                                       # head + generation byte-identical record
    acc = sorted((run / "accepted").glob("*.blend")) if (run / "accepted").exists() else []
    assert not acc, acc
    assert not (run / "edits" / rid / "commit.json").exists()
    for p, j in E.read_journals(run):
        assert j is not None and (j["op"] != "ACCEPT" or j["state"] == "ABORTED"), (p, j and j["state"])
    after = snapshot(run)
    # nothing that existed before was changed by the refused command (only refusal records / quarantine / journals are added)
    changed = sorted(k for k in before_files if before_files[k] != after.get(k))
    assert not changed, changed
    st = assert_consistent(run, v0)
    e = st["edits"][rid]
    assert e["decision"].startswith("UNDECIDED") and e["gate"] != "ACCEPT" and e["semantic_authorization"] is None, e
    if primary is not None:
        recs = sorted((run / "edits" / rid / "authorization_refusals").glob("*.json"))
        assert recs, "no refusal evidence recorded"
        rec = load(recs[-1])
        assert rec["primary"] == primary, rec
        assert f"{primary}" in r.stderr
    return st


# ------------------------------------------------------------------------------------------------- unit ---
def test_seal_and_gate_view_detect_edits_and_ignore_reserialization():
    p = {"schema": E.PROPOSAL_SCHEMA, "decision": {"decision": "REJECT", "failure_class": "INFEASIBLE_REQUEST", "reasons": ["UNREACHABLE"]},
         "status": "GATE_REJECTED", "candidate": "c" * 64, "hard_pin_residual_max_m": 0.136, "provenance": {}}
    p = reseal(p)
    assert E.proposal_kind(p) == "V2" and E._seal_ok(p, "proposal_seal")
    assert E._seal_ok(json.loads(json.dumps(p, indent=4)), "proposal_seal")           # benign reserialization
    q = dict(p, decision={"decision": "ACCEPT", "failure_class": "NONE", "reasons": []})
    assert not E._seal_ok(q, "proposal_seal")
    assert E.gate_divergence(p, q) == ["decision"]
    assert E.proposal_kind({"decision": {}}) == "LEGACY_UNSEALED"
    assert E._cj(float("nan")) == E._cj(float("nan"))                                 # NaN-safe comparison


def test_declared_reason_order_is_total_and_unique():
    assert len(set(E.AUTH_REASON_ORDER)) == len(E.AUTH_REASON_ORDER)
    for cid, (_, _, _, code, primary) in NEG.items():
        assert primary is None or primary in E.AUTH_REASON_ORDER, cid


# ---------------------------------------------------------------------------------------------- fixture ---
@pytest.fixture(scope="session")
def base_run(tmp_path_factory):
    if not HAS_BLENDER:
        pytest.skip("Blender 5.1 not available")
    run = tmp_path_factory.mktemp("w126") / "base"
    v0 = build_fixture_run(run)                       # P1, P2, P3 (valid), REACH (unreachable, real candidate file)
    bnd = load(run / "baseline" / "bindings.json")
    rw = [bid for bid, r in bnd["bound"].items() if r["semantic_joint"] == "right_wrist"][0]
    extra = {"NOOP": _req("NOOP", v0["state_id"], [12, 12], 12),
             "WB": dict(_req("WB", v0["state_id"], [12, 15], 12), binding_id=rw),
             "UA": dict(_req("UA", v0["state_id"], [12, 15], 12), authorized_channels=["lowerarm_l"])}
    for rid, rq in extra.items():
        E.dump(run / "requests" / f"{rid}.json", rq)
        E.cmd_propose(Namespace(run=str(run), request=str(run / "requests" / f"{rid}.json")))
    return run, v0, rw


@pytest.fixture
def run(base_run, tmp_path):
    src, v0, rw = base_run
    dst = tmp_path / "run"
    shutil.copytree(src, dst)
    return dst, v0, rw


def accept(run, rid, **kw):
    return cli(run, "commit", "--request-id", rid, "--decision", "accept", **kw)


@needs_blender
def test_fixture_proposals_are_sealed_and_classified_as_frozen(run):
    run, v0, rw = run
    exp = {"P1": "ACCEPT", "P2": "ACCEPT", "P3": "ACCEPT", "REACH": "REJECT", "NOOP": "ACCEPT", "WB": "REJECT", "UA": "REJECT"}
    st = status(run)
    for rid, d in exp.items():
        p = rprop(run, rid)
        assert E.proposal_kind(p) == "V2" and E._seal_ok(p, "proposal_seal"), rid
        assert p["decision"]["decision"] == d and st["edits"][rid]["gate"] == d and st["edits"][rid]["proposal_integrity"] == "OK", rid
    assert rprop(run, "REACH")["decision"]["reasons"] == ["UNREACHABLE"] and rprop(run, "REACH")["candidate"]
    assert rprop(run, "NOOP")["decision"]["reasons"] == ["TRIVIALLY_SATISFIED_NOT_EVIDENCE"]
    assert rprop(run, "WB")["candidate"] is None and rprop(run, "UA")["candidate"] is None


# ------------------------------------------------------------------------------------- negative controls ---
def _forge_accept(p):
    return dict(p, decision={"decision": "ACCEPT", "failure_class": "NONE", "reasons": []}, status="PROPOSED_GATE_PASS")


def _rehash_candidate(run, rid, p):
    """Make the proposal's candidate hash bindings consistent with whatever files are now on disk (a consistent forger)."""
    ed = run / "edits" / rid
    c = p["provenance"]["candidate"]
    c["state_file_sha256"] = E.sha256_file(ed / "candidate_state.json")
    c["state_id"] = load(ed / "candidate_state.json")["state_id"]
    c["blend_sha256"] = E.sha256_file(run / c["blend"])
    c["readback_sha256"] = E.sha256_file(run / c["readback"])
    p["candidate"] = c["state_id"]
    p["candidate_blend"] = dict(p["candidate_blend"], blend_sha256=c["blend_sha256"])
    return p


def mutate(run, cid, rw):
    """Apply control ``cid``.  Returns (request id, extra-before-command callable or None)."""
    rid = NEG[cid][0]
    ed = run / "edits" / rid
    p = rprop(run, rid)
    if cid == "N01":
        wprop(run, rid, dict(p, decision={"decision": "ACCEPT", "failure_class": "NONE", "reasons": []}))
    elif cid == "N02":
        wprop(run, rid, reseal(_forge_accept(p)))
    elif cid == "N03":
        q = _forge_accept(p)
        q["hard_pin_residual_max_m"] = 0.0
        for ev in q["evaluations"].values():
            ev["pins_satisfied"] = True
            ev["pins"][0]["candidate_residual_m"] = 0.0
        wprop(run, rid, reseal(q))
    elif cid == "N04":
        wprop(run, rid, reseal(dict(p, hard_pin_residual_max_m=0.0)))
    elif cid == "N05":
        c = dict(p["constraint"], target=[p["constraint"]["target"][0], p["constraint"]["target"][1], p["constraint"]["target"][2] + 0.05])
        wprop(run, rid, reseal(dict(p, constraint=c)))
    elif cid == "N06":
        wprop(run, rid, reseal(dict(p, pin_frames=p["pin_frames"][:2])))
    elif cid == "N07":
        wprop(run, rid, reseal(dict(p, constraint=dict(p["constraint"], constraint_digest="0" * 64))))
    elif cid == "N08":
        wprop(run, rid, reseal(dict(p, constraint=dict(p["constraint"], authorized_channels=["lowerarm_l", "spine_01", "upperarm_l"]))))
    elif cid == "N09":
        wprop(run, rid, reseal(dict(p, constraint=dict(p["constraint"], binding_id=rw))))
    elif cid == "N10":
        fp = dict(p["footprint"], frames_changed=p["footprint"]["frames_changed"][:-1],
                  channel_frames={"lowerarm_l": p["footprint"]["channel_frames"]["lowerarm_l"]})
        wprop(run, rid, reseal(dict(p, footprint=fp)))
    elif cid in ("N11", "N12", "N27"):
        rq = load(ed / "request.json")
        rq["target"]["capture_frame"] = 13
        (ed / "request.json").write_text(json.dumps(rq, indent=1), encoding="utf-8")
        if cid == "N12":
            h = E.sha256_file(ed / "request.json")
            q = dict(p, request_sha256=h)
            q["provenance"] = dict(p["provenance"], request=dict(p["provenance"]["request"], sha256=h))
            wprop(run, rid, reseal(q))
        if cid == "N27":
            wprop(run, rid, dict(p, hard_pin_residual_max_m=0.0))
    elif cid == "N13":
        b = load(run / "baseline" / "bindings.json")
        bid = p["constraint"]["binding_id"]
        b["bound"][bid] = dict(b["bound"][bid], bone="hand_l", point="head")
        (run / "baseline" / "bindings.json").write_text(json.dumps(b, indent=1), encoding="utf-8")
    elif cid == "N15":
        (ed / "request.json").write_text(json.dumps(load(ed / "request.json"), indent=4), encoding="utf-8")
    elif cid == "N16":
        st = load(ed / "candidate_state.json")
        st["values"]["lowerarm_l"][2] = [0.0, 0.0, 0.0, 1.0]
        os.chmod(ed / "candidate_state.json", 0o644)
        (ed / "candidate_state.json").write_text(json.dumps(st), encoding="utf-8")
    elif cid == "N17":
        src = run / "edits" / "REACH"
        for f in ("candidate_state.json", "candidate.blend", "candidate_readback.json"):
            if (ed / f).exists():
                os.chmod(ed / f, 0o644)
            shutil.copyfile(src / f, ed / f)
        wprop(run, rid, reseal(_rehash_candidate(run, rid, p)))
    elif cid in ("N18", "N28"):
        bp = ed / "candidate.blend"
        b = bytearray(bp.read_bytes())
        b[len(b) // 2] ^= 0x01
        bp.write_bytes(bytes(b))
        if cid == "N28":
            wprop(run, rid, dict(p, hard_pin_residual_max_m=0.0))
    elif cid == "N19":
        shutil.copyfile(run / "edits" / "REACH" / "candidate.blend", ed / "candidate.blend")
        rb = load(ed / "candidate_readback.json")
        rb["blend_sha256"] = E.sha256_file(ed / "candidate.blend")
        (ed / "candidate_readback.json").write_text(json.dumps(rb), encoding="utf-8")
        wprop(run, rid, reseal(_rehash_candidate(run, rid, p)))
    elif cid == "N21":
        wprop(run, rid, reseal(dict(p, status="PROPOSED_GATE_PASS")))
    elif cid == "N24":
        (ed / "candidate_readback.json").unlink()
    elif cid == "N25":
        rb = load(ed / "candidate_readback.json")
        rb["frames"][3]["world"]["lowerarm_l"]["tail"][2] += 1e-4
        (ed / "candidate_readback.json").write_text(json.dumps(rb), encoding="utf-8")
    elif cid == "N25b":
        rb = load(ed / "candidate_readback.json")
        m = {r["action_frame"]: r["source_frame"] for r in load(run / "baseline" / "action_sidecar.json")["frame_mapping"]}
        for fr in rb["frames"]:
            if m[fr["blender_frame"]] in set(p["pin_frames"]):
                fr["world"]["lowerarm_l"]["tail"] = list(p["constraint"]["target"])
        (ed / "candidate_readback.json").write_text(json.dumps(rb), encoding="utf-8")
        q = _forge_accept(p)
        q["hard_pin_residual_max_m"] = 0.0
        wprop(run, rid, reseal(_rehash_candidate(run, rid, q)))
    elif cid == "N26":
        wprop(run, rid, dict(p, hard_pin_residual_max_m=0.0))
    elif cid == "N29":
        q = {k: v for k, v in p.items() if k not in ("schema", "proposal_seal", "provenance")}
        wprop(run, rid, q)
    elif cid in ("N32", "N33"):
        q = dict(p, decision={"decision": "ACCEPT", "failure_class": "NONE", "reasons": []})
        if cid == "N32":
            q["status"] = "PROPOSED_GATE_PASS"
        wprop(run, rid, reseal(q))
    elif cid == "N34":
        hb = load(run / "lineage" / "refs.json")["head_blend"]["readback"]
        rb = load(run / hb)
        rb["frames"][0]["world"]["lowerarm_l"]["tail"][0] += 1e-3
        (run / hb).write_text(json.dumps(rb), encoding="utf-8")
    else:
        raise AssertionError(cid)
    return rid


CLI_NEG = [c for c in NEG if c not in ("N23", "N26", "N30")]


@needs_blender
@pytest.mark.parametrize("cid", CLI_NEG)
def test_negative_control_refused_without_any_publication(run, cid):
    run, v0, rw = run
    before = head(run)
    rid = mutate(run, cid, rw)
    files = snapshot(run)
    r = accept(run, rid)
    _, _, ex, code, primary = NEG[cid]
    assert r.returncode == ex, (cid, r.returncode, r.stderr)
    if code == "BLOCKED_HEAD_UNVERIFIED":           # the parent evidence itself is broken: every editing command stops
        assert f"STOP {code}" in r.stderr and head(run) == before and not (run / "edits" / rid / "commit.json").exists()
        assert not (run / "accepted").exists() or not list((run / "accepted").glob("*.blend"))
        return
    assert_refused_untouched(run, v0, rid, before, files, code, primary, r)
    if cid == "N19":                                 # refused AFTER staging: the transaction is rolled back + quarantined
        js = [j for _, j in E.read_journals(run) if j["op"] == "ACCEPT"]
        # POST-OUTCOME TEST CORRECTION (disclosed in the W126 Worklog): the refusal happens BEFORE promotion, so (W125 design)
        # nothing is under a final name to quarantine; the staged copy stays in the never-authoritative stage dir.
        assert len(js) == 1 and js[0]["state"] == "ABORTED" and js[0]["quarantined"] == []
        assert (run / js[0]["stage_dir"] / "accepted.blend").exists()
        assert "rolled back" in r.stderr
    if cid == "N29":                                 # legacy: not grandfathered, but an explicit REJECT stays possible
        st = status(run)
        assert st["edits"][rid]["gate"].startswith("LEGACY_UNVERIFIED")
        rr = cli(run, "commit", "--request-id", rid, "--decision", "reject")
        assert rr.returncode == 0, rr.stderr
        cm = load(run / "edits" / rid / "commit.json")
        assert cm["final_status"] == "REJECTED" and cm["proposal_integrity"]["reasons"] == ["LEGACY_UNSEALED_PROPOSAL"]
        assert head(run) == before


@needs_blender
def test_N26_stale_and_tampered_precedence(run):
    run, v0, rw = run
    assert accept(run, "P1").returncode == 0
    before = head(run)
    mutate(run, "N26", rw)
    files = snapshot(run)
    r = accept(run, "P2")
    _, _, ex, code, primary = NEG["N26"]
    assert r.returncode == ex
    assert f"STOP {code}" in r.stderr and primary in r.stderr
    assert head(run) == before and not (run / "edits/P2/commit.json").exists()
    assert not sorted(k for k in files if files[k] != snapshot(run).get(k))


@needs_blender
def test_N30_input_edit_during_transaction_is_caught_before_publication(run, tmp_path):
    run, v0, rw = run
    before = head(run)
    hang = tmp_path / "hang"
    pr = accept(run, "P1", fault="before_publish", mode="hang", hang=hang, wait=False)
    _wait(str(hang) + ".reached", 300)
    rq = run / "edits/P1/request.json"
    rq.write_bytes(rq.read_bytes() + b"\n")          # an edit while the transaction holds the run lock
    Path(str(hang) + ".release").write_text("go")
    pr.wait(300)
    err = pr.stderr.read()
    _, _, ex, code, primary = NEG["N30"]
    assert pr.returncode == ex and f"STOP {code}" in err and primary in err, err
    assert head(run) == before
    assert not list((run / "accepted").glob("*.blend")) if (run / "accepted").exists() else True
    js = [j for _, j in E.read_journals(run) if j["op"] == "ACCEPT"]
    assert len(js) == 1 and js[0]["state"] == "ABORTED" and len(js[0]["quarantined"]) == 3
    st = assert_consistent(run, v0)
    assert st["edits"]["P1"]["decision"].startswith("UNDECIDED") and st["edits"]["P1"]["gate"] != "ACCEPT"
    assert load(sorted((run / "edits/P1/authorization_refusals").glob("*.json"))[-1])["primary"] == primary


@needs_blender
def test_N23_mock_nonconverged_candidate_cannot_be_forged_into_acceptance(run, monkeypatch):
    """MOCK: the frozen resolver STATUS is stubbed to NON_CONVERGED (its solved values are kept), in-process."""
    run, v0, rw = run
    real = S.resolve_frame

    def nonconverged(*a, **k):
        st, vals, info = real(*a, **k)
        return "NON_CONVERGED", vals, dict(info, mock="W126 MOCK status")

    monkeypatch.setattr(S, "resolve_frame", nonconverged)
    E.dump(run / "requests" / "NC.json", _req("NC", v0["state_id"], [12, 15], 12))
    E.cmd_propose(Namespace(run=str(run), request=str(run / "requests" / "NC.json")))
    p = rprop(run, "NC")
    assert p["decision"] == {"decision": "REJECT", "failure_class": "SOLVER_FAILURE", "reasons": ["NON_CONVERGED"]} and p["candidate"]
    assert (run / "edits/NC/candidate.blend").exists()
    wprop(run, "NC", reseal(_forge_accept(p)))
    before = head(run)
    with pytest.raises(E.SliceStop) as ei:
        E.cmd_commit(Namespace(run=str(run), request_id="NC", decision="accept"))
    assert ei.value.code == AUTH and NEG["N23"][4] in ei.value.detail
    assert head(run) == before and not (run / "edits/NC/commit.json").exists()
    rec = load(sorted((run / "edits/NC/authorization_refusals").glob("*.json"))[-1])
    assert rec["primary"] == "RECOMPUTED_GATE_REJECT" and rec["detail"]["recomputed"]["decision"]["failure_class"] == "SOLVER_FAILURE"


# ------------------------------------------------------------------------------- recorded (preserved) cases ---
@needs_blender
def test_recorded_rejections_never_move_head(run):
    run, v0, rw = run
    r = accept(run, "REACH")                                                           # N22
    assert r.returncode == 0 and head(run)["generation"] == 0
    assert load(run / "edits/REACH/commit.json")["store_record"]["reasons"][:2] == ["GATE_REJECTED_CANNOT_ACCEPT", "UNREACHABLE"]
    p = rprop(run, "P1")                                                               # N31
    wprop(run, "P1", dict(p, decision={"decision": "ACCEPT", "failure_class": "NONE", "reasons": []}, hard_pin_residual_max_m=0.0))
    r = cli(run, "commit", "--request-id", "P1", "--decision", "reject")
    assert r.returncode == 0, r.stderr
    cm = load(run / "edits/P1/commit.json")
    assert cm["final_status"] == "REJECTED" and cm["proposal_integrity"]["reasons"][0] == "PROPOSAL_SEAL_INVALID"
    assert cm["display_final"] is None and (run / "edits/P1/timeline_final.png").exists()
    st = status(run)
    assert st["edits"]["P1"]["gate"].startswith("UNTRUSTED") and st["edits"]["P1"]["decision"] == "REJECTED"
    assert head(run)["generation"] == 0
    r = accept(run, "P2")
    assert r.returncode == 0 and head(run)["generation"] == 1                          # P2 valid -> accepted
    r = accept(run, "P3")                                                              # N20: P3 now stale
    assert r.returncode == 0 and load(run / "edits/P3/commit.json")["store_record"]["reasons"] == ["STALE_PARENT"]
    assert head(run)["generation"] == 1
    assert_consistent(run, v0)


# ------------------------------------------------------------------------------------------- valid cases ---
@needs_blender
def test_V01_V04_valid_pin_accepted_with_independent_blender_evidence_and_exact_revert(run, tmp_path):
    import numpy as np
    run, v0, rw = run
    p1 = rprop(run, "P1")
    r = accept(run, "P1")
    assert r.returncode == 0, r.stderr
    h = head(run)
    assert h["head"] == p1["candidate"] and h["generation"] == 1
    cm = load(run / "edits/P1/commit.json")
    sa = cm["semantic_authorization"]
    assert sa["authorized"] is True and sa["pre_staging_gate"]["decision"]["decision"] == "ACCEPT"
    assert sa["fresh_readback_gate"]["decision"]["decision"] == "ACCEPT" and sa["hard_pin_residual_max_m_fresh_readback"] <= 1e-3
    assert sa["fresh_readback_gate"]["changed_frames"] == cm["publication_checks"]["actual_changed_frames"] != []
    assert set(sa["fresh_readback_gate"]["changed_frames"]) <= set(p1["pin_frames"])
    assert sa["fresh_readback_gate"]["changed_bones"] == ["lowerarm_l", "upperarm_l"]
    # independent check from this process: a NEW Blender process re-opens the published head .blend
    out = tmp_path / "indep_readback.json"
    E.run_blender(BLENDER, "readback", tmp_path / "indep.log", blend=run / h["head_blend"]["blend"], report=out,
                  subframes=",".join(str(s) for s in E.SUBFRAMES))
    rb = load(out)
    assert rb["value_digest"] == h["head_blend"]["value_digest"] and rb["blend_sha256"] == h["head_blend"]["blend_sha256"]
    sidecar = load(run / "baseline/action_sidecar.json")
    X = E.readback_by_source(rb, sidecar)
    U = E.readback_by_source(load(run / "baseline/baseline_readback.json"), sidecar)
    tgt = np.array(p1["constraint"]["target"])
    resid = max(float(np.linalg.norm(np.array(X[k]["world"]["lowerarm_l"]["tail"]) - tgt)) for k in p1["pin_frames"])
    assert resid <= 1e-3
    changed = [k for k in X if X[k]["stored_wxyz"] != U[k]["stored_wxyz"] or X[k]["world"] != U[k]["world"]]
    assert sorted(changed) == sa["fresh_readback_gate"]["changed_frames"]
    sub = E.subframe_report(rb, load(run / "baseline/baseline_readback.json"), sidecar, p1["constraint"],
                            {"bone": "lowerarm_l", "point": "tail"})
    assert sub["changed_intervals_within_pin_frames"]
    st = status(run)
    assert st["edits"]["P1"]["semantic_authorization"].startswith("AUTHORIZED_AT_COMMIT") and st["edits"]["P1"]["gate"] == "ACCEPT"
    # V04 exact revert
    r = cli(run, "revert")
    assert r.returncode == 0, r.stderr
    h2 = head(run)
    rep = load(run / f"reverted/reverted_{h2['head'][:12]}_report.json")
    assert rep["exact_undo"] and rep["blender_eval_integer_frames_bit_equal"] and rep["blender_eval_subframes_bit_equal"]
    assert_consistent(run, v0)


@needs_blender
def test_V02_noop_pin_is_trivial_not_evidence(run):
    run, v0, rw = run
    r = accept(run, "NOOP")
    assert r.returncode == 0, r.stderr
    cm = load(run / "edits/NOOP/commit.json")
    assert cm["final_status"] == "ACCEPTED" and cm["semantic_authorization"]["authorized"] is True
    assert cm["semantic_authorization"]["fresh_readback_gate"]["decision"]["reasons"] == ["TRIVIALLY_SATISFIED_NOT_EVIDENCE"]
    assert cm["publication_checks"]["actual_changed_frames"] == [] and cm["publication_checks"]["actual_changed_bones"] == []
    assert_consistent(run, v0)


@needs_blender
def test_V03_explicit_reject_then_accept_flag_refused(run):
    run, v0, rw = run
    r = cli(run, "commit", "--request-id", "P3", "--decision", "reject")
    assert r.returncode == 0
    cm = load(run / "edits/P3/commit.json")
    assert cm["final_status"] == "REJECTED" and cm["proposal_integrity"] == {"kind": "V2", "reasons": []}
    r = accept(run, "P3")
    assert r.returncode == 2 and "REFUSED_ALREADY_DECIDED" in r.stderr and head(run)["generation"] == 0


@needs_blender
def test_V05_benign_reserialization_is_accepted(run):
    run, v0, rw = run
    b = load(run / "baseline/bindings.json")
    (run / "baseline/bindings.json").write_text(json.dumps(b, indent=4), encoding="utf-8")     # same content, other bytes
    wprop(run, "P1", rprop(run, "P1"), indent=4)
    r = accept(run, "P1")
    assert r.returncode == 0, r.stderr
    sa = load(run / "edits/P1/commit.json")["semantic_authorization"]
    assert sa["authorized"] is True and sa["bindings_file_bytes_equal_proposal"] is False
    assert_consistent(run, v0)


# ------------------------------------------------------------------------------------------------ recovery ---
@needs_blender
def test_R01_recovery_never_invents_authorization(run):
    run, v0, rw = run
    r = accept(run, "P1", fault="after_publish", mode="exit")
    assert r.returncode == TX.FAULT_EXIT_CODE
    jp, j = [(p, j) for p, j in E.read_journals(run) if j["op"] == "ACCEPT"][0]
    assert j["commit_result"]["semantic_authorization"]["authorized"] is True
    j2 = dict(j, commit_result=dict(j["commit_result"], semantic_authorization=dict(j["commit_result"]["semantic_authorization"],
                                                                                    authorized=False)))
    TX.atomic_dump(jp, E._sealed({k: v for k, v in j2.items() if k != "journal_digest"}, "journal_digest"))
    h = head(run)
    rc = cli(run, "recover")
    assert rc.returncode == 2 and "BLOCKED_DIVERGENT_JOURNAL" in rc.stderr
    assert head(run) == h and not (run / "edits/P1/commit.json").exists()


@needs_blender
def test_R02_pre_w126_published_journal_rolls_forward_marked_not_established(run):
    run, v0, rw = run
    r = accept(run, "P1", fault="after_publish", mode="exit")
    assert r.returncode == TX.FAULT_EXIT_CODE
    jp, j = [(p, j) for p, j in E.read_journals(run) if j["op"] == "ACCEPT"][0]
    cr = {k: v for k, v in j["commit_result"].items() if k != "semantic_authorization"}
    TX.atomic_dump(jp, E._sealed({k: v for k, v in dict(j, commit_result=cr).items() if k != "journal_digest"}, "journal_digest"))
    rc = cli(run, "recover")
    assert rc.returncode == 0 and "ROLLED_FORWARD" in rc.stdout, rc.stderr
    cm = load(run / "edits/P1/commit.json")
    assert cm["final_status"] == "ACCEPTED" and cm["semantic_authorization"]["authorized"] is None
    st = assert_consistent(run, v0)
    assert st["edits"]["P1"]["semantic_authorization"].startswith("NOT_ESTABLISHED")
