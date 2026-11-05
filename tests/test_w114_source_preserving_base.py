"""W114 scratch candidate: evidence immutability, score semantics, versioned edits/reverts, constraints, rig authority."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import w114_source_preserving_base as B  # noqa: E402
import w114_source_preserving_contract as C  # noqa: E402


@pytest.fixture()
def syn(tmp_path):
    ev, ref = C.synthetic_evidence(tmp_path)
    return ev, ref, C.synth_directives(ev), C.toy_rig()


def code_of(fn):
    try:
        fn()
        return "ACCEPT"
    except B.ContractError as e:
        return e.code


def test_score_above_one_is_preserved_and_valid(syn):
    ev, ref, _, _ = syn
    assert B.verify_evidence(ev)["frames"] == 8
    assert ev["arrays"]["obs2d_score_raw"][2, B.JI["left_knee"]] == 1.07
    assert B.verify_against_source(ev, ref)
    acc = B.score_accounting(ev, 0.3)
    assert acc["above_1"] == 1 and acc["one_minus_score_negative_if_formed"] == 1


def test_clipping_scores_is_a_source_mismatch(syn):
    ev, ref, _, _ = syn
    clipped = C.mutated(ev, lambda v: v["arrays"]["obs2d_score_raw"].__setitem__(slice(None), np.minimum(v["arrays"]["obs2d_score_raw"], 1)))
    B.verify_evidence(clipped)  # no score bound exists
    assert code_of(lambda: B.verify_against_source(clipped, ref)) == "SOURCE_MISMATCH"


def test_evidence_is_read_only_and_digest_bound(syn):
    ev = syn[0]
    with pytest.raises(ValueError):
        ev["arrays"]["obs2d_score_raw"][0, 0] = 0.0
    bad = C.mutated(ev, lambda v: v["arrays"]["obs2d_xy_px"].__setitem__((0, 0, 0), 1.0), redigest=False)
    assert code_of(lambda: B.verify_evidence(bad)) == "EVIDENCE_MODIFIED"


def test_unknown_and_manufactured_rejections(syn):
    ev = syn[0]
    zero = C.mutated(ev, lambda v: (v["arrays"]["obs2d_xy_px"].__setitem__((6, B.JI["left_ankle"]), 0.0),
                                    v["arrays"]["obs2d_score_raw"].__setitem__((6, B.JI["left_ankle"]), 0.0)))
    assert code_of(lambda: B.verify_evidence(zero)) == "UNKNOWN_COERCED_TO_ZERO"
    rel = C.mutated(ev, lambda v: v["arrays"].__setitem__("est3d_reliability", 1 - np.nan_to_num(v["arrays"]["obs2d_score_raw"])))
    assert code_of(lambda: B.verify_evidence(rel)) == "CONFIDENCE_MANUFACTURED_3D"
    nan = C.mutated(ev, lambda v: v["arrays"]["obs2d_score_raw"].__setitem__((0, 0), np.nan))
    assert code_of(lambda: B.verify_evidence(nan)) == "SCORE_MISSING"
    assert ev["arrays"]["est3d_support"][4, B.JI["left_elbow"]] == 0
    assert ev["arrays"]["est3d_support"][3, B.JI["left_wrist"]] == 2


def test_edit_locality_reversal_and_lineage(syn):
    ev, _, cs, _ = syn
    lin = B.Lineage(ev)
    root = lin.state()
    cells = [(3, "left_wrist"), (5, "left_wrist")]
    s1 = B.apply_edit(lin, ev, cs["SS1"], B.make_mock_edit(lin, ev, cs["SS1"], "E1", cells))
    loc = C.locality(root, s1, cells, ev)
    assert loc["changed_equals_requested"] and loc["unintended"] == 0 and loc["origin_changed_equals_requested"]
    assert np.array_equal(ev["arrays"]["est3d_xyz_hypothesis"], root["values"], equal_nan=True)
    s2 = B.apply_edit(lin, ev, cs["SS1"], B.make_mock_edit(lin, ev, cs["SS1"], "E2", cells[:1], (0.005, 0, 0)))
    stale = B.make_mock_edit(lin, ev, cs["SS1"], "E3", cells[:1], base=s1["state_id"])
    before = lin.snapshot()
    assert code_of(lambda: B.apply_edit(lin, ev, cs["SS1"], stale)) == "STALE_PARENT" and lin.snapshot() == before
    r2 = B.apply_revert(lin, ev, B.make_revert(lin, "E2"))
    assert r2["values"].tobytes() == s1["values"].tobytes()
    r1 = B.apply_revert(lin, ev, B.make_revert(lin, "E1"))
    assert r1["values"].tobytes() == root["values"].tobytes() and r1["origin"].tobytes() == root["origin"].tobytes()
    assert lin.audit()["ids_recomputed_equal"] and lin.audit()["states"] == 5
    assert s2["state_id"] in lin.states


def test_tampered_revert_and_revert_conflict(syn):
    ev, _, cs, _ = syn
    lin = B.Lineage(ev)
    B.apply_edit(lin, ev, cs["SS1"], B.make_mock_edit(lin, ev, cs["SS1"], "E1", [(3, "left_wrist")]))
    rv = B.make_revert(lin, "E1")
    h = bytearray(bytes.fromhex(rv["restore_values_hex"]))
    h[0] ^= 1
    rv["restore_values_hex"] = bytes(h).hex()
    assert code_of(lambda: B.apply_revert(lin, ev, rv)) == "REVERT_MISMATCH"
    B.apply_edit(lin, ev, cs["SS1"], B.make_mock_edit(lin, ev, cs["SS1"], "E2", [(3, "left_wrist")]))
    assert code_of(lambda: B.apply_revert(lin, ev, B.make_revert(lin, "E1"))) == "REVERT_CONFLICT"


def test_permission_conflict_identity_and_time(syn):
    ev, _, cs, _ = syn
    lin = B.Lineage(ev)
    mk = lambda cells, **kw: B.make_mock_edit(lin, ev, cs["SS1"], "x", cells, **kw)
    assert code_of(lambda: B.apply_edit(lin, ev, cs["SS1"], mk([(4, "left_knee")]))) == "EDIT_NOT_AUTHORIZED"
    assert code_of(lambda: B.apply_edit(lin, ev, cs["SS1"], mk([(1, "left_wrist")]))) == "LOCK_CONFLICT_EDIT"
    assert code_of(lambda: B.apply_edit(lin, ev, cs["SS1"], mk([(4, "left_elbow")]))) == "EDIT_UNKNOWN_TARGET"
    assert code_of(lambda: B.apply_edit(lin, ev, cs["SS1"], mk([(4, "left_wrist")], op="copy_from_frame", source_frame=5))) == "IDENTITY_UNVERIFIED"
    assert code_of(lambda: B.apply_edit(lin, ev, cs["SS1"], mk([(3, "left_wrist")], op="copy_from_frame", source_frame=2))) == "ACCEPT"
    rec = mk([(4, "left_wrist")])
    rec["target_timestamps"] = [t + 0.04 for t in rec["target_timestamps"]]
    assert code_of(lambda: B.apply_edit(lin, ev, cs["SS1"], rec)) == "TIME_MISMATCH"
    assert code_of(lambda: B.apply_edit(lin, ev, cs["SS1"], mk([(9, "left_wrist")]))) == "FRAME_MISMATCH"
    assert code_of(lambda: B.apply_edit(lin, ev, cs["SS1"], mk([(4, "left_wrist")], unit="pixels"))) == "COORDINATE_MISMATCH"
    assert code_of(lambda: B.apply_edit(lin, ev, cs["SS1"], mk([(4, "left_wrist")], signal="obs2d_score_raw"))) == "OBSERVATION_IMMUTABLE"
    bad = copy.deepcopy(cs["SS1"])
    bad["directives"].append({"id": "c", "kind": "permit", "joints": ["right_wrist"], "spans": [[2, 2]]})
    assert code_of(lambda: B.resolve_constraints(ev, bad)) == "LOCK_CONFLICT"


def test_rig_authority_toy(syn):
    ev, _, cs, toy = syn
    lin = B.Lineage(ev)
    cells = [(3, "left_wrist"), (4, "left_wrist"), (5, "left_wrist")]
    before = lin.snapshot()
    rec = B.make_mock_edit(lin, ev, cs["SS1"], "W", cells)
    assert code_of(lambda: B.apply_edit(lin, ev, cs["SS1"], rec, rig=toy)) == "OUTPUT_OUTSIDE_AUTHORIZED_DOMAIN"
    assert lin.snapshot() == before
    assert C.independent_authority(toy, ev, cs["SS1"], lin.state()["values"], C.source_level_after(ev, cs["SS1"], rec), False) \
        == "OUTPUT_OUTSIDE_AUTHORIZED_DOMAIN"
    rec = B.make_mock_edit(lin, ev, cs["SS1_arm"], "A", cells)
    st = B.apply_edit(lin, ev, cs["SS1_arm"], rec, rig=toy)
    assoc = lin.records["A"]["record"]["output_association"]
    assert assoc["changed_channel_frames"] == {"upper_l": 2, "lower_l": 2}
    head = B.make_mock_edit(lin, ev, cs["SS1"], "H", [(3, "head")], rig_claim=True)
    assert code_of(lambda: B.apply_edit(lin, ev, cs["SS1"], head, rig=toy)) == "NO_RIG_INFLUENCE"
    a = B.make_association(toy, ev, lin, action_fps=30.0)
    assert B.validate_association(a, toy, ev, lin)["ok"]
    a["channel_owner"]["lower_l"] = "j1x"
    assert code_of(lambda: B.validate_association(a, toy, ev, lin)) == "ASSOCIATION_INVALID"
    assert st["state_id"] == lin.head


def test_synthetic_controls_exact(syn, tmp_path):
    ev, ref, cs, toy = syn
    ok = [(3, "left_wrist"), (4, "left_wrist"), (5, "left_wrist")]
    res = C.run_controls("synthetic", ev, ref, cs["SS1"], ok, (1, "left_wrist"), (4, "left_knee"), rig=toy, cs_unconsumed=cs["SS1"],
                         unconsumed_cells=[(3, "head")], out_of_domain=ok, cross=(5, (4, "left_wrist")))
    summ = C.control_summary(res)
    assert not summ["failed"], {k: res[k] for k in summ["failed"]}
    assert summ["not_applicable"] == [] and summ["applicable"] == len(C.CONTROLS)


def test_synthetic_programme(syn):
    ev, _, cs, toy = syn
    p = C.edit_programme("synthetic", ev, cs["SS1"], [(3, "left_wrist"), (4, "left_wrist"), (5, "left_wrist")], cs_arm=cs["SS1_arm"],
                         rig=toy, cs_head=cs["SS1"], head_cells=[(3, "head"), (5, "head")], cross=(5, (4, "left_wrist")),
                         E2_filter=lambda c: c[0] == 3)
    assert p["all_ok"], p["log"]
    assert p["rig_steps_consistent"] and p["rig_steps_prediction_match"], p["rig_log"]
    assert p["RV1_equals_hypothesis"] and p["RV2_equals_post_E1"] and p["final_equals_hypothesis"]
    assert p["inplace_write_refused"] and p["evidence_unchanged_end"]


def test_quat_delta_hemisphere_and_nan():
    q = np.array([[0.0, 0.0, 0.0, 1.0], [np.nan] * 4, [0.1, 0, 0, 0.995]])
    assert B.quat_delta(q, -q)[0] == 0.0
    assert B.quat_delta(q, q)[1] == 0.0
    r = q.copy()
    r[1] = [0, 0, 0, 1]
    assert np.isinf(B.quat_delta(q, r)[1])


W105 = Path("C:/Users/dna10/animcv-w105/results/build/video_cases.json")
W102 = Path("C:/Users/dna10/animcv-w102/results/historical_video/observations_predictions.npz")


@pytest.mark.skipif(not (W105.exists() and W102.exists()), reason="W102/W105 local evidence not present")
def test_frozen_real_rig_replica_reproduces_w105():
    import w114_rig_influence as R
    rc = R.rig_context()
    z = np.load(W102)
    frames = R.real_case_frames(z["predictions"], z["timestamp"], rc, "h")
    mine = json.loads(json.dumps(C.clean(R.slot_view(frames, "SLOT"))))
    ref = R.slot_view(json.loads(W105.read_text())["cases"]["real_clip"]["frames"], "FIXED")
    assert mine == ref
