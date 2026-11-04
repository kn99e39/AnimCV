"""W113 scratch Animation Base contract: join invariants, rejections, lock/edit locality, UNKNOWN preservation."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import w113_animation_base_contract as C  # noqa: E402
import w113_animation_base_view as V  # noqa: E402


@pytest.fixture()
def syn(tmp_path):
    return C.synthetic_view(tmp_path)


def _expected(view):
    return C.expected_of(view, {"synthetic_source": view["meta"]["sources"]["synthetic_source"]})


PRESERVE = [{"id": "P_ra", "kind": "preserve", "signal": "est3d_xyz", "joints": C.RIGHT_ARM, "spans": [[0, 7]]},
            {"id": "P_f", "kind": "preserve", "signal": "est3d_xyz", "joints": list(V.JOINTS), "spans": [[0, 1]]}]
PERMIT = [{"id": "E_lw", "kind": "permit", "signal": "est3d_xyz", "joints": ["left_wrist"], "spans": [[3, 5]]},
          {"id": "E_le", "kind": "permit", "signal": "est3d_xyz", "joints": ["left_elbow"], "spans": [[4, 4]]}]


def test_synthetic_view_valid_and_roundtrip_exact(syn, tmp_path):
    assert V.validate_view(syn, expected=_expected(syn))["frames"] == 8
    rt = C.roundtrip(syn, tmp_path / "v")
    assert rt["digest_and_arrays_identical"] and rt["reloaded_valid"] is True
    back = V.load_view(tmp_path / "v")
    assert np.isnan(back["arrays"]["obs2d_xy_px"][6, V.JI["left_ankle"]]).all()  # UNKNOWN stays NaN, not zero


def test_view_never_fills_unknown_or_manufactures_3d_confidence(syn):
    a = syn["arrays"]
    assert not a["obs2d_present"][6, V.JI["left_ankle"]] and np.isnan(a["obs2d_conf"][6, V.JI["left_ankle"]])
    assert not a["est3d_finite"][4, V.JI["left_elbow"]]
    assert not any(k in a for k in ("est3d_confidence", "depth_uncertainty", "root_translation", "target_3d"))
    st = V.cell_status(syn)
    assert st["est3d_status"][3, V.JI["left_wrist"]] == 2  # finite 3D on invalid input is labelled unsupported
    assert syn["meta"]["unavailable"]["root_translation"] == {"status": "unavailable",
                                                             "reason": V.UNAVAILABLE["root_translation"]}


@pytest.mark.parametrize("name", sorted(V.VIEW_CONTROLS))
def test_each_negative_control_rejects_with_declared_code(syn, name):
    res = V.run_view_controls(syn, _expected(syn))
    assert res["positive_unchanged_view"]["ok"]
    assert res[name]["observed"] == V.VIEW_CONTROLS[name][1], res[name]


def test_stale_source_file_detected(syn, tmp_path):
    Path(syn["meta"]["sources"]["synthetic_source"]["path"]).write_bytes(b"changed")
    with pytest.raises(V.ContractError) as e:
        V.validate_view(syn)
    assert e.value.code == "PROVENANCE_STALE"


def test_nested_gt_key_rejected_and_ordinary_keys_not(syn):
    V.validate_view(syn)  # "right_*", "digest" keys present and accepted
    bad = V._clone(syn)
    bad["meta"]["producers"]["est3d"] = {"model": "x", "target_valid_used": False}
    V._redigest(bad)
    with pytest.raises(V.ContractError) as e:
        V.validate_view(bad)
    assert e.value.code == "GT_LEAKAGE"


def test_directive_masks_and_lock_conflict(syn):
    masks = V.resolve_directives(syn, V.make_directive_set(syn, "s", PRESERVE + PERMIT))
    assert masks["permit"].sum() == 4 and masks["preserve"][0].all() and masks["preserve"][:, V.JI["right_wrist"]].all()
    conflict = PERMIT + [{"id": "c", "kind": "permit", "signal": "est3d_xyz", "joints": ["left_wrist"], "spans": [[0, 5]]}]
    with pytest.raises(V.ContractError) as e:
        V.resolve_directives(syn, V.make_directive_set(syn, "s", PRESERVE + conflict))
    assert e.value.code == "LOCK_CONFLICT"


def test_edit_locality_exact(syn):
    dset = V.make_directive_set(syn, "s", PRESERVE + PERMIT)
    cells = [(3, "left_wrist"), (4, "left_wrist"), (5, "left_wrist")]
    after, log = V.apply_mock_edit(syn, dset, V.make_edit(syn, "ok", cells))
    requested = np.zeros((8, V.J), bool)
    requested[3:6, V.JI["left_wrist"]] = True
    loc = V.locality(syn, after, V.resolve_directives(syn, dset)["permit"], requested)
    assert loc["pass"] and loc["changed_cells"] == 3 and loc["unedited_est3d_byte_identical"] and loc["unknown_preserved"]
    assert after["meta"]["parent_content_digest"] == syn["meta"]["content_digest"] != after["meta"]["content_digest"]
    V.validate_view(after)
    # a vacuous locality check must fail: claiming a different requested set is detected
    wrong = requested.copy()
    wrong[3, V.JI["left_wrist"]] = False
    assert not V.locality(syn, after, V.resolve_directives(syn, dset)["permit"], wrong)["pass"]


@pytest.mark.parametrize("cells,signal,code", [
    ([(4, "right_wrist")], "est3d_xyz", "LOCK_CONFLICT_EDIT"),
    ([(4, "left_knee")], "est3d_xyz", "EDIT_NOT_AUTHORIZED"),
    ([(4, "left_elbow")], "est3d_xyz", "EDIT_UNKNOWN_TARGET"),
    ([(3, "left_wrist")], "obs2d_xy_px", "OBSERVATION_IMMUTABLE"),
    ([(3, "left_wrist"), (4, "right_wrist")], "est3d_xyz", "LOCK_CONFLICT_EDIT"),  # no partial application
])
def test_edit_rejections(syn, cells, signal, code):
    dset = V.make_directive_set(syn, "s", PRESERVE + PERMIT)
    with pytest.raises(V.ContractError) as e:
        V.apply_mock_edit(syn, dset, V.make_edit(syn, "x", cells, signal=signal))
    assert e.value.code == code
    assert syn["meta"]["edits"] == [] and (syn["arrays"]["est3d_provenance"] == 0).all()


def test_directive_identity_and_staleness(syn):
    for override, code in (({"actor_id": "a2"}, "IDENTITY_MISMATCH"), ({"sequence_id": "s2"}, "IDENTITY_MISMATCH"),
                           ({"view_content_digest": "0" * 64}, "DIRECTIVE_STALE_VIEW")):
        with pytest.raises(V.ContractError) as e:
            V.resolve_directives(syn, V.make_directive_set(syn, "s", PRESERVE + PERMIT, **override))
        assert e.value.code == code
    req = V.make_edit(syn, "x", [(3, "left_wrist")])
    req["view_content_digest"] = "1" * 64
    with pytest.raises(V.ContractError) as e:
        V.apply_mock_edit(syn, V.make_directive_set(syn, "s", PRESERVE + PERMIT), req)
    assert e.value.code == "DIRECTIVE_STALE_VIEW"


def test_mixed_unit_motion_graph_rejected():
    graph = {"frames": [{"points": {"left_wrist": {"position_2d": [10.0, 20.0], "position_3d": None}}}]}
    assert V.reject_mixed_unit_motion_graph(graph)["status"] == "unavailable_in_motion_graph"
    graph["frames"][0]["points"]["left_wrist"]["position_3d"] = [10.0, 20.0, 0.4]
    with pytest.raises(V.ContractError) as e:
        V.reject_mixed_unit_motion_graph(graph)
    assert e.value.code == "MIXED_UNIT_3D"


class _Guarded(dict):
    def __getitem__(self, key):
        if key.startswith("target"):
            raise AssertionError(f"consumer read {key}")
        return super().__getitem__(key)


def test_source_only_consumer_isolation():
    from motion.animation_semantics_v2_bridge import current_root_orientation
    from pose.framepose_bridge import build_h0_lifted_sequence
    T = 5
    z = {"frame_index": np.arange(T), "timestamp": np.arange(T) / 50.0,
         "input_2d": np.random.default_rng(0).uniform(0.2, 0.8, (T, 17, 3)).astype(np.float32),
         "input_valid": np.ones((T, 17), bool)}
    bank = C.scratch_real_bank(z, (1280, 720), 50.0)
    bank.arrays = _Guarded(bank.arrays)
    pred = np.random.default_rng(1).normal(0, .3, (T, 17, 3)).astype(np.float32)
    lifted = build_h0_lifted_sequence(bank, pred, C.REAL_SEQUENCE_ID, fps=50.0)
    assert len(current_root_orientation(lifted)) == T
    assert [f.timestamp for f in lifted.frames] == list(z["timestamp"])


def test_case_precedence():
    assert C.case_of(["repeat"], ["M1_obs2d"], {"x": True}) == "D"
    assert C.case_of([], ["M1_obs2d"], {"x": True}) == "B"
    assert C.case_of([], [], {"x": True, "y": True}) == "A"
    assert C.case_of([], [], {"x": True, "y": False}) == "C"
    assert C.case_of([], [], {}) == "C"


def test_runs_helper():
    assert V.runs(np.array([0, 1, 1, 0, 1, 0, 0, 1], bool)) == [(1, 2), (4, 4), (7, 7)]
    assert V.runs(np.zeros(3, bool)) == []


def test_frozen_confidence_domain_rejects_detector_score_above_one(syn):
    """The frozen W113 gate (declaration c3f7485c) is [0, 1]; authentic RTMPose/3DPW scores exceed it -> CASE D."""
    assert V.CONFIDENCE_UPPER == 1.0
    bad = V._clone(syn)
    bad["arrays"]["obs2d_conf"][0, V.JI["left_ankle"]] = 1.05
    V._redigest(bad)
    with pytest.raises(V.ContractError) as e:
        V.validate_view(bad)
    assert e.value.code == "CONFIDENCE_AMBIGUOUS"
    V.validate_view(syn)  # in-range scores still pass (positive)
