"""W138 qualification rule: SYNTHETIC fact tables only (no network, no model, no S1 data)."""
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("w138_generator_qualification", ROOT / "scripts" / "w138_generator_qualification.py")
Q = importlib.util.module_from_spec(spec)
sys.modules["w138_generator_qualification"] = Q
spec.loader.exec_module(Q)

ALL_TRUE = {p: Q.TRUE for p in Q.PREDICATES}
PREF = {p: True for p in Q.PREFERENCES}


def test_all_true_is_eligible():
    e = Q.evaluate_candidate(ALL_TRUE)
    assert e["eligible"] and e["status"] == "ELIGIBLE" and e["blocking"] == []


@pytest.mark.parametrize("pred", Q.PREDICATES)
def test_each_false_predicate_blocks_with_its_stop_code(pred):
    e = Q.evaluate_candidate({**ALL_TRUE, pred: Q.FALSE})
    assert not e["eligible"] and e["status"] == Q.STOP_OF[pred] and e["blocking"] == [pred]


@pytest.mark.parametrize("pred", Q.PREDICATES)
def test_unverified_or_missing_predicate_never_counts_as_true(pred):
    e = Q.evaluate_candidate({**ALL_TRUE, pred: Q.UNV})
    assert not e["eligible"] and e["status"] == f"NOT_VERIFIED:{pred}"
    missing = {k: v for k, v in ALL_TRUE.items() if k != pred}
    assert not Q.evaluate_candidate(missing)["eligible"]


def test_verified_false_outranks_not_verified_in_the_status():
    e = Q.evaluate_candidate({**ALL_TRUE, "E1c_code_rights": Q.UNV, "E4_conditioning": Q.FALSE})
    assert e["status"] == "CONDITIONING_BLOCKED" and set(e["blocking"]) == {"E1c_code_rights", "E4_conditioning"}


def test_unknown_value_is_not_eligible():
    assert not Q.evaluate_candidate({**ALL_TRUE, "E2_access": "yes"})["eligible"]


def test_no_eligible_candidate_is_qualification_blocked():
    s = Q.select({"X": {"predicates": {**ALL_TRUE, "E1c_code_rights": Q.FALSE}, "preferences": PREF},
                  "Y": {"predicates": {}, "preferences": PREF}})
    assert s["selected"] is None and s["status"] == "QUALIFICATION_BLOCKED"


def test_unique_eligible_candidate_is_selected():
    s = Q.select({"X": {"predicates": ALL_TRUE, "preferences": {}}, "Y": {"predicates": {**ALL_TRUE, "E2_access": Q.FALSE}, "preferences": PREF}})
    assert s["selected"] == "X" and s["status"] == "SELECTED" and s["not_selected_eligible"] == []


def test_preference_is_lexicographic_and_selects_at_most_one():
    p = Q.PREFERENCES
    s = Q.select({"A": {"predicates": ALL_TRUE, "preferences": {p[0]: True, p[1]: False, p[2]: True}},
                  "B": {"predicates": ALL_TRUE, "preferences": {p[0]: True, p[1]: True, p[2]: False}}})
    assert s["selected"] == "B" and s["not_selected_eligible"] == ["A"]


def test_tie_at_the_maximum_is_refused():
    s = Q.select({"A": {"predicates": ALL_TRUE, "preferences": PREF}, "B": {"predicates": ALL_TRUE, "preferences": PREF}})
    assert s["selected"] is None and s["status"] == "QUALIFICATION_BLOCKED" and "ambiguous" in s["reason"]


def test_ineligible_candidate_cannot_win_on_preference():
    s = Q.select({"A": {"predicates": ALL_TRUE, "preferences": {}}, "B": {"predicates": {**ALL_TRUE, "E1w_weight_rights": Q.UNV}, "preferences": PREF}})
    assert s["selected"] == "A"


def test_tri():
    assert Q.tri(True, True) == Q.TRUE and Q.tri(True, None) == Q.UNV and Q.tri(None, False) == Q.FALSE and Q.tri() == Q.TRUE


def test_clauses_are_whitespace_normalised_substrings():
    text = "The  Kimodo text\nencoder relies on the **gated** model."
    assert Q.clauses_present(text, ["Kimodo text encoder relies", "not there"]) == {"Kimodo text encoder relies": True, "not there": False}


@pytest.mark.parametrize("status,cls", [(200, "GRANTED"), (302, "GRANTED"), (401, "NOT_GRANTED"), (403, "NOT_GRANTED"), (404, "UNKNOWN"), (None, "UNKNOWN")])
def test_gated_head_classes(status, cls):
    assert Q.classify_gated_head(status) == cls


def test_licence_file_pattern():
    hit = [f for f in ("LICENSE", "LICENSE.md", "docs/LICENCE.txt", "COPYING", "NOTICE", "src/licenses.py", "README.md", "LICENSES/x.txt")
           if Q.LICENCE_FILE_RE.search(f)]
    assert hit == ["LICENSE", "LICENSE.md", "docs/LICENCE.txt", "COPYING", "NOTICE"]


def _recs(gate_cls="NOT_GRANTED", no_text=True, kimodo_lic=True):
    def cand(lic_files, spdx, card_name=None, card=None, author=None, full=None, gated=False, topics=()):
        return {"code": {"head_matches_remote": True, "licence_files_in_tree": lic_files, "github_api_license_spdx": spdx, "github_api_status": 200,
                         "github_full_name": full},
                "checkpoint": {"api_status": 200, "gated": gated, "weight_file_listed": True, "weight_lfs_sha256": "ab", "weight_bytes": 1e9,
                               "card_license_name": card_name, "card_license": card, "author": author},
                "clauses": {t: {"all_present": v} for t, v in topics}}
    nv = [("code_licence", True), ("weight_licence", True), ("text_gate", True), ("no_text_mode", no_text), ("resources", True), ("conditioning", True),
          ("output", True), ("per_frame", True)]
    recs = {"Kimodo": cand(["LICENSE"] if kimodo_lic else [], "Apache-2.0", "nvidia-open-model-license", topics=nv),
            "ARDY": cand(["LICENSE"], "Apache-2.0", "nvidia-open-model-agreement", topics=nv[:7] + [("autoregressive_tokenised", True)]),
            "MotionLab": cand([], None, topics=[("restricted_assets", True)]),
            "ProjFlow": cand([], None, card="mit", author="A", full="A/ProjFlow", topics=[("weight_licence", True), ("weights_hosted_by_author", True), ("conditioning", True)]),
            "GEM-X": cand(["LICENSE"], "Apache-2.0", "nvidia-open-model-license", topics=[("code_licence", True), ("weight_licence", True), ("video_estimator", True)])}
    return recs, {"class": gate_cls}


def test_table_selects_kimodo_only_in_the_no_text_mode():
    recs, gate = _recs()
    t = Q.build_table(recs, gate, 300.0)
    s = Q.select(t)
    assert s["selected"] == "Kimodo" and s["not_selected_eligible"] == ["ARDY"]
    assert t["Kimodo"]["modes"]["text_conditioned"].startswith("ACCESS_BLOCKED")
    assert s["evaluation"]["MotionLab"]["status"] == "RIGHTS_UNRESOLVED" and s["evaluation"]["ProjFlow"]["status"] == "RIGHTS_UNRESOLVED"
    assert s["evaluation"]["GEM-X"]["status"] == "CONDITIONING_BLOCKED"
    assert t["ProjFlow"]["predicates"]["E1w_weight_rights"] == Q.TRUE and t["ProjFlow"]["predicates"]["E1c_code_rights"] == Q.FALSE


def test_table_blocks_everything_when_the_no_text_mode_is_not_verified_and_the_gate_is_closed():
    recs, gate = _recs(no_text=False)
    s = Q.select(Q.build_table(recs, gate, 300.0))
    assert s["selected"] is None and s["status"] == "QUALIFICATION_BLOCKED"
    assert s["evaluation"]["Kimodo"]["status"] == "ACCESS_BLOCKED"


def test_table_blocks_kimodo_without_a_code_licence_file():
    recs, gate = _recs(kimodo_lic=False)
    s = Q.select(Q.build_table(recs, gate, 300.0))
    assert s["evaluation"]["Kimodo"]["status"] == "RIGHTS_UNRESOLVED" and s["selected"] == "ARDY"


def test_table_resource_gate():
    recs, gate = _recs()
    s = Q.select(Q.build_table(recs, gate, 5.0))
    assert s["evaluation"]["Kimodo"]["status"] == "RESOURCE_BLOCKED"
