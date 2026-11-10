"""W133 (experimental, opt-in): synthetic tests of the OmniControl preflight's pure helpers and declared terms rule.

No network, no model and no S1 data: in-memory ZIP archives and hand-written fact dictionaries only.
"""
from __future__ import annotations

import importlib.util
import io
import zipfile
from pathlib import Path

import pytest

_P = Path(__file__).resolve().parents[1] / "scripts" / "w133_omnicontrol_preflight.py"
_spec = importlib.util.spec_from_file_location("w133_omnicontrol_preflight", _P)
pf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pf)


def _zip(members):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("omnicontrol_ckpt/", b"")
        for name, data in members.items():
            z.writestr(name, data)
    return buf.getvalue()


def _listing(blob):
    eocd = pf.parse_eocd(blob[-22:])
    cd = blob[eocd["cd_offset"]:eocd["cd_offset"] + eocd["cd_size"]]
    return eocd, pf.parse_central_directory(cd, eocd["entries"])


def test_central_directory_from_tail_ranges_only():
    payload = bytes(range(256)) * 400
    blob = _zip({"omnicontrol_ckpt/model_humanml3d.pt": payload})
    eocd, members = _listing(blob)
    assert eocd["entries"] == 2
    files = [m for m in members if not m["is_dir"]]
    assert [m["name"] for m in files] == ["omnicontrol_ckpt/model_humanml3d.pt"]
    assert files[0]["uncompressed_bytes"] == len(payload)
    assert files[0]["crc32"] == f"{zipfile.ZipFile(io.BytesIO(blob)).getinfo(files[0]['name']).CRC:08x}"
    assert pf.terms_members(members) == []


@pytest.mark.parametrize("name", ["omnicontrol_ckpt/LICENSE", "omnicontrol_ckpt/README.md", "omnicontrol_ckpt/terms.txt",
                                  "omnicontrol_ckpt/MODEL_CARD", "omnicontrol_ckpt/NOTICE"])
def test_terms_member_is_detected(name):
    _, members = _listing(_zip({"omnicontrol_ckpt/model_humanml3d.pt": b"w", name: b"terms"}))
    assert pf.terms_members(members) == [name]


def test_eocd_refuses_garbage():
    with pytest.raises(ValueError, match="EOCD_NOT_FOUND"):
        pf.parse_eocd(b"\x00" * 64)


def test_readme_licence_scope():
    code = "## License\nThis code is distributed under an [MIT LICENSE](LICENSE).\nNote that our code depends on other " \
           "libraries, ... and uses datasets that each have their own respective licenses that must also be followed."
    sec, scoped, defers = pf.code_scoped_licence_section("# X\n" + code)
    assert scoped and defers and sec.startswith("## License")
    assert pf.code_scoped_licence_section("# X\n## License\nThe model weights are CC BY-NC 4.0.")[1] is False
    assert pf.code_scoped_licence_section("# no section")[1] is False


BLOCKING = {"archive_parsed": True, "archive_terms_members": [], "readme_scoped_to_code": True,
            "readme_defers_datasets": True, "readme_mentions_weight_licence": False, "project_page_fetched": True,
            "project_page_weight_terms": False, "issues_searched": True, "issues_with_licence_terms": [],
            "mdm_scoped_to_code": True, "mirror_exists": True, "mirror_publisher_is_not_rights_holder": True,
            "mirror_licence_equals_code_licence": True}


def test_rule_blocks_when_every_predicate_holds():
    rule, verdict = pf.evaluate_terms_rule(BLOCKING)
    assert all(rule.values()) and verdict == "DATA_BLOCKED_LICENSE"


@pytest.mark.parametrize("override", [
    {"archive_terms_members": ["omnicontrol_ckpt/LICENSE"]},     # terms shipped with the weights
    {"archive_parsed": False},                                   # archive identity not established
    {"readme_mentions_weight_licence": True},                    # README speaks to weight terms
    {"readme_scoped_to_code": False},
    {"project_page_weight_terms": True},
    {"issues_searched": False},                                  # un-evidenced -> no block, but also no permission
    {"issues_with_licence_terms": [{"number": 1, "title": "Licence of the weights?"}]},
    {"mdm_scoped_to_code": False},
    {"mirror_publisher_is_not_rights_holder": False},            # an author-published licensed copy
    {"mirror_licence_equals_code_licence": False},               # a distinct weight licence exists
])
def test_any_unmet_predicate_is_an_unresolved_stop_not_permission(override):
    rule, verdict = pf.evaluate_terms_rule({**BLOCKING, **override})
    assert not all(rule.values())
    assert verdict == "UNRESOLVED_TERMS_STOP"      # never "ALLOWED": permission needs a positive manual finding


def test_no_mirror_is_neutral():
    rule, verdict = pf.evaluate_terms_rule({**BLOCKING, "mirror_exists": False,
                                            "mirror_publisher_is_not_rights_holder": False,
                                            "mirror_licence_equals_code_licence": False})
    assert verdict == "DATA_BLOCKED_LICENSE"
