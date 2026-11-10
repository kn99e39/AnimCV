"""Synthetic unit tests for the W134 ARDY access preflight helpers (no network, no model, no S1 data)."""
import importlib.util
from pathlib import Path

import pytest

_P = Path(__file__).resolve().parents[1] / "scripts" / "w134_ardy_access_preflight.py"
_spec = importlib.util.spec_from_file_location("w134_pre", _P)
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)

LOAD_MODEL_SNIPPET = '''
DEFAULT_TEXT_ENCODER = "llm2vec"
TEXT_ENCODER_PRESETS = {
    "llm2vec": {
        "target": "ardy.model.LLM2VecEncoder",
        "kwargs": {
            "base_model_name_or_path": "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
            "peft_model_name_or_path": "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
        },
    }
}
'''


@pytest.mark.parametrize("st,cls", [(200, "GRANTED"), (302, "GRANTED"), (307, "GRANTED"), (401, "NOT_GRANTED"),
                                    (403, "NOT_GRANTED"), (None, "NOT_CHECKED"), (404, "UNKNOWN_STATUS"),
                                    (500, "UNKNOWN_STATUS")])
def test_classify_gated_head(st, cls):
    assert m.classify_gated_head(st) == cls


def test_text_encoder_preset_parse():
    p = m.text_encoder_preset(LOAD_MODEL_SNIPPET)
    assert p == {"default": "llm2vec", "base": "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
                 "peft": "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised"}


def test_text_encoder_preset_missing():
    assert m.text_encoder_preset("nothing here") == {"default": None, "base": None, "peft": None}


GOOD = {"requires_gated_base": True, "credential_present": True, "whoami_ok": True, "gated_head_class": "GRANTED"}


def test_access_ok_only_when_all_hold():
    rule, verdict, missing = m.evaluate_access_rule(GOOD)
    assert verdict == "ACCESS_OK" and all(rule.values()) and missing == []


@pytest.mark.parametrize("key,val,reason", [
    ("credential_present", False, "no existing local Hugging Face credential"),
    ("whoami_ok", False, "does not authenticate"),
    ("gated_head_class", "NOT_GRANTED", "no granted access"),
    ("gated_head_class", "UNKNOWN_STATUS", "no granted access"),
    ("gated_head_class", "NOT_CHECKED", "no granted access"),
])
def test_access_blocked_fails_closed(key, val, reason):
    f = dict(GOOD, **{key: val})
    _, verdict, missing = m.evaluate_access_rule(f)
    assert verdict == "ACCESS_BLOCKED"
    assert len(missing) == 1 and reason in missing[0]


def test_missing_credential_reported_before_later_predicates():
    f = dict(GOOD, credential_present=False, whoami_ok=False, gated_head_class="NOT_CHECKED")
    _, verdict, missing = m.evaluate_access_rule(f)
    assert verdict == "ACCESS_BLOCKED" and missing == ["no existing local Hugging Face credential"]


def test_not_required_is_vacuous():
    _, verdict, _ = m.evaluate_access_rule({"requires_gated_base": False})
    assert verdict == "ACCESS_OK"


def test_no_redirect_handler_refuses_redirect():
    assert m._NoRedirect().redirect_request(None, None, 302, "", {}, "x") is None
