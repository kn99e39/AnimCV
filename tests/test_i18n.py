"""Tests for the GUI translation layer (ui/i18n.py) and the language
persistence helpers in ui/gui_app.py -- all display-free, so they run
without a Tk display."""

from __future__ import annotations

import pytest

from ui import gui_app
from ui.i18n import _STRINGS, LANGUAGES, Translator


def test_translator_resolves_each_language():
    tr = Translator("en")
    assert tr("tab.frames") == "1. Frames"
    tr.set_language("ko")
    assert tr("tab.frames") == "1. 프레임"


def test_translator_falls_back_to_english_for_missing_language_entry(monkeypatch):
    monkeypatch.setitem(_STRINGS, "x.only_en", {"en": "only english"})
    tr = Translator("ko")
    assert tr("x.only_en") == "only english"


def test_translator_falls_back_to_key_when_unknown():
    tr = Translator("en")
    assert tr("no.such.key") == "no.such.key"


def test_translator_formats_placeholders():
    tr = Translator("en")
    assert tr("frames.result", count=12, out="/tmp/x") == "✓ Extracted 12 frames -> /tmp/x"
    tr.set_language("ko")
    assert "12" in tr("frames.result", count=12, out="/tmp/x")


def test_translator_ignores_invalid_language():
    tr = Translator("zz")
    assert tr.language == "en"
    tr.set_language("qq")
    assert tr.language == "en"


def test_every_string_has_all_languages():
    missing = {
        key: [lang for lang in LANGUAGES if lang not in entry]
        for key, entry in _STRINGS.items()
        if any(lang not in entry for lang in LANGUAGES)
    }
    assert missing == {}, f"strings missing a language: {missing}"


def test_language_persistence_round_trip(monkeypatch, tmp_path):
    settings = tmp_path / "settings.json"
    monkeypatch.setattr(gui_app, "_settings_path", lambda: settings)

    assert gui_app.load_saved_language() == "en"  # nothing saved yet
    gui_app.save_language("ko")
    assert gui_app.load_saved_language() == "ko"
    # an unknown code round-trips back to the safe default
    gui_app.save_language("zz")
    assert gui_app.load_saved_language() == "en"


def test_load_saved_language_tolerates_corrupt_file(monkeypatch, tmp_path):
    settings = tmp_path / "settings.json"
    settings.write_text("{ not valid json")
    monkeypatch.setattr(gui_app, "_settings_path", lambda: settings)

    assert gui_app.load_saved_language() == "en"


def test_save_language_preserves_other_settings(monkeypatch, tmp_path):
    import json

    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"other": 1}))
    monkeypatch.setattr(gui_app, "_settings_path", lambda: settings)

    gui_app.save_language("ko")
    data = json.loads(settings.read_text())
    assert data == {"other": 1, "language": "ko"}
