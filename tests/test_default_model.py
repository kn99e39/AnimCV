"""Tests for pose/default_model.py -- the RTMPose-tiny default so
estimate-pose has a working starting point without every user having to
find and download a config/checkpoint pair first."""

from __future__ import annotations

import urllib.error

import pytest

from pose import default_model


def test_default_cache_dir_is_under_home_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(default_model.Path, "home", lambda: tmp_path)

    assert default_model.default_cache_dir() == tmp_path / ".cache" / "animcv" / "models"


def test_get_default_pose_config_path_wraps_missing_mmpose(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "mmpose":
            raise ImportError("no module named mmpose")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="pip install"):
        default_model.get_default_pose_config_path()


def test_get_default_pose_config_path_raises_if_config_missing(monkeypatch, tmp_path):
    import sys
    import types

    fake_pkg_dir = tmp_path / "mmpose"
    fake_pkg_dir.mkdir()
    fake_mmpose = types.ModuleType("mmpose")
    fake_mmpose.__file__ = str(fake_pkg_dir / "__init__.py")
    monkeypatch.setitem(sys.modules, "mmpose", fake_mmpose)

    with pytest.raises(FileNotFoundError):
        default_model.get_default_pose_config_path()


def test_get_default_pose_checkpoint_path_reuses_existing_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(default_model, "default_cache_dir", lambda: tmp_path)
    cached = tmp_path / default_model._CHECKPOINT_FILENAME
    cached.write_bytes(b"already cached")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("should not attempt a download when already cached")

    monkeypatch.setattr(default_model.urllib.request, "urlretrieve", fail_if_called)

    result = default_model.get_default_pose_checkpoint_path()

    assert result == str(cached)


def test_get_default_pose_checkpoint_path_no_download_raises_when_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(default_model, "default_cache_dir", lambda: tmp_path)

    with pytest.raises(FileNotFoundError):
        default_model.get_default_pose_checkpoint_path(download=False)


def test_get_default_pose_checkpoint_path_downloads_and_caches(monkeypatch, tmp_path):
    monkeypatch.setattr(default_model, "default_cache_dir", lambda: tmp_path)

    def fake_urlretrieve(url, filename):
        with open(filename, "wb") as f:
            f.write(b"pretend checkpoint bytes")

    monkeypatch.setattr(default_model.urllib.request, "urlretrieve", fake_urlretrieve)

    result = default_model.get_default_pose_checkpoint_path()

    expected = tmp_path / default_model._CHECKPOINT_FILENAME
    assert result == str(expected)
    assert expected.read_bytes() == b"pretend checkpoint bytes"
    assert not expected.with_suffix(expected.suffix + ".part").exists()


def test_get_default_pose_checkpoint_path_download_failure_raises_runtime_error(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(default_model, "default_cache_dir", lambda: tmp_path)

    def fake_urlretrieve(url, filename):
        raise urllib.error.URLError("no network")

    monkeypatch.setattr(default_model.urllib.request, "urlretrieve", fake_urlretrieve)

    with pytest.raises(RuntimeError, match="Could not download"):
        default_model.get_default_pose_checkpoint_path()

    expected = tmp_path / default_model._CHECKPOINT_FILENAME
    assert not expected.exists()
    assert not expected.with_suffix(expected.suffix + ".part").exists()
