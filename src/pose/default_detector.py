"""Default person detector for AnimCV's production pose path."""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path

from pose.default_model import default_cache_dir

_CONFIG_RELATIVE_PATH = "rtmdet/rtmdet_tiny_8xb32-300e_coco.py"
_CHECKPOINT_FILENAME = "rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth"
_CHECKPOINT_URL = (
    "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/"
    "rtmdet_tiny_8xb32-300e_coco/" + _CHECKPOINT_FILENAME
)


def get_default_detector_config_path() -> str:
    try:
        import mmdet
    except ImportError as exc:
        raise ImportError("MMDetection is not installed. Install the optional 'pose' extra: pip install -e '.[pose]'") from exc
    path = Path(mmdet.__file__).resolve().parent / ".mim" / "configs" / _CONFIG_RELATIVE_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Expected the default RTMDet config at {path}; pass --detector-config explicitly.")
    return str(path)


def get_default_detector_checkpoint_path(download: bool = True) -> str:
    path = default_cache_dir() / _CHECKPOINT_FILENAME
    if path.is_file():
        return str(path)
    if not download:
        raise FileNotFoundError(f"No cached detector checkpoint at {path} and download=False.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    try:
        urllib.request.urlretrieve(_CHECKPOINT_URL, temporary)
    except (OSError, urllib.error.URLError) as exc:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Could not download the default RTMDet-tiny checkpoint from {_CHECKPOINT_URL}: {exc}") from exc
    temporary.rename(path)
    return str(path)
