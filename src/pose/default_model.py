"""Default MMPose model (RTMPose-tiny) so estimate-pose has a working
starting point instead of requiring every user to find and download a
matching config/checkpoint pair themselves before they can try the tool
at all. Both pieces come from mmpose's own official releases (the
config ships inside the mmpose package itself; the checkpoint is
OpenMMLab's own model zoo URL) -- nothing here is trained or hosted by
this project. This is the same config/checkpoint pair used to verify
estimate-pose end-to-end against a real model (see README_EXEC.md).

Lazy-imports mmpose inside the function bodies, same pattern as
pose/mmpose_adapter.py, so importing this module doesn't require mmpose
to be installed.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path

_CONFIG_RELATIVE_PATH = "body_2d_keypoint/rtmpose/coco/rtmpose-t_8xb256-420e_coco-256x192.py"
_CHECKPOINT_URL = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
    "rtmpose-tiny_simcc-coco_pt-aic-coco_420e-256x192-e613ba3f_20230127.pth"
)
_CHECKPOINT_FILENAME = "rtmpose-tiny_simcc-coco_pt-aic-coco_420e-256x192-e613ba3f_20230127.pth"


def default_cache_dir() -> Path:
    return Path.home() / ".cache" / "animcv" / "models"


def get_default_pose_config_path() -> str:
    try:
        import mmpose
    except ImportError as exc:
        raise ImportError(
            "MMPose is not installed. Install the optional 'pose' extra: "
            "pip install -e '.[pose]'"
        ) from exc

    config_path = Path(mmpose.__file__).resolve().parent / ".mim" / "configs" / _CONFIG_RELATIVE_PATH
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Expected the default RTMPose-tiny config at {config_path}, but it's "
            "missing -- some mmpose distributions strip .mim/configs. Pass "
            "--pose-config explicitly instead."
        )
    return str(config_path)


def get_default_pose_checkpoint_path(download: bool = True) -> str:
    """Path to a locally cached RTMPose-tiny checkpoint (~13MB),
    downloading it from OpenMMLab's model zoo on first use if not
    already cached under default_cache_dir()."""
    checkpoint_path = default_cache_dir() / _CHECKPOINT_FILENAME
    if checkpoint_path.is_file():
        return str(checkpoint_path)
    if not download:
        raise FileNotFoundError(f"No cached checkpoint at {checkpoint_path} and download=False.")

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    # Download to a temp name and rename into place atomically, so a
    # connection drop mid-download can never leave a corrupt file that
    # later runs mistake for a complete, cached checkpoint.
    tmp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".part")
    try:
        urllib.request.urlretrieve(_CHECKPOINT_URL, tmp_path)
    except (OSError, urllib.error.URLError) as exc:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download the default RTMPose-tiny checkpoint from "
            f"{_CHECKPOINT_URL}: {exc}. Check your network connection, or pass "
            "--pose-checkpoint explicitly."
        ) from exc
    tmp_path.rename(checkpoint_path)
    return str(checkpoint_path)
