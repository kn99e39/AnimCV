"""Bind a replay to the exact stored artifacts it claims to replay.

docs/34 recorded the SHA-256 of a source prediction and of its evaluation file.
Hashing an evaluation proves only that some bytes were present; it does not
check that those bytes describe the candidate, split and frame count the replay
declares. docs/35 parses it.

Where the historical evaluation schema simply has no field for something -- it
records `candidate` and `frame_count` but never a split -- that item is
reported as ``unverifiable_in_source_schema`` rather than guessed at or
silently passed. Historical evaluation files are never rewritten.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

SOURCE_IDENTITY_SCHEMA = "animcv_frame_pose_source_identity_v1"

#: Recorded instead of a verdict when the stored schema cannot answer.
UNVERIFIABLE = "unverifiable_in_source_schema"


def digest(path: str | Path) -> dict[str, Any]:
    """Bind an artifact to its exact bytes, not to the label a caller passed."""
    path = Path(path)
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def verify_source_identity(*, prediction: str | Path, evaluation: str | Path | None,
                           bank, split: str, candidate: str, frames: int,
                           joints: int) -> dict[str, Any]:
    """Check a stored prediction and its evaluation really are what is claimed.

    Raises on a genuine disagreement -- a replay filed against the wrong
    candidate, split or frame count is worse than no replay. Returns the record
    to embed in the replay artifact.
    """
    import numpy as np

    prediction_path = Path(prediction)
    array = np.load(prediction_path, mmap_mode="r")
    if tuple(array.shape) != (frames, joints, 3):
        raise ValueError(
            f"stored prediction {prediction_path} has shape {tuple(array.shape)}, but the {split!r} "
            f"split of this bank has {frames} frames x {joints} joints")

    checks: dict[str, Any] = {
        "prediction_shape_matches_split": True,
        "bank_content_digest": bank.content_digest(),
        "observation_regime": bank.regime(),
    }
    record: dict[str, Any] = {
        "schema": SOURCE_IDENTITY_SCHEMA,
        "declared_candidate": candidate,
        "split": split,
        "frames": int(frames),
        "prediction": digest(prediction_path),
        "evaluation": None,
        "checks": checks,
    }
    if evaluation is None:
        checks["evaluation_present"] = False
        for name in ("evaluation_candidate_matches", "evaluation_frame_count_matches",
                     "evaluation_regime_matches", "evaluation_split_matches"):
            checks[name] = UNVERIFIABLE
        return record

    evaluation_path = Path(evaluation)
    record["evaluation"] = digest(evaluation_path)
    payload = json.loads(evaluation_path.read_text())
    checks["evaluation_present"] = True
    checks["evaluation_schema"] = payload.get("schema", UNVERIFIABLE)

    stored_candidate = payload.get("candidate")
    if stored_candidate is None:
        checks["evaluation_candidate_matches"] = UNVERIFIABLE
    elif stored_candidate != candidate:
        raise ValueError(
            f"{evaluation_path} evaluates candidate {stored_candidate!r} but the replay declares "
            f"{candidate!r}; refusing to replay one candidate's predictions under another's name")
    else:
        checks["evaluation_candidate_matches"] = True

    stored_frames = payload.get("frame_count", payload.get("aggregate", {}).get("frame_count"))
    if stored_frames is None:
        checks["evaluation_frame_count_matches"] = UNVERIFIABLE
    elif int(stored_frames) != int(frames):
        raise ValueError(
            f"{evaluation_path} reports {stored_frames} frames but the {split!r} split has {frames}")
    else:
        checks["evaluation_frame_count_matches"] = True

    stored_regime = payload.get("observation_regime")
    if stored_regime is None:
        checks["evaluation_regime_matches"] = UNVERIFIABLE
    else:
        # The historical schema stores a list of regimes present in the run.
        regimes = stored_regime if isinstance(stored_regime, list) else [stored_regime]
        if bank.regime() not in regimes:
            raise ValueError(
                f"{evaluation_path} records regimes {regimes} but this bank is {bank.regime()!r}")
        checks["evaluation_regime_matches"] = True

    # The historical evaluation schema has no split field at all. Say so.
    checks["evaluation_split_matches"] = (
        True if "split" in payload and payload["split"] == split
        else (UNVERIFIABLE if "split" not in payload else False))
    if checks["evaluation_split_matches"] is False:
        raise ValueError(
            f"{evaluation_path} records split {payload['split']!r} but the replay declares {split!r}")
    return record
