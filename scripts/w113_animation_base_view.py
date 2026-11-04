"""W113 scratch-only Animation Base view: join, contract validation, synthetic directives and mock edit.

NON-PRODUCTION. Nothing here is a canonical schema, a migration, a Generator or an Editor. It exists only to test
whether existing AnimCV records can express a minimum reference-guided, reliability-aware, user-constrained
Animation Base without manufacturing information:

- the view joins authentic input-side records (2D observation, validity) with a frozen 3D output and labels every
  signal with a status (observed-2D / derived-2D / inferred-3D / derived / unknown / unavailable /
  synthetic-user-directive / synthetic-edit), coordinate frame, unit and source hash;
- it never fills an unknown, never creates a 3D confidence and never carries evaluation ground truth;
- synthetic user directives (preserve / permit-edit, default prohibit) are resolved into per-cell masks, a lock
  conflict is rejected rather than resolved by priority, and a deterministic mock perturbation (not animation) may
  touch only authorized cells;
- `validate_view` hard-rejects identity/time/coordinate/provenance/confidence/unknown-coercion/GT-leak violations,
  each with a distinct code so negative controls can be checked for the exact rejection.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

VIEW_SCHEMA = "w113_animation_base_view_v0_scratch_nonproduction"
DIRECTIVE_SCHEMA = "w113_synthetic_user_directive_v0_scratch"
EDIT_SCHEMA = "w113_synthetic_mock_edit_v0_scratch"
DIGEST_DOMAIN = b"w113_animation_base_view_v0"

JOINTS = ("pelvis", "left_hip", "left_knee", "left_ankle", "right_hip", "right_knee", "right_ankle", "spine", "thorax",
          "neck", "head", "left_shoulder", "left_elbow", "left_wrist", "right_shoulder", "right_elbow", "right_wrist")
J = len(JOINTS)
JI = {n: i for i, n in enumerate(JOINTS)}
# MMPose adapter canonical_v1 (16 rows) -> H36M bank joint, as consumed by W100-W105 (`SOURCE_OF`).
CANON = ("head", "neck", "pelvis", "spine", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist",
         "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle")
CANON_OF = {n: ("neck" if n == "thorax" else n) for n in JOINTS}
# How src/pose/mmpose_adapter.py `_extract_canonical_landmarks` produces each 2D joint (static code fact).
OBS2D_KIND = {n: "observed_2d" for n in JOINTS}
OBS2D_KIND.update({"head": "observed_2d_renamed_nose", "neck": "derived_2d_midpoint_shoulders",
                   "thorax": "derived_2d_midpoint_shoulders_duplicate_of_neck", "pelvis": "derived_2d_midpoint_hips",
                   "spine": "derived_2d_midpoint_neck_pelvis"})
OBS2D_STATUS = {0: "unknown_absent", 1: "observed_valid", 2: "observed_below_validity_threshold"}
EST3D_STATUS = {0: "unknown_nonfinite", 1: "inferred_3d_input_supported", 2: "inferred_3d_input_unsupported"}
EST3D_PROVENANCE = {0: "frozen_estimator_output", 1: "synthetic_mock_edit"}
UNAVAILABLE = {
    "root_translation": "not_observable: root-relative FramePose pins pelvis; no camera-motion estimate (docs/51)",
    "ground_height": "not_observable: no floor reference in FramePose coordinates (docs/51)",
    "physical_contact": "not_produced: contact-like evidence only, never proven foot-ground contact (docs/51-53)",
    "est3d_calibrated_confidence": "not_produced: no calibrated 3D reliability; 1-confidence is a 2D proxy only",
    "est3d_actual_error": "not_measurable GT-free; evaluation-only quantity never enters the view",
    "world_or_camera_motion": "not_produced: no camera/world motion estimate",
}
FORBIDDEN_TOKENS = ("target_3d", "target_valid", "ground_truth", "gt_", "_gt", "strata", "smpl", "evaluation_label")
# Frozen (declaration c3f7485c): detector confidence must lie in [0, 1]. Authentic RTMPose / 3DPW scores exceed 1, so
# the frozen real/benchmark views fail this gate (W113 CASE D). Only scripts/w113_animation_base_posthoc.py, a labelled
# post-hoc diagnostic that never yields a CASE, overrides it.
CONFIDENCE_UPPER = 1.0
EDITABLE_SIGNALS = ("est3d_xyz",)
IMMUTABLE_SIGNALS = ("obs2d_xy_px", "obs2d_conf", "obs2d_present", "obs2d_visible", "input_valid", "in_frame",
                     "frame_index", "timestamp")
ARRAY_SPEC = {  # name -> (dtype, trailing shape)
    "frame_index": ("int64", ()), "timestamp": ("float64", ()),
    "obs2d_xy_px": ("float64", (J, 2)), "obs2d_conf": ("float64", (J,)), "obs2d_present": ("bool", (J,)),
    "obs2d_visible": ("bool", (J,)), "input_valid": ("bool", (J,)), "in_frame": ("bool", (J,)),
    "est3d_xyz": ("float32", (J, 3)), "est3d_finite": ("bool", (J,)), "est3d_provenance": ("int8", (J,)),
    "root_yaw_known": ("bool", ()), "root_yaw_rad": ("float64", ()),
}


class ContractError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


def sha_file(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as s:
        for chunk in iter(lambda: s.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


# --------------------------------------------------------------------------------------------- construction ---

def build_view(*, sequence_id: str, actor_id: str, actor_identity_status: str, regime: str, backend: str,
               image_size, fps: float, fps_source: str, frame_index, timestamp, obs2d_xy_px, obs2d_conf,
               obs2d_present, obs2d_visible, input_valid, est3d_xyz, root_yaw, sources: dict[str, dict],
               producers: dict[str, Any], obs2d_kind: dict[str, str] | None = None,
               per_frame_image_digest=None, notes: dict[str, str] | None = None) -> dict:
    """Join already-authenticated input-side arrays with a frozen 3D output. Fills nothing."""
    T = len(frame_index)
    xy = np.asarray(obs2d_xy_px, np.float64).reshape(T, J, 2)
    present = np.asarray(obs2d_present, bool).reshape(T, J)
    est = np.asarray(est3d_xyz, np.float32).reshape(T, J, 3)
    w, h = image_size
    normalized = np.stack([xy[..., 0] / w, xy[..., 1] / h], -1)
    valid = np.asarray(input_valid, bool).reshape(T, J)
    # Same expression as framepose.nonlinear_reconstruction.in_frame_mask, on the retained raw coordinate.
    in_frame = (valid & np.isfinite(normalized).all(-1) & (normalized[..., 0] >= 0) & (normalized[..., 0] <= 1)
                & (normalized[..., 1] >= 0) & (normalized[..., 1] <= 1))
    yaw = np.asarray([np.nan if y is None else float(y) for y in root_yaw], np.float64)
    arrays = {
        "frame_index": np.asarray(frame_index, np.int64), "timestamp": np.asarray(timestamp, np.float64),
        "obs2d_xy_px": xy, "obs2d_conf": np.asarray(obs2d_conf, np.float64).reshape(T, J), "obs2d_present": present,
        "obs2d_visible": np.asarray(obs2d_visible, bool).reshape(T, J), "input_valid": valid, "in_frame": in_frame,
        "est3d_xyz": est, "est3d_finite": np.isfinite(est).all(-1), "est3d_provenance": np.zeros((T, J), np.int8),
        "root_yaw_known": np.isfinite(yaw), "root_yaw_rad": yaw,
    }
    meta = {
        "schema": VIEW_SCHEMA, "non_production": True, "regime": regime, "backend": backend,
        "sequence_id": sequence_id, "actor_id": actor_id, "actor_identity_status": actor_identity_status,
        "joint_names": list(JOINTS), "image_size": [int(w), int(h)], "fps": float(fps), "fps_source": fps_source,
        "coordinate_frames": {"obs2d_xy_px": "image_I", "est3d_xyz": "camera_root_relative"},
        "units": {"obs2d_xy_px": "pixels", "est3d_xyz": "metres", "timestamp": "seconds", "root_yaw_rad": "radians"},
        "signals": {
            "obs2d_xy_px": {"status": "observed_2d", "per_joint_kind": dict(obs2d_kind or OBS2D_KIND),
                            "absent": "NaN + obs2d_present=False (UNKNOWN), never zero"},
            "obs2d_conf": {"status": "observed_2d", "meaning": "detector keypoint score; NOT 3D reliability"},
            "obs2d_visible": {"status": "derived", "rule": "adapter visibility flag (score >= recorded threshold)"},
            "input_valid": {"status": "derived", "rule": "validity mask handed to the frozen estimator"},
            "in_frame": {"status": "derived", "rule": "in_frame_mask expression on the raw coordinate"},
            "est3d_xyz": {"status": "inferred_3d", "producer": producers.get("est3d"),
                          "note": "finite does not imply correct; no calibrated 3D confidence exists"},
            "root_yaw_rad": {"status": "derived", "producer": producers.get("root_yaw"),
                             "reference_frame": "camera", "unknown": "NaN + root_yaw_known=False"},
            "foot_contact_like": {"status": "unknown", "reason": producers.get("contact_status")},
        },
        "unavailable": {k: {"status": "unavailable", "reason": v} for k, v in UNAVAILABLE.items()},
        "sources": sources, "producers": producers, "per_frame_image_digest": per_frame_image_digest,
        "notes": dict(notes or {}), "edits": [], "parent_content_digest": None,
    }
    view = {"meta": meta, "arrays": arrays}
    view["meta"]["content_digest"] = content_digest(view)
    return view


def content_digest(view: dict) -> str:
    h = hashlib.sha256(DIGEST_DOMAIN)
    meta = {k: v for k, v in view["meta"].items() if k != "content_digest"}
    h.update(canonical(meta).encode())
    for key in sorted(view["arrays"]):
        a = np.ascontiguousarray(view["arrays"][key])
        h.update(f"|{key}|{a.dtype.str}|{a.shape}|".encode())
        h.update(a.tobytes())
    return h.hexdigest()


def cell_status(view: dict) -> dict[str, np.ndarray]:
    a = view["arrays"]
    obs = np.where(~a["obs2d_present"], 0, np.where(a["input_valid"], 1, 2)).astype(np.int8)
    est = np.where(~a["est3d_finite"], 0, np.where(a["input_valid"], 1, 2)).astype(np.int8)
    return {"obs2d_status": obs, "est3d_status": est}


def save_view(view: dict, stem: Path) -> tuple[Path, Path]:
    stem = Path(stem)
    np.savez(stem.with_suffix(".npz"), **view["arrays"])
    stem.with_suffix(".json").write_text(json.dumps(view["meta"], indent=1, sort_keys=True, allow_nan=False))
    return stem.with_suffix(".json"), stem.with_suffix(".npz")


def load_view(stem: Path) -> dict:
    stem = Path(stem)
    meta = json.loads(stem.with_suffix(".json").read_text())
    with np.load(stem.with_suffix(".npz"), allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    return {"meta": meta, "arrays": arrays}


# ----------------------------------------------------------------------------------------------- validation ---

def _walk_keys(value, prefix=""):
    if isinstance(value, dict):
        for k, v in value.items():
            yield f"{prefix}{k}"
            yield from _walk_keys(v, f"{prefix}{k}.")
    elif isinstance(value, list):
        for v in value:
            yield from _walk_keys(v, prefix)


def validate_view(view: dict, *, expected: dict | None = None, verify_files: bool = True) -> dict:
    """Hard-reject contract violations. `expected` binds externally authenticated identity (source hashes, fps,
    sequence/actor). Returns a summary on success."""
    meta, arrays = view.get("meta"), view.get("arrays")
    if not isinstance(meta, dict) or not isinstance(arrays, dict) or meta.get("schema") != VIEW_SCHEMA:
        raise ContractError("SCHEMA", "not a W113 scratch view")
    keys = [k.lower() for k in list(_walk_keys(meta)) + list(arrays)]
    leaked = sorted({k for k in keys for t in FORBIDDEN_TOKENS if t in k.rsplit(".", 1)[-1]})
    if leaked:
        raise ContractError("GT_LEAKAGE", f"forbidden evaluation/GT-derived keys {leaked[:5]}")
    names = meta.get("joint_names")
    if not isinstance(names, list) or len(set(names)) != len(names):
        raise ContractError("JOINT_DUPLICATE", "duplicate joint name")
    if set(names) != set(JOINTS):
        raise ContractError("JOINT_MISSING" if set(names) < set(JOINTS) else "JOINT_UNKNOWN",
                            f"joint set differs: {sorted(set(JOINTS) ^ set(names))}")
    if tuple(names) != JOINTS:
        raise ContractError("JOINT_ORDER", "joint order differs from the declared H36M order")
    for bad in ("root_translation", "ground_height", "est3d_actual_error", "physical_contact",
                "est3d_calibrated_confidence"):
        if bad in arrays:
            raise ContractError("UNAVAILABLE_COERCED", f"array for unavailable quantity {bad}")
        if meta.get("unavailable", {}).get(bad, {}).get("status") != "unavailable":
            raise ContractError("UNAVAILABLE_COERCED", f"{bad} not recorded as unavailable")
        if any(k not in ("status", "reason") for k in meta["unavailable"][bad]):
            raise ContractError("UNAVAILABLE_COERCED", f"{bad} carries a value")
    for k in arrays:
        if k in ("est3d_confidence", "depth_uncertainty", "est3d_reliability"):
            raise ContractError("CONFIDENCE_MANUFACTURED_3D", f"{k} is not produced by any estimator")
    for k in ARRAY_SPEC:
        if k not in arrays:
            raise ContractError("CONFIDENCE_MISSING" if k == "obs2d_conf" else "FIELD_MISSING", f"missing {k}")
    T = len(arrays["frame_index"])
    for k, (dtype, tail) in ARRAY_SPEC.items():
        a = arrays[k]
        if str(a.dtype) != dtype or tuple(a.shape) != (T, *tail):
            raise ContractError("SHAPE", f"{k} {a.dtype}{a.shape} != {dtype}{(T, *tail)}")
    fi, ts = arrays["frame_index"], arrays["timestamp"]
    if T == 0:
        raise ContractError("SHAPE", "empty view")
    if len(np.unique(fi)) != T:
        raise ContractError("FRAME_DUPLICATE", "duplicate frame_index")
    if (np.diff(fi) <= 0).any():
        raise ContractError("FRAME_ORDER", "frame_index not strictly increasing")
    if not np.isfinite(ts).all() or len(np.unique(ts)) != T:
        raise ContractError("TIME_DUPLICATE", "duplicate or non-finite timestamp")
    if (np.diff(ts) <= 0).any():
        raise ContractError("TIME_ORDER", "timestamps not strictly increasing")
    fps = meta.get("fps")
    if not isinstance(fps, (int, float)) or not fps > 0 or not math.isfinite(fps):
        raise ContractError("FPS_MISMATCH", "invalid fps")
    if np.abs(ts - fi / fps).max() > 1e-9:
        raise ContractError("FPS_MISMATCH", f"timestamps disagree with frame_index/fps={fps}")
    if meta.get("coordinate_frames") != {"obs2d_xy_px": "image_I", "est3d_xyz": "camera_root_relative"} \
            or meta.get("units", {}).get("est3d_xyz") != "metres" or meta["units"].get("obs2d_xy_px") != "pixels":
        raise ContractError("COORDINATE_MISMATCH", f"coordinate/unit declaration {meta.get('coordinate_frames')}")
    present, conf, xy = arrays["obs2d_present"], arrays["obs2d_conf"], arrays["obs2d_xy_px"]
    if np.isnan(conf[present]).any():
        raise ContractError("CONFIDENCE_MISSING", "present observation without confidence")
    if (conf[present] < 0).any() or (conf[present] > CONFIDENCE_UPPER).any():
        raise ContractError("CONFIDENCE_AMBIGUOUS", f"confidence outside [0,{CONFIDENCE_UPPER}]")
    if not np.isfinite(xy[present]).all():
        raise ContractError("FIELD_MISSING", "present observation without a coordinate")
    if np.isfinite(xy[~present]).any() or np.isfinite(conf[~present]).any():
        raise ContractError("UNKNOWN_COERCED_TO_ZERO" if (xy[~present] == 0).any() or (conf[~present] == 0).any()
                            else "UNKNOWN_COERCED", "absent observation carries a value")
    if (arrays["input_valid"] & ~present).any():
        raise ContractError("CONFIDENCE_AMBIGUOUS", "input_valid on an absent observation")
    if not np.array_equal(arrays["est3d_finite"], np.isfinite(arrays["est3d_xyz"]).all(-1)):
        raise ContractError("UNKNOWN_COERCED", "est3d_finite disagrees with est3d_xyz")
    yk, yaw = arrays["root_yaw_known"], arrays["root_yaw_rad"]
    if not np.array_equal(yk, np.isfinite(yaw)):
        raise ContractError("UNKNOWN_COERCED_TO_ZERO" if (yaw[~yk] == 0).any() else "UNKNOWN_COERCED",
                            "root yaw known flag disagrees with value")
    if set(np.unique(arrays["est3d_provenance"]).tolist()) - set(EST3D_PROVENANCE):
        raise ContractError("PROVENANCE_MISSING", "unknown per-cell 3D provenance code")
    sources = meta.get("sources")
    if not isinstance(sources, dict) or not sources:
        raise ContractError("PROVENANCE_MISSING", "no source provenance")
    for name, src in sources.items():
        digest = src.get("sha256") if isinstance(src, dict) else None
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ContractError("PROVENANCE_MISSING", f"source {name} lacks a sha256")
        if verify_files and src.get("path") and sha_file(src["path"]) != digest:
            raise ContractError("PROVENANCE_STALE", f"source {name} bytes differ from recorded sha256")
    if expected:
        for name, digest in expected.get("sources", {}).items():
            if sources.get(name, {}).get("sha256") != digest:
                raise ContractError("PROVENANCE_STALE", f"source {name} differs from the frozen manifest")
        if "fps" in expected and float(expected["fps"]) != float(fps):
            raise ContractError("FPS_MISMATCH", f"fps {fps} != frozen source fps {expected['fps']}")
        for k in ("sequence_id", "actor_id", "regime", "backend"):
            if k in expected and meta.get(k) != expected[k]:
                raise ContractError("IDENTITY_MISMATCH", f"{k} {meta.get(k)!r} != {expected[k]!r}")
        if "frame_count" in expected and T != expected["frame_count"]:
            raise ContractError("FRAME_ORDER", "frame count differs from the frozen manifest")
    if meta.get("content_digest") != content_digest(view):
        raise ContractError("DIGEST_STALE", "content digest does not match the view")
    return {"frames": T, "content_digest": meta["content_digest"]}


def reject_mixed_unit_motion_graph(graph: dict) -> dict:
    """MotionGraph.position_3d may be (image_x, image_y, relative_depth): never reinterpret it as metric pose."""
    mixed = sum(1 for f in graph["frames"] for p in f["points"].values() if p.get("position_3d") is not None)
    if mixed:
        raise ContractError("MIXED_UNIT_3D", f"{mixed} MotionPoint.position_3d values have mixed image/depth units "
                                             "(COORDINATE_CONVENTIONS.md); refusing metric reinterpretation")
    return {"position_3d_present": 0, "status": "unavailable_in_motion_graph"}


# ----------------------------------------------------------------------------------------------- directives ---

def resolve_directives(view: dict, dset: dict) -> dict[str, np.ndarray]:
    """Synthetic user directives -> (preserve, permit) cell masks for est3d_xyz. Default: prohibit. Conflict: reject."""
    meta, a = view["meta"], view["arrays"]
    if dset.get("schema") != DIRECTIVE_SCHEMA or dset.get("synthetic") is not True:
        raise ContractError("SCHEMA", "directive set must be the labelled synthetic scratch schema")
    if dset.get("view_content_digest") != meta["content_digest"]:
        raise ContractError("DIRECTIVE_STALE_VIEW", "directives bound to a different view digest")
    for k in ("sequence_id", "actor_id"):
        if dset.get(k) != meta[k]:
            raise ContractError("IDENTITY_MISMATCH", f"directive {k} {dset.get(k)!r} != view {meta[k]!r}")
    if dset.get("default") != "prohibit":
        raise ContractError("SCHEMA", "default must be prohibit")
    fi = a["frame_index"]
    T = len(fi)
    masks = {"preserve": np.zeros((T, J), bool), "permit": np.zeros((T, J), bool)}
    for d in dset["directives"]:
        if d.get("signal") in IMMUTABLE_SIGNALS:
            raise ContractError("OBSERVATION_IMMUTABLE", f"{d['id']}: {d.get('signal')} is an observation/identity")
        if d.get("signal") not in EDITABLE_SIGNALS:
            raise ContractError("SCHEMA", f"{d['id']}: unknown signal {d.get('signal')!r}")
        if d.get("kind") not in masks:
            raise ContractError("SCHEMA", f"{d['id']}: kind {d.get('kind')!r}")
        unknown = [n for n in d["joints"] if n not in JI]
        if unknown:
            raise ContractError("JOINT_UNKNOWN", f"{d['id']}: {unknown}")
        rows = np.zeros(T, bool)
        for s, e in d["spans"]:
            if s > e or s < fi[0] or e > fi[-1]:
                raise ContractError("FRAME_OUT_OF_RANGE", f"{d['id']}: span {s}-{e}")
            rows |= (fi >= s) & (fi <= e)
        masks[d["kind"]][np.ix_(rows, [JI[n] for n in d["joints"]])] = True
    overlap = masks["preserve"] & masks["permit"]
    if overlap.any():
        r, c = np.argwhere(overlap)[0]
        raise ContractError("LOCK_CONFLICT", f"{int(overlap.sum())} cells both preserved and permitted "
                                             f"(first frame {int(fi[r])} {JOINTS[c]}); no hidden priority")
    return masks


def apply_mock_edit(view: dict, dset: dict, request: dict) -> tuple[dict, dict]:
    """Deterministic synthetic perturbation of authorized est3d cells only. NOT animation, NOT a Generator."""
    if request.get("schema") != EDIT_SCHEMA or request.get("label") != "synthetic_mock_perturbation_not_animation":
        raise ContractError("SCHEMA", "edit request must be the labelled synthetic mock")
    if request.get("view_content_digest") != view["meta"]["content_digest"]:
        raise ContractError("DIRECTIVE_STALE_VIEW", "edit bound to a different view digest")
    if request.get("signal") in IMMUTABLE_SIGNALS:
        raise ContractError("OBSERVATION_IMMUTABLE", f"{request.get('signal')} is not editable")
    if request.get("signal") not in EDITABLE_SIGNALS:
        raise ContractError("SCHEMA", f"unknown signal {request.get('signal')!r}")
    masks = resolve_directives(view, dset)
    fi = view["arrays"]["frame_index"]
    row_of = {int(f): i for i, f in enumerate(fi)}
    cells = []
    for f, n in request["cells"]:
        if int(f) not in row_of:
            raise ContractError("FRAME_OUT_OF_RANGE", f"edit frame {f}")
        if n not in JI:
            raise ContractError("JOINT_UNKNOWN", f"edit joint {n}")
        cells.append((row_of[int(f)], JI[n]))
    if not cells or len(set(cells)) != len(cells):
        raise ContractError("SCHEMA", "edit cells empty or duplicated")
    locked = [c for c in cells if masks["preserve"][c]]
    if locked:
        raise ContractError("LOCK_CONFLICT_EDIT", f"{len(locked)} requested cells are preserved (locked)")
    unauthorized = [c for c in cells if not masks["permit"][c]]
    if unauthorized:
        raise ContractError("EDIT_NOT_AUTHORIZED", f"{len(unauthorized)} requested cells are not permitted")
    unknown = [c for c in cells if not view["arrays"]["est3d_finite"][c]]
    if unknown:
        raise ContractError("EDIT_UNKNOWN_TARGET", f"{len(unknown)} requested cells are UNKNOWN; mock may not fill")
    out = {"meta": copy.deepcopy(view["meta"]), "arrays": {k: v.copy() for k, v in view["arrays"].items()}}
    delta = np.asarray(request["delta_m"], np.float32)
    rows, cols = np.asarray(cells).T
    out["arrays"]["est3d_xyz"][rows, cols] = out["arrays"]["est3d_xyz"][rows, cols] + delta
    out["arrays"]["est3d_provenance"][rows, cols] = 1
    out["arrays"]["est3d_finite"] = np.isfinite(out["arrays"]["est3d_xyz"]).all(-1)
    log = {"edit_id": request["edit_id"], "label": request["label"], "cells": len(cells), "delta_m": request["delta_m"],
           "directive_set_id": dset["id"], "parent_content_digest": view["meta"]["content_digest"]}
    out["meta"]["edits"].append(log)
    out["meta"]["parent_content_digest"] = view["meta"]["content_digest"]
    out["meta"]["content_digest"] = content_digest(out)
    return out, log


def locality(before: dict, after: dict, permit: np.ndarray, requested: np.ndarray) -> dict:
    """Exact preservation: only requested(+permitted) est3d cells and their provenance may differ."""
    a, b = before["arrays"], after["arrays"]
    same = (a["est3d_xyz"] == b["est3d_xyz"]) | (np.isnan(a["est3d_xyz"]) & np.isnan(b["est3d_xyz"]))
    changed = ~same.all(-1)
    prov_changed = a["est3d_provenance"] != b["est3d_provenance"]
    other_arrays = {k: bool(np.array_equal(a[k], b[k], equal_nan=a[k].dtype.kind == "f"))
                    for k in a if k not in ("est3d_xyz", "est3d_provenance")}
    meta_a = {k: v for k, v in before["meta"].items() if k not in ("edits", "content_digest", "parent_content_digest")}
    meta_b = {k: v for k, v in after["meta"].items() if k not in ("edits", "content_digest", "parent_content_digest")}
    unedited = ~requested
    return {
        "changed_cells": int(changed.sum()), "requested_cells": int(requested.sum()),
        "changed_equals_requested": bool(np.array_equal(changed, requested)),
        "changed_within_permit": bool(not (changed & ~permit).any()),
        "provenance_changed_equals_requested": bool(np.array_equal(prov_changed, requested)),
        "unedited_est3d_byte_identical": bool(a["est3d_xyz"][unedited].tobytes() == b["est3d_xyz"][unedited].tobytes()),
        "other_arrays_identical": other_arrays, "all_other_arrays_identical": all(other_arrays.values()),
        "meta_identical_except_edit_log": canonical(meta_a) == canonical(meta_b),
        "unknown_preserved": bool(np.array_equal(a["est3d_finite"], b["est3d_finite"])
                                  and np.array_equal(np.isnan(a["obs2d_xy_px"]), np.isnan(b["obs2d_xy_px"]))),
        "pass": bool(np.array_equal(changed, requested) and not (changed & ~permit).any()
                     and np.array_equal(prov_changed, requested) and all(other_arrays.values())
                     and canonical(meta_a) == canonical(meta_b)),
    }


def make_directive_set(view: dict, set_id: str, directives: list[dict], **override) -> dict:
    d = {"schema": DIRECTIVE_SCHEMA, "synthetic": True, "id": set_id, "default": "prohibit",
         "view_content_digest": view["meta"]["content_digest"], "sequence_id": view["meta"]["sequence_id"],
         "actor_id": view["meta"]["actor_id"], "directives": directives}
    d.update(override)
    return d


def make_edit(view: dict, edit_id: str, cells, delta=(0.0, 0.0, 0.01), signal="est3d_xyz") -> dict:
    return {"schema": EDIT_SCHEMA, "label": "synthetic_mock_perturbation_not_animation", "edit_id": edit_id,
            "view_content_digest": view["meta"]["content_digest"], "signal": signal,
            "cells": [[int(f), n] for f, n in cells], "delta_m": list(delta)}


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Maximal [start, end] row runs of True."""
    out, i, m = [], 0, np.asarray(mask, bool)
    while i < len(m):
        if m[i]:
            j = i
            while j + 1 < len(m) and m[j + 1]:
                j += 1
            out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


# ----------------------------------------------------------------------------------------- negative controls ---

def _redigest(v):
    v["meta"]["content_digest"] = content_digest(v)
    return v


def _clone(view):
    return {"meta": copy.deepcopy(view["meta"]), "arrays": {k: v.copy() for k, v in view["arrays"].items()}}


def _nc_view(fn, redigest=True):
    def run(view):
        v = _clone(view)
        fn(v)
        return _redigest(v) if redigest else v
    return run


def _first_absent(v):
    absent = np.argwhere(~v["arrays"]["obs2d_present"])
    return tuple(absent[0]) if len(absent) else None


def _coerce_absent_zero(v):
    cell = _first_absent(v)
    if cell is None:  # no absent cell: coerce an unknown root yaw instead, or force a synthetic absent cell
        unknown = np.flatnonzero(~v["arrays"]["root_yaw_known"])
        if len(unknown):
            v["arrays"]["root_yaw_rad"][unknown[0]] = 0.0
            return
        raise RuntimeError("control not applicable: no UNKNOWN cell in this view")
    v["arrays"]["obs2d_xy_px"][cell] = 0.0
    v["arrays"]["obs2d_conf"][cell] = 0.0


def _swap_time(v):
    ts = v["arrays"]["timestamp"]
    ts[[1, 2]] = ts[[2, 1]]


def _dup_joint(v):
    v["meta"]["joint_names"][1] = v["meta"]["joint_names"][0]


def _missing_joint(v):
    v["meta"]["joint_names"] = v["meta"]["joint_names"][:-1]


def _stale(v):
    name = sorted(v["meta"]["sources"])[0]
    v["meta"]["sources"][name]["sha256"] = "0" * 64


def _no_prov(v):
    name = sorted(v["meta"]["sources"])[0]
    del v["meta"]["sources"][name]["sha256"]


def _coord(v):
    v["meta"]["coordinate_frames"]["est3d_xyz"] = "character_R"


def _unit(v):
    v["meta"]["units"]["est3d_xyz"] = "pixels_and_relative_depth"


def _no_conf(v):
    del v["arrays"]["obs2d_conf"]


def _nan_conf(v):
    present = np.argwhere(v["arrays"]["obs2d_present"])[0]
    v["arrays"]["obs2d_conf"][tuple(present)] = np.nan


def _conf3d(v):
    v["arrays"]["est3d_confidence"] = (1.0 - np.nan_to_num(v["arrays"]["obs2d_conf"])).astype(np.float64)


def _root_zero(v):
    v["arrays"]["root_translation"] = np.zeros((len(v["arrays"]["frame_index"]), 3))


def _gt_leak(v):
    v["arrays"]["target_3d"] = v["arrays"]["est3d_xyz"].copy()


def _strata_leak(v):
    v["meta"]["notes"]["strata"] = {"invalid_target_count": 0}


def _fps(v):
    v["meta"]["fps"] = v["meta"]["fps"] * 0.6


def _dup_time(v):
    v["arrays"]["timestamp"][2] = v["arrays"]["timestamp"][1]


def _shuffle_frames(v):
    v["arrays"]["frame_index"][[1, 2]] = v["arrays"]["frame_index"][[2, 1]]


def _stale_digest(v):
    v["arrays"]["est3d_xyz"][0, 0, 0] += np.float32(0.001)


# name -> (mutation, expected outcome). Frozen in the W113 declaration before any outcome is inspected.
VIEW_CONTROLS = {
    "swapped_sequence_id": ("expected_identity", "IDENTITY_MISMATCH"),
    "swapped_actor_id": ("expected_identity", "IDENTITY_MISMATCH"),
    "duplicated_joint": (_nc_view(_dup_joint), "JOINT_DUPLICATE"),
    "missing_joint": (_nc_view(_missing_joint), "JOINT_MISSING"),
    "shuffled_timestamps": (_nc_view(_swap_time), "TIME_ORDER"),
    "duplicate_timestamps": (_nc_view(_dup_time), "TIME_DUPLICATE"),
    "shuffled_frame_index": (_nc_view(_shuffle_frames), "FRAME_ORDER"),
    "changed_fps": (_nc_view(_fps), "FPS_MISMATCH"),
    "stale_provenance_hash": (_nc_view(_stale), "PROVENANCE_STALE"),
    "missing_provenance_hash": (_nc_view(_no_prov), "PROVENANCE_MISSING"),
    "coordinate_frame_mismatch": (_nc_view(_coord), "COORDINATE_MISMATCH"),
    "mixed_unit_declaration": (_nc_view(_unit), "COORDINATE_MISMATCH"),
    "missing_confidence_field": (_nc_view(_no_conf), "CONFIDENCE_MISSING"),
    "missing_confidence_cell": (_nc_view(_nan_conf), "CONFIDENCE_MISSING"),
    "manufactured_3d_confidence": (_nc_view(_conf3d), "CONFIDENCE_MANUFACTURED_3D"),
    "unknown_coerced_to_zero": (_nc_view(_coerce_absent_zero), "UNKNOWN_COERCED_TO_ZERO"),
    "unavailable_root_translation_zero": (_nc_view(_root_zero), "UNAVAILABLE_COERCED"),
    "target_3d_leakage": (_nc_view(_gt_leak), "GT_LEAKAGE"),
    "gt_strata_leakage": (_nc_view(_strata_leak), "GT_LEAKAGE"),
    "stale_content_digest": (_nc_view(_stale_digest, redigest=False), "DIGEST_STALE"),
}


def run_view_controls(view: dict, expected: dict) -> dict:
    """Positive (unchanged view passes) + every negative control must fail with exactly its declared code."""
    out = {}
    try:
        validate_view(view, expected=expected)
        out["positive_unchanged_view"] = {"expected": "PASS", "observed": "PASS", "ok": True}
    except ContractError as e:
        out["positive_unchanged_view"] = {"expected": "PASS", "observed": e.code, "ok": False, "message": str(e)}
    for name, (mutate, code) in VIEW_CONTROLS.items():
        exp = dict(expected)
        try:
            if mutate == "expected_identity":
                key = "sequence_id" if "sequence" in name else "actor_id"
                exp[key] = str(expected[key]) + "::swapped"
                candidate = view
            else:
                candidate = mutate(view)
            validate_view(candidate, expected=exp)
            observed = "PASS"
        except ContractError as e:
            observed = e.code
        except RuntimeError as e:
            observed = f"NOT_APPLICABLE: {e}"
        out[name] = {"expected": code, "observed": observed, "ok": observed == code}
    return out
