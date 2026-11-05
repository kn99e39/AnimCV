"""W114 scratch-only source-preserving Animation Base candidate: immutable evidence + versioned editable state.

NON-PRODUCTION. Not a schema, migration, Generator, Editor or version-control system. It exists only to test whether
logically separating immutable source evidence from editable animation state can carry the minimum information and
edit-authority contracts that W113 found missing:

  1. source evidence      raw 2D observation, raw detector score (exact, unbounded, never reinterpreted), validity,
                          observed/derived kind, frame/time identity, provenance (read-only arrays + content digest)
  2. reconstruction        frozen 3D estimate, coordinate frame, unit, model identity, direct-input support status;
     hypothesis           3D reliability is UNAVAILABLE (never manufactured, never 1 - score)
  3. editable state        candidate motion values (copied, never aliased), parent state id, per-cell origin
  4. user constraints      preserve / permit / prohibit (default) on joint-frame cells; conflicts are rejected
  5. output association    the channels that actually change when the frozen rig consumes a state, checked against
                          the authorized domain (observed influence, not a dependency map)

Every rejection carries a distinct code and leaves the lineage head, the stored states and the evidence untouched.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

EVIDENCE_SCHEMA = "w114_source_evidence_v0_scratch_nonproduction"
CONSTRAINT_SCHEMA = "w114_user_constraints_v0_synthetic"
EDIT_SCHEMA = "w114_edit_record_v0_mock"
ASSOC_SCHEMA = "w114_output_association_v0_scratch"
MOCK_LABEL = "MOCK_EDIT (deterministic synthetic modification; not a reconstruction correction, not animation)"

JOINTS = ("pelvis", "left_hip", "left_knee", "left_ankle", "right_hip", "right_knee", "right_ankle", "spine", "thorax",
          "neck", "head", "left_shoulder", "left_elbow", "left_wrist", "right_shoulder", "right_elbow", "right_wrist")
J = len(JOINTS)
JI = {n: i for i, n in enumerate(JOINTS)}
EST3D_SUPPORT = {0: "unknown_nonfinite", 1: "inferred_3d_input_supported", 2: "inferred_3d_input_unsupported"}
SCORE_SEMANTICS = {
    "name": "detector_keypoint_score",
    "domain": "finite real number as stored by the producer; NO upper or lower bound is assumed",
    "is_not": ["observation validity (separate producer flag)", "a calibrated probability", "3D pose reliability",
               "permission to edit"],
    "derived_quantities": "none; 1 - score is never formed",
}
UNAVAILABLE = {
    "est3d_reliability": "not produced: no calibrated 3D reliability exists; a detector score is not one",
    "est3d_calibrated_probability": "not produced",
    "est3d_actual_error": "not measurable GT-free",
    "root_translation": "not observable in root-relative FramePose",
    "physical_contact": "not produced",
}
MANUFACTURED_3D = ("est3d_reliability", "est3d_confidence", "depth_uncertainty", "est3d_calibrated_probability")
IMMUTABLE_SIGNALS = ("obs2d_xy_px", "obs2d_score_raw", "obs2d_present", "obs2d_visible", "input_valid", "frame_index",
                     "timestamp", "est3d_xyz_hypothesis")
EDITABLE_SIGNAL = "state_xyz"
EVIDENCE_ARRAYS = {  # name -> (dtype, trailing shape)
    "frame_index": ("int64", ()), "timestamp": ("float64", ()), "obs2d_xy_px": ("float64", (J, 2)),
    "obs2d_score_raw": ("float64", (J,)), "obs2d_present": ("bool", (J,)), "obs2d_visible": ("bool", (J,)),
    "input_valid": ("bool", (J,)), "est3d_xyz_hypothesis": ("float32", (J, 3)), "est3d_support": ("int8", (J,)),
    "subject_continuity_verified": ("bool", ()),
}
DIGEST_DOMAIN = b"w114_source_evidence_v0"
STATE_DOMAIN = b"w114_editable_state_v0"


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


def _hash_arrays(h, arrays: dict):
    for key in sorted(arrays):
        a = np.ascontiguousarray(arrays[key])
        h.update(f"|{key}|{a.dtype.str}|{a.shape}|".encode())
        h.update(a.tobytes())


# ------------------------------------------------------------------------------------------------- evidence ---

def build_evidence(*, sequence_id: str, regime: str, backend: str, image_size, fps: float, fps_source: str,
                   frame_index, timestamp, obs2d_xy_px, obs2d_score_raw, obs2d_present, obs2d_visible, input_valid,
                   obs2d_kind: dict, est3d_xyz, est3d_model: dict, sources: dict, subject: dict,
                   subject_continuity_verified, validity_rule: str, frame_digests=None, notes=None) -> dict:
    """Immutable evidence. Arrays are copied, made read-only and digested. Nothing is filled or rescaled."""
    T = len(frame_index)
    present = np.array(obs2d_present, bool).reshape(T, J)
    valid = np.array(input_valid, bool).reshape(T, J)
    est = np.array(est3d_xyz, np.float32).reshape(T, J, 3)
    finite = np.isfinite(est).all(-1)
    arrays = {
        "frame_index": np.array(frame_index, np.int64), "timestamp": np.array(timestamp, np.float64),
        "obs2d_xy_px": np.array(obs2d_xy_px, np.float64).reshape(T, J, 2),
        "obs2d_score_raw": np.array(obs2d_score_raw, np.float64).reshape(T, J), "obs2d_present": present,
        "obs2d_visible": np.array(obs2d_visible, bool).reshape(T, J), "input_valid": valid,
        "est3d_xyz_hypothesis": est,
        "est3d_support": np.where(~finite, 0, np.where(valid, 1, 2)).astype(np.int8),
        "subject_continuity_verified": np.array(subject_continuity_verified, bool).reshape(T),
    }
    for a in arrays.values():
        a.setflags(write=False)
    meta = {
        "schema": EVIDENCE_SCHEMA, "non_production": True, "sequence_id": sequence_id, "regime": regime,
        "backend": backend, "joint_names": list(JOINTS), "image_size": [int(image_size[0]), int(image_size[1])],
        "fps": float(fps), "fps_source": fps_source, "frame_digests": frame_digests,
        "coordinate_frames": {"obs2d_xy_px": "image_I", "est3d_xyz_hypothesis": "camera_root_relative"},
        "units": {"obs2d_xy_px": "pixels", "est3d_xyz_hypothesis": "metres", "timestamp": "seconds"},
        "obs2d_kind": dict(obs2d_kind), "score_semantics": SCORE_SEMANTICS, "validity_rule": validity_rule,
        "absent_rule": "obs2d_present=False -> xy and score NaN (UNKNOWN); never zero",
        "est3d_model": est3d_model, "est3d_note": "finite does not imply correct; support = direct input validity only",
        "subject": subject, "unavailable": {k: {"status": "unavailable", "reason": v} for k, v in UNAVAILABLE.items()},
        "sources": sources, "notes": dict(notes or {}),
    }
    ev = {"meta": meta, "arrays": arrays}
    ev["meta"]["evidence_digest"] = evidence_digest(ev)
    return ev


def evidence_digest(ev: dict) -> str:
    h = hashlib.sha256(DIGEST_DOMAIN)
    h.update(canonical({k: v for k, v in ev["meta"].items() if k != "evidence_digest"}).encode())
    _hash_arrays(h, ev["arrays"])
    return h.hexdigest()


def verify_evidence(ev: dict, *, verify_files: bool = True) -> dict:
    """Hard-reject structural, provenance, coercion and manufactured-reliability violations. No score bound."""
    meta, arrays = ev.get("meta"), ev.get("arrays")
    if not isinstance(meta, dict) or not isinstance(arrays, dict) or meta.get("schema") != EVIDENCE_SCHEMA:
        raise ContractError("SCHEMA", "not a W114 evidence store")
    for k in arrays:
        if k in MANUFACTURED_3D:
            raise ContractError("CONFIDENCE_MANUFACTURED_3D", f"{k} is not produced by any estimator")
    for k in MANUFACTURED_3D:
        if k in meta and k not in ("unavailable",):
            raise ContractError("CONFIDENCE_MANUFACTURED_3D", f"meta carries {k}")
    for k, rec in UNAVAILABLE.items():
        u = meta.get("unavailable", {}).get(k)
        if not isinstance(u, dict) or u.get("status") != "unavailable" or set(u) - {"status", "reason"}:
            raise ContractError("UNAVAILABLE_COERCED", f"{k} not recorded as unavailable without a value")
    missing = [k for k in EVIDENCE_ARRAYS if k not in arrays]
    if missing:
        raise ContractError("EVIDENCE_MISSING", f"missing evidence arrays {missing}")
    names = meta.get("joint_names")
    if not isinstance(names, list) or len(set(names)) != len(names):
        raise ContractError("JOINT_DUPLICATE", "duplicate joint identity")
    if tuple(names) != JOINTS:
        raise ContractError("JOINT_MISMATCH", "joint identities differ from the declared H36M order")
    T = len(arrays["frame_index"])
    if T == 0:
        raise ContractError("EVIDENCE_MISSING", "empty evidence")
    for k, (dtype, tail) in EVIDENCE_ARRAYS.items():
        a = arrays[k]
        if str(a.dtype) != dtype or tuple(a.shape) != (T, *tail):
            raise ContractError("SHAPE", f"{k} {a.dtype}{a.shape} != {dtype}{(T, *tail)}")
    fi, ts = arrays["frame_index"], arrays["timestamp"]
    if len(np.unique(fi)) != T or (np.diff(fi) <= 0).any():
        raise ContractError("FRAME_MISMATCH", "frame_index not unique and strictly increasing")
    if not np.isfinite(ts).all() or (np.diff(ts) <= 0).any():
        raise ContractError("TIME_MISMATCH", "timestamps not finite and strictly increasing")
    fps = meta.get("fps")
    if not isinstance(fps, (int, float)) or not fps > 0 or not math.isfinite(fps) or np.abs(ts - fi / fps).max() > 1e-9:
        raise ContractError("TIME_MISMATCH", "timestamps disagree with frame_index / fps")
    if meta.get("coordinate_frames", {}).get("est3d_xyz_hypothesis") != "camera_root_relative" \
            or meta.get("units", {}).get("est3d_xyz_hypothesis") != "metres":
        raise ContractError("COORDINATE_MISMATCH", "3D hypothesis frame/unit declaration")
    present, score, xy = arrays["obs2d_present"], arrays["obs2d_score_raw"], arrays["obs2d_xy_px"]
    if not np.isfinite(score[present]).all():
        raise ContractError("SCORE_MISSING", "present observation without a finite raw score")
    if not np.isfinite(xy[present]).all():
        raise ContractError("EVIDENCE_MISSING", "present observation without a coordinate")
    if np.isfinite(xy[~present]).any() or np.isfinite(score[~present]).any():
        zero = (xy[~present] == 0).any() or (score[~present] == 0).any()
        raise ContractError("UNKNOWN_COERCED_TO_ZERO" if zero else "UNKNOWN_COERCED", "absent observation carries a value")
    if (arrays["input_valid"] & ~present).any():
        raise ContractError("VALIDITY_WITHOUT_OBSERVATION", "input_valid on an absent observation")
    est, sup = arrays["est3d_xyz_hypothesis"], arrays["est3d_support"]
    exp_sup = np.where(~np.isfinite(est).all(-1), 0, np.where(arrays["input_valid"], 1, 2))
    if not np.array_equal(sup, exp_sup):
        raise ContractError("UNKNOWN_COERCED", "3D support status disagrees with the hypothesis/validity")
    if (np.isfinite(est).all(-1) != ~np.isnan(est).any(-1)).any():
        raise ContractError("UNKNOWN_COERCED", "partially finite 3D cell")
    sources = meta.get("sources")
    if not isinstance(sources, dict) or not sources:
        raise ContractError("PROVENANCE_MISSING", "no source provenance")
    for name, src in sources.items():
        d = src.get("sha256") if isinstance(src, dict) else None
        if not isinstance(d, str) or len(d) != 64 or any(c not in "0123456789abcdef" for c in d):
            raise ContractError("PROVENANCE_MISSING", f"source {name} lacks a sha256")
        if verify_files and src.get("path") and sha_file(src["path"]) != d:
            raise ContractError("PROVENANCE_STALE", f"source {name} bytes differ from the recorded sha256")
    if not isinstance(meta.get("subject"), dict) or "per_frame_association" not in meta["subject"]:
        raise ContractError("IDENTITY_MISSING", "subject identity status not recorded")
    if meta.get("evidence_digest") != evidence_digest(ev):
        raise ContractError("EVIDENCE_MODIFIED", "evidence content differs from its digest")
    return {"frames": T, "evidence_digest": meta["evidence_digest"]}


def verify_against_source(ev: dict, reference: dict) -> dict:
    """Byte-exact comparison of evidence arrays with independently loaded source arrays (e.g. raw scores)."""
    out = {}
    for k, ref in reference.items():
        a, r = ev["arrays"][k], np.asarray(ref)
        fl = a.dtype.kind == "f"
        rc = r.astype(a.dtype)
        out[k] = bool(a.shape == r.shape and np.array_equal(a, rc, equal_nan=fl)
                      and np.array_equal(rc.astype(r.dtype), r, equal_nan=r.dtype.kind == "f"))
    bad = [k for k, v in out.items() if not v]
    if bad:
        raise ContractError("SOURCE_MISMATCH", f"evidence differs from source for {bad}")
    return out


def score_accounting(ev: dict, threshold: float | None = None) -> dict:
    a = ev["arrays"]
    p, s = a["obs2d_present"], a["obs2d_score_raw"]
    sp = s[p]
    out = {"cells": int(p.size), "present": int(p.sum()), "absent_unknown": int((~p).sum()),
           "finite_scores": int(np.isfinite(sp).sum()), "above_1": int((sp > 1).sum()), "below_0": int((sp < 0).sum()),
           "max": float(sp.max()) if sp.size else None, "min": float(sp.min()) if sp.size else None,
           "above_1_valid": int((p & (np.nan_to_num(s) > 1) & a["input_valid"]).sum()),
           "one_minus_score_negative_if_formed": int((1 - sp < 0).sum()),
           "above_1_per_joint": {n: int((p[:, i] & (np.nan_to_num(s[:, i]) > 1)).sum()) for i, n in enumerate(JOINTS)
                                 if (p[:, i] & (np.nan_to_num(s[:, i]) > 1)).any()},
           "above_1_frames": int((p & (np.nan_to_num(s) > 1)).any(1).sum())}
    if threshold is not None:
        obs = np.array([ev["meta"]["obs2d_kind"][n].startswith("observed") for n in JOINTS])
        thr = p & (np.nan_to_num(s, nan=-np.inf) >= threshold)
        out["validity_vs_score_threshold"] = {
            "threshold": threshold, "observed_kind_cells": int(p[:, obs].sum()),
            "valid_equals_score_ge_threshold_observed": int((a["input_valid"][:, obs] == thr[:, obs]).sum()),
            "derived_kind_cells": int(p[:, ~obs].sum()),
            "valid_equals_score_ge_threshold_derived": int((a["input_valid"][:, ~obs] == thr[:, ~obs]).sum())}
    return out


# ---------------------------------------------------------------------------------------------- constraints ---

def make_constraints(ev: dict, cid: str, directives: list[dict], **override) -> dict:
    c = {"schema": CONSTRAINT_SCHEMA, "synthetic": True, "id": cid, "default": "prohibit",
         "evidence_digest": ev["meta"]["evidence_digest"], "sequence_id": ev["meta"]["sequence_id"],
         "directives": copy.deepcopy(directives)}
    c.update(override)
    return c


def constraints_digest(cs: dict) -> str:
    return hashlib.sha256(canonical(cs).encode()).hexdigest()


def resolve_constraints(ev: dict, cs: dict) -> dict[str, np.ndarray]:
    """Directives -> preserve / permit / prohibit cell masks on the editable state. Default prohibit; conflicts reject."""
    a, meta = ev["arrays"], ev["meta"]
    if cs.get("schema") != CONSTRAINT_SCHEMA or cs.get("synthetic") is not True or cs.get("default") != "prohibit":
        raise ContractError("SCHEMA", "constraints must be the labelled synthetic scratch schema, default prohibit")
    if cs.get("evidence_digest") != meta["evidence_digest"]:
        raise ContractError("STALE_EVIDENCE", "constraints bound to a different evidence digest")
    if cs.get("sequence_id") != meta["sequence_id"]:
        raise ContractError("IDENTITY_MISMATCH", "constraints bound to a different sequence")
    fi, ts = a["frame_index"], a["timestamp"]
    row = {int(f): i for i, f in enumerate(fi)}
    T = len(fi)
    m = {k: np.zeros((T, J), bool) for k in ("preserve", "permit", "prohibit")}
    for d in cs["directives"]:
        if d.get("signal", EDITABLE_SIGNAL) != EDITABLE_SIGNAL:
            raise ContractError("OBSERVATION_IMMUTABLE", f"{d.get('id')}: {d.get('signal')} is evidence, not editable")
        if d.get("kind") not in m:
            raise ContractError("SCHEMA", f"{d.get('id')}: kind {d.get('kind')!r}")
        bad = [n for n in d["joints"] if n not in JI]
        if bad or len(set(d["joints"])) != len(d["joints"]):
            raise ContractError("JOINT_MISMATCH", f"{d.get('id')}: unknown or duplicate joints {bad}")
        rows = np.zeros(T, bool)
        for k, (s, e) in enumerate(d["spans"]):
            if int(s) not in row or int(e) not in row or s > e:
                raise ContractError("FRAME_MISMATCH", f"{d.get('id')}: span {s}-{e} not bounded by evidence frames")
            if "time_spans_s" in d:
                t0, t1 = d["time_spans_s"][k]
                if abs(ts[row[int(s)]] - t0) > 1e-9 or abs(ts[row[int(e)]] - t1) > 1e-9:
                    raise ContractError("TIME_MISMATCH", f"{d.get('id')}: span time disagrees with source time")
            rows |= (fi >= s) & (fi <= e)
        m[d["kind"]][np.ix_(rows, [JI[n] for n in d["joints"]])] = True
    if (m["preserve"] & m["permit"]).any():
        raise ContractError("LOCK_CONFLICT", f"{int((m['preserve'] & m['permit']).sum())} cells preserved and permitted")
    if (m["prohibit"] & m["permit"]).any():
        raise ContractError("PERMIT_PROHIBIT_CONFLICT", "cells explicitly prohibited and permitted")
    return m


# ----------------------------------------------------------------------------------------------------- state ---

def state_id(evidence_digest_: str, parent: str | None, record_digest: str | None, values, origin) -> str:
    h = hashlib.sha256(STATE_DOMAIN)
    h.update(canonical([evidence_digest_, parent, record_digest]).encode())
    _hash_arrays(h, {"values": values, "origin": origin})
    return h.hexdigest()


def record_digest(rec: dict | None) -> str | None:
    return None if rec is None else hashlib.sha256(canonical(rec).encode()).hexdigest()


class Lineage:
    """Append-only versioned editable state over ONE immutable evidence store."""

    def __init__(self, ev: dict):
        verify_evidence(ev, verify_files=False)
        self.evidence_digest = ev["meta"]["evidence_digest"]
        values = np.array(ev["arrays"]["est3d_xyz_hypothesis"], np.float32, copy=True)
        origin = np.zeros(values.shape[:2], np.int32)  # 0 = reconstruction hypothesis passthrough; k = edit number k
        sid = state_id(self.evidence_digest, None, None, values, origin)
        values.setflags(write=False)
        origin.setflags(write=False)
        self.states = {sid: {"state_id": sid, "parent": None, "record": None, "values": values, "origin": origin}}
        self.order = [sid]
        self.head = sid
        self.records: dict[str, dict] = {}

    def state(self, sid=None) -> dict:
        return self.states[sid or self.head]

    def snapshot(self) -> tuple:
        return self.head, len(self.states), len(self.records)

    def audit(self) -> dict:
        """Recompute every state id from its parent link, record and values; verify the chain to the root."""
        ok, n = True, 0
        for sid in self.order:
            s = self.states[sid]
            ok &= state_id(self.evidence_digest, s["parent"], record_digest(s["record"]), s["values"], s["origin"]) == sid
            ok &= s["parent"] is None or s["parent"] in self.states
            n += 1
        chain, cur = [], self.head
        while cur is not None:
            chain.append(cur)
            cur = self.states[cur]["parent"]
        return {"states": n, "ids_recomputed_equal": bool(ok), "head_chain_length": len(chain),
                "root_is_hypothesis": self.states[chain[-1]]["parent"] is None}

    def _commit(self, parent: str, rec: dict, values, origin) -> dict:
        sid = state_id(self.evidence_digest, parent, record_digest(rec), values, origin)
        values.setflags(write=False)
        origin.setflags(write=False)
        self.states[sid] = {"state_id": sid, "parent": parent, "record": rec, "values": values, "origin": origin}
        self.order.append(sid)
        self.records[rec["edit_id"]] = {"record": rec, "child": sid, "parent": parent}
        self.head = sid
        return self.states[sid]


def make_mock_edit(lin: Lineage, ev: dict, cs: dict, edit_id: str, cells, delta_m=(0.0, 0.0, 0.01), *,
                   op: str = "offset", source_frame: int | None = None, rig_claim: bool = False,
                   signal: str = EDITABLE_SIGNAL, coordinate_frame: str = "camera_root_relative", unit: str = "metres",
                   base: str | None = None) -> dict:
    a = ev["arrays"]
    row = {int(f): i for i, f in enumerate(a["frame_index"])}
    return {"schema": EDIT_SCHEMA, "label": MOCK_LABEL, "edit_id": edit_id, "op": op, "signal": signal,
            "base_state_id": base or lin.head, "evidence_digest": ev["meta"]["evidence_digest"],
            "constraints_digest": constraints_digest(cs), "constraints_id": cs["id"],
            "coordinate_frame": coordinate_frame, "unit": unit,
            "targets": [[int(f), n] for f, n in cells],
            "target_timestamps": [float(a["timestamp"][row[int(f)]]) if int(f) in row else None for f, _ in cells],
            "delta_m": np.asarray(delta_m, np.float64).tolist() if op == "offset" else None,
            "source_frame": source_frame, "requires_cross_frame_identity": op == "copy_from_frame",
            "rig_claim": bool(rig_claim)}


def _prepare_edit(lin: Lineage, ev: dict, cs: dict, rec: dict):
    """All source-level checks; returns (rows, cols, new_values, new_origin, masks) without committing."""
    if rec.get("schema") != EDIT_SCHEMA or rec.get("label") != MOCK_LABEL:
        raise ContractError("SCHEMA", "edit must be the labelled mock edit record")
    verify_evidence(ev, verify_files=True)
    if rec.get("evidence_digest") != ev["meta"]["evidence_digest"] or lin.evidence_digest != ev["meta"]["evidence_digest"]:
        raise ContractError("STALE_EVIDENCE", "edit or lineage bound to a different evidence digest")
    if rec.get("base_state_id") != lin.head:
        raise ContractError("STALE_PARENT", "edit was prepared against a state that is not the lineage head")
    if rec.get("signal") in IMMUTABLE_SIGNALS:
        raise ContractError("OBSERVATION_IMMUTABLE", f"{rec.get('signal')} is immutable evidence")
    if rec.get("signal") != EDITABLE_SIGNAL:
        raise ContractError("SCHEMA", f"unknown signal {rec.get('signal')!r}")
    if rec.get("coordinate_frame") != ev["meta"]["coordinate_frames"]["est3d_xyz_hypothesis"] \
            or rec.get("unit") != ev["meta"]["units"]["est3d_xyz_hypothesis"]:
        raise ContractError("COORDINATE_MISMATCH", f"edit in {rec.get('coordinate_frame')}/{rec.get('unit')}")
    if rec.get("constraints_digest") != constraints_digest(cs):
        raise ContractError("CONSTRAINTS_STALE", "edit bound to different constraints")
    masks = resolve_constraints(ev, cs)
    a = ev["arrays"]
    row = {int(f): i for i, f in enumerate(a["frame_index"])}
    cells = []
    for (f, n), t in zip(rec["targets"], rec["target_timestamps"]):
        if int(f) not in row:
            raise ContractError("FRAME_MISMATCH", f"target frame {f} not in evidence")
        if t is None or abs(float(a["timestamp"][row[int(f)]]) - float(t)) > 1e-9:
            raise ContractError("TIME_MISMATCH", f"target frame {f} timestamp {t} != source time")
        if n not in JI:
            raise ContractError("JOINT_MISMATCH", f"unknown joint {n}")
        cells.append((row[int(f)], JI[n]))
    if not cells or len(set(cells)) != len(cells):
        raise ContractError("SCHEMA", "targets empty or duplicated")
    if rec.get("requires_cross_frame_identity"):
        sf = rec.get("source_frame")
        if sf is None or int(sf) not in row:
            raise ContractError("FRAME_MISMATCH", "cross-frame source frame not in evidence")
        involved = {row[int(sf)]} | {r for r, _ in cells}
        if not all(bool(a["subject_continuity_verified"][r]) for r in involved):
            raise ContractError("IDENTITY_UNVERIFIED", "operation needs verified same-subject continuity across frames")
    if any(masks["preserve"][c] for c in cells):
        raise ContractError("LOCK_CONFLICT_EDIT", "a target cell is preserved")
    if not all(masks["permit"][c] for c in cells):
        raise ContractError("EDIT_NOT_AUTHORIZED", "a target cell is not permitted (default prohibit)")
    cur = lin.state()
    if not all(np.isfinite(cur["values"][c]).all() for c in cells):
        raise ContractError("EDIT_UNKNOWN_TARGET", "a target cell is UNKNOWN; a mock edit may not fill it")
    rows, cols = (np.asarray(x) for x in zip(*cells))
    values = np.array(cur["values"], copy=True)
    origin = np.array(cur["origin"], copy=True)
    if rec["op"] == "offset":
        values[rows, cols] = values[rows, cols] + np.asarray(rec["delta_m"], np.float32)
    elif rec["op"] == "copy_from_frame":
        values[rows, cols] = values[row[int(rec["source_frame"])], cols]
    else:
        raise ContractError("SCHEMA", f"unknown op {rec['op']}")
    if not np.isfinite(values[rows, cols]).all():
        raise ContractError("REPLACEMENT_NONFINITE", "replacement is not finite")
    origin[rows, cols] = len(lin.records) + 1
    return rows, cols, values, origin, masks


def apply_edit(lin: Lineage, ev: dict, cs: dict, rec: dict, rig=None) -> dict:
    """Atomic: validate, (optionally) check observed rig influence against the authorized domain, then commit."""
    rows, cols, values, origin, masks = _prepare_edit(lin, ev, cs, rec)
    cur = lin.state()
    applied = dict(rec)
    applied["original_values"] = cur["values"][rows, cols].astype(np.float64).tolist()
    applied["original_values_hex"] = cur["values"][rows, cols].tobytes().hex()
    applied["replacement_values_hex"] = values[rows, cols].tobytes().hex()
    applied["original_origin"] = cur["origin"][rows, cols].tolist()
    changed = ~((values == cur["values"]) | (np.isnan(values) & np.isnan(cur["values"]))).all(-1)
    applied["changed_cells"] = int(changed.sum())
    applied["unchanged_cells"] = int((~changed).sum())
    applied["changed_outside_targets"] = int((changed & ~_mask_of(rows, cols, changed.shape)).sum())
    if rig is not None:
        assoc = rig_check(rig, ev, masks, cur["values"], values, rec)
        applied["output_association"] = assoc
    return lin._commit(lin.head, applied, values, origin)


def _mask_of(rows, cols, shape):
    m = np.zeros(shape, bool)
    m[rows, cols] = True
    return m


def make_revert(lin: Lineage, edit_id: str, revert_id: str | None = None) -> dict:
    e = lin.records[edit_id]["record"]
    return {"schema": EDIT_SCHEMA, "label": MOCK_LABEL, "op": "revert", "edit_id": revert_id or f"revert:{edit_id}",
            "reverts": edit_id, "base_state_id": lin.head, "evidence_digest": lin.evidence_digest,
            "targets": e["targets"], "restore_values_hex": e["original_values_hex"],
            "restore_origin": e["original_origin"]}


def apply_revert(lin: Lineage, ev: dict, rec: dict) -> dict:
    """Restore the reverted edit's original values; refuse a revert that would not restore its parent state."""
    if rec.get("schema") != EDIT_SCHEMA or rec.get("op") != "revert":
        raise ContractError("SCHEMA", "not a revert record")
    verify_evidence(ev, verify_files=True)
    if rec.get("evidence_digest") != ev["meta"]["evidence_digest"] or lin.evidence_digest != ev["meta"]["evidence_digest"]:
        raise ContractError("STALE_EVIDENCE", "revert bound to a different evidence digest")
    if rec.get("base_state_id") != lin.head:
        raise ContractError("STALE_PARENT", "revert prepared against a non-head state")
    if rec.get("reverts") not in lin.records:
        raise ContractError("REVERT_UNKNOWN", "reverted edit not in lineage")
    entry = lin.records[rec["reverts"]]
    e, parent = entry["record"], lin.states[entry["parent"]]
    a = ev["arrays"]
    row = {int(f): i for i, f in enumerate(a["frame_index"])}
    rows = np.array([row[int(f)] for f, _ in e["targets"]])
    cols = np.array([JI[n] for _, n in e["targets"]])
    cur = lin.state()
    if cur["values"][rows, cols].tobytes().hex() != e["replacement_values_hex"]:
        raise ContractError("REVERT_CONFLICT", "target cells changed after the reverted edit")
    restore = np.frombuffer(bytes.fromhex(rec["restore_values_hex"]), np.float32).reshape(len(rows), 3)
    if rec["restore_values_hex"] != e["original_values_hex"] or restore.tobytes() != parent["values"][rows, cols].tobytes() \
            or list(rec["restore_origin"]) != parent["origin"][rows, cols].tolist():
        raise ContractError("REVERT_MISMATCH", "revert would not restore the reverted edit's parent values")
    values = np.array(cur["values"], copy=True)
    origin = np.array(cur["origin"], copy=True)
    values[rows, cols] = restore
    origin[rows, cols] = np.asarray(rec["restore_origin"], np.int32)
    applied = dict(rec)
    applied["restores_parent_state_fully"] = bool(values.tobytes() == parent["values"].tobytes()
                                                  and origin.tobytes() == parent["origin"].tobytes())
    return lin._commit(lin.head, applied, values, origin)


# --------------------------------------------------------------------------------------- output association ---

class RigModel:
    """A frozen rig as a pure function of a (T, J, 3) state plus its declared channel facts.

    channels(values) -> {channel: (T, 4) float64 array, NaN rows where the rig emits nothing source-driven}
    depicts[channel]  -> joints whose geometry that channel's LOCAL value places (tail-joint projection)
    owners[channel]   -> declared source owner (from the implementation; confirmed or refuted by measurement)
    fixed             -> channels held at rest (no source)
    """

    def __init__(self, name, fn, depicts: dict, owners: dict, fixed=(), tol=1e-9):
        self.name, self.fn, self.depicts, self.owners, self.fixed, self.tol = name, fn, depicts, owners, tuple(fixed), tol

    def channels(self, values):
        return self.fn(values)


def quat_delta(a, b):
    """Max |component| difference after hemisphere alignment; NaN-vs-NaN = 0, NaN-vs-value = inf."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = np.isnan(a).any(-1), np.isnan(b).any(-1)
    sign = np.where((a * b).sum(-1, keepdims=True) < 0, -1.0, 1.0)
    d = np.abs(a - sign * b).max(-1)
    d = np.where(na & nb, 0.0, np.where(na ^ nb, np.inf, d))
    return d


def influence(rig: RigModel, before: dict, after: dict) -> dict:
    """Observed channel-frame changes between two channel sets (declared tolerance) plus exact-bit changes."""
    out = {}
    for c in before:
        d = quat_delta(before[c], after[c])
        bits = ~((before[c] == after[c]) | (np.isnan(before[c]) & np.isnan(after[c]))).all(-1)
        out[c] = {"changed": d > rig.tol, "bit_changed": bits, "max_delta": float(np.max(d)) if d.size else 0.0}
    return out


def rig_check(rig: RigModel, ev: dict, masks, before_values, after_values, rec) -> dict:
    """Observed influence of a candidate state vs its parent, checked against the authorized domain."""
    fi = ev["arrays"]["frame_index"]
    inf = influence(rig, rig.channels(before_values), rig.channels(after_values))
    changed = {c: [int(fi[r]) for r in np.flatnonzero(v["changed"])] for c, v in inf.items() if v["changed"].any()}
    violations = []
    for c, frames in changed.items():
        if c in rig.fixed or not rig.depicts.get(c):
            violations.append({"channel": c, "reason": "fixed/rest component changed"})
            continue
        cols = [JI[n] for n in rig.depicts[c]]
        rows = np.flatnonzero(inf[c]["changed"])
        pres = masks["preserve"][np.ix_(rows, cols)].any(1)
        notp = ~masks["permit"][np.ix_(rows, cols)].all(1)
        if pres.any() or notp.any():
            violations.append({"channel": c, "depicts": rig.depicts[c], "frames_preserved": int(pres.sum()),
                               "frames_not_permitted": int(notp.sum()), "first_frame": int(fi[rows[np.argmax(pres | notp)]])})
    assoc = {"schema": ASSOC_SCHEMA, "rig": rig.name, "changed_channel_frames": {c: len(f) for c, f in changed.items()},
             "changed_frames": changed, "max_delta": {c: v["max_delta"] for c, v in inf.items()},
             "bit_changed_channel_frames": {c: int(v["bit_changed"].sum()) for c, v in inf.items()},
             "violations": violations}
    if rec.get("rig_claim") and not changed:
        raise ContractError("NO_RIG_INFLUENCE", "edit claims rig animation but no rig channel changes under the frozen rig")
    if violations:
        raise ContractError("OUTPUT_OUTSIDE_AUTHORIZED_DOMAIN", f"{len(violations)} changed channels depict joints that "
                                                                f"are preserved or not permitted: {violations[:2]}")
    return assoc


def make_association(rig: RigModel, ev: dict, lin: Lineage, *, action_fps: float, key_offset: int = 1) -> dict:
    fi, ts = ev["arrays"]["frame_index"], ev["arrays"]["timestamp"]
    keys = fi + key_offset
    return {"schema": ASSOC_SCHEMA, "rig": rig.name, "state_id": lin.head, "evidence_digest": lin.evidence_digest,
            "channels": sorted(rig.depicts) + sorted(rig.fixed), "channel_owner": dict(rig.owners),
            "frames": {"frame_index": fi.tolist(), "source_time_s": ts.tolist(), "action_key_frame": keys.tolist(),
                       "action_time_s_at_scene_fps": ((keys - key_offset) / action_fps).tolist()}}


def validate_association(assoc: dict, rig: RigModel, ev: dict, lin: Lineage) -> dict:
    if assoc.get("schema") != ASSOC_SCHEMA or assoc.get("rig") != rig.name:
        raise ContractError("ASSOCIATION_INVALID", "schema or rig identity")
    if assoc.get("evidence_digest") != ev["meta"]["evidence_digest"] or assoc.get("state_id") not in lin.states:
        raise ContractError("ASSOCIATION_INVALID", "association bound to an unknown state or evidence")
    if assoc.get("channel_owner") != rig.owners:
        bad = {c: o for c, o in assoc.get("channel_owner", {}).items() if rig.owners.get(c) != o}
        raise ContractError("ASSOCIATION_INVALID", f"channel ownership differs from the rig configuration: {bad}")
    f = assoc["frames"]
    if f["frame_index"] != ev["arrays"]["frame_index"].tolist() or f["source_time_s"] != ev["arrays"]["timestamp"].tolist():
        raise ContractError("ASSOCIATION_INVALID", "frame/time mapping differs from the evidence")
    return {"ok": True}
