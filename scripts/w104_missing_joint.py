#!/usr/bin/env python3
"""W104: missing-joint causality and recoverability in frozen S0 (evidence-only diagnostic).

Stages (each refuses to overwrite its output directory):
  verify   frozen W102/W103 identities, raw M1 landmark provenance, M input reproduction from retained landmarks (no 3D)
  declare  2D-only intervention manifest (patterns, controls, factorial, restoration rule, gates), hashed BEFORE any
           new S0 outcome is computed; reads no GT array and no S0 output
  infer    builds the counterfactual model-visible tensors (F00/F10/F01/F11, source-only restoration R11/R10/R01/Rc, and
           separately labelled privileged references) and runs the frozen S0 on them; verifies M_Main reproduction
  analyze  paired outcome accounting under the declaration (GT is evaluation-only here); see w104_analysis.py
  video    GT-free historical real-clip consequences of the same declared interventions

No production module, checkpoint, threshold, mask policy, crop rule or normalization is changed. F01/F10 are
counterfactual diagnostics, not production observations. GT never enters a factorial or source-only restoration input;
the oracle-geometry reference is constructed in a separate, explicitly privileged function.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"), str(ROOT/"scripts")]

from w103_observability import (BASE, CONDITIONS, DIRECT, JOINTS, OBSERVATIONS_SHA, S0_CHECKPOINT_SHA, SIDE_SHA,
                                SMPL_OF, W102_FILES, W102_MANIFEST_SHA, ENVIRONMENT_FINGERPRINT, dump, geometry_all,
                                load_context, sha, video_of)

# ------------------------------------------------------------------ frozen identities
W103_MANIFEST_SHA = "3aad853cc52bc38efb85cd59833c031410df2788236fb6fd672ad1236de14f46"
W103_DECLARATION_SHA = "fda5c08bfc1c64fcb4cc476bf7f1d21cfe095322c479db588d3dfec372e0038b"
W103_PER_ROW = "results/analysis_r3/per_row.npz"
W103_PER_ROW_SHA = "3d43ed7a09335c412e3b9864def3d79619707eb9c018238fc3d7ccc8625092ca"
OBSERVE_MANIFEST_SHA = "6492b3fcf58cc434454e2bb74beb6b637c67d9e21f0169d3c39a773e53f9fb65"
VIDEO_PREDICTIONS = "results/historical_video/observations_predictions.npz"

# ------------------------------------------------------------------ fixed construction constants (not tuned)
RESTORED_FIXED_CONF = 0.3        # Rc: restored joints carry the unchanged production visibility boundary as confidence
F00_BUDGET_MM = 0.1              # W102 numerical repeatability budget (max raw joint displacement)
NATURAL_TOP = 12                 # natural missing-limb sets considered for extra patterns (by 2D-mask frequency only)
TEMPORAL_MAX_OFFSET = 5          # bank samples searched before/after a naturally missing joint
TEMPORAL_REPORT_OFFSETS = (1, 2, 3, 5)
NATURAL_MIN_ROWS = 30            # EXPLAINED gate: natural pattern classes with at least this many rows

LEG = {"left": (1, 2, 3), "right": (4, 5, 6)}
ARM = {"left": (11, 12, 13), "right": (14, 15, 16)}
MIRROR = {1: 11, 2: 12, 3: 13, 4: 14, 5: 15, 6: 16, 11: 1, 12: 2, 13: 3, 14: 4, 15: 5, 16: 6}
LIMB_OF = {1: "leg_l", 2: "leg_l", 3: "leg_l", 4: "leg_r", 5: "leg_r", 6: "leg_r",
           11: "arm_l", 12: "arm_l", 13: "arm_l", 14: "arm_r", 15: "arm_r", 16: "arm_r"}
LIMB_SEGMENTS = {"leg_l": ("thigh_l", "shin_l"), "leg_r": ("thigh_r", "shin_r"),
                 "arm_l": ("upper_arm_l", "forearm_l"), "arm_r": ("upper_arm_r", "forearm_r")}
KNEE_OF_LIMB = {"leg_l": 0, "leg_r": 1}
FACTORIAL = ("F00", "F10", "F01", "F11")
RESTORE = ("R11", "R10", "R01", "Rc")
PRIVILEGED = ("P_oracle", "P_bench")
REGIME = {
    "F00": "real_animcv_observation (frozen M_Main control)",
    "F10": "counterfactual_diagnostic (mask-only, original crop; not a valid production observation)",
    "F01": "counterfactual_diagnostic (crop-only, original validity; not a valid production observation)",
    "F11": "counterfactual_diagnostic (production-consistent missingness: mask + recrop)",
    "R11": "real_animcv_observation_source_only_restoration (evaluation-only; raw below-threshold M1 joints, recrop)",
    "R10": "counterfactual_diagnostic (restored joints, original crop)",
    "R01": "counterfactual_diagnostic (original validity, restored crop)",
    "Rc": "counterfactual_diagnostic (R11 with restored-joint confidence fixed at 0.3)",
    "P_oracle": "oracle_geometry_privileged_upper_bound (GT projections replace restored coordinates; NOT GT-free)",
    "P_bench": "benchmark_detector_observation_privileged_reference (3DPW benchmark 2D fills M-missing joints)",
}


# ------------------------------------------------------------------ pattern declaration (2D masks only)
def canon(joints):
    return tuple(sorted(int(j) for j in joints))


def pattern_name(kind, joints):
    return kind + ":" + "+".join(JOINTS[j] for j in joints)


def natural_sets(valid_masks):
    """Exact missing-limb-joint sets (H36M indices) over rows with >=1 invalid limb joint; uses validity masks only."""
    counter = Counter()
    for row in np.asarray(valid_masks, bool):
        missing = tuple(j for j in DIRECT if not row[j])
        if missing:
            counter[canon(missing)] += 1
    return counter


def declared_patterns(natural):
    """Deterministic pattern list: singles, same-limb chains, bilateral, top natural extras, mirrored controls."""
    pats = [("single", (j,)) for j in DIRECT]
    for side in ("left", "right"):
        h, k, a = LEG[side]
        s, e, w = ARM[side]
        pats += [("limb", (k, a)), ("limb", (h, k, a)), ("limb", (e, w)), ("limb", (s, e, w))]
    pats += [("bilateral", (3, 6)), ("bilateral", (2, 3, 5, 6)), ("bilateral", (13, 16)), ("bilateral", (12, 13, 15, 16))]
    pats = [(k, canon(j)) for k, j in pats]
    have = {j for _, j in pats}
    ranked = sorted(natural.items(), key=lambda kv: (-kv[1], kv[0]))[:NATURAL_TOP]
    for joints, _ in ranked:
        if joints not in have:
            pats.append(("natural", joints)); have.add(joints)
    for kind, joints in list(pats):
        m = mirror(joints)
        if m is not None and m not in have:
            pats.append(("control", m)); have.add(m)
    return pats


def mirror(joints):
    """Count-matched unrelated control: same-side leg<->arm mirror; only for pure-leg or pure-arm patterns."""
    limbs = {LIMB_OF[j][:3] for j in joints}
    if len(limbs) != 1:
        return None
    return canon(MIRROR[j] for j in joints)


def affected_limbs(joints):
    return sorted({LIMB_OF[j] for j in joints})


# ------------------------------------------------------------------ counterfactual construction (no GT argument anywhere)
def closure(remove):
    """Derived-token rule (production _midpoint): pelvis/spine need both hips; neck/thorax/spine need both shoulders."""
    R = set(int(j) for j in remove)
    if R & {1, 4}:
        R |= {0, 7}
    if R & {11, 14}:
        R |= {8, 9, 7}
    return sorted(R)


def factorial_row(x, v, size, remove):
    """Return the four (17,4) tensors and both crops for one row and one declared removal set.

    F00 original validity/original crop; F10 intervention mask/original crop; F01 original validity/recrop as though the
    joints were missing; F11 intervention mask/production-consistent recrop. Refuses non-applicable or degenerate cases.
    """
    from framepose.crops import crop_box, geometry_in_crop
    x = np.asarray(x, np.float32); v = np.asarray(v, bool)
    if any(not v[j] for j in remove):
        raise ValueError("pattern not applicable: a declared joint is not valid in the original observation")
    v1 = v.copy()
    v1[closure(remove)] = False
    if int(v1.sum()) < 2:
        raise ValueError("degenerate intervention: fewer than two valid joints remain")
    c0, c1 = crop_box(x, v, size), crop_box(x, v1, size)
    out = {"F00": geometry_in_crop(x, v, size, c0), "F10": geometry_in_crop(x, v1, size, c0),
           "F01": geometry_in_crop(x, v, size, c1), "F11": geometry_in_crop(x, v1, size, c1)}
    return out, c0, c1


def crop_shift(c0, c1):
    side_ratio = c1.side/c0.side
    centre = np.hypot((c1.x+c1.side/2)-(c0.x+c0.side/2), (c1.y+c1.side/2)-(c0.y+c0.side/2))/c0.side
    return float(side_ratio), float(centre)


def restore_row(lm, size, x, v):
    """Fixed source-availability rule: every finite raw M1 keypoint becomes valid with its raw coordinate and confidence.

    Derived tokens follow the production midpoint rule on the restored landmarks (their raw landmark rows already hold the
    pixel midpoint and min confidence). Originally valid tokens are returned bit-identical.
    """
    from w101_reference_estimator import to_input
    raw, have = to_input(lm, size, raw=True)
    x = np.asarray(x, np.float32); v = np.asarray(v, bool)
    if not (have[v].all() and np.array_equal(raw[v], x[v])):
        raise ValueError("STOP: restoration would alter an originally valid token")
    return raw, have, have & ~v


def restoration_inputs(x, v, raw, have, size):
    from framepose.crops import crop_box, geometry_in_crop
    c0, c1 = crop_box(x, v, size), crop_box(raw, have, size)
    rc = raw.copy()
    rc[have & ~v, 2] = RESTORED_FIXED_CONF
    return {"R11": geometry_in_crop(raw, have, size, c1), "R10": geometry_in_crop(raw, have, size, c0),
            "R01": geometry_in_crop(x, v, size, c1), "Rc": geometry_in_crop(rc, have, size, c1)}, c0, c1


def recompute_derived(px, valid, conf, changed):
    """Pixel-space derived tokens recomputed only where a parent changed (production midpoint/min-confidence rule)."""
    rules = ((0, (1, 4)), (9, (11, 14)), (8, (11, 14)), (7, (9, 0)))
    for d, (a, b) in rules:
        if changed[a] or changed[b] or (valid[a] and valid[b] and not valid[d]):
            valid[d] = bool(valid[a] and valid[b])
            if valid[d]:
                px[d] = (px[a]+px[b])/2
                conf[d] = min(conf[a], conf[b])
            else:
                px[d] = 0.; conf[d] = 0.
            changed[d] = True
    return px, valid, conf


def privileged_oracle_row(raw, have, new, size, uv):
    """PRIVILEGED (oracle_geometry): R11 validity/confidence, but restored base-joint coordinates = GT projection."""
    from framepose.crops import crop_box, geometry_in_crop
    W, H = float(size[0]), float(size[1])
    px = raw[:, :2].astype(np.float64)*[W, H]
    conf = raw[:, 2].astype(np.float64).copy(); valid = have.copy(); changed = np.zeros(17, bool)
    for j in BASE:
        if new[j]:
            px[j] = uv[SMPL_OF[j]]; changed[j] = True
    px, valid, conf = recompute_derived(px, valid, conf, changed)
    if not np.array_equal(valid, have):
        raise ValueError("STOP: oracle reference changed validity")
    out = raw.copy()                                   # validity and confidence stay exactly R11's
    out[changed, :2] = px[changed]/[W, H]
    return geometry_in_crop(out, valid, size, crop_box(out, valid, size))


def privileged_bench_row(x, v, xb, vb, size):
    """PRIVILEGED (benchmark_detector_observation): M-missing base joints filled with benchmark 2D where it is valid."""
    from framepose.crops import crop_box, geometry_in_crop
    W, H = float(size[0]), float(size[1])
    fill = np.zeros(17, bool)
    fill[BASE] = (~v[BASE]) & vb[BASE]
    if not fill.any():
        return None
    px = x[:, :2].astype(np.float64)*[W, H]; conf = x[:, 2].astype(np.float64).copy(); valid = v.copy()
    px[fill] = xb[fill, :2].astype(np.float64)*[W, H]; conf[fill] = xb[fill, 2]; valid[fill] = True
    changed = fill.copy()
    px, valid, conf = recompute_derived(px, valid, conf, changed)
    out = x.astype(np.float32).copy()
    ch = changed
    out[ch, :2] = np.where(valid[ch, None], px[ch]/[W, H], 0.); out[ch, 2] = np.where(valid[ch], conf[ch], 0.)
    return geometry_in_crop(out, valid, size, crop_box(out, valid, size))


# ------------------------------------------------------------------ raw M1 landmark provenance
def raw_landmarks(w102, bank, positions, obs):
    """Per population row: the retained selected M1 landmarks (16,4) [x, y, conf, visible] or NaN (no subject)."""
    from w101_reference_estimator import to_input
    observe = Path(w102)/"results/observe"
    if sha(observe/"manifest.json") != OBSERVE_MANIFEST_SHA:
        raise RuntimeError("STOP: raw M1 source manifest identity failed")
    cache, out = {}, np.full((len(positions), 16, 4), np.nan)
    audit = {"rows": len(positions), "rows_reproducing_M_input": 0, "videos": 0}
    for i, p in enumerate(positions):
        s = bank.samples[int(p)]
        video = video_of(s.sequence_id)
        if video not in cache:
            meta = json.loads((observe/(video+".json")).read_text())
            if sha(observe/(video+".npz")) != meta["npz_sha256"]:
                raise RuntimeError("STOP: raw M1 archive identity failed")
            z = np.load(observe/(video+".npz"), allow_pickle=False)
            cache[video] = (z["landmarks"], {int(f): k for k, f in enumerate(z["frame_index"])}, z["timestamp"], meta)
        lm_all, lookup, ts, meta = cache[video]
        k = lookup[s.frame_index]
        if meta["images"][k] != "3dpw/"+s.image_reference.relative_path or not np.isclose(ts[k], s.timestamp, atol=1e-12):
            raise RuntimeError("STOP: raw landmark image/timestamp alignment")
        out[i] = lm_all[k]
        lm = out[i] if np.isfinite(out[i][:, 0]).any() else None
        x, v = to_input(lm, s.image_size)
        if not (np.array_equal(x, obs["input_2d_M"][i]) and np.array_equal(v, obs["input_valid_M"][i])):
            raise RuntimeError("STOP: retained landmarks do not reproduce the M_Main input")
        audit["rows_reproducing_M_input"] += 1
    audit["videos"] = len(cache)
    return out, audit


def load_w103_labels(w103, sample_ids):
    """Frozen W103 per-row labels (quality_M, posture, same_person). S0 error columns are NOT read here."""
    path = Path(w103)/W103_PER_ROW
    if sha(path) != W103_PER_ROW_SHA:
        raise RuntimeError("STOP: W103 per-row artifact identity failed")
    z = np.load(path, allow_pickle=False)
    if not np.array_equal(z["sample_id"], sample_ids):
        raise RuntimeError("STOP: W103 row order differs")
    return {k: z[k] for k in ("quality_M", "same_person", "posture", "split", "video")}


# ------------------------------------------------------------------ stage: verify
def cmd_verify(args):
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=False)
    w102, w103 = Path(args.w102), Path(args.w103)
    report = {"schema": "w104_verify_v1", "checks": {}}
    if sha(w102/"evidence_manifest.json") != W102_MANIFEST_SHA:
        raise RuntimeError("STOP: W102 manifest identity failed")
    manifest = json.loads((w102/"evidence_manifest.json").read_text())["files"]
    observe = [k for k in manifest if k.startswith("results/observe/")]
    files = {rel: sha(w102/rel) == manifest[rel]["sha256"] for rel in W102_FILES + observe}
    report["checks"]["w102_files_match_manifest"] = {"files": len(files), "all": all(files.values()),
                                                     "raw_M1_observe_files": len(observe)}
    if not all(files.values()):
        raise RuntimeError("STOP: a frozen W102 artifact differs from its manifest")
    if sha(w103/"evidence_manifest.json") != W103_MANIFEST_SHA:
        raise RuntimeError("STOP: W103 manifest identity failed")
    m103 = json.loads((w103/"evidence_manifest.json").read_text())
    f103 = {rel: sha(w103/rel) == e["sha256"] for rel, e in m103["files"].items()}
    report["checks"]["w103_files_match_manifest"] = {"files": len(f103), "all": all(f103.values()),
                                                     "declaration_sha256": m103["declaration_sha256"]}
    if not all(f103.values()) or m103["declaration_sha256"] != W103_DECLARATION_SHA:
        raise RuntimeError("STOP: a frozen W103 artifact differs from its manifest")
    report["checks"]["checkpoint_sha256"] = sha(Path(args.w101_data)/"s0_checkpoint.pt")
    if report["checks"]["checkpoint_sha256"] != S0_CHECKPOINT_SHA:
        raise RuntimeError("STOP: S0 checkpoint identity failed")
    bank, obs, side, positions = load_context(w102, args.w101_data)
    report["checks"]["bank_content_digest"] = bank.content_digest()
    s0 = json.loads((w102/"results/s0_sources/s0_report.json").read_text())
    if s0["environment_fingerprint"] != ENVIRONMENT_FINGERPRINT:
        raise RuntimeError("STOP: W102 S0 environment record changed")
    geometry = geometry_all(bank, obs)
    geo = {c: hashlib.sha256(geometry[a].tobytes()).hexdigest() == s0["conditions"][c]["geometry_sha256"]
           for a, c in CONDITIONS.items()}
    report["checks"]["w102_geometry_reproduced"] = geo
    if not all(geo.values()):
        raise RuntimeError("STOP: model-visible input cannot be reproduced")
    lm, audit = raw_landmarks(w102, bank, positions, obs)
    report["checks"]["raw_landmark_provenance"] = audit
    same = side["identity"] == "same_person"
    vm = obs["input_valid_M"]
    finite = np.isfinite(lm[..., 0])
    present = finite.any(1)
    # canonical row -> token index map (head = nose row, thorax uses neck row; W100 SOURCE_OF)
    from eval_real_observation_estimator import CANON, SOURCE_OF
    src = [CANON.index(SOURCE_OF[n]) for n in JOINTS]
    tok_finite = finite[:, src]
    tok_conf = lm[:, src, 2]
    inv = same[:, None] & ~vm & present[:, None]
    report["raw_retention"] = {
        "same_person_rows": int(same.sum()), "same_person_rows_with_landmarks": int((same & present).sum()),
        "same_person_invalid_tokens": int((same[:, None] & ~vm).sum()),
        "invalid_tokens_with_finite_raw_coordinate": int((inv & tok_finite).sum()),
        "invalid_tokens_missing_raw_coordinate": int((inv & ~tok_finite).sum()),
        "invalid_limb_tokens": int(inv[:, DIRECT].sum()),
        "invalid_limb_tokens_finite": int((inv & tok_finite)[:, DIRECT].sum()),
        "max_conf_of_invalid_finite_token": float(np.nanmax(np.where(inv & tok_finite, tok_conf, np.nan))),
        "min_conf_of_valid_token": float(np.nanmin(np.where(same[:, None] & vm, tok_conf, np.nan))),
        "rule": "production visibility = confidence >= 0.3; retained raw landmarks keep x, y, confidence for every keypoint",
    }
    labels = load_w103_labels(w103, obs["sample_id"])
    report["checks"]["w103_labels_same_person_equal_w102_identity"] = bool(np.array_equal(labels["same_person"], same))
    dump(out/"verify_report.json", report)
    print(json.dumps(report, default=str)[:3000])


# ------------------------------------------------------------------ stage: declare (2D only)
DECLARATION_TEXT = """
W104 missing-joint intervention contract. Fixed before any W104 S0 outcome is computed. Construction reads only M1 2D
validity masks, the retained raw M1 landmarks and the frozen W103 observation-quality label; no GT array, no S0 error.

1 POPULATIONS. Same-person M rows (W102 unique association, 6,988; VAL and TEST separate). Factorial substrate: same-person
  rows whose M input has all 12 limb joints valid; PRIMARY = W103 WELL-observed (all 12 within 0.10 of the GT limb-box
  diagonal; label frozen in W103, GT-2D-referenced and S0-independent), SECONDARY = complete-but-mislocalized.
  Natural-missingness population = same-person rows with >= 1 invalid limb joint (PARTIAL). Exhaustive enumeration, no
  sampling (seed 1337 is used only for the video bootstrap). TEST is an architecture-diagnostic population, not a holdout.
2 PATTERNS. Singles: each of the 12 limb joints. Same-limb: knee+ankle, hip+knee+ankle, elbow+wrist,
  shoulder+elbow+wrist per side. Bilateral: ankles, knees+ankles, wrists, elbows+wrists. Natural: any of the 12 most
  frequent exact natural missing-limb sets (by M validity masks) not already listed. Count-matched unrelated controls:
  same-side leg<->arm mirror (hip<->shoulder, knee<->elbow, ankle<->wrist), declared for pure-leg or pure-arm patterns
  and added if absent; mixed-limb patterns have no control. Removal applies the production derived-token rule
  (hip -> pelvis, spine; shoulder -> neck, thorax, spine). Head is never removed.
3 FACTORIAL (per substrate row and pattern). F00 original validity + original crop (= W102 M_Main tensor, verified
  byte-equal); F10 intervention mask + original crop; F01 original validity + crop recomputed from the intervention
  validity; F11 intervention mask + production recrop. Coordinates/confidences of non-removed tokens are never changed.
  Crop strata: invariant (F01 == F00 bytes), small (side ratio within 1% and centre shift < 1% of side), large.
  F01 tokens outside [-1, 1] are labelled out-of-range and retained in the denominator. Patterns with a removed joint
  already invalid, or leaving < 2 valid tokens, are refused and counted.
4 SOURCE-ONLY RESTORATION (natural rows, evaluation-only). R11: every M-invalid token whose raw M1 landmark is finite is
  made valid with its raw coordinate and raw confidence (derived tokens by the production midpoint rule), production
  recrop. R10: same tokens, original crop. R01: original validity, restored crop. Rc: R11 with restored-token
  confidence fixed to 0.3 (the unchanged visibility boundary). No confidence threshold is searched; no joint is chosen
  by GT or by predicted 3D. M_Main itself is unchanged.
5 PRIVILEGED REFERENCES (never GT-free candidates; never pooled with real-source rates). P_oracle (oracle_geometry):
  R11 validity and confidence, restored base-joint coordinates replaced by the GT projection of the SMPL joint
  (COCO/SMPL semantic mismatch acknowledged); only rows whose GT root lies in front of the camera. P_bench
  (benchmark_detector_observation): M-missing base joints filled with benchmark 2D coordinates and confidences where
  benchmark-valid; derived tokens recomputed; production recrop.
6 OUTCOMES (fixed W102/W103 definitions). Root-relative MPJPE, per-joint error, SEVERE > 200 mm and tail quantiles,
  wrong knee branch (both flexions >= 20 deg, plane normals > 90 deg), segment direction error > 45 deg, limb depth-sign
  error (|dy| >= 0.25 |segment|), torso forward flip > 90 deg. LIMB FAILURE (affected limb) = any of its two segments
  > 45 deg, or for a leg a wrong knee branch. INDUCED failure for (row, pattern, condition) = newly SEVERE, or newly a limb
  failure in an affected limb, relative to F00. RECOVERED = the reverse. Temporal events on original bank neighbours with
  both rows in the substrate: S0 segment step > 45 deg while GT step < 15 deg; S0 torso-forward step > 90 deg while GT
  step < 45 deg. Paired changes per row; counterfactual copies of a frame are averaged per row before resampling.
7 STATISTICS. Video-clustered bootstrap (2,000 replicates, seed 1337), frame-weighted; no CI below 2 videos. Mask main
  effect = ((F10-F00)+(F11-F01))/2; crop main effect = ((F01-F00)+(F11-F10))/2; interaction = F11-F10-F01+F00.
  Pooled = per-row mean over the TARGET patterns (all non-control patterns), then video bootstrap.
8 GATES (fixed). REPRODUCTION: F00 tensors byte-equal to W102 M_Main geometry; stored-layout S0 forward byte-equal to
  W102 M_Main predictions; counterfactual-layout F00 within 0.1 mm of them.
  CAUSAL: (a) pooled induced rate under F10, target minus count-matched mirrored control evaluated on the target limbs,
  CI lower > 0; (b) pooled induced rate F10 minus F01, CI lower > 0; (c) pooled MPJPE(F11 - F00) CI lower >= +10 mm
  (W100 material scale). CROP_SECONDARY: |pooled crop main effect| <= 0.5 x pooled mask main effect and
  |pooled interaction| <= 0.5 x pooled mask main effect (MPJPE). EXPLAINED: over natural pattern classes with >= 30
  natural rows, natural-count-weighted F11 absolute failure rate (severe or affected-limb failure, primary substrate)
  >= 0.5 x the natural rows' M failure rate. BENEFIT: on PARTIAL rows with a restorable token, R11 - M MPJPE CI upper < 0
  and severe-rate change CI upper < 0. HARM: newly severe > 0.25 x recovered severe, or newly wrong knee > 0.25 x fixed
  knee, or p99 MPJPE(R11) > p99 MPJPE(M) on the same rows.
9 CASE. D if REPRODUCTION or provenance fails. Else if CAUSAL and CROP_SECONDARY and EXPLAINED: A if BENEFIT and not
  HARM; B if not BENEFIT; C if BENEFIT and HARM. Otherwise C. Temporal availability and privileged references are
  reported, never decisive. The reference-validity sensitivity (rows with GT root behind camera removed) recomputes
  every gate separately; the primary result is retained.
10 TEMPORAL AVAILABILITY (measurement only, no filling). For every naturally missing limb token: nearest earlier/later
  row of the same actor-sequence (original bank neighbour chain, up to 5 samples) that is same-person and has the joint
  valid in M; report offsets 1/2/3/5, bracketing, intervening non-same-person rows, missing-run length, GT-free 2D motion
  between bracketing observations; evaluation-only GT projection error of the nearest PAST observation and of the raw
  current-frame coordinate.
11 QUALITATIVE SELECTION (deterministic, ties by sample id then pattern order): (1) primary-substrate single leg
  pattern with largest MPJPE(F10-F00); (2) single pattern with largest MPJPE(F01-F00); (3) PARTIAL row with largest
  MPJPE improvement M -> R11; (4) PARTIAL row with largest MPJPE harm M -> R11.
12 REAL VIDEO (GT-free). Historical W102 clip: M reproduction, R11 on frames with an invalid token, and F11 of the
  single knee/ankle patterns and bilateral ankles on complete frames; report knee-normal flips and flexion changes vs
  the M prediction and S0 segment steps > 45 deg between consecutive frames. No rig render is required.
"""


def cmd_declare(args):
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=False)
    bank, obs, side, positions = load_context(args.w102, args.w101_data)
    labels = load_w103_labels(args.w103, obs["sample_id"])
    same = np.asarray(side["identity"]) == "same_person"
    vm = obs["input_valid_M"]
    complete = same & vm[:, DIRECT].all(1)
    natural = natural_sets(vm[same])
    pats = declared_patterns(natural)
    names = [pattern_name(k, j) for k, j in pats]
    controls = {pattern_name(k, j): (pattern_name(*next(p for p in pats if p[1] == mirror(j))) if mirror(j) else None)
                for k, j in pats if k != "control"}
    split = np.array([bank.samples[int(p)].split for p in positions])
    declaration = {
        "schema": "w104_declaration_v1", "text": DECLARATION_TEXT.strip(),
        "patterns": [{"name": n, "kind": k, "joints": list(j), "removed_with_derived": closure(j),
                      "affected_limbs": affected_limbs(j)} for n, (k, j) in zip(names, pats)],
        "controls": controls,
        "natural_missing_sets_top": [{"joints": [JOINTS[i] for i in j], "rows": c}
                                     for j, c in sorted(natural.items(), key=lambda kv: (-kv[1], kv[0]))[:NATURAL_TOP]],
        "populations": {
            "same_person": {s: int((same & (split == s)).sum()) for s in ("validation", "test")},
            "substrate_primary_well": {s: int((complete & (labels["quality_M"] == "well") & (split == s)).sum())
                                       for s in ("validation", "test")},
            "substrate_secondary_mislocalized": {s: int((complete & (labels["quality_M"] == "mislocalized") & (split == s)).sum())
                                                 for s in ("validation", "test")},
            "natural_partial": {s: int((same & ~vm[:, DIRECT].all(1) & (split == s)).sum()) for s in ("validation", "test")}},
        "constants": {"restored_fixed_conf": RESTORED_FIXED_CONF, "f00_budget_mm": F00_BUDGET_MM, "natural_top": NATURAL_TOP,
                      "temporal_max_offset": TEMPORAL_MAX_OFFSET, "temporal_report_offsets": list(TEMPORAL_REPORT_OFFSETS),
                      "natural_min_rows": NATURAL_MIN_ROWS, "bootstrap": {"replicates": 2000, "seed": 1337}},
        "regimes": REGIME,
        "inputs": {"observations_sha256": OBSERVATIONS_SHA, "side_sha256": SIDE_SHA, "w103_per_row_sha256": W103_PER_ROW_SHA,
                   "w103_columns_read": ["sample_id", "quality_M", "same_person", "posture", "split", "video"],
                   "observe_manifest_sha256": OBSERVE_MANIFEST_SHA},
        "GT_arrays_read": False, "S0_outputs_read": False}
    dump(out/"declaration.json", declaration)
    print(json.dumps({k: declaration[k] for k in ("populations", "controls")}, indent=1))
    print("patterns", len(names), names)
    print("declaration sha256", sha(out/"declaration.json"))


def require_declaration(args):
    if not args.declaration or sha(args.declaration) != args.declaration_sha:
        raise RuntimeError("STOP: this stage requires the hashed W104 declaration")
    return json.loads(Path(args.declaration).read_text())


# ------------------------------------------------------------------ stage: infer
def build_factorial(decl, bank, obs, side, positions, labels):
    """All factorial tensors. Only M 2D inputs and validity masks are read."""
    same = np.asarray(side["identity"]) == "same_person"
    vm, xm = obs["input_valid_M"], obs["input_2d_M"]
    sub = np.flatnonzero(same & vm[:, DIRECT].all(1) & np.isin(labels["quality_M"], ["well", "mislocalized"]))
    pats = [tuple(p["joints"]) for p in decl["patterns"]]
    G = {c: [] for c in ("F10", "F01", "F11")}
    F00 = np.zeros((len(sub), 17, 4), np.float32)
    rec = {"row": [], "pattern": [], "side_ratio": [], "centre_shift": [], "f01_out_of_range": [], "crop_invariant": []}
    refused = Counter()
    for n, i in enumerate(sub):
        size = bank.samples[int(positions[i])].image_size
        F00[n] = factorial_row(xm[i], vm[i], size, ())[0]["F00"]
        for p, joints in enumerate(pats):
            try:
                t, c0, c1 = factorial_row(xm[i], vm[i], size, joints)
            except ValueError as e:
                refused[(decl["patterns"][p]["name"], str(e))] += 1
                continue
            for c in G:
                G[c].append(t[c])
            r, s = crop_shift(c0, c1)
            valid = t["F01"][:, 3] > .5
            rec["row"].append(i); rec["pattern"].append(p); rec["side_ratio"].append(r); rec["centre_shift"].append(s)
            rec["f01_out_of_range"].append(bool((np.abs(t["F01"][valid, :2]) > 1).any()))
            rec["crop_invariant"].append(bool(t["F01"].tobytes() == t["F00"].tobytes()))
    return sub, F00, {c: np.stack(v) for c, v in G.items()}, {k: np.asarray(v) for k, v in rec.items()}, refused


def build_restoration(bank, obs, side, positions, lm):
    """Source-only restoration tensors on same-person rows with a restorable token. No GT is read."""
    same = np.asarray(side["identity"]) == "same_person"
    vm, xm = obs["input_valid_M"], obs["input_2d_M"]
    rows, G, new_tokens, rec = [], {c: [] for c in RESTORE}, [], {"side_ratio": [], "centre_shift": []}
    noop = 0
    for i in np.flatnonzero(same):
        size = bank.samples[int(positions[i])].image_size
        raw, have, new = restore_row(lm[i], size, xm[i], vm[i])
        if not new.any():
            noop += 1
            continue
        t, c0, c1 = restoration_inputs(xm[i], vm[i], raw, have, size)
        rows.append(i); new_tokens.append(new)
        for c in RESTORE:
            G[c].append(t[c])
        r, s = crop_shift(c0, c1); rec["side_ratio"].append(r); rec["centre_shift"].append(s)
    return (np.asarray(rows), {c: np.stack(v) for c, v in G.items()}, np.stack(new_tokens),
            {k: np.asarray(v) for k, v in rec.items()}, noop)


def build_privileged(bank, obs, positions, lm, rows, raw_refs):
    """PRIVILEGED references for the restoration rows. Reads GT projections (oracle) and benchmark 2D (bench)."""
    vm, xm = obs["input_valid_M"], obs["input_2d_M"]
    oracle, orows, bench, brows = [], [], [], []
    from w101_reference_estimator import to_input
    for i in rows:
        p = int(positions[i]); size = bank.samples[p].image_size
        raw, have = to_input(lm[i], size, raw=True)
        new = have & ~vm[i]
        if raw_refs["root"][i][1] > 0:
            oracle.append(privileged_oracle_row(raw, have, new, size, raw_refs["uv"][i])); orows.append(i)
        b = privileged_bench_row(xm[i], vm[i], bank.arrays["input_2d"][p], bank.arrays["input_valid"][p], size)
        if b is not None:
            bench.append(b); brows.append(i)
    return {"P_oracle": (np.asarray(orows), np.stack(oracle)), "P_bench": (np.asarray(brows), np.stack(bench))}


def s0_forward(model, torch, geometry):
    from framepose.signs import neutral_sign_states
    from framepose.train import predict
    out = predict(model, torch, geometry, None, np.arange(len(geometry)), torch.device("cuda"), batch_size=512,
                  amp_enabled=False, signs=neutral_sign_states(len(geometry))).astype(np.float32)
    if not np.isfinite(out).all():
        raise RuntimeError("STOP: non-finite S0 output")
    return out


def cmd_infer(args):
    import time
    import torch
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    from framepose.signs import neutral_sign_states
    from framepose.train import load_checkpoint, predict
    from run_w102_same_environment_s0 import execution_identity
    from w103_observability import load_predictions, raw_references
    decl = require_declaration(args)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    bank, obs, side, positions = load_context(args.w102, args.w101_data)
    labels = load_w103_labels(args.w103, obs["sample_id"])
    lm, audit = raw_landmarks(args.w102, bank, positions, obs)
    os.chdir(ROOT)
    environment, fingerprint = execution_identity(torch)
    lf = {p: hashlib.sha256((ROOT/p).read_bytes().replace(b"\r\n", b"\n")).hexdigest() for p in environment["model_source_sha256"]}
    s0 = json.loads((Path(args.w102)/"results/s0_sources/s0_report.json").read_text())
    report = {"schema": "w104_infer_v1", "declaration_sha256": args.declaration_sha, "environment": environment,
              "environment_fingerprint": fingerprint, "w102_environment_fingerprint": ENVIRONMENT_FINGERPRINT,
              "fields_differing_from_w102": sorted(k for k in set(environment) | set(s0["environment"])
                                                   if environment.get(k) != s0["environment"].get(k)),
              "model_source_sha256_lf_normalized": lf,
              "lf_normalized_sources_equal_w102": lf == s0["environment"]["model_source_sha256"],
              "raw_landmark_provenance": audit}
    if sha(Path(args.w101_data)/"s0_checkpoint.pt") != S0_CHECKPOINT_SHA:
        raise RuntimeError("STOP: S0 checkpoint identity failed")
    model, _ = load_checkpoint(Path(args.w101_data)/"s0_checkpoint.pt", device="cuda")
    geometry = geometry_all(bank, obs)
    stored, _ = load_predictions(args.w102, bank, positions)
    # ---- reproduction gate: stored layout, byte-equal
    rep = {}
    signs = neutral_sign_states(len(bank))
    for split in ("validation", "test"):
        pos = bank.indices(split)
        p = predict(model, torch, geometry["M"], None, pos, torch.device("cuda"), batch_size=512, amp_enabled=False,
                    signs=signs).astype(np.float32)
        ref = np.load(Path(args.w102)/f"results/s0_sources/prediction_M_Main_{split}.npy")
        rep[split] = {"rows": len(pos), "byte_equal_to_w102_M_Main": bool(p.tobytes() == ref.tobytes())}
    report["reproduction_stored_layout"] = rep
    # ---- factorial (no GT)
    sub, F00, F, rec, refused = build_factorial(decl, bank, obs, side, positions, labels)
    Mgeo = geometry["M"][positions][sub]
    report["factorial"] = {"substrate_rows": len(sub), "row_pattern_pairs": len(rec["row"]),
                           "refused": {f"{k[0]} | {k[1]}": v for k, v in refused.items()},
                           "F00_byte_equal_to_M_Main_geometry": bool(F00.tobytes() == Mgeo.tobytes()),
                           "crop_invariant_pairs": int(rec["crop_invariant"].sum()),
                           "f01_out_of_range_pairs": int(rec["f01_out_of_range"].sum())}
    pred = {"F00": s0_forward(model, torch, F00)}
    for c in F:
        pred[c] = s0_forward(model, torch, F[c])
    stored_rows = stored["M"][sub]
    report["reproduction_counterfactual_layout"] = {
        "F00_max_abs_mm_vs_stored": float(1000*np.abs(pred["F00"]-stored_rows).max()),
        "budget_mm": F00_BUDGET_MM}
    np.savez_compressed(out/"factorial.npz", substrate=sub, rec_row=rec["row"], rec_pattern=rec["pattern"],
                        side_ratio=rec["side_ratio"], centre_shift=rec["centre_shift"],
                        f01_out_of_range=rec["f01_out_of_range"], crop_invariant=rec["crop_invariant"],
                        pred_F00=pred["F00"], pred_F10=pred["F10"], pred_F01=pred["F01"], pred_F11=pred["F11"])
    np.savez_compressed(out/"factorial_inputs.npz", F00=F00, **F)
    report["factorial"]["inputs_sha256"] = sha(out/"factorial_inputs.npz")
    del F
    # ---- source-only restoration (no GT)
    rows, R, new_tokens, rrec, noop = build_restoration(bank, obs, side, positions, lm)
    rpred = {c: s0_forward(model, torch, R[c]) for c in RESTORE}
    report["restoration"] = {"rows": len(rows), "same_person_rows_without_restorable_token (no-op)": noop,
                             "restored_tokens": int(new_tokens.sum())}
    np.savez_compressed(out/"restoration_inputs.npz", rows=rows, new_tokens=new_tokens, **R)
    report["restoration"]["inputs_sha256"] = sha(out/"restoration_inputs.npz")
    # ---- privileged references (GT projection / benchmark 2D) -- separate files, separate labels
    raw_refs = raw_references(Path(args.w102)/"data/raw", bank, positions)
    P = build_privileged(bank, obs, positions, lm, rows, raw_refs)
    ppred = {c: s0_forward(model, torch, P[c][1]) for c in PRIVILEGED}
    report["privileged"] = {c: {"rows": len(P[c][0]), "regime": REGIME[c]} for c in PRIVILEGED}
    np.savez_compressed(out/"restoration.npz", rows=rows, new_tokens=new_tokens, side_ratio=rrec["side_ratio"],
                        centre_shift=rrec["centre_shift"], **{"pred_"+c: v for c, v in rpred.items()})
    np.savez_compressed(out/"privileged.npz", **{f"rows_{c}": P[c][0] for c in PRIVILEGED},
                        **{f"pred_{c}": ppred[c] for c in PRIVILEGED}, **{f"input_{c}": P[c][1] for c in PRIVILEGED})
    report["gate_reproduction_pass"] = bool(all(v["byte_equal_to_w102_M_Main"] for v in rep.values())
                                            and report["factorial"]["F00_byte_equal_to_M_Main_geometry"]
                                            and report["reproduction_counterfactual_layout"]["F00_max_abs_mm_vs_stored"] <= F00_BUDGET_MM)
    report["seconds"] = time.time()-t0
    report["gpu_max_memory_allocated_mb"] = torch.cuda.max_memory_allocated()/2**20
    report["outputs_sha256"] = {f: sha(out/f) for f in ("factorial.npz", "restoration.npz", "privileged.npz")}
    dump(out/"infer_report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("environment",)}, default=str)[:4000])
    if not report["gate_reproduction_pass"]:
        raise RuntimeError("STOP: F00 / M_Main reproduction failed (CASE D)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("verify", "declare", "infer", "analyze", "video", "posthoc"))
    ap.add_argument("--w102", required=True)
    ap.add_argument("--w101-data", required=True)
    ap.add_argument("--w103", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--declaration")
    ap.add_argument("--declaration-sha")
    ap.add_argument("--infer-dir")
    ap.add_argument("--video-dir")
    args = ap.parse_args()
    if args.stage in ("analyze", "video", "posthoc"):
        from w104_analysis import cmd_analyze, cmd_posthoc, cmd_video
        {"analyze": cmd_analyze, "video": cmd_video, "posthoc": cmd_posthoc}[args.stage](args)
    else:
        {"verify": cmd_verify, "declare": cmd_declare, "infer": cmd_infer}[args.stage](args)


if __name__ == "__main__":
    main()
