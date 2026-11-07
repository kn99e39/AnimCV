#!/usr/bin/env python3
"""W121 scratch helpers: independent real-video acquisition screen, authenticity / duplicate fingerprints, frozen
perception coverage, pre-solver segment / hold-window selection and the frozen CASE rule (NON-PRODUCTION, evidence only).

Nothing in this file solves, evaluates or tunes the W116-W120 mechanism.  Every rule below is declared before any W121
solver candidate exists and is a pure function of source metadata, decoded frames, frozen M1/S0 outputs or the
Blender-evaluated UNEDITED parent (never of a candidate result).
"""
from __future__ import annotations

import hashlib
import json
import math
import re

import numpy as np

# ------------------------------------------------------------------------------------- acquisition protocol ---
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "AnimCV-research-script/0.1 (offline evaluation)"
QUERIES = ["jumping jacks", "exercise demonstration", "boxing training", "hand wave", "aerobics exercise", "juggling",
           "kettlebell", "calisthenics", "shadow boxing", "tai chi"]
RESULTS_PER_QUERY = 30
ACTIVITY_RE = r"(jumping ?jack|exercise demonstration|boxing training|hand ?wave|aerobic|juggl|kettlebell|calisthenic|shadow ?box|tai ?chi|taiji)"
TIMESCALE_RE = r"(slow ?-?motion|slowmo|time ?-?lapse|hyperlapse|speed ?ramp)"
FREE_LICENSE_RE = r"^(CC0|Public domain|CC BY(-SA)? [0-9.]+)$"
VIDEO_MIME = ("video/webm", "application/ogg", "video/ogg")
DURATION_S = (4.0, 60.0)
MIN_SHORT_SIDE_PX = 360
MAX_DOWNLOAD_BYTES = 150 * 2 ** 20

# historical (non-independent) sources: never admissible as new evidence
EXCLUDED_SHA256 = {"0dda8a5f16c67992330a1a8534cf0bfefeb5e0f2aec438577a9e1dbace773275":
                   "W102/W105/W116-W120 source = examples/production_demo/source.mp4"}

# ------------------------------------------------------------------------ duplicate / derivative fingerprints ---
DHASH_SIZE = 8                    # 64-bit difference hash per decoded frame
DUP_HAMMING = 6                   # frame match if Hamming distance <= 6 / 64 bits
DUP_RUN_FRAMES = 10               # duplicate if >= 10 consecutive matched frames in temporal order (any time offset/scale)
CUT_MAD = 30.0
CADENCE_TOL_MS = 1.0              # container presentation-time step must equal 1000/fps within 1 ms (else the frozen
                                  # diagnostic's timestamp = index / fps would misstate source time; never repaired here)                    # shot cut: mean |difference| of 64x36 grayscale thumbnails > 30 (0..255)

# ------------------------------------------------------------------------------- perception suitability ---
LEFT_ARM = (11, 12, 13)           # S0/H36M-17 left shoulder, elbow, wrist (W105 s0_left_arm uses j[11..13])
SEG_MIN_S = 3.0
SEG_MIN_FRAMES = 90
SEG_MAX_FRAMES = 240              # bounded Action length per source (the W102 clip had 199)
ARM_VALID_MIN = 0.80              # left shoulder+elbow+wrist all valid on >= 80 % of segment frames
ARM_ACTIVITY_MIN = 0.50           # observed 2D left-wrist extent / median torso length within the segment
MAX_SOURCES = 3


def license_ok(s):
    return bool(s) and re.match(FREE_LICENSE_RE, s.strip()) is not None


def screen_metadata(c):
    """Declared metadata screen of one Commons candidate.  Returns a list of rejection reason codes ([] = pass)."""
    why = []
    if not license_ok(c.get("license")):
        why.append("R1_LICENSE_NOT_FREE_OR_UNKNOWN")
    if c.get("mime") not in VIDEO_MIME:
        why.append("R2_NOT_DECODABLE_VIDEO_CONTAINER")
    d = c.get("duration") or 0.0
    if not (DURATION_S[0] <= d <= DURATION_S[1]):
        why.append("R3_DURATION_OUTSIDE_4_60_S")
    if min(c.get("w") or 0, c.get("h") or 0) < MIN_SHORT_SIDE_PX:
        why.append("R4_RESOLUTION_BELOW_360")
    text = (c.get("title", "") + " " + re.sub("<[^>]+>", " ", c.get("desc") or "")).lower()
    if re.search(TIMESCALE_RE, text):
        why.append("R5_TIME_SCALE_NOT_REAL_TIME")
    if not re.search(ACTIVITY_RE, c.get("title", "").lower()):
        why.append("R6_TITLE_NOT_DECLARED_HUMAN_ACTIVITY")
    if (c.get("bytes") or 0) > MAX_DOWNLOAD_BYTES:
        why.append("R8_FILE_TOO_LARGE")
    return why


def dhash(gray):
    """64-bit difference hash of a grayscale frame (cv2 resize to 9x8)."""
    import cv2
    s = cv2.resize(gray, (DHASH_SIZE + 1, DHASH_SIZE), interpolation=cv2.INTER_AREA)
    bits = (s[:, 1:] > s[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a, b):
    return bin(a ^ b).count("1")


def longest_matched_run(ha, hb):
    """Longest run of consecutive frames of A each matching some frame of B with non-decreasing B index (re-timing,
    crop-free re-encode and time shifts are all caught; a crop changes the dHash and is caught by visual review)."""
    if not ha or not hb:
        return 0
    hb_arr = np.array(hb, dtype=np.uint64)
    match = []
    for h in ha:
        x = np.bitwise_xor(hb_arr, np.uint64(h))
        d = np.array([bin(int(v)).count("1") for v in x])
        match.append(set(np.flatnonzero(d <= DUP_HAMMING).tolist()))
    best = cur = 0
    last = None
    for m in match:
        nxt = [j for j in m if last is None or j >= last]
        if nxt:
            cur += 1
            last = min(nxt)
        else:
            cur, last = (1, min(m)) if m else (0, None)
        best = max(best, cur)
    return best


def shot_cuts(thumbs):
    """Indices i where a cut lies between frames i-1 and i."""
    return [i for i in range(1, len(thumbs)) if float(np.mean(np.abs(thumbs[i].astype(float) - thumbs[i - 1].astype(float)))) > CUT_MAD]


def segment_select(present, arm_valid, wrist_xy, torso_len, cuts, fps):
    """Declared pre-solver segment rule.  present[i]: M1 subject landmarks finite; arm_valid[i]: left shoulder, elbow and
    wrist all valid (frozen 0.3/0.5 policies); wrist_xy[i]: observed 2D left wrist (px) or nan; torso_len[i] px or nan.
    Candidate runs = maximal subject-present runs inside one shot.  Long runs are cut to the SEG_MAX_FRAMES window with the
    highest arm-valid fraction (earliest on ties).  Returns (all_runs, chosen or None)."""
    n = len(present)
    bounds = [0] + sorted(cuts) + [n]
    runs = []
    for s0, s1 in zip(bounds[:-1], bounds[1:]):
        i = s0
        while i < s1:
            if not present[i]:
                i += 1
                continue
            j = i
            while j < s1 and present[j]:
                j += 1
            runs.append([i, j - 1])
            i = j
    recs = []
    for lo, hi in runs:
        L = hi - lo + 1
        if L > SEG_MAX_FRAMES:
            av = np.asarray(arm_valid[lo:hi + 1], float)
            cs = np.concatenate([[0], np.cumsum(av)])
            sums = cs[SEG_MAX_FRAMES:] - cs[:-SEG_MAX_FRAMES]
            off = int(np.argmax(sums))
            wlo, whi = lo + off, lo + off + SEG_MAX_FRAMES - 1
        else:
            wlo, whi = lo, hi
        idx = list(range(wlo, whi + 1))
        frac = float(np.mean([arm_valid[i] for i in idx]))
        pts = np.array([wrist_xy[i] for i in idx if arm_valid[i]], float)
        tl = np.nanmedian([torso_len[i] for i in idx]) if np.isfinite([torso_len[i] for i in idx]).any() else float("nan")
        extent = float(np.max(np.linalg.norm(pts[:, None] - pts[None], axis=-1))) if len(pts) > 1 else 0.0
        act = extent / tl if tl and np.isfinite(tl) and tl > 0 else 0.0
        why = []
        if len(idx) < max(SEG_MIN_FRAMES, math.ceil(SEG_MIN_S * fps)):
            why.append("S4_SEGMENT_TOO_SHORT")
        if frac < ARM_VALID_MIN:
            why.append("S4_LEFT_ARM_VALID_FRACTION_BELOW_0.80")
        if act < ARM_ACTIVITY_MIN:
            why.append("S5_LEFT_WRIST_ACTIVITY_BELOW_0.50_TORSO")
        recs.append({"run": [lo, hi], "window": [wlo, whi], "frames": len(idx), "seconds": len(idx) / fps, "arm_valid_fraction": frac,
                     "wrist_extent_px": extent, "median_torso_px": tl, "activity_ratio": act, "reasons": why, "eligible": not why})
    elig = [r for r in recs if r["eligible"]]
    chosen = max(elig, key=lambda r: (r["frames"], r["arm_valid_fraction"], -r["window"][0])) if elig else None
    return recs, chosen


# ---------------------------------------------------------------------------- pre-solver hold-window design ---
HOLD_SECONDS = 0.30               # declared hold length (RH1 was 10 frames = 0.20 s at 50 fps; RH2 0.56 s)
HOLD_EDGE = 9                     # M_MAX + 1 frames kept free on each side so the full W118 budget is available
HOLD_EDITS = {"none": None, "RBpos": "SIDE_BEND_5DEG_POS", "RBneg": "SIDE_BEND_5DEG_NEG"}


def hold_windows(frames, wrist_world, fps):
    """Candidate holds: every window of L = round(HOLD_SECONDS * fps) frames whose start/end leave HOLD_EDGE frames to each
    clip end.  D(window) = max distance of the UNEDITED parent's Blender-evaluated wrist from its capture-frame position over
    the window (how far the hand travels if not held).  Selected (fixed, outcome independent): the minimum-D, median-D and
    maximum-D windows, greedily non-overlapping with >= 2*HOLD_EDGE frames between them (median = rank ceil(n/2) of the
    remaining candidates by D, earliest start on ties)."""
    L = max(2, int(round(HOLD_SECONDS * fps)))
    fr = list(frames)
    cands = []
    for s in range(HOLD_EDGE, len(fr) - HOLD_EDGE - L + 1):
        w = fr[s:s + L]
        p0 = np.array(wrist_world[w[0]], float)
        D = max(float(np.linalg.norm(np.array(wrist_world[k], float) - p0)) for k in w)
        cands.append({"capture": w[0], "frames": w, "D_m": D})
    chosen = {}

    def ok(c):
        return all(c["frames"][0] > o["frames"][-1] + 2 * HOLD_EDGE or c["frames"][-1] < o["frames"][0] - 2 * HOLD_EDGE
                   for o in chosen.values())
    byD = sorted(cands, key=lambda c: (c["D_m"], c["capture"]))
    for name, order in (("LOW", byD), ("HIGH", sorted(cands, key=lambda c: (-c["D_m"], c["capture"])))):
        c = next((c for c in order if ok(c)), None)
        if c is not None:
            chosen[name] = c
    rest = [c for c in byD if ok(c)]
    if rest:
        chosen["MID"] = rest[(len(rest) + 1) // 2 - 1]
    return {"L": L, "n_candidates": len(cands), "candidates": cands, "chosen": chosen}


# ------------------------------------------------------------------------------------------- frozen CASE rule ---
CASE_RULE = (
    "DATA_BLOCKED (no CASE) if no source passes acquisition, authentication, frozen-perception, rig-ingest and parent "
    "preflight.  Otherwise D if any validity gate V fails.  Otherwise C if any safety gate C fails.  Otherwise, per mechanism "
    "M in {GUIDE, FLOOR} with primary contrast (guide_at_F0 / floor_at_G0) and conditional contrast (guide_at_F1 / floor_at_G1), "
    "evaluated with the frozen W119 comparator compare_y on factorial-eligible NEW-SOURCE requests only: "
    "M_rep = the primary contrast meaningfully improves requests in >= 2 distinct predeclared hold windows (edit variants of one "
    "window count once) AND, when >= 2 sources were accepted, in every accepted source; "
    "M_clean = the primary contrast introduces no new bend flip or near-straight frame on any factorial-eligible new-source request; "
    "M_vis = the agent visual review marks at least one of M's primary improvements as visible with motion inspected across frames. "
    "A if some M has M_rep AND M_clean AND M_vis (labelled ONE-SOURCE TRANSFER when only one source was accepted).  "
    "B if not A and some M's primary or conditional contrast meaningfully improves at least one factorial-eligible new-source "
    "request.  C otherwise (no meaningful improvement by either mechanism on any eligible new-source request).  The W120 "
    "historical population is NOT part of any W121 denominator.")


def case_rule(V, Cg, mech, n_sources_accepted, data_blocked):
    """mech: {M: {"primary_improved": [(source, window, rid)], "conditional_improved": [...], "primary_new_defects": [rid...],
    "visible": bool}}."""
    if data_blocked:
        return {"case": "DATA_BLOCKED", "detail": "no independent source passed preflight"}
    if not all(V.values()):
        return {"case": "D", "failed_V": sorted(k for k, v in V.items() if not v)}
    if not all(Cg.values()):
        return {"case": "C", "failed_C": sorted(k for k, v in Cg.items() if not v)}
    out = {}
    for M, r in mech.items():
        wins = {(s, w) for s, w, _ in r["primary_improved"]}
        srcs = {s for s, _, _ in r["primary_improved"]}
        rep = len(wins) >= 2 and (n_sources_accepted < 2 or len(srcs) >= n_sources_accepted)
        clean = not r["primary_new_defects"]
        out[M] = {"M_rep": rep, "M_clean": clean, "M_vis": bool(r["visible"]) and bool(r["primary_improved"]),
                  "any": bool(r["primary_improved"] or r["conditional_improved"])}
    a = [M for M, x in out.items() if x["M_rep"] and x["M_clean"] and x["M_vis"]]
    if a:
        return {"case": "A", "mechanisms": out, "A_by": a, "scope": "ONE-SOURCE TRANSFER" if n_sources_accepted == 1 else "MULTI-SOURCE"}
    if any(x["any"] for x in out.values()):
        return {"case": "B", "mechanisms": out}
    return {"case": "C", "mechanisms": out, "reason": "no meaningful improvement by either mechanism"}


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
