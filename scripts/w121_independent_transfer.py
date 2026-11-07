#!/usr/bin/env python3
"""W121 scratch: independent real-video transfer of the frozen W116-W120 world-pin transition mechanism (evidence only).

Stages (each refuses to overwrite its output directory):
  acquire    declared Wikimedia Commons search + metadata screen; original bytes downloaded OUTSIDE the repository with
             SHA-256 / Commons SHA-1, licence, author, URL and access time
  inspect    decode every downloaded candidate (cv2/FFmpeg), per-frame decoded-BGR SHA-256, dHash fingerprints, shot cuts,
             duplicate / derivative screen against the W102/W105/W116-W120 source and between candidates; writes a W102
             diagnostic-compatible inspection.json per candidate
  perceive   the FROZEN W102 M1 (RTMDet+RTMPose, batch 1) + S0 video diagnostic, run UNCHANGED from the W102 code snapshot
             via a process-local launcher; then the declared segment / suitability rule and source selection
  rig        frozen W105 GT-free S0-only source-to-rig configuration (FIXED slot) for every selected source; adapter fidelity
             is proven by reproducing W105's historical real_clip FIXED values exactly from W102's stored predictions
  ... (later stages: preflight / declare / build / evaluate / assess / render / decide / manifest)

No video GT, no 3D GT, no training, no Production change.  Original video bytes stay under C:/Users/dna10/animcv-w121/data.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.error
import urllib.request

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import w121_independent_transfer_base as AB  # noqa: E402

BATCH = 121
INSTRUCTION_BLOB = "1d9f48f88b12b348b80ebb1b4a5ff15dfa403dfa"
STARTING_HEAD = "7a61b5fb05b885037f850a15eca28c358a6bf828"
BRANCH = "arch/single_frame_first"
BASE = Path("C:/Users/dna10/animcv-w121")
PY = Path("C:/Projects/AnimCV/.venv_build_full/Scripts/python.exe")
BLENDER = Path("C:/Program Files/Blender Foundation/Blender 5.1/blender.exe")
W102 = Path("C:/Users/dna10/animcv-w102")
W101_DATA = Path("C:/Users/dna10/animcv-w101/data")
W105 = Path("C:/Users/dna10/animcv-w105")
FROZEN_LAUNCHER = BASE / "setup/w121_frozen_launch.py"
FROZEN = {  # frozen W102 M1/S0 inputs (paths + identities verified in perceive)
    "weights": W102 / "data/models",
    "checkpoint": W101_DATA / "s0_checkpoint.pt",
    "gate_report": W102 / "results/m1_gate/gate_report.json",
    "lock": W102 / "results/baseline0/s0_report.json",
    "diagnostic_script": W102 / "code/scripts/w102_video_diagnostic.py",
}
HIST_CLIP = W102 / "data/historical_source.mp4"
HIST_CLIP_SHA = "0dda8a5f16c67992330a1a8534cf0bfefeb5e0f2aec438577a9e1dbace773275"
HIST_OBS = W102 / "results/historical_video/observations_predictions.npz"
HIST_OBS_SHA = "6784357f7c6145e4b60e2e9201e32accd3ae00001486283a1e76f0adae61e7bd"
W105_VIDEO_CASES = W105 / "results/build/video_cases.json"


# ------------------------------------------------------------------------------------------------ helpers ---
def clean(v):
    if isinstance(v, dict):
        return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    if isinstance(v, np.ndarray):
        return clean(v.tolist())
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.floating):
        return float(v) if np.isfinite(v) else None
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, np.integer):
        return int(v)
    return v


def dump(path, value):
    Path(path).write_text(json.dumps(clean(value), indent=1, sort_keys=True, allow_nan=False), encoding="utf-8")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha1(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def new_dir(p):
    p = Path(p)
    p.mkdir(parents=True, exist_ok=False)
    return p


def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def http_json(params):
    q = urllib.parse.urlencode(dict(params, format="json"))
    req = urllib.request.Request(AB.COMMONS_API + "?" + q, headers={"User-Agent": AB.USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as r:
        raw = r.read()
    return raw, json.loads(raw)


def safe(title):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", title.removeprefix("File:"))[:80]


# ------------------------------------------------------------------------------------------------ acquire ---
def candidate_of(page, query):
    ii = page["imageinfo"][0]
    em = ii.get("extmetadata", {})
    g = lambda k: (em.get(k) or {}).get("value")  # noqa: E731
    return {"title": page["title"], "pageid": page.get("pageid"), "query": query, "url": ii.get("url"),
            "descriptionurl": ii.get("descriptionurl"), "w": ii.get("width"), "h": ii.get("height"),
            "duration": ii.get("duration"), "bytes": ii.get("size"), "commons_sha1": ii.get("sha1"), "mime": ii.get("mime"),
            "license": g("LicenseShortName"), "license_url": g("LicenseUrl"), "usage_terms": g("UsageTerms"),
            "attribution_required": g("AttributionRequired"), "artist": re.sub("<[^>]+>", "", g("Artist") or "").strip(),
            "credit": re.sub("<[^>]+>", "", g("Credit") or "").strip(), "date_original": g("DateTimeOriginal"),
            "desc": re.sub("<[^>]+>", " ", g("ImageDescription") or "")[:600], "uploader": ii.get("user"),
            "upload_timestamp": ii.get("timestamp")}


def cmd_acquire(a):
    out = new_dir(a.out_dir)
    raw_dir = new_dir(out / "api_raw")
    orig = BASE / "data/originals"
    orig.mkdir(parents=True, exist_ok=True)
    rep = {"schema": "w121_acquire_v1", "accessed_utc": now(), "api": AB.COMMONS_API, "user_agent": AB.USER_AGENT,
           "protocol": {"queries": AB.QUERIES, "results_per_query": AB.RESULTS_PER_QUERY, "activity_re": AB.ACTIVITY_RE,
                        "timescale_re": AB.TIMESCALE_RE, "free_license_re": AB.FREE_LICENSE_RE, "mime": AB.VIDEO_MIME,
                        "duration_s": AB.DURATION_S, "min_short_side_px": AB.MIN_SHORT_SIDE_PX,
                        "max_download_bytes": AB.MAX_DOWNLOAD_BYTES,
                        "rule": "R1 free licence, R2 container, R3 4-60 s, R4 short side >= 360 px, R5 real-time (no slow motion / "
                                "time-lapse in title or description), R6 title names a declared human activity, R8 size cap; "
                                "metadata only, before any download or perception"},
           "queries": {}, "candidates": {}}
    for i, q in enumerate(AB.QUERIES):
        raw, d = http_json({"action": "query", "generator": "search", "gsrsearch": q + " filetype:video", "gsrnamespace": 6,
                            "gsrlimit": AB.RESULTS_PER_QUERY, "prop": "imageinfo",
                            "iiprop": "url|size|sha1|mime|extmetadata|timestamp|user"})
        p = raw_dir / f"q{i:02d}.json"
        p.write_bytes(raw)
        pages = sorted(d.get("query", {}).get("pages", {}).values(), key=lambda x: x.get("index", 0))
        rep["queries"][q] = {"raw_file": p.name, "raw_sha256": AB.sha256_bytes(raw), "titles": [x["title"] for x in pages]}
        for pg in pages:
            if "imageinfo" not in pg:
                continue
            t = pg["title"]
            if t in rep["candidates"]:
                rep["candidates"][t]["queries"].append(q)
                continue
            c = candidate_of(pg, q)
            c["queries"] = [q]
            c["metadata_reasons"] = AB.screen_metadata(c)
            c["metadata_pass"] = not c["metadata_reasons"]
            rep["candidates"][t] = c
        time.sleep(1.0)
    breaker = False
    for t, c in sorted(rep["candidates"].items()):
        if not c["metadata_pass"]:
            continue
        dst = orig / f"{c['pageid']}_{safe(t)}"
        t0 = time.time()
        attempts = []
        if dst.exists() and sha1(dst) == c["commons_sha1"]:
            attempts.append({"reused_complete_earlier_download": True})   # an earlier attempt's complete, Commons-SHA-1-verified bytes
        elif breaker:
            c["download"] = None
            c["download_failure"] = [{"not_attempted": "RATE_LIMIT_CIRCUIT_BREAKER (upload.wikimedia.org HTTP 429 persisted; no further "
                                                       "requests sent, to avoid disruptive access)"}]
            continue
        else:
            if dst.exists():
                raise RuntimeError(f"STOP: existing partial/different file {dst}")
            for k in range(2):
                req = urllib.request.Request(c["url"], headers={"User-Agent": AB.USER_AGENT})
                try:
                    with urllib.request.urlopen(req, timeout=300) as r, open(dst.with_suffix(dst.suffix + ".part"), "wb") as f:
                        while True:
                            b = r.read(1 << 20)
                            if not b:
                                break
                            f.write(b)
                    dst.with_suffix(dst.suffix + ".part").rename(dst)
                    attempts.append({"ok": True})
                    break
                except urllib.error.HTTPError as e:
                    wait = min(float(e.headers.get("Retry-After") or 0) or 60.0, 120.0)
                    attempts.append({"http_error": e.code, "wait_s": wait if k == 0 else 0})
                    if e.code != 429 or k == 1:
                        break
                    time.sleep(wait)
            if not dst.exists():
                c["download"] = None
                c["download_failure"] = attempts
                if any(x.get("http_error") == 429 for x in attempts):
                    breaker = True
                continue
            os.chmod(dst, 0o444)
        c["download"] = {"path": str(dst), "bytes": dst.stat().st_size, "sha256": sha(dst), "sha1": sha1(dst),
                         "sha1_matches_commons": sha1(dst) == c["commons_sha1"], "accessed_utc": now(), "seconds": time.time() - t0,
                         "attempts": attempts}
        c["excluded_historical"] = AB.EXCLUDED_SHA256.get(c["download"]["sha256"])
        if not attempts or not attempts[0].get("reused_complete_earlier_download"):
            time.sleep(5.0)
    n = len(rep["candidates"])
    rep["summary"] = {"screened": n, "metadata_pass": sum(c["metadata_pass"] for c in rep["candidates"].values()),
                      "downloaded": sum(1 for c in rep["candidates"].values() if c.get("download")),
                      "not_downloaded_rate_limited": sorted(t for t, c in rep["candidates"].items() if c["metadata_pass"] and not c.get("download")),
                      "rate_limit_circuit_breaker_tripped": breaker,
                      "reasons": {r: sum(r in c["metadata_reasons"] for c in rep["candidates"].values())
                                  for r in sorted({r for c in rep["candidates"].values() for r in c["metadata_reasons"]})}}
    dump(out / "acquire.json", rep)
    print(json.dumps(rep["summary"], indent=1))


# ------------------------------------------------------------------------------------------------ inspect ---
def decode(path):
    """Same decode as the frozen W102 diagnostic (cv2.VideoCapture, BGR uint8, timestamp = index / fps)."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    meta = {"fps": fps, "cap_frame_count": cap.get(cv2.CAP_PROP_FRAME_COUNT), "fourcc": int(cap.get(cv2.CAP_PROP_FOURCC)),
            "backend": cap.getBackendName() if cap.isOpened() else None}
    digests, hashes, thumbs, size = [], [], [], set()
    pos_ms = []
    while True:
        ok, im = cap.read()
        if not ok:
            break
        pos_ms.append(cap.get(cv2.CAP_PROP_POS_MSEC))
        size.add((im.shape[1], im.shape[0]))
        digests.append(hashlib.sha256(im.tobytes()).hexdigest())
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        hashes.append(AB.dhash(g))
        thumbs.append(cv2.resize(g, (64, 36), interpolation=cv2.INTER_AREA))
    cap.release()
    meta.update(frames=len(digests), sizes=sorted(size), pos_msec=pos_ms)
    return meta, digests, hashes, thumbs


def cmd_inspect(a):
    import cv2
    out = new_dir(a.out_dir)
    acq = load(a.acquire / "acquire.json")
    hist_meta, hist_dig, hist_hash, _ = decode(HIST_CLIP)
    rep = {"schema": "w121_inspect_v1", "historical": {"path": str(HIST_CLIP), "sha256": sha(HIST_CLIP), "frames": hist_meta["frames"],
                                                       "fps": hist_meta["fps"], "duration_s": hist_meta["frames"] / hist_meta["fps"]},
           "rules": {"dhash_bits": AB.DHASH_SIZE ** 2, "dup_hamming": AB.DUP_HAMMING, "dup_run_frames": AB.DUP_RUN_FRAMES,
                     "cut_mad": AB.CUT_MAD}, "candidates": {}}
    fp = {}
    for t, c in sorted(acq["candidates"].items()):
        if not c.get("download"):
            continue
        p = Path(c["download"]["path"])
        r = {"title": t, "path": str(p), "sha256": sha(p)}
        r["sha256_matches_acquire"] = r["sha256"] == c["download"]["sha256"]
        meta, dig, hs, th = decode(p)
        r.update({k: meta[k] for k in ("fps", "cap_frame_count", "fourcc", "backend", "frames", "sizes")})
        pm = np.array(meta["pos_msec"], float)
        r["timestamps_s"] = [i / meta["fps"] for i in range(meta["frames"])] if meta["fps"] and meta["fps"] > 0 else []
        r["container_pos_msec_minus_index_over_fps_max_abs_ms"] = (float(np.max(np.abs(pm[1:] - pm[:-1] - 1000.0 / meta["fps"])))
                                                                   if len(pm) > 2 and meta["fps"] else None)
        why = []
        if not meta["frames"] or not meta["fps"] or not np.isfinite(meta["fps"]) or meta["fps"] <= 0:
            why.append("S1_UNDECODABLE_OR_NO_CADENCE")
        if len(meta["sizes"]) != 1:
            why.append("S1_FRAME_SIZE_CHANGES")
        if meta["frames"] < AB.SEG_MIN_FRAMES:
            why.append("S1_TOO_FEW_FRAMES")
        if r["container_pos_msec_minus_index_over_fps_max_abs_ms"] is None or                 r["container_pos_msec_minus_index_over_fps_max_abs_ms"] > AB.CADENCE_TOL_MS:
            why.append("S1_DECLARED_FPS_INCONSISTENT_WITH_CONTAINER_TIMESTAMPS")
        if r["sha256"] in AB.EXCLUDED_SHA256:
            why.append("S2_EXCLUDED_HISTORICAL_SHA")
        run_hist = AB.longest_matched_run(hs, hist_hash)
        r["dup_vs_historical_longest_run"] = run_hist
        if run_hist >= AB.DUP_RUN_FRAMES:
            why.append("S2_DUPLICATE_OF_HISTORICAL_SOURCE")
        cuts = AB.shot_cuts(th) if th else []
        r["shot_cuts"] = cuts
        r["decoded_BGR_uint8_sha256"] = dig
        r["dhash"] = [format(h, "016x") for h in hs]
        r["reasons"] = why
        fp[t] = hs
        rep["candidates"][t] = r
        if dig:
            d = new_dir(out / "candidates" / safe(t))
            dump(d / "inspection.json", {"clip_sha256": r["sha256"], "frames": meta["frames"], "fps": meta["fps"],
                                         "size": list(meta["sizes"][0]) if meta["sizes"] else None, "timestamp": r["timestamps_s"],
                                         "decoded_BGR_uint8_sha256": dig, "source": f"Wikimedia Commons {t}",
                                         "GT_available": False, "unseen": True})
            # 8-frame inventory sheet (visual check only)
            cap = cv2.VideoCapture(str(p))
            ids = set(np.linspace(0, meta["frames"] - 1, 8).round().astype(int).tolist())
            sheet = np.zeros((2 * 280, 4 * 420, 3), np.uint8)
            i = n = 0
            while True:
                ok, im = cap.read()
                if not ok:
                    break
                if i in ids:
                    sheet[n // 4 * 280:n // 4 * 280 + 250, n % 4 * 420:(n % 4 + 1) * 420] = cv2.resize(im, (420, 250))
                    cv2.putText(sheet, f"frame {i} / {i / meta['fps']:.2f}s", (n % 4 * 420 + 8, n // 4 * 280 + 270),
                                cv2.FONT_HERSHEY_SIMPLEX, .5, (235, 235, 235), 1, cv2.LINE_AA)
                    n += 1
                i += 1
            cap.release()
            cv2.imwrite(str(d / "inventory.jpg"), sheet)
    # cross-candidate duplicates (the same footage published twice)
    titles = sorted(fp)
    pairs = []
    for i, t1 in enumerate(titles):
        for t2 in titles[i + 1:]:
            run = AB.longest_matched_run(fp[t1], fp[t2])
            if run >= AB.DUP_RUN_FRAMES:
                pairs.append({"a": t1, "b": t2, "longest_run": run})
    rep["cross_candidate_duplicates"] = pairs
    dump(out / "inspect.json", rep)
    print(json.dumps({t: (r["frames"], r["fps"], r["sizes"], r["dup_vs_historical_longest_run"], len(r["shot_cuts"]), r["reasons"])
                      for t, r in rep["candidates"].items()}, indent=1)[:6000])
    print("cross duplicates", pairs)


# ------------------------------------------------------------------------------------------------ perceive ---
def run_frozen(clip, inspection, out_dir, log):
    """The UNCHANGED W102 diagnostic (M1 batch 1 + frozen S0) from the W102 code snapshot."""
    args = [str(PY), "-X", "utf8", str(FROZEN_LAUNCHER), "w102_video_diagnostic.py", "run", "--clip", str(clip),
            "--inspection", str(inspection), "--weights", str(FROZEN["weights"]), "--checkpoint", str(FROZEN["checkpoint"]),
            "--gate-report", str(FROZEN["gate_report"]), "--lock", str(FROZEN["lock"]), "--out-dir", str(out_dir)]
    t0 = time.time()
    r = subprocess.run(args, capture_output=True, text=True)
    Path(log).write_text(" ".join(args) + "\n" + r.stdout + "\n" + r.stderr, encoding="utf-8")
    return r.returncode, time.time() - t0, args


def coverage(npz, size):
    """Frozen-perception coverage (GT-free).  Subject continuity is a 2D PROXY only; association stays UNKNOWN."""
    lm, x, v = npz["landmarks"], npz["input_2d"], npz["input_valid"]
    W, H = size
    n = len(lm)
    present = np.isfinite(lm[:, :, 0]).any(1)
    arm_valid = v[:, list(AB.LEFT_ARM)].all(1)
    px = x[:, :, :2] * np.array([W, H])
    wrist = np.where(v[:, 13:14], px[:, 13], np.nan)
    sh = np.where((v[:, 11] & v[:, 14])[:, None], (px[:, 11] + px[:, 14]) / 2, np.nan)
    hp = np.where((v[:, 1] & v[:, 4])[:, None], (px[:, 1] + px[:, 4]) / 2, np.nan)
    torso = np.linalg.norm(sh - hp, axis=1)
    boxes = []
    for i in range(n):
        p = px[i][v[i]]
        boxes.append(None if len(p) < 2 else [float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max())])

    def iou(a, b):
        if a is None or b is None:
            return None
        ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
        return ix * iy / u if u > 0 else None
    ious = [None] + [iou(boxes[i - 1], boxes[i]) for i in range(1, n)]
    switch = [i for i, q in enumerate(ious) if q is not None and q < 0.3]
    return {"present": present, "arm_valid": arm_valid, "wrist_xy": wrist, "torso_px": torso, "boxes": boxes, "bbox_iou_prev": ious,
            "possible_subject_switch_frames": switch, "valid_joint_count": v.sum(1)}


def rig_ctx():
    import w105_rig_utility as W5
    return W5.rig_context()


def fixed_slot(P, rc):
    """EXACTLY the W105 build_clip FIXED-slot computation (GT-free S0-only configuration): W93 torso frames from S0 as
    parent, W102 pose_export legs/spine via W105 rig_measure, S0 left arm through the existing chain (arm_export)."""
    import w105_rig_utility as W5
    from motion.torso_frame import torso_frames
    from w102_rig_review import arm_export
    j = np.asarray(P, float)
    j = j - j[0]
    Rt, Rp = torso_frames(j)
    pose = W5.rig_measure(j, Rp, Rt, rc) if Rp is not None and Rt is not None else None
    arm = None if pose is None else arm_export(j[12] - j[11], j[13] - j[11], Rp, Rt, rc)
    if pose is None:
        return None
    fr = {"object_rotation_xyzw": {"FIXED": pose["export"]["object"]}, "extra_bone_locals_xyzw": {"FIXED": dict(pose["export"]["bones"])},
          "modes": {}, "fk_points_armature": {"FIXED": pose["export"]["points"]}, "forward_cam": {"FIXED": pose["export"]["forward"]}}
    if arm is not None:
        fr["modes"]["FIXED"] = arm
    else:
        for bone in ("upperarm_l", "lowerarm_l"):
            fr["extra_bone_locals_xyzw"]["FIXED"][bone] = [0., 0., 0., 1.]
    fr["parent"] = {"FIXED": {"Rp": Rp.tolist(), "Rt": Rt.tolist()}}
    return fr


def json_eq(a, b):
    return json.dumps(clean(a), sort_keys=True) == json.dumps(clean(b), sort_keys=True)


def adapter_fidelity(rc):
    """Rig adapter control: W102's stored historical predictions through fixed_slot must equal W105's stored real_clip FIXED
    slot on every frame (object rotation, spine/leg locals, S0 left-arm mode)."""
    z = np.load(HIST_OBS)
    vc = load(W105_VIDEO_CASES)["cases"]["real_clip"]["frames"]
    bad = []
    for i, fr in enumerate(vc):
        mine = fixed_slot(z["predictions"][i], rc)
        ok = (mine is not None and json_eq(mine["object_rotation_xyzw"]["FIXED"], fr["object_rotation_xyzw"]["FIXED"])
              and json_eq(mine["extra_bone_locals_xyzw"]["FIXED"], fr["extra_bone_locals_xyzw"]["FIXED"])
              and json_eq(mine["modes"].get("FIXED"), fr["modes"].get("FIXED")))
        if not ok:
            bad.append(i)
    return {"frames": len(vc), "mismatch_frames": bad, "ok": not bad and sha(HIST_OBS) == HIST_OBS_SHA,
            "w105_video_cases_sha256": sha(W105_VIDEO_CASES), "w102_obs_sha256": sha(HIST_OBS)}


def activity_of(title):
    m = re.search(AB.ACTIVITY_RE, title.lower())
    return re.sub(r"[^a-z]", "", m.group(1)) if m else None


def cmd_perceive(a):
    out = new_dir(a.out_dir)
    ins = load(a.inspect / "inspect.json")
    acq = load(a.acquire / "acquire.json")
    rep = {"schema": "w121_perceive_v1", "frozen_inputs": {k: str(v) for k, v in FROZEN.items()},
           "frozen_identity": {"diagnostic_script_sha256": sha(FROZEN["diagnostic_script"]),
                               "repo_diagnostic_equal_LF": FROZEN["diagnostic_script"].read_bytes().replace(b"\r\n", b"\n")
                               == (ROOT / "scripts/w102_video_diagnostic.py").read_bytes().replace(b"\r\n", b"\n"),
                               "checkpoint_sha256": sha(FROZEN["checkpoint"]), "gate_report_sha256": sha(FROZEN["gate_report"]),
                               "lock_sha256": sha(FROZEN["lock"]), "launcher_sha256": sha(FROZEN_LAUNCHER)},
           "rules": {"left_arm": AB.LEFT_ARM, "seg_min_s": AB.SEG_MIN_S, "seg_min_frames": AB.SEG_MIN_FRAMES,
                     "seg_max_frames": AB.SEG_MAX_FRAMES, "arm_valid_min": AB.ARM_VALID_MIN, "arm_activity_min": AB.ARM_ACTIVITY_MIN,
                     "max_sources": AB.MAX_SOURCES, "present_mask": "M1 subject landmarks finite AND the W105 FIXED-slot rig pose + "
                     "S0 left arm defined", "selection": "eligible candidates ranked by (left-arm valid fraction desc, segment "
                     "frames desc, title); greedily keep one per author, uploader and declared activity keyword"},
           "candidates": {}}
    # frozen-stack reproduction control on the historical clip (byte-exact W102 observations)
    hd = out / "historical_reproduction"
    rc_, sec, _ = run_frozen(HIST_CLIP, W102 / "results/video_inspection/inspection.json", hd, out / "historical_reproduction.log")
    zr, zh = np.load(hd / "observations_predictions.npz"), np.load(HIST_OBS)
    rep["historical_reproduction"] = {"returncode": rc_, "seconds": sec,
                                      "arrays_byte_equal": {k: zr[k].tobytes() == zh[k].tobytes() for k in zh.files},
                                      "file_sha256": sha(hd / "observations_predictions.npz"), "expected_sha256": HIST_OBS_SHA}
    rep["historical_reproduction"]["ok"] = rc_ == 0 and all(rep["historical_reproduction"]["arrays_byte_equal"].values())
    rc = rig_ctx()
    rep["rig_adapter_fidelity"] = adapter_fidelity(rc)
    dup_pairs = ins["cross_candidate_duplicates"]
    for t, r in sorted(ins["candidates"].items()):
        rec = {"title": t, "inspect_reasons": r["reasons"]}
        rep["candidates"][t] = rec
        if r["reasons"]:
            rec["status"] = "REJECTED_AT_INSPECT"
            continue
        cdir = out / "candidates" / safe(t)
        cdir.mkdir(parents=True)
        code, sec, args = run_frozen(r["path"], a.inspect / "candidates" / safe(t) / "inspection.json", cdir / "m1s0",
                                     cdir / "m1s0.log")
        rec.update(frozen_returncode=code, frozen_seconds=sec)
        if code != 0:
            rec["status"] = "PERCEPTION_BLOCKED"
            continue
        z = np.load(cdir / "m1s0" / "observations_predictions.npz")
        diag = load(cdir / "m1s0" / "report.json")
        size = tuple(r["sizes"][0])
        cov = coverage(z, size)
        rigs = [fixed_slot(z["predictions"][i], rc) for i in range(len(z["predictions"]))]
        rig_ok = np.array([f is not None and "FIXED" in f["modes"] for f in rigs])
        usable = cov["present"] & rig_ok
        runs, chosen = AB.segment_select(usable, cov["arm_valid"], cov["wrist_xy"], cov["torso_px"], r["shot_cuts"], r["fps"])
        rec.update(status="PERCEIVED", fps=r["fps"], frames=len(usable), size=size, tracking=diag["tracking"],
                   M1_identity_checkpoints=diag["M1_identity"]["checkpoints_sha256"], environment_fingerprint=diag["environment_fingerprint"],
                   observation_prediction_sha256=diag["observation_prediction_sha256"],
                   coverage={"subject_present_frames": int(cov["present"].sum()), "rig_pose_frames": int(rig_ok.sum()),
                             "usable_frames": int(usable.sum()), "left_arm_valid_frames": int(cov["arm_valid"].sum()),
                             "possible_subject_switch_frames": cov["possible_subject_switch_frames"],
                             "valid_joint_histogram": {int(k): int(c) for k, c in zip(*np.unique(cov["valid_joint_count"], return_counts=True))},
                             "subject_association": "UNKNOWN (no GT; per-frame continuity only by 2D bbox IoU proxy)"},
                   segment_runs=runs, segment=chosen, activity=activity_of(t),
                   artist=acq["candidates"][t]["artist"], uploader=acq["candidates"][t]["uploader"],
                   license=acq["candidates"][t]["license"])
        if chosen is None:
            rec["status"] = "NO_ELIGIBLE_SEGMENT"
        np.savez_compressed(cdir / "coverage.npz", present=cov["present"], arm_valid=cov["arm_valid"], rig_ok=rig_ok,
                            wrist_xy=cov["wrist_xy"], torso_px=cov["torso_px"])
    # deterministic source selection
    elig = [t for t, c in rep["candidates"].items() if c.get("status") == "PERCEIVED"]
    elig.sort(key=lambda t: (-rep["candidates"][t]["segment"]["arm_valid_fraction"], -rep["candidates"][t]["segment"]["frames"], t))
    chosen, used_a, used_u, used_k = [], set(), set(), set()
    for t in elig:
        c = rep["candidates"][t]
        dup = [p for p in dup_pairs if t in (p["a"], p["b"]) and ({p["a"], p["b"]} - {t}) & set(chosen)]
        why = []
        if dup:
            why.append("S2_DUPLICATE_OF_SELECTED_CANDIDATE")
        if (c["artist"] or "").lower() in used_a or (c["uploader"] or "").lower() in used_u:
            why.append("SEL_SAME_AUTHOR_OR_UPLOADER_AS_SELECTED")
        if c["activity"] in used_k:
            why.append("SEL_SAME_ACTIVITY_AS_SELECTED")
        if len(chosen) >= AB.MAX_SOURCES:
            why.append("SEL_MAX_SOURCES_REACHED")
        c["selection_reasons"] = why
        if not why:
            chosen.append(t)
            used_a.add((c["artist"] or "").lower())
            used_u.add((c["uploader"] or "").lower())
            used_k.add(c["activity"])
    rep["selected"] = chosen
    rep["selection_order"] = elig
    dump(out / "perceive.json", rep)
    print(json.dumps({"historical_reproduction_ok": rep["historical_reproduction"]["ok"],
                      "rig_adapter_fidelity_ok": rep["rig_adapter_fidelity"]["ok"], "selected": chosen,
                      "status": {t: (c.get("status"), c.get("segment") and (c["segment"]["window"], round(c["segment"]["arm_valid_fraction"], 3),
                                                                           round(c["segment"]["activity_ratio"], 2)), c.get("selection_reasons"))
                                 for t, c in rep["candidates"].items()}}, indent=1, default=str)[:8000])


# ------------------------------------------------------------------------------------------------ prepare ---
RIG_FBX = ROOT / "examples/skinned_mannequin/BaseRig_Skinned.fbx"
RIG_REST = ROOT / "examples/e2e_demo/baserig_blender_rest.json"
RIG_FBX_SHA = "c2947ba7f0d088aa47dbf0e04729a991699f24c649acce252b7b902740bb61d7"
RIG_REST_SHA = "1dcdc8268777bd433d1b73d56acef9a3f67b30f16456a80618d76fb6e70f9358"
W116_V0 = Path("C:/Users/dna10/animcv-w116/results/preflight/real_parent_state.json")
W105_FIXED_DIGEST_PREFIX = "4d168d95234c"
SCENE_FPS = 30   # historical renderer / W116 render scene rate (Action playback timing reported separately)


def C_():
    import w116_semantic_pin_contract as C
    return C


def blender(*args, log=None):
    r = subprocess.run([str(BLENDER), "-b", "-t", "2", "--python-exit-code", "1", "-P", *map(str, args)], cwd=ROOT,
                       capture_output=True, text=True)
    if log:
        Path(log).write_text(r.stdout + "\n" + r.stderr, encoding="utf-8")
    if r.returncode != 0:
        raise RuntimeError(f"STOP: Blender stage failed ({r.returncode}); see {log}")
    return r


def state_values_from_cases(frames):
    """EXACTLY W116 real_parent_state's value extraction (FIXED slot; mode locals overwrite extra locals)."""
    C = C_()
    vals = {c: [] for c in C.REAL_CH}
    for fr in frames:
        m = fr["modes"]["FIXED"]
        vals["__object__"].append(fr["object_rotation_xyzw"]["FIXED"])
        ex = dict(fr["extra_bone_locals_xyzw"]["FIXED"])
        ex.update(m.get("extra_bone_locals_xyzw", {}))
        for b in ("spine_01", "thigh_l", "calf_l", "thigh_r", "calf_r"):
            vals[b].append(ex[b])
        vals["upperarm_l"].append(m["proximal_local_xyzw"])
        vals["lowerarm_l"].append(m["distal_local_xyzw"])
    return vals


def cmd_prepare(a):
    import w116_semantic_pin_base as S
    C = C_()
    out = new_dir(a.out_dir)
    per = load(a.perceive / "perceive.json")
    acq = load(a.acquire / "acquire.json")
    rep = {"schema": "w121_prepare_v1", "scope": "GT-free source-to-rig preparation and UNEDITED parent preflight only; no edit, pin "
           "resolution or candidate exists", "rig": {"fbx": str(RIG_FBX), "fbx_sha256": sha(RIG_FBX), "rest": str(RIG_REST),
                                                   "rest_sha256": sha(RIG_REST)}, "sources": {}}
    rep["rig"]["identity_ok"] = rep["rig"]["fbx_sha256"] == RIG_FBX_SHA and rep["rig"]["rest_sha256"] == RIG_REST_SHA
    # parent-construction fidelity on the historical clip: W105 cases -> W116 v0 values
    hv0 = load(W116_V0)
    hvals = state_values_from_cases(load(W105_VIDEO_CASES)["cases"]["real_clip"]["frames"])
    rep["parent_value_fidelity_w116_v0"] = S.canon({c: [[float(x) for x in q] for q in hvals[c]] for c in C.REAL_CH}) == S.canon(hv0["values"])
    # Blender rig/keying reproduction control: W116 historical v0 must key to the W105 FIXED Action digest
    C.states_file(out / "historical_states.json", {"v0": hv0}, "w105")
    blender(ROOT / "scripts/w116_blender_stages.py", "--", "probe", "--rig", RIG_FBX, "--rest", RIG_REST, "--states",
            out / "historical_states.json", "--out", out / "historical_probe.json", log=out / "historical_probe.log")
    hp = load(out / "historical_probe.json")
    rep["historical_keying_digest"] = hp["states"]["v0"]["value_digest"]
    rep["historical_keying_reproduces_w105_fixed"] = hp["states"]["v0"]["value_digest"].startswith(W105_FIXED_DIGEST_PREFIX)
    rc = rig_ctx()
    for t in per["selected"]:
        pc = per["candidates"][t]
        sid = f"S{len(rep['sources']) + 1}"
        sd = new_dir(out / sid)
        z = np.load(a.perceive / "candidates" / safe(t) / "m1s0" / "observations_predictions.npz")
        wlo, whi = pc["segment"]["window"]
        frames = list(range(wlo, whi + 1))
        fps = pc["fps"]
        cases = []
        for i in frames:
            fr = fixed_slot(z["predictions"][i], rc)
            if fr is None or "FIXED" not in fr["modes"]:
                raise RuntimeError(f"STOP: rig contract failed inside the declared segment of {t} at {i}")
            fr.update(frame_index=i, time_s=i / fps, identity="UNKNOWN (GT-free; tracked-subject continuity unverified)")
            cases.append(fr)
        dump(sd / "video_cases.json", {"rest_snapshot_digest": rc["rest"].digest(), "canonical_to_armature_xyzw": list(rc["alignment"]),
                                       "cases": {sid: {"case": sid, "sequence_id": t, "start_frame": wlo, "fps_sampled": fps,
                                                       "frames": cases, "configuration": "GT-free S0-only (W105 FIXED slot)"}}})
        c = acq["candidates"][t]
        src = {"kind": "W121_independent_source_FIXED (M1 batch-1 + frozen S0; W105 S0-only rig configuration)",
               "video_cases_sha256": sha(sd / "video_cases.json"), "title": t, "url": c["url"], "descriptionurl": c["descriptionurl"],
               "license": c["license"], "artist": c["artist"], "clip_sha256": c["download"]["sha256"], "fps_source": fps,
               "time_s": [i / fps for i in frames], "segment_source_frames": [wlo, whi],
               "observation_prediction_sha256": pc["observation_prediction_sha256"],
               "rig_identity": "W105 BaseRig_Skinned (known output rig)",
               "video_subject_identity": "UNKNOWN (per-frame tracked subject continuity not verified)"}
        v0 = S.make_state(None, "BaseRig", frames, C.REAL_CH, state_values_from_cases(cases), {"kind": "ROOT_COPY_OF_SOURCE_HYPOTHESIS"}, src)
        dump(sd / "parent_state.json", v0)
        C.states_file(sd / "parent_states.json", {"v0": v0}, "w105")
        blender(ROOT / "scripts/w116_blender_stages.py", "--", "probe", "--rig", RIG_FBX, "--rest", RIG_REST, "--states",
                sd / "parent_states.json", "--out", sd / "probe.json", log=sd / "probe.log")
        pr = load(sd / "probe.json")
        rig = C.rig_from(RIG_REST, pr["rig"])
        fk = C.fk_error(rig, v0, pr)
        ok, failed = C.bindings_for(C.REAL_BINDING_CANDIDATES, "BaseRig", pr["rig"]["rest_snapshot_digest"], pr, v0, rig)
        wrist = {fr["frame_index"]: fr["world"]["lowerarm_l"]["tail"] for fr in pr["states"]["v0"]["frames"]}
        holds = AB.hold_windows(frames, wrist, fps)
        L1 = pr["rest"]["upperarm_l"]["length"] * pr["rig"]["scale"][0]
        L2 = pr["rest"]["lowerarm_l"]["length"] * pr["rig"]["scale"][0]
        slack = [L1 + L2 - float(np.linalg.norm(np.array(fr["world"]["lowerarm_l"]["tail"]) - np.array(fr["world"]["upperarm_l"]["head"])))
                 for fr in pr["states"]["v0"]["frames"]]
        flex = []
        for fr in pr["states"]["v0"]["frames"]:
            Sx, E, W = (np.array(fr["world"]["upperarm_l"]["head"]), np.array(fr["world"]["upperarm_l"]["tail"]),
                        np.array(fr["world"]["lowerarm_l"]["tail"]))
            u, v = E - Sx, W - E
            flex.append(math.degrees(math.atan2(float(np.linalg.norm(np.cross(u, v))), float(np.dot(u, v)))))
        steps = [float(np.linalg.norm(np.array(wrist[frames[i]]) - np.array(wrist[frames[i - 1]]))) for i in range(1, len(frames))]
        rep["sources"][sid] = {
            "title": t, "frames": [wlo, whi], "n_frames": len(frames), "fps_source": fps, "source_seconds": len(frames) / fps,
            "action_frames": [1, len(frames)], "action_seconds_at_scene_fps": len(frames) / SCENE_FPS,
            "time_mapping": "Action frame index = source frame index; Blender frame = index - first + 1; keyed CONSTANT; source time = index / fps",
            "parent_state_id": v0["state_id"], "probe_value_digest": pr["states"]["v0"]["value_digest"], "fcurves": pr["states"]["v0"]["fcurves"],
            "fk_model_max_error_m": fk, "fk_ok": fk[0] <= 1e-5, "bindings_bound": {b["semantic_joint"]: k for k, b in ok.items()},
            "bindings_failed": {k: v["status"] for k, v in failed.items()},
            "left_wrist_bound": any(b["semantic_joint"] == "left_wrist" for b in ok.values()),
            "coordinate_probes": C.coord_probes(pr, pr["rig"]["scale"][0]),
            "left_arm": {"upper_m": L1, "fore_m": L2, "reach_slack_m": {"min": min(slack), "median": float(np.median(slack)), "max": max(slack)},
                         "flex_deg": {"min": min(flex), "median": float(np.median(flex)), "max": max(flex)},
                         "near_straight_frames_lt5deg": sum(1 for f in flex if f < 5.0),
                         "wrist_step_m": {"median": float(np.median(steps)), "max": max(steps)},
                         "wrist_step_m_per_s": {"median": float(np.median(steps)) * fps, "max": max(steps) * fps}},
            "holds": {"L": holds["L"], "n_candidates": holds["n_candidates"],
                      "chosen": {k: {"capture": v["capture"], "frames": [v["frames"][0], v["frames"][-1]], "D_m": v["D_m"]}
                                 for k, v in holds["chosen"].items()},
                      "D_distribution_m": {"min": min(c["D_m"] for c in holds["candidates"]), "median": float(np.median([c["D_m"] for c in holds["candidates"]])),
                                           "max": max(c["D_m"] for c in holds["candidates"])}}}
        dump(sd / "bindings.json", {"bound": ok, "failed": failed})
        dump(sd / "hold_candidates.json", holds)
    dump(out / "prepare.json", rep)
    print(json.dumps(rep, indent=1, default=str)[:7000])


# ------------------------------------------------------------------------------- request population / build ---
UPPER, LOWER = "upperarm_l", "lowerarm_l"
ARM = [UPPER, LOWER]
NEG_BASE_ORDER = ["MID_RBpos", "LOW_RBpos", "HIGH_RBpos", "MID_RBneg", "LOW_RBneg", "HIGH_RBneg", "MID_none", "LOW_none", "HIGH_none"]
CV, SF, INF, UB = ["REJECT", "CONTRACT_VIOLATION"], ["REJECT", "SOLVER_FAILURE"], ["REJECT", "INFEASIBLE_REQUEST"], ["REJECT", "UNSUPPORTED_BINDING"]
# W121 negative controls on the FIRST accepted source (base request chosen in calibrate by the declared NEG_BASE rule from frozen
# controls only).  (base-or-special, factor, description, injected conditions for the frozen W120 predictor, expected, reasons)
NEG = {
    "NG_BUDGET": ("ZB", "G1F1", "zero-budget control; factor budget built with M_MAX=8 (unequal budget)", {"fairness_bad": True}, CV,
                  ["AB_FAIRNESS:SEARCH_BUDGET_MISMATCH:Y"]),
    "NG_MARGIN": ("BASE", "G0F0", "release margin = X_r margin - 1", {"fairness_bad": True}, CV, ["AB_FAIRNESS:MARGIN_BUDGET_MISMATCH:Y"]),
    "NG_PARENT": ("BASE", "G1F0", "variant record claims another source parent (stale source binding)", {"fairness_bad": True}, CV,
                  ["AB_FAIRNESS:PARENT_MISMATCH:Y"]),
    "NG_RIG": ("BASE", "G1F1", "variant state claims another rig identity", {"fairness_bad": True}, CV, ["AB_FAIRNESS:RIG_IDENTITY_MISMATCH:Y"]),
    "NG_CDIGEST": ("BASE", "G0F0", "variant record carries another pin-constraint digest", {"fairness_bad": True}, CV,
                   ["AB_FAIRNESS:PIN_CONSTRAINT_MISMATCH:Y"]),
    "NG_AUTH": ("BASE", "G1F0", "variant record claims a narrower authorisation [upperarm_l]", {"fairness_bad": True}, CV,
                ["AB_FAIRNESS:AUTH_MISMATCH:Y"]),
    "NG_PIN_COMBINED": ("BASE", "G0F1", "lowerarm_l +2 deg on the middle pin frame: wrong pin value AND failed output pin",
                        {"fairness_bad": True, "eval_pin": True}, CV, ["EVALUATED_PIN_UNSATISFIED", "AB_FAIRNESS:PIN_VALUES_DIFFER_FROM_F:Y"]),
    "NG_FALSE": ("BASE", "G1F1", "U on every pin frame while the resolver claims SOLVED (solver failure masked as success)",
                 {"fairness_bad": True, "eval_pin": True}, CV, ["EVALUATED_PIN_UNSATISFIED"]),
    "NG_NONCONV": ("BASE", "G0F0", "fault-injected skipped forearm aim (isolated nonconvergence)", {"nonconv": True}, SF, ["NON_CONVERGED"]),
    "NG_BONE": ("BASE", "G1F0", "spine_01 (unauthorised) +1 deg on a transition frame",
                {"eval_scope": True, "fp_channel": True, "fp_outside_arm": True}, CV, ["UNAUTHORIZED_CHANNEL_CHANGE"]),
    "NG_ROOT": ("BASE", "G0F0", "__object__ +1 deg on a transition frame",
                {"eval_root": True, "eval_scope": True, "fp_root": True, "fp_channel": True, "fp_outside_arm": True}, CV, ["ROOT_MOVED"]),
    "NG_OUTSIDE": ("BASE", "G1F0", "upperarm_l +1 deg on a frame outside budget and planned domain (hidden changed frame)",
                   {"fp_budget": True, "fp_domain": True}, CV, ["FOOTPRINT_EXCEEDS_DECLARED_DOMAIN", "CHANGE_OUTSIDE_AUTOMATIC_BUDGET"]),
    "NG_PLANNED": ("BASE", "G0F1", "values unchanged; planned domain omits its outermost transition frame (false Actual Change Domain)",
                   {"fp_domain": True}, CV, ["FOOTPRINT_EXCEEDS_DECLARED_DOMAIN"]),
    "NG_DISPLAY": ("BASE", "G1F1", "false display: first automatic change relabelled UNCHANGED before the decision", {"display_bad": True}, CV,
                   ["DISPLAY_FOOTPRINT_MISMATCH"]),
    "NG_SHARED": ("BASE", "G0F0", "F_r replaced by a sibling-shared F (this request's pins + the other windows' pins of the same edit)",
                  {"isolation_bad": True}, CV, ["BASELINE_NOT_ISOLATED"]),
    "NG_REACH": ("REACH", "G1F0", "left-wrist target = capture + 1.0 m world Z (infeasible reach)", {"pin_status": "UNREACHABLE"}, INF, []),
    "NG_DOF": ("DOF", "G1F0", "forearm-only authorisation (insufficient authorised DOF)", {"missing_dof": True}, INF, ["Y_MISSING_AUTHORIZED_DOF"]),
    "NG_CONFLICT": ("CONFLICT", "G0F1", "two left-wrist hard pins 0.02 m apart", {"validation": "INFEASIBLE_REQUEST"}, INF,
                    ["CONFLICTING_HARD_CONSTRAINTS"]),
    "NG_WRONG_BINDING": ("WRONGBIND", "G0F0", "semantic left_wrist pin carrying the right_wrist binding id", {"validation": "UNSUPPORTED_BINDING"},
                         UB, ["BINDING_JOINT_MISMATCH"]),
}


def vals_of(st, k):
    import w116_semantic_pin_base as S
    return S.frame_vals(st, st["frames"].index(k))


def all_vals(st):
    import w116_semantic_pin_base as S
    return {k: S.frame_vals(st, i) for i, k in enumerate(st["frames"])}


def vlist(v):
    return {c: [float(x) for x in np.asarray(q, float)] for c, q in v.items()}


def rot_local(q, axis, deg):
    import w116_semantic_pin_base as S
    return S.q_mul(np.asarray(q, float), S.q_axis_angle(axis, math.radians(deg)))


class Src:
    """One accepted independent source: parent v0, frozen rig, bindings, capture table and the declared requests."""

    def __init__(self, sid, prep_dir):
        import w116_semantic_pin_base as S
        C = C_()
        self.S, self.C, self.sid = S, C, sid
        sd = Path(prep_dir) / sid
        self.v0 = load(sd / "parent_state.json")
        self.probe = load(sd / "probe.json")
        self.rig = C.rig_from(RIG_REST, self.probe["rig"])
        b = load(sd / "bindings.json")
        self.bound, self.failed = b["bound"], b["failed"]
        self.frames = self.v0["frames"]
        self.cap = {}
        for fr in self.probe["states"]["v0"]["frames"]:
            for bid, rec in self.bound.items():
                self.cap[(self.v0["state_id"], fr["frame_index"], bid)] = fr["world"][rec["bone"]][rec["point"]]
        self.holds = load(sd / "hold_candidates.json")["chosen"]
        self.states = {"v0": self.v0}
        self.ctrl = {}
        self.req, self.recs = {}, {}

    def bid(self, joint):
        return next((k for k, v in self.bound.items() if v["semantic_joint"] == joint), None)

    def control(self, edit):
        """U for an edit: v0 itself (no edit) or the declared W116 side bend on EVERY frame (W117 family-control rule)."""
        S, C = self.S, self.C
        key = "v0" if edit is None else f"U__{edit}"
        if key not in self.ctrl:
            base = all_vals(self.v0)
            if edit is None:
                self.ctrl[key] = base
            else:
                e = C.EDITS[edit]
                prm = {"angle_rad": e["sign"] * math.radians(e["magnitude"]["angle_deg"])}
                self.ctrl[key] = {k: S.apply_edit(self.rig, base[k], e, prm) for k in self.frames}
                self.states[key] = C.patched_state(self.v0, self.ctrl[key], {"kind": "EVAL_CONTROL", "edit": edit})
        return key, self.ctrl[key]

    def pins(self, rid, pin_decl, frames, auth, cap_frame):
        S = self.S
        out = []
        for i, (joint, kind, off, bind_as) in enumerate(pin_decl):
            bid = self.bid(bind_as or joint) or f"UNBOUND:{joint}"
            src = {"kind": "CAPTURED_FROM_EVALUATED_STATE" if kind == "capture" else "CAPTURED_PLUS_DECLARED_OFFSET",
                   "state_id": self.v0["state_id"], "frame_index": cap_frame, "binding_id": bid,
                   "evaluated_by": "w116_blender_stages probe (W121 prepare)"}
            if kind == "offset":
                src["offset_m"] = off
            t = S.captured_target(self.cap, src)
            out.append(S.make_pin(f"{rid}/c{i}", joint, bid, [0.0, 0.0, 0.0] if t is None else t.tolist(), src, frames, auth,
                                  self.v0["state_id"]))
        return out

    def vctx(self):
        roots = {n for n in self.rig.bones if self.rig.is_root(n)}
        return {"bindings": self.bound, "failed_bindings": self.failed, "capture": self.cap, "parent": self.v0, "root_bones": roots}

    def rp(self, pins, k):
        return [{"bone": self.bound[c["binding_id"]]["bone"], "point": self.bound[c["binding_id"]]["point"], "target": np.array(c["target"]),
                 "semantic_joint": c["semantic_joint"]} for c in pins if c["binding_id"] in self.bound and k in c["frames"]]

    def declare_requests(self):
        """Fixed population: windows LOW/MID/HIGH x edits none/RBpos/RBneg (evaluation), ZB zero-budget control, and the
        special negative-control requests REACH / DOF / CONFLICT / WRONGBIND on the LOW window."""
        for W in ("LOW", "MID", "HIGH"):
            if W not in self.holds:
                continue
            h = self.holds[W]
            for e, edit in AB.HOLD_EDITS.items():
                self.req[f"{self.sid}_{W}_{e}"] = {"window": W, "edit": edit, "edit_key": e, "cap": h["capture"], "pin_frames": list(h["frames"]),
                                                   "pin_decl": [("left_wrist", "capture", None, None)], "auth": ARM, "m_max": 8,
                                                   "role": "evaluation"}
        lo = self.holds["LOW"]
        self.req[f"{self.sid}_ZB"] = {"window": "LOW", "edit": "SIDE_BEND_5DEG_POS", "edit_key": "RBpos", "cap": lo["capture"],
                                      "pin_frames": list(lo["frames"]), "pin_decl": [("left_wrist", "capture", None, None)], "auth": ARM,
                                      "m_max": 0, "role": "control"}
        sp = {"REACH": ([("left_wrist", "offset", [0.0, 0.0, 1.0], None)], ARM),
              "DOF": ([("left_wrist", "capture", None, None)], [LOWER]),
              "CONFLICT": ([("left_wrist", "capture", None, None), ("left_wrist", "offset", [0.0, 0.0, 0.02], None)], ARM),
              "WRONGBIND": ([("left_wrist", "capture", None, "right_wrist")], ARM)}
        for k, (pd, auth) in sp.items():
            self.req[f"{self.sid}_{k}"] = {"window": "LOW", "edit": "SIDE_BEND_5DEG_POS", "edit_key": "RBpos", "cap": lo["capture"],
                                           "pin_frames": list(lo["frames"]), "pin_decl": pd, "auth": auth, "m_max": 8, "role": "special"}

    def fair_rec(self, R, Ust, pin_vals, margins=None, budget=None):
        return {"parent_id": self.v0["state_id"], "rig_id": self.v0["rig_id"], "U_state_id": Ust["state_id"],
                "constraint_digests": sorted(c["constraint_digest"] for c in R["request"]["constraints"]),
                "pin_frames": list(R["pin_frames"]), "auth": sorted(R["auth"]), "pin_values_digest": self.S.digest(pin_vals),
                "margins": margins, "budget_frames": budget}

    def control_record(self, rid):
        """Frozen controls: U, F_r (W116 framewise on THIS request's pin frames only), X_r (W118 offset layer)."""
        import w118_transition_edit_base as XB
        import w119_endpoint_transition_base as YB
        S, C = self.S, self.C
        D = self.req[rid]
        ukey, ctrl = self.control(D["edit"])
        Ust = self.states[ukey]
        pin = D["pin_frames"]
        pins = self.pins(rid, D["pin_decl"], pin, D["auth"], D["cap"])
        er = C.edit_record(D["edit"], self.frames, {}) if D["edit"] else None
        req = {"request_id": rid, "parent_state_id": self.v0["state_id"], "edit": er, "constraints": pins, "resolver": {}}
        errs = S.validate_request(req, self.vctx())
        R = {"side": "real", "kind": "hold", "role": D["role"], "window": D["window"], "edit_key": D["edit_key"], "parent": "v0", "U": ukey,
             "auth": D["auth"], "chain": [UPPER, LOWER], "pin_frames": pin, "pin_runs": XB.pin_runs(pin), "request": req,
             "validation_errors": errs, "edit": D["edit"], "m_max": D["m_max"], "source": self.sid}
        if errs:
            R.update(contract_invalid=True, F=ukey, X=ukey, F_statuses={},
                     x_info={"budget_frames": XB.budget_plan(pin, self.frames, D["m_max"])[1], "planned_frames": [], "sides": [], "m_max": D["m_max"]},
                     margins={}, fair=None, _ctrl=ctrl, _Fv={})
            self.recs[rid] = R
            return R
        st, Fv = {}, {}
        for k in pin:
            s_, v, _ = S.resolve_frame(self.rig, ctrl[k], self.rp(pins, k), D["auth"], method="analytic")
            st[str(k)], Fv[k] = s_, v
        R["F_statuses"] = st
        Fr = C.patched_state(self.v0, {k: (Fv[k] if k in Fv else ctrl[k]) for k in self.frames}, {"kind": "F_CANDIDATE", "request": rid})
        self.states[f"{rid}__Fr"] = Fr
        R["F"] = f"{rid}__Fr"
        R["isolation_values_F_r"] = YB.isolation_check_values(all_vals(Fr), ctrl, pin, self.frames)
        xv, xi = XB.transition_edit(self.rig, ctrl, Fv, pin, self.frames, D["auth"], LOWER, m_max=D["m_max"])
        xi = {k: v for k, v in xi.items() if k != "wall_s"}
        Xr = C.patched_state(self.v0, {k: (xv[k] if k in xv else ctrl[k]) for k in self.frames}, {"kind": "X_CANDIDATE", "request": rid})
        self.states[f"{rid}__Xr"] = Xr
        R.update(X=f"{rid}__Xr", x_info=xi, margins=YB.margins_of(xi))
        R["fair"] = {"F": self.fair_rec(R, Ust, {str(k): vlist(Fv[k]) for k in pin}),
                     "X": self.fair_rec(R, Ust, {str(k): vlist(xv[k]) for k in pin}, R["margins"], xi["budget_frames"])}
        R["_ctrl"], R["_Fv"], R["_xv"] = ctrl, Fv, xv
        self.recs[rid] = R
        return R

    def x_vals(self, rid):
        R = self.recs[rid]
        X = self.states[R["X"]]
        return {k: vals_of(X, k) for k in R["x_info"].get("planned_frames", []) if k not in set(R["pin_frames"])}

    def solve(self, rid, f, ctrl=None, margins=None, m_max=None, **kw):
        import w120_factor_attribution_base as FB
        R = self.recs[rid]
        return FB.factor_transition(self.rig, R["_ctrl"] if ctrl is None else ctrl, R["_Fv"], R["pin_frames"], self.frames, R["auth"], UPPER,
                                    LOWER, R["margins"] if margins is None else margins, R["x_info"]["m_max"] if m_max is None else m_max, f,
                                    x_vals=self.x_vals(rid), **kw)

    def factor_rec(self, rid, vals, info):
        R = self.recs[rid]
        r = self.fair_rec(R, self.states[R["U"]], {str(k): vlist(vals[k]) for k in R["pin_frames"] if k in vals}, info["margins"],
                          info["budget_frames"])
        r["solver_params_digest"] = info.get("solver_params_digest")
        return r

    def make_factors(self, rid):
        import w120_factor_attribution_base as FB
        R = self.recs[rid]
        R["factors"], R["_fv"] = {}, {}
        if R.get("contract_invalid"):
            for f in FB.FACTORS:
                R["factors"][f] = {"state": R["U"], "fair": None,
                                   "info": {"factor": f, "request_status": "NOT_RESOLVED_CONTRACT_INVALID", "planned_frames": [], "frames": {},
                                            "sides": [], "margins": {}, "budget_frames": R["x_info"]["budget_frames"], "fallback": None}}
            return
        for f in FB.FACTORS:
            vals, info = self.solve(rid, f)
            name = f"{rid}__{f}"
            self.states[name] = self.C.patched_state(self.v0, {k: (vals[k] if k in vals else R["_ctrl"][k]) for k in self.frames},
                                                     {"kind": "W121_FACTOR_CANDIDATE", "factor": f, "request": rid})
            R["factors"][f] = {"state": name, "info": info, "fair": self.factor_rec(rid, vals, info)}
            R["_fv"][f] = vals

    def build_controls(self):
        self.declare_requests()
        for rid in self.req:
            self.control_record(rid)

    def neg_base(self):
        """Declared rule (frozen controls only): the first request of NEG_BASE_ORDER that is contract-valid, has every pin status
        SOLVED/NO_CHANGE_NEEDED, and an X_r entry margin >= 1 and release margin >= 2."""
        for w in NEG_BASE_ORDER:
            rid = f"{self.sid}_{w}"
            R = self.recs.get(rid)
            if R is None or R.get("contract_invalid"):
                continue
            if set(R["F_statuses"].values()) - {"SOLVED", "NO_CHANGE_NEEDED"}:
                continue
            m = R["margins"]
            ent = [v for k, v in m.items() if k.endswith("entry")]
            rel = [v for k, v in m.items() if k.endswith("release")]
            if ent and rel and min(ent) >= 1 and min(rel) >= 2:
                return rid
        return None

    # ------------------------------------------------------------------ negative controls (frozen W120 patterns)
    def neg(self, nid, src, f, vals=None, info=None, ctrl=None, F_state=None, fair_mut=None, statuses=None, planned=None, rig_id=None,
            display_corrupt=None):
        import copy
        import w119_endpoint_transition_base as YB
        S = self.S
        _, _, desc, cond, want, reasons = NEG[nid]
        r0 = self.recs[src]
        rec = {k: copy.deepcopy(v) for k, v in r0.items() if not k.startswith("_")}
        rec.update(role="negative", negative_of=src, neg_factor=f, desc=desc, expect=want, expect_reasons=reasons, conditions=cond)
        fac = copy.deepcopy(r0["factors"][f])
        if F_state is not None:
            rec["F"] = F_state
            rec["isolation_values_F_r"] = YB.isolation_check_values(all_vals(self.states[F_state]), r0["_ctrl"], r0["pin_frames"], self.frames)
        if vals is not None:
            base_ctrl = r0["_ctrl"] if ctrl is None else ctrl
            st = self.C.patched_state(self.v0, {k: (vals[k] if k in vals else base_ctrl[k]) for k in self.frames},
                                      {"kind": "W121_NEGATIVE_CONTROL", "factor": f, "request": nid})
            if rig_id:
                st = S.make_state(st["parent_id"], rig_id, st["frames"], st["channels"], st["values"], st["op"], st.get("source"))
            self.states[f"{nid}__{f}"] = st
            fac["state"] = f"{nid}__{f}"
            fac["fair"]["pin_values_digest"] = S.digest({str(k): vlist(vals[k]) for k in r0["pin_frames"] if k in vals})
        if info is not None:
            fac["info"] = info
            fac["fair"]["margins"], fac["fair"]["budget_frames"] = info["margins"], info["budget_frames"]
            fac["fair"]["solver_params_digest"] = info.get("solver_params_digest")
        if rig_id:
            fac["fair"]["rig_id"] = rig_id
        if fair_mut:
            fair_mut(fac["fair"])
        if statuses is not None:
            rec["V_statuses_override"] = statuses
        if planned is not None:
            fac["info"] = dict(fac["info"], planned_frames=planned(fac["info"]))
        if display_corrupt:
            rec["display_corrupt"] = display_corrupt
        rec["factors"] = {f: fac}
        self.recs[nid] = rec

    def tam(self, src, f, fn):
        r0 = self.recs[src]
        vv = {k: {c: np.array(x, float) for c, x in v.items()} for k, v in r0["_fv"][f].items()}
        fn(vv, r0["_ctrl"])
        return vv

    def negatives(self, base):
        import w118_transition_edit_base as XB
        r, s = self.recs, self.sid
        zb = f"{s}_ZB"
        v, i = self.solve(zb, "G1F1", m_max=XB.M_MAX)
        self.neg("NG_BUDGET", zb, "G1F1", vals=v, info=i)
        mm = dict(r[base]["margins"])
        key = next(k for k in mm if k.endswith("release"))
        mm[key] = max(0, mm[key] - 1)
        v, i = self.solve(base, "G0F0", margins=mm)
        self.neg("NG_MARGIN", base, "G0F0", vals=v, info=i)
        self.neg("NG_PARENT", base, "G1F0", fair_mut=lambda fr: fr.__setitem__("parent_id", "0" * 64))
        self.neg("NG_RIG", base, "G1F1", vals=r[base]["_fv"]["G1F1"], rig_id="W121_OTHER_RIG")
        self.neg("NG_CDIGEST", base, "G0F0", fair_mut=lambda fr: fr.__setitem__("constraint_digests", ["W121_TAMPERED_DIGEST"]))
        self.neg("NG_AUTH", base, "G1F0", fair_mut=lambda fr: fr.__setitem__("auth", [UPPER]))
        pinb = r[base]["pin_frames"]
        mid = pinb[len(pinb) // 2]
        self.neg("NG_PIN_COMBINED", base, "G0F1", vals=self.tam(base, "G0F1", lambda vv, c: vv[mid].__setitem__(LOWER, rot_local(vv[mid][LOWER], [0, 0, 1], 2.0))))

        def false_solved(vv, c):
            for k in pinb:
                vv[k] = dict(c[k])
        self.neg("NG_FALSE", base, "G1F1", vals=self.tam(base, "G1F1", false_solved), statuses={str(k): "SOLVED" for k in pinb})
        v, i = self.solve(base, "G0F0", fault="skip_lower_aim")
        self.neg("NG_NONCONV", base, "G0F0", vals=v, info=i)
        tf = sorted(k for k in r[base]["factors"]["G1F0"]["info"]["planned_frames"] if k not in set(pinb))[0]
        self.neg("NG_BONE", base, "G1F0", vals=self.tam(base, "G1F0", lambda vv, c: vv[tf].__setitem__("spine_01", rot_local(vv[tf]["spine_01"], [1, 0, 0], 1.0))))
        self.neg("NG_ROOT", base, "G0F0", vals=self.tam(base, "G0F0", lambda vv, c: vv[tf].__setitem__("__object__", rot_local(vv[tf]["__object__"], [0, 0, 1], 1.0))))
        bud = set(r[base]["factors"]["G1F0"]["info"]["budget_frames"])
        far = next(k for k in self.frames if k not in bud)
        self.neg("NG_OUTSIDE", base, "G1F0", vals=self.tam(base, "G1F0", lambda vv, c: vv.__setitem__(far, dict(c[far], **{UPPER: rot_local(c[far][UPPER], [0, 0, 1], 1.0)}))))
        self.neg("NG_PLANNED", base, "G0F1", vals=self.tam(base, "G0F1", lambda vv, c: None),
                 planned=lambda yi_: [k for k in yi_["planned_frames"] if k != min(yi_["planned_frames"])])
        self.neg("NG_DISPLAY", base, "G1F1", display_corrupt="DY1")
        # family-shared F: this request's pins plus the other windows' pin frames of the same edit (W118 contamination pattern)
        sib = dict(r[base]["_Fv"])
        for w in ("LOW", "MID", "HIGH"):
            o = f"{s}_{w}_{r[base]['edit_key']}"
            if o != base and o in r and not r[o].get("contract_invalid"):
                sib.update(r[o]["_Fv"])
        self.states["NG_SHARED__Fshared"] = self.C.patched_state(self.v0, {k: (sib[k] if k in sib else r[base]["_ctrl"][k]) for k in self.frames},
                                                                 {"kind": "F_CANDIDATE_FAMILY_SHARED", "request": "NG_SHARED"})
        self.neg("NG_SHARED", base, "G0F0", F_state="NG_SHARED__Fshared")
        for nid, sp in (("NG_REACH", "REACH"), ("NG_DOF", "DOF"), ("NG_CONFLICT", "CONFLICT"), ("NG_WRONG_BINDING", "WRONGBIND")):
            self.neg(nid, f"{s}_{sp}", NEG[nid][1])

    def report(self, with_factors):
        recs = {r: {k: v for k, v in x.items() if not k.startswith("_")} for r, x in self.recs.items()}
        return {"records": recs, "state_ids": {n: st["state_id"] for n, st in sorted(self.states.items())},
                "state_meta": {n: {"parent_id": st["parent_id"], "op": st["op"], "rig_id": st["rig_id"]} for n, st in sorted(self.states.items())}}

    def write_states(self, path):
        self.C.states_file(path, dict(sorted(self.states.items())), "w105")


def frozen_controls_of(src):
    out = {}
    for rid, R in sorted(src.recs.items()):
        out[rid] = {"U": R["U"], "U_state_id": src.states[R["U"]]["state_id"], "F_state_id": src.states[R["F"]]["state_id"],
                    "X_state_id": src.states[R["X"]]["state_id"], "F_statuses": R.get("F_statuses"), "margins": R.get("margins"),
                    "budget_frames": R["x_info"]["budget_frames"], "planned_frames": R["x_info"].get("planned_frames"),
                    "pin_frames": R["pin_frames"], "auth": R["auth"], "validation_errors": R["validation_errors"],
                    "constraint_digests": sorted(c["constraint_digest"] for c in R["request"]["constraints"]), "m_max": R["m_max"]}
    return out


def neg_expectation_consistency():
    import w120_factor_attribution_base as FB
    out = {}
    for nid, (_, f, desc, cond, want, reasons) in NEG.items():
        p = FB.predict_decision(cond)
        out[nid] = {"predicted": p, "declared": want, "ok": [p["decision"], p["failure_class"]] == want}
    return {"controls": out, "all_ok": all(x["ok"] for x in out.values())}


def cmd_calibrate(a):
    import w120_factor_attribution as W20
    out = new_dir(a.out_dir)
    prep = load(a.prepare / "prepare.json")
    rep = {"schema": "w121_calibrate_v1", "scope": "frozen controls U/F_r/X_r (Python only, no factor variant, no Blender candidate "
           "evaluation) + outcome-independent instrument preflight", "sources": {}}
    rep["instrument_preflight"] = W20.fixture_preflight(out / "fixture_store")
    rep["negative_expectations"] = neg_expectation_consistency()
    controls = {}
    for sid in prep["sources"]:
        src = Src(sid, a.prepare)
        src.build_controls()
        base = src.neg_base()
        controls[sid] = frozen_controls_of(src)
        rep["sources"][sid] = {"requests": sorted(src.recs), "neg_base": base,
                               "F_status_summary": {rid: sorted(set((R.get("F_statuses") or {}).values())) for rid, R in src.recs.items()},
                               "margins": {rid: R.get("margins") for rid, R in src.recs.items()}}
    dump(out / "frozen_controls.json", controls)
    rep["frozen_controls_sha256"] = sha(out / "frozen_controls.json")
    rep["neg_source"] = next((sid for sid in prep["sources"] if rep["sources"][sid]["neg_base"]), None)
    dump(out / "calibration.json", rep)
    print(json.dumps({"instrument_all_ok": rep["instrument_preflight"]["all_ok"], "neg_expect_ok": rep["negative_expectations"]["all_ok"],
                      "sources": {s: {"neg_base": v["neg_base"], "status": v["F_status_summary"]} for s, v in rep["sources"].items()},
                      "neg_source": rep["neg_source"]}, indent=1))


# ------------------------------------------------------------------------------------------------ declare ---
CODE_FILES = ["scripts/w116_semantic_pin_base.py", "scripts/w116_semantic_pin_contract.py", "scripts/w116_blender_stages.py",
              "scripts/w117_temporal_pin_base.py", "scripts/w117_temporal_pin.py", "scripts/w118_transition_edit_base.py",
              "scripts/w118_transition_edit.py", "scripts/w119_endpoint_transition_base.py", "scripts/w119_endpoint_transition.py",
              "scripts/w120_factor_attribution_base.py", "scripts/w120_factor_attribution.py", "scripts/w105_rig_utility.py",
              "scripts/w102_video_diagnostic.py", "scripts/w121_independent_transfer_base.py", "scripts/w121_independent_transfer.py"]
DISPLAY_NEG = {"DG1": ("G0F0", "DY1", "first changed out-of-pin frame relabelled UNCHANGED"),
               "DG2": ("G0F1", "DY2", "a changed arm channel row dropped"),
               "DG3": ("G1F0", "DY3", "changed runs collapsed into one envelope"),
               "DG4": ("G1F1", "DY4", "pin frame relabelled as automatic"),
               "DG5": ("G1F0", "DY5", "REJECTed result shown as ACCEPTED")}
GATES = {
    "V1_authentication": "HEAD/blob/branch/lock at declare; W113-W120 Worklogs unchanged with markers; W120 decision unchanged; every new source "
                         "original re-hashes to its acquire SHA-256 and Commons SHA-1; licence free; not the W102 source (SHA + dHash run < 10)",
    "V2_frozen_perception": "the unchanged W102 diagnostic reproduces W102's historical observations/predictions byte-exactly; every new source "
                            "run passes the diagnostic's own checkpoint/config/environment identity checks; observation SHA bound in v0",
    "V3_rig_adapter_and_parent": "the W121 FIXED-slot adapter reproduces W105 real_clip FIXED values on all 199 frames; the W121 parent extraction "
                                 "reproduces W116 v0 values; the W116 keying reproduces the W105 FIXED Action digest; every source parent: FK "
                                 "<= 1e-5 m and left_wrist BOUND",
    "V4_baseline_reproduction": "every request's U/F_r/X_r state ids, statuses, margins, budget and planned frames in both builds equal the frozen "
                                "controls hashed in calibrate",
    "V5_clean_isolated_baselines": "every contract-valid evaluation/control request: F_r value-level and Blender-level isolation",
    "V6_factor_matching": "every contract-valid evaluation/control request and factor: W119 ab_fairness(F_r, X_r, variant) ok and solver digest "
                          "equal; four variants identical (cross_factor_identity); every ZB control budget = pin frames for all four",
    "V7_predeclaration_and_instrument": "declaration hash at every stage; outcomes_inspected false; frozen-controls hash; W120 fixture preflight "
                                        "all_ok (incl. 262,144-case precedence, 0 mismatches); declared negative expectations equal the "
                                        "precedence predictor; code blobs at build equal the declared blobs",
    "V8_independent_blender_evaluation": "every needed state of both builds Blender-evaluated; contrasts, residuals and footprints from Blender",
    "V9_population_complete": "every declared request has F_r/X_r and four variant records; every contract-valid request has every pin frame "
                              "Blender-evaluated for X_r and every variant",
    "V10_repeatability": "two builds byte-identical states files and identical records; two Blender evaluations identical",
    "V11_timing_semantics": "every key CONSTANT at Blender frame = index - f0 + 1; source time = index / source fps (cadence-consistent "
                            "container); Action playback seconds at scene 30 fps reported separately",
    "V12_negative_controls": "every W121 negative control returns its declared decision/class and required reasons; NG_SHARED isolation fails at "
                             "value and Blender level; display controls DG1-DG5 FAIL the truth check; source-level controls (old clip, "
                             "altered bytes, stale source/model binding, wrong FPS, unverified continuity claim, inexact rollback) detected",
    "C1_hard_pin_and_scope": "every ACCEPTed X_r and factor variant: pin-frame Blender residual <= 1 mm; changed channels only authorised; "
                             "changed frames within budget and planned domain; no outside-arm point; root unchanged",
    "C2_truthful_footprint": "every variant passes the display truth check; planned domain within budget and equal across the four variants "
                             "and X_r; preview id = committed id; result label = decision",
    "C3_atomicity_lineage_reversal": "ACCEPT iff all frames pass and fairness/isolation hold; refs unchanged on REJECT; stale parent rejected; "
                                     "ACCEPT + exact revert (values byte-equal, Blender bit-equal) on every source",
    "C4_source_immutability": "original videos, frozen W102/W105/W116-W120 inputs byte-unchanged across every stage",
    "C5_user_edit_and_collateral": "every ACCEPTed variant keeps the user-edit channel and __object__ bit-equal to U",
    "C6_no_fallback": "every variant info fallback None; no G0_REFERENCE_MISSING on an accepted variant; no infeasible request ACCEPTed",
}
VISUAL_PROCEDURE = ("after render and before decide the agent records review/visual_review.json: for the fixed review set (per source "
                    "MID_RBpos, HIGH_RBpos, LOW_none) AND for every factorial-eligible new-source request whose primary GUIDE or FLOOR "
                    "contrast is a meaningful improvement (complete, not sampled), the agent inspects the consecutive-frame sequence sheets "
                    "of U / F_r / X_r / G0F0 / G0F1 / G1F0 / G1F1 next to the source frames and marks per contrast whether the difference is "
                    "visible and whether it looks like improved continuity.  AGENT REVIEW only: no real-time playback claim, no animator claim.")


def code_blobs():
    return {f: git("hash-object", f) for f in CODE_FILES if (ROOT / f).exists()}


def cmd_declare(a):
    out = new_dir(a.out_dir)
    acq, ins, per = load(a.acquire / "acquire.json"), load(a.inspect / "inspect.json"), load(a.perceive / "perceive.json")
    prep, cal = load(a.prepare / "prepare.json"), load(a.calibrate / "calibration.json")
    if sha(a.calibrate / "frozen_controls.json") != cal["frozen_controls_sha256"]:
        raise SystemExit("STOP: frozen controls changed")
    decl = {"schema": "w121_declaration_v1", "batch": BATCH, "instruction_blob": INSTRUCTION_BLOB, "starting_head": STARTING_HEAD,
            "branch": BRANCH, "head_at_declare": git("rev-parse", "HEAD"), "outcomes_inspected": False,
            "dirs": {k: str(getattr(a, k)) for k in ("acquire", "inspect", "perceive", "prepare", "calibrate")},
            "inputs_sha256": {"acquire": sha(a.acquire / "acquire.json"), "inspect": sha(a.inspect / "inspect.json"),
                              "perceive": sha(a.perceive / "perceive.json"), "prepare": sha(a.prepare / "prepare.json"),
                              "calibration": sha(a.calibrate / "calibration.json")},
            "frozen_controls_sha256": cal["frozen_controls_sha256"],
            "sources": {sid: {k: s[k] for k in ("title", "frames", "fps_source", "parent_state_id", "holds")} |
                        {"clip_sha256": acq["candidates"][s["title"]]["download"]["sha256"], "url": acq["candidates"][s["title"]]["url"],
                         "license": acq["candidates"][s["title"]]["license"], "artist": acq["candidates"][s["title"]]["artist"]}
                        for sid, s in prep["sources"].items()},
            "population": {sid: {"requests": cal["sources"][sid]["requests"]} for sid in prep["sources"]},
            "neg_source": cal["neg_source"], "neg_base": cal["sources"][cal["neg_source"]]["neg_base"] if cal["neg_source"] else None,
            "negative_controls": {k: list(v) for k, v in NEG.items()}, "display_controls": {k: list(v) for k, v in DISPLAY_NEG.items()},
            "hold_rule": AB.hold_windows.__doc__, "hold_seconds": AB.HOLD_SECONDS, "hold_edge": AB.HOLD_EDGE, "edits": AB.HOLD_EDITS,
            "factors": __import__("w120_factor_attribution_base").FACTOR_DEF,
            "solver_params_digest": __import__("w120_factor_attribution_base").SOLVER_PARAMS_DIGEST,
            "comparator": "frozen W119 compare_y: B1 <= 0.8 ref and ref - B1 >= 10 mm, no new flip / near-straight, D2/D3/raw boundary step "
                          "not worse by > max(25 %, floor)",
            "contrasts": __import__("w120_factor_attribution_base").CONTRASTS, "gates": GATES, "case_rule": AB.CASE_RULE,
            "visual_procedure": VISUAL_PROCEDURE, "historical_w120_population": "development/history only; not in any W121 denominator",
            "code_blobs": code_blobs(), "selection_independent_of_solver": "acquire/inspect/perceive/prepare/calibrate computed no factor "
            "variant and evaluated no candidate in Blender; hold windows from the UNEDITED parent's Blender wrist path only"}
    dump(out / "declaration.json", decl)
    print(json.dumps({"declaration_sha256": sha(out / "declaration.json"), "sources": list(decl["sources"]), "neg": decl["neg_base"]}))


def check_decl(a):
    if sha(a.declaration) != a.declaration_sha:
        raise SystemExit("STOP: declaration hash mismatch")
    d = load(a.declaration)
    if d["outcomes_inspected"] is not False:
        raise SystemExit("STOP: declaration not pre-outcome")
    return d


# ------------------------------------------------------------------------------------------------ build ---
def build_sources(decl):
    out = {}
    fc = load(Path(decl["dirs"]["calibrate"]) / "frozen_controls.json")
    for sid in decl["sources"]:
        src = Src(sid, decl["dirs"]["prepare"])
        src.build_controls()
        mism = [r for r, x in frozen_controls_of(src).items() if S_canon(x) != S_canon(fc[sid].get(r))]
        for rid in src.req:
            src.make_factors(rid)
        if sid == decl["neg_source"]:
            src.negatives(decl["neg_base"])
        out[sid] = (src, mism)
    return out


def S_canon(x):
    return json.dumps(clean(x), sort_keys=True)


def cmd_build(a):
    out = new_dir(a.out_dir)
    decl = check_decl(a)
    t0 = time.time()
    built = build_sources(decl)
    rep = {"schema": "w121_build_v1", "tag": a.tag, "declaration_sha256": a.declaration_sha, "code_blobs": code_blobs(), "sources": {}}
    for sid, (src, mism) in built.items():
        src.write_states(out / f"{sid}_states.json")
        rep["sources"][sid] = dict(src.report(True), frozen_controls_mismatch=mism, states_sha256=sha(out / f"{sid}_states.json"))
    rep["wall_s"] = time.time() - t0
    dump(out / "build_report.json", rep)
    print(json.dumps({s: {"n_states": len(r["state_ids"]), "mismatch": r["frozen_controls_mismatch"]} for s, r in rep["sources"].items()}))


def cmd_evaluate(a):
    out = new_dir(a.out_dir)
    B = load(a.build_dir / "build_report.json")
    for sid in B["sources"]:
        t0 = time.time()
        blender(ROOT / "scripts/w116_blender_stages.py", "--", "evaluate", "--rig", RIG_FBX, "--rest", RIG_REST, "--states",
                a.build_dir / f"{sid}_states.json", "--out", out / f"{sid}_eval.json", log=out / f"{sid}_eval.log")
        print(sid, "evaluated", round(time.time() - t0, 1), "s")


# ------------------------------------------------------------------------------------------------ assess ---
def source_controls(decl, acq, per, prep):
    """Source-level negative controls (declared expectations: every one DETECTED)."""
    import shutil
    import w116_semantic_pin_base as S
    res = {}
    tmp = BASE / "tmp" / "w121_source_controls"
    tmp.mkdir(parents=True, exist_ok=True)
    # 1. old clip masquerading as a new source: a renamed byte copy and a re-encoded, time-shifted (drop 7 frames) copy
    import cv2
    ren = tmp / "renamed_new_source.mp4"
    if ren.exists():
        os.chmod(ren, 0o666)
        ren.unlink()
    shutil.copyfile(HIST_CLIP, ren)
    _, _, hh, _ = decode(HIST_CLIP)
    cap = cv2.VideoCapture(str(HIST_CLIP))
    fps = cap.get(cv2.CAP_PROP_FPS)
    reenc = tmp / "reencoded_shifted.avi"
    wr = None
    i = 0
    while True:
        ok, im = cap.read()
        if not ok:
            break
        if i >= 7:
            if wr is None:
                wr = cv2.VideoWriter(str(reenc), cv2.VideoWriter_fourcc(*"MJPG"), fps * 0.6, (im.shape[1], im.shape[0]))
            wr.write(im)
        i += 1
    cap.release()
    wr.release()
    _, _, hr, _ = decode(reenc)
    res["old_clip_renamed"] = {"sha_in_excluded": sha(ren) in AB.EXCLUDED_SHA256, "detected": sha(ren) in AB.EXCLUDED_SHA256}
    run = AB.longest_matched_run(hr, hh)
    res["old_clip_reencoded_time_shifted_retimed"] = {"longest_run": run, "detected": run >= AB.DUP_RUN_FRAMES}
    # 2. altered video bytes vs stored SHA
    sid0 = next(iter(decl["sources"]))
    t0 = decl["sources"][sid0]["title"]
    orig = Path(acq["candidates"][t0]["download"]["path"])
    alt = tmp / ("altered" + orig.suffix)
    if alt.exists():
        os.chmod(alt, 0o666)
        alt.unlink()
    b = bytearray(orig.read_bytes())
    b[len(b) // 2] ^= 0xFF
    alt.write_bytes(bytes(b))
    res["altered_bytes"] = {"detected": sha(alt) != decl["sources"][sid0]["clip_sha256"]}
    # 3. stale source/model binding: v0 source must bind this clip's SHA and observation SHA and the frozen checkpoint identity
    v0 = load(Path(decl["dirs"]["prepare"]) / sid0 / "parent_state.json")
    pc = per["candidates"][t0]

    def binding_ok(v, p):
        return (v["source"]["clip_sha256"] == decl["sources"][sid0]["clip_sha256"]
                and v["source"]["observation_prediction_sha256"] == p["observation_prediction_sha256"]
                and p["M1_identity_checkpoints"] == per["candidates"][t0]["M1_identity_checkpoints"] and S.verify_state(v))
    stale = json.loads(json.dumps(v0))
    stale["source"]["observation_prediction_sha256"] = HIST_OBS_SHA
    res["stale_source_binding"] = {"true_binding_ok": binding_ok(v0, pc), "detected": not binding_ok(stale, pc)}
    stale_m = dict(pc, M1_identity_checkpoints={"rtmdet_tiny": "0" * 64, "rtmpose_tiny": "0" * 64})
    res["stale_model_binding"] = {"detected": stale_m["M1_identity_checkpoints"] != per["candidates"][t0]["M1_identity_checkpoints"]}
    # 4. incorrect FPS / time offset: declared source times must equal index / fps of the cadence-consistent container
    fps0 = decl["sources"][sid0]["fps_source"]

    def time_ok(v, f):
        return all(abs(t - k / f) < 1e-9 for t, k in zip(v["source"]["time_s"], v["frames"]))
    res["wrong_fps"] = {"true_ok": time_ok(v0, fps0), "detected": not time_ok(v0, 30000.0) and not time_ok(v0, fps0 * 50 / 30)}
    off = json.loads(json.dumps(v0))
    off["source"]["time_s"] = [t + 0.1 for t in off["source"]["time_s"]]
    res["time_offset"] = {"detected": not time_ok(off, fps0)}
    # 5. unverified subject continuity claimed as verified
    claim = json.loads(json.dumps(v0))
    claim["source"]["video_subject_identity"] = "VERIFIED"
    res["unverified_continuity_claim"] = {"true_is_unknown": v0["source"]["video_subject_identity"].startswith("UNKNOWN"),
                                          "detected": not S.verify_state(claim)}
    return res


def cmd_assess(a):
    import w116_semantic_pin_base as S
    import w118_transition_edit_base as XB
    import w119_endpoint_transition as W9
    import w120_factor_attribution as W20
    C = C_()
    out = new_dir(a.out_dir)
    decl = check_decl(a)
    acq, per, prep = load(Path(decl["dirs"]["acquire"]) / "acquire.json"), load(Path(decl["dirs"]["perceive"]) / "perceive.json"), \
        load(Path(decl["dirs"]["prepare"]) / "prepare.json")
    guard_files = {f"clip_{sid}": acq["candidates"][s["title"]]["download"]["path"] for sid, s in decl["sources"].items()}
    guard_files.update(hist_clip=HIST_CLIP, hist_obs=HIST_OBS, w105_cases=W105_VIDEO_CASES, rig_fbx=RIG_FBX, rig_rest=RIG_REST,
                       w116_v0=W116_V0, w120_decision=Path("C:/Users/dna10/animcv-w120/results/decision/decision.json"))
    g = S.SourceGuard(guard_files)
    Bd = a.build_dir
    Bb = load(Bd / "build_report.json")
    store = S.Store(out / "store")
    rep = {"schema": "w121_assess_v1", "declaration_sha256": a.declaration_sha, "label": "AGENT-RUN Blender evaluation", "requests": {},
           "sources": {}}
    t0 = time.time()
    for sid, B in Bb["sources"].items():
        ev = load(a.eval_dir / f"{sid}_eval.json")
        states = {"real": load(Bd / f"{sid}_states.json")["states"]}
        v0 = load(Path(decl["dirs"]["prepare"]) / sid / "parent_state.json")
        if v0["state_id"] != states["real"]["v0"]["state_id"]:
            raise SystemExit("STOP: parent differs from the build")
        pr = load(Path(decl["dirs"]["prepare"]) / sid / "probe.json")
        rig = C.rig_from(RIG_REST, pr["rig"])
        bound = load(Path(decl["dirs"]["prepare"]) / sid / "bindings.json")["bound"]
        frames = v0["frames"]
        Bshim = {"state_meta": {"real": B["state_meta"]}}
        full_state = lambda side, name, _v0=v0: _v0  # noqa: E731  (every request's parent is this source's v0)
        for rid, rec in B["records"].items():
            A = W20.assess_record(rid, rec, ev, rig, bound, ev["rest"], ev["rig"]["scale"][0], frames, store, full_state, states, Bshim, None)
            A.update(role=rec["role"], side="real", kind=rec["kind"], pin_frames=rec["pin_frames"], pin_runs=rec["pin_runs"],
                     validation_errors=rec["validation_errors"], parent=rec["parent"], source=sid, window=rec.get("window"),
                     edit_key=rec.get("edit_key"), negative_of=rec.get("negative_of"), neg_factor=rec.get("neg_factor"), expect=rec.get("expect"),
                     expect_reasons=rec.get("expect_reasons"), margins=rec.get("margins"), correlation_groups=[f"{sid}_{rec.get('window')}"])
            rep["requests"][rid] = A
        # display controls on this source's neg base
        if sid == decl["neg_source"]:
            dcs = {}
            base = decl["neg_base"]
            rec = B["records"][base]
            for dc, (f, kind, desc) in DISPLAY_NEG.items():
                y = rep["requests"][base]["variants"][f]
                s = W9.corrupt_display(kind, y["display_spec_result"] if kind == "DY5" else y["display_spec_proposed"])
                if s is None:
                    dcs[dc] = {"executed": False, "reason": "corruption not applicable"}
                    continue
                name = rec["factors"][f]["state"]
                chk = XB.check_display(s, C.ev_frames(ev, name), C.ev_frames(ev, rec["U"]), frames, rec["pin_frames"],
                                       ev["states"][name]["state_id"], decision=y["commit"]["decision"] if kind == "DY5" else None)
                dcs[dc] = {"executed": True, "source": base, "factor": f, "desc": desc, "check": chk}
            rep["display_controls"] = dcs
        rep["sources"][sid] = {"lineage": lineage_source(out, sid, B, rep, v0, states, ev),
                               "missing_states": sorted(set(B["state_ids"]) - set(ev["states"])),
                               "timing": timing_semantics(ev, v0, decl["sources"][sid]["fps_source"])}
        del ev
    rep["assess_wall_s"] = time.time() - t0
    rep["source_controls"] = source_controls(decl, acq, per, prep)
    rep["source_guard"] = g.check()
    rep["w120_decision_after"] = {"sha256": sha(Path("C:/Users/dna10/animcv-w120/results/decision/decision.json")),
                                  "case": load(Path("C:/Users/dna10/animcv-w120/results/decision/decision.json"))["case"]}
    dump(out / "assess_report.json", rep)
    print(json.dumps({"n": len(rep["requests"]), "source": rep["source_guard"]["unchanged"], "wall": rep["assess_wall_s"]}))


def timing_semantics(ev, v0, fps):
    """Every keyed F-curve frame equals index - f0 + 1 (CONSTANT keys are audited by the W116 evaluator's stored values)."""
    st = ev["states"]["v0"]
    f0 = v0["frames"][0]
    return {"blender_frame_equals_index_minus_f0_plus_1": all(fr["blender_frame"] == fr["frame_index"] - f0 + 1 for fr in st["frames"]),
            "source_seconds": len(v0["frames"]) / fps, "action_seconds_at_scene_30fps": len(v0["frames"]) / SCENE_FPS,
            "playback_speed_factor_if_played_at_30fps": SCENE_FPS / fps}


def lineage_source(out, sid, B, rep, v0, states, ev):
    """Per source: R1 commit the first ACCEPTed G0F1 of the declared order, R2 an atomic REJECT (NG_PIN_COMBINED where present,
    otherwise the source's REACH request), R3 stale parent, R4 exact revert evaluated in Blender, R5 inexact-rollback control."""
    import w116_semantic_pin_base as S
    C = C_()
    store = S.Store(out / f"lineage_{sid}")
    Rq = rep["requests"]
    recs = B["records"]
    order = [f"{sid}_{w}" for w in NEG_BASE_ORDER]
    dec = lambda rid, f: Rq[rid]["variants"][f]["decision"]  # noqa: E731

    def cand(rid, f):
        name = recs[rid]["factors"][f]["state"]
        meta = B["state_meta"][name]
        st = S.make_state(v0["state_id"], meta["rig_id"], v0["frames"], v0["channels"], states["real"][name]["values"], meta["op"], v0.get("source"))
        if st["state_id"] != states["real"][name]["state_id"]:
            raise SystemExit("STOP: candidate does not rebuild to its id")
        return st
    store.put(v0)
    store.init_ref("main", v0["state_id"])
    store.init_ref("atomic", v0["state_id"])
    res = []
    pick = next((r for r in order if r in Rq and dec(r, "G0F1")["decision"] == "ACCEPT"), None)
    if pick:
        res.append(dict(step="R1", request=pick, variant="G0F1", **store.commit("main", v0["state_id"], cand(pick, "G0F1"), dec(pick, "G0F1"))))
    neg = "NG_PIN_COMBINED" if "NG_PIN_COMBINED" in recs and recs["NG_PIN_COMBINED"].get("source") == sid else f"{sid}_REACH"
    nf = "G0F1" if neg == "NG_PIN_COMBINED" else "G1F0"
    h = store.refs["atomic"]
    r2 = store.commit("atomic", h, cand(neg, nf), dec(neg, nf))
    res.append(dict(step="R2", request=neg, variant=nf, head_unchanged=r2["head_after"] == h, **r2))
    other = next((r for r in order if r in Rq and r != pick), None)
    if other:
        res.append(dict(step="R3", request=other, variant="G1F0", **store.commit("main", v0["state_id"], cand(other, "G1F0"), dec(other, "G1F0"))))
    rev = {}
    if store.refs["main"] != v0["state_id"]:
        rv, rrec = store.revert("main")
        res.append(dict(step="R4", **rrec))
        C.states_file(out / f"{sid}_revert_states.json", {"revert": rv}, "w105")
        blender(ROOT / "scripts/w116_blender_stages.py", "--", "evaluate", "--rig", RIG_FBX, "--rest", RIG_REST, "--states",
                out / f"{sid}_revert_states.json", "--out", out / f"{sid}_revert_eval.json", log=out / f"{sid}_revert_eval.log")
        e = load(out / f"{sid}_revert_eval.json")["states"]["revert"]
        rev = {"value_digest_equal_v0": e["value_digest"] == ev["states"]["v0"]["value_digest"],
               "frames_bit_equal_v0": e["frames"] == ev["states"]["v0"]["frames"]}
        # R5 inexact-rollback control: one quaternion component perturbed by 1e-6 must be detected as NOT a revert
        bad = json.loads(json.dumps(rv["values"]))
        ch = "upperarm_l"
        bad[ch][len(bad[ch]) // 2][0] += 1e-6
        rev["inexact_rollback_detected"] = S.canon(bad) != S.canon(v0["values"])
    return {"programme": res, "store_log": store.log, "refs": store.refs, "lineage_audit": {r: store.lineage(hh) for r, hh in store.refs.items()},
            "revert_blender": rev}


# ------------------------------------------------------------------------------------------------ render ---
VARIANTS_SHOWN = ["U", "F", "X", "G0F0", "G0F1", "G1F0", "G1F1"]


def primary_improvements(A):
    """Factorial-eligible new-source evaluation requests whose primary GUIDE/FLOOR contrast is a meaningful improvement."""
    out = {"GUIDE": [], "FLOOR": []}
    for rid, r in A["requests"].items():
        fx = r["comparisons"].get("factorial")
        if r["role"] != "evaluation" or not fx:
            continue
        for M, cn in (("GUIDE", "guide_at_F0"), ("FLOOR", "floor_at_G0")):
            if fx["contrasts"][cn]["improves"]["meaningful_improvement"]:
                out[M].append(rid)
    return out


def draw_timeline(path, rid, A, fps):
    """W120 draw_timeline layout (same rows / colours / truth source), with this source's true time axis in the footer."""
    import w120_factor_attribution as W20
    W20.draw_timeline(path, rid, A)
    from PIL import Image, ImageDraw
    im = Image.open(path).convert("RGB")
    d = ImageDraw.Draw(im)
    w, h = im.size
    d.rectangle([0, h - 22, w, h], fill=(252, 252, 251))
    d.text((10, h - 18), f"Hatched = budget or REJECTED (never applied). Axis = Action frame index = source frame index; source {fps:.3f} fps "
                         f"(1 frame = {1000 / fps:.1f} ms); W116 scene 30 fps playback is reported separately.", fill=(82, 81, 78))
    im.save(path)


def source_frames(clip, frames):
    import cv2
    cap = cv2.VideoCapture(str(clip))
    want, out, i = set(frames), {}, 0
    while want - set(out):
        ok, im = cap.read()
        if not ok:
            break
        if i in want:
            out[i] = im
        i += 1
    cap.release()
    return out


def cmd_render(a):
    import cv2
    from PIL import Image, ImageDraw
    import w120_factor_attribution as W20
    out = new_dir(a.out_dir)
    decl = check_decl(a)
    Bb = load(a.build_dir / "build_report.json")
    A = load(a.assess_dir / "assess_report.json")
    acq = load(Path(decl["dirs"]["acquire"]) / "acquire.json")
    per = load(Path(decl["dirs"]["perceive"]) / "perceive.json")
    (out / "timelines").mkdir()
    (out / "charts").mkdir()
    for rid, r in A["requests"].items():
        fps = decl["sources"][r["source"]]["fps_source"]
        draw_timeline(out / "timelines" / f"timeline_{rid}.png", rid, A, fps)
        if r["variants"]["F"].get("quality") and r["variants"].get("G0F0", {}).get("quality"):
            W20.draw_charts(out / "charts" / f"steps_{rid}.png", out / "charts" / f"flex_{rid}.png", rid, A)
    imp = primary_improvements(A)
    sel = []
    for sid in decl["sources"]:
        for w in ("MID_RBpos", "HIGH_RBpos", "LOW_none"):
            if f"{sid}_{w}" in A["requests"]:
                sel.append(f"{sid}_{w}")
    for M in imp:
        sel += [r for r in imp[M] if r not in sel]
    views = {"front": [0.0, -3.0, 0.0], "side": [3.0, 0.0, 0.0]}
    for sid in decl["sources"]:
        jobs = []
        B = Bb["sources"][sid]
        for rid in [r for r in sel if A["requests"][r]["source"] == sid]:
            rec = B["records"][rid]
            q = A["requests"][rid]["variants"]["F"].get("quality")
            if not q:
                continue
            lo, hi = q["window"]
            tgt = rec["request"]["constraints"][0]["target"]
            for v in VARIANTS_SHOWN:
                st = rec["U"] if v == "U" else rec["F"] if v == "F" else rec["X"] if v == "X" else rec["factors"][v]["state"]
                yi = rec["factors"][v]["info"]["frames"] if v in W20.FACTORS else {}
                for k in range(lo, hi + 1):
                    mk = []
                    if k in set(rec["pin_frames"]):
                        mk.append({"name": "target", "position": tgt, "color": (0.1, 0.9, 0.2, 1), "radius": 0.013})
                    if str(k) in yi:
                        mk.append({"name": "guide", "position": yi[str(k)]["guide"], "color": (0.85, 0.2, 0.75, 1), "radius": 0.008})
                    for vn, off in views.items():
                        jobs.append({"id": f"{rid}__{v}__{vn}__f{k:04d}", "state": st, "frame_index": k, "markers": mk, "centre": tgt,
                                     "view_offset": off})
        if not jobs:
            continue
        dump(out / f"{sid}_jobs.json", {"jobs": jobs})
        blender(ROOT / "scripts/w116_blender_stages.py", "--", "render", "--rig", RIG_FBX, "--rest", RIG_REST, "--states",
                a.build_dir / f"{sid}_states.json", "--jobs", out / f"{sid}_jobs.json", "--out-dir", out / sid, "--ortho-scale", "1.1",
                "--size", "240", "--hide-mesh", log=out / f"{sid}_render.log")
    # compose: source frame (+ M1 2D) | U | F | X | G0F0 | G0F1 | G1F0 | G1F1, front row + side row
    (out / "sequences").mkdir()
    for rid in sel:
        r = A["requests"][rid]
        sid = r["source"]
        q = r["variants"]["F"].get("quality")
        if not q:
            continue
        lo, hi = q["window"]
        ks = list(range(lo, hi + 1))
        t = decl["sources"][sid]["title"]
        srcf = source_frames(acq["candidates"][t]["download"]["path"], ks)
        z = np.load(Path(decl["dirs"]["perceive"]) / "candidates" / safe(t) / "m1s0" / "observations_predictions.npz")
        W_, H_ = per["candidates"][t]["size"]
        pin = set(r["pin_frames"])
        tiles = []
        for k in ks:
            im = srcf[k].copy()
            x, v = z["input_2d"][k], z["input_valid"][k]
            for (p, c) in ((11, 12), (12, 13), (14, 15), (15, 16), (11, 14), (1, 4), (11, 1), (14, 4)):
                if v[p] and v[c]:
                    cv2.line(im, (int(x[p, 0] * W_), int(x[p, 1] * H_)), (int(x[c, 0] * W_), int(x[c, 1] * H_)),
                             (0, 220, 255) if (p, c) in ((11, 12), (12, 13)) else (200, 200, 200), max(2, W_ // 400), cv2.LINE_AA)
            s = 240 / max(im.shape[:2])
            im = cv2.resize(im, (int(im.shape[1] * s), int(im.shape[0] * s)))
            pad = np.full((240, 240, 3), 25, np.uint8)
            pad[(240 - im.shape[0]) // 2:(240 - im.shape[0]) // 2 + im.shape[0], (240 - im.shape[1]) // 2:(240 - im.shape[1]) // 2 + im.shape[1]] = im
            fr = Image.new("RGB", (240 * (len(VARIANTS_SHOWN) + 1), 2 * 240 + 40), (252, 252, 251))
            dd = ImageDraw.Draw(fr)
            fr.paste(Image.fromarray(cv2.cvtColor(pad, cv2.COLOR_BGR2RGB)), (0, 40))
            dd.text((4, 4), f"source f{k} t={k / decl['sources'][sid]['fps_source']:.3f}s{' PIN' if k in pin else ''}", fill=(11, 11, 11))
            dd.text((4, 20), "M1 2D (left arm yellow)", fill=(82, 81, 78))
            for i, vv in enumerate(VARIANTS_SHOWN):
                for j, vn in enumerate(("front", "side")):
                    p_ = out / sid / f"{rid}__{vv}__{vn}__f{k:04d}.png"
                    if p_.exists():
                        fr.paste(Image.open(p_).convert("RGB"), (240 * (i + 1), 40 + 240 * j))
                cell = r["variants"][vv]["display_spec_proposed"]["cells"].get(str(k), "") if vv != "U" else ""
                dec_ = r["variants"][vv].get("commit", {}).get("decision", "") if vv not in ("U", "F") else ""
                dd.text((240 * (i + 1) + 4, 4), f"{vv} {cell}", fill=(11, 11, 11))
                dd.text((240 * (i + 1) + 4, 20), dec_, fill=(180, 30, 30) if dec_ == "REJECT" else (82, 81, 78))
            tiles.append(fr)
        tiles[0].save(out / "sequences" / f"anim_{rid}.gif", save_all=True, append_images=tiles[1:],
                      duration=int(round(1000 / decl["sources"][sid]["fps_source"])), loop=0)
        # sequence sheet: every frame of the window (front view only), one row per frame
        sh = Image.new("RGB", (120 * (len(VARIANTS_SHOWN) + 1) + 80, 120 * len(ks) + 30), (252, 252, 251))
        ds = ImageDraw.Draw(sh)
        ds.text((4, 4), f"{rid}: rows = consecutive frames; cols = source | " + " | ".join(VARIANTS_SHOWN) + " (front)", fill=(11, 11, 11))
        for i, (k, fr) in enumerate(zip(ks, tiles)):
            ds.text((4, 30 + 120 * i + 50), f"f{k}{' PIN' if k in pin else ''}", fill=(42, 120, 214) if k in pin else (82, 81, 78))
            for j in range(len(VARIANTS_SHOWN) + 1):
                sh.paste(fr.crop((240 * j, 40, 240 * (j + 1), 280)).resize((120, 120)), (80 + 120 * j, 30 + 120 * i))
        sh.save(out / "sequences" / f"sheet_{rid}.png")
    dump(out / "render_selection.json", {"selection": sel, "primary_improvements": imp, "views": views, "variants": VARIANTS_SHOWN})
    print(json.dumps({"selection": sel, "improvements": imp}))


# ------------------------------------------------------------------------------------------------ decide ---
W120_DECISION = Path("C:/Users/dna10/animcv-w120/results/decision/decision.json")
W120_DECISION_PREFIX = "1029ce925b0a5a80"
HIST_WORKLOGS = ["docs/113_WORKLOG_ANIMATION_BASE_CONTRACT_FEASIBILITY.md", "docs/114_WORKLOG_SOURCE_PRESERVING_ANIMATION_BASE_EDIT_AUTHORITY.md",
                 "docs/115_WORKLOG_ANIMATION_BASE_VALIDITY_REPEATABILITY_RESOLUTION_AUTHORITY.md",
                 "docs/116_WORKLOG_SEMANTIC_WORLD_POSITION_PIN_FEASIBILITY.md", "docs/117_WORKLOG_SEMANTIC_WORLD_PIN_TEMPORAL_COHERENCE.md",
                 "docs/118_WORKLOG_AUTOMATIC_TRANSITION_EDIT_FOOTPRINT_VISIBILITY.md",
                 "docs/119_WORKLOG_ENDPOINT_SPACE_TRANSITION_ISOLATED_BASELINES.md", "docs/120_WORKLOG_MATCHED_GUIDE_FLEX_FLOOR_ATTRIBUTION.md"]


def summary_table(A, Bb):
    import w120_factor_attribution_base as FB
    rows = {}
    for rid, r in A["requests"].items():
        if r["role"] == "negative":
            continue
        rec = Bb["sources"][r["source"]]["records"][rid]
        row = {"source": r["source"], "window": r.get("window"), "edit": r.get("edit_key"), "role": r["role"], "pin_frames": r["pin_runs"],
               "F_statuses": sorted(set((rec.get("F_statuses") or {}).values())), "margins": r.get("margins"),
               "validation_errors": r["validation_errors"], "variants": {}}
        for v in ["X"] + list(FB.FACTORS):
            x = r["variants"].get(v, {})
            q = x.get("quality") or {}
            row["variants"][v] = {"decision": x.get("decision", {}).get("decision"), "class": x.get("decision", {}).get("failure_class"),
                                  "reasons": x.get("decision", {}).get("reasons"), "target_class": x.get("class"),
                                  "pin_residual_max_m": (x.get("residual_m") or {}).get("max"),
                                  "B1_m": q.get("B1_boundary_wrist_comp_step_peak_m"), "D2_m": q.get("D2_elbow_comp_step_peak_m"),
                                  "D3_deg": q.get("D3_arm_rot_step_excess_peak_deg"), "raw_boundary_m": q.get("boundary_raw_wrist_step_peak_m"),
                                  "U_raw_boundary_m": q.get("boundary_control_raw_wrist_step_peak_m"), "min_flex_deg": q.get("min_flex_deg"),
                                  "new_flips_vs_U": q.get("introduced_flip_transitions"), "near_straight_vs_U": q.get("A4_introduced_near_straight_frames"),
                                  "mean_elbow_dev_U_m": q.get("cost_mean_elbow_dev_from_U_m"), "status_counts": (x.get("info") or {}).get("status_counts"),
                                  "flag_counts": (x.get("info") or {}).get("flag_counts"),
                                  "changed_frames": len((x.get("footprint") or {}).get("frames_changed", [])),
                                  "endpoint_jumps": x.get("endpoint_jumps")}
        fx = r["comparisons"].get("factorial")
        if fx:
            row["contrasts"] = {cn: {"dB1_m": c["delta"]["B1_boundary_wrist_comp_step_peak_m"], "rel": c["B1_relative_to_reference"],
                                     "meaningful": c["improves"]["meaningful_improvement"], "worse": c["improves"]["worse"],
                                     "new_flips": c["new_flips"], "new_near_straight": c["new_near_straight"],
                                     "reverse_meaningful": c["reverse"]["meaningful_improvement"]}
                                for cn, c in fx["contrasts"].items()}
            row["interaction_B1_m"] = fx["interaction"]["delta"]["B1_boundary_wrist_comp_step_peak_m"]
        if "vs_X" in r["comparisons"]:
            row["vs_X_meaningful"] = {f: c["meaningful_improvement"] for f, c in r["comparisons"]["vs_X"].items()}
        if "G0F0_vs_X_blender" in r["comparisons"]:
            g = r["comparisons"]["G0F0_vs_X_blender"]
            row["G0F0_vs_X"] = {"max_wrist_m": g["max_wrist_m"], "max_elbow_m": g["max_elbow_m"], "max_rot_deg": g["max_rot_deg"]}
        row["eligible_factorial"] = r.get("eligible_factorial")
        rows[rid] = row
    return rows


def cmd_decide(a):
    import w120_factor_attribution_base as FB
    out = new_dir(a.out_dir)
    decl = check_decl(a)
    per = load(Path(decl["dirs"]["perceive"]) / "perceive.json")
    prep = load(Path(decl["dirs"]["prepare"]) / "prepare.json")
    cal = load(Path(decl["dirs"]["calibrate"]) / "calibration.json")
    ins = load(Path(decl["dirs"]["inspect"]) / "inspect.json")
    acq = load(Path(decl["dirs"]["acquire"]) / "acquire.json")
    A = load(a.assess_dir / "assess_report.json")
    B1, B2 = load(a.build_dir / "build_report.json"), load(a.build_dir2 / "build_report.json")
    VR = load(a.visual_review)
    Rq = A["requests"]
    g, det = {}, {}
    srcs = list(decl["sources"])
    data_blocked = not srcs
    # ---------------- V gates
    wl_ok = all(subprocess.run(["git", "diff", "--quiet", STARTING_HEAD, "--", f], cwd=ROOT).returncode == 0
                and "AUTOMATION_BATCH_STATUS: COMPLETE" in (ROOT / f).read_text(encoding="utf-8") for f in HIST_WORKLOGS)
    v1 = {"head_blob": decl["instruction_blob"] == INSTRUCTION_BLOB and decl["starting_head"] == STARTING_HEAD,
          "worklogs_unchanged": wl_ok, "w120_decision": A["w120_decision_after"]["sha256"].startswith(W120_DECISION_PREFIX)
          and A["w120_decision_after"]["case"] == "B", "source_guard": A["source_guard"]["unchanged"]}
    for sid, s in decl["sources"].items():
        c = acq["candidates"][s["title"]]
        v1[f"{sid}_sha"] = sha(c["download"]["path"]) == s["clip_sha256"] and c["download"]["sha1_matches_commons"]
        v1[f"{sid}_licence"] = AB.license_ok(c["license"])
        v1[f"{sid}_not_historical"] = s["clip_sha256"] not in AB.EXCLUDED_SHA256 and ins["candidates"][s["title"]]["dup_vs_historical_longest_run"] < AB.DUP_RUN_FRAMES
    det["V1"] = v1
    g["V1_authentication"] = all(v1.values())
    v2 = {"historical_reproduction": per["historical_reproduction"]["ok"]}
    for sid, s in decl["sources"].items():
        pc = per["candidates"][s["title"]]
        v0 = load(Path(decl["dirs"]["prepare"]) / sid / "parent_state.json")
        v2[sid] = pc["frozen_returncode"] == 0 and v0["source"]["observation_prediction_sha256"] == pc["observation_prediction_sha256"]
    det["V2"] = v2
    g["V2_frozen_perception"] = all(v2.values())
    v3 = {"adapter": per["rig_adapter_fidelity"]["ok"], "parent_values": prep["parent_value_fidelity_w116_v0"],
          "keying": prep["historical_keying_reproduces_w105_fixed"], "rig_identity": prep["rig"]["identity_ok"]}
    for sid in srcs:
        v3[sid] = prep["sources"][sid]["fk_ok"] and prep["sources"][sid]["left_wrist_bound"]
    det["V3"] = v3
    g["V3_rig_adapter_and_parent"] = all(v3.values())
    det["V4"] = {sid: [B1["sources"][sid]["frozen_controls_mismatch"], B2["sources"][sid]["frozen_controls_mismatch"]] for sid in srcs}
    g["V4_baseline_reproduction"] = not any(x for v in det["V4"].values() for x in v)
    pop = [r for r, x in Rq.items() if x["role"] in ("evaluation", "control")]
    valid = [r for r in pop if not Rq[r]["validation_errors"]]
    recs = {r: B1["sources"][Rq[r]["source"]]["records"][r] for r in Rq}
    iso_bad = [r for r in valid if not (recs[r]["isolation_values_F_r"]["ok"] and Rq[r]["variants"]["F"]["isolation_blender"]["ok"])]
    det["V5"] = {"isolation_failed": iso_bad}
    g["V5_clean_isolated_baselines"] = not iso_bad
    unfair, cross = [], []
    for r in valid:
        for f in FB.FACTORS:
            if not Rq[r]["variants"][f]["fairness"]["ok"]:
                unfair.append(f"{r}/{f}")
        ci = FB.cross_factor_identity({f: recs[r]["factors"][f]["fair"] for f in FB.FACTORS})
        if not ci["ok"]:
            cross.append({r: ci["errors"]})
    zb = {sid: all(recs[f"{sid}_ZB"]["factors"][f]["info"]["budget_frames"] == recs[f"{sid}_ZB"]["pin_frames"] for f in FB.FACTORS)
          for sid in srcs}
    det["V6"] = {"unfair": unfair, "cross_factor": cross, "ZB_budget_is_pin": zb}
    g["V6_factor_matching"] = not unfair and not cross and all(zb.values())
    blobs_ok = B1["code_blobs"] == decl["code_blobs"] and B2["code_blobs"] == decl["code_blobs"]
    det["V7"] = {"instrument": cal["instrument_preflight"]["all_ok"], "neg_expect": cal["negative_expectations"]["all_ok"],
                 "precedence_n": cal["instrument_preflight"]["precedence_exhaustive"]["n"],
                 "precedence_mismatches": cal["instrument_preflight"]["precedence_exhaustive"]["n_mismatches"], "code_blobs_equal": blobs_ok,
                 "frozen_controls_hash": sha(Path(decl["dirs"]["calibrate"]) / "frozen_controls.json") == decl["frozen_controls_sha256"],
                 "declaration_hash": A["declaration_sha256"] == a.declaration_sha}
    g["V7_predeclaration_and_instrument"] = decl["outcomes_inspected"] is False and all(det["V7"][k] for k in
                                                                                        ("instrument", "neg_expect", "code_blobs_equal", "frozen_controls_hash", "declaration_hash"))
    e2 = {sid: set(load(a.eval_dir2 / f"{sid}_eval.json")["states"]) for sid in srcs}
    det["V8"] = {sid: {"e1_missing": A["sources"][sid]["missing_states"], "e2_missing": sorted(set(B1["sources"][sid]["state_ids"]) - e2[sid])}
                 for sid in srcs}
    g["V8_independent_blender_evaluation"] = not any(v["e1_missing"] or v["e2_missing"] for v in det["V8"].values())
    miss = []
    for sid in srcs:
        for rid in decl["population"][sid]["requests"]:
            if rid not in Rq or not {"F", "X", *FB.FACTORS} <= set(Rq[rid]["variants"]):
                miss.append(rid)
            elif not Rq[rid]["validation_errors"]:
                for v in ["X", *FB.FACTORS]:
                    if Rq[rid]["variants"][v]["residual_m"]["n"] != len(Rq[rid]["pin_frames"]) * len(recs[rid]["request"]["constraints"]):
                        miss.append(f"{rid}/{v}")
    det["V9"] = {"missing": miss}
    g["V9_population_complete"] = not miss
    rep10 = {sid: {"states_sha_equal": B1["sources"][sid]["states_sha256"] == B2["sources"][sid]["states_sha256"],
                   "records_equal": S_canon(B1["sources"][sid]["records"]) == S_canon(B2["sources"][sid]["records"]),
                   "eval_equal": sha(a.eval_dir / f"{sid}_eval.json") == sha(a.eval_dir2 / f"{sid}_eval.json")} for sid in srcs}
    det["V10"] = rep10
    g["V10_repeatability"] = all(all(v.values()) for v in rep10.values())
    det["V11"] = {sid: dict(A["sources"][sid]["timing"], cadence_reasons=ins["candidates"][decl["sources"][sid]["title"]]["reasons"]) for sid in srcs}
    g["V11_timing_semantics"] = all(v["blender_frame_equals_index_minus_f0_plus_1"] and not v["cadence_reasons"] for v in det["V11"].values())
    negs = {}
    for nid, (_, f, desc, cond, want, reasons) in NEG.items():
        if nid not in Rq:
            negs[nid] = {"executed": False, "ok": False}
            continue
        d = Rq[nid]["variants"][f]["decision"]
        ok = [d["decision"], d["failure_class"]] == want and set(reasons) <= set(d["reasons"])
        if nid == "NG_SHARED":
            ok = ok and not recs[nid]["isolation_values_F_r"]["ok"] and not Rq[nid]["variants"]["F"]["isolation_blender"]["ok"]
        negs[nid] = {"got": d, "want": want, "want_reasons": reasons, "ok": ok}
    dcs = {k: {"executed": v["executed"], "failed_truth_check": v["executed"] and not v["check"]["ok"]} for k, v in A.get("display_controls", {}).items()}
    sc = {k: v["detected"] for k, v in A["source_controls"].items()}
    rb = {sid: A["sources"][sid]["lineage"]["revert_blender"].get("inexact_rollback_detected") for sid in srcs}
    det["V12"] = {"negatives": negs, "display_controls": dcs, "source_controls": sc, "inexact_rollback_detected": rb}
    g["V12_negative_controls"] = (all(x["ok"] for x in negs.values()) and all(x["failed_truth_check"] for x in dcs.values() if x["executed"])
                                  and sum(x["executed"] for x in dcs.values()) >= 4 and all(sc.values()) and all(rb.values()))
    # ---------------- C gates
    c1, c2, c5, c6 = [], [], [], []
    for r in pop:
        R = Rq[r]
        planned = {}
        for v in ["X", *FB.FACTORS]:
            x = R["variants"][v]
            fp = x["footprint"]
            planned[v] = x["planned_frames"]
            if not set(x["planned_frames"]) <= set(x["budget_frames"]):
                c2.append(f"{r}/{v}:planned_not_in_budget")
            if not x["display_check_proposed"]["ok"] or not x.get("display_check_result", {}).get("ok", False):
                c2.append(f"{r}/{v}:display")
            if not x.get("commit", {}).get("preview_id_equals_candidate", False):
                c2.append(f"{r}/{v}:preview_id")
            if v != "X" and (x.get("info") or {}).get("fallback") is not None:
                c6.append(f"{r}/{v}:fallback")
            if x["decision"]["decision"] == "ACCEPT":
                if x["residual_m"]["n_over_tol"] or not all(x["frame_pass"].values()):
                    c1.append(f"{r}/{v}:pin_or_frame")
                if not (set(fp["frames_changed"]) <= set(x["budget_frames"]) and set(fp["frames_changed"]) <= set(x["planned_frames"])
                        and set(fp["channel_frames"]) <= set(R["variants"][v].get("info", {}).get("auth", recs[r]["auth"]))
                        and not fp["outside_arm_points_changed"] and not fp["matrix_world_changed_frames"]):
                    c1.append(f"{r}/{v}:scope")
                if not x["user_edit_and_object_equal_U"]:
                    c5.append(f"{r}/{v}")
                if set((recs[r].get("F_statuses") or {}).values()) - {"SOLVED", "NO_CHANGE_NEEDED"}:
                    c6.append(f"{r}/{v}:infeasible_accepted")
                if v != "X" and (x.get("info") or {}).get("request_status") == "G0_REFERENCE_MISSING":
                    c6.append(f"{r}/{v}:g0_missing")
        if not R["validation_errors"] and len({json.dumps(p) for p in planned.values()}) != 1:
            c2.append(f"{r}:planned_differs_across_variants")
    det["C1"], det["C2"], det["C5"], det["C6"] = c1, c2, c5, c6
    g["C1_hard_pin_and_scope"] = not c1
    g["C2_truthful_footprint"] = not c2
    c3 = {}
    for sid in srcs:
        L = A["sources"][sid]["lineage"]
        st = {p["step"]: p for p in L["programme"]}
        c3[sid] = {"R1_accept_moves_head": "R1" not in st or (st["R1"]["decision"] == "ACCEPT" and st["R1"]["head_after"] != st["R1"]["head_before"]),
                   "R1_present": "R1" in st,
                   "R2_reject_head_unchanged": st["R2"]["decision"] == "REJECT" and st["R2"]["head_unchanged"],
                   "R3_stale": "R3" not in st or st["R3"]["reasons"] == ["STALE_PARENT"] or "R1" not in st,
                   "R4_revert_exact": "R4" not in st or (st["R4"]["values_equal_restored"] and L["revert_blender"].get("value_digest_equal_v0")
                                                         and L["revert_blender"].get("frames_bit_equal_v0"))}
    det["C3"] = c3
    g["C3_atomicity_lineage_reversal"] = all(all(x for k, x in v.items() if k != "R1_present") for v in c3.values())   # R1 absence = coverage note
    g["C4_source_immutability"] = A["source_guard"]["unchanged"]
    g["C5_user_edit_and_collateral"] = not c5
    g["C6_no_fallback"] = not c6
    # ---------------- mechanisms (new-source evaluation requests only)
    mech = {}
    for M, (pc_, cc_) in {"GUIDE": ("guide_at_F0", "guide_at_F1"), "FLOOR": ("floor_at_G0", "floor_at_G1")}.items():
        pi, ci, nd = [], [], []
        for r, R in Rq.items():
            fx = R["comparisons"].get("factorial")
            if R["role"] != "evaluation" or not fx:
                continue
            c = fx["contrasts"][pc_]
            if c["improves"]["meaningful_improvement"]:
                pi.append((R["source"], R["window"], r))
            if fx["contrasts"][cc_]["improves"]["meaningful_improvement"]:
                ci.append((R["source"], R["window"], r))
            if c["new_flips"] or c["new_near_straight"]:
                nd.append(r)
        mech[M] = {"primary_improved": pi, "conditional_improved": ci, "primary_new_defects": nd,
                   "visible": bool(VR.get("mechanisms", {}).get(M, {}).get("visible_improvement"))}
    n_acc = len(srcs)
    case = AB.case_rule({k: v for k, v in g.items() if k.startswith("V")}, {k: v for k, v in g.items() if k.startswith("C")}, mech, n_acc,
                        data_blocked)
    elig = sorted(r for r, R in Rq.items() if R["role"] == "evaluation" and R["comparisons"].get("factorial"))
    rep = {"schema": "w121_decision_v1", "declaration_sha256": a.declaration_sha, "case": case["case"], "case_detail": case, "gates": g,
           "detail": det, "mechanisms": mech, "factorial_eligible_requests": elig,
           "population": {"sources": srcs, "evaluation_requests": sorted(r for r in pop if Rq[r]["role"] == "evaluation"),
                          "control_requests": sorted(r for r in pop if Rq[r]["role"] == "control")},
           "visual_review_sha256": sha(a.visual_review), "summary": summary_table(A, B1), "label": "AGENT-RUN; not Reviewer-verified"}
    dump(out / "decision.json", rep)
    print(json.dumps({"case": case, "gates": g}, indent=1, default=str)[:6000])


def cmd_manifest(a):
    files = {}
    for p in sorted(BASE.rglob("*")):
        if p.is_file() and "tmp" not in p.relative_to(BASE).parts[:1] and "cache" not in p.relative_to(BASE).parts[:1] \
                and p.name != "evidence_manifest.json":
            files[str(p.relative_to(BASE)).replace("\\", "/")] = sha(p)
    dump(BASE / "evidence_manifest.json", {"schema": "w121_manifest_v1", "declaration_sha256": a.declaration_sha, "n_files": len(files),
                                           "files": files})
    print(len(files), sha(BASE / "evidence_manifest.json"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage")
    for n in ("out_dir", "acquire", "inspect", "perceive", "prepare", "calibrate", "declaration", "build_dir", "eval_dir", "build_dir2",
              "eval_dir2", "assess_dir", "visual_review", "render_dir"):
        ap.add_argument("--" + n.replace("_", "-"), type=Path)
    ap.add_argument("--declaration-sha")
    ap.add_argument("--tag")
    a = ap.parse_args()
    {"acquire": cmd_acquire, "inspect": cmd_inspect, "perceive": cmd_perceive, "prepare": cmd_prepare,
     "calibrate": cmd_calibrate, "declare": cmd_declare, "build": cmd_build, "evaluate": cmd_evaluate, "assess": cmd_assess,
     "render": cmd_render, "decide": cmd_decide, "manifest": cmd_manifest}[a.stage](a)


if __name__ == "__main__":
    main()
