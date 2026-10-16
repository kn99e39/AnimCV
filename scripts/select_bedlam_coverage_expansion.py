#!/usr/bin/env python3
"""DIAGNOSTIC (Worklog 85): the ONE predeclared BEDLAM coverage-expansion population.

Input: the Worklog 85 row audit (``audit_bedlam_training_coverage.py``).  Output: an immutable manifest of
NEW training rows (label file, label row) plus a coverage report comparing the historical Worklog 73 TRAIN
bank, the eligible unused pool and the selection.  GT pose is used here ONLY to choose training samples
(allowed for TRAIN); it never enters inference.  No model output is read.

Policy (``POLICY``, declared before any expanded-bank model exists):
  1. eligible new row = image present in the on-disk tar AND person recovered (Worklog 73 ``SceneBodyIndex``)
     AND a TRAINING subject (never a BEDLAM validation subject) AND valid GT left arm AND the GT left
     shoulder/elbow/wrist project inside the image AND never selected by Worklog 73 (rows it selected and
     then dropped stay excluded);
  2. near-duplicate removal inside each (scene, sequence, person) track, in frame order, seeded with the
     track's historical TRAIN rows: a row is kept only if its normalized arm differs from every kept row of
     the track by ||dE|| + ||dW|| >= 0.15 (15% of the arm length, camera frame -- so camera motion counts);
  3. coverage cells = GT bend {[0,15), [15,45), [45,90), [90,180]} x |f| {[0,sin10), [sin10,0.5), [0.5,1]}
     x sign(f) x upper-arm elevation E_z/|E| {< -0.5, [-0.5,0.5], > 0.5} (72 cells);
  4. NEW_ROWS new rows are allocated to cells in proportion to sqrt(deduplicated pool count) (water-filled
     when a cell runs out) and drawn uniformly inside each cell with ``SEED`` -- denser than the historical
     bank everywhere, relatively more for rare arm configurations, never inventing a configuration.
The expanded BEDLAM TRAIN population = all 21,232 historical TRAIN rows + the new rows that survive the
unchanged Worklog 73 detector association / visibility rules at bank build.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

POLICY = {"new_rows": 80000, "dedup_min_arm_distance": 0.15, "seed": 1337,
          "bend_bins": [0, 15, 45, 90, 181], "abs_f_bins": [0.0, math.sin(math.radians(10)), 0.5, 1.01],
          "elevation_bins": [-1.01, -0.5, 0.5, 1.01], "allocation": "sqrt(deduplicated pool count), water-filled",
          "budget": "about 80k new rows: ~3 h frozen Qwen extraction (7.7 rows/s measured in Worklog 73) plus "
                    "detector passes on RTX 3080 Ti; ~125 GB token grids and ~120 GB re-encoded images on /home"}
NEW_ROWS = POLICY["new_rows"]
SEED = POLICY["seed"]


def cells(elbow: np.ndarray, wrist: np.ndarray, bend: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Coverage cell id in [0, 72) from the normalized GT arm (selection/reporting only)."""
    b = np.clip(np.digitize(bend, POLICY["bend_bins"]) - 1, 0, 3)
    a = np.clip(np.digitize(np.abs(f), POLICY["abs_f_bins"]) - 1, 0, 2)
    s = (f >= 0).astype(int)
    elev = elbow[:, 2] / np.maximum(np.linalg.norm(elbow, axis=1), 1e-12)
    e = np.clip(np.digitize(elev, POLICY["elevation_bins"]) - 1, 0, 2)
    return ((b * 3 + a) * 2 + s) * 3 + e


def dedup_tracks(track: np.ndarray, frame: np.ndarray, elbow: np.ndarray, wrist: np.ndarray, candidate: np.ndarray,
                 seed_rows: np.ndarray, threshold: float) -> np.ndarray:
    """Boolean keep-mask over candidates: greedy in frame order inside each track, seeded by ``seed_rows``."""
    keep = np.zeros(len(track), bool)
    order = np.lexsort((frame, track))
    kept_e: dict = {}
    for i in order:
        t = track[i]
        if not (candidate[i] or seed_rows[i]):
            continue
        ke = kept_e.setdefault(t, [])
        if seed_rows[i]:
            ke.append((elbow[i], wrist[i]))
            continue
        if all(np.linalg.norm(elbow[i] - e) + np.linalg.norm(wrist[i] - w) >= threshold for e, w in ke):
            keep[i] = True
            ke.append((elbow[i], wrist[i]))
    return keep


def allocate(pool_counts: np.ndarray, total: int) -> np.ndarray:
    """Integer quotas proportional to sqrt(count), capped by availability, leftover water-filled."""
    quota = np.zeros(len(pool_counts), int)
    remaining = min(total, int(pool_counts.sum()))
    open_ = pool_counts > 0
    while remaining > 0 and open_.any():
        w = np.sqrt(np.where(open_, pool_counts, 0).astype(float))
        share = np.floor(remaining * w / w.sum()).astype(int)
        if share.sum() == 0:
            share[np.argmax(np.where(open_, pool_counts - quota, -1))] = 1
        share = np.minimum(share, pool_counts - quota)
        quota += share
        remaining = min(total, int(pool_counts.sum())) - int(quota.sum())
        open_ = pool_counts - quota > 0
    return quota


def distribution(mask: np.ndarray, cell: np.ndarray) -> dict:
    c = np.bincount(cell[mask], minlength=72).astype(float)
    p = c / max(c.sum(), 1)
    nz = p[p > 0]
    return {"rows": int(mask.sum()), "occupied_cells": int((c > 0).sum()), "cells_with_100_rows": int((c >= 100).sum()),
            "entropy_bits": float(-(nz * np.log2(nz)).sum()), "counts": c.astype(int).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit("STOP: output directory is not empty; never overwrite experimental artifacts")
    z = np.load(args.audit / "rows.npz", allow_pickle=False)
    n = len(z["rgb"])
    track = np.array([f"{s}|{q}|{p}" for s, q, p in zip(z["scene"], z["sequence"], z["person"])])
    elbow, wrist = z["elbow"].astype(np.float64), z["wrist"].astype(np.float64)
    cell = cells(elbow, wrist, z["bend"], z["f"])
    identity_ok = z["identity"] == "ok"
    training_subject = identity_ok & ~z["validation_subject"]
    eligible = z["rgb"] & training_subject & z["gt_arm_ok"] & z["arm_in_image"] & ~z["hist_selected"]
    hist = z["hist_final"]
    if (hist & z["validation_subject"]).any():
        raise SystemExit("STOP: a historical TRAIN row belongs to a validation subject")
    deduped = dedup_tracks(track, z["frame"], elbow, wrist, eligible, hist, POLICY["dedup_min_arm_distance"])
    pool_counts = np.bincount(cell[deduped], minlength=72)
    quota = allocate(pool_counts, NEW_ROWS)
    rng = np.random.default_rng(SEED)
    chosen = np.zeros(n, bool)
    for c in range(72):
        idx = np.flatnonzero(deduped & (cell == c))
        if quota[c]:
            chosen[rng.choice(idx, size=quota[c], replace=False)] = True
    expanded = hist | chosen
    scene_names = z["scene_names"].tolist()
    rows = sorted((int(z["scene"][i]), int(z["label_row"][i])) for i in np.flatnonzero(chosen))
    manifest = {"schema": "animcv_w85_bedlam_expansion_manifest_v1", "policy": POLICY,
                "audit_rows_sha256": hashlib.sha256((args.audit / "rows.npz").read_bytes()).hexdigest(),
                "label_tags": z["label_tags"].tolist(), "scene_names": scene_names,
                "new_rows": [[scene_names[s], r] for s, r in rows]}
    text = json.dumps(manifest)
    manifest["manifest_sha256"] = hashlib.sha256(text.encode()).hexdigest()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "expansion_manifest.json").write_text(json.dumps(manifest))

    def track_stats(mask):
        return {"tracks": int(len(set(track[mask].tolist()))), "sequences": int(len({(a, b) for a, b in zip(z["scene"][mask], z["sequence"][mask])})),
                "persons": int(len(set(z["person"][mask].tolist()))), "subjects": int(len(set(z["subject"][mask].tolist()))),
                "scenes": int(len(set(z["scene"][mask].tolist())))}
    groups = {"historical_train": hist, "eligible_unused": eligible, "eligible_after_dedup": deduped,
              "selected_new": chosen, "expanded_train": expanded}
    report = {"manifest_sha256": manifest["manifest_sha256"], "policy": POLICY,
              "exclusions": {"rows": n, "no_rgb": int((~z["rgb"]).sum()), "identity_unrecovered": int((~identity_ok).sum()),
                             "validation_subject": int(z["validation_subject"].sum()), "gt_arm_invalid": int((~z["gt_arm_ok"]).sum()),
                             "arm_outside_image": int((~z["arm_in_image"]).sum()), "historically_selected": int(z["hist_selected"].sum()),
                             "eligible_unused": int(eligible.sum()), "removed_as_near_duplicate": int((eligible & ~deduped).sum())},
              "groups": {k: {**distribution(m, cell), **track_stats(m),
                             "abs_f_ge_0.5": float((np.abs(z["f"][m]) >= 0.5).mean()) if m.any() else None,
                             "f_negative": float((z["f"][m] < 0).mean()) if m.any() else None,
                             "bend_bins": np.bincount(np.clip(np.digitize(z["bend"][m], POLICY["bend_bins"]) - 1, 0, 3), minlength=4).tolist(),
                             "pitch_p05_p50_p95": np.percentile(z["pitch"][m], [5, 50, 95]).round(1).tolist() if m.any() else None,
                             "scene_rows": np.bincount(z["scene"][m], minlength=len(scene_names)).tolist()}
                         for k, m in groups.items()},
              "quota": quota.tolist(), "deduplicated_pool_counts": pool_counts.tolist(),
              "historical_cells_below_20_rows": int((np.bincount(cell[hist], minlength=72) < 20).sum()),
              "expanded_cells_below_20_rows": int((np.bincount(cell[expanded], minlength=72) < 20).sum()),
              "per_track_rows": {k: dict(Counter(Counter(track[m].tolist()).values()).most_common(8)) for k, m in
                                 (("historical_train", hist), ("expanded_train", expanded))}}
    (args.out / "coverage_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({"exclusions": report["exclusions"], "manifest_sha256": report["manifest_sha256"],
                      **{k: {kk: v[kk] for kk in ("rows", "occupied_cells", "cells_with_100_rows", "entropy_bits", "tracks",
                                                   "persons", "subjects", "scenes", "abs_f_ge_0.5", "bend_bins")}
                         for k, v in report["groups"].items()}}, indent=1))


if __name__ == "__main__":
    main()
