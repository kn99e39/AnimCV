#!/usr/bin/env python3
"""W103: 2D observability ceiling vs frozen-estimator failure (diagnostic only).

Stages (each refuses to overwrite its output directory):
  verify     frozen W102 artifacts, model-visible input reproduction, derived-token audit (no 3D)
  reproduce  same-stack S0 forward of B_Main/M_Main bytes + batch-one single-frame check (no GT)
  declare    2D-only similarity resolution and the full observability contract, hashed BEFORE any 3D
  analyze    3D observability / attribution / temporal / HMR / Mmask accounting under the declaration

No production module, checkpoint, threshold, mask, crop or normalization is changed. GT is evaluation-only and is
never placed in an S0 input. The neighbor-GT probe is a reference-population diagnostic, not an estimator candidate.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"), str(ROOT/"scripts")]

# ------------------------------------------------------------------ frozen identities (W102 worklog / manifest)
W102_MANIFEST_SHA = "5186a17075a1fa58382ae0c9a5233d788b8b6cda45ee435fa2554f39e1489e9f"
S0_CHECKPOINT_SHA = "6c12fea9f84ea0493fada8b9acc96cd5c70fdd5d9114603e954f9412f4247b44"
BANK_DIGEST = "75519e6394a764e3749ddaa30555b58b73a01db582ccee14f661374b9a0ed536"
POPULATION_SHA = "533895c980a080985444b703f43e0f64046391c76909dea5a2f305a26b34cdb9"
HMR_SHA = "fc5d330565205f4792cd3641671633decbab4e24d70522d4dd85f045d87a7003"
OBSERVATIONS_SHA = "7fc15fc8a575358b71fdff1c5e61470e92dc51028a98442b273d09b55bd5a0c2"
SIDE_SHA = "d0e3a8796d8e4aad08df311b2ce7da818be46d4f7f6e287371c771845ba1b276"
ENVIRONMENT_FINGERPRINT = "cd68f5389a8eb4a01db9ee258ec66d38c638f46fd2b1fb3506cdc85f190b13c9"
CONDITIONS = {"B": "B_Main", "M": "M_Main", "Moa": "Moa_Main", "Mconf": "Mconf_Main", "Mmask": "Mmask_Main"}
W102_FILES = ["results/s0_sources/s0_report.json", "results/handoff/observations.npz", "results/handoff/observations_side.npz",
              "results/evaluation/report.json", "results/evaluation/rows.json", "data/hmr_pred_full.npz",
              "data/population.json", "data/w101_transfer_image_sha256.json", "data/w101_transfer_raw_sha256.json",
              "results/semantic_preflight_row_contract.json", "results/baseline0/s0_report.json"] + [
              f"results/s0_sources/prediction_{c}_{s}.npy" for c in CONDITIONS.values() for s in ("validation", "test")]

# ------------------------------------------------------------------ canonical joints (H36M order, unchanged)
JOINTS = ["pelvis", "left_hip", "left_knee", "left_ankle", "right_hip", "right_knee", "right_ankle", "spine", "thorax",
          "neck", "head", "left_shoulder", "left_elbow", "left_wrist", "right_shoulder", "right_elbow", "right_wrist"]
DIRECT = [11, 12, 13, 14, 15, 16, 1, 2, 3, 4, 5, 6]          # 12 limb joints (W100/W101 DIRECT)
BASE = DIRECT + [10]                                         # 13 independent observed points (head token = nose)
DERIVED = {0: (1, 4), 9: (11, 14), 8: (11, 14)}              # pelvis / neck / thorax = midpoints (spine = mid(neck,pelvis))
TORSO = [1, 4, 11, 14]
SMPL_OF = {1: 1, 2: 4, 3: 7, 4: 2, 5: 5, 6: 8, 11: 16, 12: 18, 13: 20, 14: 17, 15: 19, 16: 21, 10: 15}
PARENT = {1: 0, 2: 1, 3: 2, 4: 0, 5: 4, 6: 5, 7: 0, 8: 7, 9: 8, 10: 9, 11: 8, 12: 11, 13: 12, 14: 8, 15: 14, 16: 15}
SEGMENTS = {"upper_arm_l": (11, 12), "forearm_l": (12, 13), "upper_arm_r": (14, 15), "forearm_r": (15, 16),
            "thigh_l": (1, 2), "shin_l": (2, 3), "thigh_r": (4, 5), "shin_r": (5, 6)}
KNEES = {"l": (1, 2, 3), "r": (4, 5, 6)}
# limb root, mid, end, contralateral root, ipsilateral opposite-girdle point (body context)
CHAINS = {"leg_l": (1, 2, 3, 4, 11), "leg_r": (4, 5, 6, 1, 14), "arm_l": (11, 12, 13, 14, 1), "arm_r": (14, 15, 16, 11, 4)}
SEG_CHAIN = {"thigh_l": "leg_l", "shin_l": "leg_l", "thigh_r": "leg_r", "shin_r": "leg_r",
             "upper_arm_l": "arm_l", "forearm_l": "arm_l", "upper_arm_r": "arm_r", "forearm_r": "arm_r"}
CV_TO_CANON = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
SMPL24_TO_17 = (0, 1, 4, 7, 2, 5, 8, 3, 9, 12, 15, 16, 18, 20, 17, 19, 21)

# ------------------------------------------------------------------ fixed historical decision scales (not tuned here)
SEVERE_MM = 200.0         # W100-W102 severe MPJPE
LIMB_DIR_DEG = 45.0       # W102 temporal limb-direction event scale
KNEE_BEND_DEG = 20.0      # W97 knee-plane eligibility
BRANCH_DEG = 90.0         # W97 wrong-branch / torso-flip
DEPTH_RATIO = 0.25        # W100 depth-sign eligibility |dy| >= 0.25 |segment|
GT_STEP_DEG = 15.0        # W102 GT-stable limb step
PCK_WELL = 0.10           # W101 pck10 scale (fraction of GT limb-box diagonal)
BOOT_REPLICATES, BOOT_SEED = 2000, 1337


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def dump(path, value):
    Path(path).write_text(json.dumps(clean(value), indent=1, sort_keys=False), encoding="utf-8")


def video_of(sequence_id):
    return sequence_id.split(":")[1]


# ------------------------------------------------------------------ model-visible representation
def geometry_all(bank, obs):
    """Exact W102 geometry tensors per condition, reproduced through the unchanged production crop path."""
    from w101_reference_estimator import geometry_for_condition
    return {c: geometry_for_condition(bank, obs, c) for c in CONDITIONS}


def derived_token_audit(x, v):
    """Max deviation of pelvis/neck/thorax/spine tokens from their source midpoints where all are valid (raw x/W,y/H)."""
    out = {}
    for j, (a, b) in DERIVED.items():
        m = v[:, j] & v[:, a] & v[:, b]
        out[JOINTS[j]] = float(np.abs(x[m, j, :2] - .5*(x[m, a, :2]+x[m, b, :2])).max()) if m.any() else None
    m = v[:, 7] & v[:, 9] & v[:, 0]
    out["spine"] = float(np.abs(x[m, 7, :2] - .5*(x[m, 9, :2]+x[m, 0, :2])).max()) if m.any() else None
    return out


def base_xy(F):
    """(N,17,4) geometry -> (N,13,2) crop-normalized base points; (N,13) validity."""
    return F[:, BASE, :2].astype(np.float64), F[:, BASE, 3] > .5


def full_eligible(F):
    return (F[:, TORSO, 3] > .5).all(1)


def full_distance(xa, va, xb, vb):
    """Strict full-body distance: RMS crop-normalized point distance over jointly valid base points (per pair)."""
    both = va & vb
    n = both.sum(-1)
    d2 = (((xa-xb)**2).sum(-1)*both).sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, np.sqrt(d2/np.maximum(n, 1)), np.nan)


def local_vectors(F, chain):
    """(N,8) limb vectors (mid-root, end-mid) + body context (contralateral root, opposite girdle) in crop units."""
    r, m, e, c1, c2 = chain
    xy = F[:, :, :2].astype(np.float64)
    vec = np.concatenate([xy[:, m]-xy[:, r], xy[:, e]-xy[:, m], xy[:, c1]-xy[:, r], xy[:, c2]-xy[:, r]], axis=1)
    return vec, (F[:, list(chain), 3] > .5).all(1)


def local_distance(a, b):
    return np.sqrt(((a-b)**2).sum(-1)/4.0)


def projection_features(uv):
    """(N,24,2) GT pixel projections -> (N,13,2) square-crop-normalized coordinates (same crop formula, all valid)."""
    from framepose.crops import CROP_MARGIN, CROP_MIN_PIXELS
    p = uv[:, [SMPL_OF[j] for j in BASE]]
    lo, hi = p.min(1), p.max(1)
    side = np.maximum((hi-lo).max(1)*(1+2*CROP_MARGIN), CROP_MIN_PIXELS)
    origin = (lo+hi)/2 - side[:, None]/2
    return 2*(p-origin[:, None])/side[:, None, None] - 1


# ------------------------------------------------------------------ 3D geometry (evaluation-only)
def ray_rotation(p):
    """Rotation R with R @ unit(p) = +Y (canonical forward): removes the off-axis viewing direction of the root."""
    d = np.asarray(p, float)/np.linalg.norm(p)
    f = np.array([0., 1., 0.])
    axis = np.cross(d, f)
    s, c = np.linalg.norm(axis), float(d @ f)
    if s < 1e-12:
        return np.eye(3)
    k = axis/s
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + s*K + (1-c)*K@K


def unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return v/n


def angle_deg(a, b):
    return np.degrees(np.arccos(np.clip((unit(a)*unit(b)).sum(-1), -1, 1)))


def template_pose(G, lengths):
    """Bone-length-normalized reconstruction: GT bone directions, fixed population template lengths."""
    out = np.zeros_like(G)
    for child in range(1, 17):
        parent = PARENT[child]
        out[:, child] = out[:, parent] + lengths[child]*unit(G[:, child]-G[:, parent])
    return out


def torso_forward(G):
    """Vectorized motion.torso_frames thorax forward axis (NaN where the historical rule is degenerate)."""
    from motion.torso_frame import MIN_SIDE_ANGLE_DEG, MIN_TRUNK
    trunk = G[:, 8]-G[:, 0]
    length = np.linalg.norm(trunk, axis=-1)
    u = trunk/np.maximum(length, 1e-300)[:, None]
    s = G[:, 14]-G[:, 11]
    perp = s - (s*u).sum(-1, keepdims=True)*u
    n = np.linalg.norm(perp, axis=-1)
    ok = (length >= MIN_TRUNK) & (n >= math.sin(math.radians(MIN_SIDE_ANGLE_DEG))*np.maximum(np.linalg.norm(s, axis=-1), 1e-12)) & (n >= 1e-12)
    x = perp/np.maximum(n, 1e-300)[:, None]
    y = np.cross(u, x)
    y[~ok] = np.nan
    return y


def row_geometry(G):
    """Per-row quantities used by pair metrics: segment units, depth ratios, knee normals/flexion, torso forward."""
    seg = np.stack([G[:, b]-G[:, a] for a, b in SEGMENTS.values()], 1)
    seglen = np.linalg.norm(seg, axis=-1)
    knees = []
    for h, k, a in KNEES.values():
        t, s = G[:, k]-G[:, h], G[:, a]-G[:, k]
        knees.append((angle_deg(t, s), np.cross(t, s)))
    return {"seg": unit(seg), "depth": seg[..., 1]/seglen, "flex": np.stack([k[0] for k in knees], 1),
            "normal": np.stack([unit(k[1]) for k in knees], 1), "fwd": torso_forward(G)}


def pair_metrics(ga, gb, ta=None, tb=None):
    """Material-difference flags between two geometry dicts (vectorized over pairs). ta/tb: template poses."""
    seg_ang = angle_deg(ga["seg"], gb["seg"])
    out = {"seg_angle": seg_ang, "limb_dir": (seg_ang > LIMB_DIR_DEG).any(-1)}
    knee_ok = (ga["flex"] >= KNEE_BEND_DEG) & (gb["flex"] >= KNEE_BEND_DEG)
    knee_ang = angle_deg(ga["normal"], gb["normal"])
    out["knee_eligible"], out["knee_branch"] = knee_ok, knee_ok & (knee_ang > BRANCH_DEG)
    dep_ok = (np.abs(ga["depth"]) >= DEPTH_RATIO) & (np.abs(gb["depth"]) >= DEPTH_RATIO)
    out["depth_eligible"], out["depth_flip"] = dep_ok, dep_ok & (np.sign(ga["depth"]) != np.sign(gb["depth"]))
    fwd = angle_deg(ga["fwd"], gb["fwd"])
    out["torso_eligible"], out["torso_angle"] = np.isfinite(fwd), fwd
    out["torso_flip"] = np.isfinite(fwd) & (np.nan_to_num(fwd) > BRANCH_DEG)
    out["material"] = out["limb_dir"] | out["knee_branch"].any(-1) | out["depth_flip"].any(-1) | out["torso_flip"]
    if ta is not None:
        out["template_mpjpe"] = 1000*np.linalg.norm(ta-tb, axis=-1).mean(-1)
    return out


def subset(geo, idx):
    return {k: v[idx] for k, v in geo.items()}


# ------------------------------------------------------------------ retrieval (deterministic, cross-video, distinct sequences)
def retrieve(Xq, Xr, seq_r, vid_q, vid_r, *, k=10, chunk=512, dist=None):
    """For every query: nearest reference row per OTHER-video actor-sequence, then the k nearest such sequences.

    Ties break by reference order (callers pass references sorted by sequence then sample id), then sequence order.
    Returns (idx, d) of shape (Nq, k), padded with -1 / inf.
    """
    dist = dist or (lambda a, b: np.sqrt(np.maximum(((a[:, None]-b[None])**2).sum(-1), 0)))
    seqs, starts = np.unique(seq_r, return_index=True)
    if not np.all(np.diff(np.searchsorted(seqs, seq_r)) >= 0):
        raise ValueError("references must be grouped by sequence")
    out_i = np.full((len(Xq), k), -1, np.int64)
    out_d = np.full((len(Xq), k), np.inf)
    bounds = np.append(starts, len(seq_r))
    for s0 in range(0, len(Xq), chunk):
        D = dist(Xq[s0:s0+chunk], Xr)
        D = np.where(vid_q[s0:s0+chunk, None] == vid_r[None], np.inf, D)
        per_min = np.stack([D[:, bounds[i]:bounds[i+1]].min(1) for i in range(len(seqs))], 1)
        per_arg = np.stack([bounds[i]+D[:, bounds[i]:bounds[i+1]].argmin(1) for i in range(len(seqs))], 1)
        order = np.argsort(per_min, axis=1, kind="stable")[:, :k]
        rows = np.arange(order.shape[0])[:, None]
        dd, ii = per_min[rows, order], per_arg[rows, order]
        ii = np.where(np.isfinite(dd), ii, -1)
        out_i[s0:s0+chunk, :ii.shape[1]], out_d[s0:s0+chunk, :dd.shape[1]] = ii, dd
    return out_i, out_d


def count_within(Xq, Xr, vid_q, vid_r, taus, *, chunk=512, dist=None):
    """Cross-video reference rows within each tau, per query."""
    dist = dist or (lambda a, b: np.sqrt(np.maximum(((a[:, None]-b[None])**2).sum(-1), 0)))
    out = np.zeros((len(Xq), len(taus)), np.int64)
    for s0 in range(0, len(Xq), chunk):
        D = dist(Xq[s0:s0+chunk], Xr)
        D = np.where(vid_q[s0:s0+chunk, None] == vid_r[None], np.inf, D)
        for t, tau in enumerate(taus):
            out[s0:s0+chunk, t] = (D <= tau).sum(1)
    return out


def euclid_rms(n_points):
    """Pairwise RMS point distance for flattened (N, 2*n_points) vectors (matrix identity, float64)."""
    def f(a, b):
        sq = (a*a).sum(1)[:, None] + (b*b).sum(1)[None] - 2*a@b.T
        return np.sqrt(np.maximum(sq, 0)/n_points)
    return f


# ------------------------------------------------------------------ statistics
def video_bootstrap(videos, values):
    """Frame-weighted mean, 95% interval resampling whole source videos (2,000 replicates, seed 1337)."""
    groups = defaultdict(list)
    for v, x in zip(videos, values):
        if x is not None and np.isfinite(x):
            groups[v].append(float(x))
    if not groups:
        return {"n": 0, "videos": 0, "mean": None, "ci95": None}
    keys = sorted(groups)
    ns = np.array([len(groups[k]) for k in keys])
    sums = np.array([sum(groups[k]) for k in keys])
    out = {"n": int(ns.sum()), "videos": len(keys), "mean": float(sums.sum()/ns.sum()), "ci95": None}
    if len(keys) >= 2:
        draws = np.random.default_rng(BOOT_SEED).integers(0, len(keys), (BOOT_REPLICATES, len(keys)))
        out["ci95"] = np.percentile(sums[draws].sum(1)/ns[draws].sum(1), [2.5, 97.5]).tolist()
    return out


def distribution(values):
    x = np.asarray(values, float).ravel()
    x = x[np.isfinite(x)]
    if not len(x):
        return {"n": 0}
    q = np.percentile(x, [0, 5, 10, 25, 50, 75, 90, 95, 99, 100])
    return {"n": int(len(x)), "mean": float(x.mean()),
            "quantiles": dict(zip(("min", "p05", "p10", "p25", "p50", "p75", "p90", "p95", "p99", "max"), q.tolist()))}


# ------------------------------------------------------------------ loading
def load_context(w102, w101data):
    from framepose.bank import load_bank
    bank = load_bank(Path(w101data)/"bank_3dpw_paired_v2.json")
    if bank.content_digest() != BANK_DIGEST:
        raise RuntimeError("STOP: frozen bank identity failed")
    obs = np.load(Path(w102)/"results/handoff/observations.npz", allow_pickle=False)
    side = np.load(Path(w102)/"results/handoff/observations_side.npz", allow_pickle=False)
    positions = np.asarray(obs["positions"], np.int64)
    if not (np.array_equal(positions, side["positions"]) and np.array_equal(obs["sample_id"], side["sample_id"])):
        raise RuntimeError("STOP: observation/side alignment failed")
    expected = np.concatenate([bank.indices("validation"), bank.indices("test")])
    if not np.array_equal(positions, expected):
        raise RuntimeError("STOP: population order changed")
    return bank, obs, side, positions


def load_predictions(w102, bank, positions):
    report = json.loads((Path(w102)/"results/s0_sources/s0_report.json").read_text())
    pred = {}
    for alias, cond in CONDITIONS.items():
        parts = []
        for split in ("validation", "test"):
            path = Path(w102)/f"results/s0_sources/prediction_{cond}_{split}.npy"
            if sha(path) != report["conditions"][cond]["splits"][split]["sha256"]:
                raise RuntimeError(f"STOP: {cond} {split} prediction changed")
            parts.append(np.load(path, allow_pickle=False))
        pred[alias] = np.concatenate(parts).astype(np.float64)
    if any(p.shape != (len(positions), 17, 3) for p in pred.values()):
        raise RuntimeError("STOP: prediction shape mismatch")
    return pred, report


def load_hmr(w102, positions):
    """Frozen HMR2 joints by population row; NaN where W97 marked the frame invalid."""
    path = Path(w102)/"data/hmr_pred_full.npz"
    if sha(path) != HMR_SHA:
        raise RuntimeError("STOP: frozen HMR2 identity failed")
    hp = np.load(path, allow_pickle=False)
    population = json.loads((Path(w102)/"data/population.json").read_text())["rows"]
    by_pos = {}
    for i, r in enumerate(population):
        if str(hp["population"][i]) != r["population"] or int(hp["index"][0][i]) != r["bank_position"]:
            raise RuntimeError("STOP: HMR identity mismatch")
        if r["population"].startswith("3dpw") and hp["valid"][i]:
            j = np.asarray(hp["joints24"][i], float)[list(SMPL24_TO_17)] @ CV_TO_CANON.T
            by_pos[int(r["bank_position"])] = j - j[0]
    out = np.full((len(positions), 17, 3), np.nan)
    for i, p in enumerate(positions):
        if int(p) in by_pos:
            out[i] = by_pos[int(p)]
    return out


def raw_references(raw_root, bank, positions):
    """Evaluation-only: canonical root-relative GT, root camera position, 24-joint projections, gravity (W102 row contract)."""
    from w102_reference import row_identity
    cache = {}
    G = np.zeros((len(positions), 17, 3))
    root = np.zeros((len(positions), 3))
    uv = np.zeros((len(positions), 24, 2))
    gravity = np.zeros((len(positions), 3))
    valid = np.zeros(len(positions), bool)
    for i, p in enumerate(positions):
        sample = bank.samples[int(p)]
        video = video_of(sample.sequence_id)
        if video not in cache:
            with (Path(raw_root)/"sequenceFiles"/sample.split/(video+".pkl")).open("rb") as stream:
                cache[video] = pickle.load(stream, encoding="latin1")
        d = cache[video]
        actor, k = row_identity(sample, d)
        X = np.asarray(d["jointPositions"][actor][k], float).reshape(24, 3)
        T = np.asarray(d["cam_poses"][k], float)
        Xc = X@T[:3, :3].T + T[:3, 3]
        proj = Xc@np.asarray(d["cam_intrinsics"], float).T
        uv[i] = proj[:, :2]/proj[:, 2:3]
        canon = Xc[list(SMPL24_TO_17)]@CV_TO_CANON.T
        root[i] = canon[0]
        G[i] = canon - canon[0]
        g = CV_TO_CANON@T[:3, :3]@np.array([0., 1., 0.])
        gravity[i] = g/np.linalg.norm(g)
        valid[i] = bool(d["campose_valid"][actor][k]) and np.isfinite(G[i]).all() and np.isfinite(uv[i]).all()
    return {"G": G, "root": root, "uv": uv, "gravity": gravity, "valid": valid}


# ------------------------------------------------------------------ stage: verify
def cmd_verify(args):
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=False)
    w102 = Path(args.w102)
    report = {"schema": "w103_verify_v1", "checks": {}}
    manifest_path = w102/"evidence_manifest.json"
    report["checks"]["w102_manifest_sha256"] = sha(manifest_path)
    if report["checks"]["w102_manifest_sha256"] != W102_MANIFEST_SHA:
        raise RuntimeError("STOP: W102 manifest identity failed")
    manifest = json.loads(manifest_path.read_text())["files"]
    files = {}
    for rel in W102_FILES:
        actual = sha(w102/rel)
        files[rel] = {"sha256": actual, "manifest": manifest[rel]["sha256"], "match": actual == manifest[rel]["sha256"]}
    report["checks"]["w102_files"] = files
    if not all(f["match"] for f in files.values()):
        raise RuntimeError("STOP: a frozen W102 artifact differs from its manifest")
    for rel, pinned in (("results/handoff/observations.npz", OBSERVATIONS_SHA), ("results/handoff/observations_side.npz", SIDE_SHA),
                        ("data/hmr_pred_full.npz", HMR_SHA), ("data/population.json", POPULATION_SHA)):
        if files[rel]["sha256"] != pinned:
            raise RuntimeError(f"STOP: {rel} differs from the worklog-pinned identity")
    report["checks"]["checkpoint_sha256"] = sha(Path(args.w101_data)/"s0_checkpoint.pt")
    if report["checks"]["checkpoint_sha256"] != S0_CHECKPOINT_SHA:
        raise RuntimeError("STOP: S0 checkpoint identity failed")
    raw_manifest = json.loads((w102/"data/w101_transfer_raw_sha256.json").read_text())
    raw = {}
    for key, digest in raw_manifest.items():
        rel = key.split("3dpw/", 1)[1]
        raw[rel] = sha(w102/"data/raw"/rel) == digest
    report["checks"]["raw_pickles_match"] = {"files": len(raw), "all_match": all(raw.values())}
    if not all(raw.values()):
        raise RuntimeError("STOP: raw annotation identity failed")
    bank, obs, side, positions = load_context(w102, args.w101_data)
    report["checks"]["bank_content_digest"] = bank.content_digest()
    s0 = json.loads((w102/"results/s0_sources/s0_report.json").read_text())
    if s0["environment_fingerprint"] != ENVIRONMENT_FINGERPRINT:
        raise RuntimeError("STOP: W102 S0 report environment changed")
    geometry = geometry_all(bank, obs)
    geo = {}
    for alias, cond in CONDITIONS.items():
        digest = hashlib.sha256(geometry[alias].tobytes()).hexdigest()
        geo[cond] = {"reproduced": digest, "w102": s0["conditions"][cond]["geometry_sha256"],
                     "match": digest == s0["conditions"][cond]["geometry_sha256"]}
    report["checks"]["geometry_reproduction"] = geo
    if not all(g["match"] for g in geo.values()):
        raise RuntimeError("STOP: model-visible input cannot be reproduced")
    # model-visible information audit (2D only)
    xb, vb = bank.arrays["input_2d"][positions], bank.arrays["input_valid"][positions]
    xm, vm = obs["input_2d_M"], obs["input_valid_M"]
    same = side["identity"] == "same_person"
    report["model_visible"] = {
        "tensor": "(17,4) per row: crop x, crop y in [-1,1], confidence, validity; neutral sign state (all UNKNOWN); no image tokens",
        "single_frame": "FramePoseEstimator.forward(geometry (B,17,4), image_tokens=None, sign_state) attends over the 17-joint axis only",
        "derived_tokens_max_abs_dev_B": derived_token_audit(xb, vb),
        "derived_tokens_max_abs_dev_M_same_person": derived_token_audit(xm[same], vm[same]),
        "neutral_sign_states_constant": True,
        "geometry_feature_dtype": str(geometry["B"].dtype),
        "invalid_tokens_zeroed": bool(all((geometry[a][..., :3][geometry[a][..., 3] < .5] == 0).all() for a in CONDITIONS)),
        "valid_xy_range": {a: [float(geometry[a][..., :2][geometry[a][..., 3] > .5].min()),
                               float(geometry[a][..., :2][geometry[a][..., 3] > .5].max())] for a in ("B", "M")},
        "rows_with_lt2_valid_fallback_crop": {a: int(((geometry[a][positions][..., 3] > .5).sum(1) < 2).sum()) for a in ("B", "M")},
    }
    # Mmask/Mconf composition (2D only)
    mm, vmm = obs["input_2d_Mmask"], obs["input_valid_Mmask"]
    report["probe_composition"] = {
        "Mmask_xy_equals_M_xy_where_both_valid": bool(np.array_equal(mm[vm & vmm][:, :2], xm[vm & vmm][:, :2])),
        "Mmask_conf_equals_M_conf_where_both_valid": bool(np.array_equal(mm[vm & vmm][:, 2], xm[vm & vmm][:, 2])),
        "Mmask_valid_equals_B_valid_on_same_person": bool(np.array_equal(vmm[same], (vb & np.isfinite(mm[..., 0]))[same])),
        "Mconf_valid_equals_M_valid": bool(np.array_equal(obs["input_valid_Mconf"], vm)),
    }
    dump(out/"verify_report.json", report)
    print(json.dumps({k: v for k, v in report["checks"].items() if k != "w102_files"}, default=str)[:2000])


# ------------------------------------------------------------------ stage: reproduce (GPU, same W102 stack)
def cmd_reproduce(args):
    import torch
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    from framepose.signs import neutral_sign_states
    from framepose.train import load_checkpoint, predict
    from run_w102_same_environment_s0 import execution_identity
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=False)
    bank, obs, side, positions = load_context(args.w102, args.w101_data)
    os.chdir(ROOT)
    environment, fingerprint = execution_identity(torch)
    model, checkpoint = load_checkpoint(Path(args.w101_data)/"s0_checkpoint.pt", device="cuda")
    s0 = json.loads((Path(args.w102)/"results/s0_sources/s0_report.json").read_text())
    geometry = geometry_all(bank, obs)
    signs = neutral_sign_states(len(bank))
    w102_env = s0["environment"]
    lf = {p: hashlib.sha256((ROOT/p).read_bytes().replace(b"\r\n", b"\n")).hexdigest() for p in environment["model_source_sha256"]}
    differing = sorted(k for k in set(environment) | set(w102_env) if environment.get(k) != w102_env.get(k))
    report = {"schema": "w103_reproduce_v1", "environment_fingerprint": fingerprint,
              "w102_environment_fingerprint": ENVIRONMENT_FINGERPRINT, "same_stack": fingerprint == ENVIRONMENT_FINGERPRINT,
              "identity_fields_differing_from_w102": differing,
              "model_source_sha256": environment["model_source_sha256"],
              "model_source_sha256_lf_normalized": lf,
              "lf_normalized_sources_equal_w102": lf == w102_env["model_source_sha256"], "conditions": {}}
    for alias in ("B", "M"):
        cond = CONDITIONS[alias]
        report["conditions"][cond] = {}
        for split in ("validation", "test"):
            pos = bank.indices(split)
            p = predict(model, torch, geometry[alias], None, pos, torch.device("cuda"), batch_size=512, amp_enabled=False,
                        signs=signs).astype(np.float32)
            stored = np.load(Path(args.w102)/f"results/s0_sources/prediction_{cond}_{split}.npy")
            pick = np.arange(0, len(pos), 97)
            single = np.concatenate([predict(model, torch, geometry[alias], None, pos[[i]], torch.device("cuda"), batch_size=1,
                                             amp_enabled=False, signs=signs) for i in pick]).astype(np.float32)
            report["conditions"][cond][split] = {
                "rows": len(pos), "byte_equal_to_w102": bool(p.tobytes() == stored.tobytes()),
                "sha256": hashlib.sha256(p.tobytes()).hexdigest(),
                "batch_one_rows": len(pick), "batch_one_max_abs_mm": float(1000*np.abs(single-stored[pick]).max()),
                "batch_one_byte_equal_rows": int(sum(single[i].tobytes() == stored[pick[i]].tobytes() for i in range(len(pick))))}
            print(cond, split, report["conditions"][cond][split], flush=True)
    dump(out/"reproduce_report.json", report)


# ------------------------------------------------------------------ stage: declare (2D only)
def resolution_scales(FB, FM, same):
    """Same-moment cross-detector model-visible distance (B vs M, identical masks) -> tau_S (p10), tau_N (p50).

    Uses only 2D input tensors; no target/GT array is an argument.
    """
    vb, vm = FB[..., 3] > .5, FM[..., 3] > .5
    rows = same & (vb == vm).all(1) & full_eligible(FB) & full_eligible(FM)
    xb, bb = base_xy(FB[rows]); xm, bm = base_xy(FM[rows])
    full = full_distance(xb, bb, xm, bm)
    local = {}
    for name, chain in CHAINS.items():
        a, va = local_vectors(FB, chain); b, vb_ = local_vectors(FM, chain)
        ok = same & va & vb_
        local[name] = local_distance(a[ok], b[ok])
    pooled = np.concatenate(list(local.values()))
    return {"full_rows": int(rows.sum()), "full": distribution(full),
            "tau_full": {"strict_p10": float(np.percentile(full, 10)), "near_p50": float(np.percentile(full, 50))},
            "local_instances": {k: int(len(v)) for k, v in local.items()}, "local": {k: distribution(v) for k, v in local.items()},
            "tau_local": {"strict_p10": float(np.percentile(pooled, 10)), "near_p50": float(np.percentile(pooled, 50))}}


def pools(bank, obs, side, positions, geometry):
    """Query/reference pools: B = all 10,483 actor-frames (benchmark input), M = W102 uniquely associated same-person rows."""
    seq = np.array([bank.samples[int(p)].sequence_id for p in positions])
    sid = np.array([bank.samples[int(p)].sample_id for p in positions])
    split = np.array([bank.samples[int(p)].split for p in positions])
    vid = np.array([video_of(s) for s in seq])
    same = np.asarray(side["identity"]) == "same_person"
    F = {a: geometry[a][positions] for a in CONDITIONS}
    return {"seq": seq, "sid": sid, "split": split, "vid": vid, "same": same, "F": F,
            "members": {"B": np.arange(len(positions)), "M": np.flatnonzero(same)}}


def full_groups(F, members):
    """Rows grouped by identical 17-joint validity mask, full-eligible only; returns {mask_key: member rows}."""
    keys = np.array([''.join('1' if x else '0' for x in r) for r in (F[members, :, 3] > .5)])
    ok = full_eligible(F[members])
    groups = defaultdict(list)
    for m, k, e in zip(members, keys, ok):
        if e:
            groups[k].append(m)
    return {k: np.array(v) for k, v in groups.items()}


def pair_counts(P, src, tau):
    """2D-only pair accounting under strict full-body similarity (identical masks, cross-video)."""
    F = P["F"][src]; members = P["members"][src]
    groups = full_groups(F, members)
    within = {t: 0 for t in tau}; eligible = 0; video_pairs = {t: set() for t in tau}
    for key, rows in groups.items():
        rows = rows[np.lexsort((P["sid"][rows], P["seq"][rows]))]
        x, v = base_xy(F[rows]); sel = v[0]
        X = x[:, sel].reshape(len(rows), -1)
        vid = P["vid"][rows]
        n_cross = len(rows)**2 - sum(c**2 for c in Counter(vid).values())
        eligible += n_cross//2
        for s0 in range(0, len(rows), 1024):
            D = euclid_rms(int(sel.sum()))(X[s0:s0+1024], X)
            cross = vid[s0:s0+1024, None] != vid[None]
            for t, value in tau.items():
                hit = (D <= value) & cross
                within[t] += int(hit.sum())
                a, b = np.nonzero(hit)
                for va, vb in set(zip(vid[s0+a], vid[b])):
                    video_pairs[t].add(tuple(sorted((va, vb))))
    return {"eligible_cross_video_pairs": eligible, "mask_groups": len(groups),
            "full_eligible_rows": int(sum(len(r) for r in groups.values())),
            "within": {t: within[t]//2 for t in tau}, "video_pairs_within": {t: len(video_pairs[t]) for t in tau}}


DECLARATION_TEXT = """
W103 observability contract. Fixed before any GT 3D disagreement, S0 error attribution or HMR comparison is computed.

1 MODEL-VISIBLE SPACE. Per row the exact W102 (17,4) geometry tensor (crop x, crop y in [-1,1], confidence, validity).
  Distances use the 13 independent points (12 limb joints + head token); pelvis/neck/thorax/spine tokens are verified
  midpoints and add no distinguishing information. Pools: B = all 10,483 actor-frames with the benchmark input;
  M = the 6,988 W102 uniquely associated same-person rows with the M1 input. References for a query come from the
  same pool, other videos only (primary), nearest row per other actor-sequence, k = 10 nearest sequences.
2 STRICT FULL-BODY SIMILARITY (A). Identical 17-joint validity masks; both shoulders and both hips valid (torso context);
  D_full = RMS crop-unit distance over the mask's valid base points. Confidence is not a gate: W102 Mconf (benchmark
  confidence on jointly valid joints) changed S0 MPJPE by +0.23/+0.85 mm, and it is reported per pair.
3 LOCAL LIMB SIMILARITY (B). Per limb (leg_l, leg_r, arm_l, arm_r): crop-unit vectors mid-root, end-mid,
  contralateral root-root and ipsilateral opposite-girdle-root; all five points valid in both rows;
  D_local = RMS over the four vectors. A local match never establishes full-body indistinguishability.
4 RESOLUTION. Same moment, same person, two detectors (B vs M), identical masks: D_full distribution;
  tau_full_N = median, tau_full_S = 10th percentile. Local: pooled same-moment B/M D_local over limbs;
  tau_local_N = median, tau_local_S = 10th percentile. "Near" (N) = within typical detector disagreement for one
  physical pose; "strict" (S) = within the best 10% of that agreement. Fixed bins for the descriptive curve:
  [0,.5), [.5,1), [1,2), [2,4), >=4 x tau_N. No other threshold is evaluated.
5 ORACLE-GEOMETRY CONTROL. GT 24-joint camera projections, the 13 corresponding SMPL points (head = SMPL head),
  identical square-crop formula -> D_proj with the same tau values (regime oracle_geometry, evaluation-only).
6 3D COMPARISON (evaluation-only). GT root-relative canonical camera frame, rotated so the GT root ray is +Y
  (removes off-axis perspective viewpoint; S0 cannot observe crop location). Body-scale control: population-mean
  template bone lengths (VAL+TEST GT, fixed) with GT bone directions -> template MPJPE. A pair is MATERIALLY
  DIFFERENT if any: limb direction > 45 deg; knee branch (both flexions >= 20 deg, plane normals > 90 deg);
  limb depth-sign flip (both |dy| >= 0.25 |segment|, opposite sign); torso forward > 90 deg. Near-straight knees
  (< 20 deg) and degenerate torsos are counted separately, never assigned an angle.
7 OBSERVATION QUALITY (independent of S0). Per row/source: WELL-OBSERVED = all 12 limb joints valid and every limb
  joint within 0.10 of the GT limb-box diagonal (W101 2D metric, COCO/SMPL caveat retained); PARTIAL = < 12 limb joints
  valid; MISLOCALIZED = 12 valid but a joint > 0.10.
8 OUTCOME. SEVERE = MPJPE > 200 mm (historical). ACCURATE = MPJPE <= 100 mm and no limb direction error > 45 deg,
  no wrong knee branch (W97 rule), no torso forward flip. Local failures: wrong knee branch; limb depth-sign error
  (W100 rule) per segment.
9 ATTRIBUTION of a failure at query q (source s, full tier for SEVERE, local tier of the failing limb for local):
  N_q = cross-video neighbors within tau_N (k <= 10 distinct sequences).
  AMBIGUITY-SUPPORTED: some n in N_q has GT materially different from GT_q (full) / on the failing decision opposite
  to GT_q (local), and that GT is closer to S0's output than GT_q is (full: MPJPE in q's ray frame) / agrees with
  S0's decision (local).
  MODEL-SPECIFIC-SUPPORTED: >= 2 neighbors from >= 2 videos; full: median neighbor-GT-to-GT_q MPJPE <= 0.5 x S0 MPJPE;
  local: every eligible neighbor agrees with GT_q's decision; and no neighbor supports ambiguity as defined above.
  MIXED = both; UNRESOLVED = neither (including empty N_q). Missingness and mislocalization are reported first
  (a failure whose input is not well observed is not attributed to the mapping).
10 GROUPS (same-person rows, S0 on M primary, B secondary): A WO+accurate, B WO+severe, C PARTIAL+accurate,
  D PARTIAL+severe, E B and M both severe, F M severe while B not severe (and B accurate separately).
11 TEMPORAL (diagnostic): original bank neighbors only, same pool membership, no bridging. A failure is ISOLATED if
  both neighbors exist and S0 does not fail the same decision there; 2D-STABLE if the neighbor's input is within
  tau_N (local tier for local failures, full tier with identical mask for severe).
12 STATISTICS. Query-level rates, frame-weighted, 95% intervals resampling source videos (2,000, seed 1337); no CI
  below 2 videos. Pair counts are never treated as independent samples.
13 DECISION. Among well-observed important failures (severe + wrong knee branch + depth-sign errors, M source primary),
  a mechanism has MEANINGFUL evidence if its share of attributed failures (ambiguity / model / mixed denominators
  excluding unresolved) has a video-CI lower bound >= 0.10. Ambiguity only -> CASE A (requires controlled ambiguous
  pairs: both rows well observed, D_proj <= tau_N, >= 3 independent video pairs); model only -> CASE B; both -> CASE C;
  neither or controls fail -> CASE C (unresolved); identity/reproduction failure -> CASE D.
14 QUALITATIVE SELECTION (deterministic, ties by sample id): (1) strict-S cross-video pair, materially different,
  largest template MPJPE; (2) group A row with smallest M MPJPE having >= 1 near neighbor; (3) group B row,
  model-specific-supported with largest M MPJPE (else largest M MPJPE); (4) group E row with largest min(B,M) MPJPE;
  (5) group F row with largest M-B MPJPE; (6) local-tier leg pair with smallest D_local and GT knee-branch
  disagreement; (7) local-tier arm pair with smallest D_local and GT depth-sign disagreement; (8) M torso-forward
  temporal event (S0 step > 90 deg, GT step < 45 deg) with the largest S0 step.
"""


def cmd_declare(args):
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=False)
    bank, obs, side, positions = load_context(args.w102, args.w101_data)
    geometry = geometry_all(bank, obs)
    P = pools(bank, obs, side, positions, geometry)
    scales = resolution_scales(P["F"]["B"], P["F"]["M"], P["same"])
    tau_full = scales["tau_full"]
    counts = {src: pair_counts(P, src, {"S": tau_full["strict_p10"], "N": tau_full["near_p50"]}) for src in ("B", "M")}
    declaration = {"schema": "w103_declaration_v1", "text": DECLARATION_TEXT.strip(), "resolution": scales,
                   "tau": {"full_S": tau_full["strict_p10"], "full_N": tau_full["near_p50"],
                           "local_S": scales["tau_local"]["strict_p10"], "local_N": scales["tau_local"]["near_p50"]},
                   "bins_x_tauN": [0, .5, 1, 2, 4], "k_sequences": 10,
                   "two_d_pair_counts_before_3d": counts,
                   "inputs": {"bank_content_digest": BANK_DIGEST, "observations_sha256": OBSERVATIONS_SHA,
                              "geometry_sha256": {CONDITIONS[a]: hashlib.sha256(geometry[a].tobytes()).hexdigest() for a in CONDITIONS}},
                   "GT_arrays_read": False}
    dump(out/"declaration.json", declaration)
    print(json.dumps({"tau": declaration["tau"], "counts": counts}, indent=1))
    print("declaration sha256", sha(out/"declaration.json"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("verify", "reproduce", "declare", "analyze"))
    ap.add_argument("--w102", required=True)
    ap.add_argument("--w101-data", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--declaration")
    ap.add_argument("--declaration-sha")
    ap.add_argument("--exclude-root-behind-camera", action="store_true",
                    help="analyze: sensitivity run removing rows whose GT root lies behind the camera (reported separately)")
    args = ap.parse_args()
    if args.stage == "analyze":
        from w103_analysis import cmd_analyze
        cmd_analyze(args)
    else:
        {"verify": cmd_verify, "reproduce": cmd_reproduce, "declare": cmd_declare}[args.stage](args)


if __name__ == "__main__":
    main()
