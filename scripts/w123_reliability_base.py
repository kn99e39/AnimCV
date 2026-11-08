"""W123 scratch: learned per-joint 3D error assessor around the FROZEN S0 FramePose model.

Evidence-only research code. Nothing here is imported by Production ``src/``; nothing here writes into
MotionGraph, AnimationClip, Blender/FBX or keyframe schemas. The frozen S0 checkpoint is only read.

Information ownership (kept in separate arrays / files throughout):
  * raw 2D detector observation  -> ``geometry`` (crop-normalized x, y, RAW score (never clipped), validity)
  * frozen 3D pose hypothesis    -> ``prediction`` (S0 output, camera_root_relative metres)
  * learned assessor estimate    -> ``q50`` / ``q90`` (metres; predicted per-joint 3D error quantiles)
  * evaluation-only ground truth -> ``target`` / ``error``; never an input of any runtime function here.

The runtime feature builder accepts exactly the allowlisted keys and refuses anything else, so GT cannot be
passed into inference by accident.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

JOINT_NAMES = ("pelvis", "left_hip", "left_knee", "left_ankle", "right_hip", "right_knee", "right_ankle", "spine",
               "thorax", "neck", "head", "left_shoulder", "left_elbow", "left_wrist", "right_shoulder", "right_elbow",
               "right_wrist")
PARENT = (-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15)
TARGET_JOINTS = tuple(range(1, 17))           # pelvis is the root (zero by construction)
TARGET_NAMES = tuple(JOINT_NAMES[j] for j in TARGET_JOINTS)
LIMB_JOINTS = (1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16)
ARM_JOINTS = (11, 12, 13, 14, 15, 16)
LEG_JOINTS = (1, 2, 3, 4, 5, 6)
# hinge (parent, joint, child) used for the near-straight stratum; cells inherit the hinge of their limb
HINGES = {"left_elbow": (11, 12, 13), "right_elbow": (14, 15, 16), "left_knee": (1, 2, 3), "right_knee": (4, 5, 6)}
CELL_HINGE = {12: "left_elbow", 13: "left_elbow", 15: "right_elbow", 16: "right_elbow",
              2: "left_knee", 3: "left_knee", 5: "right_knee", 6: "right_knee"}

COORDINATE_FRAME = "animcv_camera_root_relative_metres(+X right,+Y forward depth,+Z up; pelvis=joint0)"
EPS_M = 1e-3                 # log-error offset (1 mm)
QUANTILES = (0.5, 0.9)
MATERIAL_M = 0.150           # primary "materially wrong joint" event (declared)
SECONDARY_M = 0.200          # secondary, reported only
REVIEW_BUDGET = 0.10         # fraction of supported joint-cells per video flagged for review
ACCEPT_FRACTION = 0.80       # secondary selective-acceptance view
NEAR_STRAIGHT_DEG = 15.0
MIN_CELL = 30                # table baselines: minimum TRAIN samples per cell before falling back

RUNTIME_KEYS = frozenset({"geometry", "prediction"})
FORBIDDEN_TOKENS = ("target", "gt", "error", "truth", "reference_3d", "actor_id", "stratum", "split", "label")


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha_array(array: np.ndarray) -> str:
    a = np.ascontiguousarray(array)
    return sha_bytes(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def dump(path: str | Path, value: Any) -> str:
    text = json.dumps(clean(value), indent=2, allow_nan=False, sort_keys=False)
    data = text.encode("utf-8")
    Path(path).write_bytes(data)          # exact LF bytes, so the returned hash equals the file hash
    return sha_bytes(data)


# ----------------------------------------------------------------------------------------------------------------
# Runtime features (allowlisted inputs only)
# ----------------------------------------------------------------------------------------------------------------

def _check_runtime_inputs(inputs: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    keys = set(inputs)
    if keys != RUNTIME_KEYS:
        extra = sorted(keys - RUNTIME_KEYS)
        missing = sorted(RUNTIME_KEYS - keys)
        raise ValueError(f"RUNTIME_FEATURE_ALLOWLIST_VIOLATION extra={extra} missing={missing}")
    geometry = np.asarray(inputs["geometry"], dtype=np.float32)
    prediction = np.asarray(inputs["prediction"], dtype=np.float32)
    if geometry.ndim != 3 or geometry.shape[1:] != (17, 4):
        raise ValueError("geometry must be (N,17,4)")
    if prediction.shape != (geometry.shape[0], 17, 3):
        raise ValueError("prediction must be (N,17,3) aligned with geometry")
    if not (np.isfinite(geometry).all() and np.isfinite(prediction).all()):
        raise ValueError("non-finite runtime input")
    valid = geometry[:, :, 3]
    if not np.isin(valid, (0.0, 1.0)).all():
        raise ValueError("geometry validity channel must be exactly 0/1")
    return geometry, prediction


def reprojection_residual(geometry: np.ndarray, prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unlearned 2D/3D self-consistency heuristic (baseline C).

    Orthographic projection of the frozen 3D hypothesis onto the image plane (u=X, v=-Z), least-squares
    scale+translation fit to the observed crop-normalized 2D on valid joints, per-joint residual (crop units).
    Invalid joints get residual 0 and are identified by the validity channel; rows with <3 valid joints get 0s.
    """
    n = geometry.shape[0]
    valid = geometry[:, :, 3] > 0.5
    obs = geometry[:, :, :2].astype(np.float64)
    uv = np.stack([prediction[:, :, 0], -prediction[:, :, 2]], axis=-1).astype(np.float64)
    residual = np.zeros((n, 17), dtype=np.float64)
    scale = np.zeros(n, dtype=np.float64)
    w = valid.astype(np.float64)
    count = w.sum(1)
    ok = count >= 3
    cnt = np.where(ok, count, 1.0)
    mu_uv = (uv * w[..., None]).sum(1) / cnt[:, None]
    mu_ob = (obs * w[..., None]).sum(1) / cnt[:, None]
    du = (uv - mu_uv[:, None]) * w[..., None]
    do = (obs - mu_ob[:, None]) * w[..., None]
    num = (du * do).sum((1, 2))
    den = (du * du).sum((1, 2))
    s = np.where(den > 1e-12, num / np.where(den > 1e-12, den, 1.0), 0.0)
    fit = s[:, None, None] * (uv - mu_uv[:, None]) + mu_ob[:, None]
    r = np.linalg.norm(fit - obs, axis=-1)
    residual = np.where(valid & ok[:, None], r, 0.0)
    scale = np.where(ok, s, 0.0)
    return residual.astype(np.float32), scale.astype(np.float32)


def bone_lengths(prediction: np.ndarray) -> np.ndarray:
    p = np.asarray(prediction, dtype=np.float64)
    out = [np.linalg.norm(p[:, j] - p[:, PARENT[j]], axis=-1) for j in range(1, 17)]
    return np.stack(out, axis=1).astype(np.float32)


def hinge_bend_deg(prediction: np.ndarray) -> dict[str, np.ndarray]:
    p = np.asarray(prediction, dtype=np.float64)
    out = {}
    for name, (a, b, c) in HINGES.items():
        u = p[:, b] - p[:, a]
        v = p[:, c] - p[:, b]
        nu = np.linalg.norm(u, axis=-1)
        nv = np.linalg.norm(v, axis=-1)
        cos = np.clip((u * v).sum(-1) / np.maximum(nu * nv, 1e-12), -1.0, 1.0)
        out[name] = np.degrees(np.arccos(cos))   # 0 = straight
    return out


FEATURE_LAYOUT = (("geometry", 68), ("prediction_root_relative", 51), ("reprojection_residual", 17),
                  ("reprojection_scale", 1), ("bone_lengths", 16), ("invalid_limb_count", 1),
                  ("valid_score_mean", 1), ("valid_score_max", 1), ("valid_score_min", 1))
FEATURE_DIM = sum(n for _, n in FEATURE_LAYOUT)


def runtime_features(inputs: dict[str, np.ndarray]) -> np.ndarray:
    """(N, FEATURE_DIM) float32 from ONLY the frozen 2D geometry and the frozen S0 hypothesis."""
    geometry, prediction = _check_runtime_inputs(inputs)
    n = geometry.shape[0]
    root_rel = prediction - prediction[:, :1]
    residual, scale = reprojection_residual(geometry, prediction)
    valid = geometry[:, :, 3] > 0.5
    score = geometry[:, :, 2].astype(np.float64)          # RAW detector score, unclipped (may exceed 1)
    invalid_limb = (~valid[:, list(LIMB_JOINTS)]).sum(1).astype(np.float32)
    vs = np.where(valid, score, np.nan)
    with np.errstate(all="ignore"):
        mean = np.where(valid.any(1), np.nanmean(np.where(valid.any(1)[:, None], vs, 0.0), axis=1), 0.0)
        mx = np.where(valid.any(1), np.nanmax(np.where(valid.any(1)[:, None], vs, 0.0), axis=1), 0.0)
        mn = np.where(valid.any(1), np.nanmin(np.where(valid.any(1)[:, None], vs, 0.0), axis=1), 0.0)
    parts = [geometry.reshape(n, -1), root_rel.reshape(n, -1), residual, scale[:, None], bone_lengths(prediction),
             invalid_limb[:, None], mean[:, None], mx[:, None], mn[:, None]]
    out = np.concatenate([np.asarray(p, dtype=np.float32) for p in parts], axis=1)
    if out.shape[1] != FEATURE_DIM or not np.isfinite(out).all():
        raise ValueError("feature construction failed")
    return out


# ----------------------------------------------------------------------------------------------------------------
# Evaluation-only target
# ----------------------------------------------------------------------------------------------------------------

def joint_errors(prediction: np.ndarray, target: np.ndarray, root: str = "pelvis") -> np.ndarray:
    """(N,16) per-joint 3D Euclidean error in metres, both sides root-relative. EVALUATION ONLY."""
    p = np.asarray(prediction, dtype=np.float64)
    g = np.asarray(target, dtype=np.float64)
    if root == "pelvis":
        p = p - p[:, :1]
        g = g - g[:, :1]
    elif root == "hip_mid":
        p = p - (p[:, 1:2] + p[:, 4:5]) / 2
        g = g - (g[:, 1:2] + g[:, 4:5]) / 2
    else:
        raise ValueError(root)
    e = np.linalg.norm(p - g, axis=-1)[:, list(TARGET_JOINTS)]
    if not np.isfinite(e).all():
        raise ValueError("non-finite evaluation error: invalid targets must be excluded, never zero-filled")
    return e


# ----------------------------------------------------------------------------------------------------------------
# Baselines (fit on TRAIN only; quantiles in log-error space)
# ----------------------------------------------------------------------------------------------------------------

def log_err(e: np.ndarray) -> np.ndarray:
    return np.log(np.asarray(e, dtype=np.float64) + EPS_M)


def _quantiles(values: np.ndarray) -> np.ndarray:
    return np.quantile(values, QUANTILES)


class TableBaseline:
    """Nonparametric lookup of per-cell TRAIN log-error quantiles with a declared fallback hierarchy."""

    def __init__(self, name: str, keys_fn, levels: int):
        self.name = name
        self.keys_fn = keys_fn
        self.levels = levels
        self.tables: list[dict[tuple, np.ndarray]] = []
        self.edges: dict | None = None

    def fit(self, geometry, residual, y):
        self.edges = self._edges(geometry, residual)
        keys = self.keys_fn(self, geometry, residual)          # list over levels of (N,16) tuple keys
        self.tables = []
        for level in range(self.levels):
            groups: dict[tuple, list] = {}
            k = keys[level]
            for j in range(16):
                for i in range(y.shape[0]):
                    groups.setdefault((j,) + k[i][j], []).append(y[i, j])
            self.tables.append({key: _quantiles(np.asarray(v)) for key, v in groups.items() if len(v) >= MIN_CELL
                                or level == self.levels - 1})
        return self

    def _edges(self, geometry, residual):
        edges = {"score": [], "residual": []}
        for jj, j in enumerate(TARGET_JOINTS):
            valid = geometry[:, j, 3] > 0.5
            s = geometry[valid, j, 2]
            r = residual[valid, j]
            edges["score"].append(np.quantile(s, np.linspace(0.1, 0.9, 9)) if len(s) else np.zeros(9))
            edges["residual"].append(np.quantile(r, np.linspace(0.1, 0.9, 9)) if len(r) else np.zeros(9))
        return edges

    def predict(self, geometry, residual) -> np.ndarray:
        keys = self.keys_fn(self, geometry, residual)
        n = geometry.shape[0]
        out = np.zeros((n, 16, 2))
        for i in range(n):
            for j in range(16):
                for level in range(self.levels):
                    key = (j,) + keys[level][i][j]
                    if key in self.tables[level]:
                        out[i, j] = self.tables[level][key]
                        break
                else:
                    raise RuntimeError("no fallback cell")
        return out


def _score_bin(model, geometry):
    bins = np.zeros((geometry.shape[0], 16), dtype=np.int64)
    for jj, j in enumerate(TARGET_JOINTS):
        bins[:, jj] = np.searchsorted(model.edges["score"][jj], geometry[:, j, 2], side="right")
    return bins


def _residual_bin(model, residual):
    bins = np.zeros((residual.shape[0], 16), dtype=np.int64)
    for jj, j in enumerate(TARGET_JOINTS):
        bins[:, jj] = np.searchsorted(model.edges["residual"][jj], residual[:, j], side="right")
    return bins


def _valid(geometry):
    return (geometry[:, list(TARGET_JOINTS), 3] > 0.5).astype(np.int64)


def _keys_constant(model, geometry, residual):
    n = geometry.shape[0]
    return [[[() for _ in range(16)] for _ in range(n)]]


def _keys_score(model, geometry, residual):
    v = _valid(geometry)
    b = np.where(v == 1, _score_bin(model, geometry), -1)
    n = geometry.shape[0]
    return [[[(int(v[i, j]), int(b[i, j])) for j in range(16)] for i in range(n)],
            [[(int(v[i, j]),) for j in range(16)] for i in range(n)],
            [[() for _ in range(16)] for _ in range(n)]]


def _keys_score_missing(model, geometry, residual):
    v = _valid(geometry)
    b = np.where(v == 1, _score_bin(model, geometry), -1)
    m = np.minimum((geometry[:, list(LIMB_JOINTS), 3] < 0.5).sum(1), 3)
    n = geometry.shape[0]
    return [[[(int(v[i, j]), int(b[i, j]), int(m[i])) for j in range(16)] for i in range(n)],
            [[(int(v[i, j]), int(b[i, j])) for j in range(16)] for i in range(n)],
            [[(int(v[i, j]),) for j in range(16)] for i in range(n)],
            [[() for _ in range(16)] for _ in range(n)]]


def _keys_geometry(model, geometry, residual):
    v = _valid(geometry)
    b = np.where(v == 1, _residual_bin(model, residual), -1)
    n = geometry.shape[0]
    return [[[(int(v[i, j]), int(b[i, j])) for j in range(16)] for i in range(n)],
            [[(int(v[i, j]),) for j in range(16)] for i in range(n)],
            [[() for _ in range(16)] for _ in range(n)]]


def make_baselines() -> dict[str, TableBaseline]:
    return {"A_constant_per_joint": TableBaseline("A_constant_per_joint", _keys_constant, 1),
            "B_raw_score_validity": TableBaseline("B_raw_score_validity", _keys_score, 3),
            "B2_raw_score_validity_missing_count": TableBaseline("B2_raw_score_validity_missing_count",
                                                                 _keys_score_missing, 4),
            "C_reprojection_consistency": TableBaseline("C_reprojection_consistency", _keys_geometry, 3)}


# ----------------------------------------------------------------------------------------------------------------
# Learned assessor (small MLP ensemble, pinball loss on log-error)
# ----------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class AssessorConfig:
    hidden: int = 256
    depth: int = 2
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 80
    batch_size: int = 256
    seeds: tuple = (1231, 1232, 1233, 1234, 1235)
    device: str = "cpu"

    def to_dict(self):
        return dict(hidden=self.hidden, depth=self.depth, dropout=self.dropout, lr=self.lr,
                    weight_decay=self.weight_decay, epochs=self.epochs, batch_size=self.batch_size,
                    seeds=list(self.seeds), device=self.device, loss="pinball(tau=0.5,0.9) on log(e+1mm)",
                    optimizer="AdamW constant lr", activation="GELU", ensemble="mean of log-quantiles",
                    monotone_fix="q90 := max(q90, q50) after ensembling", early_stopping=None,
                    model_selection=None)


def _torch():
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    return torch


def build_mlp(torch, config: AssessorConfig, in_dim: int):
    layers = []
    width = in_dim
    for _ in range(config.depth):
        layers += [torch.nn.Linear(width, config.hidden), torch.nn.GELU(), torch.nn.Dropout(config.dropout)]
        width = config.hidden
    layers.append(torch.nn.Linear(width, 16 * len(QUANTILES)))
    return torch.nn.Sequential(*layers)


def train_assessor(features: np.ndarray, y: np.ndarray, config: AssessorConfig) -> dict:
    torch = _torch()
    mean = features.mean(0)
    std = np.maximum(features.std(0), 1e-6)
    x = torch.as_tensor(((features - mean) / std).astype(np.float32))
    t = torch.as_tensor(y.astype(np.float32))
    taus = torch.as_tensor(np.array(QUANTILES, dtype=np.float32)).view(1, 1, -1)
    states, curves = [], []
    for seed in config.seeds:
        torch.manual_seed(seed)
        model = build_mlp(torch, config, x.shape[1])
        opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        gen = torch.Generator().manual_seed(seed)
        curve = []
        for _ in range(config.epochs):
            model.train()
            order = torch.randperm(x.shape[0], generator=gen)
            total = 0.0
            for start in range(0, x.shape[0], config.batch_size):
                idx = order[start:start + config.batch_size]
                out = model(x[idx]).view(-1, 16, len(QUANTILES))
                diff = t[idx].unsqueeze(-1) - out
                loss = torch.maximum(taus * diff, (taus - 1) * diff).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss.detach()) * len(idx)
            curve.append(total / x.shape[0])
        states.append({k: v.detach().clone() for k, v in model.state_dict().items()})
        curves.append(curve)
    return {"mean": mean, "std": std, "states": states, "curves": curves, "config": config, "in_dim": x.shape[1]}


def assessor_predict(bundle: dict, features: np.ndarray) -> np.ndarray:
    """(N,16,2) ensemble log-error quantiles (uncalibrated)."""
    torch = _torch()
    x = torch.as_tensor(((features - bundle["mean"]) / bundle["std"]).astype(np.float32))
    outs = []
    for state in bundle["states"]:
        model = build_mlp(torch, bundle["config"], bundle["in_dim"])
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            outs.append(model(x).view(-1, 16, len(QUANTILES)).double().numpy())
    pred = np.mean(outs, axis=0)
    pred[..., 1] = np.maximum(pred[..., 1], pred[..., 0])
    return pred


def assessor_weights_sha(bundle: dict) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(bundle["mean"]).tobytes())
    h.update(np.ascontiguousarray(bundle["std"]).tobytes())
    for state in bundle["states"]:
        for k in sorted(state):
            h.update(k.encode())
            h.update(state[k].numpy().tobytes())
    return h.hexdigest()


# ----------------------------------------------------------------------------------------------------------------
# Split-conformal calibration (log space, per quantile, pooled over joints)
# ----------------------------------------------------------------------------------------------------------------

def conformal_shift(raw_log_q: np.ndarray, y: np.ndarray) -> list[float]:
    """Additive log-space shift per quantile so the CALIBRATION coverage reaches tau (finite-sample rank)."""
    shifts = []
    for k, tau in enumerate(QUANTILES):
        r = (y - raw_log_q[..., k]).reshape(-1)
        n = r.size
        level = min(1.0, math.ceil((n + 1) * tau) / n)
        shifts.append(float(np.quantile(r, level, method="higher")))
    return shifts


def apply_calibration(raw_log_q: np.ndarray, shifts: Iterable[float]) -> np.ndarray:
    """Metres, finite, nonnegative; q90 >= q50 is preserved by monotone exp/shift only if shift90>=shift50."""
    shifts = list(shifts)
    out = np.exp(raw_log_q + np.asarray(shifts)[None, None, :]) - EPS_M
    out = np.maximum(out, 0.0)
    out[..., 1] = np.maximum(out[..., 1], out[..., 0])
    return out


# ----------------------------------------------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------------------------------------------

def rankdata(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    s = a[order]
    ranks = np.empty(len(a), dtype=np.float64)
    boundaries = np.flatnonzero(np.diff(s) != 0) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(a)]])
    avg = (starts + ends - 1) / 2.0 + 1.0
    ranks[order] = np.repeat(avg, ends - starts)
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    rx, ry = rankdata(x), rankdata(y)
    rx -= rx.mean()
    ry -= ry.mean()
    den = math.sqrt(float((rx * rx).sum() * (ry * ry).sum()))
    return float((rx * ry).sum() / den) if den > 0 else 0.0


def average_precision(score: np.ndarray, positive: np.ndarray) -> float:
    """AP with ties grouped (a tied block is admitted together): sum over thresholds of P * dR."""
    score = np.asarray(score, dtype=np.float64)
    positive = np.asarray(positive, dtype=bool)
    total = int(positive.sum())
    if total == 0:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    s, p = score[order], positive[order]
    last = np.concatenate([np.flatnonzero(np.diff(s) != 0), [len(s) - 1]])
    tp = np.cumsum(p)[last]
    fp = (last + 1) - tp
    precision = tp / (tp + fp)
    recall = tp / total
    prev = np.concatenate([[0.0], recall[:-1]])
    return float(((recall - prev) * precision).sum())


def tiebreak_keys(cell_ids: list[str]) -> np.ndarray:
    """Label-free deterministic tie-break (blake2b of the cell identity)."""
    return np.array([int.from_bytes(hashlib.blake2b(c.encode(), digest_size=8).digest(), "big") for c in cell_ids],
                    dtype=np.uint64)


def triage_flags(risk: np.ndarray, video: np.ndarray, tiebreak: np.ndarray, budget: float = REVIEW_BUDGET) -> np.ndarray:
    """Per-video top-``ceil(budget*n)`` cells by risk (descending), ties by the label-free key."""
    flags = np.zeros(len(risk), dtype=bool)
    for v in np.unique(video):
        idx = np.flatnonzero(video == v)
        k = int(math.ceil(budget * len(idx)))
        order = np.lexsort((tiebreak[idx], -risk[idx]))
        flags[idx[order[:k]]] = True
    return flags


def coverage(e: np.ndarray, q: np.ndarray) -> float:
    return float((e <= q).mean()) if len(e) else float("nan")


def binned_calibration(e: np.ndarray, q50: np.ndarray, q90: np.ndarray, bins: int = 5) -> list[dict]:
    edges = np.quantile(q90, np.linspace(0, 1, bins + 1))
    out = []
    which = np.clip(np.searchsorted(edges[1:-1], q90, side="right"), 0, bins - 1)
    for b in range(bins):
        m = which == b
        if not m.any():
            out.append({"bin": b, "n": 0})
            continue
        out.append({"bin": b, "n": int(m.sum()), "q90_range_mm": [float(q90[m].min() * 1000), float(q90[m].max() * 1000)],
                    "coverage_q90": coverage(e[m], q90[m]), "coverage_q50": coverage(e[m], q50[m]),
                    "median_pred_q50_mm": float(np.median(q50[m]) * 1000), "median_err_mm": float(np.median(e[m]) * 1000),
                    "mean_err_mm": float(e[m].mean() * 1000), "p90_err_mm": float(np.quantile(e[m], 0.9) * 1000),
                    "material_rate": float((e[m] > MATERIAL_M).mean())})
    return out


def strata_metrics(e, q50, q90) -> dict:
    if len(e) == 0:
        return {"n": 0}
    mat = e > MATERIAL_M
    return {"n": int(len(e)), "mean_err_mm": float(e.mean() * 1000), "median_err_mm": float(np.median(e) * 1000),
            "p90_err_mm": float(np.quantile(e, 0.9) * 1000), "material_n": int(mat.sum()),
            "coverage_q90": coverage(e, q90), "coverage_q50": coverage(e, q50),
            "spearman_q50": spearman(q50, e) if len(e) >= 3 else None,
            "median_pred_q90_mm": float(np.median(q90) * 1000),
            "false_reassurance_n": int((mat & (q90 < MATERIAL_M)).sum()),
            "false_reassurance_rate": float((mat & (q90 < MATERIAL_M)).sum() / mat.sum()) if mat.any() else None}


def within_joint_spearman(q50, e, joint) -> float:
    """Mean over joints of the per-joint Spearman (removes the per-joint prior; constant predictors score 0)."""
    vals = [spearman(q50[joint == j], e[joint == j]) for j in np.unique(joint) if (joint == j).sum() >= 3]
    return float(np.mean(vals)) if vals else float("nan")


def endpoint_set(e, q50, q90, video, tiebreak, joint=None) -> dict:
    mat = e > MATERIAL_M
    flags = triage_flags(q90, video, tiebreak)
    accept = np.zeros(len(e), dtype=bool)
    order = np.lexsort((tiebreak, q90))
    accept[order[:int(math.floor(ACCEPT_FRACTION * len(e)))]] = True
    excess = np.maximum(e - MATERIAL_M, 0)
    sec = e > SECONDARY_M
    return {"n_cells": int(len(e)), "material_n": int(mat.sum()),
            "spearman_q50": spearman(q50, e), "ap_material_q90": average_precision(q90, mat),
            "spearman_within_joint_mean": within_joint_spearman(q50, e, joint) if joint is not None else None,
            "ap_material_base_rate": float(mat.mean()),
            "ap_secondary_q90": average_precision(q90, sec),
            "coverage_q90": coverage(e, q90), "coverage_q50": coverage(e, q50),
            "triage_flagged": int(flags.sum()), "triage_material_flagged": int((flags & mat).sum()),
            "triage_recall": float((flags & mat).sum() / mat.sum()) if mat.any() else None,
            "triage_precision": float((flags & mat).sum() / flags.sum()) if flags.any() else None,
            "triage_excess_capture": float(excess[flags].sum() / excess.sum()) if excess.sum() > 0 else None,
            "triage_false_negative_rate": float((mat & ~flags).sum() / mat.sum()) if mat.any() else None,
            "accept_fraction": ACCEPT_FRACTION, "accepted_mean_err_mm": float(e[accept].mean() * 1000),
            "all_mean_err_mm": float(e.mean() * 1000), "accepted_material_rate": float(mat[accept].mean()),
            "all_material_rate": float(mat.mean()),
            "false_reassurance_n": int((mat & (q90 < MATERIAL_M)).sum()),
            "false_reassurance_rate": float((mat & (q90 < MATERIAL_M)).sum() / mat.sum()) if mat.any() else None}


def cluster_bootstrap(cluster: np.ndarray, fn, reps: int = 2000, seed: int = 1337) -> dict:
    """Resample whole clusters (videos/subjects); ``fn(index_array) -> float``. No CI with <2 clusters."""
    groups = [np.flatnonzero(cluster == c) for c in np.unique(cluster)]
    point = fn(np.concatenate(groups))
    if len(groups) < 2:
        return {"point": point, "ci95": None, "clusters": len(groups)}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(groups), (reps, len(groups)))
    vals = []
    for d in draws:
        vals.append(fn(np.concatenate([groups[i] for i in d])))
    vals = np.asarray(vals, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    return {"point": point, "ci95": np.percentile(vals, [2.5, 97.5]).tolist() if len(vals) else None,
            "clusters": len(groups), "reps_finite": int(len(vals))}


# ----------------------------------------------------------------------------------------------------------------
# Scratch reliability sidecar (source-bound; never written into Production schemas)
# ----------------------------------------------------------------------------------------------------------------

SIDECAR_SCHEMA = "w123_scratch_reliability_sidecar_v1"
RELIABILITY_TYPE = "predicted_per_joint_3d_error_quantiles"
SIDECAR_FIELDS = ("frame_index", "timestamp", "q50_m", "q90_m", "support", "detector_raw_score", "input_valid")
SUPPORT_CODES = {"SUPPORTED_OBSERVED": 0, "SUPPORTED_INPUT_MISSING": 1, "UNKNOWN": 2}


def build_sidecar(*, source: dict, frame_index: np.ndarray, timestamp: np.ndarray, geometry: np.ndarray,
                  prediction: np.ndarray, q50: np.ndarray, q90: np.ndarray, checkpoint_sha256: str,
                  assessor_id: str, calibration_id: str, observation_regime: str,
                  subject_identity: str = "UNKNOWN") -> dict:
    frame_index = np.asarray(frame_index, dtype=np.int64)
    if np.any(np.diff(frame_index) <= 0):
        raise ValueError("SIDECAR_REORDERED_OR_DUPLICATE_FRAMES")
    if q50.shape != (len(frame_index), 16) or q90.shape != q50.shape:
        raise ValueError("SIDECAR_SHAPE")
    if not (np.isfinite(q50).all() and np.isfinite(q90).all()) or (q50 < 0).any() or (q90 < q50).any():
        raise ValueError("SIDECAR_RELIABILITY_INVALID")
    valid = geometry[:, list(TARGET_JOINTS), 3] > 0.5
    support = np.where(valid, SUPPORT_CODES["SUPPORTED_OBSERVED"], SUPPORT_CODES["SUPPORTED_INPUT_MISSING"])
    return {
        "schema": SIDECAR_SCHEMA,
        "source": dict(source),
        "observation_regime": observation_regime,
        "subject_identity": subject_identity,
        "coordinate_frame": COORDINATE_FRAME,
        "joint_names": list(TARGET_NAMES),
        "frozen_model": {"name": "S0_neutral_sign", "checkpoint_sha256": checkpoint_sha256},
        "pose_hypothesis_sha256": sha_array(np.asarray(prediction, dtype=np.float32)),
        "observation_sha256": sha_array(np.asarray(geometry, dtype=np.float32)),
        "reliability": {"type": RELIABILITY_TYPE, "units": "metres", "quantiles": list(QUANTILES),
                        "assessor_id": assessor_id, "calibration_id": calibration_id,
                        "semantics": "q90: predicted level that the frozen S0 joint 3D error (root-relative) stays "
                                     "below with probability 0.90 under the 3DPW-calibrated regime; NOT detector "
                                     "probability, NOT a hard constraint, NOT GT confidence"},
        "detector_score_semantics": "RAW detector score copied unchanged; not a probability; may exceed 1",
        "support_codes": SUPPORT_CODES,
        "frame_index": frame_index.tolist(),
        "timestamp": np.asarray(timestamp, dtype=np.float64).tolist(),
        "q50_m": np.asarray(q50, dtype=np.float64).tolist(),
        "q90_m": np.asarray(q90, dtype=np.float64).tolist(),
        "support": support.tolist(),
        "detector_raw_score": np.asarray(geometry[:, list(TARGET_JOINTS), 2], dtype=np.float64).tolist(),
        "input_valid": valid.tolist(),
    }


def _contains_forbidden(obj, path="") -> list[str]:
    hits = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            low = str(k).lower()
            if any(tok == low or low.startswith(tok + "_") or low.endswith("_" + tok) for tok in FORBIDDEN_TOKENS):
                hits.append(path + "/" + str(k))
            hits += _contains_forbidden(v, path + "/" + str(k))
    return hits


def validate_sidecar(sidecar: dict, *, checkpoint_sha256: str, prediction: np.ndarray, frame_index: np.ndarray,
                     geometry: np.ndarray, assessor_id: str, calibration_id: str) -> list[str]:
    """Return a list of violation codes (empty = valid)."""
    errors = []
    if sidecar.get("schema") != SIDECAR_SCHEMA:
        errors.append("SCHEMA")
    if _contains_forbidden(sidecar):
        errors.append("GT_OR_LABEL_FIELD_PRESENT")
    if sidecar.get("frozen_model", {}).get("checkpoint_sha256") != checkpoint_sha256:
        errors.append("STALE_CHECKPOINT")
    if sidecar.get("pose_hypothesis_sha256") != sha_array(np.asarray(prediction, dtype=np.float32)):
        errors.append("POSE_HYPOTHESIS_MISMATCH")
    if sidecar.get("observation_sha256") != sha_array(np.asarray(geometry, dtype=np.float32)):
        errors.append("OBSERVATION_MISMATCH")
    if sidecar.get("joint_names") != list(TARGET_NAMES):
        errors.append("JOINT_ORDER")
    fi = np.asarray(sidecar.get("frame_index", []), dtype=np.int64)
    if fi.shape != np.asarray(frame_index).shape or not np.array_equal(fi, frame_index):
        errors.append("FRAME_INDEX_MISMATCH")
    if len(fi) > 1 and np.any(np.diff(fi) <= 0):
        errors.append("REORDERED_FRAMES")
    rel = sidecar.get("reliability", {})
    if rel.get("assessor_id") != assessor_id or rel.get("calibration_id") != calibration_id:
        errors.append("STALE_ASSESSOR_OR_CALIBRATION")
    if rel.get("units") != "metres" or rel.get("type") != RELIABILITY_TYPE:
        errors.append("RELIABILITY_SEMANTICS")
    raw = np.asarray(sidecar.get("detector_raw_score", []), dtype=np.float64)
    if raw.shape != (len(frame_index), 16) or not np.array_equal(raw, geometry[:, list(TARGET_JOINTS), 2].astype(np.float64)):
        errors.append("RAW_SCORE_ALTERED")
    q50 = np.asarray(sidecar.get("q50_m", []), dtype=np.float64)
    q90 = np.asarray(sidecar.get("q90_m", []), dtype=np.float64)
    if q50.shape != (len(frame_index), 16) or q90.shape != q50.shape:
        errors.append("RELIABILITY_SHAPE")
    elif not (np.isfinite(q50).all() and np.isfinite(q90).all()) or (q50 < 0).any() or (q90 < q50).any():
        errors.append("RELIABILITY_INVALID")
    sup = np.asarray(sidecar.get("support", []))
    valid = geometry[:, list(TARGET_JOINTS), 3] > 0.5
    if sup.shape != valid.shape or not np.array_equal(sup == SUPPORT_CODES["SUPPORTED_OBSERVED"], valid):
        errors.append("SUPPORT_FLAGS")
    return errors


def illustrative_future_consumer(reference: np.ndarray, q90: np.ndarray, hard_pins: dict[tuple[int, int], np.ndarray],
                                 scale_m: float = MATERIAL_M) -> dict:
    """ILLUSTRATIVE ONLY — not a policy and not a Generator.

    Shows the separation a future Generator would need: a SOFT reference-trust weight may respond to predicted
    uncertainty; a user-authored HARD pin is a separate, binding target whose value and presence never depend on the
    reliability signal. The weight mapping here is an arbitrary placeholder, explicitly not a chosen trust policy.
    """
    q90 = np.asarray(q90, dtype=np.float64)
    soft_weight = 1.0 / (1.0 + (q90 / scale_m) ** 2)
    hard = {key: np.asarray(value, dtype=np.float64).copy() for key, value in hard_pins.items()}
    return {"soft_reference": np.asarray(reference, dtype=np.float64).copy(), "soft_weight": soft_weight,
            "hard_pins": hard, "mapping_status": "ILLUSTRATIVE_NOT_POLICY",
            "hard_pin_rule": "binding regardless of reliability; never removed, scaled or re-targeted by uncertainty"}
