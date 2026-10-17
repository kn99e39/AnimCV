"""EXPERIMENTAL (Worklog 89): optional depth-tilt stabilizer for frozen connected-limb predictions.

Not imported by production FramePose.  It postprocesses the per-frame unit chains of a frozen single-frame model
(P = 0, M = rho u, D = M + (1 - rho) v; canonical camera axes X right, Y depth, Z up) and changes exactly one thing:
the depth tilt theta = asin(d_y) of each segment direction d in {u, v}.

  * theta_u(t), theta_v(t) are smoothed per sequence with a constant-velocity Kalman model in physical time
    (white-noise acceleration, spectral density ``q_accel`` rad^2/s^3; white measurement noise ``sigma_meas`` rad).
    ``mode="offline"`` runs the forward filter plus the Rauch-Tung-Striebel backward pass (uses the whole sequence,
    zero phase lag; for offline animation generation).  ``mode="causal"`` returns the forward filter only.
  * Each segment keeps its own raw image-plane heading h = (d_x, d_z) / |(d_x, d_z)| (what the 2D detector
    observes) and its raw ratio rho; the stabilized direction is d' = (cos theta' h_x, sin theta', cos theta' h_z).
    So d' is a unit vector, M' = rho u', D' = M' + (1 - rho) v' is a connected chain with the raw segment lengths and
    the raw image-plane directions, and P stays at the origin.
  * Runtime inputs only: frame timestamps (seconds), the sequence/track key, the model's predicted M and D, and a
    per-frame observation validity flag (the detector's chain joints).  No GT, no motion labels, no sequence names.
  * Frames whose observation is invalid get no measurement update (the state is propagated); a time gap larger
    than ``reset_gap_s`` or a new track key restarts the state.  Segments with a degenerate image-plane heading
    (|(d_x, d_z)| < ``min_horizontal``) keep their raw direction.

All parameters are in seconds / radians, so their physical meaning does not change with the frame rate: with a
longer frame interval the predicted uncertainty q dt^3 grows and the filter trusts each measurement more.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class StabilizerConfig:
    sigma_meas: float  # rad: white measurement noise of a segment depth tilt
    q_accel: float  # rad^2 / s^3: white-noise-acceleration spectral density of the true depth tilt
    reset_gap_s: float = 1.0
    init_velocity_sd: float = np.pi  # rad/s: prior spread of the tilt velocity at a (re)start
    mode: str = "offline"
    min_horizontal: float = 1e-6

    def as_dict(self) -> dict:
        return asdict(self)


def decompose(e, w):
    """Unit chain -> (u, v, rho)."""
    e, w = np.asarray(e, np.float64), np.asarray(w, np.float64)
    rho = np.linalg.norm(e, axis=-1)
    u = e / np.maximum(rho[..., None], 1e-12)
    seg = w - e
    v = seg / np.maximum(np.linalg.norm(seg, axis=-1)[..., None], 1e-12)
    return u, v, rho


def depth_tilt(d) -> np.ndarray:
    return np.arcsin(np.clip(np.asarray(d, np.float64)[..., 1], -1.0, 1.0))


def with_tilt(d, theta, min_horizontal=1e-6) -> np.ndarray:
    """Replace the depth tilt of unit directions d, keeping their image-plane heading."""
    d = np.asarray(d, np.float64)
    hx, hz = d[..., 0], d[..., 2]
    n = np.hypot(hx, hz)
    ok = n >= min_horizontal
    # stop just short of +-90 deg so that the image-plane heading is never lost
    theta = np.clip(theta, -np.pi / 2 + 1e-6, np.pi / 2 - 1e-6)
    c = np.cos(theta)
    out = np.stack([c * hx / np.where(ok, n, 1.0), np.sin(theta), c * hz / np.where(ok, n, 1.0)], axis=-1)
    return np.where(ok[..., None], out, d)


def kalman_cv(t, z, valid, cfg: StabilizerConfig) -> np.ndarray:
    """Constant-velocity Kalman filter (and RTS smoother when offline) of one uninterrupted track."""
    n = len(t)
    r = cfg.sigma_meas ** 2
    xs, ps, xp, pp = np.zeros((n, 2)), np.zeros((n, 2, 2)), np.zeros((n, 2)), np.zeros((n, 2, 2))
    fs = np.zeros((n, 2, 2))
    started = False
    x, p = np.zeros(2), np.eye(2)
    for i in range(n):
        if not started:
            x = np.array([z[i] if valid[i] else 0.0, 0.0])
            p = np.diag([r if valid[i] else (np.pi / 2) ** 2, cfg.init_velocity_sd ** 2])
            f = np.eye(2)
            started = True
            xp[i], pp[i] = x, p
        else:
            dt = t[i] - t[i - 1]
            f = np.array([[1.0, dt], [0.0, 1.0]])
            q = cfg.q_accel * np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]])
            x, p = f @ x, f @ p @ f.T + q
            xp[i], pp[i] = x, p
            if valid[i]:
                s = p[0, 0] + r
                k = p[:, 0] / s
                x = x + k * (z[i] - x[0])
                p = p - np.outer(k, p[0, :])
        fs[i] = f
        xs[i], ps[i] = x, p
    if cfg.mode == "causal":
        return xs[:, 0]
    if cfg.mode != "offline":
        raise ValueError(cfg.mode)
    sm = xs.copy()
    for i in range(n - 2, -1, -1):
        g = ps[i] @ fs[i + 1].T @ np.linalg.inv(pp[i + 1])
        sm[i] = xs[i] + g @ (sm[i + 1] - xp[i + 1])
    return sm[:, 0]


def tracks(keys, times, reset_gap_s):
    """Split frame indices into uninterrupted tracks (same key, increasing time, gaps <= reset_gap_s)."""
    keys, times = np.asarray(keys), np.asarray(times, np.float64)
    order = np.lexsort((times, keys))
    out, cur = [], []
    for i in order:
        if cur and (keys[i] != keys[cur[-1]] or not (0 < times[i] - times[cur[-1]] <= reset_gap_s)):
            out.append(np.array(cur))
            cur = []
        cur.append(i)
    if cur:
        out.append(np.array(cur))
    return out


def stabilize(keys, times, e, w, valid, cfg: StabilizerConfig | None):
    """Stabilize unit chains (M = e, D = w) frame-wise; ``cfg=None`` returns the input unchanged (switch off)."""
    e, w = np.asarray(e, np.float64), np.asarray(w, np.float64)
    if cfg is None:
        return e.copy(), w.copy()
    u, v, rho = decompose(e, w)
    tu, tv = depth_tilt(u), depth_tilt(v)
    su, sv = tu.copy(), tv.copy()
    valid = np.asarray(valid, bool) & np.isfinite(tu) & np.isfinite(tv)
    for idx in tracks(keys, times, cfg.reset_gap_s):
        tt = np.asarray(times, np.float64)[idx]
        su[idx] = kalman_cv(tt, tu[idx], valid[idx], cfg)
        sv[idx] = kalman_cv(tt, tv[idx], valid[idx], cfg)
    u2, v2 = with_tilt(u, su, cfg.min_horizontal), with_tilt(v, sv, cfg.min_horizontal)
    e2 = rho[:, None] * u2
    return e2, e2 + (1 - rho)[:, None] * v2


def calibrate(theta_pred, theta_gt, keys, times, step_s, tol=1e-6) -> dict:
    """OFFLINE CALIBRATION ONLY (uses GT; never called at runtime).  Moment calibration of (sigma_meas, q_accel) from equally spaced consecutive triples of one calibration split.

    sigma_meas: robust (1.4826 MAD) scale of the second difference of the tilt error e = theta_pred - theta_gt,
      divided by sqrt(6) (a white error contributes 6 sigma^2 to the second-difference variance);
    q_accel: 1.5 mean((second difference of theta_gt)^2) / dt^3 (white-noise acceleration gives 2 q dt^3 / 3).
    The robust scale keeps detector failures from inflating the noise, the mean of squares keeps the energy of real
    motion; both choices err towards less smoothing.  theta arrays may hold several segments per frame (N, k);
    their second differences are pooled.
    """
    keys, times = np.asarray(keys), np.asarray(times, np.float64)
    tp = np.asarray(theta_pred, np.float64).reshape(len(keys), -1)
    tg = np.asarray(theta_gt, np.float64).reshape(len(keys), -1)
    order = np.lexsort((times, keys))
    d2e, d2g = [], []
    for a, b, c in zip(order[:-2], order[1:-1], order[2:]):
        if keys[a] == keys[b] == keys[c] and abs(times[b] - times[a] - step_s) < tol and abs(times[c] - times[b] - step_s) < tol:
            e = tp[[a, b, c]] - tg[[a, b, c]]
            d2e.append(e[0] - 2 * e[1] + e[2])
            d2g.append(tg[a] - 2 * tg[b] + tg[c])
    d2e, d2g = np.concatenate(d2e), np.concatenate(d2g)
    ok = np.isfinite(d2e) & np.isfinite(d2g)
    d2e, d2g = d2e[ok], d2g[ok]
    mad = 1.4826 * np.median(np.abs(d2e - np.median(d2e)))
    return {"sigma_meas": float(mad / np.sqrt(6.0)), "q_accel": float(1.5 * np.mean(d2g ** 2) / step_s ** 3),
            "second_differences": int(len(d2e)), "step_s": step_s}
