"""DIAGNOSTIC (Worklog 69): continuous left-arm forward fraction from frozen Qwen3-VL readouts.

Same inputs as Worklog 68 (13 pair-geometry features + frozen endpoint A / B /
global readouts from the unchanged Worklog 68 cache); the only conceptual
change is the output: a tanh-bounded continuous f = unit(segment).Y per
segment instead of three ordering classes.  C0 feeds exact zeros for every
visual feature, C1 the cache; nothing else differs.  Not imported by any
production module.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from framepose.learned_vision_sensor import GEOMETRY_FEATURES, HIDDEN, PAIRS, VISUAL_PROJECTION

CANDIDATES = ("C0_ZERO_VISION", "C1_QWEN_VISION")
# Worklog 68 pair order (S-E, E-W, S-W) is the Worklog 64 segment order (upper, lower, chain).
SEGMENT_OF_PAIR = tuple(seg for (_, _, _, seg) in PAIRS)


def candidate_visual(readouts: np.ndarray, candidate: str) -> np.ndarray:
    r = np.asarray(readouts)
    if candidate == "C0_ZERO_VISION":
        return np.zeros_like(r, dtype=np.float16)
    if candidate == "C1_QWEN_VISION":
        return r
    raise ValueError(f"unknown candidate {candidate!r}")


def build_regressor(channels: int):
    import torch
    from torch import nn

    class ContinuousArmDepth(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.visual = nn.Linear(channels, VISUAL_PROJECTION)
            self.heads = nn.ModuleList(
                nn.Sequential(nn.Linear(GEOMETRY_FEATURES + 3 * VISUAL_PROJECTION, HIDDEN), nn.GELU(),
                              nn.Linear(HIDDEN, 1), nn.Tanh())
                for _ in PAIRS)

        def forward(self, visual, geometry):
            """visual (B, 4, C) [S, E, W, global]; geometry (B, 3, 13) -> f (B, 3) in [-1, 1]."""
            v = self.visual(visual.float())
            out = []
            for k, (_, a, b, _) in enumerate(PAIRS):
                out.append(self.heads[k](torch.cat([geometry[:, k], v[:, a], v[:, b], v[:, 3]], dim=-1))[:, 0])
            return torch.stack(out, dim=1)

    return ContinuousArmDepth()


def sign_magnitude(pred: dict[str, np.ndarray], gt: np.ndarray, stable_sine: float) -> dict[str, Any]:
    """Sign accuracy on stable rows and | |pred| - |gt| | on sign-correct subsets.

    ``pred`` maps model name -> f predictions aligned with ``gt``.  For every
    pair of models the magnitude error is also reported on rows where both have
    the correct sign, which separates branch correction from magnitude.
    """
    gt = np.asarray(gt, float)
    stable = np.abs(gt) > stable_sine
    correct = {m: (np.sign(p) == np.sign(gt)) & stable for m, p in pred.items()}
    out: dict[str, Any] = {"stable_rows": int(stable.sum())}
    for m, p in pred.items():
        mag = np.abs(np.abs(p) - np.abs(gt))
        out[m] = {"stable_sign_accuracy": float(correct[m][stable].mean()) if stable.any() else None,
                  "magnitude_error_when_own_sign_correct": float(mag[correct[m]].mean()) if correct[m].any() else None,
                  "own_sign_correct_rows": int(correct[m].sum())}
    names = list(pred)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            both = correct[a] & correct[b]
            out[f"both_sign_correct:{a}|{b}"] = {
                "rows": int(both.sum()),
                **{f"{m}_magnitude_error": float(np.abs(np.abs(pred[m]) - np.abs(gt))[both].mean()) if both.any() else None
                   for m in (a, b)},
                "paired_improved_share_" + b + "_vs_" + a: float(np.mean(
                    np.abs(np.abs(pred[b]) - np.abs(gt))[both] < np.abs(np.abs(pred[a]) - np.abs(gt))[both])) if both.any() else None}
            only_a = correct[a] & ~correct[b]
            only_b = correct[b] & ~correct[a]
            out[f"sign_flip_counts:{a}|{b}"] = {f"{a}_only_correct": int(only_a.sum()), f"{b}_only_correct": int(only_b.sum())}
    return out


def predict(model, visual: np.ndarray, geometry: np.ndarray, positions: np.ndarray, device,
            visual_positions: np.ndarray | None = None) -> np.ndarray:
    """Inference from the cached visual readout + pair geometry only."""
    import torch

    vp = positions if visual_positions is None else visual_positions
    out = []
    model.eval()
    with torch.no_grad():
        for s in range(0, len(positions), 1024):
            v = torch.as_tensor(np.asarray(visual[np.asarray(vp[s:s + 1024])]), device=device)
            g = torch.as_tensor(geometry[np.asarray(positions[s:s + 1024])], device=device)
            out.append(model(v, g).cpu().numpy().astype(np.float64))
    return np.concatenate(out)
