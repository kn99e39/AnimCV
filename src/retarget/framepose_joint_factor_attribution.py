"""EVALUATION ONLY (Worklog 81): J0/J1 direction / reach / bend-side factorial attribution.

Every factor comes from one frozen Worklog 80 model's ONE predicted connected arm, measured by the
Worklog 62 ``measure_chain``: direction D from J_d, reach fraction R from J_r and bend-side evidence S
from J_s (``owned`` keeps every other ChainMeasure field from J0; the Worklog 61/62 solve uses only the
reach fraction and the side).  The Worklog 62 ``_runtime_side`` rule re-perpendicularizes the selected
side to the selected direction, and a degenerate source side keeps the historical rest-plane fallback.
V000 is exactly J0 and V111 exactly J1; every mixed candidate deliberately breaks the one-stage
ownership and is an attribution tool, never a runtime candidate.  Not imported by production code.
"""

from __future__ import annotations

from dataclasses import replace

from retarget.framepose_ik_gt_attribution import ChainMeasure

FACTORIAL = {f"v{d}{r}{s}": (d, r, s) for d in "01" for r in "01" for s in "01"}


def owned(j0: ChainMeasure, j1: ChainMeasure, d: str, r: str, s: str):
    """(measure, direction) for factor sources d/r/s in {"0", "1"} (J0 / J1)."""
    src = {"0": j0, "1": j1}
    if any(x not in src for x in (d, r, s)):
        raise ValueError(f"unknown factor source {d}{r}{s}")
    m = j0 if (r, s) == ("0", "0") else replace(j0, reach_fraction=src[r].reach_fraction,
                                                  side=src[s].side, degenerate=src[s].degenerate)
    return m, src[d].direction


def effects(y: dict) -> dict:
    """Factorial contrasts of one metric (y keyed v000..v111).

    total            y111 - y000
    swap_from_j0     one factor J0 -> J1 with the other two at J0
    revert_from_j1   y111 minus J1 with that one factor reverted to J0 (leave-one-out)
    main             mean over the other two factors of the J1-minus-J0 effect of that factor
    two-way (dr, ds, rs)  mean over the third factor of the difference of conditional effects
    three_way        (y111 - y011 - y101 + y001) - (y110 - y010 - y100 + y000)
    """
    v = lambda d, r, s: y[f"v{d}{r}{s}"]  # noqa: E731
    out = {"total": v(1, 1, 1) - v(0, 0, 0),
           "swap_from_j0": {"direction": v(1, 0, 0) - v(0, 0, 0), "reach": v(0, 1, 0) - v(0, 0, 0), "side": v(0, 0, 1) - v(0, 0, 0)},
           "revert_from_j1": {"direction": v(1, 1, 1) - v(0, 1, 1), "reach": v(1, 1, 1) - v(1, 0, 1), "side": v(1, 1, 1) - v(1, 1, 0)}}
    out["main"] = {
        "direction": sum(v(1, r, s) - v(0, r, s) for r in (0, 1) for s in (0, 1)) / 4,
        "reach": sum(v(d, 1, s) - v(d, 0, s) for d in (0, 1) for s in (0, 1)) / 4,
        "side": sum(v(d, r, 1) - v(d, r, 0) for d in (0, 1) for r in (0, 1)) / 4}
    out["interaction"] = {
        "direction_x_reach": sum((v(1, 1, s) - v(0, 1, s)) - (v(1, 0, s) - v(0, 0, s)) for s in (0, 1)) / 2,
        "direction_x_side": sum((v(1, r, 1) - v(0, r, 1)) - (v(1, r, 0) - v(0, r, 0)) for r in (0, 1)) / 2,
        "reach_x_side": sum((v(d, 1, 1) - v(d, 0, 1)) - (v(d, 1, 0) - v(d, 0, 0)) for d in (0, 1)) / 2,
        "three_way": (v(1, 1, 1) - v(0, 1, 1) - v(1, 0, 1) + v(0, 0, 1)) - (v(1, 1, 0) - v(0, 1, 0) - v(1, 0, 0) + v(0, 0, 0))}
    return out


def flat_effects(y: dict) -> dict:
    """``effects`` flattened to 'group:name' keys (for per-row bootstrapping)."""
    e = effects(y)
    out = {"total": e["total"]}
    for group in ("swap_from_j0", "revert_from_j1", "main", "interaction"):
        out.update({f"{group}:{k}": val for k, val in e[group].items()})
    return out
