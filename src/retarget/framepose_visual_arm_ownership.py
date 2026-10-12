"""EVALUATION ONLY (Worklog 77): direction / reach / bend-side ownership counterfactuals for the left arm.

The Worklog 61 IK consumes three geometry quantities: chain direction, reach
fraction and bend-side evidence.  ``owned_measure`` builds the Worklog 62
``ChainMeasure`` whose reach fraction and observed bend side come from the
selected sources (frozen H0, or the GT chain as an evaluation-only oracle);
everything else is unchanged.  The Worklog 61/62 solve then re-perpendicularizes
the selected side to the candidate direction with ``_runtime_side`` exactly as in
Worklogs 62/74/76 (rest-plane fallback when the selected source is degenerate).
No new geometry, heuristic or learned quantity is introduced.

``motion_residual`` compares one arm's frame-to-frame motion with the
GT-anchored reference arm's motion on the same pair; ``direction_penalty``
and ``penalty_changes`` are the factorial contrasts used for attribution.
Not imported by production code.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from retarget.axis_utils import quaternion_conjugate, quaternion_multiply
from retarget.framepose_ik_gt_attribution import ChainMeasure
from retarget.framepose_two_bone_ik import quaternion_angle_degrees

# name -> (direction source, reach source, bend-side source); "1" = E1 / GT oracle in the C-code digits.
FACTORIAL = {
    "c000": ("h0", "h0", "h0"), "c100": ("e1", "h0", "h0"),
    "c010": ("h0", "gt", "h0"), "c110": ("e1", "gt", "h0"),
    "c001": ("h0", "h0", "gt"), "c101": ("e1", "h0", "gt"),
    "c011": ("h0", "gt", "gt"), "c111": ("e1", "gt", "gt"),
}
# Depth-oracle diagnostic: GT forward fraction on the SAME detector plane as E1, H0 reach and side.
DIAGNOSTICS = {"c_gtf_h0r_h0s": ("gt_f", "h0", "h0")}
# Oracle-geometry contract checks (not candidates): GT 3D direction with H0 or GT reach/side.
CONTRACT = {"k_gtdir_h0r_h0s": ("gt_dir", "h0", "h0"), "k_gtdir_gtr_gts": ("gt_dir", "gt", "gt")}
RUNTIME = ("c000", "c100")  # the only candidates without GT components


def owned_measure(h0: ChainMeasure, gt: ChainMeasure, reach: str, side: str) -> ChainMeasure:
    """H0 measure with reach fraction and bend-side evidence taken from the selected sources."""
    sources = {"h0": h0, "gt": gt}
    if reach not in sources or side not in sources:
        raise ValueError(f"unknown ownership {reach!r}/{side!r}")
    if reach == "h0" and side == "h0":
        return h0
    r, s = sources[reach], sources[side]
    return replace(h0, reach_fraction=r.reach_fraction, bend_angle_degrees=r.bend_angle_degrees,
                   side=s.side, degenerate=s.degenerate)


def _v(a) -> np.ndarray:
    return np.asarray(a, dtype=np.float64)


def _unit(a) -> np.ndarray:
    a = _v(a)
    return a / np.linalg.norm(a)


def _relative(q1, q0):
    return quaternion_multiply(tuple(q1), quaternion_conjugate(tuple(q0)))


RESIDUAL_METRICS = ("proximal_rotation_residual_degrees", "distal_rotation_residual_degrees",
                    "upper_direction_residual_degrees", "forearm_direction_residual_degrees",
                    "chain_direction_residual_degrees", "bend_side_residual_degrees",
                    "elbow_motion_residual", "wrist_motion_residual")


def motion_residual(a0, a1, r0, r1) -> dict:
    """How far an arm's motion over one adjacent pair departs from the reference arm's motion.

    Rotations: Worklog 61 quaternion distance between the two frame-to-frame relative
    rotations.  Unit directions: |delta_candidate - delta_reference| in degrees (radians of
    chord, small-angle).  Joints: |delta_candidate - delta_reference| in cm.  Zero exactly
    when the candidate moves like the reference (a constant offset is not penalized).
    """
    out = {
        "proximal_rotation_residual_degrees": quaternion_angle_degrees(
            _relative(a1.proximal_local, a0.proximal_local), _relative(r1.proximal_local, r0.proximal_local)),
        "distal_rotation_residual_degrees": quaternion_angle_degrees(
            _relative(a1.distal_local, a0.distal_local), _relative(r1.distal_local, r0.distal_local)),
        "elbow_motion_residual": float(np.linalg.norm((_v(a1.mid) - _v(a0.mid)) - (_v(r1.mid) - _v(r0.mid)))),
        "wrist_motion_residual": float(np.linalg.norm((_v(a1.end) - _v(a0.end)) - (_v(r1.end) - _v(r0.end)))),
    }
    vectors = {"upper": lambda x: _unit(_v(x.mid) - _v(x.root)), "forearm": lambda x: _unit(_v(x.end) - _v(x.mid)),
               "chain": lambda x: _unit(x.direction), "bend_side": lambda x: _unit(x.side)}
    for name, f in vectors.items():
        d = (f(a1) - f(a0)) - (f(r1) - f(r0))
        key = f"{name}_direction_residual_degrees" if name != "bend_side" else "bend_side_residual_degrees"
        out[key] = math.degrees(float(np.linalg.norm(d)))
    return out


def direction_penalty(y: dict) -> dict:
    """E1-minus-H0 direction effect under each reach/side ownership (y keyed by factorial name)."""
    return {"h0r_h0s": y["c100"] - y["c000"], "gtr_h0s": y["c110"] - y["c010"],
            "h0r_gts": y["c101"] - y["c001"], "gtr_gts": y["c111"] - y["c011"]}


def penalty_changes(y: dict) -> dict:
    """How the E1 direction penalty changes when reach and/or side become GT-owned, plus non-additivity."""
    p = direction_penalty(y)
    return {"reach": p["gtr_h0s"] - p["h0r_h0s"], "side": p["h0r_gts"] - p["h0r_h0s"],
            "both": p["gtr_gts"] - p["h0r_h0s"],
            "three_way_interaction": p["gtr_gts"] - p["gtr_h0s"] - p["h0r_gts"] + p["h0r_h0s"]}


def ownership_effects(y: dict) -> dict:
    """Reach and side effects under each direction (GT-owned minus H0-owned), and their interaction."""
    out = {}
    for d, prefix in (("h0", "c0"), ("e1", "c1")):
        y00, y10, y01, y11 = (y[f"{prefix}{r}{s}"] for r, s in (("0", "0"), ("1", "0"), ("0", "1"), ("1", "1")))
        out[d] = {"reach": y10 - y00, "side": y01 - y00, "both": y11 - y00, "reach_x_side": y11 - y10 - y01 + y00}
    return out
