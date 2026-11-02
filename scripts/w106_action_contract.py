#!/usr/bin/env python3
"""W106 evaluation contract: pure-Python helpers shared by the host stages, the Blender stages and the tests.

Nothing here touches Blender, a rig file or a production module.  It defines

  * quaternion arithmetic (w, x, y, z) used for the E1 local-rotation edit and its exact inverse,
  * the CONSTANT-interpolation step model used to predict what an E2 key-time edit should look like,
  * the order/domain-preserving key shift (refuses instead of merging or leaving the frame domain),
  * deterministic case / window selection from Action inventory metadata (never from edit outcomes),
  * the predeclared tolerances and the CASE mapping.

The edits are scratch-only evaluation interventions on copied Actions, not authoring functionality.
"""
from __future__ import annotations

import hashlib
import json
import math

SCHEMA = "w106_action_editability_v1"

# --- predeclared contract constants (frozen into the declaration before any edit is executed) ---------------------
TARGET_BONE = "upperarm_l"            # already animated in every W91/W102/W105 export (J1X or S0 left arm)
E1_AXIS = (1.0, 0.0, 0.0)            # bone-local X: perpendicular to the bone (Blender bones point along local Y)
E1_ANGLE_DEG = 20.0                  # modest upper-arm swing
E1_WINDOW_KEYS = 5                   # central keys n//2-2 .. n//2+2
E2_GROUP_KEYS = 3                    # central keys n//2-1 .. n//2+1
E2_INTEGER_OFFSET = 1.0              # used when both group boundaries leave >= 2 frames of room
E2_SUBFRAME_OFFSET = 0.5             # used when keys are dense (every frame); never interpolates new keys
MOVING_MIN_RANGE_DEG = 15.0          # a "moving" window: target track spans >= 15 deg from its first key
SYN_FRAMES = [1.0 + 3*k for k in range(9)]   # synthetic armature keys 1, 4, ..., 25 (scene 1..25 at 30 fps)
SYN_MISSING = {"sibling": 13.0}               # synthetic missing-key case on a non-target track

TOL = {
    "key_value_untouched": 0.0,          # untouched F-curve keys must be bitwise identical
    "local_sample_untouched": 1e-7,      # evaluated local channels of untouched bones
    "world_untouched_m": 1e-6,           # non-descendant world bone heads/tails (metres; float32 FK precision)
    "world_matrix_untouched": 1e-6,      # non-descendant world rotation-matrix elements
    "e1_disp_abs_err_m": 1e-4,           # elbow displacement vs exact 2 L sin(theta/2)
    "e1_min_visible_disp_m": 0.03,       # an edit below 3 cm at the elbow is not called visible
    "render_diff_level": 16,             # 8-bit channel difference that counts a pixel as changed
    "render_min_changed_px": 100,        # per edited frame, in at least one fixed view (480x480)
    "render_max_changed_px_outside": 0,  # frames outside the CONSTANT hold interval must render identically
    "inverse_key_value": 1e-6,           # E3: |q - q_inv| per component after sign canonicalisation
    "inverse_world_m": 1e-5,             # E3: every bone, world heads/tails vs N0
    "e2_model": 1e-7,                    # E2: Blender evaluation vs CONSTANT step model of the intended key times
    "fbx_rt_parity_m": 1e-3,             # FBX baseline import vs Blender N0, deforming bone heads/tails
    "fbx_nontarget_m": 1e-5,             # FBX edited import vs FBX baseline import, non-descendant bones
    "fbx_effect_m": 1e-3,                # FBX (edited - baseline) vs Blender (edited - baseline) target-subtree motion
}


class ContractViolation(ValueError):
    """An edit that would reorder/merge keys or leave the frame domain; refused, never repaired."""


# --- quaternions (w, x, y, z) ------------------------------------------------------------------------------------
def qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw*bw - ax*bx - ay*by - az*bz,
            aw*bx + ax*bw + ay*bz - az*by,
            aw*by - ax*bz + ay*bw + az*bx,
            aw*bz + ax*by - ay*bx + az*bw)


def qconj(q):
    return (q[0], -q[1], -q[2], -q[3])


def qaxis_angle(axis, deg):
    n = math.sqrt(sum(c*c for c in axis))
    h = math.radians(deg) / 2
    s = math.sin(h) / n
    return (math.cos(h), axis[0]*s, axis[1]*s, axis[2]*s)


def qangle_deg(a, b):
    """Rotation angle between two (not necessarily unit) quaternions, sign-invariant."""
    na = math.sqrt(sum(c*c for c in a))
    nb = math.sqrt(sum(c*c for c in b))
    d = abs(sum(x*y for x, y in zip(a, b))) / (na*nb)
    return math.degrees(2*math.acos(min(1.0, d)))


def qcanon(q):
    """Sign-canonical representative (first non-zero component positive)."""
    for c in q:
        if c != 0:
            return tuple(q) if c > 0 else tuple(-v for v in q)
    return tuple(q)


def e1_offset():
    return qaxis_angle(E1_AXIS, E1_ANGLE_DEG)


def apply_local_offset(q, offset):
    """E1: post-multiply, i.e. rotate in the bone's own posed frame (exact angle change of the bone direction)."""
    return qmul(q, offset)


def apply_inverse_offset(q, offset):
    return qmul(q, qconj(offset))


def expected_elbow_displacement(bone_length_m, deg=E1_ANGLE_DEG):
    return 2.0 * bone_length_m * math.sin(math.radians(deg) / 2)


# --- windows, key shift, CONSTANT step model -------------------------------------------------------------------
def e1_window(n_keys):
    if n_keys < E1_WINDOW_KEYS:
        raise ContractViolation(f"{n_keys} keys < E1 window {E1_WINDOW_KEYS}")
    i0 = n_keys//2 - E1_WINDOW_KEYS//2
    return list(range(i0, i0 + E1_WINDOW_KEYS))


def e2_group(n_keys):
    if n_keys < E2_GROUP_KEYS + 2:
        raise ContractViolation(f"{n_keys} keys too few for an interior E2 group")
    i0 = n_keys//2 - E2_GROUP_KEYS//2
    return list(range(i0, i0 + E2_GROUP_KEYS))


def e2_offset(frames, group):
    """Integer offset when both boundary gaps are >= 2 frames, else the half-frame offset (dense tracks)."""
    lo, hi = group[0], group[-1]
    gaps = []
    if lo > 0:
        gaps.append(frames[lo] - frames[lo-1])
    if hi < len(frames) - 1:
        gaps.append(frames[hi+1] - frames[hi])
    return E2_INTEGER_OFFSET if gaps and min(gaps) >= 2*E2_INTEGER_OFFSET else E2_SUBFRAME_OFFSET


def shift_keys(frames, group, delta, frame_start, frame_end):
    """New key times with ``group`` moved by ``delta``; strict order and [frame_start, frame_end] are required."""
    out = list(frames)
    for i in group:
        out[i] = frames[i] + delta
    if any(b <= a for a, b in zip(out, out[1:])):
        raise ContractViolation(f"shift {delta:+} would reorder or merge keys")
    if min(out[i] for i in group) < frame_start or max(out[i] for i in group) > frame_end:
        raise ContractViolation(f"shift {delta:+} would leave the frame domain [{frame_start}, {frame_end}]")
    return out


def step_value(frames, values, t):
    """Blender CONSTANT interpolation: value of the last key at or before t; the first key holds before it."""
    v = values[0]
    for f, x in zip(frames, values):
        if f <= t:
            v = x
        else:
            break
    return v


def hold_interval(frames, window):
    """Frames whose CONSTANT-evaluated value depends on keys in ``window``: [f_first, f_after_last)."""
    lo = frames[window[0]] if window[0] > 0 else -math.inf  # the first key also holds before itself
    hi = frames[window[-1]+1] if window[-1]+1 < len(frames) else math.inf
    return lo, hi


def sample_times(frame_start, frame_end):
    """Every integer frame plus every half frame strictly inside the domain (between-key probes)."""
    ints = [float(f) for f in range(frame_start, frame_end+1)]
    halves = [f + 0.5 for f in range(frame_start, frame_end)]
    return sorted(ints + halves)


def render_frames(frames, window):
    lo = max(0, window[0]-2)
    hi = min(len(frames)-1, window[-1]+2)
    return list(range(int(math.floor(frames[lo])), int(math.ceil(frames[hi]))+1))


# --- case selection (inventory metadata only) ------------------------------------------------------------------
def track_range_deg(quats):
    return max(qangle_deg(quats[0], q) for q in quats) if quats else 0.0


def select_moving_window(windows):
    """First case id in stable sorted order that is not the complete control, has only same-person rows, keys the
    target bone at every sampled frame, and whose original target track spans >= MOVING_MIN_RANGE_DEG."""
    reasons = {}
    chosen = None
    for cid in sorted(windows):
        w = windows[cid]
        why = []
        if cid == "complete_control":
            why.append("complete control (selected separately)")
        if set(w["identities"]) != {"same_person"}:
            why.append(f"identities {sorted(set(w['identities']))}")
        if not w["target_keyed_every_sample"]:
            why.append("target not keyed at every sample")
        if w["target_range_deg"] < MOVING_MIN_RANGE_DEG:
            why.append(f"target range {w['target_range_deg']:.1f} < {MOVING_MIN_RANGE_DEG}")
        if not why and chosen is None:
            chosen = cid
            why.append("ELIGIBLE: first qualifying in sorted order")
        elif not why:
            why.append("eligible, not first")
        reasons[cid] = why
    return chosen, reasons


# --- digests ----------------------------------------------------------------------------------------------------
def channel_digest(channels):
    """Same hashing convention as the W105 Action audit: data_path[index] then frame:value per key."""
    h = hashlib.sha256()
    for c in sorted(channels, key=lambda c: (c["data_path"], c["index"])):
        h.update(f"{c['data_path']}[{c['index']}]".encode())
        for f, v in zip(c["frames"], c["values"]):
            h.update(f"{f:.6f}:{v:.9f};".encode())
    return h.hexdigest()


def json_sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, indent=1).encode()).hexdigest()


# --- CASE mapping -----------------------------------------------------------------------------------------------
LAYER_ORDER = ("action_representation", "rig_hierarchy", "timebase", "serialization")


def decide_case(pre):
    """``pre`` keys: preflight_ok, n0_ok (per real case), determinism_ok, synthetic_ok, real (dict case -> dict of
    gate -> bool or None for structurally untestable), first_failing_layer (dict case -> layer or None).

    D: preflight / identity / N0 baseline equivalence / determinism / no real Action testable.
    B: a real Action testable and some E1/E2/E3/E4 gate fails (first failing layer reported).
    C: synthetic fails, or only a subset of real Actions testable, or some real timing effect not isolable.
    A: every declared real Action passes every Blender gate and the synthetic probe passes.
    FBX is reported separately and never upgrades or downgrades the Blender CASE.
    """
    if not pre["preflight_ok"]:
        return "D", "preflight / artifact identity failed"
    real = pre["real"]
    testable = {k: v for k, v in real.items() if v is not None}
    if not testable:
        return "D", "no declared real Action could be opened and associated"
    if not all(pre["n0_ok"].get(k, False) for k in testable):
        return "D", "N0 null save/reopen baseline equivalence not established"
    if not pre["determinism_ok"]:
        return "D", "repeated run on fresh copies not deterministic"
    failing = {k: [g for g, ok in v.items() if ok is False] for k, v in testable.items()}
    if any(failing.values()):
        layers = [pre["first_failing_layer"].get(k) for k, f in failing.items() if f]
        first = min((l for l in layers if l), key=LAYER_ORDER.index, default="unattributed")
        return "B", f"real Action gate failure; first failing layer {first}"
    if not pre["synthetic_ok"]:
        return "C", "real Actions pass but the synthetic contract probe fails"
    if len(testable) < len(real) or any(ok is None for v in testable.values() for ok in v.values()):
        return "C", "only a subset of declared real Actions / gates testable"
    return "A", "bounded Blender Action editability supported on every declared real Action"
