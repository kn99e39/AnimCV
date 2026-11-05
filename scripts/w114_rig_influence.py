"""W114 frozen-rig influence harness (evidence only; no solver, rig, renderer or exporter change).

Two existing W105 rig configurations are evaluated as pure per-frame functions of their sources, using the unchanged
W102/W105 export functions (`w105_rig_utility.rig_measure`, `w102_rig_review.arm_export`, `motion.torso_frame`,
`run_external_orientation_handoff.handoff_frames`):

  real_s0_only   W105 real clip (GT-free): W93 torso frames from S0 -> Object (pelvis frame) and spine_01, S0 legs
                 through W95 two-bone IK, S0 left arm through the existing arm chain, right arm at rest
                 (`w105_rig_utility.build_clip`, replicated line for line below and checked byte-exactly)
  mixed_w102     W105 3DPW windows (W102 contract): HMR2 pelvis Object + HMR2 thorax frame, S0 legs, frozen J1X left
                 arm, right arm at rest (`w105_rig_utility.window_case`), checked byte-exactly against W105 cases.json

Channels: the eight W105 Action channels (Object rotation + 7 pose-bone quaternions). A channel is changed when its
hemisphere-aligned quaternion moves by more than the declared tolerance; exact-bit changes are reported separately.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

CHANNELS = ("__object__", "spine_01", "thigh_l", "calf_l", "thigh_r", "calf_r", "upperarm_l", "lowerarm_l")
REST_Q = [0.0, 0.0, 0.0, 1.0]
# Tail-joint projection: the joint(s) whose geometry a channel's LOCAL value places relative to its parent.
DEPICTS = {"__object__": ["pelvis", "left_hip", "right_hip"], "spine_01": ["thorax", "left_shoulder", "right_shoulder"],
           "thigh_l": ["left_knee"], "calf_l": ["left_ankle"], "thigh_r": ["right_knee"], "calf_r": ["right_ankle"],
           "upperarm_l": ["left_elbow"], "lowerarm_l": ["left_wrist"]}
OWNERS = {
    "real_s0_only": {"__object__": "S0 torso frame (pelvis, hips, thorax)", "spine_01": "S0 torso frames (thorax, shoulders vs pelvis frame)",
                     "thigh_l": "S0 left leg IK in S0 pelvis frame", "calf_l": "S0 left leg IK in S0 pelvis frame",
                     "thigh_r": "S0 right leg IK in S0 pelvis frame", "calf_r": "S0 right leg IK in S0 pelvis frame",
                     "upperarm_l": "S0 left arm IK in S0 thorax frame", "lowerarm_l": "S0 left arm IK in S0 thorax frame",
                     "right_arm": "fixed rest (not keyed)"},
    "mixed_w102": {"__object__": "HMR2 global_orient", "spine_01": "HMR2 pelvis vs HMR2 joints torso frame",
                   "thigh_l": "S0 left leg IK in HMR2 pelvis frame", "calf_l": "S0 left leg IK in HMR2 pelvis frame",
                   "thigh_r": "S0 right leg IK in HMR2 pelvis frame", "calf_r": "S0 right leg IK in HMR2 pelvis frame",
                   "upperarm_l": "J1X left arm IK in HMR2 thorax frame", "lowerarm_l": "J1X left arm IK in HMR2 thorax frame",
                   "right_arm": "fixed rest (not keyed)"},
}
# W113's conservative bone->S0-joint dependency map (own joints + torso frame), kept for comparison only.
W113_TORSO = ["pelvis", "left_hip", "right_hip", "thorax", "left_shoulder", "right_shoulder"]
W113_DEPENDENCIES = {
    "thigh_l": ["left_hip", "left_knee"] + W113_TORSO, "calf_l": ["left_hip", "left_knee", "left_ankle"] + W113_TORSO,
    "thigh_r": ["right_hip", "right_knee"] + W113_TORSO, "calf_r": ["right_hip", "right_knee", "right_ankle"] + W113_TORSO,
    "upperarm_l": ["left_shoulder", "left_elbow"] + W113_TORSO, "lowerarm_l": ["left_shoulder", "left_elbow", "left_wrist"] + W113_TORSO,
    "spine_01": W113_TORSO, "__object__": W113_TORSO}


def rig_context():
    from w105_rig_utility import rig_context as rc
    return rc()


# ----------------------------------------------------------------------------------------- real S0-only clip ---

def real_frame(i, P, t, rc):
    """One FIXED-slot frame of `w105_rig_utility.build_clip` (same calls, same order) for S0 pose P (17, 3)."""
    from motion.torso_frame import torso_frames
    from run_torso_frame_rig_replay import arm_pose
    from w102_rig_review import arm_export
    from w105_rig_utility import rig_measure, unit as _unit
    j = np.asarray(P, float)
    j = j - j[0]
    Rt, Rp = torso_frames(j)
    pose = rig_measure(j, Rp, Rt, rc) if Rp is not None and Rt is not None else None
    arm = None if pose is None else arm_export(j[12] - j[11], j[13] - j[11], Rp, Rt, rc)
    slot = {"pose": pose, "arm": arm}
    if pose is not None and arm is not None:
        ap = arm_pose(j[12] - j[11], j[13] - j[11], Rt, rc["alignment"], rc["target"], rc["rest"])
        slot["arm_rig"] = {"upper_w": _unit(Rt @ (rc["Am"].T @ np.asarray(ap.upper))).tolist(),
                           "fore_w": _unit(Rt @ (rc["Am"].T @ np.asarray(ap.forearm))).tolist(),
                           "elbow": float(ap.bend_angle_degrees())}
    slot["parent"] = None if Rp is None else {"Rp": Rp.tolist(), "Rt": Rt.tolist()}
    return slot


def slot_channels(pose, arm) -> np.ndarray:
    """(8, 4) channel quaternions in CHANNELS order; NaN when the rig emits no slot for the frame."""
    out = np.full((len(CHANNELS), 4), np.nan)
    if pose is None:
        return out
    e = pose["export"]
    out[0] = e["object"]
    for k, b in enumerate(CHANNELS[1:6], 1):
        out[k] = e["bones"][b]
    out[6] = arm["proximal_local_xyzw"] if arm is not None else REST_Q
    out[7] = arm["distal_local_xyzw"] if arm is not None else REST_Q
    return out


def real_channels(values, rc) -> dict:
    T = len(values)
    arr = np.stack([slot_channels(**{k: v for k, v in real_frame(i, values[i], None, rc).items() if k in ("pose", "arm")})
                    for i in range(T)])
    return {c: arr[:, k] for k, c in enumerate(CHANNELS)}


def real_case_frames(values, timestamps, rc, label):
    """Single-slot frame records in the W105 video_cases.json layout (used for the Blender replay and byte checks)."""
    frames = []
    for i in range(len(values)):
        s = real_frame(i, values[i], timestamps[i], rc)
        p, a = s["pose"], s["arm"]
        fr = {"bank_position": -1, "frame_index": i, "time_s": float(timestamps[i]), "image": f"frame_{i:04d}",
              "identity": "single_subject_GT_free", "runtime_status": "DIAGNOSTIC", "runtime_reasons": [], "yaw_deg": {},
              "modes": {}, "forward_cam": {}, "mode_lines": {}, "object_rotation_xyzw": {}, "extra_bone_locals_xyzw": {},
              "fk_points_armature": {}, "gt_valid": False}
        if p is None:
            fr["yaw_deg"]["SLOT"] = None
        else:
            fr["object_rotation_xyzw"]["SLOT"] = p["export"]["object"]
            fr["extra_bone_locals_xyzw"]["SLOT"] = dict(p["export"]["bones"])
            fr["fk_points_armature"]["SLOT"] = p["export"]["points"]
            fr["forward_cam"]["SLOT"] = p["export"]["forward"]
            fr["yaw_deg"]["SLOT"] = math.degrees(math.atan2(-p["export"]["forward"][0], p["export"]["forward"][1]))
            if a is not None:
                fr["modes"]["SLOT"] = a
            else:
                for bone in ("upperarm_l", "lowerarm_l"):
                    fr["extra_bone_locals_xyzw"]["SLOT"][bone] = list(REST_Q)
        fr["rig"] = {"SLOT": p and {"legs": p["legs"], "viol": p["viol"]}}
        fr["arm_rig"] = {"SLOT": s["arm_rig"]} if "arm_rig" in s else {}
        fr["parent"] = {"SLOT": s["parent"]}
        frames.append(fr)
    return frames


SLOT_FIELDS = ("yaw_deg", "modes", "forward_cam", "object_rotation_xyzw", "extra_bone_locals_xyzw", "fk_points_armature",
               "rig", "arm_rig", "parent")


def slot_view(frames, slot):
    """Per-frame dict of one slot's values in every slot-keyed field (for byte comparison with W105)."""
    return [{k: f.get(k, {}).get(slot) for k in SLOT_FIELDS} for f in frames]


def merge_slots(a_frames, b_frames, labels):
    """Two single-slot frame lists -> W105 two-slot records (FIXED = a, RUNTIME = b)."""
    out = []
    for fa, fb in zip(a_frames, b_frames):
        fr = {k: v for k, v in fa.items() if k not in SLOT_FIELDS}
        for k in SLOT_FIELDS:
            fr[k] = {}
            for slot, src in (("FIXED", fa), ("RUNTIME", fb)):
                if "SLOT" in src.get(k, {}):
                    fr[k][slot] = src[k]["SLOT"]
        fr["mode_lines"] = {"FIXED": [labels["FIXED"]], "RUNTIME": [labels["RUNTIME"]]}
        out.append(fr)
    return out


# ----------------------------------------------------------------------------------------- mixed W102 window ---

class MixedSources:
    """Frozen W102 window sources for one case: S0 (pred_M row), HMR2 (global_orient, joints24), J1X (elbow, wrist)."""

    def __init__(self, case, pred_M, hmr_npz, j1x_npz):
        hp = np.load(hmr_npz, allow_pickle=False)
        h = {int(p): i for i, (pop, p) in enumerate(zip(hp["population"], hp["index"][0])) if str(pop).startswith("3dpw") and hp["valid"][i]}
        j1 = np.load(j1x_npz, allow_pickle=False)
        frames = case["frames"]
        self.frames = frames
        self.rows = np.array([f["row"] for f in frames])
        self.positions = np.array([f["bank_position"] for f in frames])
        self.s0 = np.array(pred_M[self.rows], np.float32)
        self.hmr_present = np.array([int(p) in h for p in self.positions])
        idx = [h.get(int(p), 0) for p in self.positions]
        self.hmr_go = np.array(hp["global_orient"][idx], np.float32)
        self.hmr_j24 = np.array(hp["joints24"][idx], np.float32)
        # only the predicted J1X arrays are read; the npz's *_target_* arrays are evaluation GT and never touched
        self.j1x_elbow = np.array(j1["left_arm_g1_elbow"][self.positions], np.float64)
        self.j1x_wrist = np.array(j1["left_arm_g1_wrist"][self.positions], np.float64)


def mixed_slots(src: MixedSources, rc, s0=None, hmr_j24=None, j1x_wrist=None, j1x_elbow=None):
    """Per-frame (pose, arm) for the W102 contract, with optional substituted source arrays."""
    from run_external_orientation_handoff import handoff_frames
    from w102_rig_review import arm_export
    from w105_rig_utility import rig_measure
    s0 = src.s0 if s0 is None else s0
    j24 = src.hmr_j24 if hmr_j24 is None else hmr_j24
    jw = src.j1x_wrist if j1x_wrist is None else j1x_wrist
    je = src.j1x_elbow if j1x_elbow is None else j1x_elbow
    out = []
    for i in range(len(src.rows)):
        if not src.hmr_present[i]:
            out.append((None, None))
            continue
        Ro, Rt = handoff_frames(src.hmr_go[i], j24[i])[:2]
        if Rt is None:
            out.append((None, None))
            continue
        pose = rig_measure(s0[i], Ro, Rt, rc)
        arm = arm_export(je[i], jw[i], Ro, Rt, rc)
        out.append((pose, arm))
    return out


def mixed_channels(slots) -> dict:
    arr = np.stack([slot_channels(p, a) for p, a in slots])
    return {c: arr[:, k] for k, c in enumerate(CHANNELS)}


def mixed_reproduction(src: MixedSources, slots) -> dict:
    """Compare the recomputed FIXED slot with the stored W105 cases.json frame values (exact equality)."""
    eq = n = 0
    for fr, (pose, arm) in zip(src.frames, slots):
        if "FIXED" not in fr["object_rotation_xyzw"]:
            continue
        n += 1
        bones = dict(pose["export"]["bones"]) if pose else None
        if bones is not None and arm is None:
            bones.update({"upperarm_l": list(REST_Q), "lowerarm_l": list(REST_Q)})
        same = (pose is not None and fr["object_rotation_xyzw"]["FIXED"] == list(pose["export"]["object"])
                and fr["extra_bone_locals_xyzw"]["FIXED"] == bones
                and (fr["modes"].get("FIXED") == _clean(arm) if arm is not None else "FIXED" not in fr["modes"]))
        eq += int(same)
    return {"frames_with_fixed_slot": n, "byte_equal_frames": eq}


def _clean(v):
    return json.loads(json.dumps(v))


# ------------------------------------------------------------------------------------------------- utilities ---

def changed_sets(before: dict, after: dict, tol: float, rows=None) -> dict:
    from w114_source_preserving_base import quat_delta
    out = {}
    for c in CHANNELS:
        d = quat_delta(before[c], after[c])
        bits = ~((before[c] == after[c]) | (np.isnan(before[c]) & np.isnan(after[c]))).all(-1)
        sel = np.arange(len(d)) if rows is None else rows
        a, b = before[c][sel], after[c][sel]
        dot = np.abs((a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)))
        ang = np.degrees(2 * np.arccos(np.clip(dot, 0.0, 1.0)))
        out[c] = {"changed_frames": int((d[sel] > tol).sum()), "bit_changed_frames": int(bits[sel].sum()),
                  "max_delta": float(d[sel].max()) if len(sel) else 0.0,
                  "max_geodesic_deg": float(np.nanmax(ang)) if len(sel) and np.isfinite(ang).any() else None}
    return out
