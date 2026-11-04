#!/usr/bin/env python3
"""W110 evaluation contract: FBX-embedded interpolation-metadata recovery feasibility (pure Python; evidence only).

Shared by the host stages, the Blender stages and the tests.  Nothing here touches Blender, a rig file or a
production module.  It defines

  * the fixed cases: the three W106-W109 real M_Main Actions (frozen W107 V1 FBX), the W108 held-step and smooth-LINEAR
    synthetic controls (W108 V1 FBX) and ONE new predeclared mixed held/smooth synthetic Action on the W106 synthetic
    rig (exported once through the same W107 V1 path),
  * the PRODUCER rule (source-informed, allowed only here): per selected-take FBX AnimationCurve, the uniform
    interpolation of the source F-curves that drive the same (item, transform property) decides the flag --
    CONSTANT -> W109's verified CONSTANT-standard 0x6002, LINEAR -> keep 0x6104, undriven -> keep 0x6104,
    anything else -> refuse,
  * the canonical tree digest with timestamp nodes excluded (fresh-export repeat comparison),
  * the evaluated-motion metrics (W108 world error by phase / interval class, a predeclared near-boundary probe
    t = f + 0.99 in transition intervals, mixed-case held / smooth component point sets, channel-function errors),
  * the frozen tolerances and CASE mapping.

The CONSUMER adapter lives in ``w110_fbx_consumer_adapter.py`` and never imports this module.
"""
from __future__ import annotations

import hashlib
import math
import struct

import numpy as np

import w108_fbx_subframe_contract as C8          # W108 reader, sample grid, metrics (read-only reuse)
import w109_fbx_constant_import_contract as C9    # W109 flag derivation, byte patcher, canonical serialisation

SCHEMA = "w110_fbx_metadata_recovery_v1"

REAL_CASES = C8.REAL_CASES                        # complete_control, arm_natural_most, real_clip -- fixed
STEP, SMOOTH, MIXED = "syn_step", "syn_smooth", "syn_mixed"
SYN_CASES = (STEP, SMOOTH, MIXED)
ALL_CASES = REAL_CASES + SYN_CASES
HELD_CASES = REAL_CASES + (STEP,)                 # every driven source group CONSTANT

SOURCE_FLAG = C9.SOURCE_FLAG                       # 0x6104 (exporter literal, LINEAR)
CONSTANT_FLAG = C9.EXPECTED_FLAGS["CONSTANT_standard"]   # 0x6002 -- W109-verified; CONSTANT-next never used
assert CONSTANT_FLAG == 0x6002

# --- the one predeclared mixed held/smooth synthetic Action ---------------------------------------------------------
MIXED_SPEC = {
    "case": MIXED, "action": "W110 SYN mixed", "stack": "root|W110 SYN mixed", "base_action": "SYN baseline",
    "held_bones": ["base", "upperarm_l", "lowerarm_l", "hand_l"],
    "smooth_bones": ["sibling"],
    "construction": "scratch copy of the W108 input copy of the W106 synthetic N0 file; 'SYN baseline' is duplicated "
                    "(key times, values and slot identical); keyframe interpolation of the duplicate's F-curves is set "
                    "by foreach_set to CONSTANT for the held bones and LINEAR for the smooth bone (no other field); "
                    "the duplicate is assigned with the original slot identifier and the file saved as a scratch copy; "
                    "then exported once with W107's V1 call (V0 kwargs + bake_anim_simplify_factor=0.0). Rig, scene "
                    "fps and frame range untouched.",
    "key_frames": [1.0 + 3*k for k in range(9)],
    "missing_key": {"sibling": 13.0},
    "expected": "held bones: value of the latest key at or before t (Blender CONSTANT); sibling: per-component "
                "linear quaternion between its keys (Blender LINEAR), skipping its missing key at 13",
    "component_points": {
        "held": "world heads and tails of base, upperarm_l, lowerarm_l, hand_l and the world head of sibling (all "
                "determined only by held channels: base is the root, sibling's head is fixed in base's frame)",
        "smooth": "world tail of sibling (determined by held base and smooth sibling rotation; base is exact under "
                  "both the recovered and the blanket-CONSTANT condition, so error there is attributable to sibling)"},
    "expected_fbx": {"selected_stack_curves": 54, "constant_curves": 12, "linear_driven_curves": 3,
                     "undriven_curves": 39},
}


# --- producer rule (source-informed) ------------------------------------------------------------------------------
SOURCE_GROUPS = {"location": "location", "rotation_quaternion": "rotation", "rotation_euler": "rotation",
                 "rotation_axis_angle": "rotation", "scale": "scale"}


def source_group_of(data_path):
    """(item kind, item name, group) of a source F-curve data path, or None if not a transform channel."""
    if data_path.startswith('pose.bones["'):
        rest = data_path[len('pose.bones["'):]
        i = rest.index('"]')
        bone, prop = rest[:i].replace('\\"', '"').replace("\\\\", "\\"), rest[i+3:]
        g = SOURCE_GROUPS.get(prop)
        return ("bone", bone, g) if g else None
    g = SOURCE_GROUPS.get(data_path)
    return ("object", None, g) if g else None


def producer_directives(fmap, census):
    """Per selected-take FBX curve: (directive, source interpolation set).

    ``fmap``: adapter FBX-side map of the *unmodified* V1 file.  ``census``: Blender source census
    {"armature_object": name, "bones": [...], "groups": {"object|<g>" | "bone:<name>|<g>": [interpolations]}}.
    CONSTANT if the driving source group is uniformly CONSTANT; LINEAR if uniformly LINEAR; UNDRIVEN if the source
    has no F-curve for that (item, group); UNSUPPORTED otherwise (mixed / BEZIER / unknown) -- the encode refuses."""
    out = {}
    for r in fmap["curves"]:
        if r["status"] != "OK":
            out[r["uid"]] = ("UNMATCHED", [])
            continue
        if r["model_type"] == "Null":
            key = f"object|{r['group']}" if r["model_name"] == census["armature_object"] else None
        else:
            key = f"bone:{r['model_name']}|{r['group']}" if r["model_name"] in census["bones"] else None
        if key is None:
            out[r["uid"]] = ("UNMATCHED", [])
            continue
        interp = sorted(set(census["groups"].get(key, [])))
        if not interp:
            out[r["uid"]] = ("UNDRIVEN", interp)
        elif interp == ["CONSTANT"]:
            out[r["uid"]] = ("CONSTANT", interp)
        elif interp == ["LINEAR"]:
            out[r["uid"]] = ("LINEAR", interp)
        else:
            out[r["uid"]] = ("UNSUPPORTED", interp)
    return out


def producer_counts(directives):
    c = {}
    for d, _ in directives.values():
        c[d] = c.get(d, 0) + 1
    return c


# --- canonical tree digest with nodes excluded --------------------------------------------------------------------
TIMESTAMP_PATHS = ((b"FBXHeaderExtension", b"CreationTimeStamp"),)


def tree_digest(root, masked_uids=(), skip_paths=()):
    """W109 canonical serialisation (ids, property types/values, child order) with the KeyAttrFlags values of
    ``masked_uids`` masked and every node whose id path starts with one of ``skip_paths`` replaced by a token."""
    masked = {int(u) for u in masked_uids}
    h = hashlib.sha256()

    def walk(e, mask_flags, path):
        if any(path[:len(p)] == p for p in skip_paths):
            h.update(b"<SKIPPED>" + bytes(e.id))
            return
        types = bytes(e.props_type).decode()
        h.update(struct.pack("<I", len(e.id)) + bytes(e.id) + struct.pack("<I", len(types)) + types.encode())
        for t, v in zip(types, e.props):
            b = b"<MASKED>" if (mask_flags and e.id == b"KeyAttrFlags") else C9._canon(t, v)
            h.update(struct.pack("<Q", len(b)) + b)
        h.update(struct.pack("<I", len(e.elems)))
        child_mask = e.id == b"AnimationCurve" and len(e.props) > 0 and int(e.props[0]) in masked
        for c in e.elems:
            walk(c, child_mask, path + (bytes(c.id),))
    for top in root.elems:
        walk(top, False, (bytes(top.id),))
    return h.hexdigest()


def curve_arrays(root, fmap):
    """{(model name, fbx property, channel): (KeyTime int64 array, KeyValueFloat array)} for the mapped curves."""
    objs = C8._child(root, b"Objects")
    by_uid = {int(c.props[0]): c for c in objs.elems if c.id == b"AnimationCurve"}
    out = {}
    for r in fmap["curves"]:
        if r["status"] == "OK":
            c = by_uid[r["uid"]]
            out[(r["model_name"], r["fbx_property"], r["channel"])] = (
                np.asarray(C8._child(c, b"KeyTime").props[0], dtype=np.int64),
                np.asarray(C8._child(c, b"KeyValueFloat").props[0], dtype=np.float32))
    return out


# --- sample grids --------------------------------------------------------------------------------------------------
NEAR_BOUNDARY_PHASE = 0.99


def near_boundary_times(classes):
    """One probe per transition interval [f, f+1): t = f + 0.99 (held: key at f; linear: 99 % toward f+1)."""
    return [float(f) + NEAR_BOUNDARY_PHASE for f in sorted(int(k) for k in classes) if classes[f] == "transition"]


# --- tolerances (fixed a priori; metres unless noted) ---------------------------------------------------------------
TOL = {
    "recovered_m": 1e-3,              # W108 1 mm world-error threshold (held recovery / smooth preservation)
    "integer_gate_m": 1e-3,           # W107 integer-frame gate
    "deterministic_m": 1e-9,          # W109 repeat / control identity (P0 replay, adapter no-op, U_meta == U_ctrl)
    "source_fidelity_m": 0.0,         # fresh source samples vs W108 source.npz (bitwise)
    "w108_metric_reproduce_m": 1e-9,  # control subframe / integer max vs the W108 analysis value
    "harm_m": 1e-2,                   # negative controls must exceed 1 cm on the harmed component (W109 discriminable)
}


# --- metrics -------------------------------------------------------------------------------------------------------
def world_coord_err(src, imp):
    """(T, B) max |coordinate| head/tail error on common bones (W108 gate form) + bone names."""
    f = C8.error_fields(src, imp)
    return f["coord"], f["bones"], f


def overall(src, imp, src_nb, imp_nb, times, classes):
    """W108 subframe report on the main grid + near-boundary maximum + all-sample maximum."""
    rep = C8.subframe_report(src, imp, times, classes)
    nb = world_coord_err(src_nb, imp_nb)[0] if len(src_nb["times"]) else np.zeros((0, 1))
    rep["near_boundary"] = {"samples": int(nb.shape[0]), "max_abs_coord_m": float(nb.max()) if nb.size else None,
                            "samples_gt_1mm": int((nb.max(1) > TOL["recovered_m"]).sum()) if nb.size else 0}
    vals = [rep["integer"]["max_abs_coord_m"], rep["subframe"]["max_abs_coord_m"]]
    if nb.size:
        vals.append(float(nb.max()))
    rep["all_samples_max_m"] = float(max(vals))
    return rep


def _points(d, heads, tails):
    idx = {str(b): i for i, b in enumerate(d["bones"])}
    parts = [d["world_head"][:, [idx[b] for b in heads]]] if heads else []
    if tails:
        parts.append(d["world_tail"][:, [idx[b] for b in tails]])
    return np.concatenate(parts, axis=1)


def component_points(spec):
    """Held points: heads + tails of the held bones and the heads of the smooth bones (each smooth bone's parent is
    held, so its head is held-only); smooth points: tails of the smooth bones."""
    held, smooth = list(spec["held_bones"]), list(spec["smooth_bones"])
    return {"held": {"heads": held + smooth, "tails": held}, "smooth": {"heads": [], "tails": smooth}}


def component_errors(src, imp, spec):
    """Max |coordinate| world error of the mixed case's held and smooth point sets."""
    out = {}
    for k, p in component_points(spec).items():
        a = _points(src, p["heads"], p["tails"])
        b = _points(imp, p["heads"], p["tails"])
        e = np.abs(a - b).max(axis=(1, 2)) if a.size else np.zeros(0)
        out[k] = float(e.max()) if e.size else 0.0
    return out


def channel_function_error(imp, times, bone, mode):
    """Max |component| difference between the bone's local quaternion at every subframe and the declared function of
    the same import's integer samples: 'floor' (held) or 'lerp' (per-component linear).  Isolates interpolation
    semantics from key data (diagnostic, unitless)."""
    t = np.asarray(times, dtype=float)
    j = [str(b) for b in imp["bones"]].index(bone)
    q = imp["local_quat"][:, j]
    idx = {float(x): i for i, x in enumerate(t) if x == math.floor(x)}
    err = 0.0
    for i, x in enumerate(t):
        f = math.floor(x)
        if x == f or float(f + 1) not in idx:
            continue
        a, b = q[idx[float(f)]], q[idx[float(f + 1)]]
        ref = a if mode == "floor" else a + (x - f)*(b - a)
        err = max(err, float(np.abs(q[i] - ref).max()))
    return err


def max_abs_diff(a, b):
    return C8.arrays_max_diff(a, b)


# --- CASE mapping (frozen) ------------------------------------------------------------------------------------------
def decide_case(pre):
    """``pre`` keys (booleans unless noted):
      preflight_ok, encoding_valid, isolation_ok, adapter_integrity, sources_ok, p0 (case -> bool), deterministic,
      originals_unchanged, coverage (case -> bool: 100 % of selected-take FBX curves uniquely mapped, every imported
      selected-Action F-curve accounted, every producer-CONSTANT curve under a CONSTANT directive, 0 unresolved),
      recovered (case -> bool: held cases -- metadata-corrected import within 1 mm of the source at every integer,
      quarter and near-boundary sample and within 1 mm at integer frames), smooth_preserved (bool: syn_smooth corrected
      within 1 mm), mixed_ok (bool: mixed held and smooth components and all samples within 1 mm, integer within 1
      mm), mixed_held_ok (bool: mixed held component within 1 mm), controls_discriminate (bool: untouched import harms every held case and the mixed held component > 1 cm;
      blanket CONSTANT harms syn_smooth and the mixed smooth component > 1 cm), attribution (bool: adapter on the
      unmodified control FBX and the metadata FBX without adapter are both identical to the untouched control import).

    D: preflight / provenance, encoding validity, consumer isolation, adapter integrity (anything but interpolation
       of matched curves changed, non-target Actions / rig / timebase changed), source identity, P0, repeat or
       originals failure -- no viability conclusion.
    A: coverage complete on every case, held recovered on all three real Actions and syn_step, smooth preserved, mixed
       ok, controls discriminate and attribution holds (scratch, tested-Blender-only feasibility).
    B: controls discriminate and attribution holds, but no held case (real, syn_step, mixed held component) is
       recovered within 1 mm, or no case has complete coverage.
    C: otherwise (outcomes differ by case / channel, incomplete coverage on some cases, a control not discriminating)."""
    for k, why in (("preflight_ok", "preflight / provenance failed"),
                   ("encoding_valid", "metadata encoding / unrelated-semantics preservation failed"),
                   ("isolation_ok", "consumer isolation not verifiable"),
                   ("adapter_integrity", "adapter changed something other than matched-curve interpolation"),
                   ("sources_ok", "source / construction identity failed"),
                   ("deterministic", "fresh repeat disagrees"),
                   ("originals_unchanged", "an original artifact changed")):
        if not pre[k]:
            return "D", why
    if not all(pre["p0"].values()):
        return "D", f"P0 baseline failed: {sorted(c for c, v in pre['p0'].items() if not v)}"
    cov, rec = pre["coverage"], pre["recovered"]
    if (all(cov.values()) and all(rec[c] for c in HELD_CASES) and pre["smooth_preserved"] and pre["mixed_ok"]
            and pre["controls_discriminate"] and pre["attribution"]):
        return "A", "complete unique coverage; held real + syn_step recovered, smooth preserved and mixed held/smooth " \
                    "components within 1 mm from FBX metadata alone; controls discriminate; producer and consumer " \
                    "each required"
    none_recovered = not any(rec[c] for c in HELD_CASES) and not pre["mixed_held_ok"]
    if pre["controls_discriminate"] and pre["attribution"] and (none_recovered or not any(cov.values())):
        return "B", "valid controls, but the FBX-only adapter recovers no held case (or no case is fully mapped)"
    why = []
    if not all(cov.values()):
        why.append(f"incomplete coverage: {sorted(c for c, v in cov.items() if not v)}")
    if not all(rec[c] for c in HELD_CASES):
        why.append(f"held not recovered: {sorted(c for c in HELD_CASES if not rec[c])}")
    if not pre["smooth_preserved"]:
        why.append("smooth control not preserved")
    if not pre["mixed_ok"]:
        why.append("mixed case not within 1 mm")
    if not pre["controls_discriminate"]:
        why.append("a negative control does not discriminate")
    if not pre["attribution"]:
        why.append("producer/consumer attribution failed")
    return "C", "; ".join(why)


sha256_file = C8.sha256_file
load_npz = C8.load_npz
