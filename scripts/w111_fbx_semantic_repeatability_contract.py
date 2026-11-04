#!/usr/bin/env python3
"""W111 evaluation contract: FBX semantic repeatability across exporter UID and source-path variation (pure Python;
evidence only).

Shared by the host stages, the Blender stages and the tests.  It fixes

  * the cases (the six W110 cases, same source .blend bytes, Actions, selected takes, sample grids),
  * the controlled producer matrix: two explicit Python hash seeds x two absolute scratch source paths, plus an
    independent same-cell repeat (five Blender export processes per run), and a full independent second run,
  * the producer launch (``--python-use-system-env`` is required for Blender to honour PYTHONHASHSEED; the
    environment is sanitised so that nothing else from the system environment reaches Blender's Python),
  * the pairs used to attribute seed and path effects,
  * tolerances (W110 values, unchanged) and the frozen CASE rule.

The canonicaliser lives in ``w111_fbx_semantic_identity.py``; the W110 producer / consumer interventions are reused
unchanged.
"""
from __future__ import annotations

import os

import w110_fbx_metadata_recovery_contract as C10

SCHEMA = "w111_fbx_semantic_repeatability_v1"

CASES = C10.ALL_CASES                          # complete_control, arm_natural_most, real_clip, syn_step, syn_smooth, syn_mixed
REAL_CASES, MIXED, STEP, SMOOTH = C10.REAL_CASES, C10.MIXED, C10.STEP, C10.SMOOTH
HELD_CASES = C10.HELD_CASES
PRIMARY_CASES = (MIXED, "complete_control")    # the Batch's minimum full-matrix cases; every case runs the full matrix
CONTROL_CASES = (MIXED, "complete_control")    # negative / invariance controls (scratch meta.fbx of the reference cell)

SEEDS = {"s1": 1110111, "s2": 2220222}
PATHS = {"pA": "srcpaths/A", "pB": "srcpaths/B_alternate_location"}
SOURCE_NAME = "source.blend"
CELLS = (("s1_pA", "s1", "pA"), ("s2_pB", "s2", "pB"), ("s1_pB", "s1", "pB"), ("s2_pA", "s2", "pA"),
         ("s1_pA_r", "s1", "pA"))        # execution order; s1_pA_r is the independent same-cell repeat process
REF_CELL = "s1_pA"
RUNS = ("run1", "run2")
PAIRS = {"seed_only": (("s1_pA", "s2_pA"), ("s1_pB", "s2_pB")),
         "path_only": (("s1_pA", "s1_pB"), ("s2_pA", "s2_pB")),
         "same_cell": (("s1_pA", "s1_pA_r"),),
         "seed_and_path": (("s1_pA", "s2_pB"), ("s1_pB", "s2_pA"))}
CONDITIONS = ("U", "A", "A2", "K")             # untouched / adapter / adapter on a second fresh import / blanket CONSTANT
PRODUCER_LAUNCH = ["-b", "--factory-startup", "-t", "2", "--python-use-system-env", "--python-exit-code", "1"]
CONSUMER_LAUNCH = ["-b", "--factory-startup", "-t", "2", "--python-exit-code", "1"]      # W110 consumer launch
HASH_PROBES = ("w111-probe", "_key_to_uuid-probe")
TOL = dict(C10.TOL)                            # W110 tolerances, unchanged
STRIP_ENV_PREFIXES = ("PYTHON",)


def producer_env(base_env, seed):
    """Sanitised environment for a producer process: every PYTHON* variable removed, then PYTHONHASHSEED=seed and
    PYTHONNOUSERSITE=1 (no user site-packages).  Nothing else changes."""
    env = {k: v for k, v in base_env.items() if not k.upper().startswith(STRIP_ENV_PREFIXES)}
    env["PYTHONHASHSEED"] = str(int(seed))
    env["PYTHONNOUSERSITE"] = "1"
    return env


def consumer_env(base_env):
    """Consumer: the W110 launch (Blender ignores the system Python environment without --python-use-system-env)."""
    return {k: v for k, v in base_env.items() if not k.upper().startswith(STRIP_ENV_PREFIXES)}


def cell_of(name):
    for c, s, p in CELLS:
        if c == name:
            return {"cell": c, "seed_key": s, "seed": SEEDS[s], "path_key": p}
    raise KeyError(name)


def source_path(base, cid, path_key):
    return os.path.join(str(base), *PATHS[path_key].split("/"), cid, SOURCE_NAME)


# --- CASE mapping (frozen) ------------------------------------------------------------------------------------------
D_CHECKS = (("preflight_ok", "preflight / provenance failed"),
            ("environment_fixed", "an undeclared producer variable changed (Blender, add-on, exporter options, source "
                                  "bytes, Action, seed / path not as declared)"),
            ("all_complete", "a declared variant was not exported, encoded, imported or sampled"),
            ("encoding_valid", "the W110 producer metadata edit was invalid on a variant"),
            ("consumer_input_clean", "the consumer input is not source-free"),
            ("isolation_ok", "consumer isolation not verified (audit, FBX identity, Action resolution)"),
            ("adapter_integrity", "the adapter changed something other than matched-curve interpolation"),
            ("sources_ok", "source identity failed (source samples / census differ from W110 or between variants, or "
                           "a source file changed)"),
            ("canonical_injective", "the canonicaliser rejected a genuine exported variant (identities not uniquely "
                                    "mappable)"),
            ("negative_controls_ok", "a negative / invariance control was masked or not localised"),
            ("repeat_ok", "the independent second run did not reproduce the first run's decision inputs"),
            ("originals_unchanged", "an original / historical artifact or tracked file changed"))


def decide_case(pre):
    """``pre``: the D-check booleans above, and per-case dicts (case -> bool) ``semantic_identical`` (every variant's
    control and metadata FBX canonical records equal the reference's, full content), ``plans_identical`` (UID-normalised
    adapter plans of every variant and of the second adapter run equal), ``scene_identical`` (selected-Action key data
    with / without interpolation, other Actions, rig and timebase digests equal for every condition), ``motion_stable``
    (every condition's samples within deterministic_m of the reference variant), ``motion_material`` (some condition's
    samples differ from the reference by more than recovered_m), ``motion_gate`` (W110 held / smooth / mixed-component
    1 mm gates on every variant), ``coverage`` (complete unique adapter coverage, CONSTANT set == producer set, every
    variant), plus booleans ``controls_discriminate`` (untouched harms held > 1 cm, blanket CONSTANT harms smooth > 1 cm,
    every variant), ``variation_realized`` (seed pairs changed UIDs and path pairs changed the embedded path in every
    case and run) and ``attribution_clean`` (same-cell repeats UID/order/path-identical; path-only pairs UID-identical;
    seed-only pairs path-identical).

    D: any D-check false -- no viability conclusion.
    B: validity holds, but for some case the semantic records, the normalised adapter plan or the imported scene
       differ between variants, or evaluated motion differs by more than 1 mm.
    A: every case semantically identical, plan- and scene-identical, motion-stable (<= 1e-9 m) and within the W110 1 mm
       gates with complete coverage; motion controls discriminate; the intended variation occurred and is cleanly
       attributable.
    C: otherwise (variation not realised, attribution ambiguous, sub-millimetre motion drift, a motion gate / coverage /
       control failure on a stable case)."""
    for k, why in D_CHECKS:
        if not pre[k]:
            return "D", why
    per = ("semantic_identical", "plans_identical", "scene_identical")
    material = sorted(c for c in pre["semantic_identical"] if not all(pre[k][c] for k in per)
                      or pre["motion_material"][c])
    if material:
        return "B", f"material change across producer variants on {material}"
    allc = lambda k: all(pre[k].values())  # noqa: E731
    if (allc("motion_stable") and allc("motion_gate") and allc("coverage") and pre["controls_discriminate"]
            and pre["variation_realized"] and pre["attribution_clean"]):
        return "A", "every seed / path variant is semantically identical under the strict canonicaliser; normalised " \
                    "adapter plans, imported scenes and evaluated motion are identical and within the W110 gates; " \
                    "controls discriminate; variation realised and attributable (Blender 5.1 diagnostic only)"
    why = []
    for k in ("motion_stable", "motion_gate", "coverage"):
        bad = sorted(c for c, v in pre[k].items() if not v)
        if bad:
            why.append(f"{k} false on {bad}")
    for k in ("controls_discriminate", "variation_realized", "attribution_clean"):
        if not pre[k]:
            why.append(f"{k} false")
    return "C", "; ".join(why)
