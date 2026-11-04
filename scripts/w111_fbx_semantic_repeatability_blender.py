#!/usr/bin/env python3
"""W111 Blender-side stages (run inside Blender 5.1).  No original file is ever written.

  export   PRODUCER / GROUND-TRUTH side, one process per declared cell (hash seed x source path).  Records the
           Python hash environment Blender actually sees, then for every declared case opens the scratch source copy
           at the cell's path (read only), records the W110 source census and source samples (W110
           ``source_record``) and exports once through W110's ``export_v1`` (W107 V1 call).
  consume  CONSUMER side.  Reads ONLY the consumer input written by the host ``encode`` stage (metadata-FBX paths /
           SHAs, selected-take names, sample grids).  Per variant and condition: factory-empty scene, unchanged default
           import, then
             U   untouched,
             A   the unmodified W110 adapter ``recover(fbx, stack)`` (audit-logged file access, integrity snapshots),
             A2  a second fresh import + the adapter again (adapter repeatability),
             K   blanket CONSTANT on the selected imported Action (harness negative control),
           a digest of the whole imported scene right after that step, then samples on the declared grids.

    blender -b --factory-startup -t 2 [--python-use-system-env] --python-exit-code 1 -P <this> -- export ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT/"src"), str(ROOT/"scripts")]

import w110_fbx_consumer_adapter as A  # noqa: E402  (imported before any audit hook is armed)
import w110_fbx_metadata_recovery_blender as B10  # noqa: E402  (W110 producer / consumer helpers, unchanged)

sha256_bytes = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731
sha256 = B10.sha256


# =====================================================================================================================
# PRODUCER / GROUND TRUTH
# =====================================================================================================================
def export(o):
    import bpy
    import w111_fbx_semantic_repeatability_contract as C
    decl_bytes = Path(o.declaration).read_bytes()
    if sha256_bytes(decl_bytes) != o.declaration_sha:
        raise RuntimeError("STOP: declaration hash mismatch")
    decl = json.loads(decl_bytes)
    if bpy.app.version_string != decl["environment"]["blender_version"]:
        raise RuntimeError("STOP: Blender version differs from the declaration")
    cell = C.cell_of(o.cell)
    out = Path(o.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    env = {"PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"), "PYTHONNOUSERSITE": os.environ.get("PYTHONNOUSERSITE"),
           "hash_randomization": sys.flags.hash_randomization, "ignore_environment": sys.flags.ignore_environment,
           "probes": {p: hash(p) for p in C.HASH_PROBES}, "sys_path_user_site": [p for p in sys.path if "site-packages"
                                                                                in p and "Blender" not in p]}
    rep = {"schema": "w111:export", "cell": cell, "blender_version": bpy.app.version_string,
           "blender_build_hash": bpy.app.build_hash.decode() if isinstance(bpy.app.build_hash, bytes)
           else str(bpy.app.build_hash), "declaration_sha256": o.declaration_sha, "python_env": env,
           "seed_as_declared": env["PYTHONHASHSEED"] == str(cell["seed"]), "cases": {}}
    v0 = decl["exporter"]["v0_kwargs"]
    for cid, case in decl["cases"].items():
        d = out/cid
        d.mkdir()
        src = Path(case["sources"][cell["path_key"]]["path"])
        want = case["sources"][cell["path_key"]]["sha256"]
        if sha256(src) != want:
            raise RuntimeError(f"STOP: source {src} does not match its declared hash")
        t0 = time.perf_counter()
        r = B10.source_record(d, src, case["action"], case["times"], case["nb_times"])
        t1 = time.perf_counter()
        r["v1_export"] = B10.export_v1(d, src, case["action"], v0)
        t2 = time.perf_counter()
        r["blend_filepath_at_export"] = bpy.data.filepath
        r["seconds"] = {"source_record": t1 - t0, "export": t2 - t1}
        r["source_sha256_end"] = sha256(src)
        r["source_unchanged"] = r["source_sha256_end"] == want
        rep["cases"][cid] = r
    (out/"cell_report.json").write_text(json.dumps(rep, indent=1, default=str))
    return 0


# =====================================================================================================================
# CONSUMER
# =====================================================================================================================
def scene_digests(snap, selected):
    """Digests of the imported scene: selected Action key data (with / without interpolation), every other Action,
    Action/slot inventory, rig (objects, rest, bones, pose rotation modes) and timebase."""
    from w109_fbx_constant_import_blender import kd_digest
    others = sorted((a, kd_digest(kd)) for a, kd in snap["keydata"].items() if a != selected)
    rig = {k: snap[k] for k in ("objects", "rest", "bones", "pose_modes")}
    return {"selected_keydata_all": kd_digest(snap["keydata"][selected]),
            "selected_keydata_excl_interpolation": kd_digest(snap["keydata"][selected], ignore=("interpolation",)),
            "other_actions": sha256_bytes(json.dumps(others).encode()), "other_action_count": len(others),
            "actions": sha256_bytes(json.dumps(snap["actions"]).encode()),
            "rig": sha256_bytes(json.dumps(rig, sort_keys=True).encode()), "timing": snap["timing"],
            "libraries": snap["libraries"]}


def consume_one(case_dir, cond, fbx, stack, times, nb_times, allowed_roots):
    import bpy
    from w106_action_blender import assign, sample
    from w108_fbx_subframe_blender import interp_set, keydata, set_interpolation
    from w109_fbx_constant_import_blender import kd_digest
    rec = {"condition": cond, "fbx_sha256": sha256(fbx)}
    t0 = time.perf_counter()
    rec["import_result"] = B10.import_scene(fbx)
    rec["import_seconds"] = time.perf_counter() - t0
    arm, chosen, how = B10.select_action(stack)
    rec.update({"selected_action": chosen, "selection": how, "imported_actions": len(bpy.data.actions),
                "armatures": [o.name for o in bpy.data.objects if o.type == "ARMATURE"], "bones": len(arm.data.bones)})
    if chosen is None or how != "name_match":
        rec["status"] = "NO_UNIQUE_ACTION"
        return rec
    kd0 = keydata(bpy.data.actions[chosen])
    rec["import_keydata_digest_all"] = kd_digest(kd0)
    rec["import_keydata_digest_excl_interpolation"] = kd_digest(kd0, ignore=("interpolation",))
    rec["import_interpolation"] = interp_set(kd0)
    if cond in ("A", "A2"):
        pre = B10.snapshot()
        B10._AUDIT["events"].clear()
        t0 = time.perf_counter()
        B10._AUDIT["on"] = True
        try:
            rep = A.recover(str(fbx), stack)
        finally:
            B10._AUDIT["on"] = False
        rec["adapter_seconds"] = time.perf_counter() - t0
        post = B10.snapshot()
        rec["adapter"] = rep
        rec["access"] = B10.classify_access(list(B10._AUDIT["events"]), fbx, allowed_roots)
        rec["integrity"], changed = B10.integrity(pre, post, chosen, rep["applied"])
        rec["changed_curves"] = changed
        rec["selected_equals_adapter_action"] = rep["plan"]["accounting"].get("target_actions") == [chosen]
        rec["scene"] = scene_digests(post, chosen)
    else:
        if cond == "K":
            set_interpolation(bpy.data.actions[chosen], "CONSTANT")
        rec["scene"] = scene_digests(B10.snapshot(), chosen)
    act = bpy.data.actions[chosen]
    rec["assigned"] = assign(arm, act)
    kd = keydata(act)
    rec["final_interpolation"] = interp_set(kd)
    rec["final_interp_tally"] = B10.interp_tally(act)
    rec["final_keydata_digest_excl_interpolation"] = kd_digest(kd, ignore=("interpolation",))
    rec["final_keydata_digest_all"] = kd_digest(kd)
    rec["timing"] = B10._timing()
    rec["sample"] = sample(arm, times, case_dir/f"meta_{cond}.npz")
    rec["nb_sample"] = sample(arm, nb_times, case_dir/f"meta_{cond}_nb.npz")
    rec["status"] = "SAMPLED"
    (case_dir/f"meta_{cond}_meta.json").write_text(json.dumps(rec, indent=1, default=str))
    return rec


def consume(o):
    import bpy
    import w111_fbx_semantic_repeatability_contract as C
    data = Path(o.consumer_input).read_bytes()
    if sha256_bytes(data) != o.consumer_input_sha:
        raise RuntimeError("STOP: consumer input hash mismatch")
    ci = json.loads(data)
    if bpy.app.version_string != ci["blender_version"]:
        raise RuntimeError("STOP: Blender version differs")
    out = Path(o.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    allowed_roots = [str(Path(bpy.utils.resource_path("LOCAL")).parent).lower().replace("/", "\\"),
                     str(Path(sys.prefix)).lower().replace("/", "\\"), str(ROOT/"scripts").lower().replace("/", "\\")]
    sys.addaudithook(B10._audit)
    rep = {"schema": "w111:consume", "blender_version": bpy.app.version_string, "consumer_input_sha256":
           o.consumer_input_sha, "adapter_sha256": sha256(Path(A.__file__)), "allowed_module_roots": allowed_roots,
           "python_env": {"PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
                          "hash_randomization": sys.flags.hash_randomization},
           "variants": {}}
    for vid, v in ci["variants"].items():
        d = out/v["cell"]/v["case"]
        d.mkdir(parents=True)
        f = v["fbx"]
        if sha256(f["path"]) != f["sha256"]:
            raise RuntimeError(f"STOP: {f['path']} does not match the consumer input")
        rc = {}
        for cond in C.CONDITIONS:
            try:
                rc[cond] = consume_one(d, cond, Path(f["path"]), v["stack"], v["times"], v["nb_times"], allowed_roots)
            except Exception as exc:  # recorded, never retried with other settings
                import traceback
                rc[cond] = {"status": "FAILED", "error": repr(exc), "traceback": traceback.format_exc()}
        rep["variants"][vid] = rc
    (out/"consume_report.json").write_text(json.dumps(rep, indent=1, default=str))
    return 0


def main():
    argv = sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("export", "consume"))
    for name in ("declaration", "declaration_sha", "cell", "consumer_input", "consumer_input_sha", "out_dir"):
        ap.add_argument("--"+name.replace("_", "-"))
    o = ap.parse_args(argv)
    try:
        return {"export": export, "consume": consume}[o.mode](o)
    except Exception as exc:  # Blender may otherwise exit 0 after a Python exception
        import traceback
        traceback.print_exc()
        print(f"W111 Blender stage failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())
