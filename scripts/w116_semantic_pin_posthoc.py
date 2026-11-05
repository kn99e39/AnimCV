#!/usr/bin/env python3
"""W116 POST-HOC diagnostics (written AFTER the frozen decision was inspected; reporting only, assigns no CASE).

  gb5-faithful     recompute GB5 with the declared gate TEXT ("ACCEPT iff all frames pass"), counting a frame as passing
                   when the evaluator satisfies every pin and the resolver status is SOLVED or NO_CHANGE_NEEDED (the frozen
                   gate code counted NO_CHANGE_NEEDED capture frames as non-passing)
  p5-equivalent    replay real P2 -> P4 on a fresh scratch store and commit the smallest REJECTed RB_pos frame (the side-bend
                   sign that actually had rejections) against the reverted head; check the head is unchanged
  temporal-real    the declared 3x-parent-step discontinuity rule applied to the real interval candidates (RH1, RH2) and to
                   the all-frames-accepted per-frame family candidates RA_neg/RB_neg/RC_raise (wrist and elbow)
  local-vs-world   per frame: are the left-arm stored local values identical to the parent, and is the evaluated left wrist
                   within TOL_M of its parent position, for the unpinned control and the pinned candidate of RA_pos/RA_neg
  closeups         Blender bone-proxy close-ups (mesh hidden) of the declared render frames with visible markers
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import w116_semantic_pin_base as S  # noqa: E402
import w116_semantic_pin_contract as C  # noqa: E402


def frames(ev, name):
    return {fr["frame_index"]: fr for fr in ev["states"][name]["frames"]}


def steps(F, keys, b, p):
    return {k: float(np.linalg.norm(np.array(F[k]["world"][b][p]) - np.array(F[k - 1]["world"][b][p]))) for k in keys[1:]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for n in ("results", "out_dir"):
        ap.add_argument("--" + n.replace("_", "-"), required=True)
    ap.add_argument("--declaration-sha", required=True)
    a = ap.parse_args()
    R = Path(a.results)
    out = C.new_dir(a.out_dir)
    decl = C.load(R / "declare/declaration.json")
    assert S.sha_file(R / "declare/declaration.json") == a.declaration_sha
    cr = C.load(R / "commit/commit_report.json")
    rev, sev = C.load(R / "eval_e1/real_eval.json"), C.load(R / "eval_e1/syn_eval.json")
    B = C.load(R / "build_r1/build_report.json")
    rep = {"schema": "w116_posthoc_v1", "label": "POST-HOC, after the frozen decision; no gate or CASE is changed"}
    # ---- gb5 faithful
    ok = lambda st: st in ("SOLVED", "NO_CHANGE_NEEDED")  # noqa: E731
    g5 = {}
    for n, r in cr["intervals"].items():
        allpass = all(f["frame_pass"] and ok(f["resolver_status"]) for f in r["frames"].values())
        g5[n] = {"decision": r["decision"]["decision"], "all_frames_pass": allpass,
                 "consistent": (r["decision"]["decision"] == "ACCEPT") == allpass,
                 "no_change_needed_frames": [k for k, f in r["frames"].items() if f["resolver_status"] == "NO_CHANGE_NEEDED"]}
    for n in ("K8", "K8b"):
        r = cr["synthetic"][n]
        allpass = all(e["evaluation"]["pins_satisfied"] and ok(r["resolver"][k]["status"]) for k, e in r["evaluation"].items())
        g5[n] = {"decision": r["decision"]["decision"], "all_frames_pass": allpass,
                 "consistent": (r["decision"]["decision"] == "ACCEPT") == allpass,
                 "no_change_needed_frames": [k for k, v in r["resolver"].items() if v["status"] == "NO_CHANGE_NEEDED"]}
    rw = cr["whole_clip"]["RW_pos"]
    g5["RW_pos"] = {"decision": rw["decision"]["decision"], "consistent": (rw["decision"]["decision"] == "ACCEPT")
                    == (rw["frames_passing"] == rw["n_frames"])}
    rep["gb5_faithful"] = {"per_request": g5, "all_consistent": all(v["consistent"] for v in g5.values())}
    # ---- p5 equivalent
    ctx = C.Ctx(decl)
    v0 = ctx.v0
    store = S.Store(out / "store")
    store.put(v0)
    store.init_ref("real/main", v0["state_id"])
    per = cr["per_frame"]
    rc = [int(k) for k in sorted(per["RC_raise"], key=int) if per["RC_raise"][k]["decision"]["decision"] == "ACCEPT"][0]
    rbp = [int(k) for k in sorted(per["RB_pos"], key=int) if per["RB_pos"][k]["decision"]["decision"] == "REJECT"][0]

    def cand(fam, k, parent):
        return C.patched_state(parent, {k: B["requests"][fam]["candidate_values"][str(k)]}, {"kind": "PIN_EDIT", "request": f"{fam}/{k}"})

    log = [dict(step="P2", frame=rc, **store.commit("real/main", v0["state_id"], cand("RC_raise", rc, v0),
                                                   per["RC_raise"][str(rc)]["decision"]))]
    st, rr = store.revert("real/main")
    log.append(dict(step="P4", **rr))
    hp = store.refs["real/main"]
    same = S.canon(store.get(hp)["values"]) == S.canon(v0["values"])
    rec = store.commit("real/main", hp, cand("RB_pos", rbp, store.get(hp)), per["RB_pos"][str(rbp)]["decision"])
    log.append(dict(step="P5_equivalent_RB_pos", frame=rbp, head_unchanged=rec["head_after"] == hp,
                    parent_values_equal_v0=same, **rec))
    rep["p5_equivalent"] = {"log": log, "lineage_audit": store.lineage(store.refs["real/main"])}
    # ---- temporal real
    P = frames(rev, "v0")
    tr = {}
    for name, (cap, fr) in C.REAL_INTERVALS.items():
        Cn = frames(rev, f"{name}__candidate")
        keys = sorted(P)
        res = {}
        for b, p in (("lowerarm_l", "tail"), ("upperarm_l", "tail")):
            dp, dc = steps(P, keys, b, p), steps(Cn, keys, b, p)
            lim = 3 * max(dp.values())
            res[f"{b}.{p}"] = {"limit_m": lim, "parent_max_step_m": max(dp.values()),
                               "discontinuous": {f"{k - 1}->{k}": dc[k] for k in dc if dc[k] > lim},
                               "boundary_steps_m": {f"{fr[0] - 1}->{fr[0]}": dc[fr[0]], f"{fr[-1]}->{fr[-1] + 1}": dc[fr[-1] + 1]},
                               "parent_boundary_steps_m": {f"{fr[0] - 1}->{fr[0]}": dp[fr[0]], f"{fr[-1]}->{fr[-1] + 1}": dp[fr[-1] + 1]}}
        tr[name] = res
    for fam in ("RA_neg", "RB_neg", "RC_raise"):
        Cn, Kn = frames(rev, f"{fam}__candidate"), frames(rev, f"{fam}__control")
        keys = sorted(P)
        res = {}
        for b, p in (("lowerarm_l", "tail"), ("upperarm_l", "tail")):
            dp, dc, dk = steps(P, keys, b, p), steps(Cn, keys, b, p), steps(Kn, keys, b, p)
            lim = 3 * max(dp.values())
            res[f"{b}.{p}"] = {"limit_m": lim, "candidate_discontinuous": sorted(f"{k - 1}->{k}" for k in dc if dc[k] > lim),
                               "control_discontinuous": sorted(f"{k - 1}->{k}" for k in dk if dk[k] > lim),
                               "candidate_max_step_m": max(dc.values()), "control_max_step_m": max(dk.values()),
                               "parent_max_step_m": max(dp.values())}
        tr[fam + "_combined_candidate"] = res
    rep["temporal_real"] = tr
    # ---- local vs world
    lw = {}
    for fam in ("RA_pos", "RA_neg"):
        Kn, Cn = frames(rev, f"{fam}__control"), frames(rev, f"{fam}__candidate")
        cnt = {"control_local_preserved": 0, "control_world_preserved": 0, "candidate_local_preserved": 0,
               "candidate_world_preserved": 0, "n": len(P)}
        ex = None
        for k in P:
            for tag, F in (("control", Kn), ("candidate", Cn)):
                loc = all(F[k]["stored_wxyz"][c] == P[k]["stored_wxyz"][c] for c in C.ARM)
                d = float(np.linalg.norm(np.array(F[k]["world"]["lowerarm_l"]["tail"]) - np.array(P[k]["world"]["lowerarm_l"]["tail"])))
                cnt[f"{tag}_local_preserved"] += loc
                cnt[f"{tag}_world_preserved"] += d <= S.TOL_M
            if ex is None and k == 113:
                ex = {"frame": k, "parent_wrist": P[k]["world"]["lowerarm_l"]["tail"],
                      "control_wrist": Kn[k]["world"]["lowerarm_l"]["tail"], "candidate_wrist": Cn[k]["world"]["lowerarm_l"]["tail"],
                      "parent_arm_stored": {c: P[k]["stored_wxyz"][c] for c in C.ARM},
                      "control_arm_stored": {c: Kn[k]["stored_wxyz"][c] for c in C.ARM},
                      "candidate_arm_stored": {c: Cn[k]["stored_wxyz"][c] for c in C.ARM},
                      "spine_parent": P[k]["stored_wxyz"]["spine_01"], "spine_edited": Kn[k]["stored_wxyz"]["spine_01"]}
        lw[fam] = {"counts": cnt, "example_113": ex}
    rep["local_vs_world"] = lw
    C.dump(out / "posthoc.json", rep)
    # ---- close-ups (bone proxies, mesh hidden)
    jobs = []
    col = {"target": (0.1, 0.9, 0.2, 1), "unpinned": (0.95, 0.15, 0.1, 1), "pinned": (0.1, 0.4, 1.0, 1)}
    picks = [("RA_pos", 113), ("RA_pos", 0), ("RB_neg", 113), ("RB_pos", 160), ("NR1_right_wrist_left_dof", 113)]
    rb_rej = rbp
    picks.append(("RB_pos", rb_rej))
    for fam, k in picks:
        e = per[fam][str(k)]["evaluation"]["pins"][0]
        mk = [{"name": "target", "position": e["target"], "color": col["target"], "radius": 0.010},
              {"name": "unpinned", "position": e["unpinned_control"], "color": col["unpinned"], "radius": 0.007},
              {"name": "pinned", "position": e["pinned_candidate"], "color": col["pinned"], "radius": 0.005}]
        for st_name, tag in (("v0", "a_parent"), (f"{fam}__control", "b_unpinned"), (f"{fam}__candidate", "c_pinned")):
            for view, off in (("front", [0.0, -3.0, 0.0]), ("side", [3.0, 0.0, 0.0])):
                jobs.append({"id": f"{fam}_f{k:03d}__{view}__{tag}", "state": st_name, "frame_index": k, "markers": mk,
                             "centre": e["target"], "view_offset": off})
    for k in (110, 115, 118):
        e = cr["intervals"]["RH1_hold_109_118"]["frames"][str(k)]["evaluation"]["pins"][0]
        mk = [{"name": "target", "position": e["target"], "color": col["target"], "radius": 0.010},
              {"name": "unpinned", "position": e["unpinned_control"], "color": col["unpinned"], "radius": 0.007},
              {"name": "pinned", "position": e["pinned_candidate"], "color": col["pinned"], "radius": 0.005}]
        for st_name, tag in (("v0", "a_parent"), ("v0", "b_unpinned"), ("RH1_hold_109_118__candidate", "c_pinned")):
            for view, off in (("front", [0.0, -3.0, 0.0]), ("side", [3.0, 0.0, 0.0])):
                jobs.append({"id": f"RH1_f{k:03d}__{view}__{tag}", "state": st_name, "frame_index": k, "markers": mk,
                             "centre": e["target"], "view_offset": off})
    C.dump(out / "closeup_jobs.json", {"jobs": jobs})
    r = subprocess.run([str(C.BLENDER), "-b", "-t", "2", "--python-exit-code", "1", "-P", "scripts/w116_blender_stages.py", "--",
                        "render", "--rig", str(C.INPUTS["w105_rig_skinned_fbx"][0]), "--rest", str(C.INPUTS["w105_rig_rest"][0]),
                        "--states", str(R / "build_r1/real_states.json"), "--jobs", str(out / "closeup_jobs.json"),
                        "--out-dir", str(out / "closeups"), "--ortho-scale", "0.9", "--hide-mesh"], cwd=ROOT, capture_output=True, text=True)
    (out / "closeups.log").write_text(r.stdout + "\n" + r.stderr, encoding="utf-8")
    if r.returncode != 0:
        raise RuntimeError("close-up render failed")
    (out / "real").mkdir()
    for p in (out / "closeups").glob("*.png"):
        p.replace(out / "real" / p.name)
    compose_real(out)
    print(json.dumps({k: v for k, v in rep.items() if k in ("gb5_faithful",)}, indent=1)[:1500])
    print(json.dumps(rep["p5_equivalent"]["log"][-1], indent=1, default=str)[:800])
    print(json.dumps({n: {p: {k: v for k, v in x.items() if k != "limit_m"} for p, x in t.items()} for n, t in tr.items()}, indent=1)[:4000])
    print(json.dumps({f: v["counts"] for f, v in lw.items()}, indent=1))
    return 0


def compose_real(out):
    from PIL import Image, ImageDraw
    d = out / "real"
    groups = {}
    for p in sorted(d.glob("*.png")):
        groups.setdefault(p.stem.split("__")[0], []).append(p)
    for key, ps in groups.items():
        ims = {tuple(p.stem.split("__")[1:]): Image.open(p) for p in ps}
        w, h = next(iter(ims.values())).size
        grid = Image.new("RGB", (w * 3, h * 2 + 24), (20, 20, 20))
        for (view, tag), im in ims.items():
            grid.paste(im, ("abc".index(tag[0]) * w, 24 + {"front": 0, "side": 1}[view] * h))
        ImageDraw.Draw(grid).text((6, 4), f"POST-HOC close-up {key} | a parent | b unpinned | c pinned | green target, red unpinned "
                                          f"wrist, blue pinned wrist | rows: view from -Y, view from +X", fill=(255, 255, 255))
        grid.save(out / f"closeup_{key}.png")



def accounting(results, out_dir):
    """Reporting-only denominators over the frozen outputs (POST-HOC tabulation; no gate)."""
    R = Path(results)
    out = C.new_dir(out_dir)
    cr = C.load(R / "commit/commit_report.json")
    B = C.load(R / "build_r1/build_report.json")
    cal = C.load(R / "calibrate/calibration.json")
    acc = {"constraints": {"requested": 0, "valid_spec": 0, "invalid_spec": 0}, "requests": {}, "frames": {}}
    for fam, r in B["requests"].items():
        reqs = [f["request"] | {"_errs": f["validation_errors"]} for f in r["frames"].values()] if r["kind"] == "per_frame" else \
            [r["request"] | {"_errs": r["validation_errors"]}]
        for q in reqs:
            n = len(q["constraints"])
            acc["constraints"]["requested"] += n
            acc["constraints"]["invalid_spec" if q["_errs"] else "valid_spec"] += n
    for n, r in B["synthetic_requests"].items():
        k = len(r["request"]["constraints"])
        acc["constraints"]["requested"] += k
        acc["constraints"]["invalid_spec" if r["validation_errors"] else "valid_spec"] += k
    dec = {}
    for fam, per in cr["per_frame"].items():
        for k, r in per.items():
            key = (fam, r["gt"], r["decision"]["decision"], r["decision"]["failure_class"])
            dec[str(key)] = dec.get(str(key), 0) + 1
    acc["per_frame_requests"] = dec
    acc["bindings"] = {"real_bound": sorted(b["semantic_joint"] for b in cal["bindings"]["real"]["bound"].values()),
                       "real_failed": {k: v["status"] for k, v in cal["bindings"]["real"]["failed"].items()},
                       "syn_bound": sorted(b["semantic_joint"] for b in cal["bindings"]["synthetic"]["bound"].values()),
                       "syn_failed": {k: v["status"] for k, v in cal["bindings"]["synthetic"]["failed"].items()}}
    pos = ("RA_pos", "RA_neg", "RB_pos", "RB_neg", "RC_raise")
    hard = {"feasible": 0, "resolved": 0, "unresolved_feasible": 0, "infeasible": 0, "boundary": 0, "criterion_pass": 0,
            "criterion_fail": 0, "accepted": 0, "rejected": 0, "intended_channel_changes": 0, "unintended_channel_changes": 0}
    for fam in pos:
        for k, r in cr["per_frame"][fam].items():
            g = r["gt"]
            hard["feasible" if g == "FEASIBLE" else ("infeasible" if g == "INFEASIBLE" else "boundary")] += 1
            ev = r["evaluation"]
            p = ev["pins"][0]["candidate_residual_m"] <= S.TOL_M
            hard["criterion_pass" if p else "criterion_fail"] += 1
            a = r["decision"]["decision"] == "ACCEPT"
            hard["accepted" if a else "rejected"] += 1
            if g == "FEASIBLE":
                hard["resolved" if a else "unresolved_feasible"] += 1
            if a:
                hard["intended_channel_changes"] += len(ev["stored_changed_vs_control"])
                hard["unintended_channel_changes"] += len(set(ev["stored_changed_vs_control"]) - set(C.ARM))
    acc["real_positive_families_frames"] = hard
    syn = {}
    for n, r in cr["synthetic"].items():
        syn[n] = {"decision": r["decision"]["decision"], "class": r["decision"]["failure_class"], "reasons": r["decision"]["reasons"],
                  "expected": r["expected"], "gt": sorted(set(r["gt"].values())),
                  "max_candidate_residual_m": max((e["evaluation"]["pins"][0]["candidate_residual_m"] for e in r["evaluation"].values()
                                                   if e["evaluation"]), default=None),
                  "max_control_residual_m": max((e["evaluation"]["pins"][0]["control_residual_m"] for e in r["evaluation"].values()
                                                 if e["evaluation"]), default=None)}
    acc["synthetic"] = syn
    acc["identity"] = {"real_frames_video_subject_identity_UNKNOWN": 199, "real_output_rig_identity_known": 199,
                       "synthetic_rig_identity_known_frames": 20}
    C.dump(out / "accounting.json", acc)
    print(json.dumps({k: v for k, v in acc.items() if k != "per_frame_requests"}, indent=1, default=str)[:6000])
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "accounting":  # python ... accounting <results> <out_dir>
        sys.exit(accounting(sys.argv[2], sys.argv[3]))
    sys.exit(main())
