#!/usr/bin/env python3
"""W114 POST-HOC diagnostics, written AFTER the frozen W114 build outcomes were inspected. Nothing here changes a frozen
gate, tolerance, perturbation or expected outcome, and nothing here assigns a CASE by itself.

  repeat-pathnorm  The frozen `compare` stage flagged two report fields of the existing-path CONTROL that it had not
                   listed as path-induced: the legacy MotionGraph's `source_metadata.pose_source` (the absolute path of
                   the build's own reconstructed PoseSequence) and that MotionGraph file's sha256. This stage checks
                   whether the two MotionGraph files are byte-identical after replacing each build directory with one
                   placeholder, and whether every other repeat condition of the frozen report holds. The frozen report
                   is kept unchanged next to this one.
  c1-float32       Condition C1 (right_shoulder moved 0.01 m along unit(thorax - pelvis)) was predicted to change no
                   channel but changed spine_01 / upperarm_l / lowerarm_l by <= ~1.4e-8 per quaternion component (declared
                   tolerance 1e-9). This stage measures how far the float32-stored perturbation departs from the trunk
                   axis and re-evaluates the frozen rig on float64 values with the exact float64 displacement. It is a
                   mechanism check only; the frozen C1 outcome stays as recorded.
  review           reporting only: representative real-clip cases with identities and before/after values.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import w114_rig_influence as R  # noqa: E402
import w114_source_preserving_base as B  # noqa: E402
import w114_source_preserving_contract as C  # noqa: E402

LABEL = "POSTHOC_DIAGNOSTIC_NOT_A_VERDICT (written after the frozen W114 outcomes were inspected)"


def cmd_repeat_pathnorm(a):
    out = C.new_dir(a.out_dir)
    frozen = json.loads(Path(a.frozen_repeat).read_text())
    first, second = Path(a.first), Path(a.second)

    def norm(build):
        text = (build / "existing_path" / "legacy_motion.json").read_text()
        for form in (str(build / "existing_path"), str(build / "existing_path").replace("\\", "/"),
                     str(build / "existing_path").replace("/", "\\"), json.dumps(str(build / "existing_path"))[1:-1]):
            text = text.replace(form, "<BUILD>/existing_path")
        return text
    a_txt, b_txt = norm(first), norm(second)
    x = json.loads((first / "build_report.json").read_text())
    y = json.loads((second / "build_report.json").read_text())
    pose_equal = C.sha(first / "existing_path" / "legacy_pose_reconstructed.json") == C.sha(second / "existing_path" / "legacy_pose_reconstructed.json")
    explained = {
        ".existing_path.legacy.motion_graph.source_metadata.pose_source":
            Path(x["existing_path"]["legacy"]["motion_graph"]["source_metadata"]["pose_source"]).name
            == Path(y["existing_path"]["legacy"]["motion_graph"]["source_metadata"]["pose_source"]).name,
        ".existing_path.legacy.motion_graph_sha256": a_txt == b_txt,
    }
    rep = {"label": LABEL, "frozen_repeat_report": str(a.frozen_repeat), "frozen_repeatable": frozen["repeatable"],
           "frozen_unexplained": frozen["unexplained"], "path_normalized_motion_graph_identical": a_txt == b_txt,
           "reconstructed_pose_sequence_byte_identical": pose_equal, "explained_by_path": explained,
           "other_frozen_conditions": {k: frozen[k] for k in ("real_digest_equal", "benchmark_digest_equal",
                                                              "real_programme_states_equal", "rig_case_file_equal", "rig_outcomes_equal")}}
    rep["repeatable_path_normalized"] = bool(set(frozen["unexplained"]) <= {k for k, v in explained.items() if v}
                                             and all(rep["other_frozen_conditions"].values()))
    # keep the keys `decide` reads
    rep["repeatable"] = rep["repeatable_path_normalized"]
    C.dump(out / "repeat_pathnorm_report.json", rep)
    print(json.dumps({"repeatable_path_normalized": rep["repeatable_path_normalized"], "explained": explained}))
    return 0


def cmd_c1_float32(a):
    decl = C.load_declaration(a)
    out = C.new_dir(a.out_dir)
    z, wrep, insp = C.real_inputs(decl)
    ev, _ = C.real_evidence(decl, z, wrep, insp)
    rc = R.rig_context()
    fi = ev["arrays"]["frame_index"]
    rows = np.flatnonzero(fi % 2 == 0)
    base32 = np.array(ev["arrays"]["est3d_xyz_hypothesis"])
    u = base32[:, B.JI["thorax"]].astype(np.float64) - base32[:, B.JI["pelvis"]].astype(np.float64)
    u /= np.linalg.norm(u, axis=-1, keepdims=True)
    rs = B.JI["right_shoulder"]
    # (1) the candidate's float32 storage of the declared displacement
    p32 = base32.copy()
    p32[rows, rs] = p32[rows, rs] + (0.01 * u[rows]).astype(np.float32)
    stored = p32[rows, rs].astype(np.float64) - base32[rows, rs].astype(np.float64)
    par = (stored * u[rows]).sum(-1, keepdims=True) * u[rows]
    perp = np.linalg.norm(stored - par, axis=-1)
    ch_base32 = R.real_channels(base32, rc)
    ch_p32 = R.real_channels(p32, rc)
    # (2) same frozen rig on float64 values with the exact float64 displacement
    base64 = base32.astype(np.float64)
    p64 = base64.copy()
    p64[rows, rs] = p64[rows, rs] + 0.01 * u[rows]
    ch_base64 = R.real_channels(base64, rc)
    ch_p64 = R.real_channels(p64, rc)
    s32 = R.changed_sets(ch_base32, ch_p32, C.RIG_TOL, rows)
    s64 = R.changed_sets(ch_base64, ch_p64, C.RIG_TOL, rows)
    rep = {"label": LABEL, "perturbed_rows": int(len(rows)),
           "float32_stored_displacement": {"perpendicular_to_trunk_axis_max_m": float(perp.max()),
                                           "perpendicular_to_trunk_axis_median_m": float(np.median(perp)),
                                           "parallel_length_minus_0p01_max_abs_m": float(np.abs(np.linalg.norm(par, axis=-1) - 0.01).max())},
           "frozen_float32_state": {c: {"changed": v["changed_frames"], "max_delta": v["max_delta"]} for c, v in s32.items()},
           "float64_exact_displacement": {c: {"changed": v["changed_frames"], "bit_changed": v["bit_changed_frames"],
                                              "max_delta": v["max_delta"]} for c, v in s64.items()},
           "note": "float64 values change the rig's input dtype; this is a mechanism probe outside the candidate (whose state "
                   "is float32 by construction), not a replacement C1 result"}
    C.dump(out / "c1_float32_diagnostic.json", rep)
    print(json.dumps({k: rep[k] for k in ("float32_stored_displacement",)}),
          json.dumps({c: v["max_delta"] for c, v in rep["float64_exact_displacement"].items()}))
    return 0


def cmd_review(a):
    """Reporting only: representative cases with frame/joint/channel identities and before/after values."""
    decl = C.load_declaration(a)
    out = C.new_dir(a.out_dir)
    z, wrep, insp = C.real_inputs(decl)
    ev, _ = C.real_evidence(decl, z, wrep, insp)
    rc = R.rig_context()
    sets, ok_cells, _ = C.real_directives(ev)
    arr = ev["arrays"]
    row = {int(f): i for i, f in enumerate(arr["frame_index"])}
    lin = B.Lineage(ev)
    s1 = B.apply_edit(lin, ev, sets["RS1"], B.make_mock_edit(lin, ev, sets["RS1"], "E1", ok_cells, C.DELTA))
    h = lin.states[lin.order[0]]

    def cell(f, n, state=None):
        r, j = row[f], B.JI[n]
        d = {"frame_index": f, "time_s": float(arr["timestamp"][r]), "joint": n, "obs2d_kind": ev["meta"]["obs2d_kind"][n],
             "xy_px": arr["obs2d_xy_px"][r, j].tolist(), "raw_score": float(arr["obs2d_score_raw"][r, j]),
             "input_valid": bool(arr["input_valid"][r, j]), "est3d_support": B.EST3D_SUPPORT[int(arr["est3d_support"][r, j])],
             "hypothesis_xyz_m": h["values"][r, j].tolist()}
        if state is not None:
            d.update(after_E1_xyz_m=state["values"][r, j].tolist(), origin_after_E1=int(state["origin"][r, j]))
        return d

    base = R.real_channels(np.array(arr["est3d_xyz_hypothesis"]), rc)

    def rig_case(joints, f, delta=C.DELTA):
        v = np.array(arr["est3d_xyz_hypothesis"])
        for n in joints:
            v[row[f], B.JI[n]] += np.asarray(delta, np.float32)
        after = R.real_channels(v, rc)
        return {c: {"before_xyzw": base[c][row[f]].tolist(), "after_xyzw": after[c][row[f]].tolist(),
                    "geodesic_deg": float(np.degrees(2 * np.arccos(min(1.0, abs(float(np.dot(base[c][row[f]], after[c][row[f]]))))))),
                    "changed": bool(B.quat_delta(base[c][row[f]][None], after[c][row[f]][None])[0] > C.RIG_TOL)}
                for c in R.CHANNELS}
    lw = [row[f] for f in range(109, 119)]
    try:
        B.resolve_constraints(ev, sets["RS2"])
        rs2 = "ACCEPT"
    except B.ContractError as e:
        rs2 = str(e)
    e5 = B.make_mock_edit(lin, ev, sets["RS1"], "E5", [(109, "left_wrist")], op="copy_from_frame", source_frame=108)
    try:
        B.apply_edit(lin, ev, sets["RS1"], e5)
        e5c = "ACCEPT"
    except B.ContractError as e:
        e5c = str(e)
    rep = {"label": "REPORTING ONLY (representative cases; MOCK edits, not reconstruction corrections)",
           "reliable_observed_joint_unchanged": cell(150, "left_knee", s1),
           "input_unsupported_joint_permitted_for_mock_edit": cell(113, "left_wrist", s1),
           "low_score_observation_not_assumed_wrong": {
               "cells": [cell(f, "left_wrist") for f in range(109, 119)],
               "left_hip_y_px": [float(arr["obs2d_xy_px"][r, B.JI["left_hip"], 1]) for r in lw],
               "nose_y_px": [float(arr["obs2d_xy_px"][r, B.JI["head"], 1]) for r in lw],
               "candidate_label": "raw score and coordinate kept; status = observed below validity threshold; no correctness label "
                                  "is attached (GT-free)"},
           "score_above_one_cell": cell(int(arr["frame_index"][np.argwhere(np.nan_to_num(arr["obs2d_score_raw"]) > 1)[0][0]]),
                                        B.JOINTS[np.argwhere(np.nan_to_num(arr["obs2d_score_raw"]) > 1)[0][1]]),
           "conflicting_user_directive_RS2": rs2,
           "source_edit_without_rig_effect_A1_right_wrist_f110": rig_case(["right_wrist"], 110),
           "source_edit_with_rig_effect_B1_left_wrist_f110": rig_case(["left_wrist"], 110),
           "timebase_mismatch": {f: {"source_time_s": float(arr["timestamp"][row[f]]), "action_key_frame": f + 1,
                                     "action_time_s_at_30fps": f / C.ACTION_FPS} for f in (0, 100, 198)},
           "unresolved_subject_identity": {"operation": "copy left_wrist 108 -> 109 (cross-frame)", "outcome": e5c,
                                           "subject_status": ev["meta"]["subject"]}}
    C.dump(out / "review.json", rep)
    print("ok")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("repeat-pathnorm")
    for k in ("frozen-repeat", "first", "second", "out-dir"):
        p.add_argument(f"--{k}", required=True)
    p = sub.add_parser("c1-float32")
    for k in ("declaration", "declaration-sha", "out-dir"):
        p.add_argument(f"--{k}", required=True)
    p = sub.add_parser("review")
    for k in ("declaration", "declaration-sha", "out-dir"):
        p.add_argument(f"--{k}", required=True)
    a = ap.parse_args(argv)
    return {"repeat-pathnorm": cmd_repeat_pathnorm, "c1-float32": cmd_c1_float32, "review": cmd_review}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
