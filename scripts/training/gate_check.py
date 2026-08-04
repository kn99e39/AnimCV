#!/usr/bin/env python3
"""Pool per-clip audit-supervised-3d reports and compare against docs/06 runbook gates.

Reads runs/<run_name>/holdout/*/audit.json (committed, small) and writes
runs/<run_name>/holdout_gate_report.json. Defaults to the mpi_s1s6_cam0 run.
"""
import glob
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RUN_NAME = sys.argv[1] if len(sys.argv) > 1 else "mpi_s1s6_cam0"
RUN_DIR = os.path.join(REPO_ROOT, "runs", RUN_NAME)

GATES = {
    "pa_mpjpe_mm": ("<=", 80.0),
    "root_yaw_mae_degrees": ("<=", 15.0),
    "root_yaw_p95_degrees": ("<=", 30.0),
}

reports = []
for path in sorted(glob.glob(os.path.join(RUN_DIR, "holdout", "*", "audit.json"))):
    clip = os.path.basename(os.path.dirname(path))
    r = json.load(open(path))
    r["_clip"] = clip
    reports.append(r)

print(f"{len(reports)} holdout clip(s) audited\n")
total_joints = sum(r["matched_joints"] for r in reports)
total_frames = sum(r["matched_frames"] for r in reports)

# weighted pool by matched_joints (mpjpe/pa_mpjpe) and matched_frames (yaw, since yaw is per-frame)
pooled = {}
for key in ("mpjpe_mm", "pa_mpjpe_mm", "p95_joint_error_mm"):
    pooled[key] = sum(r[key] * r["matched_joints"] for r in reports) / total_joints

yaw_reports = [r for r in reports if r.get("root_yaw_mae_degrees") is not None]
if yaw_reports:
    yaw_frames = sum(r["matched_frames"] for r in yaw_reports)
    pooled["root_yaw_mae_degrees"] = sum(r["root_yaw_mae_degrees"] * r["matched_frames"] for r in yaw_reports) / yaw_frames
    # P95 cannot be pooled by weighted mean; take the max of per-clip P95 as a conservative pooled bound
    pooled["root_yaw_p95_degrees"] = max(r["root_yaw_p95_degrees"] for r in yaw_reports)
else:
    pooled["root_yaw_mae_degrees"] = None
    pooled["root_yaw_p95_degrees"] = None

print("Per-clip:")
for r in reports:
    print(f"  {r['_clip']}: frames={r['matched_frames']} mpjpe={r['mpjpe_mm']:.1f}mm "
          f"pa_mpjpe={r['pa_mpjpe_mm']:.1f}mm yaw_mae={r['root_yaw_mae_degrees']} yaw_p95={r['root_yaw_p95_degrees']}")

print(f"\nPooled ({total_frames} frames, {total_joints} joint samples):")
for k, v in pooled.items():
    print(f"  {k}: {v}")

print("\nGate comparison (docs/06 runbook thresholds):")
verdict = "PASS"
for key, (op, threshold) in GATES.items():
    value = pooled.get(key)
    if value is None:
        print(f"  {key}: NO DATA -> BLOCKED")
        verdict = "BLOCKED"
        continue
    ok = value <= threshold if op == "<=" else False
    print(f"  {key}: {value:.3f} {op} {threshold} -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        verdict = "FAIL" if verdict != "BLOCKED" else verdict

print(f"\nOVERALL VERDICT: {verdict}")
print("Note: tracker-success and knee/elbow-flip gates are not evaluated here — this run trains and")
print("evaluates directly on MPI-INF-3DHP ground-truth 2D/3D (no video/tracker/flip-stabilizer stage")
print("involved). Those two gates apply to the live estimate-pose pipeline, not to this offline lifter-accuracy holdout.")

out_path = os.path.join(RUN_DIR, "holdout_gate_report.json")
out = {
    "schema": "animcv_holdout_gate_check_v1", "clips": [r["_clip"] for r in reports],
    "total_frames": total_frames, "total_joint_samples": total_joints,
    "pooled_metrics": pooled, "gates": {k: v[1] for k, v in GATES.items()}, "verdict": verdict,
    "per_clip": reports,
}
json.dump(out, open(out_path, "w"), indent=2)
print(f"\nwrote {out_path}")
