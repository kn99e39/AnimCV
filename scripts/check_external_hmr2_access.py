#!/usr/bin/env python3
"""ACCESS GATE (Worklog 97): can the official 4DHumans / HMR 2.0b baseline be run legitimately here?

Read-only checks, no download, no inference:
  1. body model: every candidate file is inspected structurally.  HMR 2.0 needs the official SMPL v1 neutral model
     (``basicModel_neutral_lbs_10_207_0_v1.0.0.pkl``: 24 joints, 10 shape and 207 pose blend shapes).  A file that merely
     carries the name ``SMPL_NEUTRAL.pkl`` (upstream ``check_smpl_exists`` tests the filename only) is NOT accepted when
     its structure is SMPL+H (52 joints) or anything else; converting or renaming another body model is an asset
     substitution, not access.
  2. upstream code: commit of a local clone (optional) and whether ``load_hmr2`` still calls ``check_smpl_exists``.
  3. GPU: free memory and foreign compute processes (optional, via ``nvidia-smi`` output passed in or queried).
Chumpy objects in legacy pickles are stubbed so the structure can be read without installing chumpy.
"""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
from pathlib import Path

SMPL_V1 = {"joints": 24, "shape_blend": 10, "pose_blend": 207, "vertices": 6890}


class _Stub:
    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        self.__dict__["state"] = state


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("chumpy"):
            return _Stub
        return super().find_class(module, name)


def _shape(value):
    if isinstance(value, _Stub):
        x = value.__dict__.get("state", {})
        x = x.get("x") if isinstance(x, dict) else None
        return tuple(getattr(x, "shape", ())) if x is not None else None
    return tuple(getattr(value, "shape", ())) or None


def inspect_body_model(path: Path) -> dict:
    out = {"path": str(path), "exists": path.exists(), "is_symlink": path.is_symlink()}
    if not path.exists():
        return {**out, "verdict": "missing"}
    if path.is_symlink():
        out["resolves_to"] = str(path.resolve())
    try:
        if path.suffix == ".npz":
            import numpy as np

            data = dict(np.load(path, allow_pickle=True))
        else:
            with path.open("rb") as handle:
                data = _Unpickler(handle, encoding="latin1").load()
    except Exception as exc:  # noqa: BLE001 - report, do not guess
        return {**out, "verdict": f"unreadable: {type(exc).__name__}"}
    shapes = {k: _shape(data.get(k)) for k in ("J_regressor", "kintree_table", "weights", "shapedirs", "posedirs", "v_template")}
    out["shapes"] = {k: list(v) if v else None for k, v in shapes.items()}
    joints = shapes["J_regressor"][0] if shapes["J_regressor"] else None
    verts = shapes["v_template"][0] if shapes["v_template"] else None
    shape_b = shapes["shapedirs"][-1] if shapes["shapedirs"] else None
    pose_b = shapes["posedirs"][-1] if shapes["posedirs"] else None
    out["structure"] = {"joints": joints, "vertices": verts, "shape_blend": shape_b, "pose_blend": pose_b,
                        "hand_components": any(k.startswith("hands_") for k in data)}
    if joints == 52:
        family = "SMPL+H"
    elif joints == 55:
        family = "SMPL-X"
    elif joints == 24:
        family = "SMPL"
    else:
        family = "unknown"
    out["family"] = family
    ok = (family == "SMPL" and verts == SMPL_V1["vertices"] and pose_b == SMPL_V1["pose_blend"] and shape_b is not None and shape_b >= SMPL_V1["shape_blend"])
    out["verdict"] = "accepted_smpl_v1" if ok else f"rejected_{family.lower().replace('+', 'plus').replace('-', '')}"
    return out


def upstream_status(clone: Path) -> dict:
    out = {"clone": str(clone)}
    try:
        out["commit"] = subprocess.run(["git", "-C", str(clone), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        out["commit"] = f"unavailable: {type(exc).__name__}"
    init = clone / "hmr2/models/__init__.py"
    if init.exists():
        text = init.read_text()
        out["load_hmr2_checks_smpl"] = "check_smpl_exists()" in text
        out["smpl_check_is_filename_only"] = "os.path.exists(c)" in text
        out["default_checkpoint"] = next((ln.split("=", 1)[1].strip() for ln in text.splitlines() if ln.startswith("DEFAULT_CHECKPOINT")), None)
    lic = clone / "LICENSE.md"
    out["code_licence_first_line"] = lic.read_text().splitlines()[0] if lic.exists() else None
    return out


def gpu_status(nvidia_smi_csv: str | None = None) -> dict:
    """Parse ``nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits``."""
    if nvidia_smi_csv is None:
        try:
            nvidia_smi_csv = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
                                            capture_output=True, text=True, check=True).stdout
        except Exception as exc:  # noqa: BLE001
            return {"available": False, "reason": type(exc).__name__}
    gpus = []
    for line in nvidia_smi_csv.strip().splitlines():
        used, total, util = (float(v) for v in line.split(","))
        gpus.append({"memory_used_mib": used, "memory_total_mib": total, "memory_free_mib": total - used, "utilization_pct": util,
                     "busy": util >= 10.0 or used >= 1024.0})
    return {"available": bool(gpus), "gpus": gpus}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--body-model", type=Path, action="append", default=[], help="candidate SMPL file (repeatable)")
    parser.add_argument("--upstream-clone", type=Path, default=None)
    parser.add_argument("--nvidia-smi-csv", default=None, help="pre-captured nvidia-smi CSV (memory.used,memory.total,utilization.gpu)")
    parser.add_argument("--skip-gpu", action="store_true")
    args = parser.parse_args()
    report = {"schema": "worklog97_hmr2_access_gate", "required_body_model": SMPL_V1,
              "body_models": [inspect_body_model(p) for p in args.body_model]}
    report["smpl_v1_available"] = any(b["verdict"] == "accepted_smpl_v1" for b in report["body_models"])
    if args.upstream_clone:
        report["upstream"] = upstream_status(args.upstream_clone)
    if not args.skip_gpu:
        report["gpu"] = gpu_status(args.nvidia_smi_csv)
    report["blocked"] = not report["smpl_v1_available"]
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
