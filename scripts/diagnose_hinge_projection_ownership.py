#!/usr/bin/env python3
"""Is the complementary hinge side observable under the REAL projection?

docs/38. docs/37 identified the complementary bend component `c_screen` with a
2D line-side predicate on `input_2d`, on the grounds that `v_hat` has zero depth
component. `v_hat.Y == 0` proves the direction is camera-parallel; it does NOT
prove that the canonical X/Z line side equals the line side of a *perspective*
projection, because projection divides by per-joint depth.

Algebraically, with proximal/middle/distal `P, M, D` and forward depth `Y`:

    orthographic side  ~ sign(  A -     B +     C )
    perspective  side  ~ sign(Y_P*A - Y_M*B + Y_D*C )   (all depths positive)

for the same three 2x2 minors `A = M_X D_Z - M_Z D_X`, `B = P_X D_Z - P_Z D_X`,
`C = P_X M_Z - P_Z M_X`. They coincide exactly when `Y_P == Y_M == Y_D` and can
differ otherwise -- and a foreshortened limb is precisely a chain with large
within-chain depth spread.

This separates three distinct mismatches on real frames:

    canonical-plane      target c_screen sign vs predicted c_screen sign
    perspective          target c_screen sign vs the side of the ACTUAL
                         projection of the target through 3DPW's own intrinsics
    detector             projected-target side vs the shipped detector's side

The absolute camera geometry is reconstructed with the repository's own
`pose.three_dpw_adapter` conversion and 3DPW's own `cam_intrinsics`/`cam_poses`.
Nothing is inferred: no focal length, principal point or root depth is invented,
and the reconstruction is refused unless it reproduces the bank's stored
`target_3d` after the adapter's own root subtraction.

No training, no model execution, no RGB, no VLM, no SignState change.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from common.canonical_pose import JOINT_INDEX
from common.serialization import write_json
from framepose.bank import load_bank
from framepose.branch_constraints import CORRECTED, MINIMUM_NORM, apply_branch_constraints_batch
from framepose.contract import JOINT_NAMES
from framepose.hinge_plane import axis_transport, bend_components
from framepose.replay_provenance import verify_source_identity
from framepose.signs import (
    HINGE_CHAINS_BY_JOINT, SIGN_FIELD_NAMES, UNKNOWN, mask_fields, oracle_sign_states, sign_state,
)
from pose.three_dpw_adapter import _SMPL_TO_CANONICAL, _world_to_animcv_camera

SCHEMA = "animcv_frame_pose_hinge_projection_ownership_v1"

HINGE_FIELDS = tuple(name for name in SIGN_FIELD_NAMES if name.endswith("_forward_bend"))
HINGE_JOINTS = tuple(name[: -len("_forward_bend")] for name in HINGE_FIELDS)

#: Populations are named once and never reused for a different set.
POPULATIONS = ("all_frames", "all_279_residual_flips", "depth_correct_83_residual_flips",
               "type_1_corrections", "type_2_corrections")


def canonical_side(points: np.ndarray) -> int:
    """Orthographic canonical X/Z line side (depth discarded)."""
    p, m, d = points
    return int(np.sign((d[0] - p[0]) * (m[2] - p[2]) - (d[2] - p[2]) * (m[0] - p[0])))


def projected_side(pixels: np.ndarray) -> int:
    """Line side in image pixels, y downward. Uses the same handedness as
    `hinge_plane.observed_screen_side` so the two are directly comparable."""
    p, m, d = pixels
    return int(np.sign((d[1] - p[1]) * (m[0] - p[0]) - (d[0] - p[0]) * (m[1] - p[1])))


def project(points_animcv: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """AnimCV camera axes -> 3DPW OpenCV camera axes -> pixels.

    `three_dpw_adapter._world_to_animcv_camera` maps OpenCV `(x, y, z)` to
    AnimCV `(x, z, -y)`, so the inverse is `x = X`, `y = -Z`, `z = Y`.
    """
    x = points_animcv[..., 0]
    y = -points_animcv[..., 2]
    z = points_animcv[..., 1]
    u = intrinsics[0, 0] * x / z + intrinsics[0, 2]
    v = intrinsics[1, 1] * y / z + intrinsics[1, 2]
    return np.stack([u, v], axis=-1)


def load_absolute_sequences(raw_root: Path, split: str) -> dict[str, Any]:
    """Absolute AnimCV-camera joints per `3dpw:<sequence>:actor<N>`, with K."""
    sequences: dict[str, Any] = {}
    directory = raw_root / split
    for path in sorted(directory.glob("*.pkl")):
        with path.open("rb") as handle:
            raw = pickle.load(handle, encoding="latin1")
        camera = np.asarray(raw["cam_poses"], dtype=float)
        intrinsics = np.asarray(raw["cam_intrinsics"], dtype=float)
        for actor, joints in enumerate(raw["jointPositions"]):
            world = np.asarray(joints, dtype=float).reshape(-1, 24, 3)
            count = min(len(world), len(camera))
            absolute = np.stack([_world_to_animcv_camera(world[index], camera[index])
                                 for index in range(count)])
            sequences[f"3dpw:{raw['sequence']}:actor{actor}"] = {
                "absolute_smpl24": absolute, "intrinsics": intrinsics,
                "image_size": (int(round(intrinsics[0, 2] * 2)), int(round(intrinsics[1, 2] * 2))),
            }
    return sequences


def _confusion(observed: list[int], reference: list[int]) -> dict[str, Any]:
    pairs = [(o, r) for o, r in zip(observed, reference) if o != 0 and r != 0]
    unresolved = sum(1 for o, r in zip(observed, reference) if o == 0 or r == 0)
    if not pairs:
        return {"valid": 0, "unresolved": unresolved}
    matrix = Counter(pairs)
    agree = sum(count for (o, r), count in matrix.items() if o == r)
    per_class = {}
    for value in (1, -1):
        total = sum(count for (o, r), count in matrix.items() if r == value)
        per_class[str(value)] = {"reference_count": total,
                                 "recall": (matrix.get((value, value), 0) / total) if total else None}
    recalls = [entry["recall"] for entry in per_class.values() if entry["recall"] is not None]
    return {
        "valid": len(pairs), "unresolved": unresolved,
        "agreement": agree / len(pairs),
        "balanced_accuracy": float(np.mean(recalls)) if recalls else None,
        "observed_positive": sum(1 for o, _ in pairs if o > 0),
        "observed_negative": sum(1 for o, _ in pairs if o < 0),
        "reference_positive": per_class["1"]["reference_count"],
        "reference_negative": per_class["-1"]["reference_count"],
        "confusion": {f"obs{o:+d}|ref{r:+d}": count for (o, r), count in sorted(matrix.items())},
        "per_class_recall": per_class,
    }


def _quantiles(values) -> dict[str, Any]:
    values = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if not values.size:
        return {"count": 0}
    return {"count": int(values.size), "mean": float(values.mean()),
            **{f"p{q}": float(np.percentile(values, q)) for q in (50, 90)},
            "max": float(values.max())}


def main() -> int:
    parser = argparse.ArgumentParser(description="Perspective observability of the complementary hinge side")
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--source", required=True, help="LABEL=CANDIDATE:PREDICTION[:EVALUATION]")
    parser.add_argument("--raw-root", required=True, type=Path, help="3DPW sequenceFiles directory")
    parser.add_argument("--raw-split", default="validation", help="3DPW split holding the bank's test sequences")
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    bank = load_bank(args.bank)
    positions = bank.indices(args.split)
    valid = bank.arrays["target_valid"][positions]
    observation = bank.arrays["input_2d"][positions].astype(np.float64)
    observed_valid = bank.arrays["input_valid"][positions]
    targets = bank.arrays["target_3d"][positions].astype(np.float64)
    oracle = oracle_sign_states(bank.arrays["target_3d"], bank.arrays["target_valid"])[positions]
    requested = mask_fields(oracle, list(HINGE_FIELDS)).astype(np.int64)

    label, rest = args.source.split("=", 1)
    parts = rest.split(":")
    candidate, prediction = parts[0], Path(parts[1])
    evaluation = Path(parts[2]) if len(parts) > 2 and parts[2] else None
    identity = verify_source_identity(prediction=prediction, evaluation=evaluation, bank=bank,
                                      split=args.split, candidate=candidate,
                                      frames=len(positions), joints=len(JOINT_NAMES))

    before = np.load(prediction).astype(np.float64)
    after, reports = apply_branch_constraints_batch(before, valid, requested,
                                                    fields=list(HINGE_FIELDS),
                                                    hinge_write_policy=MINIMUM_NORM)
    after_signs = np.stack([sign_state(after[order], valid[order]) for order in range(len(after))])

    sequences = load_absolute_sequences(args.raw_root, args.raw_split)
    smpl_rows = np.array([_SMPL_TO_CANONICAL[name] for name in JOINT_NAMES])

    # --- reconstruct absolute canonical joints, and REFUSE unless they
    # --- reproduce the bank's own stored target_3d after root subtraction.
    absolute = np.full((len(positions), len(JOINT_NAMES), 3), np.nan)
    intrinsics_by_frame: list[np.ndarray | None] = [None] * len(positions)
    missing = Counter()
    for order, position in enumerate(positions):
        sample = bank.samples[int(position)]
        entry = sequences.get(sample.sequence_id)
        if entry is None:
            missing["sequence_not_in_raw_split"] += 1
            continue
        frame = int(sample.frame_index)
        if frame >= len(entry["absolute_smpl24"]):
            missing["frame_beyond_raw_sequence"] += 1
            continue
        absolute[order] = entry["absolute_smpl24"][frame][smpl_rows]
        intrinsics_by_frame[order] = entry["intrinsics"]

    usable = ~np.isnan(absolute).any(axis=(1, 2))
    reconstruction_error_mm = None
    if usable.any():
        rebuilt = absolute[usable] - absolute[usable][:, JOINT_INDEX["pelvis"]][:, None, :]
        difference = np.abs(rebuilt - targets[usable]) * 1000.0
        reconstruction_error_mm = {"max": float(difference.max()), "mean": float(difference.mean()),
                                   "p99": float(np.percentile(difference, 99))}
        if difference.max() > 1.0:
            raise ValueError(
                f"reconstructed absolute joints do not reproduce the bank's target_3d "
                f"(max {difference.max():.3f} mm); refusing to attribute projection behaviour to a "
                "reconstruction that is not the same geometry")

    # ---------------------------------------------------------------- populate
    per_field: dict[str, Any] = {}
    identities: dict[str, Any] = {}
    for column, field in enumerate(HINGE_FIELDS):
        chain = HINGE_CHAINS_BY_JOINT[HINGE_JOINTS[column]]
        chain_indices = [JOINT_INDEX[name] for name in chain]
        index = SIGN_FIELD_NAMES.index(field)

        buckets: dict[str, dict[str, list]] = {
            name: defaultdict(list) for name in POPULATIONS}
        counts = Counter()

        for order in range(len(positions)):
            if not all(bool(valid[order][i]) for i in chain_indices):
                continue
            target_components = bend_components(targets[order], chain)
            predicted_components = bend_components(after[order], chain)
            if target_components is None:
                continue
            target_canonical = int(np.sign(target_components["c_screen"]))
            predicted_canonical = (int(np.sign(predicted_components["c_screen"]))
                                   if predicted_components is not None else 0)

            # detector side, from input_2d only
            if all(bool(observed_valid[order][i]) for i in chain_indices):
                pixels = np.stack([observation[order][i][:2] for i in chain_indices])
                detector = projected_side(pixels)
                detector_cross = float((pixels[2][1] - pixels[0][1]) * (pixels[1][0] - pixels[0][0])
                                       - (pixels[2][0] - pixels[0][0]) * (pixels[1][1] - pixels[0][1]))
            else:
                detector, detector_cross = 0, None

            # projected-target side, through 3DPW's own intrinsics
            if usable[order]:
                chain_absolute = absolute[order][chain_indices]
                projected = project(chain_absolute, intrinsics_by_frame[order])
                projected_target = projected_side(projected)
                depth_spread = float(chain_absolute[:, 1].max() - chain_absolute[:, 1].min())
            else:
                projected_target, depth_spread = 0, None

            row = {
                "target_canonical": target_canonical,
                "predicted_canonical": predicted_canonical,
                "detector": detector,
                "projected_target": projected_target,
                "detector_cross": detector_cross,
                "target_c_screen_mm": abs(target_components["c_screen"]) * 1000.0,
                "depth_spread_m": depth_spread,
            }

            def add(name):
                for key, value in row.items():
                    buckets[name][key].append(value)

            add("all_frames")
            counts["all_frames"] += 1

            transport = axis_transport(after[order], targets[order], chain)
            entry = reports[order]["fields"][field]
            want = int(requested[order, index])

            if entry["outcome"] == CORRECTED:
                before_transport = axis_transport(before[order], targets[order], chain)
                if before_transport.get("resolved"):
                    if before_transport["historical_flipped"]:
                        add("type_1_corrections"); counts["type_1_corrections"] += 1
                    else:
                        add("type_2_corrections"); counts["type_2_corrections"] += 1

            if not transport.get("resolved") or not transport["historical_flipped"]:
                continue
            add("all_279_residual_flips")
            counts["all_279_residual_flips"] += 1
            depth_correct = (want != UNKNOWN and int(after_signs[order, index]) == want)
            if depth_correct:
                add("depth_correct_83_residual_flips")
                counts["depth_correct_83_residual_flips"] += 1

        field_report: dict[str, Any] = {"chain": list(chain), "counts": dict(counts)}
        for name in POPULATIONS:
            data = buckets[name]
            if not data:
                field_report[name] = {"count": 0}
                continue
            field_report[name] = {
                "count": len(data["target_canonical"]),
                "detector_vs_target_canonical": _confusion(data["detector"], data["target_canonical"]),
                "projected_target_vs_target_canonical": _confusion(data["projected_target"],
                                                                   data["target_canonical"]),
                "detector_vs_projected_target": _confusion(data["detector"], data["projected_target"]),
                "predicted_vs_target_canonical": _confusion(data["predicted_canonical"],
                                                            data["target_canonical"]),
                "detector_cross_magnitude": _quantiles([abs(v) for v in data["detector_cross"] if v is not None]),
                "target_c_screen_magnitude_mm": _quantiles(data["target_c_screen_mm"]),
                "chain_depth_spread_m": _quantiles(data["depth_spread_m"]),
            }
        per_field[field] = field_report
        identities[field] = {
            "all_279_residual_flips": counts["all_279_residual_flips"],
            "depth_correct_83_residual_flips": counts["depth_correct_83_residual_flips"],
            "depth_incorrect_residual_flips": counts["all_279_residual_flips"] - counts["depth_correct_83_residual_flips"],
            "identity_holds": True,
        }

    totals = Counter()
    for entry in identities.values():
        totals["all_279"] += entry["all_279_residual_flips"]
        totals["depth_correct_83"] += entry["depth_correct_83_residual_flips"]
        totals["depth_incorrect_196"] += entry["depth_incorrect_residual_flips"]

    report = {
        "schema": SCHEMA,
        "method": ("stored prediction + MINIMUM_NORM oracle-depth hinge correction; absolute camera "
                   "geometry reconstructed with pose.three_dpw_adapter's own conversion and 3DPW's "
                   "own cam_intrinsics/cam_poses; no camera parameter invented"),
        "line_side_algebra": {
            "orthographic": "sign(A - B + C)",
            "perspective": "sign(Y_P*A - Y_M*B + Y_D*C), all depths positive",
            "minors": "A = M_X D_Z - M_Z D_X, B = P_X D_Z - P_Z D_X, C = P_X M_Z - P_Z M_X",
            "equal_iff": "Y_P == Y_M == Y_D",
        },
        "provenance": {
            "bank_index": str(args.bank), "bank_content_digest": bank.content_digest(),
            "observation_regime": bank.regime(), "split": args.split,
            "frames": int(len(positions)), "source": identity,
            "raw_root": str(args.raw_root), "raw_split": args.raw_split,
            "frames_with_absolute_geometry": int(usable.sum()),
            "frames_without_absolute_geometry": dict(missing),
            "reconstruction_vs_bank_target_3d_mm": reconstruction_error_mm,
            "input_2d_space": "normalized image coordinates (pixel / image_size), y downward",
            "input_2d_origin": "3DPW poses2d, OpenPose COCO-18 detector; pelvis/spine/neck are midpoints",
            "target_3d_space": "camera_root_relative metres, AnimCV axes (+X right, +Y depth, +Z up)",
        },
        "population_identities": identities,
        "population_totals": dict(totals),
        "fields": per_field,
    }
    write_json(args.out / "hinge_projection_ownership.json", report)

    print(f"\n== perspective observability ({len(positions)} {args.split} frames)")
    print("   absolute geometry reconstructed for %d/%d frames; max |rebuilt - bank target_3d| = %s" % (
        usable.sum(), len(positions),
        "n/a" if reconstruction_error_mm is None else "%.6f mm" % reconstruction_error_mm["max"]))
    print("   population totals:", dict(totals))
    for name in POPULATIONS:
        print(f"\n   -- {name} --")
        print("   %-26s %14s %14s %14s" % ("field", "detector~target", "projected~target", "detector~projected"))
        for field, entry in per_field.items():
            data = entry.get(name) or {}
            if not data.get("count"):
                print("   %-26s %14s %14s %14s" % (field, "-", "-", "-")); continue
            def cell(key):
                value = data[key]
                if not value.get("valid"):
                    return "-"
                return "%.3f (n=%d)" % (value["balanced_accuracy"] or float("nan"), value["valid"])
            print("   %-26s %14s %14s %14s" % (
                field, cell("detector_vs_target_canonical"),
                cell("projected_target_vs_target_canonical"), cell("detector_vs_projected_target")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
