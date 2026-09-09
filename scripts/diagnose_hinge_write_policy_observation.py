#!/usr/bin/env python3
"""Which hinge write policy preserves the VISIBLE observation, not just X/Z?

docs/39. `DEPTH_ONLY` preserves canonical X and Z exactly; `MINIMUM_NORM` does
not. docs/37 read that as a position-ownership argument against MINIMUM_NORM.
docs/38 makes the objection to that reading concrete: image position is
`x ~ X/Y`, `z ~ -Z/Y`, so a write that changes ONLY the depth Y still moves the
joint in the image -- and DEPTH_ONLY's depth writes reach 3.1 m. "Canonical X/Z
unchanged" may therefore be a poor proxy for visible-position preservation.

This measures the comparison that was never made, under the real 3DPW camera
recovered exactly as in docs/38. Both operators already produce the identical
hinge branch and bend direction (docs/36), which is asserted here before
anything else is reported, so the only variable is where the middle joint lands.

Two references are kept strictly apart:

    A  projected target   the same absolute SMPL joint centre, projected
    B  observation        the stored 3DPW input_2d the FramePose Core consumed

A is like-for-like geometry. B is what the model actually saw, and its gap is
NOT pure detector error -- OpenPose COCO landmarks and SMPL joint centres are
different semantic objects -- so it is reported as observation-consistency
error, never as detector fidelity.

No training, no model execution, no new operator, no threshold, no promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from common.canonical_pose import JOINT_INDEX, bend_direction
from common.serialization import write_json
from framepose.bank import load_bank
from framepose.branch_constraints import (
    CORRECTED, DEPTH_ONLY, MINIMUM_NORM, apply_branch_constraints_batch,
)
from framepose.contract import JOINT_NAMES
from framepose.evaluate import evaluate_predictions
from framepose.replay_provenance import digest, verify_source_identity
from framepose.signs import (
    HINGE_CHAINS_BY_JOINT, SIGN_FIELD_NAMES, UNKNOWN, mask_fields, oracle_sign_states, sign_state,
)
from pose.three_dpw_adapter import _SMPL_TO_CANONICAL, _world_to_animcv_camera

SCHEMA = "animcv_frame_pose_hinge_write_policy_observation_v1"

HINGE_FIELDS = tuple(name for name in SIGN_FIELD_NAMES if name.endswith("_forward_bend"))
HINGE_JOINTS = tuple(name[: -len("_forward_bend")] for name in HINGE_FIELDS)
MIDDLE_INDEX = {field: JOINT_INDEX[HINGE_CHAINS_BY_JOINT[joint][1]]
                for field, joint in zip(HINGE_FIELDS, HINGE_JOINTS)}

#: docs/36's already-published tail ranges, carried for continuity ONLY. Nothing
#: branches on them and none is a promotion gate.
CONTINUITY_RANGES_MM = (200.0, 500.0, 1000.0)

#: The reconstruction boundary this diagnostic actually enforces, stated in the
#: same units it is computed in so the prose cannot drift from the code again.
RECONSTRUCTION_REFUSAL_MM = 1.0


def project(points_animcv: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """AnimCV camera axes -> 3DPW OpenCV axes -> pixels (see docs/38).

    `intrinsics` may be one `(3, 3)` matrix or one per point. 3DPW's test split
    mixes portrait and landscape sequences with different K, so a per-frame
    matrix is the correct form here.
    """
    x = points_animcv[..., 0]
    y = -points_animcv[..., 2]
    z = points_animcv[..., 1]
    intrinsics = np.asarray(intrinsics, dtype=float)
    if intrinsics.ndim == 2:
        fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    else:
        fx, fy = intrinsics[..., 0, 0], intrinsics[..., 1, 1]
        cx, cy = intrinsics[..., 0, 2], intrinsics[..., 1, 2]
    return np.stack([fx * x / z + cx, fy * y / z + cy], axis=-1)


def load_absolute_sequences(raw_root: Path, split: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sequences: dict[str, Any] = {}
    provenance: list[dict[str, Any]] = []
    for path in sorted((raw_root / split).glob("*.pkl")):
        data = path.read_bytes()
        with path.open("rb") as handle:
            raw = pickle.load(handle, encoding="latin1")
        camera = np.asarray(raw["cam_poses"], dtype=float)
        intrinsics = np.asarray(raw["cam_intrinsics"], dtype=float)
        provenance.append({
            "sequence": str(raw["sequence"]), "path": str(path), "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "intrinsics": [[float(v) for v in row] for row in intrinsics],
            "image_size_from_intrinsics": [int(round(intrinsics[0, 2] * 2)),
                                           int(round(intrinsics[1, 2] * 2))],
            "actors": len(raw["jointPositions"]), "frames": int(len(camera)),
        })
        for actor, joints in enumerate(raw["jointPositions"]):
            world = np.asarray(joints, dtype=float).reshape(-1, 24, 3)
            count = min(len(world), len(camera))
            absolute = np.stack([_world_to_animcv_camera(world[index], camera[index])
                                 for index in range(count)])
            sequences[f"3dpw:{raw['sequence']}:actor{actor}"] = {
                "absolute_smpl24": absolute, "intrinsics": intrinsics}
    return sequences, provenance


def _quantiles(values) -> dict[str, Any]:
    values = np.asarray([v for v in np.ravel(values) if v is not None and np.isfinite(v)], dtype=float)
    if not values.size:
        return {"count": 0}
    return {"count": int(values.size), "mean": float(values.mean()),
            **{f"p{q}": float(np.percentile(values, q)) for q in (50, 90, 99)},
            "max": float(values.max())}


def _bind_history(path: Path | None, expected_prediction_sha: str,
                  totals: dict[str, int]) -> dict[str, Any]:
    """Bind against the accepted historical residual artifact, or say we cannot."""
    if path is None or not path.is_file():
        return {"attempted": False, "verifiable": False,
                "reason": "no historical artifact supplied"}
    payload = json.loads(path.read_text())
    record: dict[str, Any] = {"attempted": True, "artifact": digest(path),
                              "schema": payload.get("schema")}
    fields = payload.get("fields") or {}
    residual = sum(len(entry.get("residual_flip_records") or []) for entry in fields.values())
    depth_correct = sum(1 for entry in fields.values()
                        for row in (entry.get("residual_flip_records") or [])
                        if row.get("depth_side_correct"))
    stored_policy = payload.get("hinge_write_policy")
    stored_source = ((payload.get("provenance") or {}).get("source") or {}).get("prediction") or {}
    missing = []
    if not fields or residual == 0:
        missing.append("residual_flip_records")
    if stored_policy is None:
        missing.append("hinge_write_policy")
    if not stored_source.get("sha256"):
        missing.append("source prediction sha256")
    if missing:
        record.update({"verifiable": False, "reason": f"artifact lacks {missing}"})
        return record
    checks = {
        "residual_total_matches": residual == totals["all_residual_flips"],
        "depth_correct_matches": depth_correct == totals["depth_correct_residual_flips"],
        "source_prediction_matches": stored_source["sha256"] == expected_prediction_sha,
        "hinge_policy_matches_minimum_norm": stored_policy == MINIMUM_NORM,
    }
    record.update({
        "verifiable": True,
        "historical_residual_total": residual,
        "historical_depth_correct": depth_correct,
        "current_residual_total": totals["all_residual_flips"],
        "current_depth_correct": totals["depth_correct_residual_flips"],
        "historical_hinge_write_policy": stored_policy,
        "checks": checks,
        "all_checks_pass": all(checks.values()),
    })
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="Hinge write policy vs Geometry Observation ownership")
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--source", required=True, help="LABEL=CANDIDATE:PREDICTION[:EVALUATION]")
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--raw-split", default="test")
    parser.add_argument("--split", default="test")
    parser.add_argument("--historical", type=Path, default=None,
                        help="accepted historical residual artifact to bind against")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    bank = load_bank(args.bank)
    positions = bank.indices(args.split)
    valid = bank.arrays["target_valid"][positions]
    observed_valid = bank.arrays["input_valid"][positions]
    observation = bank.arrays["input_2d"][positions].astype(np.float64)
    targets = bank.arrays["target_3d"][positions].astype(np.float64)
    oracle = oracle_sign_states(bank.arrays["target_3d"], bank.arrays["target_valid"])[positions]
    requested = mask_fields(oracle, list(HINGE_FIELDS)).astype(np.int64)
    opposite = requested.copy()
    known = opposite != UNKNOWN
    opposite[known] = -opposite[known]

    label, rest = args.source.split("=", 1)
    parts = rest.split(":")
    candidate, prediction_path = parts[0], Path(parts[1])
    evaluation = Path(parts[2]) if len(parts) > 2 and parts[2] else None
    identity = verify_source_identity(prediction=prediction_path, evaluation=evaluation, bank=bank,
                                      split=args.split, candidate=candidate,
                                      frames=len(positions), joints=len(JOINT_NAMES))
    h0 = np.load(prediction_path).astype(np.float64)

    states: dict[str, np.ndarray] = {"H0": h0}
    reports: dict[str, list] = {}
    for policy in (DEPTH_ONLY, MINIMUM_NORM):
        corrected, policy_reports = apply_branch_constraints_batch(
            h0, valid, requested, fields=list(HINGE_FIELDS), hinge_write_policy=policy)
        states[policy] = corrected
        reports[policy] = policy_reports
        wrong, wrong_reports = apply_branch_constraints_batch(
            h0, valid, opposite, fields=list(HINGE_FIELDS), hinge_write_policy=policy)
        states[f"{policy}__opposite"] = wrong
        reports[f"{policy}__opposite"] = wrong_reports

    # ---- semantic identity: the ONLY thing that makes this a fair comparison
    semantic: dict[str, Any] = {"checked_chain_frames": 0, "bend_direction_max_difference": 0.0,
                                "sign_state_identical": True}
    for order in range(len(positions)):
        for field in HINGE_FIELDS:
            chain = HINGE_CHAINS_BY_JOINT[field[: -len("_forward_bend")]]
            indices = [JOINT_INDEX[name] for name in chain]
            if not all(bool(valid[order][i]) for i in indices):
                continue
            first = bend_direction(states[DEPTH_ONLY][order][indices[1]],
                                   states[DEPTH_ONLY][order][indices[0]],
                                   states[DEPTH_ONLY][order][indices[2]])
            second = bend_direction(states[MINIMUM_NORM][order][indices[1]],
                                    states[MINIMUM_NORM][order][indices[0]],
                                    states[MINIMUM_NORM][order][indices[2]])
            if first is None or second is None:
                continue
            semantic["checked_chain_frames"] += 1
            semantic["bend_direction_max_difference"] = max(
                semantic["bend_direction_max_difference"], float(np.abs(first - second).max()))
    depth_signs = np.stack([sign_state(states[DEPTH_ONLY][o], valid[o]) for o in range(len(positions))])
    norm_signs = np.stack([sign_state(states[MINIMUM_NORM][o], valid[o]) for o in range(len(positions))])
    semantic["sign_state_identical"] = bool(np.array_equal(depth_signs, norm_signs))
    if semantic["bend_direction_max_difference"] > 1e-9 or not semantic["sign_state_identical"]:
        raise ValueError(
            "DEPTH_ONLY and MINIMUM_NORM no longer produce the same hinge semantics "
            f"(max bend-direction difference {semantic['bend_direction_max_difference']:.3e}, "
            f"sign states identical={semantic['sign_state_identical']}); refusing to compare "
            "write policies whose enforced branch is not the same")

    # ---- the real camera, recovered exactly as in docs/38 --------------------
    sequences, raw_provenance = load_absolute_sequences(args.raw_root, args.raw_split)
    smpl_rows = np.array([_SMPL_TO_CANONICAL[name] for name in JOINT_NAMES])
    absolute_target = np.full((len(positions), len(JOINT_NAMES), 3), np.nan)
    intrinsics = np.full((len(positions), 3, 3), np.nan)
    image_size = np.zeros((len(positions), 2))
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
        absolute_target[order] = entry["absolute_smpl24"][frame][smpl_rows]
        intrinsics[order] = entry["intrinsics"]
        image_size[order] = sample.image_size

    usable = ~np.isnan(absolute_target).any(axis=(1, 2))
    rebuilt = absolute_target[usable] - absolute_target[usable][:, JOINT_INDEX["pelvis"]][:, None, :]
    reconstruction = np.abs(rebuilt - targets[usable]) * 1000.0
    reconstruction_report = {"max_mm": float(reconstruction.max()),
                             "mean_mm": float(reconstruction.mean()),
                             "implemented_refusal_boundary_mm": RECONSTRUCTION_REFUSAL_MM}
    if reconstruction.max() > RECONSTRUCTION_REFUSAL_MM:
        raise ValueError(f"reconstruction disagrees with the bank by {reconstruction.max():.4f} mm")
    # Per frame, the intrinsics' own implied image size must agree with the
    # bank's. 3DPW's test split mixes portrait and landscape sequences, so this
    # is deliberately NOT a single-size check.
    implied = np.stack([np.round(intrinsics[:, 0, 2] * 2), np.round(intrinsics[:, 1, 2] * 2)], axis=-1)
    mismatched = int((implied[usable] != image_size[usable]).any(axis=-1).sum())
    if mismatched:
        raise ValueError(
            f"{mismatched} frames' bank image_size disagrees with the intrinsics' implied size")
    orientations = {tuple(int(v) for v in row) for row in image_size[usable]}

    # ---- ORACLE CAMERA PLACEMENT (diagnostic only) --------------------------
    root = absolute_target[:, JOINT_INDEX["pelvis"]][:, None, :]
    placed = {name: array + root for name, array in states.items()}

    per_field: dict[str, Any] = {}
    totals = Counter()
    for field in HINGE_FIELDS:
        middle = MIDDLE_INDEX[field]
        chain = HINGE_CHAINS_BY_JOINT[field[: -len("_forward_bend")]]
        indices = [JOINT_INDEX[name] for name in chain]
        index = SIGN_FIELD_NAMES.index(field)

        touched = np.array([reports[MINIMUM_NORM][o]["fields"][field]["outcome"] == CORRECTED
                            for o in range(len(positions))])
        frame_usable = usable & touched & valid[:, middle] & observed_valid[:, middle]
        totals["corrections"] += int(touched.sum())
        if not frame_usable.any():
            per_field[field] = {"corrections": int(touched.sum()), "usable": 0}
            continue

        rows = np.flatnonzero(frame_usable)
        K = intrinsics[rows]
        size = image_size[rows]
        diagonal = np.linalg.norm(size, axis=-1)

        pixels = {name: project(array[rows, middle], K) for name, array in placed.items()}
        reference_a = project(absolute_target[rows, middle], K)
        reference_b = observation[rows, middle, :2] * size          # normalized -> pixels

        def image_stats(a, b, scale=diagonal):
            delta = a - b
            magnitude = np.linalg.norm(delta, axis=-1)
            return {"pixel": _quantiles(magnitude),
                    "normalized": _quantiles(magnitude / scale),
                    "abs_delta_image_x_px": _quantiles(np.abs(delta[:, 0])),
                    "abs_delta_image_y_px": _quantiles(np.abs(delta[:, 1]))}

        entry: dict[str, Any] = {
            "corrections": int(touched.sum()), "usable": int(frame_usable.sum()),
            "middle_joint": JOINT_NAMES[middle],
            "correction_induced_image_displacement": {
                policy: image_stats(pixels[policy], pixels["H0"]) for policy in (DEPTH_ONLY, MINIMUM_NORM)},
            "target_projection_error": {
                name: image_stats(pixels[name], reference_a) for name in ("H0", DEPTH_ONLY, MINIMUM_NORM)},
            "observation_consistency_error": {
                name: image_stats(pixels[name], reference_b) for name in ("H0", DEPTH_ONLY, MINIMUM_NORM)},
            "canonical_delta_mm": {
                policy: {axis: _quantiles(np.abs(states[policy][rows, middle, column]
                                                 - h0[rows, middle, column]) * 1000.0)
                         for column, axis in ((0, "x"), (1, "y"), (2, "z"))}
                for policy in (DEPTH_ONLY, MINIMUM_NORM)},
            "canonical_xz_preserved_exactly": {
                policy: bool(np.array_equal(states[policy][rows][:, middle][:, [0, 2]],
                                            h0[rows][:, middle][:, [0, 2]]))
                for policy in (DEPTH_ONLY, MINIMUM_NORM)},
        }

        # bone lengths around the middle joint
        bones = {}
        for policy in (DEPTH_ONLY, MINIMUM_NORM):
            changes = []
            for near, far in ((indices[0], middle), (middle, indices[2])):
                before = np.linalg.norm(h0[rows, far] - h0[rows, near], axis=-1)
                after = np.linalg.norm(states[policy][rows, far] - states[policy][rows, near], axis=-1)
                changes.append(np.abs(after - before) * 1000.0)
            bones[policy] = _quantiles(np.concatenate(changes))
        entry["bone_length_abs_change_mm"] = bones

        # conditioning regime, from the geometry the closed form is built on
        axis = h0[rows, indices[2]] - h0[rows, indices[0]]
        axis_squared = (axis * axis).sum(axis=-1)
        sqrt_f = np.sqrt((axis[:, 0] ** 2 + axis[:, 2] ** 2) / axis_squared)
        depth_spread = (absolute_target[rows][:, indices, 1].max(axis=1)
                        - absolute_target[rows][:, indices, 1].min(axis=1))
        correction_norm = np.linalg.norm(states[MINIMUM_NORM][rows, middle] - h0[rows, middle], axis=-1) * 1000.0
        depth_only_norm = np.linalg.norm(states[DEPTH_ONLY][rows, middle] - h0[rows, middle], axis=-1) * 1000.0
        displacement = {policy: np.linalg.norm(pixels[policy] - pixels["H0"], axis=-1)
                        for policy in (DEPTH_ONLY, MINIMUM_NORM)}

        def correlate(a, b):
            good = np.isfinite(a) & np.isfinite(b) & (b > 0)
            if good.sum() < 3:
                return None
            return float(np.corrcoef(a[good], np.log10(b[good]))[0, 1])

        entry["conditioning"] = {
            "sqrt_f": _quantiles(sqrt_f),
            "chain_depth_spread_m": _quantiles(depth_spread),
            "correction_norm_mm": {DEPTH_ONLY: _quantiles(depth_only_norm),
                                   MINIMUM_NORM: _quantiles(correction_norm)},
            "correlation_with_log10_image_displacement": {
                policy: {"sqrt_f": correlate(sqrt_f, displacement[policy]),
                         "chain_depth_spread_m": correlate(depth_spread, displacement[policy]),
                         "correction_norm_mm": correlate(
                             np.log10(np.maximum(depth_only_norm if policy == DEPTH_ONLY else correction_norm, 1e-6)),
                             displacement[policy])}
                for policy in (DEPTH_ONLY, MINIMUM_NORM)},
            "continuity_ranges": {
                "%dmm" % int(threshold): {
                    "frames": int((depth_only_norm > threshold).sum()),
                    "median_image_displacement_px": {
                        policy: (float(np.median(displacement[policy][depth_only_norm > threshold]))
                                 if (depth_only_norm > threshold).any() else None)
                        for policy in (DEPTH_ONLY, MINIMUM_NORM)}}
                for threshold in CONTINUITY_RANGES_MM},
        }

        # Wrong-sign endpoint, in observation space. The frames the OPPOSITE
        # request corrects are a different set: where the oracle request was
        # CORRECTED the pose already holds the opposite branch, so the opposite
        # request is a no-op there. Selecting on the oracle reports would have
        # measured an empty write.
        wrong_rows = usable & np.array([
            reports[f"{MINIMUM_NORM}__opposite"][o]["fields"][field]["outcome"] == CORRECTED
            for o in range(len(positions))
        ]) & valid[:, middle] & observed_valid[:, middle]
        wrong_index = np.flatnonzero(wrong_rows)
        wrong_K = intrinsics[wrong_index]
        wrong_size = image_size[wrong_index]
        wrong_diagonal = np.linalg.norm(wrong_size, axis=-1)
        entry["wrong_sign_endpoint"] = {
            "corrected_frames": int(len(wrong_index)),
            **{policy: {
                "image_displacement": image_stats(
                    project(placed[f"{policy}__opposite"][wrong_index, middle], wrong_K),
                    project(placed["H0"][wrong_index, middle], wrong_K), wrong_diagonal),
                "target_projection_error": image_stats(
                    project(placed[f"{policy}__opposite"][wrong_index, middle], wrong_K),
                    project(absolute_target[wrong_index, middle], wrong_K), wrong_diagonal),
                "observation_consistency_error": image_stats(
                    project(placed[f"{policy}__opposite"][wrong_index, middle], wrong_K),
                    observation[wrong_index, middle, :2] * wrong_size, wrong_diagonal),
            } for policy in (DEPTH_ONLY, MINIMUM_NORM)}}
        per_field[field] = entry

    totals["all_residual_flips"] = 0
    totals["depth_correct_residual_flips"] = 0
    from framepose.hinge_plane import axis_transport

    after_signs = np.stack([sign_state(states[MINIMUM_NORM][o], valid[o]) for o in range(len(positions))])
    for order in range(len(positions)):
        for field in HINGE_FIELDS:
            chain = HINGE_CHAINS_BY_JOINT[field[: -len("_forward_bend")]]
            indices = [JOINT_INDEX[name] for name in chain]
            if not all(bool(valid[order][i]) for i in indices):
                continue
            transport = axis_transport(states[MINIMUM_NORM][order], targets[order], chain)
            if not transport.get("resolved") or not transport["historical_flipped"]:
                continue
            totals["all_residual_flips"] += 1
            index = SIGN_FIELD_NAMES.index(field)
            want = int(requested[order, index])
            if want != UNKNOWN and int(after_signs[order, index]) == want:
                totals["depth_correct_residual_flips"] += 1

    aggregate = {name: evaluate_predictions(bank, positions, states[name], candidate=f"{label}+{name}")
                 for name in ("H0", DEPTH_ONLY, MINIMUM_NORM,
                              f"{DEPTH_ONLY}__opposite", f"{MINIMUM_NORM}__opposite")}

    report = {
        "schema": SCHEMA,
        "purpose": ("compare the two EXISTING hinge write policies under the real perspective camera; "
                    "no operator is created, tuned or promoted"),
        "camera_placement": {
            "mode": "oracle_absolute_root_placement",
            "declaration": ("the stored root-relative prediction is placed at the target's absolute "
                            "camera-space pelvis, IDENTICALLY for H0/DEPTH_ONLY/MINIMUM_NORM. This is "
                            "a diagnostic device: AnimCV does not know absolute root depth and this "
                            "is not production inference"),
        },
        "references": {
            "A_projected_target": "projection of the same absolute SMPL joint centre (like-for-like)",
            "B_observation": ("stored 3DPW input_2d in pixels; OpenPose COCO landmark vs SMPL joint "
                              "centre, so its gap is observation-consistency error, NOT detector error"),
        },
        "semantic_identity": semantic,
        "reconstruction": reconstruction_report,
        "provenance": {
            "bank_index": str(args.bank), "bank_content_digest": bank.content_digest(),
            "observation_regime": bank.regime(), "split": args.split, "frames": int(len(positions)),
            "source": identity, "raw_split": args.raw_split, "raw_root": str(args.raw_root),
            "raw_sequences": raw_provenance,
            "frames_with_absolute_geometry": int(usable.sum()),
            "frames_without_absolute_geometry": dict(missing),
            "image_orientations_present": sorted(tuple(v) for v in orientations),
            "per_frame_intrinsics": True,
        },
        "historical_binding": _bind_history(args.historical,
                                            identity["prediction"]["sha256"], totals),
        "population_totals": dict(totals),
        "aggregate_3d": {name: {key: value["aggregate"][key]
                                for key in ("mpjpe_mm", "pa_mpjpe_mm", "hinge_flip_rate",
                                            "hinge_direction_mae_degrees")
                                if value["aggregate"].get(key)}
                         for name, value in aggregate.items()},
        "fields": per_field,
    }
    write_json(args.out / "hinge_write_policy_observation.json", report)

    print(f"\n== hinge write policy vs observation ({len(positions)} {args.split} frames)")
    print("   semantic identity: %d chain-frames, max bend-direction difference %.2e, sign states identical=%s"
          % (semantic["checked_chain_frames"], semantic["bend_direction_max_difference"],
             semantic["sign_state_identical"]))
    print("   reconstruction max %.6f mm (refusal boundary %.1f mm)" % (
        reconstruction_report["max_mm"], RECONSTRUCTION_REFUSAL_MM))
    binding = report["historical_binding"]
    print("   historical binding:", "verifiable, all checks pass = %s" % binding.get("all_checks_pass")
          if binding.get("verifiable") else "UNVERIFIABLE (%s)" % binding.get("reason"))
    print("\n   correction-induced IMAGE displacement from H0 (pixels)")
    print("   %-26s %28s %28s" % ("field", "DEPTH_ONLY p50/p90/max", "MINIMUM_NORM p50/p90/max"))
    for field, entry in per_field.items():
        if not entry.get("usable"):
            continue
        cells = []
        for policy in (DEPTH_ONLY, MINIMUM_NORM):
            q = entry["correction_induced_image_displacement"][policy]["pixel"]
            cells.append("%8.2f /%8.2f /%8.1f" % (q["p50"], q["p90"], q["max"]))
        print("   %-26s %28s %28s" % (field, *cells))
    print("\n   TARGET-PROJECTION error (pixels, mean)      | OBSERVATION-CONSISTENCY error (pixels, mean)")
    print("   %-26s %8s %8s %8s | %8s %8s %8s" % ("field", "H0", "DEPTH", "MINNORM", "H0", "DEPTH", "MINNORM"))
    for field, entry in per_field.items():
        if not entry.get("usable"):
            continue
        a = [entry["target_projection_error"][n]["pixel"]["mean"] for n in ("H0", DEPTH_ONLY, MINIMUM_NORM)]
        b = [entry["observation_consistency_error"][n]["pixel"]["mean"] for n in ("H0", DEPTH_ONLY, MINIMUM_NORM)]
        print("   %-26s %8.2f %8.2f %8.2f | %8.2f %8.2f %8.2f" % (field, *a, *b))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
