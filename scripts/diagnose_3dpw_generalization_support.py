#!/usr/bin/env python3
"""Close the remaining 3DPW generalization ambiguity without training.

The A9 evaluator defines the hard sets.  This diagnostic then keeps target
space (canonical GT geometry) and observation space (canonical 2D lifter
input) strictly separate while comparing sequence-disjoint nearest support.
It also attributes the fixed hard sets to signed bilateral depth residuals
and accounts for 3DPW's replacement sampling at sequence level.

No loss, sampler, augmentation, optimizer, gate, or checkpoint is changed by
this script.  The A12 checkpoint is used only as a confirmation of A9's fixed
hard examples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from pose.pose_lifter import H36M_NAMES
from training.temporal_lifter import (
    TrainingConfig,
    _arrays,
    _augment_inputs,
    _model,
    _normalize_inputs,
    _predict_batched,
    _root_yaw_error_degrees,
    _source_balanced_permutation,
    _torch,
    load_dataset,
)


WINDOW = 81
SEED = 1337
BATCH_SIZE = 128
TRAIN_CONTROL_COUNT = 512
TAILS = (0.05, 0.01)

PELVIS = H36M_NAMES.index("pelvis")
THORAX = H36M_NAMES.index("thorax")
LEFT_SHOULDER = H36M_NAMES.index("left_shoulder")
RIGHT_SHOULDER = H36M_NAMES.index("right_shoulder")
LEFT_HIP = H36M_NAMES.index("left_hip")
RIGHT_HIP = H36M_NAMES.index("right_hip")
PAIR_INDICES = {
    "shoulder": (LEFT_SHOULDER, RIGHT_SHOULDER),
    "hip": (LEFT_HIP, RIGHT_HIP),
}


def _stats(values: Any) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if not len(array):
        return {
            "count": 0, "mean": None, "std": None, "median": None,
            "p05": None, "p10": None, "p90": None, "p95": None, "p99": None,
        }
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "median": float(np.quantile(array, 0.50)),
        "p05": float(np.quantile(array, 0.05)),
        "p10": float(np.quantile(array, 0.10)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
    }


def _wrap_angle(value: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(value) + np.pi) % (2.0 * np.pi) - np.pi


def _angle_difference(first: float, second: float) -> float:
    return float(_wrap_angle(second - first))


def _finite_difference(values: np.ndarray, fps: float, *, angular: bool = False) -> np.ndarray:
    """Sequence-local finite difference; never bridges a missing observation."""
    result = np.full(len(values), np.nan, dtype=np.float64)
    if len(values) == 0 or fps <= 0:
        return result
    step = 1.0 / fps
    for index in range(len(values)):
        previous = index - 1
        following = index + 1
        if previous >= 0 and following < len(values) and np.isfinite(values[previous]) and np.isfinite(values[following]):
            delta = (_angle_difference(float(values[previous]), float(values[following]))
                     if angular else float(values[following] - values[previous]))
            result[index] = delta / (2.0 * step)
        elif following < len(values) and np.isfinite(values[index]) and np.isfinite(values[following]):
            delta = (_angle_difference(float(values[index]), float(values[following]))
                     if angular else float(values[following] - values[index]))
            result[index] = delta / step
        elif previous >= 0 and np.isfinite(values[index]) and np.isfinite(values[previous]):
            delta = (_angle_difference(float(values[previous]), float(values[index]))
                     if angular else float(values[index] - values[previous]))
            result[index] = delta / step
    return result


def _window_offsets(length: int, window: int = WINDOW) -> np.ndarray:
    radius = window // 2
    return np.clip(np.arange(length)[:, None] + np.arange(-radius, radius + 1), 0, max(length - 1, 0))


def _pair_geometry(points: np.ndarray, valid: np.ndarray, left: int, right: int) -> dict[str, np.ndarray]:
    vector = points[:, right] - points[:, left]
    pair_valid = valid[:, left] & valid[:, right]
    vector = vector.astype(np.float64, copy=True)
    vector[~pair_valid] = np.nan
    length = np.linalg.norm(vector, axis=1)
    xy_length = np.linalg.norm(vector[:, :2], axis=1)
    unit_xy = np.full((len(points), 2), np.nan, dtype=np.float64)
    stable = pair_valid & (xy_length > 1e-8)
    unit_xy[stable] = vector[stable, :2] / xy_length[stable, None]
    angle = np.full(len(points), np.nan, dtype=np.float64)
    angle[stable] = np.arctan2(unit_xy[stable, 1], unit_xy[stable, 0])
    return {
        "vector": vector,
        "valid": pair_valid,
        "length": np.where(pair_valid, length, np.nan),
        "xy_length": np.where(stable, xy_length, np.nan),
        "unit_xy": unit_xy,
        "angle": angle,
        # The requested diagnostic calls this z_right-z_left signed depth.
        "signed_z": vector[:, 2],
        # AnimCV's documented physical camera depth/forward axis is +Y.
        "signed_forward_y": vector[:, 1],
    }


def _target_frame_geometry(targets: np.ndarray, valid: np.ndarray) -> dict[str, np.ndarray]:
    shoulder = _pair_geometry(targets, valid, *PAIR_INDICES["shoulder"])
    hip = _pair_geometry(targets, valid, *PAIR_INDICES["hip"])
    axis_sum = np.zeros((len(targets), 2), dtype=np.float64)
    axis_count = np.zeros(len(targets), dtype=np.float64)
    for pair in (shoulder, hip):
        keep = np.isfinite(pair["unit_xy"]).all(axis=1)
        axis_sum[keep] += pair["unit_xy"][keep]
        axis_count[keep] += 1.0
    root_unit = np.full((len(targets), 2), np.nan, dtype=np.float64)
    keep = axis_count > 0
    norm = np.linalg.norm(axis_sum[keep], axis=1)
    stable = keep.copy()
    stable[keep] &= norm > 1e-8
    root_unit[stable] = axis_sum[stable] / np.linalg.norm(axis_sum[stable], axis=1)[:, None]
    orientation = np.full(len(targets), np.nan, dtype=np.float64)
    orientation[stable] = np.arctan2(root_unit[stable, 1], root_unit[stable, 0])
    return {
        "shoulder": shoulder,
        "hip": hip,
        "root_unit": root_unit,
        "root_orientation": orientation,
    }


def _input_frame_geometry(inputs: np.ndarray, raw_inputs: np.ndarray) -> dict[str, np.ndarray]:
    observed = inputs[..., 2] > 0
    raw_observed = raw_inputs[..., 2] > 0
    coordinates_3d = np.zeros((len(inputs), inputs.shape[1], 3), dtype=np.float64)
    coordinates_3d[..., :2] = inputs[..., :2]
    shoulder = _pair_geometry(
        coordinates_3d,
        observed,
        *PAIR_INDICES["shoulder"],
    )
    hip = _pair_geometry(
        coordinates_3d,
        observed,
        *PAIR_INDICES["hip"],
    )
    torso_vector = inputs[:, THORAX, :2] - inputs[:, PELVIS, :2]
    torso_valid = observed[:, THORAX] & observed[:, PELVIS]
    torso_height = np.linalg.norm(torso_vector, axis=1)
    torso_height[~torso_valid] = np.nan
    raw_torso_height = np.linalg.norm(raw_inputs[:, THORAX, :2] - raw_inputs[:, PELVIS, :2], axis=1)
    raw_torso_height[~(raw_observed[:, THORAX] & raw_observed[:, PELVIS])] = np.nan
    observed_count = observed.sum(axis=1).astype(np.float64)
    pair_count = (shoulder["valid"].astype(np.int64) + hip["valid"].astype(np.int64)).astype(np.float64)
    confidence_mean = np.where(observed_count > 0, (inputs[..., 2] * observed).sum(axis=1) / observed_count, np.nan)
    ordering = {}
    for name, pair in (("shoulder", shoulder), ("hip", hip)):
        delta_x = pair["vector"][:, 0]
        ordering[name] = np.where(pair["valid"], np.sign(delta_x), np.nan)
    return {
        "shoulder": shoulder,
        "hip": hip,
        "torso_height": torso_height,
        "raw_torso_height": raw_torso_height,
        "observed_count": observed_count,
        "pair_count": pair_count,
        "confidence_mean": confidence_mean,
        "ordering": ordering,
        "observed": observed,
    }


def _window_change(values: np.ndarray, window_indices: np.ndarray, *, angular: bool = False) -> tuple[np.ndarray, np.ndarray]:
    net = np.full(len(window_indices), np.nan, dtype=np.float64)
    path = np.full(len(window_indices), np.nan, dtype=np.float64)
    for center, row in enumerate(window_indices):
        sequence = values[row]
        finite = np.flatnonzero(np.isfinite(sequence))
        if not len(finite):
            continue
        first, last = int(finite[0]), int(finite[-1])
        net[center] = (_angle_difference(float(sequence[first]), float(sequence[last]))
                       if angular else float(sequence[last] - sequence[first]))
        if last == first:
            path[center] = 0.0
            continue
        total = 0.0
        segments = 0
        for left, right in zip(range(first, last), range(first + 1, last + 1)):
            if np.isfinite(sequence[left]) and np.isfinite(sequence[right]):
                total += abs(_angle_difference(float(sequence[left]), float(sequence[right]))
                             if angular else float(sequence[right] - sequence[left]))
                segments += 1
        path[center] = total if segments else np.nan
    return net, path


def _sign_transition_count(values: np.ndarray) -> float:
    previous = None
    count = 0
    for value in values:
        if not np.isfinite(value) or value == 0:
            continue
        current = 1 if value > 0 else -1
        if previous is not None and current != previous:
            count += 1
        previous = current
    return float(count)


def _longest_signed_run(values: np.ndarray) -> float:
    previous = None
    longest = 0
    current_length = 0
    for value in values:
        if not np.isfinite(value) or value == 0:
            previous = None
            current_length = 0
            continue
        current = 1 if value > 0 else -1
        if current == previous:
            current_length += 1
        else:
            previous = current
            current_length = 1
        longest = max(longest, current_length)
    return float(longest)


def _target_temporal_geometry(frame_geometry: dict[str, np.ndarray], fps: float, window_indices: np.ndarray) -> dict[str, np.ndarray]:
    orientation = frame_geometry["root_orientation"]
    shoulder_z = frame_geometry["shoulder"]["signed_z"]
    hip_z = frame_geometry["hip"]["signed_z"]
    shoulder_y = frame_geometry["shoulder"]["signed_forward_y"]
    hip_y = frame_geometry["hip"]["signed_forward_y"]
    orientation_velocity = _finite_difference(orientation, fps, angular=True)
    orientation_acceleration = _finite_difference(orientation_velocity, fps)
    shoulder_z_velocity = _finite_difference(shoulder_z, fps)
    hip_z_velocity = _finite_difference(hip_z, fps)
    shoulder_y_velocity = _finite_difference(shoulder_y, fps)
    hip_y_velocity = _finite_difference(hip_y, fps)
    orientation_net, orientation_path = _window_change(orientation, window_indices, angular=True)
    shoulder_z_net, _ = _window_change(shoulder_z, window_indices)
    hip_z_net, _ = _window_change(hip_z, window_indices)
    shoulder_y_net, _ = _window_change(shoulder_y, window_indices)
    hip_y_net, _ = _window_change(hip_y, window_indices)
    longest_run = np.asarray([_longest_signed_run(orientation_velocity[row]) for row in window_indices], dtype=np.float64)
    shoulder_transitions = np.asarray([_sign_transition_count(shoulder_z[row]) for row in window_indices], dtype=np.float64)
    hip_transitions = np.asarray([_sign_transition_count(hip_z[row]) for row in window_indices], dtype=np.float64)
    return {
        "orientation_velocity_rad_s": orientation_velocity,
        "orientation_acceleration_rad_s2": orientation_acceleration,
        "orientation_window_net_rad": orientation_net,
        "orientation_window_path_rad": orientation_path,
        "orientation_longest_signed_run_frames": longest_run,
        "shoulder_signed_z_velocity_m_s": shoulder_z_velocity,
        "hip_signed_z_velocity_m_s": hip_z_velocity,
        "shoulder_signed_z_window_net_m": shoulder_z_net,
        "hip_signed_z_window_net_m": hip_z_net,
        "shoulder_forward_y_velocity_m_s": shoulder_y_velocity,
        "hip_forward_y_velocity_m_s": hip_y_velocity,
        "shoulder_forward_y_window_net_m": shoulder_y_net,
        "hip_forward_y_window_net_m": hip_y_net,
        "shoulder_signed_z_sign_transitions_per_window": shoulder_transitions,
        "hip_signed_z_sign_transitions_per_window": hip_transitions,
    }


def _coordinate_difference(values: np.ndarray, observed: np.ndarray, fps: float) -> np.ndarray:
    result = np.full_like(values, np.nan, dtype=np.float64)
    for joint in range(values.shape[1]):
        for axis in range(values.shape[2]):
            result[:, joint, axis] = _finite_difference(values[:, joint, axis], fps)
            for index in range(len(values)):
                if not observed[index, joint]:
                    result[index, joint, axis] = np.nan
                elif index > 0 and index + 1 < len(values):
                    if not (observed[index - 1, joint] and observed[index + 1, joint]):
                        # The centered finite difference would otherwise
                        # interpret a missing landmark's zero fill as motion.
                        result[index, joint, axis] = np.nan
                elif index == 0 and len(values) > 1 and not observed[1, joint]:
                    result[index, joint, axis] = np.nan
                elif index == len(values) - 1 and len(values) > 1 and not observed[index - 1, joint]:
                    result[index, joint, axis] = np.nan
    return result


def _input_temporal_geometry(frame_geometry: dict[str, np.ndarray], inputs: np.ndarray, fps: float,
                             window_indices: np.ndarray) -> dict[str, np.ndarray]:
    coordinates = inputs[..., :2].astype(np.float64)
    velocity = _coordinate_difference(coordinates, frame_geometry["observed"], fps)
    speed = np.linalg.norm(velocity, axis=-1)
    speed[~frame_geometry["observed"]] = np.nan
    mean_speed = np.nanmean(speed, axis=1)
    mean_speed[~np.isfinite(mean_speed)] = np.nan
    shoulder_speed = _finite_difference(frame_geometry["shoulder"]["xy_length"], fps)
    hip_speed = _finite_difference(frame_geometry["hip"]["xy_length"], fps)
    torso_speed = _finite_difference(frame_geometry["torso_height"], fps)
    window_mean_speed = np.full(len(window_indices), np.nan, dtype=np.float64)
    mean_abs_velocity = np.full((len(window_indices), coordinates.shape[1], coordinates.shape[2]), np.nan, dtype=np.float64)
    net_displacement = np.full_like(mean_abs_velocity, np.nan)
    for center, row in enumerate(window_indices):
        window_speed = speed[row]
        if np.isfinite(window_speed).any():
            window_mean_speed[center] = float(np.nanmean(window_speed))
        window_velocity = velocity[row]
        mean_abs_velocity[center] = np.nanmean(np.abs(window_velocity), axis=0)
        mean_abs_velocity[center] = np.nan_to_num(mean_abs_velocity[center], nan=0.0)
        first, last = row[0], row[-1]
        net_displacement[center] = coordinates[last] - coordinates[first]
    return {
        "joint_velocity": velocity,
        "mean_joint_speed_normalized_s": mean_speed,
        "shoulder_span_velocity_normalized_s": shoulder_speed,
        "hip_span_velocity_normalized_s": hip_speed,
        "torso_height_velocity_normalized_s": torso_speed,
        "window_mean_joint_speed_normalized_s": window_mean_speed,
        "mean_abs_joint_velocity_normalized_s": mean_abs_velocity,
        "window_net_joint_displacement_normalized": net_displacement,
    }


def _target_descriptor(frame_geometry: dict[str, np.ndarray], temporal: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    shoulder, hip = frame_geometry["shoulder"], frame_geometry["hip"]
    root = frame_geometry["root_unit"]
    columns = [
        root,
        shoulder["unit_xy"], hip["unit_xy"],
        shoulder["signed_z"][:, None], hip["signed_z"][:, None],
        shoulder["signed_forward_y"][:, None], hip["signed_forward_y"][:, None],
        shoulder["length"][:, None], hip["length"][:, None],
    ]
    temporal_names = (
        "orientation_velocity_rad_s", "orientation_acceleration_rad_s2",
        "orientation_window_net_rad", "orientation_window_path_rad",
        "orientation_longest_signed_run_frames", "shoulder_signed_z_velocity_m_s",
        "hip_signed_z_velocity_m_s", "shoulder_signed_z_window_net_m",
        "hip_signed_z_window_net_m", "shoulder_forward_y_velocity_m_s",
        "hip_forward_y_velocity_m_s", "shoulder_forward_y_window_net_m",
        "hip_forward_y_window_net_m", "shoulder_signed_z_sign_transitions_per_window",
        "hip_signed_z_sign_transitions_per_window",
    )
    columns.extend(temporal[name][:, None] for name in temporal_names)
    descriptor = np.concatenate(columns, axis=1)
    valid = np.isfinite(descriptor).all(axis=1)
    return descriptor, valid


def _input_descriptor(frame_geometry: dict[str, np.ndarray], inputs: np.ndarray,
                      temporal: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    current_coordinates = inputs[..., :2].astype(np.float64).reshape(len(inputs), -1)
    confidence = inputs[..., 2].astype(np.float64)
    mean_abs_velocity = temporal["mean_abs_joint_velocity_normalized_s"].reshape(len(inputs), -1)
    net_displacement = temporal["window_net_joint_displacement_normalized"].reshape(len(inputs), -1)
    descriptor = np.concatenate((current_coordinates, confidence, mean_abs_velocity, net_displacement), axis=1)
    valid = np.isfinite(descriptor).all(axis=1)
    return descriptor, valid


def _flatten_metadata(dataset: dict[str, Any]) -> tuple[list[dict[str, Any]], list[tuple[int, int]], np.ndarray, np.ndarray, np.ndarray]:
    sequences = dataset.get("sequences", [{"frames": dataset["frames"]}])
    metadata: list[dict[str, Any]] = []
    ranges: list[tuple[int, int]] = []
    sequence_indices: list[int] = []
    local_indices: list[int] = []
    fps_values: list[float] = []
    cursor = 0
    default_fps = float(dataset.get("source_fps") or 30.0)
    for sequence_index, sequence in enumerate(sequences):
        frames = sequence["frames"]
        begin = cursor
        sequence_id = str(sequence.get("sequence_id") or f"sequence_{sequence_index}")
        fps = float(sequence.get("source_fps") or default_fps or 30.0)
        source = {**dataset.get("source", {}), **sequence.get("source", {})}
        for local_index, frame in enumerate(frames):
            frame_index = int(frame.get("frame_index", local_index))
            metadata.append({
                "sequence_id": sequence_id,
                "sequence_index": sequence_index,
                "local_index": local_index,
                "frame_index": frame_index,
                "source_fps": fps,
                "view": source.get("view") or source.get("camera"),
                "action": source.get("action") or source.get("motion"),
            })
            sequence_indices.append(sequence_index)
            local_indices.append(local_index)
            fps_values.append(fps)
        cursor += len(frames)
        ranges.append((begin, cursor))
    return metadata, ranges, np.asarray(sequence_indices, dtype=np.int64), np.asarray(local_indices, dtype=np.int64), np.asarray(fps_values, dtype=np.float64)


def _build_split(name: str, path: Path, dataset: dict[str, Any], normalization: str, window: int) -> dict[str, Any]:
    metadata, ranges, sequence_indices, local_indices, fps_values = _flatten_metadata(dataset)
    normalized_inputs, targets, valid, offsets = _arrays(
        dataset, window, include_metadata=False, coordinate_normalization=normalization,
    )
    raw_inputs = np.asarray([frame["input_2d"] for sequence in dataset.get("sequences", [{"frames": dataset["frames"]}]) for frame in sequence["frames"]], dtype=np.float64)
    raw_inputs = raw_inputs.astype(np.float64)
    normalized_inputs = normalized_inputs.astype(np.float64)
    targets = targets.astype(np.float64)
    valid = valid.astype(bool)
    target_geometry = _target_frame_geometry(targets, valid)
    input_geometry = _input_frame_geometry(normalized_inputs, raw_inputs)
    target_temporal: dict[str, np.ndarray] = {key: np.full(len(targets), np.nan, dtype=np.float64)
                                               for key in _target_temporal_geometry(target_geometry, 30.0, _window_offsets(1, window)).keys()}
    input_temporal: dict[str, np.ndarray] = {
        "joint_velocity": np.full((len(targets), len(H36M_NAMES), 2), np.nan),
        "mean_joint_speed_normalized_s": np.full(len(targets), np.nan),
        "shoulder_span_velocity_normalized_s": np.full(len(targets), np.nan),
        "hip_span_velocity_normalized_s": np.full(len(targets), np.nan),
        "torso_height_velocity_normalized_s": np.full(len(targets), np.nan),
        "window_mean_joint_speed_normalized_s": np.full(len(targets), np.nan),
        "mean_abs_joint_velocity_normalized_s": np.full((len(targets), len(H36M_NAMES), 2), np.nan),
        "window_net_joint_displacement_normalized": np.full((len(targets), len(H36M_NAMES), 2), np.nan),
    }
    # root unit (2) + pair units (4) + signed axes (4) + pair lengths (2)
    # + temporal target features (15).
    target_descriptor = np.full((len(targets), 2 + 4 + 4 + 2 + 15), np.nan, dtype=np.float64)
    input_descriptor = np.full((len(targets), 34 + 17 + 34 + 34), np.nan, dtype=np.float64)
    target_descriptor_valid = np.zeros(len(targets), dtype=bool)
    input_descriptor_valid = np.zeros(len(targets), dtype=bool)
    for begin, end in ranges:
        if end <= begin:
            continue
        sequence_window = _window_offsets(end - begin, window)
        sequence_target_geometry = {
            "shoulder": {key: value[begin:end] for key, value in target_geometry["shoulder"].items()},
            "hip": {key: value[begin:end] for key, value in target_geometry["hip"].items()},
            "root_unit": target_geometry["root_unit"][begin:end],
            "root_orientation": target_geometry["root_orientation"][begin:end],
        }
        sequence_target_temporal = _target_temporal_geometry(sequence_target_geometry, float(fps_values[begin]), sequence_window)
        sequence_input_geometry = {
            "shoulder": {key: value[begin:end] for key, value in input_geometry["shoulder"].items()},
            "hip": {key: value[begin:end] for key, value in input_geometry["hip"].items()},
            "torso_height": input_geometry["torso_height"][begin:end],
            "raw_torso_height": input_geometry["raw_torso_height"][begin:end],
            "observed_count": input_geometry["observed_count"][begin:end],
            "pair_count": input_geometry["pair_count"][begin:end],
            "confidence_mean": input_geometry["confidence_mean"][begin:end],
            "ordering": {key: value[begin:end] for key, value in input_geometry["ordering"].items()},
            "observed": input_geometry["observed"][begin:end],
        }
        sequence_inputs = normalized_inputs[begin:end]
        sequence_input_temporal = _input_temporal_geometry(sequence_input_geometry, sequence_inputs, float(fps_values[begin]), sequence_window)
        target_local_descriptor, target_local_valid = _target_descriptor(sequence_target_geometry, sequence_target_temporal)
        input_local_descriptor, input_local_valid = _input_descriptor(sequence_input_geometry, sequence_inputs, sequence_input_temporal)
        target_descriptor[begin:end] = target_local_descriptor
        input_descriptor[begin:end] = input_local_descriptor
        target_descriptor_valid[begin:end] = target_local_valid
        input_descriptor_valid[begin:end] = input_local_valid
        for key, value in sequence_target_temporal.items():
            target_temporal[key][begin:end] = value
        for key, value in sequence_input_temporal.items():
            input_temporal[key][begin:end] = value
    return {
        "name": name,
        "path": path,
        "dataset": dataset,
        "metadata": metadata,
        "ranges": ranges,
        "sequence_indices": sequence_indices,
        "local_indices": local_indices,
        "fps": fps_values,
        "inputs": normalized_inputs,
        "raw_inputs": raw_inputs,
        "targets": targets,
        "valid": valid,
        "offsets": offsets,
        "target_geometry": target_geometry,
        "input_geometry": input_geometry,
        "target_temporal": target_temporal,
        "input_temporal": input_temporal,
        "target_descriptor": target_descriptor,
        "target_descriptor_valid": target_descriptor_valid,
        "input_descriptor": input_descriptor,
        "input_descriptor_valid": input_descriptor_valid,
        "sequence_keys": np.asarray([item["sequence_id"] for item in metadata], dtype=object),
    }


def _predict_split(torch, nn, checkpoint: dict[str, Any], split: dict[str, Any], device: str) -> np.ndarray:
    model = _model(nn, int(checkpoint["channels"]), checkpoint.get("architecture", "legacy_tcn_v1")).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    x = torch.as_tensor(split["inputs"], dtype=torch.float32, device=device)
    offsets = torch.as_tensor(split["offsets"], dtype=torch.long, device=device)
    with torch.no_grad():
        prediction = _predict_batched(model, x, offsets, 1024, str(device).startswith("cuda"))
    return prediction.cpu().numpy().astype(np.float64)


def _yaw_values(prediction: np.ndarray, targets: np.ndarray, valid: np.ndarray) -> np.ndarray:
    values = np.full(len(prediction), np.nan, dtype=np.float64)
    for index, (estimate, target, frame_valid) in enumerate(zip(prediction, targets, valid)):
        value = _root_yaw_error_degrees(estimate, target, frame_valid)
        if value is not None:
            values[index] = float(value)
    return values


def _record(split: dict[str, Any], index: int, rank: int | None = None, yaw_a9: float | None = None,
            yaw_a12: float | None = None) -> dict[str, Any]:
    meta = split["metadata"][index]
    sequence_index = int(meta["sequence_index"])
    local_index = int(meta["local_index"])
    sequence_length = split["ranges"][sequence_index][1] - split["ranges"][sequence_index][0]
    radius = WINDOW // 2
    window_indices = np.clip(np.arange(local_index - radius, local_index + radius + 1), 0, sequence_length - 1)
    local_frame_ids = [split["metadata"][split["ranges"][sequence_index][0] + int(item)]["frame_index"] for item in window_indices]
    result = {
        "split": split["name"],
        "sequence_id": meta["sequence_id"],
        "sequence_index": sequence_index,
        "local_index": local_index,
        "frame_id": int(meta["frame_index"]),
        "window_center_frame_id": int(meta["frame_index"]),
        "window_start_frame_id": int(local_frame_ids[0]),
        "window_end_frame_id": int(local_frame_ids[-1]),
        "window_frame_ids": local_frame_ids,
    }
    if rank is not None:
        result["rank"] = rank
    if yaw_a9 is not None:
        result["a9_yaw_error_degrees"] = float(yaw_a9)
    if yaw_a12 is not None:
        result["a12_yaw_error_degrees"] = float(yaw_a12)
    return result


def _select_tail(split: dict[str, Any], yaw_a9: np.ndarray, yaw_a12: np.ndarray, fraction: float) -> dict[str, Any]:
    finite = np.flatnonzero(np.isfinite(yaw_a9))
    order = finite[np.lexsort((finite, -yaw_a9[finite]))]
    count = max(1, int(np.ceil(len(order) * fraction))) if len(order) else 0
    selected = order[:count]
    records = [_record(split, int(index), rank=rank + 1, yaw_a9=yaw_a9[index], yaw_a12=yaw_a12[index])
               for rank, index in enumerate(selected)]
    return {
        "fraction": fraction,
        "eligible_frame_count": int(len(order)),
        "selected_count": int(count),
        "minimum_selected_a9_yaw_error_degrees": float(yaw_a9[selected[-1]]) if len(selected) else None,
        "records": records,
        "indices": selected,
    }


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def _spearman(first: np.ndarray, second: np.ndarray) -> float | None:
    finite = np.isfinite(first) & np.isfinite(second)
    if finite.sum() < 2:
        return None
    first_rank, second_rank = _rankdata(first[finite]), _rankdata(second[finite])
    if np.std(first_rank) <= 1e-12 or np.std(second_rank) <= 1e-12:
        return None
    return float(np.corrcoef(first_rank, second_rank)[0, 1])


def _tail_keys(tail: dict[str, Any]) -> set[tuple[str, int]]:
    return {(record["sequence_id"], int(record["local_index"])) for record in tail["records"]}


def _window_coverage_keys(split: dict[str, Any], tail: dict[str, Any]) -> set[tuple[str, int]]:
    keys: set[tuple[str, int]] = set()
    for record in tail["records"]:
        sequence_index = int(record["sequence_index"])
        begin, end = split["ranges"][sequence_index]
        for local_index in range(max(0, int(record["local_index"]) - WINDOW // 2),
                                 min(end - begin, int(record["local_index"]) + WINDOW // 2 + 1)):
            keys.add((record["sequence_id"], local_index))
    return keys


def _jaccard(first: set[Any], second: set[Any]) -> float:
    union = first | second
    return len(first & second) / len(union) if union else 1.0


def _sequence_concentration(split: dict[str, Any], tail: dict[str, Any]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for record in tail["records"]:
        counts[record["sequence_id"]] = counts.get(record["sequence_id"], 0) + 1
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    total = sum(counts.values())
    return {
        "unique_sequences": len(counts),
        "top_sequence_frame_counts": dict(ordered[:10]),
        "top_sequence_share": ordered[0][1] / total if ordered else 0.0,
        "top_5_sequence_share": sum(value for _, value in ordered[:5]) / total if ordered else 0.0,
        "sequence_hhi": sum((value / total) ** 2 for value in counts.values()) if total else 0.0,
    }


def _hard_case_overlap(split: dict[str, Any], tails: dict[str, dict[str, Any]], yaw_a9: np.ndarray,
                       yaw_a12: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {"all_valid_frame_spearman_yaw_error": _spearman(yaw_a9, yaw_a12)}
    for label, tail in tails.items():
        a9_keys = _tail_keys(tail)
        # A12 selects the same fixed fraction only for overlap measurement;
        # the A9 set remains the definition of hard cases everywhere else.
        order = np.flatnonzero(np.isfinite(yaw_a12))
        order = order[np.lexsort((order, -yaw_a12[order]))]
        count = min(len(order), int(tail["selected_count"]))
        a12_indices = order[:count]
        a12_keys = {(split["metadata"][int(index)]["sequence_id"], int(split["metadata"][int(index)]["local_index"])) for index in a12_indices}
        a9_windows = _window_coverage_keys(split, tail)
        a12_tail = {"records": [_record(split, int(index), yaw_a9=yaw_a9[index], yaw_a12=yaw_a12[index]) for index in a12_indices]}
        a12_windows = _window_coverage_keys(split, a12_tail)
        result[label] = {
            "a9_selected_count": int(len(a9_keys)),
            "a12_selected_count": int(len(a12_keys)),
            "frame_center_overlap_count": int(len(a9_keys & a12_keys)),
            "frame_center_overlap_rate_of_a9": len(a9_keys & a12_keys) / len(a9_keys) if a9_keys else 0.0,
            "frame_center_jaccard": _jaccard(a9_keys, a12_keys),
            "temporal_window_frame_coverage_jaccard": _jaccard(a9_windows, a12_windows),
            "a9_sequence_concentration": _sequence_concentration(split, tail),
            "a12_sequence_concentration": _sequence_concentration(split, a12_tail),
        }
    return result


def _fit_scaler(descriptor: np.ndarray, valid: np.ndarray) -> dict[str, np.ndarray]:
    values = descriptor[valid]
    if not len(values):
        raise ValueError("cannot fit support scaler with no valid train descriptors")
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return {"mean": mean, "scale": scale}


def _nearest_support(query: np.ndarray, query_valid: np.ndarray, query_sequences: np.ndarray,
                    support: np.ndarray, support_valid: np.ndarray, support_sequences: np.ndarray,
                    scaler: dict[str, np.ndarray], query_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    query_count = len(query_indices)
    nearest_distance = np.full(query_count, np.nan, dtype=np.float64)
    nearest_index = np.full(query_count, -1, dtype=np.int64)
    support_indices = np.flatnonzero(support_valid)
    if not len(support_indices):
        return nearest_distance, nearest_index
    normalized_support = ((support[support_indices] - scaler["mean"]) / scaler["scale"]).astype(np.float32)
    for begin in range(0, query_count, 32):
        end = min(query_count, begin + 32)
        indices = query_indices[begin:end]
        valid_queries = query_valid[indices]
        if not valid_queries.any():
            continue
        normalized_query = ((query[indices] - scaler["mean"]) / scaler["scale"]).astype(np.float32)
        best_distance = np.full(end - begin, np.inf, dtype=np.float32)
        best_support = np.full(end - begin, -1, dtype=np.int64)
        for support_begin in range(0, len(support_indices), 2048):
            support_end = min(len(support_indices), support_begin + 2048)
            support_chunk = normalized_support[support_begin:support_end]
            difference = normalized_query[:, None, :] - support_chunk[None, :, :]
            distances = np.sqrt(np.mean(difference * difference, axis=2))
            allowed = support_sequences[support_indices[support_begin:support_end]][None, :] != query_sequences[indices][:, None]
            distances[~allowed] = np.inf
            local = np.argmin(distances, axis=1)
            local_distance = distances[np.arange(len(local)), local]
            improve = local_distance < best_distance
            best_distance[improve] = local_distance[improve]
            best_support[improve] = support_indices[support_begin:support_end][local[improve]]
        best_distance[~valid_queries] = np.nan
        best_support[~valid_queries] = -1
        best_distance[~np.isfinite(best_distance)] = np.nan
        nearest_distance[begin:end] = best_distance
        nearest_index[begin:end] = best_support
    return nearest_distance, nearest_index


def _support_records(query_split: dict[str, Any], query_indices: np.ndarray, support_split: dict[str, Any],
                     target_scaler: dict[str, np.ndarray], input_scaler: dict[str, np.ndarray],
                     relation: str) -> dict[str, Any]:
    target_distance, target_index = _nearest_support(
        query_split["target_descriptor"], query_split["target_descriptor_valid"], query_split["sequence_keys"],
        support_split["target_descriptor"], support_split["target_descriptor_valid"], support_split["sequence_keys"],
        target_scaler, query_indices,
    )
    input_distance, input_index = _nearest_support(
        query_split["input_descriptor"], query_split["input_descriptor_valid"], query_split["sequence_keys"],
        support_split["input_descriptor"], support_split["input_descriptor_valid"], support_split["sequence_keys"],
        input_scaler, query_indices,
    )
    controls: dict[str, Any] = {
        "relation": relation,
        "query_count": int(len(query_indices)),
        "target_valid_query_count": int(np.isfinite(target_distance).sum()),
        "input_valid_query_count": int(np.isfinite(input_distance).sum()),
        "target_distance": target_distance,
        "input_distance": input_distance,
        "target_index": target_index,
        "input_index": input_index,
    }
    return controls


def _distance_summary(values: np.ndarray, control_values: np.ndarray | None = None) -> dict[str, Any]:
    result = {"distance": _stats(values)}
    finite = values[np.isfinite(values)]
    if control_values is not None:
        controls = control_values[np.isfinite(control_values)]
        if len(controls):
            percentiles = np.asarray([(controls <= value).mean() * 100.0 for value in finite], dtype=np.float64)
            result["empirical_percentile_against_train_control"] = _stats(percentiles)
    return result


def _target_relation_values(split: dict[str, Any], index: int) -> dict[str, float]:
    geometry = split["target_geometry"]
    values: dict[str, float] = {}
    orientation = geometry["root_orientation"][index]
    if np.isfinite(orientation):
        values["root_orientation_rad"] = float(orientation)
    for pair_name in ("shoulder", "hip"):
        pair = geometry[pair_name]
        for key, output_name in (("angle", "orientation_rad"), ("signed_z", "signed_z_m"),
                                 ("signed_forward_y", "signed_forward_y_m")):
            value = pair[key][index]
            if np.isfinite(value):
                values[f"{pair_name}_{output_name}"] = float(value)
    return values


def _target_gap_records(query_split: dict[str, Any], query_indices: np.ndarray, support_split: dict[str, Any],
                        support_indices: np.ndarray) -> list[dict[str, float]]:
    records: list[dict[str, float]] = []
    for query_index, support_index in zip(query_indices, support_indices):
        if int(support_index) < 0:
            records.append({})
            continue
        query_values = _target_relation_values(query_split, int(query_index))
        support_values = _target_relation_values(support_split, int(support_index))
        gap: dict[str, float] = {}
        for key, value in query_values.items():
            if key not in support_values:
                continue
            if key == "root_orientation_rad" or key.endswith("orientation_rad"):
                gap[key.replace("_rad", "_abs_delta_degrees")] = abs(np.degrees(_angle_difference(value, support_values[key])))
            else:
                gap[key.replace("_m", "_abs_delta_m")] = abs(value - support_values[key])
        records.append(gap)
    return records


def _gap_summary(records: list[dict[str, float]], control_records: list[dict[str, float]] | None = None) -> dict[str, Any]:
    names = sorted({name for record in records for name in record})
    result: dict[str, Any] = {}
    for name in names:
        values = np.asarray([record[name] for record in records if name in record], dtype=np.float64)
        control_values = None
        if control_records is not None:
            control_values = np.asarray([record[name] for record in control_records if name in record], dtype=np.float64)
        result[name] = _distance_summary(values, control_values)
    return result


def _support_report(train: dict[str, Any], validation: dict[str, Any], test: dict[str, Any],
                    hard_sets: dict[str, dict[str, dict[str, Any]]], seed: int) -> dict[str, Any]:
    target_scaler = _fit_scaler(train["target_descriptor"], train["target_descriptor_valid"])
    input_scaler = _fit_scaler(train["input_descriptor"], train["input_descriptor_valid"])
    rng = np.random.default_rng(seed)
    train_valid = np.flatnonzero(train["target_descriptor_valid"] & train["input_descriptor_valid"])
    control_indices = np.sort(rng.choice(train_valid, size=min(TRAIN_CONTROL_COUNT, len(train_valid)), replace=False))
    relations = {
        "train_to_other_train_sequence": _support_records(train, control_indices, train, target_scaler, input_scaler, "train_to_other_train_sequence"),
        "validation_to_train": _support_records(validation, hard_sets["validation"]["top_5_percent"]["indices"], train, target_scaler, input_scaler, "validation_to_train"),
        "test_to_train": _support_records(test, hard_sets["test"]["top_5_percent"]["indices"], train, target_scaler, input_scaler, "test_to_train"),
    }
    control_target = relations["train_to_other_train_sequence"]["target_distance"]
    control_input = relations["train_to_other_train_sequence"]["input_distance"]
    query_context = {
        "train_to_other_train_sequence": (train, control_indices),
        "validation_to_train": (validation, hard_sets["validation"]["top_5_percent"]["indices"]),
        "test_to_train": (test, hard_sets["test"]["top_5_percent"]["indices"]),
    }
    control_input_target_gap = _target_gap_records(
        train, control_indices, train, relations["train_to_other_train_sequence"]["input_index"],
    )
    control_target_input_gap = _target_gap_records(
        train, control_indices, train, relations["train_to_other_train_sequence"]["target_index"],
    )
    report: dict[str, Any] = {
        "descriptor_scaling": {
            "method": "training-support mean/std; no split-specific tuning",
            "target_feature_count": int(len(target_scaler["mean"])),
            "input_feature_count": int(len(input_scaler["mean"])),
            "target_train_valid_descriptor_count": int(train["target_descriptor_valid"].sum()),
            "input_train_valid_descriptor_count": int(train["input_descriptor_valid"].sum()),
        },
        "relations": {},
    }
    for name, relation in relations.items():
        target_summary = _distance_summary(relation["target_distance"], control_target if name != "train_to_other_train_sequence" else None)
        input_summary = _distance_summary(relation["input_distance"], control_input if name != "train_to_other_train_sequence" else None)
        query_split, query_indices = query_context[name]
        input_target_gap_records = _target_gap_records(
            query_split, query_indices, train, relation["input_index"],
        )
        target_input_gap_records = _target_gap_records(
            query_split, query_indices, train, relation["target_index"],
        )
        records: list[dict[str, Any]] = []
        for position, query_index in enumerate(query_indices):
            query_index = int(query_index)
            item = _record(query_split, query_index)
            item["target_support_distance"] = float(relation["target_distance"][position]) if np.isfinite(relation["target_distance"][position]) else None
            item["input_support_distance"] = float(relation["input_distance"][position]) if np.isfinite(relation["input_distance"][position]) else None
            item["target_gap_at_input_nearest_support"] = input_target_gap_records[position]
            item["target_gap_at_target_nearest_support"] = target_input_gap_records[position]
            for space, index_key in (("target", "target_index"), ("input", "input_index")):
                support_index = int(relation[index_key][position])
                item[f"{space}_support"] = None
                if support_index >= 0:
                    item[f"{space}_support"] = {
                        "sequence_id": train["metadata"][support_index]["sequence_id"],
                        "local_index": int(train["metadata"][support_index]["local_index"]),
                        "frame_id": int(train["metadata"][support_index]["frame_index"]),
                    }
            records.append(item)
        report["relations"][name] = {
            "query_count": int(relation["query_count"]),
            "target": target_summary,
            "input": input_summary,
            "input_nearest_target_gap": _gap_summary(
                input_target_gap_records,
                None if name == "train_to_other_train_sequence" else control_input_target_gap,
            ),
            "target_nearest_target_gap": _gap_summary(
                target_input_gap_records,
                None if name == "train_to_other_train_sequence" else control_target_input_gap,
            ),
            "records": records,
        }
    return report


def _replacement_accounting(torch, direct_mix: dict[str, Any], path: Path, normalization: str, seed: int) -> dict[str, Any]:
    normalized_inputs, _targets, _valid, _offsets, source_ids, sequence_ranges = _arrays(
        direct_mix, WINDOW, include_metadata=True, coordinate_normalization=normalization,
    )
    metadata, _ranges, _sequence_indices, _local_indices, _fps = _flatten_metadata(direct_mix)
    source_labels: list[str] = []
    for sequence in direct_mix.get("sequences", [{"frames": direct_mix["frames"]}]):
        source = {**direct_mix.get("source", {}), **sequence.get("source", {})}
        label = str(source.get("dataset", "unknown"))
        if label not in source_labels:
            source_labels.append(label)
    config = TrainingConfig(
        window=WINDOW, channels=256, epochs=1, batch_size=BATCH_SIZE, seed=seed,
        source_balanced_sampling=True, input_jitter_std=0.015,
        input_dropout_probability=0.05, confidence_jitter_std=0.08,
        input_global_scale_std=0.04, input_translation_std=0.03,
        input_rotation_degrees=12.0, temporal_occlusion_probability=0.10,
        temporal_occlusion_frames=9, input_coordinate_normalization=normalization,
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    # Match train(): augmentation consumes the seeded generator before sampling.
    _augment_inputs(torch, torch.as_tensor(normalized_inputs, dtype=torch.float32), config, generator, sequence_ranges)
    indices = torch.arange(len(source_ids), dtype=torch.long)
    permutation = _source_balanced_permutation(torch, indices, torch.as_tensor(source_ids), generator).numpy()
    source_ids_np = np.asarray(source_ids)
    three_id = source_labels.index("3DPW") if "3DPW" in source_labels else None
    if three_id is None:
        return {"dataset": _fingerprint(path), "source_available": False}
    selected = permutation[source_ids_np[permutation] == three_id]
    sequence_counts: dict[str, int] = {}
    for index in selected:
        sequence_id = metadata[int(index)]["sequence_id"]
        sequence_counts[sequence_id] = sequence_counts.get(sequence_id, 0) + 1
    ordered = sorted(sequence_counts.items(), key=lambda item: (-item[1], item[0]))
    unique_source_windows = int((source_ids_np == three_id).sum())
    unique_sampled = len(set(int(index) for index in selected))
    mass = int(len(selected))
    return {
        "dataset": _fingerprint(path),
        "source": "3DPW",
        "unique_train_sequences": len(sequence_counts),
        "unique_train_windows": unique_source_windows,
        "sampled_epoch_mass": mass,
        "unique_sampled_windows": unique_sampled,
        "duplicate_sample_count": mass - unique_sampled,
        "nominal_replay_factor_mass_over_unique_windows": mass / unique_source_windows if unique_source_windows else None,
        "realized_replay_factor_mass_over_unique_sampled_windows": mass / unique_sampled if unique_sampled else None,
        "sequence_sample_counts": dict(ordered),
        "top_sequence_share": ordered[0][1] / mass if ordered else 0.0,
        "top_5_sequence_share": sum(value for _, value in ordered[:5]) / mass if ordered else 0.0,
        "sequence_hhi": sum((value / mass) ** 2 for value in sequence_counts.values()) if mass else 0.0,
        "source_labels": source_labels,
    }


def _signed_depth_report(split: dict[str, Any], predictions: dict[str, np.ndarray], hard: dict[str, Any]) -> dict[str, Any]:
    hard_indices = set(int(index) for index in hard["top_5_percent"]["indices"])
    valid_yaw = np.isfinite(predictions["a9_yaw"])
    subsets = {
        "a9_top_5_percent_hard": np.asarray([index in hard_indices for index in range(len(split["targets"]))]) & valid_yaw,
        "non_hard": np.asarray([index not in hard_indices for index in range(len(split["targets"]))]) & valid_yaw,
    }
    result: dict[str, Any] = {
        "coordinate_note": "The directive's z_right-z_left is reported verbatim as signed_z. AnimCV's documented camera-depth/forward axis is +Y, so signed_forward_y is reported alongside it.",
        "states": {},
    }
    for state, prediction in (("a9", predictions["a9"]), ("a12", predictions["a12"])):
        state_report: dict[str, Any] = {}
        pred_geometry = _target_frame_geometry(prediction, split["valid"])
        for subset_name, subset in subsets.items():
            subset_report: dict[str, Any] = {}
            for pair_name in ("shoulder", "hip"):
                pair_report: dict[str, Any] = {}
                for axis_name, key in (("signed_z", "signed_z"), ("signed_forward_y", "signed_forward_y")):
                    gt_values = split["target_geometry"][pair_name][key]
                    pred_values = pred_geometry[pair_name][key]
                    comparable = subset & np.isfinite(gt_values) & np.isfinite(pred_values)
                    residual = np.abs(pred_values - gt_values)
                    sign_disagreement = np.sign(pred_values) != np.sign(gt_values)
                    pair_report[axis_name] = {
                        "count": int(comparable.sum()),
                        "gt": _stats(gt_values[comparable]),
                        "predicted": _stats(pred_values[comparable]),
                        "absolute_residual": _stats(residual[comparable]),
                        "sign_disagreement_rate": float(sign_disagreement[comparable].mean()) if comparable.any() else None,
                    }
                pair_report["temporal_sign_transition_behavior"] = _center_transition_report(
                    split, pred_geometry[pair_name], pair_name, subset,
                )
                subset_report[pair_name] = pair_report
            state_report[subset_name] = subset_report
        result["states"][state] = state_report
    return result


def _center_transition_report(split: dict[str, Any], prediction_pair: dict[str, np.ndarray], pair_name: str,
                              subset: np.ndarray) -> dict[str, Any]:
    gt_pair = split["target_geometry"][pair_name]
    result: dict[str, Any] = {}
    for axis_name, key in (("signed_z", "signed_z"), ("signed_forward_y", "signed_forward_y")):
        gt = gt_pair[key]
        pred = prediction_pair[key]
        gt_transitions: list[bool] = []
        pred_transitions: list[bool] = []
        disagreements: list[bool] = []
        for index, meta in enumerate(split["metadata"]):
            if not subset[index] or meta["local_index"] <= 0:
                continue
            previous = index - 1
            if split["sequence_indices"][previous] != split["sequence_indices"][index]:
                continue
            if not all(np.isfinite(value) for value in (gt[previous], gt[index], pred[previous], pred[index])):
                continue
            gt_transition = np.sign(gt[previous]) != np.sign(gt[index]) and np.sign(gt[previous]) != 0 and np.sign(gt[index]) != 0
            pred_transition = np.sign(pred[previous]) != np.sign(pred[index]) and np.sign(pred[previous]) != 0 and np.sign(pred[index]) != 0
            gt_transitions.append(bool(gt_transition))
            pred_transitions.append(bool(pred_transition))
            disagreements.append(bool(gt_transition != pred_transition))
        result[axis_name] = {
            "center_edge_count": len(gt_transitions),
            "gt_transition_rate": float(np.mean(gt_transitions)) if gt_transitions else None,
            "predicted_transition_rate": float(np.mean(pred_transitions)) if pred_transitions else None,
            "transition_behavior_disagreement_rate": float(np.mean(disagreements)) if disagreements else None,
        }
    return result


def _split_target_report(split: dict[str, Any]) -> dict[str, Any]:
    geometry = split["target_geometry"]
    result = {
        "frame_count": len(split["metadata"]),
        "sequence_count": len(split["ranges"]),
        "features": {
            "torso_root_orientation_degrees": _stats(np.degrees(geometry["root_orientation"])),
            "shoulder_axis_orientation_degrees": _stats(np.degrees(geometry["shoulder"]["angle"])),
            "hip_axis_orientation_degrees": _stats(np.degrees(geometry["hip"]["angle"])),
            "shoulder_bilateral_length_m": _stats(geometry["shoulder"]["length"]),
            "hip_bilateral_length_m": _stats(geometry["hip"]["length"]),
            "shoulder_signed_z_right_minus_left_m": _stats(geometry["shoulder"]["signed_z"]),
            "hip_signed_z_right_minus_left_m": _stats(geometry["hip"]["signed_z"]),
            "shoulder_signed_forward_y_right_minus_left_m": _stats(geometry["shoulder"]["signed_forward_y"]),
            "hip_signed_forward_y_right_minus_left_m": _stats(geometry["hip"]["signed_forward_y"]),
        },
        "temporal": {},
    }
    for name, values in split["target_temporal"].items():
        multiplier = 180.0 / np.pi if "orientation" in name and ("rad" in name or "run" not in name) else 1.0
        result["temporal"][name] = _stats(values * multiplier)
    return result


def _split_input_report(split: dict[str, Any]) -> dict[str, Any]:
    geometry = split["input_geometry"]
    shoulder_ordering = np.where(np.isfinite(geometry["ordering"]["shoulder"]),
                                 geometry["ordering"]["shoulder"] == 1, np.nan)
    hip_ordering = np.where(np.isfinite(geometry["ordering"]["hip"]),
                            geometry["ordering"]["hip"] == 1, np.nan)
    result = {
        "frame_count": len(split["metadata"]),
        "sequence_count": len(split["ranges"]),
        "features": {
            "normalized_shoulder_span": _stats(geometry["shoulder"]["xy_length"]),
            "normalized_hip_span": _stats(geometry["hip"]["xy_length"]),
            "normalized_torso_projected_height": _stats(geometry["torso_height"]),
            "raw_image_torso_projected_height": _stats(geometry["raw_torso_height"]),
            "normalized_shoulder_span_over_torso": _stats(geometry["shoulder"]["xy_length"] / geometry["torso_height"]),
            "normalized_hip_span_over_torso": _stats(geometry["hip"]["xy_length"] / geometry["torso_height"]),
            "confidence_mean": _stats(geometry["confidence_mean"]),
            "valid_joint_count": _stats(geometry["observed_count"]),
            "valid_bilateral_pair_count": _stats(geometry["pair_count"]),
            "projected_shoulder_right_minus_left_x": _stats(geometry["shoulder"]["vector"][:, 0]),
            "projected_hip_right_minus_left_x": _stats(geometry["hip"]["vector"][:, 0]),
            "projected_shoulder_right_is_right_fraction": _stats(shoulder_ordering),
            "projected_hip_right_is_right_fraction": _stats(hip_ordering),
        },
        "temporal": {},
    }
    for name, values in split["input_temporal"].items():
        if values.ndim > 1:
            continue
        result["temporal"][name] = _stats(values)
    return result


def _metadata_report(splits: dict[str, dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, split in splits.items():
        views = sorted({str(item["view"]) for item in split["metadata"] if item["view"] is not None})
        actions = sorted({str(item["action"]) for item in split["metadata"] if item["action"] is not None})
        result[name] = {
            "view_labels": views,
            "view_metadata_available": bool(views),
            "semantic_action_labels": actions,
            "semantic_action_metadata_available": bool(actions),
            "sequence_ids": sorted({item["sequence_id"] for item in split["metadata"]}),
            "fps_values": sorted({float(value) for value in split["fps"]}),
        }
    return result


def _fingerprint(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest(), "byte_size": path.stat().st_size}


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-mix-train", required=True, type=Path)
    parser.add_argument("--three-dpw-train", required=True, type=Path)
    parser.add_argument("--three-dpw-validation", required=True, type=Path)
    parser.add_argument("--three-dpw-test", required=True, type=Path)
    parser.add_argument("--a9-checkpoint", required=True, type=Path)
    parser.add_argument("--a12-checkpoint", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    torch, nn = _torch()
    a9_checkpoint = torch.load(args.a9_checkpoint, map_location=args.device, weights_only=True)
    a12_checkpoint = torch.load(args.a12_checkpoint, map_location=args.device, weights_only=True)
    if int(a9_checkpoint["window"]) != int(a12_checkpoint["window"]):
        raise ValueError("A9 and A12 window contracts differ")
    window = int(a9_checkpoint["window"])
    normalization = a9_checkpoint.get("input_coordinate_normalization", "image_v1")
    if a12_checkpoint.get("input_coordinate_normalization", "image_v1") != normalization:
        raise ValueError("A9 and A12 input coordinate contracts differ")
    direct_mix_dataset = load_dataset(args.direct_mix_train)
    datasets = {
        "train": load_dataset(args.three_dpw_train),
        "validation": load_dataset(args.three_dpw_validation),
        "test": load_dataset(args.three_dpw_test),
    }
    splits = {name: _build_split(name, path, dataset, normalization, window)
              for name, (path, dataset) in zip(
                  ("train", "validation", "test"),
                  ((args.three_dpw_train, datasets["train"]), (args.three_dpw_validation, datasets["validation"]), (args.three_dpw_test, datasets["test"])),
              )}
    predictions: dict[str, dict[str, np.ndarray]] = {}
    for name, split in splits.items():
        predictions[name] = {
            "a9": _predict_split(torch, nn, a9_checkpoint, split, args.device),
            "a12": _predict_split(torch, nn, a12_checkpoint, split, args.device),
        }
        predictions[name]["a9_yaw"] = _yaw_values(predictions[name]["a9"], split["targets"], split["valid"])
        predictions[name]["a12_yaw"] = _yaw_values(predictions[name]["a12"], split["targets"], split["valid"])

    hard_sets: dict[str, dict[str, dict[str, Any]]] = {}
    for name, split in splits.items():
        hard_sets[name] = {}
        for fraction in TAILS:
            label = f"top_{int(fraction * 100):d}_percent"
            hard_sets[name][label] = _select_tail(split, predictions[name]["a9_yaw"], predictions[name]["a12_yaw"], fraction)

    overlap = {
        name: _hard_case_overlap(split, hard_sets[name], predictions[name]["a9_yaw"], predictions[name]["a12_yaw"])
        for name, split in splits.items()
    }
    support = _support_report(splits["train"], splits["validation"], splits["test"], hard_sets, args.seed)
    signed_depth = {
        name: _signed_depth_report(split, predictions[name], hard_sets[name])
        for name, split in splits.items()
    }
    replacement = _replacement_accounting(torch, direct_mix_dataset, args.direct_mix_train, normalization, args.seed)
    target_report = {name: _split_target_report(split) for name, split in splits.items()}
    input_report = {name: _split_input_report(split) for name, split in splits.items()}
    report = {
        "schema": "animcv_3dpw_generalization_support_diagnosis_v1",
        "diagnostic_only": True,
        "hard_set_contract": {
            "definition": "A9 existing root_yaw_error_degrees evaluator",
            "tails": list(TAILS),
            "window": window,
            "sequence_disjoint_support": True,
            "a12_role": "fixed-hard-case confirmation only",
        },
        "coordinate_contract": {
            "canonical_frame": "+X right, +Y forward/depth, +Z up",
            "requested_signed_depth": "z_right - z_left, reported verbatim as canonical z-axis component",
            "additional_physical_forward_component": "y_right - y_left",
            "normalization": normalization,
        },
        "datasets": {"direct_mix_train": _fingerprint(args.direct_mix_train),
                     **{name: _fingerprint(path) for name, path in (
                         ("three_dpw_train", args.three_dpw_train), ("three_dpw_validation", args.three_dpw_validation),
                         ("three_dpw_test", args.three_dpw_test), ("a9_checkpoint", args.a9_checkpoint), ("a12_checkpoint", args.a12_checkpoint))}},
        "gt_target_space": target_report,
        "input_observation_space": input_report,
        "metadata_availability": _metadata_report(splits),
        "hard_sets": {name: {label: {key: value for key, value in tail.items() if key != "indices"}
                              for label, tail in tails.items()} for name, tails in hard_sets.items()},
        "a9_a12_hard_case_overlap": overlap,
        "support": support,
        "signed_relative_depth_attribution": signed_depth,
        "sequence_diversity_and_replacement": replacement,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(_json_ready(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "schema": report["schema"],
                      "split_counts": {name: len(split["metadata"]) for name, split in splits.items()}}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
