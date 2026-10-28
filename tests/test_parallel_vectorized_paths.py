"""Worklog 99A: concurrency helpers and vectorised production paths agree with their scalar definitions."""

import math
import statistics

import numpy as np
import pytest

from common.parallel import process_map, thread_map, worker_count


def _square(x):
    return x * x


def test_thread_and_process_map_preserve_order_and_serial_fallback(monkeypatch):
    items = list(range(23))
    assert thread_map(_square, items) == [x * x for x in items]
    assert process_map(_square, items, workers=2) == [x * x for x in items]
    monkeypatch.setenv("ANIMCV_WORKERS", "1")
    assert worker_count() == 1 and worker_count("io") == 1
    assert process_map(_square, items) == [x * x for x in items]


def _quat(rng):
    q = rng.normal(size=4)
    return q / np.linalg.norm(q)


def test_collapse_segment_errors_match_scalar_interpolation():
    from optimize.collapse import _interpolation_error, _segment_errors
    from retarget.solver import BoneTransformSample

    rng = np.random.default_rng(0)
    for case in range(40):
        n = 9
        if case % 3 == 0:  # near-identity branch (dot > 0.9995) with a sign flip on the far anchor
            base = _quat(rng)
            qs = [base + 1e-3 * rng.normal(size=4) for _ in range(n)]
            qs = [q / np.linalg.norm(q) for q in qs]
            qs[-1] = -qs[-1]
        else:
            qs = [_quat(rng) for _ in range(n)]
        samples = [BoneTransformSample(i, "b", None, tuple(q), None, 1.0) for i, q in enumerate(qs)]
        vec = _segment_errors(np.array(qs), 0, n - 1, False)
        ref = [_interpolation_error(samples[0], samples[-1], samples[i], i / (n - 1), False) for i in range(1, n - 1)]
        assert np.allclose(vec, ref, atol=1e-9)
        locs = [tuple(rng.normal(size=3)) for _ in range(n)]
        ls = [BoneTransformSample(i, "b", loc, (0.0, 0.0, 0.0, 1.0), None, 1.0) for i, loc in enumerate(locs)]
        vec = _segment_errors(np.array(locs), 0, n - 1, True)
        ref = [_interpolation_error(ls[0], ls[-1], ls[i], i / (n - 1), True) for i in range(1, n - 1)]
        assert np.allclose(vec, ref, atol=1e-12)


def test_importance_matches_scalar_formula():
    from optimize.importance import quaternion_angle, score_track_importance
    from retarget.solver import AnimationTrack, BoneTransformSample

    rng = np.random.default_rng(1)
    samples = [BoneTransformSample(i, "b", None, tuple(_quat(rng)), None, float(rng.uniform())) for i in range(15)]
    track = AnimationTrack("b", samples)
    got = score_track_importance(track, {4})
    n = len(samples)
    v = [0.0] + [quaternion_angle(samples[i - 1].rotation, samples[i].rotation) for i in range(1, n)]
    a = [0.0] + [abs(v[i] - v[i - 1]) for i in range(1, n)]
    c = [0.0] + [abs(samples[i].confidence - samples[i - 1].confidence) for i in range(1, n)]
    mv, ma = max(v) or 1.0, max(a) or 1.0
    for i in range(n):
        if i in (0, n - 1, 4):
            assert got[i] == 1.0
            continue
        ext = (v[i - 1] < v[i] > v[i + 1]) or (v[i - 1] > v[i] < v[i + 1])
        ref = min(1.0, 0.35 * v[i] / mv + 0.30 * a[i] / ma + 0.15 * c[i] + 0.20 * ext)
        assert got[i] == pytest.approx(ref, abs=1e-12)


def test_median_filter_matches_scalar_definition():
    from motion.motion_graph import MotionFrame, MotionGraph, MotionPoint
    from retarget.temporal_filter import median_filter_motion_graph

    rng = np.random.default_rng(2)
    frames = []
    for f in range(12):
        pts = {}
        for name in ("a", "b"):
            if name == "b" and f == 6:
                continue  # missing landmark
            p3 = None if (name == "a" and f == 3) else tuple(rng.normal(size=3))
            pts[name] = MotionPoint(name, f, tuple(rng.normal(size=2)), p3, 0.9, not (name == "a" and f == 8))
        frames.append(MotionFrame(f, f / 30.0, pts))
    g = MotionGraph(frames=frames, tracks=[], fps=30.0, source_metadata={})
    out = median_filter_motion_graph(g, 3)
    for i, frame in enumerate(frames):
        for name, p in frame.points.items():
            win = [frames[j].points.get(name) for j in range(i - 1, i + 2)] if 0 < i < len(frames) - 1 else []
            if win and all(w is not None and w.visible for w in win):
                e2 = (statistics.median(w.position_2d[0] for w in win), statistics.median(w.position_2d[1] for w in win))
                e3 = (p.position_3d if any(w.position_3d is None for w in win)
                      else tuple(statistics.median(w.position_3d[k] for w in win) for k in range(3)))
            else:
                e2, e3 = p.position_2d, p.position_3d
            q = out.frames[i].points[name]
            assert q.position_2d == pytest.approx(e2) and (q.position_3d == e3 if e3 is None else q.position_3d == pytest.approx(e3))


def test_keyframe_parent_lookup_bisect():
    from types import SimpleNamespace

    from blender.keyframe_writer import _rotation_at_frame, _track_index

    track = SimpleNamespace(samples=[SimpleNamespace(frame_index=f, rotation=(0, 0, 0, float(f))) for f in (9, 2, 5)])
    idx = _track_index(track)
    assert idx[0] == [2, 5, 9]
    assert _rotation_at_frame(track, 5, idx) == (0, 0, 0, 5.0)
    assert _rotation_at_frame(track, 0, idx) == (0, 0, 0, 2.0)   # before the first key: first key
    assert _rotation_at_frame(track, 20, idx) == (0, 0, 0, 9.0)  # after the last key: last key
    assert _rotation_at_frame(track, 9) == (0, 0, 0, 9.0)         # index built on demand


class _FakeDepthModel:
    """infer_image / image2tensor / forward with the upstream contract (per-pixel brightness as depth)."""

    def image2tensor(self, image, input_size):
        import torch
        return torch.as_tensor(image.mean(axis=2), dtype=torch.float32)[None], image.shape[:2]

    def __call__(self, batch):
        return batch * 2.0

    def infer_image(self, image, input_size):
        return image.mean(axis=2).astype(np.float32) * 2.0


def test_depth_batched_frames_match_per_frame(monkeypatch):
    pytest.importorskip("torch")
    from pose.depth_estimator import DepthEstimator, DepthEstimatorConfig

    est = DepthEstimator(DepthEstimatorConfig(checkpoint_path="unused", batch_size=3))
    monkeypatch.setattr(est, "_load_model", lambda: _FakeDepthModel())
    rng = np.random.default_rng(3)
    images = [rng.integers(0, 255, size=(6, 8, 3)).astype(np.uint8) for _ in range(5)] + [np.zeros((4, 4, 3), np.uint8)]
    got = est.infer_frames(images)
    for image, depth in zip(images, got):
        assert depth.shape == image.shape[:2]
        assert np.allclose(depth, image.mean(axis=2) * 2.0, atol=1e-5)


def test_rest_caches_are_value_keyed():
    from retarget.framepose_fk import _bone_rest_rotation, _rest_direction, _rest_rotation
    from retarget.framepose_target_rest import RestBone, TargetRigRestPose

    eye = tuple(tuple(float(i == j) for j in range(4)) for i in range(4))
    rest = TargetRigRestPose("r", {"a": RestBone("a", None, eye, (0.0, 0.0, 0.0), (0.0, 1.0, 0.0))},
                             {"authority": "blender_imported_armature"})
    assert _rest_direction(rest, "a") == (0.0, 1.0, 0.0)
    assert _rest_rotation(rest, "a") == _bone_rest_rotation(rest.bones["a"])
    moved = TargetRigRestPose("r", {"a": RestBone("a", None, eye, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0))},
                              {"authority": "blender_imported_armature"})
    assert _rest_direction(moved, "a") == (1.0, 0.0, 0.0)  # a different bone value is a different cache entry
    assert math.isclose(sum(c * c for c in _rest_rotation(rest, "a")), 1.0)
