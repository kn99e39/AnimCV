"""Orthographic vs perspective line side, and the populations they are measured on.

docs/38. docs/37 identified the canonical X/Z line side with the observed image
line side. These pin why that identification is not exact, and pin the
population separation whose absence made docs/37's accuracy figure describe the
wrong set of frames.
"""

import numpy as np
import pytest

from framepose.hinge_plane import EMPIRICAL_AXIS_CORRESPONDENCE, IMAGE_TO_CANONICAL


def _load_script(name, relative):
    import importlib.util
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "src"))
    try:
        spec = importlib.util.spec_from_file_location(name, root / relative)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


PROJECTION = _load_script("diagnose_hinge_projection_ownership",
                          "scripts/diagnose_hinge_projection_ownership.py")


# ------------------------------------------- orthographic vs perspective ----

def _minors(P, M, D):
    return (M[0] * D[2] - M[2] * D[0],
            P[0] * D[2] - P[2] * D[0],
            P[0] * M[2] - P[2] * M[0])


def test_orthographic_and_perspective_line_sides_are_the_same_minors_differently_weighted():
    """Both are `w_P*A - w_M*B + w_D*C`; orthographic uses w = 1 everywhere,
    perspective uses each joint's own depth. That is the whole difference."""
    rng = np.random.default_rng(5)
    for _ in range(500):
        P = np.array([rng.normal(), rng.uniform(1.0, 5.0), rng.normal()])
        M = np.array([rng.normal(), rng.uniform(1.0, 5.0), rng.normal()])
        D = np.array([rng.normal(), rng.uniform(1.0, 5.0), rng.normal()])
        A, B, C = _minors(P, M, D)

        ortho = (D[0] - P[0]) * (M[2] - P[2]) - (D[2] - P[2]) * (M[0] - P[0])
        assert np.sign(ortho) == np.sign(-(A - B + C)) or abs(ortho) < 1e-12

        p, m, d = [np.array([Q[0] / Q[1], Q[2] / Q[1]]) for Q in (P, M, D)]
        perspective = (d[0] - p[0]) * (m[1] - p[1]) - (d[1] - p[1]) * (m[0] - p[0])
        weighted = P[1] * A - M[1] * B + D[1] * C
        assert np.sign(perspective) == np.sign(-weighted) or abs(perspective) < 1e-12


def test_the_two_sides_agree_exactly_when_the_chain_shares_one_depth():
    """Equal depths is exactly the orthographic case, so the identification
    docs/37 made is correct there and only there."""
    rng = np.random.default_rng(6)
    for _ in range(200):
        depth = rng.uniform(1.5, 4.0)
        P = np.array([rng.normal(), depth, rng.normal()])
        M = np.array([rng.normal(), depth, rng.normal()])
        D = np.array([rng.normal(), depth, rng.normal()])
        ortho = np.sign((D[0] - P[0]) * (M[2] - P[2]) - (D[2] - P[2]) * (M[0] - P[0]))
        p, m, d = [np.array([Q[0] / Q[1], Q[2] / Q[1]]) for Q in (P, M, D)]
        perspective = np.sign((d[0] - p[0]) * (m[1] - p[1]) - (d[1] - p[1]) * (m[0] - p[0]))
        if ortho != 0:
            assert ortho == perspective


def test_deterministic_counterexample_with_all_depths_positive():
    """A fixed chain, every depth positive, where the two signs disagree."""
    P = np.array([-2.0, 2.0, -2.0])
    M = np.array([-2.0, 1.0, -1.0])
    D = np.array([-1.0, 1.0, -2.0])
    assert (P[1] > 0) and (M[1] > 0) and (D[1] > 0)

    ortho = np.sign((D[0] - P[0]) * (M[2] - P[2]) - (D[2] - P[2]) * (M[0] - P[0]))
    p, m, d = [np.array([Q[0] / Q[1], Q[2] / Q[1]]) for Q in (P, M, D)]
    perspective = np.sign((d[0] - p[0]) * (m[1] - p[1]) - (d[1] - p[1]) * (m[0] - p[0]))

    assert ortho == 1.0
    assert perspective == -1.0
    assert ortho != perspective

    A, B, C = _minors(P, M, D)
    assert (A, B, C) == (3.0, 2.0, -2.0)
    assert A - B + C == -1.0                              # orthographic
    assert P[1] * A - M[1] * B + D[1] * C == 2.0          # perspective


# ------------------------------------------------------ the projection ----

def test_projection_uses_only_the_repositorys_own_axis_conversion():
    """`_world_to_animcv_camera` maps OpenCV (x, y, z) to AnimCV (x, z, -y), so
    the inverse used for projection must be x = X, y = -Z, z = Y."""
    from pose.three_dpw_adapter import _world_to_animcv_camera

    rng = np.random.default_rng(8)
    opencv = rng.normal(size=(6, 3))
    opencv[:, 2] = rng.uniform(1.5, 5.0, size=6)          # positive depth
    animcv = _world_to_animcv_camera(opencv, np.eye(4))
    np.testing.assert_allclose(animcv[:, 0], opencv[:, 0], atol=1e-12)
    np.testing.assert_allclose(animcv[:, 1], opencv[:, 2], atol=1e-12)
    np.testing.assert_allclose(animcv[:, 2], -opencv[:, 1], atol=1e-12)

    intrinsics = np.array([[1961.85, 0.0, 540.0], [0.0, 1969.23, 960.0], [0.0, 0.0, 1.0]])
    pixels = PROJECTION.project(animcv, intrinsics)
    expected_u = intrinsics[0, 0] * opencv[:, 0] / opencv[:, 2] + intrinsics[0, 2]
    expected_v = intrinsics[1, 1] * opencv[:, 1] / opencv[:, 2] + intrinsics[1, 2]
    np.testing.assert_allclose(pixels[:, 0], expected_u, atol=1e-9)
    np.testing.assert_allclose(pixels[:, 1], expected_v, atol=1e-9)


def test_projected_line_side_is_deterministic_and_invariant_to_intrinsic_scale():
    """A focal-length change rescales both image axes about the principal point,
    which cannot change which side of a line a point is on."""
    points = np.array([[-0.2, 3.0, 0.4], [0.05, 2.6, 0.1], [0.3, 3.4, -0.3]])
    base = np.array([[1900.0, 0.0, 540.0], [0.0, 1900.0, 960.0], [0.0, 0.0, 1.0]])
    first = PROJECTION.projected_side(PROJECTION.project(points, base))
    scaled = base.copy()
    scaled[0, 0] *= 2.0
    scaled[1, 1] *= 2.0
    assert PROJECTION.projected_side(PROJECTION.project(points, scaled)) == first
    assert PROJECTION.projected_side(PROJECTION.project(points, base)) == first
    assert first in (-1, 1)


def test_projection_invents_no_camera_parameter():
    import inspect

    source = inspect.getsource(PROJECTION.project) + inspect.getsource(PROJECTION.load_absolute_sequences)
    for banned in ("focal", "1000.0", "default_intrinsics", "assume", "fallback"):
        assert banned not in source, banned
    assert "cam_intrinsics" in source and "cam_poses" in source


# ------------------------------------------------------------ populations ----

def test_populations_are_named_once_and_do_not_overlap():
    assert PROJECTION.POPULATIONS == (
        "all_frames", "all_279_residual_flips", "depth_correct_83_residual_flips",
        "type_1_corrections", "type_2_corrections")
    # docs/37 reused one label for two different sets; the names must be distinct.
    assert len(set(PROJECTION.POPULATIONS)) == len(PROJECTION.POPULATIONS)
    assert "residual_flip_frames" not in PROJECTION.POPULATIONS


def test_empirical_axis_correspondence_is_not_a_projection_contract():
    assert EMPIRICAL_AXIS_CORRESPONDENCE["is_projection_contract"] is False
    assert EMPIRICAL_AXIS_CORRESPONDENCE["status"] == "bank_specific_diagnostic"
    assert EMPIRICAL_AXIS_CORRESPONDENCE["bank"] == "bank_3dpw_paired_v2"
    # The historical constant is retained for readability of docs/37 artifacts,
    # and carries the same axis facts.
    assert IMAGE_TO_CANONICAL["canonical_x_from"] == EMPIRICAL_AXIS_CORRESPONDENCE["canonical_x_from"]
    assert IMAGE_TO_CANONICAL["canonical_z_from"] == EMPIRICAL_AXIS_CORRESPONDENCE["canonical_z_from"]

    import inspect

    import framepose.hinge_plane as module

    text = inspect.getsource(module)
    assert "not a projection contract" in text.lower()
    assert "perspective" in text.lower()
