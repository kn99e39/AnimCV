"""Exact middle-only hinge feasibility contracts.

These tests exercise the diagnostic geometry in
``scripts/diagnose_hinge_feasibility.py``.  They do not add or select a
production branch write policy.
"""

import importlib.util
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest


def _load_geometry():
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "src"))
    try:
        spec = importlib.util.spec_from_file_location(
            "diagnose_hinge_feasibility", root / "scripts" / "diagnose_hinge_feasibility.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


GEOMETRY = _load_geometry()


def _cases():
    return {entry["name"]: entry for entry in GEOMETRY.synthetic_contracts()["cases"]}


def test_two_sphere_intersection_is_a_circle_with_explicit_plane_center_and_radius():
    locus = GEOMETRY.bone_preserving_locus([0.0, 2.0, 0.0], [1.0, 3.0, 0.0],
                                           np.sqrt(1.5), np.sqrt(1.5))
    assert locus["status"] == "circle"
    np.testing.assert_allclose(locus["circle_center"], [0.5, 2.5, 0.0])
    np.testing.assert_allclose(locus["plane_normal"], [1.0, 1.0, 0.0])
    assert locus["circle_radius"] == pytest.approx(1.0)


def test_two_sphere_degeneracies_are_not_hidden_by_a_radius_clamp():
    tangent = GEOMETRY.bone_preserving_locus([-1.0, 3.0, 0.0], [1.0, 3.0, 0.0], 1.0, 1.0)
    assert tangent["status"] == "zero_radius"
    inconsistent = GEOMETRY.bone_preserving_locus([-1.0, 3.0, 0.0], [1.0, 3.0, 0.0],
                                                    0.5, 0.5)
    assert inconsistent["status"] == "inconsistent_spheres"
    sphere = GEOMETRY.bone_preserving_locus([0.0, 3.0, 0.0], [0.0, 3.0, 0.0], 1.0, 1.0)
    assert sphere["status"] == "sphere"
    unequal = GEOMETRY.bone_preserving_locus([0.0, 3.0, 0.0], [0.0, 3.0, 0.0], 1.0, 2.0)
    assert unequal["status"] == "inconsistent_concentric_spheres"


def test_generic_camera_ray_has_one_circle_intersection_at_the_current_middle_joint():
    cases = _cases()
    for name in ("ordinary_oblique", "fronto_parallel", "depth_aligned", "near_depth_aligned"):
        case = cases[name]
        intersections = case["camera_ray_intersection"]
        assert intersections["intersection_count"] == 1, name
        np.testing.assert_allclose(intersections["points"][0], case["M"], atol=1e-12)
        assert case["opposite_branch_intersection_count"] == 0, name


def test_parallel_contained_ray_is_the_special_two_intersection_case():
    case = _cases()["ray_parallel_and_contained_in_circle_plane"]
    intersections = case["camera_ray_intersection"]
    assert intersections["relation"] == "ray_contained_in_circle_plane"
    assert intersections["intersection_count"] == 2
    assert case["opposite_branch_intersection_count"] == 1
    assert case["non_trivial_branch_changing_solution"] is True
    np.testing.assert_allclose(sorted(point[1] for point in intersections["points"]), [1.0, 5.0])


def test_depth_alignment_and_zero_radius_do_not_claim_a_branch_change():
    cases = _cases()
    assert cases["depth_aligned"]["current_branch"]["readable"] is False
    assert cases["near_depth_aligned"]["current_branch"]["readable"] is False
    assert cases["degenerate_zero_radius_circle"]["non_trivial_branch_changing_solution"] is False


def test_bone_preserving_circle_has_readable_opposite_branch_even_when_ray_does_not():
    case = _cases()["ordinary_oblique"]
    requested = case["requested_opposite_branch"]
    available = GEOMETRY.opposite_branch_available(case["bone_preserving_locus"], requested)
    assert available["available"] is True
    # The absence of a second ray intersection is therefore a camera/write-set
    # incompatibility, not an absence of a metric bone-preserving branch point.
    assert case["opposite_branch_intersection_count"] == 0


def test_camera_aware_lower_bound_is_deterministic_and_exactly_bone_preserving():
    case = _cases()["ordinary_oblique"]
    K = np.array([[1200.0, 0.0, 320.0], [0.0, 1200.0, 240.0], [0.0, 0.0, 1.0]])
    first = GEOMETRY.minimum_image_displacement_on_branch(
        case["M"], case["bone_preserving_locus"], K, case["requested_opposite_branch"])
    second = GEOMETRY.minimum_image_displacement_on_branch(
        case["M"], case["bone_preserving_locus"], K, case["requested_opposite_branch"])
    assert first["resolved"] is True
    assert first == second
    point = np.asarray(first["candidate_point"])
    p, d = np.asarray(case["P"]), np.asarray(case["D"])
    m = np.asarray(case["M"])
    np.testing.assert_allclose(np.linalg.norm(point - p), np.linalg.norm(m - p), atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(point - d), np.linalg.norm(m - d), atol=1e-12)
    assert GEOMETRY.branch_coordinates(point, case["bone_preserving_locus"])["sign"] == \
        case["requested_opposite_branch"]
    assert first["no_sampling"] is True
    assert first["no_weighted_3d_term"] is True


def test_special_second_intersection_has_zero_image_displacement_lower_bound():
    case = _cases()["ray_parallel_and_contained_in_circle_plane"]
    K = np.array([[1000.0, 0.0, 320.0], [0.0, 1000.0, 240.0], [0.0, 0.0, 1.0]])
    result = GEOMETRY.minimum_image_displacement_on_branch(
        case["M"], case["bone_preserving_locus"], K, case["requested_opposite_branch"])
    assert result["resolved"] is True
    assert result["minimum_pixel_displacement"] == pytest.approx(0.0, abs=1e-9)


def test_actual_branch_constraint_input_contract_has_no_camera_state():
    from framepose.branch_constraints import apply_branch_constraints

    audit = GEOMETRY.branch_constraint_input_contract()
    assert list(inspect.signature(apply_branch_constraints).parameters)[:3] == [
        "pose", "valid", "requested"]
    assert audit["production_default"] == "DEPTH_ONLY"
    assert set(audit["not_received"]) == {
        "input_2d", "camera intrinsics", "camera extrinsics", "absolute root depth", "image size"
    }
