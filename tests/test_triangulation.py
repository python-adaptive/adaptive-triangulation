from __future__ import annotations

import importlib.util
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

import adaptive_triangulation as rust_tri


REFERENCE_PATH = Path("/tmp/python_triangulation_reference.py")


def load_reference_module():
    spec = importlib.util.spec_from_file_location("python_triangulation_reference", REFERENCE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


REF = load_reference_module()


def as_simplex_set(value) -> set[tuple[int, ...]]:
    return {tuple(simplex) for simplex in value}


def as_vertex_to_simplices(value) -> list[set[tuple[int, ...]]]:
    return [as_simplex_set(simplices) for simplices in value]


def locate_result(value) -> tuple[int, ...]:
    if value is None:
        return ()
    return tuple(value)


def face_counter(iterator) -> Counter[tuple[int, ...]]:
    return Counter(tuple(face) for face in iterator)


def assert_points_close(lhs, rhs, atol: float = 1e-8) -> None:
    lhs = np.asarray(lhs, dtype=float)
    rhs = np.asarray(rhs, dtype=float)
    assert lhs.shape == rhs.shape
    assert np.allclose(lhs, rhs, atol=atol, rtol=1e-8)


def assert_triangulation_equal(rust, reference) -> None:
    assert rust.dim == reference.dim
    assert_points_close(rust.vertices, reference.vertices)
    assert as_simplex_set(rust.simplices) == as_simplex_set(reference.simplices)
    assert as_vertex_to_simplices(rust.vertex_to_simplices) == as_vertex_to_simplices(
        reference.vertex_to_simplices
    )
    assert set(rust.hull) == set(reference.hull)
    assert rust.reference_invariant() is True
    assert reference.reference_invariant() is True
    assert sorted(rust.volumes()) == pytest.approx(sorted(reference.volumes()))


def assert_locate_equivalent(rust, reference, point) -> None:
    rust_simplex = locate_result(rust.locate_point(point))
    ref_simplex = locate_result(reference.locate_point(np.asarray(point, dtype=float)))

    if not ref_simplex:
        assert rust_simplex == ()
        return

    assert rust_simplex in as_simplex_set(rust.simplices)
    assert ref_simplex in as_simplex_set(reference.simplices)
    assert rust.point_in_simplex(point, rust_simplex)
    assert reference.point_in_simplex(np.asarray(point, dtype=float), ref_simplex)


@pytest.mark.parametrize(
    ("coords", "probe"),
    [
        (
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.3, 0.4]],
            [0.35, 0.3],
        ),
        (
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.2, 0.2, 0.2],
            ],
            [0.2, 0.2, 0.15],
        ),
        (
            [
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.2, 0.2, 0.2, 0.1],
            ],
            [0.15, 0.15, 0.15, 0.1],
        ),
    ],
)
def test_construction_matches_reference(coords, probe):
    rust = rust_tri.Triangulation(np.asarray(coords, dtype=float))
    reference = REF.Triangulation(coords)

    assert_triangulation_equal(rust, reference)
    assert_locate_equivalent(rust, reference, probe)


def test_geometry_functions_match_reference():
    triangle = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    tetra = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    simplex4 = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    center, radius = rust_tri.circumsphere(simplex4)
    ref_center, ref_radius = REF.circumsphere(simplex4)
    assert_points_close(center, ref_center)
    assert radius == pytest.approx(ref_radius)

    fast_center, fast_radius = rust_tri.fast_2d_circumcircle(triangle)
    ref_fast_center, ref_fast_radius = REF.fast_2d_circumcircle(triangle)
    assert_points_close(fast_center, ref_fast_center)
    assert fast_radius == pytest.approx(ref_fast_radius)

    fast3_center, fast3_radius = rust_tri.fast_3d_circumsphere(tetra)
    ref_fast3_center, ref_fast3_radius = REF.fast_3d_circumcircle(tetra)
    assert_points_close(fast3_center, ref_fast3_center)
    assert fast3_radius == pytest.approx(ref_fast3_radius)

    assert rust_tri.fast_2d_point_in_simplex([0.2, 0.2], triangle) is True
    assert rust_tri.fast_2d_point_in_simplex([1.1, 0.2], triangle) is False
    assert rust_tri.point_in_simplex([0.2, 0.2], triangle) == REF.point_in_simplex(
        np.array([0.2, 0.2]),
        triangle,
    )
    assert rust_tri.point_in_simplex([0.2, 0.2, 0.2], tetra) == REF.point_in_simplex(
        np.array([0.2, 0.2, 0.2]),
        tetra,
    )

    assert rust_tri.volume(triangle) == pytest.approx(0.5)
    assert rust_tri.volume(tetra) == pytest.approx(1.0 / 6.0)
    embedded_triangle = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 4.0, 0.0]])
    assert rust_tri.simplex_volume_in_embedding(embedded_triangle) == pytest.approx(6.0)

    orientation = rust_tri.orientation([[1.0, 0.0], [0.0, 1.0]], [0.1, 0.1])
    ref_orientation = REF.orientation([[1.0, 0.0], [0.0, 1.0]], [0.1, 0.1])
    assert orientation == ref_orientation


def test_faces_containing_and_hull_match_reference():
    coords = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.4, 0.2]]
    rust = rust_tri.Triangulation(coords)
    reference = REF.Triangulation(coords)

    assert face_counter(rust.faces()) == face_counter(reference.faces())
    assert face_counter(rust.faces(dim=1)) == face_counter(reference.faces(dim=1))
    assert face_counter(rust.faces(vertices={0, 1, 4})) == face_counter(
        reference.faces(vertices={0, 1, 4})
    )

    face = next(iter(as_simplex_set(rust.simplices)))
    edge = face[:2]
    assert as_simplex_set(rust.containing(edge)) == as_simplex_set(reference.containing(edge))
    assert set(rust.hull) == set(reference.hull)


def test_add_point_inside_hull_matches_reference():
    coords = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    point = [0.3, 0.4]
    rust = rust_tri.Triangulation(coords)
    reference = REF.Triangulation(coords)

    rust_deleted, rust_added = rust.add_point(point)
    ref_deleted, ref_added = reference.add_point(point)

    assert as_simplex_set(rust_deleted) == as_simplex_set(ref_deleted)
    assert as_simplex_set(rust_added) == as_simplex_set(ref_added)
    assert_triangulation_equal(rust, reference)


def test_add_point_outside_hull_matches_reference():
    coords = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.3, 0.3]]
    point = [1.5, 0.5]
    rust = rust_tri.Triangulation(coords)
    reference = REF.Triangulation(coords)

    rust_deleted, rust_added = rust.add_point(point)
    ref_deleted, ref_added = reference.add_point(point)

    assert as_simplex_set(rust_deleted) == as_simplex_set(ref_deleted)
    assert as_simplex_set(rust_added) == as_simplex_set(ref_added)
    assert_triangulation_equal(rust, reference)


def test_add_point_with_transform_matches_reference():
    coords = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    point = [0.4, 0.25]
    transform = np.array([[2.0, 0.0], [0.0, 0.5]])

    rust = rust_tri.Triangulation(coords)
    reference = REF.Triangulation(coords)

    rust_deleted, rust_added = rust.add_point(point, transform=transform)
    ref_deleted, ref_added = reference.add_point(point, transform=transform)

    assert as_simplex_set(rust_deleted) == as_simplex_set(ref_deleted)
    assert as_simplex_set(rust_added) == as_simplex_set(ref_added)
    assert_triangulation_equal(rust, reference)


def test_circumscribed_circle_and_point_in_circumcircle_match_reference():
    coords = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.25, 0.25]]
    transform = np.array(
        [
            [math.cos(math.pi / 6), -math.sin(math.pi / 6)],
            [math.sin(math.pi / 6), math.cos(math.pi / 6)],
        ]
    )
    rust = rust_tri.Triangulation(coords)
    reference = REF.Triangulation(coords)

    simplex = next(iter(as_simplex_set(rust.simplices)))
    center, radius = rust.circumscribed_circle(simplex, transform=transform)
    ref_center, ref_radius = reference.circumscribed_circle(simplex, transform)
    assert_points_close(center, ref_center)
    assert radius == pytest.approx(ref_radius)

    for idx in range(len(coords)):
        assert rust.point_in_circumcircle(idx, simplex, transform=transform) == reference.point_in_cicumcircle(
            idx, simplex, transform
        )


def test_default_transform_and_point_in_simplex_instance_method():
    coords = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    tri = rust_tri.Triangulation(coords)

    assert_points_close(tri.default_transform, np.eye(2))
    simplex = next(iter(as_simplex_set(tri.simplices)))
    assert tri.point_in_simplex([0.2, 0.2], simplex) is True
    assert tri.point_in_simplex([1.2, 0.2], simplex) is False


def test_duplicate_point_is_rejected():
    coords = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    tri = rust_tri.Triangulation(coords)
    simplex = next(iter(as_simplex_set(tri.simplices)))

    with pytest.raises(ValueError, match="Point already in triangulation"):
        tri.add_point([0.0, 0.0], simplex=simplex)


def test_random_cross_validation_2d():
    rng = np.random.default_rng(1234)
    coords = rng.random((6, 2))
    rust = rust_tri.Triangulation(coords[:3])
    reference = REF.Triangulation(coords[:3])

    for point in coords[3:]:
        rust_deleted, rust_added = rust.add_point(point)
        ref_deleted, ref_added = reference.add_point(point)
        assert as_simplex_set(rust_deleted) == as_simplex_set(ref_deleted)
        assert as_simplex_set(rust_added) == as_simplex_set(ref_added)
        assert_triangulation_equal(rust, reference)


def test_random_cross_validation_3d():
    rng = np.random.default_rng(4321)
    coords = rng.random((7, 3))
    rust = rust_tri.Triangulation(coords[:4])
    reference = REF.Triangulation(coords[:4])

    for point in coords[4:]:
        rust_deleted, rust_added = rust.add_point(point)
        ref_deleted, ref_added = reference.add_point(point)
        assert as_simplex_set(rust_deleted) == as_simplex_set(ref_deleted)
        assert as_simplex_set(rust_added) == as_simplex_set(ref_added)
        assert_triangulation_equal(rust, reference)
