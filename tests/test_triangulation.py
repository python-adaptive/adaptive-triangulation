from __future__ import annotations

import importlib.util
import math
from collections import Counter
from pathlib import Path

import adaptive_triangulation as rust_tri
import numpy as np
import pytest

REFERENCE_PATH = Path(__file__).with_name("python_triangulation_reference.py")


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


def assert_same_exception_type_name(rust_callable, reference_callable) -> None:
    with pytest.raises(Exception) as rust_exc:  # noqa: PT011
        rust_callable()
    with pytest.raises(Exception) as ref_exc:  # noqa: PT011
        reference_callable()
    assert type(rust_exc.value).__name__ == type(ref_exc.value).__name__


class Seq:
    def __init__(self, items):
        self._items = items

    def __len__(self):
        return len(self._items)

    def __iter__(self):
        return iter(self._items)


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
    tetra = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
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
    assert rust_tri.fast_norm([3.0, 4.0]) == pytest.approx(5.0)
    fast3_alias_center, fast3_alias_radius = rust_tri.fast_3d_circumcircle(tetra)
    assert_points_close(fast3_alias_center, ref_fast3_center)
    assert fast3_alias_radius == pytest.approx(ref_fast3_radius)


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
        assert rust.point_in_circumcircle(
            idx, simplex, transform=transform
        ) == reference.point_in_cicumcircle(idx, simplex, transform)


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


def test_constructor_handles_degenerate_leading_points():
    coords = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0]]
    rust = rust_tri.Triangulation(coords)
    reference = REF.Triangulation(coords)

    assert_triangulation_equal(rust, reference)


def test_constructor_matches_reference_on_cocircular_input():
    coords = [[0.5, 0.0], [0.0, 0.5], [0.5, 1.0], [1.0, 0.5]]
    rust = rust_tri.Triangulation(coords)
    reference = REF.Triangulation(coords)

    assert_triangulation_equal(rust, reference)


def test_constructor_accepts_duplicate_trailing_points():
    coords = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
    rust = rust_tri.Triangulation(coords)
    reference = REF.Triangulation(coords)

    assert_triangulation_equal(rust, reference)


def test_get_vertices_preserves_order_and_negative_indices():
    coords = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    rust = rust_tri.Triangulation(coords)
    reference = REF.Triangulation(coords)

    assert_points_close(rust.get_vertices((2, 0, 1)), reference.get_vertices((2, 0, 1)))
    assert_points_close(rust.get_vertices((-1, 0, 1)), reference.get_vertices((-1, 0, 1)))


def test_vertices_proxy_supports_index_len_and_iteration():
    coords = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    tri = rust_tri.Triangulation(coords)
    proxy = tri.vertices

    assert type(proxy).__name__ == "VerticesProxy"
    assert len(proxy) == len(coords)
    assert_points_close(proxy[0], coords[0])
    assert_points_close(proxy[-1], coords[-1])
    assert_points_close(list(proxy), coords)


def test_has_simplex_and_simplices_proxy_membership():
    tri = rust_tri.Triangulation([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    simplex = next(iter(tri.simplices))

    assert tri.has_simplex(simplex) is True
    assert tri.has_simplex(tuple(reversed(simplex))) is True
    assert tri.has_simplex((0, 1, 2)) is False

    assert simplex in tri.simplices
    assert tuple(reversed(simplex)) in tri.simplices
    assert (0, 1, 2) not in tri.simplices


def test_simplices_proxy_iteration_and_len_match_materialized_set():
    tri = rust_tri.Triangulation([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    proxy = tri.simplices
    expected = {tuple(simplex) for simplex in proxy}

    assert type(proxy).__name__ == "SimplicesProxy"
    assert len(proxy) == len(expected)
    assert {tuple(simplex) for simplex in proxy} == expected


def test_vertex_to_simplices_for_matches_materialized_property():
    tri = rust_tri.Triangulation([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    for vertex in range(len(tri.vertices)):
        proxy = tri.vertex_to_simplices_for(vertex)
        expected = as_simplex_set(tri.vertex_to_simplices[vertex])

        assert type(proxy).__name__ == "SimplicesProxy"
        assert len(proxy) == len(expected)
        assert as_simplex_set(proxy) == expected

    assert as_simplex_set(tri.vertex_to_simplices_for(-1)) == as_simplex_set(
        tri.vertex_to_simplices[-1]
    )


def test_vertex_to_simplices_proxy_supports_index_len_and_iteration():
    tri = rust_tri.Triangulation([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    proxy = tri.vertex_to_simplices

    assert type(proxy).__name__ == "VertexToSimplicesProxy"
    assert len(proxy) == len(tri.vertices)

    for index in range(len(proxy)):
        assert isinstance(proxy[index], set)
        assert as_simplex_set(proxy[index]) == as_simplex_set(tri.vertex_to_simplices_for(index))

    assert isinstance(proxy[-1], set)
    assert as_simplex_set(proxy[-1]) == as_simplex_set(tri.vertex_to_simplices_for(-1))

    iterated = [as_simplex_set(simplices) for simplices in proxy]
    expected = [as_simplex_set(tri.vertex_to_simplices_for(index)) for index in range(len(proxy))]
    assert iterated == expected

    simplex = next(iter(tri.simplices))
    unioned = set.union(*[proxy[index] for index in simplex])
    expected_union = set.union(
        *[as_simplex_set(tri.vertex_to_simplices_for(index)) for index in simplex]
    )
    assert as_simplex_set(unioned) == expected_union


def test_get_reduced_simplex_preserves_order_and_returns_list():
    coords = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    rust = rust_tri.Triangulation(coords)
    reference = REF.Triangulation(coords)

    rust_result = rust.get_reduced_simplex((0.5, 0.0), (1, 0, 2))
    ref_result = reference.get_reduced_simplex(np.array((0.5, 0.0)), (1, 0, 2))

    assert isinstance(rust_result, list)
    assert rust_result == ref_result


def test_get_reduced_simplex_preserves_negative_indices_in_output():
    coords = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    rust = rust_tri.Triangulation(coords)
    reference = REF.Triangulation(coords)

    rust_result = rust.get_reduced_simplex((0.5, 0.0), (-3, -2, -1))
    ref_result = reference.get_reduced_simplex(np.array((0.5, 0.0)), (-3, -2, -1))
    assert rust_result == ref_result


def test_invalid_indices_raise_reference_compatible_exceptions():
    coords = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    rust = rust_tri.Triangulation(coords)
    reference = REF.Triangulation(coords)

    assert_same_exception_type_name(
        lambda: rust.get_vertices((99,)),
        lambda: reference.get_vertices((99,)),
    )
    assert_same_exception_type_name(
        lambda: rust.containing((0, 99)),
        lambda: reference.containing((0, 99)),
    )
    assert_same_exception_type_name(
        lambda: rust.volume((0, 1, 99)),
        lambda: reference.volume((0, 1, 99)),
    )
    assert_same_exception_type_name(
        lambda: rust.point_in_circumcircle(99, (0, 1, 2)),
        lambda: reference.point_in_cicumcircle(99, (0, 1, 2), reference.default_transform),
    )
    assert_same_exception_type_name(
        lambda: list(rust.faces(vertices={99})),
        lambda: list(reference.faces(vertices={99})),
    )
    assert_same_exception_type_name(
        lambda: rust.add_point((0.1, 0.1), simplex=(0, 99)),
        lambda: reference.add_point((0.1, 0.1), simplex=(0, 99)),
    )
    assert_same_exception_type_name(
        lambda: rust.get_reduced_simplex((0.1, 0.1), (0, 99)),
        lambda: reference.get_reduced_simplex(np.array((0.1, 0.1)), (0, 99)),
    )


def test_dimension_mismatches_raise_value_error():
    coords = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    rust = rust_tri.Triangulation(coords)
    reference = REF.Triangulation(coords)

    assert_same_exception_type_name(
        lambda: rust.locate_point((0.1, 0.2, 0.3)),
        lambda: reference.locate_point((0.1, 0.2, 0.3)),
    )
    assert_same_exception_type_name(
        lambda: rust.point_in_simplex((0.1, 0.2, 0.3), (0, 1, 2)),
        lambda: reference.point_in_simplex(np.array((0.1, 0.2, 0.3)), (0, 1, 2)),
    )
    assert_same_exception_type_name(
        lambda: rust.get_reduced_simplex((0.1, 0.2, 0.3), (0, 1, 2)),
        lambda: reference.get_reduced_simplex(np.array((0.1, 0.2, 0.3)), (0, 1, 2)),
    )


def test_module_point_in_simplex_matches_reference_errors_on_degenerate_inputs():
    assert_same_exception_type_name(
        lambda: rust_tri.fast_2d_point_in_simplex((0.5, 0.5), [(0, 0), (1, 1), (2, 2)]),
        lambda: REF.fast_2d_point_in_simplex((0.5, 0.5), [(0, 0), (1, 1), (2, 2)]),
    )
    assert_same_exception_type_name(
        lambda: rust_tri.point_in_simplex((0.5, 0.5), [(0, 0), (1, 1), (2, 2)]),
        lambda: REF.point_in_simplex((0.5, 0.5), [(0, 0), (1, 1), (2, 2)]),
    )
    assert_same_exception_type_name(
        lambda: rust_tri.point_in_simplex(
            np.array([0.1, 0.1, 0.0]),
            np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0]]),
        ),
        lambda: REF.point_in_simplex(
            np.array([0.1, 0.1, 0.0]),
            np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0]]),
        ),
    )


def test_tiny_triangle_point_in_simplex_matches_reference():
    triangle = np.array([[0.0, 0.0], [1e-18, 0.0], [0.0, 1e-18]])
    point = np.array([2e-19, 2e-19])

    assert rust_tri.fast_2d_point_in_simplex(point, triangle) == REF.fast_2d_point_in_simplex(
        point, triangle
    )
    assert rust_tri.point_in_simplex(point, triangle) == REF.point_in_simplex(point, triangle)

    tri = rust_tri.Triangulation(triangle)
    ref_tri = REF.Triangulation(triangle)
    assert tuple(tri.locate_point(point)) == tuple(ref_tri.locate_point(point))


def test_simplex_volume_in_embedding_matches_reference_edge_case():
    assert_same_exception_type_name(
        lambda: rust_tri.simplex_volume_in_embedding([[0.0, 0.0], [1.0, 0.0]]),
        lambda: REF.simplex_volume_in_embedding([[0.0, 0.0], [1.0, 0.0]]),
    )
    assert rust_tri.simplex_volume_in_embedding(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    ) == pytest.approx(REF.simplex_volume_in_embedding([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))


def test_orientation_matches_reference_at_large_scale():
    face = [
        [-1.41794610203929e-141, 7.406648259110687e188, -5.742753819358816e155],
        [1.2173221166511304e-141, -5.3814286111832665e188, -7.462986421855193e155],
        [-2.601736071958286e-141, -6.554531137710295e188, -3.6204403844520214e154],
    ]
    origin = [0.0, 0.0, 0.0]
    assert rust_tri.orientation(face, origin) == REF.orientation(face, origin)


def test_circumsphere_handles_tiny_4d_simplex():
    simplex = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1e-16, 0.0, 0.0, 0.0],
            [0.0, 1e-16, 0.0, 0.0],
            [0.0, 0.0, 1e-16, 0.0],
            [0.0, 0.0, 0.0, 1e-16],
        ]
    )

    center, radius = rust_tri.circumsphere(simplex)
    ref_center, ref_radius = REF.circumsphere(simplex)
    assert_points_close(center, ref_center, atol=1e-20)
    assert radius == pytest.approx(ref_radius)


def test_degenerate_4d_circumsphere_returns_nan_like_reference():
    simplex = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )

    center, radius = rust_tri.circumsphere(simplex)
    ref_center, ref_radius = REF.circumsphere(simplex)
    assert np.isnan(radius)
    assert np.isnan(ref_radius)
    assert np.array(center, dtype=float).shape == np.array(ref_center, dtype=float).shape
    assert np.all(
        np.isnan(np.array(center, dtype=float))
        | np.isinf(np.array(ref_center, dtype=float))
        | np.isnan(np.array(ref_center, dtype=float))
    )


def test_constructor_accepts_sized_custom_sequences():
    coords = Seq([Seq([0.0, 0.0]), Seq([1.0, 0.0]), Seq([0.0, 1.0]), Seq([1.0, 1.0])])
    rust = rust_tri.Triangulation(coords)
    reference = REF.Triangulation(coords)
    assert_triangulation_equal(rust, reference)


def test_constructor_raises_value_error_for_linearly_dependent_input():
    with pytest.raises(ValueError, match="Initial simplex has zero volumes"):
        rust_tri.Triangulation([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])


def test_transform_requires_sized_2d_input():
    tri = rust_tri.Triangulation([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    reference = REF.Triangulation([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

    transform = ((x for x in row) for row in ((1.0, 0.0), (0.0, 1.0)))
    assert_same_exception_type_name(
        lambda: tri.circumscribed_circle((0, 1, 2), transform=transform),
        lambda: reference.circumscribed_circle(
            (0, 1, 2), ((x for x in row) for row in ((1.0, 0.0), (0.0, 1.0)))
        ),
    )


def test_random_cross_validation_4d():
    rng = np.random.default_rng(2468)
    coords = rng.random((8, 4))
    rust = rust_tri.Triangulation(coords[:5])
    reference = REF.Triangulation(coords[:5])

    for point in coords[5:]:
        rust_deleted, rust_added = rust.add_point(point)
        ref_deleted, ref_added = reference.add_point(point)
        assert as_simplex_set(rust_deleted) == as_simplex_set(ref_deleted)
        assert as_simplex_set(rust_added) == as_simplex_set(ref_added)
        assert_triangulation_equal(rust, reference)


def test_public_bowyer_watson_is_exposed():
    tri = rust_tri.Triangulation([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    assert hasattr(tri, "bowyer_watson")
