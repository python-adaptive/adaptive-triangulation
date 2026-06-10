"""Tests for the batched queries added for adaptive's LearnerND hot loops:
``Triangulation.simplices_containing`` and ``default_loss``."""

from __future__ import annotations

import adaptive_triangulation as rust_tri
import numpy as np
import pytest
from adaptive.learner.learnerND import default_loss as reference_default_loss

UNIT_SQUARE = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]


def brute_force_containing(tri, point) -> list[tuple[int, ...]]:
    return sorted(simplex for simplex in tri.simplices if tri.point_in_simplex(point, simplex))


def random_triangulation(dim: int, n_points: int, seed: int) -> rust_tri.Triangulation:
    rng = np.random.default_rng(seed)
    coords = np.vstack(
        [
            np.zeros(dim),
            np.eye(dim),
            np.ones(dim),
            rng.random((n_points, dim)),
        ]
    )
    return rust_tri.Triangulation(coords)


def test_simplices_containing_interior_point():
    tri = rust_tri.Triangulation(UNIT_SQUARE)
    point = [0.1, 0.2]
    result = tri.simplices_containing(point)
    assert result == [tuple(tri.locate_point(point))]


def test_simplices_containing_boundary_point_returns_both_simplices():
    tri = rust_tri.Triangulation(UNIT_SQUARE)
    # The midpoint of the shared diagonal lies in both triangles.
    shared = set.intersection(*(set(simplex) for simplex in tri.simplices))
    midpoint = np.mean([tri.vertices[i] for i in shared], axis=0)
    result = tri.simplices_containing(midpoint)
    assert result == sorted(tri.simplices)


def test_simplices_containing_outside_point_is_empty():
    tri = rust_tri.Triangulation(UNIT_SQUARE)
    assert tri.simplices_containing([2.0, 2.0]) == []


@pytest.mark.parametrize("dim", [2, 3])
def test_simplices_containing_hint_matches_unhinted(dim):
    tri = random_triangulation(dim, n_points=15, seed=500 + dim)
    rng = np.random.default_rng(600 + dim)
    for point in rng.random((20, dim)):
        expected = tri.simplices_containing(point)
        located = tuple(tri.locate_point(point))
        if located:
            assert tri.simplices_containing(point, simplex=located) == expected
        # A wrong or stale hint falls back to locating the point.
        wrong_hint = next(s for s in tri.simplices if s != located)
        assert tri.simplices_containing(point, simplex=wrong_hint) == expected
        assert tri.simplices_containing(point, simplex=()) == expected


def test_simplices_containing_respects_candidates():
    tri = rust_tri.Triangulation(UNIT_SQUARE)
    point = [0.1, 0.2]
    (containing,) = tri.simplices_containing(point)
    others = [simplex for simplex in tri.simplices if simplex != containing]
    assert tri.simplices_containing(point, candidates=[containing]) == [containing]
    assert tri.simplices_containing(point, candidates=others) == []
    assert tri.simplices_containing(point, candidates=[]) == []


def test_simplices_containing_eps_is_forwarded():
    tri = rust_tri.Triangulation(UNIT_SQUARE)
    # Nudge the diagonal's midpoint perpendicularly off the diagonal: with
    # the default eps it lies in one triangle, with a loose eps in both.
    shared = sorted(set.intersection(*(set(simplex) for simplex in tri.simplices)))
    a, b = (np.array(tri.vertices[i]) for i in shared)
    normal = np.array([-(b - a)[1], (b - a)[0]])
    point = (a + b) / 2 + 1e-6 * normal / np.linalg.norm(normal)
    assert len(tri.simplices_containing(point)) == 1
    assert tri.simplices_containing(point, eps=1e-3) == sorted(tri.simplices)


@pytest.mark.parametrize("dim", [2, 3, 4])
def test_simplices_containing_matches_brute_force(dim):
    tri = random_triangulation(dim, n_points=20, seed=100 + dim)
    rng = np.random.default_rng(200 + dim)
    probes = list(rng.random((20, dim))) + [tri.vertices[i] for i in range(5)]
    for point in probes:
        assert tri.simplices_containing(point) == brute_force_containing(tri, point)


@pytest.mark.parametrize("dim", [2, 3])
def test_simplices_containing_matches_tell_pending_neighbor_loop(dim):
    # The loop this method replaces: locate the containing simplex, gather
    # every simplex sharing a vertex with it, keep those containing the point.
    tri = random_triangulation(dim, n_points=15, seed=300 + dim)
    rng = np.random.default_rng(400 + dim)
    for point in rng.random((20, dim)):
        located = tuple(tri.locate_point(point))
        if not located:
            expected = []
        else:
            neighbors = set.union(*(tri.vertex_to_simplices[i] for i in located))
            expected = sorted(
                simplex for simplex in neighbors if tri.point_in_simplex(point, simplex)
            )
        assert tri.simplices_containing(point) == expected


def test_default_loss_matches_reference_scalar_values():
    rng = np.random.default_rng(42)
    simplex = rng.random((4, 3))
    values = rng.random(4)
    expected = reference_default_loss(simplex, values, 1.0)
    assert rust_tri.default_loss(simplex, values, 1.0) == pytest.approx(expected)


def test_default_loss_matches_reference_vector_values():
    rng = np.random.default_rng(43)
    simplex = rng.random((3, 2))
    values = rng.random((3, 2))
    expected = reference_default_loss(simplex, values, 1.0)
    assert rust_tri.default_loss(simplex, values, 1.0) == pytest.approx(expected)


def test_default_loss_accepts_plain_lists_and_omitted_value_scale():
    simplex = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    values = [0.0, 1.0, 2.0]
    expected = reference_default_loss(np.array(simplex), np.array(values), 1.0)
    assert rust_tri.default_loss(simplex, values) == pytest.approx(expected)


def test_default_loss_is_the_embedded_simplex_volume():
    simplex = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    values = [0.5, 0.5, 0.5]
    embedded = [(*x, y) for x, y in zip(simplex, values, strict=False)]
    expected = rust_tri.simplex_volume_in_embedding(embedded)
    assert rust_tri.default_loss(simplex, values) == pytest.approx(expected)
