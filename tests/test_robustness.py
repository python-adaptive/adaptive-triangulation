"""Degenerate-input robustness: failed insertions must never corrupt state.

The Python reference corrupts itself on these inputs (it mutates before its
volume-conservation assert, and orphans vertices when the assert cannot see
the problem), so unlike test_triangulation.py nothing here cross-validates
against it. The contract pinned here is documented in src/tolerances.rs:

- every insertion either succeeds, or raises (ValueError/AssertionError)
  leaving the triangulation exactly as it was;
- a successful insertion always connects the new vertex;
- vertices never lose their last simplex;
- the internal indexes stay consistent (reference_invariant).
"""

from __future__ import annotations

import adaptive_triangulation as at
import numpy as np
import pytest


def mixed_scale_points(seed: int, dim: int) -> np.ndarray:
    """A unit-sized cloud plus a tiny far-away cluster: forces ~1e7 aspect
    ratio slivers, the worst known case for floating-point predicates."""
    rng = np.random.default_rng(seed)
    big = rng.random((20, dim))
    tiny = 100.0 + 1e-5 * rng.random((20, dim))
    pts = np.vstack([big, tiny])
    rng.shuffle(pts)
    return pts


def near_duplicate_points(seed: int, dim: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.random((15, dim))
    dup = base[rng.integers(0, 15, 15)] + 1e-12 * rng.random((15, dim))
    pts = np.vstack([base, dup])
    rng.shuffle(pts)
    return pts


def simplex_count_per_vertex(tri) -> list[int]:
    return [len(tri.vertex_to_simplices[i]) for i in range(len(tri.vertices))]


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("generate", [mixed_scale_points, near_duplicate_points])
def test_degenerate_insertions_never_corrupt_state(generate, seed, dim):
    pts = generate(seed, dim)
    tri = at.Triangulation(pts[: dim + 2].tolist())
    # scipy/QHull may merge near-degenerate construction points (the
    # reference does the same); those stay unconnected by design.
    construction_orphans = {
        i for i in range(len(tri.vertices)) if len(tri.vertex_to_simplices[i]) == 0
    }

    for p in pts[dim + 2 :]:
        n_vertices = len(tri.vertices)
        simplices_before = {tuple(s) for s in tri.simplices}
        try:
            tri.add_point(p.tolist())
        except (ValueError, AssertionError):
            # Rejections must be atomic: no vertex leaked, no simplex changed.
            assert len(tri.vertices) == n_vertices
            assert {tuple(s) for s in tri.simplices} == simplices_before
        else:
            # Successful insertions must connect the new vertex.
            assert len(tri.vertices) == n_vertices + 1
            assert len(tri.vertex_to_simplices[n_vertices]) > 0

    assert tri.reference_invariant()
    counts = simplex_count_per_vertex(tri)
    orphans = {i for i, count in enumerate(counts) if count == 0}
    assert orphans == construction_orphans


def test_empty_cavity_insertion_is_repaired_not_orphaned():
    # Regression (mixed-scale seed 2): the point lies inside a huge bridge
    # sliver whose floating-point circumsphere test reports it outside, so
    # the cavity came back empty and add_point "succeeded" with (set(), set()),
    # leaving the new vertex unconnected. The exact-predicate rebuild must
    # either connect it or reject the insertion outright.
    pts = mixed_scale_points(2, 2)
    dim = 2
    tri = at.Triangulation(pts[: dim + 2].tolist())
    for p in pts[dim + 2 :]:
        n_vertices = len(tri.vertices)
        try:
            deleted, added = tri.add_point(p.tolist())
        except (ValueError, AssertionError):
            assert len(tri.vertices) == n_vertices
            continue
        assert added, "successful insertion must create at least one simplex"
        assert len(tri.vertex_to_simplices[len(tri.vertices) - 1]) > 0
    assert tri.reference_invariant()


def test_well_conditioned_insertions_never_reject():
    # The repair path must not change behavior for ordinary inputs: random
    # unit-cube sweeps go through the reference-compatible fast path only.
    for dim in (2, 3):
        rng = np.random.default_rng(99)
        pts = rng.random((60, dim))
        tri = at.Triangulation(pts[: dim + 2].tolist())
        for p in pts[dim + 2 :]:
            tri.add_point(p.tolist())  # must not raise
        assert tri.reference_invariant()
        assert all(len(tri.vertex_to_simplices[i]) > 0 for i in range(len(tri.vertices)))
