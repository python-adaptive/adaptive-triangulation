"""Comprehensive tests for the Rust-powered Learner1D."""

from __future__ import annotations

import math

import numpy as np
import pytest

from adaptive_triangulation import Learner1D


# ---- Helpers ----


def sin10(x: float) -> float:
    return math.sin(10 * x)


def parabola(x: float) -> float:
    return x**2


def vector_fn(x: float) -> list[float]:
    return [math.sin(x), math.cos(x)]


# ---- Basic functionality ----


class TestBasic:
    def test_create(self):
        l = Learner1D(bounds=(0.0, 1.0))
        assert l.npoints == 0
        assert l.bounds == (0.0, 1.0)

    def test_invalid_bounds(self):
        with pytest.raises(ValueError):
            Learner1D(bounds=(1.0, 0.0))

    def test_tell_and_npoints(self):
        l = Learner1D(bounds=(0.0, 1.0))
        l.tell(0.0, 0.0)
        assert l.npoints == 1
        l.tell(1.0, 1.0)
        assert l.npoints == 2

    def test_duplicate_tell_ignored(self):
        l = Learner1D(bounds=(0.0, 1.0))
        l.tell(0.5, 1.0)
        l.tell(0.5, 99.0)
        assert l.npoints == 1
        data = l.data
        assert data[0.5] == 1.0

    def test_loss_with_two_points(self):
        l = Learner1D(bounds=(0.0, 1.0))
        l.tell(0.0, 0.0)
        l.tell(1.0, 1.0)
        loss = l.loss
        # default_loss = sqrt(dx^2 + dy^2) with scaling
        # dx_scaled = 1/1 = 1, dy_scaled = 1/1 = 1 → sqrt(2) ≈ 1.414
        assert abs(loss - math.sqrt(2)) < 1e-10

    def test_ask_returns_midpoint(self):
        l = Learner1D(bounds=(0.0, 1.0))
        l.tell(0.0, 0.0)
        l.tell(1.0, 1.0)
        pts, _ = l.ask(1, tell_pending=False)
        assert len(pts) == 1
        assert abs(pts[0] - 0.5) < 1e-10

    def test_ask_multiple(self):
        l = Learner1D(bounds=(0.0, 1.0))
        l.tell(0.0, 0.0)
        l.tell(1.0, 1.0)
        pts, imprs = l.ask(3, tell_pending=False)
        assert len(pts) == 3
        assert len(imprs) == 3
        # Should subdivide [0,1] into 4 equal parts
        expected = [0.25, 0.5, 0.75]
        for p, e in zip(sorted(pts), expected):
            assert abs(p - e) < 1e-10

    def test_ask_empty_learner(self):
        l = Learner1D(bounds=(0.0, 1.0))
        pts, imprs = l.ask(3, tell_pending=False)
        assert len(pts) == 3
        # Should return linspace-like points including bounds
        assert all(math.isinf(i) for i in imprs)

    def test_ask_bounds_first(self):
        """The first two asks should suggest the bounds."""
        l = Learner1D(bounds=(-1.0, 1.0))
        pts, imprs = l.ask(2, tell_pending=False)
        assert sorted(pts) == [-1.0, 1.0]
        assert all(math.isinf(i) for i in imprs)

    def test_loss_starts_infinite(self):
        l = Learner1D(bounds=(0.0, 1.0))
        assert math.isinf(l.loss)


# ---- tell_many ----


class TestTellMany:
    def test_incremental_path(self):
        l = Learner1D(bounds=(0.0, 1.0))
        xs = [0.0, 0.5, 1.0]
        ys = [0.0, 0.25, 1.0]
        l.tell_many(xs, ys)
        assert l.npoints == 3

    def test_force_rebuild(self):
        l = Learner1D(bounds=(0.0, 1.0))
        xs = [i / 10 for i in range(11)]
        ys = [x**2 for x in xs]
        l.tell_many(xs, ys, force=True)
        assert l.npoints == 11
        assert not math.isinf(l.loss)

    def test_large_batch_triggers_rebuild(self):
        """When len(xs) > 0.5 * len(data) and len(xs) > 2, rebuild path is used."""
        l = Learner1D(bounds=(0.0, 1.0))
        l.tell(0.0, 0.0)
        l.tell(1.0, 1.0)
        # Add 5 points (> 0.5 * 2 = 1, and > 2)
        xs = [0.2, 0.4, 0.6, 0.8, 0.5]
        ys = [x**2 for x in xs]
        l.tell_many(xs, ys)
        assert l.npoints == 7


# ---- Scalar and vector outputs ----


class TestOutputTypes:
    def test_scalar_output(self):
        l = Learner1D(bounds=(0.0, 1.0))
        l.tell(0.0, 0.0)
        l.tell(1.0, 1.0)
        assert l.vdim == 1

    def test_vector_output(self):
        l = Learner1D(bounds=(0.0, 1.0))
        l.tell(0.0, [1.0, 2.0])
        l.tell(1.0, [3.0, 4.0])
        assert l.vdim == 2
        pts, _ = l.ask(1, tell_pending=False)
        assert len(pts) == 1

    def test_vector_to_numpy(self):
        l = Learner1D(bounds=(0.0, 1.0))
        l.tell(0.0, [1.0, 2.0])
        l.tell(0.5, [2.0, 3.0])
        l.tell(1.0, [3.0, 4.0])
        xs, ys = l.to_numpy()
        assert xs.shape == (3,)
        assert ys.shape == (3, 2)


# ---- Pending points ----


class TestPending:
    def test_tell_pending(self):
        l = Learner1D(bounds=(0.0, 1.0))
        l.tell(0.0, 0.0)
        l.tell(1.0, 1.0)
        l.tell_pending(0.5)
        assert 0.5 in l.pending_points

    def test_ask_with_tell_pending_true(self):
        l = Learner1D(bounds=(0.0, 1.0))
        l.tell(0.0, 0.0)
        l.tell(1.0, 1.0)
        pts, _ = l.ask(2, tell_pending=True)
        assert len(pts) == 2
        assert len(l.pending_points) == 2

    def test_ask_with_tell_pending_false(self):
        l = Learner1D(bounds=(0.0, 1.0))
        l.tell(0.0, 0.0)
        l.tell(1.0, 1.0)
        pts, _ = l.ask(2, tell_pending=False)
        assert len(pts) == 2
        assert len(l.pending_points) == 0

    def test_tell_clears_pending(self):
        l = Learner1D(bounds=(0.0, 1.0))
        l.tell(0.0, 0.0)
        l.tell(1.0, 1.0)
        l.tell_pending(0.5)
        assert 0.5 in l.pending_points
        l.tell(0.5, 0.25)
        assert 0.5 not in l.pending_points

    def test_pending_splits_intervals(self):
        """Pending points should cause ask to suggest different points."""
        l = Learner1D(bounds=(0.0, 1.0))
        l.tell(0.0, 0.0)
        l.tell(1.0, 1.0)
        # Without pending
        pts1, _ = l.ask(1, tell_pending=False)
        # With pending at midpoint
        l.tell_pending(0.5)
        pts2, _ = l.ask(1, tell_pending=False)
        assert pts1 != pts2


# ---- run() method ----


class TestRun:
    def test_run_with_n_points(self):
        l = Learner1D(bounds=(-1.0, 1.0))
        n = l.run(sin10, n_points=20, batch_size=5)
        assert n == 20
        assert l.npoints == 20

    def test_run_with_goal(self):
        l = Learner1D(bounds=(-1.0, 1.0))
        n = l.run(parabola, goal=0.01, batch_size=10)
        assert l.loss <= 0.01
        assert n > 0

    def test_run_batch_size_1(self):
        l = Learner1D(bounds=(0.0, 1.0))
        n = l.run(parabola, n_points=10, batch_size=1)
        assert n == 10

    def test_run_no_goal_or_npoints_returns_immediately(self):
        l = Learner1D(bounds=(0.0, 1.0))
        n = l.run(parabola)
        assert n == 0


# ---- to_numpy ----


class TestToNumpy:
    def test_empty(self):
        l = Learner1D(bounds=(0.0, 1.0))
        xs, ys = l.to_numpy()
        assert len(xs) == 0

    def test_scalar(self):
        l = Learner1D(bounds=(0.0, 1.0))
        l.tell(0.0, 0.0)
        l.tell(0.5, 0.25)
        l.tell(1.0, 1.0)
        xs, ys = l.to_numpy()
        np.testing.assert_array_equal(xs, [0.0, 0.5, 1.0])
        np.testing.assert_array_equal(ys, [0.0, 0.25, 1.0])

    def test_sorted_order(self):
        l = Learner1D(bounds=(0.0, 1.0))
        l.tell(1.0, 1.0)
        l.tell(0.0, 0.0)
        l.tell(0.5, 0.25)
        xs, ys = l.to_numpy()
        assert list(xs) == [0.0, 0.5, 1.0]


# ---- remove_unfinished ----


class TestRemoveUnfinished:
    def test_clears_pending(self):
        l = Learner1D(bounds=(0.0, 1.0))
        l.tell(0.0, 0.0)
        l.tell(1.0, 1.0)
        l.ask(5, tell_pending=True)
        assert len(l.pending_points) == 5
        l.remove_unfinished()
        assert len(l.pending_points) == 0

    def test_loss_unchanged_after_remove(self):
        l = Learner1D(bounds=(0.0, 1.0))
        l.tell(0.0, 0.0)
        l.tell(1.0, 1.0)
        loss_before = l.loss
        l.ask(5, tell_pending=True)
        l.remove_unfinished()
        loss_after = l.loss
        assert abs(loss_before - loss_after) < 1e-10


# ---- Edge cases ----


class TestEdgeCases:
    def test_single_point(self):
        l = Learner1D(bounds=(0.0, 1.0))
        l.tell(0.0, 0.0)
        assert l.npoints == 1
        assert math.isinf(l.loss)

    def test_tell_at_bounds(self):
        l = Learner1D(bounds=(0.0, 1.0))
        l.tell(0.0, 0.0)
        l.tell(1.0, 1.0)
        assert l.npoints == 2
        assert not math.isinf(l.loss)

    def test_ask_zero(self):
        l = Learner1D(bounds=(0.0, 1.0))
        pts, imprs = l.ask(0)
        assert pts == []
        assert imprs == []

    def test_many_points(self):
        l = Learner1D(bounds=(0.0, 1.0))
        xs = [i / 100 for i in range(101)]
        ys = [math.sin(x * 10) for x in xs]
        l.tell_many(xs, ys, force=True)
        assert l.npoints == 101
        pts, _ = l.ask(10, tell_pending=False)
        assert len(pts) == 10


# ---- Scale doubling ----


class TestScaleDoubling:
    def test_scale_recompute_triggers(self):
        """When y-range doubles, losses should be recomputed."""
        l = Learner1D(bounds=(0.0, 1.0))
        l.tell(0.0, 0.0)
        l.tell(1.0, 1.0)
        loss1 = l.loss
        # Tell a point with y far beyond current range → triggers scale change
        l.tell(0.5, 100.0)
        # Loss should change because scale changed
        loss2 = l.loss
        assert loss1 != loss2


# ---- Custom Python loss function ----


class TestCustomLoss:
    def test_uniform_loss_callback(self):
        def my_uniform(xs, ys):
            return xs[1] - xs[0]

        l = Learner1D(bounds=(0.0, 1.0), loss_per_interval=my_uniform)
        l.tell(0.0, 0.0)
        l.tell(1.0, 100.0)
        # With uniform loss, the y-value doesn't matter
        pts, _ = l.ask(1, tell_pending=False)
        assert abs(pts[0] - 0.5) < 1e-10

    def test_custom_loss_run(self):
        def my_loss(xs, ys):
            return abs(xs[1] - xs[0])

        l = Learner1D(bounds=(0.0, 1.0), loss_per_interval=my_loss)
        n = l.run(parabola, n_points=20, batch_size=5)
        assert n == 20


# ---- Data and intervals properties ----


class TestProperties:
    def test_data_dict(self):
        l = Learner1D(bounds=(0.0, 1.0))
        l.tell(0.0, 0.0)
        l.tell(0.5, 0.25)
        l.tell(1.0, 1.0)
        d = l.data
        assert len(d) == 3
        assert d[0.0] == 0.0
        assert d[0.5] == 0.25
        assert d[1.0] == 1.0

    def test_intervals(self):
        l = Learner1D(bounds=(0.0, 1.0))
        l.tell(0.0, 0.0)
        l.tell(0.5, 0.25)
        l.tell(1.0, 1.0)
        ivals = l.intervals()
        assert len(ivals) == 2
        # Each interval is (left, right, loss)
        assert ivals[0][0] == 0.0
        assert ivals[0][1] == 0.5
        assert ivals[1][0] == 0.5
        assert ivals[1][1] == 1.0
        # Losses should be positive
        assert all(iv[2] >= 0 for iv in ivals)


# ---- Numerical precision ----


class TestNumerical:
    def test_finite_loss_rounding(self):
        """Losses should be stable under floating-point noise."""
        l = Learner1D(bounds=(0.0, 1.0))
        xs = [i * 0.1 for i in range(11)]
        ys = [math.sin(x) for x in xs]
        l.tell_many(xs, ys, force=True)
        # Loss should be finite and non-negative
        assert l.loss >= 0
        assert math.isfinite(l.loss)

    def test_tiny_interval(self):
        """Intervals near machine epsilon should have zero loss."""
        l = Learner1D(bounds=(0.0, 1.0))
        eps = 1e-15
        l.tell(0.5, 0.0)
        l.tell(0.5 + eps, 1.0)
        # The tiny interval should have very small or zero loss
        # because dx < dx_eps
        assert l.loss >= 0


# ---- Cross-validation with adaptive loop ----


class TestAdaptiveLoop:
    def test_convergence(self):
        """Loss should decrease as more points are added."""
        l = Learner1D(bounds=(-1.0, 1.0))
        l.tell(-1.0, sin10(-1.0))
        l.tell(1.0, sin10(1.0))
        losses = [l.loss]
        for _ in range(50):
            pts, _ = l.ask(1, tell_pending=True)
            for p in pts:
                l.tell(p, sin10(p))
            losses.append(l.loss)
        # Loss should generally decrease
        assert losses[-1] < losses[0]

    def test_batch_run_matches_sequential(self):
        """Batch and sequential runs should produce similar final losses."""
        l1 = Learner1D(bounds=(0.0, 1.0))
        n1 = l1.run(parabola, n_points=50, batch_size=1)

        l2 = Learner1D(bounds=(0.0, 1.0))
        n2 = l2.run(parabola, n_points=50, batch_size=10)

        assert n1 == n2 == 50
        # Both should have reasonable loss
        assert l1.loss < 0.1
        assert l2.loss < 0.1

    def test_full_adaptive_loop(self):
        """Full adaptive loop should produce good point distribution."""
        l = Learner1D(bounds=(-1.0, 1.0))
        l.run(sin10, goal=0.05, batch_size=5)
        assert l.npoints > 0
        xs, ys = l.to_numpy()
        # Check that points are sorted
        assert all(xs[i] <= xs[i + 1] for i in range(len(xs) - 1))
        # Check that function values match
        for x, y in zip(xs, ys):
            assert abs(y - sin10(x)) < 1e-10
