"""Comprehensive tests for the Rust-powered Learner1D."""

from __future__ import annotations

import math

import numpy as np
import pytest
from adaptive_triangulation import Learner1D


def sin10(x: float) -> float:
    return math.sin(10 * x)


def parabola(x: float) -> float:
    return x**2


def vector_fn(x: float) -> list[float]:
    return [math.sin(x), math.cos(x)]


class IntentionalLossError(ValueError):
    pass


def make_learner(bounds=(0.0, 1.0), points=()):
    learner = Learner1D(bounds=bounds)
    for x, y in points:
        learner.tell(x, y)
    return learner


def unit_interval_learner() -> Learner1D:
    return make_learner(points=((0.0, 0.0), (1.0, 1.0)))


def assert_points_close(points, expected) -> None:
    assert len(points) == len(expected)
    for point, expected_point in zip(sorted(points), expected, strict=True):
        assert point == pytest.approx(expected_point)


def bad_loss(_xs, _ys):
    raise IntentionalLossError


def string_loss(_xs, _ys):
    return "not a float"


@pytest.mark.parametrize("bounds", [(0.0, 1.0), (-1.0, 2.0)])
def test_create(bounds) -> None:
    learner = make_learner(bounds=bounds)
    assert learner.npoints == 0
    assert learner.bounds == bounds


def test_invalid_bounds() -> None:
    with pytest.raises(ValueError, match=r"bounds\[0\] must be strictly less than bounds\[1\]"):
        Learner1D(bounds=(1.0, 0.0))


def test_tell_counts_unique_points() -> None:
    learner = make_learner()
    learner.tell(0.0, 0.0)
    assert learner.npoints == 1
    learner.tell(1.0, 1.0)
    assert learner.npoints == 2
    learner.tell(1.0, 99.0)
    assert learner.npoints == 2
    assert learner.data[1.0] == 1.0


def test_loss_starts_infinite() -> None:
    assert math.isinf(make_learner().loss())


@pytest.mark.parametrize(
    ("points", "expected"),
    [
        (((0.0, 0.0), (1.0, 1.0)), math.sqrt(2)),
        (((0.0, 0.0), (1.0, 0.0)), 1.0),
    ],
)
def test_loss_with_two_points(points, expected: float) -> None:
    assert make_learner(points=points).loss() == pytest.approx(expected)


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (1, [0.5]),
        (3, [0.25, 0.5, 0.75]),
    ],
)
def test_ask_subdivides_interval(n: int, expected: list[float]) -> None:
    points, improvements = unit_interval_learner().ask(n, tell_pending=False)
    assert len(improvements) == n
    assert_points_close(points, expected)


@pytest.mark.parametrize(
    ("bounds", "n", "expected"),
    [
        ((0.0, 1.0), 3, [0.0, 0.5, 1.0]),
        ((0.0, 1.0), 5, [0.0, 0.25, 0.5, 0.75, 1.0]),
        ((-1.0, 1.0), 2, [-1.0, 1.0]),
    ],
)
def test_empty_ask_returns_linspace(bounds, n: int, expected: list[float]) -> None:
    points, improvements = make_learner(bounds=bounds).ask(n, tell_pending=False)
    assert all(math.isinf(improvement) for improvement in improvements)
    assert_points_close(points, expected)


def test_tell_many_incremental_path() -> None:
    learner = make_learner()
    xs = [0.0, 0.5, 1.0]
    ys = [0.0, 0.25, 1.0]
    learner.tell_many(xs, ys)
    assert learner.npoints == 3


@pytest.mark.parametrize("force", [False, True])
def test_tell_many_force_rebuild(force) -> None:
    learner = make_learner()
    xs = [i / 10 for i in range(11)]
    ys = [x**2 for x in xs]
    learner.tell_many(xs, ys, force=force)
    assert learner.npoints == 11
    assert not math.isinf(learner.loss())


def test_tell_many_large_batch_triggers_rebuild() -> None:
    learner = unit_interval_learner()
    xs = [0.2, 0.4, 0.6, 0.8, 0.5]
    ys = [x**2 for x in xs]
    learner.tell_many(xs, ys)
    assert learner.npoints == 7


def test_scalar_output_sets_vdim() -> None:
    assert unit_interval_learner().vdim == 1


def test_vector_output_supports_ask_and_to_numpy() -> None:
    learner = make_learner(points=((0.0, [1.0, 2.0]), (0.5, [2.0, 3.0]), (1.0, [3.0, 4.0])))
    assert learner.vdim == 2
    points, _ = learner.ask(1, tell_pending=False)
    assert len(points) == 1
    xs, ys = learner.to_numpy()
    assert xs.shape == (3,)
    assert ys.shape == (3, 2)


def test_tell_pending_and_tell_clears_pending() -> None:
    learner = unit_interval_learner()
    learner.tell_pending(0.5)
    assert 0.5 in learner.pending_points
    learner.tell(0.5, 0.25)
    assert 0.5 not in learner.pending_points


@pytest.mark.parametrize(("tell_pending", "expected_pending"), [(True, 2), (False, 0)])
def test_ask_pending_mode(tell_pending, expected_pending: int) -> None:
    learner = unit_interval_learner()
    points, _ = learner.ask(2, tell_pending=tell_pending)
    assert len(points) == 2
    assert len(learner.pending_points) == expected_pending


def test_pending_splits_intervals() -> None:
    learner = unit_interval_learner()
    points_without_pending, _ = learner.ask(1, tell_pending=False)
    learner.tell_pending(0.5)
    points_with_pending, _ = learner.ask(1, tell_pending=False)
    assert points_without_pending != points_with_pending


def test_run_with_n_points() -> None:
    learner = make_learner(bounds=(-1.0, 1.0))
    evaluated = learner.run(sin10, n_points=20, batch_size=5)
    assert evaluated == 20
    assert learner.npoints == 20


@pytest.mark.parametrize("batch_size", [1, 10])
def test_run_with_goal(batch_size: int) -> None:
    learner = make_learner(bounds=(-1.0, 1.0))
    evaluated = learner.run(parabola, goal=0.01, batch_size=batch_size)
    assert learner.loss() <= 0.01
    assert evaluated > 0


def test_run_batch_size_1() -> None:
    assert make_learner().run(parabola, n_points=10, batch_size=1) == 10


def test_run_no_goal_or_npoints_returns_immediately() -> None:
    assert make_learner().run(parabola) == 0


def test_to_numpy_empty() -> None:
    xs, ys = make_learner().to_numpy()
    assert len(xs) == 0
    assert len(ys) == 0


@pytest.mark.parametrize(
    "points",
    [
        ((0.0, 0.0), (0.5, 0.25), (1.0, 1.0)),
        ((1.0, 1.0), (0.0, 0.0), (0.5, 0.25)),
    ],
)
def test_to_numpy_scalar(points) -> None:
    xs, ys = make_learner(points=points).to_numpy()
    np.testing.assert_array_equal(xs, [0.0, 0.5, 1.0])
    np.testing.assert_array_equal(ys, [0.0, 0.25, 1.0])


def test_to_numpy_sorted_order() -> None:
    learner = make_learner()
    learner.tell(1.0, 1.0)
    learner.tell(0.0, 0.0)
    learner.tell(0.5, 0.25)
    xs, _ = learner.to_numpy()
    assert list(xs) == [0.0, 0.5, 1.0]


def test_remove_unfinished_clears_pending_without_changing_loss() -> None:
    learner = unit_interval_learner()
    loss_before = learner.loss()
    learner.ask(5, tell_pending=True)
    assert len(learner.pending_points) == 5
    learner.remove_unfinished()
    assert len(learner.pending_points) == 0
    assert learner.loss() == pytest.approx(loss_before)


def test_single_point_loss_stays_infinite() -> None:
    learner = make_learner(points=((0.0, 0.0),))
    assert learner.npoints == 1
    assert math.isinf(learner.loss())


def test_tell_at_bounds_yields_finite_loss() -> None:
    learner = unit_interval_learner()
    assert learner.npoints == 2
    assert not math.isinf(learner.loss())


def test_ask_zero() -> None:
    points, improvements = make_learner().ask(0)
    assert points == []
    assert improvements == []


@pytest.mark.parametrize("ask_n", [1, 10, 25])
def test_many_points(ask_n: int) -> None:
    learner = make_learner()
    xs = [i / 100 for i in range(101)]
    ys = [math.sin(x * 10) for x in xs]
    learner.tell_many(xs, ys, force=True)
    assert learner.npoints == 101
    points, _ = learner.ask(ask_n, tell_pending=False)
    assert len(points) == ask_n


def test_scale_recompute_triggers_on_large_y_change() -> None:
    learner = unit_interval_learner()
    loss_before = learner.loss()
    learner.tell(0.5, 100.0)
    assert learner.loss() != loss_before


@pytest.mark.parametrize("y_right", [1.0, 100.0])
def test_uniform_loss_callback(y_right: float) -> None:
    def my_uniform(xs, _ys):
        return xs[1] - xs[0]

    learner = Learner1D(bounds=(0.0, 1.0), loss_per_interval=my_uniform)
    learner.tell(0.0, 0.0)
    learner.tell(1.0, y_right)
    points, _ = learner.ask(1, tell_pending=False)
    assert points[0] == pytest.approx(0.5)


def test_custom_loss_run() -> None:
    def my_loss(xs, _ys):
        return abs(xs[1] - xs[0])

    learner = Learner1D(bounds=(0.0, 1.0), loss_per_interval=my_loss)
    assert learner.run(parabola, n_points=20, batch_size=5) == 20


def test_data_and_intervals_properties() -> None:
    learner = make_learner(points=((0.0, 0.0), (0.5, 0.25), (1.0, 1.0)))
    assert learner.data == {0.0: 0.0, 0.5: 0.25, 1.0: 1.0}

    intervals = learner.intervals()
    assert intervals == pytest.approx([(0.0, 0.5, intervals[0][2]), (0.5, 1.0, intervals[1][2])])
    assert all(loss >= 0 for _, _, loss in intervals)


def test_finite_loss_rounding() -> None:
    learner = make_learner()
    xs = [i * 0.1 for i in range(11)]
    ys = [math.sin(x) for x in xs]
    learner.tell_many(xs, ys, force=True)
    assert learner.loss() >= 0
    assert math.isfinite(learner.loss())


def test_tiny_interval() -> None:
    learner = make_learner()
    learner.tell(0.5, 0.0)
    learner.tell(0.5 + 1e-15, 1.0)
    assert learner.loss() >= 0


def test_convergence() -> None:
    learner = make_learner(bounds=(-1.0, 1.0), points=((-1.0, sin10(-1.0)), (1.0, sin10(1.0))))
    losses = [learner.loss()]
    for _ in range(50):
        points, _ = learner.ask(1, tell_pending=True)
        for point in points:
            learner.tell(point, sin10(point))
        losses.append(learner.loss())
    assert losses[-1] < losses[0]


def test_batch_run_matches_sequential() -> None:
    learner_seq = make_learner()
    learner_batch = make_learner()

    assert learner_seq.run(parabola, n_points=50, batch_size=1) == 50
    assert learner_batch.run(parabola, n_points=50, batch_size=10) == 50
    assert learner_seq.loss() < 0.1
    assert learner_batch.loss() < 0.1


def test_full_adaptive_loop() -> None:
    learner = make_learner(bounds=(-1.0, 1.0))
    learner.run(sin10, goal=0.05, batch_size=5)
    assert learner.npoints > 0
    xs, ys = learner.to_numpy()
    assert all(xs[i] <= xs[i + 1] for i in range(len(xs) - 1))
    for x, y in zip(xs, ys, strict=True):
        assert y == pytest.approx(sin10(x))


def test_out_of_bounds_points_are_stored_counted_and_ignored_in_loss() -> None:
    learner = make_learner(bounds=(-1.0, 1.0), points=((-1.0, 0.0), (1.0, 0.0)))
    loss_before = learner.loss()
    learner.tell(-10.0, 999.0)
    learner.tell(5.0, 42.0)
    assert learner.npoints == 4
    assert learner.data[-10.0] == 999.0
    assert learner.data[5.0] == 42.0
    assert learner.loss() == pytest.approx(loss_before)


def test_oob_does_not_affect_neighbors() -> None:
    learner = make_learner(bounds=(-1.0, 1.0))
    learner.tell(-10.0, 0.0)
    learner.tell(-1.0, 0.0)
    learner.tell(1.0, 0.0)
    intervals = learner.intervals()
    assert len(intervals) == 1
    assert intervals[0][:2] == (-1.0, 1.0)


def test_oob_duplicate_ignored() -> None:
    learner = make_learner(bounds=(-1.0, 1.0))
    learner.tell(5.0, 1.0)
    learner.tell(5.0, 99.0)
    assert learner.npoints == 1
    assert learner.data[5.0] == 1.0


@pytest.mark.parametrize("callback", [bad_loss, string_loss])
def test_callback_failure_returns_infinity(callback) -> None:
    learner = Learner1D(bounds=(0.0, 1.0), loss_per_interval=callback)
    learner.tell(0.0, 0.0)
    learner.tell(1.0, 1.0)
    assert math.isinf(learner.loss())


def test_loss_defaults_to_real() -> None:
    learner = unit_interval_learner()
    assert learner.loss() == learner.loss(real=True)


@pytest.mark.parametrize("pending_x", [0.25, 0.5])
def test_loss_real_false_accounts_for_pending(pending_x: float) -> None:
    learner = unit_interval_learner()
    real_loss = learner.loss(real=True)
    learner.tell_pending(pending_x)
    assert learner.loss(real=False) < real_loss
