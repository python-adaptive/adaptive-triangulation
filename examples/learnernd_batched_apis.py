"""Measure the LearnerND speedup from `simplices_containing` and `default_loss`.

This monkeypatches adaptive (the PR #493 branch, with the Rust backend
active) to use the two batched APIs, the way the adaptive-side integration
would:

- ``tell_pending`` distributes a pending point over the simplices containing
  it with one ``tri.simplices_containing(point, simplex=hint)`` call instead
  of a Python loop of per-neighbour ``tri.point_in_simplex`` checks.
- The default loss is the Rust ``default_loss`` instead of the Python
  wrapper around ``simplex_volume_in_embedding``.

Requirements:
    pip install adaptive-triangulation
    pip install "adaptive @ git+https://github.com/python-adaptive/adaptive@refs/pull/493/head"
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import adaptive_triangulation as at
import numpy as np
from adaptive.learner import learnerND as lnd_mod
from adaptive.learner.learnerND import LearnerND

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

reference_default_loss = lnd_mod.default_loss


def ring_of_fire(xy: Sequence[float]) -> float:
    x, y = xy
    a, d = 0.2, 0.5
    return x + np.exp(-((x**2 + y**2 - d**2) ** 2) / a**4)


def sphere_of_fire(xyz: Sequence[float]) -> float:
    x, y, z = xyz
    a, d = 0.2, 0.5
    return x + np.exp(-((x**2 + y**2 + z**2 - d**2) ** 2) / a**4)


def _add_pending_to_simplex(self: LearnerND, point: tuple, simplex: tuple) -> tuple:
    """`_try_adding_pending_point_to_simplex` without the containment check
    (the caller already knows `simplex` contains `point`)."""
    if simplex not in self._subtriangulations:
        vertices = self.tri.get_vertices(simplex)
        self._subtriangulations[simplex] = self._triangulation_class(vertices)
    self._pending_to_simplex[point] = simplex
    return self._subtriangulations[simplex].add_point(point)


def tell_pending_batched(self: LearnerND, point: tuple, *, simplex: tuple | None = None) -> None:
    """tell_pending with the point_in_simplex loop replaced by one
    `simplices_containing` call (the `simplex` argument becomes the hint)."""
    point = tuple(point)
    if not self.inside_bounds(point):
        return
    self.pending_points.add(point)
    if self.tri is None:
        return
    containing = self.tri.simplices_containing(point, simplex=simplex)
    for simpl in containing:
        _, to_add = _add_pending_to_simplex(self, point, simpl)
        if to_add is None:
            continue
        self._update_subsimplex_losses(simpl, to_add)


def run(
    func: Callable[[Sequence[float]], float],
    bounds: list[tuple[float, float]],
    n_points: int,
    *,
    rust_loss: bool,
    batched_tell: bool,
) -> tuple[float, LearnerND]:
    lnd_mod.default_loss = at.default_loss if rust_loss else reference_default_loss
    learner = LearnerND(func, bounds=bounds)
    if batched_tell:
        learner.tell_pending = tell_pending_batched.__get__(learner)
    t0 = time.perf_counter()
    for _ in range(n_points):
        points, _ = learner.ask(1)
        for p in points:
            learner.tell(p, func(p))
    elapsed = time.perf_counter() - t0
    return elapsed, learner


CONFIGS = [
    ("baseline (Rust backend, PR #493)", False, False),
    ("+ rust default_loss", True, False),
    ("+ simplices_containing tell_pending", False, True),
    ("+ both", True, True),
]

for name, func, bounds, n in [
    ("2D ring_of_fire", ring_of_fire, [(-1, 1), (-1, 1)], 3000),
    ("3D sphere_of_fire", sphere_of_fire, [(-1, 1), (-1, 1), (-1, 1)], 1500),
]:
    print(f"\n{name}, {n} points:")
    baseline = None
    base_points = None
    for label, rust_loss, batched_tell in CONFIGS:
        best = min(
            run(func, bounds, n, rust_loss=rust_loss, batched_tell=batched_tell)[0]
            for _ in range(3)
        )
        _, learner = run(func, bounds, n, rust_loss=rust_loss, batched_tell=batched_tell)
        pts = sorted(learner.data)
        if baseline is None:
            baseline, base_points = best, pts
        same = "identical points" if pts == base_points else "DIFFERENT POINTS"
        print(f"  {label:40s} {best:6.2f}s  ({baseline / best:4.2f}x, {same})")
