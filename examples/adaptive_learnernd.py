"""Using adaptive-triangulation with adaptive's LearnerND.

Drop-in replacement for adaptive's built-in Triangulation class,
providing 5× speedup for LearnerND and 7× vs Learner2D at 5K points.

Requirements:
    pip install adaptive adaptive-triangulation
"""

import time

import adaptive_triangulation as at
import numpy as np
from adaptive.learner import learnerND as lnd_mod
from adaptive.learner.learnerND import LearnerND


def ring_of_fire(xy: tuple[float, float]) -> float:
    """A 2D function with a ring-shaped feature — good test for adaptive sampling."""
    x, y = xy
    a, d = 0.2, 0.5
    return x + np.exp(-((x**2 + y**2 - d**2) ** 2) / a**4)


bounds = [(-1, 1), (-1, 1)]
n_points = 2000

# --- Baseline: pure Python triangulation ---
t0 = time.perf_counter()
learner_py = LearnerND(ring_of_fire, bounds=bounds)
for _ in range(n_points):
    points, _ = learner_py.ask(1)
    for p in points:
        learner_py.tell(p, ring_of_fire(p))
t_python = time.perf_counter() - t0
print(f"Python triangulation: {t_python:.2f}s ({n_points} points)")

# --- Rust triangulation: monkey-patch the module ---
# Replace both the Triangulation class AND the standalone geometry functions
lnd_mod.Triangulation = at.Triangulation
lnd_mod.circumsphere = at.circumsphere
lnd_mod.simplex_volume_in_embedding = at.simplex_volume_in_embedding
lnd_mod.point_in_simplex = at.point_in_simplex

t0 = time.perf_counter()
learner_rust = LearnerND(ring_of_fire, bounds=bounds)
for _ in range(n_points):
    points, _ = learner_rust.ask(1)
    for p in points:
        learner_rust.tell(p, ring_of_fire(p))
t_rust = time.perf_counter() - t0
print(f"Rust triangulation:   {t_rust:.2f}s ({n_points} points)")
print(f"Speedup: {t_python / t_rust:.1f}×")
