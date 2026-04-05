"""Benchmark: Rust vs Python triangulation at various point counts.

Compares standalone triangulation performance (incremental insertion)
across different dimensions and point counts.
"""

import time

import adaptive_triangulation as at
import numpy as np

# Try importing Python reference
try:
    from adaptive.learner.triangulation import Triangulation as PyTriangulation

    HAS_PYTHON_REF = True
except ImportError:
    HAS_PYTHON_REF = False
    print("Install 'adaptive' to compare: pip install adaptive")


def generate_points(n: int, dim: int, seed: int = 42) -> list[tuple[float, ...]]:
    """Generate a mix of uniform + clustered points (simulates adaptive refinement)."""
    rng = np.random.default_rng(seed)
    n_uniform = n // 3
    n_clustered = n - n_uniform
    uniform = rng.uniform(-1, 1, (n_uniform, dim))
    centers = rng.uniform(-0.5, 0.5, (5, dim))
    cluster_idx = rng.integers(0, 5, n_clustered)
    clustered = centers[cluster_idx] + rng.normal(0, 0.1, (n_clustered, dim))
    all_pts = np.vstack([uniform, clustered])
    return [tuple(p) for p in all_pts]


def bench_rust(points: list[tuple[float, ...]], dim: int) -> float:
    """Benchmark Rust triangulation: build from first dim+1 points, then add_point."""
    coords = points[: dim + 1]
    tri = at.Triangulation(coords)
    t0 = time.perf_counter()
    for pt in points[dim + 1 :]:
        try:
            tri.add_point(pt)
        except ValueError:
            pass  # duplicate point
    return time.perf_counter() - t0


def bench_python(points: list[tuple[float, ...]], dim: int) -> float:
    """Benchmark Python triangulation."""
    if not HAS_PYTHON_REF:
        return float("nan")
    coords = points[: dim + 1]
    tri = PyTriangulation(coords)
    t0 = time.perf_counter()
    for pt in points[dim + 1 :]:
        try:
            tri.add_point(pt)
        except ValueError:
            pass
    return time.perf_counter() - t0


configs = [
    (2, [500, 1000, 2000, 5000]),
    (3, [200, 500, 1000, 2000]),
]

print(f"{'Dim':>3} {'N':>6} {'Rust':>10} {'Python':>10} {'Speedup':>8}")
print("-" * 45)


for dim, sizes in configs:
    for n in sizes:
        points = generate_points(n, dim)
        # Best of 3
        t_rust = min(bench_rust(points, dim) for _ in range(3))
        t_py = min(bench_python(points, dim) for _ in range(3))
        speedup = t_py / t_rust if t_rust > 0 else float("nan")
        print(f"{dim:>3} {n:>6} {t_rust:>9.3f}s {t_py:>9.3f}s {speedup:>7.1f}×")
