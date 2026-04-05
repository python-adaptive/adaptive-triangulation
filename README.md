# adaptive-triangulation

[![PyPI version](https://img.shields.io/pypi/v/adaptive-triangulation.svg)](https://pypi.org/project/adaptive-triangulation/)
[![Python versions](https://img.shields.io/pypi/pyversions/adaptive-triangulation.svg)](https://pypi.org/project/adaptive-triangulation/)
[![CI](https://github.com/python-adaptive/adaptive-triangulation/actions/workflows/ci.yml/badge.svg)](https://github.com/python-adaptive/adaptive-triangulation/actions)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)

Fast N-dimensional Delaunay triangulation in Rust with Python bindings (PyO3).
Drop-in replacement for [adaptive](https://github.com/python-adaptive/adaptive)'s `Triangulation` class — **5-99× faster**.

## Performance

### Standalone triangulation (incremental insertion)
| Case | Rust | Python | Speedup |
|---|---:|---:|---:|
| 2D, 1K pts | 38.5 ms | 668 ms | **17×** |
| 2D, 5K pts | 260 ms | 8,547 ms | **33×** |
| 3D, 500 pts | 133 ms | 5,571 ms | **42×** |

### LearnerND integration (end-to-end, `ring_of_fire` 2D)
| N pts | Learner2D (scipy) | LearnerND (Python) | LearnerND (Rust) |
|---|---:|---:|---:|
| 1,000 | 0.34 s | 0.91 s | **0.23 s** |
| 2,000 | 1.17 s | 1.80 s | **0.38 s** |
| 5,000 | 6.99 s | 4.57 s | **0.99 s** |

LearnerND + Rust is **5× faster** than LearnerND + Python, and **7× faster** than Learner2D at 5K points.

## Installation

```bash
pip install adaptive-triangulation
```

Requires a Rust toolchain for building from source. Pre-built wheels are available for common platforms via CI.

## Quick start

```python
from adaptive_triangulation import Triangulation

# Build a 2D triangulation
tri = Triangulation([(0, 0), (1, 0), (0, 1), (1, 1)])

# Insert points incrementally (Bowyer-Watson)
deleted, added = tri.add_point((0.5, 0.5))

# Query properties
print(len(tri.simplices))     # number of triangles
print(tri.dim)                # 2
print(tri.reference_invariant())  # True
```

## Usage with adaptive's LearnerND

This is a drop-in replacement for `adaptive`'s built-in triangulation.
Monkey-patch the module to use Rust triangulation everywhere:

```python
import adaptive_triangulation as at
from adaptive.learner import learnerND as lnd_mod
from adaptive.learner.learnerND import LearnerND

# Replace both the class and standalone functions
lnd_mod.Triangulation = at.Triangulation
lnd_mod.circumsphere = at.circumsphere
lnd_mod.simplex_volume_in_embedding = at.simplex_volume_in_embedding
lnd_mod.point_in_simplex = at.point_in_simplex

# Now use LearnerND as normal — it's 5× faster
learner = LearnerND(my_function, bounds=[(-1, 1), (-1, 1)])
```

See [`examples/adaptive_learnernd.py`](examples/adaptive_learnernd.py) for a full working example with timing comparison.

## API

### `Triangulation` class

```python
tri = Triangulation(coords)           # Build from initial points
tri.add_point(point)                   # Incremental insertion → (deleted, added)
tri.locate_point(point)                # Find containing simplex
tri.circumscribed_circle(simplex)      # → (center, radius)
tri.volume(simplex)                    # Simplex volume
tri.volumes()                          # All simplex volumes
tri.point_in_simplex(point, simplex)   # Containment test
tri.point_in_circumcircle(pt, simplex) # Circumcircle test
tri.bowyer_watson(pt_index)            # Direct Bowyer-Watson
tri.reference_invariant()              # Consistency check
```

**Properties:** `vertices`, `simplices`, `vertex_to_simplices`, `hull`, `dim`, `default_transform`

### Standalone functions

```python
from adaptive_triangulation import (
    circumsphere,              # General circumsphere
    fast_2d_circumcircle,      # Optimized 2D
    fast_3d_circumsphere,      # Optimized 3D
    point_in_simplex,          # Containment test
    volume,                    # Simplex volume
    simplex_volume_in_embedding,  # Volume in embedding space
    orientation,               # Face orientation
)
```

## Examples

- [`examples/basic_usage.py`](examples/basic_usage.py) — Core API walkthrough
- [`examples/adaptive_learnernd.py`](examples/adaptive_learnernd.py) — LearnerND integration with timing
- [`examples/benchmark_vs_python.py`](examples/benchmark_vs_python.py) — Standalone benchmarks across dimensions

## Development

```bash
# Build (requires Rust toolchain)
pip install maturin
maturin develop --release

# Tests
cargo test                    # Rust tests
python -m pytest tests/ -v    # Python tests

# Linting
pre-commit run --all-files    # ruff, mypy, cargo fmt, cargo clippy
```

## License

BSD-3-Clause
