# adaptive-triangulation

[![PyPI version](https://img.shields.io/pypi/v/adaptive-triangulation.svg)](https://pypi.org/project/adaptive-triangulation/)
[![Python versions](https://img.shields.io/pypi/pyversions/adaptive-triangulation.svg)](https://pypi.org/project/adaptive-triangulation/)
[![CI](https://github.com/python-adaptive/adaptive-triangulation/actions/workflows/ci.yml/badge.svg)](https://github.com/python-adaptive/adaptive-triangulation/actions)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)

Fast N-dimensional Delaunay triangulation in Rust with Python bindings (PyO3).
Drop-in replacement for [adaptive](https://github.com/python-adaptive/adaptive)'s `Triangulation` class — **30-300× faster** standalone, **~3.3×** end-to-end in `LearnerND` (where adaptive's own Python code dominates). Used automatically by adaptive ≥ 1.5 when installed.

## Performance

Measured with the scripts in [`examples/`](examples/) against adaptive 1.5.0, best of 3.
Absolute times are machine-dependent; the ratios are representative.

### Standalone triangulation (incremental insertion)
| Case | Rust | Python | Speedup |
|---|---:|---:|---:|
| 2D, 1K pts | 18 ms | 730 ms | **41×** |
| 2D, 5K pts | 138 ms | 14,442 ms | **105×** |
| 3D, 500 pts | 32 ms | 3,037 ms | **94×** |
| 3D, 2K pts | 150 ms | 44,531 ms | **297×** |

### LearnerND integration (end-to-end, `ring_of_fire` 2D)
| N pts | Learner2D (scipy) | LearnerND (Python) | LearnerND (Rust) |
|---|---:|---:|---:|
| 1,000 | 0.23 s | 0.50 s | **0.16 s** |
| 2,000 | 0.90 s | 1.01 s | **0.32 s** |
| 5,000 | 5.69 s | 2.57 s | **0.79 s** |

LearnerND + Rust is **3.3× faster** than LearnerND + Python, and **7× faster** than Learner2D at 5K points.
The end-to-end ratio is smaller than the standalone one because adaptive's own Python-side loss machinery dominates once the triangulation is fast.

### Batched LearnerND APIs (not yet wired into adaptive)

`simplices_containing` and `default_loss` move two of the remaining `LearnerND`
Python hot loops into Rust. Wired in the way a future adaptive release would use
them ([`examples/learnernd_batched_apis.py`](examples/learnernd_batched_apis.py)),
they add **1.17×** (2D, 3000 pts) to **1.40×** (3D, 1500 pts) on top of the
table above, while sampling identical points.

## Installation

```bash
pip install adaptive-triangulation
```

Requires a Rust toolchain for building from source.
Pre-built wheels are available for common platforms via CI.

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

Since adaptive 1.5.0 this package is detected and used automatically — no code changes needed:

```bash
pip install "adaptive[rust]"
```

Per learner, the backend can be selected explicitly with
`LearnerND(..., triangulation_backend="auto" | "python" | "rust")`, or globally
with the `ADAPTIVE_TRIANGULATION_BACKEND` environment variable.

For adaptive < 1.5.0, monkey-patch the module instead:

```python
import adaptive_triangulation as at
from adaptive.learner import learnerND as lnd_mod
from adaptive.learner.learnerND import LearnerND

# Replace both the class and standalone functions
lnd_mod.Triangulation = at.Triangulation
lnd_mod.circumsphere = at.circumsphere
lnd_mod.simplex_volume_in_embedding = at.simplex_volume_in_embedding
lnd_mod.point_in_simplex = at.point_in_simplex

# Now use LearnerND as normal — including neighbor-aware losses
# like curvature_loss_function()
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
tri.simplices_containing(point)        # All simplices containing a point, in one call
                                       # instead of a point_in_simplex loop; pass a known
                                       # containing simplex via simplex=... to skip the
                                       # locate step, or restrict with candidates=...
tri.point_in_circumcircle(pt, simplex) # Circumcircle test
tri.bowyer_watson(pt_index)            # Direct Bowyer-Watson
tri.get_opposing_vertices(simplex)     # Facet neighbours' opposite vertices
tri.get_simplices_attached_to_points(simplex)  # Facet-sharing neighbours
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
    default_loss,              # LearnerND's default loss (embedded simplex volume),
                               # signature-compatible with loss_per_simplex
    orientation,               # Face orientation
)
```

## Examples

- [`examples/basic_usage.py`](examples/basic_usage.py) — Core API walkthrough
- [`examples/adaptive_learnernd.py`](examples/adaptive_learnernd.py) — LearnerND integration with timing
- [`examples/benchmark_vs_python.py`](examples/benchmark_vs_python.py) — Standalone benchmarks across dimensions

## Robustness on degenerate input

Point sets that mix widely separated coordinate scales force sliver simplices that no floating-point predicate can handle reliably.
Unlike the Python reference (which can corrupt its state on such input), this implementation validates every insertion before mutating: a cavity that cannot be re-triangulated is first repaired with exact predicates (Shewchuk's, via the [`robust`](https://crates.io/crates/robust) crate), and if even that fails the insertion raises with the triangulation untouched, so callers can skip the point and continue.
Well-conditioned inputs behave identically to the reference.
The full policy is documented in [`src/tolerances.rs`](src/tolerances.rs).

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
