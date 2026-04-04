# adaptive-triangulation

Fast N-dimensional Delaunay triangulation in Rust with Python bindings.

[![PyPI](https://img.shields.io/pypi/v/adaptive-triangulation)](https://pypi.org/project/adaptive-triangulation/)
[![Python](https://img.shields.io/pypi/pyversions/adaptive-triangulation)](https://pypi.org/project/adaptive-triangulation/)
[![License](https://img.shields.io/github/license/python-adaptive/adaptive-triangulation)](LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/python-adaptive/adaptive-triangulation/ci.yml)](https://github.com/python-adaptive/adaptive-triangulation/actions)
[![Downloads](https://img.shields.io/pypi/dm/adaptive-triangulation)](https://pypi.org/project/adaptive-triangulation/)

`adaptive-triangulation` is a Rust/PyO3 implementation of the incremental Bowyer-Watson
algorithm for Delaunay triangulation. It is designed as a fast drop-in triangulation backend for
`adaptive`, while remaining useful as a standalone computational geometry package for 2D, 3D, and
higher-dimensional point sets.

## Installation

```bash
pip install adaptive-triangulation
```

## Quick Start

```python
import adaptive_triangulation as at
import numpy as np

# 2D triangulation
points = [[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5]]
tri = at.Triangulation(points)
print(f"Simplices: {tri.simplices}")

# Incremental point insertion
deleted, added = tri.add_point((0.25, 0.75))
print(f"Deleted simplices: {deleted}")
print(f"Added simplices: {added}")

# Circumsphere queries
for simplex in tri.simplices:
    center, radius = tri.circumscribed_circle(simplex)
    print(simplex, center, radius)

# Works in any dimension
points_3d = np.random.rand(10, 3).tolist()
tri3d = at.Triangulation(points_3d)
print(f"3D simplices: {len(tri3d.simplices)}")
```

Standalone geometry helpers are also available:

```python
import adaptive_triangulation as at

triangle = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
center, radius = at.circumsphere(triangle)
inside = at.point_in_simplex([0.25, 0.25], triangle)
area = at.volume(triangle)
```

## Performance

The table below compares release-mode `adaptive-triangulation` against the Python reference
implementation used in the test suite on the same machine. Each case builds a triangulation from a
minimal seed simplex and inserts the remaining points incrementally.

| Case | Rust (`maturin develop --release`) | Python reference | Speedup |
| --- | ---: | ---: | ---: |
| 2D, 1,000 points | 38.5 ms | 668.1 ms | 17.4x |
| 2D, 5,000 points | 259.8 ms | 8547.2 ms | 32.9x |
| 3D, 500 points | 133.4 ms | 5570.9 ms | 41.8x |

Absolute timings will vary by machine, but the release build consistently outperforms the pure
Python reference by a wide margin on construction and incremental insertion workloads.

## API Reference

### Module attributes

- `__version__`: Package version sourced from `Cargo.toml`.

### `Triangulation`

- `Triangulation(coords)`: Build an N-dimensional Delaunay triangulation from points in general position.
- `add_point(point, simplex=None, transform=None)`: Insert one point and return `(deleted_simplices, added_simplices)`.
- `add_simplex(simplex)`: Insert a simplex directly into the triangulation state.
- `delete_simplex(simplex)`: Remove a simplex from the triangulation state.
- `locate_point(point)`: Return the simplex containing a query point, or an empty tuple when none is found.
- `get_reduced_simplex(point, simplex, eps=1e-8)`: Reduce a simplex to the smallest face containing the point.
- `circumscribed_circle(simplex, transform=None)`: Compute the circumsphere center and radius for a simplex.
- `point_in_circumcircle(pt_index, simplex, transform=None)`: Check whether a stored vertex lies inside a simplex circumsphere.
- `point_in_cicumcircle(pt_index, simplex, transform=None)`: Legacy alias for `point_in_circumcircle`.
- `point_in_simplex(point, simplex, eps=1e-8)`: Test whether a point lies inside a simplex.
- `volume(simplex)`: Compute the volume of one simplex by vertex index.
- `volumes()`: Return the volumes of all simplices in the triangulation.
- `faces(dim=None, simplices=None, vertices=None)`: Iterate over faces filtered by dimension, simplices, or vertices.
- `containing(face)`: Return the simplices that contain a face.
- `get_vertices(indices)`: Fetch vertex coordinates for a sequence of vertex indices.
- `bowyer_watson(pt_index, containing_simplex=None, transform=None)`: Run one Bowyer-Watson insertion step for an existing vertex.
- `reference_invariant()`: Check internal consistency against the reference invariants used by the tests.
- `vertex_invariant(vertex)`: Compatibility placeholder that currently raises `NotImplementedError`.
- `convex_invariant(vertex)`: Compatibility placeholder that currently raises `NotImplementedError`.
- `vertices`: Vertex coordinates stored by the triangulation.
- `simplices`: Set of simplices represented as tuples of vertex indices.
- `vertex_to_simplices`: Reverse map from vertex index to incident simplices.
- `hull`: Set of vertex indices on the convex hull.
- `dim`: Spatial dimension of the triangulation.
- `default_transform`: Identity metric transform for the current dimension.

### Standalone functions

- `circumsphere(points)`: Compute the circumsphere of a simplex given explicit coordinates.
- `fast_2d_circumcircle(points)`: Fast circumcircle routine specialized for three 2D points.
- `fast_3d_circumsphere(points)`: Fast circumsphere routine specialized for four 3D points.
- `fast_3d_circumcircle(points)`: Alias for the 3D circumsphere helper.
- `point_in_simplex(point, simplex, eps=1e-8)`: Generic point-in-simplex predicate.
- `fast_2d_point_in_simplex(point, simplex, eps=1e-8)`: Fast 2D point-in-triangle predicate.
- `volume(simplex)`: Compute the volume of a simplex from coordinates.
- `simplex_volume_in_embedding(vertices)`: Compute simplex volume in a higher-dimensional embedding space.
- `orientation(face, origin)`: Return the orientation of a face relative to an origin point.
- `fast_norm(point)`: Compute the Euclidean norm of a point.

## License

BSD-3-Clause, matching the `adaptive` project. See [LICENSE](LICENSE).
