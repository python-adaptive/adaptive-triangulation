# adaptive-triangulation

Fast N-dimensional Delaunay triangulation with incremental point insertion, written in Rust with Python bindings.

Designed as a drop-in replacement for the triangulation module in [`adaptive`](https://github.com/python-adaptive/adaptive).

## Features

- **Bowyer-Watson algorithm** for incremental Delaunay point insertion
- **N-dimensional** — works in 2D, 3D, and higher dimensions
- **Transform support** — anisotropic metrics for non-uniform meshing
- **Fast geometry** — optimized circumsphere, point-in-simplex, and volume calculations
- **Python bindings** via PyO3 — drop-in compatible with `adaptive.learner.triangulation`

## Installation

```bash
pip install adaptive-triangulation
```

## Quick Start

```python
from adaptive_triangulation import Triangulation

# Create from initial points (need at least dim+1 points in general position)
coords = [[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5]]
tri = Triangulation(coords)

# Add points incrementally (returns deleted and added simplices)
deleted, added = tri.add_point([0.3, 0.7])

# Query the triangulation
simplex = tri.locate_point([0.25, 0.25])
vol = tri.volume(simplex)
hull = tri.hull()

# Standalone geometry functions
from adaptive_triangulation import circumsphere, point_in_simplex, volume

center, radius = circumsphere([[0, 0], [1, 0], [0, 1]])
inside = point_in_simplex([0.2, 0.2], [[0, 0], [1, 0], [0, 1]])
v = volume([[0, 0], [1, 0], [0, 1]])
```

## Use with adaptive

```python
# In adaptive, set as the triangulation backend:
from adaptive import LearnerND
# adaptive will automatically use adaptive-triangulation if installed
learner = LearnerND(func, bounds=[(-1, 1), (-1, 1)])
```

## API Reference

### `Triangulation(coords)`

N-dimensional Delaunay triangulation with incremental point insertion.

- `add_point(point, simplex=None, transform=None)` — Add a point, optionally specifying containing simplex and metric transform. Returns `(deleted_simplices, added_simplices)`.
- `locate_point(point)` — Find the simplex containing the point. O(N) scan.
- `volume(simplex)` — Volume of a simplex (given as tuple of vertex indices).
- `hull` — Set of vertex indices on the convex hull.
- `simplices` — Set of all simplices (as sorted tuples of vertex indices).
- `vertices` — All vertex coordinates as numpy array.
- `dim` — Spatial dimension.
- `faces(dim=None)` — Iterator over faces of given dimension.
- `containing(face)` — Simplices containing a given face.
- `circumscribed_circle(simplex, transform=None)` — Circumsphere with optional metric transform.
- `point_in_circumcircle(pt_index, simplex, transform=None)` — Check if point is inside circumsphere.
- `get_vertices(simplex)` — Get vertex coordinates for a simplex.
- `get_reduced_simplex(point, simplex)` — Get sub-face of simplex containing point.

### Standalone functions

- `circumsphere(points)` — Circumscribed sphere of a simplex.
- `point_in_simplex(point, simplex, eps=1e-8)` — Check if point is inside simplex.
- `fast_2d_point_in_simplex(point, simplex, eps=1e-8)` — Optimized 2D version.
- `fast_2d_circumcircle(points)` — Optimized 2D circumcircle.
- `fast_3d_circumsphere(points)` — Optimized 3D circumsphere.
- `volume(simplex)` — Volume of a simplex from vertex coordinates.
- `simplex_volume_in_embedding(vertices)` — Volume in higher-dimensional embedding.
- `orientation(face, origin)` — Orientation of a face w.r.t. a point.

## License

BSD-3-Clause