"""Fast N-dimensional Delaunay triangulation with incremental point insertion.

This package provides a Rust-accelerated implementation of N-dimensional
Delaunay triangulation using the Bowyer-Watson algorithm, designed as
an optional backend for the `adaptive` package.

Usage::

    from adaptive_triangulation import Triangulation

    # Create triangulation from initial points (must be >= dim+1 points)
    coords = [[0, 0], [1, 0], [0, 1], [1, 1]]
    tri = Triangulation(coords)

    # Add points incrementally
    deleted, added = tri.add_point([0.5, 0.5])

    # Query
    simplex = tri.locate_point([0.3, 0.3])
    v = tri.volume(simplex)
"""

from __future__ import annotations

from adaptive_triangulation._rust import (
    Triangulation,
    __version__ as __version__,
    circumsphere,
    fast_2d_circumcircle,
    fast_2d_point_in_simplex,
    fast_3d_circumcircle,
    fast_3d_circumsphere,
    fast_norm,
    orientation,
    point_in_simplex,
    simplex_volume_in_embedding,
    volume,
)

__all__ = [
    "Triangulation",
    "__version__",
    "circumsphere",
    "fast_2d_circumcircle",
    "fast_2d_point_in_simplex",
    "fast_3d_circumcircle",
    "fast_3d_circumsphere",
    "fast_norm",
    "orientation",
    "point_in_simplex",
    "simplex_volume_in_embedding",
    "volume",
]
