"""Basic usage of adaptive-triangulation.

Demonstrates core Triangulation API: construction, point insertion,
simplex queries, and geometry computations.
"""

import numpy as np

from adaptive_triangulation import Triangulation, circumsphere, volume

# Create a 2D triangulation from initial points
points = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
tri = Triangulation(points)

print(f"Vertices: {len(tri.vertices)}")
print(f"Simplices: {len(tri.simplices)}")
print(f"Dimension: {tri.dim}")

# Add a point incrementally (Bowyer-Watson insertion)
deleted, added = tri.add_point((0.5, 0.5))
print(f"\nAfter adding (0.5, 0.5):")
print(f"  Deleted simplices: {len(deleted)}")
print(f"  Added simplices: {len(added)}")
print(f"  Total simplices: {len(tri.simplices)}")

# Query geometry
for simplex in tri.simplices:
    verts = tri.get_vertices(simplex)
    center, radius = tri.circumscribed_circle(simplex)
    vol = tri.volume(simplex)
    print(f"  Simplex {simplex}: volume={vol:.4f}, circumradius={radius:.4f}")

# Standalone functions work on raw point arrays
pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.866]])
center, radius = circumsphere(pts)
vol = volume(pts)
print(f"\nStandalone: circumcenter={center}, radius={radius:.4f}, volume={vol:.4f}")

# Locate which simplex contains a query point
containing = tri.locate_point((0.3, 0.3))
print(f"\nPoint (0.3, 0.3) is in simplex: {containing}")

# Check invariants
assert tri.reference_invariant(), "Triangulation invariant violated!"
print("\nAll invariants passed ✓")