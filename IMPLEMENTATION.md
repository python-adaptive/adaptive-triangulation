# Implementation Specification — adaptive-triangulation

## Overview

Standalone Rust package (PyO3/maturin) providing N-dimensional Delaunay triangulation
with Bowyer-Watson incremental insertion. This is the computational backend for
[python-adaptive](https://github.com/python-adaptive/adaptive)'s `LearnerND`.

The package must be a **drop-in replacement** for `adaptive.learner.triangulation`.
Every public method and function must match the Python API exactly in semantics.

## Architecture

```
src/
  lib.rs          — PyO3 module definition, standalone function wrappers
  geometry.rs     — Pure geometry: circumsphere, point-in-simplex, volume, orientation
  triangulation.rs — Triangulation struct: Bowyer-Watson, hull extension, point location
python/
  adaptive_triangulation/
    __init__.py   — Re-exports from _rust module
tests/
  test_triangulation.py  — Comprehensive Python tests
```

## Data Types

### Internal Rust types
- `PointIndex = usize` — index into `vertices` array
- `Simplex = Vec<PointIndex>` — always **sorted** ascending
- Coordinates: `Vec<f64>` for points, `Vec<Vec<f64>>` for point arrays
- Transform: `Option<Vec<Vec<f64>>>` — row-major N×N matrix

### Python-exposed types
- Points: `numpy.ndarray` (f64) — 1D for single point, 2D for arrays
- Simplices: `set` of `tuple[int, ...]` — frozen sets of sorted indices
- Transform: `Optional[numpy.ndarray]` — 2D f64 array

## Module: `geometry.rs`

### `circumsphere(pts: &[Vec<f64>]) -> (Vec<f64>, f64)`
Compute center and radius of circumscribed sphere through N+1 points in N dimensions.
- For 2D (3 points): use direct formula (fast path)
- For 3D (4 points): use direct formula (fast path)
- General: use Cayley-Menger determinant method

Reference — Python `circumsphere()` at line 129 of triangulation.py.

### `fast_2d_circumcircle(points: &[[f64; 2]; 3]) -> ([f64; 2], f64)`
Direct 2D circumcircle formula. Matches Python `fast_2d_circumcircle()`.

### `fast_3d_circumsphere(points: &[[f64; 3]; 4]) -> ([f64; 3], f64)`
Direct 3D circumsphere. Matches Python `fast_3d_circumcircle()` (note: Python name says "circle" but it's a sphere in 3D).

### `point_in_simplex(point: &[f64], simplex: &[Vec<f64>], eps: f64) -> bool`
Test if point lies inside simplex using barycentric coordinates (linear solve).
- 2D fast path via `fast_2d_point_in_simplex`
- General: solve `T * λ = p - v0`, check all `λ_i >= -eps` and `sum(λ) <= 1 + eps`

### `fast_2d_point_in_simplex(point: &[f64; 2], simplex: &[[f64; 2]; 3], eps: f64) -> bool`
Direct 2D barycentric coordinate test without linear algebra.

### `orientation(face: &[Vec<f64>], origin: &[f64]) -> i32`
Compute orientation of a face w.r.t. an origin point.
- Build matrix from face vectors relative to origin
- Use `slogdet` equivalent: sign of determinant
- Returns -1, 0, or 1
- If `|log_det| < -50`, treat as 0 (coplanar)

### `volume(vertices: &[Vec<f64>]) -> f64`
Simplex volume = `|det(edge_vectors)| / dim!`

### `simplex_volume_in_embedding(vertices: &[Vec<f64>]) -> f64`
Volume of simplex embedded in higher-dimensional space.
- 2D (triangle): Heron's formula from edge lengths
- General: Cayley-Menger determinant

### Helper: `fast_norm(v: &[f64]) -> f64`
Optimized vector norm. Inline for 2D/3D.

## Module: `triangulation.rs`

### `Triangulation` struct

```rust
pub struct Triangulation {
    vertices: Vec<Vec<f64>>,          // vertex coordinates
    simplices: HashSet<Vec<usize>>,   // set of sorted simplices
    vertex_to_simplices: Vec<HashSet<Vec<usize>>>,  // per-vertex simplex membership
    dim: usize,                       // spatial dimension
}
```

### Constructor: `new(coords: Vec<Vec<f64>>) -> Result<Self>`
1. Validate: non-empty, all same dimension, `dim >= 2`, at least `dim + 1` points
2. Check linear independence of initial points
3. Initialize with scipy-style Delaunay or manual initial simplex
4. For exactly `dim + 1` points: create single simplex directly
5. For more: compute initial Delaunay (can use convex hull approach or incremental)

### `add_point(&mut self, point: Vec<f64>, simplex: Option<Vec<usize>>, transform: Option<Vec<Vec<f64>>>) -> (HashSet<Vec<usize>>, HashSet<Vec<usize>>)`
Main entry point for adding a point.
1. Check point doesn't already exist (within tolerance)
2. Add to `self.vertices`
3. If no simplex provided, call `locate_point`
4. Determine if point is inside hull or outside:
   - Inside: call `bowyer_watson` directly
   - Outside: call `extend_hull` first, then `bowyer_watson`
5. Return `(deleted_simplices, added_simplices)`

The `transform` parameter is an optional N×N matrix. When provided:
- `circumscribed_circle` applies it to vertices before computing circumsphere
- `point_in_circumcircle` applies it to both the test point and the circumsphere computation

### `bowyer_watson(&mut self, pt_index: usize, containing: Option<Vec<usize>>, transform: &Option<Vec<Vec<f64>>>) -> (HashSet<Vec<usize>>, HashSet<Vec<usize>>)`
Modified Bowyer-Watson algorithm:
1. Find all "bad" simplices whose circumsphere contains the new point (BFS through face-adjacent neighbors)
2. Delete bad simplices
3. Find boundary faces of the hole (faces with multiplicity < 2)
4. Create new simplices connecting boundary faces to the new point
5. Skip degenerate simplices (volume < 1e-8)
6. Assert volume conservation: `sum(old_volumes) ≈ sum(new_volumes)`
7. Return `(deleted, added)`

**BFS neighbor filtering**: only traverse to simplices sharing a full face (dim points in common).

### `extend_hull(&mut self, pt_index: usize) -> (HashSet<Vec<usize>>, HashSet<Vec<usize>>)`
Add a point outside the current convex hull:
1. Find visible hull faces (face orientation check from new point vs interior)
2. Create new simplices connecting visible faces to new point
3. Return `(deleted={}, added=new_simplices)`

### `locate_point(&self, point: &[f64]) -> Option<Vec<usize>>`
Find which simplex contains the given point. O(N) scan.
Returns None if point is outside all simplices.

### `get_reduced_simplex(&self, point: &[f64], simplex: &[usize], eps: f64) -> Vec<usize>`
Find the lowest-dimensional sub-face of `simplex` that contains `point`.
Uses barycentric coordinates: the vertex indices where `λ_i >= -eps`.

### `circumscribed_circle(&self, simplex: &[usize], transform: &Option<Vec<Vec<f64>>>) -> (Vec<f64>, f64)`
Compute circumsphere of simplex, optionally in transformed space.
If transform is Some, multiply each vertex by the transform matrix before computing.

### `point_in_circumcircle(&self, pt_index: usize, simplex: &[usize], transform: &Option<Vec<Vec<f64>>>) -> bool`
Test if vertex `pt_index` is inside the circumsphere of `simplex`.
Uses eps = 1e-8: `dist(center, point) < radius * (1 + eps)`.
Transform applied to both circumsphere and test point.

### Properties / Getters
- `vertices(&self) -> &[Vec<f64>]`
- `simplices(&self) -> &HashSet<Vec<usize>>`
- `vertex_to_simplices(&self) -> &[HashSet<Vec<usize>>]`
- `dim(&self) -> usize`
- `hull(&self) -> HashSet<usize>` — vertices on convex hull (faces appearing in only 1 simplex)
- `faces(dim: Option<usize>, vertices: bool, simplices_set: Option<&HashSet<Vec<usize>>>) -> Iterator` — iterate over faces

### `containing(&self, face: &[usize]) -> HashSet<Vec<usize>>`
Return all simplices containing a given face (intersection of `vertex_to_simplices`).

### `volume(&self, simplex: &[usize]) -> f64`
Volume of a simplex using vertex coordinates.

### Invariant checks (debug)
- `reference_invariant(&self) -> bool` — verify `simplices` ↔ `vertex_to_simplices` consistency
- `vertex_invariant(&self) -> bool` — all simplex vertices exist
- `convex_invariant(&self) -> bool` — no overlapping simplices

### `add_simplex` / `delete_simplex`
Internal bookkeeping — maintain both `simplices` and `vertex_to_simplices`.

## PyO3 Bindings

### `PyTriangulation` class
Wraps `Triangulation`. All methods exposed to Python.

```python
class Triangulation:
    def __init__(self, coords: np.ndarray) -> None: ...
    
    def add_point(
        self,
        point: tuple[float, ...] | np.ndarray,
        simplex: tuple[int, ...] | None = None,
        transform: np.ndarray | None = None,
    ) -> tuple[set[tuple[int, ...]], set[tuple[int, ...]]]: ...
    
    @property
    def vertices(self) -> list[tuple[float, ...]]: ...
    
    @property
    def simplices(self) -> set[tuple[int, ...]]: ...
    
    @property
    def vertex_to_simplices(self) -> list[set[tuple[int, ...]]]: ...
    
    @property
    def hull(self) -> set[int]: ...
    
    @property
    def dim(self) -> int: ...
    
    def get_vertices(self, indices: tuple[int, ...]) -> list[tuple[float, ...]]: ...
    
    def locate_point(self, point: tuple[float, ...] | np.ndarray) -> tuple[int, ...] | None: ...
    
    def get_reduced_simplex(
        self, point: tuple[float, ...], simplex: tuple[int, ...], eps: float = 1e-8
    ) -> tuple[int, ...]: ...
    
    def circumscribed_circle(
        self, simplex: tuple[int, ...], transform: np.ndarray | None = None
    ) -> tuple[tuple[float, ...], float]: ...
    
    def point_in_circumcircle(
        self, pt_index: int, simplex: tuple[int, ...], transform: np.ndarray | None = None
    ) -> bool: ...
    
    def volume(self, simplex: tuple[int, ...]) -> float: ...
    
    def faces(
        self, dim: int | None = None, vertices: bool = False,
        simplices: set[tuple[int, ...]] | None = None,
    ) -> Iterator: ...
    
    def containing(self, face: tuple[int, ...]) -> set[tuple[int, ...]]: ...
    
    def reference_invariant(self) -> bool: ...
    
    @property
    def default_transform(self) -> np.ndarray: ...
    
    def point_in_simplex(self, point: tuple[float, ...], simplex: tuple[int, ...], eps: float = 1e-8) -> bool: ...
```

### Standalone functions (module-level)
All geometry functions exposed at module level matching the Python API.

## Testing Strategy

### Unit tests (`tests/test_triangulation.py`)
1. **Construction**: 2D, 3D, 4D triangulations with exact simplex counts
2. **Point insertion**: add points inside hull, on edges, on faces, outside hull
3. **Volume conservation**: total volume must be conserved after every insertion
4. **Transform support**: insertion with identity, scaling, rotation transforms
5. **Circumsphere**: verify center equidistant from all vertices, fast paths match general
6. **Point-in-simplex**: inside, outside, on face, on edge, on vertex
7. **Hull**: verify hull vertices for known geometries
8. **Faces**: enumerate faces of known triangulations
9. **locate_point**: find containing simplex for random points
10. **Invariants**: `reference_invariant` passes after every operation

### Cross-validation with Python
Test that Rust and Python implementations produce identical results for the same inputs.

## Performance Targets

The whole point is speed. Key optimizations:
- **No heap allocation in tight loops** — pre-allocate vectors, use `SmallVec` for simplices
- **Fast paths for 2D/3D** — specialized circumsphere, point-in-simplex
- **BFS with visited set** — avoid revisiting simplices in Bowyer-Watson
- **`HashSet<Vec<usize>>`** — consider `BTreeSet` or sorted arrays for simplices if hashing is slow
- **SIMD-friendly memory layout** — consider SoA for vertex storage

## Reference Code

The Python implementation in `adaptive/learner/triangulation.py` (676 lines) is the
authoritative reference. The existing Rust code on the `rust` branch (1194 lines total)
can be used as inspiration for data structures but should NOT be copied — start fresh
with modern Rust patterns.

## Environment

```bash
# Build
cd /tmp/adaptive-triangulation
maturin develop --release

# Test
export LD_LIBRARY_PATH=/nix/store/$(ls /nix/store | grep 'gcc.*lib$' | head -1)/lib:$LD_LIBRARY_PATH
source /tmp/adaptive-pr141/.venv/bin/activate
python -m pytest tests/ -x -v

# Maturin may need:
pip install maturin
```

## Deliverables

1. `src/geometry.rs` — all geometry primitives, fully implemented
2. `src/triangulation.rs` — full Triangulation struct with Bowyer-Watson
3. `src/lib.rs` — PyO3 module with all bindings
4. `tests/test_triangulation.py` — comprehensive test suite
5. Working `maturin develop --release` build
6. All tests passing

When complete, write results to `.claude/REPORT.md`.