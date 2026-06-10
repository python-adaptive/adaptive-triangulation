//! N-dimensional Delaunay triangulation with incremental Bowyer-Watson
//! point insertion.
//!
//! This is the pure-Rust core: it holds all triangulation state and
//! algorithms and never touches Python. The PyO3 surface (argument parsing,
//! result conversion, the proxy views, and the `Triangulation` Python class)
//! lives in [`crate::py`].

use std::collections::VecDeque;
use std::sync::{PoisonError, RwLock};

use rustc_hash::{FxHashMap, FxHashSet};
use thiserror::Error;

use crate::geometry::{self, GeometryError};
use crate::tolerances::{
    BARYCENTRIC_EPS, CIRCUMCIRCLE_RTOL, DEGENERATE_VOLUME_EPS, VOLUME_CONSERVATION_ATOL,
    VOLUME_CONSERVATION_RTOL,
};

/// Index of a vertex in [`Triangulation::vertices`].
pub type PointIndex = usize;
/// A simplex as a sorted list of vertex indices.
pub type Simplex = Vec<PointIndex>;

/// Errors from triangulation operations. Each variant maps onto the Python
/// exception type the reference implementation raises in the same situation.
#[derive(Debug, Error)]
pub enum TriangulationError {
    /// Maps to `ValueError`.
    #[error("{0}")]
    Value(String),
    /// Maps to `IndexError`.
    #[error("{0}")]
    Index(String),
    /// Maps to `RuntimeError`.
    #[error("{0}")]
    Runtime(String),
    /// Maps to `AssertionError` (the reference uses bare `assert`).
    #[error("{0}")]
    Assertion(String),
    /// Maps to `ValueError`.
    #[error(transparent)]
    Geometry(#[from] GeometryError),
}

pub(crate) fn index_out_of_range() -> TriangulationError {
    TriangulationError::Index("list index out of range".to_string())
}

fn validate_transform(
    transform: &Option<Vec<Vec<f64>>>,
    dim: usize,
) -> Result<(), TriangulationError> {
    let Some(transform) = transform else {
        return Ok(());
    };
    if transform.len() != dim || transform.iter().any(|row| row.len() != dim) {
        return Err(TriangulationError::Value(
            "Transform must be an N x N matrix".to_string(),
        ));
    }
    Ok(())
}

fn apply_transform(point: &[f64], transform: &[Vec<f64>]) -> Vec<f64> {
    let dim = point.len();
    let mut result = vec![0.0; dim];
    for (col, slot) in result.iter_mut().enumerate() {
        let mut value = 0.0;
        for row in 0..dim {
            value += point[row] * transform[row][col];
        }
        *slot = value;
    }
    result
}

/// Positions (`0..=dim`) of the simplex vertices kept after reducing against
/// the barycentric coordinates `alpha` (relative to vertex 0): vertices whose
/// coordinate exceeds `eps` survive, and an empty result means the point lies
/// outside the simplex. A single surviving position means the point coincides
/// with that vertex.
pub(crate) fn reduced_simplex_positions(alpha: &[f64], eps: f64) -> Vec<usize> {
    let sum_alpha = alpha.iter().sum::<f64>();
    if alpha.iter().any(|value| *value < -eps) || sum_alpha > 1.0 + eps {
        return Vec::new();
    }

    let mut positions: Vec<usize> = alpha
        .iter()
        .enumerate()
        .filter_map(|(idx, value)| (*value > eps).then_some(idx + 1))
        .collect();
    if sum_alpha < 1.0 - eps {
        positions.insert(0, 0);
    }
    positions
}

fn combinations(
    source: &[usize],
    k: usize,
    out: &mut Vec<Simplex>,
    current: &mut Vec<usize>,
    start: usize,
) {
    if current.len() == k {
        out.push(current.clone());
        return;
    }
    if k < current.len() || source.len().saturating_sub(start) < k - current.len() {
        return;
    }
    for idx in start..source.len() {
        current.push(source[idx]);
        combinations(source, k, out, current, idx + 1);
        current.pop();
    }
}

fn is_close(a: f64, b: f64) -> bool {
    // Symmetric variant of numpy.isclose (which scales by |b| only), so the
    // volume-conservation check cannot depend on argument order.
    let scale = a.abs().max(b.abs());
    (a - b).abs() <= VOLUME_CONSERVATION_ATOL + VOLUME_CONSERVATION_RTOL * scale
}

/// An N-dimensional Delaunay triangulation supporting incremental point
/// insertion via the Bowyer-Watson algorithm. Pure Rust; the Python surface
/// lives on [`crate::py::PyTriangulation`].
#[derive(Debug)]
pub struct Triangulation {
    /// Vertex coordinates, indexed by [`PointIndex`]. Vertices are never
    /// removed; a vertex may belong to no simplex (e.g. rejected duplicates).
    pub vertices: Vec<Vec<f64>>,
    /// All current simplices, each sorted ascending.
    pub simplices: FxHashSet<Simplex>,
    /// For each vertex, the simplices it belongs to (inverse of
    /// [`Self::simplices`]; see [`Self::reference_invariant`]).
    pub vertex_to_simplices: Vec<FxHashSet<Simplex>>,
    /// Dimensionality of the vertex coordinates.
    pub dim: usize,
    // Cache of the simplex found by the last point location, used as the
    // start of the next walk. Interior-mutable so lookups stay `&self`.
    last_simplex: RwLock<Option<Simplex>>,
}

impl Clone for Triangulation {
    fn clone(&self) -> Self {
        Self {
            vertices: self.vertices.clone(),
            simplices: self.simplices.clone(),
            vertex_to_simplices: self.vertex_to_simplices.clone(),
            dim: self.dim,
            last_simplex: RwLock::new(self.last_simplex().clone()),
        }
    }
}

impl Triangulation {
    // The location cache is only ever replaced wholesale, so a poisoned lock
    // (a panic while it was held) cannot leave it inconsistent; recover the
    // value rather than propagating the panic across the FFI boundary.
    fn last_simplex(&self) -> std::sync::RwLockReadGuard<'_, Option<Simplex>> {
        self.last_simplex
            .read()
            .unwrap_or_else(PoisonError::into_inner)
    }

    fn set_last_simplex(&self, simplex: Option<Simplex>) {
        *self
            .last_simplex
            .write()
            .unwrap_or_else(PoisonError::into_inner) = simplex;
    }

    pub(crate) fn validate_coords(coords: &[Vec<f64>]) -> Result<usize, TriangulationError> {
        if coords.is_empty() {
            return Err(TriangulationError::Value(
                "Please provide at least one simplex".to_string(),
            ));
        }
        let dim = coords[0].len();
        if coords.iter().any(|coord| coord.len() != dim) {
            return Err(TriangulationError::Value(
                "Coordinates dimension mismatch".to_string(),
            ));
        }
        if coords.len() < dim + 1 {
            return Err(TriangulationError::Value(
                "Please provide at least one simplex".to_string(),
            ));
        }

        let vectors: Vec<Vec<f64>> = coords[1..]
            .iter()
            .map(|coord| coord.iter().zip(&coords[0]).map(|(a, b)| a - b).collect())
            .collect();
        if geometry::numpy_matrix_rank(&vectors)? < dim {
            return Err(TriangulationError::Value(
                "Initial simplex has zero volumes (the points are linearly dependent)".to_string(),
            ));
        }

        Ok(dim)
    }

    pub(crate) fn validate_point_dim(&self, point: &[f64]) -> Result<(), TriangulationError> {
        if point.len() != self.dim {
            return Err(TriangulationError::Value(
                "Coordinates dimension mismatch".to_string(),
            ));
        }
        Ok(())
    }

    fn validate_vertex_index(&self, index: usize) -> Result<(), TriangulationError> {
        if index >= self.vertices.len() {
            return Err(index_out_of_range());
        }
        Ok(())
    }

    fn validate_simplex_indices(&self, simplex: &[usize]) -> Result<(), TriangulationError> {
        for &vertex in simplex {
            self.validate_vertex_index(vertex)?;
        }
        Ok(())
    }

    fn find_seed_simplex(coords: &[Vec<f64>], dim: usize) -> Result<Simplex, TriangulationError> {
        let source: Vec<usize> = (0..coords.len()).collect();
        let mut candidates = Vec::new();
        let mut current = Vec::new();
        combinations(&source, dim + 1, &mut candidates, &mut current, 0);

        for simplex in candidates {
            let base = &coords[simplex[0]];
            let vectors: Vec<Vec<f64>> = simplex[1..]
                .iter()
                .map(|&index| coords[index].iter().zip(base).map(|(a, b)| a - b).collect())
                .collect();
            if geometry::numpy_matrix_rank(&vectors)? == dim {
                return Ok(simplex);
            }
        }

        Err(TriangulationError::Value(
            "Initial simplex has zero volumes (the points are linearly dependent)".to_string(),
        ))
    }

    /// Build a triangulation from known simplices (e.g. scipy's Delaunay
    /// output) without running incremental insertion.
    pub fn from_simplices(
        coords: Vec<Vec<f64>>,
        simplices: impl IntoIterator<Item = Simplex>,
    ) -> Result<Self, TriangulationError> {
        let dim = Self::validate_coords(&coords)?;
        let mut triangulation = Self {
            vertex_to_simplices: vec![FxHashSet::default(); coords.len()],
            vertices: coords,
            simplices: FxHashSet::default(),
            dim,
            last_simplex: RwLock::new(None),
        };
        for simplex in simplices {
            triangulation.add_simplex(simplex)?;
        }
        Ok(triangulation)
    }

    /// 1D Delaunay triangulation is the chain of adjacent points in sorted
    /// order; building it directly avoids the O(n^2) incremental insertion.
    fn new_1d(coords: Vec<Vec<f64>>) -> Result<Self, TriangulationError> {
        let mut order: Vec<usize> = (0..coords.len()).collect();
        order.sort_unstable_by(|&a, &b| coords[a][0].total_cmp(&coords[b][0]).then(a.cmp(&b)));
        // Keep the first point among exact duplicates, matching the
        // incremental path which skips points already in the triangulation.
        let mut unique: Vec<usize> = Vec::with_capacity(order.len());
        for index in order {
            if let Some(&previous) = unique.last() {
                if coords[index][0] == coords[previous][0] {
                    continue;
                }
            }
            unique.push(index);
        }

        let segments: Vec<Simplex> = unique
            .windows(2)
            .map(|pair| {
                let mut segment = vec![pair[0], pair[1]];
                segment.sort_unstable();
                segment
            })
            .collect();
        Self::from_simplices(coords, segments)
    }

    /// Build a triangulation from scratch: sorted adjacency in 1D, otherwise
    /// a seed simplex followed by incremental insertion of the remaining
    /// points. Requires at least `dim + 1` points spanning full rank.
    pub fn new(coords: Vec<Vec<f64>>) -> Result<Self, TriangulationError> {
        let dim = Self::validate_coords(&coords)?;
        if dim == 1 {
            return Self::new_1d(coords);
        }
        let seed_simplex = Self::find_seed_simplex(&coords, dim)?;
        let seed_vertices: FxHashSet<usize> = seed_simplex.iter().copied().collect();

        let mut triangulation = Self {
            vertex_to_simplices: vec![FxHashSet::default(); coords.len()],
            vertices: coords,
            simplices: FxHashSet::default(),
            dim,
            last_simplex: RwLock::new(None),
        };
        triangulation.add_simplex(seed_simplex)?;

        for pt_index in 0..triangulation.vertices.len() {
            if seed_vertices.contains(&pt_index) {
                continue;
            }

            let point = triangulation.vertices[pt_index].clone();
            let containing = triangulation.locate_point(&point)?;
            let actual_simplex = containing.clone().unwrap_or_default();

            if actual_simplex.is_empty() {
                triangulation.extend_hull(pt_index)?;
                triangulation.bowyer_watson(pt_index, None, &None)?;
                continue;
            }

            let reduced_simplex =
                triangulation.get_reduced_simplex(&point, &actual_simplex, BARYCENTRIC_EPS)?;
            if reduced_simplex.is_empty() {
                return Err(TriangulationError::Value(
                    "Point lies outside of the specified simplex.".to_string(),
                ));
            }
            if reduced_simplex.len() == 1 {
                continue;
            }
            triangulation.bowyer_watson(pt_index, Some(actual_simplex), &None)?;
        }

        Ok(triangulation)
    }

    /// Insert a simplex (sorted automatically), keeping
    /// [`Self::vertex_to_simplices`] in sync. No geometric checks are done.
    pub fn add_simplex(&mut self, mut simplex: Simplex) -> Result<(), TriangulationError> {
        simplex.sort_unstable();
        if simplex.len() != self.dim + 1 {
            return Err(TriangulationError::Value(format!(
                "Simplex must contain {} vertices",
                self.dim + 1
            )));
        }
        self.validate_simplex_indices(&simplex)?;
        if self.simplices.insert(simplex.clone()) {
            for &vertex in &simplex {
                self.vertex_to_simplices[vertex].insert(simplex.clone());
            }
        }
        Ok(())
    }

    /// Remove a simplex, keeping [`Self::vertex_to_simplices`] in sync.
    pub fn delete_simplex(&mut self, simplex: &[usize]) -> Result<(), TriangulationError> {
        let mut simplex = simplex.to_vec();
        simplex.sort_unstable();
        if !self.simplices.remove(&simplex) {
            return Err(TriangulationError::Value("Simplex not present".to_string()));
        }
        for &vertex in &simplex {
            self.vertex_to_simplices[vertex].remove(&simplex);
        }
        Ok(())
    }

    /// Coordinates of the given vertices, cloned in order.
    pub fn get_vertices(
        &self,
        indices: &[PointIndex],
    ) -> Result<Vec<Vec<f64>>, TriangulationError> {
        self.validate_simplex_indices(indices)?;
        Ok(indices
            .iter()
            .map(|&idx| self.vertices[idx].clone())
            .collect())
    }

    fn locate_point_scan(&self, point: &[f64]) -> Result<Option<Simplex>, TriangulationError> {
        for simplex in &self.simplices {
            let vertices = self.get_vertices(simplex)?;
            if geometry::point_in_simplex(point, &vertices, BARYCENTRIC_EPS)? {
                self.set_last_simplex(Some(simplex.clone()));
                return Ok(Some(simplex.clone()));
            }
        }
        Ok(None)
    }

    /// Barycentric coordinates of `point` relative to vertices `1..=dim` of
    /// `simplex`, without cloning the vertex coordinates.
    pub(crate) fn barycentric_alpha_for_simplex(
        &self,
        simplex: &[usize],
        point: &[f64],
    ) -> Result<Vec<f64>, TriangulationError> {
        let vertices: Vec<&[f64]> = simplex
            .iter()
            .map(|&index| self.vertices[index].as_slice())
            .collect();
        geometry::barycentric_coordinates(&vertices, point).map_err(TriangulationError::Geometry)
    }

    fn next_simplex_in_walk(
        &self,
        simplex: &[usize],
        point: &[f64],
    ) -> Result<Option<Simplex>, TriangulationError> {
        let alpha = self.barycentric_alpha_for_simplex(simplex, point)?;
        let alpha0 = 1.0 - alpha.iter().sum::<f64>();

        let mut worst_idx = 0;
        let mut worst_value = alpha0;
        for (idx, value) in alpha.iter().copied().enumerate() {
            if value < worst_value {
                worst_idx = idx + 1;
                worst_value = value;
            }
        }

        if worst_value >= -BARYCENTRIC_EPS {
            self.set_last_simplex(Some(simplex.to_vec()));
            return Ok(Some(simplex.to_vec()));
        }

        let mut face = Vec::with_capacity(self.dim);
        for (idx, &vertex) in simplex.iter().enumerate() {
            if idx != worst_idx {
                face.push(vertex);
            }
        }

        let mut neighbours = self.containing(&face)?;
        neighbours.remove(simplex);
        Ok(neighbours.into_iter().next())
    }

    /// Find a simplex containing `point` by walking from the last located
    /// simplex, falling back to a linear scan if the walk hits a degenerate
    /// solve. Returns `None` when the point is outside the hull.
    pub fn locate_point(&self, point: &[f64]) -> Result<Option<Simplex>, TriangulationError> {
        self.validate_point_dim(point)?;
        let Some(mut current) = self
            .last_simplex()
            .clone()
            .filter(|simplex| self.simplices.contains(simplex))
            .or_else(|| self.simplices.iter().next().cloned())
        else {
            return Ok(None);
        };

        let mut visited = FxHashSet::default();
        while visited.insert(current.clone()) {
            match self.next_simplex_in_walk(&current, point) {
                Ok(Some(next)) if next == current => return Ok(Some(current)),
                Ok(Some(next)) => current = next,
                Ok(None) => return Ok(None),
                Err(TriangulationError::Geometry(GeometryError::SingularMatrix)) => break,
                Err(err) => return Err(err),
            }
        }

        self.locate_point_scan(point)
    }

    /// Reduce `simplex` to the face of it that `point` actually lies on
    /// (with `eps` barycentric slack): the full simplex for an interior
    /// point, a lower-dimensional face for a point on the boundary, a single
    /// vertex for a coincident point, or empty when the point lies outside.
    pub fn get_reduced_simplex(
        &self,
        point: &[f64],
        simplex: &[usize],
        eps: f64,
    ) -> Result<Simplex, TriangulationError> {
        self.validate_point_dim(point)?;
        let simplex = if simplex.len() != self.dim + 1 {
            let containing = self.containing(simplex)?;
            let Some(first) = containing.into_iter().next() else {
                return Ok(Vec::new());
            };
            first
        } else {
            self.validate_simplex_indices(simplex)?;
            simplex.to_vec()
        };

        let alpha = self.barycentric_alpha_for_simplex(&simplex, point)?;
        Ok(reduced_simplex_positions(&alpha, eps)
            .into_iter()
            .map(|position| simplex[position])
            .collect())
    }

    /// Whether `point` lies inside the simplex given by vertex `indices`,
    /// with `eps` barycentric slack.
    pub fn point_in_simplex(
        &self,
        point: &[f64],
        simplex: &[usize],
        eps: f64,
    ) -> Result<bool, TriangulationError> {
        let vertices = self.get_vertices(simplex)?;
        Ok(geometry::point_in_simplex(point, &vertices, eps)?)
    }

    /// All `dim`-vertex faces (default: (dim-1)-faces, i.e. facets) of either
    /// the whole triangulation, the given `simplices`, or the simplices
    /// touching the given `vertices` (at most one of the two filters). Faces
    /// shared by several simplices are repeated.
    pub fn faces(
        &self,
        dim: Option<usize>,
        simplices: Option<&FxHashSet<Simplex>>,
        vertices: Option<&FxHashSet<usize>>,
    ) -> Result<Vec<Simplex>, TriangulationError> {
        if simplices.is_some() && vertices.is_some() {
            return Err(TriangulationError::Value(
                "Only one of simplices and vertices is allowed.".to_string(),
            ));
        }

        let face_size = dim.unwrap_or(self.dim);
        let simplex_pool: Vec<Simplex> = if let Some(vertices) = vertices {
            let mut pool = FxHashSet::default();
            for &vertex in vertices {
                self.validate_vertex_index(vertex)?;
                pool.extend(self.vertex_to_simplices[vertex].iter().cloned());
            }
            pool.into_iter().collect()
        } else if let Some(simplices) = simplices {
            for simplex in simplices {
                self.validate_simplex_indices(simplex)?;
            }
            simplices.iter().cloned().collect()
        } else {
            self.simplices.iter().cloned().collect()
        };

        let mut faces = Vec::new();
        for simplex in simplex_pool {
            let mut current = Vec::new();
            combinations(&simplex, face_size, &mut faces, &mut current, 0);
        }

        if let Some(vertices) = vertices {
            Ok(faces
                .into_iter()
                .filter(|face| face.iter().all(|idx| vertices.contains(idx)))
                .collect())
        } else {
            Ok(faces)
        }
    }

    /// Count, for every (dim-1)-face of the given simplex pool (or of the
    /// whole triangulation when `None`), how many simplices it belongs to.
    /// In a consistent triangulation a count of 1 means a boundary face and
    /// 2 an interior face.
    fn face_multiplicities(
        &self,
        simplices: Option<&FxHashSet<Simplex>>,
    ) -> Result<FxHashMap<Simplex, usize>, TriangulationError> {
        let faces = self.faces(None, simplices, None)?;
        let mut multiplicities: FxHashMap<Simplex, usize> = FxHashMap::default();
        for face in faces {
            *multiplicities.entry(face).or_insert(0) += 1;
        }
        Ok(multiplicities)
    }

    /// All simplices that contain every vertex of `face`.
    pub fn containing(&self, face: &[usize]) -> Result<FxHashSet<Simplex>, TriangulationError> {
        if face.is_empty() {
            return Ok(FxHashSet::default());
        }
        self.validate_simplex_indices(face)?;
        let mut face_vertices = face.iter().copied();
        let first_vertex = face_vertices
            .by_ref()
            .min_by_key(|vertex| self.vertex_to_simplices[*vertex].len())
            .unwrap();
        let mut result = self.vertex_to_simplices[first_vertex].clone();
        for &vertex in face {
            if vertex == first_vertex {
                continue;
            }
            result.retain(|simplex| self.vertex_to_simplices[vertex].contains(simplex));
        }
        Ok(result)
    }

    fn simplex_points(
        &self,
        simplex: &[usize],
        transform: Option<&[Vec<f64>]>,
    ) -> Result<Vec<Vec<f64>>, TriangulationError> {
        let vertices = self.get_vertices(simplex)?;
        Ok(match transform {
            Some(matrix) => vertices
                .into_iter()
                .map(|vertex| apply_transform(&vertex, matrix))
                .collect(),
            None => vertices,
        })
    }

    fn circumscribed_circle_impl(
        &self,
        simplex: &[usize],
        transform: Option<&[Vec<f64>]>,
    ) -> Result<(Vec<f64>, f64), TriangulationError> {
        let points = self.simplex_points(simplex, transform)?;
        Ok(geometry::circumsphere(&points)?)
    }

    /// Circumsphere (center, radius) of a simplex, optionally after applying
    /// a linear `transform` to its vertices.
    pub fn circumscribed_circle(
        &self,
        simplex: &[usize],
        transform: &Option<Vec<Vec<f64>>>,
    ) -> Result<(Vec<f64>, f64), TriangulationError> {
        validate_transform(transform, self.dim)?;
        self.circumscribed_circle_impl(simplex, transform.as_deref())
    }

    fn point_in_circumcircle_impl(
        &self,
        point: &[f64],
        simplex: &[usize],
        transform: Option<&[Vec<f64>]>,
    ) -> Result<bool, TriangulationError> {
        let (center, radius) = self.circumscribed_circle_impl(simplex, transform)?;
        let transformed_point = match transform {
            Some(matrix) => apply_transform(point, matrix),
            None => point.to_vec(),
        };
        let distance = geometry::fast_norm(
            &center
                .iter()
                .zip(&transformed_point)
                .map(|(a, b)| a - b)
                .collect::<Vec<_>>(),
        );
        Ok(distance < radius * (1.0 + CIRCUMCIRCLE_RTOL))
    }

    /// Whether vertex `pt_index` lies inside the (optionally transformed)
    /// circumsphere of `simplex`, with
    /// [`CIRCUMCIRCLE_RTOL`] slack on the radius.
    pub fn point_in_circumcircle(
        &self,
        pt_index: usize,
        simplex: &[usize],
        transform: &Option<Vec<Vec<f64>>>,
    ) -> Result<bool, TriangulationError> {
        validate_transform(transform, self.dim)?;
        self.validate_vertex_index(pt_index)?;
        self.validate_simplex_indices(simplex)?;
        if simplex.contains(&pt_index) {
            // A vertex lies exactly on its own circumsphere, which the
            // (1 + eps) tolerance counts as inside; skip the numerics so
            // rounding in the circumcenter cannot flip the answer.
            return Ok(true);
        }
        self.point_in_circumcircle_impl(&self.vertices[pt_index], simplex, transform.as_deref())
    }

    /// Re-triangulate around vertex `pt_index`: delete every simplex whose
    /// circumsphere contains it (cascading through facet neighbours, except
    /// in 1D) and fill the resulting hole with simplices through the vertex.
    /// Returns `(deleted, created)` and verifies total volume is conserved.
    /// Errors without mutating anything when the insertion would strand a
    /// cavity vertex, i.e. the point is numerically a duplicate of it.
    pub fn bowyer_watson(
        &mut self,
        pt_index: usize,
        containing_simplex: Option<Simplex>,
        transform: &Option<Vec<Vec<f64>>>,
    ) -> Result<(FxHashSet<Simplex>, FxHashSet<Simplex>), TriangulationError> {
        validate_transform(transform, self.dim)?;
        self.validate_vertex_index(pt_index)?;
        if let Some(simplex) = &containing_simplex {
            self.validate_simplex_indices(simplex)?;
        }

        // Collect the cavity (every simplex whose circumsphere contains the
        // point, cascading through facet neighbours) without mutating, so
        // that pathological insertions can still be rejected cleanly below.
        let mut queue = VecDeque::new();
        let mut queued = FxHashSet::default();
        let mut bad_triangles: FxHashSet<Simplex> = FxHashSet::default();

        if let Some(simplex) = containing_simplex {
            queued.insert(simplex.clone());
            queue.push_back(simplex);
        } else {
            for simplex in self.vertex_to_simplices[pt_index].iter().cloned() {
                if queued.insert(simplex.clone()) {
                    queue.push_back(simplex);
                }
            }
        }

        while let Some(simplex) = queue.pop_front() {
            if !self.simplices.contains(&simplex) {
                continue;
            }

            if self.point_in_circumcircle(pt_index, &simplex, transform)? {
                bad_triangles.insert(simplex.clone());

                if self.dim == 1 {
                    // Inserting a point into an interval never invalidates
                    // neighbouring intervals, so there is no cascade in 1D.
                    continue;
                }

                let simplex_vertices: FxHashSet<usize> = simplex.iter().copied().collect();
                let mut neighbours = FxHashSet::default();
                for &vertex in &simplex {
                    neighbours.extend(self.vertex_to_simplices[vertex].iter().cloned());
                }

                for neighbour in neighbours {
                    let shared = neighbour
                        .iter()
                        .filter(|vertex| simplex_vertices.contains(vertex))
                        .count();
                    if shared == self.dim && queued.insert(neighbour.clone()) {
                        queue.push_back(neighbour);
                    }
                }
            }
        }

        let hole_faces: Vec<Simplex> = self
            .face_multiplicities(Some(&bad_triangles))?
            .into_iter()
            .filter_map(|(face, count)| (count == 1).then_some(face))
            .collect();

        let mut candidates: Vec<Simplex> = Vec::new();
        for face in hole_faces {
            if face.contains(&pt_index) {
                continue;
            }
            let mut simplex = face;
            simplex.push(pt_index);
            simplex.sort_unstable();

            if self.simplex_is_numerically_degenerate(&simplex)? {
                continue;
            }
            candidates.push(simplex);
        }

        // A cavity vertex that would lose all of its simplices without
        // being reconnected means the point is numerically indistinguishable
        // from that vertex (it sits within the circumcircle slack of every
        // simplex around it). Reject the insertion before mutating anything
        // rather than orphaning the vertex and corrupting the triangulation.
        let candidate_vertices: FxHashSet<usize> = candidates.iter().flatten().copied().collect();
        for &vertex in bad_triangles.iter().flatten() {
            if vertex == pt_index || candidate_vertices.contains(&vertex) {
                continue;
            }
            if self.vertex_to_simplices[vertex]
                .iter()
                .all(|simplex| bad_triangles.contains(simplex))
            {
                return Err(TriangulationError::Value(
                    "Point already in triangulation.".to_string(),
                ));
            }
        }

        for simplex in &bad_triangles {
            self.delete_simplex(simplex)?;
        }
        for simplex in candidates {
            self.add_simplex(simplex)?;
        }

        let new_triangles = self.vertex_to_simplices[pt_index].clone();
        let deleted_simplices: FxHashSet<Simplex> =
            bad_triangles.difference(&new_triangles).cloned().collect();
        let new_simplices: FxHashSet<Simplex> =
            new_triangles.difference(&bad_triangles).cloned().collect();

        let old_vol = deleted_simplices.iter().try_fold(0.0, |acc, simplex| {
            Ok::<f64, TriangulationError>(acc + self.volume(simplex)?)
        })?;
        let new_vol = new_simplices.iter().try_fold(0.0, |acc, simplex| {
            Ok::<f64, TriangulationError>(acc + self.volume(simplex)?)
        })?;
        if !is_close(old_vol, new_vol) {
            return Err(TriangulationError::Assertion(format!(
                "{old_vol} !== {new_vol}"
            )));
        }

        Ok((deleted_simplices, new_simplices))
    }

    /// Connect vertex `pt_index`, which must lie outside the convex hull, to
    /// every hull face that faces it. Returns the created simplices; errors
    /// (and rolls back) if the vertex turns out to be inside the hull.
    pub fn extend_hull(
        &mut self,
        pt_index: usize,
    ) -> Result<FxHashSet<Simplex>, TriangulationError> {
        self.validate_vertex_index(pt_index)?;
        let hull_faces: Vec<Simplex> = self
            .face_multiplicities(None)?
            .into_iter()
            .filter_map(|(face, count)| (count == 1).then_some(face))
            .collect();

        let hull_points: Vec<Vec<f64>> = hull_faces
            .iter()
            .flat_map(|face| face.iter().copied())
            .collect::<FxHashSet<_>>()
            .into_iter()
            .map(|idx| self.vertices[idx].clone())
            .collect();
        let mut pt_center = vec![0.0; self.dim];
        for point in &hull_points {
            for (coord, value) in pt_center.iter_mut().zip(point) {
                *coord += *value;
            }
        }
        for coord in &mut pt_center {
            *coord /= hull_points.len() as f64;
        }

        let new_vertex = self.vertices[pt_index].clone();
        let mut new_simplices = FxHashSet::default();

        for face in hull_faces {
            let pts_face = self.get_vertices(&face)?;
            let orientation_inside = geometry::orientation(&pts_face, &pt_center)?;
            let orientation_new = geometry::orientation(&pts_face, &new_vertex)?;
            if orientation_inside == -orientation_new {
                let mut simplex = face.clone();
                simplex.push(pt_index);
                simplex.sort_unstable();
                self.add_simplex(simplex.clone())?;
                new_simplices.insert(simplex);
            }
        }

        if new_simplices.is_empty() {
            let attached = self.vertex_to_simplices[pt_index].clone();
            for simplex in attached {
                self.delete_simplex(&simplex)?;
            }
            return Err(TriangulationError::Value(
                "Candidate vertex is inside the hull.".to_string(),
            ));
        }

        Ok(new_simplices)
    }

    /// Insert a new point, locating it first unless a containing `simplex`
    /// is supplied. Returns the `(deleted, created)` simplex sets. Errors
    /// without modifying the triangulation when the point is already present
    /// or lies outside the supplied simplex.
    pub fn add_point(
        &mut self,
        point: Vec<f64>,
        simplex: Option<Simplex>,
        transform: Option<Vec<Vec<f64>>>,
    ) -> Result<(FxHashSet<Simplex>, FxHashSet<Simplex>), TriangulationError> {
        self.validate_point_dim(&point)?;
        validate_transform(&transform, self.dim)?;

        let mut simplex = match simplex {
            Some(simplex) => simplex,
            None => self.locate_point(&point)?.unwrap_or_default(),
        };
        let actual_simplex = simplex.clone();
        self.vertex_to_simplices.push(FxHashSet::default());

        if simplex.is_empty() {
            self.vertices.push(point);
            let pt_index = self.vertices.len() - 1;
            let temporary_simplices = match self.extend_hull(pt_index) {
                Ok(simplices) => simplices,
                Err(err) => {
                    self.vertex_to_simplices.pop();
                    self.vertices.pop();
                    return Err(err);
                }
            };
            let (deleted_simplices, added_simplices) =
                match self.bowyer_watson(pt_index, None, &transform) {
                    Ok(result) => result,
                    // Value errors are raised before bowyer_watson mutates,
                    // so only the hull extension needs rolling back.
                    Err(err @ TriangulationError::Value(_)) => {
                        for simplex in self.vertex_to_simplices[pt_index].clone() {
                            self.delete_simplex(&simplex)?;
                        }
                        self.vertex_to_simplices.pop();
                        self.vertices.pop();
                        return Err(err);
                    }
                    Err(err) => return Err(err),
                };

            let deleted: FxHashSet<Simplex> = deleted_simplices
                .difference(&temporary_simplices)
                .cloned()
                .collect();
            let mut added = added_simplices;
            for simplex in temporary_simplices.difference(&deleted_simplices) {
                added.insert(simplex.clone());
            }
            return Ok((deleted, added));
        }

        let reduced_simplex = self.get_reduced_simplex(&point, &simplex, BARYCENTRIC_EPS)?;
        if reduced_simplex.is_empty() {
            self.vertex_to_simplices.pop();
            return Err(TriangulationError::Value(
                "Point lies outside of the specified simplex.".to_string(),
            ));
        }
        simplex = reduced_simplex;

        if simplex.len() == 1 {
            self.vertex_to_simplices.pop();
            return Err(TriangulationError::Value(
                "Point already in triangulation.".to_string(),
            ));
        }

        let pt_index = self.vertices.len();
        self.vertices.push(point);
        match self.bowyer_watson(pt_index, Some(actual_simplex), &transform) {
            // Value errors are raised before bowyer_watson mutates, so only
            // the freshly pushed vertex needs rolling back.
            Err(err @ TriangulationError::Value(_)) => {
                self.vertex_to_simplices.pop();
                self.vertices.pop();
                Err(err)
            }
            other => other,
        }
    }

    /// Volume of the simplex with the given vertex indices.
    pub fn volume(&self, simplex: &[usize]) -> Result<f64, TriangulationError> {
        Ok(geometry::volume(&self.get_vertices(simplex)?)?)
    }

    fn normalized_volume(&self, simplex: &[usize]) -> Result<f64, TriangulationError> {
        let vertices = self.get_vertices(simplex)?;
        let base = &vertices[0];
        let mut total_abs_coordinate_delta = 0.0;
        let mut delta_count = 0usize;
        for vertex in vertices.iter().skip(1) {
            for (coord, origin) in vertex.iter().zip(base) {
                total_abs_coordinate_delta += (coord - origin).abs();
                delta_count += 1;
            }
        }

        if delta_count == 0 {
            return Ok(0.0);
        }
        let characteristic_length = total_abs_coordinate_delta / delta_count as f64;
        if characteristic_length == 0.0 {
            return Ok(0.0);
        }

        Ok(self.volume(simplex)? / characteristic_length.powi(self.dim as i32))
    }

    fn simplex_is_numerically_degenerate(
        &self,
        simplex: &[usize],
    ) -> Result<bool, TriangulationError> {
        // Drop a candidate only when its volume is negligible in BOTH
        // senses: absolutely tiny, so removing it cannot leave a material
        // hole (this is the Python reference's criterion), AND flat relative
        // to its own extent. Either test alone is wrong in some regime: the
        // absolute cutoff alone empties finely refined or small-coordinate
        // meshes whose well-shaped simplices all sit below it, while the
        // flatness cutoff alone deletes large-but-flat simplices whose
        // volume the cavity genuinely needs (breaking volume conservation).
        Ok(self.volume(simplex)? < DEGENERATE_VOLUME_EPS
            && self.normalized_volume(simplex)? < DEGENERATE_VOLUME_EPS)
    }

    /// Whether the (order-insensitive) simplex is present.
    pub fn has_simplex(&self, simplex: &[usize]) -> Result<bool, TriangulationError> {
        let mut simplex = simplex.to_vec();
        simplex.sort_unstable();
        self.validate_simplex_indices(&simplex)?;
        Ok(self.simplices.contains(&simplex))
    }

    /// The simplices containing `vertex`, by reference.
    pub fn vertex_to_simplices_for(
        &self,
        vertex: usize,
    ) -> Result<&FxHashSet<Simplex>, TriangulationError> {
        self.validate_vertex_index(vertex)?;
        Ok(&self.vertex_to_simplices[vertex])
    }

    /// Volumes of all simplices, in iteration order of [`Self::simplices`].
    pub fn volumes(&self) -> Result<Vec<f64>, TriangulationError> {
        self.simplices
            .iter()
            .map(|simplex| self.volume(simplex))
            .collect()
    }

    /// Whether [`Self::simplices`] and [`Self::vertex_to_simplices`] are
    /// mutually consistent (a bookkeeping check, not a geometric one).
    pub fn reference_invariant(&self) -> bool {
        for vertex in 0..self.vertices.len() {
            if self.vertex_to_simplices[vertex]
                .iter()
                .any(|simplex| !simplex.contains(&vertex))
            {
                return false;
            }
        }
        for simplex in &self.simplices {
            if simplex
                .iter()
                .any(|point| !self.vertex_to_simplices[*point].contains(simplex))
            {
                return false;
            }
        }
        true
    }

    /// Vertices on the convex hull (those belonging to a boundary face).
    /// Errors when a face belongs to more than two simplices, which means
    /// the triangulation is internally inconsistent.
    pub fn hull(&self) -> Result<FxHashSet<usize>, TriangulationError> {
        let counts = self.face_multiplicities(None)?;
        if counts.values().any(|&count| count > 2) {
            return Err(TriangulationError::Runtime(
                "Broken triangulation, a (N-1)-dimensional appears in more than 2 simplices."
                    .to_string(),
            ));
        }

        let mut hull = FxHashSet::default();
        for (face, count) in counts {
            if count == 1 {
                hull.extend(face);
            }
        }
        Ok(hull)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_triangulation() -> Triangulation {
        Triangulation::from_simplices(
            vec![
                vec![0.0, 0.0],
                vec![1.0, 0.0],
                vec![0.0, 1.0],
                vec![1.0, 1.0],
            ],
            vec![vec![0, 1, 3], vec![0, 2, 3]],
        )
        .unwrap()
    }

    #[test]
    fn has_simplex_matches_current_membership() {
        let tri = sample_triangulation();

        assert!(tri.has_simplex(&[0, 1, 3]).unwrap());
        assert!(tri.has_simplex(&[3, 1, 0]).unwrap());
        assert!(!tri.has_simplex(&[0, 1, 2]).unwrap());
    }

    #[test]
    fn vertex_to_simplices_for_returns_single_vertex_view() {
        let tri = sample_triangulation();
        let simplices = tri.vertex_to_simplices_for(0).unwrap();

        assert_eq!(simplices.len(), 2);
        assert!(simplices.contains(&vec![0, 1, 3]));
        assert!(simplices.contains(&vec![0, 2, 3]));
    }

    #[test]
    fn one_dimensional_triangulation_connects_adjacent_points() {
        let tri = Triangulation::new(vec![vec![2.0], vec![0.0], vec![1.0], vec![3.0]]).unwrap();

        assert_eq!(tri.dim, 1);
        assert_eq!(
            tri.simplices,
            FxHashSet::from_iter([vec![0, 2], vec![1, 2], vec![0, 3]])
        );
        assert_eq!(tri.hull().unwrap(), FxHashSet::from_iter([1, 3]));
        assert!(tri.reference_invariant());
    }

    #[test]
    fn one_dimensional_tiny_interval_survives_bowyer_watson() {
        let mut tri = Triangulation::new(vec![vec![0.0], vec![1e-12]]).unwrap();
        let (deleted, added) = tri.add_point(vec![5e-13], None, None).unwrap();

        assert_eq!(deleted, FxHashSet::from_iter([vec![0, 1]]));
        assert_eq!(added, FxHashSet::from_iter([vec![0, 2], vec![1, 2]]));
        assert_eq!(
            tri.simplices,
            FxHashSet::from_iter([vec![0, 2], vec![1, 2]])
        );
    }

    #[test]
    fn one_dimensional_fine_grid_far_from_origin_constructs() {
        // Regression: cancellation in the circumcenter solve made the
        // volume-conservation assertion fail for small off-origin intervals.
        let coords: Vec<Vec<f64>> = (0..50).map(|i| vec![100.0 + i as f64 * 2e-5]).collect();
        let tri = Triangulation::new(coords).unwrap();

        assert_eq!(tri.simplices.len(), 49);
        assert_eq!(tri.hull().unwrap(), FxHashSet::from_iter([0, 49]));
        assert!(tri.reference_invariant());
    }

    #[test]
    fn one_dimensional_add_point_outside_hull_far_from_origin() {
        let mut tri = Triangulation::new(vec![vec![0.0], vec![0.5]]).unwrap();
        let (deleted, added) = tri.add_point(vec![0.5 + 1e-6], None, None).unwrap();

        assert!(deleted.is_empty());
        assert_eq!(added, FxHashSet::from_iter([vec![1, 2]]));
    }

    #[test]
    fn two_dimensional_small_scale_insertion_keeps_mesh() {
        // Regression: the absolute degenerate-volume cutoff dropped every
        // candidate simplex below 1e-8 volume, emptying the mesh.
        let mut tri = Triangulation::from_simplices(
            vec![
                vec![0.0, 0.0],
                vec![1e-4, 0.0],
                vec![0.0, 1e-4],
                vec![1e-4, 1e-4],
            ],
            vec![vec![0, 1, 3], vec![0, 2, 3]],
        )
        .unwrap();

        let (deleted, added) = tri.add_point(vec![5e-5, 6e-5], None, None).unwrap();

        assert_eq!(deleted.len(), 2);
        assert_eq!(added.len(), 4);
        assert_eq!(tri.simplices.len(), 4);
        assert!((0..tri.vertices.len()).all(|v| !tri.vertex_to_simplices[v].is_empty()));
    }

    #[test]
    fn two_dimensional_near_duplicate_insertion_is_rejected_without_mutation() {
        // Regression: a point within the circumcircle slack of every simplex
        // around an existing vertex orphaned that vertex.
        let mut tri = Triangulation::new(vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.5, 1.0],
            vec![0.5, 0.5],
            vec![0.5 + 1e-6, 0.5],
            vec![0.5, 0.5 + 1e-6],
        ])
        .unwrap();
        let simplices_before = tri.simplices.clone();
        let n_vertices_before = tri.vertices.len();

        let err = tri
            .add_point(vec![0.5 + 2e-10, 0.5 + 2e-10], None, None)
            .unwrap_err();

        assert!(matches!(err, TriangulationError::Value(_)));
        assert_eq!(tri.simplices, simplices_before);
        assert_eq!(tri.vertices.len(), n_vertices_before);
        assert!(tri.reference_invariant());
    }

    #[test]
    fn one_dimensional_insertion_near_shared_vertex_keeps_vertex_connected() {
        // Regression: the circumcircle cascade deleted the long neighbouring
        // interval as well, orphaning the shared vertex.
        let mut tri = Triangulation::new(vec![vec![0.0], vec![0.5], vec![0.5 + 1e-6]]).unwrap();
        let (deleted, added) = tri
            .add_point(vec![0.5 + 1e-10], Some(vec![1, 2]), None)
            .unwrap();

        assert_eq!(deleted, FxHashSet::from_iter([vec![1, 2]]));
        assert_eq!(added, FxHashSet::from_iter([vec![1, 3], vec![2, 3]]));
        assert!((0..tri.vertices.len()).all(|v| !tri.vertex_to_simplices[v].is_empty()));
    }
}
