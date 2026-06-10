//! N-dimensional Delaunay triangulation with incremental Bowyer-Watson
//! point insertion.
//!
//! This is the pure-Rust core: it holds all triangulation state and
//! algorithms and never touches Python. The PyO3 surface (argument parsing,
//! result conversion, the proxy views, and the `Triangulation` Python class)
//! lives in [`crate::py`].
//!
//! # Data layout
//!
//! Simplices are interned: each lives once in a slab and is referred to
//! everywhere else by a small integer [`SimplexId`]. Three derived indexes
//! are kept in sync incrementally by the single pair of mutation primitives
//! (`link_simplex` / `unlink_simplex`):
//!
//! - `ids`: sorted vertex list → id, for membership tests and iteration;
//! - `vertex_to_ids`: vertex → set of incident simplex ids;
//! - `facets`: (dim-1)-face → ids of the (normally ≤ 2) simplices sharing
//!   it, plus the derived `boundary_facets` set (exactly one incident
//!   simplex, i.e. the convex hull) and `overfull_facets` count (more than
//!   two incident simplices, i.e. a corrupted triangulation).
//!
//! The facet index is what makes the hot paths cheap: walking to a facet
//! neighbour ([`Triangulation::locate_point`]), cascading through neighbours
//! in [`Triangulation::bowyer_watson`], and answering
//! [`Triangulation::containing`] for a facet are all single hash lookups,
//! and [`Triangulation::extend_hull`] / [`Triangulation::hull`] read the
//! maintained boundary set instead of recounting every face of every
//! simplex.

use std::collections::VecDeque;
use std::sync::{PoisonError, RwLock};

use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;
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
/// Slab index of an interned simplex; only meaningful while that simplex is
/// present (ids are recycled after deletion).
type SimplexId = u32;
/// The simplices incident to one facet: two for an interior facet, one on
/// the hull. More than two only in inconsistent states a user can create by
/// hand through `add_simplex`.
type IncidentSimplices = SmallVec<[SimplexId; 2]>;

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

/// The facet of a sorted `simplex` obtained by removing the vertex at
/// position `skip`; the result is still sorted.
fn facet_excluding(simplex: &[usize], skip: usize) -> Simplex {
    let mut facet = Vec::with_capacity(simplex.len() - 1);
    facet.extend_from_slice(&simplex[..skip]);
    facet.extend_from_slice(&simplex[skip + 1..]);
    facet
}

fn is_close(a: f64, b: f64) -> bool {
    // Symmetric variant of numpy.isclose (which scales by |b| only), so the
    // volume-conservation check cannot depend on argument order.
    let scale = a.abs().max(b.abs());
    (a - b).abs() <= VOLUME_CONSERVATION_ATOL + VOLUME_CONSERVATION_RTOL * scale
}

/// Result of one step of the simplex walk in [`Triangulation::locate_point`].
enum WalkStep {
    /// The current simplex contains the point.
    Inside,
    /// The point lies beyond this facet neighbour; continue walking there.
    Neighbour(SimplexId),
    /// The point lies beyond a hull facet, i.e. outside the triangulation.
    OutsideHull,
}

/// An N-dimensional Delaunay triangulation supporting incremental point
/// insertion via the Bowyer-Watson algorithm. Pure Rust; the Python surface
/// lives on [`crate::py::PyTriangulation`]. See the module docs for the
/// data layout.
#[derive(Debug)]
pub struct Triangulation {
    /// Vertex coordinates, indexed by [`PointIndex`]. Vertices are never
    /// removed; a vertex may belong to no simplex (e.g. rejected duplicates).
    pub vertices: Vec<Vec<f64>>,
    /// Dimensionality of the vertex coordinates.
    pub dim: usize,
    /// Interned simplex storage; `None` entries are free slots whose ids sit
    /// in [`Self::free_ids`] awaiting reuse.
    slab: Vec<Option<Simplex>>,
    /// Recycled slab indices.
    free_ids: Vec<SimplexId>,
    /// Sorted vertex list → slab id, for membership tests and iteration.
    ids: FxHashMap<Simplex, SimplexId>,
    /// For each vertex, the ids of the simplices it belongs to.
    vertex_to_ids: Vec<FxHashSet<SimplexId>>,
    /// Every (dim-1)-face of a current simplex → ids of the simplices that
    /// share it.
    facets: FxHashMap<Simplex, IncidentSimplices>,
    /// Facets with exactly one incident simplex, i.e. the convex hull.
    boundary_facets: FxHashSet<Simplex>,
    /// Number of facets with more than two incident simplices. Non-zero only
    /// when hand-built simplices made the triangulation inconsistent;
    /// [`Self::hull`] reports that as an error.
    overfull_facets: usize,
    // Cache of the simplex found by the last point location, used as the
    // start of the next walk. A recycled id is harmless here: it then names
    // some other live simplex, which is an equally valid walk start.
    // Interior-mutable so lookups stay `&self`.
    last_simplex: RwLock<Option<SimplexId>>,
}

impl Clone for Triangulation {
    fn clone(&self) -> Self {
        Self {
            vertices: self.vertices.clone(),
            dim: self.dim,
            slab: self.slab.clone(),
            free_ids: self.free_ids.clone(),
            ids: self.ids.clone(),
            vertex_to_ids: self.vertex_to_ids.clone(),
            facets: self.facets.clone(),
            boundary_facets: self.boundary_facets.clone(),
            overfull_facets: self.overfull_facets,
            last_simplex: RwLock::new(*self.last_simplex()),
        }
    }
}

impl Triangulation {
    // The location cache is only ever replaced wholesale, so a poisoned lock
    // (a panic while it was held) cannot leave it inconsistent; recover the
    // value rather than propagating the panic across the FFI boundary.
    fn last_simplex(&self) -> std::sync::RwLockReadGuard<'_, Option<SimplexId>> {
        self.last_simplex
            .read()
            .unwrap_or_else(PoisonError::into_inner)
    }

    fn set_last_simplex(&self, id: Option<SimplexId>) {
        *self
            .last_simplex
            .write()
            .unwrap_or_else(PoisonError::into_inner) = id;
    }

    fn empty(vertices: Vec<Vec<f64>>, dim: usize) -> Self {
        Self {
            vertex_to_ids: vec![FxHashSet::default(); vertices.len()],
            vertices,
            dim,
            slab: Vec::new(),
            free_ids: Vec::new(),
            ids: FxHashMap::default(),
            facets: FxHashMap::default(),
            boundary_facets: FxHashSet::default(),
            overfull_facets: 0,
            last_simplex: RwLock::new(None),
        }
    }

    /// The interned simplex for a live id.
    fn simplex_by_id(&self, id: SimplexId) -> &Simplex {
        self.slab[id as usize]
            .as_ref()
            .expect("simplex id points at a freed slab slot")
    }

    /// Whether `id` currently names a live simplex.
    fn id_is_live(&self, id: SimplexId) -> bool {
        self.slab
            .get(id as usize)
            .is_some_and(|slot| slot.is_some())
    }

    /// Intern `simplex` (which must already be sorted) and update every
    /// derived index. Returns `false` (and changes nothing) when it is
    /// already present.
    fn link_simplex(&mut self, simplex: Simplex) -> bool {
        if self.ids.contains_key(&simplex) {
            return false;
        }
        let id = if let Some(id) = self.free_ids.pop() {
            self.slab[id as usize] = Some(simplex.clone());
            id
        } else {
            self.slab.push(Some(simplex.clone()));
            SimplexId::try_from(self.slab.len() - 1).expect("more than u32::MAX simplices")
        };

        for &vertex in &simplex {
            self.vertex_to_ids[vertex].insert(id);
        }
        for skip in 0..simplex.len() {
            let facet = facet_excluding(&simplex, skip);
            if let Some(incident) = self.facets.get_mut(&facet) {
                incident.push(id);
                match incident.len() {
                    2 => {
                        self.boundary_facets.remove(&facet);
                    }
                    3 => self.overfull_facets += 1,
                    _ => {}
                }
            } else {
                self.facets
                    .insert(facet.clone(), SmallVec::from_slice(&[id]));
                self.boundary_facets.insert(facet);
            }
        }
        self.ids.insert(simplex, id);
        true
    }

    /// Remove the interned `simplex` (which must already be sorted) and
    /// update every derived index. Returns `false` when it is not present.
    fn unlink_simplex(&mut self, simplex: &[usize]) -> bool {
        let Some(id) = self.ids.remove(simplex) else {
            return false;
        };
        self.slab[id as usize] = None;
        self.free_ids.push(id);

        for &vertex in simplex {
            self.vertex_to_ids[vertex].remove(&id);
        }
        for skip in 0..simplex.len() {
            let facet = facet_excluding(simplex, skip);
            let remaining = {
                let incident = self
                    .facets
                    .get_mut(&facet)
                    .expect("facet index out of sync with simplex storage");
                incident.retain(|&mut other| other != id);
                incident.len()
            };
            match remaining {
                0 => {
                    self.facets.remove(&facet);
                    self.boundary_facets.remove(&facet);
                }
                1 => {
                    self.boundary_facets.insert(facet);
                }
                2 => self.overfull_facets -= 1,
                _ => {}
            }
        }
        true
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
        let mut triangulation = Self::empty(coords, dim);
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

        let mut triangulation = Self::empty(coords, dim);
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

    /// Number of simplices currently in the triangulation.
    pub fn num_simplices(&self) -> usize {
        self.ids.len()
    }

    /// Iterator over all current simplices (each sorted ascending), in
    /// arbitrary order.
    pub fn simplices(&self) -> impl Iterator<Item = &Simplex> + '_ {
        self.ids.keys()
    }

    /// Whether the already-sorted `simplex` is present. See
    /// [`Self::has_simplex`] for the order-insensitive, validating variant.
    pub fn contains_simplex(&self, simplex: &[usize]) -> bool {
        self.ids.contains_key(simplex)
    }

    /// Append a vertex without connecting it to any simplex. Used while
    /// inserting points; a vertex that ends up unconnected (e.g. a rejected
    /// duplicate) must be rolled back by the caller.
    pub fn add_vertex(&mut self, point: Vec<f64>) -> PointIndex {
        self.vertices.push(point);
        self.vertex_to_ids.push(FxHashSet::default());
        self.vertices.len() - 1
    }

    fn pop_vertex(&mut self) {
        self.vertices.pop();
        self.vertex_to_ids.pop();
    }

    /// Number of simplices the given vertex belongs to.
    pub fn vertex_simplex_count(&self, vertex: PointIndex) -> usize {
        self.vertex_to_ids[vertex].len()
    }

    /// Iterator over the simplices containing `vertex`, in arbitrary order.
    /// The index must be valid.
    pub fn simplices_of(&self, vertex: PointIndex) -> impl Iterator<Item = &Simplex> + '_ {
        self.vertex_to_ids[vertex]
            .iter()
            .map(|&id| self.simplex_by_id(id))
    }

    /// Whether the already-sorted `simplex` is present and contains `vertex`.
    pub fn vertex_has_simplex(&self, vertex: PointIndex, simplex: &[usize]) -> bool {
        self.ids
            .get(simplex)
            .is_some_and(|id| self.vertex_to_ids[vertex].contains(id))
    }

    /// Insert a simplex (sorted automatically), keeping all derived indexes
    /// in sync. No geometric checks are done.
    pub fn add_simplex(&mut self, mut simplex: Simplex) -> Result<(), TriangulationError> {
        simplex.sort_unstable();
        if simplex.len() != self.dim + 1 {
            return Err(TriangulationError::Value(format!(
                "Simplex must contain {} vertices",
                self.dim + 1
            )));
        }
        self.validate_simplex_indices(&simplex)?;
        self.link_simplex(simplex);
        Ok(())
    }

    /// Remove a simplex, keeping all derived indexes in sync.
    pub fn delete_simplex(&mut self, simplex: &[usize]) -> Result<(), TriangulationError> {
        let mut simplex = simplex.to_vec();
        simplex.sort_unstable();
        if !self.unlink_simplex(&simplex) {
            return Err(TriangulationError::Value("Simplex not present".to_string()));
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
        for (simplex, &id) in &self.ids {
            let vertices = self.get_vertices(simplex)?;
            if geometry::point_in_simplex(point, &vertices, BARYCENTRIC_EPS)? {
                self.set_last_simplex(Some(id));
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
        id: SimplexId,
        point: &[f64],
    ) -> Result<WalkStep, TriangulationError> {
        let simplex = self.simplex_by_id(id);
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
            return Ok(WalkStep::Inside);
        }

        let face = facet_excluding(simplex, worst_idx);
        let neighbour = self
            .facets
            .get(&face)
            .and_then(|incident| incident.iter().copied().find(|&other| other != id));
        Ok(match neighbour {
            Some(next) => WalkStep::Neighbour(next),
            None => WalkStep::OutsideHull,
        })
    }

    /// Find a simplex containing `point` by walking from the last located
    /// simplex, falling back to a linear scan if the walk hits a degenerate
    /// solve. Returns `None` when the point is outside the hull.
    pub fn locate_point(&self, point: &[f64]) -> Result<Option<Simplex>, TriangulationError> {
        self.validate_point_dim(point)?;
        let start = self
            .last_simplex()
            .filter(|&id| self.id_is_live(id))
            .or_else(|| self.ids.values().next().copied());
        let Some(mut current) = start else {
            return Ok(None);
        };

        let mut visited = FxHashSet::default();
        while visited.insert(current) {
            match self.next_simplex_in_walk(current, point) {
                Ok(WalkStep::Inside) => {
                    self.set_last_simplex(Some(current));
                    return Ok(Some(self.simplex_by_id(current).clone()));
                }
                Ok(WalkStep::Neighbour(next)) => current = next,
                Ok(WalkStep::OutsideHull) => return Ok(None),
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
        let simplex_pool: Vec<&Simplex> = if let Some(vertices) = vertices {
            let mut pool_ids = FxHashSet::default();
            for &vertex in vertices {
                self.validate_vertex_index(vertex)?;
                pool_ids.extend(self.vertex_to_ids[vertex].iter().copied());
            }
            pool_ids
                .into_iter()
                .map(|id| self.simplex_by_id(id))
                .collect()
        } else if let Some(simplices) = simplices {
            for simplex in simplices {
                self.validate_simplex_indices(simplex)?;
            }
            simplices.iter().collect()
        } else {
            self.ids.keys().collect()
        };

        let mut faces = Vec::new();
        for simplex in simplex_pool {
            let mut current = Vec::new();
            combinations(simplex, face_size, &mut faces, &mut current, 0);
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

    /// All simplices that contain every vertex of `face`.
    pub fn containing(&self, face: &[usize]) -> Result<FxHashSet<Simplex>, TriangulationError> {
        if face.is_empty() {
            return Ok(FxHashSet::default());
        }
        self.validate_simplex_indices(face)?;

        // A facet-sized face is a single lookup in the facet index.
        if face.len() == self.dim {
            let mut sorted = face.to_vec();
            sorted.sort_unstable();
            return Ok(self
                .facets
                .get(&sorted)
                .map(|incident| {
                    incident
                        .iter()
                        .map(|&id| self.simplex_by_id(id).clone())
                        .collect()
                })
                .unwrap_or_default());
        }

        let first_vertex = face
            .iter()
            .copied()
            .min_by_key(|&vertex| self.vertex_to_ids[vertex].len())
            .unwrap();
        Ok(self.vertex_to_ids[first_vertex]
            .iter()
            .copied()
            .filter(|&id| {
                face.iter().all(|&vertex| {
                    vertex == first_vertex || self.vertex_to_ids[vertex].contains(&id)
                })
            })
            .map(|id| self.simplex_by_id(id).clone())
            .collect())
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

    /// Count, for every facet of the given simplices, how many of them it
    /// belongs to. Within a Bowyer-Watson cavity a count of 1 means a face
    /// of the hole being re-triangulated.
    fn facet_multiplicities<'a>(
        simplices: impl Iterator<Item = &'a Simplex>,
    ) -> FxHashMap<Simplex, usize> {
        let mut multiplicities: FxHashMap<Simplex, usize> = FxHashMap::default();
        for simplex in simplices {
            for skip in 0..simplex.len() {
                *multiplicities
                    .entry(facet_excluding(simplex, skip))
                    .or_insert(0) += 1;
            }
        }
        multiplicities
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
        let mut queue: VecDeque<SimplexId> = VecDeque::new();
        let mut queued: FxHashSet<SimplexId> = FxHashSet::default();
        let mut bad_ids: FxHashSet<SimplexId> = FxHashSet::default();

        if let Some(simplex) = containing_simplex {
            if let Some(&id) = self.ids.get(&simplex) {
                queued.insert(id);
                queue.push_back(id);
            }
        } else {
            for &id in &self.vertex_to_ids[pt_index] {
                if queued.insert(id) {
                    queue.push_back(id);
                }
            }
        }
        let seed_ids: Vec<SimplexId> = queue.iter().copied().collect();

        while let Some(id) = queue.pop_front() {
            let simplex = self.simplex_by_id(id);
            if self.point_in_circumcircle(pt_index, simplex, transform)? {
                bad_ids.insert(id);

                if self.dim == 1 {
                    // Inserting a point into an interval never invalidates
                    // neighbouring intervals, so there is no cascade in 1D.
                    continue;
                }

                let simplex = self.simplex_by_id(id);
                for skip in 0..simplex.len() {
                    let facet = facet_excluding(simplex, skip);
                    let Some(incident) = self.facets.get(&facet) else {
                        continue;
                    };
                    for &neighbour in incident {
                        if neighbour != id && queued.insert(neighbour) {
                            queue.push_back(neighbour);
                        }
                    }
                }
            }
        }

        let (mut bad_simplices, mut candidates) = self.cavity_candidates(pt_index, &bad_ids)?;
        let (mut deleted_simplices, mut new_simplices, mut point_connected) =
            self.cavity_outcome(pt_index, &bad_simplices, &candidates);

        let mut old_vol = self.summed_volume(&deleted_simplices)?;
        let mut new_vol = self.summed_volume(&new_simplices)?;
        // The cavity is unusable when re-triangulating it would not cover
        // the volume it frees, or when it would leave the point connected
        // to nothing: either an empty cavity (the circumsphere test falsely
        // excluded even the simplex the point lies in) or a cavity that
        // swallows every simplex of the point while producing no candidates
        // (its total volume can sit below the conservation check's absolute
        // tolerance, making that check vacuous). Both happen when sliver
        // simplices feed the floating-point circumsphere test cancellation
        // noise; see src/tolerances.rs.
        if !is_close(old_vol, new_vol) || !point_connected {
            // Repair: in 2D/3D rebuild the cavity with exact insphere
            // predicates — the exact Delaunay cavity is connected,
            // void-free, and star-shaped around the point, so filling it
            // conserves volume by construction (provided the surrounding
            // mesh itself is still consistent). Where no exact predicate
            // exists (4D+), fall back to shrinking the cavity until every
            // hole face is visible from the point. Nothing has been mutated
            // yet, so when even the repaired cavity fails validation the
            // insertion is rejected with the triangulation untouched; the
            // Python reference instead mutates first and corrupts its
            // state.
            if let Some(exact_cavity) =
                self.rebuild_cavity_exact(pt_index, &seed_ids, transform.as_deref())?
            {
                bad_ids = exact_cavity;
            } else {
                self.shrink_cavity_to_star(pt_index, &mut bad_ids, &seed_ids)?;
            }
            (bad_simplices, candidates) = self.cavity_candidates(pt_index, &bad_ids)?;
            (deleted_simplices, new_simplices, point_connected) =
                self.cavity_outcome(pt_index, &bad_simplices, &candidates);
            old_vol = self.summed_volume(&deleted_simplices)?;
            new_vol = self.summed_volume(&new_simplices)?;

            if !is_close(old_vol, new_vol) {
                return Err(TriangulationError::Assertion(format!(
                    "{old_vol} !== {new_vol}"
                )));
            }
            if !point_connected {
                // Even the repaired cavity cannot connect the point: it is
                // numerically indistinguishable from existing structure.
                return Err(TriangulationError::Value(
                    "Point already in triangulation.".to_string(),
                ));
            }
        }

        // A cavity vertex that would lose all of its simplices without
        // being reconnected means the point is numerically indistinguishable
        // from that vertex (it sits within the circumcircle slack of every
        // simplex around it). Reject the insertion before mutating anything
        // rather than orphaning the vertex and corrupting the triangulation.
        let candidate_vertices: FxHashSet<usize> = candidates.iter().flatten().copied().collect();
        for &vertex in bad_simplices.iter().flatten() {
            if vertex == pt_index || candidate_vertices.contains(&vertex) {
                continue;
            }
            if self.vertex_to_ids[vertex]
                .iter()
                .all(|id| bad_ids.contains(id))
            {
                return Err(TriangulationError::Value(
                    "Point already in triangulation.".to_string(),
                ));
            }
        }

        for simplex in &bad_simplices {
            self.delete_simplex(simplex)?;
        }
        for simplex in candidates {
            self.add_simplex(simplex)?;
        }

        Ok((deleted_simplices, new_simplices))
    }

    /// The cavity simplices (cloned out of the slab) and the candidate
    /// simplices that would fill the cavity's hole: one per hole face (a
    /// facet belonging to exactly one cavity simplex) that does not contain
    /// the point, joined with the point, minus numerically degenerate ones.
    fn cavity_candidates(
        &self,
        pt_index: usize,
        bad_ids: &FxHashSet<SimplexId>,
    ) -> Result<(Vec<Simplex>, Vec<Simplex>), TriangulationError> {
        let bad_simplices: Vec<Simplex> = bad_ids
            .iter()
            .map(|&id| self.simplex_by_id(id).clone())
            .collect();

        let hole_faces: Vec<Simplex> = Self::facet_multiplicities(bad_simplices.iter())
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
        Ok((bad_simplices, candidates))
    }

    /// The `(deleted, created, point_connected)` outcome that deleting the
    /// cavity and adding the candidates *would* produce, computed without
    /// mutating anything: the point's simplices afterwards are its current
    /// ones minus the cavity plus the candidates (`point_connected` reports
    /// whether that set is non-empty), and a simplex both deleted and
    /// recreated cancels out of the result.
    fn cavity_outcome(
        &self,
        pt_index: usize,
        bad_simplices: &[Simplex],
        candidates: &[Simplex],
    ) -> (FxHashSet<Simplex>, FxHashSet<Simplex>, bool) {
        let bad_set: FxHashSet<&Simplex> = bad_simplices.iter().collect();
        let mut new_triangles: FxHashSet<Simplex> = self.vertex_to_ids[pt_index]
            .iter()
            .map(|&id| self.simplex_by_id(id))
            .filter(|simplex| !bad_set.contains(simplex))
            .cloned()
            .collect();
        new_triangles.extend(candidates.iter().cloned());
        let point_connected = !new_triangles.is_empty();

        let deleted: FxHashSet<Simplex> = bad_simplices
            .iter()
            .filter(|simplex| !new_triangles.contains(*simplex))
            .cloned()
            .collect();
        let created: FxHashSet<Simplex> = new_triangles
            .into_iter()
            .filter(|simplex| !bad_set.contains(simplex))
            .collect();
        (deleted, created, point_connected)
    }

    /// Total volume of `simplices`, using adaptive-precision determinants
    /// (in 2D/3D) so the conservation check compares true volumes instead of
    /// cancellation noise when the cavity contains slivers.
    fn summed_volume(&self, simplices: &FxHashSet<Simplex>) -> Result<f64, TriangulationError> {
        simplices.iter().try_fold(0.0, |acc, simplex| {
            Ok::<f64, TriangulationError>(
                acc + geometry::precise_volume(&self.get_vertices(simplex)?)?,
            )
        })
    }

    /// Rebuild the cavity with exact in-circumsphere predicates (2D/3D
    /// only; `None` elsewhere): cascade through facet neighbours from the
    /// seeds, keeping the seeds themselves (the point lies in or on them)
    /// plus every simplex whose circumsphere strictly contains the point.
    /// The exact Delaunay cavity is connected, void-free, and star-shaped
    /// around the point, so filling it from the point conserves volume even
    /// when slivers fooled the floating-point tests that assembled the
    /// original cavity.
    fn rebuild_cavity_exact(
        &self,
        pt_index: usize,
        seed_ids: &[SimplexId],
        transform: Option<&[Vec<f64>]>,
    ) -> Result<Option<FxHashSet<SimplexId>>, TriangulationError> {
        if self.dim != 2 && self.dim != 3 {
            return Ok(None);
        }
        let point = match transform {
            Some(matrix) => apply_transform(&self.vertices[pt_index], matrix),
            None => self.vertices[pt_index].clone(),
        };

        let mut cavity: FxHashSet<SimplexId> = FxHashSet::default();
        let mut examined: FxHashSet<SimplexId> = FxHashSet::default();
        let mut queue: VecDeque<SimplexId> = VecDeque::new();
        for &seed in seed_ids {
            if self.id_is_live(seed) && examined.insert(seed) {
                cavity.insert(seed);
                queue.push_back(seed);
            }
        }

        while let Some(id) = queue.pop_front() {
            let simplex = self.simplex_by_id(id);
            for skip in 0..simplex.len() {
                let Some(incident) = self.facets.get(&facet_excluding(simplex, skip)) else {
                    continue;
                };
                for &neighbour in incident {
                    if neighbour == id || !examined.insert(neighbour) {
                        continue;
                    }
                    let points = self.simplex_points(self.simplex_by_id(neighbour), transform)?;
                    if geometry::robust_in_circumsphere(&points, &point)?.unwrap_or(false) {
                        cavity.insert(neighbour);
                        queue.push_back(neighbour);
                    }
                }
            }
        }
        Ok(Some(cavity))
    }

    /// Shrink the cavity until it is star-shaped as seen from `pt_index`:
    /// every hole face not containing the point must be strictly visible
    /// from it, i.e. the point lies on the same side of the face as the
    /// cavity simplex owning it. Owners of invisible faces are evicted and
    /// the hole boundary recomputed until no violation remains. Visibility
    /// uses exact orientation predicates in 2D/3D
    /// ([`geometry::robust_orientation`]), so a sliver cannot fool it.
    fn shrink_cavity_to_star(
        &self,
        pt_index: usize,
        bad_ids: &mut FxHashSet<SimplexId>,
        seed_ids: &[SimplexId],
    ) -> Result<(), TriangulationError> {
        let point = &self.vertices[pt_index];
        self.prune_to_seed_component(bad_ids, seed_ids);
        loop {
            // Hole face -> (multiplicity within the cavity, owning cavity
            // simplex, its vertex opposite the face).
            let mut faces: FxHashMap<Simplex, (usize, SimplexId, usize)> = FxHashMap::default();
            for &id in bad_ids.iter() {
                let simplex = self.simplex_by_id(id);
                for skip in 0..simplex.len() {
                    faces
                        .entry(facet_excluding(simplex, skip))
                        .and_modify(|entry| entry.0 += 1)
                        .or_insert((1, id, simplex[skip]));
                }
            }

            let mut evicted = false;
            for (face, &(count, owner, opposite)) in &faces {
                if count != 1 || face.contains(&pt_index) || !bad_ids.contains(&owner) {
                    continue;
                }
                let face_points = self.get_vertices(face)?;
                let side_of_point = geometry::robust_orientation(&face_points, point)?;
                let side_of_cavity =
                    geometry::robust_orientation(&face_points, &self.vertices[opposite])?;
                if side_of_point == 0 || side_of_point != side_of_cavity {
                    bad_ids.remove(&owner);
                    evicted = true;
                }
            }
            if !evicted || bad_ids.is_empty() {
                return Ok(());
            }
            // Evictions can split the cavity; fragments no longer reachable
            // from the seeds cannot be filled from the point.
            self.prune_to_seed_component(bad_ids, seed_ids);
            if bad_ids.is_empty() {
                return Ok(());
            }
        }
    }

    /// Restrict the cavity to the component reachable from the seed
    /// simplices through shared facets. Disconnected fragments are
    /// circumsphere false positives on slivers; keeping them makes the hole
    /// unfillable from the point.
    fn prune_to_seed_component(&self, bad_ids: &mut FxHashSet<SimplexId>, seed_ids: &[SimplexId]) {
        let mut reachable: FxHashSet<SimplexId> = FxHashSet::default();
        let mut queue: VecDeque<SimplexId> = VecDeque::new();
        for &seed in seed_ids {
            if bad_ids.contains(&seed) && reachable.insert(seed) {
                queue.push_back(seed);
            }
        }
        while let Some(id) = queue.pop_front() {
            let simplex = self.simplex_by_id(id);
            for skip in 0..simplex.len() {
                let Some(incident) = self.facets.get(&facet_excluding(simplex, skip)) else {
                    continue;
                };
                for &neighbour in incident {
                    if neighbour != id
                        && bad_ids.contains(&neighbour)
                        && reachable.insert(neighbour)
                    {
                        queue.push_back(neighbour);
                    }
                }
            }
        }
        *bad_ids = reachable;
    }

    /// Connect vertex `pt_index`, which must lie outside the convex hull, to
    /// every hull face that faces it. Returns the created simplices; errors
    /// (and rolls back) if the vertex turns out to be inside the hull.
    pub fn extend_hull(
        &mut self,
        pt_index: usize,
    ) -> Result<FxHashSet<Simplex>, TriangulationError> {
        self.validate_vertex_index(pt_index)?;
        // Snapshot the hull before mutating: the boundary set changes as new
        // simplices attach to it.
        let hull_faces: Vec<Simplex> = self.boundary_facets.iter().cloned().collect();

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
            let attached: Vec<Simplex> = self.vertex_to_ids[pt_index]
                .iter()
                .map(|&id| self.simplex_by_id(id).clone())
                .collect();
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

        if simplex.is_empty() {
            let pt_index = self.add_vertex(point);
            let temporary_simplices = match self.extend_hull(pt_index) {
                Ok(simplices) => simplices,
                Err(err) => {
                    self.pop_vertex();
                    return Err(err);
                }
            };
            let (deleted_simplices, added_simplices) = match self
                .bowyer_watson(pt_index, None, &transform)
            {
                Ok(result) => result,
                // Value and assertion errors are raised before
                // bowyer_watson mutates, so only the hull extension
                // needs rolling back.
                Err(err @ (TriangulationError::Value(_) | TriangulationError::Assertion(_))) => {
                    let attached: Vec<Simplex> = self.vertex_to_ids[pt_index]
                        .iter()
                        .map(|&id| self.simplex_by_id(id).clone())
                        .collect();
                    for simplex in attached {
                        self.delete_simplex(&simplex)?;
                    }
                    self.pop_vertex();
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
            return Err(TriangulationError::Value(
                "Point lies outside of the specified simplex.".to_string(),
            ));
        }
        simplex = reduced_simplex;

        if simplex.len() == 1 {
            return Err(TriangulationError::Value(
                "Point already in triangulation.".to_string(),
            ));
        }

        let pt_index = self.add_vertex(point);
        match self.bowyer_watson(pt_index, Some(actual_simplex), &transform) {
            // Value and assertion errors are raised before bowyer_watson
            // mutates, so only the freshly pushed vertex needs rolling back.
            Err(err @ (TriangulationError::Value(_) | TriangulationError::Assertion(_))) => {
                self.pop_vertex();
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
        Ok(self.contains_simplex(&simplex))
    }

    /// The simplices containing `vertex`, after validating the index.
    pub fn vertex_to_simplices_for(
        &self,
        vertex: usize,
    ) -> Result<Vec<&Simplex>, TriangulationError> {
        self.validate_vertex_index(vertex)?;
        Ok(self.simplices_of(vertex).collect())
    }

    /// Volumes of all simplices, in iteration order of [`Self::simplices`].
    pub fn volumes(&self) -> Result<Vec<f64>, TriangulationError> {
        self.simplices()
            .map(|simplex| self.volume(simplex))
            .collect()
    }

    /// Whether all derived indexes (slab, id map, vertex incidence, facet
    /// incidence, boundary set, overfull count) are mutually consistent
    /// (a bookkeeping check, not a geometric one).
    pub fn reference_invariant(&self) -> bool {
        // Every registered simplex lives in the slab under its id, and every
        // live slab entry is registered.
        if self.slab.iter().filter(|slot| slot.is_some()).count() != self.ids.len() {
            return false;
        }
        for (simplex, &id) in &self.ids {
            if self.slab.get(id as usize).and_then(|slot| slot.as_ref()) != Some(simplex) {
                return false;
            }
            if simplex
                .iter()
                .any(|&vertex| !self.vertex_to_ids[vertex].contains(&id))
            {
                return false;
            }
        }

        // Vertex incidence points only at live simplices containing the
        // vertex.
        for (vertex, incident) in self.vertex_to_ids.iter().enumerate() {
            for &id in incident {
                match self.slab.get(id as usize).and_then(|slot| slot.as_ref()) {
                    Some(simplex) if simplex.contains(&vertex) => {}
                    _ => return false,
                }
            }
        }

        // The facet index, boundary set, and overfull count all match a
        // recount from scratch.
        let mut expected_facets: FxHashMap<Simplex, FxHashSet<SimplexId>> = FxHashMap::default();
        for (simplex, &id) in &self.ids {
            for skip in 0..simplex.len() {
                expected_facets
                    .entry(facet_excluding(simplex, skip))
                    .or_default()
                    .insert(id);
            }
        }
        if expected_facets.len() != self.facets.len() {
            return false;
        }
        let mut expected_overfull = 0;
        for (facet, expected_ids) in &expected_facets {
            let Some(actual) = self.facets.get(facet) else {
                return false;
            };
            let actual_ids: FxHashSet<SimplexId> = actual.iter().copied().collect();
            if actual.len() != actual_ids.len() || actual_ids != *expected_ids {
                return false;
            }
            if expected_ids.len() == 1 && !self.boundary_facets.contains(facet) {
                return false;
            }
            if expected_ids.len() > 2 {
                expected_overfull += 1;
            }
        }
        let expected_boundary = expected_facets
            .values()
            .filter(|ids| ids.len() == 1)
            .count();
        self.boundary_facets.len() == expected_boundary && self.overfull_facets == expected_overfull
    }

    /// Vertices on the convex hull (those belonging to a boundary face).
    /// Errors when a face belongs to more than two simplices, which means
    /// the triangulation is internally inconsistent.
    pub fn hull(&self) -> Result<FxHashSet<usize>, TriangulationError> {
        if self.overfull_facets > 0 {
            return Err(TriangulationError::Runtime(
                "Broken triangulation, a (N-1)-dimensional appears in more than 2 simplices."
                    .to_string(),
            ));
        }

        let mut hull = FxHashSet::default();
        for facet in &self.boundary_facets {
            hull.extend(facet.iter().copied());
        }
        Ok(hull)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simplex_set(triangulation: &Triangulation) -> FxHashSet<Simplex> {
        triangulation.simplices().cloned().collect()
    }

    fn all_vertices_connected(triangulation: &Triangulation) -> bool {
        (0..triangulation.vertices.len())
            .all(|vertex| triangulation.vertex_simplex_count(vertex) > 0)
    }

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
        assert!(simplices.contains(&&vec![0, 1, 3]));
        assert!(simplices.contains(&&vec![0, 2, 3]));
    }

    #[test]
    fn add_and_delete_simplex_keep_indexes_consistent() {
        let mut tri = sample_triangulation();
        assert!(tri.reference_invariant());

        tri.delete_simplex(&[0, 1, 3]).unwrap();
        assert!(tri.reference_invariant());
        assert!(!tri.has_simplex(&[0, 1, 3]).unwrap());
        assert_eq!(tri.num_simplices(), 1);

        tri.add_simplex(vec![3, 1, 0]).unwrap();
        assert!(tri.reference_invariant());
        assert!(tri.has_simplex(&[0, 1, 3]).unwrap());

        let err = tri.delete_simplex(&[0, 1, 2]).unwrap_err();
        assert!(matches!(err, TriangulationError::Value(_)));
    }

    #[test]
    fn containing_uses_facet_index_for_facet_queries() {
        let tri = sample_triangulation();

        // Shared interior facet, in arbitrary vertex order.
        let interior = tri.containing(&[3, 0]).unwrap();
        assert_eq!(
            interior,
            FxHashSet::from_iter([vec![0, 1, 3], vec![0, 2, 3]])
        );
        // Hull facet.
        assert_eq!(
            tri.containing(&[0, 1]).unwrap(),
            FxHashSet::from_iter([vec![0, 1, 3]])
        );
        // Non-existent facet.
        assert!(tri.containing(&[1, 2]).unwrap().is_empty());
        // Sub-facet face (single vertex in 2D) still works.
        assert_eq!(tri.containing(&[1]).unwrap().len(), 1);
    }

    #[test]
    fn hull_is_maintained_incrementally() {
        let mut tri = sample_triangulation();
        assert_eq!(tri.hull().unwrap(), FxHashSet::from_iter([0, 1, 2, 3]));

        tri.add_point(vec![2.0, 0.5], None, None).unwrap();
        assert_eq!(tri.hull().unwrap(), FxHashSet::from_iter([0, 1, 2, 3, 4]));
        assert!(tri.reference_invariant());
    }

    #[test]
    fn one_dimensional_triangulation_connects_adjacent_points() {
        let tri = Triangulation::new(vec![vec![2.0], vec![0.0], vec![1.0], vec![3.0]]).unwrap();

        assert_eq!(tri.dim, 1);
        assert_eq!(
            simplex_set(&tri),
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
            simplex_set(&tri),
            FxHashSet::from_iter([vec![0, 2], vec![1, 2]])
        );
    }

    #[test]
    fn one_dimensional_fine_grid_far_from_origin_constructs() {
        // Regression: cancellation in the circumcenter solve made the
        // volume-conservation assertion fail for small off-origin intervals.
        let coords: Vec<Vec<f64>> = (0..50).map(|i| vec![100.0 + i as f64 * 2e-5]).collect();
        let tri = Triangulation::new(coords).unwrap();

        assert_eq!(tri.num_simplices(), 49);
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
        assert_eq!(tri.num_simplices(), 4);
        assert!(all_vertices_connected(&tri));
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
        let simplices_before = simplex_set(&tri);
        let n_vertices_before = tri.vertices.len();

        let err = tri
            .add_point(vec![0.5 + 2e-10, 0.5 + 2e-10], None, None)
            .unwrap_err();

        assert!(matches!(err, TriangulationError::Value(_)));
        assert_eq!(simplex_set(&tri), simplices_before);
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
        assert!(all_vertices_connected(&tri));
    }
}
