use std::collections::{HashMap, HashSet, VecDeque};

use numpy::PyArray2;
use pyo3::exceptions::{
    PyAssertionError, PyNotImplementedError, PyRuntimeError, PyTypeError, PyValueError,
};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyList, PySet, PyTuple};
use thiserror::Error;

use crate::geometry::{self, GeometryError};

pub type PointIndex = usize;
pub type Simplex = Vec<PointIndex>;

const DEFAULT_EPS: f64 = 1e-8;

#[derive(Debug, Error)]
pub enum TriangulationError {
    #[error("{0}")]
    Value(String),
    #[error("{0}")]
    Runtime(String),
    #[error("{0}")]
    Assertion(String),
    #[error(transparent)]
    Geometry(#[from] GeometryError),
}

impl TriangulationError {
    fn into_pyerr(self) -> PyErr {
        match self {
            Self::Value(message) => PyValueError::new_err(message),
            Self::Runtime(message) => PyRuntimeError::new_err(message),
            Self::Assertion(message) => PyAssertionError::new_err(message),
            Self::Geometry(error) => PyValueError::new_err(error.to_string()),
        }
    }
}

pub(crate) fn parse_point(obj: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    let Ok(iter) = obj.try_iter() else {
        return Err(PyTypeError::new_err("Expected an iterable of floats"));
    };
    let mut point = Vec::new();
    for item in iter {
        point.push(item?.extract::<f64>()?);
    }
    Ok(point)
}

pub(crate) fn parse_points(
    obj: &Bound<'_, PyAny>,
    type_error_message: &str,
) -> PyResult<Vec<Vec<f64>>> {
    let Ok(iter) = obj.try_iter() else {
        return Err(PyTypeError::new_err(type_error_message.to_string()));
    };

    let mut points = Vec::new();
    for item in iter {
        let item = item?;
        let Ok(row_iter) = item.try_iter() else {
            return Err(PyTypeError::new_err(type_error_message.to_string()));
        };
        let mut row = Vec::new();
        for value in row_iter {
            row.push(value?.extract::<f64>()?);
        }
        points.push(row);
    }
    Ok(points)
}

pub(crate) fn parse_simplex(obj: &Bound<'_, PyAny>) -> PyResult<Simplex> {
    let Ok(iter) = obj.try_iter() else {
        return Err(PyTypeError::new_err("Expected an iterable of vertex indices"));
    };
    let mut simplex = Vec::new();
    for item in iter {
        simplex.push(item?.extract::<usize>()?);
    }
    simplex.sort_unstable();
    Ok(simplex)
}

pub(crate) fn parse_simplex_set(obj: &Bound<'_, PyAny>) -> PyResult<HashSet<Simplex>> {
    let Ok(iter) = obj.try_iter() else {
        return Err(PyTypeError::new_err("Expected an iterable of simplices"));
    };
    let mut simplices = HashSet::new();
    for item in iter {
        simplices.insert(parse_simplex(&item?)?);
    }
    Ok(simplices)
}

pub(crate) fn parse_index_set(obj: &Bound<'_, PyAny>) -> PyResult<HashSet<usize>> {
    let Ok(iter) = obj.try_iter() else {
        return Err(PyTypeError::new_err("Expected an iterable of vertex indices"));
    };
    let mut indices = HashSet::new();
    for item in iter {
        indices.insert(item?.extract::<usize>()?);
    }
    Ok(indices)
}

pub(crate) fn parse_optional_transform(
    obj: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<Vec<Vec<f64>>>> {
    match obj {
        None => Ok(None),
        Some(value) if value.is_none() => Ok(None),
        Some(value) => Ok(Some(parse_points(
            value,
            "Expected an N x N transform matrix",
        )?)),
    }
}

pub(crate) fn point_tuple(py: Python<'_>, point: &[f64]) -> Py<PyTuple> {
    PyTuple::new_bound(py, point.iter().copied()).into()
}

pub(crate) fn simplex_tuple(py: Python<'_>, simplex: &[usize]) -> Py<PyTuple> {
    PyTuple::new_bound(py, simplex.iter().copied()).into()
}

pub(crate) fn simplex_set_py(py: Python<'_>, simplices: &HashSet<Simplex>) -> PyResult<Py<PyAny>> {
    let tuples: Vec<Py<PyAny>> = simplices
        .iter()
        .map(|simplex| simplex_tuple(py, simplex).into())
        .collect();
    Ok(PySet::new_bound(py, &tuples)?.into())
}

pub(crate) fn point_list_py(py: Python<'_>, points: &[Vec<f64>]) -> PyResult<Py<PyAny>> {
    let tuples: Vec<Py<PyAny>> = points
        .iter()
        .map(|point| point_tuple(py, point).into())
        .collect();
    Ok(PyList::new_bound(py, tuples).into())
}

pub(crate) fn simplex_list_py(py: Python<'_>, simplices: &[Simplex]) -> PyResult<Py<PyAny>> {
    let tuples: Vec<Py<PyAny>> = simplices
        .iter()
        .map(|simplex| simplex_tuple(py, simplex).into())
        .collect();
    Ok(PyList::new_bound(py, tuples).into())
}

fn identity_transform(dim: usize) -> Vec<Vec<f64>> {
    (0..dim)
        .map(|row| {
            (0..dim)
                .map(|col| if row == col { 1.0 } else { 0.0 })
                .collect()
        })
        .collect()
}

fn validate_transform(transform: &Option<Vec<Vec<f64>>>, dim: usize) -> Result<(), TriangulationError> {
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
    for col in 0..dim {
        let mut value = 0.0;
        for row in 0..dim {
            value += point[row] * transform[row][col];
        }
        result[col] = value;
    }
    result
}

fn barycentric_alpha(vertices: &[Vec<f64>], point: &[f64]) -> Result<Vec<f64>, TriangulationError> {
    let dim = point.len();
    let x0 = &vertices[0];
    let mut matrix = vec![vec![0.0; dim]; dim];
    let mut rhs = vec![0.0; dim];

    for row in 0..dim {
        rhs[row] = point[row] - x0[row];
        for col in 0..dim {
            matrix[row][col] = vertices[col + 1][row] - x0[row];
        }
    }

    let flat: Vec<f64> = matrix.iter().flat_map(|row| row.iter().copied()).collect();
    let mat = nalgebra::DMatrix::from_row_slice(dim, dim, &flat);
    let rhs = nalgebra::DVector::from_column_slice(&rhs);
    mat.lu()
        .solve(&rhs)
        .map(|solution| solution.iter().copied().collect())
        .ok_or_else(|| TriangulationError::Geometry(GeometryError::SingularMatrix))
}

fn combinations(source: &[usize], k: usize, out: &mut Vec<Simplex>, current: &mut Vec<usize>, start: usize) {
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
    (a - b).abs() <= 1e-8 + 1e-5 * b.abs()
}

#[derive(Debug, Clone)]
pub struct Triangulation {
    pub vertices: Vec<Vec<f64>>,
    pub simplices: HashSet<Simplex>,
    pub vertex_to_simplices: Vec<HashSet<Simplex>>,
    pub dim: usize,
}

impl Triangulation {
    pub fn new(coords: Vec<Vec<f64>>) -> Result<Self, TriangulationError> {
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
        if dim == 1 {
            return Err(TriangulationError::Value(
                "Triangulation class only supports dim >= 2".to_string(),
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
        if geometry::matrix_rank(&vectors, 1e-12)? < dim {
            return Err(TriangulationError::Value(
                "Initial simplex has zero volumes (the points are linearly dependent)".to_string(),
            ));
        }

        let mut triangulation = Self {
            vertices: coords[..=dim].to_vec(),
            simplices: HashSet::new(),
            vertex_to_simplices: vec![HashSet::new(); dim + 1],
            dim,
        };
        triangulation.add_simplex((0..=dim).collect())?;

        for point in coords.iter().skip(dim + 1) {
            triangulation.add_point(point.clone(), None, None)?;
        }

        Ok(triangulation)
    }

    pub fn add_simplex(&mut self, mut simplex: Simplex) -> Result<(), TriangulationError> {
        simplex.sort_unstable();
        if simplex.len() != self.dim + 1 {
            return Err(TriangulationError::Value(format!(
                "Simplex must contain {} vertices",
                self.dim + 1
            )));
        }
        if simplex.iter().any(|&vertex| vertex >= self.vertices.len()) {
            return Err(TriangulationError::Value(
                "Simplex references a missing vertex".to_string(),
            ));
        }
        if self.simplices.insert(simplex.clone()) {
            for &vertex in &simplex {
                self.vertex_to_simplices[vertex].insert(simplex.clone());
            }
        }
        Ok(())
    }

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

    pub fn get_vertices(&self, indices: &[PointIndex]) -> Vec<Vec<f64>> {
        indices.iter().map(|&idx| self.vertices[idx].clone()).collect()
    }

    pub fn locate_point(&self, point: &[f64]) -> Option<Simplex> {
        self.simplices.iter().find_map(|simplex| {
            let vertices = self.get_vertices(simplex);
            geometry::point_in_simplex(point, &vertices, DEFAULT_EPS).then(|| simplex.clone())
        })
    }

    pub fn get_reduced_simplex(
        &self,
        point: &[f64],
        simplex: &[usize],
        eps: f64,
    ) -> Result<Simplex, TriangulationError> {
        let simplex = if simplex.len() != self.dim + 1 {
            let containing = self.containing(simplex);
            let Some(first) = containing.into_iter().next() else {
                return Ok(Vec::new());
            };
            first
        } else {
            simplex.to_vec()
        };

        let vertices = self.get_vertices(&simplex);
        let alpha = barycentric_alpha(&vertices, point)?;
        let sum_alpha = alpha.iter().sum::<f64>();

        if alpha.iter().any(|value| *value < -eps) || sum_alpha > 1.0 + eps {
            return Ok(Vec::new());
        }

        let mut result: Vec<usize> = alpha
            .iter()
            .enumerate()
            .filter_map(|(idx, value)| (*value > eps).then_some(simplex[idx + 1]))
            .collect();
        if sum_alpha < 1.0 - eps {
            result.insert(0, simplex[0]);
        }
        Ok(result)
    }

    pub fn point_in_simplex(
        &self,
        point: &[f64],
        simplex: &[usize],
        eps: f64,
    ) -> Result<bool, TriangulationError> {
        Ok(geometry::point_in_simplex(point, &self.get_vertices(simplex), eps))
    }

    pub fn faces(
        &self,
        dim: Option<usize>,
        simplices: Option<&HashSet<Simplex>>,
        vertices: Option<&HashSet<usize>>,
    ) -> Result<Vec<Simplex>, TriangulationError> {
        if simplices.is_some() && vertices.is_some() {
            return Err(TriangulationError::Value(
                "Only one of simplices and vertices is allowed.".to_string(),
            ));
        }

        let face_size = dim.unwrap_or(self.dim);
        let simplex_pool: Vec<Simplex> = if let Some(vertices) = vertices {
            let mut pool = HashSet::new();
            for &vertex in vertices {
                if vertex < self.vertex_to_simplices.len() {
                    pool.extend(self.vertex_to_simplices[vertex].iter().cloned());
                }
            }
            pool.into_iter().collect()
        } else if let Some(simplices) = simplices {
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

    pub fn containing(&self, face: &[usize]) -> HashSet<Simplex> {
        if face.is_empty() {
            return HashSet::new();
        }
        let mut result = self.vertex_to_simplices[face[0]].clone();
        for &vertex in &face[1..] {
            result.retain(|simplex| self.vertex_to_simplices[vertex].contains(simplex));
        }
        result
    }

    pub fn circumscribed_circle(
        &self,
        simplex: &[usize],
        transform: &Option<Vec<Vec<f64>>>,
    ) -> Result<(Vec<f64>, f64), TriangulationError> {
        validate_transform(transform, self.dim)?;
        let transform = transform.clone().unwrap_or_else(|| identity_transform(self.dim));
        let points: Vec<Vec<f64>> = self
            .get_vertices(simplex)
            .into_iter()
            .map(|point| apply_transform(&point, &transform))
            .collect();
        Ok(geometry::circumsphere(&points)?)
    }

    pub fn point_in_circumcircle(
        &self,
        pt_index: usize,
        simplex: &[usize],
        transform: &Option<Vec<Vec<f64>>>,
    ) -> Result<bool, TriangulationError> {
        validate_transform(transform, self.dim)?;
        let transform = transform.clone().unwrap_or_else(|| identity_transform(self.dim));
        let (center, radius) = self.circumscribed_circle(simplex, &Some(transform.clone()))?;
        let point = apply_transform(&self.vertices[pt_index], &transform);
        let distance = geometry::fast_norm(
            &center
                .iter()
                .zip(&point)
                .map(|(a, b)| a - b)
                .collect::<Vec<_>>(),
        );
        Ok(distance < radius * (1.0 + DEFAULT_EPS))
    }

    pub fn bowyer_watson(
        &mut self,
        pt_index: usize,
        containing_simplex: Option<Simplex>,
        transform: &Option<Vec<Vec<f64>>>,
    ) -> Result<(HashSet<Simplex>, HashSet<Simplex>), TriangulationError> {
        validate_transform(transform, self.dim)?;

        let mut queue = VecDeque::new();
        let mut queued = HashSet::new();
        let mut done_simplices = HashSet::new();
        let mut bad_triangles = HashSet::new();

        if let Some(simplex) = containing_simplex {
            queued.insert(simplex.clone());
            queue.push_back(simplex);
        } else {
            for simplex in self.vertex_to_simplices[pt_index].clone() {
                if queued.insert(simplex.clone()) {
                    queue.push_back(simplex);
                }
            }
        }

        while let Some(simplex) = queue.pop_front() {
            if done_simplices.contains(&simplex) {
                continue;
            }
            done_simplices.insert(simplex.clone());
            if !self.simplices.contains(&simplex) {
                continue;
            }

            if self.point_in_circumcircle(pt_index, &simplex, transform)? {
                self.delete_simplex(&simplex)?;
                bad_triangles.insert(simplex.clone());

                let simplex_vertices: HashSet<usize> = simplex.iter().copied().collect();
                let mut neighbours = HashSet::new();
                for &vertex in &simplex {
                    neighbours.extend(self.vertex_to_simplices[vertex].iter().cloned());
                }

                for neighbour in neighbours {
                    if done_simplices.contains(&neighbour) {
                        continue;
                    }
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

        let faces = self.faces(None, Some(&bad_triangles), None)?;
        let mut multiplicities: HashMap<Simplex, usize> = HashMap::new();
        for face in &faces {
            *multiplicities.entry(face.clone()).or_insert(0) += 1;
        }
        let hole_faces: Vec<Simplex> = faces
            .into_iter()
            .filter(|face| multiplicities.get(face).copied().unwrap_or_default() < 2)
            .collect();

        for face in hole_faces {
            if face.contains(&pt_index) {
                continue;
            }
            let mut simplex = face;
            simplex.push(pt_index);
            simplex.sort_unstable();

            if self.volume(&simplex)? < 1e-8 {
                continue;
            }
            self.add_simplex(simplex)?;
        }

        let new_triangles = self.vertex_to_simplices[pt_index].clone();
        let deleted_simplices: HashSet<Simplex> =
            bad_triangles.difference(&new_triangles).cloned().collect();
        let new_simplices: HashSet<Simplex> =
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

    pub fn extend_hull(&mut self, pt_index: usize) -> Result<HashSet<Simplex>, TriangulationError> {
        let faces = self.faces(None, None, None)?;
        let mut multiplicities: HashMap<Simplex, usize> = HashMap::new();
        for face in &faces {
            *multiplicities.entry(face.clone()).or_insert(0) += 1;
        }
        let hull_faces: Vec<Simplex> = multiplicities
            .into_iter()
            .filter_map(|(face, count)| (count == 1).then_some(face))
            .collect();

        let hull = self.hull()?;
        let hull_points: Vec<Vec<f64>> = hull.iter().map(|idx| self.vertices[*idx].clone()).collect();
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
        let mut new_simplices = HashSet::new();

        for face in hull_faces {
            let pts_face = self.get_vertices(&face);
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
                self.simplices.remove(&simplex);
            }
            self.vertex_to_simplices.pop();
            self.vertices.pop();
            return Err(TriangulationError::Value(
                "Candidate vertex is inside the hull.".to_string(),
            ));
        }

        Ok(new_simplices)
    }

    pub fn add_point(
        &mut self,
        point: Vec<f64>,
        simplex: Option<Simplex>,
        transform: Option<Vec<Vec<f64>>>,
    ) -> Result<(HashSet<Simplex>, HashSet<Simplex>), TriangulationError> {
        if point.len() != self.dim {
            return Err(TriangulationError::Value(
                "Coordinates dimension mismatch".to_string(),
            ));
        }
        validate_transform(&transform, self.dim)?;

        let mut simplex = simplex.unwrap_or_else(|| self.locate_point(&point).unwrap_or_default());
        let actual_simplex = simplex.clone();
        self.vertex_to_simplices.push(HashSet::new());

        if simplex.is_empty() {
            self.vertices.push(point);
            let pt_index = self.vertices.len() - 1;
            let temporary_simplices = self.extend_hull(pt_index)?;
            let (deleted_simplices, added_simplices) =
                self.bowyer_watson(pt_index, None, &transform)?;

            let deleted: HashSet<Simplex> = deleted_simplices
                .difference(&temporary_simplices)
                .cloned()
                .collect();
            let mut added = added_simplices.clone();
            for simplex in temporary_simplices.difference(&deleted_simplices) {
                added.insert(simplex.clone());
            }
            return Ok((deleted, added));
        }

        let reduced_simplex = self.get_reduced_simplex(&point, &simplex, DEFAULT_EPS)?;
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
        self.bowyer_watson(pt_index, Some(actual_simplex), &transform)
    }

    pub fn volume(&self, simplex: &[usize]) -> Result<f64, TriangulationError> {
        Ok(geometry::volume(&self.get_vertices(simplex))?)
    }

    pub fn volumes(&self) -> Result<Vec<f64>, TriangulationError> {
        self.simplices
            .iter()
            .map(|simplex| self.volume(simplex))
            .collect()
    }

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

    pub fn hull(&self) -> Result<HashSet<usize>, TriangulationError> {
        let faces = self.faces(None, None, None)?;
        let mut counts: HashMap<Simplex, usize> = HashMap::new();
        for face in faces {
            let count = counts.entry(face).or_insert(0);
            *count += 1;
            if *count > 2 {
                return Err(TriangulationError::Runtime(
                    "Broken triangulation, a (N-1)-dimensional appears in more than 2 simplices."
                        .to_string(),
                ));
            }
        }

        let mut hull = HashSet::new();
        for (face, count) in counts {
            if count == 1 {
                hull.extend(face);
            }
        }
        Ok(hull)
    }
}

#[pyclass(name = "Triangulation")]
pub struct PyTriangulation {
    pub core: Triangulation,
}

#[pyclass]
pub struct PyFacesIter {
    items: Vec<Simplex>,
    index: usize,
}

#[pymethods]
impl PyFacesIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> Option<Py<PyTuple>> {
        let item = self.items.get(self.index)?;
        self.index += 1;
        Some(simplex_tuple(py, item))
    }
}

#[pymethods]
impl PyTriangulation {
    #[new]
    fn new(coords: &Bound<'_, PyAny>) -> PyResult<Self> {
        let coords = parse_points(coords, "Please provide a 2-dimensional list of points")?;
        let core = Triangulation::new(coords).map_err(TriangulationError::into_pyerr)?;
        Ok(Self { core })
    }

    fn add_simplex(&mut self, simplex: &Bound<'_, PyAny>) -> PyResult<()> {
        self.core
            .add_simplex(parse_simplex(simplex)?)
            .map_err(TriangulationError::into_pyerr)
    }

    fn delete_simplex(&mut self, simplex: &Bound<'_, PyAny>) -> PyResult<()> {
        let simplex = parse_simplex(simplex)?;
        self.core
            .delete_simplex(&simplex)
            .map_err(TriangulationError::into_pyerr)
    }

    #[pyo3(signature = (point, simplex=None, transform=None))]
    fn add_point(
        &mut self,
        py: Python<'_>,
        point: &Bound<'_, PyAny>,
        simplex: Option<&Bound<'_, PyAny>>,
        transform: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let point = parse_point(point)?;
        let simplex = match simplex {
            None => None,
            Some(value) if value.is_none() => None,
            Some(value) => Some(parse_simplex(value)?),
        };
        let transform = parse_optional_transform(transform)?;
        let (deleted, added) = self
            .core
            .add_point(point, simplex, transform)
            .map_err(TriangulationError::into_pyerr)?;
        Ok((simplex_set_py(py, &deleted)?, simplex_set_py(py, &added)?))
    }

    #[getter(vertices)]
    fn vertices_property(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        point_list_py(py, &self.core.vertices)
    }

    #[getter(simplices)]
    fn simplices_property(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        simplex_set_py(py, &self.core.simplices)
    }

    #[getter(vertex_to_simplices)]
    fn vertex_to_simplices_property(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let sets: Vec<Py<PyAny>> = self
            .core
            .vertex_to_simplices
            .iter()
            .map(|simplices| simplex_set_py(py, simplices))
            .collect::<PyResult<_>>()?;
        Ok(PyList::new_bound(py, sets).into())
    }

    #[getter(hull)]
    fn hull_property(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let hull = self.core.hull().map_err(TriangulationError::into_pyerr)?;
        let vertices: Vec<usize> = hull.into_iter().collect();
        Ok(PySet::new_bound(py, &vertices)?.into())
    }

    #[getter(dim)]
    fn dim_property(&self) -> usize {
        self.core.dim
    }

    #[pyo3(name = "get_vertices")]
    fn get_vertices_method(
        &self,
        py: Python<'_>,
        indices: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let indices = parse_simplex(indices)?;
        point_list_py(py, &self.core.get_vertices(&indices))
    }

    fn locate_point(&self, py: Python<'_>, point: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let point = parse_point(point)?;
        match self.core.locate_point(&point) {
            Some(simplex) => Ok(simplex_tuple(py, &simplex).into()),
            None => Ok(PyTuple::empty_bound(py).into()),
        }
    }

    #[pyo3(signature = (point, simplex, eps=DEFAULT_EPS))]
    fn get_reduced_simplex(
        &self,
        py: Python<'_>,
        point: &Bound<'_, PyAny>,
        simplex: &Bound<'_, PyAny>,
        eps: f64,
    ) -> PyResult<Py<PyAny>> {
        let point = parse_point(point)?;
        let simplex = parse_simplex(simplex)?;
        let reduced = self
            .core
            .get_reduced_simplex(&point, &simplex, eps)
            .map_err(TriangulationError::into_pyerr)?;
        Ok(simplex_tuple(py, &reduced).into())
    }

    #[pyo3(signature = (simplex, transform=None))]
    fn circumscribed_circle(
        &self,
        py: Python<'_>,
        simplex: &Bound<'_, PyAny>,
        transform: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<(Py<PyTuple>, f64)> {
        let simplex = parse_simplex(simplex)?;
        let transform = parse_optional_transform(transform)?;
        let (center, radius) = self
            .core
            .circumscribed_circle(&simplex, &transform)
            .map_err(TriangulationError::into_pyerr)?;
        Ok((point_tuple(py, &center), radius))
    }

    #[pyo3(signature = (pt_index, simplex, transform=None))]
    fn point_in_circumcircle(
        &self,
        pt_index: usize,
        simplex: &Bound<'_, PyAny>,
        transform: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<bool> {
        let simplex = parse_simplex(simplex)?;
        let transform = parse_optional_transform(transform)?;
        self.core
            .point_in_circumcircle(pt_index, &simplex, &transform)
            .map_err(TriangulationError::into_pyerr)
    }

    #[pyo3(name = "point_in_cicumcircle", signature = (pt_index, simplex, transform=None))]
    fn point_in_cicumcircle(
        &self,
        pt_index: usize,
        simplex: &Bound<'_, PyAny>,
        transform: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<bool> {
        self.point_in_circumcircle(pt_index, simplex, transform)
    }

    fn volume(&self, simplex: &Bound<'_, PyAny>) -> PyResult<f64> {
        let simplex = parse_simplex(simplex)?;
        self.core
            .volume(&simplex)
            .map_err(TriangulationError::into_pyerr)
    }

    fn volumes(&self) -> PyResult<Vec<f64>> {
        self.core.volumes().map_err(TriangulationError::into_pyerr)
    }

    #[pyo3(signature = (dim=None, simplices=None, vertices=None))]
    fn faces(
        &self,
        dim: Option<usize>,
        simplices: Option<&Bound<'_, PyAny>>,
        vertices: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyFacesIter> {
        let simplices = match simplices {
            None => None,
            Some(value) if value.is_none() => None,
            Some(value) => Some(parse_simplex_set(value)?),
        };
        let vertices = match vertices {
            None => None,
            Some(value) if value.is_none() => None,
            Some(value) => Some(parse_index_set(value)?),
        };
        let items = self
            .core
            .faces(dim, simplices.as_ref(), vertices.as_ref())
            .map_err(TriangulationError::into_pyerr)?;
        Ok(PyFacesIter { items, index: 0 })
    }

    fn containing(&self, py: Python<'_>, face: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let face = parse_simplex(face)?;
        simplex_set_py(py, &self.core.containing(&face))
    }

    fn reference_invariant(&self) -> bool {
        self.core.reference_invariant()
    }

    #[getter(default_transform)]
    fn default_transform_property<'py>(&self, py: Python<'py>) -> PyResult<Py<PyArray2<f64>>> {
        let identity = identity_transform(self.core.dim);
        Ok(PyArray2::from_vec2_bound(py, &identity)?.into())
    }

    #[pyo3(signature = (point, simplex, eps=DEFAULT_EPS))]
    fn point_in_simplex(
        &self,
        point: &Bound<'_, PyAny>,
        simplex: &Bound<'_, PyAny>,
        eps: f64,
    ) -> PyResult<bool> {
        let point = parse_point(point)?;
        let simplex = parse_simplex(simplex)?;
        self.core
            .point_in_simplex(&point, &simplex, eps)
            .map_err(TriangulationError::into_pyerr)
    }

    fn vertex_invariant(&self, _vertex: usize) -> PyResult<bool> {
        Err(PyNotImplementedError::new_err("vertex_invariant"))
    }

    fn convex_invariant(&self, _vertex: usize) -> PyResult<bool> {
        Err(PyNotImplementedError::new_err("convex_invariant"))
    }
}
