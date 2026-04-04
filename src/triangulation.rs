use std::collections::VecDeque;
use std::sync::RwLock;

use numpy::PyArray2;
use pyo3::exceptions::{
    PyAssertionError, PyIndexError, PyNotImplementedError, PyRuntimeError, PyTypeError,
    PyValueError, PyZeroDivisionError,
};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyList, PyModule, PySet, PyTuple};
use rustc_hash::{FxHashMap, FxHashSet};
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
    Index(String),
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
            Self::Index(message) => PyIndexError::new_err(message),
            Self::Runtime(message) => PyRuntimeError::new_err(message),
            Self::Assertion(message) => PyAssertionError::new_err(message),
            Self::Geometry(error) => match error {
                GeometryError::SingularMatrix => PyValueError::new_err(error.to_string()),
                _ => PyValueError::new_err(error.to_string()),
            },
        }
    }
}

fn index_out_of_range() -> TriangulationError {
    TriangulationError::Index("list index out of range".to_string())
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

fn ensure_sized(obj: &Bound<'_, PyAny>, type_error_message: &str) -> PyResult<()> {
    obj.len()
        .map(|_| ())
        .map_err(|_| PyTypeError::new_err(type_error_message.to_string()))
}

pub(crate) fn parse_points_sized(
    obj: &Bound<'_, PyAny>,
    type_error_message: &str,
) -> PyResult<Vec<Vec<f64>>> {
    parse_points_impl(obj, type_error_message, true)
}

fn parse_points_impl(
    obj: &Bound<'_, PyAny>,
    type_error_message: &str,
    require_sized: bool,
) -> PyResult<Vec<Vec<f64>>> {
    if require_sized {
        ensure_sized(obj, type_error_message)?;
    }
    let Ok(iter) = obj.try_iter() else {
        return Err(PyTypeError::new_err(type_error_message.to_string()));
    };

    let mut points = Vec::new();
    for item in iter {
        let item = item?;
        if require_sized {
            ensure_sized(&item, type_error_message)?;
        }
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

pub(crate) fn parse_signed_indices(obj: &Bound<'_, PyAny>) -> PyResult<Vec<isize>> {
    let Ok(iter) = obj.try_iter() else {
        return Err(PyTypeError::new_err(
            "Expected an iterable of vertex indices",
        ));
    };
    let mut indices = Vec::new();
    for item in iter {
        indices.push(item?.extract::<isize>()?);
    }
    Ok(indices)
}

pub(crate) fn parse_signed_simplex_set(obj: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<isize>>> {
    let Ok(iter) = obj.try_iter() else {
        return Err(PyTypeError::new_err("Expected an iterable of simplices"));
    };
    let mut simplices = Vec::new();
    for item in iter {
        simplices.push(parse_signed_indices(&item?)?);
    }
    Ok(simplices)
}

pub(crate) fn parse_signed_index_set(obj: &Bound<'_, PyAny>) -> PyResult<FxHashSet<isize>> {
    let Ok(iter) = obj.try_iter() else {
        return Err(PyTypeError::new_err(
            "Expected an iterable of vertex indices",
        ));
    };
    let mut indices = FxHashSet::default();
    for item in iter {
        indices.insert(item?.extract::<isize>()?);
    }
    Ok(indices)
}

pub(crate) fn parse_optional_transform(
    obj: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<Vec<Vec<f64>>>> {
    match obj {
        None => Ok(None),
        Some(value) if value.is_none() => Ok(None),
        Some(value) => Ok(Some(parse_points_sized(
            value,
            "Expected an N x N transform matrix",
        )?)),
    }
}

pub(crate) fn normalize_index(index: isize, len: usize) -> Result<usize, TriangulationError> {
    let normalized = if index < 0 {
        len as isize + index
    } else {
        index
    };
    if normalized < 0 || normalized >= len as isize {
        return Err(index_out_of_range());
    }
    Ok(normalized as usize)
}

pub(crate) fn normalize_indices(
    indices: &[isize],
    len: usize,
) -> Result<Vec<usize>, TriangulationError> {
    indices
        .iter()
        .map(|&index| normalize_index(index, len))
        .collect()
}

pub(crate) fn canonicalize_simplex(
    indices: &[isize],
    len: usize,
) -> Result<Simplex, TriangulationError> {
    let mut simplex = normalize_indices(indices, len)?;
    simplex.sort_unstable();
    Ok(simplex)
}

pub(crate) fn normalize_index_set(
    indices: &FxHashSet<isize>,
    len: usize,
) -> Result<FxHashSet<usize>, TriangulationError> {
    indices
        .iter()
        .map(|&index| normalize_index(index, len))
        .collect()
}

fn ordered_indices_from_py(obj: &Bound<'_, PyAny>, len: usize) -> PyResult<Vec<usize>> {
    normalize_indices(&parse_signed_indices(obj)?, len).map_err(TriangulationError::into_pyerr)
}

fn canonical_simplex_from_py(obj: &Bound<'_, PyAny>, len: usize) -> PyResult<Simplex> {
    canonicalize_simplex(&parse_signed_indices(obj)?, len).map_err(TriangulationError::into_pyerr)
}

fn simplex_set_from_py(obj: &Bound<'_, PyAny>, len: usize) -> PyResult<FxHashSet<Simplex>> {
    let simplices = parse_signed_simplex_set(obj)?;
    let mut normalized = FxHashSet::default();
    for simplex in simplices {
        normalized
            .insert(normalize_indices(&simplex, len).map_err(TriangulationError::into_pyerr)?);
    }
    Ok(normalized)
}

fn vertex_index_set_from_py(obj: &Bound<'_, PyAny>, len: usize) -> PyResult<FxHashSet<usize>> {
    normalize_index_set(&parse_signed_index_set(obj)?, len).map_err(TriangulationError::into_pyerr)
}

pub(crate) fn point_tuple(py: Python<'_>, point: &[f64]) -> Py<PyTuple> {
    PyTuple::new(py, point.iter().copied()).unwrap().into()
}

pub(crate) fn simplex_tuple(py: Python<'_>, simplex: &[usize]) -> Py<PyTuple> {
    PyTuple::new(py, simplex.iter().copied()).unwrap().into()
}

pub(crate) fn simplex_set_py(
    py: Python<'_>,
    simplices: &FxHashSet<Simplex>,
) -> PyResult<Py<PyAny>> {
    let tuples: Vec<Py<PyAny>> = simplices
        .iter()
        .map(|simplex| simplex_tuple(py, simplex).into())
        .collect();
    Ok(PySet::new(py, &tuples)?.into())
}

pub(crate) fn point_list_py(py: Python<'_>, points: &[Vec<f64>]) -> PyResult<Py<PyAny>> {
    let tuples: Vec<Py<PyAny>> = points
        .iter()
        .map(|point| point_tuple(py, point).into())
        .collect();
    Ok(PyList::new(py, tuples)?.into())
}

pub(crate) fn index_list_py(py: Python<'_>, indices: &[usize]) -> PyResult<Py<PyAny>> {
    Ok(PyList::new(py, indices.iter().copied())?.into())
}

pub(crate) fn signed_index_list_py(py: Python<'_>, indices: &[isize]) -> PyResult<Py<PyAny>> {
    Ok(PyList::new(py, indices.iter().copied())?.into())
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
    for col in 0..dim {
        let mut value = 0.0;
        for row in 0..dim {
            value += point[row] * transform[row][col];
        }
        result[col] = value;
    }
    result
}

fn numpy_linalg_error(py: Python<'_>, message: &str) -> PyErr {
    PyModule::import(py, "numpy.linalg")
        .and_then(|module| {
            let error_type = module.getattr("LinAlgError")?;
            let args = PyTuple::new(py, [message]).unwrap();
            let value = error_type.call1(args)?;
            Ok(PyErr::from_value(value))
        })
        .unwrap_or_else(|_| PyValueError::new_err(message.to_string()))
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
        .ok_or(TriangulationError::Geometry(GeometryError::SingularMatrix))
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
    (a - b).abs() <= 1e-8 + 1e-5 * b.abs()
}

#[derive(Debug)]
pub struct Triangulation {
    pub vertices: Vec<Vec<f64>>,
    pub simplices: FxHashSet<Simplex>,
    pub vertex_to_simplices: Vec<FxHashSet<Simplex>>,
    pub dim: usize,
    last_simplex: RwLock<Option<Simplex>>,
}

impl Clone for Triangulation {
    fn clone(&self) -> Self {
        Self {
            vertices: self.vertices.clone(),
            simplices: self.simplices.clone(),
            vertex_to_simplices: self.vertex_to_simplices.clone(),
            dim: self.dim,
            last_simplex: RwLock::new(self.last_simplex.read().unwrap().clone()),
        }
    }
}

impl Triangulation {
    fn validate_coords(coords: &[Vec<f64>]) -> Result<usize, TriangulationError> {
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
        if geometry::numpy_matrix_rank(&vectors)? < dim {
            return Err(TriangulationError::Value(
                "Initial simplex has zero volumes (the points are linearly dependent)".to_string(),
            ));
        }

        Ok(dim)
    }

    fn validate_point_dim(&self, point: &[f64]) -> Result<(), TriangulationError> {
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

    pub fn new(coords: Vec<Vec<f64>>) -> Result<Self, TriangulationError> {
        let dim = Self::validate_coords(&coords)?;
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
                triangulation.get_reduced_simplex(&point, &actual_simplex, DEFAULT_EPS)?;
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
            if geometry::point_in_simplex(point, &vertices, DEFAULT_EPS)? {
                *self.last_simplex.write().unwrap() = Some(simplex.clone());
                return Ok(Some(simplex.clone()));
            }
        }
        Ok(None)
    }

    fn barycentric_alpha_for_simplex(
        &self,
        simplex: &[usize],
        point: &[f64],
    ) -> Result<Vec<f64>, TriangulationError> {
        let dim = point.len();
        let x0 = &self.vertices[simplex[0]];
        let mut matrix = vec![vec![0.0; dim]; dim];
        let mut rhs = vec![0.0; dim];

        for row in 0..dim {
            rhs[row] = point[row] - x0[row];
            for col in 0..dim {
                matrix[row][col] = self.vertices[simplex[col + 1]][row] - x0[row];
            }
        }

        let flat: Vec<f64> = matrix.iter().flat_map(|row| row.iter().copied()).collect();
        let mat = nalgebra::DMatrix::from_row_slice(dim, dim, &flat);
        let rhs = nalgebra::DVector::from_column_slice(&rhs);
        mat.lu()
            .solve(&rhs)
            .map(|solution| solution.iter().copied().collect())
            .ok_or(TriangulationError::Geometry(GeometryError::SingularMatrix))
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

        if worst_value >= -DEFAULT_EPS {
            *self.last_simplex.write().unwrap() = Some(simplex.to_vec());
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

    pub fn locate_point(&self, point: &[f64]) -> Result<Option<Simplex>, TriangulationError> {
        self.validate_point_dim(point)?;
        let Some(mut current) = self
            .last_simplex
            .read()
            .unwrap()
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
        let vertices = self.get_vertices(simplex)?;
        Ok(geometry::point_in_simplex(point, &vertices, eps)?)
    }

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

    pub fn circumscribed_circle(
        &self,
        simplex: &[usize],
        transform: &Option<Vec<Vec<f64>>>,
    ) -> Result<(Vec<f64>, f64), TriangulationError> {
        validate_transform(transform, self.dim)?;
        let vertices = self.get_vertices(simplex)?;
        let points = match transform.as_deref() {
            Some(matrix) => vertices
                .iter()
                .map(|vertex| apply_transform(vertex, matrix))
                .collect(),
            None => vertices,
        };
        Ok(geometry::circumsphere(&points)?)
    }

    pub fn point_in_circumcircle(
        &self,
        pt_index: usize,
        simplex: &[usize],
        transform: &Option<Vec<Vec<f64>>>,
    ) -> Result<bool, TriangulationError> {
        validate_transform(transform, self.dim)?;
        self.validate_vertex_index(pt_index)?;
        let (center, radius) = self.circumscribed_circle(simplex, transform)?;
        let point = match transform.as_deref() {
            Some(matrix) => apply_transform(&self.vertices[pt_index], matrix),
            None => self.vertices[pt_index].clone(),
        };
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
    ) -> Result<(FxHashSet<Simplex>, FxHashSet<Simplex>), TriangulationError> {
        validate_transform(transform, self.dim)?;
        self.validate_vertex_index(pt_index)?;
        if let Some(simplex) = &containing_simplex {
            self.validate_simplex_indices(simplex)?;
        }

        let mut queue = VecDeque::new();
        let mut queued = FxHashSet::default();
        let mut done_simplices = FxHashSet::default();
        let mut bad_triangles = FxHashSet::default();

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
            if !done_simplices.insert(simplex.clone()) {
                continue;
            }
            if !self.simplices.contains(&simplex) {
                continue;
            }

            if self.point_in_circumcircle(pt_index, &simplex, transform)? {
                self.delete_simplex(&simplex)?;
                bad_triangles.insert(simplex.clone());

                let simplex_vertices: FxHashSet<usize> = simplex.iter().copied().collect();
                let mut neighbours = FxHashSet::default();
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
        let mut multiplicities: FxHashMap<Simplex, usize> = FxHashMap::default();
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

    pub fn extend_hull(
        &mut self,
        pt_index: usize,
    ) -> Result<FxHashSet<Simplex>, TriangulationError> {
        self.validate_vertex_index(pt_index)?;
        let faces = self.faces(None, None, None)?;
        let mut multiplicities: FxHashMap<Simplex, usize> = FxHashMap::default();
        for face in &faces {
            *multiplicities.entry(face.clone()).or_insert(0) += 1;
        }
        let hull_faces: Vec<Simplex> = multiplicities
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
                self.bowyer_watson(pt_index, None, &transform)?;

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
        Ok(geometry::volume(&self.get_vertices(simplex)?)?)
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

    pub fn hull(&self) -> Result<FxHashSet<usize>, TriangulationError> {
        let faces = self.faces(None, None, None)?;
        let mut counts: FxHashMap<Simplex, usize> = FxHashMap::default();
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

        let mut hull = FxHashSet::default();
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
    fn new(py: Python<'_>, coords: &Bound<'_, PyAny>) -> PyResult<Self> {
        let parsed_coords =
            parse_points_sized(coords, "Please provide a 2-dimensional list of points")?;
        Triangulation::validate_coords(&parsed_coords).map_err(TriangulationError::into_pyerr)?;

        let core = match PyModule::import(py, "scipy.spatial") {
            Ok(spatial) => {
                let coords_array = PyArray2::from_vec2(py, &parsed_coords)?;
                match spatial.getattr("Delaunay")?.call1((coords_array,)) {
                    Ok(delaunay) => {
                        let simplices = delaunay.getattr("simplices")?;
                        let mut initial = Vec::new();
                        for simplex in simplices.try_iter()? {
                            let simplex = simplex?;
                            let mut indices = Vec::new();
                            for item in simplex.try_iter()? {
                                indices.push(item?.extract::<usize>()?);
                            }
                            indices.sort_unstable();
                            initial.push(indices);
                        }
                        Triangulation::from_simplices(parsed_coords.clone(), initial)
                    }
                    Err(_) => Triangulation::new(parsed_coords.clone()),
                }
            }
            Err(_) => Triangulation::new(parsed_coords.clone()),
        }
        .map_err(TriangulationError::into_pyerr)?;
        Ok(Self { core })
    }

    fn add_simplex(&mut self, simplex: &Bound<'_, PyAny>) -> PyResult<()> {
        self.core
            .add_simplex(canonical_simplex_from_py(
                simplex,
                self.core.vertices.len(),
            )?)
            .map_err(TriangulationError::into_pyerr)
    }

    fn delete_simplex(&mut self, simplex: &Bound<'_, PyAny>) -> PyResult<()> {
        let simplex = canonical_simplex_from_py(simplex, self.core.vertices.len())?;
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
            Some(value) => Some(canonical_simplex_from_py(value, self.core.vertices.len())?),
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
        Ok(PyList::new(py, sets)?.into())
    }

    #[getter(hull)]
    fn hull_property(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let hull = self.core.hull().map_err(TriangulationError::into_pyerr)?;
        let vertices: Vec<usize> = hull.into_iter().collect();
        Ok(PySet::new(py, &vertices)?.into())
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
        let indices = ordered_indices_from_py(indices, self.core.vertices.len())?;
        point_list_py(
            py,
            &self
                .core
                .get_vertices(&indices)
                .map_err(TriangulationError::into_pyerr)?,
        )
    }

    fn locate_point(&self, py: Python<'_>, point: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let point = parse_point(point)?;
        match self
            .core
            .locate_point(&point)
            .map_err(TriangulationError::into_pyerr)?
        {
            Some(simplex) => Ok(simplex_tuple(py, &simplex).into()),
            None => Ok(PyTuple::empty(py).into()),
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
        let simplex_signed = parse_signed_indices(simplex)?;
        if simplex_signed.len() == self.core.dim + 1 {
            self.core
                .validate_point_dim(&point)
                .map_err(TriangulationError::into_pyerr)?;
            let simplex = normalize_indices(&simplex_signed, self.core.vertices.len())
                .map_err(TriangulationError::into_pyerr)?;
            let vertices = self
                .core
                .get_vertices(&simplex)
                .map_err(TriangulationError::into_pyerr)?;
            let alpha = match barycentric_alpha(&vertices, &point) {
                Ok(alpha) => alpha,
                Err(TriangulationError::Geometry(GeometryError::SingularMatrix)) => {
                    return Err(numpy_linalg_error(py, "Singular matrix"));
                }
                Err(other) => return Err(other.into_pyerr()),
            };
            let sum_alpha = alpha.iter().sum::<f64>();
            if alpha.iter().any(|value| *value < -eps) || sum_alpha > 1.0 + eps {
                return signed_index_list_py(py, &[]);
            }

            let mut reduced = Vec::new();
            for (idx, value) in alpha.iter().enumerate() {
                if *value > eps {
                    reduced.push(simplex_signed[idx + 1]);
                }
            }
            if sum_alpha < 1.0 - eps {
                reduced.insert(0, simplex_signed[0]);
            }
            return signed_index_list_py(py, &reduced);
        }

        let simplex = normalize_indices(&simplex_signed, self.core.vertices.len())
            .map_err(TriangulationError::into_pyerr)?;
        let reduced = match self.core.get_reduced_simplex(&point, &simplex, eps) {
            Ok(reduced) => reduced,
            Err(TriangulationError::Geometry(GeometryError::SingularMatrix)) => {
                return Err(numpy_linalg_error(py, "Singular matrix"));
            }
            Err(other) => return Err(other.into_pyerr()),
        };
        index_list_py(py, &reduced)
    }

    #[pyo3(signature = (simplex, transform=None))]
    fn circumscribed_circle(
        &self,
        py: Python<'_>,
        simplex: &Bound<'_, PyAny>,
        transform: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<(Py<PyTuple>, f64)> {
        let simplex = ordered_indices_from_py(simplex, self.core.vertices.len())?;
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
        pt_index: isize,
        simplex: &Bound<'_, PyAny>,
        transform: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<bool> {
        let pt_index = normalize_index(pt_index, self.core.vertices.len())
            .map_err(TriangulationError::into_pyerr)?;
        let simplex = ordered_indices_from_py(simplex, self.core.vertices.len())?;
        let transform = parse_optional_transform(transform)?;
        self.core
            .point_in_circumcircle(pt_index, &simplex, &transform)
            .map_err(TriangulationError::into_pyerr)
    }

    #[pyo3(name = "point_in_cicumcircle", signature = (pt_index, simplex, transform=None))]
    fn point_in_cicumcircle(
        &self,
        pt_index: isize,
        simplex: &Bound<'_, PyAny>,
        transform: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<bool> {
        self.point_in_circumcircle(pt_index, simplex, transform)
    }

    fn volume(&self, simplex: &Bound<'_, PyAny>) -> PyResult<f64> {
        let simplex = ordered_indices_from_py(simplex, self.core.vertices.len())?;
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
            Some(value) => Some(simplex_set_from_py(value, self.core.vertices.len())?),
        };
        let vertices = match vertices {
            None => None,
            Some(value) if value.is_none() => None,
            Some(value) => Some(vertex_index_set_from_py(value, self.core.vertices.len())?),
        };
        let items = self
            .core
            .faces(dim, simplices.as_ref(), vertices.as_ref())
            .map_err(TriangulationError::into_pyerr)?;
        Ok(PyFacesIter { items, index: 0 })
    }

    fn containing(&self, py: Python<'_>, face: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let face = ordered_indices_from_py(face, self.core.vertices.len())?;
        simplex_set_py(
            py,
            &self
                .core
                .containing(&face)
                .map_err(TriangulationError::into_pyerr)?,
        )
    }

    fn reference_invariant(&self) -> bool {
        self.core.reference_invariant()
    }

    #[getter(default_transform)]
    fn default_transform_property<'py>(&self, py: Python<'py>) -> PyResult<Py<PyArray2<f64>>> {
        let identity = identity_transform(self.core.dim);
        Ok(PyArray2::from_vec2(py, &identity)?.into())
    }

    #[pyo3(signature = (point, simplex, eps=DEFAULT_EPS))]
    fn point_in_simplex(
        &self,
        py: Python<'_>,
        point: &Bound<'_, PyAny>,
        simplex: &Bound<'_, PyAny>,
        eps: f64,
    ) -> PyResult<bool> {
        let point = parse_point(point)?;
        let simplex = ordered_indices_from_py(simplex, self.core.vertices.len())?;
        match self.core.point_in_simplex(&point, &simplex, eps) {
            Ok(result) => Ok(result),
            Err(TriangulationError::Geometry(GeometryError::DegenerateSimplex)) => {
                Err(PyZeroDivisionError::new_err("division by zero"))
            }
            Err(TriangulationError::Geometry(GeometryError::SingularMatrix)) => {
                Err(numpy_linalg_error(py, "Singular matrix"))
            }
            Err(other) => Err(other.into_pyerr()),
        }
    }

    #[pyo3(signature = (pt_index, containing_simplex=None, transform=None))]
    fn bowyer_watson(
        &mut self,
        py: Python<'_>,
        pt_index: isize,
        containing_simplex: Option<&Bound<'_, PyAny>>,
        transform: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let pt_index = normalize_index(pt_index, self.core.vertices.len())
            .map_err(TriangulationError::into_pyerr)?;
        let containing_simplex = match containing_simplex {
            None => None,
            Some(value) if value.is_none() => None,
            Some(value) => Some(canonical_simplex_from_py(value, self.core.vertices.len())?),
        };
        let transform = parse_optional_transform(transform)?;
        let (deleted, added) = self
            .core
            .bowyer_watson(pt_index, containing_simplex, &transform)
            .map_err(TriangulationError::into_pyerr)?;
        Ok((simplex_set_py(py, &deleted)?, simplex_set_py(py, &added)?))
    }

    fn vertex_invariant(&self, _vertex: usize) -> PyResult<bool> {
        Err(PyNotImplementedError::new_err("vertex_invariant"))
    }

    fn convex_invariant(&self, _vertex: usize) -> PyResult<bool> {
        Err(PyNotImplementedError::new_err("convex_invariant"))
    }
}
