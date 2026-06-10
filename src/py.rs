//! PyO3 surface for the triangulation: argument parsing, result conversion,
//! and the [`PyTriangulation`] class plus its lazy proxy/iterator views.
//!
//! Nothing in this module implements triangulation logic; it forwards to the
//! pure-Rust core in [`crate::triangulation`] and converts errors to the
//! Python exception types raised by `adaptive.learner.triangulation` in the
//! same situations.

use numpy::PyArray2;
use pyo3::exceptions::{
    PyAssertionError, PyIndexError, PyNotImplementedError, PyRuntimeError, PyTypeError,
    PyValueError, PyZeroDivisionError,
};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList, PyModule, PySet, PyTuple};
use rustc_hash::FxHashSet;

use crate::geometry::GeometryError;
use crate::tolerances::BARYCENTRIC_EPS;
use crate::triangulation::{
    index_out_of_range, reduced_simplex_positions, Simplex, Triangulation, TriangulationError,
};

impl TriangulationError {
    pub(crate) fn into_pyerr(self) -> PyErr {
        match self {
            Self::Value(message) => PyValueError::new_err(message),
            Self::Index(message) => PyIndexError::new_err(message),
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
/// Raise `numpy.linalg.LinAlgError` (falling back to `ValueError` when numpy
/// is unavailable), matching where the reference implementation lets numpy
/// solves fail.
pub(crate) fn numpy_linalg_error(py: Python<'_>, message: &str) -> PyErr {
    PyModule::import(py, "numpy.linalg")
        .and_then(|module| {
            let error_type = module.getattr("LinAlgError")?;
            let args = PyTuple::new(py, [message]).unwrap();
            let value = error_type.call1(args)?;
            Ok(PyErr::from_value(value))
        })
        .unwrap_or_else(|_| PyValueError::new_err(message.to_string()))
}
fn scipy_delaunay_simplices(
    py: Python<'_>,
    coords: &[Vec<f64>],
    dim: usize,
) -> PyResult<Option<Vec<Simplex>>> {
    if dim == 1 {
        return Ok(None);
    }

    let Ok(spatial) = PyModule::import(py, "scipy.spatial") else {
        return Ok(None);
    };
    let coords_array = PyArray2::from_vec2(py, coords)?;
    let Ok(delaunay) = spatial.getattr("Delaunay")?.call1((coords_array,)) else {
        return Ok(None);
    };

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
    Ok(Some(initial))
}
/// Python-facing `Triangulation`, a thin argument-parsing wrapper around the
/// pure-Rust [`Triangulation`] core. Drop-in compatible with
/// `adaptive.learner.triangulation.Triangulation`.
#[pyclass(name = "Triangulation")]
pub struct PyTriangulation {
    pub core: Triangulation,
}

/// Lazy, set-like view of the simplices (all of them, or those of one
/// vertex) that reads triangulation state on access instead of copying it.
#[pyclass(name = "SimplicesProxy")]
pub struct PySimplicesProxy {
    triangulation: Py<PyTriangulation>,
    vertex: Option<usize>,
}

impl PySimplicesProxy {
    fn all(triangulation: Py<PyTriangulation>) -> Self {
        Self {
            triangulation,
            vertex: None,
        }
    }

    fn for_vertex(triangulation: Py<PyTriangulation>, vertex: usize) -> Self {
        Self {
            triangulation,
            vertex: Some(vertex),
        }
    }
}

/// Lazy, sequence-like view of the vertex coordinates (supports `__array__`
/// for zero-surprise numpy conversion).
#[pyclass(name = "VerticesProxy")]
pub struct PyVerticesProxy {
    triangulation: Py<PyTriangulation>,
}

impl PyVerticesProxy {
    fn new(triangulation: Py<PyTriangulation>) -> Self {
        Self { triangulation }
    }
}

/// Lazy, sequence-like view mapping each vertex index to its simplex set.
#[pyclass(name = "VertexToSimplicesProxy")]
pub struct PyVertexToSimplicesProxy {
    triangulation: Py<PyTriangulation>,
}

impl PyVertexToSimplicesProxy {
    fn new(triangulation: Py<PyTriangulation>) -> Self {
        Self { triangulation }
    }
}

/// Iterator over a snapshot of faces/simplices as Python tuples.
#[pyclass]
pub struct PyFacesIter {
    items: Vec<Simplex>,
    index: usize,
}

#[pyclass]
pub struct PyVerticesIter {
    triangulation: Py<PyTriangulation>,
    index: usize,
}

#[pyclass]
pub struct PyVertexToSimplicesIter {
    triangulation: Py<PyTriangulation>,
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
impl PySimplicesProxy {
    fn __contains__(&self, py: Python<'_>, simplex: &Bound<'_, PyTuple>) -> bool {
        let triangulation = self.triangulation.bind(py).borrow();
        let Ok(simplex) =
            canonical_simplex_from_py(simplex.as_any(), triangulation.core.vertices.len())
        else {
            return false;
        };

        match self.vertex {
            Some(vertex) => triangulation.core.vertex_to_simplices[vertex].contains(&simplex),
            None => triangulation.core.simplices.contains(&simplex),
        }
    }

    fn __iter__(&self, py: Python<'_>) -> PyFacesIter {
        let triangulation = self.triangulation.bind(py).borrow();
        let items = match self.vertex {
            Some(vertex) => triangulation.core.vertex_to_simplices[vertex]
                .iter()
                .cloned()
                .collect(),
            None => triangulation.core.simplices.iter().cloned().collect(),
        };
        PyFacesIter { items, index: 0 }
    }

    fn __len__(&self, py: Python<'_>) -> usize {
        let triangulation = self.triangulation.bind(py).borrow();
        match self.vertex {
            Some(vertex) => triangulation.core.vertex_to_simplices[vertex].len(),
            None => triangulation.core.simplices.len(),
        }
    }
}

#[pymethods]
impl PyVerticesIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> Option<Py<PyTuple>> {
        let triangulation = self.triangulation.bind(py).borrow();
        let vertex = triangulation.core.vertices.get(self.index)?;
        self.index += 1;
        Some(point_tuple(py, vertex))
    }
}

#[pymethods]
impl PyVertexToSimplicesIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> Option<Py<PyAny>> {
        let triangulation = self.triangulation.bind(py).borrow();
        if self.index >= triangulation.core.vertices.len() {
            return None;
        }
        let simplices =
            simplex_set_py(py, &triangulation.core.vertex_to_simplices[self.index]).ok()?;
        self.index += 1;
        Some(simplices)
    }
}

#[pymethods]
impl PyVerticesProxy {
    fn __getitem__(&self, py: Python<'_>, index: isize) -> PyResult<Py<PyTuple>> {
        let triangulation = self.triangulation.bind(py).borrow();
        let index = normalize_index(index, triangulation.core.vertices.len())
            .map_err(TriangulationError::into_pyerr)?;
        Ok(point_tuple(py, &triangulation.core.vertices[index]))
    }

    fn __len__(&self, py: Python<'_>) -> usize {
        self.triangulation.bind(py).borrow().core.vertices.len()
    }

    fn __iter__(&self, py: Python<'_>) -> PyVerticesIter {
        PyVerticesIter {
            triangulation: self.triangulation.clone_ref(py),
            index: 0,
        }
    }

    #[pyo3(signature = (dtype=None, copy=None))]
    fn __array__(
        &self,
        py: Python<'_>,
        dtype: Option<&Bound<'_, PyAny>>,
        copy: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        let triangulation = self.triangulation.bind(py).borrow();
        let vertices = point_list_py(py, &triangulation.core.vertices)?;
        let numpy = PyModule::import(py, "numpy")?;
        let kwargs = PyDict::new(py);
        if let Some(dtype) = dtype {
            kwargs.set_item("dtype", dtype)?;
        }
        let array = if copy == Some(false) {
            numpy.call_method("asarray", (vertices,), Some(&kwargs))?
        } else {
            if let Some(copy) = copy {
                kwargs.set_item("copy", copy)?;
            }
            numpy.call_method("array", (vertices,), Some(&kwargs))?
        };
        Ok(array.into())
    }
}

#[pymethods]
impl PyVertexToSimplicesProxy {
    fn __getitem__(&self, py: Python<'_>, index: isize) -> PyResult<Py<PyAny>> {
        let triangulation = self.triangulation.bind(py).borrow();
        let index = normalize_index(index, triangulation.core.vertices.len())
            .map_err(TriangulationError::into_pyerr)?;
        simplex_set_py(py, &triangulation.core.vertex_to_simplices[index])
    }

    fn __len__(&self, py: Python<'_>) -> usize {
        self.triangulation.bind(py).borrow().core.vertices.len()
    }

    fn __iter__(&self, py: Python<'_>) -> PyVertexToSimplicesIter {
        PyVertexToSimplicesIter {
            triangulation: self.triangulation.clone_ref(py),
            index: 0,
        }
    }
}

#[pymethods]
impl PyTriangulation {
    #[new]
    fn new(py: Python<'_>, coords: &Bound<'_, PyAny>) -> PyResult<Self> {
        let parsed_coords =
            parse_points_sized(coords, "Please provide a 2-dimensional list of points")?;
        let dim = Triangulation::validate_coords(&parsed_coords)
            .map_err(TriangulationError::into_pyerr)?;

        let core = match scipy_delaunay_simplices(py, &parsed_coords, dim)? {
            Some(initial) => Triangulation::from_simplices(parsed_coords, initial),
            None => Triangulation::new(parsed_coords),
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
    fn vertices_property(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<Py<PyVerticesProxy>> {
        Py::new(py, PyVerticesProxy::new(slf.into()))
    }

    #[getter(simplices)]
    fn simplices_property(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<Py<PySimplicesProxy>> {
        Py::new(py, PySimplicesProxy::all(slf.into()))
    }

    #[getter(vertex_to_simplices)]
    fn vertex_to_simplices_property(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
    ) -> PyResult<Py<PyVertexToSimplicesProxy>> {
        Py::new(py, PyVertexToSimplicesProxy::new(slf.into()))
    }

    fn has_simplex(&self, simplex: &Bound<'_, PyTuple>) -> bool {
        let Ok(simplex) = canonical_simplex_from_py(simplex.as_any(), self.core.vertices.len())
        else {
            return false;
        };
        self.core.simplices.contains(&simplex)
    }

    fn vertex_to_simplices_for(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
        vertex: isize,
    ) -> PyResult<Py<PySimplicesProxy>> {
        let vertex = normalize_index(vertex, slf.core.vertices.len())
            .map_err(TriangulationError::into_pyerr)?;
        Py::new(py, PySimplicesProxy::for_vertex(slf.into(), vertex))
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

    #[pyo3(signature = (point, simplex, eps=BARYCENTRIC_EPS))]
    fn get_reduced_simplex(
        &self,
        py: Python<'_>,
        point: &Bound<'_, PyAny>,
        simplex: &Bound<'_, PyAny>,
        eps: f64,
    ) -> PyResult<Py<PyAny>> {
        let point = parse_point(point)?;
        let simplex_signed = parse_signed_indices(simplex)?;
        // For a full simplex the reference echoes back the caller's indices
        // (including negative ones), so reduce positions rather than values.
        if simplex_signed.len() == self.core.dim + 1 {
            self.core
                .validate_point_dim(&point)
                .map_err(TriangulationError::into_pyerr)?;
            let simplex = normalize_indices(&simplex_signed, self.core.vertices.len())
                .map_err(TriangulationError::into_pyerr)?;
            let alpha = match self.core.barycentric_alpha_for_simplex(&simplex, &point) {
                Ok(alpha) => alpha,
                Err(TriangulationError::Geometry(GeometryError::SingularMatrix)) => {
                    return Err(numpy_linalg_error(py, "Singular matrix"));
                }
                Err(other) => return Err(other.into_pyerr()),
            };
            let reduced: Vec<isize> = reduced_simplex_positions(&alpha, eps)
                .into_iter()
                .map(|position| simplex_signed[position])
                .collect();
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
    fn default_transform_property(&self, py: Python<'_>) -> PyResult<Py<PyArray2<f64>>> {
        let identity = identity_transform(self.core.dim);
        Ok(PyArray2::from_vec2(py, &identity)?.into())
    }

    #[pyo3(signature = (point, simplex, eps=BARYCENTRIC_EPS))]
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
