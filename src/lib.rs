// adaptive-triangulation: Fast N-dimensional Delaunay triangulation
// Python bindings via PyO3

mod geometry;
mod triangulation;

use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

/// Fast N-dimensional Delaunay triangulation module.
#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<triangulation::PyTriangulation>()?;

    // Standalone geometry functions
    m.add_function(wrap_pyfunction!(py_circumsphere, m)?)?;
    m.add_function(wrap_pyfunction!(py_fast_2d_circumcircle, m)?)?;
    m.add_function(wrap_pyfunction!(py_fast_3d_circumsphere, m)?)?;
    m.add_function(wrap_pyfunction!(py_point_in_simplex, m)?)?;
    m.add_function(wrap_pyfunction!(py_fast_2d_point_in_simplex, m)?)?;
    m.add_function(wrap_pyfunction!(py_volume, m)?)?;
    m.add_function(wrap_pyfunction!(py_simplex_volume_in_embedding, m)?)?;
    m.add_function(wrap_pyfunction!(py_orientation, m)?)?;

    Ok(())
}

// TODO: Implement all PyO3 wrapper functions
// See IMPLEMENTATION.md for full specification

#[pyfunction]
#[pyo3(name = "circumsphere")]
fn py_circumsphere(py: Python<'_>, points: PyReadonlyArray2<f64>) -> PyResult<(Py<PyArray1<f64>>, f64)> {
    todo!("Implement circumsphere")
}

#[pyfunction]
#[pyo3(name = "fast_2d_circumcircle")]
fn py_fast_2d_circumcircle(py: Python<'_>, points: PyReadonlyArray2<f64>) -> PyResult<(Py<PyArray1<f64>>, f64)> {
    todo!("Implement fast_2d_circumcircle")
}

#[pyfunction]
#[pyo3(name = "fast_3d_circumsphere")]
fn py_fast_3d_circumsphere(py: Python<'_>, points: PyReadonlyArray2<f64>) -> PyResult<(Py<PyArray1<f64>>, f64)> {
    todo!("Implement fast_3d_circumsphere")
}

#[pyfunction]
#[pyo3(name = "point_in_simplex")]
fn py_point_in_simplex(
    point: PyReadonlyArray1<f64>,
    simplex: PyReadonlyArray2<f64>,
    eps: Option<f64>,
) -> PyResult<bool> {
    todo!("Implement point_in_simplex")
}

#[pyfunction]
#[pyo3(name = "fast_2d_point_in_simplex")]
fn py_fast_2d_point_in_simplex(
    point: PyReadonlyArray1<f64>,
    simplex: PyReadonlyArray2<f64>,
    eps: Option<f64>,
) -> PyResult<bool> {
    todo!("Implement fast_2d_point_in_simplex")
}

#[pyfunction]
#[pyo3(name = "volume")]
fn py_volume(simplex: PyReadonlyArray2<f64>) -> PyResult<f64> {
    todo!("Implement volume")
}

#[pyfunction]
#[pyo3(name = "simplex_volume_in_embedding")]
fn py_simplex_volume_in_embedding(vertices: PyReadonlyArray2<f64>) -> PyResult<f64> {
    todo!("Implement simplex_volume_in_embedding")
}

#[pyfunction]
#[pyo3(name = "orientation")]
fn py_orientation(face: PyReadonlyArray2<f64>, origin: PyReadonlyArray1<f64>) -> PyResult<i32> {
    todo!("Implement orientation")
}