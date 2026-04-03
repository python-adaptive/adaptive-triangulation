mod geometry;
mod triangulation;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyTuple};

use crate::geometry as geom;
use crate::triangulation::{parse_point, parse_points, point_tuple, PyFacesIter, PyTriangulation};

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTriangulation>()?;
    m.add_class::<PyFacesIter>()?;

    m.add_function(wrap_pyfunction!(py_circumsphere, m)?)?;
    m.add_function(wrap_pyfunction!(py_fast_2d_circumcircle, m)?)?;
    m.add_function(wrap_pyfunction!(py_fast_3d_circumsphere, m)?)?;
    m.add_function(wrap_pyfunction!(py_fast_3d_circumcircle, m)?)?;
    m.add_function(wrap_pyfunction!(py_point_in_simplex, m)?)?;
    m.add_function(wrap_pyfunction!(py_fast_2d_point_in_simplex, m)?)?;
    m.add_function(wrap_pyfunction!(py_volume, m)?)?;
    m.add_function(wrap_pyfunction!(py_simplex_volume_in_embedding, m)?)?;
    m.add_function(wrap_pyfunction!(py_orientation, m)?)?;

    Ok(())
}

#[pyfunction]
#[pyo3(name = "circumsphere")]
fn py_circumsphere(
    py: Python<'_>,
    points: &Bound<'_, PyAny>,
) -> PyResult<(Py<PyTuple>, f64)> {
    let points = parse_points(points, "Please provide a 2-dimensional list of points")?;
    let (center, radius) = geom::circumsphere(&points).map_err(|err| PyValueError::new_err(err.to_string()))?;
    Ok((point_tuple(py, &center), radius))
}

#[pyfunction]
#[pyo3(name = "fast_2d_circumcircle")]
fn py_fast_2d_circumcircle(
    py: Python<'_>,
    points: &Bound<'_, PyAny>,
) -> PyResult<(Py<PyTuple>, f64)> {
    let points = parse_points(points, "Please provide a 2-dimensional list of points")?;
    if points.len() != 3 || points.iter().any(|point| point.len() != 2) {
        return Err(PyValueError::new_err(
            "fast_2d_circumcircle expects three 2D points",
        ));
    }
    let points = [
        [points[0][0], points[0][1]],
        [points[1][0], points[1][1]],
        [points[2][0], points[2][1]],
    ];
    let (center, radius) = geom::fast_2d_circumcircle(&points);
    Ok((point_tuple(py, &center), radius))
}

#[pyfunction]
#[pyo3(name = "fast_3d_circumsphere")]
fn py_fast_3d_circumsphere(
    py: Python<'_>,
    points: &Bound<'_, PyAny>,
) -> PyResult<(Py<PyTuple>, f64)> {
    let points = parse_points(points, "Please provide a 2-dimensional list of points")?;
    if points.len() != 4 || points.iter().any(|point| point.len() != 3) {
        return Err(PyValueError::new_err(
            "fast_3d_circumsphere expects four 3D points",
        ));
    }
    let points = [
        [points[0][0], points[0][1], points[0][2]],
        [points[1][0], points[1][1], points[1][2]],
        [points[2][0], points[2][1], points[2][2]],
        [points[3][0], points[3][1], points[3][2]],
    ];
    let (center, radius) = geom::fast_3d_circumsphere(&points);
    Ok((point_tuple(py, &center), radius))
}

#[pyfunction]
#[pyo3(name = "fast_3d_circumcircle")]
fn py_fast_3d_circumcircle(
    py: Python<'_>,
    points: &Bound<'_, PyAny>,
) -> PyResult<(Py<PyTuple>, f64)> {
    py_fast_3d_circumsphere(py, points)
}

#[pyfunction]
#[pyo3(name = "point_in_simplex", signature = (point, simplex, eps=None))]
fn py_point_in_simplex(
    point: &Bound<'_, PyAny>,
    simplex: &Bound<'_, PyAny>,
    eps: Option<f64>,
) -> PyResult<bool> {
    let point = parse_point(point)?;
    let simplex = parse_points(simplex, "Please provide a 2-dimensional list of points")?;
    Ok(geom::point_in_simplex(&point, &simplex, eps.unwrap_or(1e-8)))
}

#[pyfunction]
#[pyo3(name = "fast_2d_point_in_simplex", signature = (point, simplex, eps=None))]
fn py_fast_2d_point_in_simplex(
    point: &Bound<'_, PyAny>,
    simplex: &Bound<'_, PyAny>,
    eps: Option<f64>,
) -> PyResult<bool> {
    let point = parse_point(point)?;
    let simplex = parse_points(simplex, "Please provide a 2-dimensional list of points")?;
    if point.len() != 2 || simplex.len() != 3 || simplex.iter().any(|vertex| vertex.len() != 2) {
        return Err(PyValueError::new_err(
            "fast_2d_point_in_simplex expects one 2D point and one 2D simplex",
        ));
    }
    Ok(geom::fast_2d_point_in_simplex(
        &[point[0], point[1]],
        &[
            [simplex[0][0], simplex[0][1]],
            [simplex[1][0], simplex[1][1]],
            [simplex[2][0], simplex[2][1]],
        ],
        eps.unwrap_or(1e-8),
    ))
}

#[pyfunction]
#[pyo3(name = "volume")]
fn py_volume(simplex: &Bound<'_, PyAny>) -> PyResult<f64> {
    let simplex = parse_points(simplex, "Please provide a 2-dimensional list of points")?;
    geom::volume(&simplex).map_err(|err| PyValueError::new_err(err.to_string()))
}

#[pyfunction]
#[pyo3(name = "simplex_volume_in_embedding")]
fn py_simplex_volume_in_embedding(vertices: &Bound<'_, PyAny>) -> PyResult<f64> {
    let vertices = parse_points(vertices, "Please provide a 2-dimensional list of points")?;
    geom::simplex_volume_in_embedding(&vertices)
        .map_err(|err| PyValueError::new_err(err.to_string()))
}

#[pyfunction]
#[pyo3(name = "orientation")]
fn py_orientation(face: &Bound<'_, PyAny>, origin: &Bound<'_, PyAny>) -> PyResult<i32> {
    let face = parse_points(face, "Please provide a 2-dimensional list of points")?;
    let origin = parse_point(origin)?;
    geom::orientation(&face, &origin).map_err(|err| PyValueError::new_err(err.to_string()))
}
