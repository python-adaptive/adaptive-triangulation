//! Pure geometric primitives: norms, determinants, barycentric coordinates,
//! circumspheres, orientations, and simplex volumes.
//!
//! Nothing in this module touches Python or triangulation state, so every
//! function here is testable in isolation. Error messages mirror the Python
//! reference implementation in `adaptive.learner.triangulation`.

use nalgebra::{DMatrix, DVector};
use thiserror::Error;

use crate::tolerances::ORIENTATION_LOG_DET_CUTOFF;

/// Errors produced by the geometric primitives.
#[derive(Debug, Error)]
pub enum GeometryError {
    /// Input has the wrong shape (point/vertex counts or coordinate lengths).
    #[error("{0}")]
    InvalidDimensions(String),
    /// The vertices span a lower-dimensional space than a simplex requires.
    #[error("Provided vertices do not form a simplex")]
    DegenerateSimplex,
    /// A linear solve hit a (numerically) singular matrix.
    #[error("Singular matrix")]
    SingularMatrix,
}

/// Euclidean norm, with unrolled 2D and 3D fast paths.
#[inline]
pub fn fast_norm(v: &[f64]) -> f64 {
    match v.len() {
        2 => (v[0] * v[0] + v[1] * v[1]).sqrt(),
        3 => (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt(),
        _ => v.iter().map(|x| x * x).sum::<f64>().sqrt(),
    }
}

#[inline]
fn squared_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

#[inline]
fn factorial(n: usize) -> f64 {
    (1..=n).fold(1.0, |acc, value| acc * value as f64)
}

fn validate_points(points: &[Vec<f64>]) -> Result<usize, GeometryError> {
    if points.is_empty() {
        return Err(GeometryError::InvalidDimensions(
            "Expected at least one point".to_string(),
        ));
    }
    let dim = points[0].len();
    if dim == 0 {
        return Err(GeometryError::InvalidDimensions(
            "Points must have non-zero dimension".to_string(),
        ));
    }
    if points.iter().any(|pt| pt.len() != dim) {
        return Err(GeometryError::InvalidDimensions(
            "Coordinates dimension mismatch".to_string(),
        ));
    }
    Ok(dim)
}

#[inline]
fn squared_norm(point: &[f64]) -> f64 {
    point.iter().map(|coord| coord * coord).sum()
}

fn determinant(matrix: &[Vec<f64>]) -> Result<f64, GeometryError> {
    let n = matrix.len();
    if n == 0 {
        return Ok(1.0);
    }
    if matrix.iter().any(|row| row.len() != n) {
        return Err(GeometryError::InvalidDimensions(
            "Matrix must be square".to_string(),
        ));
    }

    let det = match n {
        1 => matrix[0][0],
        2 => matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0],
        3 => {
            let a = matrix[0][0];
            let b = matrix[0][1];
            let c = matrix[0][2];
            let d = matrix[1][0];
            let e = matrix[1][1];
            let f = matrix[1][2];
            let g = matrix[2][0];
            let h = matrix[2][1];
            let i = matrix[2][2];
            a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
        }
        _ => {
            let flat: Vec<f64> = matrix.iter().flat_map(|row| row.iter().copied()).collect();
            DMatrix::from_row_slice(n, n, &flat).determinant()
        }
    };

    Ok(det)
}

fn solve_square(matrix: &[Vec<f64>], rhs: &[f64]) -> Result<Vec<f64>, GeometryError> {
    let n = matrix.len();
    if rhs.len() != n || matrix.iter().any(|row| row.len() != n) {
        return Err(GeometryError::InvalidDimensions(
            "Matrix and rhs dimensions do not match".to_string(),
        ));
    }

    let flat: Vec<f64> = matrix.iter().flat_map(|row| row.iter().copied()).collect();
    let mat = DMatrix::from_row_slice(n, n, &flat);
    let vec = DVector::from_column_slice(rhs);
    mat.lu()
        .solve(&vec)
        .map(|solution| solution.iter().copied().collect())
        .ok_or(GeometryError::SingularMatrix)
}

/// Rank of the matrix whose rows are `vectors`, counting singular values
/// above `tol`. A negative `tol` selects numpy's default tolerance
/// (`eps * max(rows, cols) * largest_singular_value`).
pub fn matrix_rank(vectors: &[Vec<f64>], tol: f64) -> Result<usize, GeometryError> {
    if vectors.is_empty() {
        return Ok(0);
    }
    let cols = vectors[0].len();
    if vectors.iter().any(|row| row.len() != cols) {
        return Err(GeometryError::InvalidDimensions(
            "Coordinates dimension mismatch".to_string(),
        ));
    }
    let rows = vectors.len();
    let flat: Vec<f64> = vectors.iter().flat_map(|row| row.iter().copied()).collect();
    let svd = DMatrix::from_row_slice(rows, cols, &flat).svd(false, false);
    let max_singular = svd.singular_values.iter().copied().fold(0.0_f64, f64::max);
    let tol = if tol.is_sign_positive() {
        tol
    } else {
        f64::EPSILON * rows.max(cols) as f64 * max_singular
    };
    Ok(svd
        .singular_values
        .iter()
        .filter(|value| **value > tol)
        .count())
}

/// [`matrix_rank`] with numpy's default tolerance.
pub fn numpy_matrix_rank(vectors: &[Vec<f64>]) -> Result<usize, GeometryError> {
    matrix_rank(vectors, -1.0)
}

/// Barycentric coordinates of `point` with respect to vertices `1..=dim` of
/// `simplex`; the coordinate of vertex 0 is `1 - sum(result)`.
///
/// Accepts any slice of coordinate slices so callers can pass either owned
/// vertices or references into triangulation storage without cloning.
pub fn barycentric_coordinates<P: AsRef<[f64]>>(
    simplex: &[P],
    point: &[f64],
) -> Result<Vec<f64>, GeometryError> {
    let dim = point.len();
    let x0 = simplex[0].as_ref();
    let mut matrix = vec![vec![0.0; dim]; dim];
    let mut rhs = vec![0.0; dim];

    for row in 0..dim {
        rhs[row] = point[row] - x0[row];
        for col in 0..dim {
            matrix[row][col] = simplex[col + 1].as_ref()[row] - x0[row];
        }
    }

    solve_square(&matrix, &rhs)
}

/// 2D point-in-triangle test via explicit barycentric formulas, with `eps`
/// slack on the barycentric coordinates (see
/// [`crate::tolerances::BARYCENTRIC_EPS`]).
pub fn fast_2d_point_in_simplex(
    point: &[f64; 2],
    simplex: &[[f64; 2]; 3],
    eps: f64,
) -> Result<bool, GeometryError> {
    let [[p0x, p0y], [p1x, p1y], [p2x, p2y]] = *simplex;
    let [px, py] = *point;

    let area: f64 = 0.5 * (-p1y * p2x + p0y * (p2x - p1x) + p1x * p2y + p0x * (p1y - p2y));
    if area == 0.0 {
        return Err(GeometryError::DegenerateSimplex);
    }

    let s = 1.0 / (2.0 * area) * (p0y * p2x + (p2y - p0y) * px - p0x * p2y + (p0x - p2x) * py);
    if s < -eps || s > 1.0 + eps {
        return Ok(false);
    }

    let t = 1.0 / (2.0 * area) * (p0x * p1y + (p0y - p1y) * px - p0y * p1x + (p1x - p0x) * py);
    Ok(t >= -eps && s + t <= 1.0 + eps)
}

/// N-dimensional point-in-simplex test with `eps` slack on the barycentric
/// coordinates. Dispatches to [`fast_2d_point_in_simplex`] in 2D.
pub fn point_in_simplex(
    point: &[f64],
    simplex: &[Vec<f64>],
    eps: f64,
) -> Result<bool, GeometryError> {
    if simplex.is_empty() || simplex.len() != point.len() + 1 {
        return Err(GeometryError::InvalidDimensions(
            "Simplex dimension mismatch".to_string(),
        ));
    }
    validate_points(simplex)?;
    if simplex.iter().any(|vertex| vertex.len() != point.len()) {
        return Err(GeometryError::InvalidDimensions(
            "Simplex dimension mismatch".to_string(),
        ));
    }
    if point.len() == 2 && simplex.len() == 3 {
        let point = [point[0], point[1]];
        let simplex = [
            [simplex[0][0], simplex[0][1]],
            [simplex[1][0], simplex[1][1]],
            [simplex[2][0], simplex[2][1]],
        ];
        return fast_2d_point_in_simplex(&point, &simplex, eps);
    }

    let alpha = barycentric_coordinates(simplex, point)?;
    Ok(alpha.iter().all(|value| *value > -eps) && alpha.iter().sum::<f64>() < 1.0 + eps)
}

/// Circumcircle (center, radius) of a triangle, computed in coordinates
/// relative to the first vertex for numerical stability.
pub fn fast_2d_circumcircle(points: &[[f64; 2]; 3]) -> ([f64; 2], f64) {
    let [p0, p1, p2] = *points;
    let x1 = p1[0] - p0[0];
    let y1 = p1[1] - p0[1];
    let x2 = p2[0] - p0[0];
    let y2 = p2[1] - p0[1];

    let l1 = x1 * x1 + y1 * y1;
    let l2 = x2 * x2 + y2 * y2;

    let dx = l1 * y2 - l2 * y1;
    let dy = -l1 * x2 + l2 * x1;
    let a = 2.0 * (x1 * y2 - x2 * y1);

    let x = dx / a;
    let y = dy / a;
    let radius = (x * x + y * y).sqrt();

    ([x + p0[0], y + p0[1]], radius)
}

/// Circumsphere (center, radius) of a tetrahedron, computed in coordinates
/// relative to the first vertex for numerical stability.
pub fn fast_3d_circumsphere(points: &[[f64; 3]; 4]) -> ([f64; 3], f64) {
    let [p0, p1, p2, p3] = *points;
    let x1 = p1[0] - p0[0];
    let y1 = p1[1] - p0[1];
    let z1 = p1[2] - p0[2];
    let x2 = p2[0] - p0[0];
    let y2 = p2[1] - p0[1];
    let z2 = p2[2] - p0[2];
    let x3 = p3[0] - p0[0];
    let y3 = p3[1] - p0[1];
    let z3 = p3[2] - p0[2];

    let l1 = x1 * x1 + y1 * y1 + z1 * z1;
    let l2 = x2 * x2 + y2 * y2 + z2 * z2;
    let l3 = x3 * x3 + y3 * y3 + z3 * z3;

    let dx = l1 * (y2 * z3 - z2 * y3) - l2 * (y1 * z3 - z1 * y3) + l3 * (y1 * z2 - z1 * y2);
    let dy = l1 * (x2 * z3 - z2 * x3) - l2 * (x1 * z3 - z1 * x3) + l3 * (x1 * z2 - z1 * x2);
    let dz = l1 * (x2 * y3 - y2 * x3) - l2 * (x1 * y3 - y1 * x3) + l3 * (x1 * y2 - y1 * x2);
    let aa = x1 * (y2 * z3 - z2 * y3) - x2 * (y1 * z3 - z1 * y3) + x3 * (y1 * z2 - z1 * y2);
    let a = 2.0 * aa;

    let cx = dx / a;
    let cy = -dy / a;
    let cz = dz / a;
    let radius = (cx * cx + cy * cy + cz * cz).sqrt();

    ([cx + p0[0], cy + p0[1], cz + p0[2]], radius)
}

/// Circumsphere (center, radius) of an N-dimensional simplex with `dim + 1`
/// vertices. Dispatches to the unrolled 2D/3D paths; the generic path solves
/// in coordinates relative to the first vertex (see PR #3 — the naive
/// `|x_i|^2 - |x_0|^2` form cancels catastrophically for small simplices far
/// from the origin). Returns NaNs for degenerate input, matching numpy.
pub fn circumsphere(pts: &[Vec<f64>]) -> Result<(Vec<f64>, f64), GeometryError> {
    let dim = validate_simplex(pts)?;

    if dim == 2 {
        let points = [
            [pts[0][0], pts[0][1]],
            [pts[1][0], pts[1][1]],
            [pts[2][0], pts[2][1]],
        ];
        let (center, radius) = fast_2d_circumcircle(&points);
        return Ok((center.into_iter().collect(), radius));
    }
    if dim == 3 {
        let points = [
            [pts[0][0], pts[0][1], pts[0][2]],
            [pts[1][0], pts[1][1], pts[1][2]],
            [pts[2][0], pts[2][1], pts[2][2]],
            [pts[3][0], pts[3][1], pts[3][2]],
        ];
        let (center, radius) = fast_3d_circumsphere(&points);
        return Ok((center.into_iter().collect(), radius));
    }

    // Solve in coordinates relative to the first vertex: computing
    // |x_i|^2 - |x_0|^2 directly cancels catastrophically when the simplex
    // is small compared to its distance from the origin.
    let x0 = &pts[0];
    let mut matrix = vec![vec![0.0; dim]; dim];
    let mut rhs = vec![0.0; dim];

    for row in 0..dim {
        let translated: Vec<f64> = pts[row + 1].iter().zip(x0).map(|(a, b)| a - b).collect();
        rhs[row] = squared_norm(&translated);
        for (col, value) in translated.iter().enumerate() {
            matrix[row][col] = 2.0 * value;
        }
    }

    let relative_center = match solve_square(&matrix, &rhs) {
        Ok(center) => center,
        Err(GeometryError::SingularMatrix) => {
            return Ok((vec![f64::NAN; dim], f64::NAN));
        }
        Err(err) => return Err(err),
    };

    let radius = fast_norm(&relative_center);
    let center: Vec<f64> = relative_center.iter().zip(x0).map(|(c, o)| c + o).collect();
    Ok((center, radius))
}

fn slogdet(matrix: &[Vec<f64>]) -> Result<(f64, f64), GeometryError> {
    let n = matrix.len();
    if n == 0 {
        return Ok((1.0, 0.0));
    }
    if matrix.iter().any(|row| row.len() != n) {
        return Err(GeometryError::InvalidDimensions(
            "Matrix must be square".to_string(),
        ));
    }

    let mut work = matrix.to_vec();
    let mut sign = 1.0;
    let mut log_abs_det = 0.0;

    for pivot_col in 0..n {
        let mut pivot_row = pivot_col;
        let mut pivot_abs = work[pivot_col][pivot_col].abs();
        for (row, values) in work.iter().enumerate().skip(pivot_col + 1) {
            let candidate = values[pivot_col].abs();
            if candidate > pivot_abs {
                pivot_abs = candidate;
                pivot_row = row;
            }
        }

        if pivot_abs == 0.0 {
            return Ok((0.0, f64::NEG_INFINITY));
        }

        if pivot_row != pivot_col {
            work.swap(pivot_row, pivot_col);
            sign = -sign;
        }

        let pivot = work[pivot_col][pivot_col];
        if pivot == 0.0 {
            return Ok((0.0, f64::NEG_INFINITY));
        }
        sign *= pivot.signum();
        log_abs_det += pivot.abs().ln();

        let pivot_values = work[pivot_col].clone();
        for row_values in work.iter_mut().skip(pivot_col + 1) {
            let factor = row_values[pivot_col] / pivot;
            row_values[pivot_col] = 0.0;
            for (col, value) in row_values.iter_mut().enumerate().skip(pivot_col + 1) {
                *value -= factor * pivot_values[col];
            }
        }
    }

    Ok((sign, log_abs_det))
}

/// Sign (+1, -1, or 0) of the determinant of `face - origin`, i.e. which side
/// of the hyperplane through `face` the point `origin` lies on. Returns 0
/// when the log-determinant falls below
/// [`ORIENTATION_LOG_DET_CUTOFF`].
pub fn orientation(face: &[Vec<f64>], origin: &[f64]) -> Result<i32, GeometryError> {
    let dim = validate_points(face)?;
    if face.len() != dim || origin.len() != dim {
        return Err(GeometryError::InvalidDimensions(
            "Face and origin dimensions do not match".to_string(),
        ));
    }

    let matrix: Vec<Vec<f64>> = face
        .iter()
        .map(|point| point.iter().zip(origin).map(|(x, y)| x - y).collect())
        .collect();
    let (sign, log_det) = slogdet(&matrix)?;
    if sign == 0.0 || log_det < ORIENTATION_LOG_DET_CUTOFF {
        Ok(0)
    } else if sign.is_sign_positive() {
        Ok(1)
    } else {
        Ok(-1)
    }
}

/// `robust::Coord` from the first two coordinates of a point.
fn coord_2d(point: &[f64]) -> robust::Coord<f64> {
    robust::Coord {
        x: point[0],
        y: point[1],
    }
}

/// `robust::Coord3D` from the first three coordinates of a point.
fn coord_3d(point: &[f64]) -> robust::Coord3D<f64> {
    robust::Coord3D {
        x: point[0],
        y: point[1],
        z: point[2],
    }
}

fn sign(value: f64) -> i32 {
    if value > 0.0 {
        1
    } else if value < 0.0 {
        -1
    } else {
        0
    }
}

/// Validates that `points` are a full simplex (`dim + 1` points of equal
/// dimension) and returns `dim`.
fn validate_simplex(points: &[Vec<f64>]) -> Result<usize, GeometryError> {
    let dim = validate_points(points)?;
    if points.len() != dim + 1 {
        return Err(GeometryError::InvalidDimensions(format!(
            "Expected {} points for a {}-dimensional simplex",
            dim + 1,
            dim
        )));
    }
    Ok(dim)
}

/// Like [`orientation`], but exact in 2D and 3D: the sign is computed with
/// Shewchuk's adaptive-precision predicates (via the `robust` crate), so it
/// is correct even for slivers whose floating-point determinant rounds to
/// the wrong side of zero. Higher dimensions (and 1D) fall back to
/// [`orientation`], whose answer near zero is best-effort.
///
/// Not a drop-in replacement for [`orientation`] in reference-compatible
/// code paths: the reference applies [`ORIENTATION_LOG_DET_CUTOFF`], which
/// reports tiny-but-nonzero determinants as 0 while this reports their true
/// sign. Use it where geometric truth matters more than bug compatibility
/// (e.g. cavity repair).
pub fn robust_orientation(face: &[Vec<f64>], origin: &[f64]) -> Result<i32, GeometryError> {
    let dim = validate_points(face)?;
    if face.len() != dim || origin.len() != dim {
        return Err(GeometryError::InvalidDimensions(
            "Face and origin dimensions do not match".to_string(),
        ));
    }

    // orient2d(a, b, c) = det [[a - c], [b - c]] and orient3d(a, b, c, d) =
    // det [[a - d], [b - d], [c - d]]: exactly the determinants whose sign
    // `orientation` computes.
    match dim {
        2 => Ok(sign(robust::orient2d(
            coord_2d(&face[0]),
            coord_2d(&face[1]),
            coord_2d(origin),
        ))),
        3 => Ok(sign(robust::orient3d(
            coord_3d(&face[0]),
            coord_3d(&face[1]),
            coord_3d(&face[2]),
            coord_3d(origin),
        ))),
        _ => orientation(face, origin),
    }
}

/// Whether `point` lies strictly inside the circumsphere of `simplex`,
/// decided exactly with Shewchuk's adaptive-precision incircle/insphere
/// predicates. Returns `None` for dimensions other than 2 and 3 (no exact
/// predicate available). A degenerate simplex (zero orientation) has no
/// finite circumsphere and reports `false`, matching the NaN-radius
/// convention of [`circumsphere`].
///
/// Note the deliberate difference from the reference's tolerance test
/// (`distance < radius * (1 + CIRCUMCIRCLE_RTOL)`): this is strict
/// containment with no slack. Use it where geometric truth matters more
/// than bug compatibility (e.g. cavity repair).
pub fn robust_in_circumsphere(
    simplex: &[Vec<f64>],
    point: &[f64],
) -> Result<Option<bool>, GeometryError> {
    let dim = validate_simplex(simplex)?;
    if point.len() != dim {
        return Err(GeometryError::InvalidDimensions(
            "Coordinates dimension mismatch".to_string(),
        ));
    }

    // incircle/insphere's sign follows the simplex's orientation;
    // multiplying by it makes the test orientation-independent.
    match dim {
        2 => {
            let [a, b, c] = [&simplex[0], &simplex[1], &simplex[2]].map(|p| coord_2d(p));
            let orient = robust::orient2d(a, b, c);
            if orient == 0.0 {
                return Ok(Some(false));
            }
            Ok(Some(
                robust::incircle(a, b, c, coord_2d(point)) * orient > 0.0,
            ))
        }
        3 => {
            let [a, b, c, d] =
                [&simplex[0], &simplex[1], &simplex[2], &simplex[3]].map(|p| coord_3d(p));
            let orient = robust::orient3d(a, b, c, d);
            if orient == 0.0 {
                return Ok(Some(false));
            }
            Ok(Some(
                robust::insphere(a, b, c, d, coord_3d(point)) * orient > 0.0,
            ))
        }
        _ => Ok(None),
    }
}

/// Like [`volume`], but computed with adaptive-precision determinants in 2D
/// and 3D (the value of Shewchuk's orientation predicate is the exact edge
/// determinant, evaluated to machine relative accuracy), so cancellation
/// cannot corrupt the result for sliver simplices. Higher dimensions fall
/// back to [`volume`].
///
/// Not used for reference-compatible answers (the reference computes plain
/// floating-point determinants); use it where an accurate value matters,
/// e.g. the volume-conservation check in Bowyer-Watson.
pub fn precise_volume(vertices: &[Vec<f64>]) -> Result<f64, GeometryError> {
    match validate_simplex(vertices)? {
        2 => {
            let det = robust::orient2d(
                coord_2d(&vertices[1]),
                coord_2d(&vertices[2]),
                coord_2d(&vertices[0]),
            );
            Ok(det.abs() / 2.0)
        }
        3 => {
            let det = robust::orient3d(
                coord_3d(&vertices[1]),
                coord_3d(&vertices[2]),
                coord_3d(&vertices[3]),
                coord_3d(&vertices[0]),
            );
            Ok(det.abs() / 6.0)
        }
        _ => volume(vertices),
    }
}

/// Volume of a `dim`-simplex given its `dim + 1` vertices
/// (`|det| / dim!` of the edge matrix).
pub fn volume(vertices: &[Vec<f64>]) -> Result<f64, GeometryError> {
    let dim = validate_simplex(vertices)?;

    let mut matrix = vec![vec![0.0; dim]; dim];
    let x0 = &vertices[0];
    for row in 0..dim {
        for col in 0..dim {
            matrix[row][col] = vertices[col + 1][row] - x0[row];
        }
    }
    Ok(determinant(&matrix)?.abs() / factorial(dim))
}

/// Volume of a `k`-simplex embedded in a higher-dimensional space (segment
/// length, triangle area via Heron's formula, Cayley-Menger determinant in
/// general). Unlike [`volume`], the number of vertices may be smaller than
/// `dim + 1`.
pub fn simplex_volume_in_embedding(vertices: &[Vec<f64>]) -> Result<f64, GeometryError> {
    validate_points(vertices)?;
    if vertices.len() < 2 {
        return Err(GeometryError::InvalidDimensions(
            "Expected at least two vertices".to_string(),
        ));
    }

    if vertices.len() == 2 {
        let length_sq = squared_distance(&vertices[0], &vertices[1]);
        if length_sq == 0.0 {
            return Err(GeometryError::DegenerateSimplex);
        }
        return Ok(length_sq.sqrt());
    }

    if vertices.len() == 3 {
        let a = squared_distance(&vertices[0], &vertices[1]).sqrt();
        let b = squared_distance(&vertices[1], &vertices[2]).sqrt();
        let c = squared_distance(&vertices[2], &vertices[0]).sqrt();
        let s = 0.5 * (a + b + c);
        let area_sq = s * (s - a) * (s - b) * (s - c);
        if area_sq <= 0.0 {
            return Err(GeometryError::DegenerateSimplex);
        }
        return Ok(area_sq.sqrt());
    }

    let n = vertices.len();
    let mut matrix = vec![vec![0.0; n + 1]; n + 1];
    for value in matrix[0].iter_mut().skip(1) {
        *value = 1.0;
    }
    for row in matrix.iter_mut().skip(1) {
        row[0] = 1.0;
    }
    for row in 0..n {
        for col in 0..n {
            if row != col {
                matrix[row + 1][col + 1] = squared_distance(&vertices[row], &vertices[col]);
            }
        }
    }

    let coeff = -(-2.0f64).powi((n - 1) as i32) * factorial(n - 1).powi(2);
    let vol_square = determinant(&matrix)? / coeff;
    if vol_square <= 0.0 {
        return Err(GeometryError::DegenerateSimplex);
    }
    Ok(vol_square.sqrt())
}

/// The loss adaptive's `LearnerND` assigns to a simplex by default: the
/// volume of the simplex embedded in (input + output)-dimensional space,
/// where each vertex is extended with its function value(s). `simplex` and
/// `values` are zipped pairwise, like the reference implementation.
pub fn default_loss(simplex: &[Vec<f64>], values: &[Vec<f64>]) -> Result<f64, GeometryError> {
    let embedded: Vec<Vec<f64>> = simplex
        .iter()
        .zip(values)
        .map(|(vertex, value)| vertex.iter().chain(value.iter()).copied().collect())
        .collect();
    simplex_volume_in_embedding(&embedded)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simplex_volume_in_embedding_returns_segment_length() {
        let length =
            simplex_volume_in_embedding(&[vec![0.0, 0.0, 0.0], vec![3.0, 4.0, 0.0]]).unwrap();
        assert!((length - 5.0).abs() < 1e-12);
    }

    #[test]
    fn simplex_volume_in_embedding_rejects_identical_endpoints() {
        let err = simplex_volume_in_embedding(&[vec![1.0, 2.0], vec![1.0, 2.0]]).unwrap_err();
        assert!(matches!(err, GeometryError::DegenerateSimplex));
    }

    #[test]
    fn circumsphere_supports_one_dimensional_segment() {
        let (center, radius) = circumsphere(&[vec![0.0], vec![2.0]]).unwrap();
        assert_eq!(center, vec![1.0]);
        assert!((radius - 1.0).abs() < 1e-12);
    }

    #[test]
    fn robust_orientation_matches_orientation_on_well_conditioned_input() {
        let faces_2d = [
            (vec![vec![1.0, 0.0], vec![0.0, 1.0]], vec![0.1, 0.1]),
            (vec![vec![0.0, 1.0], vec![1.0, 0.0]], vec![0.1, 0.1]),
            (vec![vec![3.0, -2.0], vec![-1.5, 4.0]], vec![10.0, -7.0]),
        ];
        for (face, origin) in &faces_2d {
            assert_eq!(
                robust_orientation(face, origin).unwrap(),
                orientation(face, origin).unwrap()
            );
        }

        let faces_3d = [
            (
                vec![
                    vec![1.0, 0.0, 0.0],
                    vec![0.0, 1.0, 0.0],
                    vec![0.0, 0.0, 1.0],
                ],
                vec![0.0, 0.0, 0.0],
            ),
            (
                vec![
                    vec![2.0, 1.0, -1.0],
                    vec![-1.0, 3.0, 0.5],
                    vec![0.5, -2.0, 4.0],
                ],
                vec![5.0, 5.0, 5.0],
            ),
        ];
        for (face, origin) in &faces_3d {
            assert_eq!(
                robust_orientation(face, origin).unwrap(),
                orientation(face, origin).unwrap()
            );
        }
    }

    #[test]
    fn robust_orientation_resolves_signs_below_the_log_det_cutoff() {
        // A sliver whose determinant is far below exp(-50): the slogdet-based
        // test gives up and reports 0, the exact predicate knows the sign.
        let face = vec![vec![1e-12, 0.0], vec![0.0, 1e-12]];
        let origin = vec![1e-13, 1e-13];
        assert_eq!(orientation(&face, &origin).unwrap(), 0);
        assert_eq!(robust_orientation(&face, &origin).unwrap(), 1);
    }

    #[test]
    fn circumsphere_is_accurate_for_small_simplices_far_from_origin() {
        let (center, radius) = circumsphere(&[vec![0.5], vec![0.5 + 1e-6]]).unwrap();
        assert!((center[0] - (0.5 + 5e-7)).abs() < 1e-15);
        assert!((radius - 5e-7).abs() < 1e-15);

        let (center, radius) = circumsphere(&[
            vec![100.0, 100.0, 100.0, 100.0],
            vec![100.0 + 1e-5, 100.0, 100.0, 100.0],
            vec![100.0, 100.0 + 1e-5, 100.0, 100.0],
            vec![100.0, 100.0, 100.0 + 1e-5, 100.0],
            vec![100.0, 100.0, 100.0, 100.0 + 1e-5],
        ])
        .unwrap();
        for coord in &center {
            assert!((coord - (100.0 + 5e-6)).abs() < 1e-12);
        }
        assert!((radius - 1e-5).abs() < 1e-12);
    }
}
