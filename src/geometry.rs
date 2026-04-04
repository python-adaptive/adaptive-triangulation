use nalgebra::{DMatrix, DVector};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum GeometryError {
    #[error("{0}")]
    InvalidDimensions(String),
    #[error("Provided vertices do not form a simplex")]
    DegenerateSimplex,
    #[error("Singular matrix")]
    SingularMatrix,
}

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

pub fn numpy_matrix_rank(vectors: &[Vec<f64>]) -> Result<usize, GeometryError> {
    matrix_rank(vectors, -1.0)
}

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

    let x0 = &simplex[0];
    let dim = point.len();
    let mut matrix = vec![vec![0.0; dim]; dim];
    let mut rhs = vec![0.0; dim];

    for row in 0..dim {
        rhs[row] = point[row] - x0[row];
        for col in 0..dim {
            matrix[row][col] = simplex[col + 1][row] - x0[row];
        }
    }

    let alpha = solve_square(&matrix, &rhs)?;
    Ok(alpha.iter().all(|value| *value > -eps) && alpha.iter().sum::<f64>() < 1.0 + eps)
}

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

pub fn circumsphere(pts: &[Vec<f64>]) -> Result<(Vec<f64>, f64), GeometryError> {
    let dim = validate_points(pts)?;
    if pts.len() != dim + 1 {
        return Err(GeometryError::InvalidDimensions(format!(
            "Expected {} points for a {}-dimensional simplex",
            dim + 1,
            dim
        )));
    }

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

    let x0 = &pts[0];
    let x0_sq = squared_norm(x0);
    let mut matrix = vec![vec![0.0; dim]; dim];
    let mut rhs = vec![0.0; dim];

    for row in 0..dim {
        let point = &pts[row + 1];
        rhs[row] = squared_norm(point) - x0_sq;
        for col in 0..dim {
            matrix[row][col] = 2.0 * (point[col] - x0[col]);
        }
    }

    let center = match solve_square(&matrix, &rhs) {
        Ok(center) => center,
        Err(GeometryError::SingularMatrix) => {
            return Ok((vec![f64::NAN; dim], f64::NAN));
        }
        Err(err) => return Err(err),
    };

    let radius = fast_norm(
        &center
            .iter()
            .zip(&pts[0])
            .map(|(x, y)| x - y)
            .collect::<Vec<_>>(),
    );
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
    if sign == 0.0 || log_det < -50.0 {
        Ok(0)
    } else if sign.is_sign_positive() {
        Ok(1)
    } else {
        Ok(-1)
    }
}

pub fn volume(vertices: &[Vec<f64>]) -> Result<f64, GeometryError> {
    let dim = validate_points(vertices)?;
    if vertices.len() != dim + 1 {
        return Err(GeometryError::InvalidDimensions(format!(
            "Expected {} points for a {}-dimensional simplex",
            dim + 1,
            dim
        )));
    }

    let mut matrix = vec![vec![0.0; dim]; dim];
    let x0 = &vertices[0];
    for row in 0..dim {
        for col in 0..dim {
            matrix[row][col] = vertices[col + 1][row] - x0[row];
        }
    }
    Ok(determinant(&matrix)?.abs() / factorial(dim))
}

pub fn simplex_volume_in_embedding(vertices: &[Vec<f64>]) -> Result<f64, GeometryError> {
    validate_points(vertices)?;
    if vertices.len() < 2 {
        return Err(GeometryError::InvalidDimensions(
            "Expected at least two vertices".to_string(),
        ));
    }

    if vertices[0].len() == 2 && vertices.len() != 3 {
        return Err(GeometryError::InvalidDimensions(
            "Expected three 2D vertices".to_string(),
        ));
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
