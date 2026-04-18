use std::collections::{BTreeSet, HashMap};

use pyo3::prelude::*;

use super::{YValue, OF64};

// ---- Loss function enum ----

/// Built-in and custom loss functions for 1D adaptive sampling.
pub enum LossFunction {
    /// `sqrt(dx² + dy²)` with scaling.
    Default,
    /// `dx` — uniform sampling.
    Uniform,
    /// Default loss clamped by interval width.
    Resolution { min_length: f64, max_length: f64 },
    /// Weighted sum of triangle area + euclidean distance + horizontal distance.
    Curvature {
        area_factor: f64,
        euclid_factor: f64,
        horizontal_factor: f64,
    },
    /// Average area of triangles formed by 4 neighbouring points.
    Triangle,
    /// Default loss on `log(|y|)`.
    AbsMinLog,
    /// Python callable `(xs, ys) -> float`.
    PythonCallback {
        callback: PyObject,
        nth_neighbors: usize,
    },
}

impl LossFunction {
    pub fn nth_neighbors(&self) -> usize {
        match self {
            Self::Default | Self::Uniform | Self::Resolution { .. } | Self::AbsMinLog => 0,
            Self::Curvature { .. } | Self::Triangle => 1,
            Self::PythonCallback { nth_neighbors, .. } => *nth_neighbors,
        }
    }
}

impl std::fmt::Debug for LossFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Default => write!(f, "Default"),
            Self::Uniform => write!(f, "Uniform"),
            Self::Resolution {
                min_length,
                max_length,
            } => f
                .debug_struct("Resolution")
                .field("min_length", min_length)
                .field("max_length", max_length)
                .finish(),
            Self::Curvature {
                area_factor,
                euclid_factor,
                horizontal_factor,
            } => f
                .debug_struct("Curvature")
                .field("area_factor", area_factor)
                .field("euclid_factor", euclid_factor)
                .field("horizontal_factor", horizontal_factor)
                .finish(),
            Self::Triangle => write!(f, "Triangle"),
            Self::AbsMinLog => write!(f, "AbsMinLog"),
            Self::PythonCallback { nth_neighbors, .. } => f
                .debug_struct("PythonCallback")
                .field("nth_neighbors", nth_neighbors)
                .finish(),
        }
    }
}

impl Clone for LossFunction {
    fn clone(&self) -> Self {
        match self {
            Self::Default => Self::Default,
            Self::Uniform => Self::Uniform,
            Self::Resolution {
                min_length,
                max_length,
            } => Self::Resolution {
                min_length: *min_length,
                max_length: *max_length,
            },
            Self::Curvature {
                area_factor,
                euclid_factor,
                horizontal_factor,
            } => Self::Curvature {
                area_factor: *area_factor,
                euclid_factor: *euclid_factor,
                horizontal_factor: *horizontal_factor,
            },
            Self::Triangle => Self::Triangle,
            Self::AbsMinLog => Self::AbsMinLog,
            Self::PythonCallback {
                callback,
                nth_neighbors,
            } => Python::with_gil(|py| Self::PythonCallback {
                callback: callback.clone_ref(py),
                nth_neighbors: *nth_neighbors,
            }),
        }
    }
}

// ---- Loss computation dispatch ----

pub fn compute_loss(loss_fn: &LossFunction, xs: &[Option<f64>], ys: &[Option<&YValue>]) -> f64 {
    match loss_fn {
        LossFunction::Default => default_loss(xs, ys),
        LossFunction::Uniform => uniform_loss(xs),
        LossFunction::Resolution {
            min_length,
            max_length,
        } => resolution_loss(xs, ys, *min_length, *max_length),
        LossFunction::Curvature {
            area_factor,
            euclid_factor,
            horizontal_factor,
        } => curvature_loss(xs, ys, *area_factor, *euclid_factor, *horizontal_factor),
        LossFunction::Triangle => triangle_loss_fn(xs, ys),
        LossFunction::AbsMinLog => abs_min_log_loss(xs, ys),
        LossFunction::PythonCallback { callback, .. } => python_callback_loss(callback, xs, ys),
    }
}

// ---- Individual loss functions ----

fn default_loss(xs: &[Option<f64>], ys: &[Option<&YValue>]) -> f64 {
    let (x0, x1) = (xs[0].unwrap(), xs[1].unwrap());
    let (y0, y1) = (ys[0].unwrap(), ys[1].unwrap());
    let dx = x1 - x0;
    match (y0, y1) {
        (YValue::Scalar(a), YValue::Scalar(b)) => {
            let dy = b - a;
            (dx * dx + dy * dy).sqrt()
        }
        (YValue::Vector(a), YValue::Vector(b)) => a
            .iter()
            .zip(b.iter())
            .map(|(ai, bi)| {
                let dy = (bi - ai).abs();
                (dx * dx + dy * dy).sqrt()
            })
            .fold(f64::NEG_INFINITY, f64::max),
        _ => 0.0,
    }
}

fn uniform_loss(xs: &[Option<f64>]) -> f64 {
    xs[1].unwrap() - xs[0].unwrap()
}

fn resolution_loss(
    xs: &[Option<f64>],
    ys: &[Option<&YValue>],
    min_length: f64,
    max_length: f64,
) -> f64 {
    let dx = uniform_loss(xs);
    if dx < min_length {
        return 0.0;
    }
    if dx > max_length {
        return f64::INFINITY;
    }
    default_loss(xs, ys)
}

fn triangle_loss_fn(xs: &[Option<f64>], ys: &[Option<&YValue>]) -> f64 {
    let points: Vec<(f64, &YValue)> = xs
        .iter()
        .zip(ys.iter())
        .filter_map(|(x, y)| match (x, y) {
            (Some(xv), Some(yv)) => Some((*xv, *yv)),
            _ => None,
        })
        .collect();

    if points.len() <= 1 {
        return 0.0;
    }
    if points.len() == 2 {
        return points[1].0 - points[0].0;
    }

    let n_tri = points.len() - 2;
    let is_vec = matches!(points[0].1, YValue::Vector(_));

    let mut total = 0.0;
    for i in 0..n_tri {
        if is_vec {
            let mk = |p: &(f64, &YValue)| -> Vec<f64> {
                let mut v = vec![p.0];
                if let YValue::Vector(arr) = p.1 {
                    v.extend(arr);
                }
                v
            };
            total += simplex_vol_tri(&mk(&points[i]), &mk(&points[i + 1]), &mk(&points[i + 2]));
        } else {
            let gy = |p: &(f64, &YValue)| -> f64 {
                if let YValue::Scalar(v) = p.1 {
                    *v
                } else {
                    0.0
                }
            };
            total += tri_area_2d(
                points[i].0,
                gy(&points[i]),
                points[i + 1].0,
                gy(&points[i + 1]),
                points[i + 2].0,
                gy(&points[i + 2]),
            );
        }
    }
    total / n_tri as f64
}

fn curvature_loss(
    xs: &[Option<f64>],
    ys: &[Option<&YValue>],
    area_factor: f64,
    euclid_factor: f64,
    horizontal_factor: f64,
) -> f64 {
    let tri_l = triangle_loss_fn(xs, ys);
    let def_l = default_loss(&xs[1..3], &ys[1..3]);
    let dx = uniform_loss(&xs[1..3]);
    area_factor * tri_l.sqrt() + euclid_factor * def_l + horizontal_factor * dx
}

fn abs_min_log_loss(xs: &[Option<f64>], ys: &[Option<&YValue>]) -> f64 {
    let ys_log: Vec<Option<YValue>> = ys
        .iter()
        .map(|y| {
            y.map(|yv| {
                let min_abs = match yv {
                    YValue::Scalar(v) => v.abs(),
                    YValue::Vector(v) => v.iter().map(|x| x.abs()).fold(f64::INFINITY, f64::min),
                };
                YValue::Scalar(min_abs.ln())
            })
        })
        .collect();
    let refs: Vec<Option<&YValue>> = ys_log.iter().map(|y| y.as_ref()).collect();
    default_loss(xs, &refs)
}

fn python_callback_loss(
    callback: &PyObject,
    xs: &[Option<f64>],
    ys: &[Option<&YValue>],
) -> f64 {
    Python::with_gil(|py| {
        let xs_py: Vec<PyObject> = xs
            .iter()
            .map(|x| match x {
                Some(v) => v.into_pyobject(py).unwrap().into_any().unbind(),
                None => py.None(),
            })
            .collect();
        let ys_py: Vec<PyObject> = ys
            .iter()
            .map(|y| match y {
                Some(YValue::Scalar(v)) => v.into_pyobject(py).unwrap().into_any().unbind(),
                Some(YValue::Vector(v)) => {
                    pyo3::types::PyList::new(py, v).unwrap().into_any().unbind()
                }
                None => py.None(),
            })
            .collect();
        let xs_tuple = pyo3::types::PyTuple::new(py, &xs_py).unwrap();
        let ys_tuple = pyo3::types::PyTuple::new(py, &ys_py).unwrap();
        match callback.call1(py, (xs_tuple, ys_tuple)) {
            Ok(result) => result.extract::<f64>(py).unwrap_or_else(|e| {
                e.print(py);
                f64::INFINITY
            }),
            Err(err) => {
                err.print(py);
                f64::INFINITY
            }
        }
    })
}

// ---- Geometry helpers ----

fn tri_area_2d(x0: f64, y0: f64, x1: f64, y1: f64, x2: f64, y2: f64) -> f64 {
    ((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)).abs() / 2.0
}

fn simplex_vol_tri(v0: &[f64], v1: &[f64], v2: &[f64]) -> f64 {
    let dim = v0.len();
    let e1: Vec<f64> = (0..dim).map(|i| v1[i] - v0[i]).collect();
    let e2: Vec<f64> = (0..dim).map(|i| v2[i] - v0[i]).collect();
    let d11: f64 = e1.iter().map(|x| x * x).sum();
    let d12: f64 = e1.iter().zip(e2.iter()).map(|(a, b)| a * b).sum();
    let d22: f64 = e2.iter().map(|x| x * x).sum();
    (d11 * d22 - d12 * d12).abs().sqrt() / 2.0
}

// ---- Finite-loss rounding ----

pub fn round_loss(loss: f64) -> f64 {
    let fac = 1e12_f64;
    (loss * fac + 0.5).floor() / fac
}

pub fn finite_loss_value(loss: f64, left: f64, right: f64, x_scale: f64) -> f64 {
    finite_loss_with_n(loss, left, right, 1, x_scale)
}

pub fn finite_loss_with_n(loss: f64, left: f64, right: f64, n: usize, x_scale: f64) -> f64 {
    let loss = if !loss.is_finite() {
        (right - left) / x_scale / n as f64
    } else {
        loss
    };
    round_loss(loss)
}

// ---- Priority-queue entry (sorted ascending → first() = max loss) ----

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct LossEntry {
    pub neg_finite_loss: OF64,
    pub left: OF64,
    pub right: OF64,
}

// ---- Interval key ----

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Interval {
    pub left: OF64,
    pub right: OF64,
}

// ---- LossManager ----

/// Maps intervals to losses, with a priority queue sorted by finite-loss.
#[derive(Clone)]
pub struct LossManager {
    interval_to_loss: HashMap<Interval, f64>,
    queue: BTreeSet<LossEntry>,
    pub x_scale: f64,
}

impl LossManager {
    pub fn new(x_scale: f64) -> Self {
        Self {
            interval_to_loss: HashMap::new(),
            queue: BTreeSet::new(),
            x_scale,
        }
    }

    fn make_entry(&self, left: OF64, right: OF64, loss: f64) -> LossEntry {
        let fl = finite_loss_value(loss, left.into_inner(), right.into_inner(), self.x_scale);
        LossEntry {
            neg_finite_loss: OF64::from(-fl),
            left,
            right,
        }
    }

    fn loss_for_entry(&self, entry: &LossEntry) -> f64 {
        self.interval_to_loss[&Interval {
            left: entry.left,
            right: entry.right,
        }]
    }

    pub fn insert(&mut self, left: OF64, right: OF64, loss: f64) {
        let ival = Interval { left, right };
        if let Some(old_loss) = self.interval_to_loss.remove(&ival) {
            self.queue.remove(&self.make_entry(left, right, old_loss));
        }
        self.interval_to_loss.insert(ival, loss);
        self.queue.insert(self.make_entry(left, right, loss));
    }

    pub fn remove(&mut self, left: OF64, right: OF64) -> Option<f64> {
        let ival = Interval { left, right };
        if let Some(loss) = self.interval_to_loss.remove(&ival) {
            self.queue.remove(&self.make_entry(left, right, loss));
            Some(loss)
        } else {
            None
        }
    }

    pub fn get(&self, left: OF64, right: OF64) -> Option<f64> {
        self.interval_to_loss
            .get(&Interval { left, right })
            .copied()
    }

    /// Raw loss of the highest-priority interval.
    pub fn peek_max_loss(&self) -> Option<f64> {
        self.queue.iter().next().map(|entry| self.loss_for_entry(entry))
    }

    /// Iterate entries in priority order (highest loss first).
    /// Yields `(left, right, raw_loss)`.
    pub fn iter_by_priority(&self) -> impl Iterator<Item = (OF64, OF64, f64)> + '_ {
        self.queue
            .iter()
            .map(move |entry| (entry.left, entry.right, self.loss_for_entry(entry)))
    }

    /// Collect all intervals (no particular order).
    pub fn all_intervals(&self) -> Vec<(OF64, OF64)> {
        self.interval_to_loss
            .keys()
            .map(|iv| (iv.left, iv.right))
            .collect()
    }
}
