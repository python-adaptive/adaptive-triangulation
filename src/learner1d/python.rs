use numpy::{PyArray1, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::loss::LossFunction;
use super::{Learner1D, YValue};

/// Extract a `YValue` from a Python object.
fn extract_yvalue(obj: &Bound<'_, pyo3::types::PyAny>) -> PyResult<YValue> {
    if let Ok(v) = obj.extract::<f64>() {
        return Ok(YValue::Scalar(v));
    }
    if let Ok(v) = obj.extract::<Vec<f64>>() {
        return Ok(YValue::Vector(v));
    }
    Err(PyValueError::new_err(
        "y must be a float or a sequence of floats",
    ))
}

/// Detect `nth_neighbors` attribute on a Python loss callable.
fn detect_nth_neighbors(py: Python<'_>, obj: &PyObject) -> usize {
    obj.bind(py)
        .getattr("nth_neighbors")
        .and_then(|a| a.extract::<usize>())
        .unwrap_or(0)
}

#[pyclass(name = "Learner1D")]
pub struct PyLearner1D {
    pub(crate) inner: Learner1D,
}

#[pymethods]
impl PyLearner1D {
    #[new]
    #[pyo3(signature = (bounds, loss_per_interval=None))]
    fn new(
        py: Python<'_>,
        bounds: (f64, f64),
        loss_per_interval: Option<PyObject>,
    ) -> PyResult<Self> {
        if bounds.0 >= bounds.1 {
            return Err(PyValueError::new_err(
                "bounds[0] must be strictly less than bounds[1]",
            ));
        }
        let loss_fn = match loss_per_interval {
            Some(obj) => {
                let nn = detect_nth_neighbors(py, &obj);
                LossFunction::PythonCallback {
                    callback: obj,
                    nth_neighbors: nn,
                }
            }
            None => LossFunction::Default,
        };
        Ok(Self {
            inner: Learner1D::new(bounds, loss_fn),
        })
    }

    fn tell(&mut self, x: f64, y: &Bound<'_, pyo3::types::PyAny>) -> PyResult<()> {
        let yv = extract_yvalue(y)?;
        self.inner.tell(x, yv);
        Ok(())
    }

    #[pyo3(signature = (xs, ys, force=false))]
    fn tell_many(
        &mut self,
        xs: Vec<f64>,
        ys: Vec<Bound<'_, pyo3::types::PyAny>>,
        force: bool,
    ) -> PyResult<()> {
        let yvalues: Vec<YValue> = ys
            .iter()
            .map(extract_yvalue)
            .collect::<PyResult<_>>()?;
        self.inner.tell_many(&xs, &yvalues, force);
        Ok(())
    }

    fn tell_pending(&mut self, x: f64) {
        self.inner.tell_pending(x);
    }

    #[pyo3(signature = (n, tell_pending=true))]
    fn ask(&mut self, n: usize, tell_pending: bool) -> (Vec<f64>, Vec<f64>) {
        self.inner.ask(n, tell_pending)
    }

    /// Run the full adaptive loop — only user function evals cross the PyO3 boundary.
    #[pyo3(signature = (f, *, goal=None, n_points=None, batch_size=1))]
    fn run(
        &mut self,
        py: Python<'_>,
        f: PyObject,
        goal: Option<f64>,
        n_points: Option<usize>,
        batch_size: usize,
    ) -> PyResult<usize> {
        let mut n_evaluated: usize = 0;
        let bs = batch_size.max(1);
        loop {
            // Check stopping conditions
            if let Some(g) = goal {
                if self.inner.npoints() > 0 && self.inner.loss(true) <= g {
                    break;
                }
            }
            if let Some(np) = n_points {
                if n_evaluated >= np {
                    break;
                }
            }
            if goal.is_none() && n_points.is_none() {
                break;
            }

            let ask_n = if let Some(np) = n_points {
                bs.min(np - n_evaluated)
            } else {
                bs
            };

            let (xs, _) = self.inner.ask(ask_n, true);
            if xs.is_empty() {
                break;
            }

            let yvalues: Vec<YValue> = xs
                .iter()
                .map(|&x| {
                    let result = f.call1(py, (x,))?;
                    extract_yvalue(result.bind(py))
                })
                .collect::<PyResult<_>>()?;

            self.inner.tell_many(&xs, &yvalues, false);
            n_evaluated += xs.len();
        }
        Ok(n_evaluated)
    }

    #[pyo3(signature = (real=true))]
    fn loss(&self, real: bool) -> f64 {
        self.inner.loss(real)
    }

    #[getter]
    fn npoints(&self) -> usize {
        self.inner.npoints()
    }

    #[getter]
    fn vdim(&self) -> Option<usize> {
        self.inner.vdim
    }

    fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyArray1<f64>>, PyObject)> {
        let data = self.inner.to_sorted_data();
        if data.is_empty() {
            let xs = PyArray1::from_vec(py, vec![]);
            let ys: PyObject = PyArray1::from_vec(py, Vec::<f64>::new()).into_any().unbind();
            return Ok((xs, ys));
        }

        let xs: Vec<f64> = data.iter().map(|(x, _)| *x).collect();
        let xs_arr = PyArray1::from_vec(py, xs);

        let vdim = self.inner.vdim.unwrap_or(1);
        if vdim == 1 {
            let ys: Vec<f64> = data
                .iter()
                .map(|(_, y)| match y {
                    YValue::Scalar(v) => *v,
                    _ => 0.0,
                })
                .collect();
            let ys_arr: PyObject = PyArray1::from_vec(py, ys).into_any().unbind();
            Ok((xs_arr, ys_arr))
        } else {
            let rows: Vec<Vec<f64>> = data
                .iter()
                .map(|(_, y)| match y {
                    YValue::Scalar(v) => vec![*v],
                    YValue::Vector(v) => v.clone(),
                })
                .collect();
            let ys_arr: PyObject = PyArray2::from_vec2(py, &rows)?.into_any().unbind();
            Ok((xs_arr, ys_arr))
        }
    }

    fn remove_unfinished(&mut self) {
        self.inner.remove_unfinished();
    }

    #[getter]
    fn pending_points(&self) -> Vec<f64> {
        self.inner.pending.iter().map(|x| x.into_inner()).collect()
    }

    #[getter]
    fn data<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        let dict = pyo3::types::PyDict::new(py);
        // Include both in-bounds and out-of-bounds data (matches Python behavior)
        for (&x, y) in self.inner.data.iter().chain(self.inner.out_of_bounds_data.iter()) {
            let key = x.into_inner();
            let val: PyObject = match y {
                YValue::Scalar(v) => v.into_pyobject(py)?.into_any().unbind(),
                YValue::Vector(v) => pyo3::types::PyList::new(py, v)?.into_any().unbind(),
            };
            dict.set_item(key, val)?;
        }
        Ok(dict.into_any().unbind())
    }

    /// Return `[(x_left, x_right, loss), …]` for all real intervals.
    fn intervals(&self) -> Vec<(f64, f64, f64)> {
        self.inner.intervals_with_loss()
    }

    #[getter]
    fn bounds(&self) -> (f64, f64) {
        self.inner.bounds
    }
}
