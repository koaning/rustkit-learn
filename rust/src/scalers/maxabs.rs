#![allow(deprecated)]

use ndarray::{Array1, Array2, ArrayView2, Axis, Zip};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn not_fitted() -> PyErr {
    PyErr::new::<PyValueError, _>("Not fitted yet")
}

/// Per-column maximum absolute value.
fn compute_max_abs(data: ArrayView2<f64>) -> Array1<f64> {
    let mut out = Array1::<f64>::zeros(data.ncols());
    for row in data.axis_iter(Axis(0)) {
        Zip::from(&mut out).and(&row).for_each(|m, &v| {
            let a = v.abs();
            if a > *m {
                *m = a;
            }
        });
    }
    out
}

/// Multiply each column of `data` by `factor[j]` (in parallel over rows).
fn scale_columns(data: ArrayView2<f64>, factor: &Array1<f64>) -> Array2<f64> {
    let mut out = data.to_owned();
    Zip::from(out.rows_mut()).par_for_each(|mut row| {
        Zip::from(&mut row).and(factor).for_each(|v, &f| *v *= f);
    });
    out
}

/// Scale each feature by its maximum absolute value.
///
/// This estimator scales each feature individually such that the maximal
/// absolute value of each feature in the training set will be 1.0. It does
/// not shift/center the data, and thus does not destroy any sparsity.
///
/// This scaler can be used incrementally via `partial_fit`.
#[pyclass]
pub struct MaxAbsScaler {
    max_abs: Option<Array1<f64>>,
    scale: Option<Array1<f64>>,
    n_samples_seen: usize,
}

impl MaxAbsScaler {
    fn set_max_abs(&mut self, max_abs: Array1<f64>) {
        // Constant zero features are left untouched (scale of 1), like sklearn.
        self.scale = Some(max_abs.mapv(|m| if m == 0.0 { 1.0 } else { m }));
        self.max_abs = Some(max_abs);
    }

    fn check_features(&self, n_features: usize) -> PyResult<&Array1<f64>> {
        let scale = self.scale.as_ref().ok_or_else(not_fitted)?;
        if scale.len() != n_features {
            return Err(PyErr::new::<PyValueError, _>(format!(
                "X has {} features, but MaxAbsScaler is expecting {} features",
                n_features,
                scale.len()
            )));
        }
        Ok(scale)
    }
}

#[pymethods]
impl MaxAbsScaler {
    #[new]
    fn new() -> Self {
        MaxAbsScaler {
            max_abs: None,
            scale: None,
            n_samples_seen: 0,
        }
    }

    /// Compute the maximum absolute value to be used for later scaling.
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyRefMut<'py, Self> {
        let arr = x.as_array().to_owned();
        slf.n_samples_seen = arr.nrows();
        let max_abs = py.allow_threads(move || compute_max_abs(arr.view()));
        slf.set_max_abs(max_abs);
        slf
    }

    /// Online computation of max absolute value of X for later scaling.
    fn partial_fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let arr = x.as_array().to_owned();
        let n_rows = arr.nrows();
        let batch = py.allow_threads(move || compute_max_abs(arr.view()));
        let merged = match &slf.max_abs {
            Some(existing) => {
                if existing.len() != batch.len() {
                    return Err(PyErr::new::<PyValueError, _>(
                        "X has a different number of features than previous batches",
                    ));
                }
                Zip::from(existing)
                    .and(&batch)
                    .map_collect(|&a, &b| a.max(b))
            }
            None => batch,
        };
        slf.n_samples_seen += n_rows;
        slf.set_max_abs(merged);
        Ok(slf)
    }

    /// Scale the data.
    fn transform<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let arr = x.as_array().to_owned();
        let inv_scale = self.check_features(arr.ncols())?.mapv(|s| 1.0 / s);
        let out = py.allow_threads(move || scale_columns(arr.view(), &inv_scale));
        Ok(out.into_pyarray(py))
    }

    /// Fit to data, then transform it.
    fn fit_transform<'py>(
        slf: PyRefMut<'py, Self>,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let fitted = Self::fit(slf, py, x.clone());
        fitted.transform(py, x)
    }

    /// Scale back the data to the original representation.
    fn inverse_transform<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let arr = x.as_array().to_owned();
        let scale = self.check_features(arr.ncols())?.clone();
        let out = py.allow_threads(move || scale_columns(arr.view(), &scale));
        Ok(out.into_pyarray(py))
    }

    /// Per feature maximum absolute value.
    #[getter]
    fn max_abs_<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.max_abs.as_ref().map(|v| v.clone().into_pyarray(py))
    }

    /// Per feature relative scaling of the data.
    #[getter]
    fn scale_<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.scale.as_ref().map(|v| v.clone().into_pyarray(py))
    }

    /// Number of features seen during fit.
    #[getter]
    fn n_features_in_(&self) -> Option<usize> {
        self.max_abs.as_ref().map(|m| m.len())
    }

    /// Number of samples processed by the estimator.
    #[getter]
    fn n_samples_seen_(&self) -> usize {
        self.n_samples_seen
    }

    /// Get parameters for this estimator.
    #[pyo3(signature = (deep=true))]
    fn get_params<'py>(&self, py: Python<'py>, deep: bool) -> Bound<'py, PyDict> {
        let _ = deep;
        PyDict::new(py)
    }

    /// Set the parameters of this estimator.
    #[pyo3(signature = (**params))]
    fn set_params<'py>(
        slf: PyRefMut<'py, Self>,
        params: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        if let Some(p) = params {
            if let Some((k, _)) = p.iter().next() {
                return Err(PyErr::new::<PyValueError, _>(format!(
                    "Invalid parameter {} for estimator MaxAbsScaler",
                    k
                )));
            }
        }
        Ok(slf)
    }
}
