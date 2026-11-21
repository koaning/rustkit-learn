#![allow(deprecated)]

use ndarray::{Array1, Axis};
use ndarray_linalg::LeastSquaresSvd;
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn not_fitted() -> PyErr {
    PyErr::new::<PyValueError, _>("Not fitted yet")
}

/// Ordinary least squares Linear Regression.
///
/// Fits a linear model with coefficients w = (w1, ..., wp) to minimize the
/// residual sum of squares between the observed targets and the predicted
/// targets. Solved with an SVD-based least squares solver, so rank deficient
/// (collinear) inputs yield the minimum-norm solution, like sklearn.
///
/// Parameters
/// ----------
/// fit_intercept : bool, default=True
///     Whether to calculate the intercept for this model.
#[pyclass]
pub struct LinearRegression {
    fit_intercept: bool,

    coef: Option<Array1<f64>>,
    intercept: Option<f64>,
    rank: Option<i32>,
    singular: Option<Array1<f64>>,
    n_features_in: Option<usize>,
}

#[pymethods]
impl LinearRegression {
    #[new]
    #[pyo3(signature = (*, fit_intercept=true))]
    fn new(fit_intercept: bool) -> Self {
        LinearRegression {
            fit_intercept,
            coef: None,
            intercept: None,
            rank: None,
            singular: None,
            n_features_in: None,
        }
    }

    /// Fit linear model.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     Training data.
    /// y : array-like of shape (n_samples,)
    ///     Target values.
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let x_arr = x.as_array().to_owned();
        let y_arr = y.as_array().to_owned();
        let (n_samples, n_features) = x_arr.dim();

        if y_arr.len() != n_samples {
            return Err(PyErr::new::<PyValueError, _>(
                "X and y have incompatible shapes",
            ));
        }
        if n_samples == 0 || n_features == 0 {
            return Err(PyErr::new::<PyValueError, _>("X must be non-empty"));
        }

        let fit_intercept = slf.fit_intercept;
        let (coef, intercept, rank, singular) = py
            .allow_threads(move || {
                let (xc, yc, x_mean, y_mean) = if fit_intercept {
                    let x_mean = x_arr.mean_axis(Axis(0)).unwrap();
                    let y_mean = y_arr.mean().unwrap();
                    (&x_arr - &x_mean, &y_arr - y_mean, Some(x_mean), y_mean)
                } else {
                    (x_arr, y_arr, None, 0.0)
                };
                let res = xc.least_squares(&yc)?;
                let coef = res.solution;
                let intercept = match x_mean {
                    Some(m) => y_mean - m.dot(&coef),
                    None => 0.0,
                };
                Ok::<_, ndarray_linalg::error::LinalgError>((
                    coef,
                    intercept,
                    res.rank,
                    res.singular_values,
                ))
            })
            .map_err(|e| PyErr::new::<PyValueError, _>(format!("Least squares failed: {}", e)))?;

        slf.coef = Some(coef);
        slf.intercept = Some(intercept);
        slf.rank = Some(rank);
        slf.singular = Some(singular);
        slf.n_features_in = Some(n_features);
        Ok(slf)
    }

    /// Predict using the linear model.
    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let coef = self.coef.as_ref().ok_or_else(not_fitted)?;
        let intercept = self.intercept.ok_or_else(not_fitted)?;
        let x_arr = x.as_array();
        if x_arr.ncols() != coef.len() {
            return Err(PyErr::new::<PyValueError, _>(format!(
                "X has {} features, but LinearRegression is expecting {} features",
                x_arr.ncols(),
                coef.len()
            )));
        }
        Ok((x_arr.dot(coef) + intercept).into_pyarray(py))
    }

    /// Return the coefficient of determination (R^2) of the prediction.
    fn score<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<f64> {
        let y_true = y.as_array();
        let y_pred = self.predict(py, x)?.readonly();
        let y_pred = y_pred.as_array();
        let y_mean = y_true.mean().unwrap_or(0.0);
        let ss_res: f64 = y_true
            .iter()
            .zip(y_pred.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum();
        let ss_tot: f64 = y_true.iter().map(|a| (a - y_mean) * (a - y_mean)).sum();
        Ok(if ss_tot == 0.0 {
            if ss_res == 0.0 {
                1.0
            } else {
                0.0
            }
        } else {
            1.0 - ss_res / ss_tot
        })
    }

    /// Estimated coefficients for the linear regression problem.
    #[getter]
    fn coef_<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.coef.as_ref().map(|c| c.clone().into_pyarray(py))
    }

    /// Independent term in the linear model.
    #[getter]
    fn intercept_(&self) -> Option<f64> {
        self.intercept
    }

    /// Rank of the (centered) design matrix.
    #[getter]
    fn rank_(&self) -> Option<i32> {
        self.rank
    }

    /// Singular values of the (centered) design matrix.
    #[getter]
    fn singular_<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.singular.as_ref().map(|s| s.clone().into_pyarray(py))
    }

    /// Number of features seen during fit.
    #[getter]
    fn n_features_in_(&self) -> Option<usize> {
        self.n_features_in
    }

    /// Get parameters for this estimator.
    #[pyo3(signature = (deep=true))]
    fn get_params<'py>(&self, py: Python<'py>, deep: bool) -> PyResult<Bound<'py, PyDict>> {
        let _ = deep;
        let d = PyDict::new(py);
        d.set_item("fit_intercept", self.fit_intercept)?;
        Ok(d)
    }

    /// Set the parameters of this estimator.
    #[pyo3(signature = (**params))]
    fn set_params<'py>(
        mut slf: PyRefMut<'py, Self>,
        params: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        if let Some(p) = params {
            for (k, v) in p.iter() {
                let key: String = k.extract()?;
                match key.as_str() {
                    "fit_intercept" => slf.fit_intercept = v.extract()?,
                    _ => {
                        return Err(PyErr::new::<PyValueError, _>(format!(
                            "Invalid parameter {} for estimator LinearRegression",
                            key
                        )))
                    }
                }
            }
        }
        Ok(slf)
    }
}
