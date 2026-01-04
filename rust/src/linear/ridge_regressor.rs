#![allow(deprecated)]

use ndarray::{Array1, Axis};
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use ndarray_linalg::Solve;

/// Ridge Regression.
///
/// Linear least squares with L2 regularization.
///
/// Parameters
/// ----------
/// alpha : float, default=1.0
///     Regularization strength; must be non-negative.
/// fit_intercept : bool, default=True
///     Whether to calculate the intercept for this model.
/// solver : str, default='auto'
///     Solver to use. Supported: 'auto', 'cholesky'.
#[pyclass]
pub struct RidgeRegressor {
    alpha: f64,
    fit_intercept: bool,
    solver: String,

    coef: Option<Array1<f64>>,
    intercept: Option<f64>,
    n_features_in: Option<usize>,
    n_samples_fit: Option<usize>,
}

#[pymethods]
impl RidgeRegressor {
    #[new]
    #[pyo3(signature = (*, alpha=1.0, fit_intercept=true, solver="auto"))]
    fn new(alpha: f64, fit_intercept: bool, solver: &str) -> PyResult<Self> {
        if alpha < 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "alpha must be >= 0",
            ));
        }
        if solver != "auto" && solver != "cholesky" {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "solver must be 'auto' or 'cholesky'",
            ));
        }

        Ok(RidgeRegressor {
            alpha,
            fit_intercept,
            solver: solver.to_string(),
            coef: None,
            intercept: None,
            n_features_in: None,
            n_samples_fit: None,
        })
    }

    /// Fit ridge regression.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     Training data.
    /// y : array-like of shape (n_samples,)
    ///     Target values.
    ///
    /// Returns
    /// -------
    /// self : RidgeRegressor
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        _py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let x_arr = x.as_array();
        let y_arr = y.as_array();

        let n_samples = x_arr.nrows();
        let n_features = x_arr.ncols();

        if y_arr.len() != n_samples {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "X and y have incompatible shapes",
            ));
        }
        if n_samples == 0 || n_features == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "X must be non-empty",
            ));
        }

        let (x_centered, y_centered, x_mean, y_mean) = if slf.fit_intercept {
            let x_mean = x_arr.mean_axis(Axis(0)).unwrap();
            let y_mean = y_arr.sum() / y_arr.len() as f64;
            (
                &x_arr - &x_mean,
                &y_arr - y_mean,
                Some(x_mean),
                Some(y_mean),
            )
        } else {
            (x_arr.to_owned(), y_arr.to_owned(), None, None)
        };

        let x_centered = x_centered.as_standard_layout().to_owned();
        let y_centered = y_centered.as_standard_layout().to_owned();

        let mut xtx = x_centered.t().dot(&x_centered);
        let xty = x_centered.t().dot(&y_centered);

        if slf.alpha > 0.0 {
            for i in 0..n_features {
                xtx[(i, i)] += slf.alpha;
            }
        }

        let coef_vec = xtx
            .solve(&xty)
            .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Solve failed"))?;
        let intercept = if slf.fit_intercept {
            let x_mean = x_mean.unwrap();
            let y_mean = y_mean.unwrap();
            y_mean - x_mean.dot(&coef_vec)
        } else {
            0.0
        };

        slf.coef = Some(coef_vec);
        slf.intercept = Some(intercept);
        slf.n_features_in = Some(n_features);
        slf.n_samples_fit = Some(n_samples);

        Ok(slf)
    }

    /// Predict using the ridge regression model.
    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let coef = self
            .coef
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Not fitted yet"))?;
        let intercept = self
            .intercept
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Not fitted yet"))?;

        let x_arr = x.as_array();
        if x_arr.ncols() != coef.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "X has incompatible number of features",
            ));
        }

        let y_pred = x_arr.dot(coef) + intercept;
        Ok(y_pred.into_pyarray(py))
    }

    /// Return the coefficient of determination (R^2) of the prediction.
    fn score<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<f64> {
        let y_true = y.as_array();
        let y_pred_arr = self.predict(py, x)?;
        let y_pred = unsafe { y_pred_arr.as_array() };

        let y_mean: f64 = y_true.iter().sum::<f64>() / y_true.len() as f64;

        let ss_res: f64 = y_true
            .iter()
            .zip(y_pred.iter())
            .map(|(yt, yp)| {
                let diff = yt - yp;
                diff * diff
            })
            .sum();

        let ss_tot: f64 = y_true
            .iter()
            .map(|yt| {
                let diff = yt - y_mean;
                diff * diff
            })
            .sum();

        if ss_tot == 0.0 {
            if ss_res == 0.0 {
                Ok(1.0)
            } else {
                Ok(0.0)
            }
        } else {
            Ok(1.0 - ss_res / ss_tot)
        }
    }

    /// Coefficients of the model.
    #[getter]
    fn coef_<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        match &self.coef {
            Some(coef) => Ok(coef.clone().into_pyarray(py).into_any().unbind()),
            None => Ok(py.None()),
        }
    }

    /// Intercept of the model.
    #[getter]
    fn intercept_(&self) -> Option<f64> {
        self.intercept
    }

    /// Number of features seen during fit.
    #[getter]
    fn n_features_in_(&self) -> Option<usize> {
        self.n_features_in
    }

    /// Number of samples in the fitted data.
    #[getter]
    fn n_samples_fit_(&self) -> Option<usize> {
        self.n_samples_fit
    }
}
