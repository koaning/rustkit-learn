#![allow(deprecated)]

use ndarray::{Array1, Array2, Axis, Zip};
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

use ndarray_linalg::Solve;

/// Huber Regressor.
///
/// Linear regression model that is robust to outliers.
/// Uses the Huber loss function which is quadratic for small residuals
/// and linear for large residuals.
///
/// Parameters
/// ----------
/// epsilon : float, default=1.35
///     The parameter epsilon controls the number of samples that should be
///     classified as outliers. The smaller the epsilon, the more robust it is
///     to outliers. Must be greater than 1.0.
/// max_iter : int, default=100
///     Maximum number of iterations for the IRLS algorithm.
/// alpha : float, default=0.0001
///     Regularization parameter. Must be non-negative.
/// fit_intercept : bool, default=True
///     Whether to calculate the intercept for this model.
/// tol : float, default=1e-5
///     The iteration will stop when max(|coef_new - coef_old|) < tol.
#[pyclass]
pub struct HuberRegressor {
    epsilon: f64,
    max_iter: usize,
    alpha: f64,
    fit_intercept: bool,
    tol: f64,

    coef: Option<Array1<f64>>,
    intercept: Option<f64>,
    scale: Option<f64>,
    n_features_in: Option<usize>,
    n_iter: Option<usize>,
}

#[pymethods]
impl HuberRegressor {
    #[new]
    #[pyo3(signature = (*, epsilon=1.35, max_iter=100, alpha=0.0001, fit_intercept=true, tol=1e-5))]
    fn new(
        epsilon: f64,
        max_iter: usize,
        alpha: f64,
        fit_intercept: bool,
        tol: f64,
    ) -> PyResult<Self> {
        if epsilon <= 1.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "epsilon must be greater than 1.0",
            ));
        }
        if alpha < 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "alpha must be >= 0",
            ));
        }
        if max_iter == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "max_iter must be > 0",
            ));
        }
        if tol <= 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "tol must be > 0",
            ));
        }

        Ok(HuberRegressor {
            epsilon,
            max_iter,
            alpha,
            fit_intercept,
            tol,
            coef: None,
            intercept: None,
            scale: None,
            n_features_in: None,
            n_iter: None,
        })
    }

    /// Fit the Huber regressor.
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
    /// self : HuberRegressor
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

        // Center data if fit_intercept is True
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

        // Initialize coefficients with OLS solution
        let mut xtx = x_centered.t().dot(&x_centered);
        let xty = x_centered.t().dot(&y_centered);

        // Add small regularization to prevent singularity
        for i in 0..n_features {
            xtx[(i, i)] += slf.alpha;
        }

        let mut coef = xtx
            .solve(&xty)
            .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Initial solve failed"))?;

        // Compute initial residuals
        let mut residuals = &y_centered - &x_centered.dot(&coef);

        // Initialize scale using MAD (Median Absolute Deviation)
        let mut scale = compute_mad(&residuals);
        if scale < 1e-10 {
            scale = 1.0; // Fallback if all residuals are very small
        }

        let mut n_iter = 0;

        // IRLS iteration
        for iter in 0..slf.max_iter {
            n_iter = iter + 1;

            // Compute weights based on Huber loss
            let weights = compute_huber_weights(&residuals, scale, slf.epsilon);

            // Solve weighted least squares: (X^T W X + alpha*I) coef = X^T W y
            let coef_new =
                solve_weighted_ridge(&x_centered, &y_centered, &weights, slf.alpha, n_features)?;

            // Check convergence
            let max_change = (&coef_new - &coef)
                .iter()
                .map(|x| x.abs())
                .fold(0.0_f64, f64::max);

            coef = coef_new;

            // Update residuals
            residuals = &y_centered - &x_centered.dot(&coef);

            // Update scale estimate
            scale = compute_huber_scale(&residuals, scale, slf.epsilon);
            if scale < 1e-10 {
                scale = 1e-10;
            }

            if max_change < slf.tol {
                break;
            }
        }

        // Compute intercept
        let intercept = if slf.fit_intercept {
            let x_mean = x_mean.unwrap();
            let y_mean = y_mean.unwrap();
            y_mean - x_mean.dot(&coef)
        } else {
            0.0
        };

        slf.coef = Some(coef);
        slf.intercept = Some(intercept);
        slf.scale = Some(scale);
        slf.n_features_in = Some(n_features);
        slf.n_iter = Some(n_iter);

        Ok(slf)
    }

    /// Predict using the Huber regression model.
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

    /// The "scale" parameter estimated by the algorithm.
    #[getter]
    fn scale_(&self) -> Option<f64> {
        self.scale
    }

    /// Number of features seen during fit.
    #[getter]
    fn n_features_in_(&self) -> Option<usize> {
        self.n_features_in
    }

    /// Number of iterations run by the algorithm.
    #[getter]
    fn n_iter_(&self) -> Option<usize> {
        self.n_iter
    }
}

/// Compute Median Absolute Deviation (MAD) scaled to estimate standard deviation
fn compute_mad(residuals: &Array1<f64>) -> f64 {
    // Use parallel iterator for large arrays
    let mut abs_residuals: Vec<f64> = if residuals.len() > 10_000 {
        residuals
            .as_slice()
            .unwrap()
            .par_iter()
            .map(|r| r.abs())
            .collect()
    } else {
        residuals.iter().map(|r| r.abs()).collect()
    };

    // Parallel sort for large arrays
    if abs_residuals.len() > 10_000 {
        abs_residuals.par_sort_by(|a, b| a.partial_cmp(b).unwrap());
    } else {
        abs_residuals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    }

    let n = abs_residuals.len();
    let median = if n.is_multiple_of(2) {
        (abs_residuals[n / 2 - 1] + abs_residuals[n / 2]) / 2.0
    } else {
        abs_residuals[n / 2]
    };

    // Scale factor to make MAD consistent with std for normal distribution
    median / 0.6745
}

/// Compute Huber weights for IRLS
fn compute_huber_weights(residuals: &Array1<f64>, scale: f64, epsilon: f64) -> Array1<f64> {
    let threshold = epsilon * scale;

    // Use parallel computation for large arrays
    if residuals.len() > 10_000 {
        let weights: Vec<f64> = residuals
            .as_slice()
            .unwrap()
            .par_iter()
            .map(|&r| {
                let abs_r = r.abs();
                if abs_r <= threshold {
                    1.0
                } else {
                    threshold / abs_r
                }
            })
            .collect();
        Array1::from_vec(weights)
    } else {
        residuals.mapv(|r| {
            let abs_r = r.abs();
            if abs_r <= threshold {
                1.0
            } else {
                threshold / abs_r
            }
        })
    }
}

/// Solve weighted ridge regression: (X^T W X + alpha*I) coef = X^T W y
fn solve_weighted_ridge(
    x: &Array2<f64>,
    y: &Array1<f64>,
    weights: &Array1<f64>,
    alpha: f64,
    n_features: usize,
) -> PyResult<Array1<f64>> {
    // Apply weights: X_w = sqrt(W) * X, y_w = sqrt(W) * y
    let sqrt_weights = weights.mapv(|w| w.sqrt());

    // Create weighted X and y using Zip for efficiency
    let mut x_weighted = x.to_owned();
    Zip::from(x_weighted.rows_mut())
        .and(&sqrt_weights)
        .for_each(|mut row, &sw| {
            row.mapv_inplace(|v| v * sw);
        });

    let y_weighted = y * &sqrt_weights;

    // Form normal equations
    let mut xtx = x_weighted.t().dot(&x_weighted);
    let xty = x_weighted.t().dot(&y_weighted);

    // Add regularization
    for i in 0..n_features {
        xtx[(i, i)] += alpha;
    }

    xtx.solve(&xty)
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Weighted solve failed"))
}

/// Update scale estimate using Huber's proposal 2
fn compute_huber_scale(residuals: &Array1<f64>, current_scale: f64, epsilon: f64) -> f64 {
    let n = residuals.len() as f64;
    let threshold = epsilon * current_scale;
    let threshold_sq_half = threshold * threshold / 2.0;

    // Compute robust estimate of scale - use parallel for large arrays
    let sum_rho: f64 = if residuals.len() > 10_000 {
        residuals
            .as_slice()
            .unwrap()
            .par_iter()
            .map(|&r| {
                let abs_r = r.abs();
                if abs_r <= threshold {
                    r * r / 2.0
                } else {
                    threshold * abs_r - threshold_sq_half
                }
            })
            .sum()
    } else {
        residuals
            .iter()
            .map(|&r| {
                let abs_r = r.abs();
                if abs_r <= threshold {
                    r * r / 2.0
                } else {
                    threshold * abs_r - threshold_sq_half
                }
            })
            .sum()
    };

    // Scale update based on Huber's M-estimator
    let scale_squared = 2.0 * sum_rho / (n * (2.0 * epsilon.powi(2) - 1.0).max(0.1));
    scale_squared.sqrt().max(1e-10)
}
