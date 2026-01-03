#![allow(deprecated)]

use ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::utils::distances::{kneighbors_parallel, kneighbors_single, predict_parallel, predict_single};

/// K-Nearest Neighbors Regressor.
///
/// Regression based on k-nearest neighbors. The target is predicted by
/// local interpolation of the targets associated with the nearest neighbors
/// in the training set.
///
/// Parameters
/// ----------
/// n_neighbors : int, default=5
///     Number of neighbors to use for prediction.
///
/// weights : str, default='uniform'
///     Weight function used in prediction. Possible values:
///     - 'uniform': uniform weights (all neighbors weighted equally)
///     - 'distance': weight by inverse of distance
///
/// metric : str, default='euclidean'
///     Distance metric to use. Possible values:
///     - 'euclidean': Euclidean distance (L2)
///     - 'manhattan': Manhattan distance (L1)
///     - 'minkowski': Minkowski distance (Lp)
///
/// p : float, default=2.0
///     Power parameter for Minkowski metric. When p=1, this is equivalent
///     to manhattan distance, and when p=2, equivalent to euclidean distance.
///
/// n_jobs : int, default=1
///     Number of parallel jobs. Use 1 for single-threaded, any other value
///     for parallel execution.
#[pyclass]
pub struct KNeighborsRegressor {
    // Configuration parameters
    n_neighbors: usize,
    weights: String,
    metric: String,
    p: f64,
    n_jobs: i32,

    // Fitted data
    x_train_cache: Option<Array2<f64>>,
    y_train_cache: Option<Array1<f64>>,

    // Metadata
    n_features_in: Option<usize>,
    n_samples_fit: Option<usize>,
}

#[pymethods]
impl KNeighborsRegressor {
    #[new]
    #[pyo3(signature = (*, n_neighbors=5, weights="uniform", metric="euclidean", p=2.0, n_jobs=1))]
    fn new(n_neighbors: usize, weights: &str, metric: &str, p: f64, n_jobs: i32) -> Self {
        KNeighborsRegressor {
            n_neighbors,
            weights: weights.to_string(),
            metric: metric.to_string(),
            p,
            n_jobs,
            x_train_cache: None,
            y_train_cache: None,
            n_features_in: None,
            n_samples_fit: None,
        }
    }

    /// Fit the k-nearest neighbors regressor from the training dataset.
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
    /// self : KNeighborsRegressor
    ///     The fitted estimator.
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        _py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyRefMut<'py, Self> {
        let x_arr = x.as_array();
        let y_arr = y.as_array();

        // Store training data
        slf.x_train_cache = Some(x_arr.to_owned());
        slf.y_train_cache = Some(y_arr.to_owned());
        slf.n_features_in = Some(x_arr.ncols());
        slf.n_samples_fit = Some(x_arr.nrows());

        slf
    }

    /// Predict the target for the provided data.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     Test samples.
    ///
    /// Returns
    /// -------
    /// y : ndarray of shape (n_samples,)
    ///     Target values.
    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let x_train = self
            .x_train_cache
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Not fitted yet"))?;
        let y_train = self
            .y_train_cache
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Not fitted yet"))?;

        let x_test = x.as_array();
        let use_parallel = self.n_jobs != 1;

        // Clone data for thread safety
        let x_train_owned = x_train.clone();
        let y_train_owned = y_train.clone();
        let x_test_owned = x_test.to_owned();
        let k = self.n_neighbors;
        let weights = self.weights.clone();
        let metric = self.metric.clone();
        let p = self.p;

        let predictions = if use_parallel {
            py.allow_threads(move || {
                predict_parallel(
                    x_test_owned.view(),
                    x_train_owned.view(),
                    y_train_owned.view(),
                    k,
                    &weights,
                    &metric,
                    p,
                )
            })
        } else {
            py.allow_threads(move || {
                predict_single(
                    x_test_owned.view(),
                    x_train_owned.view(),
                    y_train_owned.view(),
                    k,
                    &weights,
                    &metric,
                    p,
                )
            })
        };

        let result = Array1::from_vec(predictions);
        Ok(result.into_pyarray(py))
    }

    /// Find the K-neighbors of a point.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     The query points.
    /// n_neighbors : int, optional
    ///     Number of neighbors to get. If not provided, uses the value
    ///     passed to the constructor.
    /// return_distance : bool, default=True
    ///     Whether to return distances.
    ///
    /// Returns
    /// -------
    /// neigh_dist : ndarray of shape (n_samples, n_neighbors)
    ///     Array of distances to the neighbors. Only present if return_distance=True.
    /// neigh_ind : ndarray of shape (n_samples, n_neighbors)
    ///     Indices of the nearest neighbors.
    #[pyo3(signature = (x, n_neighbors=None, return_distance=true))]
    fn kneighbors<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        n_neighbors: Option<usize>,
        return_distance: bool,
    ) -> PyResult<PyObject> {
        let x_train = self
            .x_train_cache
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Not fitted yet"))?;

        let x_test = x.as_array();
        let k = n_neighbors.unwrap_or(self.n_neighbors);
        let use_parallel = self.n_jobs != 1;

        let x_train_owned = x_train.clone();
        let x_test_owned = x_test.to_owned();
        let metric = self.metric.clone();
        let p = self.p;

        let (distances, indices) = if use_parallel {
            py.allow_threads(move || {
                kneighbors_parallel(x_test_owned.view(), x_train_owned.view(), k, &metric, p)
            })
        } else {
            py.allow_threads(move || {
                kneighbors_single(x_test_owned.view(), x_train_owned.view(), k, &metric, p)
            })
        };

        // Convert indices to i64 for numpy compatibility
        let indices_i64: Array2<i64> = indices.mapv(|x| x as i64);

        if return_distance {
            let dist_py = distances.into_pyarray(py);
            let idx_py = indices_i64.into_pyarray(py);
            Ok((dist_py, idx_py).into_pyobject(py)?.into_any().unbind())
        } else {
            let idx_py = indices_i64.into_pyarray(py);
            Ok(idx_py.into_any().unbind())
        }
    }

    /// Return the coefficient of determination (R^2) of the prediction.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     Test samples.
    /// y : array-like of shape (n_samples,)
    ///     True values for X.
    ///
    /// Returns
    /// -------
    /// score : float
    ///     R^2 of self.predict(X) w.r.t. y.
    fn score<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<f64> {
        let y_true = y.as_array();
        let y_pred_arr = self.predict(py, x)?;
        let y_pred = unsafe { y_pred_arr.as_array() };

        // Compute R^2 = 1 - SS_res / SS_tot
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
            // All y_true values are the same
            if ss_res == 0.0 {
                Ok(1.0)
            } else {
                Ok(0.0)
            }
        } else {
            Ok(1.0 - ss_res / ss_tot)
        }
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

    /// The number of neighbors to use.
    #[getter]
    fn n_neighbors_(&self) -> usize {
        self.n_neighbors
    }
}
