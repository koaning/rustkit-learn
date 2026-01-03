use ndarray::{Array1, Array2, Axis};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::utils::stats::{compute_mean_var_parallel, compute_mean_var_single, OnlineStats};

/// Standardize features by removing the mean and scaling to unit variance.
///
/// The standard score of a sample x is calculated as:
///     z = (x - mean) / std
///
/// This scaler can be used incrementally via `partial_fit` for streaming data.
#[pyclass]
pub struct StandardScaler {
    // Internal state for incremental learning
    online_stats: Option<OnlineStats>,

    // Cached computed values (for transform)
    mean_cache: Option<Array1<f64>>,
    scale_cache: Option<Array1<f64>>,
    var_cache: Option<Array1<f64>>,

    // Configuration
    with_mean: bool,
    with_std: bool,
    n_jobs: i32,
}

#[pymethods]
impl StandardScaler {
    #[new]
    #[pyo3(signature = (*, with_mean=true, with_std=true, n_jobs=1))]
    fn new(with_mean: bool, with_std: bool, n_jobs: i32) -> Self {
        StandardScaler {
            online_stats: None,
            mean_cache: None,
            scale_cache: None,
            var_cache: None,
            with_mean,
            with_std,
            n_jobs,
        }
    }

    /// Compute the mean and std to be used for later scaling.
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyRefMut<'py, Self> {
        let arr = x.as_array();
        let n_features = arr.ncols();
        let n_samples = arr.nrows();
        let use_parallel = slf.n_jobs != 1;

        // Reset state for fresh fit
        slf.online_stats = Some(OnlineStats::new(n_features));

        // Copy data to owned array for computation
        let owned_arr = arr.to_owned();

        // Use parallel or single-threaded computation based on n_jobs
        let (mean, var) = if use_parallel {
            py.allow_threads(move || compute_mean_var_parallel(owned_arr.view()))
        } else {
            py.allow_threads(move || compute_mean_var_single(owned_arr.view()))
        };

        // Update online stats to reflect the full fit
        if let Some(ref mut stats) = slf.online_stats {
            stats.mean = mean.clone();
            stats.m2 = &var * (n_samples as f64);
            stats.count = n_samples;
        }

        // Cache computed values
        slf.mean_cache = Some(mean);
        slf.var_cache = Some(var.clone());

        // Compute scale (std), handling zero variance
        let scale = var.mapv(|v| {
            let s = v.sqrt();
            if s == 0.0 {
                1.0
            } else {
                s
            }
        });
        slf.scale_cache = Some(scale);

        slf
    }

    /// Online computation of mean and std with a new batch of data.
    fn partial_fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyRefMut<'py, Self> {
        let arr = x.as_array();
        let n_features = arr.ncols();

        // Initialize if first call
        if slf.online_stats.is_none() {
            slf.online_stats = Some(OnlineStats::new(n_features));
        }

        // Copy data to owned array
        let owned_arr = arr.to_owned();

        // Get mutable reference to stats and update
        if let Some(ref mut stats) = slf.online_stats {
            // For partial_fit, we update incrementally (not parallel for correctness)
            let _ = py; // Keep py in scope but don't use allow_threads for incremental update
            stats.update(owned_arr.view());
        }

        // Update cached values - extract values first to avoid borrow issues
        let (mean, var) = if let Some(ref stats) = slf.online_stats {
            (stats.mean.clone(), stats.variance())
        } else {
            return slf;
        };

        slf.mean_cache = Some(mean);
        slf.var_cache = Some(var.clone());

        let scale = var.mapv(|v| {
            let s = v.sqrt();
            if s == 0.0 {
                1.0
            } else {
                s
            }
        });
        slf.scale_cache = Some(scale);

        slf
    }

    /// Perform standardization by centering and scaling.
    fn transform<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let mean = self
            .mean_cache
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Not fitted yet"))?
            .clone();
        let scale = self
            .scale_cache
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Not fitted yet"))?
            .clone();

        let arr = x.as_array();
        let with_mean = self.with_mean;
        let with_std = self.with_std;
        let use_parallel = self.n_jobs != 1;

        // Copy to owned array for processing
        let owned_arr = arr.to_owned();
        let n_samples = owned_arr.nrows();
        let n_features = owned_arr.ncols();

        let result = py.allow_threads(move || {
            let mut output = Array2::zeros((n_samples, n_features));

            if use_parallel {
                // Parallel transform across rows
                output
                    .axis_iter_mut(Axis(0))
                    .into_par_iter()
                    .enumerate()
                    .for_each(|(i, mut row)| {
                        for j in 0..n_features {
                            let mut val = owned_arr[[i, j]];
                            if with_mean {
                                val -= mean[j];
                            }
                            if with_std {
                                val /= scale[j];
                            }
                            row[j] = val;
                        }
                    });
            } else {
                // Single-threaded transform
                for i in 0..n_samples {
                    for j in 0..n_features {
                        let mut val = owned_arr[[i, j]];
                        if with_mean {
                            val -= mean[j];
                        }
                        if with_std {
                            val /= scale[j];
                        }
                        output[[i, j]] = val;
                    }
                }
            }

            output
        });

        Ok(result.into_pyarray(py))
    }

    /// Fit and transform in one step.
    fn fit_transform<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        // First fit
        let arr = x.as_array();
        let n_features = arr.ncols();
        let n_samples = arr.nrows();
        let use_parallel = slf.n_jobs != 1;

        slf.online_stats = Some(OnlineStats::new(n_features));

        let owned_arr = arr.to_owned();

        let (mean, var) = if use_parallel {
            py.allow_threads(|| compute_mean_var_parallel(owned_arr.view()))
        } else {
            py.allow_threads(|| compute_mean_var_single(owned_arr.view()))
        };

        if let Some(ref mut stats) = slf.online_stats {
            stats.mean = mean.clone();
            stats.m2 = &var * (n_samples as f64);
            stats.count = n_samples;
        }

        slf.mean_cache = Some(mean.clone());
        slf.var_cache = Some(var.clone());

        let scale = var.mapv(|v| {
            let s = v.sqrt();
            if s == 0.0 {
                1.0
            } else {
                s
            }
        });
        slf.scale_cache = Some(scale.clone());

        // Now transform
        let with_mean = slf.with_mean;
        let with_std = slf.with_std;

        let result = py.allow_threads(move || {
            let mut output = Array2::zeros((n_samples, n_features));

            if use_parallel {
                output
                    .axis_iter_mut(Axis(0))
                    .into_par_iter()
                    .enumerate()
                    .for_each(|(i, mut row)| {
                        for j in 0..n_features {
                            let mut val = owned_arr[[i, j]];
                            if with_mean {
                                val -= mean[j];
                            }
                            if with_std {
                                val /= scale[j];
                            }
                            row[j] = val;
                        }
                    });
            } else {
                for i in 0..n_samples {
                    for j in 0..n_features {
                        let mut val = owned_arr[[i, j]];
                        if with_mean {
                            val -= mean[j];
                        }
                        if with_std {
                            val /= scale[j];
                        }
                        output[[i, j]] = val;
                    }
                }
            }

            output
        });

        Ok(result.into_pyarray(py))
    }

    /// Scale back the data to the original representation.
    fn inverse_transform<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let mean = self
            .mean_cache
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Not fitted yet"))?
            .clone();
        let scale = self
            .scale_cache
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Not fitted yet"))?
            .clone();

        let arr = x.as_array();
        let with_mean = self.with_mean;
        let with_std = self.with_std;
        let use_parallel = self.n_jobs != 1;

        let owned_arr = arr.to_owned();
        let n_samples = owned_arr.nrows();
        let n_features = owned_arr.ncols();

        let result = py.allow_threads(move || {
            let mut output = Array2::zeros((n_samples, n_features));

            if use_parallel {
                output
                    .axis_iter_mut(Axis(0))
                    .into_par_iter()
                    .enumerate()
                    .for_each(|(i, mut row)| {
                        for j in 0..n_features {
                            let mut val = owned_arr[[i, j]];
                            if with_std {
                                val *= scale[j];
                            }
                            if with_mean {
                                val += mean[j];
                            }
                            row[j] = val;
                        }
                    });
            } else {
                for i in 0..n_samples {
                    for j in 0..n_features {
                        let mut val = owned_arr[[i, j]];
                        if with_std {
                            val *= scale[j];
                        }
                        if with_mean {
                            val += mean[j];
                        }
                        output[[i, j]] = val;
                    }
                }
            }

            output
        });

        Ok(result.into_pyarray(py))
    }

    /// Per-feature mean, computed during fit.
    #[getter]
    fn mean_<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
        Ok(self.mean_cache.as_ref().map(|m| m.clone().into_pyarray(py)))
    }

    /// Per-feature scale (standard deviation), computed during fit.
    #[getter]
    fn scale_<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
        Ok(self
            .scale_cache
            .as_ref()
            .map(|s| s.clone().into_pyarray(py)))
    }

    /// Per-feature variance, computed during fit.
    #[getter]
    fn var_<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
        Ok(self.var_cache.as_ref().map(|v| v.clone().into_pyarray(py)))
    }

    /// Number of features seen during fit.
    #[getter]
    fn n_features_in_(&self) -> Option<usize> {
        self.mean_cache.as_ref().map(|m| m.len())
    }

    /// Number of samples seen during fit.
    #[getter]
    fn n_samples_seen_(&self) -> usize {
        self.online_stats.as_ref().map(|s| s.count).unwrap_or(0)
    }
}
