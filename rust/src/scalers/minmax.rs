use ndarray::{Array1, Array2, Axis};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::utils::stats::{compute_min_max_parallel, compute_min_max_single};

/// Transform features by scaling each feature to a given range.
///
/// This estimator scales and translates each feature individually such
/// that it is in the given range on the training set, e.g. between zero and one.
///
/// The transformation is given by:
///     X_std = (X - X_min) / (X_max - X_min)
///     X_scaled = X_std * (max - min) + min
///
/// This scaler can be used incrementally via `partial_fit` for streaming data.
#[pyclass]
pub struct MinMaxScaler {
    // Learned parameters
    data_min_cache: Option<Array1<f64>>,
    data_max_cache: Option<Array1<f64>>,
    data_range_cache: Option<Array1<f64>>,

    // Computed scale and min for transform
    scale_cache: Option<Array1<f64>>,
    min_cache: Option<Array1<f64>>,

    // Configuration
    feature_range: (f64, f64),
    clip: bool,
    n_jobs: i32,

    // Sample count for tracking
    n_samples_seen: usize,
}

#[pymethods]
impl MinMaxScaler {
    #[new]
    #[pyo3(signature = (feature_range=(0.0, 1.0), *, clip=false, n_jobs=1))]
    fn new(feature_range: (f64, f64), clip: bool, n_jobs: i32) -> Self {
        MinMaxScaler {
            data_min_cache: None,
            data_max_cache: None,
            data_range_cache: None,
            scale_cache: None,
            min_cache: None,
            feature_range,
            clip,
            n_jobs,
            n_samples_seen: 0,
        }
    }

    /// Compute the minimum and maximum to be used for later scaling.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     The data used to compute the per-feature minimum and maximum.
    ///
    /// Returns
    /// -------
    /// self : MinMaxScaler
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyRefMut<'py, Self> {
        let arr = x.as_array();
        let use_parallel = slf.n_jobs != 1;

        // Reset state for fresh fit
        slf.n_samples_seen = arr.nrows();

        // Copy data to owned array for computation
        let owned_arr = arr.to_owned();

        // Use parallel or single-threaded computation based on n_jobs
        let (data_min, data_max) = if use_parallel {
            py.allow_threads(move || compute_min_max_parallel(owned_arr.view()))
        } else {
            py.allow_threads(move || compute_min_max_single(owned_arr.view()))
        };

        slf.update_computed_values(data_min, data_max);

        slf
    }

    /// Online computation of min and max with a new batch of data.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     The data used to compute the per-feature minimum and maximum.
    ///
    /// Returns
    /// -------
    /// self : MinMaxScaler
    fn partial_fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyRefMut<'py, Self> {
        let arr = x.as_array();
        let use_parallel = slf.n_jobs != 1;

        // Copy data to owned array for computation
        let owned_arr = arr.to_owned();

        // Compute min/max for this batch
        let (batch_min, batch_max) = if use_parallel {
            py.allow_threads(move || compute_min_max_parallel(owned_arr.view()))
        } else {
            py.allow_threads(move || compute_min_max_single(owned_arr.view()))
        };

        slf.n_samples_seen += arr.nrows();

        // Update global min/max
        let (data_min, data_max) = if let (Some(existing_min), Some(existing_max)) =
            (&slf.data_min_cache, &slf.data_max_cache)
        {
            // Merge with existing statistics
            let new_min = existing_min
                .iter()
                .zip(batch_min.iter())
                .map(|(&a, &b)| a.min(b))
                .collect::<Vec<_>>();
            let new_max = existing_max
                .iter()
                .zip(batch_max.iter())
                .map(|(&a, &b)| a.max(b))
                .collect::<Vec<_>>();
            (Array1::from_vec(new_min), Array1::from_vec(new_max))
        } else {
            // First batch
            (batch_min, batch_max)
        };

        slf.update_computed_values(data_min, data_max);

        slf
    }

    /// Scale features according to feature_range.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     The data to transform.
    ///
    /// Returns
    /// -------
    /// X_tr : ndarray of shape (n_samples, n_features)
    ///     Transformed array.
    fn transform<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let scale = self
            .scale_cache
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Not fitted yet"))?
            .clone();
        let min = self
            .min_cache
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Not fitted yet"))?
            .clone();

        let arr = x.as_array();
        let clip = self.clip;
        let (feature_min, feature_max) = self.feature_range;
        let use_parallel = self.n_jobs != 1;

        // Copy to owned array for processing
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
                            // X_scaled = X * scale + min (sklearn formula)
                            let mut val = owned_arr[[i, j]] * scale[j] + min[j];

                            if clip {
                                val = val.clamp(feature_min, feature_max);
                            }

                            row[j] = val;
                        }
                    });
            } else {
                for i in 0..n_samples {
                    for j in 0..n_features {
                        let mut val = owned_arr[[i, j]] * scale[j] + min[j];

                        if clip {
                            val = val.clamp(feature_min, feature_max);
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
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     The data to fit and transform.
    ///
    /// Returns
    /// -------
    /// X_tr : ndarray of shape (n_samples, n_features)
    ///     Transformed array.
    fn fit_transform<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let arr = x.as_array();
        let n_samples = arr.nrows();
        let n_features = arr.ncols();
        let use_parallel = slf.n_jobs != 1;

        slf.n_samples_seen = n_samples;

        let owned_arr = arr.to_owned();

        let (data_min, data_max) = if use_parallel {
            py.allow_threads(|| compute_min_max_parallel(owned_arr.view()))
        } else {
            py.allow_threads(|| compute_min_max_single(owned_arr.view()))
        };

        // Compute derived values
        let data_range: Array1<f64> = &data_max - &data_min;
        let (feature_min, feature_max) = slf.feature_range;
        let feature_range = feature_max - feature_min;

        let scale: Array1<f64> = data_range.mapv(|r| {
            if r == 0.0 {
                0.0
            } else {
                feature_range / r
            }
        });

        let min: Array1<f64> = data_min
            .iter()
            .zip(scale.iter())
            .map(|(&dm, &s)| feature_min - dm * s)
            .collect::<Vec<_>>()
            .into();

        slf.data_min_cache = Some(data_min.clone());
        slf.data_max_cache = Some(data_max);
        slf.data_range_cache = Some(data_range);
        slf.scale_cache = Some(scale.clone());
        slf.min_cache = Some(min.clone());

        // Now transform
        let clip = slf.clip;

        let result = py.allow_threads(move || {
            let mut output = Array2::zeros((n_samples, n_features));

            if use_parallel {
                output
                    .axis_iter_mut(Axis(0))
                    .into_par_iter()
                    .enumerate()
                    .for_each(|(i, mut row)| {
                        for j in 0..n_features {
                            // X_scaled = X * scale + min (sklearn formula)
                            let mut val = owned_arr[[i, j]] * scale[j] + min[j];

                            if clip {
                                val = val.clamp(feature_min, feature_max);
                            }

                            row[j] = val;
                        }
                    });
            } else {
                for i in 0..n_samples {
                    for j in 0..n_features {
                        let mut val = owned_arr[[i, j]] * scale[j] + min[j];

                        if clip {
                            val = val.clamp(feature_min, feature_max);
                        }

                        output[[i, j]] = val;
                    }
                }
            }

            output
        });

        Ok(result.into_pyarray(py))
    }

    /// Undo the scaling of X according to feature_range.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     The transformed data.
    ///
    /// Returns
    /// -------
    /// X_tr : ndarray of shape (n_samples, n_features)
    ///     Original data.
    fn inverse_transform<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let scale = self
            .scale_cache
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Not fitted yet"))?
            .clone();
        let min = self
            .min_cache
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Not fitted yet"))?
            .clone();
        let data_min = self
            .data_min_cache
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Not fitted yet"))?
            .clone();

        let arr = x.as_array();
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
                            // X = (X_scaled - min) / scale (sklearn formula)
                            // Handle zero scale (constant features)
                            let val = if scale[j] == 0.0 {
                                data_min[j]
                            } else {
                                (owned_arr[[i, j]] - min[j]) / scale[j]
                            };
                            row[j] = val;
                        }
                    });
            } else {
                for i in 0..n_samples {
                    for j in 0..n_features {
                        let val = if scale[j] == 0.0 {
                            data_min[j]
                        } else {
                            (owned_arr[[i, j]] - min[j]) / scale[j]
                        };
                        output[[i, j]] = val;
                    }
                }
            }

            output
        });

        Ok(result.into_pyarray(py))
    }

    /// Per feature adjustment for minimum.
    #[getter]
    fn min_<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
        Ok(self.min_cache.as_ref().map(|m| m.clone().into_pyarray(py)))
    }

    /// Per feature relative scaling of the data.
    #[getter]
    fn scale_<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
        Ok(self
            .scale_cache
            .as_ref()
            .map(|s| s.clone().into_pyarray(py)))
    }

    /// Per feature minimum seen in the data.
    #[getter]
    fn data_min_<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
        Ok(self
            .data_min_cache
            .as_ref()
            .map(|m| m.clone().into_pyarray(py)))
    }

    /// Per feature maximum seen in the data.
    #[getter]
    fn data_max_<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
        Ok(self
            .data_max_cache
            .as_ref()
            .map(|m| m.clone().into_pyarray(py)))
    }

    /// Per feature range (data_max - data_min) seen in the data.
    #[getter]
    fn data_range_<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
        Ok(self
            .data_range_cache
            .as_ref()
            .map(|r| r.clone().into_pyarray(py)))
    }

    /// Number of features seen during fit.
    #[getter]
    fn n_features_in_(&self) -> Option<usize> {
        self.data_min_cache.as_ref().map(|m| m.len())
    }

    /// Number of samples seen during fit.
    #[getter]
    fn n_samples_seen_(&self) -> usize {
        self.n_samples_seen
    }
}

impl MinMaxScaler {
    fn update_computed_values(&mut self, data_min: Array1<f64>, data_max: Array1<f64>) {
        // Compute data range
        let data_range: Array1<f64> = &data_max - &data_min;

        // Compute scale: (feature_max - feature_min) / (data_max - data_min)
        let (feature_min, feature_max) = self.feature_range;
        let feature_range = feature_max - feature_min;

        let scale: Array1<f64> = data_range.mapv(|r| {
            if r == 0.0 {
                0.0 // Handle constant features
            } else {
                feature_range / r
            }
        });

        // Compute min: feature_min - data_min * scale
        let min: Array1<f64> = data_min
            .iter()
            .zip(scale.iter())
            .map(|(&dm, &s)| feature_min - dm * s)
            .collect::<Vec<_>>()
            .into();

        self.data_min_cache = Some(data_min);
        self.data_max_cache = Some(data_max);
        self.data_range_cache = Some(data_range);
        self.scale_cache = Some(scale);
        self.min_cache = Some(min);
    }
}
