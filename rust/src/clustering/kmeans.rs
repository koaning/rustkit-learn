use ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

use super::kmeans_utils::{
    assign_clusters_parallel, assign_clusters_single, centroid_shift, compute_centroids_parallel,
    compute_centroids_single, compute_distances_parallel, compute_distances_single,
    compute_inertia, initialize_centroids_kmeans_plusplus, initialize_centroids_random, SimpleRng,
};

/// K-Means clustering.
///
/// Parameters
/// ----------
/// n_clusters : int, default=8
///     The number of clusters to form as well as the number of centroids to generate.
///
/// init : str, default='k-means++'
///     Method for initialization:
///     - 'k-means++': selects initial cluster centers using a smart algorithm
///     - 'random': choose k observations at random as initial centroids
///
/// n_init : int or None, default=None
///     Number of times the k-means algorithm will run with different centroid seeds.
///     None translates to 1 for 'k-means++' and 10 for 'random'.
///
/// max_iter : int, default=300
///     Maximum number of iterations of the k-means algorithm for a single run.
///
/// tol : float, default=1e-4
///     Relative tolerance to declare convergence based on Frobenius norm of centroid changes.
///
/// random_state : int or None, default=None
///     Seed for random number generation for reproducibility.
///
/// n_jobs : int, default=1
///     Number of parallel jobs. Use 1 for single-threaded, any other value
///     for parallel execution.
#[pyclass]
pub struct KMeans {
    // Configuration parameters
    n_clusters: usize,
    init: String,
    n_init: usize,
    max_iter: usize,
    tol: f64,
    random_state: Option<u64>,
    n_jobs: i32,

    // Fitted state
    cluster_centers: Option<Array2<f64>>,
    labels: Option<Array1<i64>>,
    inertia: Option<f64>,
    n_iter: Option<usize>,
    n_features_in: Option<usize>,
}

#[pymethods]
impl KMeans {
    #[new]
    #[pyo3(signature = (
        *,
        n_clusters=8,
        init="k-means++",
        n_init=None,
        max_iter=300,
        tol=1e-4,
        random_state=None,
        n_jobs=1
    ))]
    fn new(
        n_clusters: usize,
        init: &str,
        n_init: Option<usize>,
        max_iter: usize,
        tol: f64,
        random_state: Option<u64>,
        n_jobs: i32,
    ) -> PyResult<Self> {
        // Validate init parameter
        if init != "k-means++" && init != "random" {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "init must be 'k-means++' or 'random'",
            ));
        }

        // Resolve n_init: default is 1 for k-means++, 10 for random
        let resolved_n_init = n_init.unwrap_or(if init == "k-means++" { 1 } else { 10 });

        Ok(KMeans {
            n_clusters,
            init: init.to_string(),
            n_init: resolved_n_init,
            max_iter,
            tol,
            random_state,
            n_jobs,
            cluster_centers: None,
            labels: None,
            inertia: None,
            n_iter: None,
            n_features_in: None,
        })
    }

    /// Compute k-means clustering.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     Training instances to cluster.
    ///
    /// Returns
    /// -------
    /// self : KMeans
    ///     Fitted estimator.
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let arr = x.as_array();
        let n_samples = arr.nrows();
        let n_features = arr.ncols();

        // Validate input
        if n_samples < slf.n_clusters {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "n_samples={} should be >= n_clusters={}",
                n_samples, slf.n_clusters
            )));
        }

        slf.n_features_in = Some(n_features);

        // Copy to owned array for thread safety
        let x_owned = arr.to_owned();
        let n_clusters = slf.n_clusters;
        let init = slf.init.clone();
        let n_init = slf.n_init;
        let max_iter = slf.max_iter;
        let random_state = slf.random_state;
        let use_parallel = slf.n_jobs != 1;

        // Compute scaled tolerance (same as sklearn: tol * mean(var(X, axis=0)))
        // This makes the tolerance relative to the data scale
        let mean_var = {
            let n = n_samples as f64;
            let mut total_var = 0.0;
            for col in 0..n_features {
                let col_data = x_owned.column(col);
                let mean: f64 = col_data.iter().sum::<f64>() / n;
                let var: f64 = col_data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
                total_var += var;
            }
            total_var / n_features as f64
        };
        let tol = slf.tol * mean_var;

        // Run k-means n_init times and keep best result
        let (best_centroids, best_labels, best_inertia, best_n_iter) =
            py.detach(move || {
                let mut best_centroids: Option<Array2<f64>> = None;
                let mut best_labels: Option<Array1<i64>> = None;
                let mut best_inertia = f64::MAX;
                let mut best_n_iter = 0usize;

                for init_run in 0..n_init {
                    // Create RNG with seed derived from random_state and init_run
                    let seed = random_state.unwrap_or(0).wrapping_add(init_run as u64);
                    let mut rng = SimpleRng::new(seed);

                    // Initialize centroids
                    let mut centroids = if init == "k-means++" {
                        initialize_centroids_kmeans_plusplus(x_owned.view(), n_clusters, &mut rng)
                    } else {
                        initialize_centroids_random(x_owned.view(), n_clusters, &mut rng)
                    };

                    let mut n_iter = 0usize;

                    // Lloyd's algorithm main loop
                    for iter in 0..max_iter {
                        n_iter = iter + 1;

                        // E-step: Assign samples to nearest centroid
                        let (labels, _distances) = if use_parallel {
                            assign_clusters_parallel(x_owned.view(), centroids.view())
                        } else {
                            assign_clusters_single(x_owned.view(), centroids.view())
                        };

                        // M-step: Update centroids
                        let new_centroids = if use_parallel {
                            compute_centroids_parallel(x_owned.view(), labels.view(), n_clusters)
                        } else {
                            compute_centroids_single(x_owned.view(), labels.view(), n_clusters)
                        };

                        // Check convergence
                        let shift = centroid_shift(centroids.view(), new_centroids.view());
                        centroids = new_centroids;

                        if shift <= tol {
                            break;
                        }
                    }

                    // Final assignment for this run
                    let (final_labels, final_distances) = if use_parallel {
                        assign_clusters_parallel(x_owned.view(), centroids.view())
                    } else {
                        assign_clusters_single(x_owned.view(), centroids.view())
                    };
                    let inertia = compute_inertia(final_distances.view());

                    // Keep best result
                    if inertia < best_inertia {
                        best_inertia = inertia;
                        best_centroids = Some(centroids);
                        best_labels = Some(final_labels);
                        best_n_iter = n_iter;
                    }
                }

                (
                    best_centroids.unwrap(),
                    best_labels.unwrap(),
                    best_inertia,
                    best_n_iter,
                )
            });

        // Store fitted state
        slf.cluster_centers = Some(best_centroids);
        slf.labels = Some(best_labels);
        slf.inertia = Some(best_inertia);
        slf.n_iter = Some(best_n_iter);

        Ok(slf)
    }

    /// Predict the closest cluster each sample in X belongs to.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     New data to predict.
    ///
    /// Returns
    /// -------
    /// labels : ndarray of shape (n_samples,)
    ///     Index of the cluster each sample belongs to.
    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let centroids = self
            .cluster_centers
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Not fitted yet"))?;

        let arr = x.as_array();
        let x_owned = arr.to_owned();
        let centroids_owned = centroids.clone();
        let use_parallel = self.n_jobs != 1;

        let labels = py.detach(move || {
            let (labels, _distances) = if use_parallel {
                assign_clusters_parallel(x_owned.view(), centroids_owned.view())
            } else {
                assign_clusters_single(x_owned.view(), centroids_owned.view())
            };
            labels
        });

        Ok(labels.into_pyarray(py))
    }

    /// Compute clustering and return labels.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     Training instances to cluster.
    ///
    /// Returns
    /// -------
    /// labels : ndarray of shape (n_samples,)
    ///     Index of the cluster each sample belongs to.
    fn fit_predict<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        // Fit the model
        let arr = x.as_array();
        let n_samples = arr.nrows();
        let n_features = arr.ncols();

        if n_samples < slf.n_clusters {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "n_samples={} should be >= n_clusters={}",
                n_samples, slf.n_clusters
            )));
        }

        slf.n_features_in = Some(n_features);

        let x_owned = arr.to_owned();
        let n_clusters = slf.n_clusters;
        let init = slf.init.clone();
        let n_init = slf.n_init;
        let max_iter = slf.max_iter;
        let random_state = slf.random_state;
        let use_parallel = slf.n_jobs != 1;

        // Compute scaled tolerance (same as sklearn: tol * mean(var(X, axis=0)))
        let mean_var = {
            let n = n_samples as f64;
            let mut total_var = 0.0;
            for col in 0..n_features {
                let col_data = x_owned.column(col);
                let mean: f64 = col_data.iter().sum::<f64>() / n;
                let var: f64 = col_data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
                total_var += var;
            }
            total_var / n_features as f64
        };
        let tol = slf.tol * mean_var;

        let (best_centroids, best_labels, best_inertia, best_n_iter) =
            py.detach(move || {
                let mut best_centroids: Option<Array2<f64>> = None;
                let mut best_labels: Option<Array1<i64>> = None;
                let mut best_inertia = f64::MAX;
                let mut best_n_iter = 0usize;

                for init_run in 0..n_init {
                    let seed = random_state.unwrap_or(0).wrapping_add(init_run as u64);
                    let mut rng = SimpleRng::new(seed);

                    let mut centroids = if init == "k-means++" {
                        initialize_centroids_kmeans_plusplus(x_owned.view(), n_clusters, &mut rng)
                    } else {
                        initialize_centroids_random(x_owned.view(), n_clusters, &mut rng)
                    };

                    let mut n_iter = 0usize;

                    for iter in 0..max_iter {
                        n_iter = iter + 1;

                        let (labels, _distances) = if use_parallel {
                            assign_clusters_parallel(x_owned.view(), centroids.view())
                        } else {
                            assign_clusters_single(x_owned.view(), centroids.view())
                        };

                        let new_centroids = if use_parallel {
                            compute_centroids_parallel(x_owned.view(), labels.view(), n_clusters)
                        } else {
                            compute_centroids_single(x_owned.view(), labels.view(), n_clusters)
                        };

                        let shift = centroid_shift(centroids.view(), new_centroids.view());
                        centroids = new_centroids;

                        if shift <= tol {
                            break;
                        }
                    }

                    let (final_labels, final_distances) = if use_parallel {
                        assign_clusters_parallel(x_owned.view(), centroids.view())
                    } else {
                        assign_clusters_single(x_owned.view(), centroids.view())
                    };
                    let inertia = compute_inertia(final_distances.view());

                    if inertia < best_inertia {
                        best_inertia = inertia;
                        best_centroids = Some(centroids);
                        best_labels = Some(final_labels);
                        best_n_iter = n_iter;
                    }
                }

                (
                    best_centroids.unwrap(),
                    best_labels.unwrap(),
                    best_inertia,
                    best_n_iter,
                )
            });

        let labels_clone = best_labels.clone();

        slf.cluster_centers = Some(best_centroids);
        slf.labels = Some(best_labels);
        slf.inertia = Some(best_inertia);
        slf.n_iter = Some(best_n_iter);

        Ok(labels_clone.into_pyarray(py))
    }

    /// Transform X to a cluster-distance space.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     New data to transform.
    ///
    /// Returns
    /// -------
    /// X_new : ndarray of shape (n_samples, n_clusters)
    ///     X transformed to cluster-distance space.
    fn transform<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let centroids = self
            .cluster_centers
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Not fitted yet"))?;

        let arr = x.as_array();
        let x_owned = arr.to_owned();
        let centroids_owned = centroids.clone();
        let use_parallel = self.n_jobs != 1;

        let distances = py.detach(move || {
            if use_parallel {
                compute_distances_parallel(x_owned.view(), centroids_owned.view())
            } else {
                compute_distances_single(x_owned.view(), centroids_owned.view())
            }
        });

        Ok(distances.into_pyarray(py))
    }

    /// Fit and transform in one step.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     New data to fit and transform.
    ///
    /// Returns
    /// -------
    /// X_new : ndarray of shape (n_samples, n_clusters)
    ///     X transformed to cluster-distance space.
    fn fit_transform<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        // Fit the model first
        let arr = x.as_array();
        let n_samples = arr.nrows();
        let n_features = arr.ncols();

        if n_samples < slf.n_clusters {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "n_samples={} should be >= n_clusters={}",
                n_samples, slf.n_clusters
            )));
        }

        slf.n_features_in = Some(n_features);

        let x_owned = arr.to_owned();
        let n_clusters = slf.n_clusters;
        let init = slf.init.clone();
        let n_init = slf.n_init;
        let max_iter = slf.max_iter;
        let random_state = slf.random_state;
        let use_parallel = slf.n_jobs != 1;

        // Compute scaled tolerance (same as sklearn: tol * mean(var(X, axis=0)))
        let mean_var = {
            let n = n_samples as f64;
            let mut total_var = 0.0;
            for col in 0..n_features {
                let col_data = x_owned.column(col);
                let mean: f64 = col_data.iter().sum::<f64>() / n;
                let var: f64 = col_data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
                total_var += var;
            }
            total_var / n_features as f64
        };
        let tol = slf.tol * mean_var;

        let (best_centroids, best_labels, best_inertia, best_n_iter, distances) =
            py.detach(move || {
                let mut best_centroids: Option<Array2<f64>> = None;
                let mut best_labels: Option<Array1<i64>> = None;
                let mut best_inertia = f64::MAX;
                let mut best_n_iter = 0usize;

                for init_run in 0..n_init {
                    let seed = random_state.unwrap_or(0).wrapping_add(init_run as u64);
                    let mut rng = SimpleRng::new(seed);

                    let mut centroids = if init == "k-means++" {
                        initialize_centroids_kmeans_plusplus(x_owned.view(), n_clusters, &mut rng)
                    } else {
                        initialize_centroids_random(x_owned.view(), n_clusters, &mut rng)
                    };

                    let mut n_iter = 0usize;

                    for iter in 0..max_iter {
                        n_iter = iter + 1;

                        let (labels, _distances) = if use_parallel {
                            assign_clusters_parallel(x_owned.view(), centroids.view())
                        } else {
                            assign_clusters_single(x_owned.view(), centroids.view())
                        };

                        let new_centroids = if use_parallel {
                            compute_centroids_parallel(x_owned.view(), labels.view(), n_clusters)
                        } else {
                            compute_centroids_single(x_owned.view(), labels.view(), n_clusters)
                        };

                        let shift = centroid_shift(centroids.view(), new_centroids.view());
                        centroids = new_centroids;

                        if shift <= tol {
                            break;
                        }
                    }

                    let (final_labels, final_distances) = if use_parallel {
                        assign_clusters_parallel(x_owned.view(), centroids.view())
                    } else {
                        assign_clusters_single(x_owned.view(), centroids.view())
                    };
                    let inertia = compute_inertia(final_distances.view());

                    if inertia < best_inertia {
                        best_inertia = inertia;
                        best_centroids = Some(centroids);
                        best_labels = Some(final_labels);
                        best_n_iter = n_iter;
                    }
                }

                // Compute distances to best centroids
                let best_centroids_ref = best_centroids.as_ref().unwrap();
                let distances = if use_parallel {
                    compute_distances_parallel(x_owned.view(), best_centroids_ref.view())
                } else {
                    compute_distances_single(x_owned.view(), best_centroids_ref.view())
                };

                (
                    best_centroids.unwrap(),
                    best_labels.unwrap(),
                    best_inertia,
                    best_n_iter,
                    distances,
                )
            });

        slf.cluster_centers = Some(best_centroids);
        slf.labels = Some(best_labels);
        slf.inertia = Some(best_inertia);
        slf.n_iter = Some(best_n_iter);

        Ok(distances.into_pyarray(py))
    }

    /// Opposite of the value of X on the K-means objective (for sklearn compatibility).
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     New data.
    ///
    /// Returns
    /// -------
    /// score : float
    ///     Negative inertia.
    fn score<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<'py, f64>) -> PyResult<f64> {
        let centroids = self
            .cluster_centers
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Not fitted yet"))?;

        let arr = x.as_array();
        let x_owned = arr.to_owned();
        let centroids_owned = centroids.clone();
        let use_parallel = self.n_jobs != 1;

        let inertia = py.detach(move || {
            let (_labels, distances) = if use_parallel {
                assign_clusters_parallel(x_owned.view(), centroids_owned.view())
            } else {
                assign_clusters_single(x_owned.view(), centroids_owned.view())
            };
            compute_inertia(distances.view())
        });

        Ok(-inertia)
    }

    // Getters for fitted attributes

    #[getter]
    fn cluster_centers_<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<Bound<'py, PyArray2<f64>>>> {
        Ok(self
            .cluster_centers
            .as_ref()
            .map(|c| c.clone().into_pyarray(py)))
    }

    #[getter]
    fn labels_<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray1<i64>>>> {
        Ok(self.labels.as_ref().map(|l| l.clone().into_pyarray(py)))
    }

    #[getter]
    fn inertia_(&self) -> Option<f64> {
        self.inertia
    }

    #[getter]
    fn n_iter_(&self) -> Option<usize> {
        self.n_iter
    }

    #[getter]
    fn n_features_in_(&self) -> Option<usize> {
        self.n_features_in
    }
}
