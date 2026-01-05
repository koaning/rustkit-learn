#![allow(deprecated)]

use std::collections::HashMap;

use ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use super::ball_tree::BallTree;
use super::kd_tree::KDTree;
use super::knn_utils::{kneighbors_parallel, kneighbors_single};

/// K-Nearest Neighbors Classifier.
///
/// Classifier implementing the k-nearest neighbors vote. The target is predicted by
/// a majority vote of the nearest neighbors in the training set.
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
/// algorithm : str, default='auto'
///     Algorithm used to compute the nearest neighbors:
///     - 'auto': attempt to decide the most appropriate algorithm based on the values passed to fit method.
///     - 'ball_tree': use BallTree
///     - 'kd_tree': use KDTree
///     - 'brute': use brute-force search
///
/// leaf_size : int, default=30
///     Leaf size passed to BallTree or KDTree. This can affect the speed of the
///     construction and query, as well as the memory required to store the tree.
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
pub struct KNeighborsClassifier {
    // Configuration parameters
    n_neighbors: usize,
    weights: String,
    algorithm: String,
    leaf_size: usize,
    metric: String,
    p: f64,
    n_jobs: i32,

    // Fitted data (for brute force)
    x_train_cache: Option<Array2<f64>>,
    y_train_cache: Option<Array1<i64>>,

    // Tree structures
    kd_tree: Option<KDTree>,
    ball_tree: Option<BallTree>,

    // Metadata
    n_features_in: Option<usize>,
    n_samples_fit: Option<usize>,
    effective_algorithm: Option<String>,
    classes: Option<Vec<i64>>,
}

#[pymethods]
impl KNeighborsClassifier {
    #[new]
    #[pyo3(signature = (*, n_neighbors=5, weights="uniform", algorithm="auto", leaf_size=30, metric="euclidean", p=2.0, n_jobs=1))]
    fn new(
        n_neighbors: usize,
        weights: &str,
        algorithm: &str,
        leaf_size: usize,
        metric: &str,
        p: f64,
        n_jobs: i32,
    ) -> Self {
        KNeighborsClassifier {
            n_neighbors,
            weights: weights.to_string(),
            algorithm: algorithm.to_string(),
            leaf_size,
            metric: metric.to_string(),
            p,
            n_jobs,
            x_train_cache: None,
            y_train_cache: None,
            kd_tree: None,
            ball_tree: None,
            n_features_in: None,
            n_samples_fit: None,
            effective_algorithm: None,
            classes: None,
        }
    }

    /// Fit the k-nearest neighbors classifier from the training dataset.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     Training data.
    /// y : array-like of shape (n_samples,)
    ///     Target class labels.
    ///
    /// Returns
    /// -------
    /// self : KNeighborsClassifier
    ///     The fitted estimator.
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        _py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, i64>,
    ) -> PyRefMut<'py, Self> {
        let x_arr = x.as_array();
        let y_arr = y.as_array();

        let n_features = x_arr.ncols();
        let n_samples = x_arr.nrows();

        // Extract unique classes and sort them
        let mut classes: Vec<i64> = y_arr.iter().cloned().collect();
        classes.sort();
        classes.dedup();
        slf.classes = Some(classes);

        // Determine effective algorithm
        // Match sklearn's heuristic: use brute force for high dimensions or small datasets
        let effective_algorithm = match slf.algorithm.as_str() {
            "auto" => {
                // sklearn switches to brute force around 15-20 features
                // and also for small datasets where tree overhead isn't worth it
                if n_features > 15 || n_samples < 30 {
                    "brute".to_string()
                } else {
                    "kd_tree".to_string()
                }
            }
            alg => alg.to_string(),
        };

        // Reset tree caches
        slf.kd_tree = None;
        slf.ball_tree = None;
        slf.x_train_cache = None;

        // Build appropriate data structure
        match effective_algorithm.as_str() {
            "kd_tree" => {
                let tree = KDTree::build(x_arr.to_owned(), slf.leaf_size);
                slf.kd_tree = Some(tree);
            }
            "ball_tree" => {
                let tree = BallTree::build(x_arr.to_owned(), slf.leaf_size, &slf.metric, slf.p);
                slf.ball_tree = Some(tree);
            }
            _ => {
                // brute force - just store the data
                slf.x_train_cache = Some(x_arr.to_owned());
            }
        }

        // Always store y_train for predictions
        slf.y_train_cache = Some(y_arr.to_owned());
        slf.n_features_in = Some(n_features);
        slf.n_samples_fit = Some(n_samples);
        slf.effective_algorithm = Some(effective_algorithm);

        slf
    }

    /// Predict the class labels for the provided data.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     Test samples.
    ///
    /// Returns
    /// -------
    /// y : ndarray of shape (n_samples,)
    ///     Class labels for each sample.
    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let y_train = self
            .y_train_cache
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Not fitted yet"))?;

        let effective_algorithm = self
            .effective_algorithm
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Not fitted yet"))?;

        let x_test = x.as_array();
        let use_parallel = self.n_jobs != 1;
        let k = self.n_neighbors;
        let use_distance_weights = self.weights == "distance";

        let predictions = match effective_algorithm.as_str() {
            "kd_tree" => {
                let tree = self.kd_tree.as_ref().unwrap();
                let metric = self.metric.clone();
                let p = self.p;

                // Trees return (indices, distances)
                let (indices_2d, distances_2d) = if use_parallel {
                    tree.query_batch_parallel(x_test, k, &metric, p)
                } else {
                    tree.query_batch_single(x_test, k, &metric, p)
                };

                self.compute_predictions_from_neighbors(
                    &indices_2d,
                    &distances_2d,
                    y_train,
                    use_distance_weights,
                )
            }
            "ball_tree" => {
                let tree = self.ball_tree.as_ref().unwrap();

                // Trees return (indices, distances)
                let (indices_2d, distances_2d) = if use_parallel {
                    tree.query_batch_parallel(x_test, k)
                } else {
                    tree.query_batch_single(x_test, k)
                };

                self.compute_predictions_from_neighbors(
                    &indices_2d,
                    &distances_2d,
                    y_train,
                    use_distance_weights,
                )
            }
            _ => {
                // Brute force
                let x_train = self.x_train_cache.as_ref().unwrap();
                let x_train_owned = x_train.clone();
                let y_train_owned = y_train.clone();
                let x_test_owned = x_test.to_owned();
                let weights = self.weights.clone();
                let metric = self.metric.clone();
                let p = self.p;

                let (distances_2d, indices_2d) = if use_parallel {
                    py.allow_threads(move || {
                        kneighbors_parallel(
                            x_test_owned.view(),
                            x_train_owned.view(),
                            k,
                            &metric,
                            p,
                        )
                    })
                } else {
                    py.allow_threads(move || {
                        kneighbors_single(x_test_owned.view(), x_train_owned.view(), k, &metric, p)
                    })
                };

                self.compute_predictions_from_neighbors(
                    &indices_2d,
                    &distances_2d,
                    &y_train_owned,
                    weights == "distance",
                )
            }
        };

        let result = Array1::from_vec(predictions);
        Ok(result.into_pyarray(py))
    }

    /// Predict class probabilities for the provided data.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     Test samples.
    ///
    /// Returns
    /// -------
    /// p : ndarray of shape (n_samples, n_classes)
    ///     The class probabilities of the input samples. Classes are ordered
    ///     as they appear in self.classes_.
    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let y_train = self
            .y_train_cache
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Not fitted yet"))?;

        let classes = self
            .classes
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Not fitted yet"))?;

        let effective_algorithm = self
            .effective_algorithm
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Not fitted yet"))?;

        let x_test = x.as_array();
        let use_parallel = self.n_jobs != 1;
        let k = self.n_neighbors;
        let use_distance_weights = self.weights == "distance";

        let probabilities = match effective_algorithm.as_str() {
            "kd_tree" => {
                let tree = self.kd_tree.as_ref().unwrap();
                let metric = self.metric.clone();
                let p = self.p;

                let (indices_2d, distances_2d) = if use_parallel {
                    tree.query_batch_parallel(x_test, k, &metric, p)
                } else {
                    tree.query_batch_single(x_test, k, &metric, p)
                };

                self.compute_probabilities_from_neighbors(
                    &indices_2d,
                    &distances_2d,
                    y_train,
                    classes,
                    use_distance_weights,
                )
            }
            "ball_tree" => {
                let tree = self.ball_tree.as_ref().unwrap();

                let (indices_2d, distances_2d) = if use_parallel {
                    tree.query_batch_parallel(x_test, k)
                } else {
                    tree.query_batch_single(x_test, k)
                };

                self.compute_probabilities_from_neighbors(
                    &indices_2d,
                    &distances_2d,
                    y_train,
                    classes,
                    use_distance_weights,
                )
            }
            _ => {
                // Brute force
                let x_train = self.x_train_cache.as_ref().unwrap();
                let x_train_owned = x_train.clone();
                let y_train_owned = y_train.clone();
                let x_test_owned = x_test.to_owned();
                let weights = self.weights.clone();
                let metric = self.metric.clone();
                let p = self.p;

                let (distances_2d, indices_2d) = if use_parallel {
                    py.allow_threads(move || {
                        kneighbors_parallel(
                            x_test_owned.view(),
                            x_train_owned.view(),
                            k,
                            &metric,
                            p,
                        )
                    })
                } else {
                    py.allow_threads(move || {
                        kneighbors_single(x_test_owned.view(), x_train_owned.view(), k, &metric, p)
                    })
                };

                self.compute_probabilities_from_neighbors(
                    &indices_2d,
                    &distances_2d,
                    &y_train_owned,
                    classes,
                    weights == "distance",
                )
            }
        };

        Ok(probabilities.into_pyarray(py))
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
        let effective_algorithm = self
            .effective_algorithm
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Not fitted yet"))?;

        let x_test = x.as_array();
        let k = n_neighbors.unwrap_or(self.n_neighbors);
        let use_parallel = self.n_jobs != 1;

        // Note: trees return (indices, distances), brute force returns (distances, indices)
        let (distances, indices) = match effective_algorithm.as_str() {
            "kd_tree" => {
                let tree = self.kd_tree.as_ref().unwrap();
                let metric = self.metric.clone();
                let p = self.p;

                let (idx, dist) = if use_parallel {
                    tree.query_batch_parallel(x_test, k, &metric, p)
                } else {
                    tree.query_batch_single(x_test, k, &metric, p)
                };
                (dist, idx)
            }
            "ball_tree" => {
                let tree = self.ball_tree.as_ref().unwrap();

                let (idx, dist) = if use_parallel {
                    tree.query_batch_parallel(x_test, k)
                } else {
                    tree.query_batch_single(x_test, k)
                };
                (dist, idx)
            }
            _ => {
                // Brute force
                let x_train = self.x_train_cache.as_ref().unwrap();
                let x_train_owned = x_train.clone();
                let x_test_owned = x_test.to_owned();
                let metric = self.metric.clone();
                let p = self.p;

                if use_parallel {
                    py.allow_threads(move || {
                        kneighbors_parallel(
                            x_test_owned.view(),
                            x_train_owned.view(),
                            k,
                            &metric,
                            p,
                        )
                    })
                } else {
                    py.allow_threads(move || {
                        kneighbors_single(x_test_owned.view(), x_train_owned.view(), k, &metric, p)
                    })
                }
            }
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

    /// Return the mean accuracy on the given test data and labels.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     Test samples.
    /// y : array-like of shape (n_samples,)
    ///     True labels for X.
    ///
    /// Returns
    /// -------
    /// score : float
    ///     Mean accuracy of self.predict(X) w.r.t. y.
    fn score<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, i64>,
    ) -> PyResult<f64> {
        let y_true = y.as_array();
        let y_pred_arr = self.predict(py, x)?;
        let y_pred = unsafe { y_pred_arr.as_array() };

        // Compute accuracy = correct predictions / total predictions
        let correct: usize = y_true
            .iter()
            .zip(y_pred.iter())
            .filter(|(yt, yp)| *yt == *yp)
            .count();

        Ok(correct as f64 / y_true.len() as f64)
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

    /// Class labels known to the classifier.
    #[getter]
    fn classes_(&self) -> Option<Vec<i64>> {
        self.classes.clone()
    }

    /// Get parameters for this estimator.
    #[pyo3(signature = (deep=true))]
    fn get_params(
        &self,
        py: Python<'_>,
        deep: bool,
    ) -> std::collections::HashMap<String, PyObject> {
        let _ = deep;
        let mut params = std::collections::HashMap::new();
        params.insert(
            "n_neighbors".to_string(),
            self.n_neighbors.into_pyobject(py).unwrap().unbind().into(),
        );
        params.insert(
            "weights".to_string(),
            self.weights
                .clone()
                .into_pyobject(py)
                .unwrap()
                .unbind()
                .into(),
        );
        params.insert(
            "algorithm".to_string(),
            self.algorithm
                .clone()
                .into_pyobject(py)
                .unwrap()
                .unbind()
                .into(),
        );
        params.insert(
            "leaf_size".to_string(),
            self.leaf_size.into_pyobject(py).unwrap().unbind().into(),
        );
        params.insert(
            "metric".to_string(),
            self.metric
                .clone()
                .into_pyobject(py)
                .unwrap()
                .unbind()
                .into(),
        );
        params.insert(
            "p".to_string(),
            self.p.into_pyobject(py).unwrap().unbind().into(),
        );
        params.insert(
            "n_jobs".to_string(),
            self.n_jobs.into_pyobject(py).unwrap().unbind().into(),
        );
        params
    }
}

impl KNeighborsClassifier {
    /// Compute predictions from neighbor indices and distances using majority voting
    fn compute_predictions_from_neighbors(
        &self,
        indices: &Array2<usize>,
        distances: &Array2<f64>,
        y_train: &Array1<i64>,
        use_distance_weights: bool,
    ) -> Vec<i64> {
        let n_samples = indices.nrows();
        let mut predictions = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            let pred = if use_distance_weights {
                // Distance-weighted voting
                let mut class_weights: HashMap<i64, f64> = HashMap::new();

                for j in 0..indices.ncols() {
                    let idx = indices[[i, j]];
                    let dist = distances[[i, j]];
                    let class_label = y_train[idx];

                    let weight = if dist == 0.0 {
                        f64::INFINITY
                    } else {
                        1.0 / dist
                    };

                    if weight.is_infinite() {
                        // Exact match - this class wins
                        class_weights.clear();
                        class_weights.insert(class_label, f64::INFINITY);
                        break;
                    }

                    *class_weights.entry(class_label).or_insert(0.0) += weight;
                }

                // Find class with highest weight
                class_weights
                    .into_iter()
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .map(|(class, _)| class)
                    .unwrap()
            } else {
                // Uniform weights - simple majority vote
                let mut class_counts: HashMap<i64, usize> = HashMap::new();

                for j in 0..indices.ncols() {
                    let idx = indices[[i, j]];
                    let class_label = y_train[idx];
                    *class_counts.entry(class_label).or_insert(0) += 1;
                }

                // Find class with highest count (ties go to lowest class label)
                class_counts
                    .into_iter()
                    .max_by(|a, b| a.1.cmp(&b.1).then_with(|| b.0.cmp(&a.0)))
                    .map(|(class, _)| class)
                    .unwrap()
            };

            predictions.push(pred);
        }

        predictions
    }

    /// Compute class probabilities from neighbor indices and distances
    fn compute_probabilities_from_neighbors(
        &self,
        indices: &Array2<usize>,
        distances: &Array2<f64>,
        y_train: &Array1<i64>,
        classes: &[i64],
        use_distance_weights: bool,
    ) -> Array2<f64> {
        let n_samples = indices.nrows();
        let n_classes = classes.len();
        let mut probabilities = Array2::zeros((n_samples, n_classes));

        // Create a mapping from class label to index
        let class_to_idx: HashMap<i64, usize> =
            classes.iter().enumerate().map(|(i, &c)| (c, i)).collect();

        for i in 0..n_samples {
            let mut class_weights: Vec<f64> = vec![0.0; n_classes];
            let mut has_exact_match = false;
            let mut exact_match_class = 0;

            for j in 0..indices.ncols() {
                let idx = indices[[i, j]];
                let dist = distances[[i, j]];
                let class_label = y_train[idx];
                let class_idx = class_to_idx[&class_label];

                if use_distance_weights {
                    if dist == 0.0 {
                        // Exact match - this class gets probability 1.0
                        has_exact_match = true;
                        exact_match_class = class_idx;
                        break;
                    }
                    class_weights[class_idx] += 1.0 / dist;
                } else {
                    class_weights[class_idx] += 1.0;
                }
            }

            if has_exact_match {
                // Set probability 1.0 for exact match class
                for c in 0..n_classes {
                    probabilities[[i, c]] = if c == exact_match_class { 1.0 } else { 0.0 };
                }
            } else {
                // Normalize weights to probabilities
                let total: f64 = class_weights.iter().sum();
                for c in 0..n_classes {
                    probabilities[[i, c]] = class_weights[c] / total;
                }
            }
        }

        probabilities
    }
}
