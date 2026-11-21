use std::collections::HashMap;

use ndarray::{Array1, Array2, ArrayView2, Axis};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::kmeans_utils::{
    assign_clusters_parallel, assign_clusters_single, compute_distances_parallel,
    compute_distances_single, compute_inertia, initialize_centroids_random, SimpleRng,
};
use super::minibatch::{
    batch_law, cosine_alpha, epoch_rows, permutation, seed_subsample_kmeans_plusplus,
    select_revivals, split_row, step_law, CHECK_UNUSED_EVERY, INIT_USAGE, UNUSED_THRESHOLD,
};

/// Resolved hyperparameters plus the result of one e-means fit.
pub struct EMeansFit {
    pub centers: Array2<f64>,
    pub labels: Array1<i64>,
    pub distances: Array1<f64>,
    pub batch_size: usize,
    pub alpha: f64,
    pub n_steps: usize,
    pub n_splits: usize,
}

fn assign(x: ArrayView2<f64>, c: ArrayView2<f64>, parallel: bool) -> (Array1<i64>, Array1<f64>) {
    if parallel {
        assign_clusters_parallel(x, c)
    } else {
        assign_clusters_single(x, c)
    }
}

/// Algorithm 1 of arXiv:2611.09501: mini-batch k-means where every centroid
/// is an exponential moving average of the points assigned to it.
#[allow(clippy::too_many_arguments)]
pub fn fit_emeans(
    x: ArrayView2<f64>,
    k: usize,
    n_epochs: f64,
    batch_size: Option<usize>,
    alpha: Option<f64>,
    revival: bool,
    init: &str,
    seed: u64,
    parallel: bool,
) -> EMeansFit {
    let (n, d) = x.dim();
    let mut rng = SimpleRng::new(seed);

    let b = batch_size
        .unwrap_or_else(|| batch_law(n, k, n_epochs, d))
        .clamp(1, n);
    let alpha0 = alpha.unwrap_or_else(|| step_law(n, k, n_epochs, b));

    let mut centers = if init == "k-means++" {
        seed_subsample_kmeans_plusplus(x, k, &mut rng)
    } else {
        initialize_centroids_random(x, k, &mut rng)
    };
    let mut mass = vec![INIT_USAGE; k];

    let epochs = epoch_rows(n, n_epochs);
    let total_steps: usize = epochs.iter().map(|r| r.div_ceil(b)).sum();
    let mut step = 0usize;
    let mut n_splits = 0usize;
    let mut pass_counts = vec![0.0f64; k];
    let mut pass_sums = Array2::<f64>::zeros((k, d));

    for (e, &rows) in epochs.iter().enumerate() {
        let last = e + 1 == epochs.len();
        let perm = permutation(n, &mut rng);
        for chunk in perm[..rows].chunks(b) {
            let xb = x.select(Axis(0), chunk);
            let (labels, _) = assign(xb.view(), centers.view(), parallel);

            let mut counts = vec![0.0f64; k];
            let mut sums = Array2::<f64>::zeros((k, d));
            for (i, &l) in labels.iter().enumerate() {
                let j = l as usize;
                counts[j] += 1.0;
                let mut row = sums.row_mut(j);
                row += &xb.row(i);
            }
            if last {
                for j in 0..k {
                    pass_counts[j] += counts[j];
                }
                pass_sums += &sums;
            }

            // EMA on (N_j, S_j), carried as (N_j, c_j = S_j / N_j).
            let a = cosine_alpha(alpha0, step, total_steps);
            for j in 0..k {
                mass[j] *= 1.0 - a;
                if counts[j] > 0.0 {
                    let new_mass = mass[j] + a * counts[j];
                    let w = a * counts[j] / new_mass;
                    let inv = 1.0 / counts[j];
                    let mut c = centers.row_mut(j);
                    for (cf, &sf) in c.iter_mut().zip(sums.row(j).iter()) {
                        *cf += w * (sf * inv - *cf);
                    }
                    mass[j] = new_mass;
                }
            }
            step += 1;

            if revival && step.is_multiple_of(CHECK_UNUSED_EVERY) {
                for (src, dst) in select_revivals(&mass, UNUSED_THRESHOLD, &mut rng) {
                    split_row(&mut centers, src, dst);
                    mass[src] /= 2.0;
                    mass[dst] = mass[src];
                    n_splits += 1;
                }
            }
        }
    }

    // Finalization: replace each centroid by the plain mean of the points it
    // received during the last pass.
    for (j, &count) in pass_counts.iter().enumerate() {
        if count > 0.0 {
            let inv = 1.0 / count;
            centers
                .row_mut(j)
                .assign(&pass_sums.row(j).mapv(|v| v * inv));
        }
    }

    let (labels, distances) = assign(x, centers.view(), parallel);
    EMeansFit {
        centers,
        labels,
        distances,
        batch_size: b,
        alpha: alpha0,
        n_steps: total_steps,
        n_splits,
    }
}

/// e-means clustering: mini-batch k-means with exponentially weighted centroids.
///
/// Implements Bessa et al., "e-means: outperforming k-means (in one-tenth the
/// epochs)", arXiv:2611.09501. Each centroid is an exponential moving average
/// (EWM) of the points assigned to it, the EWM step follows a cosine schedule,
/// and the last pass replaces every centroid with a plain mean.
///
/// Parameters
/// ----------
/// n_clusters : int, default=8
///     The number of clusters to form.
///
/// n_epochs : float, default=3.0
///     Number of passes over the data. Fractional values are allowed.
///
/// batch_size : int or None, default=None
///     Mini-batch size. None picks it from the paper's fitted batch law.
///
/// alpha : float or None, default=None
///     Peak EWM step, in (0, 1]. The step decays from alpha to 0.1 * alpha
///     on a cosine schedule. None picks it from the paper's fitted step law.
///
/// revival : bool, default=False
///     Revive dead clusters by splitting a heavy one, as in the paper. A
///     cluster is dead when its EWM mass drops below 10% of the mean mass;
///     the check runs every 5 updates. Off by default: arXiv:2611.22065 shows
///     it can throw away small, isolated clusters.
///
/// init : str, default='k-means++'
///     'k-means++' seeds on a subsample of max(10000, 20 * n_clusters) rows.
///     'random' picks rows at random (the paper's setting).
///
/// random_state : int or None, default=None
///     Seed for random number generation.
///
/// n_jobs : int, default=1
///     Use 1 for single-threaded, any other value for parallel assignment.
#[pyclass]
pub struct EMeans {
    n_clusters: usize,
    n_epochs: f64,
    batch_size: Option<usize>,
    alpha: Option<f64>,
    revival: bool,
    init: String,
    random_state: Option<u64>,
    n_jobs: i32,

    cluster_centers: Option<Array2<f64>>,
    labels: Option<Array1<i64>>,
    inertia: Option<f64>,
    n_iter: Option<usize>,
    n_features_in: Option<usize>,
    batch_size_fitted: Option<usize>,
    alpha_fitted: Option<f64>,
    n_splits: Option<usize>,
}

impl EMeans {
    fn run_fit(&mut self, py: Python<'_>, x: ArrayView2<f64>) -> PyResult<()> {
        let (n, d) = x.dim();
        if n < self.n_clusters {
            return Err(PyValueError::new_err(format!(
                "n_samples={} should be >= n_clusters={}",
                n, self.n_clusters
            )));
        }
        let x_owned = x.to_owned();
        let k = self.n_clusters;
        let n_epochs = self.n_epochs;
        let batch_size = self.batch_size;
        let alpha = self.alpha;
        let revival = self.revival;
        let init = self.init.clone();
        let seed = self.random_state.unwrap_or(0);
        let parallel = self.n_jobs != 1;

        let fit = py.detach(move || {
            fit_emeans(
                x_owned.view(),
                k,
                n_epochs,
                batch_size,
                alpha,
                revival,
                &init,
                seed,
                parallel,
            )
        });

        self.n_features_in = Some(d);
        self.inertia = Some(compute_inertia(fit.distances.view()));
        self.cluster_centers = Some(fit.centers);
        self.labels = Some(fit.labels);
        self.n_iter = Some(fit.n_steps);
        self.batch_size_fitted = Some(fit.batch_size);
        self.alpha_fitted = Some(fit.alpha);
        self.n_splits = Some(fit.n_splits);
        Ok(())
    }

    fn centers(&self) -> PyResult<&Array2<f64>> {
        self.cluster_centers
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Not fitted yet"))
    }
}

#[pymethods]
impl EMeans {
    #[new]
    #[pyo3(signature = (
        *,
        n_clusters=8,
        n_epochs=3.0,
        batch_size=None,
        alpha=None,
        revival=false,
        init="k-means++",
        random_state=None,
        n_jobs=1
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n_clusters: usize,
        n_epochs: f64,
        batch_size: Option<usize>,
        alpha: Option<f64>,
        revival: bool,
        init: &str,
        random_state: Option<u64>,
        n_jobs: i32,
    ) -> PyResult<Self> {
        if init != "k-means++" && init != "random" {
            return Err(PyValueError::new_err(
                "init must be 'k-means++' or 'random'",
            ));
        }
        if n_clusters < 1 {
            return Err(PyValueError::new_err("n_clusters must be >= 1"));
        }
        if n_epochs.is_nan() || n_epochs <= 0.0 {
            return Err(PyValueError::new_err("n_epochs must be > 0"));
        }
        if batch_size == Some(0) {
            return Err(PyValueError::new_err("batch_size must be >= 1"));
        }
        if let Some(a) = alpha {
            if a.is_nan() || a <= 0.0 || a > 1.0 {
                return Err(PyValueError::new_err("alpha must be in (0, 1]"));
            }
        }
        Ok(EMeans {
            n_clusters,
            n_epochs,
            batch_size,
            alpha,
            revival,
            init: init.to_string(),
            random_state,
            n_jobs,
            cluster_centers: None,
            labels: None,
            inertia: None,
            n_iter: None,
            n_features_in: None,
            batch_size_fitted: None,
            alpha_fitted: None,
            n_splits: None,
        })
    }

    /// Compute e-means clustering.
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        slf.run_fit(py, x.as_array())?;
        Ok(slf)
    }

    /// Compute clustering and return labels.
    fn fit_predict<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        slf.run_fit(py, x.as_array())?;
        Ok(slf.labels.clone().unwrap().into_pyarray(py))
    }

    /// Fit, then return distances to each centroid.
    fn fit_transform<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        slf.run_fit(py, x.as_array())?;
        slf.transform(py, x)
    }

    /// Predict the closest cluster each sample in X belongs to.
    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let c = self.centers()?.clone();
        let x_owned = x.as_array().to_owned();
        let parallel = self.n_jobs != 1;
        let labels = py.detach(move || assign(x_owned.view(), c.view(), parallel).0);
        Ok(labels.into_pyarray(py))
    }

    /// Transform X to a cluster-distance space.
    fn transform<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let c = self.centers()?.clone();
        let x_owned = x.as_array().to_owned();
        let parallel = self.n_jobs != 1;
        let dist = py.detach(move || {
            if parallel {
                compute_distances_parallel(x_owned.view(), c.view())
            } else {
                compute_distances_single(x_owned.view(), c.view())
            }
        });
        Ok(dist.into_pyarray(py))
    }

    /// Negative inertia of X under the fitted centroids.
    fn score<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<'py, f64>) -> PyResult<f64> {
        let c = self.centers()?.clone();
        let x_owned = x.as_array().to_owned();
        let parallel = self.n_jobs != 1;
        let inertia = py.detach(move || {
            let (_, d) = assign(x_owned.view(), c.view(), parallel);
            compute_inertia(d.view())
        });
        Ok(-inertia)
    }

    /// Get parameters for this estimator.
    #[pyo3(signature = (deep=true))]
    fn get_params(&self, py: Python<'_>, deep: bool) -> PyResult<HashMap<String, Py<PyAny>>> {
        let _ = deep;
        let mut p: HashMap<String, Py<PyAny>> = HashMap::new();
        p.insert(
            "n_clusters".into(),
            self.n_clusters.into_pyobject(py)?.into_any().unbind(),
        );
        p.insert(
            "n_epochs".into(),
            self.n_epochs.into_pyobject(py)?.into_any().unbind(),
        );
        p.insert(
            "batch_size".into(),
            self.batch_size.into_pyobject(py)?.into_any().unbind(),
        );
        p.insert(
            "alpha".into(),
            self.alpha.into_pyobject(py)?.into_any().unbind(),
        );
        p.insert(
            "revival".into(),
            pyo3::types::PyBool::new(py, self.revival)
                .to_owned()
                .into_any()
                .unbind(),
        );
        p.insert(
            "init".into(),
            self.init.clone().into_pyobject(py)?.into_any().unbind(),
        );
        p.insert(
            "random_state".into(),
            self.random_state.into_pyobject(py)?.into_any().unbind(),
        );
        p.insert(
            "n_jobs".into(),
            self.n_jobs.into_pyobject(py)?.into_any().unbind(),
        );
        Ok(p)
    }

    #[getter]
    fn cluster_centers_<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f64>>> {
        self.cluster_centers
            .as_ref()
            .map(|c| c.clone().into_pyarray(py))
    }

    #[getter]
    fn labels_<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<i64>>> {
        self.labels.as_ref().map(|l| l.clone().into_pyarray(py))
    }

    #[getter]
    fn inertia_(&self) -> Option<f64> {
        self.inertia
    }

    /// Number of mini-batch updates performed.
    #[getter]
    fn n_iter_(&self) -> Option<usize> {
        self.n_iter
    }

    #[getter]
    fn n_features_in_(&self) -> Option<usize> {
        self.n_features_in
    }

    /// Batch size used in the last fit (resolved from the batch law if None).
    #[getter]
    fn batch_size_(&self) -> Option<usize> {
        self.batch_size_fitted
    }

    /// Peak EWM step used in the last fit (resolved from the step law if None).
    #[getter]
    fn alpha_(&self) -> Option<f64> {
        self.alpha_fitted
    }

    /// Number of dead clusters revived during the last fit.
    #[getter]
    fn n_splits_(&self) -> Option<usize> {
        self.n_splits
    }
}
