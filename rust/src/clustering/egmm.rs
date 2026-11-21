use std::collections::HashMap;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, Zip};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::kmeans_utils::{assign_clusters_single, initialize_centroids_random, SimpleRng};
use super::minibatch::{
    batch_law, cosine_alpha, epoch_rows, permutation, seed_subsample_kmeans_plusplus,
    select_revivals, split_row, step_law, CHECK_UNUSED_EVERY, INIT_USAGE, UNUSED_THRESHOLD,
};

const LN_2PI: f64 = 1.837_877_066_409_345_5;
/// Components with less responsibility mass than this in a batch are not updated.
const MIN_MASS: f64 = 1e-10;

/// Parameters of a diagonal-covariance Gaussian mixture.
#[derive(Clone)]
pub struct DiagGmm {
    pub means: Array2<f64>,
    pub variances: Array2<f64>,
    pub weights: Array1<f64>,
}

impl DiagGmm {
    /// Responsibilities (n, k) and per-sample log-likelihood (n,).
    pub fn e_step(&self, x: ArrayView2<f64>, parallel: bool) -> (Array2<f64>, Array1<f64>) {
        let (n, d) = x.dim();
        let k = self.means.nrows();
        let prec = self.variances.mapv(|v| 1.0 / v);
        let mean_prec = &self.means * &prec;
        let x_sq = x.mapv(|v| v * v);

        // -0.5 * sum_f (x_f - mu_f)^2 / var_f, expanded into two matmuls.
        let mut logp = Array2::<f64>::zeros((n, k));
        ndarray::linalg::general_mat_mul(-0.5, &x_sq, &prec.t(), 0.0, &mut logp);
        ndarray::linalg::general_mat_mul(1.0, &x, &mean_prec.t(), 1.0, &mut logp);

        let cst: Array1<f64> = (0..k)
            .map(|j| {
                let quad: f64 = self
                    .means
                    .row(j)
                    .iter()
                    .zip(prec.row(j).iter())
                    .map(|(m, p)| m * m * p)
                    .sum();
                let log_det: f64 = self.variances.row(j).iter().map(|v| v.ln()).sum();
                self.weights[j].max(f64::MIN_POSITIVE).ln()
                    - 0.5 * (quad + log_det + d as f64 * LN_2PI)
            })
            .collect();

        let mut loglik = Array1::<f64>::zeros(n);
        let normalize = |mut row: ndarray::ArrayViewMut1<f64>, ll: &mut f64| {
            row += &cst;
            let mx = row.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let mut s = 0.0;
            row.mapv_inplace(|v| {
                let e = (v - mx).exp();
                s += e;
                e
            });
            row.mapv_inplace(|v| v / s);
            *ll = mx + s.ln();
        };
        if parallel {
            Zip::from(logp.axis_iter_mut(Axis(0)))
                .and(&mut loglik)
                .par_for_each(normalize);
        } else {
            Zip::from(logp.axis_iter_mut(Axis(0)))
                .and(&mut loglik)
                .for_each(normalize);
        }
        (logp, loglik)
    }
}

/// Weighted sufficient statistics of a batch: mass (k,), sum x (k, d), sum x^2 (k, d).
fn batch_stats(
    xb: ArrayView2<f64>,
    resp: ArrayView2<f64>,
) -> (Array1<f64>, Array2<f64>, Array2<f64>) {
    let (_, d) = xb.dim();
    let k = resp.ncols();
    let nk = resp.sum_axis(Axis(0));
    let mut sk = Array2::<f64>::zeros((k, d));
    ndarray::linalg::general_mat_mul(1.0, &resp.t(), &xb, 0.0, &mut sk);
    let xb_sq = xb.mapv(|v| v * v);
    let mut qk = Array2::<f64>::zeros((k, d));
    ndarray::linalg::general_mat_mul(1.0, &resp.t(), &xb_sq, 0.0, &mut qk);
    (nk, sk, qk)
}

fn variances_from_moments(
    means: ArrayView2<f64>,
    second: ArrayView2<f64>,
    fallback: ArrayView1<f64>,
    reg_covar: f64,
    mass: &[f64],
) -> Array2<f64> {
    let mut var = &second - &means.mapv(|m| m * m);
    for (j, mut row) in var.axis_iter_mut(Axis(0)).enumerate() {
        if mass[j] < MIN_MASS {
            row.assign(&fallback);
        }
        row.mapv_inplace(|v| v.max(0.0) + reg_covar);
    }
    var
}

pub struct EGmmFit {
    pub model: DiagGmm,
    pub batch_size: usize,
    pub alpha: f64,
    pub n_steps: usize,
    pub n_splits: usize,
}

/// e-means applied to a diagonal Gaussian mixture: stochastic EM where each
/// component's (mass, mean, second moment) is an exponential moving average of
/// the responsibility-weighted batch statistics.
#[allow(clippy::too_many_arguments)]
pub fn fit_egmm(
    x: ArrayView2<f64>,
    k: usize,
    n_epochs: f64,
    batch_size: Option<usize>,
    alpha: Option<f64>,
    revival: bool,
    reg_covar: f64,
    init: &str,
    seed: u64,
    parallel: bool,
) -> EGmmFit {
    let (n, d) = x.dim();
    let mut rng = SimpleRng::new(seed);
    let b = batch_size
        .unwrap_or_else(|| batch_law(n, k, n_epochs, d))
        .clamp(1, n);
    let alpha0 = alpha.unwrap_or_else(|| step_law(n, k, n_epochs, b));

    // Global per-feature variance: fallback for empty components.
    let global_var = x
        .map_axis(Axis(0), |c| c.var(0.0))
        .mapv(|v| if v > 0.0 { v } else { 1.0 });

    // Initialise means by seeding, then variances and weights from a hard
    // assignment of a subsample to those seeds. Starting with the global
    // variance would make the first soft E-step pull every mean together.
    let means0 = if init == "k-means++" {
        seed_subsample_kmeans_plusplus(x, k, &mut rng)
    } else {
        initialize_centroids_random(x, k, &mut rng)
    };
    let m = n.min(10_000usize.max(20 * k));
    let mut sub_idx = permutation(n, &mut rng);
    sub_idx.truncate(m);
    let xs = x.select(Axis(0), &sub_idx);
    let (lab, _) = assign_clusters_single(xs.view(), means0.view());
    let mut hard = Array2::<f64>::zeros((m, k));
    for (i, &l) in lab.iter().enumerate() {
        hard[[i, l as usize]] = 1.0;
    }
    let (c0, s0, q0) = batch_stats(xs.view(), hard.view());
    let mut means = means0.clone();
    let mut second = Array2::<f64>::zeros((k, d));
    for j in 0..k {
        if c0[j] > 1.0 {
            means.row_mut(j).assign(&s0.row(j).mapv(|v| v / c0[j]));
            second.row_mut(j).assign(&q0.row(j).mapv(|v| v / c0[j]));
        } else {
            let mu = means.row(j).to_owned();
            second
                .row_mut(j)
                .assign(&(&global_var + &mu.mapv(|v| v * v)));
        }
    }
    let ones = vec![1.0; k];
    let variances = variances_from_moments(
        means.view(),
        second.view(),
        global_var.view(),
        reg_covar,
        &ones,
    );
    let tot = c0.sum().max(1.0);
    let weights = c0.mapv(|c| (c.max(1.0)) / tot);
    let mut model = DiagGmm {
        means,
        variances,
        weights: &weights / weights.sum(),
    };
    let mut mass = vec![INIT_USAGE; k];

    let epochs = epoch_rows(n, n_epochs);
    let total_steps: usize = epochs.iter().map(|r| r.div_ceil(b)).sum();
    let mut step = 0usize;
    let mut n_splits = 0usize;
    let mut pass_n = Array1::<f64>::zeros(k);
    let mut pass_s = Array2::<f64>::zeros((k, d));
    let mut pass_q = Array2::<f64>::zeros((k, d));

    for (e, &rows) in epochs.iter().enumerate() {
        let last = e + 1 == epochs.len();
        let perm = permutation(n, &mut rng);
        for chunk in perm[..rows].chunks(b) {
            let xb = x.select(Axis(0), chunk);
            let (resp, _) = model.e_step(xb.view(), parallel);
            let (nk, sk, qk) = batch_stats(xb.view(), resp.view());
            if last {
                pass_n += &nk;
                pass_s += &sk;
                pass_q += &qk;
            }

            let a = cosine_alpha(alpha0, step, total_steps);
            for j in 0..k {
                mass[j] *= 1.0 - a;
                if nk[j] > MIN_MASS {
                    let new_mass = mass[j] + a * nk[j];
                    let w = a * nk[j] / new_mass;
                    let inv = 1.0 / nk[j];
                    Zip::from(model.means.row_mut(j))
                        .and(second.row_mut(j))
                        .and(sk.row(j))
                        .and(qk.row(j))
                        .for_each(|mu, m2, &s, &q| {
                            *mu += w * (s * inv - *mu);
                            *m2 += w * (q * inv - *m2);
                        });
                    mass[j] = new_mass;
                }
            }
            step += 1;

            if revival && step.is_multiple_of(CHECK_UNUSED_EVERY) {
                for (src, dst) in select_revivals(&mass, UNUSED_THRESHOLD, &mut rng) {
                    let var_src: Array1<f64> = model.variances.row(src).to_owned();
                    split_row(&mut model.means, src, dst);
                    for jj in [src, dst] {
                        let mu = model.means.row(jj).to_owned();
                        second.row_mut(jj).assign(&(&var_src + &mu.mapv(|v| v * v)));
                    }
                    mass[src] /= 2.0;
                    mass[dst] = mass[src];
                    n_splits += 1;
                }
            }

            model.variances = variances_from_moments(
                model.means.view(),
                second.view(),
                global_var.view(),
                reg_covar,
                &ones,
            );
            let total: f64 = mass.iter().sum();
            model.weights = Array1::from_iter(mass.iter().map(|&w| w / total));
        }
    }

    // Finalization: one exact M-step on the statistics gathered in the last pass.
    let pass_total = pass_n.sum();
    for j in 0..k {
        if pass_n[j] > MIN_MASS {
            let inv = 1.0 / pass_n[j];
            model
                .means
                .row_mut(j)
                .assign(&pass_s.row(j).mapv(|v| v * inv));
            second.row_mut(j).assign(&pass_q.row(j).mapv(|v| v * inv));
        }
    }
    let pass_mass = pass_n.to_vec();
    model.variances = variances_from_moments(
        model.means.view(),
        second.view(),
        global_var.view(),
        reg_covar,
        &ones,
    );
    if pass_total > 0.0 {
        let floor = MIN_MASS / pass_total;
        let w: Array1<f64> =
            Array1::from_iter(pass_mass.iter().map(|&c| (c / pass_total).max(floor)));
        model.weights = &w / w.sum();
    }

    EGmmFit {
        model,
        batch_size: b,
        alpha: alpha0,
        n_steps: total_steps,
        n_splits,
    }
}

fn argmax_rows(resp: ArrayView2<f64>) -> Array1<i64> {
    resp.map_axis(Axis(1), |r| {
        let mut best = 0usize;
        for (j, &v) in r.iter().enumerate() {
            if v > r[best] {
                best = j;
            }
        }
        best as i64
    })
}

/// Gaussian mixture fitted with the e-means trick (diagonal covariances).
///
/// Stochastic EM on mini-batches: each component keeps an exponential moving
/// average (EWM) of its responsibility mass, mean and second moment, with the
/// same cosine step schedule, fitted batch/step laws, optional revival and
/// last-pass finalization as `rklearn.cluster.EMeans` (arXiv:2611.09501).
///
/// Parameters
/// ----------
/// n_components : int, default=1
///     Number of mixture components.
///
/// n_epochs : float, default=3.0
///     Number of passes over the data. Fractional values are allowed.
///
/// batch_size : int or None, default=None
///     Mini-batch size. None uses the e-means batch law.
///
/// alpha : float or None, default=None
///     Peak EWM step, in (0, 1]. None uses the e-means step law.
///
/// revival : bool, default=False
///     Split a heavy component into any component whose EWM mass falls below
///     10% of the mean mass (checked every 5 updates).
///
/// reg_covar : float, default=1e-6
///     Non-negative value added to the diagonal of every covariance.
///
/// init_params : str, default='k-means++'
///     How to seed the means: 'k-means++' (on a subsample) or 'random'.
///     Variances and weights then come from a hard assignment of a subsample.
///
/// random_state : int or None, default=None
///     Seed for random number generation.
///
/// n_jobs : int, default=1
///     Use 1 for single-threaded, any other value for a parallel E-step.
#[pyclass]
pub struct EGaussianMixture {
    n_components: usize,
    n_epochs: f64,
    batch_size: Option<usize>,
    alpha: Option<f64>,
    revival: bool,
    reg_covar: f64,
    init_params: String,
    random_state: Option<u64>,
    n_jobs: i32,

    model: Option<DiagGmm>,
    labels: Option<Array1<i64>>,
    lower_bound: Option<f64>,
    n_iter: Option<usize>,
    n_features_in: Option<usize>,
    batch_size_fitted: Option<usize>,
    alpha_fitted: Option<f64>,
    n_splits: Option<usize>,
}

impl EGaussianMixture {
    fn run_fit(&mut self, py: Python<'_>, x: ArrayView2<f64>) -> PyResult<()> {
        let (n, d) = x.dim();
        if n < self.n_components {
            return Err(PyValueError::new_err(format!(
                "n_samples={} should be >= n_components={}",
                n, self.n_components
            )));
        }
        let x_owned = x.to_owned();
        let k = self.n_components;
        let n_epochs = self.n_epochs;
        let batch_size = self.batch_size;
        let alpha = self.alpha;
        let revival = self.revival;
        let reg = self.reg_covar;
        let init = self.init_params.clone();
        let seed = self.random_state.unwrap_or(0);
        let parallel = self.n_jobs != 1;

        let (fit, labels, ll) = py.detach(move || {
            let fit = fit_egmm(
                x_owned.view(),
                k,
                n_epochs,
                batch_size,
                alpha,
                revival,
                reg,
                &init,
                seed,
                parallel,
            );
            let (resp, loglik) = fit.model.e_step(x_owned.view(), parallel);
            (
                fit,
                argmax_rows(resp.view()),
                loglik.mean().unwrap_or(f64::NAN),
            )
        });

        self.n_features_in = Some(d);
        self.model = Some(fit.model);
        self.labels = Some(labels);
        self.lower_bound = Some(ll);
        self.n_iter = Some(fit.n_steps);
        self.batch_size_fitted = Some(fit.batch_size);
        self.alpha_fitted = Some(fit.alpha);
        self.n_splits = Some(fit.n_splits);
        Ok(())
    }

    fn fitted(&self) -> PyResult<&DiagGmm> {
        self.model
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Not fitted yet"))
    }

    fn e_step_on(
        &self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<(Array2<f64>, Array1<f64>)> {
        let model = self.fitted()?.clone();
        if x.as_array().ncols() != model.means.ncols() {
            return Err(PyValueError::new_err(format!(
                "X has {} features, but EGaussianMixture was fitted with {} features",
                x.as_array().ncols(),
                model.means.ncols()
            )));
        }
        let x_owned = x.as_array().to_owned();
        let parallel = self.n_jobs != 1;
        Ok(py.detach(move || model.e_step(x_owned.view(), parallel)))
    }

    fn n_parameters(&self) -> PyResult<f64> {
        let m = self.fitted()?;
        let (k, d) = m.means.dim();
        Ok((2 * k * d + k - 1) as f64)
    }
}

#[pymethods]
impl EGaussianMixture {
    #[new]
    #[pyo3(signature = (
        *,
        n_components=1,
        n_epochs=3.0,
        batch_size=None,
        alpha=None,
        revival=false,
        reg_covar=1e-6,
        init_params="k-means++",
        random_state=None,
        n_jobs=1
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n_components: usize,
        n_epochs: f64,
        batch_size: Option<usize>,
        alpha: Option<f64>,
        revival: bool,
        reg_covar: f64,
        init_params: &str,
        random_state: Option<u64>,
        n_jobs: i32,
    ) -> PyResult<Self> {
        if init_params != "k-means++" && init_params != "random" {
            return Err(PyValueError::new_err(
                "init_params must be 'k-means++' or 'random'",
            ));
        }
        if n_components < 1 {
            return Err(PyValueError::new_err("n_components must be >= 1"));
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
        if reg_covar.is_nan() || reg_covar < 0.0 {
            return Err(PyValueError::new_err("reg_covar must be >= 0"));
        }
        Ok(EGaussianMixture {
            n_components,
            n_epochs,
            batch_size,
            alpha,
            revival,
            reg_covar,
            init_params: init_params.to_string(),
            random_state,
            n_jobs,
            model: None,
            labels: None,
            lower_bound: None,
            n_iter: None,
            n_features_in: None,
            batch_size_fitted: None,
            alpha_fitted: None,
            n_splits: None,
        })
    }

    /// Fit the mixture.
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        slf.run_fit(py, x.as_array())?;
        Ok(slf)
    }

    /// Fit, then return the most likely component for each sample.
    fn fit_predict<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        slf.run_fit(py, x.as_array())?;
        Ok(slf.labels.clone().unwrap().into_pyarray(py))
    }

    /// Most likely component for each sample.
    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let (resp, _) = self.e_step_on(py, x)?;
        Ok(argmax_rows(resp.view()).into_pyarray(py))
    }

    /// Posterior probability of each component, shape (n_samples, n_components).
    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(self.e_step_on(py, x)?.0.into_pyarray(py))
    }

    /// Per-sample log-likelihood.
    fn score_samples<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self.e_step_on(py, x)?.1.into_pyarray(py))
    }

    /// Mean per-sample log-likelihood.
    fn score<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<'py, f64>) -> PyResult<f64> {
        Ok(self.e_step_on(py, x)?.1.mean().unwrap_or(f64::NAN))
    }

    /// Bayesian information criterion on X (lower is better).
    fn bic<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<'py, f64>) -> PyResult<f64> {
        let n = x.as_array().nrows() as f64;
        let s = self.score(py, x)?;
        Ok(-2.0 * s * n + self.n_parameters()? * n.ln())
    }

    /// Akaike information criterion on X (lower is better).
    fn aic<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<'py, f64>) -> PyResult<f64> {
        let n = x.as_array().nrows() as f64;
        let s = self.score(py, x)?;
        Ok(-2.0 * s * n + 2.0 * self.n_parameters()?)
    }

    /// Get parameters for this estimator.
    #[pyo3(signature = (deep=true))]
    fn get_params(&self, py: Python<'_>, deep: bool) -> PyResult<HashMap<String, Py<PyAny>>> {
        let _ = deep;
        let mut p: HashMap<String, Py<PyAny>> = HashMap::new();
        p.insert(
            "n_components".into(),
            self.n_components.into_pyobject(py)?.into_any().unbind(),
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
            "reg_covar".into(),
            self.reg_covar.into_pyobject(py)?.into_any().unbind(),
        );
        p.insert(
            "init_params".into(),
            self.init_params
                .clone()
                .into_pyobject(py)?
                .into_any()
                .unbind(),
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
    fn weights_<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.model
            .as_ref()
            .map(|m| m.weights.clone().into_pyarray(py))
    }

    #[getter]
    fn means_<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f64>>> {
        self.model
            .as_ref()
            .map(|m| m.means.clone().into_pyarray(py))
    }

    /// Diagonal covariances, shape (n_components, n_features).
    #[getter]
    fn covariances_<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f64>>> {
        self.model
            .as_ref()
            .map(|m| m.variances.clone().into_pyarray(py))
    }

    #[getter]
    fn precisions_<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f64>>> {
        self.model
            .as_ref()
            .map(|m| m.variances.mapv(|v| 1.0 / v).into_pyarray(py))
    }

    #[getter]
    fn covariance_type(&self) -> &'static str {
        "diag"
    }

    #[getter]
    fn labels_<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<i64>>> {
        self.labels.as_ref().map(|l| l.clone().into_pyarray(py))
    }

    /// Mean log-likelihood of the training data under the fitted model.
    #[getter]
    fn lower_bound_(&self) -> Option<f64> {
        self.lower_bound
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

    #[getter]
    fn batch_size_(&self) -> Option<usize> {
        self.batch_size_fitted
    }

    #[getter]
    fn alpha_(&self) -> Option<f64> {
        self.alpha_fitted
    }

    #[getter]
    fn n_splits_(&self) -> Option<usize> {
        self.n_splits
    }
}
