use ndarray::{Array1, ArrayView2};
use rayon::prelude::*;

/// Per-feature statistics computed using Welford's online algorithm.
/// This supports incremental updates for partial_fit.
#[derive(Clone, Debug)]
pub struct OnlineStats {
    pub mean: Array1<f64>,
    pub m2: Array1<f64>, // Sum of squared differences from mean
    pub count: usize,
}

impl OnlineStats {
    pub fn new(n_features: usize) -> Self {
        OnlineStats {
            mean: Array1::zeros(n_features),
            m2: Array1::zeros(n_features),
            count: 0,
        }
    }

    /// Update statistics with a new batch of data using Welford's algorithm.
    /// This is numerically stable and supports incremental updates.
    pub fn update(&mut self, data: ArrayView2<f64>) {
        let n_samples = data.nrows();
        if n_samples == 0 {
            return;
        }

        let n_features = data.ncols();

        // Process each sample incrementally
        for row_idx in 0..n_samples {
            self.count += 1;
            let n = self.count as f64;

            for col_idx in 0..n_features {
                let x = data[[row_idx, col_idx]];
                let delta = x - self.mean[col_idx];
                self.mean[col_idx] += delta / n;
                let delta2 = x - self.mean[col_idx];
                self.m2[col_idx] += delta * delta2;
            }
        }
    }

    /// Get population variance (divide by n, not n-1)
    pub fn variance(&self) -> Array1<f64> {
        if self.count == 0 {
            return self.m2.clone();
        }
        &self.m2 / (self.count as f64)
    }

    /// Get population standard deviation
    pub fn std(&self) -> Array1<f64> {
        self.variance().mapv(f64::sqrt)
    }
}

/// Compute mean and variance in parallel across row chunks.
/// This is faster for large datasets when we have all data at once.
pub fn compute_mean_var_parallel(data: ArrayView2<f64>) -> (Array1<f64>, Array1<f64>) {
    let n_features = data.ncols();
    let n_samples = data.nrows();

    if n_samples == 0 {
        return (Array1::zeros(n_features), Array1::zeros(n_features));
    }

    let n_samples_f64 = n_samples as f64;

    // Parallel reduction: each thread computes partial sums for a chunk of rows
    let (sums, sq_diffs): (Vec<f64>, Vec<f64>) = data
        .axis_chunks_iter(ndarray::Axis(0), 10000.max(n_samples / rayon::current_num_threads()))
        .into_par_iter()
        .map(|chunk| {
            let mut local_sums = vec![0.0; n_features];
            for row in chunk.rows() {
                for (j, &val) in row.iter().enumerate() {
                    local_sums[j] += val;
                }
            }
            local_sums
        })
        .reduce(
            || vec![0.0; n_features],
            |mut acc, local| {
                for (i, val) in local.into_iter().enumerate() {
                    acc[i] += val;
                }
                acc
            },
        )
        .into_iter()
        .map(|s| {
            let mean = s / n_samples_f64;
            (s, mean)
        })
        .unzip::<_, _, Vec<_>, Vec<_>>();

    // Compute means from sums
    let means: Vec<f64> = sums.iter().map(|&s| s / n_samples_f64).collect();

    // Second pass for variance (parallel)
    let sq_diffs: Vec<f64> = data
        .axis_chunks_iter(ndarray::Axis(0), 10000.max(n_samples / rayon::current_num_threads()))
        .into_par_iter()
        .map(|chunk| {
            let mut local_sq = vec![0.0; n_features];
            for row in chunk.rows() {
                for (j, &val) in row.iter().enumerate() {
                    let diff = val - means[j];
                    local_sq[j] += diff * diff;
                }
            }
            local_sq
        })
        .reduce(
            || vec![0.0; n_features],
            |mut acc, local| {
                for (i, val) in local.into_iter().enumerate() {
                    acc[i] += val;
                }
                acc
            },
        );

    let vars: Vec<f64> = sq_diffs.iter().map(|&s| s / n_samples_f64).collect();

    (Array1::from_vec(means), Array1::from_vec(vars))
}

/// Compute min and max in parallel across row chunks.
pub fn compute_min_max_parallel(data: ArrayView2<f64>) -> (Array1<f64>, Array1<f64>) {
    let n_features = data.ncols();
    let n_samples = data.nrows();

    if n_samples == 0 {
        return (
            Array1::from_elem(n_features, f64::INFINITY),
            Array1::from_elem(n_features, f64::NEG_INFINITY),
        );
    }

    // Parallel reduction over row chunks
    let (mins, maxs): (Vec<f64>, Vec<f64>) = data
        .axis_chunks_iter(ndarray::Axis(0), 10000.max(n_samples / rayon::current_num_threads()))
        .into_par_iter()
        .map(|chunk| {
            let first_row = chunk.row(0);
            let mut local_mins: Vec<f64> = first_row.to_vec();
            let mut local_maxs: Vec<f64> = first_row.to_vec();

            for row in chunk.rows().into_iter().skip(1) {
                for (j, &val) in row.iter().enumerate() {
                    if val < local_mins[j] {
                        local_mins[j] = val;
                    }
                    if val > local_maxs[j] {
                        local_maxs[j] = val;
                    }
                }
            }
            (local_mins, local_maxs)
        })
        .reduce(
            || (vec![f64::INFINITY; n_features], vec![f64::NEG_INFINITY; n_features]),
            |(mut acc_min, mut acc_max), (local_min, local_max)| {
                for i in 0..n_features {
                    if local_min[i] < acc_min[i] {
                        acc_min[i] = local_min[i];
                    }
                    if local_max[i] > acc_max[i] {
                        acc_max[i] = local_max[i];
                    }
                }
                (acc_min, acc_max)
            },
        );

    (Array1::from_vec(mins), Array1::from_vec(maxs))
}

/// Compute mean and variance single-threaded (for n_jobs=1).
/// Uses row-major iteration for cache efficiency.
pub fn compute_mean_var_single(data: ArrayView2<f64>) -> (Array1<f64>, Array1<f64>) {
    let n_features = data.ncols();
    let n_samples = data.nrows();

    if n_samples == 0 {
        return (Array1::zeros(n_features), Array1::zeros(n_features));
    }

    let n_samples_f64 = n_samples as f64;

    // First pass: compute sums (row-major for cache efficiency)
    let mut sums = vec![0.0; n_features];
    for row in data.rows() {
        for (j, &val) in row.iter().enumerate() {
            sums[j] += val;
        }
    }

    // Compute means
    let means: Vec<f64> = sums.iter().map(|&s| s / n_samples_f64).collect();

    // Second pass: compute variance (sum of squared differences)
    let mut sq_diffs = vec![0.0; n_features];
    for row in data.rows() {
        for (j, &val) in row.iter().enumerate() {
            let diff = val - means[j];
            sq_diffs[j] += diff * diff;
        }
    }

    let vars: Vec<f64> = sq_diffs.iter().map(|&s| s / n_samples_f64).collect();

    (Array1::from_vec(means), Array1::from_vec(vars))
}

/// Compute min and max single-threaded (for n_jobs=1).
/// Uses row-major iteration for cache efficiency.
pub fn compute_min_max_single(data: ArrayView2<f64>) -> (Array1<f64>, Array1<f64>) {
    let n_features = data.ncols();
    let n_samples = data.nrows();

    if n_samples == 0 {
        return (
            Array1::from_elem(n_features, f64::INFINITY),
            Array1::from_elem(n_features, f64::NEG_INFINITY),
        );
    }

    // Initialize with first row
    let first_row = data.row(0);
    let mut mins: Vec<f64> = first_row.to_vec();
    let mut maxs: Vec<f64> = first_row.to_vec();

    // Single pass through remaining rows (row-major for cache efficiency)
    for row in data.rows().into_iter().skip(1) {
        for (j, &val) in row.iter().enumerate() {
            if val < mins[j] {
                mins[j] = val;
            }
            if val > maxs[j] {
                maxs[j] = val;
            }
        }
    }

    (Array1::from_vec(mins), Array1::from_vec(maxs))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_online_stats_basic() {
        let data = array![[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0]];
        let mut stats = OnlineStats::new(2);
        stats.update(data.view());

        assert!((stats.mean[0] - 0.5).abs() < 1e-10);
        assert!((stats.mean[1] - 0.5).abs() < 1e-10);
        assert!((stats.variance()[0] - 0.25).abs() < 1e-10);
        assert!((stats.variance()[1] - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_online_stats_incremental() {
        let data1 = array![[0.0, 0.0], [0.0, 0.0]];
        let data2 = array![[1.0, 1.0], [1.0, 1.0]];

        let mut stats = OnlineStats::new(2);
        stats.update(data1.view());
        stats.update(data2.view());

        assert!((stats.mean[0] - 0.5).abs() < 1e-10);
        assert!((stats.mean[1] - 0.5).abs() < 1e-10);
        assert!((stats.variance()[0] - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_parallel_mean_var() {
        let data = array![[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0]];
        let (mean, var) = compute_mean_var_parallel(data.view());

        assert!((mean[0] - 0.5).abs() < 1e-10);
        assert!((mean[1] - 0.5).abs() < 1e-10);
        assert!((var[0] - 0.25).abs() < 1e-10);
        assert!((var[1] - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_parallel_min_max() {
        let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let (min, max) = compute_min_max_parallel(data.view());

        assert!((min[0] - 1.0).abs() < 1e-10);
        assert!((min[1] - 2.0).abs() < 1e-10);
        assert!((max[0] - 5.0).abs() < 1e-10);
        assert!((max[1] - 6.0).abs() < 1e-10);
    }
}
