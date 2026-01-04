use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use rayon::prelude::*;

/// PCG random number generator for high-quality reproducible random numbers
/// Uses the PCG-XSH-RR algorithm which has much better statistical properties than LCG
pub struct SimpleRng {
    state: u64,
    inc: u64,
}

impl SimpleRng {
    pub fn new(seed: u64) -> Self {
        let mut rng = SimpleRng {
            state: 0,
            inc: (seed << 1) | 1, // Must be odd
        };
        // Warm up the generator
        rng.next_u32();
        rng.state = rng.state.wrapping_add(seed);
        rng.next_u32();
        rng
    }

    fn next_u32(&mut self) -> u32 {
        let old_state = self.state;
        // Advance internal state
        self.state = old_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(self.inc);
        // Calculate output function (XSH-RR)
        let xorshifted = (((old_state >> 18) ^ old_state) >> 27) as u32;
        let rot = (old_state >> 59) as u32;
        xorshifted.rotate_right(rot)
    }

    pub fn next_u64(&mut self) -> u64 {
        ((self.next_u32() as u64) << 32) | (self.next_u32() as u64)
    }

    pub fn next_f64(&mut self) -> f64 {
        // Use only 53 bits for f64 mantissa precision
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }

    pub fn choice(&mut self, n: usize) -> usize {
        // Avoid modulo bias by using rejection sampling
        let range = n as u64;
        let limit = u64::MAX - (u64::MAX % range);
        loop {
            let val = self.next_u64();
            if val < limit {
                return (val % range) as usize;
            }
        }
    }
}

/// Compute squared Euclidean distance between two points - allocation-free
#[inline(always)]
pub fn squared_euclidean(a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| {
            let d = ai - bi;
            d * d
        })
        .sum()
}

/// Compute squared norms for each row of a matrix
/// Returns Array1 of shape (n_rows,)
#[inline]
fn compute_row_norms_squared(x: ArrayView2<f64>) -> Array1<f64> {
    x.map_axis(Axis(1), |row| row.dot(&row))
}

/// Compute all pairwise squared distances using the formula:
/// ||a - b||² = ||a||² + ||b||² - 2 * a·b
/// This is much faster as it uses matrix multiplication for the dot products
fn compute_pairwise_distances_squared(
    x: ArrayView2<f64>,
    centroids: ArrayView2<f64>,
) -> Array2<f64> {
    let n_samples = x.nrows();
    let n_clusters = centroids.nrows();

    // Compute ||x_i||² for all samples
    let x_norms_sq = compute_row_norms_squared(x);

    // Compute ||c_k||² for all centroids
    let c_norms_sq = compute_row_norms_squared(centroids);

    // Compute X @ C^T using general_mat_mul for efficiency
    // Result shape: (n_samples, n_clusters)
    let mut distances = Array2::zeros((n_samples, n_clusters));
    ndarray::linalg::general_mat_mul(1.0, &x, &centroids.t(), 0.0, &mut distances);

    // distances = ||x||² + ||c||² - 2 * x·c
    // Do this row-wise for better cache locality
    for i in 0..n_samples {
        let x_norm = x_norms_sq[i];
        let mut row = distances.row_mut(i);
        for (k, val) in row.iter_mut().enumerate() {
            let dist = x_norm + c_norms_sq[k] - 2.0 * *val;
            *val = dist.max(0.0);
        }
    }

    distances
}

/// Assign clusters and get min distances directly without computing full distance matrix
/// More cache-efficient for large datasets
fn assign_clusters_direct(
    x: ArrayView2<f64>,
    centroids: ArrayView2<f64>,
) -> (Array1<i64>, Array1<f64>) {
    let n_samples = x.nrows();
    let n_clusters = centroids.nrows();

    let mut labels = Array1::zeros(n_samples);
    let mut min_distances = Array1::zeros(n_samples);

    for i in 0..n_samples {
        let sample = x.row(i);
        let mut min_dist = f64::MAX;
        let mut min_idx = 0i64;

        for k in 0..n_clusters {
            let dist = squared_euclidean(sample, centroids.row(k));
            if dist < min_dist {
                min_dist = dist;
                min_idx = k as i64;
            }
        }

        labels[i] = min_idx;
        min_distances[i] = min_dist;
    }

    (labels, min_distances)
}

/// Initialize centroids using k-means++ algorithm with local trials
/// Uses the same approach as sklearn: sample n_local_trials candidates and pick the best
/// Returns initial centroids as Array2<f64> of shape (n_clusters, n_features)
pub fn initialize_centroids_kmeans_plusplus(
    x: ArrayView2<f64>,
    n_clusters: usize,
    rng: &mut SimpleRng,
) -> Array2<f64> {
    let n_samples = x.nrows();
    let n_features = x.ncols();
    let mut centroids = Array2::zeros((n_clusters, n_features));

    // Number of local trials per center (same as sklearn: 2 + log(k))
    let n_local_trials = 2 + (n_clusters as f64).ln().floor() as usize;

    // Pre-compute ||x_i||² for all samples (used in distance calculations)
    let x_norms_sq = compute_row_norms_squared(x);

    // Step 1: Choose first centroid uniformly at random
    let first_idx = rng.choice(n_samples);
    centroids.row_mut(0).assign(&x.row(first_idx));

    // Step 2: For each subsequent centroid
    let mut min_distances: Array1<f64> = Array1::from_elem(n_samples, f64::MAX);
    let mut current_pot: f64;

    for k in 1..n_clusters {
        // Update distances to nearest centroid using vectorized ops
        // ||x - c||² = ||x||² + ||c||² - 2*x·c
        let new_centroid = centroids.row(k - 1);
        let c_norm_sq = new_centroid.dot(&new_centroid);
        let x_dot_c = x.dot(&new_centroid);

        for i in 0..n_samples {
            let dist = (x_norms_sq[i] + c_norm_sq - 2.0 * x_dot_c[i]).max(0.0);
            min_distances[i] = min_distances[i].min(dist);
        }

        // Current potential (sum of squared distances)
        current_pot = min_distances.sum();

        if current_pot == 0.0 {
            // All points are at distance 0, pick randomly
            let idx = rng.choice(n_samples);
            centroids.row_mut(k).assign(&x.row(idx));
            continue;
        }

        // Build cumulative sum for efficient sampling
        let mut cumsum = Vec::with_capacity(n_samples);
        let mut acc = 0.0;
        for &d in min_distances.iter() {
            acc += d;
            cumsum.push(acc);
        }

        // Sample n_local_trials candidates proportional to D(x)^2
        // and compute their potentials in a batch
        let mut candidate_indices = Vec::with_capacity(n_local_trials);
        for _ in 0..n_local_trials {
            let threshold = rng.next_f64() * current_pot;
            // Binary search for efficiency
            let candidate_idx =
                match cumsum.binary_search_by(|v| v.partial_cmp(&threshold).unwrap()) {
                    Ok(i) => i,
                    Err(i) => i.min(n_samples - 1),
                };
            candidate_indices.push(candidate_idx);
        }

        // Build candidate matrix for batch distance computation
        let mut candidates = Array2::zeros((n_local_trials, n_features));
        for (j, &idx) in candidate_indices.iter().enumerate() {
            candidates.row_mut(j).assign(&x.row(idx));
        }

        // Compute distances from all samples to all candidates using BLAS
        // distances_sq[i, j] = ||x_i - candidate_j||²
        let c_norms_sq = compute_row_norms_squared(candidates.view());
        let mut x_dot_candidates = Array2::zeros((n_samples, n_local_trials));
        ndarray::linalg::general_mat_mul(1.0, &x, &candidates.t(), 0.0, &mut x_dot_candidates);

        // Find best candidate (one that minimizes total potential)
        let mut best_candidate_idx = candidate_indices[0];
        let mut best_pot = f64::MAX;

        for (j, &candidate_idx) in candidate_indices.iter().enumerate() {
            let mut new_pot = 0.0;
            for i in 0..n_samples {
                let dist_to_candidate =
                    (x_norms_sq[i] + c_norms_sq[j] - 2.0 * x_dot_candidates[[i, j]]).max(0.0);
                new_pot += min_distances[i].min(dist_to_candidate);
            }

            if new_pot < best_pot {
                best_pot = new_pot;
                best_candidate_idx = candidate_idx;
            }
        }

        centroids.row_mut(k).assign(&x.row(best_candidate_idx));
    }

    centroids
}

/// Initialize centroids by randomly selecting k samples
pub fn initialize_centroids_random(
    x: ArrayView2<f64>,
    n_clusters: usize,
    rng: &mut SimpleRng,
) -> Array2<f64> {
    let n_samples = x.nrows();
    let n_features = x.ncols();
    let mut centroids = Array2::zeros((n_clusters, n_features));

    // Fisher-Yates shuffle for first k elements
    let mut indices: Vec<usize> = (0..n_samples).collect();
    for i in 0..n_clusters {
        let j = i + rng.choice(n_samples - i);
        indices.swap(i, j);
    }

    for (k, &idx) in indices.iter().take(n_clusters).enumerate() {
        centroids.row_mut(k).assign(&x.row(idx));
    }

    centroids
}

/// Assign each sample to nearest centroid (parallel version)
/// Returns (labels, distances_to_nearest)
pub fn assign_clusters_parallel(
    x: ArrayView2<f64>,
    centroids: ArrayView2<f64>,
) -> (Array1<i64>, Array1<f64>) {
    let n_samples = x.nrows();

    // Compute all pairwise distances at once using matrix ops
    let distances_sq = compute_pairwise_distances_squared(x, centroids);

    // Find argmin and min for each row in parallel
    let results: Vec<(i64, f64)> = (0..n_samples)
        .into_par_iter()
        .map(|i| {
            let row = distances_sq.row(i);
            let mut min_idx = 0i64;
            let mut min_val = row[0];
            for (k, &d) in row.iter().enumerate().skip(1) {
                if d < min_val {
                    min_val = d;
                    min_idx = k as i64;
                }
            }
            (min_idx, min_val)
        })
        .collect();

    let labels: Vec<i64> = results.iter().map(|(l, _)| *l).collect();
    let distances: Vec<f64> = results.iter().map(|(_, d)| *d).collect();

    (Array1::from_vec(labels), Array1::from_vec(distances))
}

/// Assign each sample to nearest centroid (single-threaded version)
/// Uses BLAS-accelerated matrix multiplication for large datasets
pub fn assign_clusters_single(
    x: ArrayView2<f64>,
    centroids: ArrayView2<f64>,
) -> (Array1<i64>, Array1<f64>) {
    let n_samples = x.nrows();

    // For small datasets, direct computation avoids allocation overhead
    // For larger datasets, BLAS matrix multiplication is faster
    if n_samples < 500 {
        assign_clusters_direct(x, centroids)
    } else {
        // Use BLAS-accelerated pairwise distances
        let distances_sq = compute_pairwise_distances_squared(x, centroids);

        // Find argmin and min for each row (single-threaded)
        let mut labels = Array1::zeros(n_samples);
        let mut min_distances = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let row = distances_sq.row(i);
            let mut min_idx = 0i64;
            let mut min_val = row[0];
            for (k, &d) in row.iter().enumerate().skip(1) {
                if d < min_val {
                    min_val = d;
                    min_idx = k as i64;
                }
            }
            labels[i] = min_idx;
            min_distances[i] = min_val;
        }

        (labels, min_distances)
    }
}

/// Compute new centroids from cluster assignments (parallel version)
pub fn compute_centroids_parallel(
    x: ArrayView2<f64>,
    labels: ArrayView1<i64>,
    n_clusters: usize,
) -> Array2<f64> {
    let n_features = x.ncols();

    // Parallel reduction per cluster
    let results: Vec<(Array1<f64>, usize)> = (0..n_clusters)
        .into_par_iter()
        .map(|k| {
            let k_i64 = k as i64;
            let mut sum = Array1::zeros(n_features);
            let mut count = 0usize;

            for (i, &label) in labels.iter().enumerate() {
                if label == k_i64 {
                    sum = &sum + &x.row(i);
                    count += 1;
                }
            }

            (sum, count)
        })
        .collect();

    let mut centroids = Array2::zeros((n_clusters, n_features));
    for (k, (sum, count)) in results.into_iter().enumerate() {
        if count > 0 {
            centroids.row_mut(k).assign(&(&sum / count as f64));
        }
    }

    centroids
}

/// Compute new centroids from cluster assignments (single-threaded version)
/// Optimized with better memory access patterns
pub fn compute_centroids_single(
    x: ArrayView2<f64>,
    labels: ArrayView1<i64>,
    n_clusters: usize,
) -> Array2<f64> {
    let n_features = x.ncols();
    let mut centroids = Array2::zeros((n_clusters, n_features));
    let mut counts = vec![0usize; n_clusters];

    // Accumulate sums - iterate through data once
    for (i, &label) in labels.iter().enumerate() {
        let k = label as usize;
        counts[k] += 1;
        // Use += for vectorized addition
        let row = x.row(i);
        let mut centroid_row = centroids.row_mut(k);
        centroid_row += &row;
    }

    // Divide by counts
    for (k, &count) in counts.iter().enumerate() {
        if count > 0 {
            let count = count as f64;
            centroids.row_mut(k).mapv_inplace(|v| v / count);
        }
    }

    centroids
}

/// Compute inertia (sum of squared distances to nearest centroid)
pub fn compute_inertia(distances: ArrayView1<f64>) -> f64 {
    distances.sum()
}

/// Compute Frobenius norm of centroid change for convergence check
pub fn centroid_shift(old: ArrayView2<f64>, new: ArrayView2<f64>) -> f64 {
    let diff = &old - &new;
    diff.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

/// Compute distances from each sample to each centroid (parallel version)
/// Returns Array2<f64> of shape (n_samples, n_clusters)
pub fn compute_distances_parallel(x: ArrayView2<f64>, centroids: ArrayView2<f64>) -> Array2<f64> {
    let mut distances = compute_pairwise_distances_squared(x, centroids);
    // Take sqrt - can parallelize this
    distances.par_mapv_inplace(|d| d.sqrt());
    distances
}

/// Compute distances from each sample to each centroid (single-threaded version)
pub fn compute_distances_single(x: ArrayView2<f64>, centroids: ArrayView2<f64>) -> Array2<f64> {
    let mut distances = compute_pairwise_distances_squared(x, centroids);
    distances.mapv_inplace(|d| d.sqrt());
    distances
}
