use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rayon::prelude::*;

/// Simple Linear Congruential Generator for reproducible random numbers
pub struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    pub fn new(seed: u64) -> Self {
        SimpleRng { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        // LCG parameters (same as glibc)
        self.state = self.state.wrapping_mul(1103515245).wrapping_add(12345);
        self.state
    }

    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() as f64) / (u64::MAX as f64)
    }

    pub fn choice(&mut self, n: usize) -> usize {
        (self.next_u64() as usize) % n
    }
}

/// Compute squared Euclidean distance between two points
#[inline]
pub fn squared_euclidean(a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(ai, bi)| {
            let diff = ai - bi;
            diff * diff
        })
        .sum()
}

/// Initialize centroids using k-means++ algorithm
/// Returns initial centroids as Array2<f64> of shape (n_clusters, n_features)
pub fn initialize_centroids_kmeans_plusplus(
    x: ArrayView2<f64>,
    n_clusters: usize,
    rng: &mut SimpleRng,
) -> Array2<f64> {
    let n_samples = x.nrows();
    let n_features = x.ncols();
    let mut centroids = Array2::zeros((n_clusters, n_features));

    // Step 1: Choose first centroid uniformly at random
    let first_idx = rng.choice(n_samples);
    centroids.row_mut(0).assign(&x.row(first_idx));

    // Step 2: For each subsequent centroid
    let mut distances = vec![f64::MAX; n_samples];

    for k in 1..n_clusters {
        // Update distances to nearest centroid
        let new_centroid = centroids.row(k - 1);
        for i in 0..n_samples {
            let dist = squared_euclidean(x.row(i), new_centroid);
            distances[i] = distances[i].min(dist);
        }

        // Sample next centroid with probability proportional to D(x)^2
        let total: f64 = distances.iter().sum();
        if total == 0.0 {
            // All points are at distance 0, pick randomly
            let idx = rng.choice(n_samples);
            centroids.row_mut(k).assign(&x.row(idx));
        } else {
            let threshold = rng.next_f64() * total;
            let mut cumsum = 0.0;
            let mut chosen_idx = 0;
            for (i, &d) in distances.iter().enumerate() {
                cumsum += d;
                if cumsum >= threshold {
                    chosen_idx = i;
                    break;
                }
            }
            centroids.row_mut(k).assign(&x.row(chosen_idx));
        }
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
    let n_clusters = centroids.nrows();

    let results: Vec<(i64, f64)> = (0..n_samples)
        .into_par_iter()
        .map(|i| {
            let sample = x.row(i);
            let mut min_dist = f64::MAX;
            let mut min_cluster: i64 = 0;

            for k in 0..n_clusters {
                let dist = squared_euclidean(sample, centroids.row(k));
                if dist < min_dist {
                    min_dist = dist;
                    min_cluster = k as i64;
                }
            }

            (min_cluster, min_dist)
        })
        .collect();

    let labels: Vec<i64> = results.iter().map(|(l, _)| *l).collect();
    let distances: Vec<f64> = results.iter().map(|(_, d)| *d).collect();

    (Array1::from_vec(labels), Array1::from_vec(distances))
}

/// Assign each sample to nearest centroid (single-threaded version)
pub fn assign_clusters_single(
    x: ArrayView2<f64>,
    centroids: ArrayView2<f64>,
) -> (Array1<i64>, Array1<f64>) {
    let n_samples = x.nrows();
    let n_clusters = centroids.nrows();

    let mut labels = Array1::zeros(n_samples);
    let mut distances = Array1::zeros(n_samples);

    for i in 0..n_samples {
        let sample = x.row(i);
        let mut min_dist = f64::MAX;
        let mut min_cluster: i64 = 0;

        for k in 0..n_clusters {
            let dist = squared_euclidean(sample, centroids.row(k));
            if dist < min_dist {
                min_dist = dist;
                min_cluster = k as i64;
            }
        }

        labels[i] = min_cluster;
        distances[i] = min_dist;
    }

    (labels, distances)
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
pub fn compute_centroids_single(
    x: ArrayView2<f64>,
    labels: ArrayView1<i64>,
    n_clusters: usize,
) -> Array2<f64> {
    let n_features = x.ncols();
    let mut centroids = Array2::zeros((n_clusters, n_features));
    let mut counts = vec![0usize; n_clusters];

    for (i, &label) in labels.iter().enumerate() {
        let k = label as usize;
        let row = x.row(i);
        for (j, &val) in row.iter().enumerate() {
            centroids[[k, j]] += val;
        }
        counts[k] += 1;
    }

    for k in 0..n_clusters {
        if counts[k] > 0 {
            let count = counts[k] as f64;
            for j in 0..n_features {
                centroids[[k, j]] /= count;
            }
        }
    }

    centroids
}

/// Compute inertia (sum of squared distances to nearest centroid)
pub fn compute_inertia(distances: ArrayView1<f64>) -> f64 {
    distances.iter().sum()
}

/// Compute Frobenius norm of centroid change for convergence check
pub fn centroid_shift(old: ArrayView2<f64>, new: ArrayView2<f64>) -> f64 {
    old.iter()
        .zip(new.iter())
        .map(|(a, b)| {
            let diff = a - b;
            diff * diff
        })
        .sum::<f64>()
        .sqrt()
}

/// Compute distances from each sample to each centroid (parallel version)
/// Returns Array2<f64> of shape (n_samples, n_clusters)
pub fn compute_distances_parallel(x: ArrayView2<f64>, centroids: ArrayView2<f64>) -> Array2<f64> {
    let n_samples = x.nrows();
    let n_clusters = centroids.nrows();

    let results: Vec<Vec<f64>> = (0..n_samples)
        .into_par_iter()
        .map(|i| {
            let sample = x.row(i);
            (0..n_clusters)
                .map(|k| squared_euclidean(sample, centroids.row(k)).sqrt())
                .collect()
        })
        .collect();

    let mut distances = Array2::zeros((n_samples, n_clusters));
    for (i, row) in results.into_iter().enumerate() {
        for (k, d) in row.into_iter().enumerate() {
            distances[[i, k]] = d;
        }
    }

    distances
}

/// Compute distances from each sample to each centroid (single-threaded version)
pub fn compute_distances_single(x: ArrayView2<f64>, centroids: ArrayView2<f64>) -> Array2<f64> {
    let n_samples = x.nrows();
    let n_clusters = centroids.nrows();

    let mut distances = Array2::zeros((n_samples, n_clusters));

    for i in 0..n_samples {
        let sample = x.row(i);
        for k in 0..n_clusters {
            distances[[i, k]] = squared_euclidean(sample, centroids.row(k)).sqrt();
        }
    }

    distances
}
