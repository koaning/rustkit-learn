use ndarray::{Array2, ArrayView1, ArrayView2};
use rayon::prelude::*;

/// Distance metric enum for fast dispatch (no string matching in hot path)
#[derive(Clone, Copy, Debug)]
pub enum DistanceMetric {
    Euclidean,
    Manhattan,
    Minkowski(f64),
}

impl DistanceMetric {
    pub fn from_str(metric: &str, p: f64) -> Self {
        match metric {
            "euclidean" => DistanceMetric::Euclidean,
            "manhattan" => DistanceMetric::Manhattan,
            "minkowski" => DistanceMetric::Minkowski(p),
            _ => DistanceMetric::Euclidean,
        }
    }
}

/// Compute squared Euclidean distance (avoids sqrt for comparisons)
/// Uses explicit loop unrolling for better vectorization
#[inline(always)]
pub fn squared_euclidean_distance(a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
    // Use contiguous slice access when possible
    if let (Some(a_slice), Some(b_slice)) = (a.as_slice(), b.as_slice()) {
        squared_euclidean_distance_slice(a_slice, b_slice)
    } else {
        // Fallback for non-contiguous arrays
        a.iter()
            .zip(b.iter())
            .map(|(ai, bi)| {
                let diff = ai - bi;
                diff * diff
            })
            .sum::<f64>()
    }
}

/// Compute squared Euclidean distance from raw slices (fastest path)
/// Uses explicit loop unrolling for better auto-vectorization
#[inline(always)]
pub fn squared_euclidean_distance_slice(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();

    // Use 4-way unrolling for better SIMD utilization
    let chunks = n / 4;
    let remainder = n % 4;

    let mut sum0 = 0.0;
    let mut sum1 = 0.0;
    let mut sum2 = 0.0;
    let mut sum3 = 0.0;

    // Main loop - 4 elements at a time
    for i in 0..chunks {
        let base = i * 4;
        unsafe {
            let d0 = *a.get_unchecked(base) - *b.get_unchecked(base);
            let d1 = *a.get_unchecked(base + 1) - *b.get_unchecked(base + 1);
            let d2 = *a.get_unchecked(base + 2) - *b.get_unchecked(base + 2);
            let d3 = *a.get_unchecked(base + 3) - *b.get_unchecked(base + 3);
            sum0 += d0 * d0;
            sum1 += d1 * d1;
            sum2 += d2 * d2;
            sum3 += d3 * d3;
        }
    }

    // Handle remainder
    let base = chunks * 4;
    for i in 0..remainder {
        unsafe {
            let d = *a.get_unchecked(base + i) - *b.get_unchecked(base + i);
            sum0 += d * d;
        }
    }

    sum0 + sum1 + sum2 + sum3
}

/// Compute Minkowski distance between two vectors.
/// This is the generalized distance metric where:
/// - p=1 gives Manhattan distance
/// - p=2 gives Euclidean distance
#[inline]
pub fn minkowski_distance(a: ArrayView1<f64>, b: ArrayView1<f64>, metric: &str, p: f64) -> f64 {
    let effective_p = match metric {
        "euclidean" => 2.0,
        "manhattan" => 1.0,
        "minkowski" => p,
        _ => 2.0,
    };

    if effective_p == 2.0 {
        // Optimized Euclidean distance
        squared_euclidean_distance(a, b).sqrt()
    } else if effective_p == 1.0 {
        // Optimized Manhattan distance
        a.iter()
            .zip(b.iter())
            .map(|(ai, bi)| (ai - bi).abs())
            .sum::<f64>()
    } else {
        // General Minkowski distance
        a.iter()
            .zip(b.iter())
            .map(|(ai, bi)| (ai - bi).abs().powf(effective_p))
            .sum::<f64>()
            .powf(1.0 / effective_p)
    }
}

/// Compute distance using metric enum (faster than string matching)
#[inline]
pub fn compute_distance(a: ArrayView1<f64>, b: ArrayView1<f64>, metric: DistanceMetric) -> f64 {
    match metric {
        DistanceMetric::Euclidean => squared_euclidean_distance(a, b).sqrt(),
        DistanceMetric::Manhattan => a
            .iter()
            .zip(b.iter())
            .map(|(ai, bi)| (ai - bi).abs())
            .sum::<f64>(),
        DistanceMetric::Minkowski(p) => a
            .iter()
            .zip(b.iter())
            .map(|(ai, bi)| (ai - bi).abs().powf(p))
            .sum::<f64>()
            .powf(1.0 / p),
    }
}

/// Find indices of k smallest values in a slice.
/// Returns (indices, distances) sorted by distance.
pub fn find_k_nearest(distances: &[f64], k: usize) -> (Vec<usize>, Vec<f64>) {
    let k = k.min(distances.len());

    // Create (index, distance) pairs
    let mut indexed: Vec<(usize, f64)> = distances.iter().copied().enumerate().collect();

    // Partial sort to get k smallest
    indexed.select_nth_unstable_by(k.saturating_sub(1), |a, b| {
        a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
    });

    // Take first k and sort them
    let mut nearest: Vec<(usize, f64)> = indexed.into_iter().take(k).collect();
    nearest.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let indices: Vec<usize> = nearest.iter().map(|(i, _)| *i).collect();
    let dists: Vec<f64> = nearest.iter().map(|(_, d)| *d).collect();

    (indices, dists)
}

/// Compute predictions using k-nearest neighbors (parallel version).
/// Returns predictions for each test sample.
pub fn predict_parallel(
    x_test: ArrayView2<f64>,
    x_train: ArrayView2<f64>,
    y_train: ArrayView1<f64>,
    k: usize,
    weights: &str,
    metric: &str,
    p: f64,
) -> Vec<f64> {
    let n_test = x_test.nrows();
    let n_train = x_train.nrows();
    let use_distance_weights = weights == "distance";

    (0..n_test)
        .into_par_iter()
        .map(|i| {
            let test_row = x_test.row(i);

            // Compute distances to all training samples
            let distances: Vec<f64> = (0..n_train)
                .map(|j| {
                    let train_row = x_train.row(j);
                    minkowski_distance(test_row, train_row, metric, p)
                })
                .collect();

            // Find k nearest
            let (indices, dists) = find_k_nearest(&distances, k);

            // Compute prediction
            if use_distance_weights {
                // Distance-weighted average
                let mut weighted_sum = 0.0;
                let mut weight_total = 0.0;

                for (idx, dist) in indices.iter().zip(dists.iter()) {
                    let weight = if *dist == 0.0 {
                        // If distance is 0, give infinite weight (will dominate)
                        f64::INFINITY
                    } else {
                        1.0 / dist
                    };

                    if weight.is_infinite() {
                        // Exact match - return this value
                        return y_train[*idx];
                    }

                    weighted_sum += weight * y_train[*idx];
                    weight_total += weight;
                }

                weighted_sum / weight_total
            } else {
                // Uniform weights - simple average
                let sum: f64 = indices.iter().map(|&idx| y_train[idx]).sum();
                sum / indices.len() as f64
            }
        })
        .collect()
}

/// Compute predictions using k-nearest neighbors (single-threaded version).
/// Optimized for Euclidean distance using squared distance computations.
pub fn predict_single(
    x_test: ArrayView2<f64>,
    x_train: ArrayView2<f64>,
    y_train: ArrayView1<f64>,
    k: usize,
    weights: &str,
    metric: &str,
    p: f64,
) -> Vec<f64> {
    // Use optimized path for Euclidean distance
    if metric == "euclidean" || (metric == "minkowski" && p == 2.0) {
        predict_single_euclidean(x_test, x_train, y_train, k, weights == "distance")
    } else {
        predict_single_general(x_test, x_train, y_train, k, weights, metric, p)
    }
}

/// Optimized brute-force KNN for Euclidean distance
/// Uses the formula: ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a·b
/// This leverages precomputation of norms and dot products
fn predict_single_euclidean(
    x_test: ArrayView2<f64>,
    x_train: ArrayView2<f64>,
    y_train: ArrayView1<f64>,
    k: usize,
    use_distance_weights: bool,
) -> Vec<f64> {
    let n_test = x_test.nrows();
    let n_train = x_train.nrows();
    let n_features = x_test.ncols();

    // Precompute squared norms for training data
    let train_sq_norms: Vec<f64> = (0..n_train)
        .map(|j| {
            let row = x_train.row(j);
            row.iter().map(|&x| x * x).sum()
        })
        .collect();

    let mut sq_distances = vec![0.0f64; n_train];
    let mut predictions = Vec::with_capacity(n_test);

    // Get contiguous slices if possible for faster access
    let x_test_contig = x_test.as_slice();
    let x_train_contig = x_train.as_slice();

    for i in 0..n_test {
        // Compute test point squared norm
        let test_sq_norm: f64 = if let Some(test_slice) = x_test_contig {
            let test_start = i * n_features;
            test_slice[test_start..test_start + n_features]
                .iter()
                .map(|&x| x * x)
                .sum()
        } else {
            x_test.row(i).iter().map(|&x| x * x).sum()
        };

        // Compute squared distances: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a·b
        if let (Some(test_slice), Some(train_slice)) = (x_test_contig, x_train_contig) {
            let test_start = i * n_features;
            let test_row = &test_slice[test_start..test_start + n_features];

            for j in 0..n_train {
                let train_start = j * n_features;
                let train_row = &train_slice[train_start..train_start + n_features];

                // Compute dot product
                let mut dot = 0.0;
                for f in 0..n_features {
                    dot += test_row[f] * train_row[f];
                }

                sq_distances[j] = (test_sq_norm + train_sq_norms[j] - 2.0 * dot).max(0.0);
            }
        } else {
            let test_row = x_test.row(i);
            for j in 0..n_train {
                let train_row = x_train.row(j);
                let dot: f64 = test_row
                    .iter()
                    .zip(train_row.iter())
                    .map(|(&a, &b)| a * b)
                    .sum();
                sq_distances[j] = (test_sq_norm + train_sq_norms[j] - 2.0 * dot).max(0.0);
            }
        }

        // Find k nearest using squared distances (no sqrt needed for comparison)
        let (indices, sq_dists) = find_k_nearest(&sq_distances, k);

        // Compute prediction
        let pred = if use_distance_weights {
            let mut weighted_sum = 0.0;
            let mut weight_total = 0.0;

            for (&idx, &sq_dist) in indices.iter().zip(sq_dists.iter()) {
                if sq_dist == 0.0 {
                    weighted_sum = y_train[idx];
                    weight_total = 1.0;
                    break;
                }
                // weight = 1/dist = 1/sqrt(sq_dist)
                let weight = 1.0 / sq_dist.sqrt();
                weighted_sum += weight * y_train[idx];
                weight_total += weight;
            }

            weighted_sum / weight_total
        } else {
            let sum: f64 = indices.iter().map(|&idx| y_train[idx]).sum();
            sum / indices.len() as f64
        };

        predictions.push(pred);
    }

    predictions
}

/// General brute-force KNN for non-Euclidean metrics
fn predict_single_general(
    x_test: ArrayView2<f64>,
    x_train: ArrayView2<f64>,
    y_train: ArrayView1<f64>,
    k: usize,
    weights: &str,
    metric: &str,
    p: f64,
) -> Vec<f64> {
    let n_test = x_test.nrows();
    let n_train = x_train.nrows();
    let use_distance_weights = weights == "distance";

    let mut predictions = Vec::with_capacity(n_test);

    for i in 0..n_test {
        let test_row = x_test.row(i);

        // Compute distances to all training samples
        let distances: Vec<f64> = (0..n_train)
            .map(|j| {
                let train_row = x_train.row(j);
                minkowski_distance(test_row, train_row, metric, p)
            })
            .collect();

        // Find k nearest
        let (indices, dists) = find_k_nearest(&distances, k);

        // Compute prediction
        let pred = if use_distance_weights {
            let mut weighted_sum = 0.0;
            let mut weight_total = 0.0;

            for (idx, dist) in indices.iter().zip(dists.iter()) {
                let weight = if *dist == 0.0 {
                    f64::INFINITY
                } else {
                    1.0 / dist
                };

                if weight.is_infinite() {
                    weighted_sum = y_train[*idx];
                    weight_total = 1.0;
                    break;
                }

                weighted_sum += weight * y_train[*idx];
                weight_total += weight;
            }

            weighted_sum / weight_total
        } else {
            let sum: f64 = indices.iter().map(|&idx| y_train[idx]).sum();
            sum / indices.len() as f64
        };

        predictions.push(pred);
    }

    predictions
}

/// Find k nearest neighbors for each test sample (parallel version).
/// Returns (distances, indices) where each is a 2D array of shape (n_test, k).
pub fn kneighbors_parallel(
    x_test: ArrayView2<f64>,
    x_train: ArrayView2<f64>,
    k: usize,
    metric: &str,
    p: f64,
) -> (Array2<f64>, Array2<usize>) {
    let n_test = x_test.nrows();
    let n_train = x_train.nrows();
    let k = k.min(n_train);

    let results: Vec<(Vec<f64>, Vec<usize>)> = (0..n_test)
        .into_par_iter()
        .map(|i| {
            let test_row = x_test.row(i);

            let distances: Vec<f64> = (0..n_train)
                .map(|j| {
                    let train_row = x_train.row(j);
                    minkowski_distance(test_row, train_row, metric, p)
                })
                .collect();

            let (indices, dists) = find_k_nearest(&distances, k);
            (dists, indices)
        })
        .collect();

    let mut dist_result = Array2::zeros((n_test, k));
    let mut idx_result = Array2::zeros((n_test, k));

    for (i, (dists, indices)) in results.into_iter().enumerate() {
        for (j, (d, idx)) in dists.into_iter().zip(indices.into_iter()).enumerate() {
            dist_result[[i, j]] = d;
            idx_result[[i, j]] = idx;
        }
    }

    (dist_result, idx_result)
}

/// Find k nearest neighbors for each test sample (single-threaded version).
pub fn kneighbors_single(
    x_test: ArrayView2<f64>,
    x_train: ArrayView2<f64>,
    k: usize,
    metric: &str,
    p: f64,
) -> (Array2<f64>, Array2<usize>) {
    let n_test = x_test.nrows();
    let n_train = x_train.nrows();
    let k = k.min(n_train);

    let mut dist_result = Array2::zeros((n_test, k));
    let mut idx_result = Array2::zeros((n_test, k));

    for i in 0..n_test {
        let test_row = x_test.row(i);

        let distances: Vec<f64> = (0..n_train)
            .map(|j| {
                let train_row = x_train.row(j);
                minkowski_distance(test_row, train_row, metric, p)
            })
            .collect();

        let (indices, dists) = find_k_nearest(&distances, k);

        for (j, (d, idx)) in dists.into_iter().zip(indices.into_iter()).enumerate() {
            dist_result[[i, j]] = d;
            idx_result[[i, j]] = idx;
        }
    }

    (dist_result, idx_result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_euclidean_distance() {
        let a = array![0.0, 0.0];
        let b = array![3.0, 4.0];
        let dist = minkowski_distance(a.view(), b.view(), "euclidean", 2.0);
        assert!((dist - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_manhattan_distance() {
        let a = array![0.0, 0.0];
        let b = array![3.0, 4.0];
        let dist = minkowski_distance(a.view(), b.view(), "manhattan", 1.0);
        assert!((dist - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_minkowski_distance_p3() {
        let a = array![0.0, 0.0];
        let b = array![3.0, 4.0];
        let dist = minkowski_distance(a.view(), b.view(), "minkowski", 3.0);
        let expected = (27.0_f64 + 64.0_f64).powf(1.0 / 3.0);
        assert!((dist - expected).abs() < 1e-10);
    }

    #[test]
    fn test_find_k_nearest() {
        let distances = vec![5.0, 1.0, 3.0, 2.0, 4.0];
        let (indices, dists) = find_k_nearest(&distances, 3);

        assert_eq!(indices, vec![1, 3, 2]); // Indices of 1.0, 2.0, 3.0
        assert_eq!(dists, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_find_k_nearest_k_larger_than_len() {
        let distances = vec![3.0, 1.0, 2.0];
        let (indices, dists) = find_k_nearest(&distances, 10);

        assert_eq!(indices.len(), 3);
        assert_eq!(indices, vec![1, 2, 0]);
        assert_eq!(dists, vec![1.0, 2.0, 3.0]);
    }
}
