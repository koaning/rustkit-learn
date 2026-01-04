use ndarray::{Array2, ArrayView1, ArrayView2};
use rayon::prelude::*;

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
        a.iter()
            .zip(b.iter())
            .map(|(ai, bi)| {
                let diff = ai - bi;
                diff * diff
            })
            .sum::<f64>()
            .sqrt()
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
pub fn predict_single(
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
