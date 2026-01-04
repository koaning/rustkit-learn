use ndarray::{Array2, ArrayView1, ArrayView2};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

use super::knn_utils::{
    squared_euclidean_distance, squared_euclidean_distance_slice, DistanceMetric,
};

/// A node in the KD-Tree
#[derive(Clone)]
struct KDNode {
    /// Dimension used for splitting (None for leaf nodes)
    split_dim: Option<usize>,
    /// Split value (None for leaf nodes)
    split_val: Option<f64>,
    /// Left child index
    left: Option<usize>,
    /// Right child index
    right: Option<usize>,
    /// Sample indices (only populated for leaf nodes)
    indices: Vec<usize>,
}

/// KD-Tree for efficient nearest neighbor search
pub struct KDTree {
    nodes: Vec<KDNode>,
    /// Data stored in row-major contiguous layout
    data: Array2<f64>,
    /// Raw data pointer for direct slice access
    data_slice: Vec<f64>,
    n_features: usize,
    leaf_size: usize,
}

/// Helper struct for maintaining k nearest neighbors during search
/// Stores SQUARED distances for efficient comparison
#[derive(Clone, Copy)]
struct Neighbor {
    index: usize,
    sq_distance: f64, // squared distance
}

impl PartialEq for Neighbor {
    fn eq(&self, other: &Self) -> bool {
        self.sq_distance == other.sq_distance
    }
}

impl Eq for Neighbor {}

impl PartialOrd for Neighbor {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Neighbor {
    fn cmp(&self, other: &Self) -> Ordering {
        // Max-heap: larger distances come first
        self.sq_distance
            .partial_cmp(&other.sq_distance)
            .unwrap_or(Ordering::Equal)
    }
}

impl KDTree {
    /// Build a KD-Tree from data
    pub fn build(data: Array2<f64>, leaf_size: usize) -> Self {
        let n_samples = data.nrows();
        let n_features = data.ncols();
        let indices: Vec<usize> = (0..n_samples).collect();

        // Create contiguous row-major slice for fast access
        let data_slice: Vec<f64> = data.iter().cloned().collect();

        let mut tree = KDTree {
            nodes: Vec::new(),
            data,
            data_slice,
            n_features,
            leaf_size,
        };

        tree.build_recursive(&indices);
        tree
    }

    /// Get a row as a slice from the contiguous data
    #[inline(always)]
    fn get_row(&self, idx: usize) -> &[f64] {
        let start = idx * self.n_features;
        &self.data_slice[start..start + self.n_features]
    }

    fn build_recursive(&mut self, indices: &[usize]) -> usize {
        let node_idx = self.nodes.len();

        // Create leaf node if small enough
        if indices.len() <= self.leaf_size {
            self.nodes.push(KDNode {
                split_dim: None,
                split_val: None,
                left: None,
                right: None,
                indices: indices.to_vec(),
            });
            return node_idx;
        }

        // Find dimension with maximum spread (variance) for better splits
        let split_dim = self.find_best_split_dim(indices);

        // Find median value along split dimension
        let mut values: Vec<f64> = indices.iter().map(|&i| self.data[[i, split_dim]]).collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        let median_idx = values.len() / 2;
        let split_val = values[median_idx];

        // Partition indices
        let (left_indices, right_indices): (Vec<usize>, Vec<usize>) = indices
            .iter()
            .partition(|&&i| self.data[[i, split_dim]] < split_val);

        // Handle edge case where all values are equal
        let (left_indices, right_indices) = if left_indices.is_empty() || right_indices.is_empty() {
            let mid = indices.len() / 2;
            (indices[..mid].to_vec(), indices[mid..].to_vec())
        } else {
            (left_indices, right_indices)
        };

        // Create internal node (placeholder)
        self.nodes.push(KDNode {
            split_dim: Some(split_dim),
            split_val: Some(split_val),
            left: None,
            right: None,
            indices: Vec::new(),
        });

        // Build children
        let left_child = self.build_recursive(&left_indices);
        let right_child = self.build_recursive(&right_indices);

        // Update node with children
        self.nodes[node_idx].left = Some(left_child);
        self.nodes[node_idx].right = Some(right_child);

        node_idx
    }

    /// Find dimension with maximum spread for better partitioning
    fn find_best_split_dim(&self, indices: &[usize]) -> usize {
        let n_features = self.data.ncols();
        let mut best_dim = 0;
        let mut best_spread = 0.0;

        for dim in 0..n_features {
            let values: Vec<f64> = indices.iter().map(|&i| self.data[[i, dim]]).collect();
            let min_val = values.iter().cloned().fold(f64::INFINITY, f64::min);
            let max_val = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let spread = max_val - min_val;

            if spread > best_spread {
                best_spread = spread;
                best_dim = dim;
            }
        }

        best_dim
    }

    /// Query k nearest neighbors for a single point (Euclidean distance optimized)
    /// Returns (indices, distances) where distances are actual Euclidean distances
    #[inline]
    pub fn query(
        &self,
        point: ArrayView1<f64>,
        k: usize,
        metric: &str,
        p: f64,
    ) -> (Vec<usize>, Vec<f64>) {
        // For Euclidean, use optimized squared distance path
        if metric == "euclidean" || (metric == "minkowski" && p == 2.0) {
            self.query_euclidean(point, k)
        } else {
            self.query_general(point, k, metric, p)
        }
    }

    /// Optimized query for Euclidean distance using squared distances throughout
    fn query_euclidean(&self, point: ArrayView1<f64>, k: usize) -> (Vec<usize>, Vec<f64>) {
        let mut heap: BinaryHeap<Neighbor> = BinaryHeap::with_capacity(k);

        // Get point as slice for faster access
        if let Some(point_slice) = point.as_slice() {
            self.query_euclidean_recursive_slice(0, point_slice, k, &mut heap);
        } else {
            self.query_euclidean_recursive(0, point, k, &mut heap);
        }

        // Extract results and convert squared distances to actual distances
        let mut results: Vec<Neighbor> = heap.into_vec();
        results.sort_by(|a, b| {
            a.sq_distance
                .partial_cmp(&b.sq_distance)
                .unwrap_or(Ordering::Equal)
        });

        let indices: Vec<usize> = results.iter().map(|n| n.index).collect();
        let distances: Vec<f64> = results.iter().map(|n| n.sq_distance.sqrt()).collect();

        (indices, distances)
    }

    /// Fast path using raw slices
    #[inline]
    fn query_euclidean_recursive_slice(
        &self,
        node_idx: usize,
        point: &[f64],
        k: usize,
        heap: &mut BinaryHeap<Neighbor>,
    ) {
        let node = &self.nodes[node_idx];

        // Leaf node: check all points
        if node.split_dim.is_none() {
            for &idx in &node.indices {
                let sq_dist = squared_euclidean_distance_slice(point, self.get_row(idx));

                if heap.len() < k {
                    heap.push(Neighbor {
                        index: idx,
                        sq_distance: sq_dist,
                    });
                } else if let Some(worst) = heap.peek() {
                    if sq_dist < worst.sq_distance {
                        heap.pop();
                        heap.push(Neighbor {
                            index: idx,
                            sq_distance: sq_dist,
                        });
                    }
                }
            }
            return;
        }

        let split_dim = node.split_dim.unwrap();
        let split_val = node.split_val.unwrap();
        let point_val = point[split_dim];

        // Determine which child to search first (closer one)
        let (first_child, second_child) = if point_val < split_val {
            (node.left, node.right)
        } else {
            (node.right, node.left)
        };

        // Search closer child first
        if let Some(child) = first_child {
            self.query_euclidean_recursive_slice(child, point, k, heap);
        }

        // Pruning check: squared axis distance vs squared best distance
        let axis_diff = point_val - split_val;
        let sq_axis_dist = axis_diff * axis_diff;

        let should_search_other = heap.len() < k
            || heap
                .peek()
                .map(|n| sq_axis_dist < n.sq_distance)
                .unwrap_or(true);

        if should_search_other {
            if let Some(child) = second_child {
                self.query_euclidean_recursive_slice(child, point, k, heap);
            }
        }
    }

    /// Fallback for non-contiguous arrays
    #[inline]
    fn query_euclidean_recursive(
        &self,
        node_idx: usize,
        point: ArrayView1<f64>,
        k: usize,
        heap: &mut BinaryHeap<Neighbor>,
    ) {
        let node = &self.nodes[node_idx];

        // Leaf node: check all points
        if node.split_dim.is_none() {
            for &idx in &node.indices {
                let sq_dist = squared_euclidean_distance(point, self.data.row(idx));

                if heap.len() < k {
                    heap.push(Neighbor {
                        index: idx,
                        sq_distance: sq_dist,
                    });
                } else if let Some(worst) = heap.peek() {
                    if sq_dist < worst.sq_distance {
                        heap.pop();
                        heap.push(Neighbor {
                            index: idx,
                            sq_distance: sq_dist,
                        });
                    }
                }
            }
            return;
        }

        let split_dim = node.split_dim.unwrap();
        let split_val = node.split_val.unwrap();
        let point_val = point[split_dim];

        // Determine which child to search first (closer one)
        let (first_child, second_child) = if point_val < split_val {
            (node.left, node.right)
        } else {
            (node.right, node.left)
        };

        // Search closer child first
        if let Some(child) = first_child {
            self.query_euclidean_recursive(child, point, k, heap);
        }

        // Pruning check: squared axis distance vs squared best distance
        let axis_diff = point_val - split_val;
        let sq_axis_dist = axis_diff * axis_diff;

        let should_search_other = heap.len() < k
            || heap
                .peek()
                .map(|n| sq_axis_dist < n.sq_distance)
                .unwrap_or(true);

        if should_search_other {
            if let Some(child) = second_child {
                self.query_euclidean_recursive(child, point, k, heap);
            }
        }
    }

    /// General query for non-Euclidean metrics
    fn query_general(
        &self,
        point: ArrayView1<f64>,
        k: usize,
        metric: &str,
        p: f64,
    ) -> (Vec<usize>, Vec<f64>) {
        let dist_metric = DistanceMetric::from_str(metric, p);
        let mut heap: BinaryHeap<Neighbor> = BinaryHeap::with_capacity(k);

        self.query_general_recursive(0, point, k, dist_metric, &mut heap);

        let mut results: Vec<Neighbor> = heap.into_vec();
        results.sort_by(|a, b| {
            a.sq_distance
                .partial_cmp(&b.sq_distance)
                .unwrap_or(Ordering::Equal)
        });

        let indices: Vec<usize> = results.iter().map(|n| n.index).collect();
        // For general metrics, sq_distance is actually the real distance (not squared)
        let distances: Vec<f64> = results.iter().map(|n| n.sq_distance).collect();

        (indices, distances)
    }

    fn query_general_recursive(
        &self,
        node_idx: usize,
        point: ArrayView1<f64>,
        k: usize,
        metric: DistanceMetric,
        heap: &mut BinaryHeap<Neighbor>,
    ) {
        let node = &self.nodes[node_idx];

        if node.split_dim.is_none() {
            for &idx in &node.indices {
                let dist = super::knn_utils::compute_distance(point, self.data.row(idx), metric);

                if heap.len() < k {
                    heap.push(Neighbor {
                        index: idx,
                        sq_distance: dist,
                    });
                } else if let Some(worst) = heap.peek() {
                    if dist < worst.sq_distance {
                        heap.pop();
                        heap.push(Neighbor {
                            index: idx,
                            sq_distance: dist,
                        });
                    }
                }
            }
            return;
        }

        let split_dim = node.split_dim.unwrap();
        let split_val = node.split_val.unwrap();
        let point_val = point[split_dim];

        let (first_child, second_child) = if point_val < split_val {
            (node.left, node.right)
        } else {
            (node.right, node.left)
        };

        if let Some(child) = first_child {
            self.query_general_recursive(child, point, k, metric, heap);
        }

        // For general metrics, use axis distance as lower bound
        let axis_dist = (point_val - split_val).abs();
        let should_search_other = heap.len() < k
            || heap
                .peek()
                .map(|n| axis_dist < n.sq_distance)
                .unwrap_or(true);

        if should_search_other {
            if let Some(child) = second_child {
                self.query_general_recursive(child, point, k, metric, heap);
            }
        }
    }

    /// Query k nearest neighbors for multiple points (parallel)
    pub fn query_batch_parallel(
        &self,
        points: ArrayView2<f64>,
        k: usize,
        metric: &str,
        p: f64,
    ) -> (Array2<usize>, Array2<f64>) {
        use rayon::prelude::*;

        let n_samples = points.nrows();

        let results: Vec<(Vec<usize>, Vec<f64>)> = (0..n_samples)
            .into_par_iter()
            .map(|i| self.query(points.row(i), k, metric, p))
            .collect();

        let mut indices = Array2::zeros((n_samples, k));
        let mut distances = Array2::zeros((n_samples, k));

        for (i, (idx, dist)) in results.into_iter().enumerate() {
            for j in 0..k {
                indices[[i, j]] = idx[j];
                distances[[i, j]] = dist[j];
            }
        }

        (indices, distances)
    }

    /// Query k nearest neighbors for multiple points (single-threaded)
    pub fn query_batch_single(
        &self,
        points: ArrayView2<f64>,
        k: usize,
        metric: &str,
        p: f64,
    ) -> (Array2<usize>, Array2<f64>) {
        let n_samples = points.nrows();

        let mut indices = Array2::zeros((n_samples, k));
        let mut distances = Array2::zeros((n_samples, k));

        for i in 0..n_samples {
            let (idx, dist) = self.query(points.row(i), k, metric, p);
            for j in 0..k {
                indices[[i, j]] = idx[j];
                distances[[i, j]] = dist[j];
            }
        }

        (indices, distances)
    }
}
