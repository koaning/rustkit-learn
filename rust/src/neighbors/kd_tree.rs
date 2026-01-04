use ndarray::{Array2, ArrayView1, ArrayView2};
use std::collections::BinaryHeap;
use std::cmp::Ordering;

use super::knn_utils::minkowski_distance;

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
    data: Array2<f64>,
    leaf_size: usize,
}

/// Helper struct for maintaining k nearest neighbors during search
#[derive(Clone, Copy)]
struct Neighbor {
    index: usize,
    distance: f64,
}

impl PartialEq for Neighbor {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
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
        self.distance.partial_cmp(&other.distance).unwrap_or(Ordering::Equal)
    }
}

impl KDTree {
    /// Build a KD-Tree from data
    pub fn build(data: Array2<f64>, leaf_size: usize) -> Self {
        let n_samples = data.nrows();
        let indices: Vec<usize> = (0..n_samples).collect();

        let mut tree = KDTree {
            nodes: Vec::new(),
            data,
            leaf_size,
        };

        tree.build_recursive(&indices, 0);
        tree
    }

    fn build_recursive(&mut self, indices: &[usize], depth: usize) -> usize {
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

        let n_features = self.data.ncols();
        let split_dim = depth % n_features;

        // Find median value along split dimension
        let mut values: Vec<f64> = indices.iter()
            .map(|&i| self.data[[i, split_dim]])
            .collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        let median_idx = values.len() / 2;
        let split_val = values[median_idx];

        // Partition indices
        let (left_indices, right_indices): (Vec<usize>, Vec<usize>) = indices.iter()
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
        let left_child = self.build_recursive(&left_indices, depth + 1);
        let right_child = self.build_recursive(&right_indices, depth + 1);

        // Update node with children
        self.nodes[node_idx].left = Some(left_child);
        self.nodes[node_idx].right = Some(right_child);

        node_idx
    }

    /// Query k nearest neighbors for a single point
    pub fn query(&self, point: ArrayView1<f64>, k: usize, metric: &str, p: f64) -> (Vec<usize>, Vec<f64>) {
        let mut heap: BinaryHeap<Neighbor> = BinaryHeap::with_capacity(k);

        self.query_recursive(0, point, k, metric, p, &mut heap);

        // Extract results from heap (they come out in reverse order)
        let mut results: Vec<Neighbor> = heap.into_vec();
        results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(Ordering::Equal));

        let indices: Vec<usize> = results.iter().map(|n| n.index).collect();
        let distances: Vec<f64> = results.iter().map(|n| n.distance).collect();

        (indices, distances)
    }

    fn query_recursive(
        &self,
        node_idx: usize,
        point: ArrayView1<f64>,
        k: usize,
        metric: &str,
        p: f64,
        heap: &mut BinaryHeap<Neighbor>,
    ) {
        let node = &self.nodes[node_idx];

        // Leaf node: check all points
        if node.split_dim.is_none() {
            for &idx in &node.indices {
                let dist = minkowski_distance(point, self.data.row(idx), metric, p);

                if heap.len() < k {
                    heap.push(Neighbor { index: idx, distance: dist });
                } else if let Some(worst) = heap.peek() {
                    if dist < worst.distance {
                        heap.pop();
                        heap.push(Neighbor { index: idx, distance: dist });
                    }
                }
            }
            return;
        }

        let split_dim = node.split_dim.unwrap();
        let split_val = node.split_val.unwrap();
        let point_val = point[split_dim];

        // Determine which child to search first
        let (first_child, second_child) = if point_val < split_val {
            (node.left, node.right)
        } else {
            (node.right, node.left)
        };

        // Search closer child first
        if let Some(child) = first_child {
            self.query_recursive(child, point, k, metric, p, heap);
        }

        // Check if we need to search the other child
        let axis_dist = (point_val - split_val).abs();
        let should_search_other = heap.len() < k ||
            heap.peek().map(|n| axis_dist < n.distance).unwrap_or(true);

        if should_search_other {
            if let Some(child) = second_child {
                self.query_recursive(child, point, k, metric, p, heap);
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

    /// Get reference to stored data
    pub fn data(&self) -> &Array2<f64> {
        &self.data
    }
}
