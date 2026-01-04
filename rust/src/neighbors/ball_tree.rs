use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

use super::knn_utils::minkowski_distance;

/// A node in the Ball-Tree
#[derive(Clone)]
struct BallNode {
    /// Center of the bounding ball
    center: Array1<f64>,
    /// Radius of the bounding ball
    radius: f64,
    /// Left child index
    left: Option<usize>,
    /// Right child index
    right: Option<usize>,
    /// Sample indices (only populated for leaf nodes)
    indices: Vec<usize>,
}

/// Ball-Tree for efficient nearest neighbor search
/// Works with any metric, unlike KD-Tree which is limited to Minkowski
pub struct BallTree {
    nodes: Vec<BallNode>,
    data: Array2<f64>,
    leaf_size: usize,
    metric: String,
    p: f64,
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
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(Ordering::Equal)
    }
}

impl BallTree {
    /// Build a Ball-Tree from data
    pub fn build(data: Array2<f64>, leaf_size: usize, metric: &str, p: f64) -> Self {
        let n_samples = data.nrows();
        let indices: Vec<usize> = (0..n_samples).collect();

        let mut tree = BallTree {
            nodes: Vec::new(),
            data,
            leaf_size,
            metric: metric.to_string(),
            p,
        };

        tree.build_recursive(&indices);
        tree
    }

    fn build_recursive(&mut self, indices: &[usize]) -> usize {
        let node_idx = self.nodes.len();

        // Compute center and radius for this set of points
        let (center, radius) = self.compute_bounding_ball(indices);

        // Create leaf node if small enough
        if indices.len() <= self.leaf_size {
            self.nodes.push(BallNode {
                center,
                radius,
                left: None,
                right: None,
                indices: indices.to_vec(),
            });
            return node_idx;
        }

        // Find the dimension with maximum spread
        let split_dim = self.find_split_dimension(indices);

        // Sort indices by the split dimension and partition
        let mut sorted_indices = indices.to_vec();
        sorted_indices.sort_by(|&a, &b| {
            self.data[[a, split_dim]]
                .partial_cmp(&self.data[[b, split_dim]])
                .unwrap_or(Ordering::Equal)
        });

        let mid = sorted_indices.len() / 2;
        let left_indices = &sorted_indices[..mid];
        let right_indices = &sorted_indices[mid..];

        // Create internal node (placeholder)
        self.nodes.push(BallNode {
            center,
            radius,
            left: None,
            right: None,
            indices: Vec::new(),
        });

        // Build children
        let left_child = self.build_recursive(left_indices);
        let right_child = self.build_recursive(right_indices);

        // Update node with children
        self.nodes[node_idx].left = Some(left_child);
        self.nodes[node_idx].right = Some(right_child);

        node_idx
    }

    fn compute_bounding_ball(&self, indices: &[usize]) -> (Array1<f64>, f64) {
        let n_features = self.data.ncols();

        // Compute centroid
        let mut center = Array1::zeros(n_features);
        for &idx in indices {
            center += &self.data.row(idx).to_owned();
        }
        center /= indices.len() as f64;

        // Compute radius (max distance from center to any point)
        let radius = indices
            .iter()
            .map(|&idx| minkowski_distance(center.view(), self.data.row(idx), &self.metric, self.p))
            .fold(0.0_f64, |a, b| a.max(b));

        (center, radius)
    }

    fn find_split_dimension(&self, indices: &[usize]) -> usize {
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

    /// Query k nearest neighbors for a single point
    pub fn query(&self, point: ArrayView1<f64>, k: usize) -> (Vec<usize>, Vec<f64>) {
        let mut heap: BinaryHeap<Neighbor> = BinaryHeap::with_capacity(k);

        self.query_recursive(0, point, k, &mut heap);

        // Extract results from heap (they come out in reverse order)
        let mut results: Vec<Neighbor> = heap.into_vec();
        results.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(Ordering::Equal)
        });

        let indices: Vec<usize> = results.iter().map(|n| n.index).collect();
        let distances: Vec<f64> = results.iter().map(|n| n.distance).collect();

        (indices, distances)
    }

    fn query_recursive(
        &self,
        node_idx: usize,
        point: ArrayView1<f64>,
        k: usize,
        heap: &mut BinaryHeap<Neighbor>,
    ) {
        let node = &self.nodes[node_idx];

        // Distance from query point to ball center
        let dist_to_center = minkowski_distance(point, node.center.view(), &self.metric, self.p);

        // Pruning: if the closest possible point in this ball is farther than
        // our k-th nearest neighbor, skip this subtree
        let min_possible_dist = (dist_to_center - node.radius).max(0.0);
        if heap.len() >= k {
            if let Some(worst) = heap.peek() {
                if min_possible_dist >= worst.distance {
                    return;
                }
            }
        }

        // Leaf node: check all points
        if node.left.is_none() && node.right.is_none() {
            for &idx in &node.indices {
                let dist = minkowski_distance(point, self.data.row(idx), &self.metric, self.p);

                if heap.len() < k {
                    heap.push(Neighbor {
                        index: idx,
                        distance: dist,
                    });
                } else if let Some(worst) = heap.peek() {
                    if dist < worst.distance {
                        heap.pop();
                        heap.push(Neighbor {
                            index: idx,
                            distance: dist,
                        });
                    }
                }
            }
            return;
        }

        // Internal node: visit children
        // Visit the closer child first for better pruning
        let left_dist = node
            .left
            .map(|l| minkowski_distance(point, self.nodes[l].center.view(), &self.metric, self.p));
        let right_dist = node
            .right
            .map(|r| minkowski_distance(point, self.nodes[r].center.view(), &self.metric, self.p));

        let (first_child, second_child) = match (left_dist, right_dist) {
            (Some(ld), Some(rd)) if ld <= rd => (node.left, node.right),
            (Some(_), Some(_)) => (node.right, node.left),
            (Some(_), None) => (node.left, None),
            (None, Some(_)) => (node.right, None),
            (None, None) => (None, None),
        };

        if let Some(child) = first_child {
            self.query_recursive(child, point, k, heap);
        }
        if let Some(child) = second_child {
            self.query_recursive(child, point, k, heap);
        }
    }

    /// Query k nearest neighbors for multiple points (parallel)
    pub fn query_batch_parallel(
        &self,
        points: ArrayView2<f64>,
        k: usize,
    ) -> (Array2<usize>, Array2<f64>) {
        use rayon::prelude::*;

        let n_samples = points.nrows();

        let results: Vec<(Vec<usize>, Vec<f64>)> = (0..n_samples)
            .into_par_iter()
            .map(|i| self.query(points.row(i), k))
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
    ) -> (Array2<usize>, Array2<f64>) {
        let n_samples = points.nrows();

        let mut indices = Array2::zeros((n_samples, k));
        let mut distances = Array2::zeros((n_samples, k));

        for i in 0..n_samples {
            let (idx, dist) = self.query(points.row(i), k);
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
