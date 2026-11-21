//! Shared machinery for the e-means family of mini-batch estimators.
//!
//! Follows Algorithm 1 of Bessa et al., "e-means: outperforming k-means (in
//! one-tenth the epochs)", arXiv:2611.09501. The constants match Meta's
//! reference implementation (faiss/contrib/e_means, MIT license).

use ndarray::{Array2, ArrayView2, Axis};

use super::kmeans_utils::{initialize_centroids_kmeans_plusplus, SimpleRng};

/// epsilon in Eq. 5: the cosine schedule bottoms out at this fraction of alpha0.
pub const ALPHA_MIN_FACTOR: f64 = 0.1;
/// Revival check period, in mini-batch updates.
pub const CHECK_UNUSED_EVERY: usize = 5;
/// Initial EMA mass per cluster.
pub const INIT_USAGE: f64 = 1.0 / 65536.0;
/// Relative perturbation applied when splitting a donor cluster.
pub const SPLIT_EPSILON: f64 = 1e-4;
/// tau: revive clusters whose EMA mass is below tau * mean(mass).
pub const UNUSED_THRESHOLD: f64 = 0.1;
/// Use the smaller "quality" batch law when n_features <= this.
const AUTO_QUALITY_MAX_D: usize = 16;

/// B = c * k^b_k * n^b_n * T^b_T  (Eq. 7)
const BATCH_LAW_THROUGHPUT: (f64, f64, f64, f64) = (0.035134, 0.134749, 0.876473, 0.813344);
const BATCH_LAW_QUALITY: (f64, f64, f64, f64) = (0.00439175, 0.134749, 0.876473, 0.813344);
/// alpha0 = c * k^a_k * B^a_B * n^a_n * T^a_T  (Eq. 6)
const ALPHA_LAW: (f64, f64, f64, f64, f64) = (0.8244, -0.4617, 0.9003, -0.4498, -0.1057);

fn round_pow2(value: f64) -> usize {
    if value <= 1.0 {
        return 1;
    }
    let low = 2f64.powi(value.log2().floor() as i32);
    if value / low < 2.0 * low / value {
        low as usize
    } else {
        (2.0 * low) as usize
    }
}

/// Batch size from the fitted batch law.
pub fn batch_law(n: usize, k: usize, n_epochs: f64, d: usize) -> usize {
    let quality = d <= AUTO_QUALITY_MAX_D;
    let (c, bk, bn, bt) = if quality {
        BATCH_LAW_QUALITY
    } else {
        BATCH_LAW_THROUGHPUT
    };
    let raw = c * (k as f64).powf(bk) * (n as f64).powf(bn) * n_epochs.powf(bt);
    let mut b = round_pow2(raw);
    b = if quality {
        b.min(n / 2)
    } else {
        b.min(round_pow2(n as f64 / 2.0))
    };
    b.clamp(1, n)
}

/// Peak EMA step from the fitted step law, clamped to [1e-4, 1].
pub fn step_law(n: usize, k: usize, n_epochs: f64, batch_size: usize) -> f64 {
    let (c, ak, ab, an, at) = ALPHA_LAW;
    let a = c
        * (k as f64).powf(ak)
        * (batch_size as f64).powf(ab)
        * (n as f64).powf(an)
        * n_epochs.powf(at);
    a.clamp(1e-4, 1.0)
}

/// Rows visited per epoch. A fractional epoch runs before the last full pass,
/// so the finalization pass always sees every row.
pub fn epoch_rows(n: usize, n_epochs: f64) -> Vec<usize> {
    let whole = n_epochs.floor() as usize;
    let rem = ((n_epochs - whole as f64) * n as f64).round() as usize;
    if whole == 0 {
        return vec![rem.max(1)];
    }
    let mut out = vec![n; whole - 1];
    if rem > 0 {
        out.push(rem);
    }
    out.push(n);
    out
}

/// Cosine-annealed EMA step at `step` out of `total_steps`.
pub fn cosine_alpha(alpha0: f64, step: usize, total_steps: usize) -> f64 {
    let frac = step as f64 / total_steps.max(1) as f64;
    alpha0
        * (ALPHA_MIN_FACTOR
            + (1.0 - ALPHA_MIN_FACTOR) * 0.5 * (1.0 + (std::f64::consts::PI * frac).cos()))
}

/// Fisher-Yates shuffle of 0..n.
pub fn permutation(n: usize, rng: &mut SimpleRng) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = rng.choice(i + 1);
        idx.swap(i, j);
    }
    idx
}

/// k-means++ seeding on a random subsample of max(10000, 20k) rows.
pub fn seed_subsample_kmeans_plusplus(
    x: ArrayView2<f64>,
    k: usize,
    rng: &mut SimpleRng,
) -> Array2<f64> {
    let n = x.nrows();
    let m = n.min(10_000usize.max(20 * k));
    if m == n {
        return initialize_centroids_kmeans_plusplus(x, k, rng);
    }
    let mut idx = permutation(n, rng);
    idx.truncate(m);
    idx.sort_unstable();
    let sub = x.select(Axis(0), &idx);
    initialize_centroids_kmeans_plusplus(sub.view(), k, rng)
}

/// Pick (donor, dead) pairs for revival.
///
/// Clusters with mass below `tau * mean(mass)` are dead. Donors are drawn from
/// the live clusters without replacement, proportional to their mass
/// (Gumbel top-m).
pub fn select_revivals(mass: &[f64], tau: f64, rng: &mut SimpleRng) -> Vec<(usize, usize)> {
    let k = mass.len();
    let mean = mass.iter().sum::<f64>() / k as f64;
    let threshold = tau * mean;
    let dead: Vec<usize> = (0..k).filter(|&j| mass[j] < threshold).collect();
    let mut donors: Vec<(f64, usize)> = (0..k)
        .filter(|&j| mass[j] >= threshold && mass[j] > 0.0)
        .map(|j| {
            let u = rng.next_f64().max(f64::MIN_POSITIVE);
            (mass[j].ln() - (-u.ln()).ln(), j)
        })
        .collect();
    let m = dead.len().min(donors.len());
    if m == 0 {
        return Vec::new();
    }
    donors.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    donors
        .iter()
        .take(m)
        .zip(dead.iter())
        .map(|(&(_, src), &dst)| (src, dst))
        .collect()
}

/// Split row `src` of `centers` into `src` and `dst` with a small
/// alternating-sign relative perturbation.
pub fn split_row(centers: &mut Array2<f64>, src: usize, dst: usize) {
    let d = centers.ncols();
    let mut same = true;
    for f in 0..d {
        let off = if f % 2 == 0 {
            SPLIT_EPSILON
        } else {
            -SPLIT_EPSILON
        };
        let base = centers[[src, f]];
        centers[[src, f]] = base * (1.0 - off);
        centers[[dst, f]] = base * (1.0 + off);
        same &= centers[[src, f]] == centers[[dst, f]];
    }
    if same && d > 0 {
        // All-zero centroid: nudge one coordinate so the two can separate.
        let v = centers[[dst, 0]];
        centers[[dst, 0]] = if v == 0.0 {
            f64::MIN_POSITIVE
        } else {
            v + v.abs() * f64::EPSILON
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epoch_rows_fractional() {
        assert_eq!(epoch_rows(100, 3.0), vec![100, 100, 100]);
        assert_eq!(epoch_rows(100, 1.5), vec![50, 100]);
        assert_eq!(epoch_rows(100, 0.5), vec![50]);
    }

    #[test]
    fn cosine_schedule_endpoints() {
        assert!((cosine_alpha(1.0, 0, 10) - 1.0).abs() < 1e-12);
        assert!((cosine_alpha(1.0, 10, 10) - ALPHA_MIN_FACTOR).abs() < 1e-12);
    }

    #[test]
    fn revival_pairs_dead_with_live() {
        let mut rng = SimpleRng::new(0);
        let pairs = select_revivals(&[10.0, 10.0, 0.0001, 10.0], 0.1, &mut rng);
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].1, 2);
        assert_ne!(pairs[0].0, 2);
    }
}
