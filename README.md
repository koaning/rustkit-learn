# rklearn

Fast scikit-learn compatible components in Rust.

## Installation

```bash
uv pip install git+https://github.com/koaning/rustkit-learn.git
```

## Components

### Linear Models

- **RidgeRegressor** - Linear least squares with L2 regularization

### Preprocessing

- **StandardScaler** - Standardize features by removing mean and scaling to unit variance
- **MinMaxScaler** - Scale features to a given range (default [0, 1])

### Neighbors

- **KNeighborsClassifier** - K-nearest neighbors classification
- **KNeighborsRegressor** - K-nearest neighbors regression

### Clustering

- **KMeans** - K-Means clustering
- **EMeans** - e-means ([arXiv:2611.09501](https://arxiv.org/abs/2611.09501)): mini-batch k-means with EWM centroids. Tune `batch_size`, `alpha` (peak EWM step) and `revival` (dead-cluster revival on/off)

### Mixture

- **EGaussianMixture** - diagonal Gaussian mixture fitted with the same EWM trick (stochastic EM), in `rklearn.mixture`

## Benchmarks

Performance comparison (times in milliseconds, lower is better). Speedup > 1 means rklearn is faster.

### Fit

| Component | sklearn | rklearn | rklearn (parallel) | Speedup | Speedup (parallel) |
|-----------|---------|---------|--------------------|---------|--------------------|
| StandardScaler | 11.11 | 6.65 | 1.30 | 1.67x | 8.58x |
| MinMaxScaler | 3.67 | 4.29 | 1.04 | 0.85x | 3.54x |
| RidgeRegressor | 7.29 | 19.52 | - | 0.37x | - |
| KNeighborsClassifier | 1.56 | 2.31 | 2.20 | 0.68x | 0.71x |
| KNeighborsRegressor | 1.55 | 2.35 | 2.22 | 0.66x | 0.70x |
| KMeans | 36.00 | 34.51 | 42.46 | 1.04x | 0.85x |

### Apply (transform/predict)

| Component | sklearn | rklearn | rklearn (parallel) | Speedup | Speedup (parallel) |
|-----------|---------|---------|--------------------|---------|--------------------|
| StandardScaler | 4.25 | 4.09 | 1.66 | 1.04x | 2.56x |
| MinMaxScaler | 4.17 | 1.67 | 1.57 | 2.50x | 2.65x |
| RidgeRegressor | 0.03 | 0.01 | - | 4.05x | - |
| KNeighborsClassifier | 32.63 | 24.32 | 2.21 | 1.34x | 14.76x |
| KNeighborsRegressor | 33.63 | 25.43 | 2.27 | 1.32x | 14.78x |
| KMeans | 0.13 | 0.02 | 0.16 | 5.67x | 0.78x |

*Benchmarks run on 100k samples x 50 features (scalers, ridge) and 10k samples x 10 features (KNN, KMeans).*

### e-means and EGaussianMixture

Held-out loss compared with sklearn, using default settings on a 4-core machine:

| Model | Data (n x d, k) | Fit speedup vs sklearn | Held-out quality vs sklearn |
|-------|-----------------|------------------------|-----------------------------|
| EMeans vs KMeans | 200k x 16, k=256 | 2.5x | 3.5% lower inertia |
| EMeans vs KMeans | 100k x 32, k=64 | 2.1x | 9.6% lower inertia |
| EMeans vs KMeans | 50k x 8, k=32 | 4.7x | 11.3% higher inertia |
| EGaussianMixture vs GaussianMixture(diag) | 200k x 16, k=32 | 3.1x | log-likelihood 0.26 lower |
| EGaussianMixture vs GaussianMixture(diag) | 50k x 8, k=16 | 2.8x | log-likelihood 0.05 lower |

Full tables, including the revival on/off comparison and a `batch_size` x `alpha` sweep, are in [BENCHMARKS.md](BENCHMARKS.md).

## License

MIT OR Apache-2.0
