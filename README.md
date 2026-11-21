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

## Benchmarks

Performance comparison (mean of 20 runs after 5 warmup runs, times in milliseconds, lower is better). Speedup > 1 means rklearn is faster. The `sklearn` and `rklearn` columns are single-threaded: rklearn uses `n_jobs=1` and both run inside `threadpoolctl.threadpool_limits(1)`, so numpy/scipy BLAS and OpenMP pools are capped at one thread. The parallel column uses `n_jobs=-1` without thread limits.

### Fit

| Component | sklearn | rklearn | rklearn (parallel) | Speedup | Speedup (parallel) |
|-----------|---------|---------|--------------------|---------|--------------------|
| StandardScaler | 32.78 | 28.93 | 25.94 | 1.13x | 1.26x |
| MinMaxScaler | 10.47 | 32.08 | 23.01 | 0.33x | 0.45x |
| RidgeRegressor | 36.54 | 56.74 | - | 0.64x | - |
| KNeighborsClassifier | 10.40 | 5.81 | 7.52 | 1.79x | 1.38x |
| KNeighborsRegressor | 8.67 | 5.57 | 5.57 | 1.56x | 1.55x |
| KMeans | 22.61 | 54.47 | 89.39 | 0.42x | 0.25x |

### Apply (transform/predict)

| Component | sklearn | rklearn | rklearn (parallel) | Speedup | Speedup (parallel) |
|-----------|---------|---------|--------------------|---------|--------------------|
| StandardScaler | 16.46 | 46.06 | 31.08 | 0.36x | 0.53x |
| MinMaxScaler | 13.14 | 43.40 | 30.38 | 0.30x | 0.43x |
| RidgeRegressor | 0.10 | 0.01 | - | 11.68x | - |
| KNeighborsClassifier | 91.84 | 54.82 | 14.58 | 1.68x | 6.30x |
| KNeighborsRegressor | 81.52 | 53.32 | 15.00 | 1.53x | 5.43x |
| KMeans | 0.14 | 0.03 | 0.08 | 4.45x | 1.76x |

*Benchmarks run on 100k samples x 50 features (scalers, ridge) and 10k samples x 10 features (KNN, KMeans), predicting on 1k samples, on a 4-core Linux cloud VM. Absolute numbers depend heavily on hardware.*

## License

MIT OR Apache-2.0
