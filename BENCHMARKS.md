# e-means benchmarks

Results for `rklearn.cluster.EMeans` and `rklearn.mixture.EGaussianMixture`,
produced by [`notebooks/benchmark_emeans.py`](notebooks/benchmark_emeans.py).

Reproduce with:

```bash
uv run notebooks/benchmark_emeans.py
```

## Setup

- Synthetic Gaussian mixtures with unequal cluster sizes and spreads, split 80/20
  into train and held-out sets.
- Times are best of 3 fits (best of 2 for the mixtures), wall clock, on a 4-core
  Linux container. rklearn models run with `n_jobs=-1`; sklearn uses its defaults.
- All models use `random_state=0` and a single initialization (`n_init=1`).
- EMeans and EGaussianMixture use their defaults: `n_epochs=3`, and `batch_size`
  and `alpha` from the paper's fitted laws.

## k-means

Loss is held-out inertia relative to sklearn `KMeans` (lower is better).
Speedup is sklearn `KMeans` fit time divided by the model's fit time.

| n | d | k | model | fit time (s) | held-out loss vs sklearn | speedup |
|---|---|---|---|---|---|---|
| 50,000 | 8 | 32 | sklearn KMeans | 0.189 | 0.00% | 1.0x |
| 50,000 | 8 | 32 | sklearn MiniBatchKMeans | 0.022 | -2.90% | 8.6x |
| 50,000 | 8 | 32 | rklearn KMeans | 0.670 | +1.17% | 0.3x |
| 50,000 | 8 | 32 | rklearn EMeans | 0.040 | +11.29% | 4.7x |
| 50,000 | 8 | 32 | rklearn EMeans (revival) | 0.057 | +11.32% | 3.3x |
| 100,000 | 32 | 64 | sklearn KMeans | 0.413 | 0.00% | 1.0x |
| 100,000 | 32 | 64 | sklearn MiniBatchKMeans | 0.194 | -2.58% | 2.1x |
| 100,000 | 32 | 64 | rklearn KMeans | 1.353 | -6.35% | 0.3x |
| 100,000 | 32 | 64 | rklearn EMeans | 0.196 | -9.57% | 2.1x |
| 100,000 | 32 | 64 | rklearn EMeans (revival) | 0.204 | -5.22% | 2.0x |
| 200,000 | 16 | 256 | sklearn KMeans | 2.543 | 0.00% | 1.0x |
| 200,000 | 16 | 256 | sklearn MiniBatchKMeans | 0.180 | +8.76% | 14.1x |
| 200,000 | 16 | 256 | rklearn KMeans | 15.693 | -1.06% | 0.2x |
| 200,000 | 16 | 256 | rklearn EMeans | 1.036 | -3.47% | 2.5x |
| 200,000 | 16 | 256 | rklearn EMeans (revival) | 1.007 | +2.21% | 2.5x |

Notes:

- EMeans beats sklearn's Lloyd loss on the two larger configs while being 2-2.5x
  faster. On the small 50k x 8 config it is 11% worse; this has not been
  investigated yet.
- Revival helped on none of these datasets.
- The Rust Lloyd `KMeans` is 3-6x slower than sklearn at these sizes.

## Batch size and alpha sweep

`EMeans(n_clusters=64)` on 100k x 32. `None` means the fitted law picks the value.

| batch_size | alpha | batch used | alpha used | held-out loss vs sklearn |
|---|---|---|---|---|
| None | None | 4096 | 1.000 | -2.54% |
| None | 0.05 | 4096 | 0.050 | -2.54% |
| None | 0.2 | 4096 | 0.200 | -2.54% |
| None | 0.8 | 4096 | 0.800 | -2.54% |
| 256 | None | 256 | 0.099 | +0.14% |
| 256 | 0.05 | 256 | 0.050 | +0.05% |
| 256 | 0.2 | 256 | 0.200 | -2.27% |
| 256 | 0.8 | 256 | 0.800 | -5.50% |
| 1024 | None | 1024 | 0.344 | -2.54% |
| 1024 | 0.05 | 1024 | 0.050 | -2.54% |
| 1024 | 0.2 | 1024 | 0.200 | -2.54% |
| 1024 | 0.8 | 1024 | 0.800 | -2.28% |
| 4096 | None | 4096 | 1.000 | -2.54% |
| 4096 | 0.05 | 4096 | 0.050 | -2.54% |
| 4096 | 0.2 | 4096 | 0.200 | -2.54% |
| 4096 | 0.8 | 4096 | 0.800 | -2.54% |

With batches of 1024 or more, alpha barely changes the result on this data. With
small batches it matters a lot, and the step law's pick (0.099 at batch 256) was
not the best value here.

## Gaussian mixtures (EGaussianMixture)

Metric is held-out mean log-likelihood per sample (higher is better). sklearn uses
`GaussianMixture(covariance_type="diag")` with default settings.

| n | d | k | model | fit time (s) | held-out log-likelihood | speedup |
|---|---|---|---|---|---|---|
| 50,000 | 8 | 16 | sklearn GaussianMixture (diag) | 0.329 | -14.8323 | 1.0x |
| 50,000 | 8 | 16 | rklearn EGaussianMixture | 0.118 | -14.8795 | 2.8x |
| 50,000 | 8 | 16 | rklearn EGaussianMixture (revival) | 0.105 | -14.8796 | 3.1x |
| 200,000 | 16 | 32 | sklearn GaussianMixture (diag) | 1.516 | -26.9730 | 1.0x |
| 200,000 | 16 | 32 | rklearn EGaussianMixture | 0.486 | -27.2365 | 3.1x |
| 200,000 | 16 | 32 | rklearn EGaussianMixture (revival) | 0.462 | -26.9408 | 3.3x |

Notes:

- EGaussianMixture is about 3x faster than sklearn's diagonal mixture.
- Without revival it ends slightly below sklearn's likelihood (0.05 lower on the
  small config, 0.26 lower on the larger one). With revival on, the larger config
  edges past sklearn.
- Only diagonal covariances are supported.
