"""Benchmarking utilities for rklearn."""

import time
import numpy as np


def benchmark_fn(fn, n_runs=20, warmup=5):
    """Benchmark a function with warmup runs.

    Args:
        fn: Function to benchmark (no arguments)
        n_runs: Number of timed runs
        warmup: Number of warmup runs before timing

    Returns:
        Tuple of (mean_ms, std_ms, median_ms)
    """
    for _ in range(warmup):
        fn()

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        fn()
        end = time.perf_counter()
        times.append((end - start) * 1000)

    return np.mean(times), np.std(times), np.median(times)


def generate_regression_data(n_samples, n_features, seed=42):
    """Generate random regression data (X, y)."""
    np.random.seed(seed)
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples)
    return X, y


def generate_classification_data(n_samples, n_features, n_classes=2, seed=42):
    """Generate random classification data (X, y)."""
    np.random.seed(seed)
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, n_classes, n_samples)
    return X, y


def generate_features(n_samples, n_features, seed=42):
    """Generate random feature data (X only)."""
    np.random.seed(seed)
    return np.random.randn(n_samples, n_features)


def benchmark_estimator(models, sample_sizes, feature_sizes, predict_size=1000):
    """Benchmark estimator models (fit + predict) across different data configurations.

    Args:
        models: List of (name, model_instance) tuples
        sample_sizes: List of training sample sizes
        feature_sizes: List of feature counts
        predict_size: Fixed size for predict benchmark (default 1000)

    Returns:
        List of result dicts with timing info for both fit and predict
    """
    results = []

    for n_samples in sample_sizes:
        for n_features in feature_sizes:
            print(f"  samples={n_samples:,}, features={n_features}...", end=" ", flush=True)

            X_train, y_train = generate_regression_data(n_samples, n_features, seed=42)
            X_pred, _ = generate_regression_data(predict_size, n_features, seed=123)
            base_result = {"n_samples": n_samples, "n_features": n_features}

            for name, model in models:
                # Clone the model for fresh fit
                model_clone = model.__class__(**model.get_params())

                # Benchmark fit
                mean_t, std_t, median_t = benchmark_fn(lambda m=model_clone, X=X_train, y=y_train: m.fit(X, y))
                results.append({
                    **base_result,
                    "library": name,
                    "operation": "fit",
                    "mean_ms": mean_t,
                    "std_ms": std_t,
                    "median_ms": median_t,
                })

                # Fit once for predict benchmark
                model_clone.fit(X_train, y_train)

                # Benchmark predict (fixed size)
                mean_t, std_t, median_t = benchmark_fn(lambda m=model_clone, X=X_pred: m.predict(X))
                results.append({
                    **base_result,
                    "library": name,
                    "operation": "predict",
                    "mean_ms": mean_t,
                    "std_ms": std_t,
                    "median_ms": median_t,
                })

            print("done")

    return results


def benchmark_transformer(models, sample_sizes, feature_sizes):
    """Benchmark transformer models (fit, transform, fit_transform) across different data configurations.

    Args:
        models: List of (name, model_instance) tuples
        sample_sizes: List of sample sizes
        feature_sizes: List of feature counts

    Returns:
        List of result dicts with timing info
    """
    results = []

    for n_samples in sample_sizes:
        for n_features in feature_sizes:
            print(f"  samples={n_samples:,}, features={n_features}...", end=" ", flush=True)

            X = generate_features(n_samples, n_features, seed=42)
            base_result = {"n_samples": n_samples, "n_features": n_features}

            for name, model in models:
                # Clone the model for fresh fit
                model_clone = model.__class__(**model.get_params())

                # Benchmark fit
                mean_t, std_t, median_t = benchmark_fn(lambda m=model_clone, X=X: m.fit(X))
                results.append({
                    **base_result,
                    "library": name,
                    "operation": "fit",
                    "mean_ms": mean_t,
                    "std_ms": std_t,
                    "median_ms": median_t,
                })

                # Fit once for transform benchmark
                model_clone.fit(X)

                # Benchmark transform
                mean_t, std_t, median_t = benchmark_fn(lambda m=model_clone, X=X: m.transform(X))
                results.append({
                    **base_result,
                    "library": name,
                    "operation": "transform",
                    "mean_ms": mean_t,
                    "std_ms": std_t,
                    "median_ms": median_t,
                })

                # Benchmark fit_transform (fresh clone each time)
                mean_t, std_t, median_t = benchmark_fn(
                    lambda m=model, X=X: m.__class__(**m.get_params()).fit_transform(X)
                )
                results.append({
                    **base_result,
                    "library": name,
                    "operation": "fit_transform",
                    "mean_ms": mean_t,
                    "std_ms": std_t,
                    "median_ms": median_t,
                })

            print("done")

    return results
