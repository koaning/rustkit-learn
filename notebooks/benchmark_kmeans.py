import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # KMeans Benchmark: rklearn vs sklearn

    This notebook compares the performance of the Rust-based `rklearn.KMeans`
    implementation against `sklearn.cluster.KMeans`.
    """)
    return


@app.cell
def _():
    import numpy as np
    import time
    from sklearn.cluster import KMeans as SklearnKMeans
    from rklearn.cluster import KMeans as RklearnKMeans
    return RklearnKMeans, SklearnKMeans, np, time


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Generate Test Data
    """)
    return


@app.cell
def _(np):
    def generate_clustered_data(n_samples, n_features, n_clusters, random_state=42):
        """Generate synthetic clustered data."""
        np.random.seed(random_state)
        centers = np.random.randn(n_clusters, n_features) * 10
        X = np.vstack(
            [
                np.random.randn(n_samples // n_clusters, n_features) + center
                for center in centers
            ]
        )
        return X
    return (generate_clustered_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Benchmark Function
    """)
    return


@app.cell
def _(RklearnKMeans, SklearnKMeans, np, time):
    def benchmark_kmeans(X, n_clusters, n_runs=5):
        """Benchmark both implementations and return timing results."""
        results = {}

        # Benchmark sklearn
        sklearn_times = []
        for _ in range(n_runs):
            kmeans = SklearnKMeans(n_clusters=n_clusters, n_init=1, random_state=42)
            start = time.perf_counter()
            kmeans.fit(X)
            sklearn_times.append(time.perf_counter() - start)
        results["sklearn"] = {
            "mean": np.mean(sklearn_times),
            "std": np.std(sklearn_times),
            "inertia": kmeans.inertia_,
        }

        # Benchmark rklearn (single-threaded)
        rklearn_single_times = []
        for _ in range(n_runs):
            kmeans = RklearnKMeans(
                n_clusters=n_clusters, n_init=1, n_jobs=1, random_state=42
            )
            start = time.perf_counter()
            kmeans.fit(X)
            rklearn_single_times.append(time.perf_counter() - start)
        results["rklearn_single"] = {
            "mean": np.mean(rklearn_single_times),
            "std": np.std(rklearn_single_times),
            "inertia": kmeans.inertia_,
        }

        # Benchmark rklearn (parallel)
        rklearn_parallel_times = []
        for _ in range(n_runs):
            kmeans = RklearnKMeans(
                n_clusters=n_clusters, n_init=1, n_jobs=-1, random_state=42
            )
            start = time.perf_counter()
            kmeans.fit(X)
            rklearn_parallel_times.append(time.perf_counter() - start)
        results["rklearn_parallel"] = {
            "mean": np.mean(rklearn_parallel_times),
            "std": np.std(rklearn_parallel_times),
            "inertia": kmeans.inertia_,
        }

        return results
    return (benchmark_kmeans,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Run Benchmarks
    """)
    return


@app.cell
def _(benchmark_kmeans, generate_clustered_data):
    # Test configurations
    _configs = [
        {"n_samples": 1_000, "n_features": 10, "n_clusters": 5},
        {"n_samples": 10_000, "n_features": 10, "n_clusters": 10},
        {"n_samples": 50_000, "n_features": 20, "n_clusters": 10},
        # {"n_samples": 100_000, "n_features": 50, "n_clusters": 20},
    ]

    benchmark_results = []
    for _config in _configs:
        _X = generate_clustered_data(**_config)
        _results = benchmark_kmeans(_X, _config["n_clusters"])
        benchmark_results.append({"config": _config, "results": _results})
        print(_config)
    return (benchmark_results,)


@app.cell
def _(benchmark_results, mo):
    # Format results as a table
    _rows = []
    for _item in benchmark_results:
        _config = _item["config"]
        _results = _item["results"]
        _sklearn_time = _results["sklearn"]["mean"]

        for _name, _data in _results.items():
            _speedup = _sklearn_time / _data["mean"] if _name != "sklearn" else 1.0
            _rows.append(
                {
                    "Dataset": f"{_config['n_samples']:,} x {_config['n_features']}",
                    "Clusters": _config["n_clusters"],
                    "Implementation": _name,
                    "Time (ms)": f"{_data['mean']*1000:.2f}",
                    "Std (ms)": f"{_data['std']*1000:.2f}",
                    "Speedup": f"{_speedup:.2f}x",
                    "Inertia": f"{_data['inertia']:.2e}",
                }
            )

    mo.ui.table(_rows)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Scaling Analysis
    """)
    return


@app.cell
def _(benchmark_kmeans, generate_clustered_data):
    # Test how performance scales with number of samples
    sample_sizes = [1000, 5000, 10000, 25000, 50000, 100000]
    n_features = 20
    n_clusters = 10

    scaling_results = []
    for n_samples in sample_sizes:
        X_scale = generate_clustered_data(n_samples, n_features, n_clusters)
        results_scale = benchmark_kmeans(X_scale, n_clusters, n_runs=3)
        scaling_results.append(
            {
                "n_samples": n_samples,
                **{f"{k}_time": v["mean"] for k, v in results_scale.items()},
            }
        )
    return (scaling_results,)


@app.cell
def _(mo, scaling_results):
    # Display scaling results as a table
    scaling_rows = []
    for r in scaling_results:
        speedup_single = r["sklearn_time"] / r["rklearn_single_time"]
        speedup_parallel = r["sklearn_time"] / r["rklearn_parallel_time"]
        scaling_rows.append(
            {
                "Samples": f"{r['n_samples']:,}",
                "sklearn (ms)": f"{r['sklearn_time']*1000:.2f}",
                "rklearn single (ms)": f"{r['rklearn_single_time']*1000:.2f}",
                "rklearn parallel (ms)": f"{r['rklearn_parallel_time']*1000:.2f}",
                "Speedup (single)": f"{speedup_single:.2f}x",
                "Speedup (parallel)": f"{speedup_parallel:.2f}x",
            }
        )

    mo.ui.table(scaling_rows)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Verify Correctness

    Check that both implementations produce similar clustering results.
    """)
    return


@app.cell
def _(RklearnKMeans, SklearnKMeans, generate_clustered_data):
    # Generate test data
    X_test = generate_clustered_data(1000, 10, 5)

    # Fit both models
    sklearn_kmeans = SklearnKMeans(n_clusters=5, n_init=10, random_state=42)
    rklearn_kmeans = RklearnKMeans(n_clusters=5, n_init=10, random_state=42)

    sklearn_labels = sklearn_kmeans.fit_predict(X_test)
    rklearn_labels = rklearn_kmeans.fit_predict(X_test)

    correctness_results = {
        "sklearn_inertia": sklearn_kmeans.inertia_,
        "rklearn_inertia": rklearn_kmeans.inertia_,
        "inertia_ratio": rklearn_kmeans.inertia_ / sklearn_kmeans.inertia_,
        "sklearn_n_iter": sklearn_kmeans.n_iter_,
        "rklearn_n_iter": rklearn_kmeans.n_iter_,
    }
    return (correctness_results,)


@app.cell(hide_code=True)
def _(correctness_results, mo):
    mo.md(f"""
    ### Correctness Check Results

    | Metric | Value |
    |--------|-------|
    | sklearn inertia | {correctness_results['sklearn_inertia']:.4f} |
    | rklearn inertia | {correctness_results['rklearn_inertia']:.4f} |
    | Inertia ratio | {correctness_results['inertia_ratio']:.4f} |
    | sklearn n_iter | {correctness_results['sklearn_n_iter']} |
    | rklearn n_iter | {correctness_results['rklearn_n_iter']} |
    """)
    return


if __name__ == "__main__":
    app.run()
