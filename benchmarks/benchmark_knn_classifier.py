import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # KNeighborsClassifier Benchmarks

    This notebook compares the performance of `rklearn.KNeighborsClassifier` (Rust-based) against `sklearn.neighbors.KNeighborsClassifier`.

    **Benchmark Configuration:**
    - 20 iterations per benchmark (after 5 warmup runs)
    - Compares single-threaded and parallel execution
    - Measures both fit and predict times
    - Predict uses fixed 1000 samples to isolate training size effect
    """)
    return


@app.cell
def _():
    import polars as pl
    import altair as alt
    return alt, pl


@app.cell
def _():
    from sklearn.neighbors import KNeighborsClassifier as SklearnKNN
    from rklearn import KNeighborsClassifier as RklearnKNN
    return RklearnKNN, SklearnKNN


@app.cell
def _():
    from utils import benchmark_estimator
    return (benchmark_estimator,)


@app.cell
def _():
    # Configuration
    SAMPLE_SIZES = [1_000, 10_000]
    FEATURE_SIZES = [10]
    return FEATURE_SIZES, SAMPLE_SIZES


@app.cell
def _(
    FEATURE_SIZES,
    RklearnKNN,
    SAMPLE_SIZES,
    SklearnKNN,
    benchmark_estimator,
):
    models = [
        ("sklearn", SklearnKNN(n_neighbors=5, n_jobs=1)),
        ("rklearn", RklearnKNN(n_neighbors=5, n_jobs=1)),
        ("rklearn_parallel", RklearnKNN(n_neighbors=5, n_jobs=-1)),
    ]

    print("Running KNN Classifier benchmarks:")
    results = benchmark_estimator(
        models,
        sample_sizes=SAMPLE_SIZES,
        feature_sizes=FEATURE_SIZES,
        task="classification",
    )
    print("Done!")
    return (results,)


@app.cell
def _(pl, results):
    df = pl.DataFrame(results)
    df
    return (df,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Fit Time Analysis
    """)
    return


@app.cell
def _(alt, df, pl):
    fit_df = df.filter(pl.col("operation") == "fit")
    _fit_pd = fit_df.to_pandas()
    _fit_pd["config"] = _fit_pd["n_features"].astype(str) + " features"

    _chart = alt.Chart(_fit_pd).mark_bar().encode(
        x=alt.X("n_samples:O", title="Training Samples"),
        y=alt.Y("mean_ms:Q", title="Time (ms)"),
        color=alt.Color("library:N", title="Library"),
        xOffset="library:N",
        column=alt.Column("config:N", title=""),
    ).properties(
        width=200,
        height=250,
        title="Fit Time"
    )
    _chart
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Predict Time Analysis

    Predict benchmarked on 1000 samples to isolate the effect of training set size.
    """)
    return


@app.cell
def _(alt, df, pl):
    predict_df = df.filter(pl.col("operation") == "predict")
    _pred_pd = predict_df.to_pandas()
    _pred_pd["config"] = _pred_pd["n_features"].astype(str) + " features"

    _chart = alt.Chart(_pred_pd).mark_bar().encode(
        x=alt.X("n_samples:O", title="Training Samples"),
        y=alt.Y("mean_ms:Q", title="Time (ms)"),
        color=alt.Color("library:N", title="Library"),
        xOffset="library:N",
        column=alt.Column("config:N", title=""),
    ).properties(
        width=200,
        height=250,
        title="Predict Time (1000 samples)"
    )
    _chart
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Speedup Analysis

    Values > 1 mean rklearn is faster than sklearn.
    """)
    return


@app.cell
def _(df, pl):
    # Calculate speedup
    sklearn_df = df.filter(
        (pl.col("library") == "sklearn")
    ).select(["n_samples", "n_features", "operation", "mean_ms"]).rename({"mean_ms": "sklearn_ms"})

    rklearn_df = df.filter(
        (pl.col("library") == "rklearn")
    ).select(["n_samples", "n_features", "operation", "mean_ms"]).rename({"mean_ms": "rklearn_ms"})

    rklearn_parallel_df = df.filter(
        (pl.col("library") == "rklearn_parallel")
    ).select(["n_samples", "n_features", "operation", "mean_ms"]).rename({"mean_ms": "rklearn_parallel_ms"})

    speedup_df = (
        sklearn_df
        .join(rklearn_df, on=["n_samples", "n_features", "operation"])
        .join(rklearn_parallel_df, on=["n_samples", "n_features", "operation"])
        .with_columns([
            (pl.col("sklearn_ms") / pl.col("rklearn_ms")).alias("speedup_single"),
            (pl.col("sklearn_ms") / pl.col("rklearn_parallel_ms")).alias("speedup_parallel"),
        ])
    )
    speedup_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Usage

    ```python
    from rklearn import KNeighborsClassifier

    # Single-threaded (fair comparison with sklearn)
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=1)

    # Parallel execution for faster processing
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)

    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    probabilities = knn.predict_proba(X_test)
    ```
    """)
    return


if __name__ == "__main__":
    app.run()
