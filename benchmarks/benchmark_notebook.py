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
    # rklearn vs scikit-learn Benchmarks

    This notebook compares the performance of `rklearn` (Rust-based) against `scikit-learn` for preprocessing operations.

    **Benchmark Configuration:**
    - 20 iterations per benchmark (after 5 warmup runs)
    - Fair comparison: both rklearn (n_jobs=1) and sklearn use single-threaded computation
    - Also shows rklearn with parallel computation (n_jobs=-1) for reference
    - Measures fit, transform, and fit_transform times
    """)
    return


@app.cell
def _():
    import polars as pl
    import altair as alt
    return alt, pl


@app.cell
def _():
    from sklearn.preprocessing import StandardScaler as SklearnStandardScaler
    from sklearn.preprocessing import MinMaxScaler as SklearnMinMaxScaler
    from rklearn.preprocessing import StandardScaler as RklearnStandardScaler
    from rklearn.preprocessing import MinMaxScaler as RklearnMinMaxScaler
    return (
        RklearnMinMaxScaler,
        RklearnStandardScaler,
        SklearnMinMaxScaler,
        SklearnStandardScaler,
    )


@app.cell
def _():
    from utils import benchmark_transformer
    return (benchmark_transformer,)


@app.cell
def _():
    # Configuration
    SAMPLE_SIZES = [10_000, 100_000, 1_000_000]
    FEATURE_SIZES = [50, 100, 200]
    return FEATURE_SIZES, SAMPLE_SIZES


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Running Benchmarks

    Running benchmarks with 20 iterations each (after 5 warmup runs).

    **Libraries compared:**
    - `sklearn`: scikit-learn (single-threaded)
    - `rklearn`: rklearn with n_jobs=1 (single-threaded, fair comparison)
    - `rklearn_parallel`: rklearn with n_jobs=-1 (multi-threaded via Rayon)
    """)
    return


@app.cell
def _(
    FEATURE_SIZES,
    RklearnMinMaxScaler,
    RklearnStandardScaler,
    SAMPLE_SIZES,
    SklearnMinMaxScaler,
    SklearnStandardScaler,
    benchmark_transformer,
):
    SCALERS = [
        ("StandardScaler", SklearnStandardScaler, RklearnStandardScaler),
        ("MinMaxScaler", SklearnMinMaxScaler, RklearnMinMaxScaler),
    ]

    all_results = []
    for scaler_name, sklearn_cls, rklearn_cls in SCALERS:
        print(f"Benchmarking {scaler_name}:")
        models = [
            ("sklearn", sklearn_cls()),
            ("rklearn", rklearn_cls(n_jobs=1)),
            ("rklearn_parallel", rklearn_cls(n_jobs=-1)),
        ]
        results = benchmark_transformer(
            models,
            sample_sizes=SAMPLE_SIZES,
            feature_sizes=FEATURE_SIZES
        )
        for r in results:
            r["scaler"] = scaler_name
        all_results.extend(results)
    print("Done!")
    return SCALERS, all_results, models, results, scaler_name, sklearn_cls, rklearn_cls


@app.cell
def _(all_results, pl):
    df = pl.DataFrame(all_results)
    df
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fit Time Analysis
    """)
    return


@app.cell
def _(alt, df, pl):
    fit_df = df.filter(pl.col("operation") == "fit")
    _fit_pd = fit_df.to_pandas()
    _fit_pd["n_features_label"] = _fit_pd["n_features"].astype(str) + " cols"

    _chart = alt.Chart(_fit_pd).mark_line(point=True, strokeWidth=2).encode(
        x=alt.X("n_samples:Q", title="Number of Rows", scale=alt.Scale(type="log")),
        y=alt.Y("mean_ms:Q", title="Time (ms)"),
        color=alt.Color("library:N", title="Library"),
        column=alt.Column("scaler:N", title=""),
        row=alt.Row("n_features_label:N", title=""),
    ).properties(
        width=300,
        height=200,
        title="Fit Time"
    )
    _chart
    return (fit_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Transform Time Analysis
    """)
    return


@app.cell
def _(alt, df, pl):
    transform_df = df.filter(pl.col("operation") == "transform")
    _transform_pd = transform_df.to_pandas()
    _transform_pd["n_features_label"] = _transform_pd["n_features"].astype(str) + " cols"

    _chart = alt.Chart(_transform_pd).mark_line(point=True, strokeWidth=2).encode(
        x=alt.X("n_samples:Q", title="Number of Rows", scale=alt.Scale(type="log")),
        y=alt.Y("mean_ms:Q", title="Time (ms)"),
        color=alt.Color("library:N", title="Library"),
        column=alt.Column("scaler:N", title=""),
        row=alt.Row("n_features_label:N", title=""),
    ).properties(
        width=300,
        height=200,
        title="Transform Time"
    )
    _chart
    return (transform_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fit+Transform Time Analysis
    """)
    return


@app.cell
def _(alt, df, pl):
    fit_transform_df = df.filter(pl.col("operation") == "fit_transform")
    _ft_pd = fit_transform_df.to_pandas()
    _ft_pd["n_features_label"] = _ft_pd["n_features"].astype(str) + " cols"

    _chart = alt.Chart(_ft_pd).mark_line(point=True, strokeWidth=2).encode(
        x=alt.X("n_samples:Q", title="Number of Rows", scale=alt.Scale(type="log")),
        y=alt.Y("mean_ms:Q", title="Time (ms)"),
        color=alt.Color("library:N", title="Library"),
        column=alt.Column("scaler:N", title=""),
        row=alt.Row("n_features_label:N", title=""),
    ).properties(
        width=300,
        height=200,
        title="Fit+Transform Time"
    )
    _chart
    return (fit_transform_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Speedup Analysis

    Values > 1 mean rklearn is faster than sklearn.
    """)
    return


@app.cell
def _(df, pl):
    # Calculate speedup
    sklearn_df = df.filter(pl.col("library") == "sklearn").select(
        ["scaler", "n_samples", "n_features", "operation", "mean_ms"]
    ).rename({"mean_ms": "sklearn_ms"})

    rklearn_df = df.filter(pl.col("library") == "rklearn").select(
        ["scaler", "n_samples", "n_features", "operation", "mean_ms"]
    ).rename({"mean_ms": "rklearn_ms"})

    rklearn_parallel_df = df.filter(pl.col("library") == "rklearn_parallel").select(
        ["scaler", "n_samples", "n_features", "operation", "mean_ms"]
    ).rename({"mean_ms": "rklearn_parallel_ms"})

    speedup_df = (
        sklearn_df
        .join(rklearn_df, on=["scaler", "n_samples", "n_features", "operation"])
        .join(rklearn_parallel_df, on=["scaler", "n_samples", "n_features", "operation"])
        .with_columns([
            (pl.col("sklearn_ms") / pl.col("rklearn_ms")).alias("speedup_single"),
            (pl.col("sklearn_ms") / pl.col("rklearn_parallel_ms")).alias("speedup_parallel"),
        ])
    )
    speedup_df
    return rklearn_df, rklearn_parallel_df, sklearn_df, speedup_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Findings

    ### Fair Comparison (n_jobs=1)
    - Both rklearn and sklearn run single-threaded for fair comparison
    - Performance differences come from Rust vs NumPy implementation
    - Rust's efficient memory access patterns may provide speedup on larger data

    ### Parallel Mode (n_jobs=-1)
    - rklearn uses Rayon for automatic parallelization across CPU cores
    - Speedup increases with data size and number of columns
    - GIL is released during computation, enabling true multi-threading

    ### Usage
    ```python
    from rklearn.preprocessing import StandardScaler

    # Fair single-threaded comparison (default)
    scaler = StandardScaler(n_jobs=1)

    # Enable parallelism for faster processing
    scaler = StandardScaler(n_jobs=-1)
    ```
    """)
    return


if __name__ == "__main__":
    app.run()
