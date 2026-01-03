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
    """)
    return


@app.cell
def _():
    import numpy as np
    import time
    import polars as pl
    import altair as alt
    return alt, np, pl, time


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
def _(np, time):
    def benchmark_fn(fn, data, n_runs=20, warmup=5):
        """Benchmark a function with warmup runs."""
        # Warmup
        for _ in range(warmup):
            fn(data)

        # Actual timing
        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            fn(data)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms

        return np.mean(times), np.std(times), np.median(times)
    return (benchmark_fn,)


@app.cell
def _(
    RklearnMinMaxScaler,
    RklearnStandardScaler,
    SklearnMinMaxScaler,
    SklearnStandardScaler,
):
    # Configuration - vary both rows and columns
    ROWS = [10_000, 100_000, 1_000_000]
    COLS = [50, 100, 200]

    # Scaler configurations: (name, rklearn_class, sklearn_class)
    SCALERS = [
        ("StandardScaler", RklearnStandardScaler, SklearnStandardScaler),
        ("MinMaxScaler", RklearnMinMaxScaler, SklearnMinMaxScaler),
    ]
    return COLS, ROWS, SCALERS


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Running Benchmarks

    Running `fit_transform` benchmarks with 20 iterations each (after 5 warmup runs).

    **Libraries compared:**
    - `sklearn`: scikit-learn (single-threaded)
    - `rklearn`: rklearn with n_jobs=1 (single-threaded, fair comparison)
    - `rklearn_parallel`: rklearn with n_jobs=-1 (multi-threaded via Rayon)
    """)
    return


@app.cell
def _(COLS, ROWS, SCALERS, benchmark_fn, np):
    def run_benchmarks():
        """Run fit_transform benchmarks for rklearn and sklearn."""
        results = []

        for rows in ROWS:
            for cols in COLS:
                print(f"  {rows:,} x {cols}...", end=" ", flush=True)

                # Generate data
                np.random.seed(42)
                data = np.random.randn(rows, cols)

                for scaler_name, rklearn_cls, sklearn_cls in SCALERS:
                    # Benchmark sklearn fit_transform (single-threaded)
                    fn = lambda d, cls=sklearn_cls: cls().fit_transform(d)
                    mean_t, std_t, median_t = benchmark_fn(fn, data)
                    results.append({
                        "scaler": scaler_name,
                        "rows": rows,
                        "cols": cols,
                        "library": "sklearn",
                        "mean_ms": mean_t,
                        "std_ms": std_t,
                        "median_ms": median_t,
                    })

                    # Benchmark rklearn fit_transform (single-threaded, fair comparison)
                    fn = lambda d, cls=rklearn_cls: cls(n_jobs=1).fit_transform(d)
                    mean_t, std_t, median_t = benchmark_fn(fn, data)
                    results.append({
                        "scaler": scaler_name,
                        "rows": rows,
                        "cols": cols,
                        "library": "rklearn",
                        "mean_ms": mean_t,
                        "std_ms": std_t,
                        "median_ms": median_t,
                    })

                    # Benchmark rklearn fit_transform (parallel)
                    fn = lambda d, cls=rklearn_cls: cls(n_jobs=-1).fit_transform(d)
                    mean_t, std_t, median_t = benchmark_fn(fn, data)
                    results.append({
                        "scaler": scaler_name,
                        "rows": rows,
                        "cols": cols,
                        "library": "rklearn_parallel",
                        "mean_ms": mean_t,
                        "std_ms": std_t,
                        "median_ms": median_t,
                    })

                print("done")

        return results
    return (run_benchmarks,)


@app.cell
def _(run_benchmarks):
    print("Running benchmarks:")
    benchmark_results = run_benchmarks()
    print("Done!")
    return (benchmark_results,)


@app.cell
def _(benchmark_results, pl):
    # Create DataFrame
    df = pl.DataFrame(benchmark_results)
    df
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Speedup by Data Size and Number of Columns

    Lines show different column counts. Values above 1 mean rklearn is faster than sklearn.

    **Charts:**
    1. **Fair comparison** - rklearn (n_jobs=1) vs sklearn (both single-threaded)
    2. **Parallel speedup** - rklearn_parallel (n_jobs=-1) vs sklearn
    """)
    return


@app.cell
def _(df, pl):
    # Calculate speedup for both rklearn modes
    sklearn_df = df.filter(pl.col("library") == "sklearn").select(
        ["scaler", "rows", "cols", "mean_ms"]
    ).rename({"mean_ms": "sklearn_ms"})

    rklearn_df = df.filter(pl.col("library") == "rklearn").select(
        ["scaler", "rows", "cols", "mean_ms"]
    ).rename({"mean_ms": "rklearn_ms"})

    rklearn_parallel_df = df.filter(pl.col("library") == "rklearn_parallel").select(
        ["scaler", "rows", "cols", "mean_ms"]
    ).rename({"mean_ms": "rklearn_parallel_ms"})

    speedup_df = (
        sklearn_df
        .join(rklearn_df, on=["scaler", "rows", "cols"])
        .join(rklearn_parallel_df, on=["scaler", "rows", "cols"])
        .with_columns([
            (pl.col("sklearn_ms") / pl.col("rklearn_ms")).alias("speedup_single"),
            (pl.col("sklearn_ms") / pl.col("rklearn_parallel_ms")).alias("speedup_parallel"),
        ])
    )
    speedup_df
    return (speedup_df,)


@app.cell
def _(alt, speedup_df):
    # Speedup line chart - fair comparison (single-threaded)
    _speedup_pd = speedup_df.to_pandas()
    _speedup_pd["cols_label"] = _speedup_pd["cols"].astype(str) + " cols"

    _base = alt.Chart(_speedup_pd).encode(
        x=alt.X("rows:Q", title="Number of Rows", scale=alt.Scale(type="log")),
        y=alt.Y("speedup_single:Q", title="Speedup (sklearn / rklearn)"),
        color=alt.Color("cols_label:N", title="Columns", sort=["50 cols", "100 cols", "200 cols"]),
        column=alt.Column("scaler:N", title="Scaler"),
    )

    _lines = _base.mark_line(point=True, strokeWidth=2)

    speedup_chart_single = _lines.properties(
        width=400,
        height=300,
        title="Fair Comparison: rklearn (n_jobs=1) vs sklearn (both single-threaded)"
    )
    speedup_chart_single
    return


@app.cell
def _(alt, speedup_df):
    # Speedup line chart - parallel comparison
    _speedup_pd = speedup_df.to_pandas()
    _speedup_pd["cols_label"] = _speedup_pd["cols"].astype(str) + " cols"

    _base = alt.Chart(_speedup_pd).encode(
        x=alt.X("rows:Q", title="Number of Rows", scale=alt.Scale(type="log")),
        y=alt.Y("speedup_parallel:Q", title="Speedup (sklearn / rklearn_parallel)"),
        color=alt.Color("cols_label:N", title="Columns", sort=["50 cols", "100 cols", "200 cols"]),
        column=alt.Column("scaler:N", title="Scaler"),
    )

    _lines = _base.mark_line(point=True, strokeWidth=2)

    speedup_chart_parallel = _lines.properties(
        width=400,
        height=300,
        title="Parallel Speedup: rklearn (n_jobs=-1) vs sklearn"
    )
    speedup_chart_parallel
    return


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
