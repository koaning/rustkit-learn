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
    # e-means benchmark

    Compares `rklearn.cluster.EMeans` (arXiv:2611.09501) with Lloyd k-means
    (`rklearn.cluster.KMeans`, `sklearn.cluster.KMeans`) and
    `sklearn.cluster.MiniBatchKMeans`, and `rklearn.mixture.EGaussianMixture`
    with `sklearn.mixture.GaussianMixture`.

    **Method.** Each config uses one fixed dataset, split 80/20. Every model is
    fit with 5 different `random_state` values and we report the median. Quality
    is measured on the held-out split against a single fixed reference per
    dataset: the best of 10 sklearn restarts (`n_init=10`, or `n_init=5` for the
    mixtures). On these datasets a single run's local optimum can swing the loss
    by 10-15% between seeds, so comparing single runs mostly measures seeding luck.
    """)
    return


@app.cell
def _():
    import time
    import warnings

    import numpy as np
    import polars as pl
    from sklearn.cluster import KMeans as SkKMeans
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.mixture import GaussianMixture

    from rklearn.cluster import EMeans, KMeans
    from rklearn.mixture import EGaussianMixture
    return (
        ConvergenceWarning,
        EGaussianMixture,
        EMeans,
        GaussianMixture,
        KMeans,
        MiniBatchKMeans,
        SkKMeans,
        np,
        pl,
        time,
        warnings,
    )


@app.cell
def _(ConvergenceWarning, np, time, warnings):
    SEEDS = [0, 1, 2, 3, 4]

    def make_data(n, d, k_true, seed=0):
        """Gaussian mixture with unequal sizes and spreads, split 80/20."""
        rng = np.random.default_rng(seed)
        centers = rng.normal(scale=4.0, size=(k_true, d))
        sizes = np.maximum((rng.dirichlet(np.ones(k_true) * 2) * n).astype(int), 2)
        scales = rng.uniform(0.5, 1.5, size=k_true)
        X = np.vstack(
            [rng.normal(c, s, size=(m, d)) for c, s, m in zip(centers, scales, sizes)]
        )
        X = X[rng.permutation(len(X))]
        cut = int(0.8 * len(X))
        return np.ascontiguousarray(X[:cut]), np.ascontiguousarray(X[cut:])

    def inertia(X, C):
        d = (X**2).sum(1)[:, None] - 2 * X @ C.T + (C**2).sum(1)[None, :]
        return float(np.maximum(d.min(1), 0).sum())

    def fit_timed(model, X):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            t0 = time.perf_counter()
            model.fit(X)
            return time.perf_counter() - t0, model

    def config_label(cfg):
        return f"n={cfg['n'] // 1000}k, d={cfg['d']}, k={cfg['k']}"
    return SEEDS, config_label, fit_timed, inertia, make_data


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## k-means: fit time vs. quality

    Each method is swept over its budget knob: `max_iter` for sklearn `KMeans`
    and `n_epochs` for `EMeans`. `MiniBatchKMeans`, rklearn `KMeans` and
    `EMeans(revival=True)` run at their defaults only.
    """)
    return


@app.cell
def _(
    EMeans,
    KMeans,
    MiniBatchKMeans,
    SEEDS,
    SkKMeans,
    config_label,
    fit_timed,
    inertia,
    make_data,
    pl,
):
    KMEANS_CONFIGS = [
        dict(n=50_000, d=8, k=32),
        dict(n=100_000, d=32, k=64),
        dict(n=200_000, d=16, k=256),
    ]

    _rows = []
    for _cfg in KMEANS_CONFIGS:
        _k = _cfg["k"]
        _Xtr, _Xte = make_data(_cfg["n"], _cfg["d"], k_true=_k)
        _ref = inertia(_Xte, SkKMeans(n_clusters=_k, n_init=10, random_state=0).fit(_Xtr).cluster_centers_)
        for _seed in SEEDS:
            _runs = [
                ("sklearn KMeans", it, SkKMeans(n_clusters=_k, n_init=1, max_iter=it, random_state=_seed))
                for it in [2, 5, 10, 25, 300]
            ]
            _runs += [
                ("rklearn EMeans", ep, EMeans(n_clusters=_k, n_epochs=float(ep), random_state=_seed, n_jobs=-1))
                for ep in [1, 2, 3, 5, 8]
            ]
            _runs += [
                ("sklearn MiniBatchKMeans", None, MiniBatchKMeans(n_clusters=_k, n_init=1, random_state=_seed)),
                ("rklearn KMeans", None, KMeans(n_clusters=_k, random_state=_seed, n_jobs=-1)),
                ("rklearn EMeans (revival)", 3, EMeans(n_clusters=_k, revival=True, random_state=_seed, n_jobs=-1)),
            ]
            for _model, _budget, _est in _runs:
                _t, _m = fit_timed(_est, _Xtr)
                _rows.append(
                    dict(
                        config=config_label(_cfg),
                        model=_model,
                        budget=_budget,
                        seed=_seed,
                        seconds=_t,
                        rel_loss_pct=100 * (inertia(_Xte, _m.cluster_centers_) / _ref - 1),
                    )
                )
    kmeans_raw = pl.DataFrame(_rows)
    kmeans_df = (
        kmeans_raw.group_by("config", "model", "budget", maintain_order=True)
        .agg(
            pl.col("seconds").median(),
            pl.col("rel_loss_pct").median(),
            pl.col("rel_loss_pct").min().alias("rel_loss_min"),
            pl.col("rel_loss_pct").max().alias("rel_loss_max"),
        )
        .with_columns(pl.col("seconds").round(3), pl.col("^rel_loss.*$").round(2))
    )
    return (kmeans_df,)


@app.cell
def _(kmeans_df, pl):
    # Default settings only: max_iter=300 for Lloyd, n_epochs=3 for EMeans.
    _is_default = (
        pl.col("budget").is_null()
        | ((pl.col("model") == "sklearn KMeans") & (pl.col("budget") == 300))
        | (pl.col("model").str.starts_with("rklearn EMeans") & (pl.col("budget") == 3))
    )
    _sk = kmeans_df.filter((pl.col("model") == "sklearn KMeans") & (pl.col("budget") == 300)).select(
        "config", pl.col("seconds").alias("_sk")
    )
    kmeans_summary = (
        kmeans_df.filter(_is_default)
        .join(_sk, on="config")
        .with_columns(speedup_vs_sklearn=(pl.col("_sk") / pl.col("seconds")).round(1))
        .drop("_sk", "budget")
    )
    print(kmeans_summary)
    kmeans_summary
    return


@app.cell
def _():
    import altair as alt

    # Categorical slots 1-3 of the reference palette (validated all-pairs).
    COLORS = {
        "sklearn KMeans": "#2a78d6",
        "sklearn MiniBatchKMeans": "#eb6834",
        "rklearn EMeans": "#1baf7a",
        "sklearn GaussianMixture": "#2a78d6",
        "rklearn EGaussianMixture": "#1baf7a",
    }

    def tradeoff_chart(df, y, y_title, title):
        """One panel per config: fit time (log) vs. quality, one line per model."""
        df = df.filter(df["model"].is_in(list(COLORS)))
        models = [m for m in COLORS if m in set(df["model"].to_list())]
        color = alt.Color(
            "model:N",
            scale=alt.Scale(domain=models, range=[COLORS[m] for m in models]),
            legend=alt.Legend(title=None, orient="top"),
        )
        panels = []
        for i, cfg in enumerate(df["config"].unique(maintain_order=True).to_list()):
            sub = df.filter(df["config"] == cfg).to_pandas()
            # Label only the cheapest and most expensive budget of each sweep.
            ends = sub.groupby("model")["budget"].transform(lambda b: (b == b.min()) | (b == b.max()))
            sub["end_label"] = sub["budget"].where(ends).map(
                lambda b: "" if b != b else f"{int(b)}"
            )
            base = alt.Chart(sub).encode(
                x=alt.X("seconds:Q", scale=alt.Scale(type="log", nice=False),
                        title="Fit time (s, log scale)", axis=alt.Axis(grid=False)),
                y=alt.Y(f"{y}:Q", title=y_title if i == 0 else None,
                        axis=alt.Axis(gridColor="#e4e3df")),
                color=color,
                tooltip=[
                    "model:N",
                    alt.Tooltip("budget:Q", title="max_iter / n_epochs"),
                    alt.Tooltip("seconds:Q", format=".3f", title="fit time (s)"),
                    alt.Tooltip(f"{y}:Q", format=".2f", title=y_title),
                ],
            )
            rule = alt.Chart(sub).mark_rule(color="#8c8b86", strokeDash=[4, 4]).encode(y=alt.datum(0))
            lines = base.mark_line(strokeWidth=2).encode(order="budget:Q")
            points = base.mark_point(filled=True, size=70, opacity=1, stroke="#fcfcfb", strokeWidth=2)
            labels = base.mark_text(align="left", dx=6, dy=-8, fontSize=10, color="#52514e").encode(
                text="end_label:N"
            )
            panels.append(alt.layer(rule, lines, points, labels).properties(width=260, height=220, title=cfg))
        return (
            alt.hconcat(*panels)
            .resolve_scale(y="independent")
            .properties(title=title)
            .configure_view(stroke=None)
            .configure(background="#fcfcfb")
            .configure_axis(labelColor="#52514e", titleColor="#52514e", domainColor="#c3c2b7",
                            tickColor="#c3c2b7")
            .configure_legend(labelColor="#0b0b0b")
            .configure_title(color="#0b0b0b", fontSize=12, fontWeight="normal", anchor="start")
        )
    return (tradeoff_chart,)


@app.cell
def _(kmeans_df, pl, tradeoff_chart):
    kmeans_chart = tradeoff_chart(
        kmeans_df.filter(pl.col("budget").is_not_null() | (pl.col("model") == "sklearn MiniBatchKMeans")),
        y="rel_loss_pct",
        y_title="Held-out loss above best-of-10 Lloyd (%)",
        title="k-means, median of 5 seeds. Point labels: max_iter (KMeans) / n_epochs (EMeans). Lower and further left is better.",
    )
    kmeans_chart
    return (kmeans_chart,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Sweeping batch size and alpha

    `EMeans(n_clusters=64)` on the 100k x 32 dataset, median of 5 seeds.
    `None` means the paper's fitted law picks the value.
    """)
    return


@app.cell
def _(EMeans, SEEDS, SkKMeans, inertia, make_data, pl):
    _Xtr, _Xte = make_data(100_000, 32, k_true=64)
    _ref = inertia(_Xte, SkKMeans(n_clusters=64, n_init=10, random_state=0).fit(_Xtr).cluster_centers_)
    _rows = []
    for _b in [None, 256, 1024, 4096]:
        for _a in [None, 0.05, 0.2, 0.8]:
            for _seed in SEEDS:
                _m = EMeans(n_clusters=64, batch_size=_b, alpha=_a, random_state=_seed, n_jobs=-1).fit(_Xtr)
                _rows.append(
                    dict(
                        batch_size=str(_b),
                        alpha=str(_a),
                        batch_used=_m.batch_size_,
                        alpha_used=round(_m.alpha_, 4),
                        rel_loss_pct=100 * (inertia(_Xte, _m.cluster_centers_) / _ref - 1),
                    )
                )
    sweep_df = (
        pl.DataFrame(_rows)
        .group_by("batch_size", "alpha", "batch_used", "alpha_used", maintain_order=True)
        .agg(pl.col("rel_loss_pct").median().round(2))
    )
    print(sweep_df)
    sweep_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Gaussian mixtures: fit time vs. quality

    `EGaussianMixture` runs stochastic EM on mini-batches. It keeps an EWM of
    each component's responsibility mass, mean and second moment (diagonal
    covariances) and ends with one exact M-step over the last pass. sklearn is
    swept over `max_iter`, EGaussianMixture over `n_epochs`. The y-axis is
    held-out mean log-likelihood minus that of the best of 5 sklearn restarts.
    """)
    return


@app.cell
def _(
    EGaussianMixture,
    GaussianMixture,
    SEEDS,
    config_label,
    fit_timed,
    make_data,
    pl,
):
    GMM_CONFIGS = [dict(n=50_000, d=8, k=16), dict(n=200_000, d=16, k=32)]
    _rows = []
    for _cfg in GMM_CONFIGS:
        _k = _cfg["k"]
        _Xtr, _Xte = make_data(_cfg["n"], _cfg["d"], k_true=_k, seed=1)
        _ref = GaussianMixture(n_components=_k, covariance_type="diag", n_init=5, random_state=0).fit(_Xtr).score(_Xte)
        for _seed in SEEDS:
            _runs = [
                ("sklearn GaussianMixture", it, GaussianMixture(
                    n_components=_k, covariance_type="diag", max_iter=it, random_state=_seed))
                for it in [2, 5, 10, 25, 100]
            ]
            _runs += [
                ("rklearn EGaussianMixture", ep, EGaussianMixture(
                    n_components=_k, n_epochs=float(ep), random_state=_seed, n_jobs=-1))
                for ep in [1, 2, 3, 5, 8]
            ]
            _runs += [
                ("rklearn EGaussianMixture (revival)", 3, EGaussianMixture(
                    n_components=_k, revival=True, random_state=_seed, n_jobs=-1)),
            ]
            for _model, _budget, _est in _runs:
                _t, _m = fit_timed(_est, _Xtr)
                _rows.append(
                    dict(config=config_label(_cfg), model=_model, budget=_budget, seed=_seed,
                         seconds=_t, loglik_gap=_m.score(_Xte) - _ref)
                )
    gmm_df = (
        pl.DataFrame(_rows)
        .group_by("config", "model", "budget", maintain_order=True)
        .agg(
            pl.col("seconds").median(),
            pl.col("loglik_gap").median(),
            pl.col("loglik_gap").min().alias("loglik_gap_min"),
            pl.col("loglik_gap").max().alias("loglik_gap_max"),
        )
        .with_columns(pl.col("seconds").round(3), pl.col("^loglik_gap.*$").round(4))
    )
    return (gmm_df,)


@app.cell
def _(gmm_df, pl):
    _is_default = ((pl.col("model") == "sklearn GaussianMixture") & (pl.col("budget") == 100)) | (
        pl.col("model").str.starts_with("rklearn") & (pl.col("budget") == 3)
    )
    _sk = gmm_df.filter((pl.col("model") == "sklearn GaussianMixture") & (pl.col("budget") == 100)).select(
        "config", pl.col("seconds").alias("_sk")
    )
    gmm_summary = (
        gmm_df.filter(_is_default)
        .join(_sk, on="config")
        .with_columns(speedup_vs_sklearn=(pl.col("_sk") / pl.col("seconds")).round(1))
        .drop("_sk", "budget")
    )
    print(gmm_summary)
    gmm_summary
    return


@app.cell
def _(gmm_df, tradeoff_chart):
    gmm_chart = tradeoff_chart(
        gmm_df,
        y="loglik_gap",
        y_title="Held-out log-lik vs best-of-5 sklearn",
        title="Diagonal Gaussian mixture, median of 5 seeds. Point labels: max_iter (sklearn) / n_epochs (rklearn). Higher and further left is better.",
    )
    gmm_chart
    return (gmm_chart,)


@app.cell
def _(gmm_chart, gmm_df, kmeans_chart, kmeans_df):
    # Write PNGs and the medians for BENCHMARKS.md (PNG export needs vl-convert-python).
    from pathlib import Path

    _out = Path(__file__).parent.parent / "benchmarks" / "img"
    try:
        _out.mkdir(parents=True, exist_ok=True)
        kmeans_chart.save(_out / "emeans_tradeoff.png", scale_factor=2)
        gmm_chart.save(_out / "egmm_tradeoff.png", scale_factor=2)
    except Exception as _e:  # vl-convert missing, or running in a browser
        print(f"skipped PNG export: {_e}")
    print(kmeans_df)
    print(gmm_df)
    return


if __name__ == "__main__":
    app.run()
