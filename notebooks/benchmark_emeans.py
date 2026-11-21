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
    `sklearn.cluster.MiniBatchKMeans`, then compares
    `rklearn.mixture.EGaussianMixture` with `sklearn.mixture.GaussianMixture`.

    Loss is measured on a held-out split. Relative loss is against sklearn
    `KMeans` (lower is better). For the mixtures the metric is held-out mean
    log-likelihood (higher is better).
    """)
    return


@app.cell
def _():
    import time

    import numpy as np
    import polars as pl
    from sklearn.cluster import KMeans as SkKMeans
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.mixture import GaussianMixture

    from rklearn.cluster import EMeans, KMeans
    from rklearn.mixture import EGaussianMixture
    return (
        EGaussianMixture,
        EMeans,
        GaussianMixture,
        KMeans,
        MiniBatchKMeans,
        SkKMeans,
        np,
        pl,
        time,
    )


@app.cell
def _(np):
    def make_data(n, d, k_true, seed=0, spread=1.0):
        """Gaussian mixture with unequal sizes and scales, split 80/20."""
        rng = np.random.default_rng(seed)
        centers = rng.normal(scale=4.0, size=(k_true, d))
        sizes = rng.dirichlet(np.ones(k_true) * 2) * n
        sizes = np.maximum(sizes.astype(int), 2)
        scales = rng.uniform(0.5, 1.5, size=k_true) * spread
        X = np.vstack(
            [rng.normal(c, s, size=(m, d)) for c, s, m in zip(centers, scales, sizes)]
        )
        X = X[rng.permutation(len(X))]
        cut = int(0.8 * len(X))
        return np.ascontiguousarray(X[:cut]), np.ascontiguousarray(X[cut:])

    def inertia(X, C):
        d = (X**2).sum(1)[:, None] - 2 * X @ C.T + (C**2).sum(1)[None, :]
        return float(np.maximum(d.min(1), 0).sum())
    return inertia, make_data


@app.cell
def _(time):
    def timed(fn, repeats=3):
        """Best-of-`repeats` wall time, plus the last result."""
        best, out = float("inf"), None
        for _ in range(repeats):
            t0 = time.perf_counter()
            out = fn()
            best = min(best, time.perf_counter() - t0)
        return best, out
    return (timed,)


@app.cell
def _(EMeans, KMeans, MiniBatchKMeans, SkKMeans, inertia, make_data, timed):
    CONFIGS = [
        dict(n=50_000, d=8, k=32),
        dict(n=100_000, d=32, k=64),
        dict(n=200_000, d=16, k=256),
    ]

    def kmeans_models(k):
        return {
            "sklearn KMeans": lambda: SkKMeans(n_clusters=k, n_init=1, random_state=0),
            "sklearn MiniBatchKMeans": lambda: MiniBatchKMeans(
                n_clusters=k, n_init=1, random_state=0
            ),
            "rklearn KMeans": lambda: KMeans(n_clusters=k, random_state=0, n_jobs=-1),
            "rklearn EMeans": lambda: EMeans(n_clusters=k, random_state=0, n_jobs=-1),
            "rklearn EMeans (revival)": lambda: EMeans(
                n_clusters=k, revival=True, random_state=0, n_jobs=-1
            ),
        }

    kmeans_rows = []
    for _cfg in CONFIGS:
        _Xtr, _Xte = make_data(_cfg["n"], _cfg["d"], k_true=_cfg["k"])
        _ref = None
        for _name, _make in kmeans_models(_cfg["k"]).items():
            _t, _m = timed(lambda: _make().fit(_Xtr))
            _loss = inertia(_Xte, _m.cluster_centers_)
            _ref = _loss if _ref is None else _ref
            kmeans_rows.append(
                dict(
                    **_cfg,
                    model=_name,
                    seconds=round(_t, 3),
                    test_inertia=_loss,
                    rel_loss_pct=round(100 * (_loss / _ref - 1), 2),
                )
            )
    return (kmeans_rows,)


@app.cell
def _(kmeans_rows, pl):
    kmeans_df = pl.DataFrame(kmeans_rows)
    _ref = (
        kmeans_df.filter(pl.col("model") == "sklearn KMeans")
        .select("n", "d", "k", pl.col("seconds").alias("sk_seconds"))
    )
    kmeans_df = kmeans_df.join(_ref, on=["n", "d", "k"]).with_columns(
        speedup_vs_sklearn=(pl.col("sk_seconds") / pl.col("seconds")).round(1)
    ).drop("sk_seconds", "test_inertia")
    print(kmeans_df)
    kmeans_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Sweeping batch size and alpha

    `batch_size=None` and `alpha=None` use the paper's fitted laws. The grid
    shows how sensitive the held-out loss is to both.
    """)
    return


@app.cell
def _(EMeans, SkKMeans, inertia, make_data, pl):
    _Xtr, _Xte = make_data(100_000, 32, k_true=64, seed=1)
    _ref = inertia(_Xte, SkKMeans(n_clusters=64, n_init=1, random_state=0).fit(_Xtr).cluster_centers_)
    sweep_rows = []
    for _b in [None, 256, 1024, 4096]:
        for _a in [None, 0.05, 0.2, 0.8]:
            _m = EMeans(n_clusters=64, batch_size=_b, alpha=_a, random_state=0, n_jobs=-1).fit(_Xtr)
            sweep_rows.append(
                dict(
                    batch_size=str(_b),
                    alpha=str(_a),
                    batch_used=_m.batch_size_,
                    alpha_used=round(_m.alpha_, 4),
                    rel_loss_pct=round(100 * (inertia(_Xte, _m.cluster_centers_) / _ref - 1), 2),
                )
            )
    sweep_df = pl.DataFrame(sweep_rows)
    print(sweep_df)
    sweep_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Gaussian mixtures with the same trick

    `EGaussianMixture` runs stochastic EM on mini-batches. It keeps an EWM of
    each component's responsibility mass, mean and second moment (diagonal
    covariances) and ends with one exact M-step over the last pass.
    """)
    return


@app.cell
def _(EGaussianMixture, GaussianMixture, make_data, pl, timed):
    GMM_CONFIGS = [
        dict(n=50_000, d=8, k=16),
        dict(n=200_000, d=16, k=32),
    ]
    gmm_rows = []
    for _cfg in GMM_CONFIGS:
        _Xtr, _Xte = make_data(_cfg["n"], _cfg["d"], k_true=_cfg["k"], seed=2)
        _models = {
            "sklearn GaussianMixture(diag)": lambda: GaussianMixture(
                n_components=_cfg["k"], covariance_type="diag", random_state=0
            ),
            "rklearn EGaussianMixture": lambda: EGaussianMixture(
                n_components=_cfg["k"], random_state=0, n_jobs=-1
            ),
            "rklearn EGaussianMixture (revival)": lambda: EGaussianMixture(
                n_components=_cfg["k"], revival=True, random_state=0, n_jobs=-1
            ),
        }
        for _name, _make in _models.items():
            _t, _m = timed(lambda: _make().fit(_Xtr), repeats=2)
            gmm_rows.append(
                dict(**_cfg, model=_name, seconds=round(_t, 3), test_loglik=round(_m.score(_Xte), 4))
            )
    gmm_df = pl.DataFrame(gmm_rows)
    print(gmm_df)
    gmm_df
    return


if __name__ == "__main__":
    app.run()
