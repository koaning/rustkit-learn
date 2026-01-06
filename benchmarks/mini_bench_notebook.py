import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell
def _():
    import numpy as np
    import time
    from rklearn import RidgeRegressor, HuberRegressor
    from sklearn.linear_model import Ridge as SklearnRidge
    from sklearn.linear_model import HuberRegressor as SklearnHuber

    rng = np.random.default_rng(0)
    X_small = rng.normal(size=(1_000, 8))
    y_small = X_small @ np.arange(1, 9)
    X_big = rng.normal(size=(20_000, 8))
    y_big = X_big @ np.arange(1, 9)

    t0 = time.perf_counter()
    rust = RidgeRegressor(alpha=1.0)
    rust.fit(X_small, y_small)
    ms_rust_fit = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    skl = SklearnRidge(alpha=1.0, solver="cholesky")
    skl.fit(X_small, y_small)
    ms_skl_fit = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    _ = rust.predict(X_small)
    ms_rust_pred = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    _ = skl.predict(X_small)
    ms_skl_pred = (time.perf_counter() - t0) * 1e3

    print("minibench: ridge 1k x 8")
    print(f"fit  rklearn {ms_rust_fit:.3f} ms | sklearn {ms_skl_fit:.3f} ms")
    print(f"pred rklearn {ms_rust_pred:.3f} ms | sklearn {ms_skl_pred:.3f} ms")

    t0 = time.perf_counter()
    rust.fit(X_big, y_big)
    ms_rust_fit_big = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    skl.fit(X_big, y_big)
    ms_skl_fit_big = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    _ = rust.predict(X_big)
    ms_rust_pred_big = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    _ = skl.predict(X_big)
    ms_skl_pred_big = (time.perf_counter() - t0) * 1e3

    print("minibench: ridge 20k x 8")
    print(f"fit  rklearn {ms_rust_fit_big:.3f} ms | sklearn {ms_skl_fit_big:.3f} ms")
    print(f"pred rklearn {ms_rust_pred_big:.3f} ms | sklearn {ms_skl_pred_big:.3f} ms")
    return


@app.cell
def _():
    import numpy as np
    import time
    from rklearn import HuberRegressor
    from sklearn.linear_model import HuberRegressor as SklearnHuber

    rng = np.random.default_rng(0)

    # Medium dataset: 100k x 20
    X_med = rng.normal(size=(100_000, 20))
    y_med = X_med @ np.arange(1, 21) + rng.normal(scale=0.1, size=100_000)

    # Large dataset: 500k x 50
    X_big = rng.normal(size=(500_000, 50))
    y_big = X_big @ np.arange(1, 51) + rng.normal(scale=0.1, size=500_000)

    # sklearn baseline
    skl_huber = SklearnHuber(epsilon=1.35, alpha=0.0001)

    t0 = time.perf_counter()
    skl_huber.fit(X_med, y_med)
    ms_skl_fit_med = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    _ = skl_huber.predict(X_med)
    ms_skl_pred_med = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    skl_huber.fit(X_big, y_big)
    ms_skl_fit_big = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    _ = skl_huber.predict(X_big)
    ms_skl_pred_big = (time.perf_counter() - t0) * 1e3

    # rklearn (uses rayon parallelization by default for large arrays)
    rust_huber = HuberRegressor(epsilon=1.35, alpha=0.0001)

    t0 = time.perf_counter()
    rust_huber.fit(X_med, y_med)
    ms_rust_fit_med = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    _ = rust_huber.predict(X_med)
    ms_rust_pred_med = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    rust_huber.fit(X_big, y_big)
    ms_rust_fit_big = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    _ = rust_huber.predict(X_big)
    ms_rust_pred_big = (time.perf_counter() - t0) * 1e3

    # Note: Set RAYON_NUM_THREADS=1 env var before running for single-threaded comparison
    print("minibench: huber 100k x 20")
    print(f"fit  sklearn {ms_skl_fit_med:.1f} ms | rklearn {ms_rust_fit_med:.1f} ms | speedup {ms_skl_fit_med/ms_rust_fit_med:.2f}x")
    print(f"pred sklearn {ms_skl_pred_med:.3f} ms | rklearn {ms_rust_pred_med:.3f} ms | speedup {ms_skl_pred_med/ms_rust_pred_med:.2f}x")

    print("minibench: huber 500k x 50")
    print(f"fit  sklearn {ms_skl_fit_big:.1f} ms | rklearn {ms_rust_fit_big:.1f} ms | speedup {ms_skl_fit_big/ms_rust_fit_big:.2f}x")
    print(f"pred sklearn {ms_skl_pred_big:.3f} ms | rklearn {ms_rust_pred_big:.3f} ms | speedup {ms_skl_pred_big/ms_rust_pred_big:.2f}x")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
