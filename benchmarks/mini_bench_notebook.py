import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell
def _():
    import numpy as np
    import time
    from rklearn import RidgeRegressor
    from sklearn.linear_model import Ridge as SklearnRidge

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
    return


if __name__ == "__main__":
    app.run()
