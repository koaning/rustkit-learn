import numpy as np

from rklearn import RidgeRegressor


def test_fit_predict_basic():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(200, 6))
    coef = np.array([1.5, -2.0, 0.0, 3.0, -0.5, 2.0])
    y = X @ coef + 0.7

    model = RidgeRegressor(alpha=1.0, fit_intercept=True)
    model.fit(X, y)
    preds = model.predict(X[:5])

    assert preds.shape == (5,)
    assert model.n_features_in_ == 6
    assert model.n_samples_fit_ == 200
    assert model.coef_ is not None
    assert model.intercept_ is not None


def test_alpha_zero_matches_ols():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 4))
    coef = np.array([2.0, -1.0, 0.5, 3.0])
    y = X @ coef

    model = RidgeRegressor(alpha=0.0, fit_intercept=False)
    model.fit(X, y)
    np.testing.assert_allclose(model.coef_, coef, rtol=1e-6, atol=1e-6)

