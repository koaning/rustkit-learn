"""Tests comparing RidgeRegressor against scikit-learn for numerical accuracy."""

import numpy as np
import pytest
from sklearn.linear_model import Ridge as SklearnRidge

from rklearn import RidgeRegressor


class TestRidgeRegressorSklearnCompat:
    @pytest.fixture
    def sample_data(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(300, 8))
        coef = np.array([1.5, -2.0, 0.0, 3.0, -0.5, 2.0, 1.0, -1.0])
        y = X @ coef + 0.25 + rng.normal(scale=0.01, size=300)
        return X, y

    def test_predict_matches_sklearn(self, sample_data):
        X, y = sample_data
        rust = RidgeRegressor(alpha=1.0, fit_intercept=True)
        skl = SklearnRidge(alpha=1.0, fit_intercept=True, solver="cholesky")

        rust.fit(X, y)
        skl.fit(X, y)

        rust_pred = rust.predict(X[:50])
        skl_pred = skl.predict(X[:50])

        np.testing.assert_allclose(rust_pred, skl_pred, rtol=1e-7, atol=1e-7)

    def test_coefficients_match_sklearn(self, sample_data):
        X, y = sample_data
        rust = RidgeRegressor(alpha=0.1, fit_intercept=False)
        skl = SklearnRidge(alpha=0.1, fit_intercept=False, solver="cholesky")

        rust.fit(X, y)
        skl.fit(X, y)

        np.testing.assert_allclose(rust.coef_, skl.coef_, rtol=1e-7, atol=1e-7)


def _fit_both(X, y, **kwargs):
    rust = RidgeRegressor(**kwargs).fit(X, y)
    skl = SklearnRidge(solver="cholesky", **kwargs).fit(X, y)
    return rust, skl


def _make_data(n_samples=2_000, n_features=12, offset=0.0, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features)) + offset
    coef = rng.normal(size=n_features)
    y = X @ coef + 3.0 + rng.normal(scale=0.1, size=n_samples)
    return X, y


@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("layout", ["C", "F", "strided_rows", "strided_cols"])
def test_memory_layouts_match_sklearn(fit_intercept, layout):
    """fit no longer forces a C-contiguous copy, so every layout must still be exact."""
    X, y = _make_data()
    if layout == "F":
        X = np.asfortranarray(X)
    elif layout == "strided_rows":
        X, y = X[::2], y[::2]
    elif layout == "strided_cols":
        X = X[:, ::2]
    rust, skl = _fit_both(X, y, alpha=0.5, fit_intercept=fit_intercept)
    np.testing.assert_allclose(rust.coef_, skl.coef_, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(rust.intercept_, skl.intercept_, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(rust.predict(X), skl.predict(X), rtol=1e-9, atol=1e-9)


def test_strided_y_matches_sklearn():
    X, y = _make_data()
    y_strided = np.stack([y, -y], axis=1)[:, 0]  # non-contiguous 1-D view
    assert not y_strided.flags["C_CONTIGUOUS"]
    rust, skl = _fit_both(X, y_strided, alpha=1.0)
    np.testing.assert_allclose(rust.coef_, skl.coef_, rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize("offset", [1e3, 1e6])
def test_large_feature_offset_is_stable(offset):
    """Features with mean >> spread: centering must happen before forming XᵀX."""
    X, y = _make_data(n_samples=50_000, n_features=5, offset=offset)
    rust, skl = _fit_both(X, y, alpha=1.0)
    np.testing.assert_allclose(rust.coef_, skl.coef_, rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(rust.predict(X), skl.predict(X), rtol=1e-9)


def test_does_not_modify_inputs():
    X, y = _make_data()
    X_orig, y_orig = X.copy(), y.copy()
    RidgeRegressor(alpha=1.0).fit(X, y)
    np.testing.assert_array_equal(X, X_orig)
    np.testing.assert_array_equal(y, y_orig)
