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
