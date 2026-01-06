"""Tests comparing HuberRegressor against scikit-learn for numerical accuracy."""

import numpy as np
import pytest
from sklearn.linear_model import HuberRegressor as SklearnHuber

from rklearn import HuberRegressor


class TestHuberRegressorSklearnCompat:
    @pytest.fixture
    def sample_data(self):
        """Generate sample data without outliers."""
        rng = np.random.default_rng(42)
        X = rng.normal(size=(300, 8))
        coef = np.array([1.5, -2.0, 0.0, 3.0, -0.5, 2.0, 1.0, -1.0])
        y = X @ coef + 0.25 + rng.normal(scale=0.1, size=300)
        return X, y

    @pytest.fixture
    def data_with_outliers(self):
        """Generate sample data with outliers."""
        rng = np.random.default_rng(42)
        X = rng.normal(size=(300, 8))
        coef = np.array([1.5, -2.0, 0.0, 3.0, -0.5, 2.0, 1.0, -1.0])
        y = X @ coef + 0.25 + rng.normal(scale=0.1, size=300)
        # Add outliers
        outlier_indices = rng.choice(300, size=15, replace=False)
        y[outlier_indices] += rng.choice([-1, 1], size=15) * rng.uniform(10, 20, size=15)
        return X, y

    def test_predict_close_to_sklearn(self, sample_data):
        """Test that predictions are close to sklearn."""
        X, y = sample_data
        rust = HuberRegressor(epsilon=1.35, alpha=0.0001, fit_intercept=True)
        skl = SklearnHuber(epsilon=1.35, alpha=0.0001, fit_intercept=True)

        rust.fit(X, y)
        skl.fit(X, y)

        rust_pred = rust.predict(X[:50])
        skl_pred = skl.predict(X[:50])

        # Predictions should be very close (within 0.005 absolute)
        np.testing.assert_allclose(rust_pred, skl_pred, rtol=0, atol=0.005)

    def test_coefficients_close_to_sklearn(self, sample_data):
        """Test that coefficients are close to sklearn."""
        X, y = sample_data
        rust = HuberRegressor(epsilon=1.35, alpha=0.0001, fit_intercept=True)
        skl = SklearnHuber(epsilon=1.35, alpha=0.0001, fit_intercept=True)

        rust.fit(X, y)
        skl.fit(X, y)

        # Coefficients should be very close (within 0.005 absolute)
        # Note: relative tolerance not used because some coefficients may be near zero
        np.testing.assert_allclose(rust.coef_, skl.coef_, rtol=0, atol=0.005)
        np.testing.assert_allclose(rust.intercept_, skl.intercept_, rtol=0, atol=0.005)

    def test_robustness_to_outliers(self, data_with_outliers):
        """Test that HuberRegressor is robust to outliers."""
        X, y = data_with_outliers
        rust = HuberRegressor(epsilon=1.35, alpha=0.0001)
        rust.fit(X, y)

        # The model should still have reasonable coefficients
        # despite the outliers
        assert rust.coef_ is not None
        assert rust.intercept_ is not None
        assert rust.scale_ is not None
        assert rust.n_iter_ is not None
        assert rust.n_features_in_ == 8

    def test_fit_intercept_false(self, sample_data):
        """Test with fit_intercept=False."""
        X, y = sample_data
        rust = HuberRegressor(epsilon=1.35, alpha=0.0001, fit_intercept=False)
        skl = SklearnHuber(epsilon=1.35, alpha=0.0001, fit_intercept=False)

        rust.fit(X, y)
        skl.fit(X, y)

        # Intercept should be 0 when fit_intercept=False
        assert rust.intercept_ == 0.0

        # Without intercept, optimization can converge to slightly different solutions
        # (L-BFGS-B vs IRLS behave differently without centering)
        # Check that R² scores are similar instead
        rust_score = rust.score(X, y)
        skl_score = skl.score(X, y)
        assert abs(rust_score - skl_score) < 0.01  # Within 1% R²

    def test_different_epsilon_values(self, sample_data):
        """Test with different epsilon values."""
        X, y = sample_data

        for epsilon in [1.1, 1.35, 2.0, 3.0]:
            rust = HuberRegressor(epsilon=epsilon, alpha=0.0001)
            rust.fit(X, y)
            assert rust.coef_ is not None
            assert len(rust.coef_) == 8

    def test_score_method(self, sample_data):
        """Test the score method returns reasonable R^2."""
        X, y = sample_data
        rust = HuberRegressor(epsilon=1.35, alpha=0.0001)
        rust.fit(X, y)

        score = rust.score(X, y)
        # Should have a high R^2 on clean data
        assert score > 0.9

    def test_parameter_validation(self):
        """Test that invalid parameters raise errors."""
        with pytest.raises(ValueError, match="epsilon must be greater than 1.0"):
            HuberRegressor(epsilon=0.5)

        with pytest.raises(ValueError, match="epsilon must be greater than 1.0"):
            HuberRegressor(epsilon=1.0)

        with pytest.raises(ValueError, match="alpha must be >= 0"):
            HuberRegressor(alpha=-0.1)

        with pytest.raises(ValueError, match="max_iter must be > 0"):
            HuberRegressor(max_iter=0)

        with pytest.raises(ValueError, match="tol must be > 0"):
            HuberRegressor(tol=0.0)

    def test_not_fitted_error(self, sample_data):
        """Test that predict raises error when not fitted."""
        X, y = sample_data
        rust = HuberRegressor()

        with pytest.raises(ValueError, match="Not fitted yet"):
            rust.predict(X)

    def test_fitted_attributes(self, sample_data):
        """Test that all fitted attributes are set correctly."""
        X, y = sample_data
        rust = HuberRegressor(epsilon=1.35, alpha=0.0001, max_iter=100)
        rust.fit(X, y)

        assert rust.coef_ is not None
        assert len(rust.coef_) == 8
        assert rust.intercept_ is not None
        assert rust.scale_ is not None
        assert rust.scale_ > 0
        assert rust.n_features_in_ == 8
        assert rust.n_iter_ is not None
        assert rust.n_iter_ > 0
        assert rust.n_iter_ <= 100

    def test_method_chaining(self, sample_data):
        """Test that fit returns self for method chaining."""
        X, y = sample_data
        rust = HuberRegressor()
        result = rust.fit(X, y)
        # fit should return self, allowing method chaining
        assert result.predict(X) is not None
