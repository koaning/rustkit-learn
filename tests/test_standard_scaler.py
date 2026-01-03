import numpy as np
import pytest
from rklearn import StandardScaler


class TestStandardScaler:
    def test_fit_basic(self):
        """Test basic fit functionality."""
        X = np.array([[0, 0], [0, 0], [1, 1], [1, 1]], dtype=np.float64)
        scaler = StandardScaler()
        scaler.fit(X)

        np.testing.assert_array_almost_equal(scaler.mean_, [0.5, 0.5])
        np.testing.assert_array_almost_equal(scaler.scale_, [0.5, 0.5])

    def test_transform(self):
        """Test transform produces expected output."""
        X = np.array([[0, 0], [0, 0], [1, 1], [1, 1]], dtype=np.float64)
        scaler = StandardScaler()
        scaler.fit(X)

        X_scaled = scaler.transform(X)
        expected = np.array([[-1, -1], [-1, -1], [1, 1], [1, 1]])
        np.testing.assert_array_almost_equal(X_scaled, expected)

    def test_fit_transform(self):
        """Test fit_transform produces same result as fit then transform."""
        X = np.array([[0, 0], [0, 0], [1, 1], [1, 1]], dtype=np.float64)

        scaler1 = StandardScaler()
        result1 = scaler1.fit_transform(X)

        scaler2 = StandardScaler()
        scaler2.fit(X)
        result2 = scaler2.transform(X)

        np.testing.assert_array_almost_equal(result1, result2)

    def test_inverse_transform(self):
        """Test inverse_transform recovers original data."""
        X = np.random.randn(100, 5)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_recovered = scaler.inverse_transform(X_scaled)

        np.testing.assert_array_almost_equal(X, X_recovered)

    def test_partial_fit_single_batch(self):
        """Test partial_fit with a single batch matches fit."""
        X = np.array([[0, 0], [0, 0], [1, 1], [1, 1]], dtype=np.float64)

        scaler1 = StandardScaler()
        scaler1.fit(X)

        scaler2 = StandardScaler()
        scaler2.partial_fit(X)

        np.testing.assert_array_almost_equal(scaler1.mean_, scaler2.mean_)
        np.testing.assert_array_almost_equal(scaler1.scale_, scaler2.scale_)

    def test_partial_fit_multiple_batches(self):
        """Test partial_fit with multiple batches."""
        X1 = np.array([[0, 0], [0, 0]], dtype=np.float64)
        X2 = np.array([[1, 1], [1, 1]], dtype=np.float64)

        scaler = StandardScaler()
        scaler.partial_fit(X1)
        scaler.partial_fit(X2)

        # Should get same result as fitting on combined data
        np.testing.assert_array_almost_equal(scaler.mean_, [0.5, 0.5])
        assert scaler.n_samples_seen_ == 4

    def test_with_mean_false(self):
        """Test with_mean=False doesn't center data."""
        X = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
        scaler = StandardScaler(with_mean=False)
        X_scaled = scaler.fit_transform(X)

        # Data should not be centered
        assert X_scaled.mean(axis=0).sum() != 0

    def test_with_std_false(self):
        """Test with_std=False doesn't scale data."""
        X = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
        scaler = StandardScaler(with_std=False)
        X_scaled = scaler.fit_transform(X)

        # Data should be centered but not scaled
        np.testing.assert_array_almost_equal(X_scaled.mean(axis=0), [0, 0])
        # Variance should not be 1
        assert not np.allclose(X_scaled.var(axis=0), [1, 1])

    def test_zero_variance_feature(self):
        """Test handling of constant features (zero variance)."""
        X = np.array([[1, 0], [1, 1], [1, 2]], dtype=np.float64)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Constant feature should become 0 after centering, scale should be 1
        assert scaler.scale_[0] == 1.0  # Constant feature gets scale=1
        np.testing.assert_array_almost_equal(X_scaled[:, 0], [0, 0, 0])

    def test_n_features_in(self):
        """Test n_features_in_ is set correctly."""
        X = np.random.randn(10, 5)
        scaler = StandardScaler()
        scaler.fit(X)
        assert scaler.n_features_in_ == 5

    def test_n_samples_seen(self):
        """Test n_samples_seen_ is set correctly."""
        X = np.random.randn(100, 5)
        scaler = StandardScaler()
        scaler.fit(X)
        assert scaler.n_samples_seen_ == 100

    def test_not_fitted_error(self):
        """Test error is raised when transforming before fitting."""
        scaler = StandardScaler()
        X = np.random.randn(10, 5)

        with pytest.raises(ValueError, match="Not fitted"):
            scaler.transform(X)

    def test_random_data(self):
        """Test with random data produces zero mean and unit variance."""
        np.random.seed(42)
        X = np.random.randn(1000, 10) * 5 + 3  # Non-standard data

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Check mean is approximately 0
        np.testing.assert_array_almost_equal(X_scaled.mean(axis=0), np.zeros(10), decimal=10)
        # Check variance is approximately 1
        np.testing.assert_array_almost_equal(X_scaled.var(axis=0), np.ones(10), decimal=10)
