import numpy as np
import pytest
from rklearn import MinMaxScaler


class TestMinMaxScaler:
    def test_fit_basic(self):
        """Test basic fit functionality."""
        X = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
        scaler = MinMaxScaler()
        scaler.fit(X)

        np.testing.assert_array_almost_equal(scaler.data_min_, [1, 2])
        np.testing.assert_array_almost_equal(scaler.data_max_, [5, 6])
        np.testing.assert_array_almost_equal(scaler.data_range_, [4, 4])

    def test_transform(self):
        """Test transform produces expected output."""
        X = np.array([[0, 0], [1, 1], [2, 2]], dtype=np.float64)
        scaler = MinMaxScaler()
        scaler.fit(X)

        X_scaled = scaler.transform(X)
        expected = np.array([[0, 0], [0.5, 0.5], [1, 1]])
        np.testing.assert_array_almost_equal(X_scaled, expected)

    def test_fit_transform(self):
        """Test fit_transform produces same result as fit then transform."""
        X = np.array([[0, 0], [1, 1], [2, 2]], dtype=np.float64)

        scaler1 = MinMaxScaler()
        result1 = scaler1.fit_transform(X)

        scaler2 = MinMaxScaler()
        scaler2.fit(X)
        result2 = scaler2.transform(X)

        np.testing.assert_array_almost_equal(result1, result2)

    def test_inverse_transform(self):
        """Test inverse_transform recovers original data."""
        X = np.random.randn(100, 5) * 10 + 5
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        X_recovered = scaler.inverse_transform(X_scaled)

        np.testing.assert_array_almost_equal(X, X_recovered)

    def test_custom_feature_range(self):
        """Test with custom feature_range."""
        X = np.array([[0, 0], [1, 1], [2, 2]], dtype=np.float64)
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X_scaled = scaler.fit_transform(X)

        expected = np.array([[-1, -1], [0, 0], [1, 1]])
        np.testing.assert_array_almost_equal(X_scaled, expected)

    def test_clip(self):
        """Test clip parameter clips values outside feature_range."""
        X_train = np.array([[0, 0], [1, 1], [2, 2]], dtype=np.float64)
        X_test = np.array([[-1, -1], [3, 3]], dtype=np.float64)

        scaler = MinMaxScaler(clip=True)
        scaler.fit(X_train)
        X_scaled = scaler.transform(X_test)

        # Values should be clipped to [0, 1]
        assert X_scaled.min() >= 0
        assert X_scaled.max() <= 1

    def test_partial_fit_single_batch(self):
        """Test partial_fit with a single batch matches fit."""
        X = np.array([[0, 0], [1, 1], [2, 2]], dtype=np.float64)

        scaler1 = MinMaxScaler()
        scaler1.fit(X)

        scaler2 = MinMaxScaler()
        scaler2.partial_fit(X)

        np.testing.assert_array_almost_equal(scaler1.data_min_, scaler2.data_min_)
        np.testing.assert_array_almost_equal(scaler1.data_max_, scaler2.data_max_)

    def test_partial_fit_multiple_batches(self):
        """Test partial_fit with multiple batches."""
        X1 = np.array([[0, 0], [1, 1]], dtype=np.float64)
        X2 = np.array([[2, 2], [3, 3]], dtype=np.float64)

        scaler = MinMaxScaler()
        scaler.partial_fit(X1)
        scaler.partial_fit(X2)

        # Min/max should be global across batches
        np.testing.assert_array_almost_equal(scaler.data_min_, [0, 0])
        np.testing.assert_array_almost_equal(scaler.data_max_, [3, 3])
        assert scaler.n_samples_seen_ == 4

    def test_zero_range_feature(self):
        """Test handling of constant features (zero range)."""
        X = np.array([[1, 0], [1, 1], [1, 2]], dtype=np.float64)
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        # Constant feature should map to min of feature_range (0)
        np.testing.assert_array_almost_equal(X_scaled[:, 0], [0, 0, 0])

    def test_n_features_in(self):
        """Test n_features_in_ is set correctly."""
        X = np.random.randn(10, 5)
        scaler = MinMaxScaler()
        scaler.fit(X)
        assert scaler.n_features_in_ == 5

    def test_n_samples_seen(self):
        """Test n_samples_seen_ is set correctly."""
        X = np.random.randn(100, 5)
        scaler = MinMaxScaler()
        scaler.fit(X)
        assert scaler.n_samples_seen_ == 100

    def test_not_fitted_error(self):
        """Test error is raised when transforming before fitting."""
        scaler = MinMaxScaler()
        X = np.random.randn(10, 5)

        with pytest.raises(ValueError, match="Not fitted"):
            scaler.transform(X)

    def test_random_data(self):
        """Test with random data produces values in [0, 1]."""
        np.random.seed(42)
        X = np.random.randn(1000, 10) * 5 + 3

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        # Check values are approximately in [0, 1] (allowing floating point tolerance)
        assert X_scaled.min() >= -1e-10
        assert X_scaled.max() <= 1 + 1e-10
        # Check actual min and max are approximately 0 and 1
        np.testing.assert_array_almost_equal(X_scaled.min(axis=0), np.zeros(10))
        np.testing.assert_array_almost_equal(X_scaled.max(axis=0), np.ones(10))
