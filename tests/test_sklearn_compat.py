"""Tests comparing rustscaler against scikit-learn for numerical accuracy."""

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler
from sklearn.preprocessing import MinMaxScaler as SklearnMinMaxScaler

from rklearn import StandardScaler, MinMaxScaler


class TestStandardScalerSklearnCompat:
    """Verify StandardScaler produces identical results to sklearn."""

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        return np.random.randn(1000, 20)

    def test_fit_transform_matches(self, sample_data):
        """Verify fit_transform produces identical results."""
        rust_scaler = StandardScaler()
        sklearn_scaler = SklearnStandardScaler()

        rust_result = rust_scaler.fit_transform(sample_data)
        sklearn_result = sklearn_scaler.fit_transform(sample_data)

        np.testing.assert_array_almost_equal(rust_result, sklearn_result, decimal=10)

    def test_mean_matches(self, sample_data):
        """Verify computed mean matches sklearn."""
        rust_scaler = StandardScaler()
        sklearn_scaler = SklearnStandardScaler()

        rust_scaler.fit(sample_data)
        sklearn_scaler.fit(sample_data)

        np.testing.assert_array_almost_equal(rust_scaler.mean_, sklearn_scaler.mean_, decimal=10)

    def test_scale_matches(self, sample_data):
        """Verify computed scale (std) matches sklearn."""
        rust_scaler = StandardScaler()
        sklearn_scaler = SklearnStandardScaler()

        rust_scaler.fit(sample_data)
        sklearn_scaler.fit(sample_data)

        np.testing.assert_array_almost_equal(rust_scaler.scale_, sklearn_scaler.scale_, decimal=10)

    def test_var_matches(self, sample_data):
        """Verify computed variance matches sklearn."""
        rust_scaler = StandardScaler()
        sklearn_scaler = SklearnStandardScaler()

        rust_scaler.fit(sample_data)
        sklearn_scaler.fit(sample_data)

        np.testing.assert_array_almost_equal(rust_scaler.var_, sklearn_scaler.var_, decimal=10)

    def test_inverse_transform_matches(self, sample_data):
        """Verify inverse_transform produces identical results."""
        rust_scaler = StandardScaler()
        sklearn_scaler = SklearnStandardScaler()

        rust_scaled = rust_scaler.fit_transform(sample_data)
        sklearn_scaled = sklearn_scaler.fit_transform(sample_data)

        rust_recovered = rust_scaler.inverse_transform(rust_scaled)
        sklearn_recovered = sklearn_scaler.inverse_transform(sklearn_scaled)

        np.testing.assert_array_almost_equal(rust_recovered, sklearn_recovered, decimal=10)

    def test_with_mean_false_matches(self, sample_data):
        """Verify with_mean=False produces identical results."""
        rust_scaler = StandardScaler(with_mean=False)
        sklearn_scaler = SklearnStandardScaler(with_mean=False)

        rust_result = rust_scaler.fit_transform(sample_data)
        sklearn_result = sklearn_scaler.fit_transform(sample_data)

        np.testing.assert_array_almost_equal(rust_result, sklearn_result, decimal=10)

    def test_with_std_false_matches(self, sample_data):
        """Verify with_std=False produces identical results."""
        rust_scaler = StandardScaler(with_std=False)
        sklearn_scaler = SklearnStandardScaler(with_std=False)

        rust_result = rust_scaler.fit_transform(sample_data)
        sklearn_result = sklearn_scaler.fit_transform(sample_data)

        np.testing.assert_array_almost_equal(rust_result, sklearn_result, decimal=10)

    def test_partial_fit_matches(self, sample_data):
        """Verify partial_fit produces results close to sklearn."""
        # Split data into batches
        batch_size = 100
        batches = [sample_data[i:i+batch_size] for i in range(0, len(sample_data), batch_size)]

        rust_scaler = StandardScaler()
        sklearn_scaler = SklearnStandardScaler()

        for batch in batches:
            rust_scaler.partial_fit(batch)
            sklearn_scaler.partial_fit(batch)

        # Mean should match closely
        np.testing.assert_array_almost_equal(rust_scaler.mean_, sklearn_scaler.mean_, decimal=8)
        # Variance may have small differences due to online algorithm
        np.testing.assert_array_almost_equal(rust_scaler.var_, sklearn_scaler.var_, decimal=6)


class TestMinMaxScalerSklearnCompat:
    """Verify MinMaxScaler produces identical results to sklearn."""

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        return np.random.randn(1000, 20) * 10 + 5

    def test_fit_transform_matches(self, sample_data):
        """Verify fit_transform produces identical results."""
        rust_scaler = MinMaxScaler()
        sklearn_scaler = SklearnMinMaxScaler()

        rust_result = rust_scaler.fit_transform(sample_data)
        sklearn_result = sklearn_scaler.fit_transform(sample_data)

        np.testing.assert_array_almost_equal(rust_result, sklearn_result, decimal=10)

    def test_data_min_matches(self, sample_data):
        """Verify computed data_min matches sklearn."""
        rust_scaler = MinMaxScaler()
        sklearn_scaler = SklearnMinMaxScaler()

        rust_scaler.fit(sample_data)
        sklearn_scaler.fit(sample_data)

        np.testing.assert_array_almost_equal(rust_scaler.data_min_, sklearn_scaler.data_min_, decimal=10)

    def test_data_max_matches(self, sample_data):
        """Verify computed data_max matches sklearn."""
        rust_scaler = MinMaxScaler()
        sklearn_scaler = SklearnMinMaxScaler()

        rust_scaler.fit(sample_data)
        sklearn_scaler.fit(sample_data)

        np.testing.assert_array_almost_equal(rust_scaler.data_max_, sklearn_scaler.data_max_, decimal=10)

    def test_scale_matches(self, sample_data):
        """Verify computed scale matches sklearn."""
        rust_scaler = MinMaxScaler()
        sklearn_scaler = SklearnMinMaxScaler()

        rust_scaler.fit(sample_data)
        sklearn_scaler.fit(sample_data)

        np.testing.assert_array_almost_equal(rust_scaler.scale_, sklearn_scaler.scale_, decimal=10)

    def test_inverse_transform_matches(self, sample_data):
        """Verify inverse_transform produces identical results."""
        rust_scaler = MinMaxScaler()
        sklearn_scaler = SklearnMinMaxScaler()

        rust_scaled = rust_scaler.fit_transform(sample_data)
        sklearn_scaled = sklearn_scaler.fit_transform(sample_data)

        rust_recovered = rust_scaler.inverse_transform(rust_scaled)
        sklearn_recovered = sklearn_scaler.inverse_transform(sklearn_scaled)

        np.testing.assert_array_almost_equal(rust_recovered, sklearn_recovered, decimal=10)

    def test_custom_range_matches(self, sample_data):
        """Verify custom feature_range produces identical results."""
        rust_scaler = MinMaxScaler(feature_range=(-1, 1))
        sklearn_scaler = SklearnMinMaxScaler(feature_range=(-1, 1))

        rust_result = rust_scaler.fit_transform(sample_data)
        sklearn_result = sklearn_scaler.fit_transform(sample_data)

        np.testing.assert_array_almost_equal(rust_result, sklearn_result, decimal=10)

    def test_clip_matches(self, sample_data):
        """Verify clip parameter produces identical results."""
        # Train on subset, test on full data
        train_data = sample_data[:500]
        test_data = sample_data

        rust_scaler = MinMaxScaler(clip=True)
        sklearn_scaler = SklearnMinMaxScaler(clip=True)

        rust_scaler.fit(train_data)
        sklearn_scaler.fit(train_data)

        rust_result = rust_scaler.transform(test_data)
        sklearn_result = sklearn_scaler.transform(test_data)

        np.testing.assert_array_almost_equal(rust_result, sklearn_result, decimal=10)

    def test_partial_fit_matches(self, sample_data):
        """Verify partial_fit produces results identical to sklearn."""
        # Split data into batches
        batch_size = 100
        batches = [sample_data[i:i+batch_size] for i in range(0, len(sample_data), batch_size)]

        rust_scaler = MinMaxScaler()
        sklearn_scaler = SklearnMinMaxScaler()

        for batch in batches:
            rust_scaler.partial_fit(batch)
            sklearn_scaler.partial_fit(batch)

        np.testing.assert_array_almost_equal(rust_scaler.data_min_, sklearn_scaler.data_min_, decimal=10)
        np.testing.assert_array_almost_equal(rust_scaler.data_max_, sklearn_scaler.data_max_, decimal=10)
