"""Tests for KNeighborsRegressor."""

import numpy as np
import pytest

from rklearn import KNeighborsRegressor


class TestKNeighborsRegressorBasic:
    """Basic functionality tests for KNeighborsRegressor."""

    def test_fit_predict_basic(self):
        """Test basic fit and predict."""
        X_train = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        y_train = np.array([0.0, 1.0, 2.0, 3.0])

        knn = KNeighborsRegressor(n_neighbors=1)
        knn.fit(X_train, y_train)

        # Predict on training data - should return exact values with k=1
        predictions = knn.predict(X_train)
        np.testing.assert_array_almost_equal(predictions, y_train)

    def test_fit_returns_self(self):
        """Test that fit returns self for method chaining."""
        X = np.array([[0.0, 0.0], [1.0, 1.0]])
        y = np.array([0.0, 1.0])

        knn = KNeighborsRegressor()
        result = knn.fit(X, y)
        assert result is knn

    def test_not_fitted_predict_error(self):
        """Test that predict raises error when not fitted."""
        knn = KNeighborsRegressor()
        X = np.array([[0.0, 0.0]])

        with pytest.raises(ValueError, match="Not fitted"):
            knn.predict(X)

    def test_not_fitted_kneighbors_error(self):
        """Test that kneighbors raises error when not fitted."""
        knn = KNeighborsRegressor()
        X = np.array([[0.0, 0.0]])

        with pytest.raises(ValueError, match="Not fitted"):
            knn.kneighbors(X)

    def test_n_features_in(self):
        """Test n_features_in_ attribute."""
        X = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        y = np.array([0.0, 1.0])

        knn = KNeighborsRegressor()
        assert knn.n_features_in_ is None

        knn.fit(X, y)
        assert knn.n_features_in_ == 3

    def test_n_samples_fit(self):
        """Test n_samples_fit_ attribute."""
        X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        y = np.array([0.0, 1.0, 2.0])

        knn = KNeighborsRegressor()
        assert knn.n_samples_fit_ is None

        knn.fit(X, y)
        assert knn.n_samples_fit_ == 3


class TestKNeighborsRegressorNeighbors:
    """Tests for different n_neighbors values."""

    def test_k_equals_1(self):
        """Test k=1 returns nearest neighbor's value."""
        X_train = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        y_train = np.array([0.0, 1.0, 2.0])

        knn = KNeighborsRegressor(n_neighbors=1)
        knn.fit(X_train, y_train)

        # Test point closest to first training point
        X_test = np.array([[0.1, 0.1]])
        pred = knn.predict(X_test)
        assert pred[0] == 0.0  # Nearest is [0, 0] with y=0

    def test_k_equals_3_uniform(self):
        """Test k=3 with uniform weights returns average."""
        X_train = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        y_train = np.array([0.0, 3.0, 6.0])

        knn = KNeighborsRegressor(n_neighbors=3, weights="uniform")
        knn.fit(X_train, y_train)

        # Point equidistant from all - should return average
        X_test = np.array([[0.5, 0.5]])
        pred = knn.predict(X_test)
        expected = (0.0 + 3.0 + 6.0) / 3.0
        np.testing.assert_almost_equal(pred[0], expected)


class TestKNeighborsRegressorWeights:
    """Tests for weight functions."""

    def test_uniform_weights(self):
        """Test uniform weights give equal weight to all neighbors."""
        X_train = np.array([[0.0, 0.0], [2.0, 0.0]])
        y_train = np.array([0.0, 10.0])

        knn = KNeighborsRegressor(n_neighbors=2, weights="uniform")
        knn.fit(X_train, y_train)

        # Point at [1, 0] is equidistant from both
        X_test = np.array([[1.0, 0.0]])
        pred = knn.predict(X_test)
        expected = 5.0  # (0 + 10) / 2
        np.testing.assert_almost_equal(pred[0], expected)

    def test_distance_weights(self):
        """Test distance weights give more weight to closer neighbors."""
        X_train = np.array([[0.0, 0.0], [3.0, 0.0]])
        y_train = np.array([0.0, 12.0])

        knn = KNeighborsRegressor(n_neighbors=2, weights="distance")
        knn.fit(X_train, y_train)

        # Point at [1, 0]: distance to [0,0] = 1, distance to [3,0] = 2
        # Weights: 1/1 = 1, 1/2 = 0.5
        # Prediction: (1*0 + 0.5*12) / (1 + 0.5) = 6 / 1.5 = 4
        X_test = np.array([[1.0, 0.0]])
        pred = knn.predict(X_test)
        expected = 4.0
        np.testing.assert_almost_equal(pred[0], expected)

    def test_distance_weights_zero_distance(self):
        """Test distance weights handle zero distance (exact match)."""
        X_train = np.array([[0.0, 0.0], [1.0, 0.0]])
        y_train = np.array([5.0, 10.0])

        knn = KNeighborsRegressor(n_neighbors=2, weights="distance")
        knn.fit(X_train, y_train)

        # Exact match with first training point
        X_test = np.array([[0.0, 0.0]])
        pred = knn.predict(X_test)
        assert pred[0] == 5.0  # Should return exact value


class TestKNeighborsRegressorMetrics:
    """Tests for different distance metrics."""

    def test_euclidean_metric(self):
        """Test Euclidean distance metric."""
        X_train = np.array([[0.0, 0.0], [3.0, 4.0]])
        y_train = np.array([0.0, 1.0])

        knn = KNeighborsRegressor(n_neighbors=1, metric="euclidean")
        knn.fit(X_train, y_train)

        # Point at [1.5, 2] - distance to [0,0] = sqrt(1.5^2 + 2^2) = 2.5
        # distance to [3,4] = sqrt(1.5^2 + 2^2) = 2.5 (equidistant)
        # Since k=1, it picks the first one found
        X_test = np.array([[0.0, 0.0]])
        pred = knn.predict(X_test)
        assert pred[0] == 0.0

    def test_manhattan_metric(self):
        """Test Manhattan distance metric."""
        X_train = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
        y_train = np.array([0.0, 1.0, 2.0])

        knn = KNeighborsRegressor(n_neighbors=1, metric="manhattan")
        knn.fit(X_train, y_train)

        # Point at [0.5, 0.5]
        # Manhattan distance to [0,0] = 0.5 + 0.5 = 1.0
        # Manhattan distance to [2,0] = 1.5 + 0.5 = 2.0
        # Manhattan distance to [0,2] = 0.5 + 1.5 = 2.0
        # Nearest is [0,0]
        X_test = np.array([[0.5, 0.5]])
        pred = knn.predict(X_test)
        assert pred[0] == 0.0

    def test_minkowski_metric_p3(self):
        """Test Minkowski distance with p=3."""
        X_train = np.array([[0.0, 0.0], [1.0, 1.0]])
        y_train = np.array([0.0, 10.0])

        knn = KNeighborsRegressor(n_neighbors=1, metric="minkowski", p=3.0)
        knn.fit(X_train, y_train)

        # Test point closer to [0,0]
        X_test = np.array([[0.1, 0.1]])
        pred = knn.predict(X_test)
        assert pred[0] == 0.0


class TestKNeighborsRegressorKneighbors:
    """Tests for kneighbors method."""

    def test_kneighbors_returns_correct_shape(self):
        """Test kneighbors returns arrays with correct shape."""
        X_train = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        y_train = np.array([0.0, 1.0, 2.0, 3.0])

        knn = KNeighborsRegressor(n_neighbors=2)
        knn.fit(X_train, y_train)

        X_test = np.array([[0.5, 0.5], [0.0, 0.0]])
        distances, indices = knn.kneighbors(X_test)

        assert distances.shape == (2, 2)
        assert indices.shape == (2, 2)

    def test_kneighbors_custom_n_neighbors(self):
        """Test kneighbors with custom n_neighbors."""
        X_train = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        y_train = np.array([0.0, 1.0, 2.0, 3.0])

        knn = KNeighborsRegressor(n_neighbors=2)
        knn.fit(X_train, y_train)

        X_test = np.array([[0.5, 0.5]])
        distances, indices = knn.kneighbors(X_test, n_neighbors=3)

        assert distances.shape == (1, 3)
        assert indices.shape == (1, 3)

    def test_kneighbors_return_distance_false(self):
        """Test kneighbors with return_distance=False."""
        X_train = np.array([[0.0, 0.0], [1.0, 0.0]])
        y_train = np.array([0.0, 1.0])

        knn = KNeighborsRegressor(n_neighbors=2)
        knn.fit(X_train, y_train)

        X_test = np.array([[0.5, 0.0]])
        result = knn.kneighbors(X_test, return_distance=False)

        # Should return only indices, not a tuple
        assert result.shape == (1, 2)

    def test_kneighbors_distances_sorted(self):
        """Test that kneighbors returns neighbors sorted by distance."""
        X_train = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 0.0]])
        y_train = np.array([0.0, 1.0, 2.0])

        knn = KNeighborsRegressor(n_neighbors=3)
        knn.fit(X_train, y_train)

        X_test = np.array([[0.0, 0.0]])
        distances, indices = knn.kneighbors(X_test)

        # Distances should be sorted
        assert distances[0, 0] <= distances[0, 1] <= distances[0, 2]
        # First neighbor should be the exact match
        assert distances[0, 0] == 0.0
        assert indices[0, 0] == 0


class TestKNeighborsRegressorScore:
    """Tests for score method."""

    def test_score_perfect_prediction(self):
        """Test score is 1.0 for perfect predictions."""
        X_train = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        y_train = np.array([0.0, 1.0, 2.0])

        knn = KNeighborsRegressor(n_neighbors=1)
        knn.fit(X_train, y_train)

        score = knn.score(X_train, y_train)
        np.testing.assert_almost_equal(score, 1.0)

    def test_score_returns_r2(self):
        """Test that score returns R^2 value."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = X[:, 0] + X[:, 1] + np.random.randn(100) * 0.1

        knn = KNeighborsRegressor(n_neighbors=5)
        knn.fit(X, y)

        score = knn.score(X, y)
        # Should be a valid R^2 (between -inf and 1)
        assert score <= 1.0


class TestKNeighborsRegressorParallel:
    """Tests for parallel execution."""

    @pytest.fixture
    def large_data(self):
        np.random.seed(42)
        X = np.random.randn(1000, 10)
        y = np.random.randn(1000)
        return X, y

    def test_parallel_predict_matches_single(self, large_data):
        """Test parallel predict gives same results as single-threaded."""
        X, y = large_data

        knn_single = KNeighborsRegressor(n_neighbors=5, n_jobs=1)
        knn_parallel = KNeighborsRegressor(n_neighbors=5, n_jobs=-1)

        knn_single.fit(X, y)
        knn_parallel.fit(X, y)

        X_test = X[:100]
        pred_single = knn_single.predict(X_test)
        pred_parallel = knn_parallel.predict(X_test)

        np.testing.assert_array_almost_equal(pred_single, pred_parallel)

    def test_parallel_kneighbors_matches_single(self, large_data):
        """Test parallel kneighbors gives same results as single-threaded."""
        X, y = large_data

        knn_single = KNeighborsRegressor(n_neighbors=5, n_jobs=1)
        knn_parallel = KNeighborsRegressor(n_neighbors=5, n_jobs=-1)

        knn_single.fit(X, y)
        knn_parallel.fit(X, y)

        X_test = X[:100]
        dist_single, idx_single = knn_single.kneighbors(X_test)
        dist_parallel, idx_parallel = knn_parallel.kneighbors(X_test)

        np.testing.assert_array_almost_equal(dist_single, dist_parallel)
        np.testing.assert_array_equal(idx_single, idx_parallel)
