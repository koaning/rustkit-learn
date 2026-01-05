"""Tests for KNeighborsClassifier."""

import numpy as np
import pytest

from rklearn import KNeighborsClassifier


class TestKNeighborsClassifierBasic:
    """Basic functionality tests for KNeighborsClassifier."""

    def test_fit_predict_basic(self):
        """Test basic fit and predict."""
        X_train = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        y_train = np.array([0, 1, 0, 1], dtype=np.int64)

        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(X_train, y_train)

        # Predict on training data - should return exact labels with k=1
        predictions = knn.predict(X_train)
        np.testing.assert_array_equal(predictions, y_train)

    def test_fit_returns_self(self):
        """Test that fit returns self for method chaining."""
        X = np.array([[0.0, 0.0], [1.0, 1.0]])
        y = np.array([0, 1], dtype=np.int64)

        knn = KNeighborsClassifier()
        result = knn.fit(X, y)
        assert result is knn

    def test_not_fitted_predict_error(self):
        """Test that predict raises error when not fitted."""
        knn = KNeighborsClassifier()
        X = np.array([[0.0, 0.0]])

        with pytest.raises(ValueError, match="Not fitted"):
            knn.predict(X)

    def test_not_fitted_kneighbors_error(self):
        """Test that kneighbors raises error when not fitted."""
        knn = KNeighborsClassifier()
        X = np.array([[0.0, 0.0]])

        with pytest.raises(ValueError, match="Not fitted"):
            knn.kneighbors(X)

    def test_not_fitted_predict_proba_error(self):
        """Test that predict_proba raises error when not fitted."""
        knn = KNeighborsClassifier()
        X = np.array([[0.0, 0.0]])

        with pytest.raises(ValueError, match="Not fitted"):
            knn.predict_proba(X)

    def test_n_features_in(self):
        """Test n_features_in_ attribute."""
        X = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        y = np.array([0, 1], dtype=np.int64)

        knn = KNeighborsClassifier()
        assert knn.n_features_in_ is None

        knn.fit(X, y)
        assert knn.n_features_in_ == 3

    def test_n_samples_fit(self):
        """Test n_samples_fit_ attribute."""
        X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        y = np.array([0, 1, 2], dtype=np.int64)

        knn = KNeighborsClassifier()
        assert knn.n_samples_fit_ is None

        knn.fit(X, y)
        assert knn.n_samples_fit_ == 3

    def test_classes_attribute(self):
        """Test classes_ attribute."""
        X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        y = np.array([0, 1, 0, 2], dtype=np.int64)

        knn = KNeighborsClassifier()
        assert knn.classes_ is None

        knn.fit(X, y)
        assert knn.classes_ == [0, 1, 2]


class TestKNeighborsClassifierNeighbors:
    """Tests for different n_neighbors values."""

    def test_k_equals_1(self):
        """Test k=1 returns nearest neighbor's class."""
        X_train = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        y_train = np.array([0, 1, 2], dtype=np.int64)

        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(X_train, y_train)

        # Test point closest to first training point
        X_test = np.array([[0.1, 0.1]])
        pred = knn.predict(X_test)
        assert pred[0] == 0  # Nearest is [0, 0] with y=0

    def test_k_equals_3_majority_vote(self):
        """Test k=3 with uniform weights returns majority class."""
        X_train = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
        y_train = np.array([0, 0, 1, 0], dtype=np.int64)

        knn = KNeighborsClassifier(n_neighbors=3, weights="uniform")
        knn.fit(X_train, y_train)

        # Point at [0.3, 0.3] should have 3 nearest neighbors with classes [0, 0, 1]
        # Majority is 0
        X_test = np.array([[0.3, 0.3]])
        pred = knn.predict(X_test)
        assert pred[0] == 0


class TestKNeighborsClassifierWeights:
    """Tests for weight functions."""

    def test_uniform_weights(self):
        """Test uniform weights give equal weight to all neighbors."""
        X_train = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])
        y_train = np.array([0, 1, 1], dtype=np.int64)

        knn = KNeighborsClassifier(n_neighbors=3, weights="uniform")
        knn.fit(X_train, y_train)

        # Point at [0.5, 0] - with uniform weights, class 1 wins (2 vs 1)
        X_test = np.array([[0.5, 0.0]])
        pred = knn.predict(X_test)
        assert pred[0] == 1

    def test_distance_weights(self):
        """Test distance weights give more weight to closer neighbors."""
        X_train = np.array([[0.0, 0.0], [0.1, 0.0], [1.0, 0.0]])
        y_train = np.array([0, 0, 1], dtype=np.int64)

        knn = KNeighborsClassifier(n_neighbors=3, weights="distance")
        knn.fit(X_train, y_train)

        # Point at [0.05, 0]: closer to class 0 neighbors
        # Distance to [0,0] = 0.05, to [0.1,0] = 0.05, to [1,0] = 0.95
        # Weights: 1/0.05 = 20, 1/0.05 = 20, 1/0.95 ≈ 1.05
        # Class 0 weight: 40, Class 1 weight: 1.05 -> Class 0 wins
        X_test = np.array([[0.05, 0.0]])
        pred = knn.predict(X_test)
        assert pred[0] == 0

    def test_distance_weights_zero_distance(self):
        """Test distance weights handle zero distance (exact match)."""
        X_train = np.array([[0.0, 0.0], [1.0, 0.0]])
        y_train = np.array([0, 1], dtype=np.int64)

        knn = KNeighborsClassifier(n_neighbors=2, weights="distance")
        knn.fit(X_train, y_train)

        # Exact match with first training point
        X_test = np.array([[0.0, 0.0]])
        pred = knn.predict(X_test)
        assert pred[0] == 0  # Should return exact match's class


class TestKNeighborsClassifierMetrics:
    """Tests for different distance metrics."""

    def test_euclidean_metric(self):
        """Test Euclidean distance metric."""
        X_train = np.array([[0.0, 0.0], [3.0, 4.0]])
        y_train = np.array([0, 1], dtype=np.int64)

        knn = KNeighborsClassifier(n_neighbors=1, metric="euclidean")
        knn.fit(X_train, y_train)

        # Point at [0, 0] should be closest to class 0
        X_test = np.array([[0.0, 0.0]])
        pred = knn.predict(X_test)
        assert pred[0] == 0

    def test_manhattan_metric(self):
        """Test Manhattan distance metric."""
        X_train = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
        y_train = np.array([0, 1, 2], dtype=np.int64)

        knn = KNeighborsClassifier(n_neighbors=1, metric="manhattan")
        knn.fit(X_train, y_train)

        # Point at [0.5, 0.5]
        # Manhattan distance to [0,0] = 0.5 + 0.5 = 1.0
        # Manhattan distance to [2,0] = 1.5 + 0.5 = 2.0
        # Manhattan distance to [0,2] = 0.5 + 1.5 = 2.0
        # Nearest is [0,0]
        X_test = np.array([[0.5, 0.5]])
        pred = knn.predict(X_test)
        assert pred[0] == 0

    def test_minkowski_metric_p3(self):
        """Test Minkowski distance with p=3."""
        X_train = np.array([[0.0, 0.0], [1.0, 1.0]])
        y_train = np.array([0, 1], dtype=np.int64)

        knn = KNeighborsClassifier(n_neighbors=1, metric="minkowski", p=3.0)
        knn.fit(X_train, y_train)

        # Test point closer to [0,0]
        X_test = np.array([[0.1, 0.1]])
        pred = knn.predict(X_test)
        assert pred[0] == 0


class TestKNeighborsClassifierKneighbors:
    """Tests for kneighbors method."""

    def test_kneighbors_returns_correct_shape(self):
        """Test kneighbors returns arrays with correct shape."""
        X_train = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        y_train = np.array([0, 1, 2, 3], dtype=np.int64)

        knn = KNeighborsClassifier(n_neighbors=2)
        knn.fit(X_train, y_train)

        X_test = np.array([[0.5, 0.5], [0.0, 0.0]])
        distances, indices = knn.kneighbors(X_test)

        assert distances.shape == (2, 2)
        assert indices.shape == (2, 2)

    def test_kneighbors_custom_n_neighbors(self):
        """Test kneighbors with custom n_neighbors."""
        X_train = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        y_train = np.array([0, 1, 2, 3], dtype=np.int64)

        knn = KNeighborsClassifier(n_neighbors=2)
        knn.fit(X_train, y_train)

        X_test = np.array([[0.5, 0.5]])
        distances, indices = knn.kneighbors(X_test, n_neighbors=3)

        assert distances.shape == (1, 3)
        assert indices.shape == (1, 3)

    def test_kneighbors_return_distance_false(self):
        """Test kneighbors with return_distance=False."""
        X_train = np.array([[0.0, 0.0], [1.0, 0.0]])
        y_train = np.array([0, 1], dtype=np.int64)

        knn = KNeighborsClassifier(n_neighbors=2)
        knn.fit(X_train, y_train)

        X_test = np.array([[0.5, 0.0]])
        result = knn.kneighbors(X_test, return_distance=False)

        # Should return only indices, not a tuple
        assert result.shape == (1, 2)

    def test_kneighbors_distances_sorted(self):
        """Test that kneighbors returns neighbors sorted by distance."""
        X_train = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 0.0]])
        y_train = np.array([0, 1, 2], dtype=np.int64)

        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_train, y_train)

        X_test = np.array([[0.0, 0.0]])
        distances, indices = knn.kneighbors(X_test)

        # Distances should be sorted
        assert distances[0, 0] <= distances[0, 1] <= distances[0, 2]
        # First neighbor should be the exact match
        assert distances[0, 0] == 0.0
        assert indices[0, 0] == 0


class TestKNeighborsClassifierPredictProba:
    """Tests for predict_proba method."""

    def test_predict_proba_shape(self):
        """Test predict_proba returns correct shape."""
        X_train = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        y_train = np.array([0, 1, 2], dtype=np.int64)

        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_train, y_train)

        X_test = np.array([[0.5, 0.5], [0.0, 0.0]])
        proba = knn.predict_proba(X_test)

        assert proba.shape == (2, 3)  # 2 samples, 3 classes

    def test_predict_proba_sums_to_one(self):
        """Test that probabilities sum to 1."""
        X_train = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        y_train = np.array([0, 1, 0, 1], dtype=np.int64)

        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_train, y_train)

        X_test = np.array([[0.5, 0.5], [0.2, 0.3]])
        proba = knn.predict_proba(X_test)

        # Each row should sum to 1
        np.testing.assert_array_almost_equal(proba.sum(axis=1), [1.0, 1.0])

    def test_predict_proba_uniform_weights(self):
        """Test predict_proba with uniform weights."""
        X_train = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        y_train = np.array([0, 0, 1], dtype=np.int64)

        knn = KNeighborsClassifier(n_neighbors=3, weights="uniform")
        knn.fit(X_train, y_train)

        X_test = np.array([[0.5, 0.5]])
        proba = knn.predict_proba(X_test)

        # With 3 neighbors where 2 are class 0 and 1 is class 1
        # Probability should be 2/3 for class 0, 1/3 for class 1
        np.testing.assert_array_almost_equal(proba[0], [2 / 3, 1 / 3])

    def test_predict_proba_exact_match(self):
        """Test predict_proba with exact match returns probability 1."""
        X_train = np.array([[0.0, 0.0], [1.0, 0.0]])
        y_train = np.array([0, 1], dtype=np.int64)

        knn = KNeighborsClassifier(n_neighbors=2, weights="distance")
        knn.fit(X_train, y_train)

        # Exact match with first point
        X_test = np.array([[0.0, 0.0]])
        proba = knn.predict_proba(X_test)

        # Should be 1.0 for class 0, 0.0 for class 1
        np.testing.assert_array_almost_equal(proba[0], [1.0, 0.0])


class TestKNeighborsClassifierScore:
    """Tests for score method."""

    def test_score_perfect_prediction(self):
        """Test score is 1.0 for perfect predictions."""
        X_train = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        y_train = np.array([0, 1, 2], dtype=np.int64)

        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(X_train, y_train)

        score = knn.score(X_train, y_train)
        np.testing.assert_almost_equal(score, 1.0)

    def test_score_returns_accuracy(self):
        """Test that score returns accuracy value."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(np.int64)

        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X, y)

        score = knn.score(X, y)
        # Should be a valid accuracy (between 0 and 1)
        assert 0.0 <= score <= 1.0


class TestKNeighborsClassifierParallel:
    """Tests for parallel execution."""

    @pytest.fixture
    def large_data(self):
        np.random.seed(42)
        X = np.random.randn(1000, 10)
        y = np.random.randint(0, 5, 1000).astype(np.int64)
        return X, y

    def test_parallel_predict_matches_single(self, large_data):
        """Test parallel predict gives same results as single-threaded."""
        X, y = large_data

        knn_single = KNeighborsClassifier(n_neighbors=5, n_jobs=1)
        knn_parallel = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)

        knn_single.fit(X, y)
        knn_parallel.fit(X, y)

        X_test = X[:100]
        pred_single = knn_single.predict(X_test)
        pred_parallel = knn_parallel.predict(X_test)

        np.testing.assert_array_equal(pred_single, pred_parallel)

    def test_parallel_kneighbors_matches_single(self, large_data):
        """Test parallel kneighbors gives same results as single-threaded."""
        X, y = large_data

        knn_single = KNeighborsClassifier(n_neighbors=5, n_jobs=1)
        knn_parallel = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)

        knn_single.fit(X, y)
        knn_parallel.fit(X, y)

        X_test = X[:100]
        dist_single, idx_single = knn_single.kneighbors(X_test)
        dist_parallel, idx_parallel = knn_parallel.kneighbors(X_test)

        np.testing.assert_array_almost_equal(dist_single, dist_parallel)
        np.testing.assert_array_equal(idx_single, idx_parallel)

    def test_parallel_predict_proba_matches_single(self, large_data):
        """Test parallel predict_proba gives same results as single-threaded."""
        X, y = large_data

        knn_single = KNeighborsClassifier(n_neighbors=5, n_jobs=1)
        knn_parallel = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)

        knn_single.fit(X, y)
        knn_parallel.fit(X, y)

        X_test = X[:100]
        proba_single = knn_single.predict_proba(X_test)
        proba_parallel = knn_parallel.predict_proba(X_test)

        np.testing.assert_array_almost_equal(proba_single, proba_parallel)


class TestKNeighborsClassifierGetParams:
    """Tests for get_params method."""

    def test_get_params_returns_all_params(self):
        """Test get_params returns all constructor parameters."""
        knn = KNeighborsClassifier(
            n_neighbors=7,
            weights="distance",
            algorithm="ball_tree",
            leaf_size=40,
            metric="manhattan",
            p=1.5,
            n_jobs=2,
        )

        params = knn.get_params()

        assert params["n_neighbors"] == 7
        assert params["weights"] == "distance"
        assert params["algorithm"] == "ball_tree"
        assert params["leaf_size"] == 40
        assert params["metric"] == "manhattan"
        assert params["p"] == 1.5
        assert params["n_jobs"] == 2

    def test_get_params_default_values(self):
        """Test get_params returns default values."""
        knn = KNeighborsClassifier()
        params = knn.get_params()

        assert params["n_neighbors"] == 5
        assert params["weights"] == "uniform"
        assert params["algorithm"] == "auto"
        assert params["leaf_size"] == 30
        assert params["metric"] == "euclidean"
        assert params["p"] == 2.0
        assert params["n_jobs"] == 1
