"""Tests comparing KNeighborsRegressor against scikit-learn for numerical accuracy."""

import numpy as np
import pytest

sklearn = pytest.importorskip("sklearn")
from sklearn.neighbors import KNeighborsRegressor as SklearnKNeighborsRegressor

from rklearn import KNeighborsRegressor


class TestKNeighborsRegressorSklearnCompat:
    """Verify KNeighborsRegressor produces identical results to sklearn."""

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        X = np.random.randn(200, 5)
        y = X[:, 0] * 2 + X[:, 1] - X[:, 2] * 0.5 + np.random.randn(200) * 0.1
        return X, y

    @pytest.fixture
    def train_test_split(self, sample_data):
        X, y = sample_data
        return X[:150], X[150:], y[:150], y[150:]

    def test_predict_matches_euclidean(self, train_test_split):
        """Verify predict produces identical results with Euclidean distance."""
        X_train, X_test, y_train, y_test = train_test_split

        rust_knn = KNeighborsRegressor(n_neighbors=5, metric="euclidean", weights="uniform")
        sklearn_knn = SklearnKNeighborsRegressor(n_neighbors=5, metric="euclidean", weights="uniform")

        rust_knn.fit(X_train, y_train)
        sklearn_knn.fit(X_train, y_train)

        rust_pred = rust_knn.predict(X_test)
        sklearn_pred = sklearn_knn.predict(X_test)

        np.testing.assert_array_almost_equal(rust_pred, sklearn_pred, decimal=10)

    def test_predict_matches_manhattan(self, train_test_split):
        """Verify predict produces identical results with Manhattan distance."""
        X_train, X_test, y_train, y_test = train_test_split

        rust_knn = KNeighborsRegressor(n_neighbors=5, metric="manhattan", weights="uniform")
        sklearn_knn = SklearnKNeighborsRegressor(n_neighbors=5, metric="manhattan", weights="uniform")

        rust_knn.fit(X_train, y_train)
        sklearn_knn.fit(X_train, y_train)

        rust_pred = rust_knn.predict(X_test)
        sklearn_pred = sklearn_knn.predict(X_test)

        np.testing.assert_array_almost_equal(rust_pred, sklearn_pred, decimal=10)

    def test_predict_matches_minkowski_p3(self, train_test_split):
        """Verify predict produces identical results with Minkowski p=3."""
        X_train, X_test, y_train, y_test = train_test_split

        rust_knn = KNeighborsRegressor(n_neighbors=5, metric="minkowski", p=3.0, weights="uniform")
        sklearn_knn = SklearnKNeighborsRegressor(n_neighbors=5, metric="minkowski", p=3, weights="uniform")

        rust_knn.fit(X_train, y_train)
        sklearn_knn.fit(X_train, y_train)

        rust_pred = rust_knn.predict(X_test)
        sklearn_pred = sklearn_knn.predict(X_test)

        np.testing.assert_array_almost_equal(rust_pred, sklearn_pred, decimal=10)

    def test_predict_matches_distance_weights(self, train_test_split):
        """Verify predict produces identical results with distance weights."""
        X_train, X_test, y_train, y_test = train_test_split

        rust_knn = KNeighborsRegressor(n_neighbors=5, metric="euclidean", weights="distance")
        sklearn_knn = SklearnKNeighborsRegressor(n_neighbors=5, metric="euclidean", weights="distance")

        rust_knn.fit(X_train, y_train)
        sklearn_knn.fit(X_train, y_train)

        rust_pred = rust_knn.predict(X_test)
        sklearn_pred = sklearn_knn.predict(X_test)

        np.testing.assert_array_almost_equal(rust_pred, sklearn_pred, decimal=10)

    def test_kneighbors_distances_match(self, train_test_split):
        """Verify kneighbors distances match sklearn."""
        X_train, X_test, y_train, y_test = train_test_split

        rust_knn = KNeighborsRegressor(n_neighbors=5, metric="euclidean")
        sklearn_knn = SklearnKNeighborsRegressor(n_neighbors=5, metric="euclidean")

        rust_knn.fit(X_train, y_train)
        sklearn_knn.fit(X_train, y_train)

        rust_dist, rust_idx = rust_knn.kneighbors(X_test)
        sklearn_dist, sklearn_idx = sklearn_knn.kneighbors(X_test)

        np.testing.assert_array_almost_equal(rust_dist, sklearn_dist, decimal=10)

    def test_kneighbors_indices_match(self, train_test_split):
        """Verify kneighbors indices match sklearn."""
        X_train, X_test, y_train, y_test = train_test_split

        rust_knn = KNeighborsRegressor(n_neighbors=5, metric="euclidean")
        sklearn_knn = SklearnKNeighborsRegressor(n_neighbors=5, metric="euclidean")

        rust_knn.fit(X_train, y_train)
        sklearn_knn.fit(X_train, y_train)

        rust_dist, rust_idx = rust_knn.kneighbors(X_test)
        sklearn_dist, sklearn_idx = sklearn_knn.kneighbors(X_test)

        np.testing.assert_array_equal(rust_idx, sklearn_idx)

    def test_score_matches(self, train_test_split):
        """Verify score matches sklearn."""
        X_train, X_test, y_train, y_test = train_test_split

        rust_knn = KNeighborsRegressor(n_neighbors=5)
        sklearn_knn = SklearnKNeighborsRegressor(n_neighbors=5)

        rust_knn.fit(X_train, y_train)
        sklearn_knn.fit(X_train, y_train)

        rust_score = rust_knn.score(X_test, y_test)
        sklearn_score = sklearn_knn.score(X_test, y_test)

        np.testing.assert_almost_equal(rust_score, sklearn_score, decimal=10)

    def test_different_k_values(self, train_test_split):
        """Verify predictions match for different k values."""
        X_train, X_test, y_train, y_test = train_test_split

        for k in [1, 3, 10, 20]:
            rust_knn = KNeighborsRegressor(n_neighbors=k)
            sklearn_knn = SklearnKNeighborsRegressor(n_neighbors=k)

            rust_knn.fit(X_train, y_train)
            sklearn_knn.fit(X_train, y_train)

            rust_pred = rust_knn.predict(X_test)
            sklearn_pred = sklearn_knn.predict(X_test)

            np.testing.assert_array_almost_equal(
                rust_pred, sklearn_pred, decimal=10,
                err_msg=f"Mismatch for k={k}"
            )

    def test_n_features_in_matches(self, train_test_split):
        """Verify n_features_in_ matches sklearn."""
        X_train, X_test, y_train, y_test = train_test_split

        rust_knn = KNeighborsRegressor()
        sklearn_knn = SklearnKNeighborsRegressor()

        rust_knn.fit(X_train, y_train)
        sklearn_knn.fit(X_train, y_train)

        assert rust_knn.n_features_in_ == sklearn_knn.n_features_in_

    def test_predict_on_training_data_k1(self, train_test_split):
        """Verify k=1 on training data returns exact targets."""
        X_train, X_test, y_train, y_test = train_test_split

        rust_knn = KNeighborsRegressor(n_neighbors=1)
        sklearn_knn = SklearnKNeighborsRegressor(n_neighbors=1)

        rust_knn.fit(X_train, y_train)
        sklearn_knn.fit(X_train, y_train)

        rust_pred = rust_knn.predict(X_train)
        sklearn_pred = sklearn_knn.predict(X_train)

        # Both should return exact training targets
        np.testing.assert_array_almost_equal(rust_pred, y_train, decimal=10)
        np.testing.assert_array_almost_equal(sklearn_pred, y_train, decimal=10)
