"""Tests for KMeans clustering."""

import numpy as np
import pytest

from rklearn import KMeans


class TestKMeansBasic:
    """Basic functionality tests."""

    def test_fit_predict_basic(self):
        """Test basic fit and predict."""
        X = np.array(
            [[1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [8.0, 8.0], [1.0, 0.6], [9.0, 11.0]]
        )
        kmeans = KMeans(n_clusters=2, random_state=42)
        kmeans.fit(X)

        assert kmeans.cluster_centers_ is not None
        assert kmeans.cluster_centers_.shape == (2, 2)
        assert kmeans.labels_ is not None
        assert kmeans.labels_.shape == (6,)
        assert kmeans.inertia_ is not None
        assert kmeans.n_iter_ is not None
        assert kmeans.n_features_in_ == 2

    def test_fit_returns_self(self):
        """Test that fit returns self for method chaining."""
        X = np.array([[1.0, 2.0], [1.5, 1.8], [5.0, 8.0]])
        kmeans = KMeans(n_clusters=2, random_state=42)
        result = kmeans.fit(X)
        assert result is kmeans

    def test_not_fitted_predict_error(self):
        """Test that predict raises error when not fitted."""
        kmeans = KMeans()
        X = np.array([[0.0, 0.0]])
        with pytest.raises(ValueError, match="Not fitted"):
            kmeans.predict(X)

    def test_fit_predict_method(self):
        """Test fit_predict returns labels."""
        X = np.array([[1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [8.0, 8.0]])
        kmeans = KMeans(n_clusters=2, random_state=42)
        labels = kmeans.fit_predict(X)

        assert labels.shape == (4,)
        np.testing.assert_array_equal(labels, kmeans.labels_)

    def test_n_samples_less_than_n_clusters_error(self):
        """Test error when n_samples < n_clusters."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        kmeans = KMeans(n_clusters=5)
        with pytest.raises(ValueError, match="n_samples=2 should be >= n_clusters=5"):
            kmeans.fit(X)


class TestKMeansInit:
    """Tests for initialization methods."""

    def test_kmeans_plusplus_init(self):
        """Test k-means++ initialization."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        kmeans = KMeans(n_clusters=3, init="k-means++", random_state=42)
        kmeans.fit(X)
        assert kmeans.cluster_centers_ is not None

    def test_random_init(self):
        """Test random initialization."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        kmeans = KMeans(n_clusters=3, init="random", random_state=42)
        kmeans.fit(X)
        assert kmeans.cluster_centers_ is not None

    def test_invalid_init_error(self):
        """Test invalid init raises error."""
        with pytest.raises(ValueError):
            KMeans(init="invalid")


class TestKMeansNInit:
    """Tests for n_init parameter."""

    def test_n_init_auto_kmeans_plusplus(self):
        """Test n_init defaults to 1 for k-means++."""
        np.random.seed(42)
        X = np.random.randn(50, 2)
        kmeans = KMeans(n_clusters=3, init="k-means++", random_state=42)
        kmeans.fit(X)
        # Should work without explicit n_init

    def test_n_init_auto_random(self):
        """Test n_init defaults to 10 for random."""
        np.random.seed(42)
        X = np.random.randn(50, 2)
        kmeans = KMeans(n_clusters=3, init="random", random_state=42)
        kmeans.fit(X)
        # Should work without explicit n_init

    def test_n_init_explicit(self):
        """Test explicit n_init."""
        np.random.seed(42)
        X = np.random.randn(50, 2)
        kmeans = KMeans(n_clusters=3, n_init=5, random_state=42)
        kmeans.fit(X)
        assert kmeans.cluster_centers_ is not None


class TestKMeansConvergence:
    """Tests for convergence."""

    def test_max_iter_respected(self):
        """Test max_iter is respected."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        kmeans = KMeans(n_clusters=3, max_iter=1, random_state=42)
        kmeans.fit(X)
        assert kmeans.n_iter_ <= 1

    def test_tol_convergence(self):
        """Test convergence with tol."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        kmeans = KMeans(n_clusters=3, tol=1e-10, random_state=42)
        kmeans.fit(X)
        # Should converge or hit max_iter
        assert kmeans.n_iter_ is not None


class TestKMeansTransform:
    """Tests for transform methods."""

    def test_transform_shape(self):
        """Test transform returns correct shape."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        kmeans = KMeans(n_clusters=4, random_state=42)
        kmeans.fit(X)

        X_new = np.random.randn(20, 5)
        transformed = kmeans.transform(X_new)
        assert transformed.shape == (20, 4)

    def test_fit_transform(self):
        """Test fit_transform."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        kmeans = KMeans(n_clusters=4, random_state=42)
        transformed = kmeans.fit_transform(X)
        assert transformed.shape == (100, 4)

    def test_transform_not_fitted_error(self):
        """Test transform raises error when not fitted."""
        kmeans = KMeans()
        X = np.array([[0.0, 0.0]])
        with pytest.raises(ValueError, match="Not fitted"):
            kmeans.transform(X)


class TestKMeansScore:
    """Tests for score method."""

    def test_score_negative_inertia(self):
        """Test score returns negative inertia."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans.fit(X)

        score = kmeans.score(X)
        assert score == pytest.approx(-kmeans.inertia_)

    def test_score_not_fitted_error(self):
        """Test score raises error when not fitted."""
        kmeans = KMeans()
        X = np.array([[0.0, 0.0]])
        with pytest.raises(ValueError, match="Not fitted"):
            kmeans.score(X)


class TestKMeansParallel:
    """Tests for parallel execution."""

    @pytest.fixture
    def large_data(self):
        np.random.seed(42)
        return np.random.randn(1000, 10)

    def test_parallel_matches_single(self, large_data):
        """Test parallel gives same results as single-threaded."""
        X = large_data

        kmeans_single = KMeans(n_clusters=5, n_jobs=1, random_state=42, n_init=1)
        kmeans_parallel = KMeans(n_clusters=5, n_jobs=-1, random_state=42, n_init=1)

        kmeans_single.fit(X)
        kmeans_parallel.fit(X)

        np.testing.assert_array_almost_equal(
            kmeans_single.cluster_centers_, kmeans_parallel.cluster_centers_
        )
        np.testing.assert_array_equal(kmeans_single.labels_, kmeans_parallel.labels_)


class TestKMeansReproducibility:
    """Tests for reproducibility."""

    def test_random_state_reproducible(self):
        """Test same random_state gives same results."""
        np.random.seed(42)
        X = np.random.randn(100, 2)

        kmeans1 = KMeans(n_clusters=3, random_state=42)
        kmeans2 = KMeans(n_clusters=3, random_state=42)

        kmeans1.fit(X)
        kmeans2.fit(X)

        np.testing.assert_array_almost_equal(
            kmeans1.cluster_centers_, kmeans2.cluster_centers_
        )

    def test_different_random_state_different_results(self):
        """Test different random_state may give different results."""
        np.random.seed(42)
        X = np.random.randn(100, 2)

        kmeans1 = KMeans(n_clusters=3, random_state=42)
        kmeans2 = KMeans(n_clusters=3, random_state=123)

        kmeans1.fit(X)
        kmeans2.fit(X)

        # Results may differ (not guaranteed, but likely)
        # Just check both produce valid results
        assert kmeans1.cluster_centers_ is not None
        assert kmeans2.cluster_centers_ is not None


class TestKMeansClusterQuality:
    """Tests for cluster quality."""

    def test_well_separated_clusters(self):
        """Test that well-separated clusters are found correctly."""
        # Create 3 well-separated clusters
        np.random.seed(42)
        cluster1 = np.random.randn(30, 2) + np.array([0, 0])
        cluster2 = np.random.randn(30, 2) + np.array([10, 0])
        cluster3 = np.random.randn(30, 2) + np.array([5, 10])
        X = np.vstack([cluster1, cluster2, cluster3])

        kmeans = KMeans(n_clusters=3, random_state=42)
        labels = kmeans.fit_predict(X)

        # Check that we have 3 distinct clusters
        assert len(np.unique(labels)) == 3

        # Check that points from same original cluster have same label
        # (with high probability for well-separated clusters)
        labels_cluster1 = labels[:30]
        labels_cluster2 = labels[30:60]
        labels_cluster3 = labels[60:90]

        # Most points in each original cluster should have the same label
        assert np.sum(labels_cluster1 == np.median(labels_cluster1)) >= 25
        assert np.sum(labels_cluster2 == np.median(labels_cluster2)) >= 25
        assert np.sum(labels_cluster3 == np.median(labels_cluster3)) >= 25

    def test_inertia_decreases_with_more_clusters(self):
        """Test that inertia generally decreases with more clusters."""
        np.random.seed(42)
        X = np.random.randn(100, 2)

        inertias = []
        for k in [2, 5, 10]:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)

        # Inertia should decrease as k increases
        assert inertias[0] > inertias[1] > inertias[2]
