"""Tests for EMeans and EGaussianMixture."""

import numpy as np
import pytest
from sklearn.cluster import KMeans as SkKMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

from rklearn.cluster import EMeans
from rklearn.mixture import EGaussianMixture


@pytest.fixture(scope="module")
def blobs():
    X, y = make_blobs(n_samples=6000, centers=6, n_features=5, random_state=0)
    return X, y


class TestEMeans:
    def test_fit_attributes(self, blobs):
        X, _ = blobs
        km = EMeans(n_clusters=6, random_state=0).fit(X)
        assert km.cluster_centers_.shape == (6, 5)
        assert km.labels_.shape == (6000,)
        assert km.inertia_ > 0
        assert km.n_features_in_ == 5
        assert km.batch_size_ >= 1
        assert 0 < km.alpha_ <= 1
        assert km.n_iter_ > 0
        assert km.n_splits_ == 0

    def test_recovers_blobs(self, blobs):
        X, y = blobs
        labels = EMeans(n_clusters=6, random_state=0).fit_predict(X)
        assert adjusted_rand_score(y, labels) > 0.95

    def test_inertia_close_to_lloyd(self, blobs):
        X, _ = blobs
        ref = SkKMeans(n_clusters=6, n_init=1, random_state=0).fit(X).inertia_
        got = EMeans(n_clusters=6, random_state=0).fit(X).inertia_
        assert got <= ref * 1.05

    def test_explicit_params_are_used(self, blobs):
        X, _ = blobs
        km = EMeans(n_clusters=6, batch_size=128, alpha=0.3, random_state=0).fit(X)
        assert km.batch_size_ == 128
        assert km.alpha_ == pytest.approx(0.3)
        # 3 epochs of ceil(6000 / 128) batches
        assert km.n_iter_ == 3 * 47

    def test_fractional_epochs(self, blobs):
        X, _ = blobs
        km = EMeans(n_clusters=6, n_epochs=1.5, batch_size=1000, random_state=0).fit(X)
        assert km.n_iter_ == 3 + 6

    def test_revival_splits_dead_clusters(self):
        # Random init with many clusters on few blobs leaves dead centroids.
        X, _ = make_blobs(n_samples=4000, centers=3, n_features=4, random_state=1)
        X = np.vstack([X, np.full((5, 4), 100.0)])  # far outliers attract a seed
        off = EMeans(n_clusters=20, init="random", batch_size=64, revival=False, random_state=3).fit(X)
        on = EMeans(n_clusters=20, init="random", batch_size=64, revival=True, random_state=3).fit(X)
        assert off.n_splits_ == 0
        assert on.n_splits_ > 0

    def test_deterministic(self, blobs):
        X, _ = blobs
        a = EMeans(n_clusters=6, random_state=7).fit(X).cluster_centers_
        b = EMeans(n_clusters=6, random_state=7).fit(X).cluster_centers_
        np.testing.assert_array_equal(a, b)

    def test_parallel_matches_single(self, blobs):
        X, _ = blobs
        a = EMeans(n_clusters=6, random_state=7, n_jobs=1).fit(X)
        b = EMeans(n_clusters=6, random_state=7, n_jobs=-1).fit(X)
        np.testing.assert_allclose(a.cluster_centers_, b.cluster_centers_)

    def test_predict_transform_score(self, blobs):
        X, _ = blobs
        km = EMeans(n_clusters=6, random_state=0).fit(X)
        np.testing.assert_array_equal(km.predict(X), km.labels_)
        dist = km.transform(X)
        assert dist.shape == (6000, 6)
        np.testing.assert_array_equal(dist.argmin(axis=1), km.labels_)
        assert km.score(X) == pytest.approx(-km.inertia_)

    def test_get_params(self):
        p = EMeans(n_clusters=3, batch_size=64, alpha=0.5, revival=True).get_params()
        assert p["n_clusters"] == 3
        assert p["batch_size"] == 64
        assert p["alpha"] == 0.5
        assert p["revival"] is True

    @pytest.mark.parametrize(
        "kwargs",
        [{"alpha": 0.0}, {"alpha": 1.5}, {"batch_size": 0}, {"n_epochs": 0.0}, {"init": "nope"}],
    )
    def test_invalid_params(self, kwargs):
        with pytest.raises(ValueError):
            EMeans(**kwargs)

    def test_not_fitted(self):
        with pytest.raises(ValueError, match="Not fitted"):
            EMeans().predict(np.zeros((1, 2)))

    def test_too_few_samples(self):
        with pytest.raises(ValueError, match="n_samples=2 should be >= n_clusters=5"):
            EMeans(n_clusters=5).fit(np.zeros((2, 2)))


class TestEGaussianMixture:
    def test_fit_attributes(self, blobs):
        X, _ = blobs
        gm = EGaussianMixture(n_components=6, random_state=0).fit(X)
        assert gm.means_.shape == (6, 5)
        assert gm.covariances_.shape == (6, 5)
        assert (gm.covariances_ > 0).all()
        assert gm.weights_.shape == (6,)
        assert gm.weights_.sum() == pytest.approx(1.0)
        assert gm.covariance_type == "diag"
        assert np.isfinite(gm.lower_bound_)

    def test_recovers_blobs(self, blobs):
        X, y = blobs
        labels = EGaussianMixture(n_components=6, random_state=0).fit_predict(X)
        assert adjusted_rand_score(y, labels) > 0.95

    def test_loglik_close_to_sklearn(self, blobs):
        X, _ = blobs
        ref = GaussianMixture(n_components=6, covariance_type="diag", random_state=0).fit(X).score(X)
        got = EGaussianMixture(n_components=6, random_state=0).fit(X).score(X)
        assert got >= ref - 0.05

    def test_predict_proba_rows_sum_to_one(self, blobs):
        X, _ = blobs
        gm = EGaussianMixture(n_components=6, random_state=0).fit(X)
        p = gm.predict_proba(X[:100])
        assert p.shape == (100, 6)
        np.testing.assert_allclose(p.sum(axis=1), 1.0)
        np.testing.assert_array_equal(p.argmax(axis=1), gm.predict(X[:100]))

    def test_score_samples_matches_sklearn_formula(self, blobs):
        X, _ = blobs
        gm = EGaussianMixture(n_components=3, random_state=0).fit(X)
        sk = GaussianMixture(n_components=3, covariance_type="diag")
        sk.weights_ = gm.weights_
        sk.means_ = gm.means_
        sk.covariances_ = gm.covariances_
        sk.precisions_cholesky_ = 1.0 / np.sqrt(gm.covariances_)
        np.testing.assert_allclose(gm.score_samples(X[:50]), sk.score_samples(X[:50]), rtol=1e-8)

    def test_bic_aic(self, blobs):
        X, _ = blobs
        gm = EGaussianMixture(n_components=6, random_state=0).fit(X)
        assert gm.aic(X) < gm.bic(X)

    def test_explicit_params_and_revival(self, blobs):
        X, _ = blobs
        gm = EGaussianMixture(
            n_components=6, batch_size=256, alpha=0.2, revival=True, random_state=0
        ).fit(X)
        assert gm.batch_size_ == 256
        assert gm.alpha_ == pytest.approx(0.2)
        assert gm.n_splits_ >= 0

    def test_parallel_matches_single(self, blobs):
        X, _ = blobs
        a = EGaussianMixture(n_components=6, random_state=1, n_jobs=1).fit(X)
        b = EGaussianMixture(n_components=6, random_state=1, n_jobs=-1).fit(X)
        np.testing.assert_allclose(a.means_, b.means_)

    def test_feature_mismatch(self, blobs):
        X, _ = blobs
        gm = EGaussianMixture(n_components=2, random_state=0).fit(X)
        with pytest.raises(ValueError, match="features"):
            gm.predict(np.zeros((3, 2)))

    def test_get_params(self):
        p = EGaussianMixture(n_components=4, alpha=0.1, revival=True).get_params()
        assert p["n_components"] == 4
        assert p["alpha"] == 0.1
        assert p["revival"] is True
