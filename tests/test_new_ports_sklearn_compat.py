"""sklearn-parity tests for MaxAbsScaler, Normalizer and LinearRegression."""

import numpy as np
import pytest
from sklearn import linear_model as sk_lm
from sklearn import preprocessing as sk_pp

import rklearn
from rklearn.linear_model import LinearRegression
from rklearn.preprocessing import MaxAbsScaler, Normalizer


@pytest.fixture
def X():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 6)) * [1, 5, 0.1, 10, 1, 1]
    X[:, 5] = 0.0  # constant zero column
    X[3] = 0.0  # all-zero row
    return X


@pytest.fixture
def reg_data():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 5))
    y = X @ np.array([1.5, -2.0, 0.0, 3.0, 0.5]) + 4.0 + rng.normal(scale=0.1, size=300)
    return X, y


class TestMaxAbsScaler:
    def test_matches_sklearn(self, X):
        ours, theirs = MaxAbsScaler().fit(X), sk_pp.MaxAbsScaler().fit(X)
        np.testing.assert_allclose(ours.max_abs_, theirs.max_abs_)
        np.testing.assert_allclose(ours.scale_, theirs.scale_)
        np.testing.assert_allclose(ours.transform(X), theirs.transform(X))
        np.testing.assert_allclose(MaxAbsScaler().fit_transform(X), theirs.transform(X))
        assert ours.n_features_in_ == 6 and ours.n_samples_seen_ == 200

    def test_inverse_transform(self, X):
        s = MaxAbsScaler().fit(X)
        np.testing.assert_allclose(s.inverse_transform(s.transform(X)), X)

    def test_partial_fit_matches_fit(self, X):
        s = MaxAbsScaler()
        for chunk in np.array_split(X, 4):
            s.partial_fit(chunk)
        np.testing.assert_allclose(s.max_abs_, MaxAbsScaler().fit(X).max_abs_)
        assert s.n_samples_seen_ == 200

    def test_feature_mismatch(self, X):
        with pytest.raises(ValueError):
            MaxAbsScaler().fit(X).transform(X[:, :3])

    def test_not_fitted(self, X):
        with pytest.raises(ValueError):
            MaxAbsScaler().transform(X)


class TestNormalizer:
    @pytest.mark.parametrize("norm", ["l1", "l2", "max"])
    def test_matches_sklearn(self, X, norm):
        np.testing.assert_allclose(
            Normalizer(norm=norm).fit_transform(X), sk_pp.Normalizer(norm=norm).fit_transform(X)
        )

    def test_transform_without_fit(self, X):
        np.testing.assert_allclose(Normalizer().transform(X), sk_pp.normalize(X))

    def test_invalid_norm(self):
        with pytest.raises(ValueError):
            Normalizer(norm="l3")


class TestLinearRegression:
    @pytest.mark.parametrize("fit_intercept", [True, False])
    def test_matches_sklearn(self, reg_data, fit_intercept):
        X, y = reg_data
        ours = LinearRegression(fit_intercept=fit_intercept).fit(X, y)
        theirs = sk_lm.LinearRegression(fit_intercept=fit_intercept).fit(X, y)
        np.testing.assert_allclose(ours.coef_, theirs.coef_, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(ours.intercept_, theirs.intercept_, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(ours.predict(X), theirs.predict(X), rtol=1e-8)
        np.testing.assert_allclose(ours.score(X, y), theirs.score(X, y))
        assert ours.rank_ == theirs.rank_ == 5

    def test_collinear_features_min_norm(self, reg_data):
        X, y = reg_data
        X = np.hstack([X, X[:, :1]])  # duplicate column -> rank deficient
        ours = LinearRegression().fit(X, y)
        theirs = sk_lm.LinearRegression().fit(X, y)
        np.testing.assert_allclose(ours.coef_, theirs.coef_, rtol=1e-6, atol=1e-8)
        assert ours.rank_ == 5

    def test_feature_mismatch(self, reg_data):
        X, y = reg_data
        with pytest.raises(ValueError):
            LinearRegression().fit(X, y).predict(X[:, :2])


def test_neighbors_submodule():
    import rklearn.neighbors as nb

    assert nb.KNeighborsClassifier is rklearn.KNeighborsClassifier
    assert nb.KNeighborsRegressor is rklearn.KNeighborsRegressor
