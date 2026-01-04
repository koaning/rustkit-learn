from typing import Optional, Tuple
import numpy as np
from numpy.typing import NDArray

__version__: str

class StandardScaler:
    """Standardize features by removing the mean and scaling to unit variance.

    The standard score of a sample x is calculated as:
        z = (x - mean) / std

    This scaler can be used incrementally via `partial_fit` for streaming data.
    """

    mean_: Optional[NDArray[np.float64]]
    """Per-feature mean, computed during fit."""

    scale_: Optional[NDArray[np.float64]]
    """Per-feature scale (standard deviation), computed during fit."""

    var_: Optional[NDArray[np.float64]]
    """Per-feature variance, computed during fit."""

    n_features_in_: Optional[int]
    """Number of features seen during fit."""

    n_samples_seen_: int
    """Number of samples seen during fit."""

    def __init__(
        self,
        *,
        with_mean: bool = True,
        with_std: bool = True,
        n_jobs: int = 1,
    ) -> None:
        """Initialize StandardScaler.

        Parameters
        ----------
        with_mean : bool, default=True
            If True, center the data before scaling.
        with_std : bool, default=True
            If True, scale the data to unit variance.
        n_jobs : int, default=1
            Number of parallel jobs. Use 1 for single-threaded (default).
            Use -1 or any value != 1 for parallel computation using Rayon.
        """
        ...

    def fit(
        self,
        X: NDArray[np.float64],
    ) -> "StandardScaler":
        """Compute the mean and std to be used for later scaling.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data used to compute the mean and standard deviation.

        Returns
        -------
        self : StandardScaler
        """
        ...

    def partial_fit(
        self,
        X: NDArray[np.float64],
    ) -> "StandardScaler":
        """Online computation of mean and std with a new batch of data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data used to compute the mean and standard deviation.

        Returns
        -------
        self : StandardScaler
        """
        ...

    def transform(
        self,
        X: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Perform standardization by centering and scaling.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to transform.

        Returns
        -------
        X_tr : ndarray of shape (n_samples, n_features)
            Transformed array.
        """
        ...

    def fit_transform(
        self,
        X: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Fit and transform in one step.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to fit and transform.

        Returns
        -------
        X_tr : ndarray of shape (n_samples, n_features)
            Transformed array.
        """
        ...

    def inverse_transform(
        self,
        X: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Scale back the data to the original representation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The transformed data.

        Returns
        -------
        X_tr : ndarray of shape (n_samples, n_features)
            Original data.
        """
        ...


class MinMaxScaler:
    """Transform features by scaling each feature to a given range.

    This estimator scales and translates each feature individually such
    that it is in the given range on the training set, e.g. between zero and one.

    The transformation is given by:
        X_std = (X - X_min) / (X_max - X_min)
        X_scaled = X_std * (max - min) + min

    This scaler can be used incrementally via `partial_fit` for streaming data.
    """

    min_: Optional[NDArray[np.float64]]
    """Per feature adjustment for minimum."""

    scale_: Optional[NDArray[np.float64]]
    """Per feature relative scaling of the data."""

    data_min_: Optional[NDArray[np.float64]]
    """Per feature minimum seen in the data."""

    data_max_: Optional[NDArray[np.float64]]
    """Per feature maximum seen in the data."""

    data_range_: Optional[NDArray[np.float64]]
    """Per feature range (data_max - data_min) seen in the data."""

    n_features_in_: Optional[int]
    """Number of features seen during fit."""

    n_samples_seen_: int
    """Number of samples seen during fit."""

    def __init__(
        self,
        feature_range: Tuple[float, float] = (0.0, 1.0),
        *,
        clip: bool = False,
        n_jobs: int = 1,
    ) -> None:
        """Initialize MinMaxScaler.

        Parameters
        ----------
        feature_range : tuple (min, max), default=(0, 1)
            Desired range of transformed data.
        clip : bool, default=False
            Set to True to clip transformed values to the provided feature range.
        n_jobs : int, default=1
            Number of parallel jobs. Use 1 for single-threaded (default).
            Use -1 or any value != 1 for parallel computation using Rayon.
        """
        ...

    def fit(
        self,
        X: NDArray[np.float64],
    ) -> "MinMaxScaler":
        """Compute the minimum and maximum to be used for later scaling.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data used to compute the per-feature minimum and maximum.

        Returns
        -------
        self : MinMaxScaler
        """
        ...

    def partial_fit(
        self,
        X: NDArray[np.float64],
    ) -> "MinMaxScaler":
        """Online computation of min and max with a new batch of data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data used to compute the per-feature minimum and maximum.

        Returns
        -------
        self : MinMaxScaler
        """
        ...

    def transform(
        self,
        X: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Scale features according to feature_range.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to transform.

        Returns
        -------
        X_tr : ndarray of shape (n_samples, n_features)
            Transformed array.
        """
        ...

    def fit_transform(
        self,
        X: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Fit and transform in one step.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to fit and transform.

        Returns
        -------
        X_tr : ndarray of shape (n_samples, n_features)
            Transformed array.
        """
        ...

    def inverse_transform(
        self,
        X: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Undo the scaling of X according to feature_range.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The transformed data.

        Returns
        -------
        X_tr : ndarray of shape (n_samples, n_features)
            Original data.
        """
        ...


class KMeans:
    """K-Means clustering.

    Parameters
    ----------
    n_clusters : int, default=8
        The number of clusters to form.

    init : {'k-means++', 'random'}, default='k-means++'
        Method for initialization.

    n_init : int or None, default=None
        Number of times to run with different seeds. Default is 1 for
        'k-means++' and 10 for 'random'.

    max_iter : int, default=300
        Maximum number of iterations.

    tol : float, default=1e-4
        Relative tolerance for convergence.

    random_state : int or None, default=None
        Seed for random number generation.

    n_jobs : int, default=1
        Number of parallel jobs.
    """

    cluster_centers_: Optional[NDArray[np.float64]]
    """Coordinates of cluster centers. Shape (n_clusters, n_features)."""

    labels_: Optional[NDArray[np.int64]]
    """Labels of each point. Shape (n_samples,)."""

    inertia_: Optional[float]
    """Sum of squared distances to nearest cluster center."""

    n_iter_: Optional[int]
    """Number of iterations run."""

    n_features_in_: Optional[int]
    """Number of features seen during fit."""

    def __init__(
        self,
        *,
        n_clusters: int = 8,
        init: str = "k-means++",
        n_init: Optional[int] = None,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
        n_jobs: int = 1,
    ) -> None:
        """Initialize KMeans.

        Parameters
        ----------
        n_clusters : int, default=8
            The number of clusters to form.
        init : str, default='k-means++'
            Method for initialization ('k-means++' or 'random').
        n_init : int or None, default=None
            Number of initializations. Default is 1 for 'k-means++', 10 for 'random'.
        max_iter : int, default=300
            Maximum number of iterations.
        tol : float, default=1e-4
            Relative tolerance for convergence.
        random_state : int or None, default=None
            Seed for random number generation.
        n_jobs : int, default=1
            Number of parallel jobs.
        """
        ...

    def fit(
        self,
        X: NDArray[np.float64],
    ) -> "KMeans":
        """Compute k-means clustering.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training instances to cluster.

        Returns
        -------
        self : KMeans
        """
        ...

    def predict(
        self,
        X: NDArray[np.float64],
    ) -> NDArray[np.int64]:
        """Predict the closest cluster each sample belongs to.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to predict.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        ...

    def fit_predict(
        self,
        X: NDArray[np.float64],
    ) -> NDArray[np.int64]:
        """Compute cluster centers and predict cluster index for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to transform.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        ...

    def transform(
        self,
        X: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Transform X to a cluster-distance space.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to transform.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_clusters)
            X transformed to cluster-distance space.
        """
        ...

    def fit_transform(
        self,
        X: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Fit and transform in one step.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to fit and transform.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_clusters)
            X transformed to cluster-distance space.
        """
        ...

    def score(
        self,
        X: NDArray[np.float64],
    ) -> float:
        """Opposite of inertia (for sklearn compatibility).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data.

        Returns
        -------
        score : float
            Negative inertia.
        """
        ...
