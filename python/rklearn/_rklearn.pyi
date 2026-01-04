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


class RidgeRegressor:
    """Ridge regression.

    Linear least squares with L2 regularization.
    """

    coef_: Optional[NDArray[np.float64]]
    """Estimated coefficients."""

    intercept_: Optional[float]
    """Independent term in the linear model."""

    n_features_in_: Optional[int]
    """Number of features seen during fit."""

    n_samples_fit_: Optional[int]
    """Number of samples in the fitted data."""

    def __init__(
        self,
        *,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        solver: str = "auto",
    ) -> None:
        """Initialize RidgeRegressor.

        Parameters
        ----------
        alpha : float, default=1.0
            Regularization strength; must be non-negative.
        fit_intercept : bool, default=True
            Whether to calculate the intercept for this model.
        solver : str, default='auto'
            Solver to use. Supported: 'auto', 'cholesky'.
        """
        ...

    def fit(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> "RidgeRegressor":
        """Fit ridge regression model."""
        ...

    def predict(
        self,
        X: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Predict using the ridge regression model."""
        ...

    def score(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> float:
        """Return coefficient of determination R^2 of prediction."""
        ...
