"""
rklearn - Fast scikit-learn compatible preprocessing in Rust
"""
from ._rklearn import (
    KMeans,
    KNeighborsRegressor,
    MinMaxScaler,
    RidgeRegressor,
    StandardScaler,
    __version__,
)

__all__ = [
    "KMeans",
    "KNeighborsRegressor",
    "MinMaxScaler",
    "RidgeRegressor",
    "StandardScaler",
    "__version__",
]
