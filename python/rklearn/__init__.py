"""
rklearn - Fast scikit-learn compatible preprocessing in Rust
"""
from ._rklearn import (
    KNeighborsRegressor,
    MinMaxScaler,
    RidgeRegressor,
    StandardScaler,
    __version__,
)

__all__ = [
    "KNeighborsRegressor",
    "MinMaxScaler",
    "RidgeRegressor",
    "StandardScaler",
    "__version__",
]
