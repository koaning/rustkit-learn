"""
rklearn - Fast scikit-learn compatible preprocessing in Rust
"""
from ._rklearn import KNeighborsRegressor, MinMaxScaler, StandardScaler, __version__

__all__ = [
    "KNeighborsRegressor",
    "MinMaxScaler",
    "StandardScaler",
    "__version__",
]
