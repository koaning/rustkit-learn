from rklearn._rklearn import HuberRegressor, LinearRegression, RidgeRegressor

# Aliases to match sklearn naming convention
Huber = HuberRegressor
Ridge = RidgeRegressor

__all__ = ["Huber", "HuberRegressor", "LinearRegression", "Ridge", "RidgeRegressor"]
