from rklearn._rklearn import HuberRegressor, RidgeRegressor

# Aliases to match sklearn naming convention
Huber = HuberRegressor
Ridge = RidgeRegressor

__all__ = ["Huber", "HuberRegressor", "Ridge", "RidgeRegressor"]
