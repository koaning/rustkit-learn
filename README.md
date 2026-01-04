# rklearn

Fast scikit-learn compatible components in Rust.

## Installation

```bash
uv pip install git+https://github.com/koaning/rustkit-learn.git
```

## Components

### Linear Models

- **RidgeRegressor** - Linear least squares with L2 regularization

### Preprocessing

- **StandardScaler** - Standardize features by removing mean and scaling to unit variance
- **MinMaxScaler** - Scale features to a given range (default [0, 1])

### Neighbors

- **KNeighborsRegressor** - K-nearest neighbors regression

## License

MIT OR Apache-2.0
