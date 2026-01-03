# rklearn

Fast scikit-learn compatible preprocessing in Rust.

A Python package providing scikit-learn compatible preprocessing components (StandardScaler, MinMaxScaler) written in Rust using PyO3. The goal is to provide faster performance while maintaining API compatibility with scikit-learn.

## Installation

```bash
pip install rustkit-learn
```

Or for development:

```bash
pip install -e ".[dev]"
```

## Usage

```python
import numpy as np
from rklearn.preprocessing import StandardScaler, MinMaxScaler

# StandardScaler - standardize features by removing mean and scaling to unit variance
X = np.random.randn(1000, 10)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"Mean: {X_scaled.mean(axis=0)}")  # ~0
print(f"Std: {X_scaled.std(axis=0)}")    # ~1

# MinMaxScaler - scale features to a given range (default [0, 1])
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
print(f"Min: {X_scaled.min(axis=0)}")    # 0
print(f"Max: {X_scaled.max(axis=0)}")    # 1

# Incremental learning with partial_fit
scaler = StandardScaler()
for batch in np.array_split(X, 10):
    scaler.partial_fit(batch)
X_scaled = scaler.transform(X)
```

## API

### StandardScaler

```python
StandardScaler(with_mean=True, with_std=True)
```

**Methods:**
- `fit(X)` - Compute mean and std from data
- `partial_fit(X)` - Online update of mean and std (for streaming data)
- `transform(X)` - Apply standardization
- `fit_transform(X)` - Fit and transform in one step
- `inverse_transform(X)` - Reverse the transformation

**Attributes:**
- `mean_` - Per-feature mean
- `scale_` - Per-feature standard deviation
- `var_` - Per-feature variance
- `n_features_in_` - Number of features
- `n_samples_seen_` - Number of samples processed

### MinMaxScaler

```python
MinMaxScaler(feature_range=(0.0, 1.0), clip=False)
```

**Methods:**
- `fit(X)` - Compute min and max from data
- `partial_fit(X)` - Online update of min and max (for streaming data)
- `transform(X)` - Apply min-max scaling
- `fit_transform(X)` - Fit and transform in one step
- `inverse_transform(X)` - Reverse the transformation

**Attributes:**
- `min_` - Per-feature adjustment for minimum
- `scale_` - Per-feature relative scaling
- `data_min_` - Per-feature minimum seen in data
- `data_max_` - Per-feature maximum seen in data
- `data_range_` - Per-feature range (max - min)
- `n_features_in_` - Number of features
- `n_samples_seen_` - Number of samples processed

## Performance

Benchmarks comparing rklearn vs scikit-learn on various data sizes:

| Operation | Data Size | rklearn | sklearn | Speedup |
|-----------|-----------|---------|---------|---------|
| fit_transform | 100x10 | 82 us | 55 us | 0.7x |
| fit_transform | 1000x100 | 212 us | 194 us | 0.9x |
| fit_transform | 10000x100 | 831 us | 1398 us | 1.7x |
| fit_transform | 100000x100 | 6.6 ms | 13.3 ms | 2.0x |
| fit_transform | 1000000x50 | 34 ms | 76 ms | 2.2x |

rklearn becomes faster than sklearn for larger datasets (>10k samples) due to:
- Parallel computation using rayon
- GIL release during computation
- Efficient memory access patterns

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run benchmarks
pytest benchmarks/ --benchmark-group-by=func
```

## License

MIT OR Apache-2.0
