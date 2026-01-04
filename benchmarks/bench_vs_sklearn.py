"""Benchmarks comparing rklearn against scikit-learn."""

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler
from sklearn.preprocessing import MinMaxScaler as SklearnMinMaxScaler

from rklearn.preprocessing import StandardScaler, MinMaxScaler


# Data sizes to benchmark
SIZES = [
    (100, 10),       # Small
    (1_000, 100),    # Medium
    (10_000, 100),   # Large
    (100_000, 100),  # Very large
    (1_000_000, 50), # Huge
]


@pytest.fixture(params=SIZES, ids=[f"{r}x{c}" for r, c in SIZES])
def data(request):
    rows, cols = request.param
    np.random.seed(42)
    return np.random.randn(rows, cols)


class TestStandardScalerBenchmark:
    """Benchmarks for StandardScaler."""

    def test_rust_fit(self, benchmark, data):
        """Benchmark rustscaler StandardScaler.fit()."""
        scaler = StandardScaler()
        benchmark(scaler.fit, data)

    def test_sklearn_fit(self, benchmark, data):
        """Benchmark sklearn StandardScaler.fit()."""
        scaler = SklearnStandardScaler()
        benchmark(scaler.fit, data)

    def test_rust_transform(self, benchmark, data):
        """Benchmark rustscaler StandardScaler.transform()."""
        scaler = StandardScaler()
        scaler.fit(data)
        benchmark(scaler.transform, data)

    def test_sklearn_transform(self, benchmark, data):
        """Benchmark sklearn StandardScaler.transform()."""
        scaler = SklearnStandardScaler()
        scaler.fit(data)
        benchmark(scaler.transform, data)

    def test_rust_fit_transform(self, benchmark, data):
        """Benchmark rustscaler StandardScaler.fit_transform()."""
        scaler = StandardScaler()
        benchmark(scaler.fit_transform, data)

    def test_sklearn_fit_transform(self, benchmark, data):
        """Benchmark sklearn StandardScaler.fit_transform()."""
        scaler = SklearnStandardScaler()
        benchmark(scaler.fit_transform, data)


class TestMinMaxScalerBenchmark:
    """Benchmarks for MinMaxScaler."""

    def test_rust_fit(self, benchmark, data):
        """Benchmark rustscaler MinMaxScaler.fit()."""
        scaler = MinMaxScaler()
        benchmark(scaler.fit, data)

    def test_sklearn_fit(self, benchmark, data):
        """Benchmark sklearn MinMaxScaler.fit()."""
        scaler = SklearnMinMaxScaler()
        benchmark(scaler.fit, data)

    def test_rust_transform(self, benchmark, data):
        """Benchmark rustscaler MinMaxScaler.transform()."""
        scaler = MinMaxScaler()
        scaler.fit(data)
        benchmark(scaler.transform, data)

    def test_sklearn_transform(self, benchmark, data):
        """Benchmark sklearn MinMaxScaler.transform()."""
        scaler = SklearnMinMaxScaler()
        scaler.fit(data)
        benchmark(scaler.transform, data)

    def test_rust_fit_transform(self, benchmark, data):
        """Benchmark rustscaler MinMaxScaler.fit_transform()."""
        scaler = MinMaxScaler()
        benchmark(scaler.fit_transform, data)

    def test_sklearn_fit_transform(self, benchmark, data):
        """Benchmark sklearn MinMaxScaler.fit_transform()."""
        scaler = SklearnMinMaxScaler()
        benchmark(scaler.fit_transform, data)
