"""Benchmarks comparing rklearn against scikit-learn.

All benchmarks here are single-threaded: rklearn runs with n_jobs=1 and every
test runs inside ``threadpool_limits(1)`` so that the BLAS/OpenMP thread pools
used by numpy/scipy (and therefore sklearn) are capped at one thread too.
"""

import numpy as np
import pytest
from threadpoolctl import threadpool_limits
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


@pytest.fixture(autouse=True)
def single_threaded():
    """Limit BLAS/OpenMP thread pools to one thread for a fair comparison."""
    with threadpool_limits(limits=1):
        yield


@pytest.fixture(params=SIZES, ids=[f"{r}x{c}" for r, c in SIZES])
def data(request):
    rows, cols = request.param
    np.random.seed(42)
    return np.random.randn(rows, cols)


class TestStandardScalerBenchmark:
    """Benchmarks for StandardScaler."""

    def test_rust_fit(self, benchmark, data):
        """Benchmark rustscaler StandardScaler.fit()."""
        scaler = StandardScaler(n_jobs=1)
        benchmark(scaler.fit, data)

    def test_sklearn_fit(self, benchmark, data):
        """Benchmark sklearn StandardScaler.fit()."""
        scaler = SklearnStandardScaler()
        benchmark(scaler.fit, data)

    def test_rust_transform(self, benchmark, data):
        """Benchmark rustscaler StandardScaler.transform()."""
        scaler = StandardScaler(n_jobs=1)
        scaler.fit(data)
        benchmark(scaler.transform, data)

    def test_sklearn_transform(self, benchmark, data):
        """Benchmark sklearn StandardScaler.transform()."""
        scaler = SklearnStandardScaler()
        scaler.fit(data)
        benchmark(scaler.transform, data)

    def test_rust_fit_transform(self, benchmark, data):
        """Benchmark rustscaler StandardScaler.fit_transform()."""
        scaler = StandardScaler(n_jobs=1)
        benchmark(scaler.fit_transform, data)

    def test_sklearn_fit_transform(self, benchmark, data):
        """Benchmark sklearn StandardScaler.fit_transform()."""
        scaler = SklearnStandardScaler()
        benchmark(scaler.fit_transform, data)


class TestMinMaxScalerBenchmark:
    """Benchmarks for MinMaxScaler."""

    def test_rust_fit(self, benchmark, data):
        """Benchmark rustscaler MinMaxScaler.fit()."""
        scaler = MinMaxScaler(n_jobs=1)
        benchmark(scaler.fit, data)

    def test_sklearn_fit(self, benchmark, data):
        """Benchmark sklearn MinMaxScaler.fit()."""
        scaler = SklearnMinMaxScaler()
        benchmark(scaler.fit, data)

    def test_rust_transform(self, benchmark, data):
        """Benchmark rustscaler MinMaxScaler.transform()."""
        scaler = MinMaxScaler(n_jobs=1)
        scaler.fit(data)
        benchmark(scaler.transform, data)

    def test_sklearn_transform(self, benchmark, data):
        """Benchmark sklearn MinMaxScaler.transform()."""
        scaler = SklearnMinMaxScaler()
        scaler.fit(data)
        benchmark(scaler.transform, data)

    def test_rust_fit_transform(self, benchmark, data):
        """Benchmark rustscaler MinMaxScaler.fit_transform()."""
        scaler = MinMaxScaler(n_jobs=1)
        benchmark(scaler.fit_transform, data)

    def test_sklearn_fit_transform(self, benchmark, data):
        """Benchmark sklearn MinMaxScaler.fit_transform()."""
        scaler = SklearnMinMaxScaler()
        benchmark(scaler.fit_transform, data)
