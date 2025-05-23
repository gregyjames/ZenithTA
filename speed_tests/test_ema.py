import pytest
import numpy as np
import pandas as pd
from ZenithTA import ema


def test_ema_basic():
    """Test basic EMA calculation with simple input."""
    # Simple increasing sequence
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    period = 3
    smoothing = 2.0

    result = ema(data, period, smoothing)
    expected = pd.Series(data).ewm(span=period, adjust=False).mean().values

    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_ema_constant():
    """Test EMA calculation with constant values."""
    data = np.array([10.0] * 10)
    period = 4
    smoothing = 2.0

    result = ema(data, period, smoothing)
    expected = pd.Series(data).ewm(span=period, adjust=False).mean().values

    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_ema_random():
    """Test EMA calculation with random data."""
    np.random.seed(42)
    data = np.random.normal(100, 10, 1000)
    period = 20
    smoothing = 2.0

    result = ema(data, period, smoothing)
    expected = pd.Series(data).ewm(span=period, adjust=False).mean().values

    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_ema_edge_cases():
    """Test EMA calculation with edge cases."""
    # Test with single value
    data = np.array([1.0])
    period = 1
    smoothing = 2.0

    result = ema(data, period, smoothing)
    assert len(result) == 1
    assert result[0] == data[0]

    # Test with period equal to data length
    data = np.array([1.0, 2.0, 3.0])
    period = 3
    smoothing = 2.0

    result = ema(data, period, smoothing)
    expected = pd.Series(data).ewm(span=period, adjust=False).mean().values
    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_ema_performance():
    """Test EMA performance with larger dataset."""
    # Generate larger dataset
    np.random.seed(42)
    data = np.random.normal(100, 10, 10000)
    period = 50
    smoothing = 2.0

    # Time our implementation
    import time

    start = time.time()
    result = ema(data, period, smoothing)
    our_time = time.time() - start

    # Time pandas implementation
    start = time.time()
    expected = pd.Series(data).ewm(span=period, adjust=False).mean().values
    pandas_time = time.time() - start

    # Verify results are close
    np.testing.assert_allclose(result, expected, rtol=1e-5)

    # Print performance comparison
    print(f"\nPerformance comparison (10000 points):")
    print(f"Our implementation: {our_time*1000:.2f} ms")
    print(f"Pandas implementation: {pandas_time*1000:.2f} ms")
    print(f"Speedup: {pandas_time/our_time:.2f}x")


def test_ema_different_periods():
    """Test EMA calculation with different periods."""
    np.random.seed(42)
    data = np.random.normal(100, 10, 1000)
    periods = [5, 10, 20, 50, 100]
    smoothing = 2.0

    for period in periods:
        result = ema(data, period, smoothing)
        expected = pd.Series(data).ewm(span=period, adjust=False).mean().values
        np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_ema_different_smoothing():
    """Test EMA calculation with different smoothing factors."""
    np.random.seed(42)
    data = np.random.normal(100, 10, 1000)
    period = 20
    smoothing_factors = [1.0, 2.0, 3.0]

    for smoothing in smoothing_factors:
        result = ema(data, period, smoothing)
        # Note: pandas uses alpha = 2/(span+1), so we need to adjust our comparison
        alpha = smoothing / (period + 1)
        expected = pd.Series(data).ewm(alpha=alpha, adjust=False).mean().values
        np.testing.assert_allclose(result, expected, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
