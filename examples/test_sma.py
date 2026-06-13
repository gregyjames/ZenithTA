import pytest
import numpy as np
import pandas as pd
from ZenithTA import sma
import time


def test_sma_basic():
    """Test basic SMA calculation with simple data."""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    period = 3

    result = sma(data, period)
    expected = np.array([2.0, 3.0, 4.0])  # (1+2+3)/3, (2+3+4)/3, (3+4+5)/3

    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_sma_constant_values():
    """Test SMA calculation with constant values."""
    data = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
    period = 3

    result = sma(data, period)
    expected = np.array([5.0, 5.0, 5.0])

    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_sma_increasing_values():
    """Test SMA calculation with strictly increasing values."""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    period = 4

    result = sma(data, period)
    expected = np.array([2.5, 3.5, 4.5, 5.5])  # (1+2+3+4)/4, (2+3+4+5)/4, etc.

    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_sma_decreasing_values():
    """Test SMA calculation with strictly decreasing values."""
    data = np.array([7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
    period = 4

    result = sma(data, period)
    expected = np.array([5.5, 4.5, 3.5, 2.5])  # (7+6+5+4)/4, (6+5+4+3)/4, etc.

    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_sma_oscillating_values():
    """Test SMA calculation with oscillating values."""
    data = np.array([1.0, 3.0, 1.0, 3.0, 1.0, 3.0, 1.0])
    period = 3

    result = sma(data, period)
    expected = np.array([1.666667, 2.333333, 1.666667, 2.333333, 1.666667])

    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_sma_period_equals_length():
    """Test SMA calculation when period equals data length."""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    period = 5

    result = sma(data, period)
    expected = np.array([3.0])  # (1+2+3+4+5)/5

    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_sma_period_greater_than_length():
    """Test SMA calculation when period is greater than data length."""
    data = np.array([1.0, 2.0, 3.0])
    period = 5

    with pytest.raises(ValueError):
        sma(data, period)


def test_sma_period_one():
    """Test SMA calculation with period of 1."""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    period = 1

    result = sma(data, period)
    expected = data  # SMA with period 1 should return the original data

    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_sma_performance():
    """Test SMA performance with large dataset."""
    # Generate a large dataset
    np.random.seed(42)
    data = np.random.normal(100, 10, 10000)
    period = 20

    # Time our implementation
    start = time.time()
    result = sma(data, period)
    our_time = time.time() - start

    # Time pandas implementation
    start = time.time()
    expected = pd.Series(data).rolling(window=period).mean().dropna().values
    pandas_time = time.time() - start

    # Verify results match
    np.testing.assert_allclose(result, expected, rtol=1e-5)

    # Print performance comparison
    print(f"\nSMA Performance (10000 points):")
    print(f"Our implementation: {our_time*1000:.2f} ms")
    print(f"Pandas implementation: {pandas_time*1000:.2f} ms")
    print(f"Speedup: {pandas_time/our_time:.2f}x")


def test_sma_performance_varying_periods():
    """Test SMA performance with different periods."""
    np.random.seed(42)
    data = np.random.normal(100, 10, 1000)
    periods = [5, 10, 20, 50, 100]

    print("\nSMA Performance with varying periods (1000 points):")
    print("Period | Our Time (ms) | Pandas Time (ms) | Speedup")
    print("-" * 50)

    for period in periods:
        # Time our implementation
        start = time.time()
        result = sma(data, period)
        our_time = time.time() - start

        # Time pandas implementation
        start = time.time()
        expected = pd.Series(data).rolling(window=period).mean().dropna().values
        pandas_time = time.time() - start

        # Verify results match
        np.testing.assert_allclose(result, expected, rtol=1e-5)

        # Print performance comparison
        print(
            f"{period:6d} | {our_time*1000:12.2f} | {pandas_time*1000:14.2f} | {pandas_time/our_time:8.2f}x"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
