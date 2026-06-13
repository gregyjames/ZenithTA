import pytest
import numpy as np
import pandas as pd
from ZenithTA import atr
import time


def test_atr_basic():
    """Test basic ATR calculation with simple data."""
    high = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
    low = np.array([9.0, 10.0, 11.0, 12.0, 13.0])
    close = np.array([9.5, 10.5, 11.5, 12.5, 13.5])
    period = 3

    result = atr(high, low, close, period)

    # Calculate expected ATR manually
    tr = np.zeros(len(high))
    tr[0] = high[0] - low[0]  # First TR is just high-low
    for i in range(1, len(high)):
        hl = high[i] - low[i]
        hpc = abs(high[i] - close[i - 1])
        lpc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hpc, lpc)

    # Calculate ATR using simple moving average
    expected = np.array(
        [
            np.mean(tr[0:period]),
            np.mean(tr[1 : period + 1]),
            np.mean(tr[2 : period + 2]),
        ]
    )

    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_atr_constant_values():
    """Test ATR calculation with constant values."""
    high = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
    low = np.array([9.0, 9.0, 9.0, 9.0, 9.0])
    close = np.array([9.5, 9.5, 9.5, 9.5, 9.5])
    period = 3

    result = atr(high, low, close, period)

    # Calculate expected ATR
    tr = np.zeros(len(high))
    tr[0] = high[0] - low[0]
    for i in range(1, len(high)):
        hl = high[i] - low[i]
        hpc = abs(high[i] - close[i - 1])
        lpc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hpc, lpc)

    expected = np.array(
        [
            np.mean(tr[0:period]),
            np.mean(tr[1 : period + 1]),
            np.mean(tr[2 : period + 2]),
        ]
    )

    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_atr_increasing_values():
    """Test ATR calculation with strictly increasing values."""
    high = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0])
    low = np.array([9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
    close = np.array([9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5])
    period = 3

    result = atr(high, low, close, period)

    # Calculate expected ATR
    tr = np.zeros(len(high))
    tr[0] = high[0] - low[0]
    for i in range(1, len(high)):
        hl = high[i] - low[i]
        hpc = abs(high[i] - close[i - 1])
        lpc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hpc, lpc)

    expected = np.array(
        [
            np.mean(tr[0:period]),
            np.mean(tr[1 : period + 1]),
            np.mean(tr[2 : period + 2]),
            np.mean(tr[3 : period + 3]),
            np.mean(tr[4 : period + 4]),
        ]
    )

    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_atr_gap_up():
    """Test ATR calculation with a gap up in prices."""
    high = np.array([10.0, 15.0, 16.0, 17.0, 18.0])
    low = np.array([9.0, 14.0, 15.0, 16.0, 17.0])
    close = np.array([9.5, 14.5, 15.5, 16.5, 17.5])
    period = 3

    result = atr(high, low, close, period)

    # Calculate expected ATR
    tr = np.zeros(len(high))
    tr[0] = high[0] - low[0]
    for i in range(1, len(high)):
        hl = high[i] - low[i]
        hpc = abs(high[i] - close[i - 1])
        lpc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hpc, lpc)

    expected = np.array(
        [
            np.mean(tr[0:period]),
            np.mean(tr[1 : period + 1]),
            np.mean(tr[2 : period + 2]),
        ]
    )

    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_atr_gap_down():
    """Test ATR calculation with a gap down in prices."""
    high = np.array([18.0, 13.0, 14.0, 15.0, 16.0])
    low = np.array([17.0, 12.0, 13.0, 14.0, 15.0])
    close = np.array([17.5, 12.5, 13.5, 14.5, 15.5])
    period = 3

    result = atr(high, low, close, period)

    # Calculate expected ATR
    tr = np.zeros(len(high))
    tr[0] = high[0] - low[0]
    for i in range(1, len(high)):
        hl = high[i] - low[i]
        hpc = abs(high[i] - close[i - 1])
        lpc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hpc, lpc)

    expected = np.array(
        [
            np.mean(tr[0:period]),
            np.mean(tr[1 : period + 1]),
            np.mean(tr[2 : period + 2]),
        ]
    )

    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_atr_period_equals_length():
    """Test ATR calculation when period equals data length."""
    high = np.array([10.0, 11.0, 12.0])
    low = np.array([9.0, 10.0, 11.0])
    close = np.array([9.5, 10.5, 11.5])
    period = 3

    result = atr(high, low, close, period)

    # Calculate expected ATR
    tr = np.zeros(len(high))
    tr[0] = high[0] - low[0]
    for i in range(1, len(high)):
        hl = high[i] - low[i]
        hpc = abs(high[i] - close[i - 1])
        lpc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hpc, lpc)

    expected = np.array([np.mean(tr)])

    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_atr_period_greater_than_length():
    """Test ATR calculation when period is greater than data length."""
    high = np.array([10.0, 11.0, 12.0])
    low = np.array([9.0, 10.0, 11.0])
    close = np.array([9.5, 10.5, 11.5])
    period = 5

    with pytest.raises(ValueError):
        atr(high, low, close, period)


def test_atr_period_one():
    """Test ATR calculation with period of 1."""
    high = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
    low = np.array([9.0, 10.0, 11.0, 12.0, 13.0])
    close = np.array([9.5, 10.5, 11.5, 12.5, 13.5])
    period = 1

    result = atr(high, low, close, period)

    # Calculate expected ATR
    tr = np.zeros(len(high))
    tr[0] = high[0] - low[0]
    for i in range(1, len(high)):
        hl = high[i] - low[i]
        hpc = abs(high[i] - close[i - 1])
        lpc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hpc, lpc)

    expected = tr

    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_atr_performance():
    """Test ATR performance with large dataset."""
    # Generate a large dataset
    np.random.seed(42)
    n_points = 10000
    base = np.random.normal(100, 10, n_points)
    high = base + np.random.uniform(0, 2, n_points)
    low = base - np.random.uniform(0, 2, n_points)
    close = (high + low) / 2
    period = 14

    # Time our implementation
    start = time.time()
    result = atr(high, low, close, period)
    our_time = time.time() - start

    # Time pandas implementation
    start = time.time()
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)

    tr1 = high_series - low_series
    tr2 = (high_series - close_series.shift(1)).abs()
    tr3 = (low_series - close_series.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    expected = tr.rolling(window=period).mean().dropna().values
    pandas_time = time.time() - start

    # Verify results match
    np.testing.assert_allclose(result, expected, rtol=1e-5)

    # Print performance comparison
    print(f"\nATR Performance (10000 points):")
    print(f"Our implementation: {our_time*1000:.2f} ms")
    print(f"Pandas implementation: {pandas_time*1000:.2f} ms")
    print(f"Speedup: {pandas_time/our_time:.2f}x")


def test_atr_performance_varying_periods():
    """Test ATR performance with different periods."""
    np.random.seed(42)
    n_points = 1000
    base = np.random.normal(100, 10, n_points)
    high = base + np.random.uniform(0, 2, n_points)
    low = base - np.random.uniform(0, 2, n_points)
    close = (high + low) / 2
    periods = [5, 10, 20, 50, 100]

    print("\nATR Performance with varying periods (1000 points):")
    print("Period | Our Time (ms) | Pandas Time (ms) | Speedup")
    print("-" * 50)

    for period in periods:
        # Time our implementation
        start = time.time()
        result = atr(high, low, close, period)
        our_time = time.time() - start

        # Time pandas implementation
        start = time.time()
        high_series = pd.Series(high)
        low_series = pd.Series(low)
        close_series = pd.Series(close)

        tr1 = high_series - low_series
        tr2 = (high_series - close_series.shift(1)).abs()
        tr3 = (low_series - close_series.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        expected = tr.rolling(window=period).mean().dropna().values
        pandas_time = time.time() - start

        # Verify results match
        np.testing.assert_allclose(result, expected, rtol=1e-5)

        # Print performance comparison
        print(
            f"{period:6d} | {our_time*1000:12.2f} | {pandas_time*1000:14.2f} | {pandas_time/our_time:8.2f}x"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
