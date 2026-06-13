import pytest
import numpy as np
import pandas as pd
from ZenithTA import rsi
import time


def test_rsi_basic():
    """Test basic RSI calculation with simple data."""
    # Simple price series with alternating gains and losses
    prices = np.array([10.0, 11.0, 9.0, 10.0, 8.0, 9.0, 7.0, 8.0])
    period = 3

    result = rsi(prices, period)

    # Calculate expected RSI manually
    # First calculate price changes
    changes = np.diff(prices)
    gains = np.where(changes > 0, changes, 0)
    losses = np.where(changes < 0, -changes, 0)

    # Calculate initial averages
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    rs = avg_gain / avg_loss if avg_loss != 0 else float("inf")
    expected_first = 100.0 - (100.0 / (1.0 + rs))

    # Calculate subsequent values using smoothing
    expected = [expected_first]
    for i in range(1, len(changes) - period + 1):
        avg_gain = (avg_gain * (period - 1) + gains[i + period - 1]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i + period - 1]) / period
        rs = avg_gain / avg_loss if avg_loss != 0 else float("inf")
        expected.append(100.0 - (100.0 / (1.0 + rs)))

    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_rsi_constant_values():
    """Test RSI calculation with constant values."""
    prices = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
    period = 3

    result = rsi(prices, period)
    expected = np.array([50.0, 50.0])  # RSI should be 50 for constant values

    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_rsi_increasing_values():
    """Test RSI calculation with strictly increasing values."""
    prices = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
    period = 3

    result = rsi(prices, period)

    # Calculate expected RSI
    changes = np.diff(prices)
    gains = changes  # All changes are positive
    losses = np.zeros_like(changes)

    # Calculate initial averages
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    rs = avg_gain / avg_loss if avg_loss != 0 else float("inf")
    expected_first = 100.0 - (100.0 / (1.0 + rs))

    # Calculate subsequent values using smoothing
    expected = [expected_first]
    for i in range(1, len(changes) - period + 1):
        avg_gain = (avg_gain * (period - 1) + gains[i + period - 1]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i + period - 1]) / period
        rs = avg_gain / avg_loss if avg_loss != 0 else float("inf")
        expected.append(100.0 - (100.0 / (1.0 + rs)))

    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_rsi_decreasing_values():
    """Test RSI calculation with strictly decreasing values."""
    prices = np.array([15.0, 14.0, 13.0, 12.0, 11.0, 10.0])
    period = 3

    result = rsi(prices, period)

    # Calculate expected RSI
    changes = np.diff(prices)
    gains = np.zeros_like(changes)
    losses = -changes  # All changes are negative

    # Calculate initial averages
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    rs = avg_gain / avg_loss if avg_loss != 0 else float("inf")
    expected_first = 100.0 - (100.0 / (1.0 + rs))

    # Calculate subsequent values using smoothing
    expected = [expected_first]
    for i in range(1, len(changes) - period + 1):
        avg_gain = (avg_gain * (period - 1) + gains[i + period - 1]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i + period - 1]) / period
        rs = avg_gain / avg_loss if avg_loss != 0 else float("inf")
        expected.append(100.0 - (100.0 / (1.0 + rs)))

    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_rsi_period_equals_length():
    """Test RSI calculation when period equals data length."""
    prices = np.array([10.0, 11.0, 12.0])
    period = 3

    result = rsi(prices, period)

    # Calculate expected RSI
    changes = np.diff(prices)
    gains = np.where(changes > 0, changes, 0)
    losses = np.where(changes < 0, -changes, 0)

    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)
    rs = avg_gain / avg_loss if avg_loss != 0 else float("inf")
    expected = np.array([100.0 - (100.0 / (1.0 + rs))])

    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_rsi_period_greater_than_length():
    """Test RSI calculation when period is greater than data length."""
    prices = np.array([10.0, 11.0, 12.0])
    period = 5

    with pytest.raises(ValueError):
        rsi(prices, period)


def test_rsi_period_one():
    """Test RSI calculation with period of 1."""
    prices = np.array([10.0, 11.0, 9.0, 10.0, 8.0])
    period = 1

    result = rsi(prices, period)

    # Calculate expected RSI
    changes = np.diff(prices)
    gains = np.where(changes > 0, changes, 0)
    losses = np.where(changes < 0, -changes, 0)

    # For period=1, RSI is just based on the last change
    expected = np.array(
        [
            100.0 if losses[0] == 0 else 100.0 - (100.0 / (1.0 + gains[0] / losses[0])),
            100.0 if losses[1] == 0 else 100.0 - (100.0 / (1.0 + gains[1] / losses[1])),
            100.0 if losses[2] == 0 else 100.0 - (100.0 / (1.0 + gains[2] / losses[2])),
            100.0 if losses[3] == 0 else 100.0 - (100.0 / (1.0 + gains[3] / losses[3])),
        ]
    )

    np.testing.assert_allclose(result, expected, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
