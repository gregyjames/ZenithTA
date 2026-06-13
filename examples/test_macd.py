import pytest
import numpy as np
import pandas as pd
from ZenithTA import macd
import time

def pandas_ema(series, period):
    # Matches the ema_helper logic in rust:
    # 1. First value is SMA of the first 'period' values.
    # 2. Subsequent values use EMA formula: result[i] = result[i-1] + (price[i+period-1] - result[i-1]) * alpha
    # where alpha = 2.0 / (period + 1.0)
    alpha = 2.0 / (period + 1.0)
    result = np.zeros(len(series) - period + 1)
    result[0] = series[:period].mean()
    for i in range(1, len(result)):
        result[i] = result[i-1] + (series[i+period-1] - result[i-1]) * alpha
    return result

def pandas_macd(series, period_fast, period_slow, period_signal):
    fast_ema = pandas_ema(series, period_fast)
    slow_ema = pandas_ema(series, period_slow)
    
    # Slice fast_ema to match length of slow_ema
    macd_line = fast_ema[period_slow - period_fast:] - slow_ema
    signal_line = pandas_ema(macd_line, period_signal)
    return macd_line, signal_line

def test_macd_basic():
    np.random.seed(42)
    prices = np.random.normal(100, 5, 100).astype(np.float32)
    period_fast, period_slow, period_signal = 12, 26, 9
    
    our_macd, our_signal = macd(prices, period_fast, period_slow, period_signal)
    exp_macd, exp_signal = pandas_macd(prices, period_fast, period_slow, period_signal)
    
    np.testing.assert_allclose(our_macd, exp_macd, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(our_signal, exp_signal, rtol=1e-4, atol=1e-4)

def test_macd_performance():
    np.random.seed(42)
    prices = np.random.normal(100, 5, 10000).astype(np.float32)
    period_fast, period_slow, period_signal = 12, 26, 9
    
    start = time.time()
    our_macd, our_signal = macd(prices, period_fast, period_slow, period_signal)
    our_time = time.time() - start
    
    start = time.time()
    exp_macd, exp_signal = pandas_macd(prices, period_fast, period_slow, period_signal)
    pandas_time = time.time() - start
    
    print(f"\nMACD Performance (10000 points):")
    print(f"Our implementation: {our_time*1000:.2f} ms")
    print(f"Traditional Python/Pandas: {pandas_time*1000:.2f} ms")
    print(f"Speedup: {pandas_time/our_time:.2f}x")
