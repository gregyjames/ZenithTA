import pytest
import numpy as np
import pandas as pd
from ZenithTA import roc
import time

def pandas_roc(series, period):
    series = pd.Series(series)
    # ROC = ((Price_t - Price_t-n) / Price_t-n) * 100
    # In ZenithTA:
    # result[i] = ((price_ndarray[i+period] - denominators[i]) / denominators[i]) * 100.0
    # where denominators[i] is price_ndarray[i]
    # So result[i] is ((price_ndarray[i+period] - price_ndarray[i]) / price_ndarray[i]) * 100.0
    return ((series.shift(-period) - series) / series * 100.0).dropna().values

def test_roc_basic():
    prices = np.array([10.0, 12.0, 11.0, 13.0, 15.0, 14.0], dtype=np.float32)
    period = 2
    
    result = roc(prices, period)
    expected = pandas_roc(prices, period)
    
    np.testing.assert_allclose(result, expected, rtol=1e-5)

def test_roc_performance():
    np.random.seed(42)
    prices = np.random.normal(100, 5, 10000).astype(np.float32)
    period = 10
    
    start = time.time()
    result = roc(prices, period)
    our_time = time.time() - start
    
    start = time.time()
    expected = pandas_roc(prices, period)
    pandas_time = time.time() - start
    
    print(f"\nROC Performance (10000 points):")
    print(f"Our implementation: {our_time*1000:.2f} ms")
    print(f"Pandas/Traditional implementation: {pandas_time*1000:.2f} ms")
    print(f"Speedup: {pandas_time/our_time:.2f}x")
