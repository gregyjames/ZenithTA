import pytest
import numpy as np
import pandas as pd
from ZenithTA import cmf
import time

def pandas_cmf(high, low, close, volume, period):
    high = pd.Series(high)
    low = pd.Series(low)
    close = pd.Series(close)
    volume = pd.Series(volume)
    
    # Money Flow Multiplier
    # ( (Close - Low) - (High - Close) ) / (High - Low)
    denom = high - low
    mf_mult = ((close - low) - (high - close)) / denom
    # Handle division by zero (high == low)
    mf_mult = mf_mult.fillna(0.0)
    
    # Money Flow Volume
    mf_vol = mf_mult * volume
    
    # Chaikin Money Flow
    cmf_series = mf_vol.rolling(period).sum() / volume.rolling(period).sum()
    return cmf_series.dropna().values

def test_cmf_basic():
    high = np.array([10, 12, 11, 13, 15], dtype=np.float32)
    low = np.array([8, 9, 10, 11, 12], dtype=np.float32)
    close = np.array([9, 11, 10.5, 12, 14], dtype=np.float32)
    volume = np.array([100, 150, 120, 200, 180], dtype=np.float32)
    period = 3
    
    result = cmf(high, low, close, volume, period)
    expected = pandas_cmf(high, low, close, volume, period)
    
    np.testing.assert_allclose(result, expected, rtol=1e-5)

def test_cmf_performance():
    np.random.seed(42)
    high = np.random.normal(105, 5, 10000).astype(np.float32)
    low = high - np.random.uniform(1, 5, 10000).astype(np.float32)
    close = np.random.uniform(low, high).astype(np.float32)
    volume = np.random.uniform(100, 1000, 10000).astype(np.float32)
    period = 20
    
    start = time.time()
    result = cmf(high, low, close, volume, period)
    our_time = time.time() - start
    
    start = time.time()
    expected = pandas_cmf(high, low, close, volume, period)
    pandas_time = time.time() - start
    
    print(f"\nCMF Performance (10000 points):")
    print(f"Our implementation: {our_time*1000:.2f} ms")
    print(f"Pandas/Traditional implementation: {pandas_time*1000:.2f} ms")
    print(f"Speedup: {pandas_time/our_time:.2f}x")
