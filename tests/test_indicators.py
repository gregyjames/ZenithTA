import pytest
import numpy as np
import pandas as pd
from ZenithTA import sma, ema, rsi, macd, roc, atr, cmf

# --- SMA Tests ---
def test_sma():
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    result = sma(data, 3)
    expected = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    np.testing.assert_allclose(result, expected, rtol=1e-5)

    # Constant values
    result_const = sma(np.array([5.0, 5.0, 5.0], dtype=np.float32), 2)
    np.testing.assert_allclose(result_const, [5.0, 5.0], rtol=1e-5)

    # Error on period too large
    with pytest.raises(ValueError):
        sma(data, 10)


# --- EMA Tests ---
def test_ema():
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    # Testing with basic inputs
    result = ema(data, 3, 2.0)
    
    # Calculate expected EMA manually
    # result[0] = price[0]
    # result[i] = price[i] * weight + result[i-1] * (1 - weight)
    # where weight = smoothing / (period + 1) -> 2.0 / (3 + 1) = 0.5
    expected = [1.0]
    weight = 0.5
    for val in data[1:]:
        expected.append(val * weight + expected[-1] * (1.0 - weight))
        
    np.testing.assert_allclose(result, expected, rtol=1e-5)


# --- RSI Tests ---
def test_rsi():
    prices = np.array([10.0, 11.0, 12.0, 11.0, 10.0, 11.0, 12.0])
    result = rsi(prices, 3)
    
    # Calculate RSI using pandas/traditional method
    changes = pd.Series(prices).diff().dropna()
    gains = changes.clip(lower=0)
    losses = -changes.clip(upper=0)
    
    avg_gain = gains.rolling(3).mean().dropna().values
    avg_loss = losses.rolling(3).mean().dropna().values
    
    # Wilder's smoothing
    smoothed_gain = np.zeros(len(avg_gain))
    smoothed_loss = np.zeros(len(avg_loss))
    smoothed_gain[0] = avg_gain[0]
    smoothed_loss[0] = avg_loss[0]
    
    for i in range(1, len(avg_gain)):
        smoothed_gain[i] = (smoothed_gain[i-1] * 2 + gains.values[i+2]) / 3
        smoothed_loss[i] = (smoothed_loss[i-1] * 2 + losses.values[i+2]) / 3
        
    rs = smoothed_gain / np.where(smoothed_loss == 0, 1e-9, smoothed_loss)
    expected = 100 - (100 / (1 + rs))
    
    np.testing.assert_allclose(result, expected, rtol=1e-5)


# --- MACD Tests ---
def test_macd():
    np.random.seed(42)
    prices = np.random.normal(100, 5, 50).astype(np.float32)
    period_fast, period_slow, period_signal = 12, 26, 9
    
    our_macd, our_signal = macd(prices, period_fast, period_slow, period_signal)
    
    # Verify shape consistency
    assert len(our_macd) == len(prices) - period_slow + 1
    assert len(our_signal) == len(our_macd) - period_signal + 1


# --- ROC Tests ---
def test_roc():
    prices = np.array([10.0, 12.0, 11.0, 13.0, 15.0], dtype=np.float32)
    result = roc(prices, 2)
    # ((Price[i+2] - Price[i]) / Price[i]) * 100
    expected = [
        ((11.0 - 10.0) / 10.0) * 100.0,
        ((13.0 - 12.0) / 12.0) * 100.0,
        ((15.0 - 11.0) / 11.0) * 100.0,
    ]
    np.testing.assert_allclose(result, expected, rtol=1e-5)


# --- ATR Tests ---
def test_atr():
    high = np.array([10.0, 11.0, 12.0, 13.0, 14.0], dtype=np.float32)
    low = np.array([9.0, 10.0, 11.0, 12.0, 13.0], dtype=np.float32)
    close = np.array([9.5, 10.5, 11.5, 12.5, 13.5], dtype=np.float32)
    
    result = atr(high, low, close, 3)
    
    tr = np.zeros(len(high))
    tr[0] = high[0] - low[0]
    for i in range(1, len(high)):
        hl = high[i] - low[i]
        hpc = abs(high[i] - close[i - 1])
        lpc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hpc, lpc)
        
    expected = [
        np.mean(tr[0:3]),
        np.mean(tr[1:4]),
        np.mean(tr[2:5]),
    ]
    np.testing.assert_allclose(result, expected, rtol=1e-5)


# --- CMF Tests ---
def test_cmf():
    high = np.array([10, 12, 11, 13, 15], dtype=np.float32)
    low = np.array([8, 9, 10, 11, 12], dtype=np.float32)
    close = np.array([9, 11, 10.5, 12, 14], dtype=np.float32)
    volume = np.array([100, 150, 120, 200, 180], dtype=np.float32)
    
    result = cmf(high, low, close, volume, 3)
    
    # Manual Money Flow Multiplier
    mf_mult = ((close - low) - (high - close)) / (high - low)
    mf_vol = mf_mult * volume
    
    expected = [
        np.sum(mf_vol[0:3]) / np.sum(volume[0:3]),
        np.sum(mf_vol[1:4]) / np.sum(volume[1:4]),
        np.sum(mf_vol[2:5]) / np.sum(volume[2:5]),
    ]
    np.testing.assert_allclose(result, expected, rtol=1e-5)


# --- Ergonomic and Type Dispatch Tests ---
def test_ergonomics():
    # 1. Test standard Python list
    data_list = [1.0, 2.0, 3.0, 4.0, 5.0]
    res_list = sma(data_list, 3)
    np.testing.assert_allclose(res_list, [2.0, 3.0, 4.0])

    # 2. Test Pandas Series (float64)
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    res_series = sma(series, 3)
    np.testing.assert_allclose(res_series, [2.0, 3.0, 4.0])
    assert res_series.dtype == np.float64

    # 3. Test Pandas DataFrame Column (float64)
    df = pd.DataFrame({"Close": [1.0, 2.0, 3.0, 4.0, 5.0]})
    res_df = sma(df["Close"], 3)
    np.testing.assert_allclose(res_df, [2.0, 3.0, 4.0])

    # 4. Test 2D Column Vector (shape (N, 1))
    col_vector = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]], dtype=np.float32)
    res_col = sma(col_vector, 3)
    np.testing.assert_allclose(res_col, [2.0, 3.0, 4.0])
    assert res_col.dtype == np.float32

    # 5. Test 2D Row Vector (shape (1, N))
    row_vector = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float64)
    res_row = sma(row_vector, 3)
    np.testing.assert_allclose(res_row, [2.0, 3.0, 4.0])
    assert res_row.dtype == np.float64

    # 6. Verify shape errors
    invalid_2d = np.array([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError, match="2D array must be a column/row vector"):
        sma(invalid_2d, 2)

    invalid_3d = np.array([[[1.0]]])
    with pytest.raises(ValueError, match="Input array must be 1D or a 2D column/row vector"):
        sma(invalid_3d, 1)

