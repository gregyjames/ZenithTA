use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use numpy::ndarray::prelude::*;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use numpy::ndarray::{Array1, ArrayBase, Axis, Ix1, OwnedRepr};
use pyo3::exceptions::PyValueError;
use pyo3::types::PyModule;

//Helper functions for indicators
fn sma_helper(price_ndarray: &Array1<f32>, period: usize) -> Array1<f32> {
    // Add validation for period
    if period > price_ndarray.len() {
        panic!("Period ({}) cannot be greater than data length ({})", period, price_ndarray.len());
    }
    
    let length = price_ndarray.len() - period + 1;
    let mut result = Array1::<f32>::zeros(length);
    
    // Calculate initial sum
    let mut sum: f32 = price_ndarray.slice(s![0..period]).sum();
    result[0] = sum / period as f32;
    
    // Use sliding window for subsequent values
    for i in 1..length {
        sum = sum - price_ndarray[i-1] + price_ndarray[i+period-1];
        result[i] = sum / period as f32;
    }
    
    result
}

fn ema_helper(price_ndarray: &Array1<f32>, period: usize) -> Array1<f32> {
    let length = price_ndarray.len() - period + 1;
    let mut result = Array1::<f32>::zeros(length);
    let alpha = 2.0 / (period as f32 + 1.0);
    
    // Calculate initial SMA
    let sum: f32 = price_ndarray.slice(s![0..period]).sum();
    result[0] = sum / period as f32;
    
    // Use EMA formula with optimized multiplication
    for i in 1..length {
        result[i] = result[i-1] + (price_ndarray[i+period-1] - result[i-1]) * alpha;
    }
    
    result
}

//Indicator Python function wrappers
#[pyfunction]
fn sma(price: Vec<f32>, period: usize) -> PyResult<Vec<f32>> {
    if period > price.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!("Period ({}) cannot be greater than data length ({})", period, price.len())
        ));
    }
    
    let price_ndarray = Array::from_vec(price);
    Ok(Array::to_vec(&sma_helper(&price_ndarray, period)))
}

#[pyfunction]
fn ema(price: Vec<f32>, period: usize, smoothing: f32) -> PyResult<Vec<f32>> {
    let price_ndarray = Array::from_vec(price);
    let length = price_ndarray.len();
    let mut result = Array1::<f32>::zeros(length);
    let weight = smoothing / (period + 1) as f32;
    let one_minus_weight = 1.0 - weight;
    
    result[0] = price_ndarray[0];
    for i in 1..length {
        result[i] = price_ndarray[i].mul_add(weight, result[i-1] * one_minus_weight);
    }

    Ok(Array::to_vec(&result))
}

#[pyfunction]
fn rsi(prices: Vec<f64>, period: usize) -> PyResult<Vec<f64>> {
    let prices = Array::from_vec(prices);
    let len = prices.len();

    // Input validation
    if period == 0 {
        return Err(PyValueError::new_err("Period must be greater than 0"));
    }
    if period > len {
        return Err(PyValueError::new_err(format!(
            "Period ({}) cannot be greater than data length ({})",
            period, len
        )));
    }

    // Special case: constant values
    if prices.iter().all(|&x| x == prices[0]) {
        if period == len {
            return Ok(vec![50.0]);
        } else {
            let result = Array1::from_elem(len - period, 50.0);
            return Ok(Array::to_vec(&result));
        }
    }

    // Calculate price changes
    let changes = prices.slice(s![1..]).to_owned() - prices.slice(s![..-1]);
    let gains = changes.mapv(|x| if x > 0.0 { x } else { 0.0 });
    let losses = changes.mapv(|x| if x < 0.0 { -x } else { 0.0 });

    // Special case: period = 1
    if period == 1 {
        let mut result = Array1::zeros(len - 1);
        for i in 0..len-1 {
            if losses[i] == 0.0 {
                result[i] = 100.0;
            } else {
                result[i] = 100.0 - (100.0 / (1.0 + gains[i] / losses[i]));
            }
        }
        return Ok(Array::to_vec(&result));
    }

    // Special case: period == len
    if period == len {
        let avg_gain = gains.mean().unwrap();
        let avg_loss = losses.mean().unwrap();
        let rsi = if avg_loss == 0.0 {
            100.0
        } else {
            let rs = avg_gain / avg_loss;
            100.0 - (100.0 / (1.0 + rs))
        };
        return Ok(vec![rsi]);
    }

    // Calculate initial average gain and loss
    let mut avg_gain = gains.slice(s![..period]).mean().unwrap();
    let mut avg_loss = losses.slice(s![..period]).mean().unwrap();

    // Initialize result array (len - period)
    let mut result = Array1::zeros(len - period);

    // Calculate first RSI value (at index 0)
    if avg_loss == 0.0 {
        result[0] = 100.0;
    } else {
        let rs = avg_gain / avg_loss;
        result[0] = 100.0 - (100.0 / (1.0 + rs));
    }

    // Calculate remaining RSI values using smoothing
    for i in 1..len - period {
        avg_gain = ((avg_gain * (period - 1) as f64) + gains[i + period - 1]) / period as f64;
        avg_loss = ((avg_loss * (period - 1) as f64) + losses[i + period - 1]) / period as f64;

        if avg_loss == 0.0 {
            result[i] = 100.0;
        } else {
            let rs = avg_gain / avg_loss;
            result[i] = 100.0 - (100.0 / (1.0 + rs));
        }
    }

    Ok(Array::to_vec(&result))
}

#[pyfunction]
fn macd(price: Vec<f32>, period_fast: usize, period_slow: usize, period_signal: usize) -> PyResult<(Vec<f32>, Vec<f32>)> {
    let price_ndarray = Array::from_vec(price);
    let fast_ema = ema_helper(&price_ndarray, period_fast);
    let slow_ema = ema_helper(&price_ndarray, period_slow);
    
    // Calculate MACD line
    let macd_line = fast_ema.slice(s![period_slow-period_fast..]).to_owned() - slow_ema;
    
    // Calculate signal line using EMA of MACD
    let signal = ema_helper(&macd_line, period_signal);
    
    Ok((Array::to_vec(&macd_line), Array::to_vec(&signal)))
}

#[pyfunction]
fn roc(price: Vec<f32>, period: usize) -> PyResult<Vec<f32>> {
    let price_ndarray = Array::from_vec(price);
    let length = price_ndarray.len() - period;
    let mut result = Array1::<f32>::zeros(length);
    
    // Pre-calculate denominator values to avoid division in loop
    let denominators: Vec<f32> = price_ndarray.slice(s![..length]).to_vec();
    
    for i in 0..length {
        result[i] = ((price_ndarray[i+period] - denominators[i]) / denominators[i]) * 100.0;
    }
    
    Ok(Array::to_vec(&result))
}

#[pyfunction]
fn atr(high: Vec<f32>, low: Vec<f32>, close: Vec<f32>, period: usize) -> PyResult<Vec<f32>> {
    // Validate inputs
    if period == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("Period must be greater than 0"));
    }
    if period > high.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!("Period ({}) cannot be greater than data length ({})", period, high.len())
        ));
    }
    if high.len() != low.len() || high.len() != close.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!("Input arrays must have the same length. Got high: {}, low: {}, close: {}", 
                   high.len(), low.len(), close.len())
        ));
    }

    let length = high.len();
    let high_ndarray = Array::from_vec(high);
    let low_ndarray = Array::from_vec(low);
    let close_ndarray = Array::from_vec(close);
    
    // Calculate True Range
    let mut tr = Array1::<f32>::zeros(length);
    tr[0] = high_ndarray[0] - low_ndarray[0];  // First TR is just high-low
    
    for i in 1..length {
        let hl = high_ndarray[i] - low_ndarray[i];
        let hpc = (high_ndarray[i] - close_ndarray[i-1]).abs();
        let lpc = (low_ndarray[i] - close_ndarray[i-1]).abs();
        tr[i] = hl.max(hpc).max(lpc);
    }
    
    // Calculate ATR using simple moving average
    let mut result = Array1::<f32>::zeros(length - period + 1);
    
    // Calculate initial ATR
    let mut sum: f32 = tr.slice(s![0..period]).sum();
    result[0] = sum / period as f32;
    
    // Use sliding window for subsequent values
    for i in 1..length-period+1 {
        sum = sum - tr[i-1] + tr[i+period-1];
        result[i] = sum / period as f32;
    }
    
    Ok(Array::to_vec(&result))
}

#[pyfunction]
fn cmf(high: Vec<f32>, low: Vec<f32>, close: Vec<f32>, volume: Vec<f32>, period: usize) -> PyResult<Vec<f32>> {
    let length = high.len();
    let high_ndarray = Array::from_vec(high);
    let low_ndarray = Array::from_vec(low);
    let close_ndarray = Array::from_vec(close);
    let volume_ndarray = Array::from_vec(volume);
    let mut result = Array1::<f32>::zeros(length - period + 1);
    
    // Pre-calculate money flow multiplier
    let mut mfv = Array1::<f32>::zeros(length);
    for i in 0..length {
        let range = high_ndarray[i] - low_ndarray[i];
        if range != 0.0 {
            mfv[i] = ((close_ndarray[i] - low_ndarray[i] - (high_ndarray[i] - close_ndarray[i])) / range) * volume_ndarray[i];
        }
    }
    
    // Calculate initial sum
    let mut mfv_sum: f32 = mfv.slice(s![0..period]).sum();
    let mut vol_sum: f32 = volume_ndarray.slice(s![0..period]).sum();
    result[0] = mfv_sum / vol_sum;
    
    // Use sliding window for subsequent values
    for i in 1..length-period+1 {
        mfv_sum = mfv_sum - mfv[i-1] + mfv[i+period-1];
        vol_sum = vol_sum - volume_ndarray[i-1] + volume_ndarray[i+period-1];
        result[i] = mfv_sum / vol_sum;
    }
    
    Ok(Array::to_vec(&result))
}

#[pymodule]
fn ZenithTA(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sma, m)?)?;
    m.add_function(wrap_pyfunction!(cmf, m)?)?;
    m.add_function(wrap_pyfunction!(atr, m)?)?;
    m.add_function(wrap_pyfunction!(ema, m)?)?;
    m.add_function(wrap_pyfunction!(rsi, m)?)?;
    m.add_function(wrap_pyfunction!(roc, m)?)?;
    m.add_function(wrap_pyfunction!(macd, m)?)?;
    Ok(())
}

