use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use numpy::{PyArray1, PyReadonlyArray1, PyArrayMethods, Element};
use pyo3::exceptions::PyValueError;
use pyo3::types::PyModule;

// Helper to get contiguous slice or return error
fn get_slice<'py, T: Element>(arr: &'py PyReadonlyArray1<'py, T>) -> PyResult<&'py [T]> {
    arr.as_slice().map_err(|_| {
        PyValueError::new_err("Input array must be contiguous and behave as a standard 1D slice")
    })
}

// Indicator Python function wrappers
#[pyfunction]
fn sma<'py>(py: Python<'py>, price: PyReadonlyArray1<'py, f32>, period: usize) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let price_slice = get_slice(&price)?;
    if period > price_slice.len() {
        return Err(PyValueError::new_err(
            format!("Period ({}) cannot be greater than data length ({})", period, price_slice.len())
        ));
    }
    
    let length = price_slice.len() - period + 1;
    let result_array = PyArray1::<f32>::zeros(py, length, false);
    
    // Safety: we write to the allocated array
    unsafe {
        let result_slice = result_array.as_slice_mut()?;
        let inv_period = 1.0 / period as f32;
        
        let mut sum: f32 = price_slice[0..period].iter().sum();
        result_slice[0] = sum * inv_period;
        
        for i in 1..length {
            sum = sum - price_slice[i - 1] + price_slice[i + period - 1];
            result_slice[i] = sum * inv_period;
        }
    }
    
    Ok(result_array)
}

#[pyfunction]
fn ema<'py>(py: Python<'py>, price: PyReadonlyArray1<'py, f32>, period: usize, smoothing: f32) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let price_slice = get_slice(&price)?;
    let length = price_slice.len();
    if length == 0 {
        return Ok(PyArray1::<f32>::zeros(py, 0, false));
    }
    
    let result_array = PyArray1::<f32>::zeros(py, length, false);
    unsafe {
        let result_slice = result_array.as_slice_mut()?;
        let weight = smoothing / (period + 1) as f32;
        let one_minus_weight = 1.0 - weight;
        
        result_slice[0] = price_slice[0];
        for i in 1..length {
            result_slice[i] = price_slice[i].mul_add(weight, result_slice[i - 1] * one_minus_weight);
        }
    }
    
    Ok(result_array)
}

#[pyfunction]
fn rsi<'py>(py: Python<'py>, prices: PyReadonlyArray1<'py, f64>, period: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let prices_slice = get_slice(&prices)?;
    let len = prices_slice.len();

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
    let is_constant = prices_slice.iter().all(|&x| x == prices_slice[0]);
    if is_constant {
        if period == len {
            let result_array = PyArray1::<f64>::zeros(py, 1, false);
            unsafe { result_array.as_slice_mut()?[0] = 50.0; }
            return Ok(result_array);
        } else {
            let result_array = PyArray1::<f64>::zeros(py, len - period, false);
            unsafe {
                let s = result_array.as_slice_mut()?;
                s.fill(50.0);
            }
            return Ok(result_array);
        }
    }

    // Special case: period = 1
    if period == 1 {
        let result_array = PyArray1::<f64>::zeros(py, len - 1, false);
        unsafe {
            let result_slice = result_array.as_slice_mut()?;
            for i in 0..len - 1 {
                let change = prices_slice[i + 1] - prices_slice[i];
                if change > 0.0 {
                    result_slice[i] = 100.0;
                } else if change < 0.0 {
                    result_slice[i] = 0.0;
                } else {
                    result_slice[i] = 50.0;
                }
            }
        }
        return Ok(result_array);
    }

    // Special case: period == len
    if period == len {
        let mut sum_gain = 0.0;
        let mut sum_loss = 0.0;
        for i in 0..len - 1 {
            let change = prices_slice[i + 1] - prices_slice[i];
            if change > 0.0 {
                sum_gain += change;
            } else {
                sum_loss -= change;
            }
        }
        let avg_gain = sum_gain / (len - 1) as f64;
        let avg_loss = sum_loss / (len - 1) as f64;
        let rsi_val = if avg_loss == 0.0 {
            100.0
        } else {
            100.0 - (100.0 / (1.0 + (avg_gain / avg_loss)))
        };
        let result_array = PyArray1::<f64>::zeros(py, 1, false);
        unsafe { result_array.as_slice_mut()?[0] = rsi_val; }
        return Ok(result_array);
    }

    // Standard RSI with Wilder's smoothing
    let result_array = PyArray1::<f64>::zeros(py, len - period, false);
    unsafe {
        let result_slice = result_array.as_slice_mut()?;
        
        let mut avg_gain = 0.0;
        let mut avg_loss = 0.0;
        
        // Initial window
        for i in 0..period {
            let change = prices_slice[i + 1] - prices_slice[i];
            if change > 0.0 {
                avg_gain += change;
            } else {
                avg_loss -= change;
            }
        }
        let inv_period = 1.0 / period as f64;
        avg_gain *= inv_period;
        avg_loss *= inv_period;

        if avg_loss == 0.0 {
            result_slice[0] = 100.0;
        } else {
            result_slice[0] = 100.0 - (100.0 / (1.0 + (avg_gain / avg_loss)));
        }

        let p_minus_1 = (period - 1) as f64;
        for i in 1..len - period {
            let change = prices_slice[i + period] - prices_slice[i + period - 1];
            let (gain, loss) = if change > 0.0 {
                (change, 0.0)
            } else {
                (0.0, -change)
            };
            
            avg_gain = (avg_gain * p_minus_1 + gain) * inv_period;
            avg_loss = (avg_loss * p_minus_1 + loss) * inv_period;

            if avg_loss == 0.0 {
                result_slice[i] = 100.0;
            } else {
                result_slice[i] = 100.0 - (100.0 / (1.0 + (avg_gain / avg_loss)));
            }
        }
    }

    Ok(result_array)
}

// Internal optimized EMA helper operating on raw slices
fn ema_slice_helper(price: &[f32], period: usize, dest: &mut [f32]) {
    let length = price.len() - period + 1;
    let alpha = 2.0 / (period as f32 + 1.0);
    
    // Initial SMA
    let sum: f32 = price[0..period].iter().sum();
    dest[0] = sum / period as f32;
    
    for i in 1..length {
        dest[i] = dest[i - 1] + (price[i + period - 1] - dest[i - 1]) * alpha;
    }
}

#[pyfunction]
fn macd<'py>(
    py: Python<'py>,
    price: PyReadonlyArray1<'py, f32>,
    period_fast: usize,
    period_slow: usize,
    period_signal: usize
) -> PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<f32>>)> {
    let price_slice = get_slice(&price)?;
    
    let fast_len = price_slice.len() - period_fast + 1;
    let slow_len = price_slice.len() - period_slow + 1;
    
    let mut fast_ema = vec![0.0; fast_len];
    ema_slice_helper(price_slice, period_fast, &mut fast_ema);
    
    let mut slow_ema = vec![0.0; slow_len];
    ema_slice_helper(price_slice, period_slow, &mut slow_ema);
    
    // Calculate MACD line
    let macd_len = slow_len;
    let macd_array = PyArray1::<f32>::zeros(py, macd_len, false);
    
    unsafe {
        let macd_slice = macd_array.as_slice_mut()?;
        let offset = period_slow - period_fast;
        for i in 0..macd_len {
            macd_slice[i] = fast_ema[i + offset] - slow_ema[i];
        }
        
        let signal_len = macd_len - period_signal + 1;
        let signal_array = PyArray1::<f32>::zeros(py, signal_len, false);
        let signal_slice = signal_array.as_slice_mut()?;
        
        ema_slice_helper(macd_slice, period_signal, signal_slice);
        
        Ok((macd_array, signal_array))
    }
}

#[pyfunction]
fn roc<'py>(py: Python<'py>, price: PyReadonlyArray1<'py, f32>, period: usize) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let price_slice = get_slice(&price)?;
    let length = price_slice.len() - period;
    
    let result_array = PyArray1::<f32>::zeros(py, length, false);
    unsafe {
        let result_slice = result_array.as_slice_mut()?;
        for i in 0..length {
            let denom = price_slice[i];
            result_slice[i] = ((price_slice[i + period] - denom) / denom) * 100.0;
        }
    }
    
    Ok(result_array)
}

#[pyfunction]
fn atr<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f32>,
    low: PyReadonlyArray1<'py, f32>,
    close: PyReadonlyArray1<'py, f32>,
    period: usize
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let high_slice = get_slice(&high)?;
    let low_slice = get_slice(&low)?;
    let close_slice = get_slice(&close)?;
    
    let length = high_slice.len();
    if period == 0 {
        return Err(PyValueError::new_err("Period must be greater than 0"));
    }
    if period > length {
        return Err(PyValueError::new_err(
            format!("Period ({}) cannot be greater than data length ({})", period, length)
        ));
    }
    if length != low_slice.len() || length != close_slice.len() {
        return Err(PyValueError::new_err(
            format!("Input arrays must have the same length. Got high: {}, low: {}, close: {}", 
                   length, low_slice.len(), close_slice.len())
        ));
    }

    // Calculate True Range directly in a temporary Vector to avoid multiple array overheads
    let mut tr = vec![0.0f32; length];
    tr[0] = high_slice[0] - low_slice[0];
    for i in 1..length {
        let hl = high_slice[i] - low_slice[i];
        let hpc = (high_slice[i] - close_slice[i - 1]).abs();
        let lpc = (low_slice[i] - close_slice[i - 1]).abs();
        tr[i] = hl.max(hpc).max(lpc);
    }
    
    let result_len = length - period + 1;
    let result_array = PyArray1::<f32>::zeros(py, result_len, false);
    unsafe {
        let result_slice = result_array.as_slice_mut()?;
        let inv_period = 1.0 / period as f32;
        
        let mut sum: f32 = tr[0..period].iter().sum();
        result_slice[0] = sum * inv_period;
        
        for i in 1..result_len {
            sum = sum - tr[i - 1] + tr[i + period - 1];
            result_slice[i] = sum * inv_period;
        }
    }
    
    Ok(result_array)
}

#[pyfunction]
fn cmf<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f32>,
    low: PyReadonlyArray1<'py, f32>,
    close: PyReadonlyArray1<'py, f32>,
    volume: PyReadonlyArray1<'py, f32>,
    period: usize
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let high_slice = get_slice(&high)?;
    let low_slice = get_slice(&low)?;
    let close_slice = get_slice(&close)?;
    let volume_slice = get_slice(&volume)?;
    
    let length = high_slice.len();
    if length != low_slice.len() || length != close_slice.len() || length != volume_slice.len() {
        return Err(PyValueError::new_err("All input arrays must have the same length"));
    }
    if period > length {
        return Err(PyValueError::new_err("Period cannot be greater than data length"));
    }
    
    // Calculate Money Flow Volume on the fly into a single flat vector
    let mut mfv = vec![0.0f32; length];
    for i in 0..length {
        let range = high_slice[i] - low_slice[i];
        if range != 0.0 {
            mfv[i] = ((close_slice[i] - low_slice[i] - (high_slice[i] - close_slice[i])) / range) * volume_slice[i];
        }
    }
    
    let result_len = length - period + 1;
    let result_array = PyArray1::<f32>::zeros(py, result_len, false);
    unsafe {
        let result_slice = result_array.as_slice_mut()?;
        
        let mut mfv_sum: f32 = mfv[0..period].iter().sum();
        let mut vol_sum: f32 = volume_slice[0..period].iter().sum();
        result_slice[0] = mfv_sum / vol_sum;
        
        for i in 1..result_len {
            mfv_sum = mfv_sum - mfv[i - 1] + mfv[i + period - 1];
            vol_sum = vol_sum - volume_slice[i - 1] + volume_slice[i + period - 1];
            result_slice[i] = mfv_sum / vol_sum;
        }
    }
    
    Ok(result_array)
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

