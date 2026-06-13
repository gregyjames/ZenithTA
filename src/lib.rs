use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use numpy::{PyArray1, PyReadonlyArray, PyArrayMethods, PyUntypedArrayMethods, Element, IxDyn};
use pyo3::exceptions::PyValueError;
use pyo3::types::PyModule;
use pyo3::IntoPyObject;

// Helper to resolve Python objects (Pandas Series/DataFrames, lists) to a NumPy array
fn resolve_array<'py>(py: Python<'py>, obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let obj = if obj.hasattr("to_numpy")? {
        obj.call_method0("to_numpy")?
    } else {
        obj.clone()
    };
    let np = py.import("numpy")?;
    np.call_method1("asarray", (&obj,))
}

// Helper to borrow contiguous slice from a dynamic dimension array
fn get_slice_dyn<'py, T: Element>(arr: &'py PyReadonlyArray<'py, T, IxDyn>) -> PyResult<&'py [T]> {
    let shape = arr.shape();
    
    match shape.len() {
        1 => {
            arr.as_slice().map_err(|_| {
                PyValueError::new_err("Array must be contiguous")
            })
        }
        2 => {
            if shape[0] == 1 || shape[1] == 1 {
                arr.as_slice().map_err(|_| {
                    PyValueError::new_err("Array must be contiguous")
                })
            } else {
                Err(PyValueError::new_err(format!(
                    "2D array must be a column/row vector of shape (N, 1) or (1, N). Got shape {:?}",
                    shape
                )))
            }
        }
        _ => Err(PyValueError::new_err(format!(
            "Input array must be 1D or a 2D column/row vector. Got shape {:?}",
            shape
        )))
    }
}

// Dispatch macros to avoid code duplication across Python wrappers
macro_rules! dispatch_unary {
    ($py:expr, $price:expr, $generic_func:ident, $($args:expr),*) => {{
        let resolved = resolve_array($py, $price)?;
        let dtype = resolved.getattr("dtype")?;
        let np = $py.import("numpy")?;
        
        if dtype.eq(np.getattr("float64")?)? {
            let py_arr: PyReadonlyArray<'_, f64, IxDyn> = resolved.extract()?;
            let slice = get_slice_dyn(&py_arr)?;
            let res = $generic_func($py, slice, $($args),*)?;
            let obj = res.into_pyobject($py)?;
            Ok(obj.into_any().unbind())
        } else if dtype.eq(np.getattr("float32")?)? {
            let py_arr: PyReadonlyArray<'_, f32, IxDyn> = resolved.extract()?;
            let slice = get_slice_dyn(&py_arr)?;
            let res = $generic_func($py, slice, $($args),*)?;
            let obj = res.into_pyobject($py)?;
            Ok(obj.into_any().unbind())
        } else {
            Err(PyValueError::new_err("Input array must be float32 or float64"))
        }
    }};
}

macro_rules! dispatch_multi_3 {
    ($py:expr, $arg1:expr, $arg2:expr, $arg3:expr, $generic_func:ident, $($args:expr),*) => {{
        let res1 = resolve_array($py, $arg1)?;
        let res2 = resolve_array($py, $arg2)?;
        let res3 = resolve_array($py, $arg3)?;
        
        let dtype = res1.getattr("dtype")?;
        let np = $py.import("numpy")?;
        
        if dtype.eq(np.getattr("float64")?)? {
            let py_arr1: PyReadonlyArray<'_, f64, IxDyn> = res1.extract()?;
            let py_arr2: PyReadonlyArray<'_, f64, IxDyn> = res2.extract()?;
            let py_arr3: PyReadonlyArray<'_, f64, IxDyn> = res3.extract()?;
            let s1 = get_slice_dyn(&py_arr1)?;
            let s2 = get_slice_dyn(&py_arr2)?;
            let s3 = get_slice_dyn(&py_arr3)?;
            let res = $generic_func($py, s1, s2, s3, $($args),*)?;
            let obj = res.into_pyobject($py)?;
            Ok(obj.into_any().unbind())
        } else if dtype.eq(np.getattr("float32")?)? {
            let py_arr1: PyReadonlyArray<'_, f32, IxDyn> = res1.extract()?;
            let py_arr2: PyReadonlyArray<'_, f32, IxDyn> = res2.extract()?;
            let py_arr3: PyReadonlyArray<'_, f32, IxDyn> = res3.extract()?;
            let s1 = get_slice_dyn(&py_arr1)?;
            let s2 = get_slice_dyn(&py_arr2)?;
            let s3 = get_slice_dyn(&py_arr3)?;
            let res = $generic_func($py, s1, s2, s3, $($args),*)?;
            let obj = res.into_pyobject($py)?;
            Ok(obj.into_any().unbind())
        } else {
            Err(PyValueError::new_err("Input arrays must be float32 or float64"))
        }
    }};
}

macro_rules! dispatch_multi_4 {
    ($py:expr, $arg1:expr, $arg2:expr, $arg3:expr, $arg4:expr, $generic_func:ident, $($args:expr),*) => {{
        let res1 = resolve_array($py, $arg1)?;
        let res2 = resolve_array($py, $arg2)?;
        let res3 = resolve_array($py, $arg3)?;
        let res4 = resolve_array($py, $arg4)?;
        
        let dtype = res1.getattr("dtype")?;
        let np = $py.import("numpy")?;
        
        if dtype.eq(np.getattr("float64")?)? {
            let py_arr1: PyReadonlyArray<'_, f64, IxDyn> = res1.extract()?;
            let py_arr2: PyReadonlyArray<'_, f64, IxDyn> = res2.extract()?;
            let py_arr3: PyReadonlyArray<'_, f64, IxDyn> = res3.extract()?;
            let py_arr4: PyReadonlyArray<'_, f64, IxDyn> = res4.extract()?;
            let s1 = get_slice_dyn(&py_arr1)?;
            let s2 = get_slice_dyn(&py_arr2)?;
            let s3 = get_slice_dyn(&py_arr3)?;
            let s4 = get_slice_dyn(&py_arr4)?;
            let res = $generic_func($py, s1, s2, s3, s4, $($args),*)?;
            let obj = res.into_pyobject($py)?;
            Ok(obj.into_any().unbind())
        } else if dtype.eq(np.getattr("float32")?)? {
            let py_arr1: PyReadonlyArray<'_, f32, IxDyn> = res1.extract()?;
            let py_arr2: PyReadonlyArray<'_, f32, IxDyn> = res2.extract()?;
            let py_arr3: PyReadonlyArray<'_, f32, IxDyn> = res3.extract()?;
            let py_arr4: PyReadonlyArray<'_, f32, IxDyn> = res4.extract()?;
            let s1 = get_slice_dyn(&py_arr1)?;
            let s2 = get_slice_dyn(&py_arr2)?;
            let s3 = get_slice_dyn(&py_arr3)?;
            let s4 = get_slice_dyn(&py_arr4)?;
            let res = $generic_func($py, s1, s2, s3, s4, $($args),*)?;
            let obj = res.into_pyobject($py)?;
            Ok(obj.into_any().unbind())
        } else {
            Err(PyValueError::new_err("Input arrays must be float32 or float64"))
        }
    }};
}

// Generic implementations of technical analysis indicators
fn sma_generic<'py, T>(
    py: Python<'py>,
    price: &[T],
    period: usize,
) -> PyResult<Bound<'py, PyArray1<T>>>
where
    T: Element + num_traits::Float + Copy,
{
    if period > price.len() {
        return Err(PyValueError::new_err(format!(
            "Period ({}) cannot be greater than data length ({})",
            period,
            price.len()
        )));
    }
    let length = price.len() - period + 1;
    let result_array = PyArray1::<T>::zeros(py, length, false);
    unsafe {
        let result_slice = result_array.as_slice_mut()?;
        let inv_period = T::one() / T::from(period).unwrap();
        
        let mut sum: T = price[0..period].iter().copied().fold(T::zero(), |a, b| a + b);
        *result_slice.get_unchecked_mut(0) = sum * inv_period;
        
        for i in 1..length {
            let prev = *price.get_unchecked(i - 1);
            let next = *price.get_unchecked(i + period - 1);
            sum = sum - prev + next;
            *result_slice.get_unchecked_mut(i) = sum * inv_period;
        }
    }
    Ok(result_array)
}

fn ema_generic<'py, T>(
    py: Python<'py>,
    price: &[T],
    period: usize,
    smoothing: T,
) -> PyResult<Bound<'py, PyArray1<T>>>
where
    T: Element + num_traits::Float + Copy,
{
    let length = price.len();
    if length == 0 {
        return Ok(PyArray1::<T>::zeros(py, 0, false));
    }
    
    let result_array = PyArray1::<T>::zeros(py, length, false);
    unsafe {
        let result_slice = result_array.as_slice_mut()?;
        let weight = smoothing / (T::from(period).unwrap() + T::one());
        let one_minus_weight = T::one() - weight;
        
        *result_slice.get_unchecked_mut(0) = *price.get_unchecked(0);
        for i in 1..length {
            let p_val = *price.get_unchecked(i);
            let prev_r = *result_slice.get_unchecked(i - 1);
            *result_slice.get_unchecked_mut(i) = p_val.mul_add(weight, prev_r * one_minus_weight);
        }
    }
    Ok(result_array)
}

fn rsi_generic<'py, T>(
    py: Python<'py>,
    prices: &[T],
    period: usize,
) -> PyResult<Bound<'py, PyArray1<T>>>
where
    T: Element + num_traits::Float + Copy,
{
    let len = prices.len();
    if period == 0 {
        return Err(PyValueError::new_err("Period must be greater than 0"));
    }
    if period > len {
        return Err(PyValueError::new_err(format!(
            "Period ({}) cannot be greater than data length ({})",
            period, len
        )));
    }

    let is_constant = prices.iter().all(|&x| x == prices[0]);
    let val_50 = T::from(50.0).unwrap();
    let val_100 = T::from(100.0).unwrap();

    if is_constant {
        if period == len {
            let result_array = PyArray1::<T>::zeros(py, 1, false);
            unsafe { result_array.as_slice_mut()?[0] = val_50; }
            return Ok(result_array);
        } else {
            let result_array = PyArray1::<T>::zeros(py, len - period, false);
            unsafe {
                let s = result_array.as_slice_mut()?;
                s.fill(val_50);
            }
            return Ok(result_array);
        }
    }

    if period == 1 {
        let result_array = PyArray1::<T>::zeros(py, len - 1, false);
        unsafe {
            let result_slice = result_array.as_slice_mut()?;
            for i in 0..len - 1 {
                let change = *prices.get_unchecked(i + 1) - *prices.get_unchecked(i);
                if change > T::zero() {
                    *result_slice.get_unchecked_mut(i) = val_100;
                } else if change < T::zero() {
                    *result_slice.get_unchecked_mut(i) = T::zero();
                } else {
                    *result_slice.get_unchecked_mut(i) = val_50;
                }
            }
        }
        return Ok(result_array);
    }

    if period == len {
        let mut sum_gain = T::zero();
        let mut sum_loss = T::zero();
        for i in 0..len - 1 {
            let change = unsafe { *prices.get_unchecked(i + 1) - *prices.get_unchecked(i) };
            if change > T::zero() {
                sum_gain = sum_gain + change;
            } else {
                sum_loss = sum_loss - change;
            }
        }
        let avg_gain = sum_gain / T::from(len - 1).unwrap();
        let avg_loss = sum_loss / T::from(len - 1).unwrap();
        let rsi_val = if avg_loss == T::zero() {
            val_100
        } else {
            val_100 - (val_100 / (T::one() + (avg_gain / avg_loss)))
        };
        let result_array = PyArray1::<T>::zeros(py, 1, false);
        unsafe { result_array.as_slice_mut()?[0] = rsi_val; }
        return Ok(result_array);
    }

    let result_array = PyArray1::<T>::zeros(py, len - period, false);
    unsafe {
        let result_slice = result_array.as_slice_mut()?;
        let mut avg_gain = T::zero();
        let mut avg_loss = T::zero();
        
        for i in 0..period {
            let change = *prices.get_unchecked(i + 1) - *prices.get_unchecked(i);
            if change > T::zero() {
                avg_gain = avg_gain + change;
            } else {
                avg_loss = avg_loss - change;
            }
        }
        let inv_period = T::one() / T::from(period).unwrap();
        avg_gain = avg_gain * inv_period;
        avg_loss = avg_loss * inv_period;

        if avg_loss == T::zero() {
            *result_slice.get_unchecked_mut(0) = val_100;
        } else {
            *result_slice.get_unchecked_mut(0) = val_100 - (val_100 / (T::one() + (avg_gain / avg_loss)));
        }

        let p_minus_1 = T::from(period - 1).unwrap();
        for i in 1..len - period {
            let change = *prices.get_unchecked(i + period) - *prices.get_unchecked(i + period - 1);
            let (gain, loss) = if change > T::zero() {
                (change, T::zero())
            } else {
                (T::zero(), -change)
            };
            
            avg_gain = (avg_gain * p_minus_1 + gain) * inv_period;
            avg_loss = (avg_loss * p_minus_1 + loss) * inv_period;

            if avg_loss == T::zero() {
                *result_slice.get_unchecked_mut(i) = val_100;
            } else {
                *result_slice.get_unchecked_mut(i) = val_100 - (val_100 / (T::one() + (avg_gain / avg_loss)));
            }
        }
    }
    Ok(result_array)
}

// Internal optimized EMA helper operating on raw slices
fn ema_slice_helper<T>(price: &[T], period: usize, dest: &mut [T])
where
    T: num_traits::Float + Copy,
{
    let length = price.len() - period + 1;
    let alpha = T::from(2.0).unwrap() / (T::from(period).unwrap() + T::one());
    
    // Initial SMA
    let sum: T = price[0..period].iter().copied().fold(T::zero(), |a, b| a + b);
    unsafe {
        *dest.get_unchecked_mut(0) = sum / T::from(period).unwrap();
        
        for i in 1..length {
            let prev_d = *dest.get_unchecked(i - 1);
            let p_val = *price.get_unchecked(i + period - 1);
            *dest.get_unchecked_mut(i) = prev_d + (p_val - prev_d) * alpha;
        }
    }
}

fn macd_generic<'py, T>(
    py: Python<'py>,
    price: &[T],
    period_fast: usize,
    period_slow: usize,
    period_signal: usize,
) -> PyResult<(Bound<'py, PyArray1<T>>, Bound<'py, PyArray1<T>>)>
where
    T: Element + num_traits::Float + Copy,
{
    let fast_len = price.len() - period_fast + 1;
    let slow_len = price.len() - period_slow + 1;
    
    let mut fast_ema = vec![T::zero(); fast_len];
    ema_slice_helper(price, period_fast, &mut fast_ema);
    
    let mut slow_ema = vec![T::zero(); slow_len];
    ema_slice_helper(price, period_slow, &mut slow_ema);
    
    // Calculate MACD line
    let macd_len = slow_len;
    let macd_array = PyArray1::<T>::zeros(py, macd_len, false);
    
    unsafe {
        let macd_slice = macd_array.as_slice_mut()?;
        let offset = period_slow - period_fast;
        for i in 0..macd_len {
            *macd_slice.get_unchecked_mut(i) = *fast_ema.get_unchecked(i + offset) - *slow_ema.get_unchecked(i);
        }
        
        let signal_len = macd_len - period_signal + 1;
        let signal_array = PyArray1::<T>::zeros(py, signal_len, false);
        let signal_slice = signal_array.as_slice_mut()?;
        
        ema_slice_helper(macd_slice, period_signal, signal_slice);
        
        Ok((macd_array, signal_array))
    }
}

fn roc_generic<'py, T>(
    py: Python<'py>,
    price: &[T],
    period: usize,
) -> PyResult<Bound<'py, PyArray1<T>>>
where
    T: Element + num_traits::Float + Copy,
{
    let length = price.len() - period;
    let result_array = PyArray1::<T>::zeros(py, length, false);
    unsafe {
        let result_slice = result_array.as_slice_mut()?;
        let val_100 = T::from(100.0).unwrap();
        for i in 0..length {
            let denom = *price.get_unchecked(i);
            let next_p = *price.get_unchecked(i + period);
            *result_slice.get_unchecked_mut(i) = ((next_p - denom) / denom) * val_100;
        }
    }
    Ok(result_array)
}

fn atr_generic<'py, T>(
    py: Python<'py>,
    high: &[T],
    low: &[T],
    close: &[T],
    period: usize,
) -> PyResult<Bound<'py, PyArray1<T>>>
where
    T: Element + num_traits::Float + Copy,
{
    let length = high.len();
    if period == 0 {
        return Err(PyValueError::new_err("Period must be greater than 0"));
    }
    if period > length {
        return Err(PyValueError::new_err(format!(
            "Period ({}) cannot be greater than data length ({})",
            period, length
        )));
    }
    if length != low.len() || length != close.len() {
        return Err(PyValueError::new_err(format!(
            "Input arrays must have the same length. Got high: {}, low: {}, close: {}", 
            length, low.len(), close.len()
        )));
    }

    let mut tr = vec![T::zero(); length];
    unsafe {
        *tr.get_unchecked_mut(0) = *high.get_unchecked(0) - *low.get_unchecked(0);
        for i in 1..length {
            let h = *high.get_unchecked(i);
            let l = *low.get_unchecked(i);
            let c_prev = *close.get_unchecked(i - 1);
            let hl = h - l;
            let hpc = (h - c_prev).abs();
            let lpc = (l - c_prev).abs();
            *tr.get_unchecked_mut(i) = hl.max(hpc).max(lpc);
        }
    }
    
    let result_len = length - period + 1;
    let result_array = PyArray1::<T>::zeros(py, result_len, false);
    unsafe {
        let result_slice = result_array.as_slice_mut()?;
        let inv_period = T::one() / T::from(period).unwrap();
        
        let mut sum: T = tr[0..period].iter().copied().fold(T::zero(), |a, b| a + b);
        *result_slice.get_unchecked_mut(0) = sum * inv_period;
        
        for i in 1..result_len {
            let prev_tr = *tr.get_unchecked(i - 1);
            let next_tr = *tr.get_unchecked(i + period - 1);
            sum = sum - prev_tr + next_tr;
            *result_slice.get_unchecked_mut(i) = sum * inv_period;
        }
    }
    Ok(result_array)
}

fn cmf_generic<'py, T>(
    py: Python<'py>,
    high: &[T],
    low: &[T],
    close: &[T],
    volume: &[T],
    period: usize,
) -> PyResult<Bound<'py, PyArray1<T>>>
where
    T: Element + num_traits::Float + Copy,
{
    let length = high.len();
    if length != low.len() || length != close.len() || length != volume.len() {
        return Err(PyValueError::new_err("All input arrays must have the same length"));
    }
    if period > length {
        return Err(PyValueError::new_err("Period cannot be greater than data length"));
    }
    
    let mut mfv = vec![T::zero(); length];
    unsafe {
        for i in 0..length {
            let h = *high.get_unchecked(i);
            let l = *low.get_unchecked(i);
            let c = *close.get_unchecked(i);
            let v = *volume.get_unchecked(i);
            let range = h - l;
            if range != T::zero() {
                let money_flow_multiplier = ((c - l) - (h - c)) / range;
                *mfv.get_unchecked_mut(i) = money_flow_multiplier * v;
            }
        }
    }
    
    let result_len = length - period + 1;
    let result_array = PyArray1::<T>::zeros(py, result_len, false);
    unsafe {
        let result_slice = result_array.as_slice_mut()?;
        
        let mut mfv_sum: T = mfv[0..period].iter().copied().fold(T::zero(), |a, b| a + b);
        let mut vol_sum: T = volume[0..period].iter().copied().fold(T::zero(), |a, b| a + b);
        *result_slice.get_unchecked_mut(0) = mfv_sum / vol_sum;
        
        for i in 1..result_len {
            let prev_mfv = *mfv.get_unchecked(i - 1);
            let next_mfv = *mfv.get_unchecked(i + period - 1);
            let prev_vol = *volume.get_unchecked(i - 1);
            let next_vol = *volume.get_unchecked(i + period - 1);
            mfv_sum = mfv_sum - prev_mfv + next_mfv;
            vol_sum = vol_sum - prev_vol + next_vol;
            *result_slice.get_unchecked_mut(i) = mfv_sum / vol_sum;
        }
    }
    Ok(result_array)
}

// Wrapper PyO3 functions exposed to Python
#[pyfunction]
fn sma<'py>(py: Python<'py>, price: &Bound<'py, PyAny>, period: usize) -> PyResult<PyObject> {
    dispatch_unary!(py, price, sma_generic, period)
}

#[pyfunction]
fn ema<'py>(py: Python<'py>, price: &Bound<'py, PyAny>, period: usize, smoothing: f32) -> PyResult<PyObject> {
    let resolved = resolve_array(py, price)?;
    let dtype = resolved.getattr("dtype")?;
    let np = py.import("numpy")?;
    
    if dtype.eq(np.getattr("float64")?)? {
        let py_arr: PyReadonlyArray<'_, f64, IxDyn> = resolved.extract()?;
        let slice = get_slice_dyn(&py_arr)?;
        let res = ema_generic(py, slice, period, smoothing as f64)?;
        let obj = res.into_pyobject(py)?;
        Ok(obj.into_any().unbind())
    } else if dtype.eq(np.getattr("float32")?)? {
        let py_arr: PyReadonlyArray<'_, f32, IxDyn> = resolved.extract()?;
        let slice = get_slice_dyn(&py_arr)?;
        let res = ema_generic(py, slice, period, smoothing)?;
        let obj = res.into_pyobject(py)?;
        Ok(obj.into_any().unbind())
    } else {
        Err(PyValueError::new_err("Input array must be float32 or float64"))
    }
}

#[pyfunction]
fn rsi<'py>(py: Python<'py>, prices: &Bound<'py, PyAny>, period: usize) -> PyResult<PyObject> {
    dispatch_unary!(py, prices, rsi_generic, period)
}

#[pyfunction]
fn macd<'py>(
    py: Python<'py>,
    price: &Bound<'py, PyAny>,
    period_fast: usize,
    period_slow: usize,
    period_signal: usize,
) -> PyResult<PyObject> {
    dispatch_unary!(py, price, macd_generic, period_fast, period_slow, period_signal)
}

#[pyfunction]
fn roc<'py>(py: Python<'py>, price: &Bound<'py, PyAny>, period: usize) -> PyResult<PyObject> {
    dispatch_unary!(py, price, roc_generic, period)
}

#[pyfunction]
fn atr<'py>(
    py: Python<'py>,
    high: &Bound<'py, PyAny>,
    low: &Bound<'py, PyAny>,
    close: &Bound<'py, PyAny>,
    period: usize,
) -> PyResult<PyObject> {
    dispatch_multi_3!(py, high, low, close, atr_generic, period)
}

#[pyfunction]
fn cmf<'py>(
    py: Python<'py>,
    high: &Bound<'py, PyAny>,
    low: &Bound<'py, PyAny>,
    close: &Bound<'py, PyAny>,
    volume: &Bound<'py, PyAny>,
    period: usize,
) -> PyResult<PyObject> {
    dispatch_multi_4!(py, high, low, close, volume, cmf_generic, period)
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
