use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use numpy::ndarray::prelude::*;

//Helper functions for indicators
fn sma_helper(price_ndarray: &Array1<f32>, period: usize) -> Array1<f32>{
    let length = price_ndarray.len() - period +1;
    let mut result = Array1::<f32>::zeros(length);

    for i in 0..length {
        let slice = price_ndarray.slice(s![i..i+period]);
        result[i] = slice.sum()/(period as f32);
    }
    
    result
}

fn ema_helper(price_ndarray: &Array1<f32>, period: usize) -> Array1<f32>{
    let length = price_ndarray.len() - period +1;
    let mut result = Array1::<f32>::zeros(length);
    result[0] = price_ndarray.slice(s![0..period]).sum();
    for i in 1..length{
        result[i] = result[i-1]+(price_ndarray[i+period-1]-result[i-1])*(2.0/((period as f32)+1.0));
    }
    
    result
}

//Indicator Python function wrappers
#[pyfunction]
fn sma(price: Vec<f32>, period: usize) -> PyResult<Vec<f32>> {
    let price_ndarray = Array::from_vec(price);
    let length = price_ndarray.len() - period +1;
    let mut result = Array1::<f32>::zeros(length);

    for i in 0..length {
        let slice = price_ndarray.slice(s![i..i+period]);
        result[i] = slice.sum()/(period as f32);
    }

    Ok(Array::to_vec(&result))
}

#[pyfunction]
fn ema(price: Vec<f32>, period: usize, smoothing: f32) -> PyResult<Vec<f32>> {
    let price_ndarray = Array::from_vec(price);
    let length = price_ndarray.len();
    let mut result = Array1::<f32>::zeros(length);
    
    let weight = smoothing / (period + 1) as f32;
    result[0] = price_ndarray[0];

    for i in 1..length {
        result[i] = (price_ndarray[i]*weight) + (result[i-1] * (1.0-weight));
    }

    Ok(Array::to_vec(&result))
}

#[pyfunction]
fn rsi(price: Vec<f32>, period: usize) -> PyResult<Vec<f32>>{
    let price_ndarray = Array::from_vec(price);
    let mut change = Array1::<f32>::zeros(price_ndarray.len());
    let mut gain = Array1::<f32>::zeros(price_ndarray.len());
    let mut loss = Array1::<f32>::zeros(price_ndarray.len());
    let mut ag = Array1::<f32>::zeros(price_ndarray.len());
    let mut al = Array1::<f32>::zeros(price_ndarray.len());
    let mut result = Array1::<f32>::zeros(price_ndarray.len());
    for i in 1..price_ndarray.len(){
        change[i] = price_ndarray[i] - price_ndarray[i-1];
        if change[i] == 0.0{
            gain[i] = 0.0;
            loss[i] = 0.0;
        } else if change[i]<0.0{
            gain[i] = 0.0;
            loss[i] = (change[i]).abs();
        }else{
            gain[i] = change[i];
            loss[i] = 0.0;
        }
    }
    ag[period] = gain.slice(s![1..period+1]).sum()/(period as f32);
    al[period] = loss.slice(s![1..period+1]).sum()/(period as f32);
    
    for i in period+1..price_ndarray.len(){
        ag[i] = (ag[i-1]*(period as f32-1.0)+gain[i])/period as f32;
        al[i] = (al[i-1]*(period as f32-1.0)+loss[i])/period as f32;
    }
    for i in period+1..price_ndarray.len(){
        result[i] = 100.0-(100.0/(1.0+ag[i]/al[i]))
    }

    //result.slice(s![period..]).to_owned()
    Ok(Array::to_vec(&(result.slice(s![period..]).to_owned())))
}

#[pyfunction]
fn macd(price: Vec<f32>, period_fast: usize, period_slow: usize, period_signal: usize) -> PyResult<(Vec<f32>,Vec<f32>)>{
    let price_ndarray = Array::from_vec(price);
    let line = ema_helper(&price_ndarray, period_fast).slice(s![period_slow-period_fast..]).to_owned() - ema_helper(&price_ndarray,period_slow);
    let signal = ema_helper(&price_ndarray, period_signal);

    Ok((Array::to_vec(&line), Array::to_vec(&signal)))
}

#[pyfunction]
fn roc(price: Vec<f32>, period: usize) -> PyResult<Vec<f32>> {
    let price_ndarray = Array::from_vec(price);
    let length = price_ndarray.len() - period;
    let mut result = Array1::<f32>::zeros(length);

    for i in period..price_ndarray.len() {
        result[i-period] = ((price_ndarray[i]-price_ndarray[i-period])/price_ndarray[i-period])*100.0;
    }

    Ok(Array::to_vec(&result))
}

#[pyfunction]
fn atr(high: Vec<f32>, low: Vec<f32>, close: Vec<f32>, period: usize) -> PyResult<Vec<f32>> {
    let length = high.len();
    let high_ndarray = Array::from_vec(high);
    let low_ndarray = Array::from_vec(low);
    let close_ndarray = Array::from_vec(close);
    let mut tr  = Array1::<f32>::zeros(length);
    let mut result = Array1::<f32>::zeros(length - period + 1);
    
    tr[0] = high_ndarray[0]-low_ndarray[0];
    for i in 1..high_ndarray.len() {
        let hl = high_ndarray[i] - low_ndarray[i];
        let hpc = (high_ndarray[i] - close_ndarray[i-1]).abs();
        let lpc = (low_ndarray[i] - close_ndarray[i-1]).abs();
        tr[i] = f32::max(f32::max(hl, hpc), lpc);
    }


    result[0] = tr.slice(s![0..period]).sum()/period as f32;

    for i in 1.. length -period+1 {
        result[i] = (result[i-1]*(period as f32-1.0)+tr[i+period-1])/period as f32;
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

    let a = (&close_ndarray)-(&low_ndarray);
    let b = (&high_ndarray)-(&close_ndarray);
    let c = (&high_ndarray)-(&low_ndarray);

    let mfv = ((a-b)/c)*period as f32;
    for i in 0..length-period+1 {
        result[i] =  mfv.slice(s![i..i+period]).sum()/ volume_ndarray.slice(s![i..i+period]).sum();
    }

    Ok(Array::to_vec(&result))
}

#[pymodule]
fn ZenithTA(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sma, m)?)?;
    m.add_function(wrap_pyfunction!(cmf, m)?)?;
    m.add_function(wrap_pyfunction!(atr, m)?)?;
    m.add_function(wrap_pyfunction!(ema, m)?)?;
    m.add_function(wrap_pyfunction!(rsi, m)?)?;
    m.add_function(wrap_pyfunction!(roc, m)?)?;
    m.add_function(wrap_pyfunction!(macd, m)?)?;
    Ok(())
}

