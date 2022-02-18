use numpy::ndarray::prelude::*;

fn sma(prices: &Array1<f32>, period: usize) -> Array1<f32> {
    let length = prices.len() - period +1;
    let mut result = Array1::<f32>::zeros(length);

    for i in 0..length {
        let slice = prices.slice(s![i..i+period]);
        result[i] = slice.sum()/(period as f32);
    }

    result
}

fn ema(prices: &Array1<f32>, period: usize) -> Array1<f32>{
    let length = prices.len() - period +1;
    let mut result = Array1::<f32>::zeros(length);
    result[0] = prices.slice(s![0..period]).sum();
    for i in 1..result.len(){
        result[i] = result[i-1]+(prices[i+period-1]-result[i-1])*(2.0/((period as f32)+1.0));
    }
    result
}

fn rsi(prices: &Array1<f32>, period: usize) -> Array1<f32>{
    let mut change = Array1::<f32>::zeros(prices.len());
    let mut gain = Array1::<f32>::zeros(prices.len());
    let mut loss = Array1::<f32>::zeros(prices.len());
    let mut ag = Array1::<f32>::zeros(prices.len());
    let mut al = Array1::<f32>::zeros(prices.len());
    let mut result = Array1::<f32>::zeros(prices.len());
    for i in 1..prices.len(){
        change[i] = prices[i] - prices[i-1];
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
    
    for i in period+1..prices.len(){
        ag[i] = (ag[i-1]*(period as f32-1.0)+gain[i])/period as f32;
        al[i] = (al[i-1]*(period as f32-1.0)+loss[i])/period as f32;
    }
    for i in period+1..prices.len(){
        result[i] = 100.0-(100.0/(1.0+ag[i]/al[i]))
    }
    result.slice(s![period..]).to_owned()
}

fn macd(prices: &Array1<f32>, period_fast: usize, period_slow: usize, period_signal: usize) -> (Array1<f32>,Array1<f32>){
    let line = ema(prices, period_fast).slice(s![period_slow-period_fast..]).to_owned() - ema(prices,period_slow);
    let signal = ema(&line, period_signal);

    return (line, signal)
}
