import pandas_datareader as pdr
import pandas as pd
from ZenithTA import *
from timeit import default_timer as timer
from datetime import timedelta

data = pdr.get_data_yahoo('NVDA')

print("Timing ZenithTA:")
start = timer()
a = data['Close'].tolist()
j = ema(a,4,2.0)
end = timer()
print(j[0:10:1])
print(timedelta(seconds=end-start))

print("Timing Pandas:")
start = timer()
data['4dayEWM'] = data['Close'].ewm(span=4, adjust=False).mean()
b = data['4dayEWM'].tolist()
print(b[0:10:1])
end = timer()
print(timedelta(seconds=end-start))