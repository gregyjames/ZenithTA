import pandas_datareader as pdr
import pandas as pd
from ZenithTA import *
from timeit import default_timer as timer
from datetime import timedelta

data = pdr.get_data_yahoo('NVDA')

print("Timing ZenithTA:")
a = data['Close'].tolist()
start = timer()
j = sma(a,5)
end = timer()
print(timedelta(seconds=end-start))

print("Timing Pandas:")
start = timer()
data['SMA(5)'] = data.Close.rolling(5).mean()
end = timer()
b = data['SMA(5)'].tolist()
print(timedelta(seconds=end-start))
