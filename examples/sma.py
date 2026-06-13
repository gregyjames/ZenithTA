import yfinance as yf
import pandas as pd
from ZenithTA import *
from timeit import default_timer as timer
from datetime import timedelta

data = yf.download("NVDA", start="2023-01-01", end="2024-01-01")

print("Timing ZenithTA:")
a = data["Close"].to_numpy()
start = timer()
j = sma(a, 5)
end = timer()
print(timedelta(seconds=end - start))

print("Timing Pandas:")
start = timer()
data["SMA(5)"] = data.Close.rolling(5).mean()
end = timer()
b = data["SMA(5)"].to_numpy()
print(timedelta(seconds=end - start))
