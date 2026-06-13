import yfinance as yf
import pandas as pd
from ZenithTA import ema
from timeit import default_timer as timer
from datetime import timedelta

# Download data using yfinance
data = yf.download("NVDA", start="2023-01-01", end="2024-01-01")

print("Timing ZenithTA:")
start = timer()
a = data["Close"].to_numpy().flatten().astype("float32")
j = ema(a, 4, 2.0)
end = timer()
print(j[0:10:1])
print(timedelta(seconds=end - start))

print("Timing Pandas:")
start = timer()
data["4dayEWM"] = data["Close"].ewm(span=4, adjust=False).mean()
b = data["4dayEWM"].to_numpy()
print(b[0:10:1])
end = timer()
print(timedelta(seconds=end - start))
