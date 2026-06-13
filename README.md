[![Rust](https://img.shields.io/github/actions/workflow/status/gregyjames/zenithta/rust.yml?style=for-the-badge)](https://github.com/gregyjames/ZenithTA/actions/workflows/rust.yml)
[![PyPI](https://img.shields.io/pypi/v/zenithta?color=%230s0&style=for-the-badge)](https://pypi.org/project/zenithta/)
[![License](https://img.shields.io/github/license/gregyjames/zenithta?color=%230sd&style=for-the-badge)](https://github.com/gregyjames/ZenithTA/blob/main/LICENSE)
[![Count](https://img.shields.io/tokei/lines/github/gregyjames/zenithta?color=%230fs&style=for-the-badge)](https://github.com/gregyjames/ZenithTA/)



# ZenithTA
A ultra-efficient, zero-allocation, high-performance technical analysis library written in Rust using PyO3 and rust-numpy.

## Features & Ergonomics
- **Zero-Allocation & Zero-Copy Fast-Paths**: Directly borrows 1D and 2D contiguous arrays, Pandas Series, and DataFrame columns without copying or memory allocations.
- **Dynamic Type Dispatch**: Seamlessly accepts and runs on both `float32` and `float64` input data.
- **SIMD Optimized**: Compiles targeting host CPU native instruction sets (`target-cpu=native`).
- **Bypassed Bounds Checking**: Hot calculation loops bypass bounds checks via unsafe unchecked accesses under strict safety constraints.

## Indicators
- **ATR** (Average True Range)
- **CMF** (Chaikin Money Flow)
- **SMA** (Simple Moving Average)
- **EMA** (Exponential Moving Average)
- **RSI** (Relative Strength Index)
- **MACD** (Moving Average Convergence Divergence)
- **ROC** (Rate of Change)

## How to install
```bash
pip install zenithta
```

## How to build locally
We use `uv` and `maturin` to manage, build, and run the library. To build the extension in release mode and install it in your environment:
```bash
# Rebuild and install the local package
uv sync --reinstall-package ZenithTA
```

Usage in Python:
```python
import numpy as np
import pandas as pd
from ZenithTA import sma

# Works out of the box with zero-copy on various formats:
prices = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0])
result = sma(prices, period=3)
```

## Speed
ZenithTA calculations are heavily optimized and can be significantly faster than standard Pandas/NumPy operations due to compiled SIMD instructions, unchecked loop arithmetic, and lack of layout copying/allocation overhead. Run `uv run python examples/sma.py` to see the performance benchmarks on your system.

## License
MIT License

Copyright (c) 2022 Greg James

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
