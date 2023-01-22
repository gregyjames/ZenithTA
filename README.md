[![Rust](https://img.shields.io/github/actions/workflow/status/gregyjames/zenithta/rust.yml?style=for-the-badge)](https://github.com/gregyjames/ZenithTA/actions/workflows/rust.yml)
[![PyPI](https://img.shields.io/pypi/v/zenithta?color=%230s0&style=for-the-badge)](https://pypi.org/project/zenithta/)
[![License](https://img.shields.io/github/license/gregyjames/zenithta?color=%230sd&style=for-the-badge)](https://github.com/gregyjames/ZenithTA/blob/main/LICENSE)
[![Count](https://img.shields.io/tokei/lines/github/gregyjames/zenithta?color=%230fs&style=for-the-badge)](https://github.com/gregyjames/ZenithTA/)



# ZenithTA
#### Formerly Panther
A efficient, high-performance python technical analysis library written in Rust using PyO3 and rust-numpy. 

## Indicators
- ATR
- CMF
- SMA
- EMA
- RSI
- MACD
- ROC

## How to install
`pip3 install zenithta`

## How to build (Windows)
- Run `cargo build --release` from the main directory.
- Get the generated dll from the target/release directory.
- Rename extension from .dll to .pyd.
- Place .pyd file in the same folder as script. 
- Put `from panther import *` in python script.
 
## Speed
On average, I found the Panther calculations of these indicators to be about 9x or 900% faster than the industry standard way of calculating these indicators using Pandas. Don't believe me? Install the library and run the tests in the speed_tests directory to see it for yourself :)

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
