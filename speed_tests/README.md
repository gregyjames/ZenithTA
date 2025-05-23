# ZenithTA Speed Tests

This directory contains performance tests for the ZenithTA technical analysis library. The tests compare our Rust implementation against pandas to verify both correctness and performance.

## Prerequisites

- Python 3.13 or later
- [uv](https://github.com/astral-sh/uv) package manager
- Rust toolchain (for building the native extension)

## Setup

1. Create and activate a virtual environment using uv:
```bash
uv venv .venv
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows
```

2. Install dependencies:
```bash
uv pip install -r requirements.txt
```

## Running Tests

### Using build.sh

The `build.sh` script handles building the Rust extension and running the tests in one command:

```bash
./build.sh
```

This script will:
1. Build the Rust extension
2. Install it in development mode
3. Run all tests with pytest

### Manual Testing

If you prefer to run tests manually:

1. Build and install the extension:
```bash
maturin develop
```

2. Run specific test files:
```bash
python -m pytest test_rsi.py -v  # Run RSI tests
python -m pytest test_atr.py -v  # Run ATR tests
python -m pytest -v  # Run all tests
```

## Test Structure

- `test_rsi.py`: Tests for Relative Strength Index implementation
- `test_atr.py`: Tests for Average True Range implementation
- Each test file includes:
  - Basic functionality tests
  - Edge cases
  - Performance comparisons with pandas
  - Varying period tests

## Performance Metrics

The tests measure:
- Correctness (matching pandas output)
- Execution time comparison
- Memory usage
- Speedup ratio vs pandas

## Troubleshooting

If you encounter build errors:
1. Ensure Rust toolchain is up to date: `rustup update`
2. Clean build artifacts: `cargo clean`
3. Rebuild: `maturin develop --release`

For test failures:
1. Check Python version matches requirements
2. Verify pandas version compatibility
3. Ensure virtual environment is activated

## Contributing

When adding new tests:
1. Follow the existing test structure
2. Include both correctness and performance tests
3. Add appropriate edge cases
4. Document any special test requirements
