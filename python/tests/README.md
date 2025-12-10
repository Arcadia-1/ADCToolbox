# ADCToolbox Test Suite

Comprehensive test suite for validating Python implementation with 49 pytest-compatible tests.

## Structure

```
tests/
├── unit/              # Unit tests (5 tests) - Synthetic verification
├── integration/       # Integration tests (23 tests) - Process real datasets
├── compare/           # Comparison tests (21 tests) - Python vs MATLAB validation
├── _utils.py          # Shared utilities (save_variable, save_fig)
├── conftest.py        # Pytest fixtures (project_root)
└── README.md          # This file
```

## Quick Start

```bash
# Navigate to python directory
cd python

# Run all 49 tests
pytest tests/ -v

# Run by category
pytest tests/unit/ -v         # 5 unit tests (synthetic verification)
pytest tests/integration/ -v  # 23 integration tests (real data processing)
pytest tests/compare/ -v      # 21 comparison tests (MATLAB validation)

# Run specific test
pytest tests/unit/test_verify_alias.py -v
pytest tests/integration/test_sine_fit.py -v
pytest tests/compare/test_compare_sine_fit.py -v
```

## Test Categories

### 1. Unit Tests (`unit/`) - 5 Tests

**Purpose:** Self-verification with synthetic data (NO MATLAB comparison needed)

Tests verify that functions work correctly on synthetic signals with known ground truth:

- ✅ `test_verify_alias.py` (3 tests) - Frequency aliasing across Nyquist zones
- ✅ `test_verify_jitter.py` (1 test) - Jitter measurement accuracy
- ✅ `test_verify_spec_plot.py` (2 tests) - Quantization noise vs bit depth
- ✅ `test_verify_spec_plot_phase.py` (3 tests) - Phase spectrum FFT vs LMS modes
- ✅ `test_cap2weight.py` (1 test) - Capacitor to weight conversion

**How they work:**
1. Generate synthetic signal with known parameters (e.g., jitter = 100 fs)
2. Run Python function
3. Assert measured value matches set value (within tolerance)
4. **No MATLAB needed** - Self-contained verification

**Example:**
```python
def test_verify_jitter_single_point():
    # Generate signal with 100 fs jitter
    jitter_set = 100e-15
    signal = generate_signal_with_jitter(jitter_set)

    # Measure jitter
    jitter_measured = measure_jitter(signal)

    # Verify (within 5% tolerance)
    assert abs(jitter_measured - jitter_set) / jitter_set < 0.05
```

### 2. Integration Tests (`integration/`) - 23 Tests

**Purpose:** Run Python implementations on real datasets and generate outputs

Tests process real ADC measurement data from `dataset/` folder:

**Analog Output Tests (13 tests):**
- `test_basic.py` - Basic sine fit and spectrum
- `test_sine_fit.py` - Sine wave parameter estimation
- `test_spec_plot.py` - FFT spectrum analysis
- `test_spec_plot_phase_fft.py` / `test_spec_plot_phase_lms.py` - Phase analysis
- `test_tom_decomp.py` - Thompson decomposition
- `test_err_hist_sine_code.py` / `test_err_hist_sine_phase.py` - Error histograms
- `test_err_pdf.py` - Error probability density
- `test_err_auto_correlation.py` - Autocorrelation analysis
- `test_err_envelope_spectrum.py` / `test_err_spectrum.py` - Error spectra
- `test_fit_static_nol.py` - Static nonlinearity extraction
- `test_inl_sine.py` - INL/DNL from sine wave

**Digital Output Tests (10 tests):**
- `test_bit_activity.py` - Bit toggle rate analysis
- `test_enob_bit_sweep.py` - ENoB vs bits used
- `test_fg_cal_sine.py` - Foreground calibration
- `test_fg_cal_sine_overflow_chk.py` - Calibration + overflow check
- `test_overflow_chk.py` - Overflow detection
- `test_weight_scaling.py` - Weight/radix visualization

**How they work:**
1. Load real ADC data from `dataset/aout/` or `dataset/dout/`
2. Run Python implementation
3. Save results (CSV + PNG) to `test_output/<dataset>/<test_name>/`
4. Outputs can be compared against MATLAB using comparison tests

**File locations:**
- **Input**: `dataset/aout/` (analog) or `dataset/dout/` (digital)
- **Output**: `test_output/<dataset_name>/<test_name>/`

### 3. Comparison Tests (`compare/`) - 21 Tests

**Purpose:** Validate Python outputs match MATLAB golden references

Tests compare Python vs MATLAB numerical results using CSV comparison:

**Comparison tests:**
- `test_compare_sine_fit.py` - Sine fitting parameters
- `test_compare_spec_plot.py` - Spectrum metrics
- `test_compare_fg_cal_sine.py` - Calibration results
- `test_compare_bit_activity.py` - Bit activity patterns
- Plus 17 more tests for all major functions

**Tolerance threshold:** 1e-6 absolute difference

**File locations:**
- **MATLAB reference**: `test_reference/<dataset>/<test_name>/*_matlab.csv`
- **Python output**: `test_output/<dataset>/<test_name>/*_python.csv`

**Run consolidated comparison:**
```bash
cd python
python -m tests.compare.run_all_comparisons
```

This generates a detailed log at `test_comparison_logs/all_comparisons_<timestamp>.txt`

## Test Statistics

| Category | Tests | Description |
|----------|-------|-------------|
| **Unit** | 5 | Synthetic verification (no MATLAB needed) |
| **Integration** | 23 | Process real datasets |
| **Comparison** | 21 | Python vs MATLAB validation |
| **TOTAL** | **49** | All pytest-compatible ✅ |

## Running Tests

### Option 1: pytest (Recommended)

```bash
cd python

# Run all tests
pytest tests/ -v

# Run specific category
pytest tests/unit/ -v
pytest tests/integration/ -v
pytest tests/compare/ -v

# Run single test
pytest tests/unit/test_verify_alias.py::test_verify_alias_basic -v

# Show test collection without running
pytest tests/ --collect-only
```

### Option 2: Python runner scripts

```bash
cd python

# Run all integration tests (legacy)
python tests/run_full_test_suite.py

# Run all comparison tests with consolidated log
python -m tests.compare.run_all_comparisons
```

## Requirements for Tests

```bash
# Core dependencies (from pyproject.toml)
pip install numpy matplotlib scipy

# Test framework
pip install pytest

# Install package in editable mode
pip install -e .
```

## CI Integration

GitHub Actions runs a subset of tests on every commit:

```yaml
# .github/workflows/ci.yml
- Test Basic Examples (b01-b04)
```

**Status:** ✅ All CI tests passing

## Test Data Organization

```
ADCToolbox/
├── dataset/              # Input data (real ADC measurements)
│   ├── aout/            # Analog outputs (sinewaves)
│   └── dout/            # Digital outputs (bit codes)
├── test_reference/       # MATLAB golden references
│   └── <dataset>/
│       └── <test_name>/
│           ├── freq_matlab.csv
│           └── *.png
└── test_output/          # Python test outputs (gitignored)
    └── <dataset>/
        └── <test_name>/
            ├── freq_python.csv
            └── *.png
```

## Quick Command Reference

```bash
# Navigate to python directory first!
cd python

# Run all 49 tests
pytest tests/ -v

# Run by category
pytest tests/unit/ -v         # 5 unit tests
pytest tests/integration/ -v  # 23 integration tests
pytest tests/compare/ -v      # 21 comparison tests

# Run single test
pytest tests/unit/test_verify_alias.py -v

# Run specific test function
pytest tests/unit/test_verify_alias.py::test_verify_alias_basic -v

# Collect tests without running
pytest tests/ --collect-only

# Run with detailed output
pytest tests/unit/ -vv

# Run and stop on first failure
pytest tests/ -x

# Run comparison tests with consolidated log
python -m tests.compare.run_all_comparisons
```

## Common Issues

### ImportError: No module named 'adctoolbox'

**Solution**: Install package in editable mode
```bash
cd python
pip install -e .
```

### Tests not found

**Solution**: Run pytest from `python/` directory, not project root
```bash
cd python  # IMPORTANT!
pytest tests/ -v
```

### MATLAB comparison fails: "Reference file not found"

**Solution**: Integration tests must run before comparison tests
```bash
# Step 1: Run integration test (generates Python CSV)
pytest tests/integration/test_sine_fit.py

# Step 2: Run comparison test (compares Python vs MATLAB)
pytest tests/compare/test_compare_sine_fit.py
```

## Contributing

When adding a new function:

1. **Write unit test** (if synthetic verification possible)
2. **Write integration test** (process real data)
3. **Generate MATLAB reference** (run MATLAB test)
4. **Write comparison test** (validate Python vs MATLAB)
5. **Run all tests** to verify

See `CONTRIBUTING.md` for detailed guidelines.

## Summary

- ✅ **49 pytest-compatible tests** (all categories)
- ✅ **5 unit tests** - Self-verification with synthetic data
- ✅ **23 integration tests** - Process real ADC datasets
- ✅ **21 comparison tests** - Validate against MATLAB
- ✅ **All tests runnable with pytest** from `python/` directory
- ✅ **CI integration** - Automated testing on every commit

Happy testing! 🚀
