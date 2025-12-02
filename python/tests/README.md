# ADCToolbox Test Suite

Comprehensive test suite for validating Python implementation against MATLAB reference.

## Structure

```
tests/
├── unit/              # Unit tests - run Python implementations
├── compare/           # Comparison tests - verify Python matches MATLAB
├── _utils.py          # Shared utilities (save_variable, save_fig)
└── conftest.py        # Pytest fixtures (project_root)
```

## Quick Start

```bash
# Run all tests
pytest python/tests/ -v

# Run unit tests only
pytest python/tests/unit/ -v

# Run comparison tests only
pytest python/tests/compare/ -v

# Run specific test
pytest python/tests/unit/test_sine_fit.py -v
```

## Unit Tests (`unit/`)

**Purpose:** Run Python implementations and generate outputs for validation.

- **20 test files** covering all major functions
- Processes datasets from `dataset/aout/` and `dataset/dout/`
- Outputs saved to `test_output/<dataset>/<test_name>/`
- Generates both CSV data and PNG plots

**Key tests:**
- `test_sine_fit.py` - Sine wave parameter estimation
- `test_spec_plot.py` - FFT spectrum analysis
- `test_fg_cal_sine.py` - Foreground calibration
- `test_err_hist_sine_*.py` - Error histogram analysis
- `test_bit_activity.py` - Digital bit activity

**How it works:**
1. Loads raw data from `dataset/`
2. Runs Python implementation
3. Saves results (CSV + PNG) to `test_output/`
4. Uses `_runner.py` for batch processing

## Comparison Tests (`compare/`)

**Purpose:** Validate Python outputs match MATLAB golden references.

- **11 comparison test files**
- Compares Python outputs vs MATLAB golden references
- MATLAB references stored in `test_reference/`
- Uses CSV numerical comparison with tolerance checks

**Key tests:**
- `test_compare_sine_fit.py` - Validates sine fitting accuracy
- `test_compare_spec_plot.py` - Validates spectrum metrics
- `test_compare_fg_cal_sine.py` - Validates calibration results
- Analog tests: `run_analog_comparisons.py`
- Digital tests: `run_digital_comparisons.py`

**How it works:**
1. Reads MATLAB reference from `test_reference/`
2. Reads Python output from `test_output/`
3. Compares using `csv_comparator.py`
4. Reports differences with tolerance grading

**Tolerance grading:**
- `PERFECT`: < 1e-10% difference
- `EXCELLENT`: < 0.01% difference
- `GOOD`: < 0.1% difference
- `ACCEPTABLE`: < 1.0% difference
- `NEEDS REVIEW`: ≥ 1.0% difference

## Key Utilities

### `_utils.py`
- `save_variable()` - Save numpy arrays to CSV (MATLAB-compatible format)
- `save_fig()` - Save matplotlib figures to PNG
- `auto_search_files()` - Auto-discover test datasets
- `discover_test_datasets()` - Find datasets with specific test outputs

### `compare/csv_comparator.py`
- `CSVComparator` - Compares MATLAB vs Python CSV files
- Handles numerical precision, near-zero artifacts, shape mismatches
- Generates detailed comparison reports

### `compare/_runner.py`
- `run_comparison_suite()` - Generic comparison test runner
- Supports nested (dataset/test) and flat (test only) structures
- Automated variable discovery and comparison

### `unit/_runner.py`
- `run_unit_test_batch()` - Generic unit test runner
- Batch processes multiple datasets
- Handles file I/O and output organization

## CI Integration

GitHub Actions runs subset of tests on every commit:

```yaml
# .github/workflows/ci.yml
- Test basic functionality
- Test sine_fit (unit + compare)
```

**Status:** ✅ All CI tests passing

## Development Workflow

### Adding a new test

**1. Create unit test:**
```python
# tests/unit/test_my_function.py
from adctoolbox.my_module import my_function
from tests.unit._runner import run_unit_test_batch

def _process_my_function(raw_data, sub_folder, dataset_name):
    result = my_function(raw_data)
    save_variable(sub_folder, result, 'result')
    # Generate plots if needed

def test_my_function(project_root):
    run_unit_test_batch(
        project_root=project_root,
        input_subpath="dataset/aout/sinewave",
        test_module_name="test_my_function",
        file_pattern="sinewave_*.csv",
        output_subpath="test_output",
        process_func=_process_my_function
    )
```

**2. Generate MATLAB golden reference:**
```matlab
% Run MATLAB implementation
% Save outputs to test_reference/
```

**3. Create comparison test:**
```python
# tests/compare/test_compare_my_function.py
from tests.compare._runner import run_comparison_suite

def test_compare_my_function(project_root):
    run_comparison_suite(
        project_root,
        matlab_test_name="test_myFunction",
        ref_folder="test_reference",
        out_folder="test_output",
        structure="nested"
    )
```

**4. Run tests:**
```bash
pytest python/tests/unit/test_my_function.py -v
pytest python/tests/compare/test_compare_my_function.py -v
```

## Test Data Organization

```
ADCToolbox/
├── dataset/              # Input data (full dataset)
│   ├── aout/            # Analog ADC outputs
│   └── dout/            # Digital ADC outputs
├── test_reference/       # MATLAB golden references (committed)
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

## Test Coverage

| Package | Unit Tests | Compare Tests |
|---------|-----------|---------------|
| `common` | 2 | 2 |
| `aout` | 11 | 6 |
| `dout` | 7 | 3 |
| **Total** | **20** | **11** |

## Notes

- Unit tests generate outputs, comparison tests validate them
- MATLAB golden references are source of truth
- All tests use pytest fixtures for project path resolution
- Test outputs are saved in MATLAB-compatible CSV format
- Name mapping handles MATLAB camelCase ↔ Python snake_case conversion
