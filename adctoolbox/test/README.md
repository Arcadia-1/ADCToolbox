# ADCToolbox Test Suite

This directory contains the complete test suite for the ADCToolbox Python package, structured to match the MATLAB test hierarchy.

## Test Structure

The test suite follows a 3-level hierarchy similar to the MATLAB implementation:

```
adctoolbox/test/
├── run_all_tests.py              # Top-level test runner (generates report)
├── system/                        # System tests (run unit tests by package)
│   ├── run_unit_tests_all.py     # Run all package tests
│   ├── run_unit_tests_common.py  # Run common package tests
│   ├── run_unit_tests_aout.py    # Run aout package tests
│   └── run_unit_tests_dout.py    # Run dout package tests
└── unit/                          # Unit tests (individual function tests)
    ├── test_alias.py
    ├── test_sineFit.py
    ├── test_specPlot.py
    ├── test_specPlotPhase.py
    ├── test_error_analysis.py
    ├── test_INLSine.py
    ├── test_FGCalSine.py
    ├── test_FGCalSine_overflowChk.py
    └── test_cap2weight.py
```

## Running Tests

All tests should be run from the project root directory: `d:\ADCToolbox`

### Run All Tests (Recommended)

```bash
cd d:\ADCToolbox
python adctoolbox/test/run_all_tests.py
```

This generates a comprehensive report at `test_output/TEST_REPORT.txt`

### Run Tests by Package

**Common Package** (alias, sineFit):
```bash
python adctoolbox/test/system/run_unit_tests_common.py
```

**Aout Package** (analog output analysis):
```bash
python adctoolbox/test/system/run_unit_tests_aout.py
```

**Dout Package** (digital output analysis):
```bash
python adctoolbox/test/system/run_unit_tests_dout.py
```

**All Packages**:
```bash
python adctoolbox/test/system/run_unit_tests_all.py
```

### Run Individual Unit Tests

```bash
python adctoolbox/test/unit/test_specPlot.py
python adctoolbox/test/unit/test_FGCalSine.py
# etc...
```

## Test Hierarchy Explained

### Level 1: Top-Level Runner (`run_all_tests.py`)
- Executes the complete test suite
- Generates comprehensive report with timing and statistics
- Saves report to `test_output/TEST_REPORT.txt`

### Level 2: System Tests (`system/run_unit_tests_*.py`)
- Organize tests by package (common, aout, dout)
- Run multiple related unit tests
- Provide package-level summaries

### Level 3: Unit Tests (`unit/test_*.py`)
- Test individual functions
- Generate output files and comparison data
- Auto-discover test datasets from `test_data/`

## Package Organization

### Common Package
Functions for basic ADC analysis:
- **test_alias.py** - Frequency aliasing calculations
- **test_sineFit.py** - Sine wave fitting

### Aout Package (Analog Output Analysis)
Functions for analyzing analog ADC output:
- **test_specPlot.py** - Spectrum analysis (ENoB, SNDR, SFDR, SNR, THD)
- **test_specPlotPhase.py** - Phase spectrum polar plots
- **test_error_analysis.py** - Error analysis (PDF, autocorrelation, envelope spectrum)
- **test_INLSine.py** - INL/DNL extraction from sine waves

### Dout Package (Digital Output Analysis)
Functions for analyzing digital ADC codes:
- **test_FGCalSine.py** - Foreground calibration for SAR ADCs
- **test_FGCalSine_overflowChk.py** - Calibration with overflow detection
- **test_cap2weight.py** - Capacitor to weight conversion

## Test Output

All test output is saved to `test_output/` with the following structure:

```
test_output/
├── TEST_REPORT.txt                                    # Comprehensive test report
└── <dataset_name>/                                    # Per-dataset results
    ├── test_specPlot/
    │   ├── spectrum_python.png
    │   └── metrics_python.csv
    ├── test_specPlotPhase/
    │   ├── phase_python.png
    │   └── phase_data_python.csv
    └── test_INLSine/
        └── INL_12b_<dataset_name>_python.png
```

## Test Report Contents

The comprehensive test report (`TEST_REPORT.txt`) includes:

1. **Test Hierarchy** - Visual representation of test structure
2. **Summary Statistics** - Pass/fail counts, success rate, execution time
3. **Package Test Results** - Individual package results with timing
4. **Complete Test Output** - Full output from all tests
5. **Error Output** - Detailed error messages (if any failures)

## Example Test Report

```
================================================================================
Summary Statistics
================================================================================
[Packages tested]      = 3
[Packages passed]      = 3
[Packages failed]      = 0
[Success rate]         = 100.0%
[Total execution time] = 86.9 seconds

================================================================================
Package Test Results
================================================================================
  [PASS] Common Package                 (   6.3s)
  [PASS] Aout Package                   (  67.4s)
  [PASS] Dout Package                   (  13.2s)

================================================================================
ALL TESTS PASSED!
================================================================================
```

## Path Convention

All test scripts assume execution from project root `d:\ADCToolbox`:
- Input data: `test_data/`
- Output data: `test_output/`
- No complex path calculations needed

## Comparison with MATLAB

This Python test structure matches the MATLAB test hierarchy:

| MATLAB | Python |
|--------|--------|
| `matlab/test/system/run_unit_tests_all.m` | `adctoolbox/test/system/run_unit_tests_all.py` |
| `matlab/test/system/run_unit_tests_common.m` | `adctoolbox/test/system/run_unit_tests_common.py` |
| `matlab/test/system/run_unit_tests_aout.m` | `adctoolbox/test/system/run_unit_tests_aout.py` |
| `matlab/test/system/run_unit_tests_dout.m` | `adctoolbox/test/system/run_unit_tests_dout.py` |
| `matlab/test/unit/test_*.m` | `adctoolbox/test/unit/test_*.py` |

## Maintenance

When adding new tests:

1. Create unit test in `unit/test_<function>.py`
2. Add test to appropriate package runner in `system/run_unit_tests_<package>.py`
3. Test will automatically be included when running `run_all_tests.py`

No changes needed to top-level runner or report generation.
