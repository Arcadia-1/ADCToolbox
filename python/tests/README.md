# ADCToolbox Test Suite

Complete test suite for the ADCToolbox Python package.

---

## Quick Start

```bash
cd d:\ADCToolbox

# Run all tests (recommended)
python tests/run_all_tests.py

# Run tests by package
python tests/system/run_unit_tests_common.py   # Common package
python tests/system/run_unit_tests_aout.py     # Aout package
python tests/system/run_unit_tests_dout.py     # Dout package

# Run comparison validation
python tests/system/compare_all.py
```

Results saved to `test_output/TEST_REPORT.txt`

---

## Directory Structure

```
tests/
├── run_all_tests.py              # Top-level test runner
├── check_all_tests.py            # Verify all imports work
├── system/                        # Test runners & comparison tools
│   ├── run_unit_tests_*.py       # Package test runners
│   ├── compare_*.py               # CSV comparison tools
│   └── universal_csv_compare.py  # Main comparison CLI
├── unit/                          # Individual test scripts
│   ├── test_*.py                  # Core tests (9 files)
│   └── test_err*.py               # Standalone developer tools (7 files)
├── utils/                         # Shared utilities
│   └── csv_comparator.py         # Comparison engine
└── archive/                       # Deprecated files
```

---

## Test Organization

### Integrated Tests (Included in `run_all_tests.py`)

**Common Package:**
- `test_alias.py` - Frequency aliasing
- `test_sineFit.py` - Sine wave fitting

**Aout Package:**
- `test_specPlot.py` - Spectrum analysis
- `test_specPlotPhase.py` - Phase spectrum
- `test_INLSine.py` - INL/DNL extraction

**Dout Package:**
- `test_FGCalSine.py` - Foreground calibration
- `test_FGCalSine_overflowChk.py` - Overflow detection
- `test_cap2weight.py` - Capacitor weights

### Standalone Tests (Developer Tools Only)

Run individually for detailed debugging:
- `test_errPDF.py` - Error PDF
- `test_errAutoCorrelation.py` - Autocorrelation
- `test_errEnvelopeSpectrum.py` - Envelope spectrum
- `test_errSpectrum.py` - Error spectrum
- `test_errHistSine.py` - Error histogram
- `test_tomDecomp.py` - Thompson decomposition
- `test_jitter_load.py` - Jitter analysis

**Usage:**
```bash
python tests/unit/test_errPDF.py
```

---

## Comparison Tools

**Recommended (current system):**
```bash
python tests/system/compare_all.py                           # All packages
python tests/system/universal_csv_compare.py                 # CLI with filters
python tests/system/compare_common.py                        # Common package only
```

**Deprecated (don't use):**
- ❌ `unit/compare_all_csv_pairs.py`
- ❌ `unit/analyze_comparison.py`
- ❌ `unit/compare_sineFit_results.py`

---

## Adding New Tests

1. Create `unit/test_<function>.py`
2. Add to `system/run_unit_tests_<package>.py`
3. Optionally add to `check_all_tests.py` for import validation

---

## Path Convention

All scripts run from project root `d:\ADCToolbox`:
- Input: `test_data/`
- Output: `test_output/`

---

## More Information

- MATLAB equivalents: `matlab/test/`
- Test results: `test_output/TEST_REPORT.txt`
- Project log: `project_log.md` (in project root)
