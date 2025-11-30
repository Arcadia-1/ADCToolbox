# test_basic Implementation Summary

## Overview

Implemented Python equivalent of MATLAB `test_basic.m` with two components:
1. **User example** - Verify Python environment setup
2. **CI test** - Compare Python output with MATLAB golden reference

---

## Files Created

### 1. Python Example (for users)
**File:** `python/src/adctoolbox/examples/quickstart/example_01_basic_test.py`

**Purpose:** Users can run this to verify their Python environment:
- numpy works (array operations)
- matplotlib works (plotting)
- File paths work (I/O)

**How to run:**
```bash
python -m adctoolbox.examples.quickstart.example_01_basic_test
```

**Output:**
- Console: PASS/FAIL status for numpy, matplotlib, paths
- Figure: `basic_test_sinewave.png` (full + zoomed sine wave)
- Data: `sinewave_python.csv` (first 1000 samples)

### 2. CI Test (for automated testing)
**File:** `python/tests/test_basic.py`

**Purpose:** CI automatically compares Python output with MATLAB golden reference

**How to run:**
```bash
cd python
python tests/test_basic.py
```

**What it does:**
1. Generates sine wave (same config as MATLAB)
2. Saves to `test_output/test_basic/sinewave_python.csv`
3. Loads MATLAB golden: `test_output/test_basic/sinewave_matlab.csv`
4. Compares data:
   - If diff < 1e-5: PASS (same RNG)
   - If diff > 1e-5: Check sine wave parameters (different RNG acceptable)
5. Prints detailed comparison report

---

## Test Configuration

Matches MATLAB `test_basic.m` exactly:

| Parameter | Value |
|-----------|-------|
| N (samples) | 1024 |
| Fs (Hz) | 1000 |
| Fin (Hz) | 99 |
| Amplitude | 0.49 |
| DC offset | 0.5 |
| Noise level | 1e-6 |
| Saved samples | 1000 (truncated by saveVariable) |

---

## CI Integration

**File:** `.github/workflows/ci.yml`

Added new step to `python-tests` job:
```yaml
- name: Run basic infrastructure test
  run: |
    export PYTHONPATH="${GITHUB_WORKSPACE}/python/src:${PYTHONPATH}"
    cd python
    python tests/test_basic.py
```

This runs after the existing smoke tests.

---

## Test Results

**Current status:** ✅ PASSED

```
[Comparison]
  Python shape: (1000,)
  MATLAB shape: (1000,)
  Max difference:  5.98e-06
  Mean difference: 1.14e-06
[PASS] Difference < 1e-05
```

The small difference (< 1e-5) is due to:
- Floating point precision
- Slightly different random number generators (Python uses seed(42), MATLAB may use different seed)

Since the difference is acceptable, the test passes.

---

## Directory Structure

```
ADCToolbox/
├── matlab/
│   └── tests/unit/test_basic.m              # Original MATLAB test
│
├── python/
│   ├── src/adctoolbox/examples/quickstart/
│   │   └── example_01_basic_test.py         # User example (NEW)
│   │
│   └── tests/
│       └── test_basic.py                    # CI test (NEW)
│
├── test_output/test_basic/                  # Test outputs
│   ├── sinewave_matlab.csv                  # MATLAB golden (existing)
│   └── sinewave_python.csv                  # Python output (generated)
│
└── .github/workflows/ci.yml                 # CI config (UPDATED)
```

---

## Key Design Decisions

### 1. Random Number Generation
- Python uses `np.random.seed(42)` for reproducibility
- MATLAB may use different RNG, causing small differences
- Test accepts differences < 1e-5 OR checks sine wave parameters match

### 2. Data Truncation
- Matches MATLAB's `saveVariable()` behavior (truncates to 1000 samples)
- Both save 1000 samples even though 1024 are generated

### 3. Output Locations
- User example: `examples/output/` (user-facing)
- CI test: `test_output/test_basic/` (compares with MATLAB)

### 4. Console Output Format
- Uses bracket labels: `[PASS]`, `[Comparison]`, etc.
- No unicode symbols (Windows console compatibility)
- Prints full absolute paths for saved files

---

## Usage for Users

**Verify Python environment:**
```bash
python -m adctoolbox.examples.quickstart.example_01_basic_test
```

**Expected output:**
```
======================================================================
ADCToolbox Basic Setup Test
======================================================================

[Output directory] D:\ADCToolbox\python\src\adctoolbox\examples\output

[Configuration] N=1024, Fs=1000.0 Hz, Fin=99 Hz, A=0.49, DC=0.5
[Generated sinewave] Shape=(1024,), Range=[0.009999, 0.989999]
  [save]->[D:\...\basic_test_sinewave.png]
  [save]->[D:\...\sinewave_python.csv]

======================================================================
Basic Setup Test - Results
======================================================================
[PASS] numpy: Array operations working
[PASS] matplotlib: Plotting working
[PASS] Path handling: File I/O working

[Files saved]
  - Figure: basic_test_sinewave.png
  - Data:   sinewave_python.csv (first 1000 samples)

Your Python environment is ready for ADCToolbox!
======================================================================
```

---

## Next Steps

1. ✅ Python example created and tested
2. ✅ CI test created and passing
3. ✅ CI workflow updated
4. ⏭️ Add CLI entry point (optional): `adctoolbox-test-basic`
5. ⏭️ Copy to test_reference/ (optional, if we want centralized golden references)
