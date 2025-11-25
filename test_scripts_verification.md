# Python Test Scripts Verification

## Status: ✓ ALL PASS

All 11 Python test scripts have been verified and can be imported successfully.

## Test Scripts Checked

### Unit Tests (10)
1. ✓ `adctoolbox.test.unit.test_alias` - Frequency aliasing tests
2. ✓ `adctoolbox.test.unit.test_cap2weight` - Capacitor weight calculation tests
3. ✓ `adctoolbox.test.unit.test_sineFit` - Sine wave fitting tests
4. ✓ `adctoolbox.test.unit.test_specPlot` - Spectrum plot tests
5. ✓ `adctoolbox.test.unit.test_specPlotPhase` - Spectrum with phase tests
6. ✓ `adctoolbox.test.unit.test_INLSine` - INL/DNL from sine wave tests
7. ✓ `adctoolbox.test.unit.test_FGCalSine` - Foreground calibration tests
8. ✓ `adctoolbox.test.unit.test_FGCalSine_overflowChk` - Overflow check tests
9. ✓ `adctoolbox.test.unit.test_error_analysis` - Error analysis suite
10. ✓ `adctoolbox.test.unit.compare_sineFit_results` - MATLAB vs Python comparison

### System Tests (1)
11. ✓ `adctoolbox.test.system.test_multimodal_report` - Multi-modal error report

## Fixes Applied

### 1. Import Path Updates
- Updated `test_multimodal_report.py` to use new package structure
- Changed from `ADC_Toolbox_Python.multimodal_report` to `adctoolbox.utils.multimodal_report`

### 2. Matplotlib Memory Warning Fix
Fixed "More than 20 figures" warning by:
- **Plotting functions**: Only create figure if none exists
  - `spec_plot.py`
  - `tomDecomp.py`
  - `phase_polar_plot.py`
- **Test scripts**: Added `plt.close('all')` cleanup
  - `test_error_analysis.py`
  - `test_specPlot.py`
  - `test_specPlotPhase.py`
  - `test_INLSine.py`
  - `test_FGCalSine_overflowChk.py`

### 3. MATLAB sineFit.m Bugs Fixed
- **Line 111**: Removed incorrect transpose `(A*cos(theta)+B*sin(theta)+dc)'`
- **Line 101**: Fixed variable name `A(:,end)` → `M(:,end)`
- Now returns column vectors (N,1) instead of row vectors (1,N)

## How to Run Tests

### Check All Imports
```bash
python adctoolbox/test/check_all_tests.py
```

### Run Individual Tests
```bash
# Unit tests
python adctoolbox/test/unit/test_alias.py
python adctoolbox/test/unit/test_sineFit.py
python adctoolbox/test/unit/test_error_analysis.py
# etc...

# System tests
python adctoolbox/test/system/test_multimodal_report.py
```

### Run All Tests
```bash
python adctoolbox/test/run_all_tests.py
```

## Next Steps

To run actual test execution (not just imports):
1. Ensure test data is in `test_data/` directory
2. Run MATLAB tests first to generate baseline results
3. Run Python tests and compare with MATLAB outputs
4. Use comparison scripts to verify accuracy

---
Last verified: 2025-11-25
