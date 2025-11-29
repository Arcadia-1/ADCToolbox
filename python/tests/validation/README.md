# Validation Scripts

Scripts for validating Python implementation against MATLAB and detecting regressions.

## Scripts

### compare_parity.py
Compare MATLAB vs Python outputs in `test_output/`

**Purpose:** Validates that Python implementation matches MATLAB reference

**Usage:**
```bash
python tests/validation/compare_parity.py
```

### compare_regression.py
Compare Python `test_output/` vs `test_reference/` (golden references)

**Purpose:** Detects unintended changes in Python code (regression testing)

**Usage:**
```bash
python tests/validation/compare_regression.py
```

### compare_all_results.py
Core comparison logic used by the above scripts

**Not called directly** - imported by compare_parity.py and compare_regression.py
