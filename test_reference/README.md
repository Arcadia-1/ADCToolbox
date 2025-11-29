# Golden Reference Data

Golden reference outputs for validating Python against MATLAB implementation.

## Purpose

Since CI cannot run MATLAB, we commit MATLAB outputs here. CI runs Python and compares against these references.

- **Parity test**: Python matches MATLAB (cross-platform validation)
- **Regression test**: Python output hasn't changed (detects bugs)

## Structure

```
test_reference/
  ├── golden_data_list.txt           # Datasets to process (5-10 files)
  └── <dataset>/
      └── <test>/
          ├── freq_matlab.csv
          ├── freq_python.csv
          └── *.png
```

## Golden Data List

Edit `golden_data_list.txt` to control which datasets are tested. Keep it small (5-10 files) for fast CI.

## Update Golden References

```matlab
% 1. MATLAB
cd matlab/tests/generate_golden_reference
run_matlab_tests
```

```bash
# 2. Python
cd python/tests/generate_golden_reference
python run_python_tests.py

# 3. Verify
python ../../tests/validation/compare_parity.py

# 4. Commit
git add test_reference/
git commit -m "Update golden references"
```

## CI Regression Tests

CI runs Python tests and compares against golden references:
```bash
python tests/validation/compare_regression.py
```

## Comparison Scripts

```bash
python tests/validation/compare_parity.py      # MATLAB vs Python (validates implementation)
python tests/validation/compare_regression.py  # Python vs golden (detects regressions)
```

## When to Regenerate

- ✅ Fixing bugs or improving algorithms
- ✅ Adding new datasets to `golden_data_list.txt`
- ❌ Random code changes (that's what regression tests catch)
