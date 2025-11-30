# Golden Reference Data

Golden reference outputs for validating Python against MATLAB implementation.

## Purpose

This directory contains MATLAB golden reference outputs. Since CI cannot run MATLAB, we commit MATLAB outputs here and CI compares Python against them.

- **Parity test**: Compares MATLAB vs Python outputs in test_output/ (validates Python matches MATLAB)
- **Regression test**: Compares MATLAB golden (test_reference/) vs Python current (test_output/) - detects regressions in Python code

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
% 1. Run MATLAB golden tests (outputs to test_output/)
cd matlab/tests/generate_golden_reference
run_matlab_tests

% 2. Copy MATLAB outputs to test_reference/
copy_to_golden
```

```bash
# 3. Run Python golden tests (outputs to test_output/)
cd python/tests/generate_golden_reference
python run_python_tests.py

# 4. Copy Python outputs to test_reference/
python copy_to_golden.py

# 5. Verify parity (MATLAB vs Python in test_output/)
cd ../validation
python compare_parity.py

# 6. Commit updated golden references
cd ../../..
git add test_reference/
git commit -m "Update golden references"
```

## CI Regression Tests

CI runs Python golden tests and compares against MATLAB golden references:
```bash
# 1. Generate Python outputs (to test_output/)
python python/tests/generate_golden_reference/run_python_tests.py

# 2. Compare Python current vs MATLAB golden
python python/tests/validation/compare_regression.py
```

**What this validates:**
- Python implementation matches MATLAB golden reference (the source of truth)
- No regressions introduced in Python code

## Comparison Scripts

**Parity test** - Validates Python matches MATLAB in test_output/:
```bash
python python/tests/validation/compare_parity.py
# Compares: test_output/*_matlab.csv vs test_output/*_python.csv
```

**Regression test** - Detects changes in Python vs MATLAB golden:
```bash
python python/tests/validation/compare_regression.py
# Compares: test_reference/*_matlab.csv vs test_output/*_python.csv
```

## When to Regenerate

- ✅ Fixing bugs or improving algorithms
- ✅ Adding new datasets to `golden_data_list.txt`
- ❌ Random code changes (that's what regression tests catch)
