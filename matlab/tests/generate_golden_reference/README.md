# Generate MATLAB Golden References

Generate golden reference data from MATLAB tests.

## Usage

```matlab
cd matlab/tests/generate_golden_reference
run_matlab_tests
```

Outputs saved to `test_reference/<dataset>/<test>/`

## Add Tests

Edit `run_matlab_tests.m`:
```matlab
golden_sineFit
test_alias
golden_myNewTest  % Add this
```
