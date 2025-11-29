# Generate Python Golden References

Generate golden reference data from Python tests.

## Usage

```bash
cd python/tests/generate_golden_reference
python run_python_tests.py
```

Outputs saved to `test_reference/<dataset>/<test>/`

## Add Tests

Edit `run_python_tests.py`:
```python
from golden_sine_fit import golden_sineFit
from test_alias import test_alias
from golden_my_new_test import golden_myNewTest  # Add this

golden_sineFit()
test_alias()
golden_myNewTest()  # Add this
```
