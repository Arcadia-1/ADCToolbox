# ADCToolbox

Toolbox for ADC data analysis and processing.

## Structure

- `matlab/` - MATLAB toolbox implementation
  - `src/` - Source code
  - `toolbox/` - Toolbox packaging files
- `python/` - Python implementation
  - `src/` - Source code modules
    - `aout/` - Analog output analysis tools
    - `common/` - Common utilities
    - `dout/` - Digital output analysis tools
    - `os/` - Oversampling analysis
    - `utils/` - Utility functions
  - `tests/` - Test suite
  - `examples/` - Example scripts
- `doc/` - Documentation
- `dataset/` - Datasets and test data

## Getting Started

### MATLAB

See `matlab/README.md` for MATLAB toolbox usage.

### Python

The Python implementation provides a complete port of the MATLAB ADC analysis tools.

**Installation:**

```bash
cd python
pip install -e .
```

**Quick Start:**

```python
from adctoolbox.src.aout import spec_plot, sineFit
from adctoolbox.src.common import alias, findBin

# Example: Spectrum analysis
results = spec_plot(data, fs, fin)
```

**Running Tests:**

```bash
cd python
python tests/run_all_tests.py
```

For more details, see:
- `python/src/README` - Detailed module documentation and testing status
- `python/tests/README.md` - Test suite documentation

## License



