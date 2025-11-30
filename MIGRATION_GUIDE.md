# Migration Guide: Three-Tier Data Structure

## Changes in v0.2.0

### New Features

1. **Examples now in pip package**
   - `python/examples/` → `python/src/adctoolbox/examples/`
   - **Impact:** Examples work immediately after pip install
   - **Action:** None (pip users can now run examples directly)

2. **Example data included**
   - 5 small datasets (~740 KB) included in package
   - **Impact:** Examples self-contained, no external data needed
   - **Action:** None (automatic with pip install)

### Non-Breaking Changes

- Full dataset (`dataset/`) stays in git repository (unchanged)
- Test files unchanged (use `dataset/` path)
- CI golden tests unchanged (use `test_reference/`)
- All public APIs unchanged

## Migration Steps

### For Pip Users

Upgrade and start using examples:

```bash
pip install --upgrade adctoolbox
python -m adctoolbox.examples.quickstart.basic_workflow
```

Or access example data programmatically:

```python
from adctoolbox.examples.data import get_example_data_path
import numpy as np

# Load packaged example data
data_path = get_example_data_path('sinewave_jitter_400fs.csv')
signal = np.loadtxt(data_path, delimiter=',')
```

### For Developers

No migration needed. Continue development as before:

```bash
git pull
# Unit tests continue to use dataset/ (unchanged)
# Examples now also available in package
```

## Backward Compatibility

The package maintains 100% backward compatibility:
- All public APIs unchanged
- Test suite unchanged
- Dataset paths unchanged
- Only addition: examples in package

## What Changed Under the Hood

### File Reorganization
- Examples moved from `python/examples/` to `python/src/adctoolbox/examples/`
- 5 example datasets copied into package at `adctoolbox/examples/data/`
- Full dataset (`dataset/`) remains in git, excluded from pip package

### Package Configuration
- `MANIFEST.in` updated to include examples
- `pyproject.toml` updated with package_data for CSV files
- Package size increased from ~500 KB to ~1.2 MB (due to example data)

### Example Scripts
- All examples now demonstrate loading packaged data first
- Educational synthetic data examples preserved
- New data access utilities at `adctoolbox.examples.data`

## Benefits

**For Pip Users:**
- ✅ Examples work out-of-the-box after installation
- ✅ No need to visit GitHub for basic examples
- ✅ Self-contained tutorials with real data

**For Developers:**
- ✅ No breaking changes
- ✅ Test suite unchanged
- ✅ Development workflow unaffected

## Questions?

Open an issue at: https://github.com/Arcadia-1/ADCToolbox/issues
