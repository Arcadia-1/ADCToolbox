# ADCToolbox Testing Guide

Detailed reference for writing and running tests in the ADCToolbox Python package.

---

## Writing a Unit Test

Pattern: **generate → process → assert**.

```python
import numpy as np
from adctoolbox import analyze_spectrum, find_coherent_frequency

def test_analyze_spectrum_known_snr():
    """ENOB should match theoretical SNR for known noise level."""
    N = 8192
    Fs = 800e6
    Fin, _ = find_coherent_frequency(Fs, 100e6, N)

    # Generate signal with known parameters
    t = np.arange(N) / Fs
    signal = 0.49 * np.sin(2 * np.pi * Fin * t) + 0.5
    noise_rms = 50e-6
    signal_noisy = signal + np.random.randn(N) * noise_rms

    # Run function under test
    result = analyze_spectrum(signal_noisy, fs=Fs, create_plot=False)

    # Assert against expectations
    assert result['enob'] > 12.0, f"ENOB too low: {result['enob']}"
    assert result['sndr_dbc'] > 70, f"SNDR too low: {result['sndr_dbc']}"
```

Key rules:
- Always pass `create_plot=False` in unit tests (no GUI dependency)
- Use `find_coherent_frequency` for signal generation (avoid spectral leakage)
- Test the **public API** (`from adctoolbox import ...`), not internal submodules
- Use absolute tolerances for floating-point comparisons: `np.testing.assert_allclose(a, b, atol=1e-6)`

---

## Writing an Integration Test

Pattern: **load data → process → save outputs**.

```python
import pytest
from pathlib import Path
from adctoolbox import analyze_spectrum
from tests._utils import save_fig, auto_search_files
from tests import config

def test_analyze_spectrum_real_data(project_root):
    files = auto_search_files(project_root, config.AOUT)
    for filepath in files:
        data = np.loadtxt(filepath, delimiter=',')
        result = analyze_spectrum(data, fs=800e6, create_plot=True)
        save_fig(project_root, 'test_analyze_spectrum', filepath.stem)
        assert result['enob'] > 0  # sanity check
```

---

## Writing a Comparison Test

Pattern: **load MATLAB ref → run Python → compare**.

Uses the comparison framework in `tests/compare/`:
- `_name_mapping.py` maps MATLAB variable names to Python dict keys
- `_comparator.py` performs element-wise comparison with tolerance
- `_runner.py` orchestrates load → run → compare

Tolerance: **1e-6** relative error for all numerical outputs.

---

## Shared Fixtures and Helpers

**`conftest.py`** provides:
- `project_root` — resolves to repo root for locating datasets

**`_utils.py`** provides:
- `save_fig(root, test_name, suffix)` — saves plot to `test_output/`
- `save_variable(root, test_name, var_name, data)` — saves CSV for MATLAB comparison
- `auto_search_files(root, config_entry)` — discovers test datasets by glob pattern

**`config.py`** defines dataset locations:
- `AOUT`: `reference_dataset/sinewave_*.csv`
- `DOUT`: `reference_dataset/dout_*.csv`
- `JITTER`: `test_dataset/jitter_sweep/jitter_sweep_*.csv`

---

## Skipped Comparison Tests

Four comparison tests are currently skipped due to known MATLAB/Python differences:

| Test | Reason |
|------|--------|
| `test_compare_analyze_spectrum` | MATLAB doesn't use sideBin for harmonics; Python does |
| `test_compare_inl_sine` | Minor numerical differences in INL calculation |
| `test_compare_enob_bit_sweep` | Minor numerical differences |
| `test_compare_err_envelope_spectrum` | Minor numerical differences in Hilbert transform |

---

## Test Coverage Gaps

### Functions with NO dedicated tests (need unit tests)

**Unit conversions (16 functions — none tested):**
- `db_to_mag`, `mag_to_db`, `db_to_power`, `power_to_db`
- `snr_to_enob`, `enob_to_snr`
- `lsb_to_volts`, `volts_to_lsb`
- `bin_to_freq`, `freq_to_bin`
- `dbm_to_vrms`, `vrms_to_dbm`, `dbm_to_mw`, `mw_to_dbm`
- `sine_amplitude_to_power`, `fold_bin_to_nyquist`

Note: `snr_to_nsd`, `nsd_to_snr`, and `amplitudes_to_snr` are imported in existing
tests (`test_nsd_snr_conversions.py`, spectrum tests) so have partial coverage.

**FOM metrics (3 functions — only `calculate_jitter_limit` tested indirectly):**
- `calculate_walden_fom`
- `calculate_schreier_fom`
- `calculate_thermal_noise_limit`

**Frequency utilities (1 function):**
- `estimate_frequency` — used internally by calibration and sine fitting, but no
  dedicated test exercises it through the public API.

### Functions now covered by unit tests (previously gaps)

- `analyze_weight_radix` — `tests/unit/dout/test_analyze_weight_radix.py` (6 tests)
- `plot_residual_scatter` — `tests/unit/dout/test_plot_residual_scatter.py` (7 tests)
- `sweep_performance_vs_osr` — `tests/unit/spectrum/test_sweep_performance_vs_osr.py` (6 tests)

### Integration tests using legacy imports (not public API)

All 19 integration test files import from internal submodules (`adctoolbox.dout`,
`adctoolbox.aout`, `adctoolbox.common`) using legacy function names rather than the
public `from adctoolbox import ...` API. They work correctly but don't validate the
public export path. Examples:

- `test_bit_activity.py` → `from adctoolbox.dout import check_bit_activity` (legacy name)
- `test_overflow_chk.py` → `from adctoolbox.dout import check_overflow` (legacy name)
- `test_weight_scaling.py` → `from adctoolbox.dout import plot_weight_radix` (legacy name)
- `test_sine_fit.py` → `from adctoolbox.common import ...` (legacy module)

---

## Test Output

All test artifacts go to `test_output/` (gitignored):
- PNG figures: `test_output/<test_name>/<dataset>_python.png`
- CSV variables: `test_output/<test_name>/<variable>_python.csv`

These outputs enable visual verification and MATLAB cross-comparison.
