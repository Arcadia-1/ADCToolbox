---
name: adctoolbox-contributor-guide
description: >
  ADCToolbox coding style and naming conventions for contributors. Covers both Python
  and MATLAB codebases: file naming, function naming, variable naming, parameter style,
  return conventions, docstrings, type annotations, imports, private modules, test naming,
  and legacy wrapper patterns. Use this skill when a contributor asks about naming rules,
  code style, how to name a new function or file, how to structure a new module, or
  before reviewing/writing code for consistency.
---

# ADCToolbox Code Style Guide

This guide defines the naming and style conventions for both the Python and MATLAB
codebases. All contributors should follow these rules for consistency.

---

## 1. Python Conventions

### 1.1 File & Module Naming

**Pattern:** `<verb>_<noun>_<modifier>.py`, all snake_case.

Verb prefixes carry meaning:

| Prefix | Purpose | Example |
|--------|---------|---------|
| `analyze_` | Compute metrics + optional plot | `analyze_spectrum.py` |
| `compute_` | Pure computation, no plotting | `compute_spectrum.py` |
| `plot_` | Visualization only | `plot_spectrum.py` |
| `fit_` | Curve fitting | `fit_sine_4param.py` |
| `calibrate_` | Calibration algorithm | `calibrate_weight_sine.py` |
| `generate_` | Dashboard / output generation | `generate_aout_dashboard.py` |
| `extract_` | Signal extraction | `extract_freq_components.py` |

**Private helper modules** use a `_` prefix and are never exported:

```
spectrum/
  analyze_spectrum.py        # public
  compute_spectrum.py        # public
  _window.py                 # private helper
  _harmonics.py              # private helper
  _estimate_noise_power.py   # private helper
```

### 1.2 Function Naming

- **Public functions:** snake_case, match their filename.
  `analyze_spectrum()`, `fit_sine_4param()`, `calibrate_weight_sine()`

- **Private functions:** prefixed with `_`, short descriptive names.
  `_fit_core()`, `_estimate_frequency_fft()`, `_merge_results()`

- **No classes** in the core library (except `ADC_Signal_Generator`). The codebase
  is function-based — algorithms are stateless transformations.

### 1.3 The `_export()` Registry

All public functions are flat-exported via the registry in `__init__.py`:

```python
def _export(name, obj):
    globals()[name] = obj
    __all__.append(name)

from .fundamentals import fit_sine_4param
_export('fit_sine_4param', fit_sine_4param)
```

Users import directly: `from adctoolbox import analyze_spectrum`. When adding a new
public function, always register it with `_export()`.

### 1.4 Parameter Naming

Full descriptive snake_case names. Never abbreviate beyond standard terms.

```python
def calibrate_weight_sine(
    bits: np.ndarray | list[np.ndarray],
    freq: float | np.ndarray | None = None,
    nominal_weights: np.ndarray | None = None,
    harmonic_order: int = 1,
    learning_rate: float = 0.5,
    reltol: float = 1e-12,
    max_iter: int = 100,
    verbose: int = 0
) -> dict:
```

Common parameter names used project-wide:

| Parameter | Meaning | Type |
|-----------|---------|------|
| `data` | Input signal | `np.ndarray` |
| `fs` | Sampling frequency (Hz) | `float` |
| `freq` | Normalized frequency (Fin/Fs, 0–0.5) | `float` |
| `create_plot` | Whether to generate a plot | `bool` |
| `ax` | Matplotlib axes to plot into | `plt.Axes \| None` |
| `win_type` | Window function name | `str` |
| `osr` | Oversampling ratio | `int` |
| `max_harmonic` | Harmonics to include | `int` |
| `verbose` | Verbosity level (0=silent) | `int` |
| `max_iter` | Maximum iterations | `int` |
| `resolution` or `num_bits` | ADC bit count | `int` |

### 1.5 Return Conventions

**Always return `dict` with snake_case keys** — never tuples for new functions.

```python
return {
    'fitted_signal': fitted_sig,
    'residuals': residuals,
    'frequency': freq,
    'amplitude': np.sqrt(a**2 + b**2),
    'phase': np.arctan2(-b, a),
    'dc_offset': c,
    'rmse': np.sqrt(np.mean(residuals**2))
}
```

A few legacy functions still return tuples or arrays — do not add more:
`find_coherent_frequency`, `analyze_bit_activity`, `analyze_overflow`,
`analyze_weight_radix`, `analyze_enob_sweep`, `fit_static_nonlin`.

### 1.6 Variable Naming

**Math/loop variables** — short single-letter names are fine:
`t` (time), `n` (length), `k` (bin), `A`/`B` (cos/sin coefficients),
`i`/`j` (loop index), `x`/`y` (general arrays).

**Descriptive variables** — full snake_case words:
`dc_offset`, `fitted_signal`, `design_matrix`, `cos_vec`, `sin_vec`,
`noise_floor`, `signal_power`.

### 1.7 Type Annotations

Use modern Python 3.10+ union syntax. Always annotate public function signatures.

```python
def fit_sine_4param(
    data: np.ndarray,
    frequency: float | None = None,
    max_iterations: int = 1,
    tolerance: float = 1e-7,
) -> dict:
```

- Use `|` not `Union[]`
- Use `list[...]` not `List[...]`
- Use `tuple[...]` not `Tuple[...]`
- Use `| None` not `Optional[...]`

### 1.8 Docstrings

NumPy-style with these sections: summary, Parameters, Returns, Examples (optional),
Notes (optional).

```python
def _reshape_and_stack_input(
    bits_input: np.ndarray | list[np.ndarray],
    verbose: int = 0
) -> tuple[np.ndarray, np.ndarray, int, list[np.ndarray]]:
    """Normalize input data: validate, transpose, and concatenate multi-dataset.

    Parameters
    ----------
    bits_input : ndarray or list of ndarray
        Single bits matrix or list of bits matrices.
    verbose : int, optional
        Verbosity level (default 0).

    Returns
    -------
    bits_stacked : ndarray
        Concatenated bits matrix (Ntot x bit_width).
    segment_lengths : ndarray
        Number of samples per dataset.
    ...

    Raises
    ------
    ValueError
        If datasets are empty or have inconsistent bitwidths.
    """
```

### 1.9 Imports

```python
# Standard library
import os

# Third-party (always aliased)
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import windows

# Internal — relative imports within package
from adctoolbox.spectrum._harmonics import _locate_harmonic_bins
from adctoolbox.fundamentals.frequency import estimate_frequency
```

### 1.10 Constants

Module-level constants use `_UPPER_SNAKE_CASE` (private) or `UPPER_SNAKE_CASE` (public):

```python
_SIDE_BIN_DEFAULTS = {
    'rectangular': {'enbw': 1.00, 'coherent': 0, 'non_coherent': 10},
    'hann':        {'enbw': 1.50, 'coherent': 1, 'non_coherent': 10},
}
```

Inline physics constants use descriptive comments:

```python
k = 1.38e-23   # Boltzmann constant (J/K)
T = 300         # Temperature (K)
```

### 1.11 Test Naming

**Files:** `test_verify_<function_name>.py` or `test_<topic>.py`

```
tests/unit/fundamentals/test_verify_fit_sine_4param.py
tests/unit/spectrum/test_verify_analyze_spectrum.py
tests/unit/test_nsd_snr_conversions.py
```

**Functions:** `test_<scenario>()` in snake_case:

```python
def test_verify_fit_sine_4param_clean_signal():
def test_verify_fit_sine_4param_noisy_signal():
def test_verify_fit_sine_4param_2d_input():
```

**Directory structure:**

```
tests/
  unit/           # Synthetic data, no external deps
  integration/    # Real ADC datasets from dataset/
  compare/        # Python vs MATLAB parity (tolerance 1e-6)
```

---

## 2. MATLAB Conventions

### 2.1 File & Function Naming

**Pattern:** Lowercase, short, no underscores. One function per file.

```
sinfit.m        findfreq.m      findbin.m
plotspec.m      plotwgt.m        plotphase.m
wcalsin.m       tomdec.m         errsin.m
bitchk.m        ntfperf.m        cdacwgt.m
adcpanel.m      perfosr.m        ifilter.m
```

Names are abbreviations: `wcalsin` = weight calibration sine, `tomdec` = time-domain
decomposition, `plotwgt` = plot weights, `bitchk` = bit check.

### 2.2 Variable Naming

Very concise, math-oriented. Capital `N` and `M` for dimensions.

```matlab
[N, M] = size(sig);          % N = samples, M = columns/bits
time = (0:N-1)';             % time vector
theta = 2*pi*f0*time;        % phase vector
A = x(1);  B = x(2);        % cos/sin coefficients
dc = x(3);                   % DC offset
mag = sqrt(A^2 + B^2);      % amplitude
phi = atan2(-B, A);          % phase
freq = f0;                   % normalized frequency
```

Compound names use camelCase for multi-word locals:

```matlab
freqCal       % calibrated frequency
theta_mat     % phase matrix
bits_patch    % patched bit matrix
digitalCodes  % reconstructed output codes
outputDir     % output directory path
```

### 2.3 Parameter Style (inputParser)

Name-Value arguments, lowercase or camelCase:

```matlab
addOptional(p, 'freq', 0)           % normalized frequency
addOptional(p, 'rate', 0.5)         % learning rate
addOptional(p, 'reltol', 1E-12)     % relative tolerance
addOptional(p, 'niter', 100)        % max iterations
addOptional(p, 'fsearch', 0)        % force frequency search
addOptional(p, 'verbose', 0)        % verbosity
addOptional(p, 'autotrans', 1)      % auto-transpose
addOptional(p, 'autopatch', 1)      % auto rank-deficiency patch
addOptional(p, 'nomWeight', [])     % nominal weights
addOptional(p, 'OSR', 1)            % oversampling ratio (CAPS = acronym)
addOptional(p, 'NFMethod', 'auto')  % noise floor method
addOptional(p, 'sideBin', 'auto')   % side bins around signal
addOptional(p, 'dispItem', 'all')   % display items
```

### 2.4 Output Naming

Multiple return values, all lowercase short names:

```matlab
function [fitout, freq, mag, dc, phi] = sinfit(sig, varargin)
function [weight, offset, postcal, ideal, err, freqcal] = wcalsin(bits, varargin)
```

### 2.5 Documentation Style

Cell-style block comments with `%FUNCTION_NAME` header:

```matlab
%SINFIT Four-parameter iterative sine wave fitting
%   [fitout, freq, mag, dc, phi] = SINFIT(sig)
%
%   Inputs:
%     sig - Input signal to be fitted
%       Vector (row or column) or Matrix (averaged across columns)
%
%   Outputs:
%     fitout - Fitted sine wave signal (column vector)
%     freq   - Normalized frequency (0 to 0.5)
%
%   Name-Value Arguments:
%     'niter'   - Maximum iterations (default: 100)
%     'fsearch' - Force frequency search (default: 0)
%
%   Algorithm:
%     1. Initial 3-parameter fit using linear least squares
%     2. Iterative frequency refinement via gradient descent
```

Section separators use `% ====` or `% ----` comment lines:

```matlab
% ==========================
% Multi-dataset (cell) path
% ==========================
```

### 2.6 Legacy Wrappers

Old camelCase names live in `matlab/src/legacy/` and forward to the new name:

```matlab
% legacy/sineFit.m
function [data_fit,freq,mag,dc,phi] = sineFit(data,f0,tol,rate)
%SINEFIT (legacy) — Please use sinfit() instead.
    if nargin == 1
        [data_fit,freq,mag,dc,phi] = sinfit(data);
    elseif nargin == 2
        [data_fit,freq,mag,dc,phi] = sinfit(data,f0);
    ...
```

**When renaming a MATLAB function:**
1. Create the new lowercase `.m` file in `matlab/src/`
2. Move the old file to `matlab/src/legacy/`
3. Replace the old file's body with a forwarding call
4. Add a deprecation note in the old file's docstring

### 2.7 Shortcuts

Convenience wrappers live in `matlab/src/shortcut/`:

```matlab
% shortcut/errsinv.m — shortcut for errsin with xaxis='value'
function errsinv(varargin)
    errsin(varargin{:}, 'xaxis', 'value');
end
```

---

## 3. Cross-Language Name Mapping

When the same algorithm exists in both languages:

| MATLAB | Python | Notes |
|--------|--------|-------|
| `sinfit` | `fit_sine_4param` | Python name is more descriptive |
| `plotspec` | `analyze_spectrum` / `compute_spectrum` / `plot_spectrum` | Python splits into 3 functions |
| `wcalsin` | `calibrate_weight_sine` | Python uses full words |
| `tomdec` | `decompose_harmonic_error` | Python uses full words |
| `findfreq` | `estimate_frequency` | Python uses verb prefix |
| `findbin` | `find_coherent_frequency` | Python returns both freq and bin |
| `plotwgt` | `analyze_weight_radix` | Python returns data, plot separate |
| `errsin` | `analyze_error_by_phase` / `analyze_error_by_value` | Python splits by axis mode |
| `bitchk` | `analyze_overflow` | Python uses old MATLAB name |
| `inlsin` | `compute_inl_from_sine` | Python uses full words |
| `ntfperf` | `ntf_analyzer` | Python uses full words |
| `cdacwgt` | `convert_cap_to_weight` | Python uses full words |
| `adcpanel` | `generate_aout_dashboard` / `generate_dout_dashboard` | Python splits by type |

**Rule:** MATLAB uses terse abbreviations; Python uses full descriptive snake_case.
Both must implement the same algorithm with matching numerical output (tolerance 1e-6).

---

## 4. Adding a New Function — Checklist

### Python

1. Choose the right subdirectory: `fundamentals/`, `spectrum/`, `aout/`, `dout/`,
   `calibration/`, `siggen/`, `oversampling/`, `toolset/`
2. Name the file `<verb>_<noun>.py` following the prefix table in §1.1
3. Name the function identically to the file (minus `.py`)
4. Add type annotations on the signature (Python 3.10+ style)
5. Write a NumPy-style docstring
6. Return a `dict` with snake_case keys (not a tuple)
7. Support `create_plot=True` and `ax=None` if the function produces a plot
8. Register with `_export()` in `__init__.py`
9. Create `tests/unit/<subdir>/test_verify_<name>.py`

### MATLAB

1. Name the file as a short lowercase abbreviation in `matlab/src/`
2. Use `inputParser` for optional Name-Value arguments
3. Write `%FUNCTION_NAME` block documentation
4. Return multiple outputs in `[out1, out2, ...] = func(...)` style
5. If renaming an existing function, create a legacy wrapper

---

## 5. Testing Guide

For detailed examples, fixtures, and coverage gaps, read `references/testing-guide.md`.

### 5.1 Test Directory Structure

```
python/tests/
├── conftest.py          # Shared fixtures (project_root)
├── config.py            # Dataset paths and file patterns
├── _utils.py            # Helpers: save_fig, save_variable, auto_search_files
├── unit/                # Synthetic data — no external dependencies
│   ├── aout/            # 17 files — analog error analysis
│   ├── calibration/     # 8 files — weight calibration pipeline
│   ├── dout/            # 3 files — digital output (dashboard, weight radix, residual scatter)
│   ├── fundamentals/    # 4 files — frequency, fitting, conversions
│   ├── oversampling/    # 1 file — NTF analysis
│   ├── siggen/          # 1 file — noise shaping
│   └── spectrum/        # 18 files — FFT analysis, OSR sweep
├── integration/         # 19 files — real ADC datasets from reference_dataset/
└── compare/             # 18 files — Python vs MATLAB parity (tolerance 1e-6)
```

### 5.2 Three Test Tiers

| Tier | Location | Data Source | Purpose | Command |
|------|----------|-------------|---------|---------|
| **Unit** | `tests/unit/` | Synthetic | Verify algorithm correctness | `pytest tests/unit/ -v` |
| **Integration** | `tests/integration/` | Real ADC CSVs | End-to-end pipeline | `pytest tests/integration/ -v` |
| **Comparison** | `tests/compare/` | MATLAB reference | Cross-language parity (tol 1e-6) | `pytest tests/compare/ -v` |

### 5.3 Test Naming

- **Files:** `test_verify_<function>.py` (unit), `test_<topic>.py` (integration), `test_compare_<topic>.py` (comparison)
- **Functions:** `test_<scenario>()` in snake_case

### 5.4 Key Rules

- Always `create_plot=False` in unit tests
- Use `find_coherent_frequency` when generating test signals
- Import from **public API** (`from adctoolbox import ...`), not internal submodules
- Use `np.testing.assert_allclose(a, b, atol=1e-6)` for float comparisons

### 5.5 Test Coverage Summary

**91 test files, ~212 test functions** across unit/integration/compare.

**Untested functions (20 of ~46 exports):**
- 16 unit conversion functions (trivial but untested): `db_to_mag`, `mag_to_db`,
  `db_to_power`, `power_to_db`, `snr_to_enob`, `enob_to_snr`, `lsb_to_volts`,
  `volts_to_lsb`, `bin_to_freq`, `freq_to_bin`, `dbm_to_vrms`, `vrms_to_dbm`,
  `dbm_to_mw`, `mw_to_dbm`, `sine_amplitude_to_power`, `fold_bin_to_nyquist`
- 3 FOM metrics (`calculate_walden_fom`, `calculate_schreier_fom`, `calculate_thermal_noise_limit`)
- 1 utility (`estimate_frequency` — used internally but no dedicated test)

**Integration tests use legacy internal imports** (not public API):
- 19 integration files import from `adctoolbox.dout`, `adctoolbox.aout`, `adctoolbox.common`
  instead of `from adctoolbox import ...`

**4 skipped comparison tests** due to known MATLAB/Python divergences.
