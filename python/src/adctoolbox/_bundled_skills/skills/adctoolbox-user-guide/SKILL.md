---
name: adctoolbox-user-guide
description: >
  Lightweight routing guide for using ADCToolbox from Python. Use this skill
  whenever a task involves writing, fixing, reviewing, or explaining Python
  code that uses ADCToolbox; choosing the right analysis helper; finding the
  right packaged example; using flat exports versus submodule imports; running
  dashboards; generating synthetic ADC data; or calibrating bit matrices.
---

# ADCToolbox Usage Guide

Use this skill as a router, not as a full manual.

Source-of-truth order:

1. `python/src/adctoolbox/__init__.py` for flat Python exports
2. `python/src/adctoolbox/*/__init__.py` for submodule-only Python tools
3. `python/src/adctoolbox/examples/` for runnable usage patterns

Open references only as needed:

- `references/api-quickref.md` for import paths, signatures, and return shapes
- `references/example-map.md` for actual example files to adapt

## 1. Start From Examples

- If the user wants working usage code, open `references/example-map.md` first.
- Prefer adapting a packaged example over writing an API call pattern from
  scratch.
- The packaged CLI command is `adctoolbox-get-examples [dest]`.

## 2. Python Import Rules

Use flat imports only for symbols actually exported by `adctoolbox`:

```python
from adctoolbox import analyze_spectrum, fit_sine_4param, calibrate_weight_sine
```

Use submodule imports for tools that are public but not flat-exported:

```python
from adctoolbox.siggen import ADC_Signal_Generator
from adctoolbox.toolset import generate_aout_dashboard, generate_dout_dashboard
from adctoolbox.calibration import calibrate_weight_sine_lite
from adctoolbox.fundamentals import validate_aout_data, validate_dout_data, convert_cap_to_weight
from adctoolbox.aout import analyze_phase_plane, analyze_error_phase_plane
```

Important:

- If a flat import fails, inspect the relevant submodule `__init__.py` before
  assuming the tool does not exist.

## 3. Pick The Right Tool Family

**A. Basic Operations (Essential for Testbenches):**
- **Dynamic FFT & Coherent Sampling**: 
  `analyze_spectrum`, `analyze_spectrum_polar`, `find_coherent_frequency`
- **Dashboard Summaries (Multi-Plot)**: 
  `adctoolbox.toolset.generate_aout_dashboard`, `adctoolbox.toolset.generate_dout_dashboard`

**B. Advanced Debug & Calibration:**
- **Analog Debugging**: 
  `fit_sine_4param`, error-analysis helpers, decomposition helpers, phase-plane helpers
- **Digital Calibration**: 
  `calibrate_weight_sine`, `calibrate_weight_sine_lite`, `analyze_bit_activity`, `analyze_overflow`, `analyze_enob_sweep`, `analyze_weight_radix`

**C. Utilities:**
- **Signal Generation**: 
  `adctoolbox.siggen.ADC_Signal_Generator`
- **Unit Conversions**: 
  use the flat-exported helpers directly from `adctoolbox`

Validate external inputs early:

```python
from adctoolbox.fundamentals import validate_aout_data, validate_dout_data

validate_aout_data(signal)
validate_dout_data(bits)
```

## 4. Critical Conventions

### Frequency Units

- `fs`, `Fin`, and plotting frequencies are in Hz.
- `fit_sine_4param(... )['frequency']` is normalized `Fin/Fs`, not Hz.
- `calibrate_weight_sine`, `calibrate_weight_sine_lite`, and many DOUT helpers
  expect normalized `freq=Fin/Fs`.

### Return Shapes Are Not Uniform

Most Python analysis functions return dictionaries, but notable exceptions are:

- `find_coherent_frequency` -> tuple `(fin_hz, bin_idx)`
- `analyze_bit_activity` -> ndarray
- `analyze_overflow` -> tuple
- `analyze_enob_sweep` -> tuple `(enob_sweep, n_bits_vec)`
- `fit_static_nonlin` -> tuple
- `calibrate_weight_sine_lite` -> ndarray
- `convert_cap_to_weight` -> tuple `(weights, c_total)`

Also note:

- `analyze_weight_radix` now returns a dict, not a bare array
- `compute_spectrum` returns both metrics and plot data

When docs conflict, trust source exports and packaged examples over old README
text.

## 5. What To Open Next

- Need a signature or return key: open `references/api-quickref.md`
- Need a real example file to adapt: open `references/example-map.md`
