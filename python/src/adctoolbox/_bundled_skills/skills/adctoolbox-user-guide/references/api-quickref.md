# ADCToolbox API Quick Reference

Use this file for import routing and return-shape reminders. If you need a
runnable pattern, open `example-map.md` instead.

## Basic

Spectrum / coherent-sampling / SAR-weight-cal / synthetic-stim /
buffer-validation tier — what `SKILL.md` builds the basic workflow on.

### Flat Imports

```python
from adctoolbox import (
    analyze_spectrum,
    analyze_spectrum_polar,
    find_coherent_frequency,
    fit_sine_4param,
    calibrate_weight_sine,
)
```

### Submodule Imports

```python
from adctoolbox.siggen import ADC_Signal_Generator
from adctoolbox.calibration import calibrate_weight_sine_lite
from adctoolbox.fundamentals import validate_aout_data, validate_dout_data
from adctoolbox.spectrum import compute_spectrum
```

### Default Entry Points

- Dynamic FFT metrics:
  `analyze_spectrum`, `analyze_spectrum_polar`, `compute_spectrum`
- Digital calibration:
  `calibrate_weight_sine`, `calibrate_weight_sine_lite`
- Synthetic signals:
  `ADC_Signal_Generator`
- Pre-flight checks / coherent setup:
  `validate_aout_data`, `validate_dout_data`, `find_coherent_frequency`,
  `fit_sine_4param`

## Advanced

Open `advanced-debug.md` first when working on these — it has
worked snippets organized by question.

### Flat Imports

```python
from adctoolbox import (
    analyze_error_by_value,
    analyze_error_by_phase,
    analyze_error_pdf,
    analyze_error_spectrum,
    analyze_error_autocorr,
    analyze_error_envelope_spectrum,
    analyze_inl_from_sine,
    analyze_decomposition_time,
    analyze_decomposition_polar,
    fit_static_nonlin,
    analyze_bit_activity,
    analyze_overflow,
    analyze_weight_radix,
    analyze_enob_sweep,
    plot_residual_scatter,
    calculate_walden_fom,
    calculate_schreier_fom,
    calculate_thermal_noise_limit,
    calculate_jitter_limit,
    db_to_mag,
    mag_to_db,
    db_to_power,
    power_to_db,
    snr_to_enob,
    enob_to_snr,
    snr_to_nsd,
    nsd_to_snr,
    bin_to_freq,
    freq_to_bin,
    fold_frequency_to_nyquist,
    ntf_analyzer,
)
```

### Submodule Imports

```python
from adctoolbox.toolset import generate_aout_dashboard, generate_dout_dashboard
from adctoolbox.fundamentals import convert_cap_to_weight
from adctoolbox.aout import analyze_phase_plane, analyze_error_phase_plane
```

### Default Entry Points

- Analog error debug:
  `analyze_error_*` helpers, `decompose_harmonic_error`
- Dashboards:
  `generate_aout_dashboard`, `generate_dout_dashboard`
- Phase-plane:
  `analyze_phase_plane`, `analyze_error_phase_plane`
- Bit-level / per-code:
  `analyze_bit_activity`, `analyze_overflow`, `analyze_weight_radix`,
  `analyze_enob_sweep`
- Static nonlinearity:
  `fit_static_nonlin`
- Cap-to-weight:
  `convert_cap_to_weight`

## CLI

```bash
adctoolbox-get-examples
adctoolbox-get-examples my_examples_dir
adctoolbox-install-skill
adctoolbox-install-skill --dev
```

## Key Conventions

- `fit_sine_4param(... )["frequency"]` is normalized `Fin/Fs`, not Hz.
- Calibration helpers such as `calibrate_weight_sine(..., freq=...)` and
  `calibrate_weight_sine_lite(..., freq=...)` expect normalized `Fin/Fs`.
- `find_coherent_frequency(...)` returns a tuple:
  `(fin_actual_hz, best_bin)`.
- `analyze_overflow(...)` returns a tuple.
- `analyze_enob_sweep(...)` returns a tuple:
  `(enob_sweep, n_bits_vec)`.
- `calibrate_weight_sine_lite(...)` returns weights only.
- `analyze_weight_radix(...)` returns a dict.
- `compute_spectrum(...)` returns both `metrics` and `plot_data`.

If unsure which file to copy from, open `example-map.md`.
