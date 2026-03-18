# ADCToolbox API Quick Reference

Use this file for import routing and return-shape reminders. If you need a
runnable pattern, open `example-map.md` instead.

## Flat Imports

```python
from adctoolbox import (
    analyze_spectrum,
    analyze_spectrum_polar,
    find_coherent_frequency,
    fit_sine_4param,
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
    calibrate_weight_sine,
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

## Submodule Imports

```python
from adctoolbox.siggen import ADC_Signal_Generator
from adctoolbox.toolset import generate_aout_dashboard, generate_dout_dashboard
from adctoolbox.calibration import calibrate_weight_sine_lite
from adctoolbox.fundamentals import validate_aout_data, validate_dout_data, convert_cap_to_weight
from adctoolbox.aout import analyze_phase_plane, analyze_error_phase_plane
from adctoolbox.spectrum import compute_spectrum
```

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

## Default Entry Points

- Dynamic FFT metrics:
  `analyze_spectrum`, `analyze_spectrum_polar`
- Analog debug:
  `fit_sine_4param` and the `analyze_error_*` helpers
- Digital calibration:
  `calibrate_weight_sine`, `calibrate_weight_sine_lite`
- Dashboards:
  `generate_aout_dashboard`, `generate_dout_dashboard`
- Synthetic signals:
  `ADC_Signal_Generator`

If unsure which file to copy from, open `example-map.md`.
