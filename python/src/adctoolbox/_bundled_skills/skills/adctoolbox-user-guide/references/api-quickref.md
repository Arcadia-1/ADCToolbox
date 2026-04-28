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
from adctoolbox.aout import (
    analyze_phase_plane, analyze_error_phase_plane, decompose_harmonic_error,
)
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

### Frequency

- `fit_sine_4param(...)["frequency"]` is normalized `Fin/Fs` (range 0–0.5),
  not Hz.
- `calibrate_weight_sine(bits, freq=...)` and `calibrate_weight_sine_lite(bits,
  freq)` expect normalized `freq = Fin/Fs`. The `_lite` variant takes `freq`
  positionally (not as a keyword) and is required (no auto-search).
- `analyze_enob_sweep(bits, freq=...)` and `generate_dout_dashboard(bits,
  freq=...)` also expect normalized `freq`.
- `generate_aout_dashboard(aout, fs=..., freq=...)` takes `freq` in Hz (it
  normalizes internally).
- `find_coherent_frequency(fs, fin_target, n_fft)` returns a tuple
  `(fin_actual_hz, best_bin)`. Argument order matters: `fs` first, then
  `fin_target`, then `n_fft`.
- `analyze_spectrum` does NOT take `Fin` — the fundamental is auto-detected.

### Return shapes

- Spectrum metrics dicts (`analyze_spectrum`, `analyze_spectrum_polar`,
  `compute_spectrum["metrics"]`) use lowercase `_dbc` keys: `enob`,
  `sndr_dbc`, `sfdr_dbc`, `snr_dbc`, `thd_dbc`, `sig_pwr_dbfs`,
  `noise_floor_dbfs`, `nsd_dbfs_hz`, `harmonics_dbc` — **not**
  uppercase `SNDR` / `SFDR` / `ENOB`.
- `compute_spectrum(...)` returns a dict with top-level keys `metrics` and
  `plot_data` (the latter has `freq`, `power_spectrum_db_plot`,
  `complex_spectrum`, `fundamental_bin`, …).
- `calibrate_weight_sine(...)` returns a dict: `weight`, `offset`,
  `calibrated_signal`, `ideal`, `error`, `refined_frequency`.
  `calibrate_weight_sine_lite(...)` returns just the weights ndarray.
- `analyze_bit_activity(bits)` returns an ndarray (% of 1's per bit).
- `analyze_overflow(bits, weight)` returns a 4-tuple of ndarrays
  `(range_min, range_max, ovf_pct_zero, ovf_pct_one)`. The second argument
  is the calibrated weights vector — not optional.
- `analyze_enob_sweep(bits, freq=...)` returns `(enob_sweep, n_bits_vec)`.
- `analyze_weight_radix(weights)` returns a dict (`radix`, `wgtsca`, `effres`).
- `fit_static_nonlin(sig_distorted, order)` returns
  `(k2, k3, fitted_sine, fitted_transfer)`. Input is a distorted signal,
  not INL/DNL data; `order >= 2`.
- `convert_cap_to_weight(caps_bit, caps_bridge, caps_parasitic)` takes
  three same-length arrays (LSB→MSB; pass zeros where there is no
  bridge / parasitic) and returns `(weights, c_total)`.
- `analyze_phase_plane(aout, fs=...)` and `analyze_error_phase_plane(aout,
  fs=...)` do NOT take `Fin` — they self-fit the fundamental.

If unsure which file to copy from, open `example-map.md`.
