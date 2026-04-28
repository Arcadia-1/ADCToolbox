# ADCToolbox — Advanced Debug Reference

Load this file when the basic spectrum/calibration tier in `SKILL.md`
is not enough. Each section is keyed by the user's likely question,
not by file layout.

Import conventions follow `SKILL.md` §5. Frequency conventions follow
`SKILL.md` §2 — pay particular attention: `generate_aout_dashboard`
takes `freq` in **Hz**, while `generate_dout_dashboard` takes `freq`
**normalized** to `Fin/Fs`.

## "I want one image showing all aout/dout diagnostics"

```python
from adctoolbox.toolset import generate_aout_dashboard, generate_dout_dashboard

# aout dashboard: freq in Hz
fig, axes = generate_aout_dashboard(
    aout, fs=fs, freq=fin_hz, output_path="aout_dash.png"
)

# dout dashboard: freq normalized (Fin/Fs)
fig, axes = generate_dout_dashboard(
    bits, freq=fin_hz / fs, output_path="dout_dash.png"
)
```

Use `generate_aout_dashboard` when you have reconstructed analog output
(floats); use `generate_dout_dashboard` when you only have the per-bit
decision matrix `bits`. Both return `(fig, axes_array)` and write a PNG
when `output_path=` is given.

## "I need to see nonlinearity structure, not just a single INL/DNL number"

```python
from adctoolbox.aout import analyze_phase_plane, analyze_error_phase_plane

pp  = analyze_phase_plane(aout, fs=fs)        # full signal trajectory
epp = analyze_error_phase_plane(aout, fs=fs)  # error-only (sine subtracted first)
```

Neither helper takes `Fin` — `analyze_phase_plane` infers the lag from
the FFT (`lag='auto'` default), and `analyze_error_phase_plane` calls
`fit_sine_4param` internally. Returns are dicts: `pp` has
`{lag, outliers}`, `epp` has `{residual, fitted_params, trend_coeffs,
hysteresis_gap}`. Use `epp` to isolate harmonic structure from the
fundamental.

## "I want per-bit behavior — activity, overflow, ENOB vs bit depth"

```python
from adctoolbox import (
    analyze_bit_activity, analyze_overflow,
    analyze_enob_sweep, analyze_weight_radix,
    calibrate_weight_sine,
)

activity = analyze_bit_activity(bits)                   # ndarray, length N_bits
weights  = calibrate_weight_sine(bits, freq=fin_hz/fs)["weight"]
range_min, range_max, ovf_pct_zero, ovf_pct_one = analyze_overflow(bits, weights)
enob_sweep, n_bits_vec = analyze_enob_sweep(bits, freq=fin_hz/fs)
radix_info = analyze_weight_radix(weights)              # dict: radix, wgtsca, effres
```

Notes on shapes:
- `analyze_bit_activity` returns an ndarray of length `N_bits`, not a dict.
- `analyze_overflow` is a 4-tuple of ndarrays — needs the calibrated
  `weights` as its second argument (it's measuring digital-domain over-range,
  not raw saturation).
- `analyze_enob_sweep` is a 2-tuple `(enob_sweep, n_bits_vec)` and runs
  `calibrate_weight_sine` once internally.
- `analyze_weight_radix(weights)` returns a `dict` (was an ndarray in
  pre-`v0.6` versions).

## "I have a distorted sine and want to extract HD2/HD3 coefficients"

```python
from adctoolbox import fit_static_nonlin

k2, k3, fitted_sine, fitted_transfer = fit_static_nonlin(sig_distorted, order=3)
# fitted_transfer is a tuple (x_smooth, y_smooth) of the fitted nonlinear
# transfer characteristic. k2/k3 are normalized to k1 (unity-gain).
```

Note: input is a *distorted signal* (post-nonlinearity), not INL/DNL data.
`order` must be `>=2`; `k3` is `NaN` when `order < 3`.

## "I want to decompose total error into component contributions"

```python
from adctoolbox.aout import (
    analyze_decomposition_polar,
    analyze_decomposition_time,
    decompose_harmonic_error,
    analyze_error_spectrum,
    analyze_error_envelope_spectrum,
    analyze_error_pdf,
    analyze_error_autocorr,
    analyze_error_by_phase,
    analyze_error_by_value,
    analyze_inl_from_sine,
)
```

Pick by the error view you need:

- by phase / by value → `analyze_error_by_phase`, `analyze_error_by_value`
- harmonic decomposition → `decompose_harmonic_error`
- spectral view of the error → `analyze_error_spectrum`,
  `analyze_error_envelope_spectrum`
- statistical → `analyze_error_pdf`, `analyze_error_autocorr`
- polar / time decomposition views → `analyze_decomposition_polar`,
  `analyze_decomposition_time`
- INL from sine test → `analyze_inl_from_sine`

All of these expect `aout` (or fall back to `signal`) plus `fs` in Hz.
They each fit a sine internally if no `frequency` kwarg is supplied.

## "I need cap array → weight conversion for CDAC modeling"

```python
from adctoolbox.fundamentals import convert_cap_to_weight

weights, c_total = convert_cap_to_weight(caps_bit, caps_bridge, caps_parasitic)
```

Three input arrays (LSB-to-MSB ordering, all the same length):
`caps_bit` — main capacitor sizes, `caps_bridge` — bridge caps between
sub-DAC sections (use 0 where there is no bridge), `caps_parasitic` —
parasitic caps per node. Returns a `(weights, c_total)` tuple — never
a dict.

## When to fall back to `SKILL.md`

If the task is plain spectrum analysis (SNDR / SFDR / ENOB), basic
sine fitting, or SAR weight calibration via `calibrate_weight_sine*`,
re-read `SKILL.md` — the basic tier has the cleaner entry points.
