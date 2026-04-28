# ADCToolbox — Advanced Debug Reference

Load this file when the basic spectrum/calibration tier in `SKILL.md`
is not enough. Each section below is keyed by the user's likely
question, not by file layout.

Import conventions follow `SKILL.md` §5. Frequency conventions follow
`SKILL.md` §2.

## "I want one image showing all aout/dout diagnostics"

Goal: generate a multi-plot dashboard (time-domain + spectrum + INL/DNL + extras).

```python
from adctoolbox.toolset import generate_aout_dashboard, generate_dout_dashboard

generate_aout_dashboard(aout, fs=fs, savepath="aout_dash.png")
generate_dout_dashboard(dout, n_bits=N, fs=fs, savepath="dout_dash.png")
```

Use `_aout_` when you have reconstructed analog output (floats); use
`_dout_` when you only have raw digital codes.

## "I need to see nonlinearity structure, not just a single INL/DNL number"

```python
from adctoolbox.aout import analyze_phase_plane, analyze_error_phase_plane

analyze_phase_plane(aout, fs=fs, Fin=Fin)           # full signal phase trajectory
analyze_error_phase_plane(aout, fs=fs, Fin=Fin)     # error-only phase plane
```

Use `analyze_error_phase_plane` after `fit_sine_4param` to isolate
nonlinearity from the fundamental.

## "I want per-bit behavior — activity, overflow, or ENOB vs bit depth"

```python
from adctoolbox import analyze_bit_activity, analyze_overflow, analyze_enob_sweep, analyze_weight_radix
```

- `analyze_bit_activity(dout, n_bits=N)` → `ndarray`
- `analyze_overflow(dout, n_bits=N)` → `tuple`
- `analyze_enob_sweep(dout, n_bits=N)` → `tuple (enob_sweep, n_bits_vec)`
- `analyze_weight_radix(weights)` → `dict`   (was a bare array in old versions — now a dict)

## "I have static INL/DNL data and want a nonlinearity fit"

```python
from adctoolbox import fit_static_nonlin
coef, residual = fit_static_nonlin(inl_or_dnl, order=3)   # returns tuple
```

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
- spectral view of the error → `analyze_error_spectrum`, `analyze_error_envelope_spectrum`
- statistical → `analyze_error_pdf`, `analyze_error_autocorr`
- polar / time decomposition views → `analyze_decomposition_polar`, `analyze_decomposition_time`
- INL from sine test → `analyze_inl_from_sine`

All of the above expect `fs` and `Fin` in Hz and assume a sine test
that has already passed `validate_aout_data` / `validate_dout_data`.

## "I need cap array → weight conversion for CDAC modeling"

```python
from adctoolbox.fundamentals import convert_cap_to_weight
weights, c_total = convert_cap_to_weight(cap_array)   # returns tuple
```

## When to fall back to `SKILL.md`

If the task is plain spectrum analysis (SNDR / SFDR / ENOB), basic
sine fitting, or SAR weight calibration via `calibrate_weight_sine*`,
re-read `SKILL.md` — the basic tier has the cleaner entry points.
