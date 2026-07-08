# analyze_error_envelope_spectrum

## Overview

`analyze_error_envelope_spectrum` analyzes the envelope spectrum of ADC errors to detect amplitude modulation (AM) patterns. This reveals signal-dependent errors that modulate with the input amplitude.

By default the input is treated as an ADC output signal: the function fits and
subtracts a sine internally, then analyzes the envelope of that residual. When
you already have a residual/error waveform, pass `input_kind="error"` to skip
the sine fit. That residual-input mode matches MATLAB `errevspec`.

## Syntax

```python
from adctoolbox import analyze_error_envelope_spectrum

# Basic usage
result = analyze_error_envelope_spectrum(signal, fs=100e6, create_plot=True)

# MATLAB errevspec-style usage with a precomputed residual
result = analyze_error_envelope_spectrum(error, fs=100e6, input_kind="error")
```

## Parameters

- **`signal`** (array_like) — Input ADC signal, or a residual/error waveform
  when `input_kind="error"`
- **`fs`** (float) — Sampling frequency in Hz
- **`frequency`** (float, optional) — Normalized input frequency for
  signal-input sine fitting
- **`input_kind`** (`"signal"` or `"error"`, default=`"signal"`) — Whether
  to fit/subtract a sine internally or analyze the input directly as residual
- **`create_plot`** (bool, default=True) — Display envelope spectrum
- **`ax`** (matplotlib axis, optional) — Axis for plotting

## Returns

Dictionary containing:
- spectrum metric keys such as **`enob`**, **`sndr_dbc`**, **`sfdr_dbc`**,
  **`snr_dbc`**, **`thd_dbc`**, **`sig_pwr_dbfs`**, and
  **`noise_floor_dbfs`**
- **`error_signal`** — Error signal used for envelope extraction
- **`envelope`** — Extracted envelope
- **`input_kind`** — Input contract used for the analysis

## Interpretation

| Envelope Spectrum | Likely Cause |
|-------------------|--------------|
| **DC component only** | Signal-independent error (no AM) |
| **Peak at 2×Fin** | Memory effect, residue amplifier gain error |
| **Peak at Fin** | Asymmetric nonlinearity |
| **Multiple peaks** | Complex memory effects |

## Use Cases

- Detect memory effects in pipelined/SAR ADCs
- Identify signal-dependent settling errors
- Reveal gain errors in residue amplifiers

## See Also

- [`analyze_error_autocorr`](analyze_error_autocorr.md) — Time-domain correlation
- [`analyze_error_spectrum`](analyze_error_spectrum.md) — Direct error spectrum
- [`analyze_decomposition_time`](analyze_decomposition_time.md) — Harmonic decomposition

## References

1. M. Mishali et al., "Automatic Testing of  Pipelined ADCs," Proc. IEEE Int. Test Conf., 2007
