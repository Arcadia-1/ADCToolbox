# analyze_error_envelope_spectrum

## Overview

`analyze_error_envelope_spectrum` analyzes the envelope spectrum of ADC errors to detect amplitude modulation (AM) patterns. This reveals signal-dependent errors that modulate with the input amplitude.

## Syntax

```python
from adctoolbox import analyze_error_envelope_spectrum

# Basic usage
result = analyze_error_envelope_spectrum(signal, fs=100e6, show_plot=True)

# With custom parameters
result = analyze_error_envelope_spectrum(signal, fs=100e6, resolution=12)
```

## Parameters

- **`signal`** (array_like) — Input ADC signal
- **`fs`** (float) — Sampling frequency in Hz
- **`resolution`** (int, optional) — ADC resolution in bits
- **`show_plot`** (bool, default=False) — Display envelope spectrum
- **`ax`** (matplotlib axis, optional) — Axis for plotting

## Returns

Dictionary containing:
- **`envelope_spectrum`** — Envelope spectrum magnitude
- **`envelope_freq`** — Frequency bins for envelope
- **`error`** — Error signal
- **`envelope`** — Extracted envelope

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
