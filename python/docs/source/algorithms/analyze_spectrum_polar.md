# analyze_spectrum_polar

## Overview

`analyze_spectrum_polar` performs FFT-based spectrum analysis with polar (phase) visualization. This tool is particularly useful for distinguishing between noise (random phase) and deterministic distortion (coherent phase).

## Syntax

```python
from adctoolbox import analyze_spectrum_polar

# Basic usage
result = analyze_spectrum_polar(signal, fs=100e6, show_plot=True)

# With custom parameters
result = analyze_spectrum_polar(signal, fs=100e6, harmonic=5,
                                window='blackman', nfft=8192)
```

## Parameters

- **`signal`** (array_like) — Input ADC signal
- **`fs`** (float) — Sampling frequency in Hz
- **`harmonic`** (int, default=9) — Number of harmonics to analyze
- **`window`** (str, default='blackman') — Window function
- **`nfft`** (int, optional) — FFT length
- **`show_plot`** (bool, default=False) — Display polar plot
- **`ax`** (matplotlib axis, optional) — Axis for plotting

## Returns

Dictionary containing spectrum metrics (same as `analyze_spectrum`) plus:
- **`phase`** — Phase spectrum in radians
- **`phase_unwrapped`** — Unwrapped phase for easier analysis

## Key Features

- **Polar visualization**: Magnitude vs. phase plot reveals noise vs. distortion
- **Phase coherence**: Deterministic signals have fixed phase, noise has random phase
- **Memory effects**: Phase deviations indicate signal-dependent errors

## Use Cases

- Distinguish thermal noise from harmonic distortion
- Identify memory effects in pipelined ADCs
- Analyze settling errors and inter-symbol interference

## See Also

- [`analyze_spectrum`](analyze_spectrum.md) — Standard spectrum analysis
- [`analyze_two_tone_spectrum`](analyze_two_tone_spectrum.md) — Two-tone testing

## References

1. IEEE Std 1241-2010, "IEEE Standard for Terminology and Test Methods for ADCs"
