# analyze_decomposition_time

## Overview

`analyze_decomposition_time` performs time-domain harmonic decomposition of ADC output, separating the fundamental from harmonic distortion components. This reveals the temporal structure of nonlinearity.

## Syntax

```python
from adctoolbox import analyze_decomposition_time

# Basic usage
result = analyze_decomposition_time(signal, fs=100e6, harmonic=5,
                                    show_plot=True)

# With custom parameters
result = analyze_decomposition_time(signal, fs=100e6, harmonic=9)
```

## Parameters

- **`signal`** (array_like) — Input ADC signal
- **`fs`** (float) — Sampling frequency in Hz
- **`harmonic`** (int, default=5) — Number of harmonics to decompose
- **`show_plot`** (bool, default=False) — Display decomposition plot
- **`ax`** (matplotlib axis, optional) — Axis for plotting

## Returns

Dictionary containing:
- **`fundamental`** — Fundamental component (time domain)
- **`harmonics`** — List of harmonic components
- **`residual`** — Remaining error (noise + distortion)
- **`harmonic_amplitudes`** — Amplitude of each harmonic
- **`harmonic_phases`** — Phase of each harmonic

## Algorithm

```python
# 1. Fit fundamental
result = fit_sine_4param(signal)
fundamental = result['fitted_signal']

# 2. Extract each harmonic by fitting at k×f₀
for k in range(2, harmonic + 1):
    harmonic_k = fit_sine_at_frequency(signal, freq_estimate * k)

# 3. Compute residual
residual = signal - fundamental - sum(all_harmonics)
```

## Use Cases

- Visualize harmonic distortion in time domain
- Understand nonlinearity structure (2nd vs. 3rd order)
- Compare with polar decomposition for phase insights

## See Also

- [`analyze_decomposition_polar`](../api/aout.rst) — Polar visualization of decomposition
- [`fit_sine_4param`](fit_sine_4param.md) — Core sine fitting algorithm
- [`analyze_spectrum`](analyze_spectrum.md) — Frequency-domain view

## References

1. IEEE Std 1057-2017, "IEEE Standard for Digitizing Waveform Recorders"
