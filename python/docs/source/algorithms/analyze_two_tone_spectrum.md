# analyze_two_tone_spectrum

## Overview

`analyze_two_tone_spectrum` performs two-tone spectrum analysis for ADC characterization, measuring intermodulation distortion (IMD2, IMD3) and two-tone SFDR. This is essential for evaluating ADC linearity with multi-signal inputs.

## Syntax

```python
from adctoolbox import analyze_two_tone_spectrum

# Auto-detect tones
result = analyze_two_tone_spectrum(signal, fs=100e6, show_plot=True)

# With custom parameters
result = analyze_two_tone_spectrum(signal, fs=100e6, window='blackman')
```

## Parameters

- **`signal`** (array_like) — Input two-tone signal
- **`fs`** (float) — Sampling frequency in Hz
- **`window`** (str, default='blackman') — Window function
- **`nfft`** (int, optional) — FFT length
- **`show_plot`** (bool, default=False) — Display spectrum plot
- **`ax`** (matplotlib axis, optional) — Axis for plotting

## Returns

Dictionary containing:

**Two-Tone Metrics:**
- **`imd2_db`** — 2nd-order intermodulation distortion (dB)
- **`imd3_db`** — 3rd-order intermodulation distortion (dB)
- **`imd2_freq`** — IMD2 product frequencies (Hz)
- **`imd3_freq`** — IMD3 product frequencies (Hz)
- **`sfdr_two_tone_db`** — Spurious-free dynamic range for two-tone

**Tone Information:**
- **`tone1_freq`**, **`tone2_freq`** — Fundamental frequencies
- **`tone1_power_db`**, **`tone2_power_db`** — Tone powers

## Algorithm

### Intermodulation Products

For tones at f₁ and f₂:

- **IMD2**: Products at f₁±f₂
- **IMD3**: Products at 2f₁-f₂ and 2f₂-f₁

```python
# Find two strongest tones
tone1_bin, tone2_bin = find_two_strongest_tones(spectrum)

# Calculate IMD product locations
imd2_bins = [abs(tone1_bin - tone2_bin), tone1_bin + tone2_bin]
imd3_bins = [abs(2*tone1_bin - tone2_bin), abs(2*tone2_bin - tone1_bin)]

# Measure powers
IMD2 = max_power(spectrum[imd2_bins])
IMD3 = max_power(spectrum[imd3_bins])
```

## Examples

### Example 1: Two-Tone Analysis

```python
import numpy as np
from adctoolbox import analyze_two_tone_spectrum

# Generate two-tone signal
N = 2**14
fs = 100e6
f1, f2 = 10e6, 12e6
t = np.arange(N) / fs
signal = 0.25 * np.sin(2*np.pi*f1*t) + 0.25 * np.sin(2*np.pi*f2*t)

# Analyze
result = analyze_two_tone_spectrum(signal, fs=fs, show_plot=True)

print(f"IMD2: {result['imd2_db']:.2f} dBc")
print(f"IMD3: {result['imd3_db']:.2f} dBc")
print(f"SFDR: {result['sfdr_two_tone_db']:.2f} dB")
```

## Interpretation

### IMD Performance

| IMD Level | ADC Quality |
|-----------|-------------|
| IMD3 < -80 dBc | Excellent linearity |
| -60 < IMD3 < -80 dBc | Good performance |
| IMD3 > -60 dBc | Significant nonlinearity |

### Common Issues

- **IMD2 dominant**: Even-order nonlinearity (differential pair mismatch)
- **IMD3 dominant**: Odd-order nonlinearity (compression, limiting)
- **High IMD products**: Check for:
  - Insufficient signal amplitude
  - ADC overload/clipping
  - Poor power supply rejection

## See Also

- [`analyze_spectrum`](analyze_spectrum.md) — Single-tone analysis
- [`fit_static_nonlin`](../api/aout.rst) — Extract nonlinearity coefficients

## References

1. IEEE Std 1241-2010, "IEEE Standard for Terminology and Test Methods for ADCs"
2. Application Note AN-742, "Frequency Domain Response of Switched-Capacitor ADCs," Analog Devices
