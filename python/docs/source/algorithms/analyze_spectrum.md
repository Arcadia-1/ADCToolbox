# analyze_spectrum

## Overview

`analyze_spectrum` performs FFT-based spectrum analysis for ADC characterization, computing key metrics including ENOB, SNR, SNDR, SFDR, THD, and harmonic content. This is the primary tool for frequency-domain ADC performance evaluation.

## Syntax

```python
from adctoolbox import analyze_spectrum

# Simplest usage
result = analyze_spectrum(signal, fs=100e6)

# With harmonic analysis
result = analyze_spectrum(signal, fs=100e6, harmonic=5)

# With custom windowing
result = analyze_spectrum(signal, fs=100e6, window='blackman', nfft=8192)

# With averaging
result = analyze_spectrum(signal, fs=100e6, num_avg=4, mode='coherent')

# With interactive plotting
result = analyze_spectrum(signal, fs=100e6, show_plot=True)
```

## Parameters

- **`signal`** (array_like) — Input ADC signal (1D array)
- **`fs`** (float) — Sampling frequency in Hz
- **`harmonic`** (int, default=9) — Number of harmonics to analyze (2-20)
- **`window`** (str, default='blackman') — Window function: 'rect', 'hann', 'hamming', 'blackman', 'flattop'
- **`nfft`** (int, optional) — FFT length. If None, uses len(signal)
- **`num_avg`** (int, default=1) — Number of segments for averaging
- **`mode`** (str, default='coherent') — Averaging mode: 'coherent' or 'power'
- **`show_plot`** (bool, default=False) — Display interactive spectrum plot
- **`ax`** (matplotlib axis, optional) — Axis for plotting

## Returns

Dictionary containing:

**Metrics:**
- **`enob`** — Effective Number of Bits
- **`snr_db`** — Signal-to-Noise Ratio (dB)
- **`sndr_db`** — Signal-to-Noise-and-Distortion Ratio (dB)
- **`sfdr_db`** — Spurious-Free Dynamic Range (dB)
- **`thd_db`** — Total Harmonic Distortion (dB)
- **`sinad_db`** — Signal-to-Noise-and-Distortion (alternative name for SNDR)
- **`nsd_dbfs_hz`** — Noise Spectral Density (dBFS/Hz)

**Spectrum Data:**
- **`freq`** — Frequency bins (Hz)
- **`magnitude_db`** — Spectrum magnitude (dBFS)
- **`fundamental_bin`** — FFT bin of fundamental frequency
- **`fundamental_freq`** — Fundamental frequency (Hz)
- **`harmonic_bins`** — FFT bins of harmonics (list)
- **`harmonic_powers`** — Power of each harmonic (dB, list)

## Algorithm

### 1. Signal Preprocessing

```python
# Apply window
windowed_signal = signal * window_function

# Zero-pad if nfft > len(signal)
if nfft > len(signal):
    windowed_signal = np.pad(windowed_signal, (0, nfft - len(signal)))
```

### 2. FFT Computation

```python
spectrum = np.fft.fft(windowed_signal, n=nfft)
magnitude = np.abs(spectrum[:nfft//2])
magnitude_db = 20 * np.log10(magnitude / nfft * 2)  # Convert to dBFS
```

### 3. Fundamental Detection

```python
# Find peak in first Nyquist zone (excluding DC)
fundamental_bin = np.argmax(magnitude[1:nfft//2]) + 1
fundamental_freq = fundamental_bin * fs / nfft
```

### 4. Harmonic Identification

```python
harmonic_bins = [fundamental_bin * k for k in range(2, harmonic + 1)]
# Fold harmonics above Nyquist back into spectrum
harmonic_bins_folded = [fold_bin_to_nyquist(h, nfft) for h in harmonic_bins]
```

### 5. Metric Calculation

```python
# Signal power (fundamental)
P_signal = magnitude[fundamental_bin]**2

# Harmonic power
P_harmonics = sum([magnitude[h]**2 for h in harmonic_bins_folded])

# Noise power (excluding signal, harmonics, DC)
P_noise = sum(magnitude**2) - P_signal - P_harmonics - magnitude[0]**2

# Metrics
SNR = 10 * log10(P_signal / P_noise)
THD = 10 * log10(P_harmonics / P_signal)
SNDR = 10 * log10(P_signal / (P_noise + P_harmonics))
ENOB = (SNDR - 1.76) / 6.02
SFDR = fundamental_mag_db - max_spur_mag_db
```

## Examples

### Example 1: Basic Spectrum Analysis

```python
import numpy as np
from adctoolbox import analyze_spectrum

# Generate test signal
N = 2**13
fs = 100e6
fin = 123/N * fs  # Coherent frequency
t = np.arange(N) / fs
signal = 0.5 * np.sin(2*np.pi*fin*t) + np.random.randn(N) * 10e-6

# Analyze
result = analyze_spectrum(signal, fs=fs)

print(f"ENOB: {result['enob']:.2f} bits")
print(f"SNDR: {result['sndr_db']:.2f} dB")
print(f"SFDR: {result['sfdr_db']:.2f} dB")
print(f"THD: {result['thd_db']:.2f} dB")
```

### Example 2: With Interactive Plotting

```python
result = analyze_spectrum(signal, fs=800e6, show_plot=True)
# Opens interactive plot showing spectrum, harmonics, and metrics
```

### Example 3: Windowing Comparison

```python
windows = ['rect', 'hann', 'blackman', 'flattop']

for win in windows:
    result = analyze_spectrum(signal, fs=fs, window=win)
    print(f"{win:10s}: SFDR={result['sfdr_db']:6.2f} dB, "
          f"SNDR={result['sndr_db']:6.2f} dB")
```

### Example 4: Coherent Averaging

```python
# Average 4 segments coherently
result = analyze_spectrum(signal, fs=fs, num_avg=4, mode='coherent')
# Reduces noise floor by ~6 dB (factor of 4)
```

## Window Function Selection

| Window | ENBW* | Peak Side Lobe | Use Case |
|--------|-------|----------------|----------|
| **rect** | 1.0 | -13 dB | Coherent signals only |
| **hann** | 1.5 | -32 dB | General purpose |
| **hamming** | 1.36 | -43 dB | Better side lobe rejection |
| **blackman** | 1.73 | -58 dB | High dynamic range (default) |
| **flattop** | 3.77 | -93 dB | Accurate amplitude measurement |

*ENBW = Equivalent Noise Bandwidth (bins)

## Interpretation

### ENOB (Effective Number of Bits)
- **ENOB ≈ N-bit ADC**: Ideal performance
- **ENOB < N - 2**: Poor performance, investigate noise/distortion
- **ENOB > N**: Oversampling or dithering improving effective resolution

### SFDR (Spurious-Free Dynamic Range)
- **SFDR > 80 dB**: Excellent linearity
- **60 dB < SFDR < 80 dB**: Good for most applications
- **SFDR < 60 dB**: Significant spurs, check for:
  - Power supply coupling
  - Clock feedthrough
  - Intermodulation products

### THD (Total Harmonic Distortion)
- **THD < -80 dB**: Very linear
- **-60 dB < THD < -80 dB**: Acceptable for most uses
- **THD > -60 dB**: Nonlinearity issues

## Common Issues

### Non-Coherent Sampling
```python
# BAD: Arbitrary frequency causes spectral leakage
signal = np.sin(2*np.pi*10.1234e6*t)  # Non-coherent

# GOOD: Use coherent frequency
from adctoolbox import find_coherent_frequency
fin_coh, bin_num = find_coherent_frequency(fs, 10e6, N)
signal = np.sin(2*np.pi*fin_coh*t)
```

### Insufficient FFT Length
```python
# BAD: Low resolution
result = analyze_spectrum(signal[:512], fs=fs)  # Only 512 points

# GOOD: More frequency resolution
result = analyze_spectrum(signal, fs=fs, nfft=8192)
```

## See Also

- [`analyze_spectrum_polar`](analyze_spectrum_polar.md) — Polar phase spectrum analysis
- [`analyze_two_tone_spectrum`](analyze_two_tone_spectrum.md) — Two-tone IMD testing
- [`find_coherent_frequency`](../api/fundamentals.rst) — Calculate coherent test frequencies

## References

1. IEEE Std 1241-2010, "IEEE Standard for Terminology and Test Methods for ADCs"
2. F. J. Harris, "On the Use of Windows for Harmonic Analysis with the DFT," Proc. IEEE, 1978
3. Application Note AN-9675, "Coherent Sampling Calculator," Analog Devices
