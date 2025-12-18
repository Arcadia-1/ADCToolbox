# fit_sine_4param

## Overview

`fit_sine_4param` fits a pure sine wave to ADC output using iterative least-squares with frequency refinement. This implements the IEEE Std 1057/1241 algorithm for precise sinusoidal parameter estimation.

## Syntax

```python
from adctoolbox import fit_sine_4param

# Auto-detect frequency
result = fit_sine_4param(data)

# With initial frequency estimate
result = fit_sine_4param(data, frequency_estimate=0.123)

# With custom convergence parameters
result = fit_sine_4param(data, frequency_estimate=0.123,
                         max_iterations=10, tolerance=1e-9)
```

## Parameters

- **`data`** (array_like) — ADC output signal (1D or 2D array)
- **`frequency_estimate`** (float, optional) — Initial normalized frequency (0 to 0.5)
  - If None: auto-detect using FFT peak with parabolic interpolation
- **`max_iterations`** (int, default=1) — Number of iterations for frequency refinement
- **`tolerance`** (float, default=1e-9) — Convergence threshold for frequency updates

## Returns

Dictionary containing:

- **`fitted_signal`** — Reconstructed sine wave
- **`residuals`** — data - fitted_signal
- **`frequency`** — Refined normalized frequency (0 to 0.5)
- **`amplitude`** — Signal amplitude: sqrt(A² + B²)
- **`phase`** — Phase in radians: atan2(-B, A)
- **`dc_offset`** — DC component
- **`rmse`** — Root mean square error of the fit

For 2D input, all scalar values become 1D arrays (one per column).

## Algorithm

### 1. Frequency Initialization (if not provided)

**FFT-based coarse estimate** with 3-point parabolic interpolation:
```python
spec = np.abs(np.fft.fft(data))
k0 = np.argmax(spec[1:N//2]) + 1  # Exclude DC
# Parabolic interpolation for sub-bin accuracy
direction = np.sign(spec[k0+1] - spec[k0-1])
freq_estimate = (k0 + direction * spec[k0 + direction] /
                (spec[k0] + spec[k0 + direction])) / N
```

### 2. Iterative Least-Squares Refinement

Model: **y = A·cos(ωt) + B·sin(ωt) + C**

For each iteration:

**Step 1**: Solve augmented least-squares system:
```
[cos(θ), sin(θ), ones, ∂/∂freq] × [A; B; C; δf] = data
```
where:
- `θ = 2π × freq × (0:N-1)`
- `∂/∂freq = derivative of signal w.r.t. frequency`

**Step 2**: Update frequency:
```python
freq += δf / N
```

**Step 3**: Check convergence:
```python
if abs(δf) < tolerance: break
```

### 3. Parameter Extraction

```python
amplitude = np.sqrt(A**2 + B**2)
phase = np.arctan2(-B, A)
fitted_signal = A * np.cos(θ) + B * np.sin(θ) + C
residuals = data - fitted_signal
rmse = np.sqrt(np.mean(residuals**2))
```

## Examples

### Example 1: Auto-Detect Frequency

```python
import numpy as np
from adctoolbox import fit_sine_4param

# Generate test signal
N = 1000
t = np.arange(N)
signal = 0.5 * np.sin(2*np.pi*0.123*t) + 0.1

# Fit sine wave
result = fit_sine_4param(signal)

print(f"Frequency: {result['frequency']:.6f}")
print(f"Amplitude: {result['amplitude']:.4f}")
print(f"Phase: {np.degrees(result['phase']):.2f}°")
print(f"DC Offset: {result['dc_offset']:.4f}")
print(f"RMSE: {result['rmse']:.6f}")
```

### Example 2: Known Frequency

```python
f0 = 0.1234  # Known normalized frequency
result = fit_sine_4param(data, frequency_estimate=f0)

error = result['residuals']
print(f"Fit RMS error: {result['rmse']:.6f}")
```

### Example 3: Tight Convergence

```python
result = fit_sine_4param(data, max_iterations=10, tolerance=1e-12)
```

### Example 4: Visualize Fit Quality

```python
import matplotlib.pyplot as plt

result = fit_sine_4param(signal)

# Plot one period
period = int(1/result['frequency'])
n_samples = min(period * 2, len(signal))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Original vs fitted
ax1.plot(signal[:n_samples], 'o-', label='Original', alpha=0.7)
ax1.plot(result['fitted_signal'][:n_samples], '--', label='Fitted')
ax1.set_ylabel('Amplitude')
ax1.legend()
ax1.grid(True)

# Residuals
ax2.plot(result['residuals'][:n_samples])
ax2.set_ylabel('Residuals')
ax2.set_xlabel('Sample')
ax2.grid(True)

plt.tight_layout()
plt.show()
```

## Interpretation

| Output | Interpretation |
|--------|----------------|
| `amplitude / max(data)` ≈ 0.5 | Full-scale sine wave input (peak-to-peak = 1) |
| `abs(dc_offset) << amplitude` | Well-centered signal |
| `abs(dc_offset) ≈ amplitude` | Large DC offset, check ADC biasing |
| High `rmse` | Distortion, harmonics, or noise present |
| `frequency` stable after 1 iteration | Good initial estimate |

## Use Cases

- **Error Analysis**: Remove fitted sine to analyze distortion and noise
- **Signal Characterization**: Extract amplitude, frequency, phase for testing
- **Jitter Analysis**: Phase variations indicate timing jitter
- **Pre-processing**: For INL/DNL, harmonic decomposition, etc.

## Limitations

- **Single-tone only**: Assumes pure sine wave; fails with multi-tone inputs
- **No distortion modeling**: Harmonics treated as noise in residuals
- **Convergence**: May fail if initial estimate is very far from true frequency
- **Wraparound**: Frequency must be < 0.5 (Nyquist limit)

## See Also

- [`analyze_decomposition_time`](analyze_decomposition.md) — Time-domain harmonic decomposition using fitted sine
- [`analyze_inl_from_sine`](analyze_inl_from_sine.md) — INL/DNL extraction from sine wave
- [`analyze_error_pdf`](analyze_error_pdf.md) — Error distribution analysis

## References

1. IEEE Std 1057-2017, "IEEE Standard for Digitizing Waveform Recorders"
2. IEEE Std 1241-2010, "IEEE Standard for Terminology and Test Methods for ADCs"
