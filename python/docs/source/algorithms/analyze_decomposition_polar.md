# analyze_decomposition_polar

## Overview

`analyze_decomposition_polar` performs time-domain harmonic decomposition with polar (magnitude-phase) visualization. This provides an intuitive view of harmonic distortion structure, showing how each harmonic component relates to the fundamental in both amplitude and phase.

## Syntax

```python
from adctoolbox import analyze_decomposition_polar

# Basic usage with polar plot
result = analyze_decomposition_polar(signal, harmonic=5, show_plot=True)

# Custom parameters
result = analyze_decomposition_polar(signal, harmonic=9, show_plot=True,
                                     title="Memory Effect Analysis")
```

## Parameters

- **`signal`** (array_like) — Input ADC signal (sine wave excitation)
- **`harmonic`** (int, default=5) — Number of harmonics to decompose
- **`show_plot`** (bool, default=True) — Display polar decomposition plot
- **`ax`** (matplotlib polar axis, optional) — Polar axis to plot on
- **`title`** (str, optional) — Custom title for the plot

## Returns

Dictionary containing:
- **`fundamental`** — Fundamental component (time domain)
- **`harmonics`** — List of harmonic components (time domain)
- **`residual`** — Remaining error (noise + higher-order distortion)
- **`harmonic_amplitudes`** — Amplitude of each harmonic
- **`harmonic_phases`** — Phase of each harmonic (radians)
- **`frequency`** — Detected fundamental frequency

## Algorithm

```python
# 1. Fit fundamental sine wave
result = fit_sine_4param(signal)
fundamental = result['fitted_signal']

# 2. Extract each harmonic by fitting at k×f₀
for k in range(2, harmonic + 1):
    harmonic_k = fit_sine_at_frequency(signal, freq_estimate * k)
    amplitudes[k] = sqrt(A_k**2 + B_k**2)
    phases[k] = arctan2(-B_k, A_k)

# 3. Compute residual
residual = signal - fundamental - sum(all_harmonics)
```

## Polar Visualization

The polar plot shows:
- **Angle**: Phase of each harmonic relative to fundamental
- **Radius**: Magnitude of each harmonic (dBFS or linear)
- **Markers**: Different harmonics (HD2, HD3, HD4, ...)

**Interpretation:**
- **Clustered phases**: Coherent distortion mechanism
- **Random phases**: Multiple independent error sources
- **Phase = 0° or 180°**: In-phase/anti-phase with fundamental
- **Phase = 90° or 270°**: Quadrature distortion

## Examples

### Example 1: Harmonic Decomposition with Polar Plot

```python
import numpy as np
from adctoolbox import analyze_decomposition_polar

# Analyze ADC output
result = analyze_decomposition_polar(adc_signal, harmonic=5, show_plot=True)

# Print harmonic levels
for k in range(2, 6):
    amp = result['harmonic_amplitudes'][k-2]
    phase = np.degrees(result['harmonic_phases'][k-2])
    print(f"HD{k}: {amp:.4f} ({20*np.log10(amp):.1f} dBFS), "
          f"Phase: {phase:.1f}°")
```

### Example 2: Compare Time and Polar Views

```python
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(14, 6))

# Time-domain decomposition
ax1 = fig.add_subplot(121)
result_time = analyze_decomposition_time(signal, show_plot=True, ax=ax1)

# Polar decomposition
ax2 = fig.add_subplot(122, projection='polar')
result_polar = analyze_decomposition_polar(signal, show_plot=True, ax=ax2)

plt.tight_layout()
plt.show()
```

### Example 3: Memory Effect Detection

```python
# Memory effects show characteristic phase patterns
result = analyze_decomposition_polar(signal, harmonic=9, show_plot=True,
                                     title="Residue Amplifier Memory Effect")

# Check for memory effect signature: HD2 near 180°, HD3 coherent
hd2_phase = np.degrees(result['harmonic_phases'][0])
hd3_phase = np.degrees(result['harmonic_phases'][1])

if abs(hd2_phase - 180) < 30:  # HD2 near anti-phase
    print("Possible memory effect detected")
```

## Interpretation

### Harmonic Phase Patterns

| Phase Pattern | Likely Cause |
|---------------|--------------|
| **All harmonics ~0°** | Symmetric compression/limiting |
| **HD2 at 0°, HD3 at 0°** | Positive nonlinearity (expansion) |
| **HD2 at 180°, HD3 at 0°** | Negative nonlinearity (compression) |
| **HD2 at 90°/270°** | Asymmetric transfer function |
| **Even harmonics clustered** | Differential pair mismatch |
| **Odd harmonics clustered** | Single-ended nonlinearity |

### Harmonic Magnitude Analysis

| Magnitude Pattern | Likely Cause |
|-------------------|--------------|
| **HD2 dominant** | Even-order nonlinearity (asymmetry) |
| **HD3 dominant** | Odd-order nonlinearity (curvature) |
| **HD2 = HD3** | Mixed nonlinearity |
| **High HD2, low HD3** | Differential pair mismatch |
| **Low HD2, high HD3** | Well-matched differential design |

## Use Cases

- **Distinguish nonlinearity mechanisms**: Even vs. odd order
- **Identify memory effects**: Characteristic phase signatures
- **Visualize distortion structure**: More intuitive than time domain
- **Debug ADC architectures**: Pipelined, SAR, flash
- **Compare before/after calibration**: Phase should remain stable

## See Also

- [`analyze_decomposition_time`](analyze_decomposition_time.md) — Time-domain view
- [`analyze_spectrum`](analyze_spectrum.md) — Frequency-domain metrics
- [`fit_static_nonlin`](fit_static_nonlin.md) — Extract k2/k3 coefficients

## References

1. IEEE Std 1057-2017, "IEEE Standard for Digitizing Waveform Recorders"
2. R. Schreier and G. C. Temes, "Understanding Delta-Sigma Data Converters," Wiley-IEEE Press, 2005
