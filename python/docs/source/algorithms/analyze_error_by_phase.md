# analyze_error_by_phase

## Overview

`analyze_error_by_phase` performs AM/PM (Amplitude Modulation / Phase Modulation) decomposition of ADC errors as a function of the input signal phase. This reveals whether errors are signal-dependent and whether they modulate the amplitude (AM) or timing/phase (PM) of the signal.

## Syntax

```python
from adctoolbox import analyze_error_by_phase

# Basic usage with auto-detected frequency
result = analyze_error_by_phase(signal, show_plot=True)

# With specified frequency
result = analyze_error_by_phase(signal, norm_freq=0.123, n_bins=100,
                                show_plot=True)

# Exclude base noise term
result = analyze_error_by_phase(signal, include_base_noise=False,
                                show_plot=True)
```

## Parameters

- **`signal`** (array_like) — Input ADC signal (sine wave excitation)
- **`norm_freq`** (float, optional) — Normalized frequency (f/fs), range (0, 0.5)
  - If None: auto-detected via FFT
- **`n_bins`** (int, default=100) — Number of phase bins for visualization
- **`include_base_noise`** (bool, default=True) — Include base noise term in fitting
- **`show_plot`** (bool, default=True) — Display error vs. phase plot
- **`axes`** (tuple, optional) — Tuple of (ax1, ax2) for top and bottom panels
- **`ax`** (matplotlib axis, optional) — Single axis to split into 2 panels
- **`title`** (str, optional) — Test setup description for title

## Returns

Dictionary containing:

**Numerical Results:**
- **`am_noise_rms_v`** — AM noise RMS (amplitude modulation)
- **`pm_noise_rms_v`** — PM noise RMS in voltage units
- **`pm_noise_rms_rad`** — PM noise RMS in radians
- **`base_noise_rms_v`** — Base noise RMS (signal-independent)
- **`total_rms_v`** — Total error RMS

**Validation Metrics:**
- **`r_squared_raw`** — R² for raw data fit (energy ratio)
- **`r_squared_binned`** — R² for binned data (model confidence)

**Visualization Data:**
- **`bin_error_rms_v`** — RMS error per phase bin
- **`bin_error_mean_v`** — Mean error per phase bin
- **`phase_bin_centers_rad`** — Phase bin centers (radians)

**Metadata:**
- **`amplitude`**, **`dc_offset`**, **`norm_freq`** — Fitted signal parameters
- **`fitted_signal`**, **`error`**, **`phase`** — Signal decomposition

## Algorithm

### AM/PM Model

Error is decomposed as:
```
error(φ) = AM·sin(φ) + PM·cos(φ) + base_noise
```

where:
- **AM**: Amplitude modulation error (gain variation with signal level)
- **PM**: Phase modulation error (timing jitter, settling errors)
- **base_noise**: Signal-independent noise
- **φ**: Signal phase

### Dual-Track Analysis

**Path A (Raw)**: Fit all N samples → highest precision AM/PM values
**Path B (Binned)**: Compute binned statistics → visualization

Cross-validation: Path A coefficients predict Path B trend → R² metric

## Examples

### Example 1: AM/PM Decomposition

```python
import numpy as np
from adctoolbox import analyze_error_by_phase

# Analyze ADC signal
result = analyze_error_by_phase(adc_signal, show_plot=True)

print(f"AM noise: {result['am_noise_rms_v']*1e6:.2f} µV RMS")
print(f"PM noise: {result['pm_noise_rms_rad']*1e12:.2f} pico-radians RMS")
print(f"Base noise: {result['base_noise_rms_v']*1e6:.2f} µV RMS")
print(f"Total RMS: {result['total_rms_v']*1e6:.2f} µV RMS")
print(f"R² (model fit): {result['r_squared_raw']:.4f}")
```

### Example 2: Identify Dominant Error Mechanism

```python
result = analyze_error_by_phase(signal, show_plot=False)

am = result['am_noise_rms_v']
pm = result['pm_noise_rms_v']
base = result['base_noise_rms_v']

# Determine dominant error source
errors = {'AM': am, 'PM': pm, 'Base': base}
dominant = max(errors, key=errors.get)

print(f"Dominant error: {dominant}")
print(f"  AM:   {am*1e6:.2f} µV ({am/result['total_rms_v']*100:.1f}%)")
print(f"  PM:   {pm*1e6:.2f} µV ({pm/result['total_rms_v']*100:.1f}%)")
print(f"  Base: {base*1e6:.2f} µV ({base/result['total_rms_v']*100:.1f}%)")
```

### Example 3: Compare Multiple Conditions

```python
import matplotlib.pyplot as plt

conditions = {
    'Ideal': signal_ideal,
    'With Jitter': signal_jitter,
    'With Gain Error': signal_gain_error,
}

fig, axes = plt.subplots(len(conditions), 2, figsize=(12, 4*len(conditions)))

for i, (name, sig) in enumerate(conditions.items()):
    result = analyze_error_by_phase(sig, axes=axes[i], title=name,
                                     show_plot=True)
    print(f"{name}: AM={result['am_noise_rms_v']*1e6:.1f}µV, "
          f"PM={result['pm_noise_rms_rad']*1e12:.1f}pRad")

plt.tight_layout()
plt.show()
```

## Interpretation

### Error Type Classification

| Dominant Component | Likely Cause |
|--------------------|--------------|
| **AM dominant** | Gain error, amplitude-dependent distortion, residue amplifier gain variation |
| **PM dominant** | Clock jitter, timing errors, settling time issues |
| **Base noise dominant** | Thermal noise, quantization noise (signal-independent) |
| **AM ≈ PM** | Mixed analog impairments |

### Phase Pattern Analysis

| Error vs. Phase Pattern | Interpretation |
|-------------------------|----------------|
| **Sinusoidal (AM-like)** | Amplitude-dependent error |
| **Cosinusoidal (PM-like)** | Timing/phase-dependent error |
| **Flat (uniform)** | Signal-independent noise |
| **Complex shape** | Multiple error mechanisms |

### R² Interpretation

- **R² > 0.9**: Model fits well, errors are signal-dependent
- **0.5 < R² < 0.9**: Moderate signal dependence
- **R² < 0.5**: Errors mostly signal-independent (random noise)

## Use Cases

- **Distinguish jitter from amplitude errors**
- **Identify memory effects** in pipelined ADCs
- **Validate settling time** in SAR ADCs
- **Characterize residue amplifier** gain variations
- **Debug clock quality** (PM noise indicates jitter)

## Common Patterns

### Pipelined ADC
- High AM → Residue amplifier gain error
- High PM → Inadequate settling time

### SAR ADC
- High AM → DAC mismatch, reference variation
- High PM → Comparator metastability

### Flash ADC
- Low AM, low PM → Good performance
- High base noise → Comparator noise

## See Also

- [`analyze_error_by_value`](analyze_error_by_value.md) — Error vs. ADC code
- [`analyze_error_pdf`](analyze_error_pdf.md) — Error distribution
- [`analyze_error_autocorr`](analyze_error_autocorr.md) — Temporal correlation

## References

1. IEEE Std 1241-2010, "IEEE Standard for Terminology and Test Methods for ADCs"
2. M. Soudan et al., "A Novel AM-PM-Jitter Decomposition Method for Characterizing ADC Nonidealities," IEEE Trans. IM, 2008
