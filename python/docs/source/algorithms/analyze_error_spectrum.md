# analyze_error_spectrum

## Overview

`analyze_error_spectrum` computes the FFT spectrum of the ADC error signal (data - fitted sine) to reveal frequency components in the error. This is distinct from analyzing the signal spectrum itself—here we analyze only the residual error after removing the ideal sine wave.

## Syntax

```python
from adctoolbox import analyze_error_spectrum

# Basic usage
result = analyze_error_spectrum(signal, fs=100e6, show_plot=True)

# With known frequency
result = analyze_error_spectrum(signal, fs=800e6, frequency=0.123,
                                show_plot=True)

# Custom title
result = analyze_error_spectrum(signal, fs=100e6, show_plot=True,
                                title="Error Spectrum: 25°C")
```

## Parameters

- **`signal`** (array_like) — Input ADC signal (sine wave excitation)
- **`fs`** (float, default=1) — Sampling frequency in Hz
- **`frequency`** (float, optional) — Normalized frequency (0-0.5)
  - If None: auto-detected via FFT
- **`show_plot`** (bool, default=True) — Display error spectrum plot
- **`ax`** (matplotlib axis, optional) — Axis to plot on
- **`title`** (str, optional) — Title for the plot

## Returns

Dictionary containing:

**Metrics (of error signal):**
- **`enob`** — Effective Number of Bits
- **`sndr_db`** — Signal-to-Noise and Distortion Ratio (dB)
- **`sfdr_db`** — Spurious-Free Dynamic Range (dB)
- **`snr_db`** — Signal-to-Noise Ratio (dB)
- **`thd_db`** — Total Harmonic Distortion (dB)
- **`sig_pwr_dbfs`** — Signal power (dBFS)
- **`noise_floor_dbfs`** — Noise floor (dBFS)

**Error Data:**
- **`error_signal`** — Error signal (data - fitted sine)

## Algorithm

```python
# 1. Fit ideal sine wave
if frequency is None:
    result = fit_sine_4param(signal)
else:
    result = fit_sine_4param(signal, frequency_estimate=frequency)

fitted_sine = result['fitted_signal']

# 2. Compute error
error_signal = signal - fitted_sine

# 3. Analyze spectrum of error (not signal!)
spectrum_result = analyze_spectrum(error_signal, fs=fs)
```

## Key Difference from analyze_spectrum

| Function | What it Analyzes | Use Case |
|----------|------------------|----------|
| **`analyze_spectrum`** | Original signal spectrum | Overall ADC performance (ENOB, SNR, harmonics) |
| **`analyze_error_spectrum`** | Error signal spectrum | Error characteristics, frequency-dependent errors |

## Examples

### Example 1: Error Spectrum Analysis

```python
import numpy as np
from adctoolbox import analyze_error_spectrum

# Analyze error spectrum
result = analyze_error_spectrum(adc_signal, fs=800e6, show_plot=True)

print(f"Error SNDR: {result['sndr_db']:.2f} dB")
print(f"Error SFDR: {result['sfdr_db']:.2f} dB")
print(f"Noise floor: {result['noise_floor_dbfs']:.2f} dBFS")
```

### Example 2: Compare Signal vs. Error Spectra

```python
import matplotlib.pyplot as plt
from adctoolbox import analyze_spectrum, analyze_error_spectrum

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Signal spectrum (shows fundamental + harmonics + noise)
plt.sca(ax1)
sig_result = analyze_spectrum(signal, fs=fs, show_plot=True)
ax1.set_title('Signal Spectrum')

# Error spectrum (shows harmonics + noise, NO fundamental)
plt.sca(ax2)
err_result = analyze_error_spectrum(signal, fs=fs, show_plot=True)
ax2.set_title('Error Spectrum')

plt.tight_layout()
plt.show()
```

### Example 3: Identify Frequency-Dependent Errors

```python
# Look for spurs in error spectrum
result = analyze_error_spectrum(signal, fs=800e6, show_plot=True)

# Error spectrum should be relatively flat (white noise)
# Peaks indicate frequency-dependent errors:
# - Power supply coupling
# - Clock feedthrough
# - Sampling artifacts
```

### Example 4: Batch Analysis

```python
# Analyze multiple conditions
conditions = ['Room Temp', 'High Temp', 'Low Voltage']
signals = [sig_25c, sig_85c, sig_low_vdd]

for cond, sig in zip(conditions, signals):
    result = analyze_error_spectrum(sig, fs=fs, show_plot=False,
                                     title=cond)
    print(f"{cond:15s}: Error SNDR = {result['sndr_db']:.2f} dB")
```

## Interpretation

### Error Spectrum Shape

| Spectrum Shape | Interpretation |
|----------------|----------------|
| **Flat (white)** | Random noise (thermal, quantization) |
| **1/f shape** | Flicker noise, drift |
| **Peaks at harmonics** | Nonlinearity (HD2, HD3, etc.) |
| **Peaks at non-harmonics** | Spurs (power supply, clock coupling) |
| **High at low freq** | Offset drift, 1/f noise |

### Common Error Signatures

| Peak Location | Likely Cause |
|---------------|--------------|
| **DC (0 Hz)** | Offset drift (should be minimal) |
| **f_clock** | Clock feedthrough |
| **f_supply** | Power supply ripple |
| **2×f_in, 3×f_in** | Harmonic distortion |
| **f_in ± f_clock** | Sampling artifacts |

### SFDR of Error

- **Error SFDR > 80 dB**: Excellent, noise-limited
- **60 < Error SFDR < 80 dB**: Good, some distortion
- **Error SFDR < 60 dB**: Significant spurious content

## Use Cases

- **Distinguish noise from distortion** in error
- **Identify interference sources** (spurs in error spectrum)
- **Validate fitting quality** (DC component should be near zero)
- **Debug frequency-dependent errors**
- **Measure noise floor** excluding signal energy

## Comparison with Other Error Analysis

| Function | Domain | Shows |
|----------|--------|-------|
| **`analyze_error_spectrum`** | Frequency | Error frequency components |
| **`analyze_error_pdf`** | Statistical | Error distribution |
| **`analyze_error_autocorr`** | Time | Temporal correlation |
| **`analyze_error_envelope_spectrum`** | Frequency | AM modulation in error |
| **`analyze_error_by_phase`** | Phase | AM/PM decomposition |

## See Also

- [`analyze_spectrum`](analyze_spectrum.md) — Signal spectrum (not error)
- [`analyze_error_envelope_spectrum`](analyze_error_envelope_spectrum.md) — Error envelope (AM detection)
- [`analyze_error_pdf`](analyze_error_pdf.md) — Error distribution
- [`fit_sine_4param`](fit_sine_4param.md) — Sine fitting for error extraction

## References

1. IEEE Std 1241-2010, "IEEE Standard for Terminology and Test Methods for ADCs"
2. B. Razavi, "Principles of Data Conversion System Design," IEEE Press, 1995
